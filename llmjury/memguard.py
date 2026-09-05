"""Preflight memory ceiling for local (Ollama) councils.

A local council loads every panelist concurrently -- that is the whole point of the
parallel council. But Ollama caps residency by model COUNT
(``OLLAMA_MAX_LOADED_MODELS``, default 3 on a single-GPU host), never by bytes, and
3 is exactly the size of a default panel. So nothing anywhere knows the aggregate.

When the panel does not fit, the failure mode is not a clean out-of-memory error we
could catch and report. Metal allocations are wired and cannot be paged out, so the
host compresses and swaps everything else until the kernel watchdog is starved and
panics. That is unrecoverable from inside this process, which is why this check runs
*before* the first request rather than reacting to a failure.

Measured on a 36 GB M-series Mac at ``num_ctx 8192`` x ``OLLAMA_NUM_PARALLEL 4``
(= 32768 total KV cells), resident size per ``ollama ps``::

    granite4.1:3b    2.1 GB on disk  ->   4.6 GB resident
    phi4-mini:3.8b   2.5 GB          ->   5.7 GB
    qwen3.5:4b       3.4 GB          ->   5.9 GB
    llama3.1:8b      4.9 GB          ->   9.0 GB
    qwen3:8b         5.2 GB          ->   7.9 GB
    gemma3:12b       8.1 GB          ->  11.0 GB
    phi4             9.1 GB          ->  14.0 GB

Two things that table makes obvious, and that cost a pair of kernel panics to learn:
resident size is roughly double the on-disk size, and the KV term is charged per
*total* cells, i.e. ``num_ctx x OLLAMA_NUM_PARALLEL``. A "small" 3.4 GB model pinned
to a 64k-context tag is 7.5 GB resident, not 3.4 GB.

:func:`estimate_resident` fits that data with a deliberate bias toward
over-estimating. Refusing a run that would have fit is a minor annoyance; allowing
one that does not fit takes the machine down.
"""
import json
import os
import platform
import subprocess
import urllib.error
import urllib.request

GB = 1024 ** 3

# Fitted against the measurements above, biased high. `resident ~= disk * 1.35 +
# cells * 85 KB` over-estimates every model in that table (by 4-25%) and never
# under-estimates one, which is the only direction that is safe to be wrong in.
WEIGHT_FACTOR = 1.35
KV_BYTES_PER_CELL = 85_000

# Assumed context when the caller defers to the server (`--num-ctx 0`). Ollama picks
# its default from VRAM and lands here on a 36 GB host, so this is the observed value
# rather than a guess -- and erring high is the safe direction.
SERVER_DEFAULT_CTX = 32768

# Share of physical RAM a council may occupy. The rest is not slack: it is the OS,
# the editor, the browser, and the agent session that launched this run.
#
# 0.65 matches the budget Ollama itself enforces on this host since 2026-09-05.
# Metal reports a static 28.1 GiB to Ollama regardless of what the desktop is
# using, which is how a 16.5 GB load went in on top of 8.9 GiB free and an
# exhausted swap. The launchd plist now sets OLLAMA_GPU_OVERHEAD=4GiB, so Ollama
# budgets 28.1 - 4.0 - 0.5 (minimum) = 23.6 GiB. On 36 GiB that is 0.655; 0.65 is
# 23.4 GiB, the measured top of the current panel's resident range, so the
# preflight and the server refuse the same runs. A higher fraction here would
# approve a council that Ollama then evicts a member of, which serialises the
# panel rather than crashing the host, but quietly.
DEFAULT_MEM_FRACTION = 0.65

# Set to any non-empty value except "0" to permit a local panel while an iOS
# Simulator is booted. The default is refusal: a booted simulator was measured at
# 17.6 GB resident (282 CoreSimulator processes) on the 36 GB reference host, and a
# council on top of that is exactly the co-residency that panics a machine.
SIMULATOR_OVERRIDE_ENV = "LLMJURY_ALLOW_SIMULATOR"

# Backdoor's hybrid router (the :8083 proxy that fronts Claude Code) publishes its
# circuit-breaker state here. An OPEN breaker means the router has committed the
# local Ollama server to keeping in-flight sessions alive -- the same server and
# the same VRAM a council needs, so running one anyway is a fight over ~13 GB of
# qwen tier plus ~23 GB of panel on a 36 GB host.
#
# It also means something stronger. The router opens that breaker on exactly one
# condition: this host is offline. So when it is set, the frontier ladder is
# unreachable too, and there is no tier left to escalate to -- which is why this
# refusal is marked `offline` and the CLI exits instead of falling through to the
# cloud the way a RAM or simulator refusal does.
ROUTER_STATE_PATH = os.environ.get(
    "LLMJURY_ROUTER_STATE") or os.path.expanduser("~/.backdoor/failover-state.json")
ROUTER_OVERRIDE_ENV = "LLMJURY_ALLOW_ROUTER_FAILOVER"


def _env_float(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0 < value <= 1 else default


def mem_fraction():
    return _env_float("LLMJURY_MEM_FRACTION", DEFAULT_MEM_FRACTION)


def total_ram_bytes():
    """Physical RAM, or 0 when we cannot tell (which disables the check)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) if out.returncode == 0 else 0
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0
    return 0


def num_parallel():
    """Slots Ollama decodes per model. KV is charged num_ctx x this.

    The server's setting is usually invisible here: a launchd/systemd unit exports
    ``OLLAMA_NUM_PARALLEL`` into the *server* process, not into ours. Set
    ``LLMJURY_OLLAMA_PARALLEL`` to match it and the estimate stops being pessimistic;
    without it we assume Ollama's own default of 4, since guessing low understates KV
    fourfold and understating is the direction that panics the host.
    """
    for name in ("LLMJURY_OLLAMA_PARALLEL", "OLLAMA_NUM_PARALLEL"):
        try:
            value = int(os.environ.get(name, "") or 0)
        except ValueError:
            continue
        if value > 0:
            return value
    return 4


def simulator_stack():
    """Is an iOS Simulator booted on this host, and what does its stack hold?

    Returns ``(running, rss_bytes)``. A booted simulator device always runs
    ``launchd_sim``; the resident cost is summed over every CoreSimulator-path
    process because the stack is hundreds of small XPC services, not one big one
    (282 processes / 17.6 GB measured on the host this gate was written for).

    Any probe failure returns ``(False, 0)`` -- fail open, same policy as the rest
    of this module: the guard stops a known-bad run, it must not invent failures.
    Simulators are a macOS concern, so every other platform short-circuits.
    """
    if platform.system() != "Darwin":
        return False, 0
    try:
        probe = subprocess.run(["pgrep", "-x", "launchd_sim"],
                               capture_output=True, timeout=5)
    except Exception:
        return False, 0
    if probe.returncode != 0:
        return False, 0
    rss = 0
    try:
        ps = subprocess.run(["ps", "-Axo", "rss=,comm="],
                            capture_output=True, text=True, timeout=5)
        for line in ps.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and "CoreSimulator" in parts[1]:
                rss += int(parts[0]) * 1024
    except Exception:
        rss = 0
    return True, rss


def _simulator_allowed():
    return os.environ.get(SIMULATOR_OVERRIDE_ENV, "").strip() not in ("", "0")


def _router_allowed():
    return os.environ.get(ROUTER_OVERRIDE_ENV, "").strip() not in ("", "0")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True          # exists but not ours to signal
    return True


def router_failover(path=None):
    """Is backdoor's router currently serving traffic from the local GPU?

    Returns ``(active, reason)``. A missing, unreadable, or unparseable file
    means "not failing over" -- which is also the state of a host with no router
    installed at all. Fail open, same policy as the rest of this module.

    A flag whose writer is gone is treated as inactive. The router clears the
    flag on recovery, but a router *killed* while OPEN would leave it set
    forever, and permanently disabling the council is a worse failure than the
    brief race this avoids.
    """
    try:
        with open(path or ROUTER_STATE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, ""
    if not isinstance(data, dict) or not data.get("failover_active"):
        return False, ""
    pid = data.get("pid")
    if isinstance(pid, int) and not _pid_alive(pid):
        return False, ""
    reason = data.get("reason")
    return True, reason if isinstance(reason, str) else ""


def _get_json(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def disk_sizes(host):
    """{model tag: bytes on disk} from /api/tags, or None if Ollama is unreachable."""
    data = _get_json(host.rstrip("/") + "/api/tags")
    if not isinstance(data, dict):
        return None
    sizes = {}
    for entry in data.get("models") or []:
        name, size = entry.get("name"), entry.get("size")
        if isinstance(name, str) and isinstance(size, int):
            sizes[name] = size
            # `phi4` and `phi4:latest` are the same model; index both spellings so a
            # panel written either way resolves.
            if name.endswith(":latest"):
                sizes.setdefault(name[: -len(":latest")], size)
    return sizes


def loaded_bytes(host):
    """(bytes resident now, {tag: bytes}) from /api/ps. Empty when unreachable."""
    data = _get_json(host.rstrip("/") + "/api/ps")
    if not isinstance(data, dict):
        return 0, {}
    by_model = {}
    for entry in data.get("models") or []:
        name, size = entry.get("name"), entry.get("size")
        if isinstance(name, str) and isinstance(size, int):
            by_model[name] = size
    return sum(by_model.values()), by_model


def estimate_resident(disk_bytes, cells):
    """Predicted resident bytes for a model of `disk_bytes` at `cells` KV cells."""
    return int(disk_bytes * WEIGHT_FACTOR + cells * KV_BYTES_PER_CELL)


def _canonical(tag, sizes):
    if tag in sizes:
        return tag
    if f"{tag}:latest" in sizes:
        return f"{tag}:latest"
    return None


class Report:
    """Outcome of a preflight check. `ok` False means the panel will not fit."""

    def __init__(self, ok, budget=0, projected=0, resident=0, per_model=None,
                 unknown=None, skipped=None, simulator=False, simulator_rss=0,
                 router=False, router_reason=""):
        self.ok = ok
        self.budget = budget
        self.projected = projected
        self.resident = resident
        self.per_model = per_model or []
        self.unknown = unknown or []
        self.skipped = skipped
        self.simulator = simulator
        self.simulator_rss = simulator_rss
        self.router = router
        self.router_reason = router_reason

    @property
    def offline(self):
        """Is the cloud unreachable too, so escalating cannot help?

        A RAM or simulator refusal is about THIS host's memory and leaves the
        frontier ladder perfectly usable. A router-failover refusal does not:
        the router only fails over when the host is offline, so there is no tier
        left to escalate to. Callers use this to decide between falling through
        to the frontier and stopping outright.
        """
        return self.router

    def message(self):
        """Operator-facing explanation, with the arithmetic that drove the verdict."""
        if self.router:
            because = f" ({self.router_reason})" if self.router_reason else ""
            return (f"backdoor's router has failed over to the local GPU{because}; "
                    "it opens that breaker only when this host is offline, so "
                    "neither the local council nor the cloud ladder is available")
        if self.simulator:
            held = (f" (holding ~{self.simulator_rss / GB:.1f} GB)"
                    if self.simulator_rss else "")
            return (f"an iOS Simulator is booted{held}; a local panel cannot "
                    "co-reside with the CoreSimulator stack on this host")
        lines = [
            f"panel needs ~{self.projected / GB:.1f} GB resident, "
            f"budget is {self.budget / GB:.1f} GB"
        ]
        for tag, cost in self.per_model:
            lines.append(f"  {tag:<24} ~{cost / GB:>5.1f} GB")
        if self.resident:
            lines.append(f"  {'(already loaded)':<24} ~{self.resident / GB:>5.1f} GB")
        if self.unknown:
            lines.append("  not pulled, cost unknown: " + ", ".join(self.unknown))
        return "\n".join(lines)

    def hint(self):
        if self.router:
            return ("wait for the network — the router closes its breaker on its own "
                    f"and republishes the change; inspect {ROUTER_STATE_PATH}; or set "
                    f"{ROUTER_OVERRIDE_ENV}=1 to run anyway (the council will then "
                    "contend with the router for the same VRAM)")
        if self.simulator:
            return ("shut the simulator down first: `xcrun simctl shutdown all` "
                    "(it reboots in seconds when next needed); or use a cloud "
                    "backend (--backend openrouter / --frontier-backend codex); "
                    f"or set {SIMULATOR_OVERRIDE_ENV}=1 to allow co-residency")
        cheapest = sorted(self.per_model, key=lambda kv: kv[1])[:2]
        parts = []
        if cheapest:
            parts.append("use a smaller panel, e.g. --models "
                         + ",".join(tag for tag, _ in cheapest))
        parts.append("lower --num-ctx")
        parts.append("set OLLAMA_NUM_PARALLEL=1 (KV is charged num_ctx x slots)")
        if self.resident:
            parts.append("or wait for loaded models to unload (OLLAMA_KEEP_ALIVE)")
        return "; ".join(parts)


def check(models, host="http://localhost:11434", num_ctx=8192, parallel=None,
          fraction=None):
    """Would loading `models` concurrently over-commit this host?

    Returns a :class:`Report`. Anything we cannot determine -- Ollama unreachable,
    RAM unreadable -- yields ``ok=True`` with ``skipped`` set: this guard exists to
    stop a known-bad run, not to become a new way for runs to fail.
    """
    # The router owns the GPU when it has failed over, and its failover means the
    # host is offline. Checked first because it is the only refusal that also
    # rules out escalating to the cloud -- there is nothing else to try.
    if not _router_allowed():
        failing_over, reason = router_failover()
        if failing_over:
            return Report(False, router=True, router_reason=reason)

    # A booted iOS Simulator excludes a local panel outright, before any RAM
    # arithmetic: the CoreSimulator stack is hundreds of processes whose resident
    # cost (17.6 GB measured) does not show up in any number this module models,
    # and council-plus-simulator is the co-residency that takes the host down.
    # The operator escape hatch is deliberate and explicit, never inferred.
    if not _simulator_allowed():
        sim_running, sim_rss = simulator_stack()
        if sim_running:
            return Report(False, simulator=True, simulator_rss=sim_rss)

    total = total_ram_bytes()
    if not total:
        return Report(True, skipped="cannot read physical RAM")

    sizes = disk_sizes(host)
    if sizes is None:
        return Report(True, skipped="ollama unreachable")

    slots = num_parallel() if parallel is None else parallel
    ctx = int(num_ctx) if int(num_ctx or 0) > 0 else SERVER_DEFAULT_CTX
    cells = ctx * max(int(slots), 1)
    budget = int(total * (mem_fraction() if fraction is None else fraction))

    resident_total, resident_by_model = loaded_bytes(host)

    # Callers assemble the probe list as [best] + panel, and `best` is normally a
    # member of the panel, so the same tag arrives twice. Ollama loads it once;
    # counting it twice would over-refuse a panel that actually fits.
    seen, unique = set(), []
    for tag in models:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    per_model, unknown, need = [], [], 0
    for tag in unique:
        canonical = _canonical(tag, sizes)
        if canonical is None:
            unknown.append(tag)
            continue
        cost = estimate_resident(sizes[canonical], cells)
        per_model.append((tag, cost))
        # Already resident models are counted once, via resident_total.
        if canonical not in resident_by_model and tag not in resident_by_model:
            need += cost

    projected = resident_total + need
    return Report(projected <= budget, budget=budget, projected=projected,
                  resident=resident_total, per_model=per_model, unknown=unknown)
