"""Verifiers — the heart of LLM-Jury.

A verifier checks a candidate WITHOUT being told the answer, so it can SELECT the
right one from many attempts. For code that's just: run it against tests and see if
it passes. The verifier is what makes a council beat a single model — it keeps the
correct attempt and discards the wrong ones.

SECURITY: these verifiers EXECUTE model-generated code. That is the whole point, and
it is inherently dangerous. Execution funnels through one seam, ``_run``, which by
default (``LLMJURY_SANDBOX=auto``) runs the code in a throwaway container — no network,
all capabilities dropped, non-root, with memory/cpu/pids caps — provisioning the Docker
daemon (colima or Docker Desktop) and runner image on its own, and falling back to the
host path when none can be brought up. ``LLMJURY_SANDBOX=docker`` requires the container;
``LLMJURY_SANDBOX=off`` keeps the host path. The host path is hardened (scrubbed env,
isolated temp dir, POSIX CPU/file/core limits) but is NOT a real sandbox — for untrusted
input keep the container (the default), or run inside your own VM.
"""
import os
import re
import sys
import time
import shutil
import signal
import textwrap
import subprocess
import tempfile

_MAX_OUTPUT = 1 << 20  # cap captured stdout/stderr at 1 MiB (a print-bomb can't OOM the parent)

# ---------------------------------------------------------------- code extraction
_PROG = re.compile(r"(?:^|\n)\s*(?:def |class |import |from |if |for |while |return |print\(|input\()")


def _looks_like_program(s):
    return bool(s and _PROG.search(s))


def extract_code(text):
    """Pull the model's solution out of its reply.

    Robust to two failure modes confirmed in the wild: (1) chatty models that put a
    trailing usage-example block after the real solution — we take the longest block
    that *looks like a program*, not blindly the last; (2) truncated/unterminated
    ``` fences from max_tokens cutoffs — we slice from a lone opening fence and strip
    stray fence markers.
    """
    if not text:
        return None
    blocks = re.findall(r"```(?:[A-Za-z0-9_+-]*)\s*\n?(.*?)```", text, re.DOTALL)
    prog = [b for b in blocks if _looks_like_program(b)]
    if prog:
        return max(prog, key=len).strip()
    if blocks:
        return max(blocks, key=len).strip()
    # No closed fence — handle a lone opening fence (common on truncation).
    m = re.search(r"```(?:[A-Za-z0-9_+-]*)\s*\n(.*)$", text, re.DOTALL)
    if m and _looks_like_program(m.group(1)):
        return m.group(1).strip()
    if _looks_like_program(text):
        return "\n".join(l for l in text.splitlines() if not l.strip().startswith("```")).strip()
    return None


# ---------------------------------------------------------------- sandboxing
def _safe_env(workdir):
    """An allowlist environment — the child gets none of the parent's secrets
    (API keys, tokens). HOME points inside the sandbox so `~` can't reach real files.
    """
    keep = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["HOME"] = workdir
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PATH", os.defpath)
    return env


def _limits(cpu_seconds):
    """A preexec_fn applying POSIX resource limits. Returns None on non-POSIX.

    Deliberately omits RLIMIT_NPROC (per-uid — can wedge the user's whole session)
    and RLIMIT_AS (fragile on macOS, where Python reserves huge virtual space). The
    CPU limit reaps runaway loops; the wall-clock timeout backstops it.
    """
    if not hasattr(os, "fork"):
        return None
    try:
        import resource
    except Exception:
        return None

    def apply():
        for what, soft, hard in (
            (resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1),
            (resource.RLIMIT_FSIZE, 10 * 1024 * 1024, 10 * 1024 * 1024),
            (resource.RLIMIT_CORE, 0, 0),
        ):
            try:
                resource.setrlimit(what, (soft, hard))
            except Exception:
                pass
    return apply


def _run_host(args, stdin, timeout, workdir):
    # start_new_session puts the child in its own process group so a timeout can reap
    # the WHOLE tree (forked grandchildren), not just the direct child.
    p = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=workdir, env=_safe_env(workdir),
        preexec_fn=_limits(timeout), start_new_session=hasattr(os, "setsid"))
    try:
        out, err = p.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                p.kill()
        except (ProcessLookupError, OSError):
            pass
        p.wait()
        raise
    return subprocess.CompletedProcess(args, p.returncode, out[:_MAX_OUTPUT], err[:_MAX_OUTPUT])


# ------------------------------------------------------- container sandbox (auto)
# Real isolation for model-generated code: a throwaway container with NO network,
# all capabilities dropped, non-root, and memory/cpu/pids caps. The tool provisions
# whatever it needs on its own — bringing the Docker daemon up and pulling the runner
# image on first use — then falls back to host execution if that can't be done.
_SANDBOX_IMAGE = os.environ.get("LLMJURY_SANDBOX_IMAGE", "python:3.12-slim")
_run_counter = 0
_provisioned = None        # cache: resolved sandbox mode for this process ("docker"/"host")


def _env_flag(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "off", "false", "no")


def _docker_responds():
    """True iff a Docker daemon is reachable right now."""
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=20).returncode == 0
    except Exception:
        return False


def _try_start_daemon():
    """Bring a stopped Docker daemon up on our own (colima or Docker Desktop), then
    wait for it to answer. Returns True if the daemon is reachable afterwards. Gated by
    LLMJURY_AUTOSTART (default on)."""
    if _docker_responds():
        return True
    if not _env_flag("LLMJURY_AUTOSTART", True):
        return False
    started = False
    if shutil.which("colima"):
        sys.stderr.write("[llmjury] starting the colima Docker daemon for sandboxed "
                         "execution (one-time, ~30-60s)...\n")
        try:
            subprocess.run(["colima", "start"], timeout=240,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            started = True
        except Exception:
            pass
    elif sys.platform == "darwin" and os.path.isdir("/Applications/Docker.app"):
        sys.stderr.write("[llmjury] starting Docker Desktop for sandboxed execution...\n")
        try:
            subprocess.run(["open", "-a", "Docker"], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            started = True
        except Exception:
            pass
    if not started:
        return False
    for _ in range(60):                       # poll up to ~120s for the daemon to answer
        if _docker_responds():
            return True
        time.sleep(2)
    return False


def _ensure_image():
    """Pull the runner image if it isn't present yet (one-time)."""
    try:
        have = subprocess.run(["docker", "image", "inspect", _SANDBOX_IMAGE],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=20).returncode == 0
        if have:
            return True
        sys.stderr.write(f"[llmjury] pulling sandbox image {_SANDBOX_IMAGE} (one-time)...\n")
        return subprocess.run(["docker", "pull", _SANDBOX_IMAGE], timeout=300,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def _resolve_sandbox():
    """Decide — once per process — whether code runs in a container or on the host,
    provisioning whatever the chosen mode needs.

    LLMJURY_SANDBOX: auto (default) | docker | off
      auto   — containerize whenever we can stand the daemon + image up; else host.
      docker — require the container; hard-error if it can't be provisioned.
      off    — host subprocess only (scrubbed env + POSIX limits; NOT a real sandbox).
    """
    global _provisioned
    if _provisioned is not None:
        return _provisioned
    mode = (os.environ.get("LLMJURY_SANDBOX") or "auto").strip().lower()
    if mode in ("off", "host", "none"):
        _provisioned = "host"
    elif mode in ("docker", "container", "on"):
        if not (_try_start_daemon() and _ensure_image()):
            sys.exit("[llmjury] LLMJURY_SANDBOX=docker but a container could not be "
                     "provisioned (daemon unreachable or image pull failed). "
                     "Set LLMJURY_SANDBOX=off to allow host execution.")
        _provisioned = "docker"
    else:                                      # auto
        _provisioned = "docker" if (_try_start_daemon() and _ensure_image()) else "host"
    return _provisioned


def sandbox_note():
    """(mode, human_message) for the CLI banner — provisions on first call."""
    mode = _resolve_sandbox()
    if mode == "docker":
        return ("docker", "[llmjury] sandbox=container — model code runs in a throwaway "
                f"container ({_SANDBOX_IMAGE}: no network, caps dropped, non-root, "
                "mem/cpu/pids capped).\n")
    return ("host", "[llmjury] sandbox=host — model code runs on the HOST (scrubbed env "
            "+ POSIX limits, NOT a real sandbox). Start Docker/colima, or set "
            "LLMJURY_SANDBOX=docker, for container isolation.\n")


def _run_docker(args, stdin, timeout, workdir):
    """Execute the candidate inside an ephemeral container. The per-task temp dir is
    bind-mounted at its real path so script-path args stay valid; the interpreter is
    swapped for the image's python. No network, all caps dropped, runs as the invoking
    uid (never root), with memory/cpu/pids caps and an in-container hard timeout."""
    global _run_counter
    _run_counter += 1
    wd = os.path.realpath(workdir)
    name = "llmjury-" + os.path.basename(wd) + "-" + str(_run_counter)

    def tr(a):
        if a == sys.executable:
            return "python3"
        if a == workdir:
            return wd
        if a.startswith(workdir + os.sep):
            return wd + a[len(workdir):]
        return a

    # `timeout` inside the container reaps runaway loops even though we drop the host
    # process-group plumbing; the host-side timeout below is the outer backstop.
    inner = ["timeout", "-k", "2", str(timeout)] + [tr(a) for a in args]
    docker = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "256",
        "--memory", "768m", "--memory-swap", "768m",
        "--cpus", "2",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{wd}:{wd}",
        "-w", wd,
        "-e", "HOME=" + wd,
        "-e", "PYTHONIOENCODING=utf-8",
        _SANDBOX_IMAGE,
    ] + inner
    try:
        p = subprocess.run(docker, input=stdin, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    return subprocess.CompletedProcess(args, p.returncode, p.stdout[:_MAX_OUTPUT],
                                       p.stderr[:_MAX_OUTPUT])


def _run(args, stdin, timeout, workdir):
    """Dispatch one code execution to a container (when provisioned) or the host."""
    if _resolve_sandbox() == "docker":
        return _run_docker(args, stdin, timeout, workdir)
    return _run_host(args, stdin, timeout, workdir)


def _tmproot():
    """Where per-task temp dirs live. Kept under $HOME so the bind-mount source sits
    inside the path colima / Docker Desktop share into the VM by default — macOS's
    standard TMPDIR (/var/folders/...) is NOT shared and would break the mount."""
    root = os.path.join(os.path.expanduser("~/.llmjury"), "tmp")
    try:
        os.makedirs(root, exist_ok=True)
        return root
    except Exception:
        return None


def _same_output(got, want):
    """Competitive-judge semantics: ignore trailing whitespace per line and trailing
    blank lines."""
    def norm(s):
        lines = [l.rstrip() for l in (s or "").replace("\r\n", "\n").split("\n")]
        while lines and lines[-1] == "":
            lines.pop()
        return lines
    return norm(got) == norm(want)


# ---------------------------------------------------------------- verifiers
class Verifier:
    def verify(self, candidate_text) -> bool:
        raise NotImplementedError


class FunctionalCodeVerifier(Verifier):
    """HumanEval-style. The candidate defines `entry_point`; `test` is EITHER a full
    `def check(candidate): ...` OR just its body (assert statements). Passes iff
    `check(entry_point)` runs without raising.
    """
    def __init__(self, test, entry_point, header="", timeout=20):
        self.test = test
        self.entry_point = entry_point
        self.header = header
        self.timeout = timeout

    @classmethod
    def from_cases(cls, entry_point, cases, **kw):
        """Build a verifier from (args, expected) pairs — no check() string to learn:

            FunctionalCodeVerifier.from_cases("add", [((2, 3), 5), ((-1, 1), 0)])

        `args` may be a tuple of positional args, or a single value.
        """
        lines = []
        for args, expected in cases:
            if not isinstance(args, tuple):
                args = (args,)
            lines.append(f"assert candidate({', '.join(map(repr, args))}) == {expected!r}")
        return cls("\n".join(lines), entry_point, **kw)

    def verify(self, candidate_text):
        code = extract_code(candidate_text)
        if not code:
            return False
        # Accept a full `def check(candidate): ...` OR just its body (assert lines).
        test = self.test if re.search(r"(?m)^\s*def\s+check\s*\(", self.test or "") \
            else "def check(candidate):\n" + textwrap.indent(self.test or "pass", "    ")
        program = (
            f"{self.header}\n{code}\n\n{test}\n\n"
            f"check({self.entry_point})\nprint('LLMJURY_OK')"
        )
        try:
            with tempfile.TemporaryDirectory(prefix="llmjury-", dir=_tmproot()) as wd:
                p = _run([sys.executable, "-"], program, self.timeout, wd)
            return p.returncode == 0 and "LLMJURY_OK" in p.stdout
        except Exception:
            return False


class StdioCodeVerifier(Verifier):
    """Competitive-programming style. The candidate is a full program; for each
    {"input", "output"} case we run it with `input` on stdin and compare stdout
    (per-line, ignoring trailing whitespace). Passes iff every case matches.
    """
    def __init__(self, cases, timeout=10):
        self.cases = cases
        self.timeout = timeout

    def verify(self, candidate_text):
        code = extract_code(candidate_text)
        if not code:
            return False
        try:
            with tempfile.TemporaryDirectory(prefix="llmjury-", dir=_tmproot()) as wd:
                path = os.path.join(wd, "sol.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                for c in self.cases:
                    try:
                        p = _run([sys.executable, path], c.get("input", ""), self.timeout, wd)
                    except Exception:
                        return False
                    if p.returncode != 0 or not _same_output(p.stdout, c.get("output", "")):
                        return False
            return True
        except Exception:
            return False
