"""llmjury CLI: `llmjury solve`, `llmjury reproduce`, `llmjury --version`."""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

from . import __version__
from .engine import Engine
from .verifiers import FunctionalCodeVerifier, StdioCodeVerifier

CACHE = os.path.expanduser("~/.llmjury/cache.jsonl")

WARNING = (
    "[llmjury] note: verification runs the model-generated code. v0.1 isolates it "
    "(scrubbed env, temp dir, CPU/file limits) but is NOT a real sandbox — don't point "
    "it at untrusted tasks on a machine with secrets. Use a container/VM for that.\n")


def _read(path):
    if path == "-":
        return sys.stdin.read()
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        sys.exit(f"error: cannot read {path}: {e}")


def reproduce_default_num_ctx():
    """Default `reproduce --num-ctx`, imported lazily so it stays defined in one place."""
    from .benchmarks.reproduce import DEFAULT_NUM_CTX
    return DEFAULT_NUM_CTX


def _backend(name, num_ctx=None, think=False):
    if name == "ollama":
        from .backends import OllamaBackend
        return OllamaBackend(cache_path=CACHE, num_ctx=num_ctx or None, think=think)
    if name == "codex":
        from .backends import CodexBackend
        return CodexBackend(cache_path=CACHE)
    if name == "claude":
        from .backends import ClaudeBackend
        return ClaudeBackend(cache_path=CACHE)
    from .backends import OpenRouterBackend
    return OpenRouterBackend(cache_path=CACHE)


def _refuse_root():
    if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("LLMJURY_ALLOW_ROOT") != "1":
        sys.exit("error: refusing to run as root — LLM-Jury executes model-generated code. "
                 "Set LLMJURY_ALLOW_ROOT=1 to override (not recommended).")


def _frontier_models(value, backend_name):
    """Resolve CLI shorthand without making a provider call.

    Named ladders (auto/open/opus/fable) expand to OpenRouter slugs; anything
    else is passed through to the chosen provider verbatim.
    """
    from .panels import FRONTIER_ALIASES
    if value not in FRONTIER_ALIASES:
        return value
    if backend_name != "openrouter":
        raise ValueError(
            f"--frontier {value} requires --frontier-backend openrouter")
    return list(FRONTIER_ALIASES[value])


def _codex_frontier_rescue(value, backend_name):
    """Return the authenticated Codex rescue model for `auto` inside Codex.

    OpenRouter remains the ordered first choice. A Codex-hosted run already has a
    separate authenticated provider available, so it can recover after account,
    route, or provider failures without weakening the verifier gate.
    """
    if value != "auto" or backend_name != "openrouter":
        return None
    if not (os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID")):
        return None
    if not shutil.which("codex"):
        return None
    from .panels import CODEX_BEST
    return CODEX_BEST


def _claude_frontier_rescue(value, backend_name):
    """Return the authenticated Claude rescue model inside Claude Code."""
    if value != "auto" or backend_name != "openrouter":
        return None
    if os.environ.get("CLAUDECODE", "").strip() in ("", "0"):
        return None
    if not shutil.which("claude"):
        return None
    from .panels import CLAUDE_BEST
    return CLAUDE_BEST


def _func_cases(raw):
    """Convert --cases JSON into (args_tuple, expected) pairs for a function call.

    Used when --cases is paired with --entry-point: each case calls the entry point
    with `args` and compares its RETURN VALUE to `expected` (vs the stdin/stdout
    contract --cases uses on its own). Each case is an object with args/expected
    (preferred) or input/output aliases:

        {"args": [2, 3], "expected": 5}   -> entry_point(2, 3) == 5
        {"input": 5, "output": 25}        -> entry_point(5)    == 25  (scalar = one arg)

    A list under args/input is spread as positional args; a scalar becomes a single
    arg. JSON types are preserved, so numbers stay numbers and strings stay strings.
    """
    if not isinstance(raw, list):
        sys.exit("error: --cases must be a JSON array of case objects")
    out = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            sys.exit(f"error: --cases[{i}] must be an object, got {type(c).__name__}")
        if "args" in c:
            args = c["args"]
        elif "input" in c:
            args = c["input"]
        else:
            sys.exit(f"error: --cases[{i}] needs an 'args' (or 'input') field "
                     "when --entry-point is set")
        if "expected" in c:
            expected = c["expected"]
        elif "output" in c:
            expected = c["output"]
        else:
            sys.exit(f"error: --cases[{i}] needs an 'expected' (or 'output') field "
                     "when --entry-point is set")
        out.append((tuple(args) if isinstance(args, list) else (args,), expected))
    return out


def cmd_solve(a):
    if getattr(a, "backend", None) == "ollama":
        from .memguard import local_compute_lock
        try:
            with local_compute_lock():
                return _cmd_solve(a)
        except (RuntimeError, OSError) as error:
            sys.exit(f"error: local compute unavailable: {error}")
    return _cmd_solve(a)


def _cmd_solve(a):
    _refuse_root()
    from .memguard import exclusive_compute
    exclusive, owner = exclusive_compute()
    if exclusive:
        sys.exit(
            "error: llm-jury is standing down; exclusive 27B compute is active.\n"
            f"owner: {owner}\n"
            "The local council and every frontier provider, including OpenRouter, "
            "remain disabled until the 27B route releases the host."
        )
    task = _read(a.task)
    if a.cases:
        try:
            cases = json.loads(_read(a.cases))
        except json.JSONDecodeError as e:
            sys.exit(f"error: --cases is not valid JSON: {e}")
        if a.entry_point:
            # function-call cases: call entry_point(*args) and compare its return value
            verifier = FunctionalCodeVerifier.from_cases(a.entry_point, _func_cases(cases))
        else:
            # bare --cases: stdin/stdout contract (a full program reading stdin)
            verifier = StdioCodeVerifier(cases)
    elif a.tests:
        tests_suffix = Path(a.tests).suffix.lower()
        if tests_suffix and tests_suffix not in {".py", ".pyw"}:
            sys.exit(
                f"error: invalid --tests verifier: expected a Python file, got "
                f"{tests_suffix}")
        ep = a.entry_point or "solve"
        if not a.entry_point:
            sys.stderr.write("[llmjury] note: no --entry-point given; assuming 'solve'. "
                             "Pass --entry-point if your function has another name.\n")
        try:
            verifier = FunctionalCodeVerifier(_read(a.tests), ep)
        except ValueError as error:
            sys.exit(f"error: invalid --tests verifier: {error}")
    else:
        sys.exit("error: provide --tests (functional check) or --cases (stdin/stdout JSON)")

    backend = _backend(a.backend, num_ctx=a.num_ctx, think=a.think)
    panel = a.models.split(",") if a.models else None
    route = {}
    if a.brain:
        # Opt-in: add Shawn's fine-tuned MLX brain as an EXTRA council panelist via
        # its OpenAI-compatible endpoint, routed to its own backend. Appended after
        # the default/specified panel so it's a council member, not the stage-1 best.
        from .backends import OpenAICompatBackend
        from .panels import default_panel
        if panel is None:
            _, panel = default_panel(backend.name)
            panel = list(panel)
        panel.append(a.brain_model)
        route[a.brain_model] = OpenAICompatBackend(a.brain_url, cache_path=CACHE)
        sys.stderr.write(
            f"[llmjury] brain panelist: {a.brain_model} via {a.brain_url} "
            "(extra council member, not the stage-1 best)\n")
    best = a.best or (panel[0] if panel else None)
    try:
        frontier = _frontier_models(a.frontier, a.frontier_backend)
    except ValueError as e:
        sys.exit(f"error: {e}")
    claude_rescue = _claude_frontier_rescue(a.frontier, a.frontier_backend)
    codex_rescue = _codex_frontier_rescue(a.frontier, a.frontier_backend)
    rescue_model = claude_rescue or codex_rescue
    rescue_backend = "claude" if claude_rescue else "codex" if codex_rescue else None
    frontier_route = {}
    if rescue_model:
        frontier.append(rescue_model)
        frontier_route[rescue_model] = _backend(rescue_backend)
    from .panels import FRONTIER_ALIASES
    if a.frontier in FRONTIER_ALIASES:
        sys.stderr.write(
            f"[llmjury] {a.frontier} route: local council -> "
            + " -> ".join(frontier) + " (verifier-gated)\n")
    if rescue_model:
        host = "Claude Code" if rescue_backend == "claude" else "Codex"
        provider = "Claude" if rescue_backend == "claude" else "Codex"
        sys.stderr.write(
            f"[llmjury] {host} session detected; authenticated {provider} is the final "
            "rescue if OpenRouter produces no verified candidate\n")
    fb = None
    if frontier:
        fb = _backend(a.frontier_backend, num_ctx=a.num_ctx)
    use_panel = True
    if backend.name == "ollama":
        # Preflight: flag panel tags with a baked SYSTEM prompt before spending
        # minutes prefilling one (see baked_system_warnings for the field data).
        from .backends import baked_system_warnings, show_system_chars
        probe = panel
        if probe is None:
            from .panels import default_panel
            probe_best, probe_panel = default_panel(backend.name)
            probe = [probe_best] + list(probe_panel)
        probe = ([best] if best else []) + [m for m in probe if m not in route]
        # Preflight: refuse a panel that would not fit in RAM. A council loads every
        # panelist at once and Ollama caps residency by model count, not bytes, so an
        # over-large panel takes the host down (wired GPU allocations cannot be paged
        # out; the kernel watchdog panics) rather than failing in a way we could catch.
        if a.mem_check != "off":
            from .memguard import check as memory_check
            report = memory_check(probe, host=backend.host, num_ctx=a.num_ctx)
            if not report.ok:
                sys.stderr.write("[llmjury] " + report.message() + "\n")
                if a.mem_check == "refuse":
                    # A refusal usually says the PANEL cannot load here and nothing
                    # more, leaving the frontier ladder — remote, costing this host
                    # no memory — perfectly usable. Drop the panel and escalate
                    # rather than killing a run that still has a safe path to a
                    # verified answer. Two refusals are not like that:
                    #   report.terminal  Backdoor assigned all model compute to Qwen,
                    #                    so local and frontier calls must both stop
                    #   ollama frontier  would load the very models just refused.
                    #                    Not selectable today (--frontier-backend is
                    #                    openrouter/codex); kept so that adding
                    #                    it cannot silently re-open the hole.
                    if frontier and not report.terminal and a.frontier_backend != "ollama":
                        use_panel = False
                        frontier_providers = (
                            f"{a.frontier_backend}, then {rescue_backend}"
                            if rescue_model else a.frontier_backend
                        )
                        sys.stderr.write(
                            "[llmjury] skipping the local council; escalating straight to "
                            + " -> ".join(frontier) + f" on {frontier_providers} "
                            "(remote, needs no memory on this host)\n")
                    elif report.terminal:
                        sys.exit(f"error: llm-jury is standing down.\nhint: {report.hint()}")
                    else:
                        sys.exit(
                            "error: this panel would over-commit the host, which can hang or "
                            f"panic it.\nhint: {report.hint()}\n"
                            "or: add --frontier auto to escalate to the cloud ladder, which "
                            "needs no local memory\n"
                            "override: --mem-check warn (proceed anyway) or off (skip the check)")
                else:
                    sys.stderr.write(f"[llmjury] proceeding anyway: {report.hint()}\n")
        for warning in baked_system_warnings(
                probe, lambda m: show_system_chars(backend.host, m)):
            sys.stderr.write(warning + "\n")
    from .verifiers import sandbox_note
    sys.stderr.write(sandbox_note()[1])            # provisions the sandbox on first call
    r = Engine(backend, panel=panel, best=best, k=a.k, workers=a.jobs,
               frontier=frontier, frontier_backend=fb, route=route,
               frontier_route=frontier_route,
               use_panel=use_panel).solve(task, verifier)

    if a.json:
        import dataclasses
        payload = {k: v for k, v in dataclasses.asdict(r).items() if k != "raw"}
        print(json.dumps(payload))
    else:
        status = "VERIFIED" if r.verified else "UNVERIFIED (no candidate passed)"
        sys.stderr.write(
            f"# llmjury: {status}  [stage={r.stage}, model={r.model}, attempts={r.attempts}]\n\n")
        if r.verified:
            print(r.answer)            # only verified code reaches stdout
        else:
            sys.stderr.write((r.answer or "(no code extracted)") + "\n")
    # Hard exit: an early verified win can leave samples still decoding on
    # abandoned threads; sys.exit would join them and stall for up to a full
    # generation. Dropping the process closes their connections, which is also
    # what tells Ollama to cancel the leftover decodes and free the GPU.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if r.verified else 1)


def cmd_demo(a):
    from .backends import DemoBackend
    from .verifiers import FunctionalCodeVerifier
    sys.stderr.write("[llmjury] demo — offline, no API key, no Ollama. "
                     "Running the real pipeline on a canned task...\n")
    task = "Write a function `add(a, b)` that returns the sum of two numbers."
    verifier = FunctionalCodeVerifier.from_cases("add", [((2, 3), 5), ((-1, 1), 0), ((0, 0), 0)])
    r = Engine(DemoBackend(), k=2).solve(task, verifier)
    sys.stderr.write(f"# llmjury: {'VERIFIED' if r.verified else 'UNVERIFIED'}  "
                     f"[stage={r.stage}, model={r.model}, attempts={r.attempts}]\n\n")
    print(r.answer)
    sys.stderr.write("\n(The 'weak' model returned a-b; the verifier caught it; the council's "
                     "a+b passed. That's the whole product — offline.)\n")
    sys.exit(0 if r.verified else 1)


def cmd_reproduce(a):
    from .benchmarks import reproduce
    reproduce.run(a.which, backend=a.backend, n=a.n, k=a.k, pace=a.pace, num_ctx=a.num_ctx)


def cmd_delegate(a):
    from .delegation import CodexDelegator
    task = _read(a.task)
    try:
        result = CodexDelegator(timeout=a.timeout).delegate(
            task, a.workspace, model=a.model, effort=a.effort,
            sandbox=a.sandbox, add_dirs=a.add_dir,
        )
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    if a.json:
        print(json.dumps(result.to_dict()))
    else:
        print(f"[{result.status}] {result.summary}")
        if result.changed_files:
            print("changed: " + ", ".join(result.changed_files))
        for check in result.tests:
            print("check: " + check)
        for blocker in result.blockers:
            print("blocker: " + blocker, file=sys.stderr)
    sys.exit(0 if result.status == "completed" and result.returncode == 0 else 1)


def cmd_install_claude(a):
    from .claude_integration import install_claude_agent, install_claude_skill
    try:
        for kind, install in (("skill", install_claude_skill), ("agent", install_claude_agent)):
            path, changed = install(scope=a.scope, project=a.project, force=a.force)
            state = "installed" if changed else "already current"
            print(f"Claude Code {kind} {state}: {path}")
    except (ValueError, FileExistsError, OSError) as exc:
        sys.exit(f"error: {exc}")


def cmd_plan(a):
    from .planning import ClaudePlanner
    task = _read(a.task)
    try:
        result = ClaudePlanner(timeout=a.timeout).plan(
            task, a.workspace, model=a.model, effort=a.effort)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    if a.json:
        print(json.dumps(result.to_dict()))
    else:
        print(f"[{result.status}] {result.summary}")
        for step in result.steps:
            print(f"{step['id']}: {step['objective']} [{step['acceptance']}]")
        for risk in result.risks:
            print("risk: " + risk, file=sys.stderr)
        for question in result.questions:
            print("question: " + question, file=sys.stderr)
    sys.exit(0 if result.status == "planned" and result.returncode == 0 else 1)


def cmd_install_codex(a):
    from .codex_integration import install_codex_skill
    try:
        path, changed = install_codex_skill(force=a.force)
    except (FileExistsError, OSError) as exc:
        sys.exit(f"error: {exc}")
    state = "installed" if changed else "already current"
    print(f"Codex skill {state}: {path}")


def cmd_preflight(a):
    """Read-only admission probe for other local-model consumers."""
    from .memguard import check, DEFAULT_OLLAMA_HOST
    report = check(a.models.split(","), host=a.host or DEFAULT_OLLAMA_HOST,
                   num_ctx=a.num_ctx)
    print(json.dumps({"ok": report.ok, "terminal": report.terminal,
                      "reason": "admitted" if report.ok else report.message()}))
    sys.exit(0 if report.ok else 1)


def main():
    p = argparse.ArgumentParser(prog="llmjury",
                                description="Local verified answers. Don't vote, verify.")
    p.add_argument("--version", action="version", version=f"llmjury {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    guard = sub.add_parser("preflight", help="check local model admission without inference")
    guard.add_argument("--models", required=True, help="comma-separated model tags")
    guard.add_argument("--num-ctx", type=int, default=8192)
    guard.add_argument("--host", help="Ollama base URL")
    guard.set_defaults(func=cmd_preflight)

    s = sub.add_parser("solve", help="solve a verifiable task with a verified small-model council")
    s.add_argument("--task", required=True, help="file containing the problem statement")
    s.add_argument("--tests", help="functional test file: a full check(candidate) OR just its body")
    s.add_argument("--entry-point", help="function name the tests call (default: solve)")
    s.add_argument("--cases", help='JSON file of cases. Alone: [{"input": "...", "output": "..."}] '
                   'stdin/stdout cases. With --entry-point: [{"args": [2, 3], "expected": 5}] '
                   'function-call cases (entry_point(*args) == expected)')
    s.add_argument("--backend", default="openrouter",
                   choices=["openrouter", "ollama", "codex"])
    s.add_argument("--k", type=int, default=4, help="samples per model (best-of-k)")
    s.add_argument("--jobs", type=int, default=None,
                   help="concurrent generation threads across a stage "
                   "(default: k x panel size, capped at 16)")
    s.add_argument("--num-ctx", type=int, default=8192,
                   help="context window per request, --backend ollama only (default 8192; "
                   "0 = the server's default). Ollama sizes each model's KV cache as "
                   "num_ctx x OLLAMA_NUM_PARALLEL at load, so a lean value here is what "
                   "lets the whole council decode in parallel without evictions")
    s.add_argument("--mem-check", choices=["refuse", "warn", "off"], default="refuse",
                   help="preflight the local panel against physical RAM, --backend ollama "
                   "only (default refuse). A panel that does not fit does not fail "
                   "cleanly: it over-commits unified memory and can panic the host. "
                   "Tune the ceiling with LLMJURY_MEM_FRACTION (default 0.70)")
    s.add_argument("--think", action="store_true",
                   help="let thinking-capable Ollama models spend tokens on reasoning; "
                   "disabled by default so the verifier receives answer code")
    s.add_argument("--models", help="comma-separated council models (overrides the default panel)")
    s.add_argument("--best", help="model to try first (default: first of --models, or the panel best)")
    s.add_argument("--json", action="store_true", help="emit a JSON result instead of the human banner")
    s.add_argument("--frontier", help="opt-in: a cloud model to try when the council can't verify. "
                   "Named ladders: 'auto' (open-weight, then a proprietary top tier), "
                   "'open' (open-weight only), 'opus', 'fable'. Any other value is a provider slug")
    s.add_argument("--frontier-backend", default="openrouter",
                   choices=["openrouter", "codex"],
                   help="provider for --frontier (default: openrouter; codex reuses "
                   "its own CLI's auth, so escalation costs nothing beyond that subscription)")
    s.add_argument("--brain", action="store_true",
                   help="add Shawn's fine-tuned MLX brain as an extra council panelist via an "
                   "OpenAI-compatible endpoint (default the local mlx_lm.server). It joins the "
                   "council stage, not the stage-1 best.")
    s.add_argument("--brain-url", default="http://127.0.0.1:8801/v1",
                   help="OpenAI-compatible base URL for --brain (default: local MLX server)")
    s.add_argument("--brain-model", default="mlx-community/Qwen3.5-4B-MLX-4bit",
                   help="model id the --brain endpoint serves")
    s.set_defaults(func=cmd_solve)

    sub.add_parser("demo", help="run the full pipeline offline — no API key, no Ollama") \
        .set_defaults(func=cmd_demo)

    r = sub.add_parser("reproduce", help="reproduce the benchmark numbers from the post")
    r.add_argument("which", choices=["humaneval", "lcb"])
    r.add_argument("--backend", default="openrouter", choices=["openrouter", "ollama"])
    r.add_argument("--n", type=int, default=None, help="number of problems (default: full bundled slice)")
    r.add_argument("--k", type=int, default=4, help="samples per model (best-of-k)")
    r.add_argument("--pace", type=float, default=0.0,
                   help="seconds to pause between problems (duty-cycle to keep the GPU cool)")
    r.add_argument("--num-ctx", type=int, default=reproduce_default_num_ctx(),
                   help="Ollama context cap per request (default: 8192). Ollama sizes KV as "
                        "num_ctx x OLLAMA_NUM_PARALLEL at load, so leaving this to the "
                        "server default (often 32k) inflates every panelist.")
    r.set_defaults(func=cmd_reproduce)

    d = sub.add_parser(
        "delegate", help="run a bounded Claude-planned task through a workspace-confined Codex agent")
    d.add_argument("--task", required=True, help="task brief file, or - to read it from stdin")
    d.add_argument("--workspace", default=".", help="repository/workspace Codex may access")
    d.add_argument("--model", help="Codex model override (default: Codex configuration)")
    d.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    d.add_argument("--sandbox", default="workspace-write", choices=["read-only", "workspace-write"])
    d.add_argument("--add-dir", action="append", default=[],
                   help="additional writable directory (repeatable; use sparingly)")
    d.add_argument("--timeout", type=int, default=1800, help="execution timeout in seconds")
    d.add_argument("--json", action="store_true", help="emit the structured Codex handoff as JSON")
    d.set_defaults(func=cmd_delegate)

    i = sub.add_parser(
        "install-claude",
        help="install the LLM-Jury skill and fusion agent for Claude Code")
    i.add_argument("--scope", default="user", choices=["user", "project"])
    i.add_argument("--project", help="project root for --scope project (default: current directory)")
    i.add_argument("--force", action="store_true", help="replace a differing existing skill/agent")
    i.set_defaults(func=cmd_install_claude)

    q = sub.add_parser("plan", help="ask Claude Code for a read-only structured execution plan")
    q.add_argument("--task", required=True, help="task and current evidence file, or - for stdin")
    q.add_argument("--workspace", default=".", help="repository Claude may inspect read-only")
    q.add_argument("--model", help="Claude model override (default: Claude configuration)")
    q.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    q.add_argument("--timeout", type=int, default=900, help="planning timeout in seconds")
    q.add_argument("--json", action="store_true", help="emit the structured Claude plan as JSON")
    q.set_defaults(func=cmd_plan)

    c = sub.add_parser("install-codex", help="install automatic Claude planning for Codex")
    c.add_argument("--force", action="store_true", help="replace a differing existing skill")
    c.set_defaults(func=cmd_install_codex)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
