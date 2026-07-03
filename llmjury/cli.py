"""llmjury CLI: `llmjury solve`, `llmjury reproduce`, `llmjury --version`."""
import os
import sys
import json
import argparse

from . import __version__
from .engine import Engine
from .verifiers import FunctionalCodeVerifier, StdioCodeVerifier

CACHE = os.path.expanduser("~/.llmjury/cache.jsonl")

WARNING = (
    "[llmjury] note: verification runs the model-generated code. v0.1 isolates it "
    "(scrubbed env, temp dir, CPU/file limits) but is NOT a real sandbox — don't point "
    "it at untrusted tasks on a machine with secrets. Use a container/VM for that.\n")


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        sys.exit(f"error: cannot read {path}: {e}")


def _backend(name, num_ctx=None):
    if name == "ollama":
        from .backends import OllamaBackend
        return OllamaBackend(cache_path=CACHE, num_ctx=num_ctx or None)
    from .backends import OpenRouterBackend
    return OpenRouterBackend(cache_path=CACHE)


def _refuse_root():
    if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("LLMJURY_ALLOW_ROOT") != "1":
        sys.exit("error: refusing to run as root — LLM-Jury executes model-generated code. "
                 "Set LLMJURY_ALLOW_ROOT=1 to override (not recommended).")


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
    _refuse_root()
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
        ep = a.entry_point or "solve"
        if not a.entry_point:
            sys.stderr.write("[llmjury] note: no --entry-point given; assuming 'solve'. "
                             "Pass --entry-point if your function has another name.\n")
        verifier = FunctionalCodeVerifier(_read(a.tests), ep)
    else:
        sys.exit("error: provide --tests (functional check) or --cases (stdin/stdout JSON)")

    backend = _backend(a.backend, num_ctx=a.num_ctx)
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
    fb = None
    if a.frontier:
        from .backends import OpenRouterBackend
        fb = OpenRouterBackend(cache_path=CACHE)   # the frontier tier is a cloud model
    from .verifiers import sandbox_note
    sys.stderr.write(sandbox_note()[1])            # provisions the sandbox on first call
    r = Engine(backend, panel=panel, best=best, k=a.k, workers=a.jobs,
               frontier=a.frontier, frontier_backend=fb, route=route).solve(task, verifier)

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
    reproduce.run(a.which, backend=a.backend, n=a.n, k=a.k, pace=a.pace)


def main():
    p = argparse.ArgumentParser(prog="llmjury",
                                description="Local verified answers. Don't vote, verify.")
    p.add_argument("--version", action="version", version=f"llmjury {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="solve a verifiable task with a verified small-model council")
    s.add_argument("--task", required=True, help="file containing the problem statement")
    s.add_argument("--tests", help="functional test file: a full check(candidate) OR just its body")
    s.add_argument("--entry-point", help="function name the tests call (default: solve)")
    s.add_argument("--cases", help='JSON file of cases. Alone: [{"input": "...", "output": "..."}] '
                   'stdin/stdout cases. With --entry-point: [{"args": [2, 3], "expected": 5}] '
                   'function-call cases (entry_point(*args) == expected)')
    s.add_argument("--backend", default="openrouter", choices=["openrouter", "ollama"])
    s.add_argument("--k", type=int, default=4, help="samples per model (best-of-k)")
    s.add_argument("--jobs", type=int, default=None,
                   help="concurrent generation threads across a stage "
                   "(default: k x panel size, capped at 16)")
    s.add_argument("--num-ctx", type=int, default=8192,
                   help="context window per request, --backend ollama only (default 8192; "
                   "0 = the server's default). Ollama sizes each model's KV cache as "
                   "num_ctx x OLLAMA_NUM_PARALLEL at load, so a lean value here is what "
                   "lets the whole council decode in parallel without evictions")
    s.add_argument("--models", help="comma-separated council models (overrides the default panel)")
    s.add_argument("--best", help="model to try first (default: first of --models, or the panel best)")
    s.add_argument("--json", action="store_true", help="emit a JSON result instead of the human banner")
    s.add_argument("--frontier", help="opt-in: a strong cloud model (e.g. deepseek/deepseek-v4-pro) "
                   "to escalate to when the local council can't verify (needs OPENROUTER_API_KEY)")
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
    r.set_defaults(func=cmd_reproduce)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
