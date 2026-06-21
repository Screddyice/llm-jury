"""litmus CLI: `litmus solve`, `litmus reproduce`, `litmus --version`."""
import os
import sys
import json
import argparse

from . import __version__
from .engine import Engine
from .verifiers import FunctionalCodeVerifier, StdioCodeVerifier

CACHE = os.path.expanduser("~/.litmus/cache.jsonl")

WARNING = (
    "[litmus] note: verification runs the model-generated code. v0.1 isolates it "
    "(scrubbed env, temp dir, CPU/file limits) but is NOT a real sandbox — don't point "
    "it at untrusted tasks on a machine with secrets. Use a container/VM for that.\n")


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        sys.exit(f"error: cannot read {path}: {e}")


def _backend(name):
    if name == "ollama":
        from .backends import OllamaBackend
        return OllamaBackend(cache_path=CACHE)
    from .backends import OpenRouterBackend
    return OpenRouterBackend(cache_path=CACHE)


def _refuse_root():
    if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("LITMUS_ALLOW_ROOT") != "1":
        sys.exit("error: refusing to run as root — Litmus executes model-generated code. "
                 "Set LITMUS_ALLOW_ROOT=1 to override (not recommended).")


def cmd_solve(a):
    _refuse_root()
    task = _read(a.task)
    if a.cases:
        try:
            cases = json.loads(_read(a.cases))
        except json.JSONDecodeError as e:
            sys.exit(f"error: --cases is not valid JSON: {e}")
        verifier = StdioCodeVerifier(cases)
    elif a.tests:
        ep = a.entry_point or "solve"
        if not a.entry_point:
            sys.stderr.write("[litmus] note: no --entry-point given; assuming 'solve'. "
                             "Pass --entry-point if your function has another name.\n")
        verifier = FunctionalCodeVerifier(_read(a.tests), ep)
    else:
        sys.exit("error: provide --tests (functional check) or --cases (stdin/stdout JSON)")

    panel = a.models.split(",") if a.models else None
    best = a.best or (panel[0] if panel else None)
    fb = None
    if a.frontier:
        from .backends import OpenRouterBackend
        fb = OpenRouterBackend(cache_path=CACHE)   # the frontier tier is a cloud model
    sys.stderr.write(WARNING)
    r = Engine(_backend(a.backend), panel=panel, best=best, k=a.k,
               frontier=a.frontier, frontier_backend=fb).solve(task, verifier)

    if a.json:
        import dataclasses
        payload = {k: v for k, v in dataclasses.asdict(r).items() if k != "raw"}
        print(json.dumps(payload))
    else:
        status = "VERIFIED" if r.verified else "UNVERIFIED (no candidate passed)"
        sys.stderr.write(
            f"# litmus: {status}  [stage={r.stage}, model={r.model}, attempts={r.attempts}]\n\n")
        if r.verified:
            print(r.answer)            # only verified code reaches stdout
        else:
            sys.stderr.write((r.answer or "(no code extracted)") + "\n")
    sys.exit(0 if r.verified else 1)


def cmd_demo(a):
    from .backends import DemoBackend
    from .verifiers import FunctionalCodeVerifier
    sys.stderr.write("[litmus] demo — offline, no API key, no Ollama. "
                     "Running the real pipeline on a canned task...\n")
    task = "Write a function `add(a, b)` that returns the sum of two numbers."
    verifier = FunctionalCodeVerifier.from_cases("add", [((2, 3), 5), ((-1, 1), 0), ((0, 0), 0)])
    r = Engine(DemoBackend(), k=2).solve(task, verifier)
    sys.stderr.write(f"# litmus: {'VERIFIED' if r.verified else 'UNVERIFIED'}  "
                     f"[stage={r.stage}, model={r.model}, attempts={r.attempts}]\n\n")
    print(r.answer)
    sys.stderr.write("\n(The 'weak' model returned a-b; the verifier caught it; the council's "
                     "a+b passed. That's the whole product — offline.)\n")
    sys.exit(0 if r.verified else 1)


def cmd_reproduce(a):
    from .benchmarks import reproduce
    reproduce.run(a.which, backend=a.backend, n=a.n, k=a.k, pace=a.pace)


def main():
    p = argparse.ArgumentParser(prog="litmus",
                                description="Local verified answers. Don't vote, verify.")
    p.add_argument("--version", action="version", version=f"litmus {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="solve a verifiable task with a verified small-model council")
    s.add_argument("--task", required=True, help="file containing the problem statement")
    s.add_argument("--tests", help="functional test file: a full check(candidate) OR just its body")
    s.add_argument("--entry-point", help="function name the tests call (default: solve)")
    s.add_argument("--cases", help='JSON file: [{"input": "...", "output": "..."}] stdin/stdout cases')
    s.add_argument("--backend", default="openrouter", choices=["openrouter", "ollama"])
    s.add_argument("--k", type=int, default=4, help="samples per model (best-of-k)")
    s.add_argument("--models", help="comma-separated council models (overrides the default panel)")
    s.add_argument("--best", help="model to try first (default: first of --models, or the panel best)")
    s.add_argument("--json", action="store_true", help="emit a JSON result instead of the human banner")
    s.add_argument("--frontier", help="opt-in: a strong cloud model (e.g. deepseek/deepseek-v4-pro) "
                   "to escalate to when the local council can't verify (needs OPENROUTER_API_KEY)")
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
