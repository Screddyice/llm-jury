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


def cmd_solve(a):
    task = _read(a.task)
    if a.cases:
        try:
            cases = json.loads(_read(a.cases))
        except json.JSONDecodeError as e:
            sys.exit(f"error: --cases is not valid JSON: {e}")
        verifier = StdioCodeVerifier(cases)
    elif a.tests:
        verifier = FunctionalCodeVerifier(_read(a.tests), a.entry_point or "solve")
    else:
        sys.exit("error: provide --tests (functional check) or --cases (stdin/stdout JSON)")

    sys.stderr.write(WARNING)
    r = Engine(_backend(a.backend), k=a.k).solve(task, verifier)

    status = "VERIFIED" if r.verified else "UNVERIFIED (no candidate passed)"
    sys.stderr.write(
        f"# litmus: {status}  [stage={r.stage}, model={r.model}, attempts={r.attempts}]\n\n")
    if r.verified:
        print(r.answer)            # only verified code reaches stdout
    else:
        # Don't pipe an unverified, possibly-broken answer to stdout; keep it on stderr.
        sys.stderr.write((r.answer or "(no code extracted)") + "\n")
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
    s.add_argument("--tests", help="functional test file: the body of check(candidate)")
    s.add_argument("--entry-point", help="function name the tests call (default: solve)")
    s.add_argument("--cases", help='JSON file: [{"input": "...", "output": "..."}] stdin/stdout cases')
    s.add_argument("--backend", default="openrouter", choices=["openrouter", "ollama"])
    s.add_argument("--k", type=int, default=4, help="samples per model (best-of-k)")
    s.set_defaults(func=cmd_solve)

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
