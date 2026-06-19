"""Reproduce the published benchmark numbers with the Litmus engine.

Runs the council on a bundled benchmark slice and reports how many problems the
single best model solves (verified best-of-k) versus what the diverse council adds
on escalation — the exact "council adds coverage" story from the write-up. Uses the
response cache, so reruns are near-free.

    litmus reproduce humaneval        # full bundled slice (25)
    litmus reproduce lcb --n 5        # quick 5-problem check
    litmus reproduce lcb --backend ollama   # run it locally
"""
import os
import json

from ..engine import Engine
from ..verifiers import FunctionalCodeVerifier, StdioCodeVerifier

DATA = os.path.join(os.path.dirname(__file__), "data")
CACHE = "~/.litmus/cache.jsonl"

REFERENCE = {
    "humaneval": "Published full-run (larger sample): council 97.6% = frontier (DeepSeek) one-shot 97.6%  [HumanEval+, n=82]",
    "lcb": "Published full-run (larger sample): council 75.6% > frontier (DeepSeek) one-shot 62.2%  [LiveCodeBench hard, n=45]",
}


def _backend(name):
    if name == "ollama":
        from ..backends import OllamaBackend
        return OllamaBackend(cache_path=CACHE)
    from ..backends import OpenRouterBackend
    return OpenRouterBackend(cache_path=CACHE)


def _verifier(which, p):
    if which == "humaneval":
        return FunctionalCodeVerifier(p["test"], p["entry_point"], header=p.get("header", ""))
    return StdioCodeVerifier(p["cases"])


def run(which, backend="openrouter", n=None, k=4):
    with open(os.path.join(DATA, f"{which}_slice.json"), encoding="utf-8") as fh:
        probs = json.load(fh)
    if n:
        probs = probs[:n]
    eng = Engine(_backend(backend), k=k)

    single = council = 0
    print(f"Reproducing {which} on {len(probs)} problems (backend={backend}, k={k})...\n", flush=True)
    for i, p in enumerate(probs):
        r = eng.solve(p["task"], _verifier(which, p))
        if r.verified and r.stage == "single":
            single += 1
        elif r.verified and r.stage == "council":
            council += 1
        mark = "PASS" if r.verified else "----"
        print(f"  [{i+1:2}/{len(probs)}] {p['id']:16} {mark}  ({r.stage:10} {r.model})", flush=True)

    nn = len(probs)
    total = single + council
    print(f"\n== litmus reproduce: {which}  (n={nn}, backend={backend}, k={k}) ==")
    print(f"  single best model + verified best-of-{k}:   {single}/{nn} = {single/nn:.1%}")
    print(f"  + diverse council (escalation):            +{council}  ->  {total}/{nn} = {total/nn:.1%}")
    print(f"\n  This run reproduces the METHOD on a bundled {nn}-problem slice.")
    print(f"  {REFERENCE[which]}")
    return {"which": which, "n": nn, "single": single, "council_added": council, "total": total}
