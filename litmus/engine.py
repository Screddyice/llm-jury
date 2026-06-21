"""The Litmus engine: generate -> verify -> select -> escalate.

Default to the single best model + best-of-k (fast, fits memory). Escalate to the
full diverse council only when nothing verifies — that's the regime where the
council actually pays, and it keeps the common case cheap and memory-light.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .verifiers import extract_code
from .panels import default_panel

CODE_PROMPT = (
    "Solve this problem. Return ONE complete, self-contained Python solution "
    "(including any imports and the full function or program) in a single ```python "
    "code block. No explanation outside the block.\n\n{task}"
)


@dataclass
class Result:
    answer: str | None      # extracted code, or None
    raw: str | None         # full model text of the chosen sample
    verified: bool          # did it pass the verifier?
    model: str | None       # which model produced the chosen sample
    stage: str              # "single", "council", "frontier", or "unverified"
    attempts: int           # total samples generated


class Engine:
    def __init__(self, backend, panel=None, best=None, prompt_template=CODE_PROMPT,
                 k=4, max_tokens=4000, temperature=0.7, frontier=None, frontier_backend=None):
        self.backend = backend
        b, p = default_panel(backend.name)
        self.best = best or b
        self.panel = panel or p
        self.prompt_template = prompt_template
        self.k = k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.frontier = frontier                          # opt-in strong model for the last tier
        self.frontier_backend = frontier_backend or backend

    def _gen(self, model, prompt, n):
        texts = self.backend.complete(model, prompt, n=n,
                                      temperature=self.temperature, max_tokens=self.max_tokens)
        return [(model, t) for t in texts]

    def solve(self, task, verifier, escalate=True):
        prompt = self.prompt_template.format(task=task)
        attempts = 0

        # Stage 1: single best model, best-of-k.
        cand = self._gen(self.best, prompt, self.k)
        attempts += len(cand)
        for model, text in cand:
            if verifier.verify(text):
                return Result(extract_code(text), text, True, model, "single", attempts)

        # Stage 2: escalate to the diverse council (skip best, already tried).
        if escalate:
            for m in [m for m in self.panel if m != self.best]:
                for model, text in self._gen(m, prompt, self.k):
                    attempts += 1
                    if verifier.verify(text):
                        return Result(extract_code(text), text, True, model, "council", attempts)

        # Stage 3: opt-in frontier escalation — one strong (cloud) model, only when the local
        # council couldn't verify. This is what lets a local-first setup match a cloud fusion's
        # accuracy while paying for a frontier call on the hard minority, not on every problem.
        if escalate and self.frontier:
            for text in self.frontier_backend.complete(
                    self.frontier, prompt, n=self.k,
                    temperature=self.temperature, max_tokens=self.max_tokens):
                attempts += 1
                if verifier.verify(text):
                    return Result(extract_code(text), text, True, self.frontier, "frontier", attempts)

        # Nothing verified — return the most complete best-effort (longest extractable
        # code), flagged unverified, rather than blindly the first sample.
        best = None
        for model, text in cand:
            code = extract_code(text)
            if code and (best is None or len(code) > len(best[2])):
                best = (model, text, code)
        if best:
            return Result(best[2], best[1], False, best[0], "unverified", attempts)
        first_model, first_text = cand[0] if cand else (None, None)
        return Result(None, first_text, False, first_model, "unverified", attempts)


def solve(task, verifier, backend=None, **kw):
    """Litmus in one call. Defaults to the OpenRouter backend; pass backend= for Ollama."""
    if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("LITMUS_ALLOW_ROOT") != "1":
        raise PermissionError(
            "Litmus refuses to run as root — it executes model-generated code. "
            "Set LITMUS_ALLOW_ROOT=1 to override (not recommended).")
    if backend is None:
        from .backends import OpenRouterBackend
        backend = OpenRouterBackend(cache_path="~/.litmus/cache.jsonl")
    return Engine(backend, **kw).solve(task, verifier)
