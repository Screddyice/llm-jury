"""The LLM-Jury engine: generate -> verify -> select -> escalate.

Default to the single best model + best-of-k (fast, fits memory). Escalate to the
full diverse council only when nothing verifies — that's the regime where the
council actually pays, and it keeps the common case cheap and memory-light.

Within a stage everything is concurrent: all of the stage's samples (across all
of its models) are queued at once, and verification runs in completion order —
the sandbox checks finished samples while the backend keeps decoding the rest,
and the first verified sample wins the stage. Across stages the escalation
ladder stays strictly sequential; that's the cost model.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    attempts: int           # samples that finished generating before the verdict


class Engine:
    def __init__(self, backend, panel=None, best=None, prompt_template=CODE_PROMPT,
                 k=4, max_tokens=4000, temperature=0.7, frontier=None, frontier_backend=None,
                 route=None, workers=None, frontier_max_tokens=None, use_panel=True):
        self.backend = backend
        b, p = default_panel(backend.name)
        self.best = best or b
        self.panel = panel or p
        self.prompt_template = prompt_template
        self.k = k
        self.max_tokens = max_tokens
        self.temperature = temperature
        # One model or an ordered verifier-gated ladder. Each model is attempted
        # only after every cheaper/local tier failed the same oracle.
        self.frontier = ([frontier] if isinstance(frontier, str) else list(frontier or []))
        self.frontier_backend = frontier_backend or backend
        # Reasoning models spend part of `max_tokens` on private thinking before
        # emitting a single character of code, and providers count those tokens
        # against the same budget. The frontier tier runs only on the hard tail —
        # exactly where thinking is longest — so a budget sized for the council
        # truncates the answer precisely when it matters most. Give it headroom.
        self.frontier_max_tokens = frontier_max_tokens or max(max_tokens, 8000)
        # Per-model backend overrides: a panelist named here generates through its own
        # backend instead of the shared one. Lets a custom local model — e.g. a
        # fine-tuned MLX brain on an OpenAI-compatible endpoint — sit in the council
        # alongside the default panel. Empty by default, so the common path is unchanged.
        self.route = route or {}
        # Stages 1-2 (best-of-k, then the council) run the panel on `backend`.
        # `use_panel=False` skips both and starts at the frontier ladder. That is for
        # a caller which has determined the panel cannot be loaded SAFELY — see the
        # memguard preflight, where a panel that would over-commit the host is refused
        # before any model loads. The frontier tier runs on a remote provider and
        # costs this host no memory, so it stays a valid path when the panel is not.
        self.use_panel = use_panel
        # Generation threads shared by all of a solve()'s stages. The default is
        # sized so an entire council stage (every panelist x k samples) can be
        # in flight at once.
        self.workers = workers or min(16, max(4, self.k * max(1, len(self.panel))))

    def _submit(self, ex, pairs, prompt, max_tokens=None):
        """Queue k samples for every (model, backend) pair; return {future: model}.

        Backends that expose `submit` (all the built-ins) give one future per
        sample, so decoding interleaves across models and samples. A duck-typed
        backend that only has `complete` gets one future wrapping its whole
        batch — same result, just coarser overlap.
        """
        mt = max_tokens or self.max_tokens
        futs = {}
        for model, backend in pairs:
            if hasattr(backend, "submit"):
                for f in backend.submit(ex, model, prompt, n=self.k,
                                        temperature=self.temperature,
                                        max_tokens=mt):
                    futs[f] = model
            else:
                futs[ex.submit(backend.complete, model, prompt, self.k,
                               self.temperature, mt)] = model
        return futs

    def solve(self, task, verifier, escalate=True):
        """Solve a task; returns the first verified Result the ladder produces.

        On an early verified exit, samples still decoding are abandoned — their
        threads finish (and are discarded) in the background. The CLI exits the
        process right after printing, which closes those connections and lets
        the backend cancel the leftover decodes.
        """
        prompt = self.prompt_template.format(task=task)
        seen = []       # every completed (model, text): attempt count + fallback pool

        def backend_for(m):
            return self.route.get(m, self.backend)

        def run_stage(ex, pairs, stage, max_tokens=None):
            for fut, model in _in_completion_order(
                    self._submit(ex, pairs, prompt, max_tokens)):
                out = fut.result()
                for text in ([out] if isinstance(out, str) else out):
                    seen.append((model, text))
                    if verifier.verify(text):
                        return Result(extract_code(text), text, True, model, stage, len(seen))
            return None

        ex = ThreadPoolExecutor(max_workers=self.workers)
        try:
            if self.use_panel:
                # Stage 1: single best model, best-of-k.
                r = run_stage(ex, [(self.best, backend_for(self.best))], "single")
                if r:
                    return r

                # Stage 2: the rest of the diverse council, all panelists at once
                # (skip best, already tried).
                if escalate:
                    rest = [(m, backend_for(m)) for m in self.panel if m != self.best]
                    if rest:
                        r = run_stage(ex, rest, "council")
                        if r:
                            return r

            # Stage 3: opt-in frontier escalation — one strong (cloud) model, only when
            # the local council couldn't verify. This is what lets a local-first setup
            # match a cloud fusion's accuracy while paying for a frontier call on the
            # hard minority, not on every problem.
            if escalate and self.frontier:
                for model in self.frontier:
                    r = run_stage(ex, [(model, self.frontier_backend)], "frontier",
                                  self.frontier_max_tokens)
                    if r:
                        return r
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        # Nothing verified — return the most complete best-effort (longest extractable
        # code) across everything generated, flagged unverified, rather than blindly
        # the first sample.
        best = None
        for model, text in seen:
            code = extract_code(text)
            if code and (best is None or len(code) > len(best[2])):
                best = (model, text, code)
        if best:
            return Result(best[2], best[1], False, best[0], "unverified", len(seen))
        first_model, first_text = seen[0] if seen else (None, None)
        return Result(None, first_text, False, first_model, "unverified", len(seen))


def _in_completion_order(futmap):
    """Yield (future, model) pairs as generation finishes, not submission order."""
    for fut in as_completed(futmap):
        yield fut, futmap[fut]


def solve(task, verifier, backend=None, **kw):
    """LLM-Jury in one call. Defaults to the OpenRouter backend; pass backend= for Ollama."""
    if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("LLMJURY_ALLOW_ROOT") != "1":
        raise PermissionError(
            "LLM-Jury refuses to run as root — it executes model-generated code. "
            "Set LLMJURY_ALLOW_ROOT=1 to override (not recommended).")
    if backend is None:
        from .backends import OpenRouterBackend
        backend = OpenRouterBackend(cache_path="~/.llmjury/cache.jsonl")
    return Engine(backend, **kw).solve(task, verifier)
