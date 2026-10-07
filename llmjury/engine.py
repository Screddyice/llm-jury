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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
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
    analyst_model: str | None = None
    analyst_summary: dict | None = None
    routing: dict | None = None


def sample_counts(k, frontier_k=None):
    def validate(value):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("sample counts must be positive integers")
        return value

    local_samples = validate(k)
    frontier_samples = local_samples if frontier_k is None else validate(frontier_k)
    return local_samples, frontier_samples


def frontier_sample_count(k, frontier_k=None, backend_name="", defaults=None):
    """Resolve the actual provider's default unless a shared budget was supplied."""
    samples = frontier_k
    if samples is None:
        samples = k if defaults is None else defaults.get(backend_name, k)
    if samples is None:
        raise ValueError("sample counts must be positive integers")
    return sample_counts(k, samples)[1]


class Engine:
    def __init__(self, backend, panel=None, best=None, prompt_template=CODE_PROMPT,
                 k=4, max_tokens=4000, temperature=0.7, frontier=None, frontier_backend=None,
                 route=None, frontier_route=None, workers=None, frontier_max_tokens=None,
                 use_panel=True, frontier_k=None, frontier_defaults=None,
                 local_scheduler=None, rebalance=True, analyst_model=None,
                 analyst_backend=None, analyst_max_tokens=1200,
                 pathway=None, routing=None):
        self.backend = backend
        b, p = default_panel(backend.name)
        self.best = best or b
        self.panel = panel or p
        self.prompt_template = prompt_template
        self.k, self.frontier_k = sample_counts(k, frontier_k)
        # Provider defaults apply only when the caller did not set a shared budget.
        self.frontier_defaults = dict(frontier_defaults or {}) if frontier_k is None else {}
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
        # Frontier overrides stay separate so a rescue model name cannot accidentally
        # reroute a same-named local panelist.
        self.frontier_route = frontier_route or {}
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
        self.local_scheduler = local_scheduler
        self.rebalance = rebalance
        self.analyst_model = analyst_model
        self.analyst_backend = analyst_backend or backend
        self.analyst_max_tokens = analyst_max_tokens
        if pathway not in (None, "single", "council", "analyst"):
            raise ValueError("unknown jury pathway")
        self.pathway = pathway
        self.routing = routing or {}

    def _result(self, answer, raw, verified, model, stage, attempts,
                analyst_model=None, analyst_summary=None):
        return Result(answer, raw, verified, model, stage, attempts,
                      analyst_model, analyst_summary, self.routing)

    def _submit(self, ex, pairs, prompt, max_tokens=None, samples=None):
        """Queue stage samples for each (model, backend); return {future: model}.

        Backends that expose `submit` (all the built-ins) give one future per
        sample, so decoding interleaves across models and samples. A duck-typed
        backend that only has `complete` gets one future wrapping its whole
        batch — same result, just coarser overlap.
        """
        mt = max_tokens or self.max_tokens
        n = self.k if samples is None else samples
        futs = {}
        for index in range(n):
            for model, backend in pairs:
                if hasattr(backend, "submit_sample"):
                    future = backend.submit_sample(ex, model, prompt, index,
                                                   self.temperature, mt)
                    futs[future] = model
                elif index == 0:
                    # Preserve compatibility with third-party batch backends.
                    if hasattr(backend, "submit"):
                        for future in backend.submit(ex, model, prompt, n=n,
                                                     temperature=self.temperature,
                                                     max_tokens=mt):
                            futs[future] = model
                    else:
                        futs[ex.submit(backend.complete, model, prompt, n,
                                       self.temperature, mt)] = model
        return futs

    def solve(self, task, verifier, escalate=True):
        """Solve a task; returns the first verified Result the ladder produces.

        Without a local scheduler, samples still decoding are abandoned — their
        threads finish (and are discarded) in the background. The CLI exits the
        process right after printing, which closes those connections and lets
        the backend cancel the leftover decodes.
        Scheduled stages retain their model reservation until running samples
        finish, so another task cannot reuse memory still occupied by a decode.
        """
        prompt = self.prompt_template.format(task=task)
        seen = []       # every completed (model, text): attempt count + fallback pool

        def backend_for(m):
            return self.route.get(m, self.backend)

        def frontier_backend_for(m):
            return self.frontier_route.get(m, self.frontier_backend)

        def select_with_analyst(completed, stage, attempts, analyst):
            if len(completed) > 1:
                from .analysis import parse, prompt as analyst_prompt
                if self.local_scheduler and getattr(analyst, "name", None) == "ollama":
                    with self.local_scheduler.reserve([self.analyst_model]) as admitted:
                        if admitted is None:
                            return next((self._result(extract_code(text), text, True, model, stage,
                                                      attempts, self.analyst_model, None)
                                         for model, text, passed in completed if passed), None)
                        self.routing.setdefault("admitted_models", []).append([admitted])
                        raw = analyst.complete(
                            self.analyst_model, analyst_prompt(task, completed), n=1,
                            temperature=0, max_tokens=self.analyst_max_tokens)[0]
                else:
                    raw = analyst.complete(
                        self.analyst_model, analyst_prompt(task, completed), n=1,
                        temperature=0, max_tokens=self.analyst_max_tokens)[0]
                order, summary = parse(raw, len(completed))
                for index in order:
                    model, text, passed = completed[index]
                    if passed:
                        return self._result(extract_code(text), text, True, model, stage,
                                            attempts, self.analyst_model, summary)
            else:
                for model, text, passed in completed:
                    if passed:
                        return self._result(extract_code(text), text, True, model, stage,
                                            attempts, self.analyst_model, None)
            return None

        def run_stage(ex, pairs, stage, max_tokens=None, samples=None, analyst=None,
                      defer_verdict=False):
            if self.local_scheduler:
                self.local_scheduler.require_available()
            futures = self._submit(ex, pairs, prompt, max_tokens, samples)
            completed = []
            try:
                for fut, model in _in_completion_order(futures):
                    out = fut.result()
                    for text in ([out] if isinstance(out, str) else out):
                        seen.append((model, text))
                        passed = verifier.verify(text)
                        completed.append((model, text, passed))
                        if analyst is None and not defer_verdict and passed:
                            return self._result(extract_code(text), text, True, model, stage, len(seen))
            finally:
                if self.local_scheduler:
                    # Keep the reservation until running samples finish. A winning
                    # candidate cannot release memory still used by other decodes.
                    for future in futures:
                        future.cancel()
                    wait(futures)
            if defer_verdict:
                return completed, len(seen)
            if analyst is not None:
                return select_with_analyst(completed, stage, len(seen), analyst)
            return None

        def scheduled_local(ex):
            remaining = list(dict.fromkeys([self.best, *self.panel]))
            selected = None
            legacy = self.pathway is None
            if self.pathway in ("council", "analyst"):
                r = None
            elif backend_for(self.best).name != "ollama":
                selected = self.best
                r = run_stage(ex, [(selected, backend_for(selected))], "single")
            else:
                choices = [m for m in remaining if backend_for(m).name == "ollama"]
                if not self.rebalance:
                    choices = [self.best]
                with self.local_scheduler.reserve(choices, preferred=self.best) as selected:
                    if selected is not None:
                        self.routing.setdefault("admitted_models", []).append([selected])
                    r = (run_stage(ex, [(selected, backend_for(selected))], "single")
                         if selected is not None else None)
            if self.pathway == "single" and (r or not escalate):
                return r
            if self.pathway == "single":
                if selected is not None:
                    remaining.remove(selected)
                return None
            if legacy and r:
                return r
            if selected is not None:
                remaining.remove(selected)
            elif not self.rebalance:
                # A pinned first model cannot be replaced by another local
                # lane merely because it is busy or refused. Routed council
                # members and the configured frontier retain their own policy.
                remaining = [m for m in remaining if backend_for(m).name != "ollama"]

            # Match the cloud council stage: complementary panelists share one
            # prompt and oracle, with first verified completion winning. Local
            # admission bounds concurrent residency; explicit routed members
            # still participate when no local model fits.
            while remaining:
                local = [m for m in remaining if backend_for(m).name == "ollama"]
                routed = [m for m in remaining if backend_for(m).name != "ollama"]
                reserve_candidates = list(local)
                deferred = None
                with self.local_scheduler.reserve_many(reserve_candidates) as admitted:
                    models = admitted + routed
                    if admitted:
                        self.routing.setdefault("admitted_models", []).append(list(admitted))
                    if routed:
                        self.routing.setdefault("routed_models", []).extend(routed)
                    generated = [m for m in models if m in local or m in routed]
                    analyst = (self.analyst_backend if self.analyst_model and
                               self.pathway in (None, "analyst") and len(generated) > 1 else None)
                    if generated:
                        deferred = run_stage(
                            ex, [(m, backend_for(m)) for m in generated], "council",
                            defer_verdict=analyst is not None)
                        if analyst is None and deferred:
                            return deferred
                if analyst is not None and deferred:
                    r = select_with_analyst(deferred[0], "council", deferred[1], analyst)
                    if r:
                        return r
                if not admitted:
                    break
                remaining = [m for m in remaining if m not in models]
            return None

        ex = ThreadPoolExecutor(max_workers=self.workers)
        try:
            if self.use_panel and self.local_scheduler:
                r = scheduled_local(ex)
                if r:
                    return r
            elif self.use_panel:
                if self.pathway in (None, "single"):
                    r = run_stage(ex, [(self.best, backend_for(self.best))], "single")
                    if r:
                        return r
                if self.pathway is None:
                    pairs = [(m, backend_for(m)) for m in self.panel if m != self.best]
                    r = run_stage(ex, pairs, "council",
                                  analyst=(self.analyst_backend
                                           if self.analyst_model else None)) if pairs else None
                    if r:
                        return r
                elif self.pathway in ("council", "analyst"):
                    pairs = [(m, backend_for(m)) for m in self.panel]
                    r = run_stage(ex, pairs, "council",
                                  analyst=(self.analyst_backend
                                           if self.pathway == "analyst" else None))
                    if r:
                        return r

            # Stage 3: opt-in frontier escalation — one strong (cloud) model, only when
            # the local council couldn't verify. This is what lets a local-first setup
            # match a cloud fusion's accuracy while paying for a frontier call on the
            # hard minority, not on every problem.
            if escalate and self.frontier:
                for model in self.frontier:
                    provider = frontier_backend_for(model)
                    samples = frontier_sample_count(
                        self.frontier_k, backend_name=provider.name,
                        defaults=self.frontier_defaults)
                    r = run_stage(ex, [(model, provider)], "frontier",
                                  self.frontier_max_tokens, samples)
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
            return self._result(best[2], best[1], False, best[0], "unverified", len(seen))
        first_model, first_text = seen[0] if seen else (None, None)
        return self._result(None, first_text, False, first_model, "unverified", len(seen))


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
