"""Default cross-lineage panels.

Diversity is the point: models from different labs make *different* mistakes, and a
verifier can harvest that. A panel of near-identical models gains you nothing.
"""
import os

# Cloud (OpenRouter slugs) — what the published benchmarks were run with.
CLOUD_PANEL = ["microsoft/phi-4", "google/gemma-3-12b-it", "meta-llama/llama-3.1-8b-instruct"]
CLOUD_BEST = "microsoft/phi-4"

# Local (Ollama tags) — pull with:
#   ollama pull gemma3:12b && ollama pull llama3.1:8b && ollama pull granite4.1:3b
#
# Cross-lineage (Google / Meta / IBM). REQUIRES OLLAMA_NUM_PARALLEL=2, which is a
# documented, enforced requirement rather than an assumption — see below.
#
#   36 GiB host, budget 25.2 GiB (70% of RAM)
#     2 slots -> 23.0 GiB  fits
#     4 slots -> 26.8 GiB  refused by memguard, with a hint
#
# KV is charged num_ctx x slots, so parallelism multiplies memory for every model on
# the server. Ollama ships 4 slots; this panel needs 2. That requirement is safe to
# take because it degrades into a *refusal*, not a crash: memguard.check() runs before
# any model loads, so an untuned host gets an actionable error naming the smaller
# panel, never a kernel panic. test_memguard_* pins both halves of that contract.
#
# History, because this line has been wrong in both directions. It was
# phi4 + gemma3:12b + llama3.1:8b, which projects 35.6 GiB at 4 slots — essentially
# the whole machine. It survived on timing rather than headroom, since the three were
# rarely all resident at once. On 2026-07-31 two unrelated changes consumed the
# remaining slack (a diff-review hook pinned resident, and a second CLI firing that
# same hook) and the host kernel-panicked. The immediate fix then over-corrected to
# llama3.1:8b + phi4-mini:3.8b + granite4.1:3b at 19.7 GiB, giving up the only 12B
# panelist and leaving ~5 GiB of budget unused.
#
# phi4 is what cannot come back: 12.7 GiB alone, so no 3-model panel holding it fits
# at any parallelism worth running.
#
# Panel strength is not free but it is not decisive either: llm-jury verifies rather
# than votes, so a weaker panelist escalates to the frontier ladder more often instead
# of returning a worse answer. That trade costs escalation spend, not correctness —
# which is why fitting the host is the constraint and maximising within it is the
# objective. Per-run overrides go through `--models`, gated by the same preflight.
LOCAL_PANEL = ["gemma3:12b", "llama3.1:8b", "granite4.1:3b"]
LOCAL_BEST = "gemma3:12b"

# Codex is a single frontier provider, not a diverse council by itself. Use it as
# the final tier after the local council, or sample it best-of-k directly.
CODEX_BEST = os.environ.get("LLMJURY_CODEX_MODEL", "gpt-5.6-sol")
CODEX_PANEL = [CODEX_BEST]

# Grok is the same shape as Codex: a single authenticated frontier provider, not a
# diverse council. Its value is that the Grok CLI's own session auth covers it, so
# inside a Grok session this is the escalation tier that spends no OpenRouter credit.
GROK_BEST = os.environ.get("LLMJURY_GROK_MODEL", "grok-4.5")
GROK_PANEL = [GROK_BEST]

# Verifier-gated OpenRouter escalation. These are deliberately ordered by role,
# not fanned out on every task: Flash is the low-cost first cloud recovery, then
# Pro is the benchmark-backed accuracy tier for the genuinely hard remainder.
# Both are open-weight models; the local council remains the default first line.
OPEN_SOURCE_FRONTIER = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
]

# The last-resort tier for the hard tail nothing cheaper could verify. This is a
# proprietary model and ~35x the per-token cost of the open-weight ladder above,
# which is why it is last: every earlier tier must have failed the same oracle
# before a single token is spent here.
TOP_FRONTIER = os.environ.get("LLMJURY_TOP_FRONTIER", "anthropic/claude-opus-5")

# What `--frontier auto` resolves to: cheap open-weight recovery first, then one
# proprietary top tier. Use `--frontier open` to keep escalation open-weight only.
AUTO_FRONTIER = OPEN_SOURCE_FRONTIER + [TOP_FRONTIER]

# Shorthands accepted by `--frontier`, so callers don't have to memorise slugs.
# Anything not listed here is passed through to the provider verbatim.
FRONTIER_ALIASES = {
    "auto": AUTO_FRONTIER,
    "open": OPEN_SOURCE_FRONTIER,
    "opus": ["anthropic/claude-opus-5"],
    "fable": ["anthropic/claude-fable-5"],
}


def default_panel(backend_name):
    if backend_name == "ollama":
        return LOCAL_BEST, LOCAL_PANEL
    if backend_name == "demo":
        return "demo-weak", ["demo-weak", "demo-council"]
    if backend_name == "codex":
        return CODEX_BEST, CODEX_PANEL
    if backend_name == "grok":
        return GROK_BEST, GROK_PANEL
    return CLOUD_BEST, CLOUD_PANEL
