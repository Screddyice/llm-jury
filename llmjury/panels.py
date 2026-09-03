"""Default cross-lineage panels.

Diversity is the point: models from different labs make *different* mistakes, and a
verifier can harvest that. A panel of near-identical models gains you nothing.
"""
import os

# Cloud (OpenRouter slugs) — what the published benchmarks were run with.
CLOUD_PANEL = ["microsoft/phi-4", "google/gemma-3-12b-it", "meta-llama/llama-3.1-8b-instruct"]
CLOUD_BEST = "microsoft/phi-4"

# Local (Ollama tags) — pull with:
#   ollama pull gemma3:12b && ollama pull llama3.1:8b && ollama pull phi4-mini:3.8b
#
# This panel mirrors CLOUD_PANEL's lineages — Google / Meta / Microsoft, the same three
# labs the published benchmarks were run with. That mirror is deliberate: the local
# council is meant to be the benchmarked council, run locally, so the measured numbers
# describe something a user can actually reproduce off-cloud.
#
# The mirror is no longer exact, and cannot be. CLOUD_BEST is phi-4; locally phi4 is
# 12.7 GiB on its own, and the benchmarked trio (phi4 + gemma3:12b + llama3.1:8b)
# projects 31.7 GiB at 2 slots against a 25.2 GiB budget on a 36 GiB host. There is no
# num_ctx or slot count that fits it — the weights alone are ~28 GiB. So phi-4 is
# substituted by phi4-mini:3.8b, its own family, which keeps all three lineages rather
# than swapping a whole lab out.
#
#   36 GiB host, budget 25.2 GiB (70% of RAM)
#     2 slots -> 23.4 GiB  fits
#     4 slots -> 27.3 GiB  refused by memguard, with a hint
#
# REQUIRES OLLAMA_NUM_PARALLEL=2. KV is charged num_ctx x slots, so parallelism is part
# of a panel's spec rather than ambient config. Depending on a non-stock slot count is
# safe only because the preflight runs before any model loads: an untuned 4-slot host
# gets an actionable refusal naming a smaller panel, never a kernel panic.
# test_memguard_* pins both halves of that contract.
#
# History, because this line has been wrong in both directions. The benchmarked trio
# was the default and projects 35.6 GiB at 4 slots, essentially the whole machine. It
# survived on timing rather than headroom, since the three were rarely all resident at
# once — meaning this host was never really running a concurrent council. On 2026-07-31
# two unrelated changes consumed the remaining slack (a diff-review hook pinned
# resident, and a corrupt 196 GB vector index mmap'd on every memory sync) and the host
# kernel-panicked. The immediate fix over-corrected to
# llama3.1:8b + phi4-mini:3.8b + granite4.1:3b at 19.7 GiB, dropping the only 12B model.
#
# Panel strength is not free but it is not decisive either: llm-jury verifies rather
# than votes, so a weaker panelist escalates to the frontier ladder more often instead
# of returning a worse answer. That trade costs escalation spend, not correctness. For
# exact benchmark fidelity use --backend openrouter, which runs CLOUD_PANEL unchanged.
LOCAL_PANEL = ["gemma3:12b", "llama3.1:8b", "phi4-mini:3.8b"]
LOCAL_BEST = "gemma3:12b"

# Codex is a single frontier provider, not a diverse council by itself. Use it as
# the final tier after the local council, or sample it best-of-k directly.
CODEX_BEST = os.environ.get("LLMJURY_CODEX_MODEL", "gpt-5.6-sol")
CODEX_PANEL = [CODEX_BEST]

# Claude Code is also a single authenticated frontier provider. It is used as
# the final rescue for `--frontier auto` when the caller is a Claude Code session.
CLAUDE_BEST = os.environ.get("LLMJURY_CLAUDE_MODEL", "opus")

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
    return CLOUD_BEST, CLOUD_PANEL
