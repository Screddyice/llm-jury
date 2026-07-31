"""Default cross-lineage panels.

Diversity is the point: models from different labs make *different* mistakes, and a
verifier can harvest that. A panel of near-identical models gains you nothing.
"""
import os

# Cloud (OpenRouter slugs) — what the published benchmarks were run with.
CLOUD_PANEL = ["microsoft/phi-4", "google/gemma-3-12b-it", "meta-llama/llama-3.1-8b-instruct"]
CLOUD_BEST = "microsoft/phi-4"

# Local (Ollama tags) — pull with: ollama pull phi4 && ollama pull gemma3:12b && ollama pull llama3.1:8b
LOCAL_PANEL = ["phi4", "gemma3:12b", "llama3.1:8b"]
LOCAL_BEST = "phi4"

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
