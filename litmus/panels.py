"""Default cross-lineage panels.

Diversity is the point: models from different labs make *different* mistakes, and a
verifier can harvest that. A panel of near-identical models gains you nothing.
"""

# Cloud (OpenRouter slugs) — what the published benchmarks were run with.
CLOUD_PANEL = ["microsoft/phi-4", "google/gemma-3-12b-it", "meta-llama/llama-3.1-8b-instruct"]
CLOUD_BEST = "microsoft/phi-4"

# Local (Ollama tags) — pull with: ollama pull phi4 && ollama pull gemma3:12b && ollama pull llama3.1:8b
LOCAL_PANEL = ["phi4", "gemma3:12b", "llama3.1:8b"]
LOCAL_BEST = "phi4"


def default_panel(backend_name):
    if backend_name == "ollama":
        return LOCAL_BEST, LOCAL_PANEL
    return CLOUD_BEST, CLOUD_PANEL
