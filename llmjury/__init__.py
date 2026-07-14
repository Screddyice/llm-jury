"""LLM-Jury — a local engine that returns answers it can prove are right.

Don't vote, verify: sample a diverse small-model council, run a real verifier,
return the answer that provably passes. Use Codex, local Ollama, or OpenRouter.
"""
from .engine import Engine, Result, solve
from .verifiers import FunctionalCodeVerifier, StdioCodeVerifier, extract_code
from .backends import CodexBackend, OpenRouterBackend, OllamaBackend, DemoBackend
from .panels import CODEX_PANEL, CODEX_BEST, CLOUD_PANEL, CLOUD_BEST, LOCAL_PANEL, LOCAL_BEST

__version__ = "0.1.0"
__all__ = [
    "Engine", "Result", "solve",
    "FunctionalCodeVerifier", "StdioCodeVerifier", "extract_code",
    "CodexBackend", "OpenRouterBackend", "OllamaBackend", "DemoBackend",
    "CODEX_PANEL", "CODEX_BEST",
    "CLOUD_PANEL", "CLOUD_BEST", "LOCAL_PANEL", "LOCAL_BEST",
]
