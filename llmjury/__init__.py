"""LLM-Jury — a local engine that returns answers it can prove are right.

Don't vote, verify: sample a diverse small-model council, run a real verifier,
return the answer that provably passes. Fully local (Ollama) or cloud (OpenRouter).
"""
from .engine import Engine, Result, solve
from .verifiers import FunctionalCodeVerifier, StdioCodeVerifier, extract_code
from .backends import OpenRouterBackend, OllamaBackend, DemoBackend
from .panels import CLOUD_PANEL, CLOUD_BEST, LOCAL_PANEL, LOCAL_BEST

__version__ = "0.1.0"
__all__ = [
    "Engine", "Result", "solve",
    "FunctionalCodeVerifier", "StdioCodeVerifier", "extract_code",
    "OpenRouterBackend", "OllamaBackend", "DemoBackend",
    "CLOUD_PANEL", "CLOUD_BEST", "LOCAL_PANEL", "LOCAL_BEST",
]
