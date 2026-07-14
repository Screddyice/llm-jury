# LLM-Jury

LLM-Jury generates candidate Python solutions and returns only code that passes an
independent verifier. Provider diversity is useful only when a real oracle exists.

## Verification

```bash
python3 tests/test_llmjury.py
python3 -m llmjury.cli demo
git diff --check
```

Tests must remain offline by default. Mock provider processes and HTTP calls in unit tests.

## Provider Boundaries

- Ollama is the local/private council backend.
- Codex CLI is an authenticated OpenAI provider and may be used directly or as the
  frontier tier. It must run ephemeral and read-only for candidate generation.
- OpenRouter is a model-agnostic cloud provider. Do not couple it to Anthropic or any
  single model family.
- The recommended hybrid route is `--backend ollama --frontier auto`: local council
  first, then the ordered open-weight OpenRouter ladder only when verification fails.
- The verifier, not the provider, decides which candidate is accepted.
- Never expose auth files, API keys, or the parent process environment to generated code.
