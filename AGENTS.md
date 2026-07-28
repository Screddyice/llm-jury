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
- OpenRouter is a model-agnostic cloud provider. The backend must stay model-agnostic:
  no Anthropic-specific request shapes, response parsing, or auth paths. Model choice
  is policy, expressed as data in `llmjury.panels`, never wired into transport code.
- `--frontier auto` ends on a proprietary top tier (`TOP_FRONTIER`, Claude Opus 5) after
  the open-weight ladder. That is a deliberate default-policy choice, not a coupling:
  it is one env-overridable constant (`LLMJURY_TOP_FRONTIER`), and `--frontier open`
  keeps escalation open-weight only. Keep both paths working.
- Cost ordering is a correctness property of the ladder, not a preference. The
  proprietary tier costs ~35× the open-weight tiers, so it must stay last and stay
  verifier-gated; a test asserts that ordering. Never fan out across tiers.
- The recommended hybrid route is `--backend ollama --frontier auto`: local council
  first, then the ordered OpenRouter ladder only when verification fails.
- The verifier, not the provider, decides which candidate is accepted.
- Never expose auth files, API keys, or the parent process environment to generated code.
