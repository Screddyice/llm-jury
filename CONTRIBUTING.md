# Contributing to LLM-Jury

Thanks for your interest. LLM-Jury is small, stdlib-only, and easy to hack on.

## Dev setup

```bash
git clone https://github.com/ajsai47/llm-jury
cd llmjury
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

There are **no runtime dependencies** — keep it that way unless there's a strong reason.

## Running the tests

```bash
python tests/test_llmjury.py        # offline — no API keys, no network
```

CI runs these on every push and PR (`.github/workflows/test.yml`).

## Ground rules

- **Stdlib only** for the package runtime. Dev/test tooling can use extras.
- **The verifier is the product.** Changes to `llmjury/verifiers.py` need a test, and must
  preserve the execution sandbox (scrubbed env, isolated temp cwd, resource limits).
- New verifiers/backends should follow the existing interfaces (`Verifier.verify`,
  `Backend.complete`).
- Run the tests before opening a PR.

## Security

LLM-Jury executes model-generated code to verify it. If you find a way the sandbox leaks (reads
host env vars, escapes the temp dir, evades the resource limits), please report it privately
rather than opening a public issue first.
