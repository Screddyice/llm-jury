"""Spend ledger: llm-jury records what its metered backends cost.

The consumer is backdoor's weekly model-economics report, which reads this
ledger instead of OPENROUTER_API_KEY. Recording must never be able to fail a
solve — a billing side-effect that breaks the answer is worse than no ledger.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from llmjury import spend  # noqa: E402


def test_record_appends_one_line_per_call(tmp_path, monkeypatch):
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))

    spend.record("openrouter", "deepseek/deepseek-v4-flash",
                 {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0025})
    spend.record("openrouter", "anthropic/claude-opus-5",
                 {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.5})

    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["backend"] == "openrouter"
    assert first["model"] == "deepseek/deepseek-v4-flash"
    assert first["prompt_tokens"] == 100
    assert first["completion_tokens"] == 20
    assert first["cost_usd"] == 0.0025
    assert first["ts"].endswith("+00:00")


def test_record_is_a_no_op_without_usage(tmp_path, monkeypatch):
    """No usage block means nothing measured. An invented zero is a false claim."""
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))
    spend.record("openrouter", "m", None)
    spend.record("openrouter", "m", {})
    assert not ledger.exists()


def test_record_never_raises_when_the_ledger_cannot_be_written(tmp_path, monkeypatch):
    """A billing side-effect must not be able to fail the solve it was measuring."""
    unwritable = tmp_path / "nodir" / "deeper"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(unwritable / "spend.jsonl"))
    unwritable.parent.mkdir()
    unwritable.parent.chmod(0o500)
    try:
        spend.record("openrouter", "m", {"prompt_tokens": 1, "cost": 0.1})
    finally:
        unwritable.parent.chmod(0o700)


def test_record_defaults_cost_to_zero_when_the_provider_omits_it(tmp_path, monkeypatch):
    """Tokens are still worth recording when cost accounting is off."""
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))
    spend.record("openrouter", "m", {"prompt_tokens": 7, "completion_tokens": 3})
    rec = json.loads(ledger.read_text().splitlines()[0])
    assert rec["cost_usd"] == 0.0
    assert rec["prompt_tokens"] == 7


def test_openrouter_backend_records_the_call_it_was_billed_for(tmp_path, monkeypatch):
    """The wiring, not just the module: a successful call lands in the ledger."""
    import io
    import json as _json
    import urllib.request

    from llmjury import backends

    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    payload = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "cost": 0.0031},
    }

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp(_json.dumps(payload).encode()))

    backend = backends.OpenRouterBackend()
    assert backend._one("deepseek/deepseek-v4-flash", "p", 0.2, 64) == "answer"

    rec = _json.loads(ledger.read_text().splitlines()[0])
    assert rec["model"] == "deepseek/deepseek-v4-flash"
    assert rec["cost_usd"] == 0.0031
    assert rec["prompt_tokens"] == 42
