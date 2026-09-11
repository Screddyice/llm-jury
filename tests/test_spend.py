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


# --- subscription-served escalations: the spend that did NOT happen ---------


def test_record_subscription_logs_the_openrouter_spend_avoided(tmp_path, monkeypatch):
    """Escalating to the Codex CLI costs nothing beyond a subscription already paid for.

    That is the whole point of `--frontier-backend codex`, and it was invisible:
    no OpenRouter call is made, so nothing reached the ledger and the weekly
    report showed OpenRouter usage as zero — true, but it read as "the frontier
    ladder was never used" rather than "it ran for free".
    """
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))

    spend.record_subscription("codex", "gpt-5.6-sol", prompt="x" * 4000, completion="y" * 800)

    rec = json.loads(ledger.read_text().splitlines()[0])
    assert rec["backend"] == "codex"
    assert rec["billing"] == "subscription"
    assert rec["cost_usd"] == 0.0            # nothing was metered
    assert rec["avoided_usd"] > 0            # but something was avoided
    assert rec["estimated"] is True          # and it is an estimate, flagged


def test_subscription_records_are_separable_from_metered_ones(tmp_path, monkeypatch):
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))

    spend.record("openrouter", "deepseek/deepseek-v4-flash",
                 {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.25})
    spend.record_subscription("codex", "gpt-5.6-sol", prompt="x" * 400, completion="y" * 80)

    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    metered = [r for r in rows if r.get("billing") != "subscription"]
    free = [r for r in rows if r.get("billing") == "subscription"]
    assert len(metered) == 1 and metered[0]["cost_usd"] == 0.25
    assert len(free) == 1 and free[0]["cost_usd"] == 0.0


def test_record_subscription_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(tmp_path / "nodir" / "x.jsonl"))
    spend.record_subscription("codex", "m", prompt="a", completion="b")


def test_claude_cli_escalation_is_recorded_as_avoided_spend(tmp_path, monkeypatch):
    """The Claude rescue is the same economics as the Codex one, and was equally silent.

    Inside Claude Code, `_claude_frontier_rescue` diverts the ladder's paid
    Anthropic rung to the authenticated CLI — the rung that costs roughly 35x the
    open-weight tiers on OpenRouter. Nothing was metered, so nothing was recorded,
    so the largest avoided cost in the whole ladder was the least visible.
    """
    import types
    from llmjury import backends

    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))

    def fake_runner(cmd, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="def add(a, b):\n    return a + b\n", stderr="")

    backend = backends.ClaudeBackend(runner=fake_runner)
    out = backend._one("claude-opus-5", "write add()", 0.2, 256)
    assert "return a + b" in out

    rec = json.loads(ledger.read_text().splitlines()[0])
    assert rec["backend"] == "claude"
    assert rec["billing"] == "subscription"
    assert rec["cost_usd"] == 0.0
    assert rec["avoided_usd"] > 0
