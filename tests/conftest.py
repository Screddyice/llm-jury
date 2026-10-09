"""Shared test isolation for local model admission."""

import pytest

from llmjury import memguard


@pytest.fixture(autouse=True)
def _isolate_router_state(tmp_path, monkeypatch):
    monkeypatch.setattr(memguard, "COMPUTE_LEASE_DIR", str(tmp_path / "compute-leases"))

@pytest.fixture(autouse=True)
def _ledger_stays_out_of_the_real_home(tmp_path_factory, monkeypatch):
    """No test may append to the developer's real spend ledger.

    `~/.llmjury/spend.jsonl` is a durable record a weekly cost report reads.
    Once the Codex and Claude backends began recording subscription-served
    escalations, every existing test that drove them through a fake runner
    started writing real-looking rows into it -- four landed before this was
    caught, with models `gpt-test` and `opus`, which is precisely the kind of
    invented row a cost report cannot distinguish from a real one.

    Same category as redirecting any other side effect that leaves the process.
    """
    ledger = tmp_path_factory.mktemp("llmjury-ledger") / "spend.jsonl"
    monkeypatch.setenv("LLMJURY_SPEND_LEDGER", str(ledger))
    return ledger
