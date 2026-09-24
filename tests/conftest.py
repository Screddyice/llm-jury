"""Shared test isolation.

`memguard.check` consults backdoor's published router state at a REAL path
(`~/.backdoor/failover-state.json` by default) to decide whether to stand down.
Any test that reaches `check` without stubbing `router_failover` therefore
inherits the developer machine's live network state: if the router happens to be
mid-failover, the guard correctly refuses, and a test asserting `ok` fails for a
reason that has nothing to do with the code under test.

Pin the path somewhere empty for every test, so the suite's result never depends
on whether the machine running it has internet.

Note this covers `pytest` only. `python tests/test_llmjury.py` runs the same
tests through the module's own `__main__` runner, where fixtures do not apply —
so individual tests still stub `router_failover` explicitly, and this fixture is
the backstop for anything that forgets.
"""

import pytest

from llmjury import memguard


@pytest.fixture(autouse=True)
def _isolate_router_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memguard, "ROUTER_STATE_PATH", str(tmp_path / "failover-state.json")
    )
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
