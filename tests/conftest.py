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
