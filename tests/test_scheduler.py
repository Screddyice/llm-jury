"""Offline concurrency tests using real file locks and fake model generations."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import os
from pathlib import Path
import select
import subprocess
import sys
import threading

import pytest

from llmjury import memguard
from llmjury.engine import Engine
from llmjury.scheduler import LocalScheduler, choose_model


@pytest.fixture
def scheduler_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMJURY_LOCAL_LOCK", str(tmp_path / "compute.lock"))
    monkeypatch.setattr(memguard, "exclusive_compute", lambda *_: (False, ""))
    probes = []

    def check(models, **settings):
        probes.append((list(models), settings))
        return memguard.Report(True)

    monkeypatch.setattr(memguard, "check", check)
    return probes


def test_choose_idle_model_and_preserve_explicit_preference():
    models, busy = ["qwen", "phi"], ["qwen"]
    assert choose_model(models, "qwen", busy) == "phi"
    assert choose_model(models, "phi", []) == "phi"
    assert choose_model(models, "absent", []) == "qwen"
    assert choose_model(models, "qwen", models) is None
    assert models == ["qwen", "phi"] and busy == ["qwen"]


def test_two_independent_tasks_overlap_on_distinct_models(scheduler_environment):
    barrier = threading.Barrier(2)
    calls = []

    class Backend:
        name = "ollama"

        def complete(self, model, prompt, *args):
            calls.append((model, prompt))
            barrier.wait(timeout=5)
            # Each task receives its own answer, so sharing one oracle or prompt fails.
            value = 11 if "task-A" in prompt else 22
            return [f"def solve():\n    return {value}"]

    class Verifier:
        def __init__(self, expected):
            self.expected = expected

        def verify(self, text):
            return f"return {self.expected}" in text

    def run(task, expected):
        with memguard.local_compute_lock(shared=True):
            engine = Engine(Backend(), panel=["qwen", "phi"], best="qwen", k=1,
                            local_scheduler=LocalScheduler("http://localhost:11434"))
            return engine.solve(task, Verifier(expected))

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(run, "task-A", 11)
        b = pool.submit(run, "task-B", 22)
        results = [a.result(timeout=10), b.result(timeout=10)]
    assert all(result.verified for result in results)
    assert {result.model for result in results} == {"qwen", "phi"}
    assert sorted(result.answer for result in results) == [
        "def solve():\n    return 11", "def solve():\n    return 22"]
    assert len(calls) == 2
    assert any(set(models) == {"qwen", "phi"} for models, _ in scheduler_environment)


def test_aggregate_reservations_use_largest_context(scheduler_environment):
    a = LocalScheduler("http://localhost:11434", 8192, wait_seconds=0)
    b = LocalScheduler("http://localhost:11434", 2048, wait_seconds=0)
    with a.reserve(["qwen", "phi"], "qwen") as first:
        with b.reserve(["qwen", "phi"], "qwen") as second:
            assert (first, second) == ("qwen", "phi")
    assert scheduler_environment[-1] == (["qwen", "phi"], {
        "host": "http://localhost:11434", "num_ctx": 8192})


def test_two_slots_and_aliases_never_admit_third_task(scheduler_environment):
    with ExitStack() as stack:
        a = LocalScheduler("http://localhost:11434", wait_seconds=0)
        assert stack.enter_context(a.reserve(["qwen:latest"])) == "qwen:latest"
        assert stack.enter_context(a.reserve(["qwen", "phi"])) == "phi"
        assert stack.enter_context(a.reserve(["third"])) is None
    with a.reserve(["qwen"]) as selected:
        assert selected == "qwen"


def test_combined_memory_refusal_prevents_second_generation(scheduler_environment, monkeypatch):
    scheduler = LocalScheduler("http://localhost:11434", wait_seconds=0)
    with scheduler.reserve(["qwen"]) as selected:
        assert selected == "qwen"
        monkeypatch.setattr(memguard, "check", lambda models, **_: memguard.Report(
            False, pressure_reason="insufficient aggregate headroom"))
        with scheduler.reserve(["phi"]) as second:
            assert second is None


def test_smaller_admitted_model_can_replace_preferred_one(scheduler_environment, monkeypatch):
    monkeypatch.setattr(memguard, "check", lambda models, **_: memguard.Report(
        models == ["phi"], pressure_reason="preferred model does not fit"))
    with LocalScheduler("http://localhost:11434").reserve(["qwen", "phi"], "qwen") as model:
        assert model == "phi"


def test_exclusive_ownership_and_unknown_active_metadata_fail_closed(scheduler_environment, monkeypatch):
    scheduler = LocalScheduler("http://localhost:11434", wait_seconds=0)
    monkeypatch.setattr(memguard, "exclusive_compute", lambda *_: (True, "27B lease"))
    with pytest.raises(RuntimeError, match="27B"):
        with scheduler.reserve(["qwen"]):
            pytest.fail("exclusive owner must stop generation")
    monkeypatch.setattr(memguard, "exclusive_compute", lambda *_: (False, ""))
    with scheduler.reserve(["qwen"]):
        (scheduler.root / "slot-0.json").write_text("broken metadata")
        with pytest.raises(RuntimeError, match="active local reservation"):
            with scheduler.reserve(["phi"]):
                pytest.fail("unknown reservations must not be ignored")


def test_shared_solves_preserve_legacy_exclusive_lock(scheduler_environment):
    with memguard.local_compute_lock(shared=True):
        with memguard.local_compute_lock(shared=True):
            with pytest.raises(RuntimeError):
                with memguard.local_compute_lock():
                    pytest.fail("legacy exclusive job must wait")
    with memguard.local_compute_lock():
        with pytest.raises(RuntimeError):
            with memguard.local_compute_lock(shared=True):
                pytest.fail("scheduled solve must respect legacy owner")
        with pytest.raises(RuntimeError):
            with LocalScheduler("http://localhost:11434").reserve(["qwen"]):
                pytest.fail("direct scheduler use must respect legacy owner")


def test_new_exclusive_ownership_stops_frontier_after_local_failure(scheduler_environment, monkeypatch):
    calls = []

    class Backend:
        name = "ollama"

        def complete(self, model, *_args):
            calls.append(model)
            monkeypatch.setattr(memguard, "exclusive_compute", lambda *_: (True, "new 27B lease"))
            return ["def solve():\n    return 0"]

    class Frontier(Backend):
        name = "codex"

    class Verifier:
        def verify(self, text):
            return False

    with pytest.raises(RuntimeError, match="27B"):
        Engine(Backend(), panel=["qwen"], best="qwen", k=1,
               frontier=["codex"], frontier_backend=Frontier(),
               local_scheduler=LocalScheduler("http://localhost:11434")).solve("task", Verifier())
    assert calls == ["qwen"]


def test_cross_process_assignment_and_crash_release(scheduler_environment):
    script = '''
from llmjury import memguard
from llmjury.scheduler import LocalScheduler
import sys
memguard.exclusive_compute = lambda *_: (False, "")
memguard.check = lambda *_args, **_kw: memguard.Report(True)
with memguard.local_compute_lock(shared=True):
    with LocalScheduler("http://localhost:11434", wait_seconds=0).reserve(
            ["qwen", "phi"], "qwen") as model:
        print(model, flush=True)
        sys.stdin.readline()
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    processes = []

    def start(expected):
        process = subprocess.Popen([sys.executable, "-c", script], env=environment,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        processes.append(process)
        assert select.select([process.stdout], [], [], 10)[0], "worker did not start"
        assert process.stdout.readline().strip() == expected
        return process

    try:
        a = start("qwen")
        start("phi")
        a.kill()
        a.wait(timeout=5)
        with LocalScheduler("http://localhost:11434", wait_seconds=0).reserve(["qwen"]) as model:
            assert model == "qwen"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


def test_ordered_fallback_uses_own_oracle_and_preserves_sampling(scheduler_environment):
    calls = []

    class Backend:
        name = "ollama"

        def complete(self, model, prompt, n, *_args):
            calls.append((model, n))
            return ["def solve():\n    return 0"] * n

    class Frontier(Backend):
        name = "codex"

        def complete(self, model, prompt, n, *_args):
            calls.append((model, n))
            return ["def solve():\n    return 1"] * n

    class Verifier:
        def verify(self, text):
            return "return 1" in text

    result = Engine(Backend(), panel=["qwen", "phi"], best="qwen", k=3,
                    frontier=["codex"], frontier_backend=Frontier(), frontier_k=1,
                    local_scheduler=LocalScheduler("http://localhost:11434"))
    result = result.solve("task", Verifier())
    assert result.verified and result.stage == "frontier"
    assert calls == [("qwen", 3), ("phi", 3), ("codex", 1)]


def test_running_samples_keep_reservation_after_first_verifies(scheduler_environment):
    from llmjury.backends import Backend
    started, release, verified = threading.Event(), threading.Event(), threading.Event()

    class Model(Backend):
        name = "ollama"

        def _sample(self, model, prompt, temperature, max_tokens, index):
            if index == 1:
                started.set()
                release.wait(5)
            else:
                assert started.wait(5)
            return "def solve():\n    return 1"

    class Verifier:
        def verify(self, text):
            verified.set()
            return True

    scheduler = LocalScheduler("http://localhost:11434", wait_seconds=0)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(Engine(Model(), panel=["qwen"], best="qwen", k=2,
                                    local_scheduler=scheduler).solve, "task", Verifier())
        try:
            assert verified.wait(5)
            assert not future.done()
            with scheduler.reserve(["qwen"]) as selected:
                assert selected is None
        finally:
            release.set()
        assert future.result(timeout=5).verified


def test_explicit_best_is_not_rebalanced(scheduler_environment):
    calls = []

    class Backend:
        name = "ollama"

        def complete(self, model, *_):
            calls.append(model)
            return ["def solve():\n    return 1"]

    class Verifier:
        def verify(self, text):
            return True

    scheduler = LocalScheduler("http://localhost:11434", wait_seconds=0)
    with scheduler.reserve(["qwen"]):
        result = Engine(Backend(), panel=["qwen", "phi"], best="qwen", k=1,
                        rebalance=False, local_scheduler=scheduler).solve("task", Verifier())
    assert not result.verified and not calls
