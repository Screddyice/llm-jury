"""Offline concurrency tests using real file locks and fake model generations."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import os
import json
from pathlib import Path
import select
import subprocess
import sys
import threading

import pytest

from llmjury import memguard
from llmjury.engine import Engine
from llmjury.backends import Backend
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


def test_group_reservation_counts_both_lanes_and_releases_after_exception(scheduler_environment):
    scheduler = LocalScheduler("http://localhost:11434", wait_seconds=0)
    with pytest.raises(ValueError, match="generation failed"):
        with scheduler.reserve_many(["phi", "phi:latest", "gemma"]) as selected:
            assert selected == ["phi", "gemma"]
            assert scheduler_environment[-1][0] == ["phi", "gemma"]
            with scheduler.reserve(["third"]) as third:
                assert third is None
            raise ValueError("generation failed")
    with scheduler.reserve_many(["phi", "gemma"]) as selected:
        assert selected == ["phi", "gemma"]


def test_group_admission_terminal_refusal_releases_partial_reservation(scheduler_environment, monkeypatch):
    def check(models, **_):
        return memguard.Report(len(models) == 1, router=len(models) > 1, router_reason="27B lease")

    monkeypatch.setattr(memguard, "check", check)
    scheduler = LocalScheduler("http://localhost:11434", wait_seconds=0)
    with pytest.raises(RuntimeError):
        with scheduler.reserve_many(["phi", "gemma"]):
            pytest.fail("terminal refusal must stop the complete stage")
    with scheduler.reserve(["phi"]) as selected:
        assert selected == "phi"


def test_complementary_local_council_overlaps_with_two_workers(scheduler_environment):
    barrier = threading.Barrier(2)
    calls = []

    class Model(Backend):
        name = "ollama"

        def _sample(self, model, prompt, temperature, max_tokens, index):
            calls.append((model, index))
            if model != "best" and index == 0:
                barrier.wait(timeout=5)
            return "def solve():\n    return " + ("1" if model == "gemma" else "0")

    class Verifier:
        def verify(self, text):
            return "return 1" in text

    result = Engine(Model(), panel=["best", "phi", "gemma"], best="best",
                    k=3, workers=2, local_scheduler=LocalScheduler(
                        "http://localhost:11434", wait_seconds=0)).solve("task", Verifier())
    assert result.verified and result.stage == "council" and result.model == "gemma"
    assert set(calls[:3]) == {("best", 0), ("best", 1), ("best", 2)}
    assert set(calls[3:5]) == {("phi", 0), ("gemma", 0)}


def test_memory_constrained_council_tries_remaining_members_in_later_stages(
        scheduler_environment, monkeypatch):
    monkeypatch.setattr(memguard, "check", lambda models, **_: memguard.Report(len(models) == 1))
    calls = []

    class Model(Backend):
        name = "ollama"

        def _one(self, model, *_):
            calls.append(model)
            return "def solve():\n    return " + ("1" if model == "gemma" else "0")

    class Verifier:
        def verify(self, text):
            return "return 1" in text

    result = Engine(Model(), panel=["best", "phi", "gemma"], best="best", k=1,
                    local_scheduler=LocalScheduler("http://localhost:11434", wait_seconds=0))
    result = result.solve("task", Verifier())
    assert result.verified and result.model == "gemma" and result.stage == "council"
    assert calls == ["best", "phi", "gemma"]


@pytest.mark.parametrize("rebalance", [True, False])
def test_local_refusal_preserves_explicitly_routed_member(
        scheduler_environment, monkeypatch, rebalance):
    monkeypatch.setattr(memguard, "check", lambda *_args, **_: memguard.Report(False))
    calls = []

    class Local(Backend):
        name = "ollama"

        def _one(self, *_):
            pytest.fail("refused local model must not generate")

    class Routed(Backend):
        name = "codex"

        def _one(self, model, *_):
            calls.append(model)
            return "def solve():\n    return 1"

    class Verifier:
        def verify(self, text):
            return "return 1" in text

    result = Engine(Local(), panel=["qwen", "phi", "brain"], best="qwen", k=1,
                    route={"brain": Routed()}, frontier=["frontier"],
                    frontier_backend=Routed(), rebalance=rebalance,
                    local_scheduler=LocalScheduler("http://localhost:11434", wait_seconds=0))
    result = result.solve("task", Verifier())
    assert result.verified and result.stage == "council" and result.model == "brain"
    assert calls == ["brain"]


@pytest.mark.parametrize("winner,escalate", [
    ("best", True), ("phi", True), ("gemma", True),
    ("cheap", True), ("top", True), (None, True), ("phi", False),
])
def test_scheduled_local_and_cloud_share_verification_and_escalation_contract(
        scheduler_environment, winner, escalate):
    class Model(Backend):
        def __init__(self, name):
            super().__init__()
            self.name, self.calls = name, []

        def _sample(self, model, prompt, temperature, max_tokens, index):
            self.calls.append((model, index))
            return "def solve():\n    return " + ("1" if model == winner else "0")

    class Verifier:
        def verify(self, text):
            return "return 1" in text

    results = []
    for name in ("openrouter", "ollama"):
        provider, frontier = Model(name), Model("codex")
        result = Engine(provider, panel=["best", "phi", "gemma"], best="best", k=2,
                        frontier=["cheap", "top"], frontier_backend=frontier, frontier_k=1,
                        local_scheduler=(LocalScheduler("http://localhost:11434", wait_seconds=0)
                                         if name == "ollama" else None))
        result = result.solve("same task and oracle", Verifier(), escalate=escalate)
        results.append((result.verified, result.stage, result.model, result.answer))
        expected_frontier = ([("cheap", 0)] if winner == "cheap" else
                             [("cheap", 0), ("top", 0)] if winner in ("top", None) else [])
        assert frontier.calls == expected_frontier
    assert results[0] == results[1]


def test_interleaved_sampling_keeps_distinct_cache_indices(tmp_path):
    calls, barrier = [], threading.Barrier(2)

    class Model(Backend):
        name = "ollama"

        def _one(self, model, *_):
            calls.append(model)
            if len(calls) <= 2:
                barrier.wait(timeout=5)
            return "def solve():\n    return 0"

    cache_path = tmp_path / "cache.jsonl"
    backend = Model(cache_path=str(cache_path))
    engine = Engine(backend, k=3, workers=2)
    pairs = [(model, backend) for model in ("phi", "gemma")]
    for _ in range(2):
        with ThreadPoolExecutor(2) as pool:
            futures = engine._submit(pool, pairs, "task")
            assert len(futures) == 6
            assert all(f.result(timeout=10) for f in futures)
    assert len(calls) == 6 and set(calls[:2]) == {"phi", "gemma"}
    rows = [json.loads(line) for line in cache_path.read_text().splitlines()]
    expected_keys = {backend.cache.key("ollama", model, 0.7, 4000, index, "task")
                     for model in ("phi", "gemma") for index in range(3)}
    assert {row["k"] for row in rows} == expected_keys


def test_council_keeps_both_reservations_until_running_samples_finish(scheduler_environment):
    started, release, verified = threading.Event(), threading.Event(), threading.Event()

    class Model(Backend):
        name = "ollama"

        def _one(self, model, *_):
            if model == "best":
                return "def solve():\n    return 0"
            if model == "gemma":
                started.set()
                release.wait(timeout=5)
            else:
                assert started.wait(timeout=5)
            return "def solve():\n    return 1"

    class Verifier:
        def verify(self, text):
            if "return 1" in text:
                verified.set()
                return True
            return False

    scheduler = LocalScheduler("http://localhost:11434", wait_seconds=0)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(Engine(Model(), panel=["best", "phi", "gemma"], best="best",
                                    k=1, workers=2, local_scheduler=scheduler).solve, "task", Verifier())
        try:
            assert verified.wait(timeout=5) and not future.done()
            with scheduler.reserve(["third"]) as selected:
                assert selected is None
        finally:
            release.set()
        assert future.result(timeout=5).verified


def test_cli_full_panel_refusal_retains_routed_member(
        scheduler_environment, monkeypatch, tmp_path, capsys):
    from llmjury import cli, backends, verifiers
    from llmjury.engine import Result

    task, cases = tmp_path / "task.txt", tmp_path / "cases.json"
    task.write_text("implement solve")
    cases.write_text('[{"args": [], "expected": 1}]')
    monkeypatch.setattr(memguard, "check", lambda *_args, **_: memguard.Report(False))
    monkeypatch.setattr(backends, "baked_system_warnings", lambda *_: [])
    monkeypatch.setattr(verifiers, "sandbox_note", lambda: ("container", ""))
    monkeypatch.setattr(cli, "_refuse_root", lambda: None)
    monkeypatch.setattr(sys, "argv", ["llmjury", "solve", "--task", str(task),
        "--cases", str(cases), "--entry-point", "solve", "--backend", "ollama",
        "--models", "qwen,phi", "--brain", "--brain-model", "brain",
        "--mem-check", "refuse", "--json"])
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    class CaptureEngine:
        def __init__(self, backend, **settings):
            assert settings["use_panel"] and settings["panel"] == ["brain"]
            assert settings["best"] == "brain" and "brain" in settings["route"]

        def solve(self, *_):
            return Result("def solve():\n    return 1", None, True, "brain", "single", 1)

    monkeypatch.setattr(cli, "Engine", CaptureEngine)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["verified"]


@pytest.mark.parametrize("explicit_panel", [True, False])
def test_cli_subset_diagnostic_keeps_members_for_later_stages(
        scheduler_environment, monkeypatch, tmp_path, explicit_panel):
    from llmjury import cli, backends, verifiers, panels
    from llmjury.engine import Result

    task, cases = tmp_path / "task.txt", tmp_path / "cases.json"
    task.write_text("implement solve")
    cases.write_text('[{"args": [], "expected": 1}]')
    monkeypatch.setattr(memguard, "check", lambda models, **_: memguard.Report(
        len(set(models)) == 1))
    monkeypatch.setattr(backends, "baked_system_warnings", lambda *_: [])
    monkeypatch.setattr(verifiers, "sandbox_note", lambda: ("container", ""))
    monkeypatch.setattr(cli, "_refuse_root", lambda: None)
    arguments = ["llmjury", "solve", "--task", str(task), "--cases", str(cases),
                 "--entry-point", "solve", "--backend", "ollama", "--mem-check", "refuse"]
    if explicit_panel:
        arguments += ["--models", "qwen,phi"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    class CaptureEngine:
        def __init__(self, backend, **settings):
            assert settings["use_panel"]
            configured = settings["panel"] or panels.LOCAL_PANEL
            assert configured == (["qwen", "phi"] if explicit_panel else panels.LOCAL_PANEL)

        def solve(self, *_):
            return Result("def solve():\n    return 1", None, True, "member", "council", 1)

    monkeypatch.setattr(cli, "Engine", CaptureEngine)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0


def test_analyst_ranks_council_candidates_before_verifier_selection(scheduler_environment):
    calls = []

    class Model(Backend):
        name = "ollama"

        def _one(self, model, prompt, *_):
            calls.append((model, prompt))
            if prompt.startswith("You are the local council analyst"):
                return '{"order":[1,0],"consensus":"same shape","conflicts":"none","gaps":"none"}'
            return ("def solve():\n    return 1" if model == "gemma" else
                    "def solve():\n    return 0")

    class Verifier:
        def verify(self, text):
            return "return 1" in text

    model = Model()
    result = Engine(model, panel=["best", "phi", "gemma"], best="best", k=1,
                    analyst_model="phi", analyst_backend=model, workers=2,
                    local_scheduler=LocalScheduler("http://localhost:11434", wait_seconds=0)) \
        .solve("choose the correct implementation", Verifier())
    assert result.verified and result.stage == "council" and result.model == "gemma"
    assert result.analyst_model == "phi"
    assert result.analyst_summary == {
        "consensus": "same shape", "conflicts": "none", "gaps": "none"}
    assert any(prompt.startswith("You are the local council analyst") for _, prompt in calls)


def test_explicit_analyst_pathway_admits_qwen_worker_after_generation(scheduler_environment):
    calls = []

    class Model(Backend):
        name = "ollama"

        def _one(self, model, prompt, *_):
            calls.append((model, prompt))
            if prompt.startswith("You are the local council analyst"):
                return '{"order":[1,0],"consensus":"ok","conflicts":"none","gaps":"none"}'
            return "def solve():\n    return " + ("1" if model == "gemma" else "0")

    class Verifier:
        def verify(self, text):
            return "return 1" in text

    model = Model()
    result = Engine(
        model, panel=["phi", "gemma"], best="phi", k=1, pathway="analyst",
        analyst_model="qwen", analyst_backend=model, workers=2,
        local_scheduler=LocalScheduler("http://localhost:11434", wait_seconds=0),
    ).solve("choose the correct implementation", Verifier())

    assert result.verified and result.model == "gemma"
    assert result.analyst_model == "qwen"
    assert result.analyst_summary == {"consensus": "ok", "conflicts": "none", "gaps": "none"}
    assert any(model == "qwen" and prompt.startswith("You are the local council analyst")
               for model, prompt in calls)


def test_analyst_parse_fails_closed_to_generation_order():
    from llmjury.analysis import parse

    assert parse("not json", 3) == ([0, 1, 2], None)
    assert parse('{"order":[0,0]}', 2) == ([0, 1], None)
    assert parse('{"order":[1,0],"consensus":"ok"}', 2)[0] == [1, 0]
