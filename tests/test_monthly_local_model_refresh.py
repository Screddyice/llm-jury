import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/monthly-local-model-refresh.py"
SPEC = importlib.util.spec_from_file_location("monthly_refresh", SCRIPT)
REFRESH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFRESH)


def test_refresh_selects_largest_admitted_local_subset(monkeypatch):
    responses = {
        ("best", "other"): {"ok": False, "terminal": False, "reason": "too large"},
        ("best",): {"ok": True, "terminal": False, "reason": "admitted"},
    }
    monkeypatch.setattr(
        REFRESH, "preflight", lambda python, models: responses[tuple(models)])

    selected, result = REFRESH.largest_admitted_panel("python", ["best", "other"])

    assert selected == ["best"]
    assert result["ok"]


def test_refresh_stops_subset_probing_under_host_pressure(monkeypatch):
    calls = []

    def refusal(python, models):
        calls.append(models)
        return {"ok": False, "terminal": False,
                "reason": "host memory pressure is elevated"}

    monkeypatch.setattr(REFRESH, "preflight", refusal)

    selected, result = REFRESH.largest_admitted_panel("python", ["best", "other"])

    assert selected == []
    assert calls == [["best", "other"]]
    assert "pressure" in result["reason"]


def test_scheduled_subprocesses_use_conservative_hardware_defaults(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs["env"])
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(REFRESH.subprocess, "run", fake_run)
    monkeypatch.delenv("LLMJURY_OLLAMA_PARALLEL", raising=False)
    monkeypatch.delenv("LLMJURY_PROMPT_CACHE_MIB", raising=False)

    REFRESH.run(["ollama", "list"], 1)

    assert captured["LLMJURY_OLLAMA_PARALLEL"] == "2"
    assert captured["LLMJURY_PROMPT_CACHE_MIB"] == "1024"
