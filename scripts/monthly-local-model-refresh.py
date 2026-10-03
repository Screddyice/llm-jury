#!/usr/bin/env python3
"""Refresh the local Ollama panel and run a guarded verifier smoke test."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
HOME = Path.home()
LOCK_PATH = HOME / ".llmjury/monthly-refresh.lock"
REPORT_PATH = HOME / ".llmjury/monthly-refresh.json"
NUM_CTX = 8192


def run(command, timeout):
    environment = os.environ.copy()
    environment.pop("OPENROUTER_API_KEY", None)
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPOSITORY) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else "")
    return subprocess.run(command, capture_output=True, text=True,
                          timeout=timeout, env=environment)


def installed_models(ollama):
    result = run([ollama, "list"], 60)
    if result.returncode:
        raise RuntimeError("ollama list failed")
    models = {}
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2:
            models[fields[0]] = fields[1]
    return models


def preflight(python, models):
    result = run([python, "-m", "llmjury.cli", "preflight",
                  "--models", ",".join(models), "--num-ctx", str(NUM_CTX)], 120)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"ok": False, "reason": "preflight returned invalid JSON"}
    payload["models"] = models
    return payload


def largest_admitted_panel(python, models):
    for size in range(len(models), 0, -1):
        for candidate in combinations(models, size):
            result = preflight(python, list(candidate))
            if result.get("ok"):
                return list(candidate), result
            if result.get("terminal") or "pressure" in result.get("reason", "").lower():
                return [], result
    return [], {"ok": False, "reason": "no local model was admitted"}


def smoke_test(python, models):
    with tempfile.TemporaryDirectory(prefix="llmjury-monthly-") as directory:
        root = Path(directory)
        task = root / "task.txt"
        tests = root / "tests.py"
        task.write_text("Implement solve(a, b) that returns the sum of two integers.\n")
        tests.write_text("def check(candidate):\n    assert candidate(2, 3) == 5\n")
        result = run([
            python, "-m", "llmjury.cli", "solve", "--task", str(task),
            "--tests", str(tests), "--entry-point", "solve", "--backend", "ollama",
            "--models", ",".join(models), "--best", models[0], "--k", "1",
            "--jobs", "1", "--num-ctx", str(NUM_CTX), "--json",
        ], 900)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {"verified": False}
        return {"verified": bool(payload.get("verified")),
                "model": payload.get("model"), "stage": payload.get("stage"),
                "returncode": result.returncode}


def main():
    ollama = shutil.which("ollama")
    if not ollama:
        raise RuntimeError("ollama executable not found")
    python = sys.executable
    sys.path.insert(0, str(REPOSITORY))
    from llmjury.panels import LOCAL_PANEL

    REPORT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = installed_models(ollama)
        pulls = []
        for model in LOCAL_PANEL:
            result = run([ollama, "pull", model], 1800)
            after = installed_models(ollama) if result.returncode == 0 else before
            pulls.append({"model": model, "ok": result.returncode == 0,
                          "changed": before.get(model) != after.get(model)})
            before = after
        admitted, admission = largest_admitted_panel(python, LOCAL_PANEL)
        smoke = smoke_test(python, admitted) if admitted else {"verified": False}
        report = {
            "at": datetime.now(timezone.utc).isoformat(),
            "configured_panel": LOCAL_PANEL,
            "pulls": pulls,
            "admitted_panel": admitted,
            "preflight": admission,
            "local_smoke": smoke,
        }
        temporary = REPORT_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, REPORT_PATH)
        print(json.dumps(report, sort_keys=True))
        return 0 if all(item["ok"] for item in pulls) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"monthly local model refresh failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
