"""Explainable pathway policy and local verified-outcome model rankings.

No inference or provider calls happen here. The scheduler remains responsible
for atomic admission; rankings never grant permission to load a model.
"""
import json
import math
import os
from pathlib import Path
import re
from statistics import median

TASK_KINDS = ("general", "parsing", "algorithms", "transformation")
PATHWAYS = ("auto", "single", "council", "analyst")


def task_profile(task, verifier, kind="auto"):
    if kind not in ("auto", *TASK_KINDS):
        raise ValueError("unknown task kind")
    text = task.lower()
    if kind == "auto":
        # Explicit labels from the orchestrator override these bounded hints.
        kind = ("parsing" if re.search(r"\b(parse|parser|tokenize|grammar|csv|json)\b", text) else
                "algorithms" if re.search(r"\b(graph|algorithm|shortest path|dynamic programming)\b|o\(", text) else
                "transformation" if re.search(r"\b(sort|filter|group|encode|decode|normalize|transform)\b", text) else
                "general")
        source = "contract_hints"
    else:
        source = "explicit"
    oracle = ("functional" if hasattr(verifier, "entry_point") else
              "stdio" if hasattr(verifier, "cases") else "custom")
    return {"task_kind": kind, "kind_source": source, "oracle": oracle}


def initial_pathway(task, requested="auto"):
    if requested not in PATHWAYS:
        raise ValueError("unknown jury pathway")
    if requested != "auto":
        return "single" if requested == "single" else "council"
    # Explicit complexity constraints justify diversity sooner. Other tasks
    # keep the cheap single-model first tier and escalate on oracle failures.
    return "council" if re.search(r"\b(worst.case|dynamic programming|shortest path)\b|\bO\(", task,
                                  re.IGNORECASE) else "single"


class RoutingHistory:
    """Rolling metrics, containing no task, candidate, or credential text."""
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("LLMJURY_ROUTING_HISTORY") or
                         Path.home() / ".llmjury/routing.jsonl").expanduser()

    def rows(self):
        try:
            with self.path.open("rb") as stream:
                stream.seek(0, 2)
                start = max(0, stream.tell() - 256_000)
                stream.seek(start)
                if start:
                    stream.readline()
                lines = stream.readlines()
        except OSError:
            return []
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
                seconds = row.get("elapsed_seconds") if isinstance(row, dict) else None
                if (row.get("schema") == 1 and row.get("backend") == "ollama" and
                        isinstance(row.get("model"), str) and
                        type(row.get("verified")) is bool and
                        type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0):
                    rows.append(row)
            except (ValueError, AttributeError):
                continue
        return rows

    def rank(self, models, profile, samples, num_ctx):
        models = list(dict.fromkeys(models))
        rows = self.rows()
        scores = {}
        for model in models:
            matching = [r for r in rows if r["model"] == model.removesuffix(":latest") and
                        r.get("task_kind") == profile["task_kind"] and
                        r.get("oracle") == profile["oracle"] and
                        r.get("samples") == samples and r.get("num_ctx") == num_ctx][-30:]
            if len(matching) >= 3:
                probability = (sum(r["verified"] for r in matching) + 1) / (len(matching) + 2)
                scores[model] = probability / median(r["elapsed_seconds"] for r in matching)
        # Avoid pretending an unmeasured member has lower quality. Keep the
        # configured order until all competing members have comparable data.
        if models and len(scores) == len(models):
            return sorted(models, key=lambda m: -scores[m]), "verified_history"
        return models, "configured_order_insufficient_history"

    def record(self, model, profile, samples, num_ctx, verified, elapsed_seconds):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            row = {"schema": 1, "backend": "ollama", "model": model.removesuffix(":latest"),
                   "task_kind": profile["task_kind"], "oracle": profile["oracle"],
                   "samples": samples, "num_ctx": num_ctx, "verified": bool(verified),
                   "elapsed_seconds": max(0.001, elapsed_seconds)}
            data = (json.dumps(row) + "\n").encode()
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except OSError:
            pass
