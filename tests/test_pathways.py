import json

import pytest

from llmjury.pathways import RoutingHistory, initial_pathway, task_profile


class FunctionalVerifier:
    entry_point = "solve"


def test_task_profile_uses_explicit_kind_and_oracle_shape():
    profile = task_profile("parse a graph", FunctionalVerifier(), "parsing")

    assert profile == {
        "task_kind": "parsing",
        "kind_source": "explicit",
        "oracle": "functional",
    }


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("sort these records", "single"),
        ("find the shortest path in a graph", "council"),
    ],
)
def test_auto_pathway_chooses_by_task_shape(task, expected):
    assert initial_pathway(task) == expected


def test_routing_history_requires_comparable_observations(tmp_path):
    history = RoutingHistory(tmp_path / "routing.jsonl")
    profile = {"task_kind": "general", "oracle": "functional"}

    for model, verified, elapsed in (
        ("qwen", False, 1.0),
        ("qwen", False, 1.0),
        ("qwen", False, 1.0),
        ("phi", True, 1.5),
        ("phi", True, 1.5),
        ("phi", True, 1.5),
    ):
        history.record(model, profile, 4, 8192, verified, elapsed)

    order, basis = history.rank(["qwen", "phi"], profile, 4, 8192)

    assert order == ["phi", "qwen"]
    assert basis == "verified_history"
    assert all("task" not in row for row in history.rows())


def test_routing_history_ignores_malformed_rows(tmp_path):
    path = tmp_path / "routing.jsonl"
    path.write_text(json.dumps({"schema": 1, "model": "bad"}) + "\n")

    assert RoutingHistory(path).rows() == []
