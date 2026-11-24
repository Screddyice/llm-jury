"""The Claude Code <-> Cursor peer hooks in deploy/peer-hooks."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "deploy" / "peer-hooks"
sys.path.insert(0, str(HOOKS))

import peer_consult_mcp as consult  # noqa: E402
import provider_review_hook as peer  # noqa: E402


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "remote", "add", "origin", "https://github.com/teamnebula-ai/some-app.git")
    (root / "main.go").write_text("package main\n\nfunc avg(xs []int) int { return 0 }\n")
    git(root, "add", ".")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return root


def test_repo_allowed_is_limited_to_first_party_owners(repo):
    assert peer.repo_allowed(repo)
    for origin, allowed in [
        ("git@github.com:Screddyice/tool.git", True),
        ("https://github.com/reddy2help/site.git", True),
        ("https://github.com/rs21io/app.git", False),
        ("https://github.com/some-client/app.git", False),
    ]:
        git(repo, "remote", "set-url", "origin", origin)
        assert peer.repo_allowed(repo) is allowed, origin


def test_edited_context_covers_go_files(repo):
    (repo / "main.go").write_text("package main\n\nfunc avg(xs []int) int { return 1 }\n")
    context = consult.edited_context(str(repo), "main.go")
    assert "return 1" in context


def test_edited_context_skips_credential_files(repo):
    (repo / "secrets.yml").write_text("token: x\n")
    assert consult.edited_context(str(repo), "secrets.yml") == ""


def test_first_edit_review_runs_outside_llm_jury_once_per_session(repo, monkeypatch):
    calls = []

    def fake_tool_call(reviewer, arguments):
        calls.append(arguments["repo_path"])
        return {"content": [{"type": "text", "text": json.dumps({"verdict": "approve"})}]}

    monkeypatch.setattr(consult, "tool_call", fake_tool_call)
    (repo / "main.go").write_text("package main\n\nfunc avg(xs []int) int { return 2 }\n")
    event = {"session_id": "s1", "tool_input": {"file_path": str(repo / "main.go")}}

    first = consult.review_first_edit("claude", event, str(repo))
    second = consult.review_first_edit("claude", event, str(repo))

    assert "In-session peer review from Cursor" in first
    assert second == ""
    assert calls == [str(repo)]


def test_consult_retries_once_when_the_peer_reply_is_unusable(repo):
    replies = [None, {"verdict": "approve", "summary": "ok", "findings": []}]

    def flaky(question, context, scratch):
        return replies.pop(0)

    result = consult.tool_call(flaky, {"repo_path": str(repo), "question": "Sort first?"})

    assert result["isError"] is False
    assert json.loads(result["content"][0]["text"])["summary"] == "ok"
    assert replies == []


def test_consult_reports_the_peer_unavailable_after_two_unusable_replies(repo):
    calls = []

    def dead(question, context, scratch):
        calls.append(question)

    result = consult.tool_call(dead, {"repo_path": str(repo), "question": "Sort first?"})

    assert result["isError"] is True
    assert len(calls) == 2


def test_mcp_lists_the_consult_tool_for_each_peer():
    for target in ("claude", "cursor"):
        listed = consult.reply({"id": 1, "method": "tools/list"}, target)
        assert [t["name"] for t in listed["result"]["tools"]] == ["consult_" + target]
