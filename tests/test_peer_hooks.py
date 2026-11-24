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


OWNER_CONFIG = {"owners": ["example-org", "example-user"]}


def write_config(home, data):
    path = home / ".config" / "llmjury" / "peer-review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(peer, "CONFIG", home / ".config" / "llmjury" / "peer-review.json")
    write_config(home, OWNER_CONFIG)
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "remote", "add", "origin", "https://github.com/example-org/some-app.git")
    (root / "main.go").write_text("package main\n\nfunc avg(xs []int) int { return 0 }\n")
    git(root, "add", ".")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return root


def test_repo_allowed_is_limited_to_configured_owners(repo):
    assert peer.repo_allowed(repo)
    for origin, allowed in [
        ("git@github.com:Example-User/tool.git", True),
        ("https://github.com/rs21io/app.git", False),
        ("https://github.com/some-client/app.git", False),
    ]:
        git(repo, "remote", "set-url", "origin", origin)
        assert peer.repo_allowed(repo) is allowed, origin


def test_no_config_means_no_repo_is_reviewed(repo, tmp_path):
    peer.CONFIG.unlink()
    assert not peer.repo_allowed(repo)


def test_claude_profile_comes_from_config(repo, tmp_path):
    assert peer.claude_config_dir() is None
    profile = tmp_path / "profile"
    profile.mkdir()
    write_config(tmp_path / "home", {**OWNER_CONFIG, "claudeConfigDir": str(profile)})
    assert peer.claude_config_dir() == str(profile)


def test_shipped_files_hold_no_machine_specific_values():
    root = HOOKS.parents[1]
    shipped = list(HOOKS.glob("*.py")) + list((root / "deploy").glob("*.plist*"))
    for path in shipped:
        text = path.read_text()
        assert "/Users/" not in text and "/home/" not in text, f"{path.name} hard-codes a home path"
        assert "OWNERS =" not in text, f"{path.name} hard-codes the allowed owners"
    template = (root / "deploy" / "llmjury-monthly-refresh.plist.template").read_text()
    assert "__HOME__" in template and "__REPO__" in template


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


def test_cursor_secondary_accepts_a_tilde_project_path(repo, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    caches = home / ".t3" / "caches"
    caches.mkdir(parents=True)
    (caches / "alt.json").write_text(json.dumps({
        "auth": {"status": "authenticated", "email": "alt@example.com"},
        "models": [{"slug": "composer-2.5"}],
    }))
    (caches / "cursor.json").write_text(json.dumps({"auth": {"email": "main@example.com"}}))
    write_config(home, {**OWNER_CONFIG, "cursorSecondary": {"instance": "alt", "project": "~/proj"}})
    monkeypatch.setattr(peer, "ROOT", home)
    commands = []
    monkeypatch.setattr(peer, "run", lambda cmd, **kwargs: commands.append(cmd))

    peer.cursor_secondary("review this")

    assert len(commands) == 1
    assert str(home / "proj") in [str(part) for part in commands[0]]
