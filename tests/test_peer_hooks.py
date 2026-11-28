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


def test_first_edit_review_runs_once_per_session_through_the_chain(repo, monkeypatch):
    calls = fake_peers(monkeypatch, {"codex": APPROVE})
    (repo / "main.go").write_text("package main\n\nfunc avg(xs []int) int { return 2 }\n")
    event = {"session_id": "s1", "tool_input": {"file_path": str(repo / "main.go")}}

    first = consult.review_first_edit("claude", event, str(repo))
    second = consult.review_first_edit("claude", event, str(repo))

    assert "In-session peer review from Codex" in first
    assert second == ""
    assert [c[0] for c in calls] == ["codex"]


def test_codex_first_edit_finds_the_change_through_git(repo, monkeypatch):
    calls = fake_peers(monkeypatch, {"claude": REJECT})
    (repo / "main.go").write_text("package main\n\nfunc avg(xs []int) int { return 3 }\n")
    event = {"session_id": "cx1", "tool_name": "Bash", "tool_input": {"command": "python3 edit.py"}}

    review = consult.review_first_edit("codex", event, str(repo))

    assert "In-session peer review from Claude" in review and "wrong sign" in review
    assert calls[0][0] == "claude"


def test_consult_retries_a_peer_once_then_falls_through(repo, monkeypatch):
    answers = {"codex": [None, None], "grok": [None, APPROVE]}
    seen = []

    def ask(name, instructions, content, use, author):
        seen.append(name)
        queue = answers.get(name, [])
        return queue.pop(0) if queue else None

    monkeypatch.setattr(peer, "ask_peer", ask)
    result = consult.tool_call("claude", {"repo_path": str(repo), "question": "Sort first?"})

    assert result["isError"] is False
    assert json.loads(result["content"][0]["text"])["peer"] == "Grok"
    assert seen == ["codex", "codex", "grok", "grok"]


def test_consult_reports_every_peer_unavailable(repo, monkeypatch):
    fake_peers(monkeypatch, {})
    result = consult.tool_call("cursor", {"repo_path": str(repo), "question": "Sort first?"})
    assert result["isError"] is True


def test_mcp_serves_consult_peer_and_keeps_the_old_names():
    for action, author, tool in [("serve", "codex", "consult_peer"), ("cursor", "claude", "consult_cursor"),
                                 ("claude", "cursor", "consult_claude")]:
        resolved_author, name = consult.server_identity(action, author if action == "serve" else None)
        listed = consult.reply({"id": 1, "method": "tools/list"}, resolved_author, name)
        assert resolved_author == author
        assert [t["name"] for t in listed["result"]["tools"]] == [tool]


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


def fake_peers(monkeypatch, answers):
    """answers: peer -> review dict or None (unavailable). Records (peer, use, model)."""
    calls = []

    def ask(name, instructions, content, use, author):
        calls.append((name, use, peer.model_for(name, use, author)))
        return answers.get(name)

    monkeypatch.setattr(peer, "ask_peer", ask)
    return calls


APPROVE = {"verdict": "approve", "summary": "fine", "findings": []}
REJECT = {"verdict": "changes_requested", "summary": "bug", "findings": ["a.py: wrong sign"]}


def test_claude_reviews_go_to_codex_first(repo, monkeypatch):
    calls = fake_peers(monkeypatch, {"codex": REJECT, "cursor": APPROVE})
    result = peer.PeerChain("claude").review("task", "diff", str(repo))
    assert [c[0] for c in calls] == ["codex"]
    assert result.verdict == "changes_requested"
    assert result.findings == ["a.py: wrong sign"]


def test_unavailable_peer_falls_through_to_the_next(repo, monkeypatch):
    calls = fake_peers(monkeypatch, {"codex": None, "cursor": None, "grok": APPROVE})
    chain = peer.PeerChain("claude")
    result = chain.review("task", "diff", str(repo))
    assert [c[0] for c in calls] == ["codex", "cursor", "grok"]
    assert result.verdict == "approve" and chain.name == "Grok"


def test_no_peer_available_is_an_error_not_an_approval(repo, monkeypatch):
    fake_peers(monkeypatch, {})
    assert peer.PeerChain("cursor").review("task", "diff", str(repo)).verdict == "error"


def test_each_author_has_its_own_default_order():
    assert peer.chain("claude", "review") == ["codex", "cursor", "grok"]
    assert peer.chain("claude", "consult") == ["codex", "grok", "cursor"]
    assert peer.chain("cursor", "review") == ["claude", "codex"]
    assert peer.chain("codex", "review") == ["claude", "cursor"]


def test_cursor_defaults_to_a_plan_included_model():
    assert peer.model_for("cursor", "review", "claude") == "composer-2.5"
    assert peer.model_for("cursor", "consult", "codex") == "composer-2.5"


def test_a_model_can_depend_on_the_author(repo, tmp_path):
    write_config(tmp_path / "home", {**OWNER_CONFIG, "models": {
        "cursor": {"review": {"codex": "claude-x", "default": "gpt-x"}}}})
    assert peer.model_for("cursor", "review", "codex") == "claude-x"
    assert peer.model_for("cursor", "review", "claude") == "gpt-x"


def test_config_can_override_order_and_models(repo, tmp_path):
    write_config(tmp_path / "home", {
        **OWNER_CONFIG,
        "chains": {"claude": {"review": ["cursor"]}},
        "models": {"cursor": {"review": "my-model"}},
    })
    assert peer.chain("claude", "review") == ["cursor"]
    assert peer.chain("claude", "consult") == ["codex", "grok", "cursor"]
    assert peer.model_for("cursor", "review", "codex") == "my-model"


def test_codex_stop_hook_blocks_with_the_review(repo, monkeypatch, capsys):
    fake_peers(monkeypatch, {"claude": REJECT})
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base", "--allow-empty")
    (repo / "main.go").write_text("package main\n\nfunc avg(xs []int) int { return -1 }\n")
    event = {"cwd": str(repo), "session_id": "codex-1", "hook_event_name": "Stop"}
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(event)))
    monkeypatch.delenv("LLMJURY_NESTED", raising=False)
    peer.main("codex")
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "block" and "Claude" in out["reason"]


def test_codex_is_found_outside_a_short_path(tmp_path, monkeypatch):
    stale = tmp_path / "stale-codex"
    stale.write_text("not a program\n")
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\n")
    codex.chmod(0o755)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))  # a short PATH with no codex on it, on any machine
    monkeypatch.setattr(peer, "CODEX_PLACES", (tmp_path / "missing", stale, codex))
    assert peer.codex_path() == str(codex)
    monkeypatch.setattr(peer, "CODEX_PLACES", (tmp_path / "missing", stale))
    assert peer.codex_path() is None


LONG_ANSWER = "The average is computed with integer division, so avg([1, 2]) returns 1. " * 5


def claude_transcript(path, prompt, answer):
    lines = [
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "<system-reminder>x</system-reminder>"}},
        {"type": "user", "message": {"role": "user", "content": prompt}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash"}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]}},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def run_stop(monkeypatch, capsys, host, event):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(event)))
    monkeypatch.delenv("LLMJURY_NESTED", raising=False)
    peer.main(host)
    out = capsys.readouterr().out.strip()
    return json.loads(out) if out else None


def test_chat_turn_is_reviewed_outside_any_repo(repo, tmp_path, monkeypatch, capsys):
    calls = fake_peers(monkeypatch, {"codex": REJECT})
    folder = tmp_path / "notes"
    folder.mkdir()
    transcript = claude_transcript(tmp_path / "t.jsonl", "What does avg return?", LONG_ANSWER)
    out = run_stop(monkeypatch, capsys, "claude", {
        "cwd": str(folder), "session_id": "chat-1", "transcript_path": str(transcript)})
    assert out["decision"] == "block" and "wrong sign" in out["reason"] and "Codex" in out["reason"]
    assert [c[:2] for c in calls] == [("codex", "review")]


def test_chat_review_sends_the_request_and_the_answer(repo, tmp_path, monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(peer, "ask_peer", lambda name, instructions, content, use, author: sent.append(content) or APPROVE)
    transcript = claude_transcript(tmp_path / "t.jsonl", "What does avg return?", "old answer")
    assert run_stop(monkeypatch, capsys, "claude", {
        "cwd": str(tmp_path), "session_id": "chat-2", "transcript_path": str(transcript),
        "last_assistant_message": LONG_ANSWER}) is None
    assert "What does avg return?" in sent[0] and LONG_ANSWER.strip() in sent[0] and "old answer" not in sent[0]


def test_chat_review_skips_client_and_rs21_checkouts(repo, tmp_path, monkeypatch, capsys):
    calls = fake_peers(monkeypatch, {"codex": REJECT})
    transcript = claude_transcript(tmp_path / "t.jsonl", "Explain this", LONG_ANSWER)
    for origin in ("https://github.com/some-client/app.git", "https://github.com/rs21io/app.git",
                   "https://gitlab.com/example-org/app.git"):
        git(repo, "remote", "set-url", "origin", origin)
        assert run_stop(monkeypatch, capsys, "claude", {
            "cwd": str(repo), "session_id": "c", "transcript_path": str(transcript)}) is None
    git(repo, "remote", "set-url", "origin", "https://github.com/example-org/some-app.git")
    git(repo, "remote", "add", "upstream", "https://github.com/some-client/app.git")
    assert run_stop(monkeypatch, capsys, "claude", {
        "cwd": str(repo), "session_id": "c", "transcript_path": str(transcript)}) is None
    assert calls == []


def test_location_check_fails_closed_when_git_cannot_answer(repo, monkeypatch):
    monkeypatch.setattr(peer, "run", lambda *a, **k: None)
    assert peer.location_allowed(str(repo)) is False
    assert peer.location_allowed(None) is True


def test_short_answers_are_not_reviewed(repo, tmp_path, monkeypatch, capsys):
    calls = fake_peers(monkeypatch, {"codex": REJECT})
    transcript = claude_transcript(tmp_path / "t.jsonl", "Thanks", "You're welcome.")
    assert run_stop(monkeypatch, capsys, "claude", {
        "cwd": str(tmp_path), "session_id": "s", "transcript_path": str(transcript)}) is None
    assert calls == []


def test_repo_turn_without_code_changes_still_gets_a_chat_review(repo, tmp_path, monkeypatch, capsys):
    calls = fake_peers(monkeypatch, {"codex": APPROVE})
    transcript = claude_transcript(tmp_path / "t.jsonl", "How is avg computed?", LONG_ANSWER)
    assert run_stop(monkeypatch, capsys, "claude", {
        "cwd": str(repo), "session_id": "r", "transcript_path": str(transcript)}) is None
    assert [c[0] for c in calls] == ["codex"]


def test_cursor_chat_without_a_workspace_is_reviewed(repo, tmp_path, monkeypatch, capsys):
    calls = fake_peers(monkeypatch, {"claude": REJECT})
    transcript = tmp_path / "cursor.jsonl"
    transcript.write_text("\n".join(json.dumps(line) for line in [
        {"role": "user", "message": {"content": [{"type": "text", "text":
            "<timestamp>now</timestamp>\n<user_query>\nWhat does avg return?\n</user_query>"}]}},
        {"role": "assistant", "message": {"content": [{"type": "text", "text": LONG_ANSWER}]}},
    ]))
    out = run_stop(monkeypatch, capsys, "cursor", {
        "status": "completed", "loop_count": 0, "workspace_roots": [],
        "conversation_id": "cur-1", "transcript_path": str(transcript)})
    assert "wrong sign" in out["followup_message"] and calls[0][0] == "claude"


def test_codex_transcript_turn_is_read(tmp_path):
    transcript = tmp_path / "rollout.jsonl"
    def message(role, kind, text):
        return {"type": "response_item", "payload": {"type": "message", "role": role,
                "content": [{"type": kind, "text": text}]}}
    transcript.write_text("\n".join(json.dumps(line) for line in [
        message("user", "input_text", "<environment_context>cwd</environment_context>"),
        message("user", "input_text", "Is avg right?"),
        message("assistant", "output_text", "No, it truncates."),
    ]))
    assert peer.last_turn(str(transcript)) == ("Is avg right?", "No, it truncates.")


def test_chat_review_can_be_turned_off(repo, tmp_path, monkeypatch, capsys):
    write_config(tmp_path / "home", {**OWNER_CONFIG, "chatReview": False})
    calls = fake_peers(monkeypatch, {"codex": REJECT})
    transcript = claude_transcript(tmp_path / "t.jsonl", "Explain", LONG_ANSWER)
    assert run_stop(monkeypatch, capsys, "claude", {
        "cwd": str(tmp_path), "session_id": "o", "transcript_path": str(transcript)}) is None
    assert calls == []


def test_consult_works_in_any_folder_but_not_client_checkouts(repo, tmp_path, monkeypatch):
    fake_peers(monkeypatch, {"codex": APPROVE})
    folder = tmp_path / "scratch"
    folder.mkdir()
    assert consult.tool_call("claude", {"repo_path": str(folder), "question": "Sort first?"})["isError"] is False
    git(repo, "remote", "set-url", "origin", "https://github.com/some-client/app.git")
    assert consult.tool_call("claude", {"repo_path": str(repo), "question": "Sort first?"})["isError"] is True


def test_location_check_denies_a_checkout_git_refuses_to_read(repo, monkeypatch):
    # e.g. git's "dubious ownership" refusal: git fails, but the folder is still a checkout.
    monkeypatch.setattr(peer, "run", lambda *a, **k: subprocess.CompletedProcess(a, 128, "", "fatal"))
    assert peer.location_allowed(str(repo / "sub")) is False
    assert peer.location_allowed(str(repo)) is False


def test_location_check_looks_at_every_enclosing_checkout(repo, tmp_path):
    git(repo, "remote", "set-url", "origin", "https://github.com/some-client/app.git")
    inner = repo / "vendor" / "tool"
    inner.mkdir(parents=True)
    git(inner, "init", "-q")
    git(inner, "remote", "add", "origin", "https://github.com/example-org/tool.git")
    assert peer.location_allowed(str(inner)) is False
    plain = tmp_path / "plain" / "deep"
    plain.mkdir(parents=True)
    assert peer.location_allowed(str(plain)) is True
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    git(fresh, "init", "-q")
    assert peer.location_allowed(str(fresh)) is True
