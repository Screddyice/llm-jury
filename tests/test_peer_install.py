"""The peer-hooks installer in deploy/peer-hooks/install.py, run against a fake home."""

import json
from pathlib import Path
import sys

HOOKS = Path(__file__).resolve().parents[1] / "deploy" / "peer-hooks"
sys.path.insert(0, str(HOOKS))

import install  # noqa: E402


def seed(home):
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "other-stop-hook"}]}]}}))
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    (home / ".cursor").mkdir()
    (home / ".cursor" / "hooks.json").write_text(json.dumps({"version": 1, "hooks": {"stop": [
        {"command": "python /old/provider_review_hook.py cursor", "timeout": 90}]}}))
    (home / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {}}))
    (home / ".codex").mkdir()
    (home / ".codex" / "hooks.json").write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": '"$HOME/.local/bin/llmjury" review-hook --host codex'}]},
        {"hooks": [{"type": "command", "command": "claude-mem hook codex summarize"}]}]}}))
    (home / ".codex" / "config.toml").write_text('model = "x"\n\n[mcp_servers.peer-consult]\ncommand = "old"\n\n[mcp_servers.other]\ncommand = "keep"\n')


def commands(entries, nested=True):
    found = []
    for entry in entries:
        for hook in (entry.get("hooks", []) if nested else [entry]):
            found.append(hook["command"])
    return found


def test_install_registers_every_agent_and_keeps_other_hooks(tmp_path):
    home = tmp_path / "home"
    seed(home)
    install.install(home, python="/py")

    claude = json.loads((home / ".claude" / "settings.json").read_text())["hooks"]
    assert "other-stop-hook" in commands(claude["Stop"])
    assert any(c.endswith("provider_review_hook.py claude") for c in commands(claude["Stop"]))
    assert any(c.endswith("after-edit-claude") for c in commands(claude["PostToolUse"]))
    mcp = json.loads((home / ".claude.json").read_text())["mcpServers"]
    assert mcp["peer-consult"]["args"][-2:] == ["serve", "claude"] and "other" in mcp

    cursor = json.loads((home / ".cursor" / "hooks.json").read_text())["hooks"]
    stops = [h for h in cursor["stop"] if "provider_review_hook" in h["command"]]
    assert len(stops) == 1 and stops[0]["timeout"] >= 300 and "/old/" not in stops[0]["command"]
    assert json.loads((home / ".cursor" / "mcp.json").read_text())["mcpServers"]["peer-consult"]["args"][-1] == "cursor"

    codex = json.loads((home / ".codex" / "hooks.json").read_text())["hooks"]
    stop_cmds = commands(codex["Stop"])
    assert not any("review-hook --host codex" in c for c in stop_cmds)
    assert "claude-mem hook codex summarize" in stop_cmds
    assert any(c.endswith("provider_review_hook.py codex") for c in stop_cmds)
    assert any(c.endswith("after-edit-codex") for c in commands(codex["PostToolUse"]))
    toml = (home / ".codex" / "config.toml").read_text()
    assert toml.count("[mcp_servers.peer-consult]") == 1 and '"serve", "codex"' in toml
    assert 'command = "keep"' in toml and 'command = "old"' not in toml

    assert (home / ".local" / "lib" / "llmjury" / "peer_consult_mcp.py").exists()


def test_install_is_idempotent_and_backs_up_only_on_change(tmp_path):
    home = tmp_path / "home"
    seed(home)
    first = install.install(home, python="/py")
    snapshot = {p: p.read_text() for p in home.rglob("*") if p.is_file() and ".bak-" not in p.name}
    second = install.install(home, python="/py")
    assert first and not second
    assert all(p.read_text() == text for p, text in snapshot.items())
    assert list(home.rglob("settings.json.bak-*"))


def test_dry_run_changes_nothing(tmp_path):
    home = tmp_path / "home"
    seed(home)
    before = {p: p.read_text() for p in home.rglob("*") if p.is_file()}
    changed = install.install(home, python="/py", dry_run=True)
    assert changed
    assert {p: p.read_text() for p in home.rglob("*") if p.is_file()} == before


def test_rerun_leaves_installed_files_alone_when_only_layout_differs(tmp_path):
    home = tmp_path / "home"
    seed(home)
    install.install(home, python="/py")
    # Another tool re-added its own hook after ours, a setting holds non-ASCII text,
    # and a file lost its final newline. None of that changes what the hooks do.
    hooks_file = home / ".cursor" / "hooks.json"
    hooks = json.loads(hooks_file.read_text())
    hooks["hooks"]["stop"].append({"command": "claude-mem hook cursor summarize"})
    hooks_file.write_text(json.dumps(hooks, indent=2))
    settings_file = home / ".claude" / "settings.json"
    settings = json.loads(settings_file.read_text())
    settings["note"] = "keep this — as typed"
    settings_file.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    mcp_file = home / ".cursor" / "mcp.json"
    mcp_file.write_text(mcp_file.read_text().rstrip("\n"))
    before = {p: p.read_text() for p in (hooks_file, settings_file, mcp_file)}

    assert install.install(home, python="/py") == []
    assert {p: p.read_text() for p in before} == before


def test_changed_file_keeps_non_ascii_text_readable(tmp_path):
    home = tmp_path / "home"
    seed(home)
    settings_file = home / ".claude" / "settings.json"
    settings = json.loads(settings_file.read_text())
    settings["note"] = "keep this — as typed"
    settings_file.write_text(json.dumps(settings))
    install.install(home, python="/py")
    assert "keep this — as typed" in settings_file.read_text()
