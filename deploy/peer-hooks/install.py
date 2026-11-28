"""Register the Claude Code, Cursor and Codex peer hooks on this machine.

Run with the Python that has llm-jury-verify installed:

    python deploy/peer-hooks/install.py [--dry-run]

It copies the two hook scripts to ~/.local/lib/llmjury/, then adds or refreshes the
peer-consult MCP server and the first-edit and end-of-turn hooks for each agent. Other
hooks and servers are left alone, a file is backed up before it changes, and a second
run changes nothing. New or changed Codex hooks run only after you trust them in
Codex's /hooks screen.
"""

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
SCRIPTS = ("provider_review_hook.py", "peer_consult_mcp.py")
OURS = ("provider_review_hook.py", "peer_consult_mcp.py", "review-hook --host codex")
REVIEW_TIMEOUT = 600
EDIT_TIMEOUT = 300


def claude_dirs(home):
    """~/.claude plus the Claude profile named in peer-review.json, if any."""
    dirs = [home / ".claude"]
    try:
        config = json.loads((home / ".config" / "llmjury" / "peer-review.json").read_text())
        extra = Path(config.get("claudeConfigDir", "")).expanduser()
        if str(extra) not in ("", ".") and extra.is_dir() and extra not in dirs:
            dirs.append(extra)
    except (OSError, ValueError, AttributeError):
        pass
    return dirs


class Writer:
    def __init__(self, dry_run):
        self.dry_run, self.changed = dry_run, []
        self.stamp = time.strftime("%Y%m%d-%H%M%S")

    def write(self, path, text):
        old = path.read_text() if path.exists() else None
        if old == text:
            return
        self.changed.append(path)
        if self.dry_run:
            return
        if old is not None:
            path.with_name(f"{path.name}.bak-{self.stamp}-peer").write_text(old)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def json(self, path, update):
        """Apply update; leave the file untouched when its parsed content would not change."""
        data = json.loads(path.read_text()) if path.exists() else {}
        before = copy.deepcopy(data)
        update(data)
        if path.exists() and data == before:
            return
        self.write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def is_ours(command):
    return any(marker in command for marker in OURS)


def nested(command, timeout, matcher=None):
    entry = {"hooks": [{"type": "command", "command": command, "timeout": timeout}]}
    if matcher:
        entry["matcher"] = matcher
    return entry


def already_installed(items, entry, ours):
    """True when entry is our only registration, wherever it sits among other tools' hooks."""
    mine = [item for item in items if ours(item)]
    return mine == [entry]


def set_nested_hook(data, event, entry):
    """Claude/Codex layout: hooks[event] = [{matcher?, hooks: [{command}]}]."""
    current = data.setdefault("hooks", {}).get(event, [])
    if already_installed(current, entry, lambda item: any(
            is_ours(hook.get("command", "")) for hook in item.get("hooks", []))):
        return
    entries = []
    for item in current:
        kept = [hook for hook in item.get("hooks", []) if not is_ours(hook.get("command", ""))]
        if kept:
            entries.append({**item, "hooks": kept})
    data["hooks"][event] = entries + [entry]


def set_flat_hook(data, event, entry):
    """Cursor layout: hooks[event] = [{command, matcher?, timeout?}]."""
    current = data.setdefault("hooks", {}).get(event, [])
    if already_installed(current, entry, lambda hook: is_ours(hook.get("command", ""))):
        return
    kept = [hook for hook in current if not is_ours(hook.get("command", ""))]
    data["hooks"][event] = kept + [entry]


def mcp_server(python, lib, author):
    return {"type": "stdio", "command": python,
            "args": [str(lib / "peer_consult_mcp.py"), "serve", author], "env": {}}


def codex_toml(text, python, lib):
    """Replace any [mcp_servers.peer-consult] tables with one current block."""
    text = re.sub(r"(?ms)^\[mcp_servers\.peer-consult(?:\.[^\]]+)?\]\n.*?(?=^\[|\Z)", "", text)
    block = (f'[mcp_servers.peer-consult]\ncommand = {json.dumps(python)}\n'
             f'args = [{json.dumps(str(lib / "peer_consult_mcp.py"))}, "serve", "codex"]\n')
    return text.rstrip("\n") + "\n\n" + block


def install(home, python, dry_run=False):
    """Install or refresh everything; returns the list of files that changed."""
    home = Path(home)
    lib = home / ".local" / "lib" / "llmjury"
    out = Writer(dry_run)
    for name in SCRIPTS:
        out.write(lib / name, (HERE / name).read_text())
    review = lambda author: f"{python} {lib / 'provider_review_hook.py'} {author}"  # noqa: E731
    edit = lambda author: f"{python} {lib / 'peer_consult_mcp.py'} after-edit-{author}"  # noqa: E731

    for directory in claude_dirs(home):
        def claude_hooks(data):
            set_nested_hook(data, "PostToolUse", nested(edit("claude"), EDIT_TIMEOUT, "Edit|Write|MultiEdit"))
            set_nested_hook(data, "Stop", nested(review("claude"), REVIEW_TIMEOUT))
        out.json(directory / "settings.json", claude_hooks)
        mcp_file = home / ".claude.json" if directory == home / ".claude" else directory / ".claude.json"
        out.json(mcp_file, lambda d: d.setdefault("mcpServers", {}).__setitem__(
            "peer-consult", mcp_server(python, lib, "claude")))

    cursor = home / ".cursor"
    if cursor.is_dir():
        def cursor_hooks(data):
            data.setdefault("version", 1)
            set_flat_hook(data, "postToolUse", {"command": edit("cursor"), "timeout": EDIT_TIMEOUT,
                                                "matcher": "Write|Edit|StrReplace|ApplyPatch"})
            set_flat_hook(data, "stop", {"command": review("cursor"), "timeout": REVIEW_TIMEOUT,
                                         "loop_limit": 1})
        out.json(cursor / "hooks.json", cursor_hooks)
        out.json(cursor / "mcp.json", lambda d: d.setdefault("mcpServers", {}).__setitem__(
            "peer-consult", mcp_server(python, lib, "cursor")))

    codex = home / ".codex"
    if codex.is_dir():
        def codex_hooks(data):
            set_nested_hook(data, "PostToolUse", nested(edit("codex"), EDIT_TIMEOUT))
            set_nested_hook(data, "Stop", nested(review("codex"), REVIEW_TIMEOUT))
        out.json(codex / "hooks.json", codex_hooks)
        config = codex / "config.toml"
        out.write(config, codex_toml(config.read_text() if config.exists() else "", python, lib))
    return out.changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list the files that would change")
    parser.add_argument("--python", default=sys.executable, help="Python with llm-jury-verify installed")
    args = parser.parse_args(argv)
    changed = install(Path.home(), args.python, args.dry_run)
    verb = "Would change" if args.dry_run else "Changed"
    print(f"{verb} {len(changed)} file(s):" if changed else "Already up to date.")
    for path in changed:
        print(f"  {path}")
    if changed and not args.dry_run and shutil.which("codex"):
        print("Open Codex and trust the new peer hooks in /hooks; restart running agents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
