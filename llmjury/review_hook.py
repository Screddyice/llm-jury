"""Stop hook: when one agent finishes a turn that changed code, the other reviews it.

Claude Code and Codex share the same Stop-hook contract. The hook reads one JSON object
on stdin and, to keep the agent working, prints ``{"decision": "block", "reason": ...}``;
the reason becomes the agent's next prompt. Every other outcome exits 0 silently, so a
missing CLI, a timeout, or an unreadable repository never blocks the user's session.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .delegation import (ClaudeDelegator, CodexDelegator, MAX_REVIEW_DIFF,
                         snapshot_tree, tree_diff)

# Set for every agent the hook or `delegate` starts, so their own Stop hooks stay quiet.
NESTED_ENV = "LLMJURY_NESTED"
OPT_OUT_ENV = "LLMJURY_CROSS_REVIEW"
STATE_DIR = Path("~/.llmjury/review-hook").expanduser()

REVIEW_TASK = """\
Review the uncommitted change that {author} made in this repository during its last turn.
The diff is everything {author} changed since the previous review in this session.

{author}'s own summary of the turn:
{summary}
"""

FIX_REASON = """\
{reviewer} reviewed the change you just made and requested fixes:

{findings}

Fix each finding that is correct and rerun the relevant checks. If a finding is wrong,
say why in one line instead of changing the code.
"""


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                          check=False)


def _excluded(cwd):
    """RS21 repositories take no automation of any kind."""
    remote = _git(cwd, "remote", "get-url", "origin").stdout.lower()
    return "rs21" in remote


def _state_path(session_id, cwd):
    digest = hashlib.sha256(f"{session_id}\0{cwd}".encode()).hexdigest()[:24]
    return STATE_DIR / f"{digest}.tree"


def decide(event, author_name, reviewer, snapshot=snapshot_tree, diff=tree_diff):
    """Return the hook's stdout payload, or None to let the agent stop."""
    if os.environ.get(NESTED_ENV) or os.environ.get(OPT_OUT_ENV, "").lower() == "off":
        return None
    # One review per turn: a turn the hook already continued is the fix pass.
    if event.get("stop_hook_active"):
        return None
    cwd = event.get("cwd") or os.getcwd()
    if _excluded(cwd):
        return None
    current = snapshot(cwd)
    if current is None:
        return None
    state = _state_path(event.get("session_id") or "nosession", cwd)
    try:
        before = state.read_text(encoding="utf-8").strip()
    except OSError:
        head = _git(cwd, "rev-parse", "HEAD^{tree}")
        before = head.stdout.strip() if head.returncode == 0 else None
    if not before or before == current:
        _remember(state, current)
        return None
    change = diff(cwd, before, current)
    _remember(state, current)
    if not change.strip():
        return None
    if len(change) > MAX_REVIEW_DIFF:
        change = change[:MAX_REVIEW_DIFF] + "\n[diff truncated; read the files directly]"

    os.environ[NESTED_ENV] = "1"
    summary = (event.get("last_assistant_message") or "(none)").strip()[:4000]
    task = REVIEW_TASK.format(author=author_name, summary=summary)
    review = reviewer.review(task, change, cwd)
    sys.stderr.write(f"[llmjury] {reviewer.name} review: {review.verdict}\n")
    if review.verdict != "changes_requested" or not review.findings:
        return None
    findings = "\n".join(f"- {item}" for item in review.findings)
    return {"decision": "block",
            "reason": FIX_REASON.format(reviewer=reviewer.name, findings=findings)}


def _remember(state, tree):
    try:
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(tree, encoding="utf-8")
    except OSError:
        pass


def main(host, stdin=sys.stdin, timeout=540):
    """Entry point for ``llmjury review-hook --host claude|codex``."""
    try:
        event = json.loads(stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    if host == "claude":
        author_name, reviewer_cls = ClaudeDelegator.name, CodexDelegator
    else:
        author_name, reviewer_cls = CodexDelegator.name, ClaudeDelegator
    try:
        payload = decide(event, author_name, reviewer_cls(timeout=timeout))
    except Exception as exc:  # a review hook must never break the session
        sys.stderr.write(f"[llmjury] cross-review skipped: {exc}\n")
        return 0
    if payload:
        print(json.dumps(payload))
    return 0
