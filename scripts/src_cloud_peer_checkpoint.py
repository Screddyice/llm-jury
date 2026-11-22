"""Return a peer's cloud review to the author immediately after a coding edit."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


SERVICE = "https://os.shawn-reddy.com/peer-review"
REPO = "Screddyice/llm-jury"
ORIGIN = re.compile(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)screddyice/llm-jury(?:\.git)?", re.I)
TOKEN_PATTERN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-|AKIA)[\w-]{16,}", re.I)
SENSITIVE_NAME = re.compile(r"secret|credential|private|id_ed25519|id_rsa", re.I)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def git(cwd, *args):
    return subprocess.run(["git", "--literal-pathspecs", *args], cwd=cwd, capture_output=True,
                          text=True, check=True, timeout=5).stdout.strip()


def excerpt(cwd, event):
    if not ORIGIN.fullmatch(git(cwd, "remote", "get-url", "origin")):
        return ""
    root = Path(git(cwd, "rev-parse", "--show-toplevel")).resolve()
    inputs = event.get("tool_input") or {}
    filename = inputs.get("file_path") or inputs.get("path") or event.get("file_path")
    if not isinstance(filename, str) or not filename:
        return ""
    touched = (Path(cwd) / filename).resolve()
    if not touched.is_relative_to(root) or not touched.is_file():
        return ""
    path = touched.relative_to(root)
    if (path.suffix not in {".py", ".js", ".ts", ".tsx", ".sh"}
        or any(part.startswith(".") for part in path.parts) or SENSITIVE_NAME.search(str(path))):
        return ""
    changed = inputs.get("new_string") or inputs.get("content") or inputs.get("streamContent")
    if not isinstance(changed, str) or not changed.strip():
        changed = git(root, "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--", str(path))
    if not changed and touched.stat().st_size <= 8000:
        changed = touched.read_text() if git(root, "ls-files", "--others", "--exclude-standard", "--", str(path)) else ""
    if not changed or len(changed) > 8000 or TOKEN_PATTERN.search(changed):
        return ""
    return f"Edited file {path.as_posix()}\n{changed}"


def once(event, host):
    session = event.get("generation_id") or event.get("session_id") or event.get("conversation_id")
    if not isinstance(session, str) or not session:
        return False
    state = Path.home() / ".cache" / "src-peer-consult"
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        with (state / hashlib.sha256(f"{host}:{session}".encode()).hexdigest()).open("x"):
            return True
    except FileExistsError:
        return False


def request(path, payload, token):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(f"{SERVICE}/{path}", data=json.dumps(payload).encode(), headers=headers)
    with build_opener(NoRedirect).open(req, timeout=20) as response:
        return json.loads(response.read(65536))


def consult(host, token, context):
    target = "claude" if host == "cursor" else "cursor"
    reviewer = "Claude Code Cloud" if target == "claude" else "Cursor Cloud"
    try:
        receipt = request(f"{target}/start", {"repository": REPO, "question": "Check this first coding edit for correctness before I continue.", "context": context}, token)
        deadline = monotonic() + 180
        while receipt["status"] == "pending" and monotonic() < deadline:
            receipt = request(f"{target}/result", {"request_id": receipt["request_id"]}, token)
        if receipt["status"] == "complete":
            return (f"Same-turn review from {reviewer}. Treat the review as advisory and verify it against your code.\n"
                    + json.dumps(receipt))
        return f"The in-session review from {reviewer} did not return a verdict. Continue with independent validation."
    except HTTPError as exc:
        return f"The in-session review from {reviewer} was unavailable (HTTP {exc.code}). Continue with independent validation."
    except (OSError, ValueError, KeyError):
        return f"The in-session review from {reviewer} was unavailable. Continue with independent validation."


def main(host):
    token = os.environ.get("SRC_PEER_CURSOR_TOKEN") if host == "cursor" else None
    if ((host == "cursor" and not token) or
        (host == "claude" and (os.environ.get("CLAUDE_CODE_REMOTE") != "true" or os.environ.get("SRC_PEER_AUTHOR") != "1"))):
        return
    event = json.load(sys.stdin)
    roots = event.get("workspace_roots") or []
    cwd = event.get("cwd") or (roots[0] if len(roots) == 1 else None)
    if not cwd:
        return
    try:
        context = excerpt(cwd, event)
        if not context or not once(event, host):
            return
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    reply = consult(host, token, context)
    if host == "cursor":
        print(json.dumps({"additional_context": reply}))
    else:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": reply}}))


if __name__ == "__main__":
    main(sys.argv[1])
