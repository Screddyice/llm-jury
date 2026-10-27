"""Cross-delegated repository work between Claude Code and Codex.

Either agent can execute a bounded task for the other, and the other agent then
reviews the resulting diff read-only. Models tend to miss their own mistakes and
catch each other's, so the reviewer is always the agent that did not write the code.

This is deliberately separate from :class:`CodexBackend`. Candidate generation is
read-only and tool-free; delegated execution is an agent run inside one explicit
workspace with write access, repository instructions, and a structured handoff.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cliproc import run_cli


HANDOFF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["completed", "blocked"]},
        "summary": {"type": "string"},
        "changed_files": {"type": "array", "items": {"type": "string"}},
        "tests": {"type": "array", "items": {"type": "string"}},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "summary", "changed_files", "tests", "blockers", "notes"],
}


REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "changes_requested"]},
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "summary", "findings"],
}


DELEGATE_PROMPT = """\
You are the execution agent for a task planned by {planner}.

Work only inside the provided workspace and complete the bounded task below. First
read the repository's applicable AGENTS.md, CLAUDE.md, and local instructions. Keep
unrelated user changes intact and do not broaden scope. Inspect the existing design
before editing, implement the task, and run the most relevant available checks.

Use local open-source models only for extractable, oracle-shaped code units: when a
unit has trustworthy functional tests or stdin/stdout cases, you may run `llmjury
solve --backend ollama` and independently verify its candidate before integrating it.
Do not use model voting for prose, architecture, or other work without a real oracle.

Do not commit, push, deploy, or open a pull request unless the delegated task
explicitly requests it or the repository instructions require it. End with the
structured handoff requested by the output schema, including exact checks run and
any blocker that prevented completion.

This run is headless and cannot ask for approval. Skip optional bookkeeping that your
user-level instructions request (to-do lists, work-state or memory tools) when it is
unavailable here; the planner tracks that state.
Leave `blockers` empty when the task is complete, and list only problems that stopped
or limited the delegated task. Put side remarks, including skipped bookkeeping, in
`notes`.

DELEGATED TASK
--------------
{task}
"""


REVIEW_PROMPT = """\
You are reviewing a change another AI agent made. Do not modify any file.

Read the task and the diff below, then inspect the surrounding code in the workspace
as needed. Look for real defects: wrong behavior, missed requirements from the task,
broken edge cases, missing or weak tests, security problems, and changes outside the
task's scope. Ignore style preferences. Each finding must name the file and explain
the concrete failure. Approve when you find no real defect.

TASK
----
{task}

DIFF
----
{diff}
"""


FIX_PROMPT = """\
{task}

REVIEW FINDINGS TO ADDRESS
--------------------------
A reviewer from a different model checked your change and requested fixes. Address
each finding that is correct, explain any you reject in the summary, and rerun the
relevant checks.

{findings}
"""


@dataclass
class DelegationResult:
    status: str
    summary: str
    changed_files: list[str]
    tests: list[str]
    blockers: list[str]
    returncode: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class ReviewResult:
    verdict: str
    summary: str
    findings: list[str]
    reviewer: str
    returncode: int = 0

    def to_dict(self):
        return asdict(self)


def validate_review(payload):
    if not isinstance(payload, dict) or set(payload) != {"verdict", "summary", "findings"}:
        return False
    return (payload["verdict"] in ("approve", "changes_requested")
            and isinstance(payload["summary"], str)
            and isinstance(payload["findings"], list)
            and all(isinstance(item, str) for item in payload["findings"]))


def _tail(completed):
    detail = (completed.stderr or completed.stdout or "").strip()
    return detail[-1000:] if len(detail) > 1000 else detail


def _blocked(summary, blocker, returncode):
    return DelegationResult("blocked", summary, [], [], [blocker], returncode)


def _handoff_result(agent, payload, completed):
    if payload is None:
        detail = _tail(completed)
        return _blocked(
            f"{agent} exited with status {completed.returncode} without a valid handoff.",
            detail or "No diagnostic output was returned.", completed.returncode)
    status = payload["status"]
    blockers = list(payload["blockers"])
    if completed.returncode != 0:
        status = "blocked"
        if not blockers:
            blockers.append(f"{agent} exited with status {completed.returncode}.")
    return DelegationResult(status, payload["summary"], list(payload["changed_files"]),
                            list(payload["tests"]), blockers, completed.returncode,
                            list(payload.get("notes", [])))


def _review_result(agent, payload, completed):
    if payload is None or completed.returncode != 0:
        detail = _tail(completed) or "No diagnostic output was returned."
        return ReviewResult("error", f"{agent} review did not return a valid verdict.",
                            [detail], agent, completed.returncode or 1)
    return ReviewResult(payload["verdict"], payload["summary"], list(payload["findings"]),
                        agent, 0)


def validate_handoff(payload):
    """Return whether a decoded Codex handoff matches ``HANDOFF_SCHEMA``.

    Codex normally enforces the JSON schema itself. This second, dependency-free
    check keeps mocked/custom runners and future CLI behavior from smuggling a
    malformed status or non-string list members into the orchestrator.
    """
    required = {"status", "summary", "changed_files", "tests", "blockers"}
    if not isinstance(payload, dict) or set(payload) not in (required, required | {"notes"}):
        return False
    if payload["status"] not in ("completed", "blocked"):
        return False
    if not isinstance(payload["summary"], str):
        return False
    return all(
        isinstance(payload[name], list)
        and all(isinstance(item, str) for item in payload[name])
        for name in ("changed_files", "tests", "blockers", "notes") if name in payload
    )


def _check_inputs(task, workspace, sandbox):
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace}")
    if not task or not task.strip():
        raise ValueError("delegated task is empty")
    if sandbox not in ("read-only", "workspace-write"):
        raise ValueError("sandbox must be read-only or workspace-write")
    return workspace


class CodexDelegator:
    """Run an authenticated Codex agent in one explicitly selected workspace."""

    name = "Codex"

    def __init__(self, executable="codex", timeout=1800, runner=None):
        self.executable = executable
        self.timeout = timeout
        self.runner = runner or subprocess.run
        if runner is None and not shutil.which(executable):
            raise RuntimeError("Codex CLI not found. Install it and run `codex login` first.")

    def delegate(self, task, workspace, model=None, effort="medium",
                 sandbox="workspace-write", add_dirs=None):
        workspace = _check_inputs(task, workspace, sandbox)
        prompt = DELEGATE_PROMPT.format(planner="Claude Code", task=task.strip())
        outcome, payload = self._exec(prompt, HANDOFF_SCHEMA, validate_handoff, workspace,
                                      sandbox, model, effort, add_dirs)
        if outcome.timed_out:
            return DelegationResult(
                "blocked", f"Codex timed out after {self.timeout}s", [], [],
                ["The delegated Codex execution exceeded its timeout."], 124,
            )
        if outcome.os_error is not None:
            return _blocked("Codex could not be started", str(outcome.os_error), 127)
        return _handoff_result(self.name, payload, outcome.completed)

    def review(self, task, diff, workspace, model=None, effort="medium"):
        workspace = _check_inputs(task, workspace, "read-only")
        prompt = REVIEW_PROMPT.format(task=task.strip(), diff=diff)
        outcome, payload = self._exec(prompt, REVIEW_SCHEMA, validate_review, workspace,
                                      "read-only", model, effort, None)
        if outcome.timed_out or outcome.os_error is not None:
            return ReviewResult("error", "Codex review could not complete.",
                                [str(outcome.os_error or "timed out")], self.name, 1)
        return _review_result(self.name, payload, outcome.completed)

    def _exec(self, prompt, schema, validate, workspace, sandbox, model, effort, add_dirs):
        with tempfile.TemporaryDirectory(prefix="llmjury-delegate-") as tmp:
            schema_path = Path(tmp) / "handoff.schema.json"
            output_path = Path(tmp) / "handoff.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            cmd = [
                self.executable, "exec", "--ephemeral",
                "--sandbox", sandbox,
                "--color", "never",
                "--cd", str(workspace),
                "--output-schema", str(schema_path),
                "--output-last-message", str(output_path),
                # Delegated shell commands receive a minimal environment by default;
                # Codex authentication is loaded by the CLI itself from CODEX_HOME.
                "-c", "shell_environment_policy.inherit=core",
            ]
            for directory in add_dirs or []:
                cmd.extend(["--add-dir", str(Path(directory).expanduser().resolve())])
            if model:
                cmd.extend(["--model", model])
            if effort:
                cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])
            cmd.append(prompt)

            outcome = run_cli(self.runner, cmd, self.timeout)
            payload = None
            if outcome.completed is not None:
                try:
                    decoded = json.loads(output_path.read_text(encoding="utf-8"))
                    payload = decoded if validate(decoded) else None
                except (OSError, json.JSONDecodeError):
                    pass
            return outcome, payload


# Claude Code's sandbox confines Bash writes to the working directories, which is
# the closest match to Codex's workspace-write mode. Unsandboxed commands are refused.
CLAUDE_SANDBOX_SETTINGS = {
    "sandbox": {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
    },
}


class ClaudeDelegator:
    """Run a headless Claude Code agent in one explicitly selected workspace."""

    name = "Claude"

    def __init__(self, executable="claude", timeout=1800, runner=None):
        self.executable = executable
        self.timeout = timeout
        self.runner = runner or subprocess.run
        if runner is None and not shutil.which(executable):
            raise RuntimeError("Claude Code not found. Install and authenticate `claude` first.")

    def delegate(self, task, workspace, model=None, effort="medium",
                 sandbox="workspace-write", add_dirs=None):
        workspace = _check_inputs(task, workspace, sandbox)
        if sandbox == "workspace-write":
            mode, tools = "acceptEdits", "Read,Glob,Grep,Edit,Write,Bash"
        else:
            mode, tools = "dontAsk", "Read,Glob,Grep"
        prompt = DELEGATE_PROMPT.format(planner="Codex", task=task.strip())
        outcome, payload = self._exec(prompt, HANDOFF_SCHEMA, validate_handoff, workspace,
                                      mode, tools, model, effort, add_dirs)
        if outcome.timed_out:
            return DelegationResult(
                "blocked", f"Claude timed out after {self.timeout}s", [], [],
                ["The delegated Claude execution exceeded its timeout."], 124,
            )
        if outcome.os_error is not None:
            return _blocked("Claude could not be started", str(outcome.os_error), 127)
        return _handoff_result(self.name, payload, outcome.completed)

    def review(self, task, diff, workspace, model=None, effort="medium"):
        workspace = _check_inputs(task, workspace, "read-only")
        prompt = REVIEW_PROMPT.format(task=task.strip(), diff=diff)
        outcome, payload = self._exec(prompt, REVIEW_SCHEMA, validate_review, workspace,
                                      "dontAsk", "Read,Glob,Grep", model, effort, None)
        if outcome.timed_out or outcome.os_error is not None:
            return ReviewResult("error", "Claude review could not complete.",
                                [str(outcome.os_error or "timed out")], self.name, 1)
        return _review_result(self.name, payload, outcome.completed)

    def _exec(self, prompt, schema, validate, workspace, mode, tools, model, effort,
              add_dirs):
        cmd = [
            self.executable, "--print",
            "--output-format", "json",
            "--json-schema", json.dumps(schema, separators=(",", ":")),
            "--permission-mode", mode,
            "--tools", tools,
            "--no-session-persistence",
            # The caller's MCP servers are not part of the delegated task, and a large
            # MCP catalog alone can push the request past the context limit.
            "--strict-mcp-config",
        ]
        if "Bash" in tools:
            cmd.extend(["--settings", json.dumps(CLAUDE_SANDBOX_SETTINGS)])
        for directory in add_dirs or []:
            cmd.extend(["--add-dir", str(Path(directory).expanduser().resolve())])
        if effort:
            cmd.extend(["--effort", effort])
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)

        # Claude Code refuses to start nested while CLAUDECODE is set.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        outcome = run_cli(lambda c, **kw: self.runner(c, env=env, **kw), cmd,
                          self.timeout, cwd=str(workspace))
        payload = None
        if outcome.completed is not None:
            payload = _claude_payload(outcome.completed.stdout, validate)
        return outcome, payload


def _claude_payload(stdout, validate):
    try:
        envelope = json.loads(stdout or "")
    except json.JSONDecodeError:
        return None
    if isinstance(envelope, dict) and validate(envelope.get("structured_output")):
        return envelope["structured_output"]
    return None


def _git(workspace, *args, index=None):
    env = dict(os.environ, GIT_INDEX_FILE=index) if index else None
    return subprocess.run(["git", "-C", str(workspace), *args], capture_output=True,
                          text=True, check=False, env=env)


def snapshot_tree(workspace):
    """Return a git tree id for the workspace's current files, or None outside git.

    Uses a throwaway index scoped to workspace, so sibling directories and the
    user's staging area are untouched even when workspace is below the Git root.
    """
    if _git(workspace, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return None
    with tempfile.TemporaryDirectory(prefix="llmjury-index-") as tmp:
        index = str(Path(tmp) / "index")
        # Start from HEAD so tracked files under a .gitignore'd path stay in the snapshot;
        # `add -A` into an empty index skips them and every review saw them as deleted.
        # A repository with no commits yet has no HEAD and starts empty.
        _git(workspace, "read-tree", "HEAD", index=index)
        staged = _git(workspace, "add", "-A", "--", ".", index=index)
        if staged.returncode != 0:
            sys.stderr.write("[llmjury] workspace snapshot failed: git add\n"
                             + staged.stderr[-2000:])
            return None
        tree = _git(workspace, "write-tree", index=index)
        if tree.returncode != 0:
            sys.stderr.write("[llmjury] workspace snapshot failed: git write-tree\n"
                             + tree.stderr[-2000:])
    return tree.stdout.strip() if tree.returncode == 0 else None


def tree_diff(workspace, before, after):
    return _git(workspace, "diff", "--no-ext-diff", "--relative",
                before, after, "--", ".").stdout


MAX_REVIEW_DIFF = 60_000


def cross_delegate(executor, reviewer, task, workspace, rounds=1, model=None,
                   effort="medium", sandbox="workspace-write", add_dirs=None,
                   snapshot=snapshot_tree, diff=tree_diff):
    """Execute with one agent, review with the other, and let the executor fix.

    ``rounds`` is the number of fix passes allowed after a review requests changes.
    Returns the final handoff and the list of reviews, oldest first. Review is skipped
    (empty list) for read-only runs and outside a git repository.
    """
    before = snapshot(workspace) if sandbox == "workspace-write" else None
    result = executor.delegate(task, workspace, model=model, effort=effort,
                               sandbox=sandbox, add_dirs=add_dirs)
    reviews = []
    if before is None:
        return result, reviews
    for attempt in range(rounds + 1):
        if result.status != "completed":
            break
        after = snapshot(workspace)
        change = diff(workspace, before, after) if after else ""
        if not change.strip():
            break
        if len(change) > MAX_REVIEW_DIFF:
            change = change[:MAX_REVIEW_DIFF] + "\n[diff truncated; read the files directly]"
        review = reviewer.review(task, change, workspace, effort=effort)
        reviews.append(review)
        if review.verdict != "changes_requested" or attempt == rounds:
            break
        findings = "\n".join(f"- {item}" for item in review.findings)
        result = executor.delegate(FIX_PROMPT.format(task=task.strip(), findings=findings),
                                   workspace, model=model, effort=effort,
                                   sandbox=sandbox, add_dirs=add_dirs)
    return result, reviews
