"""Claude-planned, Codex-executed repository work.

This is deliberately separate from :class:`CodexBackend`. Candidate generation is
read-only and tool-free; delegated execution is an agent run inside one explicit
workspace with write access, repository instructions, and a structured handoff.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
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
    },
    "required": ["status", "summary", "changed_files", "tests", "blockers"],
}


DELEGATE_PROMPT = """\
You are the execution agent for a task planned by Claude Code.

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

DELEGATED TASK
--------------
{task}
"""


@dataclass
class DelegationResult:
    status: str
    summary: str
    changed_files: list[str]
    tests: list[str]
    blockers: list[str]
    returncode: int = 0

    def to_dict(self):
        return asdict(self)


def validate_handoff(payload):
    """Return whether a decoded Codex handoff matches ``HANDOFF_SCHEMA``.

    Codex normally enforces the JSON schema itself. This second, dependency-free
    check keeps mocked/custom runners and future CLI behavior from smuggling a
    malformed status or non-string list members into the orchestrator.
    """
    required = {"status", "summary", "changed_files", "tests", "blockers"}
    if not isinstance(payload, dict) or set(payload) != required:
        return False
    if payload["status"] not in ("completed", "blocked"):
        return False
    if not isinstance(payload["summary"], str):
        return False
    return all(
        isinstance(payload[name], list)
        and all(isinstance(item, str) for item in payload[name])
        for name in ("changed_files", "tests", "blockers")
    )


class CodexDelegator:
    """Run an authenticated Codex agent in one explicitly selected workspace."""

    def __init__(self, executable="codex", timeout=1800, runner=None):
        self.executable = executable
        self.timeout = timeout
        self.runner = runner or subprocess.run
        if runner is None and not shutil.which(executable):
            raise RuntimeError("Codex CLI not found. Install it and run `codex login` first.")

    def delegate(self, task, workspace, model=None, effort="medium",
                 sandbox="workspace-write", add_dirs=None):
        workspace = Path(workspace).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        if not task or not task.strip():
            raise ValueError("delegated task is empty")
        if sandbox not in ("read-only", "workspace-write"):
            raise ValueError("sandbox must be read-only or workspace-write")

        with tempfile.TemporaryDirectory(prefix="llmjury-delegate-") as tmp:
            schema_path = Path(tmp) / "handoff.schema.json"
            output_path = Path(tmp) / "handoff.json"
            schema_path.write_text(json.dumps(HANDOFF_SCHEMA), encoding="utf-8")
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
            cmd.append(DELEGATE_PROMPT.format(task=task.strip()))

            outcome = run_cli(self.runner, cmd, self.timeout)
            if outcome.timed_out:
                return DelegationResult(
                    "blocked", f"Codex timed out after {self.timeout}s", [], [],
                    ["The delegated Codex execution exceeded its timeout."], 124,
                )
            if outcome.os_error is not None:
                return DelegationResult(
                    "blocked", "Codex could not be started", [], [],
                    [str(outcome.os_error)], 127,
                )
            completed = outcome.completed

            payload = self._read_handoff(output_path, completed)
            status = payload.get("status", "blocked")
            blockers = list(payload.get("blockers") or [])
            if completed.returncode != 0:
                status = "blocked"
                if not blockers:
                    blockers.append(f"Codex exited with status {completed.returncode}.")
            return DelegationResult(
                status=status,
                summary=payload.get("summary", "Codex did not return a valid handoff."),
                changed_files=list(payload.get("changed_files") or []),
                tests=list(payload.get("tests") or []),
                blockers=blockers,
                returncode=completed.returncode,
            )

    @staticmethod
    def _read_handoff(output_path, completed):
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if validate_handoff(payload):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > 1000:
            detail = detail[-1000:]
        return {
            "status": "blocked",
            "summary": f"Codex exited with status {completed.returncode} without a valid handoff.",
            "changed_files": [],
            "tests": [],
            "blockers": [detail] if detail else ["No diagnostic output was returned."],
        }
