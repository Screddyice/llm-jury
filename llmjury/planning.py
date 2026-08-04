"""Read-only Claude planning for Codex execution sessions."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .cliproc import run_cli


PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["planned", "blocked"]},
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "objective": {"type": "string"},
                    "acceptance": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "objective", "acceptance", "files"],
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "summary", "steps", "risks", "questions"],
}


PLAN_PROMPT = """\
You are the planning agent for a Codex execution session. Do not modify files.

Read the applicable AGENTS.md, CLAUDE.md, and repository instructions, then inspect
only enough code to produce a concrete execution plan for the task below. Resolve
architecture, ordering, boundaries, likely files, acceptance criteria, and checks.
Keep steps bounded enough for Codex to execute and verify independently.

If the task includes execution evidence from an earlier attempt, dynamically replan:
preserve completed work, address the new evidence, and return only the remaining or
corrective steps. Mark the plan blocked only when a user decision or unavailable
external fact is genuinely required. Do not write code, edit files, or invoke another
agent. Return the structured plan requested by the output schema.

TASK AND CURRENT EVIDENCE
-------------------------
{task}
"""


@dataclass
class PlanResult:
    status: str
    summary: str
    steps: list[dict]
    risks: list[str]
    questions: list[str]
    returncode: int = 0

    def to_dict(self):
        return asdict(self)


def validate_plan(payload):
    required = {"status", "summary", "steps", "risks", "questions"}
    if not isinstance(payload, dict) or set(payload) != required:
        return False
    if payload["status"] not in ("planned", "blocked"):
        return False
    if not isinstance(payload["summary"], str):
        return False
    if not all(isinstance(payload[name], list) for name in ("steps", "risks", "questions")):
        return False
    if not all(isinstance(item, str) for name in ("risks", "questions") for item in payload[name]):
        return False
    step_keys = {"id", "objective", "acceptance", "files"}
    return all(
        isinstance(step, dict)
        and set(step) == step_keys
        and all(isinstance(step[name], str) for name in ("id", "objective", "acceptance"))
        and isinstance(step["files"], list)
        and all(isinstance(path, str) for path in step["files"])
        for step in payload["steps"]
    )


class ClaudePlanner:
    """Ask Claude Code for a schema-validated plan without granting write tools."""

    def __init__(self, executable="claude", timeout=900, runner=None):
        self.executable = executable
        self.timeout = timeout
        self.runner = runner or subprocess.run
        if runner is None and not shutil.which(executable):
            raise RuntimeError("Claude Code not found. Install and authenticate `claude` first.")

    def plan(self, task, workspace, model=None, effort="high"):
        workspace = Path(workspace).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        if not task or not task.strip():
            raise ValueError("planning task is empty")
        cmd = [
            self.executable, "--print",
            "--output-format", "json",
            "--json-schema", json.dumps(PLAN_SCHEMA, separators=(",", ":")),
            "--permission-mode", "plan",
            "--tools", "Read,Glob,Grep",
            "--no-session-persistence",
            "--effort", effort,
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append(PLAN_PROMPT.format(task=task.strip()))
        outcome = run_cli(self.runner, cmd, self.timeout, cwd=str(workspace))
        if outcome.timed_out:
            return PlanResult("blocked", f"Claude timed out after {self.timeout}s", [], [],
                              ["The planning call exceeded its timeout."], 124)
        if outcome.os_error is not None:
            return PlanResult("blocked", "Claude could not be started", [], [],
                              [str(outcome.os_error)], 127)
        completed = outcome.completed

        payload = self._payload(completed.stdout)
        if not validate_plan(payload):
            detail = (completed.stderr or completed.stdout or "").strip()
            if len(detail) > 1000:
                detail = detail[-1000:]
            return PlanResult(
                "blocked", f"Claude exited with status {completed.returncode} without a valid plan.",
                [], [], [detail or "No diagnostic output was returned."], completed.returncode,
            )
        status = payload["status"] if completed.returncode == 0 else "blocked"
        questions = list(payload["questions"])
        if completed.returncode != 0 and not questions:
            questions.append(f"Claude exited with status {completed.returncode}.")
        return PlanResult(status, payload["summary"], list(payload["steps"]),
                          list(payload["risks"]), questions, completed.returncode)

    @staticmethod
    def _payload(stdout):
        try:
            envelope = json.loads(stdout or "")
        except json.JSONDecodeError:
            return None
        if validate_plan(envelope):
            return envelope
        if isinstance(envelope, dict):
            structured = envelope.get("structured_output")
            if validate_plan(structured):
                return structured
            result = envelope.get("result")
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    pass
        return None
