"""Install the Codex skill that delegates planning to Claude."""
from __future__ import annotations

import os
from pathlib import Path


SKILL = """\
---
name: llm-jury-orchestrate
description: Automatically use Claude Code to plan non-trivial Codex implementation work, dynamically replan when execution evidence changes, and use the local LLM jury for verifier-shaped code units.
---

# Dynamic LLM-Jury Orchestration

For non-trivial implementation work, use Claude as the planning agent and remain the
execution agent yourself. Skip this only for trivial one-step edits, pure questions,
or when the user explicitly says to skip planning or just execute.

## Initial plan

1. Inspect enough local context to state the user's task accurately.
2. Run `llmjury plan --workspace "$PWD" --task - --json`, passing the task, known
   constraints, and requested outcome on stdin.
3. If Claude returns `blocked`, surface its questions when they genuinely require the
   user. Otherwise execute the returned steps in order and verify each acceptance
   criterion.

Do not call Claude again when a delegated brief already contains a concrete Claude
plan. That path came from `llmjury delegate`; execute it directly unless new evidence
invalidates the plan.

## Dynamic replanning

Call `llmjury plan` again when tests expose a wrong assumption, the relevant code
differs materially from the plan, scope must change, or two focused execution attempts
fail. Include the original goal, completed steps, current diff summary, exact failure
output, and the decision that needs replanning. Continue with only Claude's remaining
or corrective steps; do not discard verified work.

## Local execution assistance

For a self-contained Python unit with a trustworthy functional or stdin/stdout oracle,
run `llmjury solve --backend ollama` before using a cloud frontier. Integrate a local
candidate only after the independent verifier accepts it. Never use the council as a
vote for architecture, prose, product judgment, or other work without a real oracle.

Codex owns edits, tests, diff inspection, and the final handoff. Claude owns planning
and replanning. Repository instructions and the user's authorization govern commits,
external actions, and deployments.
"""


def skill_path():
    root = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    return root / "skills" / "llm-jury-orchestrate" / "SKILL.md"


def install_codex_skill(force=False):
    destination = skill_path()
    if destination.exists():
        current = destination.read_text(encoding="utf-8")
        if current == SKILL:
            return destination, False
        if not force:
            raise FileExistsError(
                f"{destination} already exists with different content; pass --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(SKILL, encoding="utf-8")
    temporary.replace(destination)
    return destination, True
