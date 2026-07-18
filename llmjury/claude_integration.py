"""Install the Claude Code skill that exposes LLM-Jury delegation."""
from __future__ import annotations

import os
from pathlib import Path


SKILL = """\
---
name: llm-jury-delegate
description: Delegate bounded implementation and execution tasks from Claude Code to Codex while keeping local Ollama models verifier-gated for testable code units.
---

# LLM-Jury Delegation

Use Claude for planning, architecture, ambiguity resolution, and final review. Delegate
bounded repository execution to Codex when the work has a clear outcome and can be
checked with tests, lint, builds, or a focused diff review.

## Workflow

1. Inspect the repository and read its applicable instructions.
2. Produce a bounded execution brief with scope, constraints, acceptance criteria,
   relevant files, and required checks.
3. Run `llmjury delegate --workspace "$PWD" --task - --json`, passing the brief on
   stdin. Add `--model MODEL` only when a specific Codex model is required.
4. Inspect Codex's changed files and rerun or extend the important checks yourself.
5. Resolve planning-level issues in Claude. Delegate another bounded execution pass
   only when concrete work remains.

For a self-contained Python unit with a trustworthy oracle, let the execution agent
use `llmjury solve --backend ollama` so Phi-4 and the local cross-lineage council get
the first chance. The verifier, never a vote, decides whether their code is usable.
Do not route prose, product judgment, architecture, secrets, deployments, or work
without a reliable oracle through the local jury.

Do not delegate destructive operations or permission expansion. Codex defaults to
workspace-write confinement and receives only a minimal shell environment. External
writes, commits, pushes, PRs, and deployments remain governed by the user's request
and the repository instructions.
"""


def skill_path(scope="user", project=None):
    if scope == "user":
        root = Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()
    elif scope == "project":
        root = Path(project or os.getcwd()).expanduser().resolve() / ".claude"
    else:
        raise ValueError("scope must be user or project")
    return root / "skills" / "llm-jury-delegate" / "SKILL.md"


def install_claude_skill(scope="user", project=None, force=False):
    destination = skill_path(scope, project)
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
