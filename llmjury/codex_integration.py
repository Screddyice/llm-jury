"""Install the Codex skill for verified jury runs and optional Claude planning."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


MANAGED_MARKER = "<!-- managed by llmjury install-codex; version: 2 -->"
MANAGED_PREFIX = "<!-- managed by llmjury install-codex; version:"
LEGACY_MANAGED_DIGESTS = {
    "a2e067b1e03119e0ac9ea267b86db3f11e2a13c0a14a9cfc2f463011b3109a50",
}


SKILL = """\
---
name: llm-jury-orchestrate
description: Use LLM-Jury inside the Codex app for verifier-shaped code tasks, local-first fusion, and optional Claude planning.
---
<!-- managed by llmjury install-codex; version: 2 -->

# LLM-Jury in the Codex app

Use this skill when the user asks for LLM-Jury or fusion, or when a code unit has a
trustworthy functional or stdin/stdout oracle. Codex remains the implementation agent.
LLM-Jury proposes code, and its verifier decides whether Codex may use that candidate.

## Choose a jury-shaped unit

Run the jury for a function, method, parser, algorithm, or bug reproduced by focused
tests. Extract the smallest unit that has deterministic inputs and outputs. Use an
existing test when it matches LLM-Jury's verifier format, or write a focused `check`
function or JSON cases file in a temporary directory.

Skip the jury for prose, architecture, UI judgment, configuration, or code without a
trustworthy oracle. Honor requests to skip fusion or write the change without it.

## Run from the Codex app

1. Read the repository instructions and inspect the target code and tests.
2. Tell the user which unit and oracle you will send to the jury.
3. Create a task file plus a verifier file or cases file outside tracked source.
4. Use the Codex app's terminal execution tool to run a local-first jury:

```bash
llmjury solve --task "$task_file" --tests "$tests_file" \\
  --entry-point function_name --backend ollama --frontier auto --json
```

Use `--cases "$cases_file"` instead of `--tests` for JSON cases. Omit
`--entry-point` for stdin/stdout programs. If the user requests a private local run,
or no OpenRouter credential exists, omit `--frontier auto` and keep
`--backend ollama`. Report any OpenRouter escalation and the model that produced the
accepted candidate.

5. Accept output only when the command exits with status 0 and the JSON contains
   `"verified": true`. Do not integrate an unverified answer. Inspect verified code
   before applying it. If you alter its behavior, run the jury verifier again.
6. Run the repository's focused tests after integration, inspect the diff, and remove
   temporary files.

Keep the default generated-code sandbox enabled. Avoid `--backend codex` from inside
the Codex app unless the user requests a Codex-provider comparison. A nested Codex
session adds cost without adding model-family diversity.

## Optional planning

For implementation work that needs a separate plan, Codex may ask Claude Code for a
read-only structured plan:

```bash
llmjury plan --workspace "$PWD" --task - --json
```

Planning is independent of the jury run. Do not call Claude when the user asks only
for local fusion, when the task already has a concrete plan, or when a delegated brief
contains a Claude plan. Replan only when execution evidence invalidates the current
plan. Include the failed command, output, completed work, and the decision that needs
a new plan.

Codex owns edits, tests, and the final handoff. Repository instructions and user
authorization govern commits, external actions, and deployments.
"""


def skill_path():
    root = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    return root / "skills" / "llm-jury-orchestrate" / "SKILL.md"


def _is_managed_skill(content):
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return MANAGED_PREFIX in content or digest in LEGACY_MANAGED_DIGESTS


def install_codex_skill(force=False):
    destination = skill_path()
    if destination.exists():
        current = destination.read_text(encoding="utf-8")
        if current == SKILL:
            return destination, False
        if not force and not _is_managed_skill(current):
            raise FileExistsError(
                f"{destination} already exists with different content; pass --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(SKILL, encoding="utf-8")
    temporary.replace(destination)
    return destination, True
