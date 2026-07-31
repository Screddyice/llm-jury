"""Install the Grok CLI skill for verified jury runs inside a Grok session."""
from __future__ import annotations

import os
from pathlib import Path


MANAGED_MARKER = "<!-- managed by llmjury install-grok; version: 1 -->"
MANAGED_PREFIX = "<!-- managed by llmjury install-grok; version:"


SKILL = """\
---
name: llm-jury-orchestrate
description: "Use LLM-Jury inside the Grok CLI for verifier-shaped code tasks: a local council proposes code and a real verifier, not a vote, decides what you may use."
---
<!-- managed by llmjury install-grok; version: 1 -->

# LLM-Jury in the Grok CLI

Use this skill when the user asks for LLM-Jury or fusion, or when a code unit has a
trustworthy functional or stdin/stdout oracle. Grok remains the implementation agent.
LLM-Jury proposes code, and its verifier decides whether Grok may use that candidate.

## Choose a jury-shaped unit

Run the jury for a function, method, parser, algorithm, or bug reproduced by focused
tests. Extract the smallest unit that has deterministic inputs and outputs. Use an
existing test when it matches LLM-Jury's verifier format, or write a focused `check`
function or JSON cases file in a temporary directory.

Skip the jury for prose, architecture, UI judgment, configuration, or code without a
trustworthy oracle. Honor requests to skip fusion or write the change without it.

## Run from the Grok CLI

1. Read the repository instructions and inspect the target code and tests.
2. Tell the user which unit and oracle you will send to the jury.
3. Create a task file plus a verifier file or cases file outside tracked source.
4. Run a local-first jury with the `bash` tool:

```bash
llmjury solve --task "$task_file" --tests "$tests_file" \\
  --entry-point function_name --backend ollama --json
```

Use `--cases "$cases_file"` instead of `--tests` for JSON cases. Omit
`--entry-point` for stdin/stdout programs.

5. Accept output only when the command exits with status 0 and the JSON contains
   `"verified": true`. Do not integrate an unverified answer. Inspect verified code
   before applying it. If you alter its behavior, run the jury verifier again.
6. Run the repository's focused tests after integration, inspect the diff, and remove
   temporary files.

## Escalation without leaving the subscription

When the local council cannot verify a candidate, escalate to Grok itself before
spending OpenRouter credit. The Grok backend reuses this CLI's own session auth, so
it costs nothing beyond the subscription already in place:

```bash
llmjury solve --task "$task_file" --tests "$tests_file" \\
  --backend ollama --frontier grok-4.5 --frontier-backend grok --json
```

Escalate to `--frontier auto --frontier-backend openrouter` (DeepSeek, then a
proprietary top tier) only when the user accepts metered spend, and say so first.

Report which stage produced the accepted candidate and whether any escalation ran.

Keep the default generated-code sandbox enabled. Do not use `--backend grok` for the
council itself: one provider sampled k times is not a cross-lineage panel, and the
local Ollama council is both free and more diverse.
"""


def skill_path():
    root = Path(os.environ.get("GROK_HOME", "~/.grok")).expanduser()
    return root / "skills" / "llm-jury-orchestrate" / "SKILL.md"


def _is_managed_skill(content):
    return MANAGED_PREFIX in content


def install_grok_skill(force=False):
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
