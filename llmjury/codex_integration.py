"""Install the Codex skill for verified jury runs and optional Claude planning."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


MANAGED_MARKER = "<!-- managed by llmjury install-codex; version: 3 -->"
MANAGED_PREFIX = "<!-- managed by llmjury install-codex; version:"
LEGACY_MANAGED_DIGESTS = {
    "a2e067b1e03119e0ac9ea267b86db3f11e2a13c0a14a9cfc2f463011b3109a50",
}


SKILL = """\
---
name: llm-jury-orchestrate
description: Use LLM-Jury inside the Codex app for verifier-shaped code tasks, local-first fusion, and optional Claude planning.
---
<!-- managed by llmjury install-codex; version: 3 -->

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

## Use JEV for bounded judgments

Use the registered JEV tool for semantic classification, evidence checks, relevance,
record matching, and choices from explicit permitted options. Supply the relevant
input and rubric; omit conversation history and unrelated files. Batch up to ten
independent questions about the same state. Start with a short evidence excerpt and
expand only when it cannot support the decision. Keep the provider's request limits.
Use lower_snake_case question keys and send state and questions as JSON objects,
with description strings under each choice criterion.

For advisory choices, confidence below 0.80 or a top-two probability gap below 0.20
requires review. JEV cannot approve access, sends, merges, deployments, or spending.
Use code for arithmetic, parsing, permissions, memory admission, and test verdicts.
Keep writing, code implementation, architecture, and final review with Codex. A clear
route does not need an extra JEV call. Preserve opt-outs and data restrictions.

If JEV is unavailable, disclose the fallback and use Codex for that judgment. Never
retry an uncertain billed request. Retain the compact result receipt when validating
the route; avoid copying full tool payloads into later generation prompts.

## Run from the Codex app

1. Read the repository instructions and inspect the target code and tests.
2. Tell the user which unit and oracle you will send to the jury.
3. Create a task file plus a verifier file or cases file outside tracked source.
4. Use the Codex app's terminal execution tool to run the local council first,
   then the authenticated Codex CLI only if the verifier rejects local candidates:

```bash
llmjury solve --task "$task_file" --tests "$tests_file" \\
  --entry-point function_name --backend ollama \\
  --frontier "${LLMJURY_CODEX_MODEL:-gpt-5.6-sol}" --frontier-backend codex \\
  --frontier-k 1 --jobs 2 --num-ctx 8192 --json
```

Use `--cases "$cases_file"` instead of `--tests` for JSON cases. Omit
`--entry-point` for stdin/stdout programs. If the user requests a private local run,
omit both frontier flags. This Codex workflow uses OpenRouter only when the user
explicitly requests an OpenRouter model or comparison. Report when the authenticated
Codex CLI produced the accepted candidate.

Keep the local best-of-k budget while starting the frontier with one candidate. If
that candidate fails the oracle, inspect the failure before requesting a larger
frontier budget. This avoids four concurrent subscription generations on routine
fallbacks. Send only the function's contract and required context, then run repository
tests after integration. Reuse identical tasks through the cache; use a unique task
when live provider validation is required.

On Macs, memory admission caps model residency at 65% of physical RAM and preserves
4 GiB beyond current desktop needs. Warning pressure, unreadable probes, and exclusive
27B ownership still refuse local inference. Never bypass a refusal or raise context
to make a local run appear successful. Ordinary memory pressure may use the configured
Codex frontier; exclusive ownership stops the entire jury run.
Jury Ollama requests retain models for 30 seconds after generation, then let the
server unload them; this does not change retention for other Ollama clients.

5. Accept output only when the command exits with status 0 and the JSON contains
   `"verified": true`. Do not integrate an unverified answer. Inspect verified code
   before applying it. If you alter its behavior, run the jury verifier again.
6. Run the repository's focused tests after integration, inspect the diff, and remove
   temporary files.

Keep the default generated-code sandbox enabled. The Codex CLI fallback uses the
existing Codex login and subscription, without an OpenRouter API key. Use it only
after local verification fails; avoid `--backend codex` as the first tier unless the
user requests a Codex-provider comparison.

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
