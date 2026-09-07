# LLM-Jury

[![PyPI](https://img.shields.io/pypi/v/llm-jury-verify)](https://pypi.org/project/llm-jury-verify/)
[![Python](https://img.shields.io/pypi/pyversions/llm-jury-verify)](https://pypi.org/project/llm-jury-verify/)
[![CI](https://img.shields.io/github/actions/workflow/status/ajsai47/llm-jury/ci.yml?label=ci)](https://github.com/ajsai47/llm-jury/actions)
[![dependencies](https://img.shields.io/badge/dependencies-0-success)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[Quickstart](#quickstart) · [Claude ↔ Codex](#bidirectional-claude--codex-orchestration) · [Codex Fusion](#codex-fusion) · [How it works](#how-it-works) · [Benchmarks](#benchmarks) · [Write-up →](https://app.notion.com/p/3844834c4d7881d1adaeed9c3a81dcbb) · [Paper →](https://app.notion.com/p/3874834c4d78817f99a0fc26088ed7e4)

**Local verified answers. Don't vote, verify.**

A frontier API gives you one answer and asks you to trust it. LLM-Jury gives you an answer it can
**prove** — by running the tests — and does most of the work on your own laptop, for free.

It runs model-generated code through a real verifier and returns only what **provably passes** —
not the answer that got the most votes. The result: a council of small open models on your machine,
plus opt-in, verifier-gated cloud escalation for the hard minority. The benchmarked hybrid
**matches a commercial frontier-model fusion on hard code at ~38× lower cost.**

> **Measured head-to-head — 45 hard LiveCodeBench problems, same oracle** ([full numbers + methodology →](BENCHMARKS.md)):

| | Accuracy (n=45) | Cost | Solved locally |
|---|---|---|---|
| **LLM-Jury hybrid** — local council + 1 opt-in frontier escalation | **44/45 (97.8%)** | **$0.49** | 75% |
| OpenRouter Fusion — commercial frontier fusion | 41/45 (91.1%) | $18.31 | 0% |

![Cost vs. accuracy on 45 hard LiveCodeBench problems — LLM-Jury hybrid 44/45 at $0.49, up-and-left of OpenRouter Fusion 41/45 at $18.31 (~38× cheaper); LLM-Jury council 75.6% at ~$0.03, local and free.](assets/cost_vs_accuracy.svg)

**Same accuracy, ~38× cheaper.** LLM-Jury *matches* the commercial frontier fusion — a strict
superset, solving every problem it does plus 3 — for **$0.49 vs $18.31**, running 75% of problems
fully on your laptop and escalating only the hard minority to a single cloud call. *(The +3-problem
edge is a clean sweep but not statistically significant at n=45 — McNemar p = 0.25 — so read
accuracy as parity-or-better; the **cost gap is the decisive, measured win.**)*

And the local council **alone**, no cloud at all, already beats a frontier model's one-shot on hard
code — **75.6% vs 62.2%** on LiveCodeBench — and matches it on HumanEval+ (**97.6% vs 97.6%**).
Free, private, zero dependencies (stdlib only).

```bash
pip install llm-jury-verify   # zero dependencies, stdlib only
llmjury demo                 # 5-second offline demo — no API key, nothing to download
```

`llmjury demo` runs the real generate → verify → escalate pipeline on a canned task with a built-in
offline backend: a "weak" model returns a wrong answer, the verifier catches it, and the council's
answer passes. That's the whole product, offline.

> **⚠️ Security.** LLM-Jury *executes model-generated code* to verify it. The default
> `LLMJURY_SANDBOX=auto` provisions a throwaway Docker/Colima container with no network,
> dropped capabilities, a non-root user, and resource limits. If a container cannot start,
> it falls back to the hardened host runner and prints that fact. Use
> `LLMJURY_SANDBOX=docker` to require container isolation; never run as root.

---

## Quickstart

**CLI** (examples ship in the repo):

```bash
# functional tests (the body of a check(candidate) function)
llmjury solve --task examples/add_task.txt --tests examples/add_test.py --entry-point add

# or competitive-programming style (stdin/stdout cases as JSON)
llmjury solve --task examples/sum_stdin_task.txt --cases examples/cases.example.json

# hybrid: local council first, then a verifier-gated OpenRouter ladder
llmjury solve --task examples/sum_stdin_task.txt --cases examples/cases.example.json \
    --backend ollama --frontier auto   # needs OPENROUTER_API_KEY

# same, but keep every escalation open-weight (no proprietary top tier)
llmjury solve --task examples/sum_stdin_task.txt --cases examples/cases.example.json \
    --backend ollama --frontier open

# Codex-native hybrid: local council first, then the authenticated Codex CLI.
# No OpenAI API key is required; this reuses `codex login` / ChatGPT-managed auth.
llmjury solve --task examples/add_task.txt --tests examples/add_test.py --entry-point add \
    --backend ollama --frontier gpt-5.6-sol --frontier-backend codex

# Codex as the generation provider for every tier (single model, best-of-k).
llmjury solve --task examples/add_task.txt --tests examples/add_test.py --entry-point add \
    --backend codex --models gpt-5.6-sol
```

`--tests` accepts Python only: either a complete `check(candidate)` function or its
body. LLM-Jury compiles that oracle before it loads a backend, starts Ollama, or sends
an OpenRouter request. A TypeScript, malformed, or otherwise invalid test file exits
with a line-specific error instead of making every generated candidate fail.

**Python:**

```python
from llmjury import solve, FunctionalCodeVerifier, OllamaBackend

task = "Write a function `add(a, b)` that returns the sum of two numbers."
tests = "def check(c):\n    assert c(2, 3) == 5\n    assert c(-1, 1) == 0\n"

# Local + free (needs Ollama running). Omit backend= to use the OpenRouter cloud
# backend instead — that one needs OPENROUTER_API_KEY set.
result = solve(task, FunctionalCodeVerifier(tests, entry_point="add"), backend=OllamaBackend())
print(result.verified)   # True
print(result.answer)     # def add(a, b): return a + b
print(result.stage)      # "single"  (escalates to "council" on hard problems)
```

**Bring your own tests** — skip the `check()` string entirely with `(args, expected)` pairs:

```python
from llmjury import FunctionalCodeVerifier
verifier = FunctionalCodeVerifier.from_cases("add", [((2, 3), 5), ((-1, 1), 0)])
```

(`--tests` also accepts either a full `def check(candidate): ...` or just its body. And
`llmjury solve --json` emits a machine-readable result for piping into CI/agents.)

## Bidirectional Claude ↔ Codex orchestration

LLM-Jury installs skills into both agent harnesses so the workflow works from either
side: Claude plans and reviews; authenticated Codex executes; the local Ollama council
assists only on code units with a trustworthy oracle.

```bash
llmjury install-claude
llmjury install-codex
```

Restart Claude Code and Codex after the first install so each discovers its skill.

### Agent frontmatter rules

These two rules exist because a host that parses agent frontmatter differently from
Claude Code fails **silently**, not loudly. Both are cheap to honour and expensive to
rediscover.

> **Quote the description — agents only.** Claude Code parses agent frontmatter
> leniently. Under a strict YAML parser an unquoted `description:` containing a
> colon-space (`": "`) is not a string — it reads as a mapping, and the agent is dropped
> with no error anywhere. Keep agent descriptions as quoted YAML scalars; the shipped
> agent does, and a test enforces it. Skill frontmatter is parsed leniently and needs no
> quoting.

> **Ship no `model:` pin.** Hosts disagree about a pin they cannot resolve. Some reject
> it outright (sessions on the official Anthropic API — the Claude desktop app, cron);
> others ignore it and run the agent on the session model instead, so an agent pinned to
> a local model quietly executes on a cloud one. `llm-jury-fusion` therefore ships with
> no pin at all. If you keep locally-pinned agents in `~/.claude/agents/`, confirm each
> host actually resolves the pin rather than assuming it does.

```text
Start in Claude:  Claude plan → Codex execute ─┐
                                               ├→ tests/diff → done
Start in Codex:   Claude plan ← Codex request ─┘       │
                         ↑                              │ evidence changed
                         └──────── dynamic replan ──────┘

Within Codex execution: testable Python unit → local Ollama jury → verifier → integrate
```

### Starting in Claude Code

`install-claude` installs two entry points:

- **`llm-jury-fusion` agent** — a subagent that frames a verifiable task, derives the
  oracle, and drives `llmjury solve --backend ollama` itself. It deliberately carries
  no `model:` pin, so it inherits the session model and works in every Claude Code
  session type — terminal, the Claude desktop app's hosted sessions, and cron — with
  no dependency on a local model router or a custom `ANTHROPIC_BASE_URL`. (Sessions
  pinned to the official API, like the desktop app's, cannot resolve local-router
  model names; an unpinned agent sidesteps that entirely.)
- **`llm-jury-delegate` skill** — has Claude pass a bounded execution brief to Codex:

```bash
llmjury delegate --workspace "$PWD" --task - --json <<'TASK'
Implement the parser described in the current plan.

Scope: src/parser.py and tests/test_parser.py.
Acceptance: preserve the public API and make the focused parser tests pass.
Checks: python -m pytest tests/test_parser.py.
TASK
```

### Starting in Codex

The `llm-jury-orchestrate` skill gives the Codex app a native jury workflow. Codex
extracts a code unit and its oracle, runs the local Ollama council through the app's
terminal tool, and accepts a candidate only when the JSON result contains
`"verified": true`.

```bash
llmjury solve --task task.txt --tests tests.py --entry-point solve \
    --backend ollama --frontier auto --json
```

The skill keeps private runs local when the user requests them or OpenRouter has no
credential. Codex runs the repository's tests after it applies a verified candidate.
It does not send prose, architecture, UI judgment, or code without a trustworthy
oracle to the jury.

Claude planning remains available for work that needs a separate plan:

```bash
llmjury plan --workspace "$PWD" --task - --json <<'TASK'
Implement resumable uploads without changing the public client API.
Include repository constraints, acceptance criteria, and checks.
TASK
```

Claude receives read and search tools in plan permission mode and returns structured
steps, risks, and blocking questions. Planning does not run before each jury call.
Codex invokes it when the user requests a plan or execution evidence invalidates the
current one.

`delegate` is a different security mode from the Codex generation backend:

- It runs an ephemeral Codex agent with `workspace-write` confinement in the one
  workspace Claude names.
- It keeps Codex user configuration and repository instructions enabled, so the
  executor reads the applicable `AGENTS.md` and `CLAUDE.md`.
- Shell commands receive Codex's minimal `core` environment by default instead of
  inheriting the caller's credential-rich environment.
- Codex returns a schema-validated handoff: status, summary, changed files, checks,
  and blockers. Claude remains responsible for inspecting the diff and final result.
- Destructive operations and permission bypasses are not exposed. Extra writable
  directories require an explicit repeatable `--add-dir` argument.

The delegation prompt allows Codex to call the existing local jury for an extractable
Python unit with functional tests or stdin/stdout cases. It does not ask local models
to vote on architecture or other uncheckable work. This preserves the project's core
rule: local model output becomes implementation input only after an independent
verifier accepts it.

Use `--scope project` to install the Claude skill in only the current repository, or
`--force` to replace a locally modified installed copy. Both installers are idempotent.

## Codex Fusion

The recommended Codex workflow keeps candidate generation local until the verifier proves
the local council needs help:

```text
The host agent frames the task and oracle
  → Phi-4 on local Ollama
  → Gemma 3 12B + Llama 3.1 8B only if Phi-4 fails
  → DeepSeek V4 Flash on OpenRouter only if the local council fails
  → DeepSeek V4 Pro only if Flash also fails
  → the host's authenticated Codex or Claude CLI if OpenRouter fails
  → return the first candidate that passes the oracle
```

Run that policy with:

```bash
export OPENROUTER_API_KEY="..."   # or store it in ~/.llmjury/.env
llmjury solve --task task.txt --tests tests.py --entry-point solve \
    --backend ollama --frontier auto --json
```

The distinction matters: **the council is local and private through Ollama; the DeepSeek
models are open-weight but remotely hosted through OpenRouter.** A task leaves the machine
only after every local candidate fails verification. The CLI prints the selected stage and
model, so an orchestrating Codex session can report when paid escalation actually occurred.

`auto` uses a capability ladder instead of guessing difficulty from task keywords. The
verifier is the router: Flash gets the first inexpensive recovery attempt, Pro receives only
the unresolved tail, and neither can introduce an accepted regression because its output must
pass the same tests.

To use authenticated Codex itself as the final provider instead of OpenRouter:

```bash
llmjury solve --task task.txt --tests tests.py --entry-point solve \
    --backend ollama --frontier gpt-5.6-sol --frontier-backend codex
```

That path reuses `codex login`, launches an ephemeral read-only generation session, ignores
repository rules and user configuration, and disables shell tools. Pin any explicit
OpenRouter slug with `--frontier <provider/model>` when reproducing a benchmark or comparing
a particular model.

## How it works

Three small models you can run for free on a laptop, plus a verifier, match or beat a frontier
model on verifiable work. The engine is a tiered pipeline that *gates fusion on the verifier*:

```
task + verifier
  → GENERATE   sample candidates across a cross-lineage small-model council
  → VERIFY     run the verifier on each (code = run the tests)
  → SELECT     return the one that provably passes
  → ESCALATE   one model first → add the council only when nothing passes
               → (optional) ordered frontier models only when earlier tiers fail
```

That tiered escalation is what keeps it laptop-friendly *and* lets it reach frontier-class
accuracy: the common case runs **one** model fast, hard problems load the full local council, and —
if you opt in with `--frontier` — the small hard minority the council can't verify escalates to a
cloud model or ordered ladder. You pay for cloud inference only on the problems that need it, and
each later model runs only when the earlier one still cannot produce a verified answer.

**The rule LLM-Jury is built on:** combining models helps *exactly when you can check the answer
cheaply and independently.* Code has a real oracle (run the tests), so the council wins. Tasks
with no oracle (open-ended reasoning) collapse to voting, where a council *hurts* — so there,
LLM-Jury uses the single best model. The verifier, not the model count, is what decides.

## Benchmarks

Full numbers behind the headline — the measured head-to-head on all 45 hard LiveCodeBench
problems, judged by the same oracle ([methodology + caveats](BENCHMARKS.md)):

| | Accuracy (n=45) | 95% CI | Cost | Local | Verified |
|---|---|---|---|---|---|
| **LLM-Jury hybrid** (council + 1 frontier escalation) | **44/45 (97.8%)** | [88%, 99.6%] | **$0.49** | ◐ | ✓ |
| OpenRouter Fusion (cloud) | 41/45 (91.1%) | [79%, 97%] | $18.31 | ✗ | ✗ |
| **LLM-Jury council** (small open council, local) | 34/45 (75.6%) | [61%, 86%] | **$0.031** · free local | ✓ | ✓ |

Stated honestly: the **cost** gap is measured and decisive; the **accuracy** edge (+3 problems, 0
losses) is a clean sweep on the sample but **not statistically significant** (McNemar p = 0.25 at
n=45) — so read it as parity-or-better, not a proven win. And we *gave Fusion a token-budget
advantage* on 5 problems it first truncated, which only helped it. "Correct" here means "passes the
public sample tests," and the escalated ~25% do leave the device. Full caveats — confidence
intervals, the public-test oracle, the fairness disclosure — are in [BENCHMARKS.md](BENCHMARKS.md).

The table reports the historical benchmark configuration: local council followed by a pinned
DeepSeek V4 Pro best-of-4 escalation. The newer `--frontier auto` Flash → Pro ladder is designed
to reduce the cost of the unresolved tail without weakening acceptance, but it should not be
read as a separately measured benchmark result until that exact policy is reproduced.

## Backends

- **Ollama (local, free, private)** — `--backend ollama`. Pull the council first:
  ```bash
  ollama pull gemma3:12b && ollama pull llama3.1:8b && ollama pull phi4-mini:3.8b
  ```
  The local council mirrors the lineages of the benchmarked cloud panel — Google /
  Meta / Microsoft — so the measured numbers describe something reproducible
  off-cloud. The mirror is not exact and cannot be: `phi-4` is 12.7 GiB locally, and
  the benchmarked trio `phi4 + gemma3:12b + llama3.1:8b` projects 31.7 GiB against a
  25.2 GiB budget on a 36 GiB host. No `num_ctx` or slot count fits it, since the
  weights alone are ~28 GiB. `phi-4` is therefore substituted by `phi4-mini:3.8b`
  from the same family, keeping all three labs on the council. **For exact benchmark
  fidelity use `--backend openrouter`, which runs `CLOUD_PANEL` unchanged.**

  The default **requires `OLLAMA_NUM_PARALLEL=2`**. KV cache is charged
  `num_ctx x slots`, so parallelism multiplies memory for every model on the server
  and is part of a panel's spec:

  | slots | projected | 36 GiB host, budget 25.2 GiB |
  |-------|-----------|------------------------------|
  | 2     | 23.4 GiB  | fits                         |
  | 4 (Ollama default) | 27.3 GiB | refused, with a hint |

  Set it on the server and tell the client, then restart Ollama:
  ```bash
  # server: launchd plist / systemd unit
  OLLAMA_NUM_PARALLEL=2
  # client: or the preflight assumes 4 and over-refuses panels that would fit
  export LLMJURY_OLLAMA_PARALLEL=2
  ```
  Leaving Ollama at 4 slots is safe, just smaller: the preflight runs before any
  model loads, so you get an actionable refusal naming a smaller panel rather than a
  host that swaps itself to death. Use
  `--models llama3.1:8b,phi4-mini:3.8b,granite4.1:3b` (19.7 GiB at 4 slots) if you
  would rather not tune the server. `--models` is gated by the same preflight.

  Pass any Ollama completion tag through `--models`, including Qwen, custom
  Modelfiles, and fine-tunes. LLM-Jury disables model thinking by default so the
  generation budget produces verifier-ready code. Add `--think` when you want an
  Ollama model to spend part of that budget on reasoning. The cache keeps thinking
  and non-thinking runs separate.
- **OpenRouter (cloud)** — `--backend openrouter` (default). Set `OPENROUTER_API_KEY`.
- **Codex CLI (authenticated OpenAI provider)** — `--backend codex`, or use
  `--frontier-backend codex` after a local council. It runs ephemeral, read-only
  `codex exec` sessions and reuses the existing Codex login. The default model for
  this checkout is `gpt-5.6-sol`; set `LLMJURY_CODEX_MODEL` to change that default,
  or override it per invocation with `--models` or `--frontier`.
  Candidate-generation sessions disable Codex's shell tools as an additional guard
  against reading unrelated host files.
  Jury calls default to low reasoning effort because the independent verifier supplies
  the acceptance gate; Python callers can override `CodexBackend(reasoning_effort=...)`.
- **Claude Code CLI (authenticated Anthropic provider)** — used as the last rescue for
  `--frontier auto` inside Claude Code. It removes the parent-session nesting marker,
  starts in a temporary directory with safe mode, disables tools and permission prompts,
  and keeps session persistence off. Set `LLMJURY_CLAUDE_MODEL` to change the default
  from `opus`.

The frontier provider is explicit. `--frontier-backend openrouter` accepts OpenRouter
slugs and requires `OPENROUTER_API_KEY`; `--frontier-backend codex` accepts a model
available to the installed Codex CLI and uses Codex authentication. OpenRouter is not
coupled to Anthropic: any compatible OpenRouter model slug can be used.

Inside Codex or Claude Code, `--frontier auto` keeps the ordered OpenRouter ladder and
adds the host's authenticated model as its final rescue. An exhausted OpenRouter balance
or provider outage can no longer end the run while the host provider remains available.
Every candidate still has to pass the same verifier. `--frontier open`, `opus`, `fable`,
and explicit model slugs keep their stated provider boundary; named OpenRouter ladders
remain invalid with `--frontier-backend codex` because Codex cannot serve OpenRouter
slugs.

The completion cache stores generated candidates, not transport failures. Empty results
from timeouts, account errors, and unavailable routes are retried on the next run. This
also repairs empty cache entries written by older releases without deleting valid cached
generations.

### Fitting the council in RAM

A local council loads every panelist at once, and Ollama caps residency by model
**count** (`OLLAMA_MAX_LOADED_MODELS`, default 3 on a single-GPU host), never by bytes.
Three models is exactly a default panel, so nothing in the stack knows the aggregate.

That matters because an over-large panel does not fail cleanly. Metal allocations are
wired and cannot be paged out, so the host compresses and swaps everything else until
the kernel watchdog is starved and panics. On 2026-07-31 that took a 36 GB Mac down
twice before the cause was found: the previous default panel measured **34 GB**
resident.

Two rules of thumb, both measured with `ollama ps`:

- Resident size is roughly **double** the on-disk size.
- The KV cache is charged per **total** cells, meaning `num_ctx × OLLAMA_NUM_PARALLEL`.
  A 3.4 GB model pinned to a 64k-context tag costs 7.5 GB resident, not 3.4 GB. This is
  the trap: raising `OLLAMA_NUM_PARALLEL` for throughput multiplies KV for every model
  on the server.

`llmjury solve --backend ollama` therefore preflights the panel against physical RAM and
**refuses** to start a run that would over-commit the host:

```
$ llmjury solve --backend ollama --models phi4,gemma3:12b,llama3.1:8b ...
[llmjury] panel needs ~35.6 GB resident, budget is 25.2 GB
  phi4                     ~ 14.0 GB
  gemma3:12b               ~ 12.8 GB
  llama3.1:8b              ~  8.8 GB
error: this panel would over-commit the host, which can hang or panic it.
hint: use a smaller panel, e.g. --models llama3.1:8b,gemma3:12b; lower --num-ctx;
      set OLLAMA_NUM_PARALLEL=1 (KV is charged num_ctx x slots)
```

The same preflight also refuses a local panel while an **iOS Simulator is booted**.
The CoreSimulator stack is hundreds of XPC services whose resident cost (17.6 GB
measured across 282 processes on the 36 GB reference host) appears in none of the
numbers this module can model, and a council loaded on top of it is precisely the
co-residency that panics a machine. The refusal names the stack's measured size and
the ways out: `xcrun simctl shutdown all` (a simulator reboots in seconds), a cloud
backend, or the explicit `LLMJURY_ALLOW_SIMULATOR=1` escape hatch for hosts with the
RAM to hold both. Detection fails open — a host without `pgrep` never starts
refusing panels — and non-macOS hosts skip the probe entirely.

A refusal is not the end of the run. It says the *panel* cannot load here, which says
nothing about the frontier ladder — that runs on a remote provider and costs this host
no memory. So when `--frontier` is set, a refusal **skips the local council and
escalates straight to the ladder** instead of killing a run that still has a safe path
to a verified answer:

```
[llmjury] an iOS Simulator is booted (holding ~18.7 GB); a local panel cannot co-reside …
[llmjury] skipping the local council; escalating straight to deepseek/deepseek-v4-flash
          -> deepseek/deepseek-v4-pro -> anthropic/claude-opus-5 on openrouter
          (remote, needs no memory on this host)
# llmjury: VERIFIED  [stage=frontier, model=deepseek/deepseek-v4-flash, attempts=1]
```

Without a `--frontier` ladder there is nothing to escalate to, so the run still stops —
and the error now points at `--frontier auto` as the way through.

#### Standing down for Qwen 27B

There is one refusal that does *not* escalate. [Backdoor](https://github.com/Screddyice/backdoor)
gives `qwen3.8:27b-obliterated` exclusive compute when a Claude or Codex session selects
it directly or fails over to it. Backdoor publishes circuit-breaker state to
`~/.backdoor/failover-state.json` and writes short ownership leases under
`~/.backdoor/compute-leases/` before local inference begins. LLM-Jury also checks
Ollama's `/api/ps` output as a residency backstop.

```
error: llm-jury is standing down; exclusive 27B compute is active.
owner: claude-explicit owns qwen3.8:27b-obliterated
The local council and every frontier provider, including OpenRouter, remain disabled
until the 27B route releases the host.
```

This gate runs before backend construction and ignores `--mem-check`, so it also blocks
direct OpenRouter runs and verifier-gated frontier escalation. The lease closes the gap
before Ollama reports the model as resident. An expired lease or one from a dead router
process is ignored. Missing or unreadable state fails open. Point the probes elsewhere
with `LLMJURY_ROUTER_STATE` and `LLMJURY_COMPUTE_LEASE_DIR` when testing an isolated
router.

One wrinkle worth setting up: a launchd or systemd unit exports `OLLAMA_NUM_PARALLEL`
into the *server* process, not into the client, so the preflight cannot read it and
assumes Ollama's default of 4. Export `LLMJURY_OLLAMA_PARALLEL` to match your server and
the estimate stops being pessimistic:

```bash
export LLMJURY_OLLAMA_PARALLEL=2      # match OLLAMA_NUM_PARALLEL on the server
```

The budget defaults to 70% of physical RAM, since the remainder is not slack: it is the
OS, the editor, the browser, and the agent session that launched the run. Models another
session already has resident are counted too. Tune with `LLMJURY_MEM_FRACTION`, or relax
the guard with `--mem-check warn` (proceed anyway) or `--mem-check off`. When the check
cannot determine an answer, because Ollama is unreachable or RAM is unreadable, it skips
rather than blocking: it exists to stop a known-bad run, not to invent new failures.

The shipped panel is sized to fit a 36 GB host at ~19 GB and stays cross-lineage
(Meta / Microsoft / IBM). Panel strength matters less here than it would in a voting
council, because LLM-Jury verifies rather than votes: weaker panelists escalate to the
frontier ladder more often instead of returning worse answers. On a larger host, pass a
stronger panel through `--models`.

### Caching

Generations are cached in `~/.llmjury/cache.jsonl`, keyed on backend, model,
temperature, `max_tokens`, sample index, and the prompt. Re-running the same solve is
therefore free and near-instant: it replays stored samples and never calls the provider.
The verifier still executes every time, so a cached run reports exactly like a fresh
one — convenient for iterating on a verifier, and easy to mistake for a live test.

To confirm a run actually reached the provider, compare `wc -l` on the cache before and
after; zero new lines means nothing was generated. To force a cold run, move the cache
aside and restore it afterwards.

`--frontier auto` is the recommended local-first policy. It keeps the cross-lineage
Ollama council first, then tries DeepSeek V4 Flash for a low-cost cloud recovery,
DeepSeek V4 Pro for the hard remainder, and finally Claude Opus 5 for the tail that
nothing cheaper could verify. Every later tier runs only when all earlier candidates
failed the same oracle, so the faster model cannot lower accuracy — and the tail that
reaches the last tier is small by construction.

Cost note: the top tier is proprietary and roughly 35× the per-token price of the
open-weight tiers below it ($5/$25 per M vs $0.14/$0.28). It is last precisely so it
bills only on problems two cheaper tiers already failed. Named ladders:

| `--frontier` | Escalation order after the local council |
|---|---|
| `auto` | DeepSeek V4 Flash → DeepSeek V4 Pro → Claude Opus 5 |
| `open` | DeepSeek V4 Flash → DeepSeek V4 Pro (open-weight only, no proprietary tier) |
| `opus` | Claude Opus 5 |
| `fable` | Claude Fable 5 |
| any other value | passed to the provider verbatim as a model slug |

Use `open` to cap escalation spend, or pin a slug when reproducing a historical
benchmark or testing a specific model. `LLMJURY_TOP_FRONTIER` overrides which model
`auto` ends on, so the top tier is configurable rather than baked in.

Reasoning models spend part of their token budget on private thinking before emitting
any code, and providers count those tokens against `max_tokens`. Since the frontier
tier runs only on the hard tail — where thinking is longest — it gets its own wider
budget (`frontier_max_tokens`, default `max(max_tokens, 8000)`) so a long deliberation
cannot truncate the answer.

OpenRouter requests have a 180-second socket timeout. A timeout ends that sample
without the prior silent retry loop, so one stalled provider request cannot hold a
frontier tier for up to 20 minutes. Set `LLMJURY_OPENROUTER_TIMEOUT` to a positive
number of seconds when a slower provider needs more room.

A diverse, cross-lineage panel (different labs → different mistakes) is the point. The defaults
are Phi-4 (Microsoft) + Gemma-3-12B (Google) + Llama-3.1-8B (Meta); swap your own in
`llmjury.panels` or by passing `panel=[...]` to `Engine`.

```bash
llmjury solve --task task.txt --tests tests.py --entry-point solve \
  --backend ollama --models qwen3:8b,qwen3.5:4b,gemma3:12b --best qwen3.5:4b
```

## Reproduce the benchmarks

```bash
llmjury reproduce humaneval            # bundled 25-problem slice
llmjury reproduce lcb --n 5            # quick check
llmjury reproduce lcb --backend ollama  # run it locally
llmjury reproduce lcb --backend ollama --num-ctx 4096   # tighter KV on a small host
```

It runs the council on a bundled benchmark slice and reports how many problems the single best
model solves versus what the diverse council adds on escalation — the "council adds coverage"
story, in one command. (The bundled slices are 25 problems each; the headline 97.6% / 75.6%
figures are the larger full runs from the write-up.)

The Ollama path pins `--num-ctx` (default 8192) and runs the same RAM preflight as `solve`.
Both matter: Ollama sizes KV as `num_ctx x OLLAMA_NUM_PARALLEL` **at load**, so inheriting a
server default of 32k inflates every panelist by 1.6-1.9x, and a benchmark sweep is the most
likely way to hold the whole council resident at once.

Measured on a 36 GiB Mac with the shipped local panel at `OLLAMA_NUM_PARALLEL=2`:

```
llmjury reproduce lcb --backend ollama       # gemma3:12b + llama3.1:8b + phi4-mini:3.8b
  single best model + verified best-of-4:   19/25 = 76.0%
  + diverse council (escalation):            +0  ->  19/25 = 76.0%
```

Two honest caveats. This is the 25-problem slice, not the 45-problem run behind the headline,
so 76.0% here and the published 75.6% are not the same measurement. And on this slice the
council added **nothing** over the best model alone: every pass came from `gemma3:12b` at the
`single` stage. Council escalation earns its keep on harder distributions than the bundled
slice — treat `+0` as a property of this sample, not a refutation of the method.

## Status

`v0.1` — working engine, functional and stdin/stdout verifiers, auto-provisioned container
sandbox, Codex + Ollama + OpenRouter backends, verifier-gated frontier ladders,
bidirectional Claude/Codex orchestration, and `llmjury reproduce`. Next: more verifier
types and benchmark runs for adaptive routing policies.

### Contributing from an agent harness

If you drive this repo with a coding agent, keep its scaffolding out of your commits.
`.claude-harness/` and a generated `CLAUDE.md` are per-machine session state, not project
configuration — both are gitignored. `AGENTS.md` is the tracked, shared agent guidance;
put anything a contributor needs there instead. Prefer staging explicit paths over
`git add -A`, which sweeps this kind of local state into a PR.

### Internals

CLI subprocess execution is centralized in `llmjury/cliproc.py`: one `run_cli` helper
owns the run-with-timeout mechanics (injected runner, captured text output,
`TimeoutExpired`/`OSError` classification) for `CodexBackend`, `CodexDelegator`, and
`ClaudePlanner`, while each call site keeps its own command construction and error
messages.

## License

MIT © The AI Collective

### Memory preflight budget (2026-09-05)

`DEFAULT_MEM_FRACTION` is 0.65, down from 0.70, to match the budget Ollama now enforces on the
host. Metal reports a static 28.1 GiB to Ollama whatever the desktop is using; the launchd plist
sets `OLLAMA_GPU_OVERHEAD` to 4 GiB, so Ollama budgets 23.6 GiB. The preflight refuses the same runs
the server would evict, instead of approving a council Ollama then silently serialises. Override
per run with `LLMJURY_MEM_FRACTION`.

### Desktop pressure and host prompt caches

Ollama's `/api/ps` reports model/GPU allocations, not the full process footprint.
The llama-server runner also defaults to an 8 GiB host prompt cache. On a 36 GiB
Mac, the 27B runner reached a 26.4 GiB footprint while `/api/ps` reported 16.4 GiB;
the logs showed 7.3 GiB of saved prompts. One cache update took 51 seconds.

The preflight now reserves that cache bound per loaded/requested runner, refuses
unreadable memory probes, and checks current desktop headroom as well as the
static model budget. macOS warning/critical pressure blocks local work. On Linux
the guard uses `MemAvailable`. It preserves 2 GiB beyond current desktop needs.
An unknown or unlimited cache bound also refuses local admission.

To reduce cache overhead, set `LLAMA_ARG_CACHE_RAM=1024` in the **Ollama server's**
environment. It limits saved prompt states to 1 GiB without shrinking the model
or its context window. An operator must restart Ollama at an idle point before
the new environment applies. Preserve a rollback copy of the service config;
do not restart Backdoor. Verify the runner log says `limits: 1024.000 MiB` before
setting `LLMJURY_PROMPT_CACHE_MIB=1024` in clients. Until then, clients reserve the
full default 8 GiB. A saved plist alone is not proof of the active limit.

Other local clients can share the read-only admission check:

```bash
llmjury preflight --models qwen3.5:4b --num-ctx 24576
```

It returns JSON and exits 0 only on admission, without inference or cloud calls.
The check includes Backdoor's live-process leases and 27B residency. Exclusive
ownership blocks all jury providers; memory refusals can still use an explicitly
configured remote frontier. `solve --backend ollama` and `reproduce --backend ollama` hold a nonblocking process
lock at `~/.cache/llmjury/local-compute.lock`; cooperating background reviewers
hold the same lock through their check and inference. `LLMJURY_LOCAL_LOCK` can
override that path, but consumers must use the same value. Kernel locks disappear
on process exit. This coordinates participating clients, not arbitrary direct
Ollama calls, and does not preempt an inference if Qwen starts afterward.

Verification: `python tests/test_llmjury.py` and
`python tests/test_memory_pressure.py` run without model inference. Existing
panel-fit measurements above exclude the extra prompt-cache reserve; use the
current preflight before starting a council on a busy desktop.

## Working in this repo

Python `>=3.9`, managed with **uv** (`uv.lock` committed). The package installs the
`llmjury` and `jury` console scripts. No Node toolchain.

```bash
uv sync
uv run pytest
uv run llmjury solve --task <f> --tests <f> --backend ollama --frontier auto
```

`solve` requires a verifier (`--tests`, `--cases`, or `--entry-point`) and refuses to
answer without one — that constraint is the product, not a limitation to work around.

**Before changing the council, read `BENCHMARKS.md`.** Resident memory runs about 2x
on-disk and KV is charged `num_ctx x OLLAMA_NUM_PARALLEL`. The original panel measured
34 GB resident on a 36 GB Mac and panicked the kernel twice; the package now refuses a
panel that will not fit. Substitute within a lab, never swap labs — a test enforces the
cross-lineage shape. Agent instructions live in `CLAUDE.md`.
