# Changelog

## Unreleased

- **Refuse local panels that do not fit in RAM.** A council loads every panelist at
  once, but Ollama caps residency by model count (default 3, exactly a panel), never by
  bytes, so nothing knew the aggregate. The previous default panel measured 34 GB
  resident on a 36 GB host and panicked it twice on 2026-07-31: wired GPU allocations
  cannot be paged out, so the machine starves its kernel watchdog instead of raising an
  out-of-memory error a caller could catch. `llmjury solve --backend ollama` now
  preflights the panel against physical RAM and exits with the arithmetic and a
  remediation hint. Budget defaults to 70% of RAM (`LLMJURY_MEM_FRACTION`); relax with
  `--mem-check warn|off`. The check skips rather than blocking when it cannot tell.
- **Resize the default local panel** from `phi4 + gemma3:12b + llama3.1:8b` (34 GB) to
  `llama3.1:8b + phi4-mini:3.8b + granite4.1:3b` (~19 GB), still cross-lineage. Because
  LLM-Jury verifies rather than votes, weaker panelists escalate more often rather than
  returning worse answers, so this trades escalation spend for host stability. Pass
  `--models` to restore a larger panel on a larger host.
- **Restore a 12B panelist now that the ceiling is enforced.** The resize above was
  measured against Ollama's stock 4 parallel slots and, at ~19.7 GiB, left roughly a
  fifth of the budget unused. The default is now
  `gemma3:12b + llama3.1:8b + granite4.1:3b` (Google / Meta / IBM) at 23.0 GiB, which
  **requires `OLLAMA_NUM_PARALLEL=2`** — stated as a requirement because KV is charged
  `num_ctx x slots`, so parallelism is part of a panel's spec rather than ambient
  config. Depending on it is safe only because the preflight runs before any model
  loads: a stock 4-slot host is *refused* with a hint naming a smaller panel, never
  panicked. Both halves are pinned by tests, so the default cannot silently become a
  crash risk for an untuned host.

- Add an authenticated Grok CLI backend (`--backend grok`, `--frontier-backend grok`)
  for direct generation or frontier escalation. Like the Codex backend it generates in
  an isolated session — read-only sandbox, no tools, no subagents, no memory, one turn,
  empty temp cwd — and authenticates from the Grok CLI's own session, so escalation
  costs nothing beyond an existing Grok subscription and needs no `XAI_API_KEY`. The
  model default is `grok-4.5`, overridable via `LLMJURY_GROK_MODEL`.
- Add `llmjury install-grok`, which installs an idempotent `llm-jury-orchestrate` skill
  into `~/.grok/skills/` (honouring `GROK_HOME`) so the Grok CLI drives verified jury
  runs and prefers subscription-auth escalation over metered OpenRouter spend.
- Quote the `description:` frontmatter of the shipped Claude agent and skills. The Grok
  CLI parses frontmatter as strict YAML and **silently drops** any file whose unquoted
  description contains a colon-space — the agent simply never appears in `grok inspect`.
  Claude Code is lenient, so this was invisible from that side; a test now enforces it.

- `solve --backend ollama` now preflights the panel with Ollama `/api/show` and warns
  when a tag has a large SYSTEM prompt baked into its Modelfile: jury prompts are bare
  user messages at a lean num_ctx, so a baked system slows every prefill and gets
  truncated — measured live, 186 KB baked prompts made one 12B tag time out a
  50-token probe. The warning names the tag and size; the run still proceeds.
- `llmjury install-claude` now also installs a router-independent `llm-jury-fusion`
  Claude Code agent. It carries no `model:` pin, so fusion runs work in sessions that
  pin the official Anthropic API (the Claude desktop app's hosted sessions, cron)
  where local-router model names cannot resolve; the council still runs on local
  Ollama either way.

- Add `llmjury delegate` for Claude-planned, workspace-confined Codex execution with
  schema-validated handoffs and minimal shell-environment inheritance.
- Add `llmjury install-claude` to install an idempotent Claude Code delegation skill
  that keeps local Ollama assistance limited to verifier-backed code units.
- Add `llmjury plan` and `llmjury install-codex` so Codex automatically delegates
  non-trivial planning and evidence-driven replanning to read-only Claude Code.
- Add an authenticated Codex CLI backend for direct generation or frontier escalation.
- Add `--frontier-backend` so Codex and OpenRouter are explicit, interchangeable
  frontier providers instead of coupling escalation to OpenRouter.
- Allow `LLMJURY_CODEX_MODEL` to override the checked-in Codex model default and
  disable shell tools in Codex candidate-generation sessions.
- Add `--frontier auto`, a verifier-gated OpenRouter ladder that tries open-weight
  DeepSeek V4 Flash before the benchmark-backed V4 Pro accuracy tier.
- Allow ordered frontier ladders in `Engine` while preserving single-model callers.
- Disable Ollama thinking by default so Qwen and other reasoning models return code
  within the generation budget. Add `solve --think` for opt-in reasoning, and isolate
  cached responses by Ollama host, context size, and thinking mode.

Notable changes to LLM-Jury. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Stages decode concurrently and verify in completion order: all of a stage's samples
  (across all of its models) are queued at once, the sandbox checks finished samples
  while the backend keeps decoding the rest, and the first verified sample wins the
  stage. The CLI exits as soon as a sample verifies, dropping leftover decodes (Ollama
  cancels them on disconnect). Escalation across stages is still strictly sequential.
- `Backend.submit()` — per-sample futures alongside the batch `complete()`; custom
  duck-typed backends with only `complete()` keep working. `Result.attempts` now counts
  samples that finished generating before the verdict.
- `OllamaBackend(num_ctx=...)` and `solve --num-ctx` (default 8192): cap each request's
  context window. Ollama sizes a model's KV cache as num_ctx x OLLAMA_NUM_PARALLEL at
  load, so a lean value is what lets the whole council stay resident and decode in
  parallel (at a 32k server default with 4 slots, one 8B model ballooned to 22 GB and
  evicted every other panelist).
- `solve --jobs N` — generation threads across a stage (default: k x panel size, cap 16).
- Measured on an M5 Max (36 GB, `OLLAMA_NUM_PARALLEL=4`, 3-model council, k=4): the
  full-ladder worst case dropped 20.9s -> 14.1s, with the stage-2 panelists co-resident
  at 100% GPU; the verified happy path now returns on the first passing sample.

## [0.1.0] — 2026-06-19

First public release.

### Added
- Verified best-of-N engine: sample a diverse small-model council, run a real verifier on each
  candidate, return the verified-best answer. Escalates from a single model to the full council
  only when nothing verifies (fast common case, council where it pays).
- Code verifiers: `FunctionalCodeVerifier` (HumanEval-style `check(candidate)`) and
  `StdioCodeVerifier` (competitive-programming stdin/stdout, per-line comparison).
- Backends: `OllamaBackend` (local) and `OpenRouterBackend` (cloud) — stdlib only.
- CLI: `llmjury solve`, `llmjury reproduce`, `llmjury --version`.
- `llmjury reproduce humaneval|lcb` with bundled benchmark slices; `--pace` to duty-cycle local
  GPU load on long runs.
- Sandboxed execution of model-generated code: scrubbed environment, isolated temp working
  directory, POSIX CPU/file-size/core limits. Not a full sandbox — see the README's security note.
- `llmjury demo` — runs the full pipeline offline with no API key (built-in `DemoBackend`).
- Bring-your-own-tests: `FunctionalCodeVerifier.from_cases([(args, expected), ...])`, and `--tests`
  now accepts a full `def check(candidate): ...` OR just its body.
- `llmjury solve --json` (machine-readable result), `--models` / `--best` (custom council from the CLI).
- Hardened execution: process-group kill on timeout (reaps forked grandchildren), 1 MiB output cap,
  and refuses to run as root unless `LLMJURY_ALLOW_ROOT=1`.
- Opt-in **hybrid frontier escalation**: `llmjury solve --frontier MODEL` (and `Engine(frontier=...)`)
  escalates *only* the problems the local council can't verify to one frontier cloud model — a
  frontier call on the hard minority, not on every problem.
- Head-to-head [BENCHMARKS.md](BENCHMARKS.md) vs frontier one-shot and OpenRouter Fusion, including
  a measured 45-problem LiveCodeBench run: the hybrid matched-or-beat OpenRouter Fusion on every
  problem (44/45 vs 41/45, a strict superset) at ~38× lower cost.
- Offline test suite (15 tests); zero runtime dependencies.

[0.1.0]: https://github.com/ajsai47/llm-jury/releases/tag/v0.1.0
