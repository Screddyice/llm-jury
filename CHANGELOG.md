# Changelog

## Unreleased

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
