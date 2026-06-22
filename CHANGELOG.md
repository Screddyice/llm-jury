# Changelog

Notable changes to Litmus. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/).

## [0.1.0] — 2026-06-19

First public release.

### Added
- Verified best-of-N engine: sample a diverse small-model council, run a real verifier on each
  candidate, return the verified-best answer. Escalates from a single model to the full council
  only when nothing verifies (fast common case, council where it pays).
- Code verifiers: `FunctionalCodeVerifier` (HumanEval-style `check(candidate)`) and
  `StdioCodeVerifier` (competitive-programming stdin/stdout, per-line comparison).
- Backends: `OllamaBackend` (local) and `OpenRouterBackend` (cloud) — stdlib only.
- CLI: `litmus solve`, `litmus reproduce`, `litmus --version`.
- `litmus reproduce humaneval|lcb` with bundled benchmark slices; `--pace` to duty-cycle local
  GPU load on long runs.
- Sandboxed execution of model-generated code: scrubbed environment, isolated temp working
  directory, POSIX CPU/file-size/core limits. Not a full sandbox — see the README's security note.
- `litmus demo` — runs the full pipeline offline with no API key (built-in `DemoBackend`).
- Bring-your-own-tests: `FunctionalCodeVerifier.from_cases([(args, expected), ...])`, and `--tests`
  now accepts a full `def check(candidate): ...` OR just its body.
- `litmus solve --json` (machine-readable result), `--models` / `--best` (custom council from the CLI).
- Hardened execution: process-group kill on timeout (reaps forked grandchildren), 1 MiB output cap,
  and refuses to run as root unless `LITMUS_ALLOW_ROOT=1`.
- Opt-in **hybrid frontier escalation**: `litmus solve --frontier MODEL` (and `Engine(frontier=...)`)
  escalates *only* the problems the local council can't verify to one frontier cloud model — a
  frontier call on the hard minority, not on every problem.
- Head-to-head [BENCHMARKS.md](BENCHMARKS.md) vs frontier one-shot and OpenRouter Fusion, including
  a measured 45-problem LiveCodeBench run: the hybrid matched-or-beat OpenRouter Fusion on every
  problem (44/45 vs 41/45, a strict superset) at ~38× lower cost.
- Offline test suite (15 tests); zero runtime dependencies.

[0.1.0]: https://github.com/ajsai47/litmus/releases/tag/v0.1.0
