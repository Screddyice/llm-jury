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
- Offline test suite; zero runtime dependencies.

[0.1.0]: https://github.com/ajsai47/litmus/releases/tag/v0.1.0
