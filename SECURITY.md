# Security Policy

LLM-Jury **executes model-generated code** to verify it — that is the core mechanism, and it is
inherently dangerous. Read this before pointing it at anything you don't trust.

## What v0.1 does to contain verification

By default (`LLMJURY_SANDBOX=auto`), each candidate runs in a throwaway container
with no network, dropped capabilities, a non-root user, memory/CPU/process limits,
and a read-only root filesystem. If Docker or Colima cannot be provisioned, auto mode
prints a warning and uses the hardened host runner. Set `LLMJURY_SANDBOX=docker` to
require the container instead of allowing that fallback.

The hardened host runner uses:

- a **scrubbed environment** — none of the parent's secrets/API keys are inherited, and `HOME` is
  repointed into a throwaway temp directory,
- an **isolated temp working directory** (it can't read or clobber your project files),
- **POSIX resource limits** — `RLIMIT_CPU`, `RLIMIT_FSIZE`, and no core dumps,
- a hard **timeout** that kills the entire **process group** (so forked grandchildren are reaped),
- **captured output capped at 1 MiB** (a print-bomb can't OOM the parent).

It also **refuses to run as root** unless you set `LLMJURY_ALLOW_ROOT=1`.

## What the host fallback is NOT

The host fallback is **not a real sandbox.** It has no namespaces, seccomp, or VM; the network is
not blocked, and memory is not hard-capped (a reliable `RLIMIT_AS` is fragile on macOS). A
determined adversary controlling the model output, the task, or the tests can likely still do harm.

**For untrusted input, run LLM-Jury inside a container or a VM.** Don't run it as root, and don't run
it on a machine holding secrets you can't afford to expose.

## Delegated repository execution

`llmjury delegate` intentionally has a different trust boundary from candidate
verification. It launches an authenticated Codex agent with `workspace-write` access
to the directory selected by the caller. That agent can read and modify files in the
workspace and run commands there. Use it only in repositories you trust and inspect
the resulting diff.

The delegation command does not expose Codex's danger-full-access or approval-bypass
modes. It runs ephemerally, keeps repository instructions enabled, requests Codex's
minimal `core` shell environment, and requires explicit `--add-dir` flags for any
additional writable directory. These controls reduce scope; they do not make an
untrusted task or repository safe. Never include secrets in a delegated task brief.

`llmjury plan` gives Claude Code only `Read`, `Glob`, and `Grep` tools in plan
permission mode, runs without session persistence, and does not grant Bash or edit
tools. It can still read the selected repository, so use it only with workspaces and
task briefs that the configured Claude account is allowed to inspect.

## Reporting a vulnerability

If you find a way past the containment (reads host env, escapes the temp dir, evades the limits,
reaches the network in a way that matters), please report it privately via GitHub's **private
vulnerability reporting** (the repository's *Security* tab) rather than opening a public issue.
We'll acknowledge within a few days.

## Supported versions

`v0.1.x` is pre-1.0 and best-effort. Pin a version if you depend on it.
