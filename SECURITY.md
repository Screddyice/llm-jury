# Security Policy

LLM-Jury **executes model-generated code** to verify it — that is the core mechanism, and it is
inherently dangerous. Read this before pointing it at anything you don't trust.

## What v0.1 does to contain execution

Each candidate runs in a subprocess with:

- a **scrubbed environment** — none of the parent's secrets/API keys are inherited, and `HOME` is
  repointed into a throwaway temp directory,
- an **isolated temp working directory** (it can't read or clobber your project files),
- **POSIX resource limits** — `RLIMIT_CPU`, `RLIMIT_FSIZE`, and no core dumps,
- a hard **timeout** that kills the entire **process group** (so forked grandchildren are reaped),
- **captured output capped at 1 MiB** (a print-bomb can't OOM the parent).

It also **refuses to run as root** unless you set `LLMJURY_ALLOW_ROOT=1`.

## What v0.1 is NOT

This is **not a real sandbox.** No containers, namespaces, seccomp, or VM. The network is not
blocked, and memory is not hard-capped (a reliable `RLIMIT_AS` is fragile on macOS). A determined
adversary controlling the model output, the task, or the tests can likely still do harm.

**For untrusted input, run LLM-Jury inside a container or a VM.** Don't run it as root, and don't run
it on a machine holding secrets you can't afford to expose.

## Reporting a vulnerability

If you find a way past the containment (reads host env, escapes the temp dir, evades the limits,
reaches the network in a way that matters), please report it privately via GitHub's **private
vulnerability reporting** (the repository's *Security* tab) rather than opening a public issue.
We'll acknowledge within a few days.

## Supported versions

`v0.1.x` is pre-1.0 and best-effort. Pin a version if you depend on it.
