# LLM-Jury

LLM-Jury generates candidate Python solutions and returns only code that passes an
independent verifier. Provider diversity is useful only when a real oracle exists.

## Verification

```bash
python3 tests/test_llmjury.py
python3 -m llmjury.cli demo
git diff --check
```

Tests must remain offline by default. Mock provider processes and HTTP calls in unit tests.

### Verifying against a live provider

The completion cache turns a repeated live run into a replay. `~/.llmjury/cache.jsonl`
keys on (backend, model, temperature, max_tokens, sample index, prompt), so re-running
an identical `llmjury solve` returns stored generations and never reaches the provider.
The verifier still executes, so the run looks and reports exactly like a fresh one.

Before claiming a change works against a real provider, prove the generation was live:

```bash
before=$(wc -l < ~/.llmjury/cache.jsonl)
llmjury solve --task task.txt --cases cases.json --entry-point solve \
    --backend ollama --frontier auto
after=$(wc -l < ~/.llmjury/cache.jsonl)
echo "new cache entries: $((after - before))"   # 0 means nothing was generated
```

Two checks that look convincing and prove nothing:

- **The cache file's mtime.** It reflects whichever run wrote last, which is easy to
  read as the current one when runs are minutes apart.
- **Grepping the cache for a model name.** Keys are hashed, so a grep returns nothing
  whether or not the model ran.

For a genuinely cold run, move the cache aside and restore it from a shell trap, so an
interrupted or failed run cannot leave it missing:

```bash
trap 'mv -f ~/.llmjury/cache.jsonl.bak ~/.llmjury/cache.jsonl' EXIT INT TERM
mv ~/.llmjury/cache.jsonl ~/.llmjury/cache.jsonl.bak
# ... run ...
```

Confirm the restored file matches the original with `shasum -a 256`.

`LLMJURY_SANDBOX=off` prints a `sandbox=host` warning whether or not a container runtime
is available. It reports that the sandbox was disabled, not that Docker is down — check
`docker info` before concluding otherwise. A sandboxed run prints `sandbox=container`.

## Provider Boundaries

- Ollama is the local/private council backend.
- Codex CLI is an authenticated OpenAI provider and may be used directly or as the
  frontier tier. It must run ephemeral and read-only for candidate generation: no tools,
  no subagents, no memory, one turn, empty temp cwd. It authenticates from its own
  session, so escalating to it spends no metered credit — inside Codex, prefer it over an
  OpenRouter ladder. Named ladders resolve to OpenRouter slugs and must keep rejecting
  `--frontier-backend codex`.
- Agent frontmatter must keep `description:` a quoted YAML scalar. Claude Code parses it
  leniently, but under a strict YAML parser an unquoted description containing a
  colon-space reads as a mapping and the agent is silently dropped, with no error
  anywhere. A test enforces the quoting. The rule is agents only; skill frontmatter is
  parsed leniently.
- The shipped agent must stay free of a `model:` pin. Hosts disagree about a pin they
  cannot resolve, and at least one of them fails silently: a host that ignores the pin
  runs the agent on its own session model, so an agent pinned to a local model quietly
  executes on a cloud one. Sessions pinned to the official Anthropic API (the Claude
  desktop app, cron) instead reject the unknown model outright. Either way an unpinned
  agent is correct everywhere, while a pinned one is wrong somewhere — and possibly wrong
  without saying so, which is the case worth designing against.
- OpenRouter is a model-agnostic cloud provider. The backend must stay model-agnostic:
  no Anthropic-specific request shapes, response parsing, or auth paths. Model choice
  is policy, expressed as data in `llmjury.panels`, never wired into transport code.
- `--frontier auto` ends on a proprietary top tier (`TOP_FRONTIER`, Claude Opus 5) after
  the open-weight ladder. That is a deliberate default-policy choice, not a coupling:
  it is one env-overridable constant (`LLMJURY_TOP_FRONTIER`), and `--frontier open`
  keeps escalation open-weight only. Keep both paths working.
- Cost ordering is a correctness property of the ladder, not a preference. The
  proprietary tier costs ~35× the open-weight tiers, so it must stay last and stay
  verifier-gated; a test asserts that ordering. Never fan out across tiers.
- The ordering earns its keep, measured rather than assumed. Well-known algorithmic
  problems are solved by the local council alone, even under an oracle that enforces an
  O(n log n) bound — their solutions are memorised. What escalates is a task whose shape
  is genuinely new, and the first cloud tier absorbs most of that. Treat any change that
  moves work to a later tier by default as a regression until a benchmark says otherwise.
- The recommended hybrid route is `--backend ollama --frontier auto`: local council
  first, then the ordered OpenRouter ladder only when verification fails.
- The verifier, not the provider, decides which candidate is accepted.
- Never expose auth files, API keys, or the parent process environment to generated code.
