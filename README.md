# LLM-Jury

[![PyPI](https://img.shields.io/pypi/v/llm-jury-verify)](https://pypi.org/project/llm-jury-verify/)
[![Python](https://img.shields.io/pypi/pyversions/llm-jury-verify)](https://pypi.org/project/llm-jury-verify/)
[![CI](https://img.shields.io/github/actions/workflow/status/ajsai47/llm-jury/ci.yml?label=ci)](https://github.com/ajsai47/llm-jury/actions)
[![dependencies](https://img.shields.io/badge/dependencies-0-success)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[Quickstart](#quickstart) · [How it works](#how-it-works) · [Benchmarks](#benchmarks) · [Write-up →](https://app.notion.com/p/3844834c4d7881d1adaeed9c3a81dcbb) · [Paper →](https://app.notion.com/p/3874834c4d78817f99a0fc26088ed7e4)

**Local verified answers. Don't vote, verify.**

A frontier API gives you one answer and asks you to trust it. LLM-Jury gives you an answer it can
**prove** — by running the tests — and does most of the work on your own laptop, for free.

It runs model-generated code through a real verifier and returns only what **provably passes** —
not the answer that got the most votes. The result: a council of small open models on your machine,
plus *one* opt-in frontier escalation on the hard minority, **matches a commercial frontier-model
fusion on hard code — at ~38× lower cost.**

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

> **⚠️ Security.** LLM-Jury *executes model-generated code* to verify it — that's the whole
> point, and it's inherently risky. v0.1 isolates execution (scrubbed environment, isolated
> temp directory, CPU/file-size limits) but is **not a real sandbox**. Don't run untrusted
> tasks on a machine with secrets, and don't run as root. For real isolation, run it inside a
> container or VM.

---

## Quickstart

**CLI** (examples ship in the repo):

```bash
# functional tests (the body of a check(candidate) function)
llmjury solve --task examples/add_task.txt --tests examples/add_test.py --entry-point add

# or competitive-programming style (stdin/stdout cases as JSON)
llmjury solve --task examples/sum_stdin_task.txt --cases examples/cases.example.json

# hybrid: run the local council, and escalate ONLY what it can't verify to one
# frontier model — Fusion-class accuracy, a frontier call on the hard minority, not on everything
llmjury solve --task examples/sum_stdin_task.txt --cases examples/cases.example.json \
    --backend ollama --frontier deepseek/deepseek-v4-pro   # needs OPENROUTER_API_KEY
```

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

## How it works

Three small models you can run for free on a laptop, plus a verifier, match or beat a frontier
model on verifiable work. The engine is a tiered pipeline that *gates fusion on the verifier*:

```
task + verifier
  → GENERATE   sample candidates across a cross-lineage small-model council
  → VERIFY     run the verifier on each (code = run the tests)
  → SELECT     return the one that provably passes
  → ESCALATE   one model first → add the council only when nothing passes
               → (optional) one frontier model only when the council can't either
```

That tiered escalation is what keeps it laptop-friendly *and* lets it reach frontier-class
accuracy: the common case runs **one** model fast, hard problems load the full local council, and —
if you opt in with `--frontier` — the small hard minority the council can't verify escalates to a
single cloud model. You pay for the frontier only on the problems that need it, not on every call.

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

## Backends

- **Ollama (local, free, private)** — `--backend ollama`. Pull the council first:
  ```bash
  ollama pull phi4 && ollama pull gemma3:12b && ollama pull llama3.1:8b
  ```
- **OpenRouter (cloud)** — `--backend openrouter` (default). Set `OPENROUTER_API_KEY`.

A diverse, cross-lineage panel (different labs → different mistakes) is the point. The defaults
are Phi-4 (Microsoft) + Gemma-3-12B (Google) + Llama-3.1-8B (Meta); swap your own in
`llmjury.panels` or by passing `panel=[...]` to `Engine`.

## Reproduce the benchmarks

```bash
llmjury reproduce humaneval            # bundled 25-problem slice
llmjury reproduce lcb --n 5            # quick check
llmjury reproduce lcb --backend ollama  # run it locally
```

It runs the council on a bundled benchmark slice and reports how many problems the single best
model solves versus what the diverse council adds on escalation — the "council adds coverage"
story, in one command. (The bundled slices are 25 problems each; the headline 97.6% / 75.6%
figures are the larger full runs from the write-up.)

## Status

`v0.1` — working engine, code verifiers (functional + stdin/stdout), Ollama + OpenRouter
backends, escalating council, and `llmjury reproduce`. Next: a real sandbox (container) for
untrusted input, more verifiers (math, citations), and a task classifier that picks the
strategy automatically.

## License

MIT © The AI Collective
