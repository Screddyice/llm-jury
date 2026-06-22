# Litmus

**Local verified answers. Don't vote, verify.**

Litmus runs a diverse council of small open models *on your own machine*, checks every
attempt against a real verifier, and hands back the answer that **provably passes**. On hard
code it matches — and beats — a frontier model's out-of-the-box answer. Fully local, fully
free, fully private.

Zero dependencies. Stdlib only.

```bash
pip install litmus-verify
```

**Try it in 5 seconds — no API key, nothing to download:**

```bash
litmus demo
```

It runs the real generate → verify → escalate pipeline on a canned task with a built-in offline
backend: a "weak" model returns a wrong answer, the verifier catches it, and the council's answer
passes. That's the whole product, offline.

> **⚠️ Security.** Litmus *executes model-generated code* to verify it — that's the whole
> point, and it's inherently risky. v0.1 isolates execution (scrubbed environment, isolated
> temp directory, CPU/file-size limits) but is **not a real sandbox**. Don't run untrusted
> tasks on a machine with secrets, and don't run as root. For real isolation, run it inside a
> container or VM.

---

## Why

A frontier API gives you one answer and asks you to trust it. Litmus gives you an answer it
can **prove** — by running the tests. And it turns out three small models you can run for free
on a laptop, plus a verifier, match or beat a frontier model on verifiable work:

| Benchmark | Litmus (laptop council) | Frontier model, one shot |
|---|---|---|
| HumanEval+ (code) | **97.6%** | 97.6% |
| LiveCodeBench (hard code) | **75.6%** | 62.2% |

On *hard* code, the small-model council **beats** the frontier model by 13 points. The harder
the problem, the bigger the win. ([Read the full write-up →](https://app.notion.com/p/3844834c4d7881d1adaeed9c3a81dcbb))

> Caveat, stated honestly: the frontier number is a single shot; Litmus spends best-of-N + a
> verifier (your laptop's compute). The claim is "a laptop matches/beats the frontier model's
> *out-of-the-box* answer," not "wins sample-for-sample." For someone choosing between their
> laptop and an API bill, that's the comparison that matters.

## Benchmarks

Measured head-to-head on the same hard LiveCodeBench problems, judged by the same oracle — the
40 problems where OpenRouter Fusion actually ran ([full numbers + methodology + caveats](BENCHMARKS.md)):

| | Accuracy (n=40) | 95% CI | Cost | Local | Verified |
|---|---|---|---|---|---|
| **Litmus hybrid** (council + 1 frontier escalation) | **39/40 (97.5%)** | [87%, 99.6%] | **$0.49** | ◐ | ✓ |
| OpenRouter Fusion (cloud) | 37/40 (92.5%) | [80%, 97.4%] | $16.96 | ✗ | ✗ |
| **Litmus council** (small open council, local) | 34/45 (75.6%) | — | **$0.031** · free local | ✓ | ✓ |

**The hybrid matches OpenRouter Fusion's accuracy — measured, not extrapolated — at ~35× less
money** (39/40 vs 37/40, $0.49 vs $16.96). It solves 75% of problems with a local council and
escalates only the hard minority to a single frontier model. The plain **council**, with no cloud
at all, gets 75.6% on hard code for 3 cents — and on easier code (HumanEval+) **matches frontier
outright** (97.6%).

Stated honestly: the **cost** gap is measured and large; the **accuracy** result is a *statistical
tie* — the hybrid is nominally ahead by 2 problems, but the confidence intervals overlap, so read
it as parity, not a win. We don't claim Litmus is *more* accurate than a frontier fusion. (Fusion
ran on 40 of 45 — 5 problems hit a credit cap and are excluded from both arms, not counted as
losses; the hybrid solved all 5.) "Correct" here means "passes the public sample tests," and the
escalated ~25% do leave the device. Full caveats — confidence intervals, the public-test oracle,
the credit cap — are in [BENCHMARKS.md](BENCHMARKS.md).

## Quickstart

**CLI** (examples ship in the repo):

```bash
# functional tests (the body of a check(candidate) function)
litmus solve --task examples/add_task.txt --tests examples/add_test.py --entry-point add

# or competitive-programming style (stdin/stdout cases as JSON)
litmus solve --task examples/sum_stdin_task.txt --cases examples/cases.example.json

# hybrid: run the local council, and escalate ONLY what it can't verify to one
# frontier model — Fusion-class accuracy, a frontier call on the hard minority, not on everything
litmus solve --task examples/sum_stdin_task.txt --cases examples/cases.example.json \
    --backend ollama --frontier deepseek/deepseek-v4-pro   # needs OPENROUTER_API_KEY
```

**Python:**

```python
from litmus import solve, FunctionalCodeVerifier, OllamaBackend

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
from litmus import FunctionalCodeVerifier
verifier = FunctionalCodeVerifier.from_cases("add", [((2, 3), 5), ((-1, 1), 0)])
```

(`--tests` also accepts either a full `def check(candidate): ...` or just its body. And
`litmus solve --json` emits a machine-readable result for piping into CI/agents.)

## How it works

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

**The rule Litmus is built on:** combining models helps *exactly when you can check the answer
cheaply and independently.* Code has a real oracle (run the tests), so the council wins. Tasks
with no oracle (open-ended reasoning) collapse to voting, where a council *hurts* — so there,
Litmus uses the single best model. The verifier, not the model count, is what decides.

## Backends

- **Ollama (local, free, private)** — `--backend ollama`. Pull the council first:
  ```bash
  ollama pull phi4 && ollama pull gemma3:12b && ollama pull llama3.1:8b
  ```
- **OpenRouter (cloud)** — `--backend openrouter` (default). Set `OPENROUTER_API_KEY`.

A diverse, cross-lineage panel (different labs → different mistakes) is the point. The defaults
are Phi-4 (Microsoft) + Gemma-3-12B (Google) + Llama-3.1-8B (Meta); swap your own in
`litmus.panels` or by passing `panel=[...]` to `Engine`.

## Reproduce the benchmarks

```bash
litmus reproduce humaneval            # bundled 25-problem slice
litmus reproduce lcb --n 5            # quick check
litmus reproduce lcb --backend ollama  # run it locally
```

It runs the council on a bundled benchmark slice and reports how many problems the single best
model solves versus what the diverse council adds on escalation — the "council adds coverage"
story, in one command. (The bundled slices are 25 problems each; the headline 97.6% / 75.6%
figures are the larger full runs from the write-up.)

## Status

`v0.1` — working engine, code verifiers (functional + stdin/stdout), Ollama + OpenRouter
backends, escalating council, and `litmus reproduce`. Next: a real sandbox (container) for
untrusted input, more verifiers (math, citations), and a task classifier that picks the
strategy automatically.

## License

MIT © The AI Collective
