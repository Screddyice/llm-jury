# Benchmarks

Head-to-head on the **same problems, judged by the same oracle**. We don't claim Litmus is the
most *accurate* tool on the market — a frontier cloud fusion is more accurate on hard problems.
We claim it's the best **price-performance** option, and the **only** one that's local, private,
and verified. The numbers below are why.

## Hard code — LiveCodeBench (medium/hard), 12 problems, public-test oracle

| Approach | Accuracy | Cost (12 problems) | Local | Private | Verified |
|---|---|---|---|---|---|
| Single frontier model (DeepSeek V4-Pro), one shot | 66.7% | ~$0.12 | ✗ | ✗ | ✗ |
| OpenRouter Fusion (closed, frontier panel + judge) | **100%** | **$6.04** | ✗ | ✗ | ✗ |
| **Litmus** (small open council + real verifier) | **75.0%** | **$0.008** · $0 local | ✓ | ✓ | ✓ |

- **Fusion is the most accurate** (frontier-model panel) — and the most expensive by far.
- **Litmus beats the frontier *default*** (the single one-shot call most people actually make).
- **Litmus is ~700× cheaper than Fusion** on identical work ($0.008 vs $6.04), and **free** run locally.
- **Litmus is the only one you can run offline** — the others send your code to the cloud and return
  an answer you must trust; Litmus proves its answer passes the tests.

Fusion's 100% is on 12 problems judged by *public* tests (lenient, small sample) — read it as
"clearly stronger on hard code," not "perfect."

## Full LiveCodeBench run (45 hard/medium problems)

| Arm | Accuracy | Cost (45) | $/correct |
|---|---|---|---|
| One small model, one shot (Phi-4) | 11% | — | — |
| One small model + verified best-of-9 | 62.2% | — | — |
| **Small council + verified best-of-9** | **75.6%** | **$0.031** | **$0.0009** |
| Frontier (DeepSeek) one shot | 62.2% | ~$0.45 | — |

The council's **entire cost for 45 hard problems was 3.1 cents.**

## Easy code — HumanEval+ (82 problems)

| Approach | Accuracy |
|---|---|
| **Litmus** (small council + verifier) | **97.6%** |
| Frontier (DeepSeek) one shot | 97.6% |

On easier code where the small models are capable, Litmus **matches frontier** outright.

## What this means

There are two different jobs, and two different winners:

- **Maximum accuracy, cost no object, cloud is fine** → a frontier fusion (OpenRouter Fusion) wins.
- **Frontier-default-or-better accuracy on verifiable code, for ~1/700th the cost, fully local,
  private, and provably correct** → **nothing beats Litmus**, because nothing else is even in that
  quadrant.

Litmus is the efficiency-and-trust champion, not the accuracy champion. For most people writing
verifiable code on a machine they already own, that's the trade that matters.

## Reproduce it

- Litmus / single-model / council numbers: `litmus reproduce lcb` and `litmus reproduce humaneval`.
- The Fusion and frontier comparison harnesses live in the research repo (`crucible-ablation/`):
  `fusion_bench.py` (Fusion via `openrouter/fusion`, same oracle) and `lcb_ablation.py`.
- Every comparison uses the identical public-test oracle, so the accuracy column is apples-to-apples;
  the cost column is each approach's real measured spend.
