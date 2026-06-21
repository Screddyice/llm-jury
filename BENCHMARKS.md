# Benchmarks

Head-to-head on the **same problems, judged by the same oracle**. Two honest headlines:

1. **Cost (measured, rock-solid):** the local-first **hybrid** ties OpenRouter Fusion on a
   12-problem head-to-head — both 12/12 — for **~36× less money** ($0.169 vs $6.04). On the full
   45 it scores **44/45 (97.8%)** for **$0.49**, an **extrapolated ~46×** cost advantage.
2. **What we do *not* claim:** that Litmus is *more accurate* than a frontier cloud fusion. A
   12/12-vs-12/12 tie on n=12 cannot prove accuracy parity (the sample is too small — see the CI
   caveat). The cost gap is measured; the accuracy comparison is a tie on a small sample, no more.

The numbers below, and the caveats that bound them, are why.

## The three configurations

| Config | What runs | Local | Private | Verified |
|---|---|---|---|---|
| **Council** | 3 small open models (phi-4, gemma-3-12b, llama-3.1-8b), verified best-of-9 | ✓ | ✓ | ✓ |
| **Hybrid** | Council, then escalate *only* the problems the verifier can't pass to **one** frontier model (best-of-4), re-verified | ✓ for ~75%; cloud for the hard ~25% | partial | ✓ |
| **OpenRouter Fusion** | Closed cloud: fans out to ~8 frontier models + a judge, on **every** problem | ✗ | ✗ | ✗ |

## Hard code — LiveCodeBench, 12-problem head-to-head

The only direct, same-problems, same-oracle comparison with Fusion (problems abc301–303, the
first 12 in sequence):

| Approach | Accuracy | Cost (12 problems) | Local | Private | Verified |
|---|---|---|---|---|---|
| Single frontier model (DeepSeek V4-Pro), one shot | 66.7% | ~$0.12 | ✗ | ✗ | ✗ |
| OpenRouter Fusion (closed, frontier panel + judge) | 12/12 (100%) | **$6.04** | ✗ | ✗ | ✗ |
| **Litmus hybrid** (local council + 1 frontier escalation) | 12/12 (100%) | **$0.169** | ◐ | ◐ | ✓ |
| **Litmus council** (small open council, no escalation) | 9/12 (75%) | **$0.008** · $0 local | ✓ | ✓ | ✓ |

**Litmus hybrid ties OpenRouter Fusion (both 12/12) at ~36× lower cost** ($0.169 vs $6.04 —
$0.008 council + $0.161 frontier). The cost ratio is measured to the cent. The accuracy is a
**tie on n=12** — see the caveat: this sample cannot resolve which system is actually more
accurate; each arm's 95% confidence interval runs roughly 74–100%.

## Hybrid escalation — full LiveCodeBench (45 medium/hard problems)

The council solves 34/45 locally. The hybrid escalates **only the 11 it can't verify** to one
frontier model (DeepSeek V4-Pro, best-of-4), re-checked by the same oracle:

| Stage | Result | Cost |
|---|---|---|
| Council alone (local, verified best-of-9) | 34/45 (75.6%) | $0.031 · $0 local |
| **+ frontier escalation on the 11 failures** | **+10 recovered → 44/45 (97.8%)** | + $0.456 |
| **Hybrid total** | **44/45 (97.8%)** · 95% CI ≈ 88–100% | **$0.487** |
| OpenRouter Fusion (*extrapolated* to 45) | — | **~$22.65** |

- **34 solved locally** (free, private); **10 of the 11 council failures recovered** by one
  frontier escalation; one genuine remaining failure (`abc314_e` — all 4 frontier samples
  returned real code, none dropped).
- **~46× cheaper than Fusion — extrapolated.** Fusion was run only on the first 12 problems;
  $22.65 = ($6.04 / 12) × 45. The difficulty mix is close (first-12: 42% hard; full-45: 40%
  hard), so the projection is reasonable but **was not measured on the other 33**. The 36× on the
  12-problem head-to-head *is* measured.

## Easy code — HumanEval+ (82 problems)

| Approach | Accuracy |
|---|---|
| **Litmus** (small council + verifier) | **97.6%** |
| Frontier (DeepSeek) one shot | 97.6% |

On easier code where the small models are capable, the council **matches frontier** outright with
no escalation needed.

## Caveats — read these next to the numbers

These bound exactly what the headline means. They came out of an adversarial audit of our own
claims; we kept the ones the data forced us to keep.

1. **Small-n on accuracy.** The 12/12-vs-12/12 tie is on **n=12**; each arm's 95% CI spans
   ~74–100%. The sample establishes the **cost** gap, **not** accuracy parity — it cannot
   distinguish the two systems' accuracy. Telling 97.8% from 100% would take ~350 problems.
2. **Full-run interval.** 44/45 = 97.8% carries a 95% Wilson CI of about **[88%, 100%]**. The
   recovery sub-rate driving it (10/11 = 91%) has a CI of roughly [62%, 98%]. Read the point
   estimate with its interval.
3. **The ~46× is extrapolated; the ~36× is measured.** Fusion ran on 12 problems only (abc301–303).
   $22.65 is projected from its measured per-problem cost. The 36× head-to-head is direct.
4. **The oracle is public sample tests only.** Both systems are judged by the **same** oracle —
   the problem's public sample cases (1–4 per problem, mean 2.8; two problems have a single case).
   This is **not** AtCoder's hidden judge suite. "100%" and "97.8%" mean "passes the public
   samples," not "provably correct in competition." The leniency is **identical for both systems**,
   so the *comparison* is fair; only the absolute accuracy is inflated vs full hidden-test scoring.
5. **"Local" describes the product, not this run.** These numbers were measured with the council
   served via OpenRouter for reproducibility (measured council cost $0.031; a genuine local run is
   $0). The 75% "local" figure is an architectural property of local-capable models (Ollama:
   phi-4 / gemma-3-12b / llama-3.1-8b), not of the run that produced these dollars.
6. **Escalated problems leave the device.** The ~24% the council can't verify — disproportionately
   the hardest (7 of 11 are "hard") — are sent to a cloud frontier model. **"Local-first" is not
   "fully private."** The hybrid keeps most work local; it does not keep all of it local.
7. **Not sample-for-sample.** The hybrid spends best-of-9 (council) + best-of-4 (frontier on
   failures); Fusion is one shot per problem. The claim is **system-vs-system at the stated cost**,
   not per-sample parity.

*Footnote (verified non-issue):* the council arm scored whole-output `.strip()` while the hybrid
oracle is per-line; re-running all 855 cached council candidates under the strict per-line oracle
yields the **identical 34/45** — zero verdict flips, so no result depends on it. *Available upside
not claimed:* 9 of the 10 recoveries passed on the first frontier sample, so a sequential
early-stop escalation would cost less than the best-of-4 figure above; we report the higher
measured cost.

## What this means

Three jobs, three winners:

- **Maximum accuracy, cost no object, cloud is fine** → a frontier fusion (OpenRouter Fusion).
- **Cheapest verifiable answers, fully local and private** → the **council** (75.6% on hard code
  for 3 cents, $0 local).
- **Fusion-class results without the Fusion bill** → the **hybrid**: ties Fusion on the measured
  head-to-head for ~1/36th the cost, keeps 75% of work local, and pays for a frontier call only on
  the hard minority it can't verify itself.

The honest one-liner: **same accuracy as the cloud leader on the problems we could compare,
at a fraction of the cost — with most of the work never leaving your machine.** The cost claim is
measured; the accuracy claim is a tie on a small sample, stated as exactly that.

## Reproduce it

- Council / single-model numbers: `litmus reproduce lcb` and `litmus reproduce humaneval`.
- Hybrid + Fusion comparison harnesses live in the research repo (`crucible-ablation/`):
  `hybrid_bench.py` (council + frontier escalation, same oracle) and `fusion_bench.py` (Fusion via
  `openrouter/fusion`). Both share byte-identical `oracle()` / `_norm()` and the same public-test
  cases, so the cost column is apples-to-apples and the accuracy column is same-oracle.
