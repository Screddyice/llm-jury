# Benchmarks

Head-to-head on the **same problems, judged by the same oracle**. Two honest headlines:

1. **The hybrid matches the cloud leader's accuracy — now measured, not extrapolated.** On a
   40-problem head-to-head where OpenRouter Fusion actually ran, the local-first hybrid scores
   **39/40 (97.5%)** vs Fusion's **37/40 (92.5%)** — a statistical tie (the hybrid is nominally
   ahead by 2; the confidence intervals overlap) — for **~35× less money** ($0.49 vs $16.96).
2. **What we still do *not* claim:** that the hybrid is *more* accurate than Fusion. +2 problems
   at n=40 is within noise. The honest claim is **parity at a fraction of the cost**, and now the
   accuracy comparison is backed by a real 40-problem run with genuine statistical power — not the
   earlier n=12 tie that couldn't resolve anything.

The numbers below, and the caveats that bound them, are why.

## The three configurations

| Config | What runs | Local | Private | Verified |
|---|---|---|---|---|
| **Council** | 3 small open models (phi-4, gemma-3-12b, llama-3.1-8b), verified best-of-9 | ✓ | ✓ | ✓ |
| **Hybrid** | Council, then escalate *only* the problems the verifier can't pass to **one** frontier model (best-of-4), re-verified | ✓ for ~75%; cloud for the hard ~25% | partial | ✓ |
| **OpenRouter Fusion** | Closed cloud: fans out to ~8 frontier models + a judge, on **every** problem | ✗ | ✗ | ✗ |

## Measured head-to-head — LiveCodeBench, 40 problems

The full comparison, same problems, byte-identical oracle. (Fusion ran on 40 of the 45; the last
5 hit a credit cap and never executed — see below. They are **excluded from both arms** rather
than scored as Fusion losses.)

| Approach | Accuracy | 95% CI | Cost | Local | Private | Verified |
|---|---|---|---|---|---|---|
| **Litmus hybrid** (council + 1 frontier escalation) | **39/40 (97.5%)** | [87.1%, 99.6%] | **$0.49** | ◐ | ◐ | ✓ |
| OpenRouter Fusion (closed, ~8-model panel + judge) | 37/40 (92.5%) | [80.1%, 97.4%] | $16.96 | ✗ | ✗ | ✗ |
| **Litmus council** (small open council, local, no cloud) | 34/45 (75.6%) | [60.8%, 85.9%] | $0.031 · $0 local | ✓ | ✓ | ✓ |

- **The hybrid is tied with Fusion, nominally ahead by 2, at ~35× lower cost.** It solves two
  problems Fusion gets wrong (`abc311_c`, `abc314_f`); both miss exactly one genuinely brutal
  problem (`abc314_e`). The cost figures are measured to the cent — the hybrid's $0.49 covers all
  45 problems; Fusion's $16.96 covers the 40 it completed, so ~35× is if anything conservative.
- **Statistical honesty:** 39/40 vs 37/40 is a 2-problem gap with overlapping CIs — read it as a
  *tie*, not a win. But it is a tie established on **n=40 measured**, which is the upgrade: the
  accuracy comparison is no longer extrapolated or underpowered.

### The credit cap (full disclosure)

Fusion's run exhausted its OpenRouter credits at $16.96, and the last **5 problems never executed**
(HTTP 402 → empty responses): `abc312_e`, `abc312_f`, `abc313_b`, `abc314_d`, `abc315_d`. Counting
those as Fusion failures would have produced a dishonest "37/45 = 82%." They are un-run, not wrong,
so they are dropped from **both** systems. (For the record, the hybrid solved all 5 — so completing
the Fusion run could only hold or widen the hybrid's edge, never reverse it.)

## 12-problem head-to-head (the original direct subset)

The first 12 problems (abc301–303), where both systems went 12/12 — kept for continuity:

| Approach | Accuracy | Cost (12) |
|---|---|---|
| OpenRouter Fusion | 12/12 (100%) | $6.04 |
| **Litmus hybrid** | 12/12 (100%) | **$0.169** |
| Litmus council | 9/12 (75%) | $0.008 · $0 local |

On this slice the cost ratio is ~36×. The 40-problem run above supersedes it for the accuracy
comparison (n=12 had no power to resolve a tie).

## Hybrid escalation — how the 44/45 is built

The council solves 34/45 locally. The hybrid escalates **only the 11 it can't verify** to one
frontier model (DeepSeek V4-Pro, best-of-4), re-checked by the same oracle:

| Stage | Result | Cost |
|---|---|---|
| Council alone (local, verified best-of-9) | 34/45 (75.6%) | $0.031 · $0 local |
| **+ frontier escalation on the 11 failures** | **+10 recovered → 44/45 (97.8%)** | + $0.456 |
| **Hybrid total (all 45)** | **44/45 (97.8%)** · 95% CI ≈ 88–100% | **$0.487** |

**34 solved locally** (free, private); **10 of the 11 council failures recovered** by one frontier
escalation; one genuine remaining failure (`abc314_e` — all 4 frontier samples returned real code,
none dropped; Fusion misses it too).

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

1. **The accuracy result is a tie, not a win.** 39/40 vs 37/40 is +2 problems with overlapping 95%
   CIs ([87.1%, 99.6%] vs [80.1%, 97.4%]). Read it as parity. It is now measured at n=40 (real
   power), which is the only change from the earlier underpowered n=12 tie — but it still does not
   license "beats Fusion."
2. **Point estimates carry intervals.** Hybrid 44/45 = 97.8% has a 95% Wilson CI of ~[88%, 100%];
   the recovery sub-rate driving it (10/11 = 91%) has a CI of ~[62%, 98%]. Read points with bands.
3. **The cost gap is measured; 5 Fusion problems are credit-capped.** Fusion ran on 40 of 45 (the
   rest hit a 402 credit cap and are excluded from both arms, not counted as losses). The ~35× cost
   advantage is from the measured $16.96 (Fusion, 40 problems) vs $0.49 (hybrid, all 45). No
   extrapolation — the earlier "~46× projected" is retired in favor of this measured ~35×.
4. **The oracle is public sample tests only.** Both systems are judged by the **same** oracle —
   the problem's public sample cases (1–4 per problem, mean 2.8; two problems have a single case).
   This is **not** AtCoder's hidden judge suite. "97.5%" and "92.5%" mean "passes the public
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

Three jobs, three configurations:

- **Cheapest verifiable answers, fully local and private** → the **council** (75.6% on hard code
  for 3 cents, $0 local, nothing leaves your machine).
- **Cloud-leader accuracy without the cloud-leader bill** → the **hybrid**: tied with OpenRouter
  Fusion on a measured 40-problem head-to-head (39/40 vs 37/40, nominally ahead) for ~1/35th the
  cost, keeping 75% of work local and paying for a frontier call only on the hard minority it
  can't verify itself.
- **A pure cloud fusion** (OpenRouter Fusion) remains a strong, simple option if cost and locality
  don't matter — but on this benchmark it was neither more accurate nor remotely cheaper.

The honest one-liner: **matches the cloud market leader's accuracy on a measured 40-problem
head-to-head, at ~35× lower cost, with three-quarters of the work never leaving your machine.**
Both halves are now measured; the accuracy half is a statistical tie, stated as exactly that.

## Reproduce it

- Council / single-model numbers: `litmus reproduce lcb` and `litmus reproduce humaneval`.
- Hybrid + Fusion comparison harnesses live in the research repo (`crucible-ablation/`):
  `hybrid_bench.py` (council + frontier escalation, same oracle) and `fusion_bench.py` (Fusion via
  `openrouter/fusion`). Both share byte-identical `oracle()` / `_norm()` and the same public-test
  cases, so the cost column is apples-to-apples and the accuracy column is same-oracle.
