# ADR 0016: Stage 4, adversarial regime certification: evaluated, rejected as a kill stage

Status: **rejected as a kill stage (2026-10-29, user)**. This ADR is a findings record. No
validation study was run. The pre-registration drafted for it is kept below, marked **not run**.
Stage 4 survives only as a reported-only stress diagnostic attached to Stage 3 (see
"Consequences").

## Conclusion
Stage 4 was meant to kill a strategy that is profitable on average but fragile to a market
regime, by finding the worst plausible regime and testing whether the strategy loses money
there. On the planted-signal benchmark (about 24 years of daily data, T = 6000), the design
cannot do that, for reasons that are properties of the data, not of the implementation.

1. **Latent regimes are not identifiable** from the equal-weight return at this sample size and
   regime strength. A Markov-switching GARCH-t labeller chose one regime on all 20 regime seeds:
   two regimes gained a median 6.1 log-likelihood units against BIC's 21.75 penalty, with
   overlapping BIC gaps against no-regime seeds.
2. **An observable volatility state** (vol21, q = .70) is only a noisy proxy. Just 39% of its
   high-volatility days fall in the true stress regime, and the regime-blind oracle's s_regime
   Sharpe in that state is −0.01, against −1.35 in true stress.
3. **Path generators that preserve the observed stylized facts could not be built.** Three
   attempts failed G0, the last against the observed series themselves (McNemar p = .002).
4. **Without paths (option 4), the plausible set is a sampling-uncertainty region** for the
   state's share and spell. At T ≈ 4739 it reaches a horizon-average high-volatility share of only
   .41–.46. A strategy whose edge reverses in high volatility (s_volfrag, oracle Sharpe −0.87
   there) breaks even only at a share of about **.70**. So even with perfectly known per-state
   moments its worst-case SR_H* is **+0.90 [+0.82, +0.98]**, and the kill rule could essentially
   never fire.

**The general statement.** At about 24 years of daily data, the plausible variation in a
strategy's exposure to high-volatility periods is too small to overturn a strategy that is
profitable on average. A rule that killed for "loses money in stress" without such an
adversary, for example a negative Sharpe in the high-volatility state alone, would penalise
ordinary market exposure (most long strategies earn less, or lose, when volatility spikes)
rather than fragility in any decision-relevant sense. Stage 4 is therefore not adopted as a kill
stage. Its per-state measurement is kept as a diagnostic.

## Labeler validation targets (fixed 2026-10-29, before the MS-GARCH labeler ran)

**Background.** A Gaussian 2-state Markov-switching labeler (Hamilton 1989, EM) was tried first,
on 20 dev seeds from root **20261101**, equal-weight daily returns. It fit cleanly (20 of 20
converged) and agreed with statsmodels, but:
- median daily accuracy against the true regime was .83 (.73–.91);
- its fitted stress spells were too short: median 36 days, against an in-sample mean true spell
  of 92;
- on `multi_asset`, which has no regimes, it "found" stress regimes (median 19-day spells, .22
  share) by mistaking GARCH volatility clusters for regime changes.

Labelling 5–10-day sums fixed the spell length (80–99 days) but not accuracy (.80–.82), and it
invented long regimes on `multi_asset`. The user chose a Markov-switching GARCH labeler instead
(below).

**Labeler.** A Markov-switching GARCH(1,1) with Student-t errors (Haas, Mittnik & Paolella 2004):
- each regime has its own GARCH recursion, run in parallel on the same observed shocks;
- the number of regimes, 1 or 2, is chosen by BIC;
- with one regime, every day is labelled calm, the adversary has nothing to move, and Stage 4
  passes. That is how scenario S5 (`multi_asset`) is meant to pass.

**Targets.** These are evaluated once on the same 20 dev seeds per preset (root 20261101,
`RngFactory(20261101).child(f"dev/{i}")`, i = 0..19), equal-weight daily return, hard labels by
smoothed stress probability > .5:

| # | preset | target |
|---|---|---|
| L1 | `regime` | median over seeds of daily label accuracy against the true regime **≥ .90** |
| L2 | `regime` | median over seeds of (fitted expected stress spell, 1/P(stress→calm)) / (in-sample mean true stress spell) **within [0.7, 1.3]** |
| L3 | `multi_asset` | BIC selects **one regime on all 20 seeds** (no invented regimes) |

Reported, not a target: the number of `regime` seeds on which BIC selects two regimes. A seed on
which it selects one is scored as all calm, so it counts against L1.

**Rule.** If any target is missed, I stop and show the user. There is no iteration on these dev
seeds, and no change to the targets. The work is time-boxed to about a day.

## Result: the latent regime is not identifiable from the equal-weight return (2026-10-29)
One run, labeler code `d36c6b9`, targets `41b71c0`. Record:
[`docs/results/0016-labeler-dev/labeler_dev.json`](../results/0016-labeler-dev/labeler_dev.json).

| target | result | met |
|---|---|---|
| L1 median accuracy ≥ .90 | .851 (every day labelled calm) | no |
| L2 spell ratio in [0.7, 1.3] | no two-regime model chosen | no |
| L3 one regime on all 20 `multi_asset` seeds | 20 of 20 | yes |
| reported: two regimes chosen on `regime` | **0 of 20** | – |

**This is a finding about identification, not a fitting failure.**
- **The fits are fine.** The two-regime MS-GARCH-t fits converged on all 40 seeds, and on
  `regime` seeds their stress share and spell were roughly right: median .20 and 83 days,
  against a true in-sample .15 and 92.
- **Two regimes barely improve the fit.**
  - On `regime` seeds, two regimes improve the log-likelihood over one GARCH-t by a median of
    **6.1** units (range 1.1–14.9).
  - On `multi_asset` seeds, which have no regimes, by **3.3** (0.0–6.6).
  - BIC's penalty for the five extra parameters at T = 6000 is **21.75** units.
- **No threshold separates them.** The BIC gaps overlap: 13.8–41.2 on `regime` against 30.3–43.4
  on `multi_asset`. Only 8 of 20 `regime` seeds have a gain above the largest no-regime gain, so a
  likelihood threshold chosen after the fact would detect the planted regimes about 40% of the time.
- **Why.** A persistent GARCH-t (β = .92) absorbs a 2–2.5× switch in the variance intercept
  almost entirely. The same happened on independent synthetic data in the unit tests
  (`test_optimum_is_at_least_as_likely_as_the_truth`).
- **Implication.** A Stage 4 built on a latent-regime labeler would label every day calm on this
  benchmark. The adversary would have nothing to move, and Stage 4 would pass every strategy, so
  its power gate would fail by construction.
- **Likely also on real data.** About 20 years of daily data probably cannot separate regimes of
  this kind from GARCH clustering. Stage 4 must not claim to detect latent regimes.

The MS-GARCH labeler (`stages/adversarial/regimes.py`) stays in the codebase, tested, as an
evaluated alternative. Stage 4 does not use it.

## Decision: an observable volatility state (user, 2026-10-29)
Fixed before any run that uses it.

**Definition.**
- `v_t` = the sample sd of the equal-weight daily return over days t−20..t (trailing, 21 days).
- Day t is in the **high-volatility state** iff `v_t` exceeds the **q = .70** quantile of `v` over
  the Stage 3 out-of-sample days.
- The P&L earned on day t+1 (weights dated t, return dated t+1) is labelled with the state of day
  **t**. The label is therefore fixed before the return it labels, so no day is selected on its own
  outcome. Using `v_{t+1}` would include r_{t+1}, select on |r_{t+1}|, and distort the
  per-state P&L moments.

**Why q = .70** (argued without reference to the benchmark's planted stress share).
- Stage 4's kill rests on the strategy's Sharpe in the high-volatility state. That estimate is only
  decision-relevant if its standard error is no larger than the Sharpe hurdle Stage 3 already
  uses: c = 0.5 annualised (ADR 0013).
- At Sharpe ≈ 0, the annualised SE is about √(252/n_s), so n_s ≥ 1008 days are needed in the state.
- The smallest panel Stage 3 admits (`min_obs` = 5000 rows; window 1260) has 3740 out-of-sample
  days. That needs a state share of at least 1008/3740 = .27, which on a grid of .05 gives a top
  share of **.30**.
- On the benchmark (4739 out-of-sample days), n_s ≈ 1422 and the SE is ≈ .42. On the real ETF
  panel (about 4020 out-of-sample days), n_s ≈ 1206 and the SE is ≈ .46.
- **Stated explicitly:** a .30 share is above the benchmark's planted stress share (.20; the
  in-sample median on the dev seeds is .15). So the high-volatility state necessarily includes
  calm-regime days, and the labels are noisy against the truth by design. This dilutes s_regime's
  fragility in the state, and the pilot and the study measure the resulting power rather than
  assuming it.

**Consequences.**
- The generator, the realism gate and the adversary are unchanged, with "regime" replaced by this
  state.
  - θ̂ is the empirical transition frequency of the binary state sequence.
  - The adversary moves the state's share and its expected spell length.
- **S5 changes.** No "one regime, so pass" path exists. `multi_asset` with {s_real} is graded like
  any robust strategy (pass rate, threshold from the pilot rule).
- **Stage 4's claim** becomes: "survives the worst plausible mix of high- and low-volatility
  periods". It does not claim to recover a latent regime.

## Result: the observable state overlaps the true regime only partly (2026-10-29)
Reported, not gated. The state definition was fixed in `73367a7` before this ran.
Record: [`docs/results/0016-vol-state-dev/overlap.json`](../results/0016-vol-state-dev/overlap.json).

On the 20 `regime` dev seeds (root 20261101), with decision dates from 1260 and P&L day t + 1
compared with the true regime of t + 1:

| quantity | median | range |
|---|---|---|
| accuracy | .786 | .714–.832 |
| true stress days caught (TPR) | .763 | .576–.909 |
| calm days left calm (TNR) | .784 | .729–.855 |
| **precision** (share of high-vol days that are true stress) | **.392** | .177–.671 |
| mean high spell, state against truth | 23.5 against 91.7 days | |

**Consequence, on population quantities** (regime-blind oracle, 10⁶-day path). s_regime's
oracle Sharpe is −0.01 in the vol21, q = .70 high-volatility state, against −1.35 in the true
stress regime (+2.56 and +2.69 in the calm counterparts). No variant tried did materially
better:

| state | precision | s_regime SR in the high state |
|---|---|---|
| vol21 q .80 | .62 | −0.23 |
| vol21 q .90 | .68 | −0.30 |
| vol63 q .70 | .51 | +0.06 |
| vol63 q .80 | .58 | −0.10 |
| 63-day correlation q .70 | .28 | +0.83 |

The latent-regime signal's fragility is invisible in observable volatility states at this
benchmark's strength. That led to s_volfrag (ADR 0015 amendment), whose fragility is defined on
the market's conditional volatility.

## Result: G0 failed with within-state blocks (2026-10-29)
**The first generator:**
- a Markov chain over the day-level state;
- each spell filled by stationary-bootstrap blocks (mean ⌈n^{1/3}⌉ = 17) restricted to observed
  same-state segments.

**What G0 found** (`tests/benchmark/test_adversarial_generator.py`: `volfrag` seeds 0–4, every
asset, one path at θ̂ of the observed length). The generated paths keep heavy tails and the
absence of linear autocorrelation, but **not volatility clustering**. Mean |r| ACF over lags 1–20
(the checker's threshold is .05):

| path | mean ACF |
|---|---|
| observed | .070 |
| generator as designed (within-state blocks, mean 17) | .023 |
| same, block mean 63 | .028 |
| plain stationary bootstrap, ignoring states, block 17 | .042 |
| plain stationary bootstrap, ignoring states, block 63 | .063 |

**Cause.** The q = .70 state has short spells (about 16 days high and 38 low). Blocks that may not
cross a state boundary were cut to an effective **8.7 days**, so longer nominal blocks barely
helped. The clustering the checker measures spans up to 20 days; the binary state chain carries
little of it, and the block joins destroyed the rest. Even a plain bootstrap needs blocks of
about 63 days to keep it.

**The smoke numbers from that generator are void:** seed 1, s_volfrag killed (UB −0.05), s_real
passed (+1.46). No study or pilot seed ever ran on it.

## Decision: a block-level chain, with a data-driven block length (user, 2026-10-29)
Fixed before G0 is rerun. G0 itself is unchanged.

**Block length L.** The Politis & White (2004) automatic selector, with the correction of Patton,
Politis & White (2009):
- **Input:** |r| of the equal-weight return on the observed out-of-sample P&L days.
- **Why |r|:**
  - the dependence the blocks must preserve is volatility clustering (the stylized fact G0 checks
    and the quantity the realism features and the state are built on);
  - raw returns are nearly serially uncorrelated, so the selector on r would pick blocks of about
    1 day, which is the failure above.
- **Rule:** L = max(2, min(round(b̂_CB), b_max)).
  - b̂_CB is the circular- and moving-block estimate, because the blocks are fixed-length moving
    blocks.
  - b_max = ⌈min(3√n, n/3)⌉ is the selector's own cap.
- **Settings:** the selector's published choices (K_N = max(5, ⌈√log₁₀ n⌉), c = 2,
  m_max = ⌈√n⌉ + K_N, flat-top kernel), with no tuning.
- **Computed per run, and reported.**

**Generator.**
- **Pool.** The pool is every moving block of L consecutive observed P&L days (Künsch 1989). A
  block's **type** is high if more than half of its days are in the high-volatility state, and
  low otherwise (a tie counts as low).
- **Paths.** A generated path is a chain over block types, filled slot by slot with a uniformly
  drawn pool block of that type and truncated to length. Each day keeps its observed row and its
  own day-level state label.
- **Fitted chain θ̂.** θ̂ comes from the non-overlapping tiling of the observed path into blocks of
  L: the share of high tiles, and the mean run of high tiles in blocks.
- **Adversary.** It moves the block-type share, the block-type mean run (in blocks), and the
  starting type. The grid's spell axis is in blocks, from 1 to 1000/L, log-spaced as before.
- **Objective.** Per block type k, the P&L moments are pooled over the days of all type-k pool
  blocks. The horizon weights are the expected type shares over the ⌈H/L⌉ slots of the horizon,
  with the last slot weighted by its length. SR_H is the mixture Sharpe of those.
- **Interval.** Each bootstrap replicate redraws, for each type, as many type-k pool blocks as
  the observed tiling has type-k tiles, recomputes the per-type moments, and re-minimises over
  the plausible set.
- **Unchanged:** the realism gate (features, MMD², null at θ̂, α = .05, 199 replicates, 4× length)
  and kill rule A.

**Coarseness of spell control (stated in advance).**
- At L of about 60, a 252-day horizon is about 4 slots. Spells are controlled only in whole
  blocks, and a "high" block contains low-state days, and the reverse.
- So the adversary's lever is coarser and more diluted than with the day-level chain. The worst
  plausible horizon is "about 4 blocks, mostly of the high type", not "252 consecutive
  high-volatility days".
- This is the price of keeping volatility clustering, and it is expected to lower Stage 4's power
  against s_volfrag. The pilot measures how much.

## Result: G0 failed again with the block-level generator (2026-10-29)
One rerun, with G0 unchanged.

**Block length L (reported).** The Politis–White rule picked L = 154, 127, 128, 153 and 89 days
on `volfrag` seeds 0–4, about **2×** the 63 days the diagnostic suggested. `multi_asset` gives
similar values (92–156), so this reflects the market's volatility persistence, not s_volfrag.

**Result.**
- Clustering is mostly preserved: the mean clustering statistic is .060 generated, against .070
  observed (and .023 for the first generator).
- But 13 of 25 asset series failed:
  - `volatility_clustering`, 10 of 25 series: generated .032–.049 against a threshold of .05;
  - `no_linear_autocorrelation`, 3 of 25 series: robust t 3.88, 4.24 and 4.24 against a bound
    of 3.48, where the observed values are 1.54, 3.62 and 2.92.

**Likely cause of the new linear-autocorrelation failures** (not verified separately). The
moving-block pool was sampled with replacement. A path's neighbouring slots can draw overlapping
observed stretches, so the same observed day reappears a few positions later. A duplicated
heavy-tailed day creates spurious positive autocorrelation at that lag.

## Decision: local no-overlap within a path (user, 2026-10-29)
**Strict no-overlap is infeasible.** A path would need more mutually disjoint observed blocks
than the observed path contains (greedy maximum of disjoint pool blocks, per type):

| seed | L | slots needed (path of n / of 4n) | most disjoint blocks (low / high) | high slots needed at θ̂ (n / 4n) |
|---|---|---|---|---|
| 0 | 154 | 31 / 124 | 26 / 10 | ~7 / ~28 |
| 1 | 127 | 38 / 150 | 30 / 15 | ~9 / ~36 |
| 2 | 128 | 38 / 149 | 31 / 12 | ~8 / ~32 |
| 3 | 153 | 31 / 124 | 28 / 10 | ~7 / ~28 |
| 4 | 89 | 54 / 213 | 44 / 18 | ~13 / ~52 |

The realism gate's 4×-length paths could not be generated at all. Even 1×-length paths would
need low-type blocks close to a perfect packing, and random sequential packing falls well short
of the maximum.

**The rule.**
- A block may not overlap the blocks of the previous m − 1 slots, with m = ⌈63/L⌉ + 1
  (rejection sampling; `generator.no_overlap_slots`).
- Blocks d slots apart can share a day only at positions more than (d − 1)L apart, so **no
  observed day appears twice within 63 positions** of a generated path.
- 63 is the longest lag any check uses: the realism features' three 21-day windows, which also
  covers the stylized-facts checker's 20 lags.
- At the selected L, m = 2: only neighbouring slots are restricted.
- Repetition more than 63 positions apart remains (ordinary bootstrap reuse).
- The interval's resampling is unaffected: it computes only per-type moments, for which block
  adjacency is irrelevant.

## Deviation: G0 re-specified as a paired comparison with the observed series (user, 2026-10-29)
**Justification.** G0 as committed holds a generated path to fixed per-asset thresholds. Applied
exactly as committed to the **observed** series (`volfrag` seeds 0–4, every asset, the
out-of-sample P&L days), it fails **6 of 25 series (6 of 75 checks)**:

| seed | asset | check | statistic | threshold |
|---|---|---|---|---|
| 1 | asset_01 | volatility_clustering | .045 | .05 |
| 2 | asset_04 | volatility_clustering | .049 | .05 |
| 4 | asset_00 | volatility_clustering | .047 | .05 |
| 4 | asset_02 | volatility_clustering | .047 | .05 |
| 4 | asset_03 | no_linear_autocorrelation | 3.616 | 3.481 |
| 4 | asset_04 | volatility_clustering | .042 | .05 |

The data the generator resamples do not pass G0 themselves, so as committed G0 cannot separate a
faithful generator from a broken one.

**G0 now.**
- **Same paths:** `volfrag` seeds 0–4, every asset, one path at θ̂ of the observed length, same
  seeds.
- **The comparison:** each (seed, asset, check) outcome of the generated path is paired with the
  same check on the observed series.
- **The test:** "generated is not significantly worse than observed", by the exact one-sided
  McNemar test (`tests/fixtures/paired_facts.mcnemar_worse`).
- **Pass:** iff p ≥ .05.

Fixed before G0's third run. If it fails, the work stops for review.

## Result: G0 failed a third time (2026-10-29)
One run, local no-overlap rule and McNemar G0 (`e3edf33`).
- Generated paths failed **15 of 75** checks; the observed series failed **6 of 75**.
- McNemar: 9 checks fail only on the generated paths and 0 only on the observed series,
  **p = .002**. The generated paths are significantly worse.

| seed | asset | check | generated | observed | threshold |
|---|---|---|---|---|---|
| 1 | 0, 2, 4 | volatility_clustering | .039–.040 | .053–.057 | .05 |
| 3 | 1 | volatility_clustering | .049 | .067 | .05 |
| 2 | 0, 2 | no_linear_autocorrelation | 3.78, 3.78 | 2.93, 3.28 | 3.48 |
| 3 | 1 | no_linear_autocorrelation | 3.55 | 2.98 | 3.48 |
| 4 | 0, 1 | no_linear_autocorrelation | 3.50, 3.48 | 2.35, 2.16 | 3.48 |

The duplicate-day explanation of the earlier autocorrelation failures is **refuted**: with no
observed day able to repeat within 63 positions, five such failures remain. Their mechanism was
not identified. About 15% of the volatility clustering is still lost (.059 against .070).

## Findings: three generator attempts
| attempt | generator | G0 | what it showed |
|---|---|---|---|
| 1 | day-level chain, stationary-bootstrap blocks inside same-state segments | failed (clustering .023 against .070) | short state spells cut blocks to 8.7 days |
| 2 | block-type chain, Politis–White blocks of 89–154 days, moving-block pool | failed (13 of 25 series) | clustering mostly kept; spurious short-lag autocorrelation |
| 3 | as 2, plus local no-overlap (no day repeats within 63 positions) | failed (McNemar p = .002 against observed) | the autocorrelation failures are not duplicate days |

**Deviation (recorded above):** G0 became a paired McNemar test against the observed series,
because the observed series fail the fixed thresholds on 6 of 75 checks.

**Why paths were dropped (user, 2026-10-29; option 4).** Stage 4's decision never needed
generated paths:
- SR_H is analytic, from the per-state P&L moments and the chain's horizon weights;
- the bootstrap interval resamples the observed data;
- paths fed only the MMD realism gate.

A generator that cannot reproduce the observed stylized facts would make that gate's "plausible
set" a statement about the generator, not the data. The gate is therefore replaced by a
confidence region for the chain itself, estimated from the observed state sequence.

**Code.** The three generators, the MMD gate, their tests and the failing G0 test are archived
on the branch **`archive/stage4-block-generator`** (commit `273b517`), outside `src/` on `main`,
for reproducibility of these findings. Kept on `main`:
- the Politis–White selector (`blocklength.py`, cross-checked against `arch` and the AR(1)
  closed form);
- the chain (`chain.py`);
- the objective (`objective.py`, now day level only);
- the state (`state.py`).

## Decision: a confidence region for the state chain, in place of path generation (option 4)
**Region** (`stages/adversarial/region.py`).
- From the observed vol21, q = .70 state sequence x_t:
  - π̂ = mean(x_t);
  - d̂ = 1/p̂, with p̂ = Σ x_t(1 − x_{t+1}) / Σ x_t (high-to-low transitions).
- Both are smooth functions of means of the pairs (x_t, x_{t+1}). A moving-block bootstrap of
  the pairs (Künsch 1989) gives valid replicates, and because pairs are resampled whole, block
  joins add no spurious transitions.
- The 95% region is the replicates' own Mahalanobis contour in (π, log d): q is the
  ⌈(B + 1)·.95⌉-th smallest replicate distance under the replicates' covariance, not a χ²
  quantile.
- A replicate with no high-to-low transition is dropped and counted.
- Unit test: on Markov chains with known (π, d) = (.3, 20) at n = 4739, the region missed the
  truth in fewer than 12% of 120 replications (Wilson upper bound). This is a check of the
  implementation; the study's S1 is the real test.

**Block length.** L = max(2, round(b̂_CB)), taking b̂_CB as the larger of the Politis & White
(2004) / Patton, Politis & White (2009) circular estimates on the state indicator and on
|strategy P&L|. The bootstrap must keep both series' dependence. This replaces the |EW r| rule of
the abandoned generator, whose resampled series was different.

**Adversary.** An exhaustive grid:
- the chains are a 25 × 25 grid over the in-region replicates' range of (π, log d), keeping valid
  chains that lie in the region, plus the point estimate;
- each chain is crossed with the starting state {low, high, stationary};
- SR_H* is the minimum day-level horizon Sharpe ratio (H = 252).

**Markov approximation.** The observed state is a thresholded 21-day trailing volatility, so it
is not a first-order Markov chain and its spells are not geometric. The horizon weights use the
first-order chain with the region's (π, d). The approximation affects only how fast a start in
one state mixes toward π over the horizon. It does not affect the per-state P&L moments, which
are empirical.

**Interval and kill rule A** (unchanged):
- the interval comes from moving-block replicates of the joint daily series (P&L, state); each
  re-estimates the per-state moments and re-minimises over the region's chains, with the region
  held fixed;
- the strategy is killed iff the 95th percentile of the replicates' SR_H* is below 0.

**Reported:**
- SR_H* − SR_H*(1/N);
- the bootstrap quantiles of SR_H* (the worst-case Sharpe distribution);
- GPD tail fits of observed daily losses, overall and in the high-volatility state, with
  moving-block intervals.

**S1 changes.** The region must contain the state's **population** (π_s, d_s), computed once
from a 10⁶-day `volfrag` path (seed 20261103, the same estimator): **π = .300, d = 16.40 days**.
S1's miss rate must have a Wilson upper bound ≤ .075.

## Result: population feasibility check, which the design fails (2026-10-29)
Run before the pre-registration was approved, on population quantities only, with no pilot or
study seed. Script: `scripts/stage4_feasibility.py`. Record:
[`docs/results/0016-feasibility/feasibility.json`](../results/0016-feasibility/feasibility.json).

**Method.**
- **Path:** one 60k-day `volfrag` path, seed 20261104, used by nothing else.
- **Population moments:** Stage 3's walk-forward (Ridge-CV and 1/N, `base.yaml`, lag 0) gives
  per-state daily P&L moments of {s_volfrag} and {s_real}.
- **Typical regions:** the path is cut into twelve non-overlapping 4739-day windows. Each gets
  its own vol21, q = .70 state and its own 95% region, exactly as `certify` builds them.
- **Adversary:** run with the population moments, so these are SR_H* values with no estimation
  error.

| | low-vol SR | high-vol SR | worst SR_H* (median [range], 12 windows) | worst horizon high-vol share |
|---|---|---|---|---|
| {s_volfrag} | +2.90 | **−0.87** | **+0.90 [+0.82, +0.98]** | .43 (.41–.46) |
| {s_real} | +1.43 | +1.12 | +1.24 [+1.24, +1.25] | .43 |

**Typical region:** π between about .19 and .41, and a mean high spell d between about 10 and
30 days.

**The worst point, in every window:** the region's top share (π ≈ .40), its longest spell
(d ≈ 20–30 days) and a start in the high state. Spells of 20–30 days barely move a 252-day
horizon away from π.

**Break-even:** s_volfrag's mixture Sharpe crosses 0 only at a horizon high-volatility share of
**.698**. Its high-volatility days have 1.44× the P&L sd of its low-volatility days. The region
never gets close, so S2 (the power gate) would fail by construction, and the study was not run.

## Consequences
- **Stage 4 is not a kill stage.** The pipeline's kill stages are Stages 1–3; Stage 5 (the
  memo) is next. `StageName.ADVERSARIAL` remains in the enum and is unused by the ledger.
- **A reported-only stress diagnostic** is attached to Stage 3. Each portfolio's Ridge-CV and
  1/N Sharpe in the vol21, q = .70 high-volatility state goes in the ledger metrics, with a
  moving-block bootstrap interval (`stages/adversarial/diagnostic.py`). It never decides.
- **Benchmark.** s_volfrag and the `volfrag` / `volfrag_null` presets stay (ADR 0015) as the
  evidence behind this conclusion. Like s_regime, s_volfrag is marked `identifiable = false`, so
  `score_run` reports it and excludes it from the kill counts. Its expected verdict within
  Stages 1–3 is PASS: a true, average-positive edge.
- **Code.** The certification (`chain`, `region`, `objective`, `blocklength`, `certify`) and the
  study harness stay in `stages/adversarial/`, tested, as an evaluated design, like Stage 3's
  DRO arms. The study's task smokes move to the slow marker, since no study will run. The path
  generators are on `archive/stage4-block-generator`.
- **The memo must say** that Stage 4 was evaluated and not adopted, and why (this
  "Conclusion"), and must report the stress diagnostic as a description, never as a
  certification.

## Pre-registration (DRAFT; NOT RUN). It failed the feasibility check above
It is kept as written for the record. It was never committed as a pre-registration, no pilot
ran, and none of its thresholds or gates were applied.

Everything below was to be fixed before the pilot. The pilot sets only the S2, S4 and S5 thresholds, by
the rule below, and is recorded in a separate "Pilot" section pushed before the full run. Results
go in a separate section afterwards and are never edited into this one.

### Method under test (`stages/adversarial/`)
**Inputs.** Stage 3's walk-forward daily P&L for Ridge-CV and 1/N:
- `configs/base.yaml`'s portfolio settings at execution lag 0 (window 1260, refit 21, CV every
  252, c = 0.5/√252, gross ≤ 1);
- over decision dates 1260 … T − 2, so 4739 days at T = 6000.

**State.** The vol21, q = .70 volatility state of each P&L day (`state.py`, fixed in `73367a7`).
It is labelled with day t's trailing volatility for the P&L earned on t + 1.

**Certification** (`certify.py`, all of it above):
1. the Politis–White block length;
2. the 95% bootstrap region for (π_s, d_s), with 999 replicates;
3. the grid adversary over the region × {low, high, stationary}, with H = 252;
4. SR_H* with 999 joint moving-block replicates, re-minimised over the region;
5. **kill rule A:** kill iff the 95th percentile of the replicates' SR_H* is < 0.

`configs/adversarial_study.yaml` pins every value (`tests/unit/test_adversarial_study.py`).

### Tasks
Replication i uses `RngFactory(root).child(f"seed/{i}")`, with dataset and pipeline seeds drawn
from it. The harness is `stages/adversarial/study.py` and `scripts/adversarial_study.py`, using
joblib/loky with chunked JSONL checkpoints that resume.

| task | market (preset) | members | role |
|---|---|---|---|
| `volfrag` | `volfrag` | {s_volfrag} | **S2**: Stage 4 kill rate |
| | same market | {s_real} | **S4**: Stage 4 pass rate; **S1**: does the {s_real} certification's region contain the population (π, d) = (.300, 16.40)? |
| `null` | `volfrag_null` | {s_real, s_volfrag}, both with zero edge and μ = 0 | **S3**: Stage 4 kill rate (false kills) |
| `noregime` | `multi_asset` (unchanged) | {s_real} | **S5**: Stage 4 pass rate |
| `regime` | `regime` | {s_regime} | reported: the unidentifiable latent-regime signal |
| `pipeline` | `volfrag` | Stages 1–3 on the production path; Stage 4 on the members that passed Stage 3 | reported |

The forced-member tasks also record Stage 3's kill-rule-A p-value (one-sided Ledoit–Wolf, 999
replicates) on the same paths, so "Stage 4 kills given Stage 3 passed" is reported.

**Why S3's null is H0's boundary.** With μ = 0 and no edge,
E[r_{t+1} | F_t] = 0. Ridge-CV's weights are F_t-measurable, and the state of day t is
F_t-measurable. So the population mean of the P&L is 0 in each state, and SR_H = 0 for every chain
and every start. A kill there is a false kill, and S3 measures the rule's size *after* the
adversary's minimisation, which biases the point estimate downward.

### Gates (Stage 4 is accepted on the benchmark only if all five hold)
| gate | rate | criterion |
|---|---|---|
| S1 | region misses the population (π, d) | Wilson 95% upper bound ≤ **.075** |
| S3 | kill rate on `null` | Wilson 95% upper bound ≤ **.075** |
| S2 | kill rate of {s_volfrag} on `volfrag` | Wilson 95% lower bound ≥ **T2** (pilot rule) |
| S4 | pass rate of {s_real} on `volfrag` | Wilson 95% lower bound ≥ **T4** (pilot rule) |
| S5 | pass rate of {s_real} on `multi_asset` | Wilson 95% lower bound ≥ **T5** (pilot rule) |

**M = `required_seeds(.05, .075, n_gating = 2, all_pass = .90)` = 1273**, for the two size-type
gates (S1, S3), as in ADR 0014. All three gated tasks run on M seeds.

Exact pass probability per size gate at M = 1273:

| true rate | .030 | .040 | **.045** | **.050** | .060 | .075 | .100 |
|---|---|---|---|---|---|---|---|
| P(gate passes) | 1.000 | 1.000 | .996 | **.959** | .559 | .025 | .000 |

Both size gates at a true .05 pass with probability ≥ .918 (union bound).

### The pilot and the threshold rule (user, 2026-10-29)
- **Pilot:** 50 seeds from root **20261030**, every task kind (the reported kinds for timing).
- **Rule, per gate (S2, S4, S5):** T = ⌊(Wilson 95% lower bound of the pilot rate − .05)⌋ to .01.
  - Examples: 50 of 50 gives .929 − .05, so .87; 45 of 50 gives .786 − .05, so .73.
- **Floors:** if T < **.50** for S2, or T < **.80** for S4 or S5, the study is not run and the
  design returns to review.
- **Recorded with each T:** the exact binomial pass probability at M = 1273, at the pilot's
  estimate, at its Wilson lower bound, and at T (`pilot_thresholds`).
- **The pilot's rates set the thresholds only.** They are never reported as results.

Illustration (M = 1273): with T = .73, P(pass) is .024 at a true .73, .357 at .75 and .985 at
.78. With T = .87, it is .021 at .87, .557 at .89 and .906 at .90. A pilot Wilson lower bound on
50 seeds sits about .08–.14 below the pilot estimate, so a gate whose true rate equals the pilot
estimate passes almost surely, while one whose true rate has fallen to T passes about 2% of the
time.

### Reported, never gated
- Stage 4 kill rate of {s_regime} on `regime`, at M_rep = 200 seeds. It is expected low (ADR
  0015: its fragility is invisible in the observable state). This is the documented
  unidentifiable case.
- For the `volfrag` {s_volfrag} task: Stage 3's pass rate, and Stage 4's kill rate given that
  Stage 3 passed.
- **Pipeline (M_rep = 200):**
  - per signal, the rate of reaching Stage 4 and Stage 4's kill rate given reached;
  - the member sets, and the passenger and collateral rates (joint verdict);
  - Stage 2's kills of s_volfrag.
- Per task:
  - SR_H*, its upper bound and bootstrap quantiles;
  - SR_H*(baseline at θ̂);
  - SR_H* − SR_H*(1/N);
  - the worst (π, d, start) and its position in the region;
  - the region's ranges;
  - L;
  - the GPD ξ and ES₉₉ with intervals.
- Runtime.

### Compute budget (mechanical; not revisited after results)
- **Budget:** 24 h wall clock on 8 workers.
- **Projection:** the pilot's mean task seconds per kind, measured under the full 8-worker load,
  replace the idle-timing allowance (ADR 0014's backlog lesson).
- **Core:** the three gated tasks at M = 1273.
- **Reported tasks** run at M_rep = 200 unless that overruns, then at the largest count that fits.
- **If the core overruns, or M_rep would fall below 100,** the study is not run (`budget_plan`).
  The gates and M are never reduced.

### Fast-suite coverage
- `tests/integration/test_adversarial_study_tasks.py` runs every task kind on one tiny smoke seed
  (root 7; n_obs 2500, window 300, small bootstraps; Stage 3's α at .49 so the pipeline's Stage 4
  branch runs), plus the pilot rule, the budget plan and the summary.
- `tests/unit/test_adversarial_study.py` tests the config pins, M, the threshold rule (with its
  feasibility floor), acceptance across all five gates, and the budget plan.
- `tests/unit/test_adversarial_certify.py` tests the chain, the objective, the region (including
  its coverage on known chains) and the certification.
- `tests/leakage/test_adversarial_inputs.py` checks the state's timing.

### Limitations stated in advance
- **The narrowed claim:** "survives the worst mix of high- and low-volatility periods consistent
  with the observed state sequence". Not latent regimes, which proved unidentifiable.
- **The Markov approximation** for the horizon weights.
- **The fixed-policy approximation:** no refitting under a changed mix.
- **The region is a sampling-uncertainty region at T ≈ 4739.** It does not allow occupancies
  far from those observed; a 2008-length crisis in a calm sample is outside it by design.
- **The benchmark's fragile signal is serially independent (ρ = 0),** and its C1 margin is thin
  (ADR 0015).
- **Gross Sharpe only,** as in Stage 3.

### Run discipline
- This pre-registration is committed and pushed before the pilot.
- The pilot runs from a git worktree at that commit. Its thresholds and budget go in a "Pilot"
  section and the config, pushed before the full run, which runs from a worktree at that commit.
- Post-launch fixes are recorded with their cause, the fix commit, and whether the run resumed or
  restarted.
- If any gate fails, Stage 4 is not built and the design returns to review. No threshold changes
  after the pilot.
