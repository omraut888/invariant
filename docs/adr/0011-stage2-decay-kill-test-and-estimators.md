# ADR 0011: Stage 2 decay validation: kill test and half-life estimator (pre-registered study)

Status: **accepted (2026-10-23)**.
- **The kill rule is adopted** (sup-split decline test, p < .05).
- **No half-life estimator is admissible**, so Stage 2 reports no half-life.
- The design, rules and compute-budget rule were pre-registered and pushed on 2026-10-21
  (`8d133ce`), before any smoke or full run, and nothing above "KSG budget decision" changed
  after a run.
- One post-launch bug fix (`fe3bd87`) did not change the design. It is described below.

## Context
Stage 2 takes each signal that passed Stage 1 (ADR 0010) and decides whether its edge is
decaying, killing it if the edge is gone or going. On the benchmark, s_decay (coefficient
half-life H = 250, onset at 0.6 T, so `GroundTruth.te_half_life_approx` = H/2 = 125) should be
killed here, and s_real (constant coefficient) should pass.

The spec (docs/invariant-spec.html, section 04) proposes this method: transfer entropy over
rolling 60/120/252-day windows, an exponential fitted to the resulting curve, and the
half-life with a CI as the evidence. Before writing the stage, this ADR fixes two things:
1. a **kill rule**;
2. a comparison of three half-life estimators: (a) KSG TE with surrogate significance, (b)
   Gaussian TE, and (c) the rolling hybrid partial correlation on Stage 1's conditioning set.

The comparison is pre-registered the way ADR 0009 chose the CI test.

## Findings before the study (analytic; `scripts/decay_prestudy_analysis.py`)
These are why the design differs from the spec.

1. **At the default SNR, no estimator can identify the half-life.** Take the Cramér–Rao bound
   for the decay rate λ in `β(t) = β0 · 2^{-(t-τ)/H}`, at r0 = .08, T = 6000, onset .6 T,
   oracle conditioning:
   - relative SE(λ) is **1.33** when τ is known and **1.86** when it is not;
   - for the TE half-life (truth 125 days), that is an SE of about **167** and **232 days**.

   No estimator does better than this bound. At the default preset, a bias, variance or
   coverage comparison could only show that all three estimators are uninformative. The
   comparison is therefore decided on **identifiable** scenarios: decay ic .25, where the
   relative SE bound is .43 at H = 500 and .59 at H = 250.
2. **The spec's short windows carry no information at this SNR.**

   | window | SE(r) | Gaussian TE floor `1/(2w)` (nats) |
   |---|---|---|
   | 60 | .129 | .0083 |
   | 120 | .091 | .0042 |
   | 252 | .063 | .0020 |

   The planted r0 is .08 and the planted TE0 is .0032. Overlapping windows fitted as if they
   were independent would also give CIs that are too narrow. This study uses
   **non-overlapping 252-row windows**, and the model is fitted to the window *averages* of
   the decay curve.
3. **Two natural kill rules fail.**
   - *"The recent window is not significant while the early window was."* This treats absence
     of evidence as decay. For s_real (ic .06), P(recent window not significant) is .84 at
     w = 250, .73 at 500 and .52 at 1000. That is its false-decay rate on a signal that does
     not decay.
   - *"The half-life's upper CI bound is below H_min."* Given finding 1, its power on s_decay
     is about .12, even with a generous H_min of 1260 days.
4. **A one-sided test of decline has false-decay rate α by construction and good power.**
   Gaussian simulation, 1500 replications, r0 = .08, power at α = .05:

   | onset / H | .3/250 | .6/125 | .6/250 | .6/500 | .8/125 | .8/250 | .8/500 |
   |---|---|---|---|---|---|---|---|
   | sup-split (chosen) | .78 | .81 | .75 | .65 | .54 | .44 | .26 |
   | linear trend | .79 | .81 | .76 | .68 | .43 | .37 | .25 |

   **A decline that started recently is the hardest to detect, and it is also the case that
   matters most economically.** This is a power limit, not a defect, and the memo must say so
   whenever Stage 2 passes a signal.
5. **Selection by Stage 1 should not distort the decline test.** Under a constant coefficient,
   the full-sample estimate and the early-minus-late contrast are uncorrelated (precision
   weights). So whether a signal passed Stage 1 is close to independent of its decline
   statistic. The study measures this rather than assuming it: it reports rates conditional
   on passing Stage 1.

## Decision: kill rule (decides)
**Kill with `decay.edge_declining` iff the decline test's p < α_decay = 0.05.**

The test is `invariant.stages.decay.decline.decline_test`. It runs on Stage 1's best link (the
asset, lag and conditioning set, held fixed) and has these parts:
- **Residualisation.** x and y are residualised on `[1, Z]` with full-sample coefficients.
  This is the partial structural-change model (Bai & Perron 2003): only the signal's
  coefficient may break.
- **Statistic.** For every break row k on a grid over [.15 n, .85 n] in steps of 25, `t(k) =
  (b_pre − b_post) / sqrt(V_pre + V_post)` with White HC0 segment variances. The statistic is
  `S = max_k t(k)`: one-sided, with an unknown break (Andrews 1993).
  - HC0 and not HC3: the cumulative-sum algorithm needs HC0. A segment has at least 900 rows,
    so the leverage correction would change V by well under 1%.
- **p-value.** Hansen's (2000) fixed-regressor wild bootstrap: Rademacher weights on the
  no-break residuals, B = 999, and `p = (1 + #{S* ≥ S}) / (B + 1)`.
- **Reported, never decide:**
  - the break fraction at the maximum, which is not an onset estimate;
  - the one-sided trend-interaction t.

Justification:
- The false-decay rate on a non-decaying signal is controlled at α. There is no correction
  across signals. A correction would make kills harder as candidates are added, which would
  make the pipeline more permissive. Stage 1 already controls the FDR of what passes.
- The test needs neither a half-life nor an onset, and both are unidentifiable at realistic
  SNR (finding 1).
- The user chose sup-split over the linear trend on 2026-10-21. Both have the same power for
  early onsets, and sup-split is better for recent decay (.54 vs .43 at onset .8, H 125).

**The half-life estimate and its CI are reported and never decide.** Stage 1's bootstrap
interval follows the same precedent. Stage 3 and the memo consume them.

## Study design
Code: `invariant.stages.decay.{windows, te, halflife, decline, study}`; script:
`scripts/decay_study.py`; config: `configs/decay_study.yaml`.

**Replication.** Each replication runs Stage 1's method on the dataset: PCMCI with
`hc3|hybrid`, Bonferroni over lags 1..5, then BH over the four candidates. That records
whether the signal passed Stage 1. For s_real and s_decay the study then takes the best lag
and its conditioning set and builds `lagged_design`, as Stage 1 hands them on. The kill test
and the half-life estimators run on that design **whatever Stage 1's verdict was**
(unconditional rates). The rates conditional on passing Stage 1 are reported as well.

**Half-life estimators.**
- **Windows.** 252 rows, non-overlapping and end-aligned. Each window's statistic uses only
  its own rows (tests/leakage/test_decay_window_inputs.py).
- **Model.** Constant up to an onset τ, then exponential decay with half-life HL ∈ (0, ∞].
- **Fit.** Profile likelihood on a grid: τ every 21 rows, and HL at 121 log-spaced values in
  [5, 50 000] plus ∞.
- **Interval.** The 95% interval comes from inverting the profile likelihood ratio (χ²₁).
  Results are stored as TE-scale decay rates λ = ln 2 / HL, so "no decay" is λ = 0.

The three estimators:
- **(c) `partial_corr`.** The window partial correlation as Fisher z, Gaussian with variance
  1/(w − k − 3) and mean ζ0·s_j. Here s_j is the window average of m at the *coefficient*
  half-life 2·HL, and ζ0 takes either sign. The TE half-life is the low-SNR mapping HL_coef/2.
  At r0 = .25 the exact Gaussian TE half-life is 2.3% shorter than H/2 (244.2 vs 250), which
  is far below the estimators' SE. The approximation is documented, and scoring uses
  `te_half_life_approx` as instructed.
- **(b) `gaussian_te`.** The window TE `−½ log(1 − r²) = log cosh z` is a function of |z|.
  Its likelihood is therefore the folded normal of |z|, with `μ_j = atanh(sqrt(1 −
  exp(−2·TE0·s_j)))`, TE0 ≥ 0, and s_j taken at HL itself. The folded normal absorbs the
  estimator's positive floor near TE = 0.
- **(a) `ksg`.** tigramite `CMIknn` (Frenzel–Pompe/KSG CMI) at tigramite's defaults:
  - knn = 0.2·w, rank transform, 5 shuffle neighbours, 500 local-permutation surrogates
    (Runge 2018), seeded per window;
  - each window's value minus its surrogate mean is modelled as Gaussian, with the surrogate
    SD and mean TE0·s_j, TE0 ≥ 0.

  This required adding `numba` (`uv add numba`, 0.68.0): tigramite's `cmiknn` module imports
  it, and it is not a declared dependency of tigramite. numpy stays at 2.4.6.

**Scenarios.** Root seed **20261021**, which is fresh. Seed i's dataset is shared by every
scenario (common random numbers). T = 6000.

| scenario | DGP | role |
|---|---|---|
| d1_base | default preset | **gate**: kill test on s_real; **gate**: estimator false-decay on s_real; s_decay reported (power, unidentifiable half-life) |
| d2_ident | decay ic .25, H 500 (TE HL 250) | **gate**: estimator false-decay on s_real; **gate**: s_decay coverage; **selection** |
| d3_ident_fast | decay ic .25, H 250 (TE HL 125) | **gate**: s_decay coverage |
| k_garch_hi | GARCH .04/.95 | **gate**: kill test on s_real |
| k_rho99 | s_real ρ .99 | **gate**: kill test on s_real |
| k_realistic | realistic preset (ic .02) | **gate**: kill test on s_real |
| k_hidden | hidden confounder | reported: the decline test when the residuals carry an omitted persistent parent |
| p_onset{.3,.6,.8}_h{125,250,500} | as in finding 4, six cells (.6/250 is d1) | reported: kill-test power on s_decay |

KSG runs on d1–d3. (b) and (c) run on d1–d3 and the k_* scenarios.

**Decision units.**
- Kill test: reject iff p < .05, one Bernoulli per (scenario, signal, seed).
- Estimator "claims decay": the 95% interval excludes no decay (λ_low > 0).
- Estimator "covers": the interval contains `te_half_life_approx`.
- Point accuracy: bias, SD and RMSE of λ̂ against ln 2 / (H/2). These are finite even when an
  estimate is "no decay". Medians are reported on the half-life scale.

## Rules (pre-registered)
- **Kill test admissible** iff the Wilson 95% upper bound of the s_real rejection rate is
  ≤ .075 in every one of d1_base, k_garch_hi, k_rho99 and k_realistic. **If not, Stage 2 is
  not built and the decision returns to review.**
- **Estimator admissible** iff both hold:
  - its false-decay Wilson upper bound on s_real is ≤ .075 in d1_base and d2_ident;
  - its coverage Wilson lower bound on s_decay is ≥ .90 in d2_ident and d3_ident_fast.
- **Selection.** Among admissible estimators, the lowest RMSE(λ̂) on s_decay in d2_ident
  wins. Two estimators tie when the 95% seed-bootstrap CI of their RMSE difference contains
  0 (2000 replicates, seeded from the root seed). Ties go to the simpler estimator, in the
  order **(c) partial_corr < (b) gaussian_te < (a) ksg**:
  - (b) is a transform of (c) that discards the sign;
  - (a) adds a nonparametric estimator, a neighbour count and surrogates.

  If no estimator is admissible, Stage 2 reports no half-life and the question returns to
  review. The kill rule does not depend on this outcome.
- **M** = `required_seeds(0.05, 0.075, 6, 0.90)` = **1610**, counting 4 kill-test gates and 2
  estimator false-decay gates. At M = 1610:
  - a true-.05 gate passes with probability .987, and all 6 with probability ≥ .92 (union
    bound);
  - a true-.06 gate passes with .66, and a true-.075 gate with .025.
  - The coverage gate (lower bound ≥ .90) passes with probability ≈ 1.00 at true coverage
    .95, .99 at .93, .79 at .92 and .02 at .90.

## KSG compute budget (pre-registered before the smoke run)
- **Budget**: 6 hours of wall-clock time for the KSG arm at the required M = 1610, on all
  cores of the run machine. The run machine has 8 logical cores, and the report records the
  worker count.
- **Projection**: the M = 8 smoke run's mean KSG wall time per seed, summed over the KSG tasks
  of d1–d3, times 1610, divided by the number of workers (`study.ksg_budget`).
- **Within budget**: KSG runs on all 1610 seeds and is eligible for selection.
- **Over budget**: KSG is a **reported-only arm**:
  - it runs on the first `M_ksg = floor(6 h × workers / seconds per seed)` seeds, the same
    datasets as (b) and (c);
  - its rates, Wilson CIs, bias, SD and RMSE are reported;
  - it is **not eligible for selection**, so selection is between (c) and (b) with the same
    rule and tie-break;
  - this ADR will then state that KSG was not fully evaluated for compute reasons, and will
    record M_ksg, the projection and the smoke timings.
- The decision is mechanical, made from the smoke timings, and recorded in this ADR and in
  `ksg_n_seeds` in the config. Both are committed and pushed before the full run, and it is
  not revisited after any result.
- The smoke run's rates are not used for anything.

## Assumptions and where they are violated
| Assumption | Violation | Consequence / what is measured |
|---|---|---|
| Residuals are an MDS given Stage 1's conditioning set | An omitted persistent parent (hidden confounder; in markets, likely) | The wild bootstrap ignores serial correlation, so size may inflate. Measured on k_hidden (reported). |
| Only the signal's coefficient changes | Conditioning coefficients drift too | A correlated drift leaks into t(k). Not tested here. |
| Stationary volatility | GARCH persistence near 1 | Gated in k_garch_hi. |
| Persistent regressor | ρ .99 | Gated in k_rho99 (Stambaugh-like persistence, independent innovations). |
| Exponential decay after a single onset | Step changes, partial decay to a floor, recovery | The decline test still detects declines. The half-life model is misspecified and is reported, never decides. |
| Best (asset, lag) | Multi-asset signals; a lag that is not identified (ADR 0009, finding 5) | Stage 2 tests only Stage 1's best link. |
| Profile-LR χ² calibration | τ is unidentified under no decay (Davies 1987) | Measured as estimator false-decay (gated). |
| Window z ~ Gaussian, independent windows | GARCH-t residuals; persistent regressors | Measured by coverage (gated). |

**Look-ahead.** Like Stage 1's verdict, the decline test and the half-life fit are full-sample
statements about the sample, and not values used for a trade at date t. Their inputs are the
PIT panel's point-in-time features, and they never read `fwd_*`. Each window's statistic uses
only rows inside that window.

## Alternatives considered
- **The spec's method as written** (60/120/252-day rolling windows, an exponential fit on
  overlapping estimates, a half-life-based kill): rejected by findings 1–3.
- **A linear-trend decline test**: simpler, with equal power for early onsets but less for
  recent ones. It is kept as a diagnostic.
- **A fixed recent-window contrast** (the last W rows against the rest): its power depends on
  a W that could only be tuned against the benchmark's known onset. Rejected.
- **A "lost at least half" shifted null** (H0: β_late ≥ ½ β_early): it is less powerful and
  needs a tuned fraction. Not adopted. The kill reason states "declining", not "gone".
- **Correcting kills for multiplicity**: makes the pipeline more permissive as candidates are
  added (see the justification above).

## KSG budget decision (smoke run, 2026-10-21; before the full run)
The smoke run was M = 8, at commit `8d133ce` (clean tree), with 8 workers. Its timings are in
`runs/decay_study/3d48cc9bc74b7ef8/ksg_budget.json`.
- KSG wall time is **512 s per seed**, summed over d1–d3 under 8-way parallel load. The median
  KSG task (one scenario) took 169 s, and the median main-arm task 7.2 s.
- The projection is 512 × 1610 / 8 = **28.6 h**, against a budget of 6 h. **Over budget.**
- By the pre-registered rule, **KSG is a reported-only arm** on the first **M_ksg =
  floor(6 × 3600 × 8 / 512.2) = 337** seeds (`ksg_n_seeds: 337` in the config). It is **not
  eligible for selection**. Selection is between (c) partial_corr and (b) gaussian_te, with
  the same rule and tie-break.
- **KSG was not fully evaluated, for compute reasons.** Its rates and intervals are reported
  at M = 337, where a Wilson interval near .05 is about ±.025 wide. Its gates are shown but do
  not decide anything.
- The smoke run's rates were not used for anything.

## Post-launch bug fix (2026-10-22): KSG crash on an empty conditioning set
This fix does **not change the pre-registered design.** The estimators, scenarios, seeds, M,
M_ksg, rules and budget are all as above. No partial result or rate was looked at before or
during the fix.

**What happened.** The full run (commit `f7b4e5c`, clean tree, run directory
`runs/decay_study/01df4c82cfd33b73`) stopped with `TypeError: corrcoef() got an unexpected
keyword argument 'ddof'`, raised in a joblib worker. The crash came in the first chunk of the
KSG arm. Main-arm tasks run first, then KSG tasks, and at that point the checkpoint held
20,928 of the 20,930 main-arm records: everything except `p_onset0.8_h500` seeds 1608 and
1609. It held **no** KSG records.

**Cause.** The bug is in tigramite 5.2.10.1, not in our code.
- Location: `tigramite/independence_tests/independence_tests_base.py`, line 1093, in
  `CondIndTest._get_acf`. Our code never calls `corrcoef`.
- When Z is empty (Stage 1's hybrid conditioning set for the best link can be empty),
  `CMIknn.get_shuffle_significance` falls back to a block shuffle.
- Its block length comes from `CondIndTest._get_acf`, which calls `np.corrcoef(y1, y2,
  ddof=0)`.
- numpy 2.x removed `ddof` from `corrcoef`. It had been a documented no-op since numpy 1.10,
  because the normalisation cancels in a correlation.
- ADR 0009's numpy compatibility check never exercised this path, and neither did our tests:
  every KSG test used a non-empty Z, and the fast suite never ran the KSG arm.

**Fix (smallest correct change; commit `fe3bd87`, pushed before the restart).** `te.CMIknnNumpy2` subclasses `CMIknn` and overrides only
`_get_acf`. The body is the same code without `ddof`. `ksg_windows` uses it. Tests:
- an empty-Z `ksg_windows` regression test. It fails with the original `CMIknn` and the same
  `TypeError`.
- the overridden ACF against an independent reference (Hypothesis);
- `CMIknnNumpy2` against `CMIknn` with a non-empty Z: the value, p and null distribution are
  bit-identical;
- an end-to-end run of **every** study arm, KSG included, on one tiny seed in the fast suite.

**Restart: resumed from the checkpoint, not rerun from scratch.**
- *Static:* the fix changes only `stages/decay/te.py`. `ksg_windows` and `CMIknnNumpy2` are
  reached only from the `arm == "ksg"` branch of `study.run_seed`. The main arm (PCMCI,
  windows, the half-life fits (c) and (b), the decline test) does not call them.
- *Empirical:* 4 checkpointed main-arm records, sampled at random (`random.Random(0)`), were
  recomputed with the fixed code and are bit-identical, wall time excluded. That compared
  equality only, and printed no rates.
- *No KSG records existed*, so nothing produced by the broken path is reused.

The 20,928 main-arm records were copied byte for byte (checked with `cmp`) into the new run
directory, `runs/decay_study/3566ebefd08713f9`. The run key includes the git SHA, so the
directory is new. The restarted run reported `n_reused=20928`. It ran the remaining 2
main-arm tasks and all 1,011 KSG tasks at `fe3bd87` and finished without error. The old
directory (`01df4c82cfd33b73`) and both logs are kept.

## Results (M = 1610, M_ksg = 337, commit `fe3bd87`, clean tree; report `runs/decay_study/3566ebefd08713f9/report.json`; archived at [`docs/results/0011-decay-study/report.json`](../results/0011-decay-study/report.json))
Main-arm records 0–20,927 came from the run at `f7b4e5c` (see the post-launch bug fix above).
Everything else ran at `fe3bd87`. No crash. All rates are decision-unit rates at α = .05 with
95% Wilson CIs.

### Kill test (sup-split, wild bootstrap): **admissible, all 4 gates pass**
False-decay on s_real (`*` = gate). Each cell gives the unconditional rate, then the rate
given a Stage 1 pass.

| scenario | s_real rate | given a Stage 1 pass | trend diagnostic |
|---|---|---|---|
| d1_base * | .043 [.035, .055] | .044 [.034, .056] | .055 |
| k_garch_hi * | .044 [.035, .055] | .045 [.036, .058] | .055 |
| k_rho99 * | .048 [.038, .059] | .048 [.038, .061] | .040 |
| k_realistic * | .051 [.041, .063] | .046 [.025, .082] | .049 |
| k_hidden | .048 [.039, .060] | .050 [.040, .063] | .053 |
| d2_ident | .055 [.045, .068] | .059 [.048, .072] | .057 |
| d3_ident_fast | .058 [.047, .070] | .061 [.050, .075] | .055 |

Power on s_decay:

| scenario (onset / H) | unconditional | given a Stage 1 pass | trend | pre-study prediction |
|---|---|---|---|---|
| p .3 / 250 | .606 | .638 | .610 | .78 |
| p .6 / 125 | .632 | .661 | .639 | .81 |
| **d1_base .6 / 250** | **.593 [.569, .617]** | **.622 [.596, .648]** | .607 | .75 |
| p .6 / 500 | .517 | .544 | .525 | .65 |
| p .8 / 125 | .443 | .464 | .358 | .54 |
| p .8 / 250 | .348 | .366 | .300 | .44 |
| p .8 / 500 | .221 | .229 | .214 | .26 |
| k_realistic (ic .02) | .104 | .207 | .122 | – |
| d2 / d3 (ic .25) | .986 / .991 | .986 / .993 | .982 / .988 | – |

### Half-life estimators: **none admissible**
| scenario / signal | partial_corr (c) | gaussian_te (b) | ksg (a), M = 337, reported only |
|---|---|---|---|
| d1 s_real false-decay * | **.116 [.101, .132]** | **.104 [.090, .120]** | .116 [.086, .154] |
| d2 s_real false-decay * | **.121 [.106, .138]** | **.118 [.103, .135]** | .107 [.078, .144] |
| d2 s_decay coverage * | **.889 [.873, .903]** | **.900 [.884, .914]** | .843 [.800, .878] |
| d3 s_decay coverage * | .915 [.900, .928] | .951 [.939, .960] | .920 [.886, .944] |
| d2 λ bias / SD / RMSE (truth .00277) | +.020 / .043 / .047 | +.038 / .054 / .066 | +.050 / .060 / .078 |
| d2 median TE half-life (truth 250) | 215 | 171 | 79 |
| d1 s_decay claims decay | .726 | .487 | .243 |

Bold marks a failed gate. **Rule outcome: no estimator is admissible, so none is selected.**
As pre-registered, Stage 2 reports no half-life, and the half-life question returns to review.
The kill rule does not depend on this outcome. KSG was not fully evaluated, for compute
reasons. It would not have changed the outcome: it fails the same gates at M = 337.

## Findings
1. **The kill test is calibrated across every null, including the ones outside its
   assumptions.**
   - The four gates sit at .043 to .051.
   - The hidden confounder (an omitted persistent parent, which violates the MDS assumption)
     gives .048.
   - Conditioning on a Stage 1 pass changes nothing measurable (finding 5 of the pre-study).
   - Mild size elevation, not gated: s_real in d2 and d3 rejects at .055 and .058. There
     s_decay is a strong, decaying *co-parent* of R. Its coefficient is held constant in the
     partial-change residualisation, which violates the "only the signal's coefficient
     changes" assumption. Both CIs include .05, so this is consistent with a small leak and
     not established.
2. **Power is about 15 points below the pre-study prediction** (.59 against .75 at the
   default preset). The prediction used an oracle conditioning set, i.i.d. Gaussian x and
   Gaussian noise. The benchmark has GARCH-t(6) residuals, which a wild bootstrap with HC0
   segment variances uses less efficiently than the Gaussian likelihood, plus a persistent
   x and PC1's estimated conditioning set. The ordering across onsets is as predicted: recent
   decay is the hardest case (.22 to .44 at onset .8). The trend test is level with sup-split
   for early onsets and worse for recent ones (.36 vs .44 at .8 / 125), which supports the
   choice made on 2026-10-21.
3. **Through Stage 1 and Stage 2 together, s_decay is killed at Stage 2 in about .52 of runs**
   at the default preset: a Stage 1 pass rate of .834 (ADR 0010) times .622. Most of the
   remaining s_decay runs reach Stage 3 with a declining edge. **This is a power limit at
   ic .08 and T = 6000, not a defect,** and the memo must say so for any signal Stage 2
   passes.
4. **The profile-LR half-life interval is anti-conservative under no decay, as the Davies
   caveat warned.** On s_real, every estimator excludes "no decay" 10–12% of the time against
   a nominal 5%. τ is unidentified there, and the χ²₁ critical value is too small for a
   supremum over τ. Coverage also falls just short in d2 (.889 and .900 against the ≥ .90
   lower-bound rule) and passes in d3. Every estimator overestimates the decay rate (positive
   λ bias: half-lives too short), which is consistent with τ being selected on the noise.
5. **At the default SNR the half-life is uninformative, as finding 1 predicted.** On d1,
   s_decay's median MLE sits at the grid floor (5 days): a step change fits as well as an
   exponential. Coverage of .997 to 1.000 comes from intervals that span almost the whole
   grid.
6. **The simpler estimators were the more accurate ones.** RMSE(λ) on d2 is .047 for (c),
   .066 for (b) and .078 for KSG. This was not a selection, because none was admissible.

## Decision (accepted 2026-10-23)
- **Kill rule: adopted as pre-registered.** Stage 2 kills `decay.edge_declining` iff the
  sup-split decline test gives p < .05.
- **Half-life: not reported by Stage 2** until a calibrated interval exists (see Backlog).
- Stage 2's M = 500 validation acceptance threshold for s_decay is set from the measured
  power: .622 given a Stage 1 pass at the default preset.
- Stage 2 itself, and its validation, are in ADR 0012.

## Backlog (future work, deferred 2026-10-23)
- **A calibrated half-life interval.** This needs a new pre-registered study. The leading
  candidate replaces the χ²₁ cutoff of the profile LR with one simulated under the null (a
  parametric bootstrap of the sup-over-τ LR). That addresses finding 4 directly, the Davies
  problem behind the 10–12% false-decay rate. The study must:
  - use a fresh root seed, never 20261021;
  - keep this study's gates (false-decay ≤ .075 upper bound; coverage lower bound ≥ .90 on
    identifiable scenarios);
  - take (c) partial_corr as the first candidate, since it was the most accurate here.
- **KSG at full M.** It is only worth doing if a compute budget allows it and a calibrated
  interval exists to compare. At M = 337 it failed the same gates as (b) and (c).
- **Co-parent leak in the decline test** (finding 1): s_real rejects at .055 and .058 when a
  strong decaying co-parent is held at a constant coefficient. A regime-aware residualisation
  could be measured against this.
- **The decline test at T < 6000.** Its size and power are unmeasured there, so Stage 2's
  `min_obs` is 5000 design rows (ADR 0012). A shorter-sample study would allow a lower floor.
