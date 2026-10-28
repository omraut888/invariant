# ADR 0009: Stage 1 CI test and decision test (size studies)

Status: accepted (2026-10-18). Three pre-registered studies; study 3's decision is a recorded
deviation from its pre-registered rule. Study 1 (CI test) is complete and its results
are recorded below. Study 2 (decision test x CI test) was pre-registered and committed before
its full-M run. A confirmation of study 3's deviation on a fresh root seed was
pre-registered and pushed on 2026-10-18, before its run. **It confirmed `hc3|hybrid`** (last
section). Each study's design and rules were written before any of its full-M results existed.

## Context
Stage 1 runs PCMCI (Runge et al. 2019) and kills a signal when no lag in [1, tau_max] has a
significant MCI link into returns. The benchmark's returns have GARCH(1,1)-t(6) residuals. The
question is whether that inflates ParCorr's false-positive rate, and whether a
heteroskedasticity-aware test (tigramite `ParCorrWLS`) or a rank-based one (`RobustParCorr`)
should replace it.

A prior expectation, stated so the study can refute it: for a link X -> R, the OLS t-test is
oversized only when `X^2` is correlated with `Var(R | past)` (White 1980). The benchmark's
signals are independent of volatility by construction, so on S -> R links ParCorr should be
close to nominal size, and the inflation, if any, should appear on R -> R and R -> S links. A
benchmark-only study would therefore flatter ParCorr. That is why E4 exists.

## Dependencies
- `tigramite` 5.2.10.1. Verified against numpy 2.4.6, pandas 3.0.6 and scipy 1.17.1 with
  DeprecationWarning and FutureWarning raised as errors.
- `joblib`: tigramite imports it without declaring it (a packaging bug in 5.2.10.1). It is
  also used to run the study's seeds in parallel.

## FastParCorrWLS
tigramite's `ParCorrWLS._get_single_residuals` builds a dense `T x T` diagonal weight matrix:
O(T^2), ~290 MB and ~37 s per PCMCI run at T = 6000. `FastParCorrWLS` overrides only that
method, multiplying by the weight vector elementwise. The p- and value matrices of a full
PCMCI run are bit-identical to the original (tests/unit/test_ci_tests.py), at 0.8 s per run.
Without it the study would need ~3 days of CPU for WLS alone. The estimator itself is
unchanged, including two properties worth knowing about: the trailing-window weight at t
includes `|resid_t|` itself, and the first `window - 1` weights are 1.

## Design
**Decision unit.** A *link family* is the set of lags of one `source -> target` link. Its
decision on one dataset is the Stage 1 decision unit: reject iff `min_lag p <= alpha / n_lags`
(Bonferroni, valid under any dependence between lags). A one-lag family is a fixed-lag test at
alpha. Each family gives one Bernoulli decision per seed, and seeds are independent, so the
Wilson interval of the rejection rate is honest. Correlated lags are never pooled as if they
were independent trials.

**Scenarios** (every scenario uses the same dataset seed for seed i, common random numbers):

| | DGP | Role |
|---|---|---|
| E1 | Benchmark, all ICs 0, GARCH off (i.i.d. t(6)) | control; nothing gates |
| E2 | Benchmark, all ICs 0, GARCH-t | **gates**: z, s_real, s_conf, s_decay, s_noise -> R (lags 1..5) |
| E3 | Default preset, Z observed | **gates**: s_conf -> R (lags 1..5); power: s_real, s_decay |
| E4 | Test-only: `S_t = log h_{t+1}`, `R_t = mu + a_t` | **gates**: s_vol -> R (lags 1..5) |
| E5 | Realistic preset (IC 0.02) | power only |

E1 uses i.i.d. t(6) rather than Gaussian residuals, because `GarchParams` requires nu > 4.
Comparing E1 with E2 therefore isolates the GARCH effect from the heavy tails.
Reported but not gating: fixed lag-1 families, R -> R, R -> S, s_noise in E3, s_real at lags
2..5 in E3, KS tests of p-value uniformity for one-lag nulls.

E4 (tests/fixtures/vol_null.py) is a vol-timing signal under the null. `S_t` is a function of
returns up to t (tests/leakage/test_vol_null_timing.py), it has no linear relation to `R_{t+1}`,
and `S_t^2` is correlated with `Var(R_{t+1})`. It does not change the benchmark generator.

**Candidates**: `parcorr`, `wls10` and `wls50` (`ParCorrWLS`, time-dependent
heteroskedasticity, windows 10 and 50), `robust` (`RobustParCorr`). Analytic p-values throughout.
PCMCI: tau_min 1, tau_max 5, pc_alpha 0.2, T = 6000. The study reads `dataset.features` and
`dataset.returns` directly: the PIT panel's `syn_*` columns equal them bit for bit (ADR 0008).

## Rules (pre-registered)
**Admissible** iff, on every one of the 7 gating families, the upper bound of the 95% Wilson
interval of the rejection rate at alpha = 0.05 is <= 0.075.

**Number of seeds.** The rule is a test of "size >= 0.075" against "size < 0.075", so a test
that is truly at 0.05 needs enough seeds to pass. M is the smallest seed count from which a
true-0.05 test passes each gate with probability >= 1 - 0.10/7 for every M' in [M, M + 200].
Over 7 gates (union bound, any dependence) that gives an all-pass probability >= 0.90.
`required_seeds` computes **M = 1654**. At M = 1654, per gate:

| true size | P(pass) |
|---|---|
| 0.050 | 0.9886 (all 7 gates: >= 0.920) |
| 0.060 | 0.675 |
| 0.075 | 0.025 |
| 0.100 | 3e-8 |

Bonferroni families have true size <= 0.05, so these are lower bounds for a well-sized test.
Limitation: the rule cannot tell a size of 0.06 from 0.05.

Smaller M fails well-sized tests. At M = 500 a true-0.05 test passes one gate with probability
0.55, and at M = 100 with probability 0.12. A rule run at M of a few hundred would rule tests
out on sample size, not behaviour.

**Selection.** Among admissible tests, power is the mean decision-unit rejection rate over
s_real -> R and s_decay -> R in E3. The highest-power test wins unless another admissible
test's paired per-seed power difference from it has a 95% CI containing 0. Tied tests go to the
lowest median runtime. If no test is admissible, none is selected and the decision returns to
review.

**Reproducibility.** Seed i's dataset seed and CI-test seeds come from
`RngFactory(root_seed).child("seed/i")`, so serial and parallel runs give identical records
(tests/unit/test_calibration.py). The output directory is keyed by a hash of (study config, M,
git state), and the uncommitted patch is saved next to the report.

## Results (M = 1654, T = 6000, report `runs/causal_ci_study/487866d2fda16623/report.json`; archived at [`docs/results/0009-causal-ci-study-1/report.json`](../results/0009-causal-ci-study-1/report.json))
Code: commit `09d3f44`. The run was recorded as `54a4ffe` + `dirty.patch`, and that tree is
identical to `09d3f44` outside `docs/adr/` (verified).

Rejection rate at alpha = 0.05 with 95% Wilson CI. Gates are marked *. Bold marks a failed gate.

| Family (decision unit) | parcorr | wls10 | wls50 | robust |
|---|---|---|---|---|
| E2 s_real->R * | .054 [.044,.066] | .057 [.047,.069] | .059 [.049,.072] | .054 [.044,.066] |
| E2 s_conf->R * | .053 [.043,.064] | .048 [.038,.059] | .047 [.037,.058] | .047 [.038,.058] |
| E2 s_decay->R * | .042 [.034,.053] | .045 [.036,.056] | .043 [.034,.054] | .043 [.034,.054] |
| E2 s_noise->R * | .046 [.037,.057] | .045 [.036,.056] | .045 [.036,.056] | .040 [.031,.050] |
| E2 z->R * | **.066 [.055,.079]** | .046 [.037,.057] | .062 [.051,.074] | .062 [.051,.074] |
| E3 s_conf->R * | .055 [.045,.067] | .048 [.039,.060] | .048 [.038,.059] | .048 [.039,.060] |
| E4 s_vol->R * | **.198 [.179,.218]** | .034 [.026,.044] | **.126 [.111,.143]** | **.074 [.063,.088]** |
| E1 R->R (no GARCH) | .056 | .039 | .056 | .056 |
| E2 R->R | .244 | .047 | .099 | .118 |
| E1 z->R (no GARCH) | .065 [.054,.078] | .045 | .065 | .065 |
| E3 s_real->R lags 2..5 | .102 | .092 | .099 | .099 |
| E3 power s_real->R | .346 | .401 | .413 | .403 |
| E3 power s_decay->R | .358 | .397 | .396 | .414 |
| E5 power s_real->R | .082 | .097 | .099 | .102 |
| E5 power s_decay->R | .059 | .081 | .070 | .071 |
| median s / PCMCI run | 1.19 | 1.51 | 1.61 | 3.90 |

Selection-score power (mean of the two E3 families): parcorr .352, wls10 .399, wls50 .404,
robust .409. wls10 minus parcorr: +.047 [+.031, +.064] (paired).

**Rule outcome: `wls10` is the only admissible test and is selected.**

## Findings
1. **The prior expectation held.** In E2, ParCorr is at nominal size on every S -> R link
   whose regressor is independent of volatility. Its GARCH inflation sits on R -> R (.244, against
   .056 without GARCH in E1), and on the vol-correlated null E4 (.198, four times nominal).
   wls10 is calibrated on both (.047, .034), and its fixed-lag E4 rate is .050 (KS p .16).
2. **The window result is not monotone: wls10 passes E4, wls50 fails it.** A longer window
   should estimate a GARCH variance with persistence 0.98 *better*. The likely reason is that
   tigramite's weight at t includes `|resid_t|` itself, so with window 10 each residual is
   partly self-normalised (bounded influence). With window 50 that effect is 5x weaker. If so,
   wls10's calibration may come from that self-normalisation and not from variance estimation,
   and it may not carry over to other vol-correlated nulls. Not yet tested.
3. **z -> R is oversized for parcorr, wls50 and robust in E1, without GARCH** (.065
   [.054, .078]). This is not a heteroskedasticity effect. Z has phi = 0.99. A likely cause is
   post-selection: PC1 picks a near-unit-root regressor's lags on the same data that MCI then
   tests. Not established. wls10 is at .045 here.
4. **MCI has low power on persistent signals. This is a property of the method, not of the CI
   test.** MCI conditions on the parents of the cause, so it uses only the cause's innovation.
   Its t-stat is about `sqrt(1 - rho^2)` times the predictive-regression t-stat that
   `GroundTruth.expected_t` reports. From population moments, the ratio is 0.43 for
   s_real/s_decay (rho 0.9) and 0.15 for Z (phi 0.99). The expected MCI t is 1.92 for s_real
   against an expected_t of 4.43. The predicted lag-1 power of 0.485 matches the observed .518
   (parcorr). Consequences:
   - In the default preset Stage 1 will pass s_real and s_decay only ~40% of the time, before
     any cross-signal FDR. `GroundTruth.low_power` (expected_t < 3) does not flag this.
   - s_conf is still killed at nominal rates even though Z -> R is rarely found (power .12).
     MCI conditions on s_conf's own parent `Z_{t-2}`, and that blocks the confounding path.
5. **Lags are not identified.** s_real at lags 2..5 rejects ~.10 given the conditioning set.
   When PC1 misses `s_real_{t-1}` (it often does at this power), the lag-2 value stands in for
   it. The signal-level decision is unaffected, but a "best lag" must not be read as the true lag.

## Decision (study 1)
Not adopted on its own. On review (2026-10-14), findings 2 and 4 were judged to need a second
study before any choice. The CI test and the signal -> R decision test are now chosen jointly
in study 2.

---

# Study 2: decision test x CI test (pre-registered 2026-10-14, before the full-M run)

## Questions
- **A.** Which conditioning set should decide signal -> R, given finding 4 (MCI's power loss on
  persistent signals) and the risks a smaller set brings back: confounding through a parent of
  X that PC1 did not select for R, and Stambaugh bias?
- **B.** Is wls10's E4 calibration robust, or a side effect of its weight at t including
  `|resid_t|` (finding 2)?

## Candidates: (CI test, decision test) pairs, 7 x 3 = 21
Every decision test uses the same PCMCI run: the same PC1 parents P(.) and the same CI test.
For `X_{t-tau} -> R_t`:

| Decision test | Conditions on |
|---|---|
| (a) `mci` | `P(R) \ {X_{t-tau}}` plus `P(X)` shifted by tau (tigramite's MCI set; reproduces `p_matrix` exactly, tests/unit/test_pcmci.py) |
| (b) `parents_r` | `P(R) \ {X_{t-tau}}` |
| (c) `hybrid` | (b) plus the shifted parents of X other than X's own lags |

CI tests:
- `parcorr`.
- `wls5`, `wls10`, `wls20`: tigramite's `ParCorrWLS` estimator, whose weight at t includes
  `|resid_t|`.
- `loo5`, `loo10`, `loo20`: `LeaveCurrentOutParCorrWLS`, whose weights come from
  `|resid_{t-w..t-1}|` only. The weights are a one-step variance forecast with no
  self-normalisation. The first w rows use the mean of the first w residuals.

## Scenarios
Root seed 20261014, the same as study 1, so seed i gives the same datasets. Each scenario has
T = 6000. E1 is dropped: it was a control, and it answers neither A nor B.

| | DGP | Gates (Bonferroni over lags 1..5) |
|---|---|---|
| E2 | benchmark, all ICs 0, GARCH-t | s_real, s_conf, s_decay, s_noise -> R (z is not a candidate: reported only) |
| E3 | default preset, Z observed | s_conf -> R, s_noise -> R; power: s_real, s_decay |
| E4 | log h_{t+1} signal, GARCH .06/.92 | s_vol -> R |
| E4-lo | as E4, GARCH .10/.80 (persistence .90) | s_vol -> R |
| E4-hi | as E4, GARCH .04/.95 (persistence .99) | s_vol -> R |
| E4-rv | as E4, signal = log 22-day trailing realized variance | s_vol -> R |
| E6 | test-only Stambaugh null: x AR(1) rho .99, R = mu + e, corr(u, e) = -0.8, Gaussian | s_stam -> R |
| E5 | realistic preset | none (power, reported) |

That is 11 gates. E2's s_real and s_decay gates go beyond the A list asked for. Under the
global null they are candidate signals Stage 1 decides on, and leaving them out would test
less, not more.

E4 variants and E6 are test-only (tests/fixtures/). The realized-vol signal and E6's x use
only data up to t (tests/leakage/test_vol_null_timing.py, by construction for E6).

## Rules
- **Admissible**: the Wilson 95% upper bound of the rejection rate at alpha = 0.05 is <= 0.075
  on all 11 gates. This is study 1's rule.
- **M** = `required_seeds(0.05, 0.075, 11, 0.90)` = **1785**. A true-0.05 pair passes one gate
  with probability 0.9928, and all 11 with probability >= 0.92 (union bound). A true-0.06 pair
  passes one gate with probability 0.708, a true-0.075 pair with 0.025, a true-0.10 pair with
  1.5e-8.
- **Selection**: among admissible pairs, the highest mean E3 power over s_real -> R and
  s_decay -> R (decision unit). Pairs whose paired per-seed power difference from the best has a
  95% CI containing 0 are tied. Ties go to the simpler pair: first by decision test
  `parents_r < hybrid < mci` (fewer conditioning variables), then by CI test `parcorr < wls10 <
  wls5 < wls20 < loo10 < loo5 < loo20` (unweighted, then the published estimator with its
  default window first, then our variant). If no pair is admissible, nothing is selected and
  the decision returns to review.
- **B is answered by the same table.** If wls_w passes the E4 variants and loo_w does not, the
  self-normalisation hypothesis is supported. If both pass, the variance estimate is enough. In
  either case the rule above makes the choice, and B's reading is reported.

## Consistency check (pre-registered)
With the same root seed, every `parcorr|mci` and `wls10|mci` cell on E2, E3, E4 and E5 for
seeds < 1654 must equal study 1's record bit for bit. A smoke run at M = 8 matched on all 720
overlapping cells. Its rates (CI width ~ +/-0.3) were not used for anything. Any mismatch in the
full run invalidates it.

## Follow-ups bound by this study, whatever wins
1. Stage 1 records the MCI p-value (decision test (a)) in the kill ledger as a diagnostic next to
   the deciding p-value, even when (a) is not the deciding test.
2. GroundTruth gains the expected t of the chosen decision test (the population partial
   correlation for its conditioning set, times sqrt(n - 1)), so `low_power` flags the method
   actually used.

## Study 2 results (M = 1785, commit `e3217bc`, clean tree; report `runs/causal_ci_study/c7cb0210a18a30fd/report.json`; archived at [`docs/results/0009-causal-ci-study-2/report.json`](../results/0009-causal-ci-study-2/report.json))
The consistency check passed: 148,860 overlapping `*|mci` cells equal study 1 bit for bit.

The rule selected `wls20|hybrid`. **That selection is void: see the defect below.**

Rejection rates (decision unit, alpha .05). `*` marks a failed gate (Wilson upper > .075).

| pair | E2 real/conf/dec/noise | E3 conf/noise | E4 / lo / hi / rv | E6 | E3 power real/dec | E5 power real/dec |
|---|---|---|---|---|---|---|
| parcorr\|mci | .056 .053 .045 .046 | .055 .045 | .198* .235* .160* .096* | .013 | .342 .361 | .084 .063 |
| parcorr\|parents_r | .041 .026 .035 .030 | **.446*** .026 | .147* .206* .118* .089* | .019 | .915 .896 | .284 .107 |
| parcorr\|hybrid | .043 .029 .036 .029 | .031 .026 | .148* .207* .118* .090* | .020 | .894 .895 | .239 .110 |
| wls10\|mci | .057 .048 .048 .045 | .049 .044 | .033 .047 .027 .051 | .010 | .404 .398 | .095 .082 |
| wls10\|hybrid | .049 .042 .044 .040 | .040 .045 | **.000 .000 .000 .000** | .024 | .901 .908 | .232 .125 |
| wls20\|hybrid | .047 .038 .043 .041 | .039 .039 | **.000 .000 .000 .000** | .023 | .907 .929 | .252 .128 |
| loo10\|hybrid | .042 .041 .052 .042 | .039 .045 | .018 .030 .010 .027 | .026 | .894 .882 | .210 .124 |

The full 21-pair table is in the report.

## Defect: tigramite's ParCorrWLS is not location-invariant (found 2026-10-15)
Every tigramite-WLS pair with `parents_r` or `hybrid` rejects **0 of 1785** seeds on all four E4
variants. Its lag-1 p-values lie in [.78, .99] (1st percentile .778). That is a degenerate test,
not a calibrated one. Cause, in tigramite 5.2.10.1:
- `ParCorrWLS._get_single_residuals` defaults to `standardize=False` (ParCorr's default is
  `True`), and `get_dependence_measure` calls it without the flag.
- The regressions have no intercept.
- With an empty conditioning set, the "time-dependent std" is a window mean of `|level|`,
  not of `|residual|`.

So a variable whose mean is far from 0 (E4's `log h` ~ -9.2) is weighted into a near-constant.
Check: adding +10 to X changes `parcorr_wls`'s p-value from .27 to .44 (Z empty) or .70 (Z = X's
own lag). `parcorr` and `robust_parcorr` are unchanged. `FastParCorrWLS` and
`LeaveCurrentOutParCorrWLS` inherit the defect (they mirror tigramite exactly).

Consequences:
- **Void**: the study 2 selection; every WLS/LOO cell on E4 (both studies), including study 1's
  "wls10 is calibrated on E4" and finding 2 (window non-monotonicity), which are likely artifacts.
- **Likely small, not verified**: WLS cells on E2/E3/E5. Benchmark signals are ~N(0, 1), and
  returns have mean .0003 against sd .01, so the level effect there is small but not zero.
- **Valid**: every `parcorr` and `robust_parcorr` cell, and every comparison between decision
  tests within `parcorr`.

## What the valid evidence (parcorr) says
1. **parents_r is ruled out.** It falsely passes S_conf in E3 at .446 (Wilson upper .469), as
   predicted on review: Z -> R is rarely selected, so Z is rarely in P(R).
2. **hybrid keeps S_conf killed (.031) and recovers power**: .894/.895 against MCI's .342/.361
   on s_real/s_decay in E3. In E5 its power is .239/.110 against .084/.063.
3. **Stambaugh (E6) is not a problem at T = 6000** for any decision test (.013 to .020, lag-1
   .045). The `T(1 - rho) = 60` horizon is long enough that the slope bias is negligible.
   Limitation: a shorter sample or rho closer to 1 could change this, and it was not tested.
4. **The heteroskedasticity question is still open.** parcorr fails every E4 variant with every
   decision test (.09 to .24). No valid heteroskedasticity-robust test has been measured yet.

## Decision (study 2)
None. The CI-test choice returns to review with a corrected weighted test and/or an HC-robust
test. The decision test is provisionally `hybrid`, pending a valid CI test that passes E4.

## Upstream assessment of the ParCorrWLS defect (2026-10-16)
**Verdict: a bug, not a documented or an undocumented assumption.**
- *Not documented as an assumption.* The class docstring, the CI-test tutorial's assumption
  table and the dedicated ParCorrWLS tutorial (tigramite repo, `tutorials/causal_discovery/`)
  never require zero-mean or standardized inputs. The tutorial's toy data has
  `noise_means=[0.]`, so the case never arises there.
- *Contradicted by tigramite's own docstring.* `ParCorrWLS._get_single_residuals` documents
  `standardize ... (default: True) ... Must be used for partial correlation`, but its signature
  is `standardize=False`.
- *Location invariance is a property of partial correlation.* `ParCorr` and `RobustParCorr`
  have it. Reproduction with tigramite alone (`scripts/repro_tigramite_parcorrwls_location.py`,
  T = 1000, 400 reps): with Z = {} and a true effect, power is .988 at mean(X) = 0 and **.048**
  at mean(X) = 10. Size is .045 and .015.
- No upstream issue mentions ParCorrWLS. The fix is not on `master`. The issue is drafted in
  `docs/upstream/tigramite-parcorrwls-location-dependence.md` and has not been filed.

---

# Study 3: corrected weighted tests and HC3 (pre-registered 2026-10-16, before the full-M run)

## Question
Which CI test is valid on the vol-correlated nulls (E4 variants) without giving up calibration
elsewhere? And, given that test, is `hybrid` still the decision test? `parents_r` is dropped:
it fails S_conf (.446 under parcorr), and that failure comes from its conditioning set, not
its CI test.

## Candidates: 8 CI tests x {hybrid, mci} = 16 pairs
- `parcorr`.
- `hc3`: `ParCorrHC3`. ParCorr's statistic, with the p-value from White's HC3 t-test on X's
  coefficient in `Y ~ 1 + X + Z`. Equals statsmodels' HC3 p-value (tests/unit/test_ci_invariance.py).
- `cwls5`, `cwls10`, `cwls20`: `CenteredParCorrWLS`, with an intercept in both regressions and
  the variance from `|residual|`. The trailing window includes `|resid_t|`.
- `cloo5`, `cloo10`, `cloo20`: as `cwlsW`, but the window ends at t - 1.

Every candidate's p-value is invariant to affine transforms of X, Y and Z (Hypothesis property
test, all three conditioning-set shapes). The defective classes are pinned as still defective.

## Scenarios and gates
As study 2: the same 11 gates (E2 s_real/s_conf/s_decay/s_noise, E3 s_conf/s_noise, E4 /
E4-lo / E4-hi / E4-rv, E6) and the same root seed, so M = **1785**, with the same pass
probabilities.

Added, **non-gating diagnostic**: `e6_stambaugh_short`, E6 at T = 500 (two trading years,
`T(1 - rho) = 5`), where Stambaugh bias should matter. It reports size for each pair.

## Rules
- Admissible, power and ties: as study 2. Power is the mean E3 decision-unit rejection over
  s_real and s_decay, and a tie is a paired 95% CI on the difference that contains 0.
- **Tie-break (simplicity)**: by decision test `hybrid < mci` (fewer conditioning variables),
  then by CI test `parcorr < hc3 < cwls10 < cwls5 < cwls20 < cloo10 < cloo5 < cloo20` (fewer
  tuning parameters: none, then none with a robust SE, then one window, then one window plus
  the leave-current-out choice; tigramite's default window first).
- If nothing is admissible, nothing is selected and the decision returns to review.

## Consistency check (pre-registered)
Every `parcorr|hybrid` and `parcorr|mci` cell at T = 6000 must equal study 2's record bit for
bit (same seeds, same code path). A smoke run at M = 8 matched on all 944 overlapping cells.
Its rates were not used. Any mismatch in the full run invalidates it.

## Bound follow-ups (unchanged from study 2)
The MCI p-value is recorded in the ledger as a diagnostic. GroundTruth gains the chosen
decision test's expected t, so `low_power` matches the method in use.

## Study 3 results (M = 1785, commit `ff38fc2`, clean tree; report `runs/causal_ci_study/188421e7d7adadc3/report.json`; archived at [`docs/results/0009-causal-ci-study-3/report.json`](../results/0009-causal-ci-study-3/report.json))
The consistency check passed: 210,630 `parcorr|*` cells equal study 2 bit for bit. No admissible
pair is degenerate: the 1% and 5% quantiles of its lag-1 null p-values are ~.01 and ~.05.

Rejection rates (decision unit, alpha .05). `*` marks a failed gate.

| pair | E2 re/co/de/no | E3 co/no | E4 / lo / hi / rv | E6 | E3 power re/de | E5 power re/de | E6 T=500 (lag 1) | E2 R->R |
|---|---|---|---|---|---|---|---|---|
| parcorr\|hybrid | .043 .029 .036 .029 | .031 .026 | .148* .207* .118* .090* | .020 | .894 .895 | .239 .110 | .039 (.082) | .246 |
| hc3\|hybrid | .045 .029 .033 .029 | .029 .026 | .030 .046 .024 .021 | .020 | .890 .887 | .237 .111 | .039 (.082) | .049 |
| hc3\|mci | .058 .053 .041 .045 | .054 .046 | .054 .049 .055 .043 | .013 | .352 .359 | .086 .062 | .012 (.017) | .049 |
| cwls10\|hybrid | .044 .041 .045 .039 | .038 .044 | .018 .029 .019 .029 | .024 | .903 .912 | .232 .127 | .036 (.056) | .050 |
| **cwls20\|hybrid** | .045 .037 .041 .040 | .034 .035 | .030 .055 .026 .032 | .024 | .907 .933 | .243 .134 | .032 (.058) | .068 |
| cwls20\|mci | .058 .045 .046 .044 | .048 .044 | .066* .076* .055 .066* | .012 | .428 .419 | .105 .072 | .013 (.019) | .066 |
| cloo10\|hybrid | .038 .044 .048 .047 | .042 .050 | .018 .030 .015 .031 | .024 | .889 .886 | .209 .123 | .029 (.046) | .048 |
| cloo20\|hybrid | .041 .040 .043 .040 | .037 .040 | .035 .073* .029 .043 | .021 | .900 .922 | .234 .126 | .033 (.053) | .073 |

The full 16-pair table is in the report. Admissible: hc3 x {hybrid, mci}, cwls5/10 x {hybrid,
mci}, cwls20|hybrid, cloo5/10 x {hybrid, mci}.

**Rule outcome: `cwls20|hybrid`** (power .920). It beats the next admissible pairs on paired
power difference: cwls10|hybrid +.013 [+.003, +.023], hc3|hybrid +.032 [+.020, +.043].

## Findings (study 3)
1. **hybrid is confirmed as the decision test under every CI test.** Its power is .81 to .92,
   against .31 to .43 for mci, with no admissibility cost.
2. **HC3 fixes what ParCorr gets wrong.** Its E4 variants are at .021 to .046 (parcorr .090 to
   .207), and E2 R -> R is at .049 (parcorr .246). Its lag-1 null p-values are KS-uniform on
   every vol null (KS p .14 to .44). It gives up .032 of power against cwls20|hybrid, and is
   level with it in E5.
3. **cwls20|hybrid passes, but sits at the edge.** Its siblings fail the same nulls:
   cwls20|mci fails three E4 gates and cloo20|hybrid fails E4-lo (.073). Its own E4-lo is .055,
   E2 R -> R is .068, and its lag-1 null p-values are *not* uniform on E4, E4-rv and E2 (KS p
   .000 to .001). Window 20 is also the largest of the windows tried, so it is the most
   likely to be the winner's curse of a 6-way window search.
4. **The two-sided decision unit holds Stambaugh at T = 500** (.029 to .039 for hybrid pairs).
   The fixed lag-1 test does not for parcorr/hc3 hybrid (.082): bias appears in short samples,
   and MCI's conditioning on X's past absorbs it (.012 to .028).
5. **Realistic-preset power stays low** (.21 to .24 for s_real, .11 to .13 for s_decay). This is
   a power result at IC .02 and T = 6000, as pre-registered.

## Decision (study 3), 2026-10-18
**Stage 1 uses `hc3|hybrid`: CI test `ParCorrHC3`, decision test `hybrid`.** This is a
**recorded deviation** from the pre-registered rule, whose outcome was `cwls20|hybrid`.

Reasons, all from results reported (not gated) under the pre-registration:
- cwls20|hybrid sits at the edge of admissibility. Its nearest variants fail the same nulls:
  cwls20|mci fails three E4 gates, and cloo20|hybrid fails E4-lo.
- Its lag-1 null p-values are not uniform on E4, E4-rv and E2 (KS p <= .001). Its E2 R -> R is
  .068.
- Window 20 is the largest of six window settings tried, so its power lead (+.032 [+.020,
  +.043]) is partly a winner's curse.
- hc3|hybrid passes every gate with margin (E4 variants .021 to .046) and has no tuning
  parameter. It matches statsmodels' HC3 exactly. Its lag-1 null p-values are KS-uniform on
  every vol null (p .14 to .44), and it fixes parcorr's R -> R inflation (.049).

Cost of the deviation: .032 of E3 power (s_real .890 vs .907, s_decay .887 vs .933). E5
power is level.

Consequences:
- Stage 1 decides with the Bonferroni family over lags 1..tau_max, never with a single lag:
  at T = 500 the lag-1 test is oversized under Stambaugh (.082).
- The MCI p-value (`hc3|mci`) is recorded in the ledger as a diagnostic.
- GroundTruth gains the expected t of the hybrid test.
- The tigramite ParCorrWLS-derived classes are not used for decisions.

---

# Confirmation of the study 3 deviation (pre-registered 2026-10-18, before any run)

## Question
Study 3's decision (`hc3|hybrid`) departs from its pre-registered rule, and every number behind
the decision came from one root seed, so it may be partly a winner's curse. Does `hc3|hybrid`
pass all 11 gates on data it has never seen?

## Design
- Config `configs/causal_study3_confirm.yaml`. It has study 3's scenario set: the same 11 gates
  and the non-gating T = 500 Stambaugh diagnostic.
- **Fresh root seed 20261018.** Studies 1-3 all used 20261014. No dataset is shared with study 3,
  so there is no consistency check.
- Pairs: `hc3|hybrid` (under test) and `cwls20|hybrid` (the rule's outcome in study 3,
  reported alongside). Both are run on the same datasets (common random numbers).
- M = `required_seeds(0.05, 0.075, 11, 0.90)` = **1785**, T = 6000, tau_max 5, pc_alpha 0.2,
  alpha 0.05, analytic p-values, as in study 3.
- It is run from a clean worktree at the commit that adds this section, after that commit is pushed.

## Rule
- **Confirmed** iff `hc3|hybrid` is admissible: the upper bound of the 95% Wilson interval of
  its decision-unit rejection rate is <= 0.075 on every one of the 11 gates.
- `cwls20|hybrid` gates nothing. Its rates, E3/E5 power and the paired power difference are
  reported. The report's `selected` field is not used.
- **If not confirmed**, the CI-test decision returns to review. There is no automatic switch to
  `cwls20|hybrid` or to any other pair.
- Known error rate of the rule: a pair truly at size 0.05 passes one gate with probability
  0.9928 and all 11 with probability >= 0.92 (union bound). A failure is therefore strong
  evidence of miscalibration, not proof. A true-0.06 pair passes one gate with probability 0.708.

## Confirmation results (M = 1785, commit `694c75f`, clean tree; report `runs/causal_ci_study/617e5f4a76f2f39a/report.json`; archived at [`docs/results/0009-causal-ci-confirmation/report.json`](../results/0009-causal-ci-confirmation/report.json))
**Outcome under the rule: confirmed.** `hc3|hybrid` is admissible on the fresh seed: no gate
failed, and the largest Wilson upper bound is .058, on E2 s_decay. `cwls20|hybrid` is also
admissible. The report's `selected` field (`cwls20|hybrid`) is not used, as pre-registered.

Rejection rates (decision unit, alpha .05, 95% Wilson CI). Every row is a gate unless marked
otherwise.

| family | hc3\|hybrid | cwls20\|hybrid | study 3 hc3 / cwls20 |
|---|---|---|---|
| E2 s_real | .043 [.034, .053] | .044 [.036, .055] | .045 / .045 |
| E2 s_conf | .029 [.022, .037] | .035 [.028, .045] | .029 / .037 |
| E2 s_decay | .047 [.038, .058] | .050 [.041, .061] | .033 / .041 |
| E2 s_noise | .034 [.027, .044] | .029 [.022, .037] | .029 / .040 |
| E3 s_conf | .029 [.022, .038] | .036 [.029, .046] | .029 / .034 |
| E3 s_noise | .031 [.024, .041] | .029 [.022, .037] | .026 / .035 |
| E4 | .024 [.018, .032] | .026 [.019, .034] | .030 / .030 |
| E4-lo | .039 [.031, .049] | .053 [.043, .064] | .046 / .055 |
| E4-hi | .022 [.016, .030] | .021 [.016, .029] | .024 / .026 |
| E4-rv | .026 [.020, .035] | .036 [.028, .046] | .021 / .032 |
| E6 | .020 [.015, .028] | .020 [.014, .027] | .020 / .024 |
| *E3 power s_real / s_decay* | .884 / .881 | .898 / .915 | .890 .887 / .907 .933 |
| *E5 power s_real / s_decay* | .248 / .105 | .252 / .119 | .237 .111 / .243 .134 |
| *E6 T = 500, family (lag 1)* | .050 (.077) | .035 (.053) | .039 (.082) / .032 (.058) |
| *E2 R -> R* | .056 | .068 | .049 / .068 |

E3 power (mean of the two families): hc3|hybrid .883, cwls20|hybrid .906. The paired difference
is +.024 [+.012, +.035], against +.032 [+.020, +.043] in study 3.

## Findings (confirmation)
1. **Study 3's numbers replicate on new data.** Every `hc3|hybrid` gate rate is within about
   .015 of study 3's.
2. **The cost of the deviation replicates:** .024 of E3 power, with E5 level.
3. **cwls20's edge behaviour replicates.** It is at .053 on E4-lo, and its R -> R rates are
   .068 to .082 across scenarios, against .052 to .062 for hc3. This supports the deviation's
   reasoning, but was not part of the rule.
4. **The short-sample lag-1 problem replicates.** At T = 500, the fixed lag-1 hc3|hybrid test
   rejects at .077 [.066, .091] (study 3: .082). The Bonferroni family is at .050. This
   confirms the decision to decide on the family, never on a single lag. At T = 500 the family
   is at nominal size, with no margin.
