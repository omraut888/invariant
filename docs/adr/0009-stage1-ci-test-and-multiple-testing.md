# ADR 0009: Stage 1 CI test and decision test (size studies)

Status: in progress. Two pre-registered studies. Study 1 (CI test) is complete and its results
are recorded below. Study 2 (decision test x CI test) was pre-registered and committed before
its full-M run. Each study's design and rules were written before any of its full-M results
existed.

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

## Results (M = 1654, T = 6000, report `runs/causal_ci_study/487866d2fda16623/report.json`)
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

## Study 2 results
Pending: `uv run python -m scripts.causal_ci_size_study --config configs/causal_decision_study.yaml`.
