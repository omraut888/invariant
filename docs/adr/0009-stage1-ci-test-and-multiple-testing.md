# ADR 0009: Stage 1 CI test choice (size study)

Status: in progress. The pre-registered design (this section through "Rules") was written before
any full-M result existed. Results and the decision are appended below it.

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

## Results
Pending: `uv run python -m scripts.causal_ci_size_study`.
