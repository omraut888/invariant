# Draft upstream issue: tigramite `ParCorrWLS` p-values depend on the mean of X

Status: **draft, not filed.** Target: https://github.com/jakobrunge/tigramite/issues.
No existing issue mentions ParCorrWLS (searched 2026-10-16).

---

**Title:** ParCorrWLS is not location-invariant: adding a constant to X changes size and power

**Version:** tigramite 5.2.10.1 (also present on `master`, `parcorr_wls.py` line 309), numpy
2.4.6, scipy 1.17.1, Python 3.11.

**Summary.** With `expert_knowledge="time-dependent heteroskedasticity"`, `ParCorrWLS` gives
different p-values when a constant is added to X. Partial correlation is location-invariant,
and `ParCorr` is unaffected. For a variable whose mean is far from 0, the test loses nearly all
power and its size falls toward 0.

**Minimal reproduction** (numpy + tigramite only): X, Y i.i.d. N(0, 1), T = 1000, 400
replications, alpha = 0.05, testing X_{t-1} -> Y_t with `run_test`.

| | Z | ParCorr, mean(X) = 0 / 10 | ParCorrWLS, mean(X) = 0 / 10 |
|---|---|---|---|
| size | {} | .040 / .040 | .045 / **.015** |
| size | {X_{t-2}} | .043 / .043 | .043 / .055 |
| power (Y_t = .15 X_{t-1} + e) | {} | .998 / .998 | .988 / **.048** |
| power | {X_{t-2}} | .995 / .995 | .995 / .907 |

Script: `scripts/repro_tigramite_parcorrwls_location.py` (attach inline when filing).

**Cause.**
1. `ParCorrWLS._get_single_residuals(self, array, target_var, standardize=False, ...)`. Its
   docstring says `standardize : bool, optional (default: True)` and "Must be used for partial
   correlation". `ParCorr.get_dependence_measure` calls it without the argument, so no
   centering happens.
2. Neither the variance-estimation regression (`_estimate_std_time`) nor the weighted
   regression includes an intercept.
3. With an empty conditioning set, `_estimate_std_time` uses `abs(y)` (the level), not the
   absolute residual about the mean.

Together these make the weight `1 / window_mean(|X|)` with X's mean in it. For X far from 0,
the weighted X is close to a constant (about -1 for X around -9), and the test degenerates.
The first `window_size - 1` weights are set to 1, so they also break scale invariance.

**Suggested fix.** Add an intercept column to both regressions (or demean before weighting),
and estimate the variance from residuals about the fitted mean. For rows without a full window,
use the first full window's value, not 1.

**Is it documented?** No. The class docstring, the CI-test tutorial's assumption table
("continuous X, Y, Z with linear dependencies, can account for heteroskedastic dependencies")
and the ParCorrWLS tutorial do not require zero-mean or standardized inputs. The tutorial's
toy data has `noise_means=[0.]`, so it never exercises a nonzero mean.
