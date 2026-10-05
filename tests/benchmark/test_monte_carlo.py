"""200-seed Monte Carlo: size and power of the tests the benchmark is meant to exercise.

Results are printed (run with ``-s``) and recorded as junit properties; the numbers are
reported in docs/adr/0003-planted-signal-benchmark-scm.md. ``INVARIANT_MC_SEEDS`` overrides the
seed count (seeds are 0..N-1) for a higher-resolution run; the asserted bands scale with N.

Asserted: the partial-correlation test of S_conf given the true parents is correctly sized
(that is the claim "zero partial correlation"), S_conf is marginally significant in most
samples (the claim "it is a real trap"), and empirical power for S_real / S_decay agrees with
the ``expected_t`` exported in GroundTruth.

Measured, not asserted: the empirical size of a naive OLS t-test vs a Newey-West t-test of
``R_{t+1}`` on the persistent ``S_noise_t``. Granger-Newbold spurious regression needs *both*
series persistent; returns are near white noise, so any distortion is expected to be mild, and
whatever the number is, it is the finding.
"""

import math
import os
from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.diagnostics import CovType, ols
from invariant.benchmark.synthetic import (
    SyntheticDataset,
    decay_multiplier,
    generate_market,
    stationary_covariance,
    transition_matrices,
)

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

N_SEEDS = int(os.environ.get("INVARIANT_MC_SEEDS", "200"))
ALPHA = 0.05
Z_CRIT = float(stats.norm.ppf(1 - ALPHA / 2))
PARENTS = ["z", "s_real", "s_decay"]


def _design(
    d: SyntheticDataset, regressors: list[str], n_return_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    f = d.features[regressors].to_numpy()
    r = d.returns["asset_00"].to_numpy()
    j = np.arange(max(n_return_lags - 1, 0), len(r) - 1)
    x = np.column_stack([f[j]] + [r[j - lag] for lag in range(n_return_lags)])
    return r[j + 1], x


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def _naive_size_prediction(cfg: SyntheticMarketConfig) -> float:
    """Asymptotic size of the naive OLS t-test of R_{t+1} on S_noise_t.

    The univariate error ``e_t = c'X_t + a_{t+1}`` contains the omitted persistent predictable
    part of returns, so ``Var(t) -> [Var(e) + 2 sum_k rho^k c'A^k Sigma c] / Var(e)`` with
    ``sum_k (rho A)^k = (I - rho A)^{-1} - I``. S_decay's loading uses its RMS multiplier.
    """
    a, _ = transition_matrices(cfg)
    sigma = stationary_covariance(cfg)
    mult = decay_multiplier(cfg)[cfg.burn_in : cfg.burn_in + cfg.n_obs - 1]
    sb = cfg.garch.sigma_bar
    c = (
        np.array(
            [
                cfg.factor.ic,
                cfg.real.ic,
                0.0,
                cfg.decay.ic * math.sqrt(float((mult**2).mean())),
                0.0,
            ]
        )
        * sb
    )
    var_e = float(c @ sigma @ c) + sb**2
    geo = np.linalg.inv(np.eye(5) - cfg.noise.rho * a) - np.eye(5)
    var_t = (var_e + 2.0 * float(c @ geo @ sigma @ c)) / var_e
    return float(2.0 * stats.norm.sf(Z_CRIT / math.sqrt(var_t)))


def _two_sided_power(expected_t: float) -> float:
    return float(stats.norm.sf(Z_CRIT - expected_t) + stats.norm.cdf(-Z_CRIT - expected_t))


Test = Callable[[SyntheticDataset], float]


def _t(regressors: list[str], cov: CovType, lags: int) -> Test:
    def run(d: SyntheticDataset) -> float:
        y, x = _design(d, regressors, lags)
        return float(ols(y, x, cov=cov).tstat[1])

    return run


TESTS: dict[str, Test] = {
    "s_noise_naive_ols": _t(["s_noise"], CovType.IID, 0),
    "s_noise_newey_west": _t(["s_noise"], CovType.HAC, 0),
    "s_conf_marginal_hac": _t(["s_conf"], CovType.HAC, 0),
    "s_conf_partial_ols": _t(["s_conf", *PARENTS], CovType.IID, 5),
    "s_conf_partial_hac": _t(["s_conf", *PARENTS], CovType.HAC, 5),
    "s_real_partial_hac": _t(["s_real", "z", "s_decay"], CovType.HAC, 5),
    "s_decay_partial_hac": _t(["s_decay", "z", "s_real"], CovType.HAC, 5),
}


def test_size_and_power_monte_carlo(record_property: Callable[[str, object], None]) -> None:
    cfg = load_benchmark_config()
    rejections = dict.fromkeys(TESTS, 0)
    truth = None
    for seed in range(N_SEEDS):
        d = generate_market(cfg, seed)
        truth = d.ground_truth
        for name, test in TESTS.items():
            rejections[name] += abs(test(d)) > Z_CRIT
    assert truth is not None

    print(f"\nMonte Carlo: {N_SEEDS} seeds, n_obs={cfg.n_obs}, two-sided alpha={ALPHA}")
    rates = {}
    for name, k in rejections.items():
        rate = k / N_SEEDS
        lo, hi = _wilson(k, N_SEEDS)
        rates[name] = rate
        record_property(name, rate)
        print(f"  {name:<22} rejection rate {rate:.3f}  (95% Wilson CI {lo:.3f}-{hi:.3f})")

    # Size of the partial test under the true null: binomial 99.9% band around ALPHA.
    lo_band = stats.binom.ppf(0.0005, N_SEEDS, ALPHA) / N_SEEDS
    hi_band = stats.binom.isf(0.0005, N_SEEDS, ALPHA) / N_SEEDS
    for name in ("s_conf_partial_ols", "s_conf_partial_hac"):
        assert lo_band <= rates[name] <= hi_band, (name, rates[name], (lo_band, hi_band))

    assert rates["s_conf_marginal_hac"] > 0.9, "S_conf must usually look significant marginally"

    for sid in ("s_real", "s_decay"):
        predicted = _two_sided_power(truth.signal(sid).expected_t[0])
        record_property(f"{sid}_predicted_power", predicted)
        print(f"  {sid} predicted power from GroundTruth.expected_t: {predicted:.3f}")
        assert rates[f"{sid}_partial_hac"] == pytest.approx(predicted, abs=0.1)

    # S_noise: measured, not asserted (see module docstring).
    predicted_naive = _naive_size_prediction(cfg)
    record_property("s_noise_naive_ols_predicted_size", predicted_naive)
    print(f"  s_noise naive-OLS size predicted analytically: {predicted_naive:.3f}")
    for name in ("s_noise_naive_ols", "s_noise_newey_west"):
        assert 0.0 <= rates[name] <= 1.0
