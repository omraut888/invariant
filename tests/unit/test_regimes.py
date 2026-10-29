"""The Stage 4 Markov-switching GARCH labeller (stages/adversarial/regimes.py, ADR 0016).

Seeds here are unrelated to the dev seeds (root 20261101) the labeller is validated on.
"""

import math
import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from invariant.stages.adversarial.regimes import (
    MarkovSwitchingGarchFit,
    fit_ms_garch,
    label_regimes,
    loglik,
    select_regimes,
    smoothed_probabilities,
)


def _chain(n: int, p00: float, p11: float, gen: np.random.Generator) -> np.ndarray:
    xi = np.empty(n, dtype=np.int64)
    state = 0
    for t, u in enumerate(gen.random(n)):
        if t > 0:
            state = int(u >= p00) if state == 0 else int(u < p11)
        xi[t] = state
    return xi


def _ms_garch(
    n: int,
    seed: int,
    *,
    omega: tuple[float, float],
    alpha: tuple[float, float],
    beta: tuple[float, float],
    p: tuple[float, float],
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    gen = np.random.default_rng(seed)
    xi = _chain(n, p[0], p[1], gen)
    z = gen.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu)
    h = np.array([w / (1 - a - b) for w, a, b in zip(omega, alpha, beta, strict=True)])
    eps = np.empty(n)
    for t in range(n):
        if t > 0:
            h = np.array(omega) + np.array(alpha) * eps[t - 1] ** 2 + np.array(beta) * h
        eps[t] = math.sqrt(h[xi[t]]) * z[t]
    return eps, xi


def _garch_t_loglik(r: np.ndarray, mu: float, w: float, a: float, b: float, nu: float) -> float:
    """Independent single-regime GARCH(1,1)-t log-likelihood (scipy's t density)."""
    eps = r - mu
    h = w / (1 - a - b)
    scale_c = math.sqrt((nu - 2.0) / nu)
    total = 0.0
    for t in range(len(r)):
        if t > 0:
            h = w + a * eps[t - 1] ** 2 + b * h
        s = math.sqrt(h) * scale_c
        total += float(stats.t.logpdf(eps[t], nu, scale=s))
    return total


# --- statsmodels cross-check (restricted model: Hamilton 1989 MS variance) ------------------


@pytest.fixture(scope="module")
def ms_variance() -> np.ndarray:
    gen = np.random.default_rng(314)
    xi = _chain(3000, 0.98, 0.95, gen)
    return 0.0002 + np.where(xi == 1, 0.025, 0.01) * gen.standard_normal(3000)


def _statsmodels_fit(r: np.ndarray) -> tuple[MarkovSwitchingGarchFit, float]:
    """statsmodels' MarkovRegression fit, expressed as our restricted model, and its llf."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.tsa.MarkovRegression(
            r, k_regimes=2, trend="c", switching_trend=False, switching_variance=True
        ).fit(disp=False)
    names = list(res.model.param_names)
    par = dict(zip(names, res.params, strict=True))
    p00, p10 = par["p[0->0]"], par["p[1->0]"]
    var = (par["sigma2[0]"], par["sigma2[1]"])
    order = (0, 1) if var[0] <= var[1] else (1, 0)
    p_mat = np.array([[p00, 1 - p00], [p10, 1 - p10]])[np.ix_(order, order)]
    fit = MarkovSwitchingGarchFit(
        n_regimes=2,
        mu=float(par["const"]),
        omega=tuple(float(var[k]) for k in order),
        alpha=(0.0, 0.0),
        beta=(0.0, 0.0),
        nu=math.inf,
        transition=tuple(tuple(float(v) for v in row) for row in p_mat),
        loglik=float(res.llf),
        n_params=6,
        n_obs=len(r),
        converged=True,
    )
    return fit, float(res.llf)


def test_likelihood_matches_statsmodels_at_its_estimates(ms_variance: np.ndarray) -> None:
    # Same model (switching variance, common mean, steady-state start): our exact likelihood at
    # statsmodels' parameters must equal its llf.
    sm_fit, sm_llf = _statsmodels_fit(ms_variance)
    assert loglik(ms_variance, sm_fit) == pytest.approx(sm_llf, abs=1e-6)


def test_fit_matches_statsmodels(ms_variance: np.ndarray) -> None:
    sm_fit, sm_llf = _statsmodels_fit(ms_variance)
    ours = fit_ms_garch(ms_variance, 2, gaussian=True, garch=False)
    assert ours.loglik >= sm_llf - 1e-4  # at least as good an optimum
    assert ours.loglik == pytest.approx(sm_llf, abs=1e-2)
    np.testing.assert_allclose(ours.omega, sm_fit.omega, rtol=1e-3)
    np.testing.assert_allclose(ours.transition, sm_fit.transition, atol=1e-3)
    assert ours.mu == pytest.approx(sm_fit.mu, abs=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.tsa.MarkovRegression(
            ms_variance, k_regimes=2, trend="c", switching_trend=False, switching_variance=True
        ).fit(disp=False)
    sm_probs = res.smoothed_marginal_probabilities
    high = int(np.argmax([res.params[-2], res.params[-1]]))
    np.testing.assert_allclose(
        smoothed_probabilities(ms_variance, ours)[:, 1], sm_probs[:, high], atol=1e-3
    )


# --- single regime = GARCH-t -----------------------------------------------------------------


def test_one_regime_likelihood_is_garch_t() -> None:
    r, _ = _ms_garch(
        1500, 7, omega=(2e-6, 2e-6), alpha=(0.06, 0.06), beta=(0.92, 0.92), p=(0.5, 0.5), nu=6.0
    )
    fit = fit_ms_garch(r, 1)
    expected = _garch_t_loglik(r, fit.mu, fit.omega[0], fit.alpha[0], fit.beta[0], fit.nu)
    assert fit.loglik == pytest.approx(expected, rel=1e-10)
    assert fit.stress_share == 0.0 and fit.n_params == 5
    assert not label_regimes(r, fit).any()


# --- recovery and selection on planted processes ---------------------------------------------


@pytest.fixture(scope="module")
def planted() -> tuple[np.ndarray, np.ndarray]:
    # Well separated: unconditional variances 10x apart, stress share 25%, moderate persistence.
    return _ms_garch(
        8000,
        2718,
        omega=(3e-6, 3e-5),
        alpha=(0.10, 0.10),
        beta=(0.80, 0.80),
        p=(0.99, 0.97),
        nu=6.0,
    )


def test_recovers_a_planted_ms_garch(planted: tuple[np.ndarray, np.ndarray]) -> None:
    r, xi = planted
    fit = select_regimes(r).chosen
    assert fit.n_regimes == 2
    v = fit.unconditional_variance
    assert v[0] == pytest.approx(3e-5, rel=0.25)
    assert v[1] == pytest.approx(3e-4, rel=0.25)
    assert fit.transition[0][0] == pytest.approx(0.99, abs=0.005)
    assert fit.transition[1][1] == pytest.approx(0.97, abs=0.015)
    assert (label_regimes(r, fit) == xi).mean() > 0.9


def test_optimum_is_at_least_as_likely_as_the_truth() -> None:
    # A weakly identified case (13% stress, persistent GARCH): BIC may prefer one regime, but
    # the two-regime optimiser must still reach at least the true parameters' likelihood.
    omega, alpha, beta, p = (1.2e-6, 1.2e-5), (0.05, 0.05), (0.90, 0.90), (0.995, 0.98)
    r, _ = _ms_garch(8000, 2718, omega=omega, alpha=alpha, beta=beta, p=p, nu=6.0)
    truth = MarkovSwitchingGarchFit(
        n_regimes=2,
        mu=0.0,
        omega=omega,
        alpha=alpha,
        beta=beta,
        nu=6.0,
        transition=((p[0], 1 - p[0]), (1 - p[1], p[1])),
        loglik=0.0,
        n_params=10,
        n_obs=len(r),
        converged=True,
    )
    assert fit_ms_garch(r, 2).loglik >= loglik(r, truth)


def test_bic_selects_one_regime_for_plain_garch() -> None:
    r, _ = _ms_garch(
        6000, 99, omega=(1e-6, 1e-6), alpha=(0.06, 0.06), beta=(0.92, 0.92), p=(0.5, 0.5), nu=6.0
    )
    sel = select_regimes(r)
    assert sel.chosen.n_regimes == 1


def test_smoothed_probabilities_are_distributions(planted: tuple[np.ndarray, np.ndarray]) -> None:
    r, _ = planted
    fit = fit_ms_garch(r[:2000], 2)
    probs = smoothed_probabilities(r[:2000], fit)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert fit.unconditional_variance[0] <= fit.unconditional_variance[1]


def test_fit_is_deterministic(planted: tuple[np.ndarray, np.ndarray]) -> None:
    r = planted[0][:1500]
    assert fit_ms_garch(r, 2) == fit_ms_garch(r, 2)


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_ms_garch(np.full(200, np.nan), 2)
    with pytest.raises(ValueError, match="n_regimes"):
        fit_ms_garch(np.zeros(200) + np.arange(200) * 1e-4, 3)
