from __future__ import annotations

import numpy as np
import pytest

from invariant.core.run_hash import RngFactory
from invariant.stages.dro.walkforward import (
    Arm,
    WalkForwardConfig,
    fc_radius,
    fit_linear,
    run_walk_forward,
    solve_path,
)

SMALL = WalkForwardConfig(
    c=0.03,
    window=150,
    refit_every=10,
    cv_every=50,
    cv_valid=50,
    ridge_grid=(0.0, 0.1, 1.0),
    eps_grid=(0.0, 0.01, 0.1),
    fc_draws=2000,
)


def _data(seed: int, t: int = 400, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((t, 2))
    for s in range(1, t):
        x[s] = 0.9 * x[s - 1] + np.sqrt(1 - 0.81) * rng.standard_normal(2)
    beta = np.array([[0.002, 0.0], [0.001, 0.001], [0.0, 0.002]])[:n]
    r = np.zeros((t, n))
    r[1:] = 0.0003 + x[:-1] @ beta.T + 0.01 * rng.standard_normal((t - 1, n))
    return x, r


def test_fit_matches_per_asset_ols_and_hc0_sandwich() -> None:
    x, r = _data(0)
    fit = fit_linear(x[:-1], r[1:], hac_lags=0)
    d = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    for i in range(r.shape[1]):
        b, *_ = np.linalg.lstsq(d, r[1:, i], rcond=None)
        np.testing.assert_allclose(fit.coef[i], b, rtol=1e-10)
    e = r[1:] - d @ fit.coef.T
    a = np.linalg.inv(d.T @ d)
    manual = a @ (d.T * (e[:, 0] * e[:, 1])) @ d @ a  # HC0 cross-covariance, assets 0 and 1
    np.testing.assert_allclose(fit.coef_cov[0, :, 1, :], manual, rtol=1e-10)
    x_t = np.array([0.3, -1.2])
    dd = np.r_[1.0, x_t]
    assert fit.forecast_cov(x_t)[0, 1] == pytest.approx(dd @ manual @ dd, rel=1e-10)
    np.testing.assert_allclose(fit.resid_cov, e.T @ e / len(e), rtol=1e-12)


def test_radius_covers_the_true_conditional_mean_at_nominal_rate() -> None:
    """i.i.d. Gaussian regression: ||mu_hat - mu||_2 <= eps in ~95% of repeated samples."""
    rng = np.random.default_rng(1)
    draws = np.random.default_rng(2).standard_normal((4000, 3))
    beta = np.array([[0.002, 0.0], [0.001, 0.001], [0.0, 0.002]])
    x_t = np.array([1.0, -0.5])
    hits = []
    for _ in range(400):
        x = rng.standard_normal((300, 2))
        y = x @ beta.T + 0.01 * rng.standard_normal((300, 3))
        fit = fit_linear(x, y, hac_lags=0)
        eps = fc_radius(fit.forecast_cov(x_t), draws, 0.05)
        hits.append(np.linalg.norm(fit.forecast(x_t) - beta @ x_t) <= eps)
    assert 0.92 <= float(np.mean(hits)) <= 0.98


def test_zero_penalty_arms_reproduce_markowitz() -> None:
    x, r = _data(3)
    cfg = WalkForwardConfig(**{**SMALL.__dict__, "ridge_grid": (0.0,), "eps_grid": (0.0,)})
    paths = run_walk_forward(x, r, list(Arm), cfg, RngFactory(1))
    for a in (Arm.RIDGE_CV, Arm.DRO_CV):
        np.testing.assert_allclose(paths[a].weights, paths[Arm.MARKOWITZ].weights, atol=1e-7)
    np.testing.assert_allclose(paths[Arm.EQUAL_WEIGHT].weights, 1 / 3)
    assert (np.abs(paths[Arm.MARKOWITZ].weights).sum(axis=1) <= 1 + 1e-9).all()
    fc = paths[Arm.DRO_FC]
    assert fc.radius is not None and (fc.radius > 0).all()
    assert len(paths[Arm.DRO_CV].chosen) == len(range(0, len(fc.dates), cfg.cv_every))


def test_deterministic_and_oracle_path() -> None:
    x, r = _data(4)
    a = run_walk_forward(x, r, list(Arm), SMALL, RngFactory(9))
    b = run_walk_forward(x, r, list(Arm), SMALL, RngFactory(9))
    for arm in Arm:
        np.testing.assert_array_equal(a[arm].weights, b[arm].weights)
    dates = a[Arm.MARKOWITZ].dates
    mean = np.zeros((len(r), 3)) + 1e-3
    cov = np.repeat(np.eye(3)[None] * 1e-4, len(r), axis=0)
    w, rets = solve_path(mean, cov, r, dates, 0.03)
    assert w.shape == (len(dates), 3)
    np.testing.assert_allclose(rets, np.einsum("tn,tn->t", w, r[dates + 1]))


def test_config_validated() -> None:
    with pytest.raises(ValueError):
        WalkForwardConfig(c=-1.0)
    with pytest.raises(ValueError):
        WalkForwardConfig(c=0.1, window=100, cv_valid=100)
    x, r = _data(5, t=100)
    with pytest.raises(ValueError):
        run_walk_forward(x, r, [Arm.MARKOWITZ], SMALL, RngFactory(1))
