from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.decay.halflife import (
    DecayGrid,
    HalfLifeFit,
    fit_gaussian_te,
    fit_ksg,
    fit_partial_corr,
    make_grid,
    window_decay_mean,
)
from invariant.stages.decay.te import KsgWindows
from invariant.stages.decay.windows import WindowPartialCorr, window_bounds

N, W = 6000, 250


def _m(i: np.ndarray, tau: float, hl: float) -> np.ndarray:
    return np.where(i < tau, 1.0, np.exp2(-np.maximum(i - tau, 0) / hl))


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 1200), st.floats(1.0, 5000.0), st.integers(5, 200))
def test_window_mean_matches_brute_force(tau: int, hl: float, width: int) -> None:
    bounds = window_bounds(1200, width)
    s = window_decay_mean(bounds, np.array([tau]), np.array([hl]))[0, 0]
    brute = [_m(np.arange(a, b), tau, hl).mean() for a, b in bounds]
    np.testing.assert_allclose(s, brute, rtol=1e-10, atol=1e-14)


def _grid() -> DecayGrid:
    return make_grid(N, tau_step=50, hl_min=5.0, hl_max=50_000.0, n_hl=121)


def _wpc(z: np.ndarray, var: float) -> WindowPartialCorr:
    bounds = window_bounds(N, W)
    return WindowPartialCorr(
        bounds=bounds, r=np.tanh(z), fisher_z=z, fisher_var=np.full(len(z), var)
    )


def test_noiseless_partial_corr_curve_recovers_half_life_and_onset() -> None:
    grid = _grid()
    hl_te = float(grid.half_lives[70])  # on the grid
    tau = 3600
    shapes = window_decay_mean(window_bounds(N, W), np.array([tau]), np.array([2.0 * hl_te]))
    fit = fit_partial_corr(_wpc(0.2 * shapes[0, 0], 1e-6), grid)
    assert fit.lam_hat == pytest.approx(math.log(2) / hl_te)
    assert fit.tau_hat == tau
    assert fit.amplitude_hat == pytest.approx(0.2)
    assert fit.claims_decay and fit.covers(hl_te)


def test_constant_curve_includes_no_decay() -> None:
    fit = fit_partial_corr(_wpc(np.full(N // W, 0.08), 1.0 / 247), _grid())
    assert fit.lam_hat == 0.0 and fit.lam_low == 0.0 and not fit.claims_decay
    assert fit.covers(1e12)


def test_noiseless_gaussian_te_and_ksg_recover_te_half_life() -> None:
    grid = _grid()
    hl_te = float(grid.half_lives[60])
    bounds = window_bounds(N, W)
    s = window_decay_mean(bounds, np.array([3000]), np.array([hl_te]))[0, 0]
    te = 0.03 * s
    z = np.arctanh(np.sqrt(-np.expm1(-2.0 * te)))
    fit_b = fit_gaussian_te(_wpc(z, 1e-6), grid)
    assert fit_b.lam_hat == pytest.approx(math.log(2) / hl_te)
    assert fit_b.amplitude_hat == pytest.approx(0.03, rel=1e-4)
    k = len(bounds)
    ksg = KsgWindows(
        bounds=bounds,
        value=0.01 + te,
        null_mean=np.full(k, 0.01),
        null_sd=np.full(k, 1e-5),
        p=np.zeros(k),
    )
    fit_a = fit_ksg(ksg, grid)
    assert fit_a.lam_hat == pytest.approx(math.log(2) / hl_te)
    assert fit_a.amplitude_hat == pytest.approx(0.03)


def test_ksg_amplitude_is_non_negative() -> None:
    bounds = window_bounds(N, W)
    k = len(bounds)
    ksg = KsgWindows(
        bounds=bounds,
        value=np.full(k, 0.0),
        null_mean=np.full(k, 0.01),
        null_sd=np.full(k, 0.005),
        p=np.ones(k),
    )
    fit = fit_ksg(ksg, _grid())
    assert fit.amplitude_hat == 0.0 and not fit.claims_decay


def test_gaussian_te_profile_matches_brute_force_amplitude_grid() -> None:
    rng = np.random.default_rng(5)
    bounds = window_bounds(N, W)
    z = 0.1 + rng.standard_normal(len(bounds)) / math.sqrt(247)
    grid = DecayGrid(taus=np.array([0, 3000]), half_lives=np.array([100.0, 1000.0]))
    fit = fit_gaussian_te(_wpc(z, 1.0 / 247), grid)
    # brute force over a fine TE0 grid at every (tau, HL) and at no decay
    shapes = window_decay_mean(bounds, grid.taus, grid.half_lives).reshape(-1, len(bounds))
    shapes = np.vstack([shapes, np.ones(len(bounds))])
    best = -np.inf
    for te0 in np.linspace(0.0, 0.05, 20001):
        mu = np.arctanh(np.sqrt(-np.expm1(-2.0 * te0 * shapes)))
        a, v = np.abs(z), 1.0 / 247
        ll = (-((a - mu) ** 2) / (2 * v) + np.log1p(np.exp(-2 * a * mu / v))).sum(axis=1)
        best = max(best, float(ll.max()))
    assert fit.loglik_max == pytest.approx(best, abs=1e-6)
    assert fit.loglik_max >= best - 1e-9


def test_partial_corr_interval_covers_in_gaussian_monte_carlo() -> None:
    """With exactly Gaussian window z's the profile-LR interval should cover near 95%.

    Davies' problem is absent here because the truth decays. 200 replications resolve coverage
    to about +-3 points.
    """
    rng = np.random.default_rng(20261021)
    grid = make_grid(N, tau_step=100, hl_min=5.0, hl_max=50_000.0, n_hl=61)
    bounds = window_bounds(N, W)
    hl_te, var = 250.0, 1.0 / 247
    mean = 0.25 * window_decay_mean(bounds, np.array([3600]), np.array([2 * hl_te]))[0, 0]
    covered = [
        fit_partial_corr(
            _wpc(mean + rng.standard_normal(len(mean)) * math.sqrt(var), var), grid
        ).covers(hl_te)
        for _ in range(200)
    ]
    assert 0.88 <= float(np.mean(covered)) <= 1.0


def test_fit_model_is_frozen_and_rates_are_finite() -> None:
    fit = fit_partial_corr(_wpc(np.full(N // W, 0.08), 1.0 / 247), _grid())
    assert isinstance(fit, HalfLifeFit)
    assert all(math.isfinite(v) for v in (fit.lam_hat, fit.lam_low, fit.lam_high))
    with pytest.raises(ValueError):
        fit.lam_hat = 1.0  # type: ignore[misc]


def test_grid_validated() -> None:
    with pytest.raises(ValueError):
        DecayGrid(taus=np.array([0]), half_lives=np.array([5.0]))
    with pytest.raises(ValueError):
        DecayGrid(taus=np.array([0]), half_lives=np.array([5.0, 4.0]))
