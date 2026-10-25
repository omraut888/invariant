from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.dro.robust import (
    MeanStdProblem,
    mean_std_risk,
    robust_mean_std_risk,
    worst_case_samples,
)


def _instance(seed: int, n: int = 80, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = rng.standard_t(5, (n, k)) * 0.01 + rng.standard_normal(k) * 0.001
    w = rng.standard_normal(k)
    return samples, w / np.abs(w).sum()


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 2**32 - 1), st.floats(0.0, 4.0), st.floats(1e-5, 1e-2))
def test_explicit_worst_case_attains_the_closed_form(seed: int, c: float, eps: float) -> None:
    samples, w = _instance(seed)
    moved = worst_case_samples(samples, w, c, eps)
    cost = float(np.mean(np.sum((moved - samples) ** 2, axis=1)))
    assert cost == pytest.approx(eps**2, rel=1e-9)
    mu, cov = samples.mean(axis=0), np.cov(samples.T, bias=True)
    closed = robust_mean_std_risk(mu, cov, w, c, eps)
    assert mean_std_risk(moved, w, c) == pytest.approx(closed, rel=1e-9, abs=1e-14)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**32 - 1), st.floats(0.0, 4.0), st.floats(1e-5, 1e-2))
def test_no_feasible_perturbation_exceeds_the_closed_form(seed: int, c: float, eps: float) -> None:
    samples, w = _instance(seed)
    mu, cov = samples.mean(axis=0), np.cov(samples.T, bias=True)
    closed = robust_mean_std_risk(mu, cov, w, c, eps)
    rng = np.random.default_rng(seed + 1)
    for _ in range(25):
        e = rng.standard_normal(samples.shape)
        e *= eps / np.sqrt(np.mean(np.sum(e**2, axis=1)))  # transport cost exactly eps²
        assert mean_std_risk(samples + e, w, c) <= closed + 1e-12


def test_closed_form_matches_the_gelbrich_sdp() -> None:
    """The direct formulation over (mean, covariance) in the Gelbrich ball, in percent units
    (Nguyen et al., arXiv:2112.09959, Thm 5). Raw daily units make SCS inaccurate."""
    rng = np.random.default_rng(3)
    for _ in range(6):
        n, k = 150, int(rng.integers(2, 5))
        samples = rng.standard_t(5, (n, k))
        w = rng.standard_normal(k)
        w /= np.abs(w).sum()
        c, eps = float(rng.uniform(0.2, 3.0)), float(rng.uniform(0.05, 0.5))
        mu, cov = samples.mean(axis=0), np.cov(samples.T, bias=True)
        m = cp.Variable(k)
        s = cp.Variable((k, k), symmetric=True)
        x = cp.Variable((k, k))
        prob = cp.Problem(
            cp.Maximize(-m @ w + c * cp.sqrt(cp.quad_form(w, s, assume_PSD=True))),
            [
                cp.sum_squares(m - mu) + cp.trace(s) + np.trace(cov) - 2 * cp.trace(x) <= eps**2,
                cp.bmat([[s, x], [x.T, cov]]) >> 0,
                s >> 0,
            ],
        )
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        assert prob.value == pytest.approx(robust_mean_std_risk(mu, cov, w, c, eps), rel=1e-5)


def test_portfolio_matches_brute_force_on_two_assets() -> None:
    """Homogeneous objective: the optimum is max(0, best point on the L1 sphere)."""
    rng = np.random.default_rng(7)
    prob = MeanStdProblem(2, c=0.3)
    for _ in range(20):
        a = rng.standard_normal((2, 2))
        cov = (a @ a.T + 0.2 * np.eye(2)) * 1e-4
        mu = rng.standard_normal(2) * 1e-3
        kappa = float(rng.uniform(0.0, 2e-3))
        sol = prob.solve(mu, cov, kappa)
        t = np.linspace(0, 2 * np.pi, 20001)
        grid = np.c_[np.cos(t), np.sin(t)]
        grid /= np.abs(grid).sum(axis=1, keepdims=True)
        vals = (
            grid @ mu
            - 0.3 * np.sqrt(np.einsum("ij,jk,ik->i", grid, cov, grid))
            - kappa * (np.linalg.norm(grid, axis=1))
        )
        best = max(0.0, float(vals.max()))
        assert sol.value == pytest.approx(best, abs=1e-8)
        if best == 0.0:
            assert not sol.invested
        else:
            assert np.abs(sol.w).sum() == pytest.approx(1.0, abs=1e-6)


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2**32 - 1), st.floats(1e-3, 1e3))
def test_solution_is_invariant_to_rescaling_units(seed: int, k: float) -> None:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((4, 4))
    cov = (a @ a.T + 0.1 * np.eye(4)) * 1e-4
    mu, kappa = rng.standard_normal(4) * 1e-3, 1e-4
    prob = MeanStdProblem(4, c=0.1)
    base = prob.solve(mu, cov, kappa)
    scaled = prob.solve(mu * k, cov * k * k, kappa * k)
    np.testing.assert_allclose(scaled.w, base.w, atol=1e-6)
    assert scaled.value == pytest.approx(base.value * k, rel=1e-5, abs=1e-12)


def test_a_high_hurdle_or_radius_means_cash() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 3))
    cov = (a @ a.T + 0.1 * np.eye(3)) * 1e-4
    mu = np.array([1e-3, -5e-4, 2e-4])
    assert MeanStdProblem(3, c=0.0).solve(mu, cov, 0.0).invested
    assert not MeanStdProblem(3, c=50.0).solve(mu, cov, 0.0).invested
    assert not MeanStdProblem(3, c=0.0).solve(mu, cov, 1.0).invested


def test_validation() -> None:
    with pytest.raises(ValueError):
        MeanStdProblem(2, c=-1.0)
    with pytest.raises(ValueError):
        robust_mean_std_risk(np.zeros(2), np.eye(2), np.ones(2), 1.0, -1.0)
    with pytest.raises(ValueError):
        MeanStdProblem(2, c=0.1).solve(np.zeros(3), np.eye(2), 0.0)


def _cvxpy_reference(mu: np.ndarray, cov: np.ndarray, c: float, kappa: float) -> float:
    """The same program written in cvxpy, in percent units (the formulation of ADR 0013)."""
    n = len(mu)
    w = cp.Variable(n)
    chol = np.linalg.cholesky(cov * 1e4).T
    prob = cp.Problem(
        cp.Maximize(mu * 100 @ w - c * cp.norm(chol @ w, 2) - kappa * 100 * cp.norm(w, 2)),
        [cp.norm(w, 1) <= 1],
    )
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.OPTIMAL
    return max(0.0, float(prob.value) / 100)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(2, 6), st.floats(0.0, 0.2), st.floats(0.0, 2e-3))
def test_direct_clarabel_matches_the_cvxpy_formulation(
    seed: int, n: int, c: float, kappa: float
) -> None:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    cov = (a @ a.T + 0.1 * np.eye(n)) * 1e-4
    mu = rng.standard_normal(n) * 1e-3
    sol = MeanStdProblem(n, c).solve(mu, cov, kappa)
    ref = _cvxpy_reference(mu, cov, c, kappa)
    assert sol.value == pytest.approx(ref, rel=1e-6, abs=1e-10)
    if sol.invested:
        realised = float(
            mu @ sol.w - c * np.sqrt(sol.w @ cov @ sol.w) - kappa * np.linalg.norm(sol.w)
        )
        assert realised == pytest.approx(ref, rel=1e-6, abs=1e-10)
        assert np.abs(sol.w).sum() <= 1.0 + 1e-12
