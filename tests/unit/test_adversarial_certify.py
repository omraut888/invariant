"""Stage 4 core (option 4): chain, objective, confidence region and certification (ADR 0016)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from invariant.core.run_hash import RngFactory
from invariant.stages.adversarial.blocklength import optimal_block_length
from invariant.stages.adversarial.certify import (
    CertificationConfig,
    certification_block_length,
    certify,
)
from invariant.stages.adversarial.chain import Start, StateChain, state_path
from invariant.stages.adversarial.objective import horizon_weights, mixture_sharpe, sr_h
from invariant.stages.adversarial.region import chain_estimate, fit_region, moving_block_indices

SMALL = CertificationConfig(n_region_boot=199, n_grid=9, n_boot=199, gpd_n_boot=5)


def _states(n: int, seed: int, share: float = 0.3, duration: float = 20.0) -> np.ndarray:
    chain = StateChain(share=share, duration=duration)
    return state_path(chain, Start.STATIONARY, n, np.random.default_rng(seed))


# --- chain and objective ----------------------------------------------------------------------


def test_chain_is_stochastic_with_stationary_share() -> None:
    c = StateChain(share=0.3, duration=20.0)
    np.testing.assert_allclose(c.transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(c.stationary @ c.transition, c.stationary, atol=1e-15)
    with pytest.raises(ValidationError, match="P\\(low -> high\\)"):
        StateChain(share=0.9, duration=2.0)


def test_horizon_weights() -> None:
    c = StateChain(share=0.3, duration=20.0)
    np.testing.assert_allclose(horizon_weights(c, Start.STATIONARY, 252), c.stationary)
    np.testing.assert_allclose(horizon_weights(c, Start.HIGH, 1), c.transition[1])
    w = horizon_weights(c, Start.HIGH, 252)
    assert w.sum() == pytest.approx(1.0) and w[1] > c.share


@settings(max_examples=50, deadline=None)
@given(mu=st.floats(-0.01, 0.01), sd=st.floats(1e-4, 0.05), w=st.floats(0.0, 1.0))
def test_one_state_mixture_is_the_plain_sharpe(mu: float, sd: float, w: float) -> None:
    m, s = np.array([mu, mu]), np.array([sd, sd])
    assert mixture_sharpe(m, s, np.array([1 - w, w])) == pytest.approx(mu / sd * math.sqrt(252))


@settings(max_examples=50, deadline=None)
@given(w1=st.floats(0.0, 1.0), w2=st.floats(0.0, 1.0))
def test_mixture_sharpe_falls_as_the_bad_state_grows(w1: float, w2: float) -> None:
    mu, sd = np.array([0.001, -0.001]), np.array([0.01, 0.02])
    lo, hi = sorted((w1, w2))
    a = mixture_sharpe(mu, sd, np.array([1 - lo, lo]))
    b = mixture_sharpe(mu, sd, np.array([1 - hi, hi]))
    assert b <= a + 1e-12


def test_sr_h_matches_a_simulated_markov_horizon() -> None:
    c = StateChain(share=0.4, duration=30.0)
    mu, sd = np.array([0.001, -0.0005]), np.array([0.01, 0.02])
    gen = np.random.default_rng(11)
    pooled = []
    for _ in range(400):
        z = state_path(c, Start.HIGH, 252, gen)
        pooled.append(mu[z] + sd[z] * gen.standard_normal(252))
    x = np.concatenate(pooled)
    assert sr_h(mu, sd, c, Start.HIGH, 252) == pytest.approx(
        x.mean() / x.std() * math.sqrt(252), abs=0.15
    )


# --- region ---------------------------------------------------------------------------------


def test_chain_estimate_counts_exits() -> None:
    x = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0])
    share, dur = chain_estimate(x[:-1], x[1:])
    assert share == pytest.approx(5 / 8) and dur == pytest.approx(5 / 2)
    assert chain_estimate(np.ones(5, dtype=np.int64), np.ones(5, dtype=np.int64))[1] == math.inf


def test_moving_block_indices() -> None:
    idx = moving_block_indices(100, 7, np.random.default_rng(1))
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100
    assert (np.diff(idx[:7]) == 1).all()


def test_region_covers_the_true_chain_at_about_its_level() -> None:
    # Markov chains with a known (pi, d); 120 replications, 95% region. The miss rate's Wilson
    # upper bound must stay below .12 (a correct region misses about 5%).
    misses = 0
    n_rep = 120
    for rep in range(n_rep):
        x = _states(4739, 1000 + rep)
        block = max(2, round(optimal_block_length(x.astype(float)).circular))
        region = fit_region(x, block, 199, 0.95, np.random.default_rng(rep))
        misses += not region.contains(0.3, 20.0)
    p = misses / n_rep
    z = 1.96
    upper = (
        p + z * z / (2 * n_rep) + z * math.sqrt(p * (1 - p) / n_rep + z * z / (4 * n_rep**2))
    ) / (1 + z * z / n_rep)
    assert upper < 0.12, (misses, upper)


def test_region_grid_lies_in_the_region_and_includes_the_estimate() -> None:
    x = _states(3000, 5)
    region = fit_region(x, 40, 199, 0.95, np.random.default_rng(6))
    grid = region.grid(9)
    assert grid[0].share == region.share_hat and len(grid) > 1
    assert all(region.contains(c.share, c.duration) for c in grid[1:])


def test_region_rejects_a_single_state() -> None:
    with pytest.raises(ValueError, match="both states"):
        fit_region(np.zeros(500, dtype=np.int64), 10, 99, 0.95, np.random.default_rng(0))


# --- certification --------------------------------------------------------------------------


def _market(seed: int, high_mean: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = np.random.default_rng(seed)
    n = 4000
    states = _states(n, seed + 1)
    pnl = np.where(states == 1, high_mean, 0.002) + 0.01 * gen.standard_normal(n)
    ew = 0.0003 + np.where(states == 1, 0.02, 0.01) * gen.standard_normal(n)
    return pnl, ew, states


def test_certify_kills_a_strategy_that_loses_in_high_volatility() -> None:
    pnl, ew, states = _market(20, -0.004)
    res = certify(pnl, ew, states, SMALL, RngFactory(1))
    assert res.kill and res.sr_h_star_upper < 0
    assert res.worst_start is Start.HIGH
    assert res.sr_h_star <= res.sr_h_baseline


def test_certify_passes_a_strategy_that_earns_in_both_states() -> None:
    pnl, ew, states = _market(21, 0.002)
    res = certify(pnl, ew, states, SMALL, RngFactory(2))
    assert not res.kill and res.sr_h_star_quantiles[0] > 0


def test_certify_is_deterministic_and_checks_alignment() -> None:
    pnl, ew, states = _market(22, -0.001)
    assert certify(pnl, ew, states, SMALL, RngFactory(3)) == certify(
        pnl, ew, states, SMALL, RngFactory(3)
    )
    with pytest.raises(ValueError, match="aligned"):
        certify(pnl[:-1], ew, states, SMALL, RngFactory(4))


def test_block_length_rule_takes_the_larger_dependence() -> None:
    pnl, _, states = _market(23, 0.0)
    b = certification_block_length(pnl, states)
    assert b == max(2, round(optimal_block_length(states.astype(float)).circular))
