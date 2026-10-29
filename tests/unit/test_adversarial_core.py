"""Stage 4 core: generator, realism gate, objective and certification (docs/adr/0016)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from invariant.core.run_hash import RngFactory
from invariant.stages.adversarial.certify import CertificationConfig, certify
from invariant.stages.adversarial.generator import BlockPool, Start, StateChain, state_path
from invariant.stages.adversarial.objective import horizon_weights, mixture_sharpe, sr_h
from invariant.stages.adversarial.realism import (
    RealismReference,
    mmd2_unbiased,
    null_threshold,
    window_features,
)

SMALL = CertificationConfig(
    n_null=19, n_share=4, n_duration=4, n_boot=99, n_report_paths=20, gpd_n_boot=5
)


def _states(n: int, seed: int) -> np.ndarray:
    return state_path(
        StateChain(share=0.3, duration=20.0), Start.STATIONARY, n, np.random.default_rng(seed)
    )


# --- generator --------------------------------------------------------------------------------


def test_chain_is_stochastic_with_stationary_share() -> None:
    c = StateChain(share=0.3, duration=20.0)
    np.testing.assert_allclose(c.transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(c.stationary @ c.transition, c.stationary, atol=1e-15)
    with pytest.raises(ValidationError, match="P\\(low -> high\\)"):
        StateChain(share=0.9, duration=2.0)
    assert not StateChain.is_valid(0.9, 2.0) and StateChain.is_valid(0.3, 20.0)


def test_state_path_occupancy_and_spells() -> None:
    z = _states(400_000, 1)
    est = StateChain.from_states(z)
    # Batch-means SE of the share; spells are geometric with sd sqrt(d (d - 1)).
    per = [b.mean() for b in np.array_split(z, 100)]
    assert abs(est.share - 0.3) < 4 * np.std(per, ddof=1) / 10
    n_spells = 400_000 * 0.3 / 20
    assert abs(est.duration - 20) < 4 * math.sqrt(20 * 19) / math.sqrt(n_spells)


def test_pool_types_tiles_and_coverage() -> None:
    obs = _states(3000, 2)
    pool = BlockPool.from_states(obs, 40)
    assert len(pool.tile_types) == 75
    for k in (0, 1):
        for start in pool.starts[k][:50]:
            assert (obs[start : start + 40].mean() > 0.5) == bool(k)
    assert pool.coverage[0].sum() + pool.coverage[1].sum() == (3000 - 40 + 1) * 40
    with pytest.raises(ValueError, match="block length"):
        BlockPool.from_states(obs, 2000)


def test_generated_paths_follow_the_type_chain() -> None:
    obs = _states(3000, 3)
    pool = BlockPool.from_states(obs, 30)
    chain = StateChain(share=0.6, duration=3.0)
    idx = pool.generate(chain, Start.HIGH, 3000, np.random.default_rng(4))
    types = state_path(chain, Start.HIGH, 100, np.random.default_rng(4))
    assert len(idx) == 3000
    for j in range(100):
        block = idx[j * 30 : (j + 1) * 30]
        np.testing.assert_array_equal(np.diff(block), 1)  # one contiguous observed block
        assert (obs[block].mean() > 0.5) == bool(types[j])


@pytest.mark.parametrize("block", [10, 30, 70])
def test_no_observed_day_repeats_within_63_positions(block: int) -> None:
    obs = _states(4000, 12)
    pool = BlockPool.from_states(obs, block)
    idx = pool.generate(
        StateChain(share=0.5, duration=3.0), Start.HIGH, 8000, np.random.default_rng(13)
    )
    last: dict[int, int] = {}
    for pos, day in enumerate(idx.tolist()):
        if day in last:
            assert pos - last[day] > 63
        last[day] = pos


def test_resample_draws_one_block_per_tile_of_each_type() -> None:
    obs = _states(3000, 5)
    pool = BlockPool.from_states(obs, 25)
    i0, i1 = pool.resample(np.random.default_rng(6))
    n_high = int(pool.tile_types.sum())
    assert len(i1) == n_high * 25 and len(i0) == (len(pool.tile_types) - n_high) * 25


def test_type_moments_are_coverage_weighted() -> None:
    obs = _states(600, 7)
    pool = BlockPool.from_states(obs, 20)
    pnl = np.random.default_rng(8).standard_normal(600)
    mu, _ = pool.type_moments(pnl)
    direct = np.concatenate([pnl[s : s + 20] for s in pool.starts[1]]).mean()
    assert mu[1] == pytest.approx(direct)


def test_segments_reject_a_single_type() -> None:
    with pytest.raises(ValueError, match="both block types"):
        BlockPool.from_states(np.zeros(500, dtype=np.int64), 20)


# --- realism ---------------------------------------------------------------------------------


def test_window_features_match_a_direct_computation() -> None:
    r = np.random.default_rng(7).standard_normal((21 * 6, 3)) * 0.01
    f = window_features(r)
    assert f.shape == (4, 5)
    w = r[21 * 5 : 21 * 6]
    ew = w.mean(axis=1)
    c = np.corrcoef(w.T)
    assert f[-1, 0] == pytest.approx(math.log(ew.std(ddof=1)))
    assert f[-1, 2] == pytest.approx(math.log(r[21 * 3 : 21 * 4].mean(axis=1).std(ddof=1)))
    assert f[-1, 3] == pytest.approx((c.sum() - 3) / 6)
    assert f[-1, 4] == pytest.approx(ew.sum())


def test_mmd_is_near_zero_for_one_distribution_and_positive_for_a_shift() -> None:
    gen = np.random.default_rng(8)
    x, y = gen.standard_normal((400, 2)), gen.standard_normal((400, 2))
    same = mmd2_unbiased(x, y, 1.0)
    shifted = mmd2_unbiased(x, y + 1.0, 1.0)
    assert abs(same) < 0.01 < shifted
    assert mmd2_unbiased(x, y, 1.0) == pytest.approx(mmd2_unbiased(y, x, 1.0))


def test_null_threshold_is_the_rank_order_statistic() -> None:
    assert null_threshold(np.arange(199.0), 0.05) == 189.0  # the 190th of 199
    with pytest.raises(ValueError, match="cannot give"):
        null_threshold(np.arange(5.0), 0.05)


def test_realism_rejects_a_clearly_different_volatility_mix() -> None:
    gen = np.random.default_rng(9)
    obs_state = _states(4000, 10)
    returns = gen.standard_normal((4000, 4)) * np.where(obs_state == 1, 0.03, 0.01)[:, None]
    ref = RealismReference.from_returns(returns)
    pool = BlockPool.from_states(obs_state, 20)
    hat = pool.chain_hat
    null = [
        ref.statistic(
            returns[pool.generate(hat, Start.STATIONARY, 16000, np.random.default_rng(100 + b))]
        )
        for b in range(39)
    ]
    thr = null_threshold(np.array(null), 0.05)
    far = StateChain(share=0.8, duration=10.0)
    stat = ref.statistic(
        returns[pool.generate(far, Start.STATIONARY, 16000, np.random.default_rng(1))]
    )
    assert stat > thr


# --- objective ------------------------------------------------------------------------------


def test_horizon_weights() -> None:
    c = StateChain(share=0.3, duration=20.0)
    np.testing.assert_allclose(horizon_weights(c, Start.STATIONARY, 252), c.stationary)
    np.testing.assert_allclose(horizon_weights(c, Start.HIGH, 1), c.transition[1])
    w = horizon_weights(c, Start.HIGH, 252)
    assert w.sum() == pytest.approx(1.0) and w[1] > c.share
    # Blocks of 100 days over 252: slots of 100, 100 and 52 days.
    p = c.transition
    row = np.array([0.0, 1.0])
    expect = (100 * (row @ p) + 100 * (row @ p @ p) + 52 * (row @ p @ p @ p)) / 252
    np.testing.assert_allclose(horizon_weights(c, Start.HIGH, 252, 100), expect)


@settings(max_examples=50, deadline=None)
@given(
    mu=st.floats(-0.01, 0.01),
    sd=st.floats(1e-4, 0.05),
    w=st.floats(0.0, 1.0),
)
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


def test_sr_h_matches_a_simulated_mixture() -> None:
    c = StateChain(share=0.4, duration=30.0)
    mu, sd = np.array([0.001, -0.0005]), np.array([0.01, 0.02])
    gen = np.random.default_rng(11)
    pooled = []
    for _ in range(400):
        z = state_path(c, Start.HIGH, 252, gen)
        pooled.append(mu[z] + sd[z] * gen.standard_normal(252))
    x = np.concatenate(pooled)
    sim = x.mean() / x.std() * math.sqrt(252)
    assert sr_h(mu, sd, c, Start.HIGH, 252) == pytest.approx(sim, abs=0.15)


# --- certification --------------------------------------------------------------------------


def _market(seed: int, high_mean: float) -> tuple[np.ndarray, ...]:
    gen = np.random.default_rng(seed)
    n = 3000
    states = _states(n, seed + 1)
    vol = np.where(states == 1, 0.02, 0.01)
    returns = gen.standard_normal((n, 4)) * vol[:, None] + 0.0003
    pnl = np.where(states == 1, high_mean, 0.002) + 0.01 * gen.standard_normal(n)
    return pnl, returns.mean(axis=1), returns, states


def test_certify_kills_a_strategy_that_loses_in_high_volatility() -> None:
    # Loses enough in high volatility that even theta_hat (share ~.3) is net negative.
    pnl, ew, returns, states = _market(20, -0.01)
    res = certify(pnl, ew, returns, states, SMALL, RngFactory(1))
    assert res.kill and res.sr_h_star_upper < 0
    assert res.worst_start is Start.HIGH
    assert res.sr_h_star <= res.sr_h_baseline


def test_certify_passes_a_strategy_that_earns_in_both_states() -> None:
    pnl, ew, returns, states = _market(21, 0.002)
    res = certify(pnl, ew, returns, states, SMALL, RngFactory(2))
    assert not res.kill and res.sr_h_star_ci[0] > 0
    assert res.n_plausible >= 1 and len(res.path_sharpe_quantiles) == 5


def test_certify_is_deterministic_and_checks_alignment() -> None:
    pnl, ew, returns, states = _market(22, -0.001)
    assert certify(pnl, ew, returns, states, SMALL, RngFactory(3)) == certify(
        pnl, ew, returns, states, SMALL, RngFactory(3)
    )
    with pytest.raises(ValueError, match="aligned"):
        certify(pnl[:-1], ew, returns, states, SMALL, RngFactory(4))
