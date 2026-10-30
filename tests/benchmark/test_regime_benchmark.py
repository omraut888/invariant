"""Validation of benchmark v4 (ADR 0015): the regime chain, the Markov-switching-intercept GARCH,
and S_regime, against their own ground truth.

The chain and ``h`` are strongly autocorrelated, so every tolerance is 4 batch-means standard
errors (100 contiguous batches) computed in the test, not an i.i.d. SE. Fixed seeds throughout.
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError
from tests.fixtures.paired_facts import failed_checks, mcnemar_worse

from invariant.benchmark.config import (
    GarchParams,
    RegimeParams,
    RegimeSignalParams,
    load_benchmark_config,
)
from invariant.benchmark.ground_truth import GroundTruth
from invariant.benchmark.stylized_facts import check_stylized_facts
from invariant.benchmark.synthetic import (
    REGIME_SIGNAL,
    _ms_garch_path,
    generate_market,
    population_return_acf,
    regime_conditional_variance,
    regime_path,
)
from invariant.core.types import StageName, Verdict

pytestmark = pytest.mark.benchmark

N_BATCH = 100


def _batch_se(values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """(ratio estimate, batch-means SE) of ``sum(w v) / sum(w)`` over contiguous batches."""
    w = np.ones_like(values) if weights is None else weights
    num = np.array([b.sum() for b in np.array_split(w * values, N_BATCH)])
    den = np.array([b.sum() for b in np.array_split(w, N_BATCH)])
    est = num.sum() / den.sum()
    per = num / den
    return float(est), float(per.std(ddof=1) / math.sqrt(N_BATCH))


def _regime(**updates: object) -> RegimeParams:
    base = {
        "stress_share": 0.2,
        "stress_duration": 120.0,
        "resid_vol_ratio": 2.0,
        "market_vol_ratio": 2.5,
        "signal": RegimeSignalParams(rho=0.9, ic=0.1, stress_ratio=-1.5),
    }
    return RegimeParams.model_validate({**base, **updates})


# --- the chain --------------------------------------------------------------------------------


def test_chain_is_stochastic_and_pi_is_stationary() -> None:
    r = _regime()
    p = np.array(r.transition_matrix)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    pi = np.array(r.stationary)
    np.testing.assert_allclose(pi @ p, pi, atol=1e-15)
    assert r.calm_duration == pytest.approx(480.0)  # d_s (1 - pi_s) / pi_s


def test_chain_without_a_valid_calm_exit_is_rejected() -> None:
    # pi_s = .9 with d_s = 2 needs P(calm -> stress) = 4.5.
    with pytest.raises(ValidationError, match="no such chain"):
        _regime(stress_share=0.9, stress_duration=2.0)


def test_regime_path_occupancy_and_spell_lengths() -> None:
    r = _regime()
    xi = regime_path(r, np.random.default_rng(5).random(2_000_000))
    share, se = _batch_se(xi.astype(float))
    assert abs(share - r.stress_share) < 4 * se
    starts = np.flatnonzero(np.diff(xi) == 1) + 1
    ends = np.flatnonzero(np.diff(xi) == -1) + 1
    ends = ends[ends > starts[0]][: len(starts) - 1]
    spells = ends - starts[: len(ends)]
    # Geometric spells: sd = sqrt(d (d - 1)); spells are independent given the chain.
    d = r.stress_duration
    assert abs(spells.mean() - d) < 4 * math.sqrt(d * (d - 1)) / math.sqrt(len(spells))


# --- Markov-switching-intercept GARCH ---------------------------------------------------------


def test_conditional_variance_by_regime_matches_simulation() -> None:
    r = _regime()
    garch = GarchParams(nu=6, alpha=0.06, beta=0.92, sigma_bar=0.01)
    n = 2_000_000
    xi = regime_path(r, np.random.default_rng(11).random(n))
    eps = np.random.default_rng(12).standard_normal(n)
    h, _ = _ms_garch_path(eps, garch, np.where(xi == 1, r.resid_vol_ratio, 1.0))
    exact = regime_conditional_variance(r, garch, r.resid_vol_ratio)
    for k in (0, 1):
        est, se = _batch_se(h, (xi == k).astype(float))
        assert abs(est - exact[k]) < 4 * se, (k, est, exact[k], se)
    # pi-weighted, they are the unconditional mixture variance.
    pi_c, pi_s = r.stationary
    mixture = garch.sigma_bar**2 * (pi_c + pi_s * r.resid_vol_ratio**2)
    assert pi_c * exact[0] + pi_s * exact[1] == pytest.approx(mixture, rel=1e-12)
    # h remembers the previous regime: E[h | stress] lies below the long-run stress level.
    assert exact[0] > garch.sigma_bar**2
    assert exact[1] < (r.resid_vol_ratio * garch.sigma_bar) ** 2


# --- the planted signal against GroundTruth ---------------------------------------------------


@pytest.fixture(scope="module")
def long_regime() -> tuple[np.ndarray, np.ndarray, np.ndarray, GroundTruth]:
    cfg = load_benchmark_config("regime").model_copy(update={"n_obs": 400_000})
    d = generate_market(cfg, 21)
    return (
        d.features[REGIME_SIGNAL].to_numpy()[:-1],
        d.returns["asset_00"].to_numpy()[1:],
        d.latents["regime"].to_numpy()[1:],
        d.ground_truth,
    )


def _corr_and_se(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Correlation within ``w`` and its batch-means SE (per-batch correlations)."""
    mask = np.ones_like(x, dtype=bool) if w is None else w.astype(bool)
    xs, ys = x[mask], y[mask]
    est = float(np.corrcoef(xs, ys)[0, 1])
    per = [
        float(np.corrcoef(a, b)[0, 1])
        for a, b in zip(np.array_split(xs, N_BATCH), np.array_split(ys, N_BATCH), strict=True)
    ]
    return est, float(np.std(per, ddof=1) / math.sqrt(N_BATCH))


@pytest.mark.slow  # 400k-day path
def test_s_regime_marginal_ic_matches_ground_truth(long_regime: tuple) -> None:  # type: ignore[type-arg]
    s, r, _, gt = long_regime
    est, se = _corr_and_se(s, r)
    assert abs(est - gt.signal(REGIME_SIGNAL).marginal_ic[0]) < 4 * se


@pytest.mark.slow  # 400k-day path
def test_per_regime_ics_match_regime_truth(long_regime: tuple) -> None:  # type: ignore[type-arg]
    s, r, xi, gt = long_regime
    assert gt.regime is not None
    truth = {ic.signal_id: ic for ic in gt.regime.regime_ic}[REGIME_SIGNAL]
    for k, expected in ((0, truth.calm[0]), (1, truth.stress[0])):
        est, se = _corr_and_se(s, r, xi == k)
        assert abs(est - expected) < 4 * se, (k, est, expected, se)
    assert truth.calm[0] > 0 > truth.stress[0]  # the planted reversal


def test_regime_ground_truth_verdicts_and_round_trip() -> None:
    gt = generate_market(
        load_benchmark_config("regime").model_copy(update={"n_obs": 500}), 1
    ).ground_truth
    s = gt.signal(REGIME_SIGNAL)
    assert s.has_causal_edge
    assert (s.expected_verdict, s.expected_kill_stage) == (Verdict.KILL, StageName.ADVERSARIAL)
    assert gt.signal("s_real").expected_verdict is Verdict.PASS
    assert gt.state_order[-1] == REGIME_SIGNAL and len(gt.stationary_covariance) == 6
    assert GroundTruth.model_validate_json(gt.model_dump_json()) == gt
    null = generate_market(
        load_benchmark_config("regime_null").model_copy(update={"n_obs": 500}), 1
    ).ground_truth
    assert {x.expected_kill_stage for x in null.signals} == {StageName.CAUSAL}


def test_regime_streams_leave_shared_components_unchanged() -> None:
    # Common random numbers: changing the reversal moves returns only, never the state or chain.
    cfg = load_benchmark_config("regime").model_copy(update={"n_obs": 800})
    assert cfg.regime is not None
    other = cfg.model_copy(
        update={
            "regime": cfg.regime.model_copy(
                update={"signal": cfg.regime.signal.model_copy(update={"stress_ratio": 0.5})}
            )
        }
    )
    a, b = generate_market(cfg, 4), generate_market(other, 4)
    np.testing.assert_array_equal(a.features.to_numpy(), b.features.to_numpy())
    np.testing.assert_array_equal(a.latents.to_numpy(), b.latents.to_numpy())
    assert not np.allclose(a.returns.to_numpy(), b.returns.to_numpy())


@pytest.mark.slow  # 400k-day paths
@pytest.mark.parametrize("preset", ["multi_asset", "regime"])
def test_population_return_acf_matches_simulation(preset: str) -> None:
    cfg = load_benchmark_config(preset).model_copy(update={"n_obs": 400_000})
    d = generate_market(cfg, 8)
    r = d.returns["asset_00"].to_numpy()
    pop = population_return_acf(cfg, d.ground_truth.loadings, 0, 3)
    for k in (1, 2, 3):
        est, se = _corr_and_se(r[k:], r[:-k])
        assert abs(est - pop[k - 1]) < 4 * se, (k, est, pop[k - 1], se)


def test_stylized_facts_hold_on_every_regime_asset_seed_0() -> None:
    # The claim ADR 0003 makes for multi_asset (seed 0, every asset), for the regime preset.
    d = generate_market(load_benchmark_config("regime"), 0)
    for asset in d.returns.columns:
        report = check_stylized_facts(d.returns[asset])
        assert report.passed, (asset, [c for c in report.checks if not c.passed])


def test_regime_preset_is_not_significantly_worse_than_multi_asset() -> None:
    # Exact one-sided McNemar on the paired (seed, asset, check) outcomes, alpha = .05
    # (tests/fixtures/paired_facts.py). ADR 0015 records the counts.
    ours, ref = failed_checks("regime"), failed_checks("multi_asset")
    only_ours, only_ref, p = mcnemar_worse(ours, ref)
    assert p >= 0.05, (only_ours, only_ref, p)
