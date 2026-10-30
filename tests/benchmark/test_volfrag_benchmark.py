"""Validation of benchmark v4.1 (ADR 0015 amendment): s_volfrag against its own ground truth.

Tolerances are 4 batch-means SEs computed in the test (h_M is persistent). Fixed seeds.
"""

import numpy as np
import pytest
from tests.fixtures.paired_facts import failed_checks, mcnemar_worse

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.ground_truth import GroundTruth
from invariant.benchmark.stylized_facts import check_stylized_facts
from invariant.benchmark.synthetic import (
    VOLFRAG_SIGNAL,
    generate_market,
    population_return_acf,
    volfrag_threshold,
)
from invariant.core.types import StageName, Verdict

pytestmark = pytest.mark.benchmark

N_BATCH = 100


def _corr_and_se(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float]:
    m = np.ones_like(x, dtype=bool) if mask is None else mask
    xs, ys = x[m], y[m]
    per = [
        float(np.corrcoef(a, b)[0, 1])
        for a, b in zip(np.array_split(xs, N_BATCH), np.array_split(ys, N_BATCH), strict=True)
    ]
    return float(np.corrcoef(xs, ys)[0, 1]), float(np.std(per, ddof=1) / np.sqrt(N_BATCH))


@pytest.fixture(scope="module")
def long_volfrag() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GroundTruth]:
    cfg = load_benchmark_config("volfrag").model_copy(update={"n_obs": 400_000})
    d = generate_market(cfg, 23)
    return (
        d.features[VOLFRAG_SIGNAL].to_numpy()[:-1],
        d.returns.to_numpy()[1:],
        d.latents["h_market"].to_numpy()[1:],
        d.returns["asset_00"].to_numpy(),
        d.ground_truth,
    )


@pytest.mark.slow  # 400k-day path
def test_tau_is_the_population_quantile(long_volfrag: tuple) -> None:  # type: ignore[type-arg]
    _, _, h, _, gt = long_volfrag
    assert gt.volfrag is not None
    share, se = (h > gt.volfrag.tau).mean(), None
    per = [float((b > gt.volfrag.tau).mean()) for b in np.array_split(h, N_BATCH)]
    se = float(np.std(per, ddof=1) / np.sqrt(N_BATCH))
    assert abs(share - (1 - gt.volfrag.h_quantile)) < 4 * se


@pytest.mark.slow  # 400k-day path
def test_marginal_and_per_state_ics_match_ground_truth(long_volfrag: tuple) -> None:  # type: ignore[type-arg]
    s, r, h, _, gt = long_volfrag
    assert gt.volfrag is not None
    est, se = _corr_and_se(s, r[:, 0])
    assert abs(est - gt.signal(VOLFRAG_SIGNAL).marginal_ic[0]) < 4 * se
    truth = {ic.signal_id: ic for ic in gt.volfrag.state_ic}
    high = h > gt.volfrag.tau
    for sid, x in ((VOLFRAG_SIGNAL, s),):
        for mask, expected in ((~high, truth[sid].calm[0]), (high, truth[sid].stress[0])):
            est, se = _corr_and_se(x, r[:, 0], mask)
            assert abs(est - expected) < 4 * se, (sid, est, expected, se)
    assert truth[VOLFRAG_SIGNAL].calm[0] > 0 > truth[VOLFRAG_SIGNAL].stress[0]


@pytest.mark.slow  # 400k-day path
def test_population_return_acf_matches_simulation(long_volfrag: tuple) -> None:  # type: ignore[type-arg]
    _, _, _, r0, gt = long_volfrag
    cfg = gt.config
    pop = population_return_acf(cfg, gt.loadings, 0, 3)
    for k in (1, 2, 3):
        est, se = _corr_and_se(r0[k:], r0[:-k])
        assert abs(est - pop[k - 1]) < 4 * se, (k, est, pop[k - 1], se)


def test_threshold_is_cached_and_deterministic() -> None:
    cfg = load_benchmark_config("volfrag")
    assert cfg.market is not None and cfg.volfrag is not None
    a = volfrag_threshold(cfg.market.garch, cfg.volfrag.h_quantile)
    assert a == volfrag_threshold(cfg.market.garch, cfg.volfrag.h_quantile)
    assert a[1] < a[0] < a[2]  # E[h | low] < tau < E[h | high]


def test_volfrag_ground_truth() -> None:
    gt = generate_market(
        load_benchmark_config("volfrag").model_copy(update={"n_obs": 500}), 1
    ).ground_truth
    s = gt.signal(VOLFRAG_SIGNAL)
    assert s.has_causal_edge and s.identifiable
    assert (s.expected_verdict, s.expected_kill_stage) == (Verdict.KILL, StageName.ADVERSARIAL)
    assert GroundTruth.model_validate_json(gt.model_dump_json()) == gt
    regime = generate_market(
        load_benchmark_config("regime").model_copy(update={"n_obs": 500}), 1
    ).ground_truth
    assert not regime.signal("s_regime").identifiable  # ADR 0016: reported-only


def test_volfrag_and_regime_are_exclusive() -> None:
    cfg = load_benchmark_config("volfrag")
    reg = load_benchmark_config("regime").regime
    with pytest.raises(ValueError, match="mutually exclusive"):
        type(cfg).model_validate({**cfg.model_dump(), "regime": reg})


def test_stylized_facts_hold_on_every_volfrag_asset_seed_0() -> None:
    d = generate_market(load_benchmark_config("volfrag"), 0)
    for asset in d.returns.columns:
        report = check_stylized_facts(d.returns[asset])
        assert report.passed, (asset, [c for c in report.checks if not c.passed])


def test_volfrag_preset_is_not_significantly_worse_than_multi_asset() -> None:
    # Exact one-sided McNemar on the paired (seed, asset, check) outcomes, alpha = .05
    # (tests/fixtures/paired_facts.py). ADR 0015 records the counts.
    ours, ref = failed_checks("volfrag"), failed_checks("multi_asset")
    only_ours, only_ref, p = mcnemar_worse(ours, ref)
    assert p >= 0.05, (only_ours, only_ref, p)
