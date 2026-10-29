"""Statistical validation of the planted-signal benchmark against its own ground truth.

Every tolerance below is derived from an analytic or Monte Carlo standard error with a margin
of >= 4 SE (documented inline), and every test uses fixed seeds, so these are deterministic.
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.diagnostics import CovType, ols
from invariant.benchmark.ground_truth import GroundTruth
from invariant.benchmark.stylized_facts import check_stylized_facts
from invariant.benchmark.synthetic import (
    STATE_NODES,
    Innovations,
    SimulatedPaths,
    SyntheticDataset,
    asset_names,
    build_ground_truth,
    decay_multiplier,
    draw_innovations,
    draw_loadings,
    generate_market,
    population_moments,
    simulate,
    state_nodes,
    stationary_covariance,
)
from invariant.core.run_hash import RngFactory

pytestmark = pytest.mark.benchmark

PARENTS = ["z", "s_real", "s_decay"]


def _predictive_regression(
    d: SyntheticDataset,
    regressors: list[str],
    *,
    n_return_lags: int = 5,
    asset: str = "asset_00",
    rows: slice = slice(None),
) -> tuple[np.ndarray, np.ndarray]:
    """``y_j = R_{j+1}``, ``X_j = [features_j, R_j, ..., R_{j-p+1}]`` (timing convention).

    ``rows`` slices the signal dates ``j`` that have a next-day return and ``p`` return lags.
    """
    f = d.features[regressors].to_numpy()
    r = d.returns[asset].to_numpy()
    j = np.arange(max(n_return_lags - 1, 0), len(r) - 1)
    x = np.column_stack([f[j]] + [r[j - lag] for lag in range(n_return_lags)])
    return r[j + 1][rows], x[rows]


# --- the generation graph IS the ground-truth graph -------------------------------------------


def _node_values(paths: SimulatedPaths, cfg: SyntheticMarketConfig) -> dict[str, np.ndarray]:
    values = {name: paths.state[:, k] for k, name in enumerate(state_nodes(cfg))}
    for i, a in enumerate(asset_names(cfg.n_assets)):
        values[a] = paths.returns[:, i]
    if cfg.market is not None:
        values["market"] = paths.market
    return values


def _shock(innov: Innovations, cfg: SyntheticMarketConfig, node: str, s: int) -> Innovations:
    if node == "s_regime":
        assert innov.s_regime is not None
        s_reg = innov.s_regime.copy()
        s_reg[s] += 1.0
        return replace(innov, s_regime=s_reg)
    if node == "s_volfrag":
        assert innov.s_volfrag is not None
        s_vf = innov.s_volfrag.copy()
        s_vf[s] += 1.0
        return replace(innov, s_volfrag=s_vf)
    if node in STATE_NODES:
        state = innov.state.copy()
        state[s, STATE_NODES.index(node)] += 1.0
        return replace(innov, state=state)
    if node == "market":
        assert innov.eps_market is not None
        eps_m = innov.eps_market.copy()
        eps_m[s] += 1.0
        return replace(innov, eps_market=eps_m)
    eps = innov.eps.copy()
    eps[s, asset_names(cfg.n_assets).index(node)] += 1.0
    return replace(innov, eps=eps)


def _children(gt: GroundTruth, nodes: set[str], lag: int) -> set[str]:
    return {e.target for e in gt.edges if e.lag == lag and e.source in nodes}


def _lag0_closure(gt: GroundTruth, nodes: set[str]) -> set[str]:
    out = set(nodes)
    while True:
        grown = out | _children(gt, out, 0)
        if grown == out:
            return out
        out = grown


@pytest.mark.parametrize("preset", [None, "multi_asset", "regime", "volfrag"])
def test_intervention_support_matches_ground_truth_graph(preset: str | None) -> None:
    """Shock each node's innovation at step s; what moves at s and s+1 must be what GT predicts.

    This tests the simulator against the declared graph (not the graph against itself): with
    all edges at lag <= 1, the nodes changed at s are the lag-0 closure of the shocked node, and
    the nodes changed at s+1 are the lag-0 closure of the lag-1 children of those.
    """
    cfg = load_benchmark_config(preset).model_copy(update={"n_obs": 300, "burn_in": 20})
    rng = RngFactory(99)
    loadings = draw_loadings(cfg, rng)
    innov = draw_innovations(cfg, rng)
    dates = pd.bdate_range(cfg.start_date, periods=cfg.n_obs)
    gt = build_ground_truth(cfg, 99, loadings, dates, "0" * 64)
    base = _node_values(simulate(cfg, innov, loadings), cfg)
    s = cfg.burn_in + 50  # pre-decay
    shockable = [n.name for n in gt.nodes if n.name in base]
    assert set(shockable) == set(base)
    for node in shockable:
        moved = _node_values(simulate(cfg, _shock(innov, cfg, node, s), loadings), cfg)
        changed = {
            lag: {n for n in base if abs(moved[n][s + lag] - base[n][s + lag]) > 1e-15}
            for lag in (0, 1)
        }
        at_s = _lag0_closure(gt, {node})
        at_s1 = _lag0_closure(gt, _children(gt, at_s, 1))
        assert changed[0] == at_s, f"shock to {node}: lag-0 support"
        assert changed[1] == at_s1, f"shock to {node}: lag-1 support"


# --- population moments are what the data are drawn from --------------------------------------


def test_sample_moments_match_population_moments_on_a_long_path() -> None:
    cfg = load_benchmark_config().model_copy(update={"n_obs": 200_000})
    rng = RngFactory(2024)
    loadings = draw_loadings(cfg, rng)
    paths = simulate(cfg, draw_innovations(cfg, rng), loadings)
    x = paths.state[cfg.burn_in :]
    # Z mixes slowest (phi=.99): effective n for a variance ~ n(1-phi^2)/(1+phi^2) ~ 2000, so
    # SE(cov entry) <~ sqrt(2/2000) ~ 0.032. Tolerance 0.15 ~ 4.7 SE.
    np.testing.assert_allclose(np.cov(x, rowvar=False), stationary_covariance(cfg), atol=0.15)

    # Marginal ICs, pre-decay segment only (coefficients at their initial values).
    moments = population_moments(cfg, loadings)[0]
    t0 = cfg.burn_in + cfg.decay_start_index
    r_next = paths.returns[cfg.burn_in + 1 : t0, 0]
    x_pre = paths.state[cfg.burn_in : t0 - 1]
    n = len(r_next)
    # R is ~white noise, so SE(corr) ~ 1/sqrt(n) inflated by GARCH (x1.5 margin): ~0.0043.
    tol = 4 * 1.5 / math.sqrt(n)
    for k, name in enumerate(STATE_NODES):
        sample = np.corrcoef(x_pre[:, k], r_next)[0, 1]
        assert sample == pytest.approx(moments.marginal_ic[k], abs=tol), name


# --- S_decay: the effective coefficient halves at the stated half-life ------------------------


def test_decay_coefficient_halves_at_half_life_in_rolling_regression() -> None:
    """High-SNR config: at IC 0.06 the relative SE of a 500-day slope is ~0.9 (uninformative)."""
    base = load_benchmark_config()
    cfg = base.model_copy(
        update={
            "decay": base.decay.model_copy(
                update={"ic": 0.5, "half_life": 1000.0, "start_frac": 0.3},
            ),
        },
    )
    t0, h, w = cfg.decay_start_index, 1000, 500
    windows = {
        "pre": slice(t0 - w, t0),
        "one": slice(t0 + h - w // 2, t0 + h + w // 2),
        "two": slice(t0 + 2 * h - w // 2, t0 + 2 * h + w // 2),
    }
    mult = decay_multiplier(cfg)[cfg.burn_in :]
    expected = {
        k: float(mult[windows[k]].mean() / mult[windows["pre"]].mean()) for k in ("one", "two")
    }
    assert expected["one"] == pytest.approx(0.5, abs=0.01)  # window-averaging effect is tiny
    ratios: dict[str, list[float]] = {"one": [], "two": []}
    for seed in range(20):
        d = generate_market(cfg, seed)
        slopes = {}
        for name, rows in windows.items():
            y, x = _predictive_regression(d, ["s_decay", "z", "s_real"], n_return_lags=0, rows=rows)
            slopes[name] = ols(y, x).coef[1]
        for k in ratios:
            ratios[k].append(slopes[k] / slopes["pre"])
    for k, values in ratios.items():
        arr = np.array(values)
        se = arr.std(ddof=1) / math.sqrt(len(arr))
        assert se < 0.04, f"{k}: test has no power (SE {se:.3f})"
        assert arr.mean() == pytest.approx(expected[k], abs=4 * se), (k, arr.mean(), se)


# --- stylized facts ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_stylized_facts_hold_single_asset(seed: int) -> None:
    d = generate_market(load_benchmark_config(), seed)
    report = check_stylized_facts(d.returns["asset_00"])
    assert report.passed, report.checks


def test_stylized_facts_hold_multi_asset() -> None:
    d = generate_market(load_benchmark_config("multi_asset"), 0)
    for asset in d.returns.columns:
        report = check_stylized_facts(d.returns[asset])
        assert report.passed, (asset, report.checks)


# --- S_conf is a real trap, and conditioning on the parents disarms it ------------------------


def _trap_cfg(preset: str | None = None) -> SyntheticMarketConfig:
    return load_benchmark_config(preset).model_copy(update={"n_obs": 20_000})


@pytest.mark.parametrize("seed", range(5))
def test_s_conf_marginally_significant_but_partially_null(seed: int) -> None:
    d = generate_market(_trap_cfg(), seed)
    expected_t = d.ground_truth.signal("s_conf").marginal_ic[0] * math.sqrt(20_000)
    assert expected_t > 8  # population trap strength at this n

    y, x = _predictive_regression(d, ["s_conf"], n_return_lags=0)
    marginal_t = ols(y, x, cov=CovType.HAC).tstat[1]
    assert marginal_t > 4.0, "S_conf must be significant marginally (it is a trap)"

    y, x = _predictive_regression(d, ["s_conf", *PARENTS], n_return_lags=5)
    partial_t = ols(y, x, cov=CovType.HAC).tstat[1]
    # Under the true null |t| > 3 has probability ~0.27% per seed.
    assert abs(partial_t) < 3.0, f"partial t {partial_t:.2f} given Z, S_real, S_decay, R lags"


def test_hidden_confounder_leaves_s_conf_significant() -> None:
    """Documents the limitation: with Z hidden no observed adjustment set removes the bias."""
    partial_corrs = []
    for seed in range(5):
        d = generate_market(_trap_cfg("hidden_confounder"), seed)
        y, x = _predictive_regression(d, ["s_conf", "s_real", "s_decay"], n_return_lags=5)
        res = ols(y, x, cov=CovType.HAC)
        t = res.tstat[1]
        assert t > 3.0, "with Z hidden, S_conf survives a correct CI test"
        partial_corrs.append(t / math.sqrt(t**2 + res.nobs - x.shape[1] - 1))
    gt_bias = d.ground_truth.signal("s_conf").partial_ic[0]
    # SE of a sample partial corr ~ 1/sqrt(n) (x1.5 GARCH margin) ~ 0.011; mean over 5 ~ 0.005.
    assert float(np.mean(partial_corrs)) == pytest.approx(gt_bias, abs=0.02)
