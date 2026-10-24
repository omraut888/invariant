from __future__ import annotations

import numpy as np
import pytest

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.diagnostics import CovType, ols
from invariant.benchmark.synthetic import STATE_NODES, SyntheticDataset, generate_market
from invariant.stages.dro.oracle import projection_oracle

OBSERVED = ["z", "s_real", "s_conf", "s_decay", "s_noise"]


@pytest.fixture(scope="module")
def market() -> SyntheticDataset:
    cfg = load_benchmark_config("multi_asset").model_copy(update={"n_obs": 20_000})
    return generate_market(cfg, 11)


def test_full_information_mean_matches_an_independent_reconstruction(
    market: SyntheticDataset,
) -> None:
    """With every node observed, the projection is the SCM's own mean term (exact)."""
    truth, f = market.ground_truth, market.features
    full = projection_oracle(truth, f, OBSERVED)
    load = truth.loadings
    m = market.latents["decay_multiplier"].to_numpy()
    manual = (
        truth.config.mu
        + f[["z"]].to_numpy() * np.array(load.beta_z)
        + f[["s_real"]].to_numpy() * np.array(load.beta_real)
        + (f["s_decay"].to_numpy() * m)[:, None] * np.array(load.beta_decay_initial)
    )
    np.testing.assert_allclose(full.mean, manual, rtol=1e-12, atol=1e-15)
    b = np.array(load.market_beta)
    cov = np.outer(b, b) * truth.config.market.garch.sigma_bar**2 + np.eye(len(b)) * (  # type: ignore[union-attr]
        truth.config.garch.sigma_bar**2
    )
    np.testing.assert_allclose(full.cov[0], cov, rtol=1e-12)


def test_full_information_residual_is_market_plus_garch(market: SyntheticDataset) -> None:
    truth, f = market.ground_truth, market.features
    full = projection_oracle(truth, f, OBSERVED)
    r = market.returns.to_numpy()
    b = np.array(truth.loadings.market_beta)
    resid = r[1:] - full.mean[:-1] - np.outer(market.latents["market"].to_numpy()[1:], b)
    z = resid / np.sqrt(market.latents[[f"h_{a}" for a in market.returns.columns]].to_numpy()[1:])
    assert np.abs(z.mean(axis=0)).max() < 0.05  # standardized GARCH-t innovations
    np.testing.assert_allclose(z.var(axis=0), 1.0, atol=0.08)
    corr = np.corrcoef(np.c_[resid, f.to_numpy()[:-1]].T)[: resid.shape[1], resid.shape[1] :]
    assert np.abs(corr).max() < 0.04  # 20k rows: ~3 SE


def test_partial_information_slopes_match_ols(market: SyntheticDataset) -> None:
    """On a window after s_decay has fully decayed, OLS of R_{t+1} on [1, s_real, s_decay]
    recovers the oracle's projection slopes within 4 HAC standard errors.

    HAC with a long bandwidth: the projection residual holds the omitted, persistent Z
    (phi .99), so the naive OLS standard error is about 3x too small (ADR 0003)."""
    truth, f = market.ground_truth, market.features
    o1 = projection_oracle(truth, f, ["s_real", "s_decay"])
    h = slice(16_000, 19_999)  # onset 12,000, H 250: m(t) <= 2**-16 here
    x = f[["s_real", "s_decay"]].to_numpy()[h]
    r = market.returns.to_numpy()[h.start + 1 : h.stop + 1]
    orc, *_ = np.linalg.lstsq(np.c_[np.ones(len(x)), x], o1.mean[h], rcond=None)
    for i in range(r.shape[1]):
        res = ols(r[:, i], x, cov=CovType.HAC, hac_lags=200)
        assert abs(res.coef[1] - orc[1, i]) < 4 * res.se[1]
    np.testing.assert_allclose(orc[0], truth.config.mu, atol=1e-9)


def test_oracle_validates_information_set(market: SyntheticDataset) -> None:
    with pytest.raises(ValueError):
        projection_oracle(market.ground_truth, market.features, [])
    with pytest.raises(ValueError):
        projection_oracle(market.ground_truth, market.features, ["nope"])
    assert tuple(market.ground_truth.state_order) == STATE_NODES
