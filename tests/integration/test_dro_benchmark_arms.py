"""Every Stage 3 study arm, the two oracles and every reported metric, end to end on one tiny
multi_asset benchmark seed. So the pilot cannot be the first thing to run a code path
(CLAUDE.md, "Method choices")."""

from __future__ import annotations

import math

import numpy as np

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.core.run_hash import RngFactory
from invariant.stages.dro.metrics import (
    deflated_sharpe_ratio,
    exposure_shares,
    sharpe,
    sharpe_difference_test,
    turnover,
)
from invariant.stages.dro.oracle import projection_oracle
from invariant.stages.dro.walkforward import Arm, WalkForwardConfig, run_walk_forward, solve_path

SIGNALS = ["s_real", "s_decay"]
OBSERVED = ["z", "s_real", "s_conf", "s_decay", "s_noise"]


def test_all_arms_oracles_and_metrics_on_a_tiny_seed() -> None:
    cfg_m = load_benchmark_config("multi_asset").model_copy(update={"n_obs": 700})
    ds = generate_market(cfg_m, 5)
    x = ds.features[SIGNALS].to_numpy()
    r = ds.returns.to_numpy()
    cfg = WalkForwardConfig(
        c=0.5 / math.sqrt(252),
        window=300,
        refit_every=21,
        cv_every=200,
        cv_valid=100,
        ridge_grid=(0.0, 0.3),
        eps_grid=(0.0, 0.05),
        fc_draws=1000,
    )
    paths = run_walk_forward(x, r, list(Arm), cfg, RngFactory(1))
    dates = paths[Arm.MARKOWITZ].dates
    o1 = projection_oracle(ds.ground_truth, ds.features, SIGNALS)
    o2 = projection_oracle(ds.ground_truth, ds.features, OBSERVED)
    _, r_o1 = solve_path(o1.mean, o1.cov, r, dates, cfg.c)
    _, r_o2 = solve_path(o2.mean, o2.cov, r, dates, cfg.c)
    rng = RngFactory(2).generator("lw")
    sharpes = []
    for arm, p in paths.items():
        assert p.weights.shape == (len(dates), r.shape[1])
        assert np.isfinite(p.returns).all()
        shares = exposure_shares(p.weights)
        assert math.isclose(sum(shares.values()), 1.0)
        assert (turnover(p.weights) >= 0).all()
        s = sharpe(p.returns)
        sharpes.append(s if not math.isnan(s) else 0.0)
        if arm is not Arm.EQUAL_WEIGHT:
            _, pval = sharpe_difference_test(
                p.returns, paths[Arm.EQUAL_WEIGHT].returns, rng, n_boot=99
            )
            assert 0.0 < pval <= 1.0
    regret = sharpe(r_o1) - sharpe(paths[Arm.MARKOWITZ].returns)
    assert math.isfinite(regret) and math.isfinite(sharpe(r_o2))
    var = float(np.var(np.array(sharpes) / math.sqrt(252)))
    dsr = deflated_sharpe_ratio(r_o1, n_trials=len(Arm), trial_sharpe_variance=var)
    assert 0.0 <= dsr <= 1.0
    fc = paths[Arm.DRO_FC]
    assert fc.mu_hat is not None and fc.radius is not None
    err = np.linalg.norm(fc.mu_hat - o1.mean[dates], axis=1)
    coverage = float(np.mean(err <= fc.radius))
    assert 0.0 <= coverage <= 1.0
