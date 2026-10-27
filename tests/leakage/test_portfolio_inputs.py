"""Stage 3 look-ahead guard: forward targets (``fwd_*``) never influence Stage 3.

Replacing them with noise or NaN, or putting a ``fwd_`` column among the features, leaves the
PortfolioResult identical. Stage 3 reads only its members' columns and ``ret_*``. That every
weight dated t uses rows <= t, at every execution lag, is pinned in test_dro_no_lookahead.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.causal import benchmark_panel

from invariant.config.settings import load_config
from invariant.core.run_hash import RngFactory
from invariant.data.pit import PitPanel
from invariant.stages.causal.stage import run_causal_stage
from invariant.stages.decay.stage import run_decay_stage
from invariant.stages.portfolio.stage import run_portfolio_stage

pytestmark = pytest.mark.leakage


def _with(
    panel: PitPanel, *, features: pd.DataFrame | None = None, targets: pd.DataFrame | None = None
) -> PitPanel:
    return PitPanel(
        features if features is not None else panel.features,
        panel.obs_dates,
        targets if targets is not None else panel.targets,
        panel.config,
    )


def test_forward_targets_never_influence_stage3(tmp_path: Path) -> None:
    bp = benchmark_panel(tmp_path, n_obs=2500)  # seed 7: s_real, s_decay pass Stage 1
    causal = run_causal_stage(bp.panel, bp.config.causal, RngFactory(5))
    decay_cfg = load_config().decay.model_copy(update={"min_obs": 500, "n_boot": 99})
    decay = run_decay_stage(bp.panel, causal, decay_cfg, RngFactory(5))
    cfg = load_config().portfolio.model_copy(
        update={
            "window": 300,
            "cv_every": 1000,
            "cv_valid": 100,
            "ridge_grid": (0.0, 1.0),
            "eps_grid": (0.0, 0.05),
            "fc_draws": 500,
            "lw_n_boot": 99,
            "b_n_boot": 99,
            "min_obs": 600,
        }
    )
    base = run_portfolio_stage(bp.panel, decay, cfg, RngFactory(5))
    assert base.evidence is not None  # something reached Stage 3 and it ran
    targets = bp.panel.targets
    noise = pd.DataFrame(
        np.random.default_rng(0).standard_normal(targets.shape),
        index=targets.index,
        columns=targets.columns,
    )
    for replaced in (noise, targets * np.nan, targets.iloc[:, :0]):
        assert (
            run_portfolio_stage(_with(bp.panel, targets=replaced), decay, cfg, RngFactory(5))
            == base
        )
    leaked = bp.panel.features.copy()
    leaked["fwd_SYN00"] = targets["fwd_SYN00"].fillna(0.0)
    assert run_portfolio_stage(_with(bp.panel, features=leaked), decay, cfg, RngFactory(5)) == base
