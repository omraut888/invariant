"""Stage 1 look-ahead guards.

1. Forward targets (``fwd_*``) never influence Stage 1: replacing them with noise or NaN, or
   placing a ``fwd_`` column among the features, leaves the CausalResult identical.
2. Lag orientation: every tested link runs from a feature dated ``t - lag`` (lag >= 1) to the
   return dated ``t``. A feature that leaks the next return (``S_t ~ R_{t+1}``) is found at
   lag 1, and a contemporaneous one (``S_t ~ R_t``) is not, because lag 0 is never tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.causal import benchmark_panel

from invariant.core.run_hash import RngFactory
from invariant.core.types import Verdict
from invariant.data.pit import PitPanel
from invariant.stages.causal.stage import run_causal_stage

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


def test_forward_targets_never_influence_stage1(tmp_path: Path) -> None:
    bp = benchmark_panel(tmp_path, n_obs=800)
    cfg = bp.config.causal
    base = run_causal_stage(bp.panel, cfg, RngFactory(5))
    targets = bp.panel.targets
    assert list(targets.columns) and all(str(c).startswith("fwd_") for c in targets.columns)

    noise = pd.DataFrame(
        np.random.default_rng(0).standard_normal(targets.shape),
        index=targets.index,
        columns=targets.columns,
    )
    nan = targets * np.nan
    for replaced in (noise, nan, targets.iloc[:, :0]):
        assert run_causal_stage(_with(bp.panel, targets=replaced), cfg, RngFactory(5)) == base

    leaked = bp.panel.features.copy()
    leaked["fwd_SYN00"] = targets["fwd_SYN00"].fillna(0.0)
    assert run_causal_stage(_with(bp.panel, features=leaked), cfg, RngFactory(5)) == base


def test_tested_links_are_lagged_from_feature_to_later_return(tmp_path: Path) -> None:
    bp = benchmark_panel(tmp_path, n_obs=1000)
    feats = bp.panel.features.copy()
    ret = feats["ret_SYN00"]
    # Independent noise per canary: a shared draw would itself link the two canaries.
    noise = np.random.default_rng(1).standard_normal((len(ret), 2)) * float(ret.std())
    feats["syn_future"] = (ret.shift(-1) + noise[:, 0]).fillna(0.0)  # S_t ~ R_{t+1}: a leak
    feats["syn_now"] = (ret + noise[:, 1]).fillna(0.0)  # S_t ~ R_t: contemporaneous only
    cfg = bp.config.causal.model_copy(
        update={"signals": (*(bp.config.causal.signals or ()), "future", "now")}
    )
    result = run_causal_stage(_with(bp.panel, features=feats), cfg, RngFactory(2))
    by_id = {s.signal_id: s for s in result.signals}

    future = by_id["future"]
    assert future.verdict is Verdict.PASS and future.evidence is not None
    lag1 = next(t for t in future.evidence.tests if t.lag == 1)
    assert lag1.r > 0.5  # S_{t-1} ~ R_t is the tested pair at lag 1
    assert future.evidence.best_lag == 1

    now = by_id["now"].evidence
    assert now is not None
    assert all(abs(t.r) < 0.15 for t in now.tests)  # would be ~0.7 if lag 0 were tested
