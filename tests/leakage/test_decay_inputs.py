"""Stage 2 look-ahead guard: forward targets (``fwd_*``) never influence Stage 2.

Replacing them with noise or NaN, or putting a ``fwd_`` column among the features, leaves the
DecayResult identical. Stage 2 reuses Stage 1's input selection, which never reads ``fwd_*``.
Window-level locality of the per-window statistics is pinned in test_decay_window_inputs.py.
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


def test_forward_targets_never_influence_stage2(tmp_path: Path) -> None:
    bp = benchmark_panel(tmp_path, n_obs=2500)  # seed 7: s_real, s_decay pass
    causal = run_causal_stage(bp.panel, bp.config.causal, RngFactory(5))
    cfg = load_config().decay.model_copy(update={"min_obs": 500, "n_boot": 99})
    base = run_decay_stage(bp.panel, causal, cfg, RngFactory(5))
    assert base.signals  # something reached Stage 2
    targets = bp.panel.targets
    noise = pd.DataFrame(
        np.random.default_rng(0).standard_normal(targets.shape),
        index=targets.index,
        columns=targets.columns,
    )
    for replaced in (noise, targets * np.nan, targets.iloc[:, :0]):
        assert (
            run_decay_stage(_with(bp.panel, targets=replaced), causal, cfg, RngFactory(5)) == base
        )
    leaked = bp.panel.features.copy()
    leaked["fwd_SYN00"] = targets["fwd_SYN00"].fillna(0.0)
    assert run_decay_stage(_with(bp.panel, features=leaked), causal, cfg, RngFactory(5)) == base
