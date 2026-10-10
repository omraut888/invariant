"""Targets for scoring a stage against the planted-signal benchmark.

Stages are scored on the benchmark through ``scoring_targets`` only. It builds forward returns
with ``invariant.data.pit.forward_returns``, the same function real-data runs use, after
requiring that the timing matches the one ``GroundTruth`` computed its expected values for
(``GroundTruth.target_timing``, execution lag 0 and horizon 1, because the SCM plants
``S_t -> R_{t+1}``). With any other timing a stage would be graded against a decayed version
of the planted effect, so the mismatch raises instead of producing a misleading score.
See docs/adr/0006-point-in-time-panel.md.
"""

from __future__ import annotations

import pandas as pd

from invariant.benchmark.synthetic import SyntheticDataset
from invariant.config.settings import TargetConfig
from invariant.data.pit import forward_returns

__all__ = ["scoring_targets"]


def scoring_targets(dataset: SyntheticDataset, timing: TargetConfig) -> pd.DataFrame:
    """Forward returns of ``dataset`` under ``timing``; raises ``TargetTimingMismatchError``
    unless ``timing`` equals ``dataset.ground_truth.target_timing``."""
    dataset.ground_truth.require_target_timing(timing)
    return forward_returns(dataset.returns, timing)
