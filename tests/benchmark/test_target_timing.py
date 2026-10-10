"""Benchmark runs must be scored with the target timing GroundTruth's expected values assume.

The SCM plants ``S_t -> R_{t+1}``. Scoring with the real-data execution lag (1) would compare
stages against ``Corr(S_t, R_{t+2})``, a decayed version of the planted effect.
"""

import numpy as np
import pytest
from tests.fixtures.pit import bit_identical

from invariant.benchmark.config import (
    list_benchmark_presets,
    load_benchmark_config,
    load_benchmark_target_timing,
)
from invariant.benchmark.ground_truth import (
    EXPECTED_VALUES_TARGET_TIMING,
    TargetTimingMismatchError,
)
from invariant.benchmark.scoring import scoring_targets
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import TargetConfig, load_config

pytestmark = pytest.mark.benchmark


def _small(preset: str | None = None) -> SyntheticDataset:
    cfg = load_benchmark_config(preset).model_copy(update={"n_obs": 300, "burn_in": 10})
    return generate_market(cfg, 3)


def test_benchmark_yaml_timing_matches_ground_truth() -> None:
    timing = load_benchmark_target_timing()
    assert timing == TargetConfig(execution_lag=0, horizon=1)
    for preset in (None, *list_benchmark_presets()):
        assert _small(preset).ground_truth.target_timing == timing


def test_real_data_timing_is_rejected_for_benchmark_scoring() -> None:
    dataset = _small()
    real = load_config().pit.targets
    assert real.execution_lag == 1  # base.yaml: the real-data convention
    with pytest.raises(TargetTimingMismatchError, match="execution_lag=1"):
        scoring_targets(dataset, real)


@pytest.mark.parametrize(
    "timing",
    [TargetConfig(execution_lag=1, horizon=1), TargetConfig(execution_lag=0, horizon=5)],
    ids=["execution_lag", "horizon"],
)
def test_any_mismatch_raises(timing: TargetConfig) -> None:
    with pytest.raises(TargetTimingMismatchError):
        scoring_targets(_small(), timing)


def test_matching_timing_targets_are_next_return() -> None:
    dataset = _small("multi_asset")
    y = scoring_targets(dataset, EXPECTED_VALUES_TARGET_TIMING)
    assert bit_identical(y, dataset.returns.shift(-1))


def test_scored_targets_recover_planted_marginal_ic() -> None:
    # Ground truth first: under the matching timing, the sample IC of S_real agrees with the
    # population IC GroundTruth exports (|error| < 4 standard errors, SE ~ 1/sqrt(n)).
    dataset = generate_market(load_benchmark_config(), 20260105)
    truth = dataset.ground_truth
    y = scoring_targets(dataset, load_benchmark_target_timing()).iloc[:-1, 0]
    s = dataset.features["s_real"].iloc[:-1]
    ic = float(np.corrcoef(s, y)[0, 1])
    want = truth.signal("s_real").marginal_ic[0]
    assert abs(ic - want) < 4 / np.sqrt(len(y))
