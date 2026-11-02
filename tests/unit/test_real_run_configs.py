"""The first real-data run's configs (docs/adr/0018): pinned so they differ only where stated."""

from __future__ import annotations

from pathlib import Path

from invariant.config.settings import load_config
from invariant.stages.memo.validation import load_memo_validation_config


def test_the_sensitivity_config_differs_only_in_the_lag_rule() -> None:
    """docs/adr/0018: the sensitivity run is the primary run with the max-lag rule."""
    primary = load_config(override_path=Path("configs/real_0018.yaml"))
    sens = load_config(override_path=Path("configs/real_0018_maxlag.yaml"))
    assert primary.data.fred_market is not None and sens.data.fred_market is not None
    assert primary.data.fred_market.lag_rule == "exact_in_archive"
    assert sens.data.fred_market.lag_rule == "max_lag"
    unpin = {"data": {"fred_market": {"lag_rule"}}}
    assert primary.model_dump(exclude=unpin) == sens.model_dump(exclude=unpin)
    assert primary.snapshot_id == sens.snapshot_id == "930504ace5464752"


def test_the_real_shaped_config_changes_only_the_seed_and_the_shape() -> None:
    """docs/adr/0018: every preset real-data-shaped, a fresh seed; everything else as the
    accepted deviation-1 validation."""
    dev = load_memo_validation_config(Path("configs/memo_validation_deviation1.yaml"))
    real = load_memo_validation_config(Path("configs/memo_validation_real.yaml"))
    assert real.root_seed == 20261113 not in (dev.root_seed, 20261105)
    assert all(p.real_shaped for p in real.presets)
    # Same names: the coverage rule maps categories to preset names (COVERAGE_PRESETS).
    assert [(p.name, p.preset, p.n_obs) for p in real.presets] == [
        (p.name, p.preset, p.n_obs) for p in dev.presets
    ]
    same = {"root_seed", "presets"}
    assert real.model_dump(exclude=same) == dev.model_dump(exclude=same)
