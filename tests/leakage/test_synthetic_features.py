"""Look-ahead and oracle-leak guards for synthetic benchmark features, end to end.

These go through the real path (payload -> immutable snapshot -> ``build_pit_panel``). The
generic look-ahead property over random inputs, including synthetic features, lives in
tests/leakage/test_point_in_time.py.

1. Perturbing synthetic features dated after t leaves every panel value at or before t
   bit-identical, and perturbing the value dated t changes row t (the lag is exactly 0).
2. The panel's ``syn_*`` values equal the dataset's observed features bit for bit.
3. Oracle-only latents (``z`` when hidden, ``decay_multiplier``, ``h_*``, ``market``) never
   reach the snapshot or the panel, and the detector catches them when they are planted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.pit import bit_identical
from tests.fixtures.synthetic_snapshot import (
    PRESETS,
    benchmark_config,
    latent_leaks,
    small_dataset,
    with_feature_table,
    write_benchmark_snapshot,
)

from invariant.benchmark.payload import synthetic_feature_payload
from invariant.benchmark.synthetic import SyntheticDataset
from invariant.data.pit import PitPanel, build_pit_panel
from invariant.data.snapshot import Snapshot
from invariant.data.sources.base import SourceName

pytestmark = pytest.mark.leakage


def _panel(dataset: SyntheticDataset, snapshot: Snapshot) -> PitPanel:
    return build_pit_panel(snapshot, benchmark_config(dataset))


def _perturbed_features(dataset: SyntheticDataset, rows: slice, seed: int) -> pd.DataFrame:
    """The genuine ``features`` table with ``rows`` replaced by unrelated values."""
    table = synthetic_feature_payload(dataset).tables["features"].copy()
    cols = list(table.columns[1:])
    gen = np.random.default_rng(seed)
    n = len(table.iloc[rows])
    table.loc[table.index[rows], cols] = gen.standard_normal((n, len(cols)))
    return table


# --- 1. perturbing the future ----------------------------------------------------------------


@pytest.mark.parametrize("t", [0, 1, 150, 298])
def test_features_after_t_do_not_change_panel_at_or_before_t(tmp_path: Path, t: int) -> None:
    ds = small_dataset()
    genuine = write_benchmark_snapshot(ds, tmp_path / "a")
    future = _perturbed_features(ds, slice(t + 1, None), seed=t)
    perturbed = write_benchmark_snapshot(
        ds, tmp_path / "b", features=with_feature_table(synthetic_feature_payload(ds), future)
    )
    assert perturbed.snapshot_id != genuine.snapshot_id
    pa, pb = _panel(ds, genuine), _panel(ds, perturbed)
    assert bit_identical(pa.features.iloc[: t + 1], pb.features.iloc[: t + 1])
    assert bit_identical(pa.targets, pb.targets)
    syn = [c for c in pa.features.columns if c.startswith("syn_")]
    assert not pa.features[syn].iloc[t + 1 :].equals(pb.features[syn].iloc[t + 1 :])


@pytest.mark.parametrize("t", [0, 150, 299])
def test_feature_dated_t_is_used_at_t(tmp_path: Path, t: int) -> None:
    # The converse: a change dated t must show up at t, so the lag is 0, not merely >= 0.
    ds = small_dataset()
    genuine = write_benchmark_snapshot(ds, tmp_path / "a")
    changed = _perturbed_features(ds, slice(t, t + 1), seed=t)
    perturbed = write_benchmark_snapshot(
        ds, tmp_path / "b", features=with_feature_table(synthetic_feature_payload(ds), changed)
    )
    pa, pb = _panel(ds, genuine), _panel(ds, perturbed)
    syn = [c for c in pa.features.columns if c.startswith("syn_")]
    assert bit_identical(pa.features.iloc[:t], pb.features.iloc[:t])
    assert (pa.features[syn].iloc[t] != pb.features[syn].iloc[t]).all()
    assert bit_identical(pa.features.iloc[t + 1 :], pb.features.iloc[t + 1 :])


# --- 2. the panel carries the observed features exactly --------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_panel_features_are_the_observed_features_bit_for_bit(
    tmp_path: Path, preset: str | None
) -> None:
    ds = small_dataset(preset)
    panel = _panel(ds, write_benchmark_snapshot(ds, tmp_path))
    syn = [c for c in panel.features.columns if c.startswith("syn_")]
    assert syn == [f"syn_{c}" for c in ds.features.columns]
    got = panel.features[syn].set_axis(list(ds.features.columns), axis=1)
    want = ds.features.set_axis(pd.DatetimeIndex(ds.features.index, name="date"))
    assert got.index.equals(want.index)
    for col in want.columns:
        assert got[col].to_numpy().tobytes() == want[col].to_numpy(dtype="float64").tobytes()


# --- 3. oracle-only latents ------------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_no_oracle_latent_reaches_snapshot_or_panel(tmp_path: Path, preset: str | None) -> None:
    ds = small_dataset(preset)
    snapshot = write_benchmark_snapshot(ds, tmp_path)
    assert latent_leaks(snapshot, _panel(ds, snapshot), ds) == []


def test_hidden_confounder_omits_z_from_the_snapshot_entirely(tmp_path: Path) -> None:
    ds = small_dataset("hidden_confounder")
    snapshot = write_benchmark_snapshot(ds, tmp_path)
    for entry in snapshot.manifest.sources:
        for f in entry.files:
            assert "z" not in f.columns, f.path
    params = snapshot.manifest.source(SourceName.SYNTHETIC).params
    assert params["columns"] == ["s_real", "s_conf", "s_decay", "s_noise"]
    assert "z" not in snapshot.synthetic_report().columns
    assert "syn_z" not in _panel(ds, snapshot).features.columns


@pytest.mark.parametrize(
    ("preset", "planted", "renamed"),
    [
        ("hidden_confounder", "z", False),
        (None, "decay_multiplier", False),
        (None, "h_asset_00", True),
        ("multi_asset", "market", True),
    ],
)
def test_leak_detector_catches_planted_latents(
    tmp_path: Path, preset: str | None, planted: str, renamed: bool
) -> None:
    # Bypass the payload allowlist and write a latent into the features table, under its own
    # name or disguised as an innocuous one; the detector must flag it either way.
    ds = small_dataset(preset)
    genuine = synthetic_feature_payload(ds)
    table = genuine.tables["features"].copy()
    table["s_extra" if renamed else planted] = ds.latents[planted].to_numpy(dtype="float64")
    snapshot = write_benchmark_snapshot(ds, tmp_path, features=with_feature_table(genuine, table))
    leaks = latent_leaks(snapshot, _panel(ds, snapshot), ds)
    assert any(repr(planted) in leak for leak in leaks), leaks
