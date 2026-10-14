"""The ``synthetic`` feature source: payload allowlist, snapshot reader, PIT placement, links."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.ledger import CLEAN_GIT, temp_settings
from tests.fixtures.pit import SESSIONS, pit_config
from tests.fixtures.synthetic_snapshot import (
    benchmark_config,
    small_dataset,
    with_feature_table,
    write_benchmark_snapshot,
)

from invariant.benchmark.payload import (
    observable_feature_names,
    synthetic_etf_payload,
    synthetic_feature_payload,
)
from invariant.config.settings import load_config
from invariant.core.run_hash import RunContext
from invariant.data.build import FETCHED_SOURCES, build_snapshot
from invariant.data.pit import PitInputs, assemble_panel, build_pit_panel, synthetic_on_sessions
from invariant.data.snapshot import write_snapshot
from invariant.data.sources.base import DATE_DTYPE, DataValidationError, SourceName
from invariant.data.sources.synthetic import SYNTHETIC_PROVIDER, validate_feature_table
from invariant.ledger.benchmark_score import BenchmarkMismatchError, score_run

# --- payload ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (None, {"s_real", "s_conf", "s_decay", "s_noise", "z"}),
        ("hidden_confounder", {"s_real", "s_conf", "s_decay", "s_noise"}),
    ],
)
def test_observable_features_follow_ground_truth(preset: str | None, expected: set[str]) -> None:
    ds = small_dataset(preset)
    assert observable_feature_names(ds.ground_truth) == expected
    assert set(ds.features.columns) == expected


def test_payload_carries_features_and_dataset_link() -> None:
    ds = small_dataset()
    payload = synthetic_feature_payload(ds)
    assert payload.source is SourceName.SYNTHETIC
    assert payload.params["provider"] == SYNTHETIC_PROVIDER
    assert payload.params["dataset_sha256"] == ds.ground_truth.dataset_sha256
    assert payload.params["columns"] == list(ds.features.columns)
    table = payload.tables["features"]
    assert list(table.columns) == ["date", *ds.features.columns]
    assert str(table["date"].dtype) == DATE_DTYPE and len(table) == len(ds.features)


@pytest.mark.parametrize(
    ("preset", "latent"),
    [("hidden_confounder", "z"), (None, "decay_multiplier"), (None, "h_asset_00")],
)
def test_payload_refuses_non_observable_columns(preset: str | None, latent: str) -> None:
    ds = small_dataset(preset)
    leaky = dataclasses.replace(ds, features=ds.features.assign(**{latent: ds.latents[latent]}))
    with pytest.raises(ValueError, match=latent):
        synthetic_feature_payload(leaky)


def test_payload_refuses_a_tampered_dataset() -> None:
    ds = small_dataset()
    edited = ds.features.copy()
    edited["s_real"] = edited["s_real"] + 1.0
    with pytest.raises(ValueError, match="content hash"):
        synthetic_feature_payload(dataclasses.replace(ds, features=edited))


def test_feature_table_validation() -> None:
    table = synthetic_feature_payload(small_dataset()).tables["features"]
    with pytest.raises(DataValidationError, match="date"):
        validate_feature_table(table.drop(columns="date"))
    with pytest.raises(DataValidationError, match="float64"):
        validate_feature_table(table.astype({"s_real": "float32"}))
    with pytest.raises(ValueError, match="sorted"):
        validate_feature_table(table.iloc[::-1].reset_index(drop=True))


# --- snapshot ---------------------------------------------------------------------------------


def test_snapshot_round_trip(tmp_path: Path) -> None:
    ds = small_dataset()
    snap = write_benchmark_snapshot(ds, tmp_path)
    frame = snap.synthetic_features()
    assert frame.index.equals(pd.DatetimeIndex(ds.features.index, name="date"))
    np.testing.assert_array_equal(frame.to_numpy(), ds.features.to_numpy())
    assert snap.synthetic_report().columns == tuple(ds.features.columns)


def test_build_snapshot_never_fetches_synthetic(tmp_path: Path) -> None:
    assert SourceName.SYNTHETIC not in FETCHED_SOURCES
    with pytest.raises(ValueError, match="no fetcher"):
        build_snapshot(load_config(), temp_settings(tmp_path), sources=[SourceName.SYNTHETIC])


# --- PIT placement ----------------------------------------------------------------------------


def test_synthetic_on_sessions_is_lag_zero_without_carry_forward() -> None:
    frame = pd.DataFrame({"s_real": [1.0, 2.0]}, index=SESSIONS[[3, 5]])
    out = synthetic_on_sessions(frame, SESSIONS)
    assert out["s_real"].iloc[3] == 1.0 and out["s_real"].iloc[5] == 2.0
    assert out["s_real"].drop(SESSIONS[[3, 5]]).isna().all()


def test_synthetic_on_sessions_rejects_off_calendar_dates() -> None:
    saturday = pd.DatetimeIndex([pd.Timestamp("2025-03-08")], name="date")
    with pytest.raises(DataValidationError, match="not on panel sessions"):
        synthetic_on_sessions(pd.DataFrame({"s_real": [1.0]}, index=saturday), SESSIONS)


def test_assemble_panel_prefixes_synthetic_columns() -> None:
    adj = pd.DataFrame({"AAA": 100.0}, index=SESSIONS)
    syn = pd.DataFrame({"s_real": 0.5, "z": -0.5}, index=SESSIONS)
    panel = assemble_panel(PitInputs(adj_close=adj, synthetic=syn), pit_config())
    assert list(panel.features.columns) == ["ret_AAA", "syn_s_real", "syn_z"]


# --- dataset links ----------------------------------------------------------------------------


def test_panel_refuses_features_from_another_dataset(tmp_path: Path) -> None:
    ds, other = small_dataset(seed=5), small_dataset(seed=6)
    snap = write_snapshot(
        [synthetic_etf_payload(ds), synthetic_feature_payload(other)], tmp_path / "snapshots"
    )
    with pytest.raises(DataValidationError, match="synthetic features are from dataset"):
        build_pit_panel(snap, benchmark_config(ds))


def test_panel_refuses_synthetic_features_with_real_prices(tmp_path: Path) -> None:
    ds = small_dataset()
    etf = synthetic_etf_payload(ds)
    real = dataclasses.replace(etf, params={**etf.params, "provider": "yfinance"})
    snap = write_snapshot([real, synthetic_feature_payload(ds)], tmp_path / "snapshots")
    with pytest.raises(DataValidationError, match="non-synthetic prices"):
        build_pit_panel(snap, benchmark_config(ds))


def test_scorer_refuses_features_from_another_dataset(tmp_path: Path) -> None:
    ds, other = small_dataset(seed=5), small_dataset(seed=6)
    snap = write_snapshot(
        [synthetic_etf_payload(ds), synthetic_feature_payload(other)], tmp_path / "snapshots"
    )
    settings = temp_settings(tmp_path)
    ctx = RunContext.create(
        benchmark_config(ds), snap.snapshot_id, settings=settings, git_state=CLEAN_GIT
    )
    with pytest.raises(BenchmarkMismatchError, match="features of dataset"):
        score_run(ctx.run_dir, ds.ground_truth, snap)


def test_panel_without_synthetic_source_has_no_syn_columns(tmp_path: Path) -> None:
    ds = small_dataset()
    snap = write_snapshot([synthetic_etf_payload(ds)], tmp_path / "snapshots")
    panel = build_pit_panel(snap, benchmark_config(ds))
    assert not any(c.startswith("syn_") for c in panel.features.columns)


def test_replaced_feature_table_still_links(tmp_path: Path) -> None:
    # Fixture sanity: a substituted features table keeps the genuine dataset link.
    ds = small_dataset()
    payload = synthetic_feature_payload(ds)
    snap = write_benchmark_snapshot(
        ds, tmp_path, features=with_feature_table(payload, payload.tables["features"])
    )
    assert build_pit_panel(snap, benchmark_config(ds)).features.shape[0] == len(ds.returns)
