import dataclasses
import json
import os
import stat
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.fixtures.fakes import (
    FRENCH_CONFIG,
    UNRATE_ROWS,
    FakeAlfredClient,
    FakeHttp,
    FakePriceClient,
    fixed_now,
    french_zips,
    price_frame,
)

from invariant.data import snapshot as snapshot_mod
from invariant.data.snapshot import (
    MANIFEST_NAME,
    SnapshotError,
    SnapshotIntegrityError,
    load_snapshot,
    write_snapshot,
)
from invariant.data.sources.alfred import fetch_alfred
from invariant.data.sources.base import SourceName, SourcePayload
from invariant.data.sources.etf import fetch_etf_prices
from invariant.data.sources.french import fetch_french_factors

SESSIONS = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
START, END = date(2025, 1, 2), date(2025, 1, 7)


def _etf(spy: list[float] | None = None) -> SourcePayload:
    raw = price_frame(SESSIONS, {"SPY": spy or [1.0, 2.0, 3.0, 4.0], "GLD": [5.0, 6.0, 7.0, 8.0]})
    return fetch_etf_prices(FakePriceClient(raw), ["SPY", "GLD"], START, END, now=fixed_now)


def _french() -> SourcePayload:
    return fetch_french_factors(FakeHttp(french_zips()), FRENCH_CONFIG, now=fixed_now)


def _alfred() -> SourcePayload:
    return fetch_alfred(FakeAlfredClient({"UNRATE": UNRATE_ROWS}), ["UNRATE"], now=fixed_now)


def _all() -> list[SourcePayload]:
    return [_etf(), _french(), _alfred()]


# --- round trip -------------------------------------------------------------------------------


def test_round_trip_preserves_every_table(tmp_path: Path) -> None:
    payloads = _all()
    snap = write_snapshot(payloads, tmp_path)
    assert len(snap.snapshot_id) == 16
    for payload in payloads:
        for name, frame in payload.tables.items():
            pd.testing.assert_frame_equal(snap.table(payload.source, name), frame)
    reloaded = load_snapshot(snap.snapshot_id, tmp_path)
    assert reloaded.manifest == snap.manifest


def test_typed_accessors(tmp_path: Path) -> None:
    snap = write_snapshot(_all(), tmp_path)
    assert snap.etf_report().complete
    assert list(snap.etf_adj_close().columns) == ["SPY", "GLD"]
    returns = snap.etf_log_returns()
    assert returns["SPY"].iloc[1] == pytest.approx(np.log(2.0))
    factors = snap.french_factors()
    assert factors.index.name == "date"
    assert factors.index.is_monotonic_increasing
    assert snap.french_report().files[0].crsp_vintage == "202608"
    assert len(snap.alfred_vintages()) == 4
    assert snap.alfred_report().series[0].n_missing_values == 1


def test_manifest_contents(tmp_path: Path) -> None:
    snap = write_snapshot(_all(), tmp_path, now=fixed_now)
    m = json.loads((snap.path / MANIFEST_NAME).read_text())
    assert m["snapshot_id"] == snap.snapshot_id
    assert m["digest"].startswith(snap.snapshot_id)
    assert [s["source"] for s in m["sources"]] == ["alfred", "etf", "french"]
    etf = next(s for s in m["sources"] if s["source"] == "etf")
    assert etf["fetched_at"] == "2026-01-02T03:04:05Z"
    assert etf["params"]["tickers"] == ["SPY", "GLD"]
    assert {f["path"] for f in etf["files"]} == {"etf/prices.parquet", "etf/sessions.parquet"}
    prices = next(f for f in etf["files"] if f["path"] == "etf/prices.parquet")
    assert prices["n_rows"] == 8
    assert len(prices["sha256"]) == 64
    assert "tickers" in etf["report"]
    assert "pyarrow" in m["environment"]


def test_missing_source_raises(tmp_path: Path) -> None:
    snap = write_snapshot([_etf()], tmp_path)
    with pytest.raises(SnapshotError, match="no source alfred"):
        snap.alfred_vintages()
    with pytest.raises(SnapshotError, match="no table"):
        snap.table(SourceName.ETF, "nope")


# --- identity ---------------------------------------------------------------------------------


def test_id_is_content_addressed_across_directories(tmp_path: Path) -> None:
    a = write_snapshot(_all(), tmp_path / "a")
    b = write_snapshot(list(reversed(_all())), tmp_path / "b")
    assert a.snapshot_id == b.snapshot_id


def test_fetch_time_does_not_change_id(tmp_path: Path) -> None:
    later = dataclasses.replace(_etf(), fetched_at=datetime(2030, 1, 1, tzinfo=UTC))
    assert (
        write_snapshot([_etf()], tmp_path / "a").snapshot_id
        == write_snapshot([later], tmp_path / "b").snapshot_id
    )


def test_data_change_changes_id(tmp_path: Path) -> None:
    a = write_snapshot([_etf()], tmp_path)
    b = write_snapshot([_etf(spy=[1.0, 2.0, 3.0, 4.5])], tmp_path)
    assert a.snapshot_id != b.snapshot_id


def test_params_change_changes_id(tmp_path: Path) -> None:
    base = _etf()
    other = dataclasses.replace(base, params={**base.params, "provider": "other"})
    assert write_snapshot([base], tmp_path).snapshot_id != (
        write_snapshot([other], tmp_path).snapshot_id
    )


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.permutations(["tickers", "start", "end", "calendar", "provider", "fields"]))
def test_id_independent_of_params_key_order(tmp_path: Path, order: list[str]) -> None:
    base = _etf()
    shuffled = dataclasses.replace(base, params={k: base.params[k] for k in order})
    a = write_snapshot([base], tmp_path)
    b = write_snapshot([shuffled], tmp_path)
    assert a.snapshot_id == b.snapshot_id


def test_parquet_bytes_are_deterministic() -> None:
    frame = _etf().tables["prices"]
    assert snapshot_mod._parquet_bytes(frame, "x") == snapshot_mod._parquet_bytes(frame, "x")


# --- immutability -----------------------------------------------------------------------------


def test_rewrite_keeps_existing_snapshot_untouched(tmp_path: Path) -> None:
    first = write_snapshot(_all(), tmp_path, now=fixed_now)
    manifest_bytes = (first.path / MANIFEST_NAME).read_bytes()
    later = [dataclasses.replace(p, fetched_at=datetime(2030, 1, 1, tzinfo=UTC)) for p in _all()]
    second = write_snapshot(later, tmp_path, now=lambda: datetime(2030, 1, 1, tzinfo=UTC))
    assert second.snapshot_id == first.snapshot_id
    assert (first.path / MANIFEST_NAME).read_bytes() == manifest_bytes


def test_files_are_read_only_and_no_temp_dirs_remain(tmp_path: Path) -> None:
    snap = write_snapshot(_all(), tmp_path)
    for path in snap.path.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert [p.name for p in tmp_path.iterdir()] == [snap.snapshot_id]


def test_failed_write_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: Path, data: bytes) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(snapshot_mod, "_fsync_write", boom)
    with pytest.raises(OSError, match="disk full"):
        write_snapshot(_all(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def _make_writable(path: Path) -> None:
    os.chmod(path, 0o644)


def test_tampered_table_is_detected(tmp_path: Path) -> None:
    snap = write_snapshot(_all(), tmp_path)
    target = snap.path / "etf" / "prices.parquet"
    _make_writable(target)
    target.write_bytes(target.read_bytes() + b"\0")
    with pytest.raises(SnapshotIntegrityError, match="does not match its manifest hash"):
        load_snapshot(snap.snapshot_id, tmp_path)
    with pytest.raises(SnapshotIntegrityError):
        snap.etf_prices()  # an already-loaded handle re-verifies on read


def test_missing_table_file_is_detected(tmp_path: Path) -> None:
    snap = write_snapshot(_all(), tmp_path)
    (snap.path / "french" / "factors.parquet").unlink()
    with pytest.raises(SnapshotIntegrityError, match="missing file"):
        load_snapshot(snap.snapshot_id, tmp_path)


def test_edited_manifest_is_detected(tmp_path: Path) -> None:
    snap = write_snapshot(_all(), tmp_path)
    path = snap.path / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["sources"][1]["params"]["tickers"] = ["SPY"]
    _make_writable(path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotIntegrityError, match="does not hash to its id"):
        load_snapshot(snap.snapshot_id, tmp_path)


def test_renamed_snapshot_dir_is_detected(tmp_path: Path) -> None:
    snap = write_snapshot(_all(), tmp_path)
    snap.path.rename(tmp_path / "0123456789abcdef")
    with pytest.raises(SnapshotIntegrityError, match="does not hash to its id"):
        load_snapshot("0123456789abcdef", tmp_path)


@pytest.mark.parametrize("bad", ["../escape", "ABCDEF0123456789", "abc", "0123456789abcdef0"])
def test_invalid_ids_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(SnapshotError, match="invalid snapshot id"):
        load_snapshot(bad, tmp_path)


def test_unknown_id_not_found(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="not found"):
        load_snapshot("0123456789abcdef", tmp_path)


# --- payload validation -----------------------------------------------------------------------


def test_payload_validation(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="at least one"):
        write_snapshot([], tmp_path)
    with pytest.raises(SnapshotError, match="duplicate sources"):
        write_snapshot([_etf(), _etf()], tmp_path)
    etf = _etf()
    indexed = dataclasses.replace(etf, tables={"prices": etf.tables["prices"].set_index("date")})
    with pytest.raises(SnapshotError, match="RangeIndex"):
        write_snapshot([indexed], tmp_path)
    bad_name = dataclasses.replace(etf, tables={"Bad-Name": etf.tables["prices"]})
    with pytest.raises(SnapshotError, match="invalid table name"):
        write_snapshot([bad_name], tmp_path)
    with pytest.raises(SnapshotError, match="no tables"):
        write_snapshot([dataclasses.replace(etf, tables={})], tmp_path)
