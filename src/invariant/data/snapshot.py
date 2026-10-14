"""Immutable, content-addressed data snapshots.

Layout
------
``<data_dir>/snapshots/<snapshot_id>/`` holds ``manifest.json`` and one Parquet file per table,
``<source>/<table>.parquet`` (e.g. ``etf/prices.parquet``, ``alfred/vintages.parquet``). The
manifest records, per source: fetch time (UTC), request params, library versions, the source's
typed report (coverage, missing values, API counts), and per file its path, SHA-256, row count
and columns.

Identity
--------
``digest = sha256(canonical_json({snapshot_schema, sources: {name: {params, files: {path:
sha256}}}}))`` and ``snapshot_id = digest[:16]``. Fetch times, reports and library versions are
deliberately excluded: fetching identical data twice yields the same id, and the first snapshot
is kept untouched. Reports are derived from the data, so they add nothing to identity.

File hashes are of the Parquet bytes. They are deterministic within a locked environment (fixed
writer options, schema metadata stripped), but a different pyarrow version may encode the same
data differently. That gives the same data a *different* id. It fails safe: two snapshots
can never share an id unless their bytes match, so a run hash is never ambiguous. The
cost is only lost deduplication. The writer's pandas/pyarrow versions are in the manifest.

Immutability
------------
A snapshot is written into a hidden temporary directory, fsynced, made read-only, then renamed
into place in one atomic ``os.rename``, so a reader never sees a partial snapshot. An existing
id is never overwritten: re-writing verifies the existing snapshot and returns it. Every load
re-hashes every file and recomputes the id, so tampering or corruption raises
``SnapshotIntegrityError`` instead of silently feeding altered data into a run.

See docs/adr/0005-immutable-snapshots.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, JsonValue
from structlog.typing import FilteringBoundLogger

from invariant.config.settings import Settings, load_settings
from invariant.core.logging import get_logger
from invariant.core.run_hash import canonical_json, canonicalize
from invariant.data.sources.alfred import VINTAGE_COLUMNS, AlfredReport
from invariant.data.sources.base import SourceName, SourcePayload, library_versions
from invariant.data.sources.etf import (
    PRICE_COLUMNS,
    EtfReport,
    adj_close_panel,
    log_returns,
)
from invariant.data.sources.french import FACTOR_COLUMNS, FrenchReport
from invariant.data.sources.synthetic import FEATURES_TABLE, SyntheticFeatureReport, features_frame

__all__ = [
    "MANIFEST_NAME",
    "SNAPSHOT_ID_LENGTH",
    "SNAPSHOT_SCHEMA_VERSION",
    "FileEntry",
    "Snapshot",
    "SnapshotError",
    "SnapshotIntegrityError",
    "SnapshotManifest",
    "SourceEntry",
    "compute_snapshot_digest",
    "default_snapshots_dir",
    "load_snapshot",
    "write_snapshot",
]

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_ID_LENGTH = 16
MANIFEST_NAME = "manifest.json"
_SNAPSHOT_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_TABLE_RE = re.compile(r"^[a-z0-9_]+$")
_READ_ONLY = 0o444


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


class SnapshotError(Exception):
    """A snapshot could not be written or located."""


class SnapshotIntegrityError(SnapshotError):
    """A snapshot on disk does not match its manifest or its id."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FileEntry(_Frozen):
    path: str
    sha256: str
    n_rows: int
    columns: tuple[str, ...]


class SourceEntry(_Frozen):
    source: SourceName
    fetched_at: datetime
    params: dict[str, JsonValue]
    library_versions: dict[str, str | None]
    report: dict[str, JsonValue]
    files: tuple[FileEntry, ...]


class SnapshotManifest(_Frozen):
    snapshot_schema: int
    snapshot_id: str
    digest: str
    created_at: datetime
    environment: dict[str, str | None]
    sources: tuple[SourceEntry, ...]

    def source(self, name: SourceName) -> SourceEntry:
        for entry in self.sources:
            if entry.source is name:
                return entry
        raise SnapshotError(f"snapshot {self.snapshot_id} has no source {name!s}")


def compute_snapshot_digest(sources: Sequence[SourceEntry]) -> str:
    """SHA-256 of the identity payload (schema version, params, file hashes; see module doc)."""
    payload = {
        "snapshot_schema": SNAPSHOT_SCHEMA_VERSION,
        "sources": {
            str(s.source): {"params": s.params, "files": {f.path: f.sha256 for f in s.files}}
            for s in sources
        },
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def default_snapshots_dir(settings: Settings | None = None) -> Path:
    return (settings if settings is not None else load_settings()).data_dir / "snapshots"


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def _parquet_bytes(frame: pd.DataFrame, where: str) -> bytes:
    if not isinstance(frame.index, pd.RangeIndex) or frame.index.start != 0:
        raise SnapshotError(f"{where}: tables must have a default RangeIndex; use columns")
    if not all(isinstance(c, str) for c in frame.columns) or frame.columns.has_duplicates:
        raise SnapshotError(f"{where}: column names must be unique strings")
    # Schema metadata (pandas version, index info) is stripped so bytes depend on data only,
    # within one pyarrow version.
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd", compression_level=3, write_statistics=True)
    return buf.getvalue()


def _fsync_write(path: Path, data: bytes) -> None:
    with path.open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _manifest_bytes(manifest: SnapshotManifest) -> bytes:
    return (json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def write_snapshot(
    payloads: Sequence[SourcePayload],
    snapshots_dir: Path | None = None,
    *,
    now: Callable[[], datetime] = _utcnow,
) -> Snapshot:
    """Persist ``payloads`` as one immutable snapshot and return it (loaded and verified).

    If a snapshot with the same id exists it is verified and returned unchanged; the new
    payloads' fetch times and reports are discarded (and logged).
    """
    root = snapshots_dir if snapshots_dir is not None else default_snapshots_dir()
    if not payloads:
        raise SnapshotError("a snapshot needs at least one source")
    names = [p.source for p in payloads]
    if len(set(names)) != len(names):
        raise SnapshotError(f"duplicate sources in snapshot: {[str(n) for n in names]}")

    blobs: dict[str, bytes] = {}
    entries: list[SourceEntry] = []
    for payload in sorted(payloads, key=lambda p: str(p.source)):
        if not payload.tables:
            raise SnapshotError(f"source {payload.source!s} has no tables")
        files: list[FileEntry] = []
        for table_name in sorted(payload.tables):
            if not _TABLE_RE.fullmatch(table_name):
                raise SnapshotError(f"invalid table name {table_name!r}")
            frame = payload.tables[table_name]
            rel = f"{payload.source}/{table_name}.parquet"
            data = _parquet_bytes(frame, rel)
            blobs[rel] = data
            files.append(
                FileEntry(
                    path=rel,
                    sha256=hashlib.sha256(data).hexdigest(),
                    n_rows=len(frame),
                    columns=tuple(str(c) for c in frame.columns),
                )
            )
        params = canonicalize(dict(payload.params))
        if not isinstance(params, dict):  # unreachable: canonicalize maps dicts to dicts
            raise TypeError(f"params of {payload.source!s} did not canonicalize to a mapping")
        entries.append(
            SourceEntry(
                source=payload.source,
                fetched_at=payload.fetched_at,
                params=params,
                library_versions=dict(payload.library_versions),
                report=payload.report.model_dump(mode="json"),
                files=tuple(files),
            )
        )

    digest = compute_snapshot_digest(entries)
    snapshot_id = digest[:SNAPSHOT_ID_LENGTH]
    final = root / snapshot_id
    if final.exists():
        return _existing(final.parent, snapshot_id, digest)

    manifest = SnapshotManifest(
        snapshot_schema=SNAPSHOT_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        digest=digest,
        created_at=now(),
        environment=library_versions("pandas", "pyarrow"),
        sources=tuple(entries),
    )
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp-{snapshot_id}-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        for rel, data in blobs.items():
            path = tmp / rel
            path.parent.mkdir(exist_ok=True)
            _fsync_write(path, data)
            path.chmod(_READ_ONLY)
        _fsync_write(tmp / MANIFEST_NAME, _manifest_bytes(manifest))
        (tmp / MANIFEST_NAME).chmod(_READ_ONLY)
        for sub in {Path(rel).parent for rel in blobs}:
            _fsync_dir(tmp / sub)
        _fsync_dir(tmp)
        try:
            os.rename(tmp, final)
        except OSError:
            if final.exists():  # lost a race with a concurrent writer of the same id
                shutil.rmtree(tmp)
                return _existing(root, snapshot_id, digest)
            raise
        _fsync_dir(root)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    _log().info(
        "snapshot_written",
        snapshot_id=snapshot_id,
        sources=[str(e.source) for e in entries],
        path=str(final),
    )
    return load_snapshot(snapshot_id, root)


def _existing(root: Path, snapshot_id: str, digest: str) -> Snapshot:
    snap = load_snapshot(snapshot_id, root)
    if snap.manifest.digest != digest:
        raise SnapshotIntegrityError(
            f"snapshot {snapshot_id} exists with a different full digest (id collision); "
            "refusing to reuse it"
        )
    _log().info("snapshot_exists", snapshot_id=snapshot_id, detail="kept existing, unchanged")
    return snap


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def load_snapshot(snapshot_id: str, snapshots_dir: Path | None = None) -> Snapshot:
    """Load snapshot ``snapshot_id``, verifying every file hash and the id itself."""
    if not _SNAPSHOT_ID_RE.fullmatch(snapshot_id):
        raise SnapshotError(f"invalid snapshot id {snapshot_id!r}: expected 16 lowercase hex")
    root = snapshots_dir if snapshots_dir is not None else default_snapshots_dir()
    path = root / snapshot_id
    try:
        raw = (path / MANIFEST_NAME).read_bytes()
    except FileNotFoundError:
        raise SnapshotError(f"snapshot {snapshot_id} not found in {root}") from None
    try:
        manifest = SnapshotManifest.model_validate_json(raw)
    except ValueError as exc:
        raise SnapshotIntegrityError(f"snapshot {snapshot_id}: invalid manifest: {exc}") from exc
    if manifest.snapshot_schema != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError(
            f"snapshot {snapshot_id} has schema {manifest.snapshot_schema}, "
            f"this code reads {SNAPSHOT_SCHEMA_VERSION}"
        )
    digest = compute_snapshot_digest(manifest.sources)
    if digest != manifest.digest or digest[:SNAPSHOT_ID_LENGTH] != snapshot_id:
        raise SnapshotIntegrityError(
            f"snapshot {snapshot_id}: manifest does not hash to its id (edited manifest?)"
        )
    snap = Snapshot(manifest=manifest, path=path)
    for entry in manifest.sources:
        for f in entry.files:
            snap._read_verified(f)
    return snap


class Snapshot:
    """A verified, read-only view of one snapshot. Construct with ``load_snapshot``."""

    def __init__(self, manifest: SnapshotManifest, path: Path) -> None:
        self.manifest = manifest
        self.path = path

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id

    def _read_verified(self, entry: FileEntry) -> bytes:
        try:
            data = (self.path / entry.path).read_bytes()
        except FileNotFoundError:
            raise SnapshotIntegrityError(
                f"snapshot {self.snapshot_id}: missing file {entry.path}"
            ) from None
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise SnapshotIntegrityError(
                f"snapshot {self.snapshot_id}: {entry.path} does not match its manifest hash"
            )
        return data

    def table(self, source: SourceName, name: str) -> pd.DataFrame:
        """Read one table, re-verifying its hash on the bytes that are parsed."""
        rel = f"{source}/{name}.parquet"
        entry = next((f for f in self.manifest.source(source).files if f.path == rel), None)
        if entry is None:
            raise SnapshotError(f"snapshot {self.snapshot_id} has no table {rel}")
        frame: pd.DataFrame = pq.read_table(io.BytesIO(self._read_verified(entry))).to_pandas()
        if tuple(frame.columns) != entry.columns or len(frame) != entry.n_rows:
            raise SnapshotIntegrityError(f"snapshot {self.snapshot_id}: {rel} shape mismatch")
        return frame

    def _report(self, source: SourceName) -> Mapping[str, JsonValue]:
        return self.manifest.source(source).report

    # --- etf ----------------------------------------------------------------------------------

    def etf_report(self) -> EtfReport:
        return EtfReport.model_validate(self._report(SourceName.ETF))

    def etf_prices(self) -> pd.DataFrame:
        """Long price table (``PRICE_COLUMNS``), exactly as fetched."""
        return self.table(SourceName.ETF, "prices")[list(PRICE_COLUMNS)]

    def etf_sessions(self) -> pd.DatetimeIndex:
        """The exchange sessions the coverage report was computed against."""
        return pd.DatetimeIndex(self.table(SourceName.ETF, "sessions")["date"], name="date")

    def etf_adj_close(self) -> pd.DataFrame:
        """Session-aligned adjusted-close panel; NaN marks a missing session (never filled)."""
        tickers = [t.ticker for t in self.etf_report().tickers]
        return adj_close_panel(self.etf_prices(), self.etf_sessions(), tickers)

    def etf_log_returns(self) -> pd.DataFrame:
        return log_returns(self.etf_adj_close())

    # --- french -------------------------------------------------------------------------------

    def french_report(self) -> FrenchReport:
        return FrenchReport.model_validate(self._report(SourceName.FRENCH))

    def french_factors(self) -> pd.DataFrame:
        """Daily factor returns (decimal), indexed by date. Not point-in-time: see french.py."""
        frame = self.table(SourceName.FRENCH, "factors")
        return frame.set_index("date")[list(FACTOR_COLUMNS)]

    # --- alfred -------------------------------------------------------------------------------

    def alfred_report(self) -> AlfredReport:
        return AlfredReport.model_validate(self._report(SourceName.ALFRED))

    def alfred_vintages(self) -> pd.DataFrame:
        """Long vintage table (``VINTAGE_COLUMNS``) for all series."""
        return self.table(SourceName.ALFRED, "vintages")[list(VINTAGE_COLUMNS)]

    # --- synthetic (benchmark only) -----------------------------------------------------------

    def synthetic_report(self) -> SyntheticFeatureReport:
        return SyntheticFeatureReport.model_validate(self._report(SourceName.SYNTHETIC))

    def synthetic_features(self) -> pd.DataFrame:
        """Observed planted-signal features indexed by date; a value dated t is known at t."""
        return features_frame(self.table(SourceName.SYNTHETIC, FEATURES_TABLE))
