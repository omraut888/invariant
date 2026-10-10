"""Macro series from ALFRED: every vintage of every observation, with its real-time period.

Method
------
``fetch_alfred`` pulls all releases of each configured series through ``AlfredClient`` (FRED
``series/observations`` over the full real-time period, paginated; see ``clients``) and keeps
one row per ``(series_id, observation_date, realtime_start)`` with ``realtime_end`` and
``value``. A row means: from ``realtime_start`` through ``realtime_end`` (inclusive), the
published value for ``observation_date`` was ``value``. ``realtime_end = 9999-12-31`` marks the
current vintage. Nothing is collapsed to final revised values. FRED's missing-value marker ``.``
becomes NaN, is kept as a row (the vintage did publish "no value") and is counted.

Validation per series: ``realtime_start <= realtime_end``; no duplicate
``(observation_date, realtime_start)``; for each observation, real-time periods are disjoint
(the next vintage starts after the previous one ends). Gaps between periods are allowed: an
observation can be withdrawn from a vintage.

Assumptions and limitations
---------------------------
- ``realtime_start`` is a date with no time of day. Most releases are at 08:30 ET, but some are
  not, and the API does not say. The point-in-time layer (``pit.py``) therefore treats a vintage
  as usable from the **next trading session after** ``realtime_start``. This is conservative:
  it costs at most one session of timeliness and cannot leak.
- ALFRED's archive begins at a series-specific first vintage (INDPRO 1927-01-26, UNRATE
  1960-03-15, CPIAUCSL 1972-07-21, PAYEMS 1955-05-06). Observations dated before it carry that
  first vintage date as ``realtime_start``, later than their true publication. Again
  conservative.
- Vintages reflect what FRED/ALFRED archived, which can differ in rare cases from the agency's
  press release (e.g. later-discovered transcription errors).

Reference: Croushore, D. and Stark, T. (2001), "A real-time data set for macroeconomists",
Journal of Econometrics 105(1), 111-130. ALFRED: https://alfred.stlouisfed.org/
See docs/adr/0004-data-sources-and-pit-limitations.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from structlog.typing import FilteringBoundLogger

from invariant.core.logging import get_logger
from invariant.data.sources.base import (
    DATE_DTYPE,
    DataValidationError,
    SourceName,
    SourcePayload,
    library_versions,
)
from invariant.data.sources.clients import (
    ALFRED_REALTIME_END,
    ALFRED_REALTIME_START,
    AlfredClient,
    AlfredRawSeries,
)

__all__ = [
    "VINTAGE_COLUMNS",
    "AlfredReport",
    "SeriesReport",
    "fetch_alfred",
    "parse_alfred_rows",
    "validate_vintages",
]

VINTAGE_COLUMNS = ("series_id", "observation_date", "realtime_start", "realtime_end", "value")
_MISSING = "."


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SeriesReport(_Frozen):
    series_id: str
    n_rows: int
    api_count: int
    pages: int
    n_observations: int
    n_vintages: int
    n_missing_values: int
    first_observation: date
    last_observation: date
    first_realtime_start: date
    last_realtime_start: date


class AlfredReport(_Frozen):
    series: tuple[SeriesReport, ...]


def _dates(rows: Sequence[Mapping[str, str]], column: str, series_id: str) -> pd.Series:
    try:
        return pd.to_datetime(pd.Series([r[column] for r in rows]), format="%Y-%m-%d").astype(
            DATE_DTYPE
        )
    except ValueError as exc:
        raise DataValidationError(f"{series_id}: invalid {column}: {exc}") from None


def parse_alfred_rows(raw: AlfredRawSeries) -> pd.DataFrame:
    """Convert raw FRED rows to a ``VINTAGE_COLUMNS`` table, sorted and validated.

    The parse is row-wise and never alters ``realtime_start``/``realtime_end``.
    """
    sid = raw.series_id
    if not raw.rows:
        raise DataValidationError(f"{sid}: no observations returned")
    values: list[float] = []
    for r in raw.rows:
        v = r["value"]
        try:
            values.append(np.nan if v == _MISSING else float(v))
        except ValueError:
            raise DataValidationError(f"{sid}: non-numeric value {v!r}") from None
    frame = pd.DataFrame(
        {
            "series_id": pd.Series([sid] * len(raw.rows), dtype="str"),
            "observation_date": _dates(raw.rows, "date", sid),
            "realtime_start": _dates(raw.rows, "realtime_start", sid),
            "realtime_end": _dates(raw.rows, "realtime_end", sid),
            "value": np.asarray(values, dtype="float64"),
        }
    )
    return validate_vintages(frame, sid)


def validate_vintages(frame: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Sort one series' ``VINTAGE_COLUMNS`` table and check its real-time periods.

    Raises ``DataValidationError`` if a period ends before it starts, an
    ``(observation_date, realtime_start)`` pair repeats, or two periods of one observation
    overlap. Returns a sorted copy with a default index.
    """
    sid = series_id
    missing = [c for c in VINTAGE_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"{sid}: vintage table lacks columns {missing}")
    if (frame["series_id"] != sid).any():
        raise DataValidationError(f"{sid}: vintage table contains other series")
    frame = frame.sort_values(["observation_date", "realtime_start"], kind="stable")
    frame = frame.reset_index(drop=True)

    if (frame["realtime_start"] > frame["realtime_end"]).any():
        raise DataValidationError(f"{sid}: realtime_start after realtime_end")
    if frame.duplicated(["observation_date", "realtime_start"]).any():
        raise DataValidationError(f"{sid}: duplicate (observation_date, realtime_start) rows")
    same_obs = frame["observation_date"].eq(frame["observation_date"].shift())
    overlap = same_obs & (frame["realtime_start"] <= frame["realtime_end"].shift())
    if overlap.any():
        row = frame[overlap].iloc[0]
        raise DataValidationError(
            f"{sid}: overlapping real-time periods for {row['observation_date'].date()}"
        )
    return frame


def _report(raw: AlfredRawSeries, frame: pd.DataFrame) -> SeriesReport:
    obs = pd.DatetimeIndex(frame["observation_date"])
    rts = pd.DatetimeIndex(frame["realtime_start"])
    return SeriesReport(
        series_id=raw.series_id,
        n_rows=len(frame),
        api_count=raw.count,
        pages=raw.pages,
        n_observations=int(obs.nunique()),
        n_vintages=int(rts.nunique()),
        n_missing_values=int(frame["value"].isna().sum()),
        first_observation=obs.min().date(),
        last_observation=obs.max().date(),
        first_realtime_start=rts.min().date(),
        last_realtime_start=rts.max().date(),
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def fetch_alfred(
    client: AlfredClient,
    series_ids: Sequence[str],
    *,
    now: Callable[[], datetime] = _utcnow,
) -> SourcePayload:
    """Fetch all vintages of ``series_ids``; return the ``alfred`` payload (table ``vintages``)."""
    if not series_ids:
        raise ValueError("series_ids must not be empty")
    fetched_at = now()
    frames: list[pd.DataFrame] = []
    reports: list[SeriesReport] = []
    for sid in series_ids:
        raw = client.all_releases(sid)
        if raw.series_id != sid:
            raise DataValidationError(f"requested {sid}, client returned {raw.series_id}")
        frame = parse_alfred_rows(raw)
        rep = _report(raw, frame)
        if rep.n_missing_values:
            _log().warning("alfred_missing_values", series_id=sid, n=rep.n_missing_values)
        frames.append(frame)
        reports.append(rep)
    table = pd.concat(frames, ignore_index=True)[list(VINTAGE_COLUMNS)]
    _log().info("alfred_fetched", series=list(series_ids), rows=len(table))
    return SourcePayload(
        source=SourceName.ALFRED,
        params={
            "series": list(series_ids),
            "realtime_start": ALFRED_REALTIME_START,
            "realtime_end": ALFRED_REALTIME_END,
            "endpoint": "fred/series/observations",
        },
        report=AlfredReport(series=tuple(reports)),
        tables={"vintages": table.astype({"series_id": "str"})},
        fetched_at=fetched_at,
        library_versions=library_versions("httpx"),
    )
