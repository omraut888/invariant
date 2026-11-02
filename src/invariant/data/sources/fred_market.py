"""Daily FRED market series: archived vintages, current values and a posting-lag audit.

Why a separate source (docs/adr/0019)
-------------------------------------
ALFRED's archive of every daily market series begins after the 2005 sample start (DTB3
2005-06-28; VIXCLS 2010-11-22; T10Y3M 2014-01-27; DBAA, DAAA 2014-04-02). Under ADR 0004's rule
an observation is usable only from its first archived vintage, so these series would be
unusable for years of the sample. And the existing ALFRED fetch cannot request them: FRED
refuses a request spanning more than 2000 vintage dates. This source splits each series at its
first vintage date V:

- **Archived period** (observations dated on or after V): the vintages themselves (table
  ``vintages``, ALFRED's row format), used under ADR 0004's rule exactly: a value is usable
  from the session after it was first posted, and the value used is the one in force.
- **Before the archive** (observations dated before V): the current values (table
  ``observations``), usable ``pre_archive_lag`` sessions after their date. No vintage exists
  for them, so this lag is an assumption, set from the archive (below).

Method
------
``fetch_fred_market`` takes, for each series and for observations dated in ``[start, end]``,
the current vintage and every archived vintage. The archive is requested one calendar year of
real time at a time; ``merge_chunks`` joins a row split at a year boundary back into one
real-time period (two consecutive chunks with the same value are one period, as ALFRED would
report it), and a row still current on the fetch date gets ALFRED's open end, 9999-12-31.

**Posting lag.** For an observation dated d first posted with a value on date r, the lag is
the number of exchange sessions in ``(d, r]``; under ADR 0004 the value is usable at the first
session after r. It is measured for observations dated on or after V (older ones were loaded on
the archive's first day). Two kinds of observation are not *regular* postings and are excluded
from the pre-archive lag:

- **backfilled**: first published as missing (FRED's ``.``) and filled later. In real time the
  value was missing; the archived rows reproduce that.
- **batch-posted**: first posted on a date that first-posted ``BATCH_MIN`` or more observations
  at once, an outage (DBAA and DAAA: October 2016 to March 2017, posted together on 2017-03-22).

``pre_archive_lag`` = 1 + the largest regular lag. A value FRED posted *before* its own date (a
non-trading day filled with the previous close, e.g. VIXCLS on NYSE holidays) has lag 0 and is
flagged ``posted_early``. The audit (table ``archive``, one row per archived observation, and
the report) records first and last values, the number of distinct values, the largest revision,
the lag and the flags.

Assumptions and limitations
---------------------------
- Before the archive, point-in-time correctness rests on the assumption that FRED posted as it
  regularly did later. An outage before the archive would not be seen.
- Before the archive the current value is used; a value revised after posting is used in its
  revised form. The audit reports how often and how much archived values were revised.
- The two exclusions were defined after the lags were measured (and before any stage ran on
  real data); docs/adr/0019 records why.

Reference: FRED/ALFRED API, https://fred.stlouisfed.org/docs/api/fred/ ; Croushore, D. and
Stark, T. (2001), "A real-time data set for macroeconomists", Journal of Econometrics 105(1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from structlog.typing import FilteringBoundLogger

from invariant.core.logging import get_logger
from invariant.data.sources.alfred import VINTAGE_COLUMNS, validate_vintages
from invariant.data.sources.base import (
    DATE_DTYPE,
    DataValidationError,
    SourceName,
    SourcePayload,
    library_versions,
)
from invariant.data.sources.clients import ALFRED_REALTIME_END, FredMarketClient, FredMarketRaw
from invariant.data.sources.etf import session_calendar

__all__ = [
    "ARCHIVE_COLUMNS",
    "BATCH_MIN",
    "OBSERVATION_COLUMNS",
    "FredMarketReport",
    "MarketSeriesReport",
    "audit_archive",
    "fetch_fred_market",
    "merge_chunks",
    "parse_current",
]

OBSERVATION_COLUMNS = ("series_id", "observation_date", "value")
ARCHIVE_COLUMNS = (
    "series_id",
    "observation_date",
    "first_realtime_start",
    "first_value",
    "last_value",
    "n_values",
    "max_abs_revision",
    "posting_lag_sessions",
    "backfilled",
    "batch_posted",
    "posted_early",
)
BATCH_MIN: Final = 10
"""Observations first posted on one date at or above which the posting is an outage batch."""
_MISSING = "."
_OPEN_END: Final = np.datetime64(ALFRED_REALTIME_END, "D").astype(DATE_DTYPE)


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MarketSeriesReport(_Frozen):
    series_id: str
    n_observations: int
    n_missing_values: int
    first_observation: date
    last_observation: date
    first_vintage: date
    n_archived: int
    """Archived observations dated on or after the first vintage."""
    posting_lag_counts: dict[str, int]
    """Posting lag (sessions, as a string key) -> number of archived observations."""
    max_posting_lag: int
    n_backfilled: int
    n_batch_posted: int
    n_posted_early: int
    max_regular_lag: int
    """Largest lag over observations neither backfilled nor batch-posted."""
    pre_archive_lag: int
    """Sessions after its date at which a pre-archive value is usable: 1 + max_regular_lag."""
    n_revised: int
    """Archived observations whose published value changed."""
    max_abs_revision: float
    n_current_mismatch: int
    """Archived observations whose current value differs from their last archived value."""
    requests: int


class FredMarketReport(_Frozen):
    series: tuple[MarketSeriesReport, ...]

    def get(self, series_id: str) -> MarketSeriesReport:
        for s in self.series:
            if s.series_id == series_id:
                return s
        raise KeyError(series_id)


def _value(text: str, series_id: str) -> float:
    try:
        return np.nan if text == _MISSING else float(text)
    except ValueError:
        raise DataValidationError(f"{series_id}: non-numeric value {text!r}") from None


def _date_column(rows: Sequence[Mapping[str, str]], column: str, series_id: str) -> pd.Series:
    try:
        return pd.to_datetime(pd.Series([r[column] for r in rows]), format="%Y-%m-%d").astype(
            DATE_DTYPE
        )
    except ValueError as exc:
        raise DataValidationError(f"{series_id}: invalid {column}: {exc}") from None


def parse_current(raw: FredMarketRaw) -> pd.DataFrame:
    """The current vintage as an ``OBSERVATION_COLUMNS`` table, sorted, one row per date."""
    sid = raw.series_id
    if not raw.current:
        raise DataValidationError(f"{sid}: no observations returned")
    frame = pd.DataFrame(
        {
            "series_id": pd.Series([sid] * len(raw.current), dtype="str"),
            "observation_date": _date_column(raw.current, "date", sid),
            "value": np.asarray([_value(r["value"], sid) for r in raw.current], dtype="float64"),
        }
    )
    if frame["observation_date"].duplicated().any():
        raise DataValidationError(f"{sid}: duplicate observation dates")
    return frame.sort_values("observation_date", kind="stable").reset_index(drop=True)


def merge_chunks(raw: FredMarketRaw, today: date) -> pd.DataFrame:
    """The archive's yearly chunks as one ALFRED ``VINTAGE_COLUMNS`` table (module docstring).

    Consecutive rows of one observation merge when the second starts the day after the first
    ends and has the same value (missing equals missing). A row ending on ``today`` is open.
    """
    sid = raw.series_id
    if not raw.archive:
        raise DataValidationError(f"{sid}: no archived vintages")
    rows = pd.DataFrame(
        {
            "observation_date": _date_column(raw.archive, "date", sid),
            "realtime_start": _date_column(raw.archive, "realtime_start", sid),
            "realtime_end": _date_column(raw.archive, "realtime_end", sid),
            "value": np.asarray([_value(r["value"], sid) for r in raw.archive], dtype="float64"),
        }
    ).sort_values(["observation_date", "realtime_start"], kind="stable")
    rows = rows.drop_duplicates(["observation_date", "realtime_start"]).reset_index(drop=True)
    obs = rows["observation_date"].to_numpy()
    start = rows["realtime_start"].to_numpy()
    end = rows["realtime_end"].to_numpy()
    val = rows["value"].to_numpy()
    same = np.r_[False, obs[1:] == obs[:-1]]
    adjacent = np.r_[False, start[1:] == end[:-1] + np.timedelta64(1, "D")]
    equal = np.r_[False, (val[1:] == val[:-1]) | (np.isnan(val[1:]) & np.isnan(val[:-1]))]
    new_run = ~(same & adjacent & equal)
    run = np.cumsum(new_run) - 1
    merged = pd.DataFrame(
        {
            "observation_date": obs[new_run],
            "realtime_start": start[new_run],
            "realtime_end": pd.Series(end).groupby(run).max().to_numpy(),
            "value": val[new_run],
        }
    )
    today64 = np.datetime64(today.isoformat(), "D").astype(DATE_DTYPE)
    current = merged["realtime_end"].to_numpy() >= today64
    merged["realtime_end"] = np.where(current, _OPEN_END, merged["realtime_end"].to_numpy())
    merged.insert(0, "series_id", pd.Series([sid] * len(merged), dtype="str"))
    merged = merged.astype({"observation_date": DATE_DTYPE, "realtime_start": DATE_DTYPE})
    merged["realtime_end"] = merged["realtime_end"].astype(DATE_DTYPE)
    return validate_vintages(merged[list(VINTAGE_COLUMNS)], sid)


def audit_archive(
    vintages: pd.DataFrame, first_vintage: date, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """One ``ARCHIVE_COLUMNS`` row per observation dated on or after ``first_vintage``.

    ``vintages`` is one series' ``VINTAGE_COLUMNS`` table; ``sessions`` must cover every
    observation and posting date.
    """
    sid = str(vintages["series_id"].iloc[0]) if len(vintages) else "?"
    fv = np.datetime64(first_vintage.isoformat(), "D").astype(DATE_DTYPE)
    table = vintages[vintages["observation_date"].to_numpy() >= fv]
    if table.empty:
        raise DataValidationError(f"{sid}: no observation dated on or after the first vintage")
    # The first row as published, missing or not (groupby().first() would skip a NaN).
    first_any = table.drop_duplicates("observation_date", keep="first").set_index(
        "observation_date"
    )
    valued = table[table["value"].notna()]
    grouped = valued.groupby("observation_date", sort=True)["value"]
    out = pd.DataFrame(
        {
            "first_realtime_start": valued.groupby("observation_date", sort=True)[
                "realtime_start"
            ].min(),
            "first_value": grouped.first(),
            "last_value": grouped.last(),
            "n_values": grouped.nunique(),
            "max_abs_revision": grouped.max() - grouped.min(),
        }
    )
    out["backfilled"] = first_any.loc[out.index, "value"].isna().to_numpy()
    obs = out.index.to_numpy(dtype=DATE_DTYPE)
    posted = out["first_realtime_start"].to_numpy(dtype=DATE_DTYPE)
    sess = sessions.to_numpy(dtype=DATE_DTYPE)
    if len(sess) == 0 or obs.min() < sess[0] or posted.max() > sess[-1]:
        raise DataValidationError(f"{sid}: the session calendar does not cover the archive")
    lag = np.searchsorted(sess, posted, side="right") - np.searchsorted(sess, obs, side="right")
    # A value posted before its own date is FRED filling a non-trading day with the previous
    # close (VIXCLS on NYSE holidays). Its information is older than its date, so lag 0.
    out["posted_early"] = posted < obs
    out["posting_lag_sessions"] = np.maximum(lag, 0).astype("int64")
    first_posts = pd.Series(posted).map(pd.Series(posted).value_counts()).to_numpy()
    out["batch_posted"] = first_posts >= BATCH_MIN
    out = out.reset_index()
    out.insert(0, "series_id", pd.Series([sid] * len(out), dtype="str"))
    out["n_values"] = out["n_values"].astype("int64")
    return out[list(ARCHIVE_COLUMNS)]


def _report(raw: FredMarketRaw, current: pd.DataFrame, audit: pd.DataFrame) -> MarketSeriesReport:
    lags = audit["posting_lag_sessions"]
    regular = lags[~(audit["backfilled"] | audit["batch_posted"])]
    if regular.empty:
        raise DataValidationError(f"{raw.series_id}: no regular posting to measure a lag on")
    merged = audit.merge(current, on="observation_date", how="inner")
    mismatch = merged["value"].notna() & (merged["value"] != merged["last_value"])
    obs = pd.DatetimeIndex(current["observation_date"])
    return MarketSeriesReport(
        series_id=raw.series_id,
        n_observations=len(current),
        n_missing_values=int(current["value"].isna().sum()),
        first_observation=obs.min().date(),
        last_observation=obs.max().date(),
        first_vintage=date.fromisoformat(raw.first_vintage),
        n_archived=len(audit),
        posting_lag_counts={str(k): int(v) for k, v in lags.value_counts().sort_index().items()},
        max_posting_lag=int(lags.max()),
        n_backfilled=int(audit["backfilled"].sum()),
        n_batch_posted=int(audit["batch_posted"].sum()),
        n_posted_early=int(audit["posted_early"].sum()),
        max_regular_lag=int(regular.max()),
        pre_archive_lag=int(regular.max()) + 1,
        n_revised=int((audit["n_values"] > 1).sum()),
        max_abs_revision=float(audit["max_abs_revision"].max()),
        n_current_mismatch=int(mismatch.sum()),
        requests=raw.requests,
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def fetch_fred_market(
    client: FredMarketClient,
    series_ids: Sequence[str],
    start: date,
    end: date,
    *,
    calendar: str,
    now: Callable[[], datetime] = _utcnow,
) -> SourcePayload:
    """Fetch ``series_ids`` for observations dated in ``[start, end]``; return the
    ``fred_market`` payload (tables ``observations``, ``vintages`` and ``archive``)."""
    if not series_ids:
        raise ValueError("series_ids must not be empty")
    fetched_at = now()
    today = fetched_at.date()
    sessions = session_calendar(calendar, start, today)
    currents: list[pd.DataFrame] = []
    vintage_tables: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    reports: list[MarketSeriesReport] = []
    for sid in series_ids:
        raw = client.market_series(sid, start, end, today)
        if raw.series_id != sid:
            raise DataValidationError(f"requested {sid}, client returned {raw.series_id}")
        current = parse_current(raw)
        vintages = merge_chunks(raw, today)
        audit = audit_archive(vintages, date.fromisoformat(raw.first_vintage), sessions)
        rep = _report(raw, current, audit)
        _log().info(
            "fred_market_audit",
            series_id=sid,
            pre_archive_lag=rep.pre_archive_lag,
            max_posting_lag=rep.max_posting_lag,
            n_backfilled=rep.n_backfilled,
            n_batch_posted=rep.n_batch_posted,
            n_revised=rep.n_revised,
        )
        currents.append(current)
        vintage_tables.append(vintages)
        audits.append(audit)
        reports.append(rep)
    return SourcePayload(
        source=SourceName.FRED_MARKET,
        params={
            "series": list(series_ids),
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "realtime_end": today.isoformat(),
            "calendar": calendar,
            "endpoint": "fred/series/observations",
        },
        report=FredMarketReport(series=tuple(reports)),
        tables={
            "observations": pd.concat(currents, ignore_index=True)[list(OBSERVATION_COLUMNS)],
            "vintages": pd.concat(vintage_tables, ignore_index=True)[list(VINTAGE_COLUMNS)],
            "archive": pd.concat(audits, ignore_index=True)[list(ARCHIVE_COLUMNS)],
        },
        fetched_at=fetched_at,
        library_versions=library_versions("httpx", "exchange-calendars"),
    )
