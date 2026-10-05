"""Daily ETF prices from Yahoo Finance (via ``PriceClient``), coverage reporting, log returns.

Method
------
``fetch_etf_prices`` downloads ``Close``, ``Adj Close`` and ``Volume`` for the universe and
stores them as a long table ``(date, ticker, close, adj_close, volume)``, one row per ticker per
date Yahoo returned. Coverage is checked against the *exact* exchange session calendar
(exchange-calendars, default XNYS), not against the union of the tickers' own dates, so a
session missing from every ticker is still caught. Nothing is filled: a session without an
adjusted close is listed in the ``EtfReport`` and logged; a ticker with no data at all is an
error. Dates Yahoo returns that are not sessions are reported and kept in the raw table, then
excluded (with a warning) from the session-aligned panel.

``log_returns`` computes r_t = ln(P_t / P_{t-1}) on a session-aligned panel. A missing price at
t makes r_t and r_{t+1} NaN; a return is never stretched across a gap.

Assumptions and limitations
---------------------------
- Adjusted price *levels* are not point-in-time. Yahoo back-adjusts history multiplicatively for
  every split and dividend, so a level seen today embeds distributions paid after that date.
  Returns between adjacent sessions are unaffected (the common adjustment factor cancels), so
  only returns, never adjusted levels, may be used as features.
- Yahoo revises and rounds its history without notice; the snapshot is the only record of what
  was fetched.
- Survivorship: the universe was chosen knowing every ticker still trades (configs/base.yaml).
- Exchange calendars are code: a new special closure (e.g. a national day of mourning) appears
  only after the library is updated. The snapshot stores the session list it used.

See docs/adr/0004-data-sources-and-pit-limitations.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from structlog.typing import FilteringBoundLogger

from invariant.core.logging import get_logger
from invariant.core.types import Frequency, validate_time_index
from invariant.data.sources.base import (
    DATE_DTYPE,
    DataValidationError,
    SourceName,
    SourcePayload,
    library_versions,
)
from invariant.data.sources.clients import PriceClient

__all__ = [
    "PRICE_COLUMNS",
    "EtfReport",
    "TickerCoverage",
    "adj_close_panel",
    "coverage_report",
    "fetch_etf_prices",
    "log_returns",
    "parse_price_frame",
    "session_calendar",
]

_FIELDS = {"Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
PRICE_COLUMNS = ("date", "ticker", "close", "adj_close", "volume")


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TickerCoverage(_Frozen):
    """Coverage of one ticker against the session calendar.

    ``missing_dates`` lists every session in the window without an adjusted close, including
    sessions before the first or after the last observation; ``n_missing_interior`` counts only
    those strictly between ``first_date`` and ``last_date`` (holes, as opposed to late listing).
    """

    ticker: str
    n_sessions: int
    n_present: int
    first_date: date | None
    last_date: date | None
    missing_dates: tuple[date, ...]
    n_missing_interior: int
    off_calendar_dates: tuple[date, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_dates and not self.off_calendar_dates


class EtfReport(_Frozen):
    calendar: str
    start: date
    end: date
    n_sessions: int
    tickers: tuple[TickerCoverage, ...]

    @property
    def complete(self) -> bool:
        return all(t.complete for t in self.tickers)


# ---------------------------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------------------------


def session_calendar(calendar: str, start: date, end: date) -> pd.DatetimeIndex:
    """Sessions of ``calendar`` in ``[start, end]`` as a tz-naive ``DATE_DTYPE`` index.

    The calendar is built with explicit bounds: exchange-calendars defaults to the last 20
    years only, which would silently truncate the 2005 start of the default window.
    """
    import exchange_calendars as xc  # lazy: slow to import

    cal = xc.get_calendar(calendar, start=start.isoformat(), end=end.isoformat())
    # Built with these bounds, the calendar's own sessions are exactly those in [start, end];
    # sessions_in_range would reject a start/end that is not itself a session.
    sessions = cal.sessions
    out = pd.DatetimeIndex(sessions.tz_localize(None) if sessions.tz else sessions)
    return pd.DatetimeIndex(out.astype(DATE_DTYPE), name="date")


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


def parse_price_frame(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Convert a ``PriceClient`` frame to the long ``PRICE_COLUMNS`` table.

    Rows where all of close, adj_close and volume are NaN are dropped (they carry no data and
    are reported as missing by ``coverage_report``). A ticker absent from ``raw`` yields no
    rows. Raises ``DataValidationError`` on a malformed frame, duplicate dates, non-positive
    prices or negative volume.
    """
    empty = pd.DataFrame(
        {
            "date": pd.Series(dtype=DATE_DTYPE),
            "ticker": pd.Series(dtype="str"),
            "close": pd.Series(dtype="float64"),
            "adj_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    if raw.empty:
        return empty
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise DataValidationError(f"price index must be a DatetimeIndex, got {type(raw.index)}")
    if not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels != 2:
        raise DataValidationError("price columns must be a (field, ticker) MultiIndex")
    index = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    if not (index == index.normalize()).all():
        raise DataValidationError("daily price timestamps must be midnight dates")
    if index.has_duplicates:
        raise DataValidationError(f"duplicate price dates, e.g. {index[index.duplicated()][0]}")
    missing_fields = set(_FIELDS) - set(raw.columns.get_level_values(0))
    if missing_fields:
        raise DataValidationError(f"price frame lacks fields {sorted(missing_fields)}")

    parts: list[pd.DataFrame] = []
    present = set(raw.columns.get_level_values(1))
    for ticker in tickers:
        if ticker not in present:
            continue
        cols = {
            new: pd.to_numeric(raw[(old, ticker)], errors="raise").to_numpy(dtype="float64")
            for old, new in _FIELDS.items()
        }
        part = pd.DataFrame({"date": index.astype(DATE_DTYPE), "ticker": ticker, **cols})
        parts.append(part.dropna(how="all", subset=["close", "adj_close", "volume"]))
    if not parts:
        return empty
    long = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"], kind="stable")
    long = long.astype({"ticker": "str"}).reset_index(drop=True)[list(PRICE_COLUMNS)]
    for col in ("close", "adj_close"):
        bad = long[long[col] <= 0]
        if not bad.empty:
            row = bad.iloc[0]
            raise DataValidationError(
                f"non-positive {col} {row[col]} for {row['ticker']} on {row['date'].date()}"
            )
    if (long["volume"] < 0).any():
        raise DataValidationError("negative volume in price data")
    return long


# ---------------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------------


def coverage_report(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    tickers: Sequence[str],
    *,
    calendar: str,
    start: date,
    end: date,
) -> EtfReport:
    """Per-ticker coverage of ``prices`` (long table) against ``sessions``.

    A session counts as present iff the ticker has a non-NaN ``adj_close`` on it.
    """
    coverage: list[TickerCoverage] = []
    for ticker in tickers:
        rows = prices[prices["ticker"] == ticker]
        dates = pd.DatetimeIndex(rows.loc[rows["adj_close"].notna(), "date"])
        on_cal = dates[dates.isin(sessions)]
        off_cal = pd.DatetimeIndex(rows["date"])
        off_cal = off_cal[~off_cal.isin(sessions)]
        missing = sessions[~sessions.isin(on_cal)]
        first: pd.Timestamp | None
        last: pd.Timestamp | None
        if len(on_cal):
            first, last = on_cal.min(), on_cal.max()
            interior = int(((missing > first) & (missing < last)).sum())
        else:
            first = last = None
            interior = 0
        coverage.append(
            TickerCoverage(
                ticker=ticker,
                n_sessions=len(sessions),
                n_present=len(on_cal),
                first_date=first.date() if first is not None else None,
                last_date=last.date() if last is not None else None,
                missing_dates=tuple(d.date() for d in missing),
                n_missing_interior=interior,
                off_calendar_dates=tuple(d.date() for d in off_cal),
            )
        )
    return EtfReport(
        calendar=calendar,
        start=start,
        end=end,
        n_sessions=len(sessions),
        tickers=tuple(coverage),
    )


# ---------------------------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def fetch_etf_prices(
    client: PriceClient,
    tickers: Sequence[str],
    start: date,
    end: date,
    *,
    calendar: str = "XNYS",
    now: Callable[[], datetime] = _utcnow,
) -> SourcePayload:
    """Fetch, parse and coverage-check daily prices; return the ``etf`` snapshot payload.

    Tables: ``prices`` (long, ``PRICE_COLUMNS``) and ``sessions`` (the calendar used).
    Raises ``DataValidationError`` if any ticker returned no data at all.
    """
    fetched_at = now()
    raw = client.download(tickers, start, end)
    prices = parse_price_frame(raw, tickers)
    sessions = session_calendar(calendar, start, end)
    report = coverage_report(prices, sessions, tickers, calendar=calendar, start=start, end=end)
    empty = [t.ticker for t in report.tickers if t.n_present == 0]
    if empty:
        raise DataValidationError(f"no price data returned for {empty} in [{start}, {end}]")
    for cov in report.tickers:
        if not cov.complete:
            _log().warning(
                "etf_missing_data",
                ticker=cov.ticker,
                n_missing=len(cov.missing_dates),
                n_missing_interior=cov.n_missing_interior,
                first_missing=str(cov.missing_dates[0]) if cov.missing_dates else None,
                n_off_calendar=len(cov.off_calendar_dates),
            )
    _log().info("etf_fetched", tickers=len(tickers), rows=len(prices), complete=report.complete)
    return SourcePayload(
        source=SourceName.ETF,
        params={
            "tickers": list(tickers),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "calendar": calendar,
            "provider": "yahoo",
            "fields": list(sorted(_FIELDS.values())),
        },
        report=report,
        tables={"prices": prices, "sessions": pd.DataFrame({"date": sessions.to_numpy()})},
        fetched_at=fetched_at,
        library_versions=library_versions("yfinance", "exchange-calendars"),
    )


# ---------------------------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------------------------


def adj_close_panel(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Wide ``sessions x tickers`` adjusted-close panel; NaN where a session has no price.

    Off-calendar rows are excluded with a warning (they are listed in the ``EtfReport``).
    The NaNs are explicit absences, not fills: every row is a real session.
    """
    validate_time_index(pd.Series(0, index=sessions), Frequency.DAILY)
    off = ~prices["date"].isin(sessions)
    if off.any():
        _log().warning("etf_off_calendar_rows_excluded", n_rows=int(off.sum()))
    wide = prices.loc[~off].pivot(index="date", columns="ticker", values="adj_close")
    wide = wide.reindex(index=sessions, columns=list(tickers))
    wide.index.name = "date"
    wide.columns.name = "ticker"
    return wide.astype("float64")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns r_t = ln(P_t / P_{t-1}) of a session-aligned price panel.

    Assumes consecutive rows are consecutive sessions (as ``adj_close_panel`` guarantees).
    r_t uses only P_t and P_{t-1}; NaN in either gives NaN; the first row is NaN.
    Raises ``DataValidationError`` on non-positive prices.
    """
    validate_time_index(prices, Frequency.DAILY)
    values = prices.to_numpy(dtype="float64")
    if (values[~np.isnan(values)] <= 0).any():
        raise DataValidationError("log returns need strictly positive prices")
    logp = np.log(values)
    out = np.full_like(logp, np.nan)
    out[1:] = logp[1:] - logp[:-1]
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)
