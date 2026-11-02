"""The FRED market source: chunk merging, the posting-lag audit, the client (docs/adr/0019)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr
from tests.fixtures.fakes import FakeMarketClient, alfred_row, fixed_now, market_raw

from invariant.data.sources.base import DataValidationError, SourceName
from invariant.data.sources.clients import (
    FRED_OBSERVATIONS_URL,
    FRED_VINTAGEDATES_URL,
    FredMarketHttpClient,
    HttpError,
)
from invariant.data.sources.etf import session_calendar
from invariant.data.sources.fred_market import (
    BATCH_MIN,
    audit_archive,
    fetch_fred_market,
    merge_chunks,
    parse_current,
)

OPEN = "9999-12-31"
SESSIONS = session_calendar("XNYS", date(2023, 12, 1), date(2024, 3, 29))
TODAY = date(2024, 3, 29)

# First vintage 2024-01-10. 2024-01-15 is a holiday (MLK day).
ARCHIVE = (
    alfred_row("2024-01-05", "2024-01-10", OPEN, "0.5"),  # pre-archive, loaded on day one
    alfred_row("2024-01-10", "2024-01-11", OPEN, "1.0"),  # lag 1
    alfred_row("2024-01-11", "2024-01-12", "2024-01-18", "."),  # backfilled: lag 5
    alfred_row("2024-01-11", "2024-01-19", OPEN, "2.0"),
    alfred_row("2024-01-12", "2024-01-12", OPEN, "3.0"),  # posted the same day: lag 0
    alfred_row("2024-01-16", "2024-01-17", "2024-01-17", "1.5"),  # lag 1, revised next day
    alfred_row("2024-01-16", "2024-01-18", OPEN, "1.6"),
)
CURRENT = (
    ("2024-01-05", "0.5"),
    ("2024-01-10", "1.0"),
    ("2024-01-11", "2.0"),
    ("2024-01-12", "3.0"),
    ("2024-01-16", "1.6"),
)


def _raw(archive: tuple[Mapping[str, str], ...] = ARCHIVE) -> object:
    return market_raw("SER", CURRENT, archive, "2024-01-10")


def _audit(archive: tuple[Mapping[str, str], ...] = ARCHIVE) -> pd.DataFrame:
    raw = market_raw("SER", CURRENT, archive, "2024-01-10")
    return audit_archive(merge_chunks(raw, TODAY), date(2024, 1, 10), SESSIONS)


def test_chunks_split_at_a_year_boundary_merge_back() -> None:
    raw = market_raw(
        "SER",
        [("2023-12-01", "1.0")],
        [
            alfred_row("2023-12-01", "2023-12-04", "2023-12-31", "1.0"),
            alfred_row("2023-12-01", "2024-01-01", "2024-02-05", "1.0"),
            alfred_row("2023-12-01", "2024-02-06", "2024-03-29", "1.1"),
        ],
        "2023-12-04",
    )
    v = merge_chunks(raw, TODAY)
    assert list(v["realtime_start"].dt.strftime("%Y-%m-%d")) == ["2023-12-04", "2024-02-06"]
    assert list(v["realtime_end"].dt.strftime("%Y-%m-%d")) == ["2024-02-05", OPEN]
    assert list(v["value"]) == [1.0, 1.1]


def test_missing_rows_merge_like_values() -> None:
    raw = market_raw(
        "SER",
        [("2023-12-01", ".")],
        [
            alfred_row("2023-12-01", "2023-12-04", "2023-12-31", "."),
            alfred_row("2023-12-01", "2024-01-01", "2024-03-29", "."),
        ],
        "2023-12-04",
    )
    v = merge_chunks(raw, TODAY)
    assert len(v) == 1 and np.isnan(v["value"].iloc[0])


def test_the_audit_measures_lags_and_flags() -> None:
    a = _audit().set_index(a_col := "observation_date")
    assert a_col == "observation_date"
    assert pd.Timestamp("2024-01-05") not in a.index  # pre-archive: no lag measured
    lag = a["posting_lag_sessions"]
    assert lag[pd.Timestamp("2024-01-10")] == 1
    assert lag[pd.Timestamp("2024-01-11")] == 5  # 12, 16, 17, 18, 19 (15th is a holiday)
    assert lag[pd.Timestamp("2024-01-12")] == 0
    assert bool(a.loc[pd.Timestamp("2024-01-11"), "backfilled"])
    assert not a["backfilled"].drop(pd.Timestamp("2024-01-11")).any()
    rev = a.loc[pd.Timestamp("2024-01-16")]
    assert rev["n_values"] == 2 and rev["max_abs_revision"] == pytest.approx(0.1)
    assert rev["first_value"] == 1.5 and rev["last_value"] == 1.6
    assert not a["batch_posted"].any() and not a["posted_early"].any()


def test_a_value_posted_before_its_date_has_lag_zero_and_is_flagged() -> None:
    a = _audit((*ARCHIVE, alfred_row("2024-01-15", "2024-01-12", OPEN, "3.0")))
    row = a.set_index("observation_date").loc[pd.Timestamp("2024-01-15")]
    assert row["posting_lag_sessions"] == 0 and bool(row["posted_early"])


def test_a_batch_posting_is_flagged() -> None:
    days = SESSIONS[SESSIONS.searchsorted(pd.Timestamp("2024-02-01")) :][:BATCH_MIN]
    batch = tuple(alfred_row(d.strftime("%Y-%m-%d"), "2024-03-01", OPEN, "1.0") for d in days)
    a = _audit((*ARCHIVE, *batch)).set_index("observation_date")
    assert a.loc[days, "batch_posted"].all()
    assert not a.drop(days)["batch_posted"].any()
    short = _audit((*ARCHIVE, *batch[:-1])).set_index("observation_date")
    assert not short["batch_posted"].any()  # one fewer than BATCH_MIN is not a batch


def test_the_pre_archive_lag_ignores_backfills_and_batches() -> None:
    days = SESSIONS[SESSIONS.searchsorted(pd.Timestamp("2024-02-01")) :][:BATCH_MIN]
    batch = tuple(alfred_row(d.strftime("%Y-%m-%d"), "2024-03-01", OPEN, "1.0") for d in days)
    current = (*CURRENT, *((d.strftime("%Y-%m-%d"), "1.0") for d in days))
    raw = market_raw("SER", current, (*ARCHIVE, *batch), "2024-01-10")
    payload = fetch_fred_market(
        FakeMarketClient({"SER": raw}),
        ["SER"],
        date(2023, 12, 1),
        date(2024, 3, 1),
        calendar="XNYS",
        now=fixed_now,
    )
    rep = payload.report.get("SER")  # type: ignore[attr-defined]
    assert rep.max_posting_lag > 5  # the batch
    assert rep.n_backfilled == 1 and rep.n_batch_posted == BATCH_MIN
    assert rep.max_regular_lag == 1 and rep.pre_archive_lag == 2
    assert rep.n_revised == 1 and rep.n_current_mismatch == 0
    assert payload.source is SourceName.FRED_MARKET
    assert set(payload.tables) == {"observations", "vintages", "archive"}


def test_current_values_parse_missing_as_nan_and_refuse_duplicates() -> None:
    frame = parse_current(market_raw("SER", [("2024-01-02", "."), ("2024-01-03", "1")], [], "x"))
    assert np.isnan(frame["value"].iloc[0]) and frame["value"].iloc[1] == 1.0
    with pytest.raises(DataValidationError, match="duplicate"):
        parse_current(market_raw("SER", [("2024-01-02", "1"), ("2024-01-02", "2")], [], "x"))


# --- the client -------------------------------------------------------------------------------


class RoutedHttp:
    """``HttpClient`` answering by (url, params); records every request."""

    def __init__(self, route: Callable[[str, Mapping[str, str]], object]) -> None:
        self.route = route
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_bytes(self, url: str, params: Mapping[str, str] | None = None) -> bytes:
        p = dict(params or {})
        self.calls.append((url, p))
        out = self.route(url, p)
        if isinstance(out, Exception):
            raise out
        return json.dumps(out).encode()


def _route(url: str, p: Mapping[str, str]) -> object:
    if url == FRED_VINTAGEDATES_URL:
        return {"vintage_dates": ["2023-06-01"]}
    assert url == FRED_OBSERVATIONS_URL
    if "realtime_start" not in p:
        return {"count": 1, "observations": [{"date": "2023-06-01", "value": "1.0"}]}
    row = {"date": "2023-06-01", "realtime_start": p["realtime_start"]}
    return {"count": 1, "observations": [{**row, "realtime_end": p["realtime_end"], "value": "1"}]}


def test_the_client_requests_one_year_of_real_time_at_a_time() -> None:
    http = RoutedHttp(_route)
    sleeps: list[float] = []
    clock = iter(float(i) * 0.1 for i in range(100))
    client = FredMarketHttpClient(
        http, SecretStr("k"), min_interval_s=0.5, sleep=sleeps.append, clock=lambda: next(clock)
    )
    raw = client.market_series("SER", date(2023, 1, 1), date(2024, 12, 31), date(2025, 3, 1))
    windows = [
        (p["realtime_start"], p["realtime_end"]) for _, p in http.calls if "realtime_start" in p
    ]
    assert windows == [
        ("2023-06-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-03-01"),
    ]
    assert raw.first_vintage == "2023-06-01" and len(raw.archive) == 3
    assert raw.requests == len(http.calls) == 5
    assert sleeps and all(s > 0 for s in sleeps)  # spaced for the rate limit


def test_the_client_redacts_the_key() -> None:
    def fail(url: str, p: Mapping[str, str]) -> object:
        return HttpError(f"GET {url}?api_key={p['api_key']} failed", 400)

    client = FredMarketHttpClient(RoutedHttp(fail), SecretStr("secret-key"), min_interval_s=0)
    with pytest.raises(HttpError) as info:
        client.market_series("SER", date(2023, 1, 1), date(2023, 2, 1), date(2025, 3, 1))
    assert "secret-key" not in str(info.value) and info.value.__context__ is None
