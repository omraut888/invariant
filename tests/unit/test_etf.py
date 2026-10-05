from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.fixtures.fakes import FakePriceClient, fixed_now, price_frame

from invariant.core.types import TimeSeriesValidationError
from invariant.data.sources.base import DataValidationError, SourceName
from invariant.data.sources.etf import (
    PRICE_COLUMNS,
    EtfReport,
    adj_close_panel,
    coverage_report,
    fetch_etf_prices,
    log_returns,
    parse_price_frame,
    session_calendar,
)

# XNYS sessions 2025-01-02 .. 2025-01-14: 2025-01-09 closed (national day of mourning).
SESSIONS = [
    "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
    "2025-01-10", "2025-01-13", "2025-01-14",
]  # fmt: skip
START, END = date(2025, 1, 1), date(2025, 1, 14)
NAN = float("nan")


def _sessions() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(SESSIONS).astype("datetime64[us]"), name="date")


# --- calendar ---------------------------------------------------------------------------------


def test_session_calendar_exact_xnys() -> None:
    sessions = session_calendar("XNYS", START, END)  # starts on a holiday, not a session
    assert list(sessions.strftime("%Y-%m-%d")) == SESSIONS
    assert sessions.dtype == "datetime64[us]"
    assert sessions.tz is None


def test_session_calendar_reaches_back_before_default_window() -> None:
    # exchange-calendars defaults to 20 years of history; the config window starts in 2005.
    sessions = session_calendar("XNYS", date(2005, 1, 3), date(2005, 1, 7))
    assert len(sessions) == 5


# --- parsing ----------------------------------------------------------------------------------


def test_parse_to_long_table() -> None:
    raw = price_frame(SESSIONS[:3], {"SPY": [1.0, 2.0, 3.0], "TLT": [4.0, NAN, 6.0]})
    long = parse_price_frame(raw, ["SPY", "TLT"])
    assert tuple(long.columns) == PRICE_COLUMNS
    assert len(long) == 5  # the all-NaN TLT row is dropped
    assert long["date"].dtype == "datetime64[us]"
    tlt = long[long["ticker"] == "TLT"]
    assert list(tlt["adj_close"]) == [4.0, 6.0]
    assert tlt["close"].iloc[0] == pytest.approx(4.04)


def test_parse_ignores_unrequested_and_absent_tickers() -> None:
    raw = price_frame(SESSIONS[:2], {"SPY": [1.0, 2.0], "XLK": [3.0, 4.0]})
    long = parse_price_frame(raw, ["SPY", "GLD"])
    assert set(long["ticker"]) == {"SPY"}


def test_parse_tz_aware_index_keeps_local_date() -> None:
    raw = price_frame(SESSIONS[:2], {"SPY": [1.0, 2.0]})
    raw.index = pd.DatetimeIndex(raw.index).tz_localize("America/New_York")
    long = parse_price_frame(raw, ["SPY"])
    assert list(long["date"].dt.strftime("%Y-%m-%d")) == SESSIONS[:2]


def test_parse_empty_frame() -> None:
    long = parse_price_frame(pd.DataFrame(), ["SPY"])
    assert long.empty
    assert tuple(long.columns) == PRICE_COLUMNS


@pytest.mark.parametrize("field", ["Close", "Adj Close"])
@pytest.mark.parametrize("bad", [0.0, -2.0])
def test_parse_rejects_non_positive_prices(field: str, bad: float) -> None:
    raw = price_frame(SESSIONS[:2], {"SPY": [1.0, 2.0]})
    raw.loc[raw.index[1], (field, "SPY")] = bad
    name = {"Close": "close", "Adj Close": "adj_close"}[field]
    with pytest.raises(DataValidationError, match=f"non-positive {name} {bad} for SPY"):
        parse_price_frame(raw, ["SPY"])


def test_parse_rejects_malformed_frames() -> None:
    raw = price_frame([SESSIONS[0], SESSIONS[0]], {"SPY": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="duplicate"):
        parse_price_frame(raw, ["SPY"])
    flat = pd.DataFrame({"SPY": [1.0]}, index=pd.to_datetime([SESSIONS[0]]))
    with pytest.raises(DataValidationError, match="MultiIndex"):
        parse_price_frame(flat, ["SPY"])
    no_adj = price_frame(SESSIONS[:1], {"SPY": [1.0]}).drop(columns="Adj Close", level=0)
    with pytest.raises(DataValidationError, match="lacks fields"):
        parse_price_frame(no_adj, ["SPY"])
    intraday = price_frame(SESSIONS[:1], {"SPY": [1.0]})
    intraday.index = intraday.index + pd.Timedelta(hours=9)
    with pytest.raises(DataValidationError, match="midnight"):
        parse_price_frame(intraday, ["SPY"])


# --- coverage ---------------------------------------------------------------------------------


def test_coverage_distinguishes_late_start_holes_and_off_calendar() -> None:
    # SPY: complete. TLT: starts late (2 leading misses), one interior hole, and a row on the
    # 2025-01-09 closure (off-calendar).
    dates = [*SESSIONS[:5], "2025-01-09", *SESSIONS[5:]]
    raw = price_frame(
        dates,
        {
            "SPY": [1, 2, 3, 4, 5, NAN, 6, 7, 8],
            "TLT": [NAN, NAN, 3, NAN, 5, 9, 6, 7, 8],
        },
    )
    long = parse_price_frame(raw, ["SPY", "TLT"])
    report = coverage_report(
        long, _sessions(), ["SPY", "TLT"], calendar="XNYS", start=START, end=END
    )
    spy, tlt = report.tickers
    assert spy.complete
    assert (spy.n_present, spy.n_sessions) == (8, 8)
    assert not tlt.complete
    assert tlt.n_present == 5
    assert [str(d) for d in tlt.missing_dates] == ["2025-01-02", "2025-01-03", "2025-01-07"]
    assert tlt.n_missing_interior == 1
    assert str(tlt.first_date) == "2025-01-06"
    assert [str(d) for d in tlt.off_calendar_dates] == ["2025-01-09"]
    assert not report.complete


def test_coverage_catches_a_session_missing_from_every_ticker() -> None:
    dates = [d for d in SESSIONS if d != "2025-01-07"]
    raw = price_frame(dates, {"SPY": [1.0] * 7, "TLT": [2.0] * 7})
    long = parse_price_frame(raw, ["SPY", "TLT"])
    report = coverage_report(
        long, _sessions(), ["SPY", "TLT"], calendar="XNYS", start=START, end=END
    )
    assert all([str(d) for d in t.missing_dates] == ["2025-01-07"] for t in report.tickers)


# --- fetch ------------------------------------------------------------------------------------


def test_fetch_etf_prices_payload() -> None:
    one_to_eight = [float(x) for x in range(1, 9)]
    raw = price_frame(SESSIONS, {"SPY": one_to_eight, "GLD": [NAN, *one_to_eight[1:]]})
    client = FakePriceClient(raw)
    payload = fetch_etf_prices(client, ["SPY", "GLD"], START, END, now=fixed_now)
    assert client.calls == [(("SPY", "GLD"), START, END)]
    assert payload.source is SourceName.ETF
    assert set(payload.tables) == {"prices", "sessions"}
    assert len(payload.tables["sessions"]) == 8
    report = payload.report
    assert isinstance(report, EtfReport)
    assert [t.ticker for t in report.tickers] == ["SPY", "GLD"]
    assert [str(d) for d in report.tickers[1].missing_dates] == ["2025-01-02"]
    assert payload.params["tickers"] == ["SPY", "GLD"]
    assert payload.params["calendar"] == "XNYS"


def test_fetch_raises_when_a_ticker_has_no_data() -> None:
    raw = price_frame(SESSIONS, {"SPY": [float(x) for x in range(1, 9)], "GLD": [NAN] * 8})
    with pytest.raises(DataValidationError, match=r"no price data returned for \['GLD'\]"):
        fetch_etf_prices(FakePriceClient(raw), ["SPY", "GLD"], START, END)


# --- transforms -------------------------------------------------------------------------------


def test_panel_marks_missing_sessions_as_nan_and_drops_off_calendar() -> None:
    dates = [*SESSIONS[:3], "2025-01-09"]
    long = parse_price_frame(price_frame(dates, {"SPY": [1.0, 2.0, 3.0, 9.0]}), ["SPY"])
    panel = adj_close_panel(long, _sessions(), ["SPY", "TLT"])
    assert list(panel.index) == list(_sessions())
    assert list(panel.columns) == ["SPY", "TLT"]
    assert panel["SPY"].iloc[:3].tolist() == [1.0, 2.0, 3.0]
    assert panel["SPY"].iloc[3:].isna().all()
    assert panel["TLT"].isna().all()
    assert 9.0 not in panel["SPY"].to_numpy()


def test_log_returns_values_and_gaps() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, NAN, 121.0, 133.1]}, index=_sessions()[:5])
    r = log_returns(prices)["A"]
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert np.isnan(r.iloc[2]) and np.isnan(r.iloc[3])  # never stretched across the gap
    assert r.iloc[4] == pytest.approx(np.log(1.1))


def test_log_returns_rejects_bad_input() -> None:
    with pytest.raises(DataValidationError, match="positive"):
        log_returns(pd.DataFrame({"A": [1.0, 0.0]}, index=_sessions()[:2]))
    with pytest.raises(TimeSeriesValidationError):
        log_returns(pd.DataFrame({"A": [1.0, 2.0]}, index=_sessions()[:2][::-1]))


@given(st.lists(st.floats(min_value=0.01, max_value=1e4), min_size=2, max_size=8))
def test_log_returns_telescope(prices: list[float]) -> None:
    frame = pd.DataFrame({"A": prices}, index=_sessions()[: len(prices)])
    total = log_returns(frame)["A"].iloc[1:].sum()
    assert total == pytest.approx(np.log(prices[-1] / prices[0]), abs=1e-9)
