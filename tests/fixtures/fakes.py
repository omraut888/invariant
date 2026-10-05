"""Offline fakes and fixture builders for the data layer. No function here touches the network."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from invariant.config.settings import FrenchConfig
from invariant.data.sources.clients import AlfredRawSeries

FIXTURES = Path(__file__).resolve().parent
FIVE_FACTOR_CSV = FIXTURES / "french" / "F-F_Research_Data_5_Factors_2x3_daily.csv"
MOMENTUM_CSV = FIXTURES / "french" / "F-F_Momentum_Factor_daily.csv"
FRENCH_CONFIG = FrenchConfig(
    five_factor_url="https://example.test/ff5.zip",
    momentum_url="https://example.test/mom.zip",
)
FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def fixed_now() -> datetime:
    return FIXED_NOW


def zip_bytes(members: Mapping[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return buf.getvalue()


def french_zips() -> dict[str, bytes]:
    """URL -> zip bytes for both fixture CSVs, keyed by ``FRENCH_CONFIG`` URLs."""
    return {
        FRENCH_CONFIG.five_factor_url: zip_bytes(
            {FIVE_FACTOR_CSV.name: FIVE_FACTOR_CSV.read_text()}
        ),
        FRENCH_CONFIG.momentum_url: zip_bytes({MOMENTUM_CSV.name: MOMENTUM_CSV.read_text()}),
    }


class FakeHttp:
    """``HttpClient`` serving fixed bodies by URL and recording requests."""

    def __init__(self, bodies: Mapping[str, bytes]) -> None:
        self.bodies = dict(bodies)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_bytes(self, url: str, params: Mapping[str, str] | None = None) -> bytes:
        self.calls.append((url, dict(params or {})))
        return self.bodies[url]


def price_frame(
    dates: Sequence[str],
    adj_close: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """A yfinance-shaped frame: (field, ticker) columns; close = adj_close * 1.01, volume 1e6."""
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)), name="Date")
    cols: dict[tuple[str, str], np.ndarray[tuple[int], np.dtype[np.float64]]] = {}
    for ticker, values in adj_close.items():
        adj = np.asarray(values, dtype="float64")
        cols[("Adj Close", ticker)] = adj
        cols[("Close", ticker)] = adj * 1.01
        cols[("Volume", ticker)] = np.where(np.isnan(adj), np.nan, 1e6)
    frame = pd.DataFrame(cols, index=index)
    frame.columns = pd.MultiIndex.from_tuples(list(cols), names=["Price", "Ticker"])
    return frame


class FakePriceClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[tuple[str, ...], date, date]] = []

    def download(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self.calls.append((tuple(tickers), start, end))
        return self.frame


def alfred_row(obs: str, start: str, end: str, value: str) -> dict[str, str]:
    return {"date": obs, "realtime_start": start, "realtime_end": end, "value": value}


def alfred_raw(series_id: str, rows: Sequence[Mapping[str, str]]) -> AlfredRawSeries:
    return AlfredRawSeries(series_id=series_id, rows=tuple(rows), count=len(rows), pages=1)


UNRATE_ROWS = (
    alfred_row("2025-08-01", "2025-09-05", "2025-10-02", "4.3"),
    alfred_row("2025-08-01", "2025-10-03", "9999-12-31", "4.2"),
    alfred_row("2025-09-01", "2025-10-03", "9999-12-31", "."),
    alfred_row("2025-07-01", "2025-08-01", "9999-12-31", "4.2"),
)


class FakeAlfredClient:
    def __init__(self, series: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        self.series = dict(series)

    def all_releases(self, series_id: str) -> AlfredRawSeries:
        return alfred_raw(series_id, self.series[series_id])
