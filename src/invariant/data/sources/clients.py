"""The network boundary: every byte the data layer fetches passes through one of these clients.

Source modules depend only on the ``Protocol``s, so unit tests pass fakes built from fixtures
and never touch the network. The concrete classes are thin adapters:

- ``HttpxClient`` (``HttpClient``): GET with a timeout and bounded retries. Transport errors and
  HTTP 429/5xx are retried with exponential backoff ``backoff_s * 2**attempt``; other non-2xx
  statuses fail immediately. Error messages and logs never include query strings, which carry
  API keys, and the underlying httpx exception is not chained for the same reason.
- ``FredAlfredClient`` (``AlfredClient``): the FRED ``series/observations`` endpoint with the
  full real-time period (``realtime_start=1776-07-04``, ``realtime_end=9999-12-31``), i.e. every
  ALFRED vintage. Pages through ``limit``/``offset`` until every row is retrieved, then checks
  that the row count equals the API-reported ``count``, that ``count`` did not change between
  pages (a release landing mid-fetch), and that no (date, realtime_start) row repeats (unstable
  ordering across pages). Any failure raises: a partial vintage history is worse than none.
- ``YFinanceClient`` (``PriceClient``): ``yfinance.download`` with ``auto_adjust=False`` (both
  ``Close`` and ``Adj Close``), ``repair=False`` (no heuristic data edits) and ``actions=False``.
  yfinance has no retry knob and reports per-ticker failures by returning all-NaN columns
  rather than raising; the ETF source detects those as missing data.

Reference: FRED API, https://fred.stlouisfed.org/docs/api/fred/series_observations.html
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import httpx
import pandas as pd
from pydantic import SecretStr
from structlog.typing import FilteringBoundLogger

from invariant.config.settings import HttpConfig
from invariant.core.logging import get_logger

__all__ = [
    "ALFRED_REALTIME_END",
    "ALFRED_REALTIME_START",
    "AlfredClient",
    "AlfredFetchError",
    "AlfredRawSeries",
    "FredAlfredClient",
    "HttpClient",
    "HttpError",
    "HttpxClient",
    "PriceClient",
    "YFinanceClient",
]

ALFRED_REALTIME_START = "1776-07-04"
ALFRED_REALTIME_END = "9999-12-31"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


class HttpError(Exception):
    """An HTTP request failed after all retries. The message never contains the query string."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class AlfredFetchError(Exception):
    """An ALFRED response was malformed or its pagination was inconsistent."""


# ---------------------------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------------------------


class HttpClient(Protocol):
    def get_bytes(self, url: str, params: Mapping[str, str] | None = None) -> bytes:
        """GET ``url`` and return the body of a 2xx response; raise ``HttpError`` otherwise."""
        ...


class PriceClient(Protocol):
    def download(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Daily bars for ``[start, end]`` (both inclusive).

        Contract: a ``DatetimeIndex`` of dates and ``(field, ticker)`` MultiIndex columns where
        ``field`` includes ``"Adj Close"``, ``"Close"`` and ``"Volume"``.
        """
        ...


@dataclass(frozen=True)
class AlfredRawSeries:
    """Every ALFRED row of one series as returned by FRED (all fields still strings)."""

    series_id: str
    rows: tuple[Mapping[str, str], ...]
    count: int
    pages: int


class AlfredClient(Protocol):
    def all_releases(self, series_id: str) -> AlfredRawSeries:
        """All vintages of ``series_id``; rows carry date, realtime_start/end, value."""
        ...


# ---------------------------------------------------------------------------------------------
# httpx
# ---------------------------------------------------------------------------------------------


class HttpxClient:
    """``HttpClient`` over ``httpx.Client``; ``transport`` and ``sleep`` are injectable."""

    def __init__(
        self,
        config: HttpConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=config.timeout_s,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpxClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_bytes(self, url: str, params: Mapping[str, str] | None = None) -> bytes:
        attempts = self._config.max_retries + 1
        failure = ""
        status: int | None = None
        for attempt in range(attempts):
            if attempt > 0:
                delay = self._config.backoff_s * 2 ** (attempt - 1)
                _log().warning(
                    "http_retry", url=url, attempt=attempt, delay_s=delay, reason=failure
                )
                self._sleep(delay)
            try:
                response = self._client.get(url, params=dict(params) if params else None)
            except httpx.TransportError as exc:
                # Not chained: the exception's request URL carries the query string.
                failure, status = f"{type(exc).__name__}", None
                continue
            if response.is_success:
                return response.content
            status = response.status_code
            failure = f"HTTP {status}: {response.text[:300]}"
            if status not in _RETRY_STATUSES:
                break
        raise HttpError(f"GET {url} failed after {attempt + 1} attempt(s): {failure}", status)


# ---------------------------------------------------------------------------------------------
# FRED / ALFRED
# ---------------------------------------------------------------------------------------------

_ROW_FIELDS = ("date", "realtime_start", "realtime_end", "value")


class FredAlfredClient:
    """``AlfredClient`` over the FRED ``series/observations`` endpoint (see module docstring)."""

    def __init__(
        self,
        http: HttpClient,
        api_key: SecretStr,
        *,
        page_size: int = 100_000,
        url: str = FRED_OBSERVATIONS_URL,
    ) -> None:
        if not 1 <= page_size <= 100_000:
            raise ValueError(f"page_size must be in [1, 100000], got {page_size}")
        if not api_key.get_secret_value():
            raise ValueError("FRED API key is empty; set FRED_API_KEY in .env")
        self._http = http
        self._api_key = api_key
        self._page_size = page_size
        self._url = url

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key.get_secret_value(), "***")

    def _page(self, series_id: str, offset: int) -> tuple[list[Mapping[str, str]], int]:
        params = {
            "series_id": series_id,
            "api_key": self._api_key.get_secret_value(),
            "file_type": "json",
            "realtime_start": ALFRED_REALTIME_START,
            "realtime_end": ALFRED_REALTIME_END,
            "sort_order": "asc",
            "order_by": "observation_date",
            "limit": str(self._page_size),
            "offset": str(offset),
        }
        redacted: HttpError | None = None
        try:
            body = self._http.get_bytes(self._url, params)
        except HttpError as exc:
            redacted = HttpError(self._redact(str(exc)), exc.status)
        if redacted is not None:
            # Raised outside the except block so the unredacted error is not kept as context.
            raise redacted
        try:
            payload = json.loads(body)
            count = payload["count"]
            observations = payload["observations"]
        except (ValueError, KeyError, TypeError) as exc:
            raise AlfredFetchError(
                f"{series_id}: malformed FRED response at offset {offset}: {exc!r}"
            ) from None
        if not isinstance(count, int) or not isinstance(observations, list):
            raise AlfredFetchError(f"{series_id}: unexpected types in FRED response")
        rows: list[Mapping[str, str]] = []
        for obs in observations:
            if not isinstance(obs, dict) or any(
                not isinstance(obs.get(k), str) for k in _ROW_FIELDS
            ):
                raise AlfredFetchError(f"{series_id}: malformed observation {obs!r}")
            rows.append({k: obs[k] for k in _ROW_FIELDS})
        return rows, count

    def all_releases(self, series_id: str) -> AlfredRawSeries:
        rows, count = self._page(series_id, 0)
        pages = 1
        while len(rows) < count:
            page, page_count = self._page(series_id, len(rows))
            pages += 1
            if page_count != count:
                raise AlfredFetchError(
                    f"{series_id}: API count changed from {count} to {page_count} during "
                    "pagination (a new release landed mid-fetch); retry the fetch"
                )
            if not page:
                break
            rows.extend(page)
        if len(rows) != count:
            raise AlfredFetchError(
                f"{series_id}: retrieved {len(rows)} rows but the API reported {count}"
            )
        keys = {(r["date"], r["realtime_start"]) for r in rows}
        if len(keys) != len(rows):
            raise AlfredFetchError(
                f"{series_id}: {len(rows) - len(keys)} duplicate (date, realtime_start) rows; "
                "page ordering was unstable"
            )
        _log().info("alfred_fetched", series_id=series_id, rows=count, pages=pages)
        return AlfredRawSeries(series_id=series_id, rows=tuple(rows), count=count, pages=pages)


# ---------------------------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------------------------


class YFinanceClient:
    """``PriceClient`` over ``yfinance.download`` (see module docstring)."""

    def __init__(self, timeout_s: float) -> None:
        self._timeout_s = timeout_s

    def download(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        import yfinance  # imported lazily: heavy, and unit tests never need it

        frame = yfinance.download(
            list(tickers),
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
            auto_adjust=False,
            actions=False,
            repair=False,
            group_by="column",
            multi_level_index=True,
            progress=False,
            threads=False,
            timeout=self._timeout_s,
        )
        if not isinstance(frame, pd.DataFrame):  # yfinance returns None when nothing downloads
            return pd.DataFrame()
        return frame
