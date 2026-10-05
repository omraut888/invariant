"""Fetch every configured source with the real clients and write one snapshot.

This is the only place that wires concrete network clients to the source modules. Any client
can be injected (tests, or a caller that wants a subset of sources via ``sources``).
"""

from __future__ import annotations

from collections.abc import Sequence

from invariant.config.settings import PipelineConfig, Settings, load_settings
from invariant.data.snapshot import Snapshot, default_snapshots_dir, write_snapshot
from invariant.data.sources.alfred import fetch_alfred
from invariant.data.sources.base import SourceName, SourcePayload
from invariant.data.sources.clients import (
    AlfredClient,
    FredAlfredClient,
    HttpClient,
    HttpxClient,
    PriceClient,
    YFinanceClient,
)
from invariant.data.sources.etf import fetch_etf_prices
from invariant.data.sources.french import fetch_french_factors

__all__ = ["build_snapshot"]


def build_snapshot(
    config: PipelineConfig,
    settings: Settings | None = None,
    *,
    sources: Sequence[SourceName] = tuple(SourceName),
    price_client: PriceClient | None = None,
    http: HttpClient | None = None,
    alfred_client: AlfredClient | None = None,
) -> Snapshot:
    """Fetch ``sources`` as configured and write them as one immutable snapshot.

    Clients default to the real adapters; ALFRED requires ``FRED_API_KEY`` unless
    ``alfred_client`` is injected.
    """
    settings = settings if settings is not None else load_settings()
    data_cfg = config.data
    owned: HttpxClient | None = None
    if http is None:
        owned = HttpxClient(data_cfg.http)
        http = owned
    try:
        payloads: list[SourcePayload] = []
        if SourceName.ETF in sources:
            client = price_client or YFinanceClient(timeout_s=data_cfg.http.timeout_s)
            payloads.append(
                fetch_etf_prices(
                    client,
                    config.universe.tickers,
                    config.dates.start,
                    config.dates.end,
                    calendar=data_cfg.calendar,
                )
            )
        if SourceName.FRENCH in sources:
            payloads.append(fetch_french_factors(http, data_cfg.french))
        if SourceName.ALFRED in sources:
            if alfred_client is None:
                if settings.fred_api_key is None:
                    raise ValueError("FRED_API_KEY is not set; required for ALFRED")
                alfred_client = FredAlfredClient(
                    http, settings.fred_api_key, page_size=data_cfg.alfred.page_size
                )
            payloads.append(fetch_alfred(alfred_client, data_cfg.alfred.series))
        return write_snapshot(payloads, default_snapshots_dir(settings))
    finally:
        if owned is not None:
            owned.close()
