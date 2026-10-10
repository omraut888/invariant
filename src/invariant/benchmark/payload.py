"""Express a synthetic benchmark dataset as an ETF snapshot payload.

This lets a benchmark run go through the same path as real data: ``write_snapshot``,
``Snapshot.etf_adj_close`` and ``build_pit_panel``. Asset ``i`` becomes ticker ``SYN<ii>``
(``UniverseConfig`` accepts only uppercase symbols). Its adjusted close is
``100 * exp(cumsum(returns))``, so the log return between adjacent rows recovers the synthetic
return up to floating-point rounding (not bit-exactly; stages scored on the benchmark use
``scoring_targets`` on the dataset's own returns). The level at t uses only returns up to t.

The payload reuses ``SourceName.ETF`` with ``params.provider = "synthetic"`` and the dataset's
content hash, so the snapshot id is tied to the exact dataset and ``score_run`` can check that
a run's snapshot is the one a ``GroundTruth`` describes. The session list is the dataset's
consecutive weekdays (calendar ``"synthetic-weekdays"``), not an exchange calendar.
See docs/adr/0007-kill-ledger-and-benchmark-scoring.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Final

import numpy as np
import pandas as pd

from invariant.benchmark.synthetic import SyntheticDataset
from invariant.core.types import Frequency, validate_time_index
from invariant.data.sources.base import DATE_DTYPE, SourceName, SourcePayload, library_versions
from invariant.data.sources.etf import PRICE_COLUMNS, coverage_report

__all__ = [
    "SYNTHETIC_CALENDAR",
    "SYNTHETIC_PROVIDER",
    "synthetic_etf_payload",
    "synthetic_prices",
    "synthetic_tickers",
]

SYNTHETIC_PROVIDER: Final = "synthetic"
SYNTHETIC_CALENDAR: Final = "synthetic-weekdays"
_START_PRICE: Final = 100.0


def synthetic_tickers(asset_names: Sequence[str]) -> tuple[str, ...]:
    """``SYN00, SYN01, ...``, one per asset, in order."""
    return tuple(f"SYN{i:02d}" for i in range(len(asset_names)))


def synthetic_prices(returns: pd.DataFrame) -> pd.DataFrame:
    """Price levels ``100 * exp(r_0 + ... + r_t)``; row t uses only returns up to t."""
    validate_time_index(returns, Frequency.DAILY)
    values = returns.to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise ValueError("synthetic returns must not contain NaN")
    levels = _START_PRICE * np.exp(np.cumsum(values, axis=0))
    return pd.DataFrame(levels, index=returns.index, columns=returns.columns)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def synthetic_etf_payload(
    dataset: SyntheticDataset, *, now: Callable[[], datetime] = _utcnow
) -> SourcePayload:
    """The ``etf`` payload (tables ``prices``, ``sessions``) for ``dataset``."""
    dataset.verify()
    truth = dataset.ground_truth
    tickers = synthetic_tickers(truth.asset_names)
    wide = synthetic_prices(dataset.returns)
    sessions = pd.DatetimeIndex(dataset.returns.index.astype(DATE_DTYPE), name="date")
    parts = [
        pd.DataFrame(
            {
                "date": sessions,
                "ticker": ticker,
                "close": wide[asset].to_numpy(),
                "adj_close": wide[asset].to_numpy(),
                "volume": np.nan,
            }
        )
        for ticker, asset in zip(tickers, truth.asset_names, strict=True)
    ]
    prices = pd.concat(parts, ignore_index=True).astype({"ticker": "str"})[list(PRICE_COLUMNS)]
    report = coverage_report(
        prices,
        sessions,
        tickers,
        calendar=SYNTHETIC_CALENDAR,
        start=truth.first_date,
        end=truth.last_date,
    )
    return SourcePayload(
        source=SourceName.ETF,
        params={
            "provider": SYNTHETIC_PROVIDER,
            "dataset_sha256": truth.dataset_sha256,
            "seed": truth.seed,
            "tickers": dict(zip(tickers, truth.asset_names, strict=True)),
        },
        report=report,
        tables={"prices": prices, "sessions": pd.DataFrame({"date": sessions.to_numpy()})},
        fetched_at=now(),
        library_versions=library_versions("numpy"),
    )
