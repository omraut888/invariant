"""Real-data candidate signals and the signal panel (docs/adr/0018, 0019).

Each signal is one common time series at session t, tested against every asset's return. Its
definition is frozen in ``REGISTRY`` (ADR 0018); a run selects signals by id
(``signals.ids``). The panel holds only ``ret_*`` and ``sig_*`` columns, because Stage 1 puts
every non-return feature into its graph as a conditioning variable.

Method
------
Inputs are computed on the full fetched session range, which starts at ``signals.warmup_start``
(before the panel), so every lookback is complete at the first panel session. Then every
column is cut to the panel's sessions (``dates``). Information times are the sources':

- **ETF returns** (``r_t`` of SPY): known at the close of t (ADR 0006). Windows are exact
  sums over a fixed number of past sessions (``_window_sum``); a missing return anywhere in a
  window makes the signal missing (nothing is filled).
- **FRED market series** (``pit.market_asof``, ADR 0019): the newest value usable at t, under
  ``data.fred_market.lag_rule`` (the primary rule, or ADR 0018's max-lag sensitivity rule).
- **ALFRED macro series** (``pit.within_vintage_change``, ADR 0019): a change computed inside
  the vintage usable at t.

Every signal at t is a function of inputs usable at or before t (tests/leakage/).

Assumptions and limitations
---------------------------
- Signals are levels or changes of persistent series (momentum and volatility windows, yield
  spreads; first-order autocorrelation about .996 to .999). Stage 1's size was validated up
  to rho = .99 at T = 6000 and in the T(1 - rho) = 5 Stambaugh diagnostic (ADR 0009).
- Monthly macro changes are step functions on the daily panel: they change only on release
  days.
- Signal definitions follow the cited papers in spirit, at a daily sampling the papers mostly
  did not use; the exact definitions are below and in ADR 0018.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from invariant.config.settings import PipelineConfig
from invariant.core.logging import get_logger
from invariant.data.pit import (
    PitPanel,
    forward_returns,
    market_asof,
    within_vintage_change,
)
from invariant.data.snapshot import Snapshot
from invariant.data.sources.base import DataValidationError, SourceName
from invariant.data.sources.etf import log_returns

__all__ = [
    "REGISTRY",
    "SIGNAL_PREFIX",
    "SignalDefinition",
    "SignalInputs",
    "build_signal_panel",
    "compute_signals",
]

SIGNAL_PREFIX: Final = "sig_"
TRADING_DAYS: Final = 252
_FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SignalInputs:
    """Everything a signal reads, on one session index (the full fetched range).

    ``returns``: ETF log returns (``ticker`` columns). ``market``: FRED market series id ->
    ``market_asof`` frame. ``alfred``: ALFRED series id -> that series' vintage table.
    """

    sessions: pd.DatetimeIndex
    returns: pd.DataFrame
    market: Mapping[str, pd.DataFrame]
    alfred: Mapping[str, pd.DataFrame]


SignalValue = tuple["pd.Series[float]", "pd.Series[pd.Timestamp] | None"]


@dataclass(frozen=True)
class SignalDefinition:
    """One frozen candidate signal (ADR 0018). ``compute`` returns the signal and, for a dated
    source, the observation date behind each value (else ``None``)."""

    id: str
    definition: str
    """The exact definition, as the memo and ADR 0018 state it."""
    cites: tuple[str, ...]
    """Literature corpus keys of the papers that motivate it."""
    etf: tuple[str, ...]
    market: tuple[str, ...]
    alfred: tuple[str, ...]
    compute: Callable[[SignalInputs], SignalValue]


def _window_sum(x: pd.Series, newest: int, oldest: int) -> pd.Series:
    """``sum_{k=newest..oldest} x_{t-k}``, NaN unless all of those values exist.

    An explicit sum over each window (not a running sum), so the value at t depends only on
    the values in its window, bit for bit.
    """
    values = x.to_numpy(dtype="float64")
    width = oldest - newest + 1
    out: _FloatArray = np.full(len(values), np.nan)
    if len(values) >= oldest + 1:
        sums = sliding_window_view(values, width).sum(axis=1)
        # sums[j] covers values[j .. j+width-1]; at t that window is t-oldest .. t-newest.
        out[oldest:] = sums[: len(values) - oldest]
    return pd.Series(out, index=x.index)


def _spy(inputs: SignalInputs) -> pd.Series:
    return inputs.returns["SPY"]


def _tsmom_spy(inputs: SignalInputs) -> SignalValue:
    return _window_sum(_spy(inputs), 21, TRADING_DAYS - 1), None


def _rv_spy(inputs: SignalInputs) -> SignalValue:
    rv = _window_sum(_spy(inputs) ** 2, 0, 20)
    return 0.5 * np.log((TRADING_DAYS / 21) * rv), None


def _vrp(inputs: SignalInputs) -> SignalValue:
    vix = inputs.market["VIXCLS"]
    rv = _window_sum(_spy(inputs) ** 2, 0, 20)
    return (vix["value"] / 100.0) ** 2 / 12.0 - rv, vix["obs_date"]


def _term_spread(inputs: SignalInputs) -> SignalValue:
    s = inputs.market["T10Y3M"]
    return s["value"], s["obs_date"]


def _default_spread(inputs: SignalInputs) -> SignalValue:
    baa, aaa = inputs.market["DBAA"], inputs.market["DAAA"]
    stale = np.minimum(baa["obs_date"].to_numpy(), aaa["obs_date"].to_numpy())
    return baa["value"] - aaa["value"], pd.Series(stale, index=baa.index)


def _rel_bill(inputs: SignalInputs) -> SignalValue:
    bill = inputs.market["DTB3"]
    prior_mean = _window_sum(bill["value"], 1, TRADING_DAYS) / TRADING_DAYS
    return bill["value"] - prior_mean, bill["obs_date"]


def _macro(series: str, months: int, log: bool) -> Callable[[SignalInputs], SignalValue]:
    def compute(inputs: SignalInputs) -> SignalValue:
        out = within_vintage_change(
            inputs.alfred[series], series, inputs.sessions, months=months, log=log
        )
        return out["value"], out["obs_date"]

    return compute


REGISTRY: Final[Mapping[str, SignalDefinition]] = {
    d.id: d
    for d in (
        SignalDefinition(
            id="tsmom_spy",
            definition="sum of SPY daily log returns over sessions t-251 to t-21 (12-1 month "
            "time-series momentum)",
            cites=("moskowitz2012",),
            etf=("SPY",),
            market=(),
            alfred=(),
            compute=_tsmom_spy,
        ),
        SignalDefinition(
            id="rv_spy",
            definition="half the log of annualised SPY realised variance over sessions t-20 to "
            "t: 0.5 log((252/21) sum r^2)",
            cites=("french1987", "moreira2017"),
            etf=("SPY",),
            market=(),
            alfred=(),
            compute=_rv_spy,
        ),
        SignalDefinition(
            id="vrp",
            definition="variance risk premium: (VIX/100)^2/12 minus the sum of squared SPY "
            "daily log returns over sessions t-20 to t",
            cites=("bollerslev2009",),
            etf=("SPY",),
            market=("VIXCLS",),
            alfred=(),
            compute=_vrp,
        ),
        SignalDefinition(
            id="term_spread",
            definition="10-year minus 3-month Treasury constant-maturity yield (FRED T10Y3M), "
            "the newest value usable at t",
            cites=("campbell1987", "fama1989"),
            etf=(),
            market=("T10Y3M",),
            alfred=(),
            compute=_term_spread,
        ),
        SignalDefinition(
            id="default_spread",
            definition="Moody's Baa minus Aaa corporate bond yield (FRED DBAA - DAAA), the "
            "newest values usable at t",
            cites=("keim1986", "fama1989"),
            etf=(),
            market=("DBAA", "DAAA"),
            alfred=(),
            compute=_default_spread,
        ),
        SignalDefinition(
            id="rel_bill",
            definition="3-month Treasury bill rate (FRED DTB3) minus its mean over the prior "
            "252 sessions, each value as usable at its session",
            cites=("ang2007",),
            etf=(),
            market=("DTB3",),
            alfred=(),
            compute=_rel_bill,
        ),
        SignalDefinition(
            id="ip_growth",
            definition="industrial production (ALFRED INDPRO): log change from month o-1 to "
            "o, both from the vintage usable at t",
            cites=("fama1990",),
            etf=(),
            market=(),
            alfred=("INDPRO",),
            compute=_macro("INDPRO", 1, log=True),
        ),
        SignalDefinition(
            id="payrolls_growth",
            definition="nonfarm payrolls (ALFRED PAYEMS): log change from month o-1 to o, both "
            "from the vintage usable at t",
            cites=("flannery2002",),
            etf=(),
            market=(),
            alfred=("PAYEMS",),
            compute=_macro("PAYEMS", 1, log=True),
        ),
        SignalDefinition(
            id="unrate_change",
            definition="unemployment rate (ALFRED UNRATE): change from month o-1 to o, both "
            "from the vintage usable at t",
            cites=("boyd2005",),
            etf=(),
            market=(),
            alfred=("UNRATE",),
            compute=_macro("UNRATE", 1, log=False),
        ),
        SignalDefinition(
            id="inflation",
            definition="CPI inflation (ALFRED CPIAUCSL): log change from month o-12 to o, both "
            "from the vintage usable at t",
            cites=("fama1977",),
            etf=(),
            market=(),
            alfred=("CPIAUCSL",),
            compute=_macro("CPIAUCSL", 12, log=True),
        ),
    )
}
"""The frozen candidate definitions (ADR 0018). ``vix`` was considered and dropped: vrp is
built from VIX squared and realised variance, so with rv_spy the three are near-redundant, and
conditioning on two in PCMCI could kill the third for collinearity, not lack of evidence."""


def compute_signals(
    inputs: SignalInputs, ids: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ``sig_<id>`` columns and their observation dates on ``inputs.sessions``."""
    unknown = [i for i in ids if i not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown signal ids {unknown}; known: {sorted(REGISTRY)}")
    values: dict[str, pd.Series[float]] = {}
    dates: dict[str, pd.Series[pd.Timestamp]] = {}
    for sid in ids:
        value, obs = REGISTRY[sid].compute(inputs)
        values[SIGNAL_PREFIX + sid] = value.astype("float64")
        if obs is not None:
            dates[SIGNAL_PREFIX + sid] = obs
    return (
        pd.DataFrame(values, index=inputs.sessions),
        pd.DataFrame(dates, index=inputs.sessions),
    )


def _inputs(snapshot: Snapshot, config: PipelineConfig, ids: tuple[str, ...]) -> SignalInputs:
    defs = [REGISTRY[i] for i in ids]
    need_market = sorted({s for d in defs for s in d.market})
    need_alfred = sorted({s for d in defs for s in d.alfred})
    adj = snapshot.etf_adj_close()
    sessions = pd.DatetimeIndex(adj.index)
    if sessions[0] > pd.Timestamp(config.fetch_start) + pd.Timedelta(days=7):
        raise DataValidationError(
            f"snapshot {snapshot.snapshot_id} starts {sessions[0].date()}, after the warm-up "
            f"start {config.fetch_start}"
        )
    market: dict[str, pd.DataFrame] = {}
    if need_market:
        report = snapshot.fred_market_report()
        vintages, observations = (
            snapshot.fred_market_vintages(),
            snapshot.fred_market_observations(),
        )
        rule = config.data.fred_market.lag_rule if config.data.fred_market else None
        for sid in need_market:
            rep = report.get(sid)
            if rule == "max_lag":
                # Every observation is "before the archive": current value, uniform lag.
                first_vintage, lag = date.max, rep.max_posting_lag + 1
            else:
                first_vintage, lag = rep.first_vintage, rep.pre_archive_lag
            market[sid] = market_asof(
                vintages[vintages["series_id"] == sid].reset_index(drop=True),
                observations,
                sid,
                first_vintage=first_vintage,
                pre_archive_lag=lag,
                sessions=sessions,
            )
    alfred: dict[str, pd.DataFrame] = {}
    if need_alfred:
        table = snapshot.alfred_vintages()
        for sid in need_alfred:
            alfred[sid] = table[table["series_id"] == sid].reset_index(drop=True)
            if alfred[sid].empty:
                raise DataValidationError(f"snapshot lacks ALFRED series {sid}")
    return SignalInputs(sessions=sessions, returns=log_returns(adj), market=market, alfred=alfred)


def build_signal_panel(snapshot: Snapshot, config: PipelineConfig) -> PitPanel:
    """The real-data panel: ``ret_*`` of the universe and ``sig_*`` of ``config.signals.ids``
    over the sessions in ``config.dates``, with ``fwd_*`` targets (module docstring)."""
    if config.signals is None:
        raise ValueError("config.signals is not set")
    ids = config.signals.ids
    if any(s.source is SourceName.SYNTHETIC for s in snapshot.manifest.sources):
        raise DataValidationError("a signal panel is built from real data only")
    inputs = _inputs(snapshot, config, ids)
    signals, obs_dates = compute_signals(inputs, ids)
    adj = snapshot.etf_adj_close()
    tickers = list(config.universe.tickers)
    absent = [t for t in tickers if t not in adj.columns]
    if absent:
        raise DataValidationError(f"snapshot {snapshot.snapshot_id} lacks tickers {absent}")
    in_range = (adj.index >= pd.Timestamp(config.dates.start)) & (
        adj.index <= pd.Timestamp(config.dates.end)
    )
    adj = adj.loc[in_range, tickers]
    returns = log_returns(adj)
    features = pd.concat([returns.add_prefix("ret_"), signals.loc[adj.index]], axis=1)
    targets = forward_returns(returns, config.pit.targets).add_prefix("fwd_")
    panel = PitPanel(
        features=features,
        obs_dates=obs_dates.loc[adj.index],
        targets=targets,
        config=config.pit,
    )
    get_logger(__name__).info(
        "signal_panel_built",
        snapshot_id=snapshot.snapshot_id,
        sessions=len(features),
        signals=list(ids),
        complete_rows=int(features.notna().all(axis=1).sum()),
    )
    return panel
