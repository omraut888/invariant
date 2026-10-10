"""Point-in-time feature panel and forward-return targets.

Every value the panel holds for session t uses only information usable at the close of t.
Targets are built only from returns dated after t. The rules come from
docs/adr/0004-data-sources-and-pit-limitations.md and docs/adr/0006-point-in-time-panel.md.

Method
------
Each raw value gets an *information time*: the first session at which it may be used. A
feature at t is then a function of the values whose information time is at or before t.

- **ETF returns** (``ret_<TICKER>``): ``r_t = ln(P_t / P_{t-1})`` from adjusted closes. Known at
  the close of t (lag 0). Adjusted *levels* are never features (ADR 0004: they embed later
  splits and dividends).
- **French factors** (``ff_<factor>``): no release dates exist, so the row dated d is used at
  the session ``publication_lag_sessions.french`` sessions after d (the first session on or
  after d if d is not a session). The value is placed at exactly that session, not carried
  forward, so a missing row stays missing. Rows dated before the first panel session are
  dropped: their true lagged session cannot be counted without the earlier calendar, so the
  first ``lag`` sessions are NaN. This is conservative.
- **ALFRED macro series** (``macro_<SERIES>``, with the observation date in ``obs_dates``):
  ``realtime_start`` has no time of day, so a vintage row is usable from the first session
  *strictly after* ``realtime_start`` (ADR 0004). It stops being usable at the first session
  strictly after ``realtime_end + 1 day``, the date of the vintage that superseded or withdrew
  it. That is exactly when the next vintage of the same observation becomes usable, so
  consecutive vintages hand over without a gap or an overlap. At each session, among the rows
  usable then, the feature is the value of the newest observation, so it is never a final
  revised value unless that vintage was already published.

  ``realtime_end`` of a row still current at t is future information. It is only ever compared
  as ``t < stop``. A ``stop`` after t gives the same answer at t whatever its exact value, so
  that future information never reaches the panel (tested in tests/leakage/test_point_in_time.py).

  NaN vintage rows (FRED's ``.``: the vintage published "no value") are never used as values.
  If the newest observation is NaN, the feature is the newest observation that has a value
  *in the same vintages*. A NaN row still ends the previous vintage of its observation, so an
  older value is never resurrected. ``obs_dates`` shows how stale each value is.

- **Targets** (``fwd_<TICKER>``): ``y_t = r_{t+1+e} + ... + r_{t+e+h}`` with ``e =
  execution_lag`` and ``h = horizon`` (``TargetConfig``). A window with any NaN, or one that
  runs past the end of the sample, gives NaN. ``y_t`` reads only returns dated after t. In
  price terms it is ``ln(P_{t+e+h} / P_{t+e})``, so with ``e = 0`` it depends on ``P_t`` through
  ``r_{t+1}`` (that is what a return is). With ``e >= 1`` it reads no price at or before t.

Assumptions and limitations
---------------------------
- Information times are session-granular. Same-session use of an 08:30 ET release would need
  release timestamps (ADR 0004).
- The French lag is a fixed session count. Real library releases are irregular, and every
  release revises history. The lag bounds timeliness, not revisions (ADR 0004).
- Sessions come from the snapshot's exchange calendar. Calendar changes between library
  versions are covered by storing the calendar in the snapshot.
- The macro feature is a *level*. Changes must be computed within one vintage, never by
  differencing this as-of series, which mixes vintages (a follow-up task).

Reference: Croushore, D. and Stark, T. (2001), "A real-time data set for macroeconomists",
Journal of Econometrics 105(1), 111-130.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from structlog.typing import FilteringBoundLogger

from invariant.config.settings import PipelineConfig, PitConfig, TargetConfig
from invariant.core.logging import get_logger
from invariant.core.types import Frequency, validate_time_index
from invariant.data.snapshot import Snapshot
from invariant.data.sources.alfred import validate_vintages
from invariant.data.sources.base import DATE_DTYPE, DataValidationError, SourceName
from invariant.data.sources.etf import log_returns

__all__ = [
    "PitInputs",
    "PitPanel",
    "alfred_asof",
    "assemble_panel",
    "build_pit_panel",
    "forward_returns",
    "lag_by_sessions",
]

_ONE_DAY: Final = np.timedelta64(1, "D")
_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.intp]
_DateArray = npt.NDArray[np.datetime64]


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


def _dates(values: object) -> _DateArray:
    return np.asarray(values, dtype=DATE_DTYPE)


def _check_sessions(sessions: pd.DatetimeIndex) -> _DateArray:
    validate_time_index(pd.Series(0, index=sessions), Frequency.DAILY)
    return _dates(sessions)


# ---------------------------------------------------------------------------------------------
# Per-source transforms (pure)
# ---------------------------------------------------------------------------------------------


def alfred_asof(vintages: pd.DataFrame, series_id: str, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """As-known value of ALFRED series ``series_id`` at each session (see module docstring).

    ``vintages`` is that series' ``VINTAGE_COLUMNS`` table (it may be empty). Returns a frame
    indexed by ``sessions`` with ``value`` (float, NaN before anything is usable) and
    ``obs_date`` (the observation that value belongs to, NaT where ``value`` is NaN).
    """
    sess = _check_sessions(sessions)
    table = validate_vintages(vintages, series_id)
    table = table[table["value"].notna()]

    # Usable on sessions [start, stop): from the session after realtime_start until the session
    # after the superseding vintage's date (realtime_end + 1 day). An open end (9999-12-31)
    # maps past every session.
    start: _IntArray = np.searchsorted(sess, _dates(table["realtime_start"]), side="right")
    stop: _IntArray = np.searchsorted(sess, _dates(table["realtime_end"]) + _ONE_DAY, side="right")
    keep = start < stop
    start, stop = start[keep], stop[keep]
    obs = _dates(table["observation_date"])[keep]
    values = table["value"].to_numpy(dtype="float64")[keep]

    n = len(sess)
    out_value: _FloatArray = np.full(n, np.nan)
    out_obs: _DateArray = np.full(n, np.datetime64("NaT"), dtype=DATE_DTYPE)
    # The usable set is constant between consecutive start/stop positions.
    cuts = np.unique(np.concatenate([start, stop, [0, n]]))
    cuts = cuts[(cuts >= 0) & (cuts <= n)]
    for lo, hi in itertools.pairwise(cuts):
        active = np.flatnonzero((start <= lo) & (stop > lo))
        if active.size:
            # Periods of one observation are disjoint, so the newest observation is unique.
            i = active[np.argmax(obs[active])]
            out_value[lo:hi] = values[i]
            out_obs[lo:hi] = obs[i]
    return pd.DataFrame({"value": out_value, "obs_date": out_obs}, index=sessions)


def lag_by_sessions(frame: pd.DataFrame, sessions: pd.DatetimeIndex, lag: int) -> pd.DataFrame:
    """Place each row of ``frame`` (indexed by date) ``lag`` sessions after its date.

    A date that is not a session counts from the next session (logged). Rows dated before
    ``sessions[0]``, or landing after the last session, are dropped. If two rows land on one
    session, the later-dated one wins. Sessions with no row are NaN: nothing is carried forward.
    """
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    sess = _check_sessions(sessions)
    if len(frame):
        validate_time_index(frame)
    dates = _dates(frame.index)
    pos: _IntArray = np.searchsorted(sess, dates, side="left") + lag
    in_panel = dates >= sess[0]
    off_calendar = in_panel & ~np.isin(dates, sess)
    if off_calendar.any():
        _log().warning("pit_off_calendar_rows", n_rows=int(off_calendar.sum()))
    keep = in_panel & (pos < len(sess))
    out = pd.DataFrame(np.nan, index=sessions, columns=frame.columns, dtype="float64")
    if keep.any():
        values = frame.to_numpy(dtype="float64")[keep]
        target = pos[keep]
        # frame is sorted by date, so the last row per target session is the latest-dated.
        last = np.r_[target[1:] != target[:-1], True]
        out.iloc[target[last]] = values[last]
    return out


def forward_returns(returns: pd.DataFrame, target: TargetConfig) -> pd.DataFrame:
    """``y_t = r_{t+1+e} + ... + r_{t+e+h}`` for a session-aligned return panel.

    NaN where the window holds a NaN or runs past the last row. Summation is in a fixed order
    (``k = 1+e, ..., e+h``), so results are bit-reproducible.
    """
    validate_time_index(returns, Frequency.DAILY)
    values = returns.to_numpy(dtype="float64")
    first = 1 + target.execution_lag
    last = target.execution_lag + target.horizon
    n_full = len(values) - last
    out: _FloatArray = np.full_like(values, np.nan)
    if n_full > 0:
        acc = np.zeros((n_full, values.shape[1]))
        for k in range(first, last + 1):
            acc += values[k : k + n_full]
        out[:n_full] = acc
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


# ---------------------------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PitInputs:
    """Raw inputs to ``assemble_panel``.

    ``adj_close``: session-aligned adjusted closes (``sessions x tickers``; its index is the panel
    index). ``french``: factor returns indexed by date, or ``None``. ``alfred``: series id ->
    that series' ``VINTAGE_COLUMNS`` table. The keys, not the rows, decide the macro columns, so
    the panel's columns never depend on which vintages exist yet.
    """

    adj_close: pd.DataFrame
    french: pd.DataFrame | None = None
    alfred: Mapping[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True)
class PitPanel:
    """A point-in-time panel. All frames share one validated daily session index.

    ``features``: ``ret_*``, ``ff_*`` and ``macro_*`` columns. ``obs_dates``: for each
    ``macro_*`` column, the observation date of the value used. ``targets``: ``fwd_*`` columns.
    """

    features: pd.DataFrame
    obs_dates: pd.DataFrame
    targets: pd.DataFrame
    config: PitConfig

    def __post_init__(self) -> None:
        validate_time_index(self.features, Frequency.DAILY)
        for name, frame in (("obs_dates", self.obs_dates), ("targets", self.targets)):
            if not frame.index.equals(self.features.index):
                raise ValueError(f"{name} index differs from the features index")


def assemble_panel(inputs: PitInputs, config: PitConfig) -> PitPanel:
    """Build the point-in-time panel from raw inputs (pure; see the module docstring)."""
    sessions = pd.DatetimeIndex(inputs.adj_close.index)
    returns = log_returns(inputs.adj_close)
    parts = [returns.add_prefix("ret_")]
    if inputs.french is not None:
        lagged = lag_by_sessions(inputs.french, sessions, config.publication_lag_sessions.french)
        parts.append(lagged.add_prefix("ff_"))
    macro_values: dict[str, pd.Series[float]] = {}
    macro_obs: dict[str, pd.Series[pd.Timestamp]] = {}
    for sid in sorted(inputs.alfred):
        asof = alfred_asof(inputs.alfred[sid], sid, sessions)
        macro_values[f"macro_{sid}"] = asof["value"]
        macro_obs[f"macro_{sid}"] = asof["obs_date"]
    if macro_values:
        parts.append(pd.DataFrame(macro_values, index=sessions))
    features = pd.concat(parts, axis=1)
    obs_dates = pd.DataFrame(macro_obs, index=sessions)
    targets = forward_returns(returns, config.targets).add_prefix("fwd_")
    return PitPanel(features=features, obs_dates=obs_dates, targets=targets, config=config)


def _has_source(snapshot: Snapshot, name: SourceName) -> bool:
    return any(entry.source is name for entry in snapshot.manifest.sources)


def build_pit_panel(snapshot: Snapshot, config: PipelineConfig) -> PitPanel:
    """Read ``snapshot`` and assemble the panel over the sessions in ``config.dates``.

    ETF prices are required. French factors and ALFRED series are included when the snapshot
    has them (a missing source is logged). Every configured ALFRED series must be present.
    """
    adj = snapshot.etf_adj_close()
    tickers = list(config.universe.tickers)
    absent = [t for t in tickers if t not in adj.columns]
    if absent:
        raise DataValidationError(f"snapshot {snapshot.snapshot_id} lacks tickers {absent}")
    in_range = (adj.index >= pd.Timestamp(config.dates.start)) & (
        adj.index <= pd.Timestamp(config.dates.end)
    )
    adj = adj.loc[in_range, tickers]
    if adj.empty:
        raise DataValidationError(f"snapshot {snapshot.snapshot_id} has no sessions in range")

    french: pd.DataFrame | None = None
    if _has_source(snapshot, SourceName.FRENCH):
        french = snapshot.french_factors()
    else:
        _log().warning("pit_source_absent", source=str(SourceName.FRENCH))
    alfred: dict[str, pd.DataFrame] = {}
    if _has_source(snapshot, SourceName.ALFRED):
        alfred = _select_series(snapshot.alfred_vintages(), config.data.alfred.series)
    else:
        _log().warning("pit_source_absent", source=str(SourceName.ALFRED))

    panel = assemble_panel(PitInputs(adj_close=adj, french=french, alfred=alfred), config.pit)
    _log().info(
        "pit_panel_built",
        snapshot_id=snapshot.snapshot_id,
        sessions=len(panel.features),
        features=len(panel.features.columns),
        execution_lag=config.pit.targets.execution_lag,
        horizon=config.pit.targets.horizon,
    )
    return panel


def _select_series(vintages: pd.DataFrame, series: Sequence[str]) -> dict[str, pd.DataFrame]:
    present = set(vintages["series_id"].unique())
    missing = [s for s in series if s not in present]
    if missing:
        raise DataValidationError(f"snapshot lacks ALFRED series {missing}")
    return {sid: vintages[vintages["series_id"] == sid].reset_index(drop=True) for sid in series}
