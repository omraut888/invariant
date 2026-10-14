"""Builders, strategies and comparisons for point-in-time tests. Offline only.

Macro data are generated as a *release history*: each release is a complete vintage (a mapping
observation -> value; a missing key means the observation is not in that vintage, NaN means
it was published as "."). ``releases_to_vintages`` turns a history into ALFRED's row format,
merging consecutive releases with an unchanged value into one real-time period exactly as
ALFRED does. Because a release's rows depend only on that release and the ones before it,
"the future" is just the releases dated after the cut, and its effect on ``realtime_end`` of
earlier rows comes for free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from tests.fixtures.fakes import alfred_raw, alfred_row

from invariant.config.settings import PitConfig, PublicationLags, TargetConfig
from invariant.data.pit import PitInputs, PitPanel
from invariant.data.sources.alfred import VINTAGE_COLUMNS, parse_alfred_rows
from invariant.data.sources.base import DATE_DTYPE

PAYEMS_CSV = Path(__file__).resolve().parent / "alfred" / "PAYEMS_2024Q1_vintages.csv"

N_SESSIONS = 40
SESSIONS = pd.DatetimeIndex(
    pd.bdate_range("2025-03-03", periods=N_SESSIONS).astype(DATE_DTYPE), name="date"
)
TICKERS = ("AAA", "BBB")
FACTORS = ("mkt_rf", "smb")
SERIES = ("SER1", "SER2")
SYN_FEATURES = ("s_real", "z")
OBS_DATES = tuple(pd.Timestamp(d) for d in ("2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"))
OPEN_END = "9999-12-31"

Release = tuple[pd.Timestamp, Mapping[pd.Timestamp, float]]


def pit_config(*, execution_lag: int = 1, horizon: int = 1, french_lag: int = 42) -> PitConfig:
    return PitConfig(
        targets=TargetConfig(execution_lag=execution_lag, horizon=horizon),
        publication_lag_sessions=PublicationLags(french=french_lag),
    )


def payems_vintages() -> pd.DataFrame:
    """The real PAYEMS excerpt in ``VINTAGE_COLUMNS`` form (see the CSV header)."""
    raw = pd.read_csv(PAYEMS_CSV, comment="#", dtype=str)
    rows = [
        alfred_row(
            str(r["date"]), str(r["realtime_start"]), str(r["realtime_end"]), str(r["value"])
        )
        for r in raw.to_dict("records")
    ]
    return parse_alfred_rows(alfred_raw("PAYEMS", rows))


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def empty_vintages() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": pd.Series(dtype="str"),
            "observation_date": pd.Series(dtype=DATE_DTYPE),
            "realtime_start": pd.Series(dtype=DATE_DTYPE),
            "realtime_end": pd.Series(dtype=DATE_DTYPE),
            "value": pd.Series(dtype="float64"),
        }
    )[list(VINTAGE_COLUMNS)]


def releases_to_vintages(series_id: str, releases: Sequence[Release]) -> pd.DataFrame:
    """ALFRED rows for a release history (see module docstring)."""
    ordered = sorted(releases, key=lambda r: r[0])
    observations = sorted({o for _, vintage in ordered for o in vintage})
    rows: list[dict[str, str]] = []

    def emit(obs: pd.Timestamp, start: pd.Timestamp, end: str, value: float) -> None:
        text = "." if math.isnan(value) else repr(value)
        rows.append(alfred_row(obs.strftime("%Y-%m-%d"), start.strftime("%Y-%m-%d"), end, text))

    for obs in observations:
        run: tuple[pd.Timestamp, float] | None = None
        for released, vintage in ordered:
            if run is not None and not (obs in vintage and _same(vintage[obs], run[1])):
                end = (released - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                emit(obs, run[0], end, run[1])
                run = None
            if run is None and obs in vintage:
                run = (released, vintage[obs])
        if run is not None:
            emit(obs, run[0], OPEN_END, run[1])
    if not rows:
        return empty_vintages()
    return parse_alfred_rows(alfred_raw(series_id, rows))


def oracle_asof(releases: Sequence[Release], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Brute-force reference for ``alfred_asof``, computed from the releases, not the rows.

    At session s the known vintage is the latest release dated strictly before s (a release
    dated s is usable only after s). The value is that vintage's newest non-NaN observation.
    """
    values: list[float] = []
    obs: list[pd.Timestamp | None] = []
    ordered = sorted(releases, key=lambda r: r[0])
    for s in sessions:
        known = [v for d, v in ordered if d < s]
        candidates = {o: x for o, x in (known[-1] if known else {}).items() if not math.isnan(x)}
        newest = max(candidates, default=None)
        obs.append(newest)
        values.append(np.nan if newest is None else candidates[newest])
    return pd.DataFrame(
        {
            "value": np.asarray(values, dtype="float64"),
            "obs_date": pd.Series(obs, dtype=DATE_DTYPE).to_numpy(),
        },
        index=sessions,
    )


# ---------------------------------------------------------------------------------------------
# Bit-level comparison
# ---------------------------------------------------------------------------------------------


def bit_identical(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Same index, columns, dtypes and *bytes* (stricter than assert_frame_equal: -0.0 != 0.0)."""
    if not (a.index.equals(b.index) and list(a.columns) == list(b.columns)):
        return False
    if list(a.dtypes) != list(b.dtypes):
        return False
    return all(
        np.ascontiguousarray(a[c].to_numpy()).tobytes()
        == np.ascontiguousarray(b[c].to_numpy()).tobytes()
        for c in a.columns
    )


def prefix_identical(a: PitPanel, b: PitPanel, t: int) -> bool:
    """Features and observation dates at sessions ``0..t`` are bit-identical."""
    return bit_identical(a.features.iloc[: t + 1], b.features.iloc[: t + 1]) and bit_identical(
        a.obs_dates.iloc[: t + 1], b.obs_dates.iloc[: t + 1]
    )


# ---------------------------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------------------------

_price = st.one_of(st.floats(min_value=0.5, max_value=500.0), st.just(math.nan))
_factor = st.one_of(st.floats(min_value=-0.05, max_value=0.05), st.just(math.nan))
_syn = st.one_of(st.floats(min_value=-5.0, max_value=5.0), st.just(math.nan))
_macro = st.sampled_from([1.0, 2.0, 3.0, math.nan])  # few values, so runs merge


def releases(
    lo: pd.Timestamp, hi: pd.Timestamp, max_size: int = 5
) -> st.SearchStrategy[list[Release]]:
    """Release histories with calendar dates (weekends included) in ``[lo, hi]``."""
    span = (hi - lo).days
    vintage = st.dictionaries(st.sampled_from(OBS_DATES), _macro, max_size=len(OBS_DATES))
    days = st.lists(st.integers(0, span), unique=True, max_size=max_size).map(sorted)
    return days.flatmap(
        lambda ds: st.tuples(*(vintage for _ in ds)).map(
            lambda vs: [(lo + pd.Timedelta(days=d), v) for d, v in zip(ds, vs, strict=True)]
        )
    )


@dataclass(frozen=True)
class Scenario:
    """Two raw input sets that agree on every value whose information time is <= ``t``."""

    a: PitInputs
    b: PitInputs
    t: int
    config: PitConfig


def _price_frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=SESSIONS, columns=list(TICKERS), dtype="float64")


@st.composite
def scenarios(draw: st.DrawFn, max_french_lag: int = 5) -> Scenario:
    """Raw inputs ``a``/``b`` that differ only in information dated after session ``t``.

    "After t" per source: ETF prices and synthetic features dated after t; French rows dated
    after the session ``lag`` sessions before t (all rows if t < lag); ALFRED releases dated on
    or after t, because a release dated t is usable only from the next session. The future side
    may change values, add or remove rows, and (through the release history) move
    ``realtime_end``.
    """
    t = draw(st.integers(0, N_SESSIONS - 1))
    lag = draw(st.integers(0, max_french_lag))
    config = pit_config(
        execution_lag=draw(st.integers(0, 2)), horizon=draw(st.integers(1, 3)), french_lag=lag
    )
    row = st.lists(_price, min_size=len(TICKERS), max_size=len(TICKERS))
    prices = draw(st.lists(row, min_size=N_SESSIONS, max_size=N_SESSIONS))
    other = draw(st.lists(row, min_size=N_SESSIONS, max_size=N_SESSIONS))
    prices_b = prices[: t + 1] + other[t + 1 :]

    # French: one optional row per session.
    cutoff = SESSIONS[t - lag] if t >= lag else SESSIONS[0] - pd.Timedelta(days=1)
    frow = st.one_of(st.none(), st.lists(_factor, min_size=len(FACTORS), max_size=len(FACTORS)))
    fa = draw(st.lists(frow, min_size=N_SESSIONS, max_size=N_SESSIONS))
    fb_future = draw(st.lists(frow, min_size=N_SESSIONS, max_size=N_SESSIONS))
    fb = [fa[i] if SESSIONS[i] <= cutoff else fb_future[i] for i in range(N_SESSIONS)]

    # Synthetic features: one optional row per session (a missing row must stay missing).
    srow = st.one_of(
        st.none(), st.lists(_syn, min_size=len(SYN_FEATURES), max_size=len(SYN_FEATURES))
    )
    sa = draw(st.lists(srow, min_size=N_SESSIONS, max_size=N_SESSIONS))
    sb_future = draw(st.lists(srow, min_size=N_SESSIONS, max_size=N_SESSIONS))
    sb = sa[: t + 1] + sb_future[t + 1 :]

    t_date = SESSIONS[t]
    alfred_a: dict[str, pd.DataFrame] = {}
    alfred_b: dict[str, pd.DataFrame] = {}
    for sid in SERIES:
        past = draw(releases(SESSIONS[0] - pd.Timedelta(days=45), t_date - pd.Timedelta(days=1)))
        future_hi = SESSIONS[-1] + pd.Timedelta(days=5)
        fut_a = draw(releases(t_date, future_hi))
        fut_b = draw(releases(t_date, future_hi))
        alfred_a[sid] = releases_to_vintages(sid, past + fut_a)
        alfred_b[sid] = releases_to_vintages(sid, past + fut_b)

    return Scenario(
        a=PitInputs(
            adj_close=_price_frame(prices),
            french=_french_frame(fa),
            alfred=alfred_a,
            synthetic=_dated_frame(sa, SYN_FEATURES),
        ),
        b=PitInputs(
            adj_close=_price_frame(prices_b),
            french=_french_frame(fb),
            alfred=alfred_b,
            synthetic=_dated_frame(sb, SYN_FEATURES),
        ),
        t=t,
        config=config,
    )


def _french_frame(rows: list[list[float] | None]) -> pd.DataFrame:
    return _dated_frame(rows, FACTORS)


def _dated_frame(rows: list[list[float] | None], columns: Sequence[str]) -> pd.DataFrame:
    keep = [i for i, r in enumerate(rows) if r is not None]
    values = [rows[i] for i in keep]
    index = pd.DatetimeIndex(SESSIONS[keep], name="date")
    return pd.DataFrame(values, index=index, columns=list(columns), dtype="float64")
