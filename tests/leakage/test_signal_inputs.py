"""Look-ahead guards for the real-data signal inputs (docs/adr/0019).

1. ``market_asof``: archived values follow ADR 0004's rule exactly, pre-archive values become
   usable ``pre_archive_lag`` sessions after their date, and nothing dated after t moves a value
   at t, including the current (final) values of archived observations, which are never used.
2. ``within_vintage_change``: the change at t is computed inside the vintage usable at t, and
   nothing released on or after t moves it.
3. Each property is itself tested: ``hypothesis.find`` must expose deliberately leaky variants
   (a pre-archive value one session early, same-day archived and macro releases, the base
   value of a change taken from the final vintage).
4. Mixed-vintage differencing uses no future data, so no look-ahead property can catch it. The
   oracle comparison does: ``find`` must expose a scenario where it differs from the change
   computed inside one vintage, and a hand-built revision shows the difference.
5. Signal windows: a return after t never moves a signal at t.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, Phase, find, given, settings
from hypothesis import strategies as st
from tests.fixtures.pit import (
    OBS_DATES,
    SESSIONS,
    Release,
    bit_identical,
    releases,
    releases_to_vintages,
)

from invariant.data.pit import alfred_asof, market_asof, within_vintage_change
from invariant.data.signals import REGISTRY, SignalInputs, _window_sum

pytestmark = pytest.mark.leakage

SID = "SER"
_FIND = settings(
    max_examples=3000,
    database=None,
    derandomize=True,
    phases=[Phase.generate],
    suppress_health_check=[HealthCheck.too_slow],
)
_value = st.sampled_from([1.0, 2.0, 3.0, math.nan])


def _prefix_equal(a: pd.DataFrame, b: pd.DataFrame, t: int) -> bool:
    return bit_identical(a.iloc[: t + 1], b.iloc[: t + 1])


# --- 1. market_asof ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketCase:
    """Two input sets that agree on everything usable at or before session ``t``."""

    t: int
    k: int
    """The first vintage is ``SESSIONS[k]``."""
    lag: int
    releases_a: list[Release]
    releases_b: list[Release]
    current_a: Mapping[pd.Timestamp, float]
    current_b: Mapping[pd.Timestamp, float]

    @property
    def first_vintage(self) -> date:
        return SESSIONS[self.k].date()


def _market_releases(
    draw: st.DrawFn, obs: Sequence[pd.Timestamp], lo: pd.Timestamp, hi: pd.Timestamp
) -> list[Release]:
    if lo > hi or not obs:
        return []
    days = draw(st.lists(st.integers(0, (hi - lo).days), unique=True, max_size=4).map(sorted))
    vintage = st.dictionaries(st.sampled_from(list(obs)), _value, max_size=len(obs))
    return [(lo + pd.Timedelta(days=d), draw(vintage)) for d in days]


@st.composite
def market_cases(draw: st.DrawFn) -> MarketCase:
    n = len(SESSIONS)
    t = draw(st.integers(0, n - 1))
    k = draw(st.integers(0, n - 1))
    lag = draw(st.integers(1, 4))
    archived = list(SESSIONS[k : k + 6])
    fv, t_date = SESSIONS[k], SESSIONS[t]
    past = _market_releases(draw, archived, fv, t_date - pd.Timedelta(days=1))
    hi = SESSIONS[-1] + pd.Timedelta(days=5)
    fut_a = _market_releases(draw, archived, max(t_date, fv), hi)
    fut_b = _market_releases(draw, archived, max(t_date, fv), hi)
    # Current values: every pre-archive session plus the archived dates. A pre-archive value is
    # usable at its index + lag; the current value of an archived date is never usable.
    dates = [*SESSIONS[:k], *archived]
    cur_a = {d: draw(_value) for d in dates}
    cur_b = {
        d: (cur_a[d] if (d < fv and i + lag <= t) else draw(_value)) for i, d in enumerate(dates)
    }
    return MarketCase(t, k, lag, past + fut_a, past + fut_b, cur_a, cur_b)


def _observations(current: Mapping[pd.Timestamp, float]) -> pd.DataFrame:
    dates = sorted(current)
    return pd.DataFrame(
        {
            "series_id": pd.Series([SID] * len(dates), dtype="str"),
            "observation_date": pd.DatetimeIndex(dates).astype("datetime64[us]"),
            "value": np.asarray([current[d] for d in dates], dtype="float64"),
        }
    )


MarketBuilder = Callable[[list[Release], Mapping[pd.Timestamp, float], MarketCase], pd.DataFrame]


def _market(
    rels: list[Release], current: Mapping[pd.Timestamp, float], c: MarketCase
) -> pd.DataFrame:
    return market_asof(
        releases_to_vintages(SID, rels),
        _observations(current),
        SID,
        first_vintage=c.first_vintage,
        pre_archive_lag=c.lag,
        sessions=SESSIONS,
    )


def _market_oracle(c: MarketCase) -> pd.DataFrame:
    """At session s: the latest archived release dated strictly before s, plus every
    pre-archive value whose index + lag is <= s; the newest observation with a value."""
    fv = SESSIONS[c.k]
    ordered = sorted(c.releases_a, key=lambda r: r[0])
    values, obs = [], []
    for s, day in enumerate(SESSIONS):
        known = [v for d, v in ordered if d < day]
        view = {o: x for o, x in (known[-1] if known else {}).items() if not math.isnan(x)}
        for i, d in enumerate(SESSIONS[: c.k]):
            if d < fv and i + c.lag <= s and not math.isnan(c.current_a[d]):
                view[d] = c.current_a[d]
        newest = max(view, default=None)
        obs.append(newest)
        values.append(np.nan if newest is None else view[newest])
    return pd.DataFrame(
        {
            "value": np.asarray(values),
            "obs_date": pd.Series(obs, dtype="datetime64[us]").to_numpy(),
        },
        index=SESSIONS,
    )


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(market_cases())
def test_market_values_at_or_before_t_ignore_everything_after_t(c: MarketCase) -> None:
    a = _market(c.releases_a, c.current_a, c)
    b = _market(c.releases_b, c.current_b, c)
    assert _prefix_equal(a, b, c.t)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(market_cases())
def test_market_asof_matches_the_oracle(c: MarketCase) -> None:
    got = _market(c.releases_a, c.current_a, c)
    assert bit_identical(got, _market_oracle(c))


def _market_lag_minus_one(
    rels: list[Release], cur: Mapping[pd.Timestamp, float], c: MarketCase
) -> pd.DataFrame:
    """A pre-archive value one session before its lag allows."""
    return _market(rels, cur, dataclasses.replace(c, lag=c.lag - 1) if c.lag > 1 else c)


def _market_same_day(
    rels: list[Release], cur: Mapping[pd.Timestamp, float], c: MarketCase
) -> pd.DataFrame:
    """An archived value usable on the day it is posted (violates ADR 0004)."""
    shifted = [(d - pd.Timedelta(days=1), v) for d, v in rels]
    return _market(shifted, cur, c)


def _market_final_values(
    rels: list[Release], cur: Mapping[pd.Timestamp, float], c: MarketCase
) -> pd.DataFrame:
    """The current (final) value of an archived observation from its first posting on."""
    fv = SESSIONS[c.k]
    final: list[Release] = [
        (d, {o: (cur[o] if o >= fv else x) for o, x in v.items()}) for d, v in rels
    ]
    return _market(final, cur, c)


@pytest.mark.parametrize(
    "leaky",
    [_market_lag_minus_one, _market_same_day, _market_final_values],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_market_property_detects_known_leaks(leaky: MarketBuilder) -> None:
    def leaks(c: MarketCase) -> bool:
        return not _prefix_equal(
            leaky(c.releases_a, c.current_a, c), leaky(c.releases_b, c.current_b, c), c.t
        )

    example = find(market_cases(), leaks, settings=_FIND)
    clean = _prefix_equal(
        _market(example.releases_a, example.current_a, example),
        _market(example.releases_b, example.current_b, example),
        example.t,
    )
    assert clean


def test_archive_rows_of_pre_archive_observations_are_ignored() -> None:
    fv = SESSIONS[5]
    vintages = releases_to_vintages(SID, [(fv, {SESSIONS[2]: 9.0, SESSIONS[5]: 1.0})])
    out = market_asof(
        vintages,
        _observations({SESSIONS[2]: 2.0}),
        SID,
        first_vintage=fv.date(),
        pre_archive_lag=2,
        sessions=SESSIONS,
    )
    assert np.isnan(out["value"].iloc[3])
    assert out["value"].iloc[4] == 2.0  # the current value, 2 sessions after its date
    assert out["value"].iloc[6] == 1.0  # the archived observation, the session after fv


# --- 2. within_vintage_change -----------------------------------------------------------------


@dataclass(frozen=True)
class MacroCase:
    t: int
    months: int
    log: bool
    releases_a: list[Release]
    releases_b: list[Release]


@st.composite
def macro_cases(draw: st.DrawFn) -> MacroCase:
    t = draw(st.integers(0, len(SESSIONS) - 1))
    t_date = SESSIONS[t]
    past = draw(releases(SESSIONS[0] - pd.Timedelta(days=45), t_date - pd.Timedelta(days=1)))
    hi = SESSIONS[-1] + pd.Timedelta(days=5)
    fut_a, fut_b = draw(releases(t_date, hi)), draw(releases(t_date, hi))
    return MacroCase(t, draw(st.integers(1, 2)), draw(st.booleans()), past + fut_a, past + fut_b)


def _month_back(o: pd.Timestamp, months: int) -> pd.Timestamp:
    return o - pd.DateOffset(months=months)


def _macro_oracle(
    rels: list[Release], months: int, log: bool, *, base: str = "same"
) -> pd.DataFrame:
    """At session s, the vintage is the latest release dated strictly before s. ``base``:
    ``same`` (the method), ``final`` (o - months from the last release: look-ahead), or
    ``diagonal`` (o - months as known when it was the newest observation: mixed vintages)."""
    ordered = sorted(rels, key=lambda r: r[0])
    f = np.log if log else (lambda x: x)
    values, obs = [], []
    for day in SESSIONS:
        known = [v for d, v in ordered if d < day]
        view = {o: x for o, x in (known[-1] if known else {}).items() if not math.isnan(x)}
        pick = None
        for o in sorted(view, reverse=True):
            b = _month_back(o, months)
            if base == "same":
                bv = view.get(b)
            elif base == "final":
                last = {o2: x for o2, x in ordered[-1][1].items() if not math.isnan(x)}
                bv = last.get(b) if b in view else None
            else:  # the newest-observation series, read when b was the newest
                older = [
                    {o2: x for o2, x in v.items() if not math.isnan(x)}
                    for d, v in ordered
                    if d < day
                ]
                diag = [w[b] for w in older if w and max(w) == b]
                bv = diag[-1] if diag and b in view else None
            if bv is not None:
                pick = (o, float(f(view[o]) - f(bv)))
                break
        obs.append(None if pick is None else pick[0])
        values.append(np.nan if pick is None else pick[1])
    return pd.DataFrame(
        {
            "value": np.asarray(values),
            "obs_date": pd.Series(obs, dtype="datetime64[us]").to_numpy(),
        },
        index=SESSIONS,
    )


def _macro(rels: list[Release], c: MacroCase) -> pd.DataFrame:
    return within_vintage_change(
        releases_to_vintages(SID, rels), SID, SESSIONS, months=c.months, log=c.log
    )


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(macro_cases())
def test_changes_at_or_before_t_ignore_releases_after_t(c: MacroCase) -> None:
    assert _prefix_equal(_macro(c.releases_a, c), _macro(c.releases_b, c), c.t)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(macro_cases())
def test_the_change_is_computed_inside_one_vintage(c: MacroCase) -> None:
    got = _macro(c.releases_a, c)
    want = _macro_oracle(c.releases_a, c.months, c.log)
    assert got["obs_date"].equals(want["obs_date"])
    np.testing.assert_allclose(got["value"], want["value"], rtol=0, atol=1e-15)


MacroBuilder = Callable[[list[Release], MacroCase], pd.DataFrame]


def _macro_same_day(rels: list[Release], c: MacroCase) -> pd.DataFrame:
    """A release usable on its own date (violates ADR 0004)."""
    return _macro([(d - pd.Timedelta(days=1), v) for d, v in rels], c)


def _macro_final_base(rels: list[Release], c: MacroCase) -> pd.DataFrame:
    """The base value of the change taken from the final vintage."""
    return _macro_oracle(rels, c.months, c.log, base="final")


@pytest.mark.parametrize(
    "leaky", [_macro_same_day, _macro_final_base], ids=lambda f: f.__name__.lstrip("_")
)
def test_change_property_detects_known_leaks(leaky: MacroBuilder) -> None:
    def leaks(c: MacroCase) -> bool:
        return not _prefix_equal(leaky(c.releases_a, c), leaky(c.releases_b, c), c.t)

    example = find(macro_cases(), leaks, settings=_FIND)
    assert _prefix_equal(
        _macro(example.releases_a, example), _macro(example.releases_b, example), example.t
    )


# --- 4. mixed vintages ------------------------------------------------------------------------


def test_the_oracle_comparison_detects_mixed_vintage_differencing() -> None:
    def mixes(c: MacroCase) -> bool:
        mixed = _macro_oracle(c.releases_a, c.months, c.log, base="diagonal")
        same = _macro(c.releases_a, c)
        return not np.array_equal(mixed["value"], same["value"], equal_nan=True)

    find(macro_cases(), mixes, settings=_FIND)


def test_a_revision_released_with_the_new_month_is_used() -> None:
    dec, jan, feb = OBS_DATES[0], OBS_DATES[1], OBS_DATES[2]
    rels: list[Release] = [
        (SESSIONS[2], {dec: 100.0, jan: 101.0}),
        (SESSIONS[10], {dec: 100.0, jan: 103.0, feb: 104.0}),  # revises January
    ]
    vintages = releases_to_vintages(SID, rels)
    change = within_vintage_change(vintages, SID, SESSIONS, months=1, log=True)
    asof = alfred_asof(vintages, SID, SESSIONS)
    after = 11  # the first session after the second release
    assert change["value"].iloc[after] == pytest.approx(math.log(104 / 103))
    mixed = math.log(asof["value"].iloc[after]) - math.log(asof["value"].iloc[after - 1])
    assert mixed == pytest.approx(math.log(104 / 101))  # what differencing the as-of gives
    assert change["value"].iloc[after - 1] == pytest.approx(math.log(101 / 100))


# --- 5. signal windows ------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    st.lists(st.floats(-0.1, 0.1), min_size=30, max_size=30),
    st.lists(st.floats(-0.1, 0.1), min_size=30, max_size=30),
    st.integers(0, 29),
    st.integers(0, 5),
    st.integers(0, 10),
)
def test_a_window_sum_ignores_values_after_t(
    base: list[float], other: list[float], t: int, newest: int, width: int
) -> None:
    a = pd.Series(base)
    b = pd.Series(base[: t + 1] + other[t + 1 :])
    wa, wb = _window_sum(a, newest, newest + width), _window_sum(b, newest, newest + width)
    assert np.array_equal(wa.iloc[: t + 1], wb.iloc[: t + 1], equal_nan=True)


def test_a_window_reaching_t_plus_one_is_caught() -> None:
    def leaky(x: pd.Series, newest: int, oldest: int) -> pd.Series:
        return _window_sum(x.shift(-1), newest, oldest)

    a = pd.Series(np.arange(30, dtype="float64"))
    b = a.copy()
    b.iloc[11:] += 1.0  # the future after t = 10
    same = np.array_equal(
        _window_sum(a, 0, 4).iloc[:11], _window_sum(b, 0, 4).iloc[:11], equal_nan=True
    )
    assert same
    assert not np.array_equal(leaky(a, 0, 4).iloc[:11], leaky(b, 0, 4).iloc[:11], equal_nan=True)


def test_registry_signals_at_t_ignore_returns_after_t() -> None:
    sessions = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=400), name="date")
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 400)
    t = 300
    r_b = r.copy()
    r_b[t + 1 :] = rng.normal(0, 0.05, 400 - t - 1)
    flat = pd.DataFrame(
        {"value": 20.0, "obs_date": sessions.astype("datetime64[us]")}, index=sessions
    )
    market = {s: flat for s in ("VIXCLS", "T10Y3M", "DBAA", "DAAA", "DTB3")}

    def run(ret: np.ndarray) -> dict[str, pd.Series]:
        inputs = SignalInputs(
            sessions=sessions,
            returns=pd.DataFrame({"SPY": ret}, index=sessions),
            market=market,
            alfred={},
        )
        return {i: d.compute(inputs)[0] for i, d in REGISTRY.items() if not d.alfred}

    a, b = run(r), run(r_b)
    for sid in a:
        assert np.array_equal(a[sid].iloc[: t + 1], b[sid].iloc[: t + 1], equal_nan=True), sid
        assert a[sid].iloc[260:].notna().all(), sid  # windows are complete after warm-up
