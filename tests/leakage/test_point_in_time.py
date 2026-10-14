"""Look-ahead guards for the point-in-time panel (``invariant.data.pit``).

1. Perturbing every raw input whose information time is after t leaves every feature and
   observation date at sessions <= t bit-identical (Hypothesis).
2. The property test detects leaks: deliberately leaky variants of the panel fail it.
3. On a real ALFRED excerpt (PAYEMS), the value used is the vintage in force, not the final
   revised value, and a release is invisible on its own release date.
4. No target at t depends on returns dated at or before t.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, Phase, find, given, settings
from hypothesis import strategies as st
from tests.fixtures.pit import (
    Scenario,
    bit_identical,
    payems_vintages,
    pit_config,
    prefix_identical,
    scenarios,
)

from invariant.config.settings import PitConfig, TargetConfig
from invariant.data.pit import PitInputs, PitPanel, assemble_panel, forward_returns
from invariant.data.sources.etf import session_calendar

pytestmark = pytest.mark.leakage

Builder = Callable[[PitInputs, PitConfig], PitPanel]


def _leaks(build: Builder, s: Scenario) -> bool:
    return not prefix_identical(build(s.a, s.config), build(s.b, s.config), s.t)


# --- 1. the property --------------------------------------------------------------------------


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_features_at_or_before_t_ignore_everything_after_t(s: Scenario) -> None:
    pa, pb = assemble_panel(s.a, s.config), assemble_panel(s.b, s.config)
    assert prefix_identical(pa, pb, s.t)


# --- 2. the property catches leaks ------------------------------------------------------------


def _same_day_alfred(inputs: PitInputs, config: PitConfig) -> PitPanel:
    """Treat a vintage as usable on its release date (violates ADR 0004's next-session rule)."""
    day = pd.Timedelta(days=1)
    shifted = {
        sid: t.assign(
            realtime_start=t["realtime_start"] - day, realtime_end=t["realtime_end"] - day
        )
        for sid, t in inputs.alfred.items()
    }
    return assemble_panel(dataclasses.replace(inputs, alfred=shifted), config)


def _final_values(inputs: PitInputs, config: PitConfig) -> PitPanel:
    """Use each observation's final revised value from its first release on."""
    final = {
        sid: t.assign(value=t.groupby("observation_date")["value"].transform("last"))
        for sid, t in inputs.alfred.items()
    }
    return assemble_panel(dataclasses.replace(inputs, alfred=final), config)


def _french_without_lag(inputs: PitInputs, config: PitConfig) -> PitPanel:
    """Use French factors on their own date, ignoring the publication lag."""
    unlagged = config.model_copy(
        update={
            "publication_lag_sessions": config.publication_lag_sessions.model_copy(
                update={"french": 0}
            )
        }
    )
    return assemble_panel(inputs, unlagged)


def _synthetic_next_day(inputs: PitInputs, config: PitConfig) -> PitPanel:
    """Use the synthetic feature dated t+1 at t (lag -1)."""
    assert inputs.synthetic is not None
    shifted = inputs.synthetic.reindex(inputs.adj_close.index).shift(-1)
    return assemble_panel(dataclasses.replace(inputs, synthetic=shifted), config)


def _next_close(inputs: PitInputs, config: PitConfig) -> PitPanel:
    """Compute the return at t from the close of t+1."""
    return assemble_panel(dataclasses.replace(inputs, adj_close=inputs.adj_close.shift(-1)), config)


@pytest.mark.parametrize(
    "leaky",
    [_same_day_alfred, _final_values, _french_without_lag, _next_close, _synthetic_next_day],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_property_detects_known_leaks(leaky: Builder) -> None:
    # find() raises NoSuchExample if the scenario strategy cannot expose the leak, which would
    # mean the property test above is too weak to be trusted.
    example = find(
        scenarios(),
        lambda s: _leaks(leaky, s),
        settings=settings(
            max_examples=2000,
            database=None,
            derandomize=True,
            phases=[Phase.generate],
            suppress_health_check=[HealthCheck.too_slow],
        ),
    )
    assert not _leaks(assemble_panel, example)  # the real panel is clean on the same scenario


# --- 3. a real revised series -----------------------------------------------------------------


def _payems_panel() -> PitPanel:
    sessions = session_calendar(
        "XNYS", pd.Timestamp("2024-01-02").date(), pd.Timestamp("2026-06-30").date()
    )
    sessions = pd.DatetimeIndex(sessions.astype("datetime64[us]"), name="date")
    adj = pd.DataFrame({"SPY": 100.0}, index=sessions)
    return assemble_panel(
        PitInputs(adj_close=adj, alfred={"PAYEMS": payems_vintages()}), pit_config()
    )


def _at(panel: PitPanel, day: str) -> tuple[float, pd.Timestamp]:
    ts = pd.Timestamp(day)
    value = float(panel.features["macro_PAYEMS"].loc[ts])
    return value, pd.Timestamp(panel.obs_dates["macro_PAYEMS"].loc[ts])


def test_vintage_value_differs_from_final_revised_value() -> None:
    panel = _payems_panel()
    final = payems_vintages().groupby("observation_date")["value"].last()
    # Every session before the last benchmark revision becomes usable (2026-02-12, the session
    # after its 2026-02-11 release) uses a value that differs from the final revised one.
    before = panel.features.index < pd.Timestamp("2026-02-12")
    used = panel.features.loc[before, "macro_PAYEMS"]
    obs = panel.obs_dates.loc[before, "macro_PAYEMS"]
    known = used.notna()
    assert known.sum() > 400
    assert (used[known].to_numpy() != final.loc[obs[known]].to_numpy()).all()
    # Concrete points, values as published in ALFRED.
    assert _at(panel, "2024-03-07") == (157700.0, pd.Timestamp("2024-01-01"))  # Jan, 1st print
    assert _at(panel, "2025-02-06") == (158106.0, pd.Timestamp("2024-03-01"))  # Mar, 3rd print
    assert _at(panel, "2025-02-10") == (157517.0, pd.Timestamp("2024-03-01"))  # 2025 benchmark
    assert _at(panel, "2026-02-12") == (157466.0, pd.Timestamp("2024-03-01"))  # final


def test_release_is_invisible_on_its_own_release_date() -> None:
    panel = _payems_panel()
    # The February 2024 report was released Friday 2024-03-08: usable from Monday 2024-03-11.
    assert _at(panel, "2024-03-08") == (157700.0, pd.Timestamp("2024-01-01"))
    assert _at(panel, "2024-03-11") == (157808.0, pd.Timestamp("2024-02-01"))
    # Nothing is known before the first release (2024-02-02), nor on its date.
    assert np.isnan(_at(panel, "2024-02-02")[0])
    assert _at(panel, "2024-02-05") == (157700.0, pd.Timestamp("2024-01-01"))


# --- 4. targets -------------------------------------------------------------------------------

_N = 30
_INDEX = pd.DatetimeIndex(
    pd.bdate_range("2025-03-03", periods=_N).astype("datetime64[us]"), name="date"
)
_ret = st.one_of(st.floats(min_value=-0.2, max_value=0.2), st.just(float("nan")))


@settings(max_examples=200, deadline=None)
@given(
    st.lists(_ret, min_size=_N, max_size=_N),
    st.lists(_ret, min_size=_N, max_size=_N),
    st.integers(0, _N - 1),
    st.integers(0, 3),
    st.integers(1, 5),
)
def test_no_target_depends_on_returns_at_or_before_t(
    base: list[float], other: list[float], t: int, execution_lag: int, horizon: int
) -> None:
    timing = TargetConfig(execution_lag=execution_lag, horizon=horizon)
    a = pd.DataFrame({"X": base}, index=_INDEX)
    b = pd.DataFrame({"X": other[: t + 1] + base[t + 1 :]}, index=_INDEX)  # rewrite 0..t
    ya, yb = forward_returns(a, timing), forward_returns(b, timing)
    assert bit_identical(ya.iloc[t:], yb.iloc[t:])  # targets at t and later: unchanged


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_panel_targets_ignore_prices_before_entry(s: Scenario) -> None:
    # y_t = ln(P_{t+e+h} / P_{t+e}), so prices dated before t+e must not matter. (With e = 0
    # the target uses P_t through r_{t+1}: that is the definition of a return, not a leak.)
    e = s.config.targets.execution_lag
    cut = s.t + e  # rewrite prices at sessions < cut
    prices_b = s.a.adj_close.copy()
    prices_b.iloc[:cut] = s.b.adj_close.iloc[::-1].to_numpy()[:cut]
    pa = assemble_panel(s.a, s.config)
    pb = assemble_panel(dataclasses.replace(s.a, adj_close=prices_b), s.config)
    assert bit_identical(pa.targets.iloc[s.t :], pb.targets.iloc[s.t :])
