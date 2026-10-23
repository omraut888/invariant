import re
from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from invariant.core.types import (
    Frequency,
    KillReason,
    StageName,
    TimeSeriesValidationError,
    Verdict,
    make_signal_id,
    validate_time_index,
)


def _series(index: pd.Index) -> "pd.Series[float]":
    return pd.Series(np.arange(len(index), dtype=float), index=index)


# --- validate_time_index: contract violations -------------------------------------------------


def test_valid_series_returned_unchanged() -> None:
    s = _series(pd.bdate_range("2020-01-01", periods=10))
    assert validate_time_index(s, Frequency.DAILY) is s


def test_valid_dataframe_accepted() -> None:
    df = pd.DataFrame({"a": range(5)}, index=pd.bdate_range("2020-01-01", periods=5))
    assert validate_time_index(df, Frequency.DAILY) is df


@pytest.mark.parametrize(
    ("index", "match"),
    [
        (pd.RangeIndex(3), "DatetimeIndex"),
        (pd.Index(["2020-01-01", "2020-01-02"]), "DatetimeIndex"),
        (pd.DatetimeIndex([]), "empty"),
        (pd.date_range("2020-01-01", periods=3, tz="UTC"), "tz-naive"),
        (pd.DatetimeIndex(["2020-01-01", pd.NaT]), "NaT"),
        (pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02"]), "duplicate"),
        (pd.DatetimeIndex(["2020-01-02", "2020-01-01"]), "not sorted"),
    ],
)
def test_contract_violations(index: pd.Index, match: str) -> None:
    with pytest.raises(TimeSeriesValidationError, match=match):
        validate_time_index(_series(index))


def test_error_is_value_error() -> None:
    assert issubclass(TimeSeriesValidationError, ValueError)


def test_no_frequency_check_when_freq_none() -> None:
    # Weekend and intraday stamps are fine when no frequency is asserted.
    idx = pd.DatetimeIndex(["2020-01-04 10:00", "2020-01-05 12:30"])
    validate_time_index(_series(idx))


# --- daily rule ------------------------------------------------------------------------------


def test_daily_with_exchange_holidays_passes() -> None:
    # 2019-12-23 .. 2020-01-10 NYSE days: Christmas and New Year missing.
    days = pd.bdate_range("2019-12-23", "2020-01-10").drop(
        pd.DatetimeIndex(["2019-12-25", "2020-01-01"])
    )
    validate_time_index(_series(days), Frequency.DAILY)


def test_daily_sept_2001_closure_passes() -> None:
    days = pd.DatetimeIndex(["2001-09-07", "2001-09-10", "2001-09-17", "2001-09-18"])
    validate_time_index(_series(days), Frequency.DAILY)


def test_daily_weekend_rejected() -> None:
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-04"])
    with pytest.raises(TimeSeriesValidationError, match="weekend"):
        validate_time_index(_series(idx), Frequency.DAILY)


def test_daily_intraday_rejected() -> None:
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03 16:00"])
    with pytest.raises(TimeSeriesValidationError, match="midnight"):
        validate_time_index(_series(idx), Frequency.DAILY)


def test_daily_gap_too_large_rejected() -> None:
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-13"])  # 7 business days
    with pytest.raises(TimeSeriesValidationError, match="gap of 7 business days"):
        validate_time_index(_series(idx), Frequency.DAILY)
    validate_time_index(_series(idx), Frequency.DAILY, max_gap_bdays=7)


def test_max_gap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_gap_bdays"):
        validate_time_index(_series(pd.bdate_range("2020-01-01", periods=2)), max_gap_bdays=0)


def test_single_observation_daily_passes() -> None:
    validate_time_index(_series(pd.DatetimeIndex(["2020-01-02"])), Frequency.DAILY)


@st.composite
def _bday_indices(draw: st.DrawFn, max_gap: int = 5) -> pd.DatetimeIndex:
    start = draw(st.dates(min_value=date(1990, 1, 1), max_value=date(2040, 1, 1)))
    gaps = draw(st.lists(st.integers(min_value=1, max_value=max_gap), min_size=0, max_size=60))
    first = np.busday_offset(np.datetime64(start, "D"), 0, roll="forward")
    offsets = np.concatenate([[0], np.cumsum(gaps, dtype=np.int64)])
    days = np.busday_offset(first, offsets)
    return pd.DatetimeIndex(days.astype("datetime64[ns]"))


@given(_bday_indices())
def test_property_sorted_weekdays_with_small_gaps_pass(index: pd.DatetimeIndex) -> None:
    validate_time_index(_series(index), Frequency.DAILY)


@given(_bday_indices().filter(lambda i: len(i) >= 2).flatmap(lambda i: st.permutations(list(i))))
def test_property_unsorted_permutation_fails(perm: list[pd.Timestamp]) -> None:
    assume(perm != sorted(perm))
    with pytest.raises(TimeSeriesValidationError, match="not sorted"):
        validate_time_index(_series(pd.DatetimeIndex(perm)), Frequency.DAILY)


# --- weekly / monthly rules ------------------------------------------------------------------


def test_weekly_holiday_shift_passes() -> None:
    fridays = pd.date_range("2020-03-27", periods=5, freq="W-FRI")
    # Good Friday 2020-04-10 -> sample on Thursday 2020-04-09 instead.
    shifted = fridays.where(fridays != pd.Timestamp("2020-04-10"), pd.Timestamp("2020-04-09"))
    validate_time_index(_series(pd.DatetimeIndex(shifted)), Frequency.WEEKLY)


def test_weekly_skipped_week_rejected() -> None:
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-17"])
    with pytest.raises(TimeSeriesValidationError, match="spans 2 periods"):
        validate_time_index(_series(idx), Frequency.WEEKLY)


def test_weekly_two_in_one_week_rejected() -> None:
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-08"])
    with pytest.raises(TimeSeriesValidationError, match="spans 0 periods"):
        validate_time_index(_series(idx), Frequency.WEEKLY)


def test_monthly_business_month_end_passes() -> None:
    validate_time_index(
        _series(pd.date_range("2019-01-01", periods=24, freq="BME")), Frequency.MONTHLY
    )


@pytest.mark.parametrize(
    "dates",
    [["2020-01-31", "2020-03-31"], ["2020-01-02", "2020-01-31"]],
)
def test_monthly_irregular_rejected(dates: list[str]) -> None:
    with pytest.raises(TimeSeriesValidationError, match="one observation per consecutive"):
        validate_time_index(_series(pd.DatetimeIndex(dates)), Frequency.MONTHLY)


# --- identifiers and enums -------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["mom_12_1", "a", "macro:cpi.yoy-v2", "0x", "a" * 128])
def test_valid_signal_ids(raw: str) -> None:
    assert make_signal_id(raw) == raw


@pytest.mark.parametrize("raw", ["", "Mom", "_x", "-x", "a b", "a/b", "é", "a" * 129])
def test_invalid_signal_ids(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid signal id"):
        make_signal_id(raw)


def test_enum_values_pinned() -> None:
    # Values are persisted in the kill ledger; renaming one is a breaking change.
    assert [s.value for s in StageName] == ["causal", "decay", "dro", "adversarial", "memo"]
    assert [v.value for v in Verdict] == ["pass", "kill"]
    assert {r.value for r in KillReason} == {
        "general.insufficient_data",
        "general.data_validation_failed",
        "general.assumption_violation",
        "general.placeholder",
        "causal.no_significant_link",
        "causal.not_significant_after_correction",
        "decay.edge_declining",
    }
    assert [f.value for f in Frequency] == ["daily", "weekly", "monthly"]


_PREFIXES = "|".join(["general", *(s.value for s in StageName)])
_REASON_RE = re.compile(rf"^({_PREFIXES})\.[a-z][a-z0-9_]*$")


@pytest.mark.parametrize("reason", list(KillReason))
def test_kill_reason_naming_convention(reason: KillReason) -> None:
    assert _REASON_RE.fullmatch(reason.value), reason.value
    prefix = reason.value.split(".", 1)[0]
    if prefix == "general":
        assert reason.stage is None
    else:
        assert reason.stage is StageName(prefix)


def test_enums_are_str() -> None:
    assert KillReason("general.insufficient_data") == "general.insufficient_data"
    assert StageName("dro") is StageName.DRO
