"""Core typed contracts shared across config, data, stages, and the kill ledger.

Time-series validation
----------------------
``validate_time_index`` enforces the module-boundary contract for every time series in the
system: a non-empty, tz-naive, strictly increasing ``DatetimeIndex`` with no duplicates or NaT,
optionally conforming to a sampling frequency.

The daily check is a *trading-day* check, not ``pd.infer_freq``: exchange calendars have
holidays, so real daily data never has a regular pandas frequency. The rule is that every
timestamp is a weekday at midnight and no two consecutive observations are more than
``max_gap_bdays`` business days apart (default 5, which admits the four-day NYSE closure after
11 September 2001). Limitation: this does not verify membership in the exact NYSE calendar, so
a missing single trading day or an observation on an exchange holiday passes. Exact calendar
checks belong to the data layer (see docs/adr/0001-config-layering-and-core-contracts.md).

Weekly and monthly checks require midnight timestamps and exactly one observation per
consecutive calendar week / month (no skipped or repeated periods), which tolerates
holiday-shifted sampling days (e.g. Thursday instead of Good Friday).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, NewType, TypeVar

import numpy as np
import pandas as pd

__all__ = [
    "Frequency",
    "KillReason",
    "SignalId",
    "StageName",
    "TimeSeriesValidationError",
    "Verdict",
    "make_signal_id",
    "validate_time_index",
]


class Frequency(StrEnum):
    """Sampling frequency of a time series."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class StageName(StrEnum):
    """The five pipeline stages, in execution order."""

    CAUSAL = "causal"
    DECAY = "decay"
    DRO = "dro"
    ADVERSARIAL = "adversarial"
    MEMO = "memo"


class Verdict(StrEnum):
    """Outcome of a stage for one candidate signal."""

    PASS = "pass"
    KILL = "kill"


_GENERAL_PREFIX = "general"


class KillReason(StrEnum):
    """Reason code recorded in the kill ledger when a stage kills a candidate.

    Convention: every value is ``"<prefix>.<snake_case_code>"`` where ``<prefix>`` is either a
    ``StageName`` value (stage-specific reasons) or ``"general"`` (reasons any stage may emit).
    Python enums with members cannot be subclassed, so stages extend this enum by adding members
    here. tests/unit/test_types.py enforces the convention for every member.
    """

    INSUFFICIENT_DATA = "general.insufficient_data"
    DATA_VALIDATION_FAILED = "general.data_validation_failed"
    ASSUMPTION_VIOLATION = "general.assumption_violation"
    # Only for ledger entries marked ``placeholder=True`` (smoke runs, tests); the kill ledger
    # rejects it otherwise. Never emitted by a real stage.
    PLACEHOLDER = "general.placeholder"

    @property
    def stage(self) -> StageName | None:
        """The stage this reason belongs to, or ``None`` for cross-cutting reasons."""
        prefix = self.value.split(".", 1)[0]
        return None if prefix == _GENERAL_PREFIX else StageName(prefix)


SignalId = NewType("SignalId", str)
"""Identifier of a candidate signal. Construct only via ``make_signal_id``."""

_SIGNAL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


def make_signal_id(raw: str) -> SignalId:
    """Validate ``raw`` and return it as a ``SignalId``.

    Allowed: 1-128 chars of lowercase ASCII letters, digits, ``_ . : -``, starting with a letter
    or digit. The restricted alphabet keeps ids safe as file names and ledger keys.
    """
    if not _SIGNAL_ID_RE.fullmatch(raw):
        raise ValueError(
            f"invalid signal id {raw!r}: must match {_SIGNAL_ID_RE.pattern}",
        )
    return SignalId(raw)


class TimeSeriesValidationError(ValueError):
    """A time series violates the module-boundary index contract."""


# Any: pandas-stubs makes Series generic over its dtype; validation is dtype-agnostic.
_TS = TypeVar("_TS", "pd.Series[Any]", pd.DataFrame)


def validate_time_index(
    obj: _TS,
    freq: Frequency | None = None,
    *,
    max_gap_bdays: int = 5,
) -> _TS:
    """Check the index contract of ``obj`` and return ``obj`` unchanged.

    Raises ``TimeSeriesValidationError`` with a specific message for the first violation found.
    See the module docstring for the frequency rules and their limitations.
    """
    if max_gap_bdays < 1:
        raise ValueError(f"max_gap_bdays must be >= 1, got {max_gap_bdays}")
    index = obj.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TimeSeriesValidationError(
            f"index must be a DatetimeIndex, got {type(index).__name__}",
        )
    if len(index) == 0:
        raise TimeSeriesValidationError("index is empty")
    if index.tz is not None:
        raise TimeSeriesValidationError(f"index must be tz-naive, got tz={index.tz}")
    if index.hasnans:
        raise TimeSeriesValidationError("index contains NaT")
    if index.has_duplicates:
        dupes = index[index.duplicated()].unique()[:5]
        raise TimeSeriesValidationError(f"index has duplicate timestamps, e.g. {list(dupes)}")
    if not index.is_monotonic_increasing:
        raise TimeSeriesValidationError("index is not sorted in increasing order")
    if freq is not None:
        _check_frequency(index, freq, max_gap_bdays)
    return obj


def _check_frequency(index: pd.DatetimeIndex, freq: Frequency, max_gap_bdays: int) -> None:
    not_midnight = index[index != index.normalize()]
    if len(not_midnight) > 0:
        raise TimeSeriesValidationError(
            f"{freq} series must have midnight timestamps, e.g. {not_midnight[0]}",
        )
    if freq is Frequency.DAILY:
        weekend = index[index.dayofweek >= 5]
        if len(weekend) > 0:
            raise TimeSeriesValidationError(
                f"daily series contains weekend dates, e.g. {weekend[0].date()}",
            )
        days = index.to_numpy().astype("datetime64[D]")
        gaps = np.busday_count(days[:-1], days[1:])
        if len(gaps) > 0 and int(gaps.max()) > max_gap_bdays:
            i = int(gaps.argmax())
            raise TimeSeriesValidationError(
                f"daily series has a gap of {int(gaps[i])} business days between "
                f"{index[i].date()} and {index[i + 1].date()} (max {max_gap_bdays})",
            )
        return
    if freq is Frequency.WEEKLY:
        # Monday-start (ISO) week number: days since epoch of that week's Monday, // 7.
        days = index.to_numpy().astype("datetime64[D]").astype(np.int64)
        ordinals = (days - index.dayofweek.to_numpy()) // 7
    else:
        ordinals = index.year.to_numpy() * 12 + index.month.to_numpy()
    steps = np.diff(ordinals)
    bad = np.flatnonzero(steps != 1)
    if len(bad) > 0:
        i = int(bad[0])
        raise TimeSeriesValidationError(
            f"{freq} series must have exactly one observation per consecutive period; "
            f"{index[i].date()} -> {index[i + 1].date()} spans {int(steps[i])} periods",
        )
