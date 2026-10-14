"""Contract for the ``synthetic`` source: observed features of a planted-signal dataset.

The payload is built by ``invariant.benchmark.payload.synthetic_feature_payload``; this module
holds only what the data layer needs to read it back, so ``invariant.data`` never imports
``invariant.benchmark``.

Table ``features``: one row per date (``date`` column, ``DATE_DTYPE``, sorted, unique) and one
float64 column per *observed* feature, named as in ``GroundTruth`` (``s_real``, ``s_conf``,
``s_decay``, ``s_noise``, and ``z`` only when the confounder is observed). Oracle-only latents
never appear here. Params carry ``provider``, ``dataset_sha256``, ``seed`` and ``columns``.

Timing: a value dated t is known at the close of t (``GroundTruth.timing_convention``). See
docs/adr/0008-synthetic-feature-source.md.
"""

from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from invariant.core.types import Frequency, validate_time_index
from invariant.data.sources.base import DATE_DTYPE, DataValidationError

__all__ = [
    "FEATURES_TABLE",
    "SYNTHETIC_PROVIDER",
    "SyntheticFeatureReport",
    "features_frame",
    "validate_feature_table",
]

SYNTHETIC_PROVIDER: Final = "synthetic"
FEATURES_TABLE: Final = "features"


class SyntheticFeatureReport(BaseModel):
    """What the ``synthetic`` payload holds (recorded in the snapshot manifest)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: tuple[str, ...]
    n_rows: int
    first_date: date
    last_date: date
    n_missing: int


def validate_feature_table(table: pd.DataFrame) -> pd.DataFrame:
    """Check a ``features`` table: ``date`` first, sorted unique dates, float64 values."""
    if list(table.columns[:1]) != ["date"]:
        raise DataValidationError("synthetic features table must start with a 'date' column")
    value_cols = list(table.columns[1:])
    if not value_cols or len(set(value_cols)) != len(value_cols):
        raise DataValidationError(f"synthetic feature columns must be unique: {value_cols}")
    if str(table["date"].dtype) != DATE_DTYPE:
        raise DataValidationError(f"synthetic 'date' must be {DATE_DTYPE}")
    bad = [c for c in value_cols if table[c].dtype != np.float64]
    if bad:
        raise DataValidationError(f"synthetic feature columns must be float64: {bad}")
    validate_time_index(features_frame(table, validate=False), Frequency.DAILY)
    return table


def features_frame(table: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """The ``features`` table indexed by date (a tz-naive ``DatetimeIndex`` named ``date``)."""
    if validate:
        validate_feature_table(table)
    frame = table.set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame
