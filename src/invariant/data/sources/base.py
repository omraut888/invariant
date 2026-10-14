"""Contracts shared by every data source and the snapshot writer.

A source's fetch function returns a ``SourcePayload``: plain-column tables (no meaningful index)
plus the request params and a typed report of what was fetched and what was missing. The
snapshot layer persists payloads verbatim; it never transforms data.

Dates in the data layer use ``DATE_DTYPE`` (``datetime64[us]``): nanosecond resolution cannot
represent ALFRED's open-ended ``realtime_end`` of 9999-12-31, and one unit everywhere avoids
mixed-resolution surprises after a Parquet round trip.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final

import pandas as pd
from pydantic import BaseModel, JsonValue

__all__ = [
    "DATE_DTYPE",
    "DataValidationError",
    "SourceName",
    "SourcePayload",
    "library_versions",
]

DATE_DTYPE: Final = "datetime64[us]"


class DataValidationError(ValueError):
    """Fetched data violates a source's contract (shape, dtype, ordering, plausibility)."""


class SourceName(StrEnum):
    ETF = "etf"
    FRENCH = "french"
    ALFRED = "alfred"
    SYNTHETIC = "synthetic"  # planted-signal benchmark features only; never fetched


@dataclass(frozen=True)
class SourcePayload:
    """One fetched source, ready to be written into a snapshot.

    ``tables`` maps a table name (a file stem, ``[a-z0-9_]+``) to a frame with a default
    ``RangeIndex``; all information lives in columns so Parquet round-trips are exact.
    ``params`` must be canonical-JSON-able: it is part of the snapshot id.
    """

    source: SourceName
    params: Mapping[str, JsonValue]
    report: BaseModel
    tables: Mapping[str, pd.DataFrame]
    fetched_at: datetime
    library_versions: Mapping[str, str | None] = field(default_factory=dict)


def library_versions(*packages: str) -> dict[str, str | None]:
    """Installed versions of ``packages`` (``None`` if absent), recorded in the manifest."""
    out: dict[str, str | None] = {}
    for pkg in packages:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out
