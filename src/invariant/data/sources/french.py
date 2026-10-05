"""Fama-French 5 factors (2x3) and momentum, daily, from the Ken French Data Library.

Method
------
Both zip files are downloaded through ``HttpClient`` (URLs in ``data.french`` of the config) and
must contain exactly one CSV. The CSV layout is: free-text description lines, a header row
starting with a comma (``,Mkt-RF,SMB,HML,RMW,CMA,RF`` / ``,Mom``), one row per day
``YYYYMMDD, v1, ...`` with values in percent, then a blank line and a copyright footer. The
parser requires the exact expected header, stops at the first non-data line, and fails if any
dated row appears after that point (a layout change, e.g. an extra section). Values are
converted from percent to decimal returns. The library's missing-value sentinels, -99.99 and
-999, become NaN and are counted per column in the report. The two files are outer-joined on
date; dates present in only one file (momentum starts 1926-11-03, the 5 factors 1963-07-01)
keep NaN for the other file's columns and are counted in the report. Nothing is filled.

Each file's SHA-256 and its CRSP build ("created by using the 202608 CRSP database") are recorded,
which is the only available identifier of the library revision that was fetched.

Not point-in-time
-----------------
The library is not a point-in-time source:

- Every CRSP update rebuilds the full history. Past factor returns change through data
  corrections, delisting-return fixes, and reconstituted breakpoints and portfolios.
- Each release covers data through the end of the prior month and is published weeks later,
  so a factor return for day t was not available on day t.
- Only the current revision can be downloaded, so earlier revisions cannot be reconstructed.

Safe uses: contemporaneous risk adjustment and performance attribution. Using factor returns as
*predictors* needs at least a one-month-plus publication lag and must acknowledge revision
noise. Daily rows before 1952 include Saturday sessions, so the weekday-only daily index check
is not applied; only sortedness and uniqueness are validated.

Reference: Fama, E. F. and French, K. R. (2015), "A five-factor asset pricing model", Journal of
Financial Economics 116(1), 1-22; Carhart, M. M. (1997), "On Persistence in Mutual Fund
Performance", Journal of Finance 52(1), 57-82. Data library:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
See docs/adr/0004-data-sources-and-pit-limitations.md.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from structlog.typing import FilteringBoundLogger

from invariant.config.settings import FrenchConfig
from invariant.core.logging import get_logger
from invariant.core.types import TimeSeriesValidationError, validate_time_index
from invariant.data.sources.base import (
    DATE_DTYPE,
    DataValidationError,
    SourceName,
    SourcePayload,
    library_versions,
)
from invariant.data.sources.clients import HttpClient

__all__ = [
    "FACTOR_COLUMNS",
    "FIVE_FACTOR_HEADER",
    "MOMENTUM_HEADER",
    "FrenchFileReport",
    "FrenchReport",
    "ParsedFrenchFile",
    "extract_single_csv",
    "fetch_french_factors",
    "join_factor_files",
    "parse_french_csv",
]

FIVE_FACTOR_HEADER = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")
MOMENTUM_HEADER = ("Mom",)
_RENAME = {
    "Mkt-RF": "mkt_rf",
    "SMB": "smb",
    "HML": "hml",
    "RMW": "rmw",
    "CMA": "cma",
    "RF": "rf",
    "Mom": "mom",
}
FACTOR_COLUMNS = ("mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom")
_SENTINELS = (-99.99, -999.0)
_DATE_RE = re.compile(r"^\d{8}$")
_CRSP_RE = re.compile(r"(\d{6})\s+CRSP database")


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FrenchFileReport(_Frozen):
    name: str
    url: str
    member: str
    zip_sha256: str
    crsp_vintage: str | None
    n_rows: int
    first_date: date
    last_date: date
    missing_values: dict[str, int]


class FrenchReport(_Frozen):
    files: tuple[FrenchFileReport, ...]
    n_rows: int
    dates_only_in: dict[str, int]


@dataclass(frozen=True)
class ParsedFrenchFile:
    """A parsed CSV: ``frame`` has a ``date`` column plus decimal-return factor columns."""

    frame: pd.DataFrame
    crsp_vintage: str | None
    missing_values: dict[str, int]


def extract_single_csv(data: bytes) -> tuple[str, str]:
    """Return ``(member name, decoded text)`` of the only CSV in a zip archive."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DataValidationError(f"not a zip archive: {exc}") from None
    with archive:
        members = [n for n in archive.namelist() if not n.endswith("/")]
        csvs = [n for n in members if n.lower().endswith(".csv")]
        if len(members) != 1 or len(csvs) != 1:
            raise DataValidationError(f"expected exactly one CSV in the zip, found {members}")
        raw = archive.read(csvs[0])
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    return csvs[0], text


def parse_french_csv(text: str, header: Sequence[str]) -> ParsedFrenchFile:
    """Parse one daily Ken French CSV whose header must equal ``header`` (see module docstring).

    The parse is row-wise: each output row depends only on its own input line.
    """
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith(",")), None)
    if header_idx is None:
        raise DataValidationError("no header row (a line starting with ',') found")
    found = tuple(c.strip() for c in lines[header_idx].split(",")[1:])
    if found != tuple(header):
        raise DataValidationError(f"unexpected header {found}, expected {tuple(header)}")

    dates: list[str] = []
    values: list[list[float]] = []
    end = len(lines)
    for i in range(header_idx + 1, len(lines)):
        fields = [f.strip() for f in lines[i].split(",")]
        if not _DATE_RE.fullmatch(fields[0]):
            end = i
            break
        if len(fields) != len(header) + 1:
            raise DataValidationError(f"line {i + 1}: expected {len(header) + 1} fields")
        try:
            values.append([float(f) for f in fields[1:]])
        except ValueError:
            raise DataValidationError(f"line {i + 1}: non-numeric value in {fields}") from None
        dates.append(fields[0])
    trailing = [ln for ln in lines[end:] if _DATE_RE.fullmatch(ln.split(",")[0].strip())]
    if trailing:
        raise DataValidationError(
            f"{len(trailing)} dated rows after the first data block; layout changed"
        )
    if not dates:
        raise DataValidationError("no data rows after the header")

    try:
        index = pd.to_datetime(pd.Series(dates), format="%Y%m%d").astype(DATE_DTYPE)
    except ValueError as exc:
        raise DataValidationError(f"invalid date: {exc}") from None
    arr = np.asarray(values, dtype="float64")
    sentinel = np.isin(arr, _SENTINELS)
    arr[sentinel] = np.nan
    names = [_RENAME[h] for h in header]
    frame = pd.DataFrame(arr / 100.0, columns=names)
    frame.insert(0, "date", index.to_numpy())
    try:
        validate_time_index(frame.set_index("date"))
    except TimeSeriesValidationError as exc:
        raise DataValidationError(f"factor dates: {exc}") from None
    vintage = _CRSP_RE.search("\n".join(lines[:header_idx]))
    return ParsedFrenchFile(
        frame=frame,
        crsp_vintage=vintage.group(1) if vintage else None,
        missing_values={n: int(sentinel[:, j].sum()) for j, n in enumerate(names)},
    )


def join_factor_files(five: pd.DataFrame, mom: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Outer-join the two parsed frames on ``date``; return the table and one-sided date counts."""
    joined = five.merge(mom, on="date", how="outer", sort=True, indicator=True)
    counts = {
        "five_factor": int((joined["_merge"] == "left_only").sum()),
        "momentum": int((joined["_merge"] == "right_only").sum()),
    }
    table = joined.drop(columns="_merge")[["date", *FACTOR_COLUMNS]].reset_index(drop=True)
    return table.astype({"date": DATE_DTYPE}), counts


def _utcnow() -> datetime:
    return datetime.now(UTC)


def fetch_french_factors(
    http: HttpClient,
    config: FrenchConfig,
    *,
    now: Callable[[], datetime] = _utcnow,
) -> SourcePayload:
    """Download and parse both files; return the ``french`` snapshot payload (table ``factors``)."""
    fetched_at = now()
    parsed: dict[str, ParsedFrenchFile] = {}
    reports: list[FrenchFileReport] = []
    for name, url, header in (
        ("five_factor", config.five_factor_url, FIVE_FACTOR_HEADER),
        ("momentum", config.momentum_url, MOMENTUM_HEADER),
    ):
        data = http.get_bytes(url)
        member, text = extract_single_csv(data)
        try:
            p = parse_french_csv(text, header)
        except DataValidationError as exc:
            raise DataValidationError(f"{name} ({member}): {exc}") from exc
        parsed[name] = p
        if p.crsp_vintage is None:
            _log().warning("french_crsp_vintage_not_found", file=name)
        if any(p.missing_values.values()):
            _log().warning("french_missing_values", file=name, counts=p.missing_values)
        dates = pd.DatetimeIndex(p.frame["date"])
        reports.append(
            FrenchFileReport(
                name=name,
                url=url,
                member=member,
                zip_sha256=hashlib.sha256(data).hexdigest(),
                crsp_vintage=p.crsp_vintage,
                n_rows=len(p.frame),
                first_date=dates[0].date(),
                last_date=dates[-1].date(),
                missing_values=p.missing_values,
            )
        )
    table, only_in = join_factor_files(parsed["five_factor"].frame, parsed["momentum"].frame)
    report = FrenchReport(files=tuple(reports), n_rows=len(table), dates_only_in=only_in)
    _log().info("french_fetched", rows=len(table), dates_only_in=only_in)
    return SourcePayload(
        source=SourceName.FRENCH,
        params={
            "five_factor_url": config.five_factor_url,
            "momentum_url": config.momentum_url,
        },
        report=report,
        tables={"factors": table},
        fetched_at=fetched_at,
        library_versions=library_versions("httpx"),
    )
