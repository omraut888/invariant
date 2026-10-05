import numpy as np
import pandas as pd
import pytest
from tests.fixtures.fakes import (
    UNRATE_ROWS,
    FakeAlfredClient,
    alfred_raw,
    alfred_row,
    fixed_now,
)

from invariant.data.sources.alfred import (
    VINTAGE_COLUMNS,
    AlfredReport,
    fetch_alfred,
    parse_alfred_rows,
)
from invariant.data.sources.base import DataValidationError, SourceName
from invariant.data.sources.clients import AlfredRawSeries


def test_parse_keeps_every_vintage_sorted() -> None:
    frame = parse_alfred_rows(alfred_raw("UNRATE", UNRATE_ROWS))
    assert tuple(frame.columns) == VINTAGE_COLUMNS
    assert len(frame) == 4
    assert list(frame["observation_date"].dt.strftime("%Y-%m-%d")) == [
        "2025-07-01",
        "2025-08-01",
        "2025-08-01",
        "2025-09-01",
    ]
    aug = frame[frame["observation_date"] == pd.Timestamp("2025-08-01")]
    assert list(aug["value"]) == [4.3, 4.2]  # first release, then the revision
    assert list(aug["realtime_start"]) == [pd.Timestamp("2025-09-05"), pd.Timestamp("2025-10-03")]


def test_parse_dtypes_and_open_ended_vintage() -> None:
    frame = parse_alfred_rows(alfred_raw("UNRATE", UNRATE_ROWS))
    for col in ("observation_date", "realtime_start", "realtime_end"):
        assert frame[col].dtype == "datetime64[us]"
    assert frame["realtime_end"].max() == pd.Timestamp("9999-12-31")
    assert frame["series_id"].dtype == "str"
    assert frame["value"].dtype == "float64"


def test_missing_marker_becomes_nan_row() -> None:
    frame = parse_alfred_rows(alfred_raw("UNRATE", UNRATE_ROWS))
    sep = frame[frame["observation_date"] == pd.Timestamp("2025-09-01")]
    assert len(sep) == 1
    assert np.isnan(sep["value"].iloc[0])


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([], "no observations"),
        ([alfred_row("2025-01-01", "2025-02-01", "9999-12-31", "abc")], "non-numeric"),
        ([alfred_row("2025-13-01", "2025-02-01", "9999-12-31", "1")], "invalid date"),
        ([alfred_row("2025-01-01", "2025-03-01", "2025-02-01", "1")], "after realtime_end"),
        (
            [
                alfred_row("2025-01-01", "2025-02-01", "9999-12-31", "1"),
                alfred_row("2025-01-01", "2025-02-01", "9999-12-31", "2"),
            ],
            "duplicate",
        ),
        (
            [
                alfred_row("2025-01-01", "2025-02-01", "2025-03-10", "1"),
                alfred_row("2025-01-01", "2025-03-05", "9999-12-31", "2"),
            ],
            "overlapping",
        ),
    ],
)
def test_invalid_rows_rejected(rows: list[dict[str, str]], match: str) -> None:
    with pytest.raises(DataValidationError, match=match):
        parse_alfred_rows(alfred_raw("X", rows))


def test_gaps_between_vintages_are_allowed() -> None:
    rows = [
        alfred_row("2025-01-01", "2025-02-01", "2025-02-27", "1"),
        alfred_row("2025-01-01", "2025-04-01", "9999-12-31", "2"),  # withdrawn in March
    ]
    assert len(parse_alfred_rows(alfred_raw("X", rows))) == 2


def test_fetch_alfred_payload_and_report() -> None:
    other = [alfred_row("2025-08-01", "2025-09-16", "9999-12-31", "103.9")]
    client = FakeAlfredClient({"UNRATE": UNRATE_ROWS, "INDPRO": other})
    payload = fetch_alfred(client, ["UNRATE", "INDPRO"], now=fixed_now)
    assert payload.source is SourceName.ALFRED
    table = payload.tables["vintages"]
    assert list(table["series_id"].unique()) == ["UNRATE", "INDPRO"]
    assert len(table) == 5
    assert payload.params["realtime_start"] == "1776-07-04"
    assert payload.params["realtime_end"] == "9999-12-31"
    report = payload.report
    assert isinstance(report, AlfredReport)
    unrate = report.series[0]
    assert unrate.series_id == "UNRATE"
    assert (unrate.n_rows, unrate.n_observations, unrate.n_vintages) == (4, 3, 3)
    assert unrate.n_missing_values == 1
    assert unrate.api_count == 4
    assert str(unrate.first_realtime_start) == "2025-08-01"


def test_fetch_alfred_rejects_wrong_series_and_empty_request() -> None:
    class Wrong(FakeAlfredClient):
        def all_releases(self, series_id: str) -> AlfredRawSeries:
            return alfred_raw("OTHER", UNRATE_ROWS)

    with pytest.raises(DataValidationError, match="returned OTHER"):
        fetch_alfred(Wrong({}), ["UNRATE"])
    with pytest.raises(ValueError, match="empty"):
        fetch_alfred(FakeAlfredClient({}), [])
