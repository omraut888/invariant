import numpy as np
import pandas as pd
import pytest
from tests.fixtures.fakes import (
    FIVE_FACTOR_CSV,
    FRENCH_CONFIG,
    MOMENTUM_CSV,
    FakeHttp,
    fixed_now,
    french_zips,
    zip_bytes,
)

from invariant.data.sources.base import DataValidationError, SourceName
from invariant.data.sources.french import (
    FACTOR_COLUMNS,
    FIVE_FACTOR_HEADER,
    MOMENTUM_HEADER,
    FrenchReport,
    extract_single_csv,
    fetch_french_factors,
    join_factor_files,
    parse_french_csv,
)

HEADER_ONLY = "Some description\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n"


def test_parse_five_factor_fixture() -> None:
    parsed = parse_french_csv(FIVE_FACTOR_CSV.read_text(), FIVE_FACTOR_HEADER)
    frame = parsed.frame
    assert list(frame.columns) == ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert len(frame) == 8
    assert frame["date"].dtype == "datetime64[us]"
    assert frame["date"].iloc[0] == pd.Timestamp("2024-12-27")
    # Percent -> decimal: 20241227 Mkt-RF is -1.17%.
    assert frame["mkt_rf"].iloc[0] == pytest.approx(-0.0117)
    assert frame["rf"].iloc[-1] == pytest.approx(0.0002)
    assert parsed.crsp_vintage == "202608"
    assert parsed.missing_values == dict.fromkeys(frame.columns[1:], 0)


def test_parse_momentum_fixture_with_long_description() -> None:
    parsed = parse_french_csv(MOMENTUM_CSV.read_text(), MOMENTUM_HEADER)
    assert list(parsed.frame.columns) == ["date", "mom"]
    assert len(parsed.frame) == 9
    assert parsed.frame["mom"].iloc[1] == pytest.approx(-0.0084)
    assert parsed.crsp_vintage == "202608"


def test_crlf_line_endings_parse_identically() -> None:
    text = FIVE_FACTOR_CSV.read_text()
    a = parse_french_csv(text, FIVE_FACTOR_HEADER).frame
    b = parse_french_csv(text.replace("\n", "\r\n"), FIVE_FACTOR_HEADER).frame
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("sentinel", ["-99.99", "-999", "-999.00"])
def test_sentinels_become_nan_and_are_counted(sentinel: str) -> None:
    text = HEADER_ONLY + f"20250102, {sentinel}, 0.1, 0.2, 0.3, 0.4, 0.01\n"
    parsed = parse_french_csv(text, FIVE_FACTOR_HEADER)
    assert np.isnan(parsed.frame["mkt_rf"].iloc[0])
    assert parsed.frame["smb"].iloc[0] == pytest.approx(0.001)
    assert parsed.missing_values["mkt_rf"] == 1
    assert parsed.missing_values["smb"] == 0


def test_missing_crsp_line_gives_none() -> None:
    parsed = parse_french_csv(HEADER_ONLY + "20250102,1,1,1,1,1,1\n", FIVE_FACTOR_HEADER)
    assert parsed.crsp_vintage is None


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("no header here\n20250102,1\n", "no header"),
        ("x\n,Mkt-RF,SMB\n20250102,1,2\n", "unexpected header"),
        (HEADER_ONLY, "no data rows"),
        (HEADER_ONLY + "20250102,1,1,1,1,1\n", "expected 7 fields"),
        (HEADER_ONLY + "20250102,1,x,1,1,1,1\n", "non-numeric"),
        (HEADER_ONLY + "20251302,1,1,1,1,1,1\n", "invalid date"),
        (HEADER_ONLY + "20250103,1,1,1,1,1,1\n20250102,1,1,1,1,1,1\n", "not sorted"),
        (HEADER_ONLY + "20250102,1,1,1,1,1,1\n20250102,1,1,1,1,1,1\n", "duplicate"),
        (
            HEADER_ONLY
            + "20250102,1,1,1,1,1,1\n\n Annual\n,Mkt-RF\n2024,1\n20250103,1,1,1,1,1,1\n",
            "layout changed",
        ),
    ],
)
def test_malformed_csv_rejected(text: str, match: str) -> None:
    with pytest.raises(DataValidationError, match=match):
        parse_french_csv(text, FIVE_FACTOR_HEADER)


def test_saturday_sessions_before_1952_are_accepted() -> None:
    text = ",Mom\n19261105, 1.17\n19261106, -0.03\n19261108, -0.01\n"  # 1926-11-06 is a Saturday
    assert len(parse_french_csv(text, MOMENTUM_HEADER).frame) == 3


def test_extract_single_csv() -> None:
    name, text = extract_single_csv(zip_bytes({"a.CSV": "hello"}))
    assert (name, text) == ("a.CSV", "hello")


def test_extract_rejects_multiple_members_and_non_zip() -> None:
    with pytest.raises(DataValidationError, match="exactly one CSV"):
        extract_single_csv(zip_bytes({"a.csv": "1", "b.csv": "2"}))
    with pytest.raises(DataValidationError, match="exactly one CSV"):
        extract_single_csv(zip_bytes({"a.txt": "1"}))
    with pytest.raises(DataValidationError, match="not a zip"):
        extract_single_csv(b"<html>moved</html>")


def test_join_keeps_one_sided_dates_as_nan() -> None:
    five = parse_french_csv(FIVE_FACTOR_CSV.read_text(), FIVE_FACTOR_HEADER).frame
    mom = parse_french_csv(MOMENTUM_CSV.read_text(), MOMENTUM_HEADER).frame
    table, only_in = join_factor_files(five, mom)
    assert only_in == {"five_factor": 0, "momentum": 1}  # 2024-12-26 is in Mom only
    assert list(table.columns) == ["date", *FACTOR_COLUMNS]
    assert len(table) == 9
    first = table.iloc[0]
    assert first["date"] == pd.Timestamp("2024-12-26")
    assert np.isnan(first["mkt_rf"])
    assert first["mom"] == pytest.approx(0.0)
    assert table["date"].is_monotonic_increasing


def test_fetch_french_factors_payload() -> None:
    http = FakeHttp(french_zips())
    payload = fetch_french_factors(http, FRENCH_CONFIG, now=fixed_now)
    assert payload.source is SourceName.FRENCH
    assert [url for url, _ in http.calls] == [
        FRENCH_CONFIG.five_factor_url,
        FRENCH_CONFIG.momentum_url,
    ]
    assert set(payload.tables) == {"factors"}
    assert len(payload.tables["factors"]) == 9
    report = payload.report
    assert isinstance(report, FrenchReport)
    assert [f.name for f in report.files] == ["five_factor", "momentum"]
    assert all(f.crsp_vintage == "202608" and len(f.zip_sha256) == 64 for f in report.files)
    assert report.dates_only_in == {"five_factor": 0, "momentum": 1}
    assert payload.fetched_at == fixed_now()
    assert payload.params == {
        "five_factor_url": FRENCH_CONFIG.five_factor_url,
        "momentum_url": FRENCH_CONFIG.momentum_url,
    }


def test_fetch_names_the_failing_file() -> None:
    bodies = french_zips()
    bodies[FRENCH_CONFIG.momentum_url] = zip_bytes({"m.csv": "garbage\n"})
    with pytest.raises(DataValidationError, match=r"momentum \(m.csv\)"):
        fetch_french_factors(FakeHttp(bodies), FRENCH_CONFIG)
