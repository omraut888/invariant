"""Real network fetches. Slow, and dependent on Yahoo, the Ken French library and FRED.

Run with ``uv run pytest -m slow tests/integration``. ALFRED tests need FRED_API_KEY in .env.
"""

from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr

from invariant.config.settings import PipelineConfig, Settings, load_config, load_settings
from invariant.data.build import build_snapshot
from invariant.data.snapshot import load_snapshot
from invariant.data.sources.alfred import AlfredReport, fetch_alfred
from invariant.data.sources.base import SourceName
from invariant.data.sources.clients import FredAlfredClient, HttpxClient, YFinanceClient
from invariant.data.sources.etf import EtfReport, fetch_etf_prices
from invariant.data.sources.french import FrenchReport, fetch_french_factors

pytestmark = pytest.mark.slow

CONFIG = load_config()


def _fred_key() -> SecretStr:
    key = load_settings().fred_api_key
    if key is None:
        pytest.skip("FRED_API_KEY not set")
    return key


def test_yahoo_prices_cover_every_session() -> None:
    payload = fetch_etf_prices(
        YFinanceClient(timeout_s=30), ["SPY", "GLD"], date(2025, 1, 2), date(2025, 1, 31)
    )
    report = payload.report
    assert isinstance(report, EtfReport)
    assert report.n_sessions == 20  # 22 weekdays minus the 2025-01-09 closure and MLK Day
    assert report.complete, report


def test_french_library_layout_unchanged() -> None:
    with HttpxClient(CONFIG.data.http) as http:
        payload = fetch_french_factors(http, CONFIG.data.french)
    report = payload.report
    assert isinstance(report, FrenchReport)
    assert all(f.crsp_vintage is not None for f in report.files)
    five, mom = report.files
    assert five.first_date == date(1963, 7, 1)
    assert mom.first_date == date(1926, 11, 3)
    assert five.last_date.year >= 2025
    assert report.dates_only_in["five_factor"] == 0


def test_alfred_pagination_against_real_api() -> None:
    with HttpxClient(CONFIG.data.http) as http:
        client = FredAlfredClient(http, _fred_key(), page_size=1000)
        payload = fetch_alfred(client, ["UNRATE"])
    report = payload.report
    assert isinstance(report, AlfredReport)
    unrate = report.series[0]
    assert unrate.pages >= 3
    assert unrate.n_rows == unrate.api_count
    assert unrate.first_realtime_start == date(1960, 3, 15)
    assert unrate.n_vintages > 700


def test_end_to_end_snapshot(tmp_path: Path) -> None:
    _fred_key()
    cfg: PipelineConfig = CONFIG.model_copy(
        update={
            "universe": CONFIG.universe.model_copy(update={"tickers": ("SPY", "TLT")}),
            "dates": CONFIG.dates.model_copy(
                update={"start": date(2024, 12, 2), "end": date(2024, 12, 31)}
            ),
            "data": CONFIG.data.model_copy(
                update={"alfred": CONFIG.data.alfred.model_copy(update={"series": ("UNRATE",)})}
            ),
        }
    )
    real = load_settings()
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        runs_dir=tmp_path / "runs",
        fred_api_key=real.fred_api_key,
    )
    snap = build_snapshot(cfg, settings)
    again = load_snapshot(snap.snapshot_id, tmp_path / "snapshots")
    assert {s.source for s in again.manifest.sources} == set(SourceName)
    assert again.etf_log_returns().shape == (21, 2)  # Dec 2024: 22 weekdays minus Christmas
    assert not again.french_factors().empty
    assert len(again.alfred_vintages()) == again.alfred_report().series[0].api_count
