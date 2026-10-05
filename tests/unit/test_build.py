from pathlib import Path

import pytest
from tests.fixtures.fakes import (
    FRENCH_CONFIG,
    UNRATE_ROWS,
    FakeAlfredClient,
    FakeHttp,
    FakePriceClient,
    french_zips,
    price_frame,
)

from invariant.config.settings import PipelineConfig, Settings, load_config
from invariant.data.build import build_snapshot
from invariant.data.snapshot import default_snapshots_dir
from invariant.data.sources.base import SourceName


def _config() -> PipelineConfig:
    cfg = load_config()
    return cfg.model_copy(
        update={
            "universe": cfg.universe.model_copy(update={"tickers": ("SPY",)}),
            "dates": cfg.dates.model_copy(
                update={"start": cfg.dates.start, "end": cfg.dates.start.replace(day=7)}
            ),
            "data": cfg.data.model_copy(
                update={
                    "french": FRENCH_CONFIG,
                    "alfred": cfg.data.alfred.model_copy(update={"series": ("UNRATE",)}),
                }
            ),
        }
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, runs_dir=tmp_path / "runs")


def _prices() -> FakePriceClient:
    sessions = ["2005-01-03", "2005-01-04", "2005-01-05", "2005-01-06", "2005-01-07"]
    return FakePriceClient(price_frame(sessions, {"SPY": [1.0, 2.0, 3.0, 4.0, 5.0]}))


def test_build_snapshot_with_injected_clients(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    snap = build_snapshot(
        _config(),
        settings,
        price_client=_prices(),
        http=FakeHttp(french_zips()),
        alfred_client=FakeAlfredClient({"UNRATE": UNRATE_ROWS}),
    )
    assert snap.path.parent == default_snapshots_dir(settings) == tmp_path / "snapshots"
    assert {s.source for s in snap.manifest.sources} == set(SourceName)
    assert snap.etf_report().complete
    assert snap.alfred_report().series[0].series_id == "UNRATE"


def test_build_subset_of_sources(tmp_path: Path) -> None:
    snap = build_snapshot(
        _config(),
        _settings(tmp_path),
        sources=(SourceName.FRENCH,),
        http=FakeHttp(french_zips()),
    )
    assert [s.source for s in snap.manifest.sources] == [SourceName.FRENCH]


def test_alfred_without_key_or_client_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FRED_API_KEY"):
        build_snapshot(
            _config(),
            _settings(tmp_path),
            sources=(SourceName.ALFRED,),
            http=FakeHttp({}),
        )
