from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from pydantic import ValidationError
from tests.fixtures.fakes import (
    FRENCH_CONFIG,
    UNRATE_ROWS,
    FakeAlfredClient,
    FakeHttp,
    FakePriceClient,
    fixed_now,
    french_zips,
    price_frame,
)
from tests.fixtures.pit import (
    SESSIONS,
    bit_identical,
    empty_vintages,
    oracle_asof,
    pit_config,
    releases,
    releases_to_vintages,
)

from invariant.config.settings import PipelineConfig, PitConfig, TargetConfig, load_config
from invariant.data.pit import (
    PitInputs,
    PitPanel,
    alfred_asof,
    assemble_panel,
    build_pit_panel,
    forward_returns,
    lag_by_sessions,
)
from invariant.data.snapshot import Snapshot, write_snapshot
from invariant.data.sources.alfred import fetch_alfred
from invariant.data.sources.base import DataValidationError
from invariant.data.sources.etf import fetch_etf_prices
from invariant.data.sources.french import fetch_french_factors

TS = pd.Timestamp
JAN, FEB, MAR = TS("2025-01-01"), TS("2025-02-01"), TS("2025-03-01")


def _asof(rels: list[tuple[pd.Timestamp, dict[pd.Timestamp, float]]]) -> pd.DataFrame:
    return alfred_asof(releases_to_vintages("S", rels), "S", SESSIONS)


def _value(frame: pd.DataFrame, day: str) -> tuple[float, pd.Timestamp]:
    return float(frame["value"].loc[TS(day)]), TS(frame["obs_date"].loc[TS(day)])


# --- alfred_asof ------------------------------------------------------------------------------


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(releases(SESSIONS[0] - pd.Timedelta(days=30), SESSIONS[-1] + pd.Timedelta(days=5), 8))
def test_alfred_asof_matches_brute_force_oracle(
    rels: list[tuple[pd.Timestamp, dict[pd.Timestamp, float]]],
) -> None:
    # The oracle works from the release history, the implementation from ALFRED's merged rows.
    got = alfred_asof(releases_to_vintages("S", rels), "S", SESSIONS)
    assert bit_identical(got, oracle_asof(rels, SESSIONS))


def test_release_usable_from_next_session() -> None:
    # SESSIONS start Monday 2025-03-03.
    friday = _asof([(TS("2025-03-07"), {JAN: 1.0})])
    assert np.isnan(_value(friday, "2025-03-07")[0])
    assert _value(friday, "2025-03-10") == (1.0, JAN)  # Monday
    weekend = _asof([(TS("2025-03-08"), {JAN: 1.0})])  # Saturday
    assert _value(weekend, "2025-03-10") == (1.0, JAN)


def test_revision_replaces_value_from_the_session_after_it() -> None:
    frame = _asof([(TS("2025-03-04"), {JAN: 1.0}), (TS("2025-03-06"), {JAN: 2.0})])
    assert _value(frame, "2025-03-06") == (1.0, JAN)
    assert _value(frame, "2025-03-07") == (2.0, JAN)


def test_newer_observation_wins_and_obs_date_tracks_it() -> None:
    frame = _asof([(TS("2025-03-04"), {JAN: 1.0}), (TS("2025-03-06"), {JAN: 1.0, FEB: 5.0})])
    assert _value(frame, "2025-03-06") == (1.0, JAN)
    assert _value(frame, "2025-03-07") == (5.0, FEB)


def test_withdrawn_observation_falls_back_once_withdrawal_is_usable() -> None:
    frame = _asof([(TS("2025-03-04"), {JAN: 1.0, FEB: 5.0}), (TS("2025-03-06"), {JAN: 1.0})])
    assert _value(frame, "2025-03-06") == (5.0, FEB)
    assert _value(frame, "2025-03-07") == (1.0, JAN)


def test_nan_print_never_resurrects_an_older_vintage() -> None:
    # FEB is published as "." on 03-06: the feature falls back to JAN in the *same* vintage,
    # never to FEB's earlier value 5.0.
    frame = _asof(
        [(TS("2025-03-04"), {JAN: 1.0, FEB: 5.0}), (TS("2025-03-06"), {JAN: 1.5, FEB: np.nan})]
    )
    assert _value(frame, "2025-03-07") == (1.5, JAN)


def test_release_before_panel_is_known_from_first_session() -> None:
    frame = _asof([(TS("2025-01-15"), {JAN: 1.0}), (TS("2025-06-01"), {JAN: 9.0})])
    assert _value(frame, str(SESSIONS[0].date())) == (1.0, JAN)
    assert (frame["value"] == 1.0).all()  # the release after the last session never appears


def test_empty_vintages_give_all_nan() -> None:
    frame = alfred_asof(empty_vintages(), "S", SESSIONS)
    assert frame["value"].isna().all()
    assert frame["obs_date"].isna().all()


def test_alfred_asof_rejects_other_series_and_overlaps() -> None:
    table = releases_to_vintages("S", [(TS("2025-03-04"), {JAN: 1.0})])
    with pytest.raises(DataValidationError, match="other series"):
        alfred_asof(table, "T", SESSIONS)
    overlap = pd.concat([table, table.assign(realtime_start=TS("2025-03-05"))], ignore_index=True)
    with pytest.raises(DataValidationError, match="overlapping"):
        alfred_asof(overlap, "S", SESSIONS)


# --- lag_by_sessions --------------------------------------------------------------------------


def _factor(dates: list[str], values: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(dates).astype("datetime64[us]"), name="date")
    return pd.DataFrame({"f": values}, index=index)


def test_lag_shifts_by_sessions_without_filling() -> None:
    frame = _factor(["2025-03-03", "2025-03-05"], [1.0, 2.0])
    out = lag_by_sessions(frame, SESSIONS, 2)
    assert out.loc[TS("2025-03-05"), "f"] == 1.0
    assert out.loc[TS("2025-03-07"), "f"] == 2.0
    assert out["f"].notna().sum() == 2  # 03-06 is not filled from 03-05


def test_lag_zero_is_identity_on_sessions() -> None:
    frame = _factor(["2025-03-03", "2025-03-04"], [1.0, 2.0])
    out = lag_by_sessions(frame, SESSIONS, 0)
    assert list(out["f"].iloc[:2]) == [1.0, 2.0]


def test_lag_drops_pre_panel_rows_and_maps_off_calendar_forward() -> None:
    # 2025-02-28 precedes the panel; Saturday 03-08 counts from Monday 03-10 and, landing on
    # the same session as the 03-10 row, loses to the later-dated row.
    frame = _factor(["2025-02-28", "2025-03-08", "2025-03-10"], [9.0, 1.0, 2.0])
    out = lag_by_sessions(frame, SESSIONS, 1)
    assert out["f"].notna().sum() == 1
    assert out.loc[TS("2025-03-11"), "f"] == 2.0


def test_lag_validates_arguments() -> None:
    frame = _factor(["2025-03-03"], [1.0])
    with pytest.raises(ValueError, match="lag"):
        lag_by_sessions(frame, SESSIONS, -1)
    assert lag_by_sessions(frame.iloc[:0], SESSIONS, 3)["f"].isna().all()


# --- forward_returns --------------------------------------------------------------------------


@pytest.mark.parametrize(("execution_lag", "horizon"), [(0, 1), (1, 1), (1, 5), (2, 3)])
def test_forward_return_is_log_price_ratio_from_entry(execution_lag: int, horizon: int) -> None:
    prices = pd.DataFrame({"X": np.linspace(100.0, 140.0, len(SESSIONS))}, index=SESSIONS)
    returns = pd.DataFrame(np.log(prices.to_numpy()), index=SESSIONS, columns=["X"]).diff()
    y = forward_returns(returns, TargetConfig(execution_lag=execution_lag, horizon=horizon))
    e, h = execution_lag, horizon
    p = prices["X"].to_numpy()
    n_valid = len(p) - e - h
    want = np.log(p[e + h : e + h + n_valid] / p[e : e + n_valid])
    np.testing.assert_allclose(y["X"].to_numpy()[:n_valid], want, rtol=1e-12)
    assert y["X"].iloc[n_valid:].isna().all()  # no partial windows at the end


def test_forward_return_propagates_nan_in_window() -> None:
    returns = pd.DataFrame({"X": [0.01, 0.02, np.nan, 0.04, 0.05, 0.06]}, index=SESSIONS[:6])
    y = forward_returns(returns, TargetConfig(execution_lag=0, horizon=2))["X"].to_numpy()
    # y_0 = r_1 + r_2 and y_1 = r_2 + r_3 touch the NaN; y_4, y_5 run past the end.
    assert np.isnan(y[[0, 1, 4, 5]]).all()
    assert y[2] == 0.04 + 0.05
    assert y[3] == 0.05 + 0.06


# --- assemble / build -------------------------------------------------------------------------


def test_assemble_columns_and_empty_series() -> None:
    adj = pd.DataFrame({"AAA": np.linspace(10.0, 20.0, len(SESSIONS))}, index=SESSIONS)
    french = _factor(["2025-03-03"], [0.01]).rename(columns={"f": "mkt_rf"})
    panel = assemble_panel(
        PitInputs(adj_close=adj, french=french, alfred={"ZZZ": empty_vintages()}),
        pit_config(french_lag=1),
    )
    assert list(panel.features.columns) == ["ret_AAA", "ff_mkt_rf", "macro_ZZZ"]
    assert list(panel.obs_dates.columns) == ["macro_ZZZ"]
    assert list(panel.targets.columns) == ["fwd_AAA"]
    assert panel.features["macro_ZZZ"].isna().all()
    assert panel.features.loc[TS("2025-03-04"), "ff_mkt_rf"] == 0.01


def test_panel_rejects_misaligned_frames() -> None:
    frame = pd.DataFrame({"x": 1.0}, index=SESSIONS)
    with pytest.raises(ValueError, match="targets index"):
        PitPanel(features=frame, obs_dates=frame, targets=frame.iloc[1:], config=pit_config())


def _snapshot_config() -> PipelineConfig:
    cfg = load_config()
    return cfg.model_copy(
        update={
            "universe": cfg.universe.model_copy(update={"tickers": ("SPY", "GLD")}),
            "dates": cfg.dates.model_copy(
                update={"start": date(2025, 1, 2), "end": date(2025, 1, 7)}
            ),
            "data": cfg.data.model_copy(
                update={"alfred": cfg.data.alfred.model_copy(update={"series": ("UNRATE",)})}
            ),
            "pit": pit_config(french_lag=1),
        }
    )


def _write(tmp_path: Path, *, alfred: bool = True) -> Snapshot:
    sessions = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    raw = price_frame(sessions, {"SPY": [1.0, 2.0, 3.0, 4.0], "GLD": [5.0, 6.0, 7.0, 8.0]})
    payloads = [
        fetch_etf_prices(
            FakePriceClient(raw), ["SPY", "GLD"], date(2025, 1, 2), date(2025, 1, 7), now=fixed_now
        ),
        fetch_french_factors(FakeHttp(french_zips()), FRENCH_CONFIG, now=fixed_now),
    ]
    if alfred:
        payloads.append(
            fetch_alfred(FakeAlfredClient({"UNRATE": UNRATE_ROWS}), ["UNRATE"], now=fixed_now)
        )
    return write_snapshot(payloads, tmp_path)


def test_build_from_snapshot(tmp_path: Path) -> None:
    snap = _write(tmp_path)
    panel = build_pit_panel(snap, _snapshot_config())
    assert len(panel.features) == 4
    assert panel.features["ret_SPY"].iloc[1] == pytest.approx(np.log(2.0))
    # Execution lag 1: y at 01-02 is r(01-06) = ln(3/2).
    assert panel.targets["fwd_SPY"].iloc[0] == pytest.approx(np.log(1.5))
    # French lag 1: the factor dated 01-02 is used on 01-03; nothing is usable on 01-02.
    ff = panel.features["ff_mkt_rf"]
    assert np.isnan(ff.iloc[0])
    assert ff.iloc[1] == snap.french_factors().loc[TS("2025-01-02"), "mkt_rf"]
    # Every UNRATE vintage is dated after the panel, so nothing is known yet.
    assert panel.features["macro_UNRATE"].isna().all()


def test_build_requires_configured_series_and_tickers(tmp_path: Path) -> None:
    snap = _write(tmp_path)
    cfg = _snapshot_config()
    bad_series = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(
                update={"alfred": cfg.data.alfred.model_copy(update={"series": ("PAYEMS",)})}
            )
        }
    )
    with pytest.raises(DataValidationError, match="PAYEMS"):
        build_pit_panel(snap, bad_series)
    bad_ticker = cfg.model_copy(
        update={"universe": cfg.universe.model_copy(update={"tickers": ("XLE",)})}
    )
    with pytest.raises(DataValidationError, match="XLE"):
        build_pit_panel(snap, bad_ticker)


def test_build_without_alfred_source(tmp_path: Path) -> None:
    panel = build_pit_panel(_write(tmp_path, alfred=False), _snapshot_config())
    assert not any(c.startswith("macro_") for c in panel.features.columns)


# --- config -----------------------------------------------------------------------------------


def test_base_config_pit_defaults() -> None:
    pit = load_config().pit
    assert pit.targets == TargetConfig(execution_lag=1, horizon=1)
    assert pit.publication_lag_sessions.french == 42


@pytest.mark.parametrize(
    "targets",
    [{"execution_lag": -1, "horizon": 1}, {"execution_lag": 0, "horizon": 0}, {"horizon": 1}],
)
def test_target_config_validation(targets: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        PitConfig.model_validate({"targets": targets, "publication_lag_sessions": {"french": 1}})
