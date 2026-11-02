"""Signal definitions and the real-data signal panel (docs/adr/0018, 0019)."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.fakes import (
    FakeAlfredClient,
    FakeMarketClient,
    FakePriceClient,
    alfred_row,
    market_raw,
    price_frame,
)

from invariant.config.settings import (
    FredMarketConfig,
    PipelineConfig,
    Settings,
    SignalsConfig,
    load_config,
)
from invariant.data.build import build_snapshot
from invariant.data.signals import (
    REGISTRY,
    SIGNAL_PREFIX,
    SignalInputs,
    _window_sum,
    build_signal_panel,
    compute_signals,
)
from invariant.data.sources.base import SourceName
from invariant.data.sources.etf import session_calendar
from invariant.stages.memo.literature import load_corpus

FROZEN_IDS = (
    "tsmom_spy",
    "rv_spy",
    "vrp",
    "term_spread",
    "default_spread",
    "rel_bill",
    "ip_growth",
    "payrolls_growth",
    "unrate_change",
    "inflation",
)
MARKET = ("VIXCLS", "T10Y3M", "DBAA", "DAAA", "DTB3")
MACRO = ("INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL")


def test_the_registry_is_the_frozen_list() -> None:
    assert tuple(REGISTRY) == FROZEN_IDS
    assert "vix" not in REGISTRY  # dropped as redundant with vrp and rv_spy (ADR 0018)
    corpus = load_corpus()
    for d in REGISTRY.values():
        assert d.definition and d.cites, d.id
        assert set(d.cites) <= set(corpus), d.id


def test_window_sums_are_exact_and_need_a_full_window() -> None:
    x = pd.Series(np.arange(10, dtype="float64"))
    w = _window_sum(x, 1, 3)  # x[t-3] + x[t-2] + x[t-1]
    assert w.iloc[:3].isna().all()
    assert list(w.iloc[3:]) == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    x.iloc[5] = np.nan
    assert _window_sum(x, 1, 3).iloc[6:9].isna().all()


def _inputs(r: float = 0.001, n: int = 300) -> SignalInputs:
    sessions = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n), name="date")
    obs = sessions.astype("datetime64[us]")
    bill = pd.DataFrame({"value": np.arange(n, dtype="float64"), "obs_date": obs}, index=sessions)
    flat = {s: pd.DataFrame({"value": 20.0, "obs_date": obs}, index=sessions) for s in MARKET}
    flat["DBAA"] = pd.DataFrame({"value": 6.5, "obs_date": obs}, index=sessions)
    flat["DAAA"] = pd.DataFrame({"value": 5.25, "obs_date": obs}, index=sessions)
    flat["DTB3"] = bill
    return SignalInputs(
        sessions=sessions,
        returns=pd.DataFrame({"SPY": r}, index=sessions),
        market=flat,
        alfred={},
    )


def test_price_and_market_signals_on_known_inputs() -> None:
    inputs = _inputs(r=0.001)
    ids = ("tsmom_spy", "rv_spy", "vrp", "term_spread", "default_spread", "rel_bill")
    sig, obs = compute_signals(inputs, ids)
    assert list(sig.columns) == [SIGNAL_PREFIX + i for i in ids]
    assert math.isnan(sig["sig_tsmom_spy"].iloc[250])
    assert sig["sig_tsmom_spy"].iloc[251] == pytest.approx(231 * 0.001)
    assert sig["sig_rv_spy"].iloc[20] == pytest.approx(math.log(0.001 * math.sqrt(252)))
    assert sig["sig_vrp"].iloc[20] == pytest.approx(0.2**2 / 12 - 21 * 0.001**2)
    assert sig["sig_term_spread"].iloc[0] == 20.0
    assert sig["sig_default_spread"].iloc[0] == 1.25
    # A linear bill rate b_t = t: b_t minus the mean of b_{t-252..t-1} is 126.5.
    assert math.isnan(sig["sig_rel_bill"].iloc[251])
    assert sig["sig_rel_bill"].iloc[252] == pytest.approx(126.5)
    assert set(obs.columns) == {"sig_vrp", "sig_term_spread", "sig_default_spread", "sig_rel_bill"}


def test_unknown_signal_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown signal ids"):
        compute_signals(_inputs(), ("vix",))


# --- the panel, end to end through a snapshot -------------------------------------------------


WARMUP = date(2018, 11, 1)  # rel_bill needs 252 prior sessions plus the posting lag
START, END = date(2020, 1, 2), date(2020, 6, 30)


def _monthly_vintages(sid: str, sessions: pd.DatetimeIndex) -> list[dict[str, str]]:
    """Each month m released on the 5th of m+1, revising m-1 by +0.1 (values > 0)."""
    rows = []
    months = pd.date_range("2018-10-01", "2020-06-01", freq="MS")
    for i, m in enumerate(months):
        rel = (m + pd.DateOffset(months=1, days=4)).strftime("%Y-%m-%d")
        nxt = (m + pd.DateOffset(months=2, days=3)).strftime("%Y-%m-%d")
        v = 100.0 + i
        rows.append(alfred_row(m.strftime("%Y-%m-%d"), rel, nxt, repr(v)))
        rows.append(
            alfred_row(
                m.strftime("%Y-%m-%d"),
                (m + pd.DateOffset(months=2, days=4)).strftime("%Y-%m-%d"),
                "9999-12-31",
                repr(v + 0.1),
            )
        )
    return rows


def _config() -> PipelineConfig:
    cfg = load_config()
    return cfg.model_copy(
        update={
            "universe": cfg.universe.model_copy(update={"tickers": ("SPY", "TLT")}),
            "dates": cfg.dates.model_copy(update={"start": START, "end": END}),
            "data": cfg.data.model_copy(
                update={
                    "alfred": cfg.data.alfred.model_copy(update={"series": MACRO}),
                    "fred_market": FredMarketConfig(series=MARKET, min_interval_s=0),
                }
            ),
            "signals": SignalsConfig(ids=FROZEN_IDS, warmup_start=WARMUP),
        }
    )


def _snapshot(tmp_path: Path):  # type: ignore[no-untyped-def]
    sessions = session_calendar("XNYS", WARMUP, END)
    days = [d.strftime("%Y-%m-%d") for d in sessions]
    rng = np.random.default_rng(3)
    prices = {
        t: list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))) for t in ("SPY", "TLT")
    }
    fv = "2019-06-03"
    market = {}
    for s in MARKET:
        current = [(d, "2.0") for d in days]
        archive = [
            alfred_row(
                d,
                (sessions[i + 1] if i + 1 < len(sessions) else sessions[i]).strftime("%Y-%m-%d"),
                "9999-12-31",
                "2.0",
            )
            for i, d in enumerate(days)
            if d >= fv
        ]
        market[s] = market_raw(s, current, archive, fv)
    settings = Settings(_env_file=None, data_dir=tmp_path, runs_dir=tmp_path / "runs")
    return build_snapshot(
        _config(),
        settings,
        sources=(SourceName.ETF, SourceName.ALFRED, SourceName.FRED_MARKET),
        price_client=FakePriceClient(price_frame(days, prices)),
        alfred_client=FakeAlfredClient({s: _monthly_vintages(s, sessions) for s in MACRO}),
        market_client=FakeMarketClient(market),
    )


def test_the_signal_panel_holds_returns_and_signals_only(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    cfg = _config()
    panel = build_signal_panel(snap, cfg)
    cols = list(panel.features.columns)
    assert cols == ["ret_SPY", "ret_TLT", *(SIGNAL_PREFIX + i for i in FROZEN_IDS)]
    assert panel.features.index[0] == pd.Timestamp(START)
    # With the warm-up, only the first return is missing: every signal is complete at START.
    sig = panel.features[[c for c in cols if c.startswith(SIGNAL_PREFIX)]]
    assert sig.notna().all().all()
    assert panel.features["ret_SPY"].iloc[1:].notna().all()
    assert list(panel.targets.columns) == ["fwd_SPY", "fwd_TLT"]
    # A macro change is computed inside the vintage usable at t: the monthly series rises by 1
    # per month and every release revises the previous month by +0.1.
    v = panel.features["sig_ip_growth"]
    o = panel.obs_dates["sig_ip_growth"]
    i = 30
    month = (pd.Timestamp(o.iloc[i]).year - 2018) * 12 + pd.Timestamp(o.iloc[i]).month - 10
    assert v.iloc[i] == pytest.approx(math.log((100 + month) / (100 + month - 1 + 0.1)))


def test_the_signal_panel_refuses_a_late_snapshot(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    late = _config().model_copy(
        update={"signals": SignalsConfig(ids=FROZEN_IDS, warmup_start=date(2018, 1, 2))}
    )
    with pytest.raises(ValueError, match="warm-up"):
        build_signal_panel(snap, late)


def test_warmup_after_the_sample_start_is_refused() -> None:
    cfg = _config()
    with pytest.raises(ValueError, match="warmup_start"):
        PipelineConfig.model_validate(
            {
                **cfg.model_dump(),
                "signals": {"ids": list(FROZEN_IDS), "warmup_start": date(2021, 1, 4)},
            }
        )


def test_the_max_lag_rule_uses_current_values_with_the_maximum_lag(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    cfg = _config()
    assert cfg.data.fred_market is not None
    mx = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(
                update={
                    "fred_market": cfg.data.fred_market.model_copy(update={"lag_rule": "max_lag"})
                }
            )
        }
    )
    a, b = build_signal_panel(snap, cfg), build_signal_panel(snap, mx)
    # The fake archive posts every value the next session (max lag 1 -> usable 2 sessions after
    # its date); before the archive the primary rule uses pre_archive_lag = 2 as well, so the two
    # rules agree here except where archived vintages and current values differ: nowhere.
    assert a.features["sig_term_spread"].equals(b.features["sig_term_spread"])
    od_a, od_b = a.obs_dates["sig_term_spread"], b.obs_dates["sig_term_spread"]
    sessions = snap.etf_sessions()  # the full fetched range: observations may predate the panel
    lag_b = [sessions.get_loc(t) - sessions.get_loc(pd.Timestamp(o)) for t, o in od_b.items()]
    assert set(lag_b) == {2}  # usable max_posting_lag + 1 = 2 sessions after its date
    assert od_a.equals(od_b)
