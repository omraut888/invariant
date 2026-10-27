from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from invariant.benchmark.ground_truth import TargetTimingMismatchError
from invariant.config.settings import (
    PitConfig,
    PortfolioConfig,
    PublicationLags,
    TargetConfig,
    load_config,
)
from invariant.core.run_hash import RngFactory
from invariant.core.types import KillReason, StageName, Verdict
from invariant.data.pit import PitPanel
from invariant.stages.dro.metrics import sharpe, sharpe_difference_interval
from invariant.stages.dro.study import load_dro_study_config
from invariant.stages.dro.walkforward import Arm, run_walk_forward
from invariant.stages.portfolio.result import MemberSignal, PortfolioResult
from invariant.stages.portfolio.stage import (
    decide,
    ledger_metrics,
    run_portfolio_members,
    walk_config,
)

TINY = PortfolioConfig(
    c_annual=0.5,
    window=120,
    refit_every=7,
    cv_every=60,
    cv_valid=40,
    ridge_grid=(0.0, 0.3),
    eps_grid=(0.0, 0.05),
    fc_alpha=0.05,
    fc_draws=200,
    lw_alpha=0.05,
    lw_n_boot=99,
    b_n_boot=99,
    confidence=0.95,
    cost_bp=5.0,
    min_obs=200,
)


def _panel(
    x: np.ndarray, r: np.ndarray, *, lag: int, horizon: int = 1, prefix: str = "macro_"
) -> PitPanel:
    index = pd.bdate_range("2001-01-01", periods=len(r))
    feats = pd.DataFrame(x, index=index, columns=[f"{prefix}x{k}" for k in range(x.shape[1])])
    rets = pd.DataFrame(r, index=index, columns=[f"ret_A{k}" for k in range(r.shape[1])])
    features = pd.concat([feats, rets], axis=1)
    config = PitConfig(
        targets=TargetConfig(execution_lag=lag, horizon=horizon),
        publication_lag_sessions=PublicationLags(french=42),
    )
    empty = pd.DataFrame(index=index)
    return PitPanel(features, empty, empty, config)


def _planted(n: int, lead: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """i.i.d. x whose value at t moves every asset's return at t + 1 + lead."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1))
    r = 0.01 * rng.standard_normal((n, 3))
    r[1 + lead :] += 0.006 * x[: n - 1 - lead]
    return x, r


def _members(panel: PitPanel) -> list[tuple[str, str]]:
    cols = [str(c) for c in panel.features.columns if not str(c).startswith("ret_")]
    return [(c.lower(), c) for c in cols]


def test_base_config_is_the_one_adr_0013_evaluated() -> None:
    port, study = load_config().portfolio, load_dro_study_config()
    w = study.walk
    assert port.c_annual == study.c_annual
    assert (port.window, port.refit_every, port.cv_every, port.cv_valid) == (
        w.window,
        w.refit_every,
        w.cv_every,
        w.cv_valid,
    )
    assert port.ridge_grid == w.ridge_grid and port.eps_grid == w.eps_grid
    assert (port.fc_alpha, port.fc_draws) == (w.fc_alpha, w.fc_draws)
    assert (port.lw_alpha, port.lw_n_boot, port.cost_bp) == (
        study.lw_alpha,
        study.lw_n_boot,
        study.cost_bp,
    )
    assert walk_config(port, 0) == study.walk.config(study.c)


def test_decide_passes_only_below_alpha() -> None:
    assert decide(0.049, 0.05) == (Verdict.PASS, None)
    kill = (Verdict.KILL, KillReason.PORTFOLIO_NO_EDGE_OVER_EQUAL_WEIGHT)
    assert decide(0.05, 0.05) == kill and decide(0.9, 0.05) == kill
    assert KillReason.PORTFOLIO_NO_EDGE_OVER_EQUAL_WEIGHT.stage is StageName.PORTFOLIO


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_stage_timing_follows_the_pit_execution_lag(lag: int) -> None:
    x, r = _planted(600, lead=lag)
    panel = _panel(x, r, lag=lag)
    result = run_portfolio_members(panel, _members(panel), TINY, RngFactory(3))
    ev = result.evidence
    assert ev is not None and ev.execution_lag == lag
    direct = run_walk_forward(x, r, [Arm.RIDGE_CV], walk_config(TINY, lag), RngFactory(0))
    path = direct[Arm.RIDGE_CV]
    assert ev.n_test_days == len(path.dates)
    assert ev.deciding.sharpe == pytest.approx(sharpe(path.returns))
    assert ev.first_decision == panel.features.index[int(path.dates[0])].date()
    # The planted lead is found only at the matching lag.
    other = run_portfolio_members(
        _panel(x, r, lag=(lag + 1) % 3), _members(panel), TINY, RngFactory(3)
    )
    assert other.evidence is not None
    assert ev.deciding.sharpe > 2.0 > other.evidence.deciding.sharpe


def test_a_benchmark_panel_with_a_mismatched_lag_is_refused() -> None:
    x, r = _planted(400, lead=0)
    syn = _panel(x, r, lag=1, prefix="syn_")
    with pytest.raises(TargetTimingMismatchError, match="execution_lag=1"):
        run_portfolio_members(syn, _members(syn), TINY, RngFactory(3))
    ok = run_portfolio_members(
        _panel(x, r, lag=0, prefix="syn_"), _members(syn), TINY, RngFactory(3)
    )
    assert ok.evidence is not None and ok.evidence.execution_lag == 0
    real = _panel(x, r, lag=1)  # a non-benchmark panel may use lag 1
    assert run_portfolio_members(real, _members(real), TINY, RngFactory(3)).evidence is not None


def test_horizon_other_than_one_is_refused() -> None:
    x, r = _planted(400, lead=0)
    panel = _panel(x, r, lag=0, horizon=2)
    with pytest.raises(ValueError, match="horizon"):
        run_portfolio_members(panel, _members(panel), TINY, RngFactory(3))


def test_short_or_gappy_data_kills_every_member_with_a_general_reason() -> None:
    x, r = _planted(150, lead=0)
    x = np.column_stack([x, x[:, 0] * 0.5])
    panel = _panel(x, r, lag=0)
    short = run_portfolio_members(panel, _members(panel), TINY, RngFactory(3))
    assert short.evidence is None and len(short.members) == 2
    assert {m.reason for m in short.members} == {KillReason.INSUFFICIENT_DATA}
    x, r = _planted(400, lead=0)
    x[200, 0] = np.nan
    gappy = _panel(x, r, lag=0)
    res = run_portfolio_members(gappy, _members(gappy), TINY, RngFactory(3))
    assert res.evidence is None and res.members[0].reason is KillReason.DATA_VALIDATION_FAILED
    assert "missing" in (res.members[0].data_issue or "")


def test_no_member_means_no_verdict() -> None:
    x, r = _planted(400, lead=0)
    res = run_portfolio_members(_panel(x, r, lag=0), [], TINY, RngFactory(3), not_reached=("a",))
    assert res.members == () and res.verdict is None and res.not_reached == ("a",)


def test_members_share_the_verdict_and_b_is_always_defined() -> None:
    x, r = _planted(600, lead=0)
    x = np.column_stack([x, np.random.default_rng(9).standard_normal(len(x))])
    panel = _panel(x, r, lag=0)
    two = run_portfolio_members(panel, _members(panel), TINY, RngFactory(3))
    assert len({m.verdict for m in two.members}) == 1
    assert all(m.baseline == "other_members" for m in two.members)
    b = {m.signal_id: m.marginal_sharpe for m in two.members}
    assert b["macro_x0"] is not None and b["macro_x1"] is not None
    assert b["macro_x0"] > 1.0 > abs(b["macro_x1"])  # the planted signal carries the value
    one = run_portfolio_members(panel, _members(panel)[:1], TINY, RngFactory(3))
    m = one.members[0]
    assert m.baseline == "intercept_only" and m.marginal_sharpe is not None
    assert m.marginal_ci_low is not None and m.marginal_ci_low <= m.marginal_sharpe
    assert two == run_portfolio_members(panel, _members(panel), TINY, RngFactory(3))


def test_ledger_metrics_are_flat_finite_scalars() -> None:
    x, r = _planted(600, lead=0)
    panel = _panel(x, r, lag=0)
    res = run_portfolio_members(panel, _members(panel), TINY, RngFactory(3))
    metrics = ledger_metrics(res.members[0], res)
    assert metrics["deciding_arm"] == "ridge_cv"
    assert metrics["method"] == "ridge_cv_selected_wasserstein_dro_evaluated"
    assert "ridge_cv_lw_p" in metrics and "dro_fc_sharpe" in metrics
    for v in metrics.values():
        assert isinstance(v, (bool, int, float, str))
        if isinstance(v, float):
            assert math.isfinite(v)


def test_result_rejects_members_with_different_verdicts() -> None:
    a = MemberSignal(signal_id="a", column="a", verdict=Verdict.PASS, reason=None)
    b = MemberSignal(
        signal_id="b",
        column="b",
        verdict=Verdict.KILL,
        reason=KillReason.PORTFOLIO_NO_EDGE_OVER_EQUAL_WEIGHT,
    )
    with pytest.raises(ValidationError, match="share"):
        PortfolioResult(
            config=TINY,
            decay_config=None,
            n_obs=1,
            first_date=date(2001, 1, 1),
            last_date=date(2001, 1, 1),
            members=(a, b),
            evidence=None,
            not_reached=(),
        )
    with pytest.raises(ValidationError, match="Stage 3"):
        MemberSignal(
            signal_id="a", column="a", verdict=Verdict.KILL, reason=KillReason.DECAY_EDGE_DECLINING
        )


def test_sharpe_difference_interval() -> None:
    rng = np.random.default_rng(1)
    a = 0.01 * rng.standard_normal(800) + 0.001
    b = 0.01 * rng.standard_normal(800)
    d, lo, hi = sharpe_difference_interval(
        a, a, np.random.default_rng(0), n_boot=99, confidence=0.9
    )
    assert d == lo == hi == 0.0
    d, lo, hi = sharpe_difference_interval(
        a, b, np.random.default_rng(0), n_boot=199, confidence=0.9
    )
    assert lo < d < hi
    d2, lo2, hi2 = sharpe_difference_interval(
        b, a, np.random.default_rng(0), n_boot=199, confidence=0.9
    )
    assert d2 == pytest.approx(-d) and lo2 == pytest.approx(-hi) and hi2 == pytest.approx(-lo)
    cash = np.zeros(800)
    assert (
        sharpe_difference_interval(cash, cash, np.random.default_rng(0), n_boot=9, confidence=0.9)[
            0
        ]
        == 0.0
    )
