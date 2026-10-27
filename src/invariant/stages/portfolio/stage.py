r"""Stage 3, portfolio construction: does a portfolio of the surviving signals beat 1/N?
(docs/adr/0013 for the method study, 0014 for the stage)

Wasserstein DRO was implemented, verified and evaluated in ADR 0013, and **ridge
regularization was selected**: the deciding arm is Ridge-CV. Markowitz, DRO-FC and DRO-CV run on
the same data and are reported only. A memo must describe the portfolio as Ridge-CV, never as
distributionally robust.

Method
------
1. Inputs: Stage 2's ``DecayResult`` and the same PIT panel. Stage 1's input selection is
   reapplied and its row count and dates must match the ``DecayResult``; otherwise
   ``ValueError``, because the result would belong to another panel. ``fwd_*`` is never read.
2. Members: the signals that passed Stage 2. The data are their columns and every ``ret_*``
   column, with leading incomplete rows trimmed. No member: nothing runs.
3. Timing from the PIT config: a decision at the close of t uses rows <= t, is entered at the
   close of t + e (``pit.targets.execution_lag``) and earns ``ret[t+1+e]``
   (``dro.walkforward``). Only ``horizon = 1`` is supported (daily rebalancing, as validated).
   A benchmark panel (``syn_*`` members) must use the timing ``GroundTruth`` was computed for;
   anything else raises ``TargetTimingMismatchError``.
4. Walk-forward over all five arms (1/N, Markowitz, DRO-FC, Ridge-CV, DRO-CV) with ADR 0013's
   settings: window 1260, refit every 21, walk-forward CV every 252 on the last 252 rows, c =
   c_annual / sqrt(252), gross exposure <= 1.
5. Kill rule A: kill every member with ``portfolio.no_edge_over_equal_weight`` when the
   one-sided Ledoit-Wolf (2008) test of Ridge-CV's Sharpe against 1/N gives ``p >= lw_alpha``.
   A general kill comes first: ``general.data_validation_failed`` for a missing value inside
   the sample, ``general.insufficient_data`` below ``min_obs`` complete rows.
6. Diagnostic B (never decides): for each member, Ridge-CV's Sharpe minus Ridge-CV's Sharpe
   without that member (with one member: an intercept-only forecast), with a paired block
   bootstrap interval.
7. Also reported: every arm's gross and net-of-cost Sharpe, exposure and turnover, LW p-value
   and the Deflated Sharpe Ratio over the four candidates (Bailey & Lopez de Prado 2014).

Assumptions and limitations (ADRs 0013-0014):
- Linear forecasts from the members dated t; Stage 1's best lag is not used (it is not
  identified, and ADR 0013 validated lag 0 features).
- Portfolio returns are ``w . r`` with log returns, a first-order approximation; position drift
  between rebalances is ignored.
- The kill test's size was gated at 6000 rows (ADR 0013) and is validated at 5000 (ADR 0014).
  Its power at realistic signal strength is close to zero (ADR 0013, finding 2).
- Selection was on gross Sharpe; net of 5 bp DRO-FC was better (ADR 0013, finding 1).

Look-ahead: every weight dated t uses rows <= t (tests/leakage/test_dro_no_lookahead.py). The
verdict is a full-sample statement about the out-of-sample path and is not used to trade at t.

References: Ledoit & Wolf (2008), J. Empirical Finance 15(5); Bailey & Lopez de Prado (2014),
J. Portfolio Management 40(5).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from invariant.benchmark.ground_truth import (
    EXPECTED_VALUES_TARGET_TIMING,
    TargetTimingMismatchError,
)
from invariant.config.settings import PortfolioConfig, TargetConfig
from invariant.core.logging import get_logger
from invariant.core.run_hash import RngFactory, canonical_json
from invariant.core.types import KillReason, StageName, Verdict
from invariant.data.pit import PitPanel
from invariant.ledger.kill_ledger import KillLedger, LedgerEntry, MetricValue
from invariant.stages.causal.stage import RETURN_PREFIX, SYNTHETIC_PREFIX, select_inputs
from invariant.stages.decay.result import DecayResult
from invariant.stages.dro.metrics import (
    TRADING_DAYS,
    deflated_sharpe_ratio,
    exposure_shares,
    sharpe,
    sharpe_difference_interval,
    sharpe_difference_test,
    turnover,
)
from invariant.stages.dro.walkforward import Arm, ArmPath, WalkForwardConfig, run_walk_forward
from invariant.stages.portfolio.result import (
    DECIDING_ARM,
    ArmReport,
    MemberSignal,
    PortfolioEvidence,
    PortfolioResult,
)

__all__ = [
    "ARMS",
    "PORTFOLIO_RESULT_NAME",
    "construct_portfolio",
    "decide",
    "ledger_metrics",
    "record_portfolio_ledger",
    "require_benchmark_timing",
    "run_portfolio_members",
    "run_portfolio_stage",
    "walk_config",
    "write_portfolio_result",
]

PORTFOLIO_RESULT_NAME: Final = "portfolio_result.json"
ARMS: Final = (Arm.EQUAL_WEIGHT, Arm.MARKOWITZ, Arm.DRO_FC, Arm.RIDGE_CV, Arm.DRO_CV)
_CANDIDATES: Final = ARMS[1:]

_FloatArray = npt.NDArray[np.float64]


def walk_config(config: PortfolioConfig, lag: int) -> WalkForwardConfig:
    """The walk-forward settings of ``config`` at execution lag ``lag``."""
    return WalkForwardConfig(
        c=config.c_annual / math.sqrt(TRADING_DAYS),
        window=config.window,
        refit_every=config.refit_every,
        cv_every=config.cv_every,
        cv_valid=config.cv_valid,
        fc_alpha=config.fc_alpha,
        fc_draws=config.fc_draws,
        ridge_grid=config.ridge_grid,
        eps_grid=config.eps_grid,
        lag=lag,
    )


def decide(p: float, alpha: float) -> tuple[Verdict, KillReason | None]:
    """Kill rule A from the LW p-value of Ridge-CV against 1/N: pass iff ``p < alpha``. A tie at
    alpha kills, because the strategy has not shown it beats 1/N."""
    if p < alpha:
        return Verdict.PASS, None
    return Verdict.KILL, KillReason.PORTFOLIO_NO_EDGE_OVER_EQUAL_WEIGHT


def require_benchmark_timing(columns: Sequence[str], targets: TargetConfig) -> None:
    """Refuse a benchmark panel (any ``syn_*`` column) whose timing is not the one
    ``GroundTruth``'s expected values assume (``S_t -> R_{t+1}``, execution lag 0)."""
    if any(c.startswith(SYNTHETIC_PREFIX) for c in columns) and (
        targets != EXPECTED_VALUES_TARGET_TIMING
    ):
        raise TargetTimingMismatchError(
            f"benchmark panel with execution_lag={targets.execution_lag}, "
            f"horizon={targets.horizon}; GroundTruth assumes "
            f"execution_lag={EXPECTED_VALUES_TARGET_TIMING.execution_lag}, "
            f"horizon={EXPECTED_VALUES_TARGET_TIMING.horizon}"
        )


def _sr(r: _FloatArray) -> float:
    s = sharpe(r)
    return 0.0 if math.isnan(s) else s


def _report(path: ArmPath, cost_bp: float) -> ArmReport:
    to = turnover(path.weights)
    shares = exposure_shares(path.weights)
    return ArmReport(
        arm=path.arm,
        decides=path.arm is DECIDING_ARM,
        sharpe=_sr(path.returns),
        sharpe_net=_sr(path.returns - cost_bp * 1e-4 * to),
        mean=float(path.returns.mean()),
        sd=float(path.returns.std(ddof=1)),
        invested=shares["invested"],
        cash=shares["cash"],
        partial=shares["partial"],
        turnover=float(to.mean()),
    )


def construct_portfolio(
    x: _FloatArray,
    r: _FloatArray,
    index: pd.DatetimeIndex,
    return_columns: Sequence[str],
    member_ids: Sequence[str],
    config: PortfolioConfig,
    *,
    lag: int,
    rng: RngFactory,
) -> tuple[PortfolioEvidence, list[tuple[float, float, float]]]:
    """Every arm on members ``x`` (T, K) and returns ``r`` (T, N), plus diagnostic B per member
    as ``(marginal Sharpe, low, high)``. Deterministic given ``rng``."""
    wf = walk_config(config, lag)
    paths = run_walk_forward(x, r, ARMS, wf, rng.child("wf"))
    reports = {a: _report(p, config.cost_bp) for a, p in paths.items()}
    base = paths[Arm.EQUAL_WEIGHT].returns
    daily = [sharpe(paths[a].returns, annualize=False) for a in _CANDIDATES]
    var = float(np.var([0.0 if math.isnan(s) else s for s in daily], ddof=1))
    lw = rng.child("lw")
    for a in _CANDIDATES:
        delta, p = sharpe_difference_test(
            paths[a].returns, base, lw.generator(a.value), n_boot=config.lw_n_boot
        )
        dsr = deflated_sharpe_ratio(
            paths[a].returns, n_trials=len(_CANDIDATES), trial_sharpe_variance=var
        )
        reports[a] = reports[a].model_copy(
            update={
                "lw_delta": None if math.isnan(delta) else delta,
                "lw_p": p,
                "dsr": None if math.isnan(dsr) else dsr,
            }
        )

    ridge = paths[DECIDING_ARM]
    b_rng = rng.child("b")
    marginals = []
    for k, sid in enumerate(member_ids):
        keep = [j for j in range(x.shape[1]) if j != k]
        drop = run_walk_forward(x[:, keep], r, [DECIDING_ARM], wf, rng.child(f"drop/{sid}"))
        marginals.append(
            sharpe_difference_interval(
                ridge.returns,
                drop[DECIDING_ARM].returns,
                b_rng.generator(sid),
                n_boot=config.b_n_boot,
                confidence=config.confidence,
            )
        )
    evidence = PortfolioEvidence(
        execution_lag=lag,
        n_rows=len(r),
        n_test_days=len(ridge.dates),
        first_decision=index[int(ridge.dates[0])].date(),
        last_decision=index[int(ridge.dates[-1])].date(),
        arms=tuple(reports[a] for a in ARMS),
        ridge_lambdas=tuple(ridge.chosen),
        final_weights=tuple(
            (c, float(w)) for c, w in zip(return_columns, ridge.weights[-1], strict=True)
        ),
    )
    return evidence, marginals


def _general_kill(
    members: Sequence[tuple[str, str]], reason: KillReason, issue: str
) -> tuple[MemberSignal, ...]:
    return tuple(
        MemberSignal(
            signal_id=sid, column=col, verdict=Verdict.KILL, reason=reason, data_issue=issue
        )
        for sid, col in members
    )


def run_portfolio_members(
    panel: PitPanel,
    members: Sequence[tuple[str, str]],
    config: PortfolioConfig,
    rng: RngFactory,
    *,
    decay: DecayResult | None = None,
    not_reached: Sequence[str] = (),
) -> PortfolioResult:
    """Stage 3 on the given ``(signal_id, column)`` members of ``panel`` (module docstring,
    steps 2-7). ``run_portfolio_stage`` takes the members from Stage 2; the size validation
    (ADR 0014) gives them directly. Claims the ``portfolio`` child of ``rng``."""
    log = get_logger(__name__)
    targets = panel.config.targets
    if targets.horizon != 1:
        raise ValueError(f"Stage 3 rebalances daily; horizon {targets.horizon} is not supported")
    columns = [str(c) for c in panel.features.columns]
    returns = [c for c in columns if c.startswith(RETURN_PREFIX)]
    member_cols = [col for _, col in members]
    require_benchmark_timing(member_cols, targets)
    stage_rng = rng.child(StageName.PORTFOLIO.value)
    frame = panel.features.loc[:, [*member_cols, *returns]]
    complete = frame.notna().all(axis=1).to_numpy()
    start = int(np.argmax(complete)) if complete.any() else len(frame)
    frame = frame.iloc[start:]
    index = pd.DatetimeIndex(frame.index)
    common = {
        "config": config,
        "decay_config": None if decay is None else decay.config,
        "n_obs": len(frame),
        "first_date": index[0].date() if len(index) else None,
        "last_date": index[-1].date() if len(index) else None,
        "not_reached": tuple(not_reached),
    }
    if not members:
        return PortfolioResult(**common, members=(), evidence=None)
    if not returns:
        raise ValueError("the panel has no return column (ret_*)")
    interior = [str(c) for c in frame.columns[frame.isna().any(axis=0).to_numpy()]]
    if interior:
        issue = f"missing values inside the sample in {interior}"
        log.warning("portfolio_data_invalid", columns=interior)
        kills = _general_kill(members, KillReason.DATA_VALIDATION_FAILED, issue)
        return PortfolioResult(**common, members=kills, evidence=None)
    min_rows = max(config.min_obs, config.window + 2 + 2 * targets.execution_lag)
    if len(frame) < min_rows:
        issue = f"{len(frame)} complete rows < min_obs {min_rows}"
        log.warning("portfolio_insufficient_data", n_rows=len(frame))
        kills = _general_kill(members, KillReason.INSUFFICIENT_DATA, issue)
        return PortfolioResult(**common, members=kills, evidence=None)

    t0 = time.perf_counter()
    # C-contiguous copies: the panel's block layout must not change BLAS summation order, so the
    # result depends only on the values (tests/leakage/test_portfolio_inputs.py).
    evidence, marginals = construct_portfolio(
        np.ascontiguousarray(frame.loc[:, member_cols].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(frame.loc[:, returns].to_numpy(dtype=np.float64)),
        index,
        returns,
        [sid for sid, _ in members],
        config,
        lag=targets.execution_lag,
        rng=stage_rng,
    )
    p = evidence.deciding.lw_p
    assert p is not None  # every candidate arm is tested
    verdict, reason = decide(p, config.lw_alpha)
    baseline = "other_members" if len(members) > 1 else "intercept_only"
    out = tuple(
        MemberSignal(
            signal_id=sid,
            column=col,
            verdict=verdict,
            reason=reason,
            marginal_sharpe=b,
            marginal_ci_low=lo,
            marginal_ci_high=hi,
            baseline=baseline,
        )
        for (sid, col), (b, lo, hi) in zip(members, marginals, strict=True)
    )
    log.info(
        "portfolio_stage_done",
        n_members=len(members),
        verdict=verdict.value,
        lw_p=p,
        seconds=round(time.perf_counter() - t0, 3),
    )
    return PortfolioResult(**common, members=out, evidence=evidence)


def run_portfolio_stage(
    panel: PitPanel, decay: DecayResult, config: PortfolioConfig, rng: RngFactory
) -> PortfolioResult:
    """Run Stage 3 on ``panel`` for the signals ``decay`` passed (module docstring)."""
    inputs = select_inputs(panel.features, decay.causal_config)
    index = inputs.frame.index
    first = index[0].date() if len(index) else None
    last = index[-1].date() if len(index) else None
    if (len(index), first, last) != (decay.n_obs, decay.first_date, decay.last_date):
        raise ValueError("the DecayResult was not produced from this panel")
    members = [(s.signal_id, s.column) for s in decay.signals if s.verdict is Verdict.PASS]
    killed = [s.signal_id for s in decay.signals if s.verdict is not Verdict.PASS]
    return run_portfolio_members(
        panel,
        members,
        config,
        rng,
        decay=decay,
        not_reached=(*decay.not_reached, *killed),
    )


def ledger_metrics(member: MemberSignal, result: PortfolioResult) -> dict[str, MetricValue]:
    """Flat, finite, scalar metrics of one member's Stage 3 decision, for its ledger entry."""
    cfg = result.config
    metrics: dict[str, MetricValue] = {
        "n_obs": result.n_obs,
        "min_obs": cfg.min_obs,
        "n_members": len(result.members),
        "deciding_arm": DECIDING_ARM.value,
        "method": "ridge_cv_selected_wasserstein_dro_evaluated",
        "test": "ledoit_wolf_one_sided_vs_equal_weight",
        "lw_alpha": cfg.lw_alpha,
        "lw_n_boot": cfg.lw_n_boot,
        "c_annual": cfg.c_annual,
        "cost_bp": cfg.cost_bp,
    }
    ev = result.evidence
    if ev is None:
        metrics["data_issue"] = member.data_issue or "not run"
        return metrics
    metrics |= {"execution_lag": ev.execution_lag, "n_test_days": ev.n_test_days}
    for a in ev.arms:
        key = a.arm.value
        metrics[f"{key}_sharpe"] = a.sharpe
        metrics[f"{key}_sharpe_net"] = a.sharpe_net
        metrics[f"{key}_invested"] = a.invested
        metrics[f"{key}_cash"] = a.cash
        metrics[f"{key}_turnover"] = a.turnover
        if a.lw_p is not None:
            metrics[f"{key}_lw_p"] = a.lw_p
        if a.lw_delta is not None:
            metrics[f"{key}_lw_delta"] = a.lw_delta
        if a.dsr is not None:
            metrics[f"{key}_dsr"] = a.dsr
    if member.marginal_sharpe is not None:
        assert member.marginal_ci_low is not None and member.marginal_ci_high is not None
        assert member.baseline is not None
        metrics |= {
            "b_marginal_sharpe": member.marginal_sharpe,
            "b_ci_low": member.marginal_ci_low,
            "b_ci_high": member.marginal_ci_high,
            "b_baseline": member.baseline,
        }
    return metrics


def record_portfolio_ledger(ledger: KillLedger, result: PortfolioResult) -> list[LedgerEntry]:
    """One Stage 3 ledger entry per member, all with the portfolio's verdict."""
    return [
        ledger.record(
            m.signal_id,
            StageName.PORTFOLIO,
            m.verdict,
            reason=m.reason,
            metrics=ledger_metrics(m, result),
        )
        for m in result.members
    ]


def write_portfolio_result(run_dir: Path, result: PortfolioResult) -> Path:
    """Write ``result`` as canonical JSON to ``<run_dir>/portfolio_result.json``."""
    path = run_dir / PORTFOLIO_RESULT_NAME
    path.write_text(canonical_json(result.model_dump(mode="json")) + "\n", encoding="utf-8")
    return path
