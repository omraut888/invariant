from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from invariant.stages.dro.study import (
    CANDIDATES,
    ArmMetrics,
    DroStudyConfig,
    OracleMetrics,
    TaskRecord,
    VariantResult,
    dro_scenarios,
    gate_seeds,
    load_dro_study_config,
    plan_budget,
    selection_seeds,
    study_tasks,
    summarize,
)
from invariant.stages.dro.walkforward import Arm


def test_preregistered_config_loads_with_its_stated_values() -> None:
    cfg = load_dro_study_config()
    assert cfg.c == pytest.approx(0.5 / math.sqrt(252))
    assert cfg.n_gating == 8
    assert gate_seeds(cfg) == 1683
    assert cfg.root_seed != cfg.pilot_root_seed


def test_scenarios_validate_and_plant_what_they_claim() -> None:
    by = {s.name: s.config(None) for s in dro_scenarios()}
    assert by["d1_base"].n_assets == 5 and by["d1_base"].n_obs == 6000
    for null in ("d3_null", "d5_null_heavy"):
        c = by[null]
        assert c.mu == 0.0 and c.factor.ic == c.real.ic == c.decay.ic == 0.0
    for heavy in ("d4_heavy", "d5_null_heavy"):
        c = by[heavy]
        assert c.garch.nu == 5.0 and c.market is not None and c.market.garch.nu == 5.0
    assert by["d2_realistic"].real.ic == 0.02


def test_config_rejects_a_scenario_in_two_roles_or_a_bad_order() -> None:
    data = load_dro_study_config().model_dump()
    with pytest.raises(ValidationError):
        DroStudyConfig.model_validate({**data, "reported_scenarios": ["d1_base"]})
    with pytest.raises(ValidationError):
        DroStudyConfig.model_validate({**data, "simplicity": ["markowitz", "dro_fc"]})
    with pytest.raises(ValidationError):
        DroStudyConfig.model_validate({**data, "size_gates": ["nope"]})


def test_selection_seeds_formula() -> None:
    # sigma_U = sigma * sqrt(19 / chi2_.05(19)) = sigma * 1.3770; z = 1.95996 + 1.28155
    m = selection_seeds(0.3, 20, delta=0.1, power=0.9, sigma_confidence=0.95, floor=1)
    sigma_u = 0.3 * math.sqrt(19 / 10.117013)
    assert m == math.ceil(((1.959964 + 1.281552) * sigma_u / 0.1) ** 2)
    assert selection_seeds(0.01, 20, delta=0.1, power=0.9, sigma_confidence=0.95, floor=200) == 200
    a = selection_seeds(0.2, 20, delta=0.1, power=0.9, sigma_confidence=0.95, floor=1)
    b = selection_seeds(0.4, 20, delta=0.1, power=0.9, sigma_confidence=0.95, floor=1)
    assert b > 3 * a


def _seconds(cfg: DroStudyConfig, t: float) -> dict[str, float]:
    keys = [f"{s.name}/main" for s in dro_scenarios()]
    keys += [f"{cfg.selection_scenario}/{k}" for k in ("loso", "pipeline", "c_sens")]
    return dict.fromkeys(keys, t)


def test_budget_plan_shrinks_reported_seeds_then_declares_infeasible() -> None:
    recorded = load_dro_study_config()
    assert recorded.n_report_seeds is not None
    kept = plan_budget(_seconds(recorded, 1.0), recorded, n_select=300, workers=8)
    assert kept.n_report_seeds == recorded.n_report_seeds  # a recorded value is honoured
    cfg = recorded.model_copy(update={"n_report_seeds": None})
    roomy = plan_budget(_seconds(cfg, 1.0), cfg, n_select=300, workers=8)
    assert roomy.feasible and roomy.n_report_seeds == 300
    # core = 2*1683 + 300 = 3666 task-units; reported = 5 per seed
    t = cfg.budget_hours * 3600 * 8 / (3666 + 5 * 150)
    tight = plan_budget(_seconds(cfg, t), cfg, n_select=300, workers=8)
    assert tight.feasible and tight.n_report_seeds == 150
    assert tight.projected_hours <= cfg.budget_hours + 1e-9
    over = plan_budget(_seconds(cfg, 300.0), cfg, n_select=300, workers=8)
    assert not over.feasible


def test_study_tasks_counts() -> None:
    cfg = load_dro_study_config()
    tasks = study_tasks(cfg, n_gate=10, n_select=4, n_report=3)
    assert len(tasks) == 2 * 10 + 4 + 2 * 3 + 3 * 3
    assert len(set(tasks)) == len(tasks)


def _arm(arm: Arm, sr: float, p: float | None) -> ArmMetrics:
    return ArmMetrics(
        arm=arm,
        sharpe=sr,
        sharpe_net=sr - 0.05,
        mean=0.0,
        sd=0.01,
        invested=1.0,
        cash=0.0,
        partial=0.0,
        turnover=0.1,
        lw_p=p,
        dsr=0.5 if p is not None else None,
    )


def _record(scenario: str, i: int, sharpes: dict[Arm, float], p: dict[Arm, float]) -> TaskRecord:
    arms = [_arm(Arm.EQUAL_WEIGHT, 0.4, None)]
    arms += [_arm(a, sharpes[a], p[a]) for a in CANDIDATES]
    v = VariantResult(
        label="main",
        signals=("s_real", "s_decay"),
        c_annual=0.5,
        arms=tuple(arms),
        oracles=(OracleMetrics(info="o1", sharpe=1.3, invested=1.0, cash=0.0, turnover=0.1),),
    )
    return TaskRecord(
        scenario=scenario, index=i, kind="main", dataset_seed=i, variants=(v,), seconds=1.0
    )


def _records(
    n: int, null_p: dict[Arm, float], sel: dict[Arm, float], noise: float = 0.05
) -> list[TaskRecord]:
    """Selection-scenario Sharpe ratios ``sel[a] + noise * (shared + own)``, with every noise
    series centred so that the sample mean differences are exactly the ``sel`` differences."""
    rng = np.random.default_rng(0)
    shared = rng.normal(size=n)
    own = {a: rng.normal(size=n) for a in CANDIDATES}
    own = {a: v - v.mean() for a, v in own.items()}
    recs = []
    for i in range(n):
        for s in ("d3_null", "d5_null_heavy"):
            recs.append(_record(s, i, dict.fromkeys(CANDIDATES, 0.0), null_p))
        sh = {a: sel[a] + noise * (shared[i] + own[a][i]) for a in CANDIDATES}
        recs.append(_record("d1_base", i, sh, dict.fromkeys(CANDIDATES, 0.01)))
    return recs


def test_selection_takes_the_simplest_tied_admissible_arm() -> None:
    cfg = load_dro_study_config()
    ok = dict.fromkeys(CANDIDATES, 0.5)
    sel = {Arm.MARKOWITZ: 0.30, Arm.DRO_FC: 0.70, Arm.RIDGE_CV: 0.705, Arm.DRO_CV: 0.71}
    report = summarize(_records(200, ok, sel, noise=0.2), cfg, git_sha=None, git_dirty=False)
    assert report.failed_gates == ()
    assert set(report.admissible) == set(CANDIDATES)
    tied = {r.arm for r in report.selection if r.tied}
    assert Arm.MARKOWITZ not in tied and Arm.DRO_CV in tied and Arm.DRO_FC in tied
    assert report.selected is Arm.DRO_FC


def test_an_arm_that_fails_a_size_gate_cannot_be_selected() -> None:
    cfg = load_dro_study_config()
    null_p = {Arm.MARKOWITZ: 0.5, Arm.DRO_FC: 0.01, Arm.RIDGE_CV: 0.5, Arm.DRO_CV: 0.5}
    sel = {Arm.MARKOWITZ: 0.2, Arm.DRO_FC: 0.9, Arm.RIDGE_CV: 0.6, Arm.DRO_CV: 0.61}
    report = summarize(_records(200, null_p, sel, noise=0.2), cfg, git_sha=None, git_dirty=False)
    assert set(report.failed_gates) == {"d3_null/dro_fc", "d5_null_heavy/dro_fc"}
    assert Arm.DRO_FC not in report.admissible
    assert report.selected in (Arm.RIDGE_CV, Arm.DRO_CV)
    assert report.selected is Arm.RIDGE_CV  # tied with dro_cv; simpler


def test_no_admissible_arm_means_no_selection() -> None:
    cfg = load_dro_study_config()
    report = summarize(
        _records(50, dict.fromkeys(CANDIDATES, 0.01), dict.fromkeys(CANDIDATES, 0.5)),
        cfg,
        git_sha=None,
        git_dirty=False,
    )
    assert report.admissible == () and report.selected is None
    assert "not built" in report.selection_reason


def test_summary_regret_and_kill_rates() -> None:
    cfg = load_dro_study_config()
    recs = _records(40, dict.fromkeys(CANDIDATES, 0.5), dict.fromkeys(CANDIDATES, 0.5), 0.0)
    report = summarize(recs, cfg, git_sha=None, git_dirty=False)
    row = next(a for a in report.arms if a.scenario == "d1_base" and a.arm is Arm.MARKOWITZ)
    assert row.regret_o1 == pytest.approx(0.8)
    assert row.beats_equal_weight is not None and row.beats_equal_weight.estimate == 1.0
    assert not row.gate
    null = next(a for a in report.arms if a.scenario == "d3_null" and a.arm is Arm.MARKOWITZ)
    assert null.gate and null.beats_equal_weight is not None
    assert null.beats_equal_weight.estimate == 0.0
