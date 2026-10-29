"""Stage 4 study rules (ADR 0016): config pins, M, the pilot threshold rule, acceptance, budget."""

import math

import pytest

from invariant.ledger.benchmark_score import wilson_interval
from invariant.stages.adversarial.certify import CertificationConfig
from invariant.stages.adversarial.study import (
    Outcome,
    TaskRecord,
    Thresholds,
    budget_plan,
    load_study_config,
    pilot_thresholds,
    summarize_study,
)
from invariant.stages.causal.calibration import required_seeds


def _outcome(members: tuple[str, ...], kill: bool, contains: bool | None = None) -> Outcome:
    return Outcome(
        members=members,
        stage3_p=0.01,
        kill=kill,
        sr_h_star=-1.0 if kill else 1.0,
        sr_h_star_upper=-0.1 if kill else 1.5,
        sr_h_baseline=1.0,
        sharpe_by_state=(1.0, 0.0),
        min_difference_to_equal_weight=0.0,
        worst_share=0.4,
        worst_duration=20.0,
        worst_start="high",
        block_length=100,
        region_share=(0.2, 0.4),
        region_duration=(10.0, 25.0),
        contains_population=contains,
    )


def _records(
    n: int, frag_kills: int, robust_kills: int, noreg_kills: int, null_kills: int, misses: int
) -> list[TaskRecord]:
    recs = []
    for i in range(n):
        recs.append(
            TaskRecord(
                kind="volfrag",
                index=i,
                dataset_seed=i,
                pipeline_seed=i,
                seconds=30.0,
                outcomes=(
                    _outcome(("s_volfrag",), i < frag_kills),
                    _outcome(("s_real",), i < robust_kills, contains=i >= misses),
                ),
            )
        )
        recs.append(
            TaskRecord(
                kind="null",
                index=i,
                dataset_seed=i,
                pipeline_seed=i,
                seconds=20.0,
                outcomes=(_outcome(("s_real", "s_volfrag"), i < null_kills),),
            )
        )
        recs.append(
            TaskRecord(
                kind="noregime",
                index=i,
                dataset_seed=i,
                pipeline_seed=i,
                seconds=20.0,
                outcomes=(_outcome(("s_real",), i < noreg_kills),),
            )
        )
        recs.append(
            TaskRecord(
                kind="regime",
                index=i,
                dataset_seed=i,
                pipeline_seed=i,
                seconds=20.0,
                outcomes=(_outcome(("s_regime",), False),),
            )
        )
        recs.append(
            TaskRecord(
                kind="pipeline",
                index=i,
                dataset_seed=i,
                pipeline_seed=i,
                seconds=120.0,
                outcomes=(),
            )
        )
    return recs


def test_study_config_is_the_preregistered_one() -> None:
    cfg = load_study_config()
    assert (cfg.root_seed, cfg.pilot_root_seed, cfg.population_seed) == (
        20261029,
        20261030,
        20261103,
    )
    assert cfg.n_seeds == required_seeds(0.05, 0.075, 2, 0.90) == 1273
    assert (cfg.n_report_seeds, cfg.n_pilot) == (200, 50)
    assert (cfg.size_upper_max, cfg.pilot_margin, cfg.s2_floor, cfg.pass_floor) == (
        0.075,
        0.05,
        0.5,
        0.8,
    )
    assert cfg.certification == CertificationConfig()
    assert cfg.overrides == {} and cfg.n_obs is None and cfg.causal_n_boot is None
    assert cfg.population_n_obs == 1_000_000


def test_pilot_threshold_rule() -> None:
    cfg = load_study_config().model_copy(update={"n_seeds": 200})  # small M keeps it fast
    th = {t.gate: t for t in pilot_thresholds(_records(50, 45, 1, 0, 2, 2), cfg)}
    lower = wilson_interval(45, 50).ci_low
    assert lower is not None
    assert th["S2_frag_kill"].threshold == math.floor((lower - 0.05) * 100) / 100 == 0.73
    assert th["S2_frag_kill"].feasible
    assert th["S5_noregime_pass"].threshold == 0.87  # 50 of 50: .929 - .05
    infeasible = {t.gate: t for t in pilot_thresholds(_records(50, 20, 1, 0, 2, 2), cfg)}
    assert not infeasible["S2_frag_kill"].feasible  # 20/50: lower .27 - .05 < .50
    assert 0.0 <= th["S2_frag_kill"].pass_probability_at_threshold <= 1.0


def test_acceptance_needs_thresholds_and_applies_every_gate() -> None:
    cfg = load_study_config()
    recs = _records(400, 380, 4, 2, 10, 10)
    with pytest.raises(ValueError, match="pilot"):
        summarize_study(recs, cfg)
    good = cfg.model_copy(
        update={
            "thresholds": Thresholds(
                s2_kill_lower_min=0.9, s4_pass_lower_min=0.95, s5_pass_lower_min=0.95
            )
        }
    )
    report = summarize_study(recs, good)
    assert report.accepted
    bad = summarize_study(_records(400, 380, 4, 2, 40, 10), good)  # null kill 10%
    assert not bad.accepted
    assert {c.name: c.passed for c in bad.criteria}["S3_null_kill"] is False


def test_budget_plan() -> None:
    cfg = load_study_config()
    plan = budget_plan(_records(10, 0, 0, 0, 0, 0), cfg, n_jobs=8)
    # gated: 30 + 20 + 20 s per seed at M = 1273 on 8 workers
    assert plan["core_hours"] == pytest.approx(70 * 1273 / 8 / 3600)
    assert plan["n_report_seeds"] == 200 and plan["feasible"]
    slow = [r.model_copy(update={"seconds": 2000.0}) for r in _records(10, 0, 0, 0, 0, 0)]
    assert not budget_plan(slow, cfg, n_jobs=8)["feasible"]
