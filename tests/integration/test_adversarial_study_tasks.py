"""Every Stage 4 study task kind on one tiny seed (ADR 0016: no arm or task may first run in the
full study). n_obs 2500, Stage 3 window 300, small bootstraps. Not a statistical check."""

from functools import cache

import pytest

from invariant.core.run_hash import read_git_state
from invariant.stages.adversarial.certify import CertificationConfig
from invariant.stages.adversarial.study import (
    GATED,
    REPORTED,
    StudyConfig,
    TaskRecord,
    Thresholds,
    budget_plan,
    load_study_config,
    pilot_thresholds,
    run_task,
    summarize_study,
)

ROOT = 7  # a smoke root seed: not the study's, the pilot's or any earlier study's


@cache
def _cfg() -> StudyConfig:
    return load_study_config().model_copy(
        update={
            "n_seeds": 10,
            "n_report_seeds": 10,
            "n_obs": 2500,
            "overrides": {
                "portfolio": {"window": 300, "min_obs": 600, "lw_n_boot": 99, "lw_alpha": 0.49},
                "decay": {"min_obs": 1000, "n_boot": 99},
            },
            "population_n_obs": 20_000,
            "causal_n_boot": 99,
            "certification": CertificationConfig(
                n_region_boot=99, n_grid=5, n_boot=99, gpd_n_boot=2
            ),
        }
    )


@cache
def _record(kind: str) -> TaskRecord:
    git, patch = read_git_state()
    return run_task(kind, 0, _cfg(), root_seed=ROOT, git_state=git, patch=patch)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", GATED + REPORTED)
def test_every_task_kind_runs(kind: str) -> None:
    rec = _record(kind)
    assert rec.kind == kind
    if kind == "volfrag":
        assert {o.members for o in rec.outcomes} == {("s_volfrag",), ("s_real",)}
        robust = rec.outcome(("s_real",))
        assert robust is not None and robust.contains_population is not None
    elif kind == "pipeline":
        assert rec.signals  # Stages 1-3 ran on the production path
    else:
        assert len(rec.outcomes) == 1 and rec.outcomes[0].block_length >= 2


def test_pipeline_reaches_stage4_on_this_seed() -> None:
    rec = _record("pipeline")
    assert rec.outcomes, "pick a smoke seed on which some signal passes Stage 3"
    assert any(s.stage4 is not None for s in rec.signals)


def test_summary_pilot_and_budget_run_on_the_records() -> None:
    records = [_record(k) for k in GATED + REPORTED]
    cfg = _cfg()
    thresholds = pilot_thresholds(records, cfg)
    assert [t.gate for t in thresholds] == ["S2_frag_kill", "S4_robust_pass", "S5_noregime_pass"]
    plan = budget_plan(records, cfg, n_jobs=8)
    assert plan["core_hours"] > 0
    report = summarize_study(
        records,
        cfg.model_copy(
            update={
                "thresholds": Thresholds(
                    s2_kill_lower_min=0.5, s4_pass_lower_min=0.8, s5_pass_lower_min=0.8
                )
            }
        ),
    )
    assert [c.name for c in report.criteria][:2] == ["S1_population_miss", "S3_null_kill"]
