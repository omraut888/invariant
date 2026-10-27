"""Stage 3 through the production path on tiny seeds: every task kind of the ADR 0014 validation,
the ledger chain over (causal, decay, portfolio), scoring, and the summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from invariant.core.run_hash import GitState
from invariant.core.types import StageName, Verdict
from invariant.stages.dro.walkforward import Arm
from invariant.stages.portfolio.validation import (
    PortfolioValidationConfig,
    TaskRecord,
    load_portfolio_validation_config,
    run_task,
    summarize_validation,
)

GIT = GitState(sha="a" * 40, dirty=False)
TINY = {
    "window": 300,
    "refit_every": 21,
    "cv_every": 1000,
    "cv_valid": 100,
    "ridge_grid": [0.0, 1.0],
    "eps_grid": [0.0, 0.05],
    "fc_draws": 500,
    "lw_n_boot": 99,
    "b_n_boot": 99,
    "min_obs": 600,
}
SEED = 2  # root seed at which s_real reaches Stage 3 at n_obs 2500


def _cfg() -> PortfolioValidationConfig:
    return PortfolioValidationConfig.model_validate(
        {
            "root_seed": SEED,
            "n_seeds": 1,
            "confidence": 0.95,
            "n_boot": 99,
            "pipeline_n_obs": 2500,
            "overrides": {"portfolio": TINY, "decay": {"min_obs": 1000, "n_boot": 99}},
            "size": {
                "scenarios": ["d3_null", "d5_null_heavy"],
                "n_obs": 800,
                "members": ["s_real", "s_decay"],
                "upper_max": 0.075,
            },
            "power": {"signal": "s_real", "lower_min": 0.7},
        }
    )


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> list[TaskRecord]:
    cfg = _cfg()
    work: Path = tmp_path_factory.mktemp("portfolio")
    recs = [run_task("size", s, 0, cfg, git_state=GIT, work_dir=work) for s in cfg.size.scenarios]
    recs.append(run_task("pipeline", "d1_base", 0, cfg, git_state=GIT, work_dir=work))
    return recs


def test_the_preregistered_config_loads() -> None:
    cfg = load_portfolio_validation_config()
    assert cfg.n_seeds == 1273 and cfg.size.n_obs == 5001 and cfg.power.lower_min == 0.70
    assert cfg.overrides == {} and cfg.pipeline_n_obs is None


def test_size_tasks_force_both_members_and_report_every_arm(records: list[TaskRecord]) -> None:
    for rec in records[:2]:
        assert rec.kind == "size" and rec.members == ("s_real", "s_decay")
        assert rec.verdict in (Verdict.PASS, Verdict.KILL)
        assert {a.arm for a in rec.arms} == set(Arm)
        assert [a.arm for a in rec.arms if a.decides] == [Arm.RIDGE_CV]
        assert {m.signal_id for m in rec.marginals} == {"s_real", "s_decay"}
        assert rec.score is None


def test_pipeline_task_runs_three_stages_and_scores_them(records: list[TaskRecord]) -> None:
    rec = records[2]
    assert rec.members == ("s_real",) and rec.score is not None and rec.score.complete
    assert rec.score.stages == (StageName.CAUSAL, StageName.DECAY, StageName.PORTFOLIO)
    by = {s.signal_id: s for s in rec.signals}
    assert by["s_real"].stage3 is rec.verdict
    assert all(s.stage3 is None for sid, s in by.items() if sid != "s_real")
    assert rec.marginals[0].signal_id == "s_real"  # intercept-only baseline


def test_summary_and_criteria(records: list[TaskRecord]) -> None:
    report = summarize_validation(
        records, _cfg(), n_seeds=1, lw_alpha=0.05, git_sha="a" * 40, git_dirty=False
    )
    assert [c.name for c in report.criteria] == ["V1_size", "V1_size", "V2_power", "V3_complete"]
    assert report.criteria[-1].passed
    pipe = report.scenarios[-1]
    assert pipe.kind == "pipeline" and pipe.member_sets == {"s_real": 1}
    assert pipe.aggregate is not None
