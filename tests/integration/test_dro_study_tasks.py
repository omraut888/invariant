"""Every Stage 3 study scenario and task kind, end to end on one tiny seed, then the summary
rules on those records. So neither the pilot nor the full study can be the first thing to run
a code path (CLAUDE.md, "Method choices")."""

from __future__ import annotations

import pytest

from invariant.core.run_hash import read_git_state
from invariant.stages.dro.study import (
    CANDIDATES,
    DroStudyConfig,
    load_dro_study_config,
    pilot_report,
    plan_budget,
    run_records,
    summarize,
)
from invariant.stages.dro.walkforward import Arm

# Stage 1 passes s_real on this seed at this length, so the pipeline path runs arms too.
N_OBS, SEED = 2500, 1


def _tiny() -> DroStudyConfig:
    cfg = load_dro_study_config()
    data = cfg.model_dump()
    data.update(
        n_obs=N_OBS,
        lw_n_boot=99,
        n_diff_boot=200,
        pilot_seeds=2,
        walk={
            **data["walk"],
            "window": 300,
            "cv_every": 1000,
            "cv_valid": 100,
            "fc_draws": 500,
            "ridge_grid": (0.0, 0.3),
            "eps_grid": (0.0, 0.05),
        },
        pipeline={"min_obs": 1000},
    )
    return DroStudyConfig.model_validate(data)


@pytest.fixture(scope="module")
def records() -> list:  # type: ignore[type-arg]
    cfg = _tiny()
    scen = (*cfg.size_gates, cfg.selection_scenario, *cfg.reported_scenarios)
    tasks = [(s, SEED, "main") for s in scen]
    tasks += [(cfg.selection_scenario, SEED, k) for k in ("loso", "pipeline", "c_sens")]
    git, patch = read_git_state()
    return run_records(tasks, cfg, root_seed=cfg.root_seed, git_state=git, patch=patch)  # type: ignore[arg-type]


def test_every_task_kind_produces_complete_records(records: list) -> None:  # type: ignore[type-arg]
    cfg = _tiny()
    by = {(r.scenario, r.kind): r for r in records}
    assert len({r.dataset_seed for r in records}) == 1  # common random numbers
    main = by[(cfg.selection_scenario, "main")].variants[0]
    assert [m.arm for m in main.arms] == [Arm.EQUAL_WEIGHT, *CANDIDATES]
    assert {o.info for o in main.oracles} == {"o1", "o2"}
    for a in CANDIDATES:
        m = main.arm(a)
        assert m.lw_p is not None and 0.0 < m.lw_p <= 1.0
        assert abs(m.invested + m.cash + m.partial - 1.0) < 1e-9
    fc = main.arm(Arm.DRO_FC)
    assert fc.fc_coverage_o1 is not None and fc.fc_coverage_o2 is not None
    assert [v.label for v in by[(cfg.selection_scenario, "loso")].variants] == [
        "drop:s_real",
        "drop:s_decay",
    ]
    assert [v.c_annual for v in by[(cfg.selection_scenario, "c_sens")].variants] == [0.25, 1.0]
    pipe = by[(cfg.selection_scenario, "pipeline")]
    assert pipe.passed is not None and "s_real" in pipe.passed
    assert pipe.variants and pipe.variants[0].signals == pipe.passed
    for s in cfg.size_gates:  # nulls: no drift, no signal -> the oracle stays in cash
        o1 = next(o for o in by[(s, "main")].variants[0].oracles if o.info == "o1")
        assert o1.cash == 1.0 and o1.sharpe == 0.0


def test_summary_pilot_and_budget_run_on_the_records(records: list) -> None:  # type: ignore[type-arg]
    cfg = _tiny()
    report = summarize(records, cfg, git_sha=None, git_dirty=False)
    assert report.selected is None or report.selected in report.admissible
    assert len(report.selection) == len(CANDIDATES)
    assert {r.dropped for r in report.loso} == set(cfg.signals)
    assert sum(report.pipeline_sets.values()) == 1
    sel = [r for r in records if r.scenario == cfg.selection_scenario and r.kind == "main"]
    pilot = pilot_report([*sel, *[r.model_copy(update={"index": 1}) for r in sel]], cfg, 1)
    assert pilot.n_select_seeds >= cfg.min_select_seeds
    seconds = {f"{r.scenario}/{r.kind}": r.seconds for r in records}
    plan = plan_budget(seconds, cfg, n_select=pilot.n_select_seeds, workers=8)
    assert plan.n_gate_seeds == 1683
