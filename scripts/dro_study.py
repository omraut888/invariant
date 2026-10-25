"""Stage 3 pre-registered study (docs/adr/0013): arm selection and kill-test size.

    uv run python -m scripts.dro_study --pilot [--n-jobs N]
    uv run python -m scripts.dro_study [--n-jobs N] [--config PATH]

``--pilot`` runs every scenario and task kind on ``pilot_seeds`` seeds from ``pilot_root_seed``.
It prints only what the pilot is for: the paired-difference SDs, ``n_select_seeds`` and the
budget plan. Its rates and means are neither printed nor used.

The full run needs ``n_select_seeds`` recorded in the config. It writes
``runs/dro_study/<key>/report.json``, where ``<key>`` hashes the config, the seed counts and the
git state. Records are checkpointed to ``records.jsonl``, so a rerun with the same key resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from invariant.config.settings import find_project_root
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import canonical_json, read_git_state
from invariant.ledger.benchmark_score import RateEstimate
from invariant.stages.dro.study import (
    REPORTED_KINDS,
    DroStudyReport,
    Task,
    gate_seeds,
    load_dro_study_config,
    pilot_report,
    plan_budget,
    run_records,
    study_tasks,
    summarize,
    write_report,
)


def _r(rate: RateEstimate | None) -> str:
    if rate is None or rate.estimate is None:
        return "-"
    return f"{rate.estimate:.3f} [{rate.ci_low:.3f},{rate.ci_high:.3f}]"


def _f(v: float | None, fmt: str = "+.3f") -> str:
    return "-" if v is None else format(v, fmt)


def _print(report: DroStudyReport) -> None:
    print(
        "\nscenario       variant      arm           n    SR   SRnet  regO1  regO2  "
        "beats 1/N (kill-test rejections)  inv  cash  turn   DSR  FCcov(O1/O2)"
    )
    for a in report.arms:
        print(
            f"{a.scenario:14s} {a.variant:12s} {a.arm.value:12s}{'*' if a.gate else ' '}"
            f"{a.n:5d} {a.sharpe:+.3f} {a.sharpe_net:+.3f} {_f(a.regret_o1)} {_f(a.regret_o2)} "
            f"{_r(a.beats_equal_weight):26s} {a.invested:.2f} {a.cash:.2f} {a.turnover:.3f} "
            f"{_f(a.dsr_mean, '.2f')} {_f(a.fc_coverage_o1, '.3f')}/{_f(a.fc_coverage_o2, '.3f')}"
        )
    print(f"\nfailed size gates: {list(report.failed_gates)}")
    print(f"admissible: {[a.value for a in report.admissible]}")
    for s in report.selection:
        print(
            f"select {s.arm.value:10s} adm={s.admissible!s:5s} SR {s.mean_sharpe:+.3f} "
            f"best-minus {s.diff_from_best:+.3f} [{s.diff_ci_low:+.3f},{s.diff_ci_high:+.3f}] "
            f"tied={s.tied}"
        )
    print(f"selected: {report.selected} ({report.selection_reason})")
    for row in report.loso:
        print(
            f"B {row.arm.value:10s} drop {row.dropped:8s} n={row.n} marginal SR "
            f"{row.marginal_sharpe:+.3f} [{row.ci_low:+.3f},{row.ci_high:+.3f}]"
        )
    print(f"pipeline signal sets: {report.pipeline_sets}")
    print(f"median seconds per task: {report.median_seconds}")


def _out_dir(payload: dict[str, object]) -> Path:
    key = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]
    out = find_project_root() / "runs" / "dro_study" / key
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)
    cfg = load_dro_study_config(args.config)
    git, patch = read_git_state()
    base = {"config": cfg.model_dump(mode="json"), "git": git.model_dump()}

    if args.pilot:
        m = cfg.pilot_seeds
        scen = (*cfg.size_gates, cfg.selection_scenario, *cfg.reported_scenarios)
        tasks: list[Task] = [(s, i, "main") for s in scen for i in range(m)]
        tasks += [(cfg.selection_scenario, i, k) for k in REPORTED_KINDS for i in range(m)]
        out = _out_dir({**base, "pilot": m})
        if patch is not None:
            (out / "dirty.patch").write_bytes(patch)
        recs = run_records(
            tasks,
            cfg,
            root_seed=cfg.pilot_root_seed,
            git_state=git,
            patch=patch,
            n_jobs=args.n_jobs,
            checkpoint=out / "pilot_records.jsonl",
        )
        pilot = pilot_report(recs, cfg, workers=args.n_jobs)
        plan = plan_budget(pilot.seconds, cfg, n_select=pilot.n_select_seeds, workers=args.n_jobs)
        write_report(pilot, out / "pilot.json")
        write_report(plan, out / "budget.json")
        print(f"\npilot SDs of paired Sharpe differences: {pilot.pair_sd}")
        print(f"sigma_hat = {pilot.sigma_hat:.4f} -> n_select_seeds = {pilot.n_select_seeds}")
        print(f"mean task seconds: {pilot.seconds}")
        print(f"budget plan: {plan.model_dump()}")
        print(f"pilot: {out}")
        return 0

    if cfg.n_select_seeds is None:
        print("n_select_seeds is not recorded: run --pilot and record it first", file=sys.stderr)
        return 2
    m_gate = gate_seeds(cfg)
    m_rep = cfg.n_report_seeds if cfg.n_report_seeds is not None else cfg.n_select_seeds
    tasks = study_tasks(cfg, n_gate=m_gate, n_select=cfg.n_select_seeds, n_report=m_rep)
    print(f"M gate {m_gate}, M select {cfg.n_select_seeds}, M report {m_rep}: {len(tasks)} tasks")
    out = _out_dir({**base, "m": [m_gate, cfg.n_select_seeds, m_rep]})
    if patch is not None:
        (out / "dirty.patch").write_bytes(patch)
    recs = run_records(
        tasks,
        cfg,
        root_seed=cfg.root_seed,
        git_state=git,
        patch=patch,
        n_jobs=args.n_jobs,
        checkpoint=out / "records.jsonl",
    )
    report = summarize(recs, cfg, git_sha=git.sha, git_dirty=git.dirty)
    write_report(report, out / "report.json")
    _print(report)
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
