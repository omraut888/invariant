"""Stage 2 pre-registered study (docs/adr/0011): kill-test size and power, and half-life
estimators.

Writes ``runs/decay_study/<key>/report.json`` (``<key>`` hashes the study config, M, M_ksg
and the git state) and prints the tables. Records are checkpointed to ``records.jsonl``, so a
rerun with the same key resumes.

    uv run python -m scripts.decay_study [--config PATH] [--n-jobs N] [--smoke]

``--smoke`` runs M = 8 on every arm and applies the KSG compute-budget rule. It prints the
``ksg_n_seeds`` to record in the config and ADR 0011 before the full run. Its rates are not
used for anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from invariant.config.settings import find_project_root
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import canonical_json, read_git_state
from invariant.ledger.benchmark_score import RateEstimate
from invariant.stages.causal.calibration import required_seeds
from invariant.stages.decay.halflife import Estimator
from invariant.stages.decay.study import (
    Arm,
    DecayStudyReport,
    Scenario,
    decay_scenarios,
    ksg_budget,
    load_decay_study_config,
    run_records,
    summarize,
    write_report,
)

SMOKE_SEEDS = 8


def _r(rate: RateEstimate | None) -> str:
    if rate is None or rate.estimate is None:
        return "-"
    return f"{rate.estimate:.3f} [{rate.ci_low:.3f},{rate.ci_high:.3f}]"


def _hl(lam: float | None) -> str:
    if lam is None:
        return "-"
    return "inf" if lam == 0.0 else f"{math.log(2.0) / lam:.0f}"


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:+.5f}"


def _print(report: DecayStudyReport) -> None:
    print("\nKill test (p < alpha). * = gate. rate | given Stage 1 pass | trend diagnostic")
    for r in report.rejections:
        mark = "*" if r.gate else " "
        print(
            f"{r.scenario:18s} {r.signal:8s}{mark} {_r(r.rate):26s} "
            f"{_r(r.rate_given_stage1_pass):26s} {_r(r.trend_rate)}"
        )
    print(f"kill-test gates failed: {list(report.kill_failed_gates)}")
    print("\nEstimators: claims-decay | coverage | bias / sd / rmse (lambda) | median HL | true HL")
    for e in report.estimators:
        print(
            f"{e.scenario:14s} {e.signal:8s} {e.estimator.value:13s} n={e.n:5d} "
            f"{_r(e.claims_decay):26s} {_r(e.coverage):26s} "
            f"{_fmt(e.lam_bias)} {_fmt(e.lam_sd)} {_fmt(e.lam_rmse)} "
            f"{_hl(e.median_lam_hat):>6s} {_hl(e.lam_true):>6s}"
        )
    print(f"\nadmissible: {report.admissible}\nfailed: {report.estimator_failed_gates}")
    for c in report.rmse:
        print(
            f"rmse {c.estimator.value:13s} {c.rmse:.5f}  minus best {c.diff_from_best:+.5f} "
            f"[{c.diff_ci_low:+.5f},{c.diff_ci_high:+.5f}]"
        )
    print(f"KSG on {report.ksg_n_seeds}/{report.n_seeds} seeds; eligible: {report.ksg_eligible}")
    print(f"selected: {report.selected} ({report.selection_reason})")
    print(f"median seconds per task: {report.median_seconds}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="M = 8 and the KSG budget rule")
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)

    cfg = load_decay_study_config(args.config)
    m_req = required_seeds(
        cfg.decline.alpha, cfg.size_cutoff, cfg.n_gating, cfg.all_pass_prob, cfg.confidence
    )
    m = SMOKE_SEEDS if args.smoke else (cfg.n_seeds or m_req)
    m_ksg = SMOKE_SEEDS if args.smoke else min(cfg.ksg_n_seeds or m, m)
    print(f"gates: {cfg.n_gating}; required M: {m_req}; M = {m}; M_ksg = {m_ksg}", flush=True)

    scenarios: tuple[Scenario, ...] = decay_scenarios()
    tasks: list[tuple[Scenario, int, Arm]] = [(s, i, "main") for s in scenarios for i in range(m)]
    tasks += [
        (s, i, "ksg") for s in scenarios if Estimator.KSG in s.estimators for i in range(m_ksg)
    ]
    git, patch = read_git_state()
    key_src = canonical_json(
        {
            "config": cfg.model_dump(mode="json"),
            "m": m,
            "m_ksg": m_ksg,
            "smoke": args.smoke,
            "git": git.model_dump(),
        }
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    out_dir = find_project_root() / "runs" / "decay_study" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    if patch is not None:
        (out_dir / "dirty.patch").write_bytes(patch)
    records = run_records(tasks, cfg, n_jobs=args.n_jobs, checkpoint=out_dir / "records.jsonl")
    report = summarize(records, scenarios, cfg, git_sha=git.sha, git_dirty=git.dirty)
    write_report(report, out_dir / "report.json")
    _print(report)
    if args.smoke:
        budget = ksg_budget(
            records,
            budget_hours=cfg.ksg_budget_hours,
            workers=args.n_jobs,
            required_seeds=cfg.n_seeds or m_req,
        )
        (out_dir / "ksg_budget.json").write_text(
            json.dumps(budget.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nKSG budget rule: {budget.model_dump()}")
        print(
            f"record in config and ADR 0011: ksg_n_seeds: {budget.ksg_n_seeds}"
            + ("" if not budget.full else " (full: leave ksg_n_seeds null)")
        )
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
