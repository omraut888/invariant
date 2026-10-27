"""Stage 3 validation on the planted-signal benchmark (docs/adr/0014).

Runs the size tasks (null scenarios, members forced) and the pipeline tasks (Stages 1-3 through
the production path) on M seeds, then writes ``runs/portfolio_stage_validation/<key>/report.json``
and prints the results. ``<key>`` hashes the validation config, M and the git state. Records are
checkpointed to ``records.jsonl``, so a rerun with the same key resumes.

    uv run python -m scripts.portfolio_stage_validation [--config PATH] [--n-jobs N] [--n-seeds M]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from invariant.config.settings import find_project_root, load_config
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import canonical_json, read_git_state
from invariant.ledger.benchmark_score import RateEstimate
from invariant.stages.portfolio.validation import (
    load_portfolio_validation_config,
    run_validation,
    summarize_validation,
)


def _r(rate: RateEstimate | None) -> str:
    if rate is None or rate.estimate is None:
        return "-"
    return f"{rate.estimate:.3f} [{rate.ci_low:.3f},{rate.ci_high:.3f}] (n={rate.trials})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--n-seeds", type=int, default=None, help="override M (smoke runs)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)

    cfg = load_portfolio_validation_config(args.config)
    m = args.n_seeds or cfg.n_seeds
    git, patch = read_git_state()
    key_src = canonical_json(
        {"config": cfg.model_dump(mode="json"), "m": m, "git": git.model_dump()}
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    out_dir = find_project_root() / "runs" / "portfolio_stage_validation" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    if patch is not None:
        (out_dir / "dirty.patch").write_bytes(patch)
    records = run_validation(
        cfg,
        git_state=git,
        patch=patch,
        n_seeds=m,
        n_jobs=args.n_jobs,
        checkpoint=out_dir / "records.jsonl",
    )
    report = summarize_validation(
        records,
        cfg,
        n_seeds=m,
        lw_alpha=load_config().portfolio.lw_alpha,
        git_sha=git.sha or "",
        git_dirty=bool(git.dirty),
    )
    (out_dir / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    for s in report.scenarios:
        print(
            f"\n{s.kind} {s.scenario}: runs {s.n_runs}, built {s.n_ran}, "
            f"stage pass {_r(s.stage_pass)}, median {s.median_seconds:.1f}s"
        )
        for a in s.arms:
            print(
                f"  {a.arm.value:12s}{'*' if a.decides else ' '} SR {a.sharpe:+.3f} "
                f"net {a.sharpe_net:+.3f} inv {a.invested:.2f} cash {a.cash:.2f} "
                f"turn {a.turnover:.3f} beats 1/N {_r(a.beats_equal_weight)}"
            )
        for b in s.marginals:
            print(
                f"  B {b.signal_id:8s} mean {b.mean_marginal_sharpe:+.3f} "
                f"CI excludes 0 {_r(b.ci_excludes_zero)}"
            )
        for g in s.signals:
            print(
                f"  {g.signal_id:8s} reached S3 {_r(g.reached_stage3)}; "
                f"S3 pass | reached {_r(g.stage3_pass_given_reached)}; all {_r(g.passes_all)}"
            )
        print(f"  member sets: {s.member_sets}")
    print("\nAcceptance:")
    for c in report.criteria:
        print(
            f"  {c.name:12s} {c.scope:14s} {_r(c.rate)} bound {c.bound} threshold "
            f"{c.threshold} -> {'PASS' if c.passed else 'FAIL'}"
        )
    print(f"accepted: {report.accepted}\nreport: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
