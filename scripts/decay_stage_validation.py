"""Stage 2 validation on the planted-signal benchmark (docs/adr/0012).

Runs Stage 1 and Stage 2 through the production path on M seeds of each preset, then writes
``runs/decay_stage_validation/<key>/report.json`` and prints the results. ``<key>`` hashes the
validation config, M and the git state. Records are checkpointed to ``records.jsonl``.

    uv run python -m scripts.decay_stage_validation [--config PATH] [--n-jobs N] [--n-seeds M]
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
from invariant.stages.decay.validation import (
    load_decay_validation_config,
    run_validation,
    summarize_validation,
)


def _r(rate: RateEstimate) -> str:
    if rate.estimate is None:
        return "-"
    return f"{rate.estimate:.3f} [{rate.ci_low:.3f},{rate.ci_high:.3f}] (n={rate.trials})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--n-seeds", type=int, default=None, help="override M (smoke runs)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)

    cfg = load_decay_validation_config(args.config)
    m = args.n_seeds or cfg.n_seeds
    git, patch = read_git_state()
    key_src = canonical_json(
        {"config": cfg.model_dump(mode="json"), "m": m, "git": git.model_dump()}
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    out_dir = find_project_root() / "runs" / "decay_stage_validation" / key
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
        alpha=load_config().decay.alpha,
        git_sha=git.sha or "",
        git_dirty=bool(git.dirty),
    )
    (out_dir / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    for p in report.presets:
        agg = p.aggregate
        print(
            f"\n{p.preset}: runs {p.n_runs}, incomplete {p.n_incomplete}, "
            f"median {p.median_seconds:.1f}s; precision {_r(agg.kill_precision)}, "
            f"recall {_r(agg.kill_recall)}"
        )
        for s in p.signals:
            print(f"  {s.signal_id:8s} S1 pass {_r(s.stage1_pass_rate)}")
            print(f"  {'':8s} S2 kill | S1 pass {_r(s.stage2_kill_given_stage1_pass)} {s.reasons}")
            print(f"  {'':8s} trend   | S1 pass {_r(s.trend_kill_given_stage1_pass)}")
            print(f"  {'':8s} passes both {_r(s.pipeline_pass_rate)}")
    print("\nAcceptance:")
    for c in report.criteria:
        print(
            f"  {c.name:18s} {c.signal_id or '':8s} bound {c.bound} threshold {c.threshold} "
            f"-> {'PASS' if c.passed else 'FAIL'}"
        )
    print(f"accepted: {report.accepted}\nreport: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
