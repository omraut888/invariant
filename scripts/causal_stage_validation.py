"""Stage 1 Monte Carlo validation on the planted-signal benchmark (docs/adr/0010).

Runs every preset of configs/causal_stage_validation.yaml on M replications through the
production path (snapshot -> PIT panel -> Stage 1 -> kill ledger -> score_run), writes
records.jsonl and report.json under runs/causal_stage_validation/<key>/, and prints the
pre-registered acceptance decision. The key hashes (config, M, git state); a dirty tree's patch
is saved next to the report. An interrupted run resumes from records.jsonl.

Usage: uv run python -m scripts.causal_stage_validation [--config PATH] [--n-jobs N]
                                                         [--n-seeds M]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from invariant.config.settings import find_project_root, load_config
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import canonical_json, read_git_state
from invariant.stages.causal.validation import (
    ValidationReport,
    load_validation_config,
    run_validation,
    summarize_validation,
)


def _fmt(rate: object) -> str:
    est, lo, hi = (getattr(rate, k) for k in ("estimate", "ci_low", "ci_high"))
    return "n/a" if est is None else f"{est:.3f} [{lo:.3f},{hi:.3f}]"


def _print(report: ValidationReport) -> None:
    for p in report.presets:
        agg = p.aggregate
        print(
            f"\n== {p.preset} (M = {p.n_runs}, incomplete {p.n_incomplete}, "
            f"median {p.median_seconds:.1f} s/run)"
        )
        print(f"kill precision {_fmt(agg.kill_precision)}  recall {_fmt(agg.kill_recall)}")
        for s in p.signals:
            t_real = "n/a" if s.realized_t_lag1_mean is None else f"{s.realized_t_lag1_mean:.2f}"
            cov = "n/a" if s.bootstrap_coverage is None else _fmt(s.bootstrap_coverage)
            pred = "n/a" if s.predicted_power is None else f"{s.predicted_power:.3f}"
            print(
                f"  {s.signal_id:8s} expect {s.expected_verdict.value:4s} "
                f"pass {_fmt(s.pass_rate)}  mci-family {_fmt(s.mci_family_rate)}  "
                f"t_hyb exp {s.expected_t_hybrid_mean:.2f} real {t_real}  "
                f"pred power {pred}  boot cov {cov}  low_power {s.low_power_runs}"
            )
            print(f"           reasons {s.reasons}")
    print("\nacceptance:")
    for c in report.criteria:
        print(
            f"  {c.name:16s} {c.signal_id or '-':8s} bound {c.bound} threshold {c.threshold} "
            f"{'PASS' if c.passed else 'FAIL'}"
        )
    print(f"accepted: {report.accepted}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--n-seeds", type=int, default=None, help="override M (smoke runs)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)

    cfg = load_validation_config(args.config)
    m = args.n_seeds or cfg.n_seeds
    git, patch = read_git_state()
    key_src = canonical_json(
        {"config": cfg.model_dump(mode="json"), "m": m, "git": git.model_dump()}
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    out_dir = find_project_root() / "runs" / "causal_stage_validation" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    if patch is not None:
        (out_dir / "dirty.patch").write_bytes(patch)
    print(f"running M = {m} on {list(cfg.presets)} -> {out_dir}", flush=True)
    records = run_validation(
        cfg,
        git_state=git,
        patch=patch,
        n_seeds=m,
        n_jobs=args.n_jobs,
        checkpoint=out_dir / "records.jsonl",
    )
    causal = load_config().causal
    report = summarize_validation(
        records,
        cfg,
        alpha=causal.alpha,
        tau_max=causal.tau_max,
        git_sha=git.sha,
        git_dirty=git.dirty,
        tigramite_version=importlib.metadata.version("tigramite"),
    )
    text = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    (out_dir / "report.json").write_text(text, encoding="utf-8")
    _print(report)
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
