"""Stage 4 validation study on the planted-signal benchmark (docs/adr/0016).

``--pilot`` runs every task kind on ``n_pilot`` seeds of the pilot root seed and writes
``runs/adversarial_study/<key>/pilot.json``: the S2/S4/S5 thresholds from ADR 0016's rule, with
their pass probabilities, plus per-kind timings and the budget plan. It prints only those. The
pilot's rates set thresholds and are not results.

Without ``--pilot`` it runs the study (the thresholds must be recorded in the config) and writes
``report.json``. Records are checkpointed to ``records.jsonl``, so a rerun with the same key
(config, mode, M, git state) resumes.

    uv run python -m scripts.adversarial_study [--pilot] [--config PATH] [--n-jobs N] [--n-seeds M]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from invariant.config.settings import find_project_root
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import canonical_json, read_git_state
from invariant.stages.adversarial.study import (
    budget_plan,
    load_study_config,
    pilot_thresholds,
    run_study,
    summarize_study,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--n-seeds", type=int, default=None, help="override M (smoke runs)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)

    cfg = load_study_config(args.config)
    if args.pilot:
        root, m, n_rep = (
            cfg.pilot_root_seed,
            args.n_seeds or cfg.n_pilot,
            args.n_seeds or cfg.n_pilot,
        )
    else:
        if cfg.thresholds is None:
            raise SystemExit("record the pilot's thresholds in the config before the full run")
        root, m, n_rep = cfg.root_seed, args.n_seeds or cfg.n_seeds, cfg.n_report_seeds
    git, patch = read_git_state()
    key_src = canonical_json(
        {
            "config": cfg.model_dump(mode="json"),
            "pilot": args.pilot,
            "m": m,
            "git": git.model_dump(),
        }
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    out_dir = find_project_root() / "runs" / "adversarial_study" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    if patch is not None:
        (out_dir / "dirty.patch").write_bytes(patch)
    records = run_study(
        cfg,
        root_seed=root,
        n_seeds=m,
        n_report=n_rep,
        git_state=git,
        patch=patch,
        n_jobs=args.n_jobs,
        checkpoint=out_dir / "records.jsonl",
    )
    if args.pilot:
        thresholds = pilot_thresholds(records, cfg)
        plan = budget_plan(records, cfg, args.n_jobs)
        pilot = {
            "root_seed": root,
            "n_pilot": m,
            "git_sha": git.sha,
            "git_dirty": bool(git.dirty),
            "thresholds": [t.model_dump(mode="json") for t in thresholds],
            "budget": plan,
        }
        (out_dir / "pilot.json").write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
        for t in thresholds:
            print(
                f"{t.gate}: threshold {t.threshold:.2f} (floor {t.floor:.2f}, "
                f"{'feasible' if t.feasible else 'INFEASIBLE'}); P(pass) at the estimate "
                f"{t.pass_probability_at_estimate:.3f}, at the Wilson lower "
                f"{t.pass_probability_at_wilson_lower:.3f}, at the threshold "
                f"{t.pass_probability_at_threshold:.3f}"
            )
        print(f"budget: {plan}\npilot: {out_dir / 'pilot.json'}")
        return 0
    report = summarize_study(records, cfg)
    (out_dir / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    for c in report.criteria:
        print(
            f"{c.name:20s} {c.rate.estimate} [{c.rate.ci_low}, {c.rate.ci_high}]"
            f" (n={c.rate.trials}) bound {c.bound:.3f} {c.side} {c.threshold}"
            f" -> {'PASS' if c.passed else 'FAIL'}"
        )
    print(f"accepted: {report.accepted}\nreported: {report.reported}\nreport: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
