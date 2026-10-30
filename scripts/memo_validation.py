"""Stage 5 memo verifier validation on the planted-signal benchmark (docs/adr/0017).

Runs Stages 1-3 on every preset and seed (with the coverage rule), verifies each run's template
memo and its seeded corruptions, verifies the hand-written sets against the fixtures, and writes
``runs/memo_validation/<key>/report.json``. ``<key>`` hashes the config and the git state;
records are checkpointed to ``records.jsonl``, so a rerun with the same key resumes.

    uv run python -m scripts.memo_validation [--config PATH] [--n-jobs N]
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
from invariant.stages.memo.validation import (
    load_memo_validation_config,
    run_hand_sets,
    run_validation,
    summarize_validation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)

    cfg = load_memo_validation_config(args.config)
    git, patch = read_git_state()
    key_src = canonical_json({"config": cfg.model_dump(mode="json"), "git": git.model_dump()})
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    out_dir = find_project_root() / "runs" / "memo_validation" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    if patch is not None:
        (out_dir / "dirty.patch").write_bytes(patch)
    records = run_validation(
        cfg,
        git_state=git,
        patch=patch,
        n_jobs=args.n_jobs,
        checkpoint=out_dir / "records.jsonl",
        keep_dir=out_dir / "kept_runs",
    )
    hand = run_hand_sets(cfg)
    (out_dir / "hand_sets.jsonl").write_text(
        "".join(h.model_dump_json() + "\n" for h in hand), encoding="utf-8"
    )
    report = summarize_validation(
        records, hand, cfg, git_sha=git.sha or "", git_dirty=bool(git.dirty)
    )
    (out_dir / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    print(f"runs {report.n_runs} {report.runs_per_preset}")
    print(f"categories {report.categories} (coverage met: {report.coverage_met})")
    print(f"recommendations {report.recommendations}")
    for c in report.classes:
        print(f"  {c.cls}: {c.misses}/{c.n} missed (upper {c.miss_rate_upper:.4f}) {c.checks}")
    for s in report.hand_sets:
        print(
            f"  set {s.set}: {len(s.missed)}/{s.n_corruptions} missed {list(s.missed)}; "
            f"benign rejected {len(s.benign_rejected)}/{s.n_benign} {list(s.benign_rejected)}"
        )
    for g in report.gates:
        print(f"  {g.name}: {'PASS' if g.passed else 'FAIL'} ({g.detail})")
    print(f"accepted: {report.accepted}\nreport: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
