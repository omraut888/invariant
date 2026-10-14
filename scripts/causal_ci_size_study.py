"""Stage 1 CI-test size and power study (E1-E5, docs/adr/0009).

Runs PCMCI with every candidate CI test on M seeds of each scenario, then writes
``runs/causal_ci_study/<key>/report.json`` and prints the results table. ``<key>`` hashes the
study config, M and the git state. Records are checkpointed to ``records.jsonl`` in the same
directory, so a rerun with the same key resumes.

Run from the repo root as a module, so the test-only E4 DGP (tests/fixtures/vol_null.py) is
importable:

    uv run python -m scripts.causal_ci_size_study [--n-jobs N] [--n-seeds M] [--n-obs T]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from tests.fixtures.vol_null import vol_null_scenario

from invariant.config.settings import find_project_root
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import canonical_json, read_git_state
from invariant.stages.causal.calibration import (
    FamilyRole,
    StudyReport,
    benchmark_scenarios,
    load_size_study_config,
    required_seeds,
    run_scenarios,
    summarize,
    write_report,
)


def _table(report: StudyReport) -> str:
    labels = [t.label for t in report.config.tests]
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for f in report.families:
        r = f.rate
        assert r.estimate is not None and r.ci_low is not None and r.ci_high is not None
        cell = f"{r.estimate:.3f} [{r.ci_low:.3f},{r.ci_high:.3f}]"
        if f.ks_uniform_p is not None:
            cell += f" ks={f.ks_uniform_p:.2f}"
        rows.setdefault((f.scenario, f.family, f.role.value), {})[f.test] = cell
    width = 34
    lines = [
        f"{'scenario':14s} {'family':15s} {'role':11s} "
        + " ".join(f"{lab:{width}s}" for lab in labels)
    ]
    for (sc, fam, role), cells in rows.items():
        mark = "*" if role == FamilyRole.GATING_NULL.value else " "
        lines.append(
            f"{sc:14s} {fam:15s} {role:11s}{mark}"
            + " ".join(f"{cells[lab]:{width}s}" for lab in labels)
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--n-seeds", type=int, default=None, help="override M (smoke runs)")
    parser.add_argument("--n-obs", type=int, default=None, help="override T (smoke runs)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)

    cfg = load_size_study_config(args.config)
    if args.n_obs is not None:
        cfg = cfg.model_copy(update={"n_obs": args.n_obs})
    scenarios = (*benchmark_scenarios(cfg.tau_max), vol_null_scenario(cfg.tau_max))
    n_gating = sum(f.role is FamilyRole.GATING_NULL for s in scenarios for f in s.families)
    m_req = required_seeds(cfg.alpha, cfg.size_cutoff, n_gating, cfg.all_pass_prob, cfg.confidence)
    m = args.n_seeds or cfg.n_seeds or m_req
    print(f"gating families: {n_gating}; required M: {m_req}; running M = {m}", flush=True)

    git, patch = read_git_state()
    key_src = canonical_json(
        {"config": cfg.model_dump(mode="json"), "m": m, "git": git.model_dump()}
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    out_dir = find_project_root() / "runs" / "causal_ci_study" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    if patch is not None:  # uncommitted edits: keep them, as RunContext does
        (out_dir / "dirty.patch").write_bytes(patch)
    records = run_scenarios(
        scenarios, cfg, m, n_jobs=args.n_jobs, checkpoint=out_dir / "records.jsonl"
    )
    report = summarize(
        records,
        scenarios,
        cfg,
        tigramite_version=importlib.metadata.version("tigramite"),
        git_sha=git.sha,
        git_dirty=git.dirty,
    )
    write_report(report, out_dir / "report.json")
    print(_table(report))
    print(f"\nadmissible: {report.admissible}\nfailed gates: {report.failed_gates}")
    for pc in report.power:
        print(
            f"power {pc.test:8s} {pc.power:.3f}  best-minus-this {pc.diff_from_best:+.3f} "
            f"[{pc.diff_ci_low:+.3f},{pc.diff_ci_high:+.3f}]"
        )
    print(f"median seconds/run: {report.median_seconds}")
    print(f"selected: {report.selected} ({report.selection_reason})")
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
