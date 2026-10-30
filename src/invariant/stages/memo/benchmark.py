"""Stages 1-3 on one planted-signal dataset, kept as a run directory a memo can be built from.

This is the production path the stage validations ran (ADRs 0010, 0012, 0014): snapshot -> PIT
panel -> ``RunContext`` -> Stage 1 -> ledger -> Stage 2 -> ledger -> Stage 3 -> ledger, with
every result file written. It is used by the memo's validation (docs/adr/0017) and to build test
fixtures. GroundTruth is used only for the target-timing guard every benchmark run passes; it is
not written to the run directory, so a memo cannot read it.
"""

from __future__ import annotations

from pathlib import Path

from invariant.benchmark.payload import synthetic_etf_payload, synthetic_feature_payload
from invariant.benchmark.synthetic import SyntheticDataset
from invariant.config.settings import Settings, load_config
from invariant.core.run_hash import GitState, RunContext
from invariant.data.pit import build_pit_panel
from invariant.data.snapshot import write_snapshot
from invariant.ledger.kill_ledger import KillLedger
from invariant.stages.causal.stage import (
    record_causal_ledger,
    run_causal_stage,
    write_causal_result,
)
from invariant.stages.causal.validation import pipeline_config
from invariant.stages.decay.stage import record_decay_ledger, run_decay_stage, write_decay_result
from invariant.stages.portfolio.stage import (
    record_portfolio_ledger,
    run_portfolio_stage,
    write_portfolio_result,
)

__all__ = ["run_benchmark_pipeline"]


def run_benchmark_pipeline(
    dataset: SyntheticDataset,
    seed: int,
    root: Path,
    *,
    git_state: GitState | None = None,
    patch: bytes | None = None,
) -> Path:
    """Run Stages 1-3 on ``dataset`` with pipeline seed ``seed`` under ``root``; return the run
    directory."""
    settings = Settings(_env_file=None, data_dir=root / "data", runs_dir=root / "runs")
    config = pipeline_config(load_config(), dataset, seed=seed, n_boot=None)
    dataset.ground_truth.require_target_timing(config.pit.targets)
    snapshot = write_snapshot(
        [synthetic_etf_payload(dataset), synthetic_feature_payload(dataset)],
        settings.data_dir / "snapshots",
    )
    panel = build_pit_panel(snapshot, config)
    ctx = RunContext.create(
        config, snapshot.snapshot_id, settings=settings, git_state=git_state, patch=patch
    )
    try:
        causal = run_causal_stage(panel, config.causal, ctx.rng)
        ledger = KillLedger.open(ctx)
        record_causal_ledger(ledger, causal)
        write_causal_result(ctx.run_dir, causal)
        decay = run_decay_stage(panel, causal, config.decay, ctx.rng)
        record_decay_ledger(ledger, decay)
        write_decay_result(ctx.run_dir, decay)
        portfolio = run_portfolio_stage(panel, decay, config.portfolio, ctx.rng)
        record_portfolio_ledger(ledger, portfolio)
        write_portfolio_result(ctx.run_dir, portfolio)
        ctx.mark_completed()
    except BaseException as exc:
        ctx.mark_failed(exc)
        raise
    return ctx.run_dir
