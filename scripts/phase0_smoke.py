"""Phase 0 smoke run: every foundation component, end to end, on the planted-signal benchmark.

1. Generate a synthetic dataset and its GroundTruth.
2. Write it as an immutable snapshot.
3. Create (or resume) a RunContext keyed by that snapshot and the benchmark target timing.
4. Build the point-in-time panel from the snapshot and check its returns against the dataset.
5. Write *placeholder* ledger entries: a deliberately mixed set covering a correct pass, a
   correct kill at the right stage, a correct kill at the wrong stage and a missed kill. Every
   entry has ``placeholder=True``. No stage logic exists yet.
6. Score the ledger against GroundTruth and require the known result (exit 1 otherwise).
7. Mark the run completed and print its manifest and the score.

Usage: uv run python scripts/phase0_smoke.py [--preset NAME] [--n-obs N]
                                             [--data-dir DIR] [--runs-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from invariant.benchmark.config import (
    SyntheticMarketConfig,
    load_benchmark_config,
    load_benchmark_target_timing,
)
from invariant.benchmark.payload import synthetic_etf_payload, synthetic_tickers
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import PipelineConfig, Settings, load_config, load_settings
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import RunContext
from invariant.core.types import KillReason, StageName, Verdict
from invariant.data.pit import build_pit_panel
from invariant.data.snapshot import write_snapshot
from invariant.ledger.benchmark_score import ConfusionMatrix, score_run
from invariant.ledger.kill_ledger import STAGE_ORDER, KillLedger

# Placeholder fates: signal -> stage it is killed at, or None to pass every stage.
PLACEHOLDER_FATES: dict[str, StageName | None] = {
    "s_real": None,  # expected pass     -> correct pass
    "s_conf": StageName.CAUSAL,  # expected causal   -> correct kill, right stage
    "s_decay": StageName.CAUSAL,  # expected decay    -> correct kill, wrong stage
    "s_noise": None,  # expected causal   -> missed kill
}
# The one expected-pass signal passes here, so wrong_kill is 0 by construction; unit tests
# cover it.
EXPECTED_MATRIX = ConfusionMatrix(
    correct_kill=2, wrong_kill=0, correct_pass=1, missed_kill=1, kills_at_expected_stage=1
)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default=None, help="benchmark preset (default: base)")
    parser.add_argument("--n-obs", type=int, default=None, help="override n_obs (speed)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    update: dict[str, Path] = {}
    if args.data_dir is not None:
        update["data_dir"] = args.data_dir.resolve()
    if args.runs_dir is not None:
        update["runs_dir"] = args.runs_dir.resolve()
    return settings.model_copy(update=update)


def _dataset(args: argparse.Namespace, seed: int) -> SyntheticDataset:
    cfg = load_benchmark_config(args.preset)
    if args.n_obs is not None:
        # Re-validate rather than model_copy, which would skip the config's validators.
        cfg = SyntheticMarketConfig.model_validate({**cfg.model_dump(), "n_obs": args.n_obs})
    return generate_market(cfg, seed)


def _pipeline_config(base: PipelineConfig, dataset: SyntheticDataset) -> PipelineConfig:
    truth = dataset.ground_truth
    return base.model_copy(
        update={
            "universe": base.universe.model_copy(
                update={"tickers": synthetic_tickers(truth.asset_names)}
            ),
            "dates": base.dates.model_copy(
                update={"start": truth.first_date, "end": truth.last_date}
            ),
            "pit": base.pit.model_copy(update={"targets": load_benchmark_target_timing()}),
        }
    )


def _write_placeholders(ledger: KillLedger) -> None:
    for signal_id, kill_stage in PLACEHOLDER_FATES.items():
        for stage in STAGE_ORDER:
            if stage is kill_stage:
                ledger.record(
                    signal_id,
                    stage,
                    Verdict.KILL,
                    reason=KillReason.PLACEHOLDER,
                    metrics={"placeholder": True},
                    placeholder=True,
                )
                break
            ledger.record(
                signal_id, stage, Verdict.PASS, metrics={"placeholder": True}, placeholder=True
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    configure_logging(LogMode.DEV)
    settings = _settings(args)
    base = load_config()

    dataset = _dataset(args, base.seed)
    truth = dataset.ground_truth
    snapshot = write_snapshot([synthetic_etf_payload(dataset)], settings.data_dir / "snapshots")
    config = _pipeline_config(base, dataset)
    ctx = RunContext.create(config, snapshot.snapshot_id, settings=settings)
    try:
        panel = build_pit_panel(snapshot, config)
        tickers = synthetic_tickers(truth.asset_names)
        got = panel.features[[f"ret_{t}" for t in tickers]].to_numpy()[1:]
        want = dataset.returns.to_numpy()[1:]
        if not np.allclose(got, want, rtol=0.0, atol=1e-12):
            raise RuntimeError("PIT panel returns do not reproduce the synthetic returns")

        _write_placeholders(KillLedger.open(ctx))
        score = score_run(ctx.run_dir, truth, snapshot)
        if score.matrix != EXPECTED_MATRIX or not score.complete:
            raise RuntimeError(f"unexpected score {score.matrix} (complete={score.complete})")
        if not score.contains_placeholders:
            raise RuntimeError("placeholder entries were not detected by the scorer")
        ctx.mark_completed()
    except BaseException as exc:
        ctx.mark_failed(exc)
        raise

    print(json.dumps(ctx.manifest().model_dump(mode="json"), indent=2, sort_keys=True))
    print(f"\nsnapshot {snapshot.snapshot_id}  panel {panel.features.shape}  run {ctx.run_dir}")
    print(f"score (placeholders={score.contains_placeholders}): {score.matrix.model_dump()}")
    for row in score.signals:
        print(
            f"  {row.signal_id:8s} {row.outcome!s:13s} expected={row.expected_stage} "
            f"observed={row.observed_stage} stage_match={row.stage_match}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
