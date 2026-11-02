"""Stages 1-3 on one planted-signal dataset, kept as a run directory a memo can be built from.

This is the production path the stage validations ran (ADRs 0010, 0012, 0014): snapshot -> PIT
panel -> ``RunContext`` -> Stage 1 -> ledger -> Stage 2 -> ledger -> Stage 3 -> ledger, with
every result file written. It is used by the memo's validation (docs/adr/0017) and to build test
fixtures. GroundTruth is used only for the target-timing guard every benchmark run passes; it is
not written to the run directory, so a memo cannot read it.

**Real-data-shaped runs** (``real_shaped=True``, docs/adr/0018): the same pipeline, with the
panel's ``syn_*`` columns renamed ``sig_*`` before Stage 1 and a ``signals.json`` of stand-in
definitions written to the run, so the memo takes its real-data path (the real-data
``data_source`` statement, the signals table with definitions and citations) on data whose
answer is known. The stand-in definitions carry digits, so the verifier's check of text facts
is exercised, and they rotate over every source kind, so every variant of the real-data
statement appears. Execution lag stays the benchmark's.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Final

from invariant.benchmark.payload import synthetic_etf_payload, synthetic_feature_payload
from invariant.benchmark.synthetic import SyntheticDataset
from invariant.config.settings import Settings, load_config
from invariant.core.run_hash import GitState, RunContext
from invariant.data.pit import build_pit_panel
from invariant.data.snapshot import write_snapshot
from invariant.orchestration.pipeline import run_kill_stages
from invariant.stages.causal.validation import pipeline_config
from invariant.stages.memo.facts import SignalDoc, SourceKind, write_signal_docs

__all__ = ["STAND_INS", "real_shaped_id", "run_benchmark_pipeline"]

STAND_INS: Final[tuple[tuple[str, tuple[str, ...], tuple[SourceKind, ...]], ...]] = (
    (
        "stand-in for a 12-1 month momentum sum of SPY log returns over sessions t-251 to t-21",
        ("moskowitz2012",),
        ("etf",),
    ),
    (
        "stand-in for the 10-year minus 3-month Treasury spread, usable 5 sessions after its date",
        ("campbell1987", "fama1989"),
        ("fred_market",),
    ),
    (
        "stand-in for a 1-month log change of a monthly series within the vintage usable at t",
        ("fama1990",),
        ("alfred",),
    ),
    (
        "stand-in for (VIX/100)^2/12 minus 21-session realised variance of SPY",
        ("bollerslev2009",),
        ("etf", "fred_market"),
    ),
)
"""Real-data-shaped runs: candidate i gets ``STAND_INS[i % 4]``."""


def real_shaped_id(signal_id: str) -> str:
    """The id benchmark signal ``signal_id`` (a GroundTruth id) carries in a real-data-shaped
    run, as in its ledger and fact sheet."""
    return "sig_" + signal_id


def run_benchmark_pipeline(
    dataset: SyntheticDataset,
    seed: int,
    root: Path,
    *,
    git_state: GitState | None = None,
    patch: bytes | None = None,
    real_shaped: bool = False,
) -> Path:
    """Run Stages 1-3 on ``dataset`` with pipeline seed ``seed`` under ``root``; return the run
    directory (module docstring for ``real_shaped``)."""
    settings = Settings(_env_file=None, data_dir=root / "data", runs_dir=root / "runs")
    config = pipeline_config(load_config(), dataset, seed=seed, n_boot=None)
    dataset.ground_truth.require_target_timing(config.pit.targets)
    snapshot = write_snapshot(
        [synthetic_etf_payload(dataset), synthetic_feature_payload(dataset)],
        settings.data_dir / "snapshots",
    )
    panel = build_pit_panel(snapshot, config)
    if real_shaped:
        renamed = {
            c: "sig_" + c.removeprefix("syn_")
            for c in panel.features.columns
            if c.startswith("syn_")
        }
        panel = dataclasses.replace(panel, features=panel.features.rename(columns=renamed))
        signals = tuple(real_shaped_id(s) for s in config.causal.signals or ())
        config = config.model_copy(
            update={"causal": config.causal.model_copy(update={"signals": signals})}
        )
    ctx = RunContext.create(
        config, snapshot.snapshot_id, settings=settings, git_state=git_state, patch=patch
    )
    try:
        run_kill_stages(panel, config, ctx)
        if real_shaped:
            write_signal_docs(
                ctx.run_dir,
                [
                    SignalDoc(id=s, definition=d, cites=c, sources=k)
                    for i, s in enumerate(config.causal.signals or ())
                    for d, c, k in (STAND_INS[i % len(STAND_INS)],)
                ],
            )
        ctx.mark_completed()
    except BaseException as exc:
        ctx.mark_failed(exc)
        raise
    return ctx.run_dir
