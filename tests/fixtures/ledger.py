"""Helpers for kill-ledger tests: a RunContext in a temp dir and fate writers. Offline only."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from invariant.config.settings import PipelineConfig, Settings, load_config
from invariant.core.run_hash import GitState, RunContext
from invariant.core.types import KillReason, StageName, Verdict
from invariant.ledger.kill_ledger import STAGE_ORDER, KillLedger

CLEAN_GIT = GitState(sha="a" * 40, dirty=False)


def temp_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")


def run_context(
    tmp_path: Path, *, config: PipelineConfig | None = None, snapshot_id: str = "snap-1"
) -> RunContext:
    return RunContext.create(
        config if config is not None else load_config(),
        snapshot_id,
        settings=temp_settings(tmp_path),
        git_state=CLEAN_GIT,
    )


def write_fates(
    ledger: KillLedger,
    fates: Mapping[str, StageName | None],
    *,
    stop_before: Mapping[str, StageName] | None = None,
    placeholder: bool = False,
) -> None:
    """Record each signal's decisions: PASS until its kill stage (``None``: pass all five).

    ``stop_before[sid]`` leaves ``sid`` undecided from that stage on.
    """
    stops = dict(stop_before or {})
    reason = KillReason.PLACEHOLDER if placeholder else KillReason.INSUFFICIENT_DATA
    for sid, kill_stage in fates.items():
        for stage in STAGE_ORDER:
            if stops.get(sid) is stage:
                break
            if stage is kill_stage:
                ledger.record(
                    sid,
                    stage,
                    Verdict.KILL,
                    reason=reason,
                    metrics={"stat": 1.5, "p_value": 0.2},
                    placeholder=placeholder,
                )
                break
            ledger.record(sid, stage, Verdict.PASS, metrics={"stat": 4.0}, placeholder=placeholder)
