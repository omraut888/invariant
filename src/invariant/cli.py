"""Command line: ``invariant snapshot`` and ``invariant run`` (docs/adr/0018).

- ``invariant snapshot --config PATH`` fetches every configured source and writes one immutable
  snapshot (``data.build``), then prints its id. A real-data run's snapshot is fixed in its ADR
  before any stage runs, so fetching is a separate command: re-fetching later gives a different
  snapshot (Yahoo revises history silently).
- ``invariant run --config PATH`` runs the real-data pipeline on the snapshot pinned in the
  config (``snapshot_id``): the signal panel (``data.signals``), Stages 1-3
  (``orchestration.pipeline``), ``signals.json`` (each candidate's frozen definition), and the
  Stage 5 memo, written only if the verifier accepts it. It refuses a dirty git tree unless
  ``--allow-dirty`` (the run would then carry a patch, not a commit).
- ``invariant sensitivity --primary RUN_DIR --sensitivity RUN_DIR`` attaches ADR 0018's
  sensitivity run to the primary run: it checks that the two runs are completed and identical
  but for ``data.fred_market.lag_rule`` (primary ``exact_in_archive``, sensitivity ``max_lag``),
  copies the sensitivity run's manifest and ledger into ``<primary>/sensitivity/``, and rewrites
  the primary memo, which then states whether any signal's outcome depends on the lag rule.

Both commands load ``configs/base.yaml`` with ``PATH`` merged on top.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from invariant.config.settings import PipelineConfig, load_config, load_settings
from invariant.core.logging import LogMode, configure_logging, get_logger
from invariant.core.run_hash import (
    MANIFEST_NAME,
    RunContext,
    RunManifest,
    RunStatus,
    read_git_state,
)
from invariant.data.build import build_snapshot
from invariant.data.signals import REGISTRY, SIGNAL_PREFIX, build_signal_panel
from invariant.data.snapshot import load_snapshot
from invariant.ledger.kill_ledger import HEAD_NAME, LEDGER_NAME
from invariant.orchestration.pipeline import run_kill_stages
from invariant.stages.memo.facts import (
    SENSITIVITY_DIR,
    SignalDoc,
    SourceKind,
    write_signal_docs,
)
from invariant.stages.memo.stage import write_memo

__all__ = ["attach_sensitivity", "main", "signal_docs"]


def signal_docs(config: PipelineConfig) -> list[SignalDoc]:
    """``signals.json`` entries for the configured candidates, from the frozen registry."""
    if config.signals is None:
        raise ValueError("config.signals is not set")
    docs = []
    for sid in config.signals.ids:
        d = REGISTRY[sid]
        sources: list[SourceKind] = []
        if d.etf:
            sources.append("etf")
        if d.market:
            sources.append("fred_market")
        if d.alfred:
            sources.append("alfred")
        docs.append(
            SignalDoc(
                id=SIGNAL_PREFIX + sid,
                definition=d.definition,
                cites=d.cites,
                sources=tuple(sources),
            )
        )
    return docs


def _snapshot(args: argparse.Namespace) -> int:
    config = load_config(override_path=args.config)
    snap = build_snapshot(config, load_settings())
    print(snap.snapshot_id)
    print(snap.path)
    return 0


def _run(args: argparse.Namespace) -> int:
    log = get_logger(__name__)
    config = load_config(override_path=args.config)
    if config.signals is None or config.snapshot_id is None:
        raise SystemExit("a real-data run needs signals and a pinned snapshot_id in its config")
    git, patch = read_git_state()
    if git.dirty and not args.allow_dirty:
        raise SystemExit("the git tree is dirty; commit first, or pass --allow-dirty")
    settings = load_settings()
    snapshot = load_snapshot(config.snapshot_id)
    panel = build_signal_panel(snapshot, config)
    ctx = RunContext.create(
        config, snapshot.snapshot_id, settings=settings, git_state=git, patch=patch
    )
    try:
        run_kill_stages(panel, config, ctx)
        write_signal_docs(ctx.run_dir, signal_docs(config))
        ctx.mark_completed()
    except BaseException as exc:
        ctx.mark_failed(exc)
        raise
    memo = write_memo(ctx.run_dir)
    log.info("real_run_done", run_id=ctx.run_id, recommendation=str(memo.sheet.recommendation))
    print(ctx.run_dir)
    print(memo.md_path)
    return 0


def attach_sensitivity(primary: Path, sensitivity: Path) -> Path:
    """Copy the sensitivity run into ``primary`` after checking it (module docstring); rewrite
    the primary memo. Returns the memo path."""
    a = RunManifest.model_validate_json((primary / MANIFEST_NAME).read_bytes())
    b = RunManifest.model_validate_json((sensitivity / MANIFEST_NAME).read_bytes())
    if a.status is not RunStatus.COMPLETED or b.status is not RunStatus.COMPLETED:
        raise SystemExit("both runs must be completed")
    rules = [_lag_rule(m.inputs.config) for m in (a, b)]
    if rules != ["exact_in_archive", "max_lag"]:
        raise SystemExit(f"lag rules are {rules}, expected exact_in_archive then max_lag")
    if _without_rule(a.inputs.config) != _without_rule(b.inputs.config):
        raise SystemExit("the runs' configs differ beyond the lag rule")
    if (a.inputs.snapshot_id, a.inputs.seed, a.inputs.git) != (
        b.inputs.snapshot_id,
        b.inputs.seed,
        b.inputs.git,
    ):
        raise SystemExit("the runs differ in snapshot, seed or code")
    dest = primary / SENSITIVITY_DIR
    if dest.exists():
        raise SystemExit(f"{dest} exists; a sensitivity run is attached once")
    dest.mkdir()
    for name in (MANIFEST_NAME, LEDGER_NAME, HEAD_NAME):
        shutil.copy2(sensitivity / name, dest / name)
    return write_memo(primary).md_path


def _lag_rule(config: Mapping[str, object]) -> object:
    data = config.get("data")
    market = data.get("fred_market") if isinstance(data, dict) else None
    return market.get("lag_rule") if isinstance(market, dict) else None


def _without_rule(config: Mapping[str, object]) -> object:
    out = copy.deepcopy(dict(config))
    data = out.get("data")
    if isinstance(data, dict) and isinstance(data.get("fred_market"), dict):
        data["fred_market"].pop("lag_rule", None)
    return out


def _sensitivity(args: argparse.Namespace) -> int:
    print(attach_sensitivity(args.primary, args.sensitivity))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="invariant", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="fetch every configured source into one snapshot")
    snap.add_argument("--config", type=Path, required=True)
    run = sub.add_parser("run", help="run Stages 1-3 and the memo on the pinned snapshot")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--allow-dirty", action="store_true")
    sens = sub.add_parser("sensitivity", help="attach the sensitivity run to the primary run")
    sens.add_argument("--primary", type=Path, required=True)
    sens.add_argument("--sensitivity", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV)
    commands = {"snapshot": _snapshot, "run": _run, "sensitivity": _sensitivity}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
