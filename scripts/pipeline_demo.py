"""Stages 1-3 end to end on one planted-signal market: what each signal's fate was, and why.

1. Generate ADR 0013's selection market (``d1_base``: ``multi_asset``, T = 6000) on ``--seed``,
   with its GroundTruth.
2. Write it as an immutable snapshot and build the point-in-time panel.
3. Create (or resume) a run keyed by the snapshot, the config, the git state and the seed.
4. Stage 1 (causal) -> ledger -> Stage 2 (decay) -> ledger -> Stage 3 (portfolio) -> ledger,
   the same production path the validations ran (ADRs 0010, 0012, 0014).
5. Score the ledger against GroundTruth over (causal, decay, portfolio); this verifies the
   ledger's hash chain.
6. Print, per signal, the truth, the stage that killed it (or that it passed every stage), and
   the numbers behind each decision, read back from the ledger.

One seed is an illustration, not evidence: the rates are in docs/results/. The run directory
(manifest, ledger, stage results) is kept under the runs dir for inspection.

Usage: uv run python scripts/pipeline_demo.py [--seed N] [--data-dir DIR] [--runs-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from invariant.benchmark.payload import synthetic_etf_payload, synthetic_feature_payload
from invariant.benchmark.synthetic import generate_market
from invariant.config.settings import Settings, load_config, load_settings
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import RunContext
from invariant.core.types import StageName, Verdict
from invariant.data.pit import build_pit_panel
from invariant.data.snapshot import write_snapshot
from invariant.ledger.benchmark_score import score_run
from invariant.ledger.kill_ledger import KillLedger, LedgerEntry
from invariant.stages.causal.stage import (
    record_causal_ledger,
    run_causal_stage,
    write_causal_result,
)
from invariant.stages.causal.validation import pipeline_config
from invariant.stages.decay.stage import record_decay_ledger, run_decay_stage, write_decay_result
from invariant.stages.dro.study import dro_scenarios
from invariant.stages.portfolio.stage import (
    record_portfolio_ledger,
    run_portfolio_stage,
    write_portfolio_result,
)

SCENARIO = "d1_base"
STAGES = (StageName.CAUSAL, StageName.DECAY, StageName.PORTFOLIO)


def _f(value: object) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def evidence(entry: LedgerEntry) -> str:
    """The numbers behind one ledger decision, as one line."""
    m = entry.metrics
    if "data_issue" in m:
        return f"data issue: {m['data_issue']}"
    if entry.stage is StageName.CAUSAL:
        return (
            f"BH q {_f(m['q'])} vs alpha {_f(m['alpha'])}; partial r {_f(m['r_best'])} "
            f"[{_f(m['r_best_ci_low'])}, {_f(m['r_best_ci_high'])}] at lag {m['best_lag']}"
        )
    if entry.stage is StageName.DECAY:
        return (
            f"decline p {_f(m['p'])} vs alpha {_f(m['alpha'])}; coefficient before/after the "
            f"break {_f(m['b_pre'])} / {_f(m['b_post'])} (break at {_f(m['break_fraction'])} T)"
        )
    line = (
        f"Ridge-CV Sharpe {_f(m['ridge_cv_sharpe'])} (net {_f(m['ridge_cv_sharpe_net'])}) vs "
        f"1/N {_f(m['equal_weight_sharpe'])}; LW p {_f(m['ridge_cv_lw_p'])} vs "
        f"{_f(m['lw_alpha'])}; {m['n_members']} member(s)"
    )
    if "b_marginal_sharpe" in m:
        line += (
            f"; B {_f(m['b_marginal_sharpe'])} [{_f(m['b_ci_low'])}, {_f(m['b_ci_high'])}] "
            f"vs {m['b_baseline']}"
        )
    return line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="dataset and pipeline seed")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV, level="WARNING")

    settings = load_settings()
    if args.data_dir is not None or args.runs_dir is not None:
        settings = Settings(
            _env_file=None,
            data_dir=args.data_dir or settings.data_dir,
            runs_dir=args.runs_dir or settings.runs_dir,
        )
    t0 = time.perf_counter()
    scenario = next(s for s in dro_scenarios() if s.name == SCENARIO)
    dataset = generate_market(scenario.config(None), args.seed)
    truth = dataset.ground_truth
    config = pipeline_config(load_config(), dataset, seed=args.seed, n_boot=None)
    truth.require_target_timing(config.pit.targets)
    snapshot = write_snapshot(
        [synthetic_etf_payload(dataset), synthetic_feature_payload(dataset)],
        settings.data_dir / "snapshots",
    )
    panel = build_pit_panel(snapshot, config)
    ctx = RunContext.create(config, snapshot.snapshot_id, settings=settings)
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
    score = score_run(ctx.run_dir, truth, snapshot, stages=STAGES)
    entries = KillLedger.read(ctx.run_dir).entries()

    print(f"\n{SCENARIO}, seed {args.seed}: {panel.features.shape[0]} panel rows, run {ctx.run_id}")
    print(f"run dir: {ctx.run_dir}\n")
    for sig in truth.signals:
        expected = (
            "should pass every stage"
            if sig.expected_verdict is Verdict.PASS
            else f"should be killed at {sig.expected_kill_stage}"
        )
        mine = [e for e in entries if e.signal_id == sig.signal_id]
        killed = next((e for e in mine if e.verdict is Verdict.KILL), None)
        fate = (
            f"KILLED at {killed.stage} ({killed.reason})"
            if killed is not None
            else f"passed {', '.join(str(e.stage) for e in mine)}"
        )
        print(f"{sig.signal_id}: {expected}; {fate}")
        for e in mine:
            print(f"  {e.stage!s:9s} {e.verdict!s:4s}  {evidence(e)}")
    m = score.matrix
    print(
        f"\nscore over {', '.join(str(s) for s in STAGES)}: correct kills {m.correct_kill}, "
        f"wrong kills {m.wrong_kill}, correct passes {m.correct_pass}, missed kills "
        f"{m.missed_kill}; ledger chain verified; complete {score.complete}"
    )
    print(f"{time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
