"""The kill stages in order, on one PIT panel, inside one run (the production path).

Stage 1 -> ledger -> Stage 2 -> ledger -> Stage 3 -> ledger, every result file written to the
run directory. The benchmark runner (``stages.memo.benchmark``) and the real-data run
(``invariant.cli``) both call ``run_kill_stages``, so the path the stages were validated on is
the path real data takes.
"""

from __future__ import annotations

from invariant.config.settings import PipelineConfig
from invariant.core.run_hash import RunContext
from invariant.data.pit import PitPanel
from invariant.ledger.kill_ledger import KillLedger
from invariant.stages.causal.stage import (
    record_causal_ledger,
    run_causal_stage,
    write_causal_result,
)
from invariant.stages.decay.stage import record_decay_ledger, run_decay_stage, write_decay_result
from invariant.stages.portfolio.stage import (
    record_portfolio_ledger,
    run_portfolio_stage,
    write_portfolio_result,
)

__all__ = ["run_kill_stages"]


def run_kill_stages(panel: PitPanel, config: PipelineConfig, ctx: RunContext) -> None:
    """Stages 1-3 on ``panel``; ledger entries and result files go to ``ctx``'s run."""
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
