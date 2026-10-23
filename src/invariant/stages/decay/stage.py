r"""Stage 2: is the edge of a signal Stage 1 passed declining? (docs/adr/0011, 0012)

Method
------
1. Inputs: Stage 1's ``CausalResult`` and the same PIT panel. Stage 1's input selection is
   reapplied (``causal.stage.select_inputs``), so the data matrix is the one Stage 1 tested.
   Stage 2 checks that the columns, row count and dates match the ``CausalResult``, and raises
   ``ValueError`` if they do not, because the result would then belong to a different panel.
   Forward targets (``fwd_*``) are never read.
2. For every signal Stage 1 passed, Stage 2 takes its best link: asset, lag and conditioning
   set, all held fixed. It builds the same time-aligned design (``causal.pcmci.lagged_design``)
   and runs the one-sided sup-split decline test (``decline.decline_test``): an unknown
   break, HC0 segment variances, and a fixed-regressor wild bootstrap.
3. Verdict:
   - kill ``general.insufficient_data`` when the design has fewer than ``min_obs`` rows (the
     test was validated only at T = 6000);
   - otherwise kill ``decay.edge_declining`` when ``p < alpha``;
   - otherwise pass.
   There is no correction across signals: a correction would make kills harder as candidates
   are added (ADR 0011).
4. No half-life is reported. No estimator had a calibrated interval in ADR 0011. The break
   fraction at the maximum is descriptive, and is not an onset estimate.

Assumptions, and where they are violated, are in docs/adr/0011. In short:
- the residuals are an MDS given Stage 1's conditioning set;
- only the signal's coefficient changes;
- only the best link is tested.
Power is lowest for a decline that started recently (.22 to .44 at onset .8 T on the
benchmark).

Look-ahead: like Stage 1's, the verdict is a full-sample statement about the sample and is not
used for a trade at date t. Its inputs are point-in-time, and the design runs from a feature
dated ``t - lag`` to the return dated t.

References: Andrews (1993), Econometrica 61(4); Hansen (2000), J. Econometrics 97(1).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Final

import numpy as np

from invariant.config.settings import DecayConfig
from invariant.core.logging import get_logger
from invariant.core.run_hash import RngFactory, canonical_json
from invariant.core.types import KillReason, StageName, Verdict
from invariant.data.pit import PitPanel
from invariant.ledger.kill_ledger import KillLedger, LedgerEntry, MetricValue
from invariant.stages.causal.pcmci import lagged_design
from invariant.stages.causal.result import CausalResult
from invariant.stages.causal.stage import RETURN_PREFIX, select_inputs
from invariant.stages.decay.decline import decline_test
from invariant.stages.decay.result import DecayResult, SignalDecay

__all__ = [
    "DECAY_RESULT_NAME",
    "decide",
    "ledger_metrics",
    "record_decay_ledger",
    "run_decay_stage",
    "write_decay_result",
]

DECAY_RESULT_NAME: Final = "decay_result.json"


def decide(p: float, alpha: float) -> tuple[Verdict, KillReason | None]:
    """Stage 2's verdict from the decline test's p-value. A tie at alpha passes: ``p < alpha``
    is the pre-registered kill rule (ADR 0011)."""
    if p < alpha:
        return Verdict.KILL, KillReason.DECAY_EDGE_DECLINING
    return Verdict.PASS, None


def run_decay_stage(
    panel: PitPanel, causal: CausalResult, config: DecayConfig, rng: RngFactory
) -> DecayResult:
    """Run Stage 2 on ``panel`` for the signals ``causal`` passed (module docstring).

    Deterministic given ``rng``'s seed. Claims the ``decay`` child of ``rng``, with one
    generator per signal id under it.
    """
    log = get_logger(__name__)
    decay_rng = rng.child(StageName.DECAY.value)
    inputs = select_inputs(panel.features, causal.config)
    var_names = tuple(map(str, inputs.frame.columns))
    index = inputs.frame.index
    first = index[0].date() if len(index) else None
    last = index[-1].date() if len(index) else None
    if (var_names, len(index), first, last) != (
        causal.var_names,
        causal.n_obs,
        causal.first_date,
        causal.last_date,
    ):
        raise ValueError("the CausalResult was not produced from this panel")
    data = inputs.frame.to_numpy(dtype=np.float64)
    t0 = time.perf_counter()
    out: list[SignalDecay] = []
    for link in causal.signals:
        ev = link.evidence
        if link.verdict is not Verdict.PASS:
            continue
        assert ev is not None  # a Stage 1 pass always carries evidence
        design = lagged_design(
            data,
            var_names,
            link.column,
            ev.best_asset,
            ev.best_lag,
            ev.best_conditions,
            tau_max=causal.config.tau_max,
        )
        common = {
            "signal_id": link.signal_id,
            "column": link.column,
            "asset": ev.best_asset,
            "lag": ev.best_lag,
            "conditions": ev.best_conditions,
        }
        n_rows = len(design.x)
        if n_rows < config.min_obs:
            log.warning("decay_insufficient_data", signal=link.signal_id, n_rows=n_rows)
            out.append(
                SignalDecay(
                    **common,
                    verdict=Verdict.KILL,
                    reason=KillReason.INSUFFICIENT_DATA,
                    test=None,
                    data_issue=f"{n_rows} design rows < min_obs {config.min_obs}",
                )
            )
            continue
        test = decline_test(
            design.x,
            design.y,
            design.z,
            decay_rng.generator(link.signal_id),
            n_boot=config.n_boot,
            trim=config.trim,
            step=config.break_step,
        )
        verdict, reason = decide(test.p, config.alpha)
        out.append(SignalDecay(**common, verdict=verdict, reason=reason, test=test))
    result = DecayResult(
        config=config,
        causal_config=causal.config,
        n_obs=len(index),
        first_date=first,
        last_date=last,
        signals=tuple(out),
        not_reached=tuple(s.signal_id for s in causal.signals if s.verdict is not Verdict.PASS),
    )
    log.info(
        "decay_stage_done",
        n_signals=len(out),
        n_pass=sum(s.verdict is Verdict.PASS for s in out),
        seconds=round(time.perf_counter() - t0, 3),
    )
    return result


def ledger_metrics(sig: SignalDecay, result: DecayResult) -> dict[str, MetricValue]:
    """Flat, finite, scalar metrics of one signal's Stage 2 decision, for its ledger entry."""
    cfg = result.config
    metrics: dict[str, MetricValue] = {
        "n_obs": result.n_obs,
        "min_obs": cfg.min_obs,
        "alpha": cfg.alpha,
        "n_boot": cfg.n_boot,
        "trim": cfg.trim,
        "break_step": cfg.break_step,
        "test": "sup_split_wild_bootstrap",
        "half_life": "not_reported",
        "asset": sig.asset.removeprefix(RETURN_PREFIX),
        "lag": sig.lag,
        "n_conditions": len(sig.conditions),
    }
    t = sig.test
    if t is None:
        metrics["data_issue"] = sig.data_issue or "not tested"
        return metrics
    metrics |= {
        "p": t.p,
        "statistic": t.statistic,
        "n_rows": t.n_rows,
        "break_row": t.break_row,
        "break_fraction": t.break_fraction,
        "b_full": t.b_full,
        "b_pre": t.b_pre,
        "b_post": t.b_post,
        "se_pre": t.se_pre,
        "se_post": t.se_post,
        "trend_t": t.trend_t,
        "trend_p": t.trend_p,
    }
    return metrics


def record_decay_ledger(ledger: KillLedger, result: DecayResult) -> list[LedgerEntry]:
    """One Stage 2 ledger entry per signal that reached Stage 2, pass or kill."""
    return [
        ledger.record(
            sig.signal_id,
            StageName.DECAY,
            sig.verdict,
            reason=sig.reason,
            metrics=ledger_metrics(sig, result),
        )
        for sig in result.signals
    ]


def write_decay_result(run_dir: Path, result: DecayResult) -> Path:
    """Write ``result`` as canonical JSON to ``<run_dir>/decay_result.json``."""
    path = run_dir / DECAY_RESULT_NAME
    path.write_text(canonical_json(result.model_dump(mode="json")) + "\n", encoding="utf-8")
    return path
