r"""Stage 1: is a candidate signal structurally upstream of returns? (docs/adr/0010)

Method
------
1. Inputs: the PIT panel's feature columns (``syn_*``, ``ff_*``, ``macro_*``) and its returns
   (``ret_*``). Forward targets (``fwd_*``) are never read: they hold future returns.
2. One PCMCI run (Runge et al. 2019) over every feature and every return, with tigramite's
   ParCorr statistic and White's HC3 p-value (``ParCorrHC3``; chosen in docs/adr/0009).
3. For every candidate signal X, every return R^a and every lag in 1..tau_max, the deciding
   test (``hybrid`` by default) of ``X_{t-lag} -> R^a_t``. It conditions on the PC1 parents of
   R^a and on the parents of X other than X's own lags. The MCI test of the same link is
   recorded next to it as a diagnostic.
4. Bonferroni over each signal's (asset, lag) tests gives its family p-value. The family
   p-values are then adjusted for the FDR across the run's signals (BH, or BY).
5. Verdict:
   - kill ``causal.no_significant_link`` when the family p is >= alpha (no link survives
     even before the correction across signals);
   - otherwise kill ``causal.not_significant_after_correction`` when the q-value is >= alpha;
   - otherwise pass.
   It is a kill, not a pass, on an exact tie at alpha.
6. A circular block-bootstrap interval of the partial correlation at the best (asset, lag).

Before testing: leading rows with any missing value are dropped (``ret_*`` is missing on the
first session). A missing value after that kills every signal with
``general.data_validation_failed``, because dropping interior rows would join non-adjacent
sessions as if they were one lag apart. Fewer than ``min_obs`` complete rows kill every
signal with ``general.insufficient_data``.

Assumptions (where each is violated is in docs/adr/0010): causal sufficiency over the panel's
variables, the causal Markov condition and faithfulness, no contemporaneous effects, a
stationary causal structure, linear mean effects. HC3 makes the p-values robust to
heteroskedasticity, not to these.

Look-ahead: the verdict is a full-sample statement about the sample's causal structure, not a
value used for trading at any date t. Every input value dated t is point-in-time (the panel's
contract), and every tested link runs from a lagged feature to a later return.

Reference: Runge, Nowack, Kretschmer, Flaxman & Sejdinovic (2019), "Detecting and quantifying
causal associations in large nonlinear time series datasets", Science Advances 5(11).
"""

from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from invariant.config.settings import CausalConfig
from invariant.core.logging import get_logger
from invariant.core.run_hash import RngFactory, canonical_json
from invariant.core.types import KillReason, StageName, Verdict, make_signal_id
from invariant.data.pit import PitPanel
from invariant.ledger.kill_ledger import KillLedger, LedgerEntry, MetricValue
from invariant.stages.causal.bootstrap import partial_corr_ci
from invariant.stages.causal.ci_tests import CITestName, make_ci_test
from invariant.stages.causal.multiple_testing import bonferroni_family_p, fdr_qvalues
from invariant.stages.causal.pcmci import DecisionTest, PcmciOutput, lagged_design, run_pcmci
from invariant.stages.causal.result import (
    CausalResult,
    LaggedLink,
    LagTest,
    SignalCausalLink,
    SignalEvidence,
)

__all__ = [
    "CAUSAL_RESULT_NAME",
    "CausalInputs",
    "decide",
    "ledger_metrics",
    "record_causal_ledger",
    "run_causal_stage",
    "select_inputs",
    "signal_id_of",
    "write_causal_result",
]

CAUSAL_RESULT_NAME: Final = "causal_result.json"
RETURN_PREFIX: Final = "ret_"
TARGET_PREFIX: Final = "fwd_"
SYNTHETIC_PREFIX: Final = "syn_"


def signal_id_of(column: str) -> str:
    """Ledger id of a feature column: lower-cased, and a ``syn_`` column loses its prefix so
    that its id is the benchmark's signal id (``syn_s_real`` -> ``s_real``, ``macro_INDPRO``
    -> ``macro_indpro``). ``select_inputs`` rejects columns whose ids collide."""
    return make_signal_id(column.removeprefix(SYNTHETIC_PREFIX).lower())


@dataclass(frozen=True)
class CausalInputs:
    """The columns Stage 1 reads, with leading incomplete rows trimmed."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    signal_columns: tuple[str, ...]
    return_columns: tuple[str, ...]
    interior_missing: tuple[str, ...]
    """Columns with a missing value after the first complete row."""


def select_inputs(features: pd.DataFrame, config: CausalConfig) -> CausalInputs:
    """Pick Stage 1's columns from a PIT panel's ``features``. Never touches ``fwd_*``.

    Raises ``ValueError`` when the panel has no return or no candidate column, or when a
    configured signal is not a feature column.
    """
    columns = [str(c) for c in features.columns]
    returns = tuple(c for c in columns if c.startswith(RETURN_PREFIX))
    feats = tuple(
        c for c in columns if not c.startswith(RETURN_PREFIX) and not c.startswith(TARGET_PREFIX)
    )
    if not returns:
        raise ValueError("the panel has no return column (ret_*)")
    ids = [signal_id_of(c) for c in feats]
    if len(set(ids)) != len(ids):
        raise ValueError(f"feature columns map to duplicate ids: {sorted(ids)}")
    by_id = dict(zip(ids, feats, strict=True))
    if config.signals is None:
        signals = feats
    else:
        missing = sorted(set(config.signals) - set(by_id))
        if missing:
            raise ValueError(f"configured signals are not feature columns: {missing}")
        signals = tuple(by_id[s] for s in config.signals)
    if not signals:
        raise ValueError("the panel has no candidate signal column")
    frame = features.loc[:, [*feats, *returns]]
    complete = frame.notna().all(axis=1).to_numpy()
    start = int(np.argmax(complete)) if complete.any() else len(frame)
    trimmed = frame.iloc[start:]
    interior = tuple(str(c) for c in trimmed.columns[trimmed.isna().any(axis=0).to_numpy()])
    return CausalInputs(
        frame=trimmed,
        feature_columns=feats,
        signal_columns=signals,
        return_columns=returns,
        interior_missing=interior,
    )


def decide(family_p: float, q: float, alpha: float) -> tuple[Verdict, KillReason | None]:
    """Stage 1's verdict from a signal's family p-value and its FDR q-value."""
    if family_p >= alpha:
        return Verdict.KILL, KillReason.CAUSAL_NO_SIGNIFICANT_LINK
    if q >= alpha:
        return Verdict.KILL, KillReason.CAUSAL_NOT_SIGNIFICANT_AFTER_CORRECTION
    return Verdict.PASS, None


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def _untested(
    inputs: CausalInputs,
    config: CausalConfig,
    reason: KillReason,
    issue: str,
) -> CausalResult:
    index = inputs.frame.index
    return CausalResult(
        config=config,
        tigramite_version=importlib.metadata.version("tigramite"),
        n_obs=len(index),
        n_rows_tested=None,
        first_date=index[0].date() if len(index) else None,
        last_date=index[-1].date() if len(index) else None,
        var_names=tuple(map(str, inputs.frame.columns)),
        signal_columns=inputs.signal_columns,
        return_columns=inputs.return_columns,
        parents={},
        graph=(),
        signals=tuple(
            SignalCausalLink(
                signal_id=signal_id_of(c),
                column=c,
                verdict=Verdict.KILL,
                reason=reason,
                evidence=None,
            )
            for c in inputs.signal_columns
        ),
        data_issue=issue,
    )


def _graph(out: PcmciOutput, alpha: float) -> tuple[LaggedLink, ...]:
    links = []
    n = len(out.var_names)
    for i in range(n):
        for j in range(n):
            for lag in range(1, out.tau_max + 1):
                p = float(out.p[i, j, lag])
                if p <= alpha:
                    links.append(
                        LaggedLink(
                            source=out.var_names[i],
                            target=out.var_names[j],
                            lag=lag,
                            p=p,
                            r=float(out.val[i, j, lag]),
                        )
                    )
    return tuple(links)


def _lag_tests(
    out: PcmciOutput, decision: DecisionTest, signal: str, returns: Sequence[str]
) -> tuple[LagTest, ...]:
    lags = list(range(1, out.tau_max + 1))
    tests = []
    for asset in returns:
        p = out.decision_p[(decision, signal, asset)]
        r = out.decision_val[(decision, signal, asset)]
        p_mci, r_mci = out.link_p(signal, asset, lags), out.link_val(signal, asset, lags)
        for k, lag in enumerate(lags):
            values = (p[k], r[k], p_mci[k], r_mci[k])
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite test result for {signal} -> {asset} at lag {lag}")
            tests.append(
                LagTest(
                    asset=asset,
                    lag=lag,
                    p=float(p[k]),
                    r=float(r[k]),
                    p_mci=float(p_mci[k]),
                    r_mci=float(r_mci[k]),
                )
            )
    return tuple(tests)


def run_causal_stage(panel: PitPanel, config: CausalConfig, rng: RngFactory) -> CausalResult:
    """Run Stage 1 on ``panel`` (see module docstring). Deterministic given ``rng``'s seed.

    Claims the ``causal`` child stream of ``rng``, with streams ``ci_test`` and ``bootstrap``
    (one generator per signal id) under it.
    """
    log = get_logger(__name__)
    causal_rng = rng.child(StageName.CAUSAL.value)
    ci_seed = _u32(causal_rng, "ci_test")
    boot_rng = causal_rng.child("bootstrap")

    inputs = select_inputs(panel.features, config)
    if inputs.interior_missing:
        issue = f"missing values after the first complete row in {list(inputs.interior_missing)}"
        log.warning("causal_data_invalid", columns=list(inputs.interior_missing))
        return _untested(inputs, config, KillReason.DATA_VALIDATION_FAILED, issue)
    n_obs = len(inputs.frame)
    if n_obs < config.min_obs:
        issue = f"{n_obs} complete rows < min_obs {config.min_obs}"
        log.warning("causal_insufficient_data", n_obs=n_obs, min_obs=config.min_obs)
        return _untested(inputs, config, KillReason.INSUFFICIENT_DATA, issue)

    decision = DecisionTest(config.decision)
    var_names = tuple(map(str, inputs.frame.columns))
    data = inputs.frame.to_numpy(dtype=np.float64)
    t0 = time.perf_counter()
    out = run_pcmci(
        data,
        var_names,
        make_ci_test(CITestName(config.ci_test), seed=ci_seed),
        tau_max=config.tau_max,
        pc_alpha=config.pc_alpha,
        decision_links=[(s, a) for s in inputs.signal_columns for a in inputs.return_columns],
        decisions=[decision],
    )

    per_signal = {
        s: _lag_tests(out, decision, s, inputs.return_columns) for s in inputs.signal_columns
    }
    family_p = {s: bonferroni_family_p([t.p for t in tests]) for s, tests in per_signal.items()}
    q_values = fdr_qvalues([family_p[s] for s in inputs.signal_columns], config.fdr)
    links = []
    for signal, q in zip(inputs.signal_columns, q_values, strict=True):
        tests = per_signal[signal]
        best = tests[int(np.argmin([t.p for t in tests]))]  # first minimum: asset, then lag order
        conditions = out.conditions(decision, signal, best.asset, best.lag)
        design = lagged_design(
            data, var_names, signal, best.asset, best.lag, conditions, tau_max=config.tau_max
        )
        sid = signal_id_of(signal)
        ci = partial_corr_ci(
            design.x,
            design.y,
            design.z,
            n_boot=config.bootstrap.n_boot,
            block_length=config.bootstrap.block_length,
            confidence=config.bootstrap.confidence,
            rng=boot_rng.generator(sid),
        )
        verdict, reason = decide(family_p[signal], float(q), config.alpha)
        links.append(
            SignalCausalLink(
                signal_id=sid,
                column=signal,
                verdict=verdict,
                reason=reason,
                evidence=SignalEvidence(
                    tests=tests,
                    n_family_tests=len(tests),
                    family_p=family_p[signal],
                    mci_family_p=bonferroni_family_p([t.p_mci for t in tests]),
                    # q >= p in exact arithmetic; max() absorbs a last-bit rounding of p*m/m.
                    q=max(float(q), family_p[signal]),
                    best_asset=best.asset,
                    best_lag=best.lag,
                    best_conditions=conditions,
                    r_best_ci=ci,
                ),
            )
        )
    index = inputs.frame.index
    result = CausalResult(
        config=config,
        tigramite_version=importlib.metadata.version("tigramite"),
        n_obs=n_obs,
        n_rows_tested=n_obs - 2 * config.tau_max,
        first_date=index[0].date(),
        last_date=index[-1].date(),
        var_names=var_names,
        signal_columns=inputs.signal_columns,
        return_columns=inputs.return_columns,
        parents={k: tuple(v) for k, v in out.parents.items()},
        graph=_graph(out, config.alpha),
        signals=tuple(links),
    )
    log.info(
        "causal_stage_done",
        n_obs=n_obs,
        n_signals=len(links),
        n_pass=sum(link.verdict is Verdict.PASS for link in links),
        seconds=round(time.perf_counter() - t0, 3),
    )
    return result


def _asset_key(column: str) -> str:
    return column.removeprefix(RETURN_PREFIX)


def ledger_metrics(link: SignalCausalLink, result: CausalResult) -> dict[str, MetricValue]:
    """Flat, finite, scalar metrics of one signal's decision, for its ledger entry."""
    cfg = result.config
    metrics: dict[str, MetricValue] = {
        "n_obs": result.n_obs,
        "min_obs": cfg.min_obs,
        "alpha": cfg.alpha,
        "ci_test": cfg.ci_test,
        "decision": cfg.decision,
        "fdr": cfg.fdr,
        "tau_max": cfg.tau_max,
    }
    ev = link.evidence
    if ev is None:
        metrics["data_issue"] = result.data_issue or "not tested"
        return metrics
    metrics |= {
        "n_rows_tested": result.n_rows_tested if result.n_rows_tested is not None else 0,
        "n_family_tests": ev.n_family_tests,
        "n_fdr_signals": len(result.signals),
        "family_p": ev.family_p,
        "q": ev.q,
        "mci_family_p": ev.mci_family_p,
        "best_asset": _asset_key(ev.best_asset),
        "best_lag": ev.best_lag,
        "r_best": ev.r_best_ci.estimate,
        "r_best_ci_low": ev.r_best_ci.low,
        "r_best_ci_high": ev.r_best_ci.high,
        "r_best_ci_confidence": ev.r_best_ci.confidence,
        "bootstrap_block_length": ev.r_best_ci.block_length,
    }
    for t in ev.tests:
        key = f"{_asset_key(t.asset)}.lag{t.lag}"
        metrics[f"p.{key}"] = t.p
        metrics[f"r.{key}"] = t.r
        metrics[f"p_mci.{key}"] = t.p_mci
    return metrics


def record_causal_ledger(ledger: KillLedger, result: CausalResult) -> list[LedgerEntry]:
    """One Stage 1 ledger entry per signal, pass or kill, with every number as a metric."""
    return [
        ledger.record(
            link.signal_id,
            StageName.CAUSAL,
            link.verdict,
            reason=link.reason,
            metrics=ledger_metrics(link, result),
        )
        for link in result.signals
    ]


def write_causal_result(run_dir: Path, result: CausalResult) -> Path:
    """Write ``result`` as canonical JSON to ``<run_dir>/causal_result.json``."""
    path = run_dir / CAUSAL_RESULT_NAME
    path.write_text(canonical_json(result.model_dump(mode="json")) + "\n", encoding="utf-8")
    return path
