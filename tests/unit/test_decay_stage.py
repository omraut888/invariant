from __future__ import annotations

import math
from pathlib import Path

import pytest
from tests.fixtures.causal import BenchmarkPanel, benchmark_panel
from tests.fixtures.ledger import CLEAN_GIT

from invariant.config.settings import DecayConfig, Settings, load_config
from invariant.core.run_hash import RngFactory, RunContext
from invariant.core.types import KillReason, StageName, Verdict
from invariant.ledger.kill_ledger import KillLedger
from invariant.stages.causal.result import CausalResult
from invariant.stages.causal.stage import record_causal_ledger, run_causal_stage
from invariant.stages.decay.result import DecayResult, SignalDecay
from invariant.stages.decay.stage import (
    DECAY_RESULT_NAME,
    decide,
    ledger_metrics,
    record_decay_ledger,
    run_decay_stage,
    write_decay_result,
)


def _decay_cfg(**update: object) -> DecayConfig:
    base = load_config().decay.model_copy(update={"min_obs": 500, "n_boot": 199})
    return base.model_copy(update=update)


@pytest.fixture(scope="module")
def stage1(tmp_path_factory: pytest.TempPathFactory) -> tuple[BenchmarkPanel, CausalResult]:
    bp = benchmark_panel(tmp_path_factory.mktemp("bp"), n_obs=2500)  # seed 7: two passes
    return bp, run_causal_stage(bp.panel, bp.config.causal, RngFactory(3))


def test_decide_kills_only_below_alpha() -> None:
    assert decide(0.049, 0.05) == (Verdict.KILL, KillReason.DECAY_EDGE_DECLINING)
    assert decide(0.05, 0.05) == (Verdict.PASS, None)


def test_only_stage1_passes_are_tested_and_result_is_deterministic(
    stage1: tuple[BenchmarkPanel, CausalResult],
) -> None:
    bp, causal = stage1
    passed = [s.signal_id for s in causal.signals if s.verdict is Verdict.PASS]
    assert passed  # the fixture must exercise the stage
    a = run_decay_stage(bp.panel, causal, _decay_cfg(), RngFactory(9))
    b = run_decay_stage(bp.panel, causal, _decay_cfg(), RngFactory(9))
    assert a == b
    assert [s.signal_id for s in a.signals] == passed
    assert set(a.not_reached) == {s.signal_id for s in causal.signals} - set(passed)
    by_id = {s.signal_id: s for s in causal.signals}
    for sig in a.signals:
        ev = by_id[sig.signal_id].evidence
        assert ev is not None and sig.test is not None
        assert (sig.asset, sig.lag, sig.conditions) == (
            ev.best_asset,
            ev.best_lag,
            ev.best_conditions,
        )
        assert sig.verdict is decide(sig.test.p, 0.05)[0]


def test_too_short_a_design_is_insufficient_data(
    stage1: tuple[BenchmarkPanel, CausalResult],
) -> None:
    bp, causal = stage1
    res = run_decay_stage(bp.panel, causal, _decay_cfg(min_obs=5000), RngFactory(9))
    assert res.signals and all(
        s.verdict is Verdict.KILL and s.reason is KillReason.INSUFFICIENT_DATA and s.test is None
        for s in res.signals
    )


def test_rejects_a_causal_result_from_another_panel(
    stage1: tuple[BenchmarkPanel, CausalResult], tmp_path: Path
) -> None:
    _, causal = stage1
    other = benchmark_panel(tmp_path, n_obs=1400)
    with pytest.raises(ValueError, match="not produced from this panel"):
        run_decay_stage(other.panel, causal, _decay_cfg(), RngFactory(9))


def test_signal_decay_contract() -> None:
    common = {"signal_id": "s", "column": "syn_s", "asset": "ret_A", "lag": 1, "conditions": ()}
    with pytest.raises(ValueError):
        SignalDecay(**common, verdict=Verdict.KILL, reason=None, test=None)
    with pytest.raises(ValueError):
        SignalDecay(
            **common,
            verdict=Verdict.KILL,
            reason=KillReason.CAUSAL_NO_SIGNIFICANT_LINK,
            test=None,
        )
    with pytest.raises(ValueError):
        SignalDecay(
            **common,
            verdict=Verdict.KILL,
            reason=KillReason.DECAY_EDGE_DECLINING,
            test=None,
        )


def test_ledger_entries_follow_stage1_and_metrics_are_finite(
    stage1: tuple[BenchmarkPanel, CausalResult], tmp_path: Path
) -> None:
    bp, causal = stage1
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    ctx = RunContext.create(
        bp.config, bp.snapshot.snapshot_id, settings=settings, git_state=CLEAN_GIT
    )
    ledger = KillLedger.open(ctx)
    record_causal_ledger(ledger, causal)
    result = run_decay_stage(bp.panel, causal, _decay_cfg(), ctx.rng)
    entries = record_decay_ledger(ledger, result)
    assert [e.stage for e in entries] == [StageName.DECAY] * len(result.signals)
    for e, sig in zip(entries, result.signals, strict=True):
        assert e.verdict is sig.verdict and e.reason is sig.reason
        assert e.metrics["half_life"] == "not_reported"
        assert all(math.isfinite(v) for v in e.metrics.values() if isinstance(v, float))
        assert ledger_metrics(sig, result) == e.metrics
    reread = KillLedger.read(ctx.run_dir)
    assert len(reread.entries()) == len(causal.signals) + len(result.signals)
    path = write_decay_result(ctx.run_dir, result)
    assert path.name == DECAY_RESULT_NAME
    assert DecayResult.model_validate_json(path.read_text(encoding="utf-8")) == result
