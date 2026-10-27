import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pytest
from statsmodels.stats.proportion import proportion_confint
from tests.fixtures.fakes import FakePriceClient, fixed_now, price_frame
from tests.fixtures.ledger import CLEAN_GIT, run_context, temp_settings, write_fates

from invariant.benchmark.config import load_benchmark_config, load_benchmark_target_timing
from invariant.benchmark.ground_truth import GroundTruth
from invariant.benchmark.payload import synthetic_etf_payload, synthetic_tickers
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import PipelineConfig, TargetConfig, load_config
from invariant.core.run_hash import RunContext
from invariant.core.types import StageName, Verdict
from invariant.data.snapshot import Snapshot, write_snapshot
from invariant.data.sources.etf import fetch_etf_prices
from invariant.ledger.benchmark_score import (
    BENCHMARK_SCORE_NAME,
    BenchmarkMismatchError,
    BenchmarkScore,
    ConfusionMatrix,
    Outcome,
    aggregate_scores,
    score_fates,
    score_ledger,
    score_run,
    wilson_interval,
)
from invariant.ledger.kill_ledger import Fate, FateStatus, KillLedger

CAUSAL, DECAY, PORTFOLIO = StageName.CAUSAL, StageName.DECAY, StageName.PORTFOLIO
# Expected (all five stages): s_real pass; s_conf, s_noise kill at causal; s_decay kill at decay.
ORACLE: dict[str, StageName | None] = {
    "s_real": None,
    "s_conf": CAUSAL,
    "s_decay": DECAY,
    "s_noise": CAUSAL,
}


def _dataset(preset: str | None = None, seed: int = 5) -> SyntheticDataset:
    cfg = load_benchmark_config(preset).model_copy(update={"n_obs": 300, "burn_in": 10})
    return generate_market(cfg, seed)


def _truth(preset: str | None = None) -> GroundTruth:
    return _dataset(preset).ground_truth


def _fates(spec: Mapping[str, StageName | str | None]) -> dict[str, Fate]:
    """``None``: passed; a stage: killed there; ``"incomplete"``: undecided."""
    out: dict[str, Fate] = {}
    for sid, v in spec.items():
        if v is None:
            out[sid] = Fate(sid, FateStatus.PASSED, StageName.MEMO)
        elif v == "incomplete":
            out[sid] = Fate(sid, FateStatus.INCOMPLETE, CAUSAL)
        else:
            assert isinstance(v, StageName)
            out[sid] = Fate(sid, FateStatus.KILLED, v)
    return out


def _row(score: BenchmarkScore, sid: str) -> Outcome:
    return next(r.outcome for r in score.signals if r.signal_id == sid)


# --- score_fates ------------------------------------------------------------------------------


def test_oracle_fates_score_perfectly() -> None:
    score = score_fates(_fates(dict(ORACLE)), _truth())
    assert score.matrix == ConfusionMatrix(
        correct_kill=3, wrong_kill=0, correct_pass=1, missed_kill=0, kills_at_expected_stage=3
    )
    assert score.complete and not score.undecided and not score.contains_placeholders


def test_every_confusion_cell_and_stage_match() -> None:
    score = score_fates(
        _fates({"s_real": PORTFOLIO, "s_conf": CAUSAL, "s_decay": CAUSAL, "s_noise": None}),
        _truth(),
    )
    assert score.matrix == ConfusionMatrix(
        correct_kill=2, wrong_kill=1, correct_pass=0, missed_kill=1, kills_at_expected_stage=1
    )
    rows = {r.signal_id: r for r in score.signals}
    assert rows["s_real"].outcome is Outcome.WRONG_KILL and rows["s_real"].stage_match is None
    assert rows["s_conf"].stage_match is True
    assert rows["s_decay"].outcome is Outcome.CORRECT_KILL and rows["s_decay"].stage_match is False
    assert rows["s_decay"].expected_stage is DECAY and rows["s_decay"].observed_stage is CAUSAL
    assert rows["s_noise"].outcome is Outcome.MISSED_KILL


def test_stage_prefix_changes_expected_outcomes() -> None:
    # Graded on causal only: s_decay should pass causal; its decay kill is out of scope.
    fates = _fates({"s_real": None, "s_conf": CAUSAL, "s_decay": None, "s_noise": CAUSAL})
    score = score_fates(fates, _truth(), stages=[CAUSAL])
    assert score.matrix.correct_pass == 2 and score.matrix.correct_kill == 2
    assert score.stages == (CAUSAL,)
    rows = {r.signal_id: r for r in score.signals}
    assert (
        rows["s_decay"].expected_verdict is Verdict.PASS and rows["s_decay"].expected_stage is None
    )


def test_undecided_and_absent_signals_are_never_passes() -> None:
    score = score_fates(_fates({"s_real": None, "s_conf": "incomplete"}), _truth())
    assert score.undecided == ("s_conf", "s_decay", "s_noise")
    assert not score.complete
    assert score.matrix.total == 1  # only s_real counted
    assert _row(score, "s_conf") is Outcome.UNDECIDED


def test_unidentifiable_signal_is_scored_but_kept_out_of_matrix() -> None:
    truth = _truth("hidden_confounder")
    assert not truth.signal("s_conf").identifiable
    score = score_fates(_fates({**ORACLE, "s_conf": None}), truth)  # "wrongly" passed
    assert score.unidentifiable == ("s_conf",)
    assert _row(score, "s_conf") is Outcome.MISSED_KILL  # reported per signal...
    assert score.matrix.missed_kill == 0 and score.matrix.total == 3  # ...not counted


def test_unknown_signal_rejected() -> None:
    with pytest.raises(BenchmarkMismatchError, match="s_other"):
        score_fates(_fates({**ORACLE, "s_other": None}), _truth())


def test_low_power_flag() -> None:
    weak = _truth("realistic")  # n_obs 300: every true edge is far below t = 3
    rows = {r.signal_id: r for r in score_fates(_fates(dict(ORACLE)), weak).signals}
    assert rows["s_real"].low_power and rows["s_decay"].low_power
    assert not rows["s_noise"].low_power  # no true edge: power is not the question


# --- ledger and run ---------------------------------------------------------------------------


def _benchmark_run(
    tmp_path: Path, *, timing: TargetConfig | None = None, dataset: SyntheticDataset | None = None
) -> tuple[RunContext, Snapshot, GroundTruth]:
    ds = dataset if dataset is not None else _dataset()
    snap = write_snapshot([synthetic_etf_payload(ds)], tmp_path / "data" / "snapshots")
    base = load_config()
    config: PipelineConfig = base.model_copy(
        update={
            "universe": base.universe.model_copy(
                update={"tickers": synthetic_tickers(ds.ground_truth.asset_names)}
            ),
            "pit": base.pit.model_copy(
                update={"targets": timing if timing is not None else load_benchmark_target_timing()}
            ),
        }
    )
    ctx = RunContext.create(
        config, snap.snapshot_id, settings=temp_settings(tmp_path), git_state=CLEAN_GIT
    )
    return ctx, snap, ds.ground_truth


def test_score_ledger_flags_placeholders(tmp_path: Path) -> None:
    ctx, _, truth = _benchmark_run(tmp_path)
    ledger = KillLedger.open(ctx)
    write_fates(ledger, ORACLE, placeholder=True)
    score = score_ledger(ledger, truth)
    assert score.contains_placeholders
    assert score.matrix.correct_kill == 3 and score.run_id == ctx.run_id


def test_placeholder_on_pass_only_entries_is_flagged(tmp_path: Path) -> None:
    ctx, _, truth = _benchmark_run(tmp_path)
    ledger = KillLedger.open(ctx)
    write_fates(ledger, {"s_real": None}, placeholder=True)  # PASS entries only, no reason
    write_fates(ledger, {"s_conf": CAUSAL, "s_decay": DECAY, "s_noise": CAUSAL})
    assert score_ledger(ledger, truth).contains_placeholders


def test_score_run_writes_score(tmp_path: Path) -> None:
    ctx, snap, truth = _benchmark_run(tmp_path)
    write_fates(KillLedger.open(ctx), ORACLE)
    score = score_run(ctx.run_dir, truth, snap)
    on_disk = BenchmarkScore.model_validate(
        json.loads((ctx.run_dir / BENCHMARK_SCORE_NAME).read_text())
    )
    assert on_disk == score and score.complete and not score.contains_placeholders


def test_score_run_rejects_wrong_timing(tmp_path: Path) -> None:
    ctx, snap, truth = _benchmark_run(tmp_path, timing=load_config().pit.targets)  # lag 1
    with pytest.raises(BenchmarkMismatchError, match="targets"):
        score_run(ctx.run_dir, truth, snap)


def test_score_run_rejects_other_dataset(tmp_path: Path) -> None:
    ctx, snap, _ = _benchmark_run(tmp_path)
    other = _dataset(seed=6).ground_truth
    with pytest.raises(BenchmarkMismatchError, match="describes"):
        score_run(ctx.run_dir, other, snap)


def test_score_run_rejects_other_snapshot(tmp_path: Path) -> None:
    ctx, _, truth = _benchmark_run(tmp_path)
    _, other_snap, _ = _benchmark_run(tmp_path / "b", dataset=_dataset(seed=6))
    with pytest.raises(BenchmarkMismatchError, match="used snapshot"):
        score_run(ctx.run_dir, truth, other_snap)


def test_score_run_rejects_real_data_snapshot(tmp_path: Path) -> None:
    _, _, truth = _benchmark_run(tmp_path)
    raw = price_frame(["2025-01-02", "2025-01-03"], {"SPY": [1.0, 2.0]})
    payload = fetch_etf_prices(
        FakePriceClient(raw), ["SPY"], date(2025, 1, 2), date(2025, 1, 3), now=fixed_now
    )
    real_snap = write_snapshot([payload], tmp_path / "real" / "snapshots")
    real = run_context(tmp_path / "real", snapshot_id=real_snap.snapshot_id)
    with pytest.raises(BenchmarkMismatchError, match="not synthetic"):
        score_run(real.run_dir, truth, real_snap)


# --- aggregation ------------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "n"), [(0, 10), (3, 10), (10, 10), (1, 1), (37, 50)])
def test_wilson_matches_statsmodels(k: int, n: int) -> None:
    got = wilson_interval(k, n)
    low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert got.ci_low == pytest.approx(low, abs=1e-12)
    assert got.ci_high == pytest.approx(high, abs=1e-12)
    assert got.estimate == k / n


def test_wilson_edge_cases() -> None:
    empty = wilson_interval(0, 0)
    assert empty.estimate is None and empty.ci_low is None
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, confidence=1.0)


def test_aggregate_scores() -> None:
    truth = _truth()
    perfect = score_fates(_fates(dict(ORACLE)), truth)
    mixed = score_fates(
        _fates({"s_real": PORTFOLIO, "s_conf": CAUSAL, "s_decay": CAUSAL, "s_noise": None}), truth
    )
    agg = aggregate_scores([perfect, mixed, perfect])
    assert agg.n_runs == 3 and agg.n_incomplete_runs == 0
    assert agg.per_signal_correct["s_real"].successes == 2
    assert agg.per_signal_correct["s_real"].trials == 3
    assert agg.per_signal_stage_match["s_decay"].successes == 2  # wrong stage in `mixed`
    assert agg.per_signal_stage_match["s_decay"].trials == 3
    assert (agg.kill_precision.successes, agg.kill_precision.trials) == (8, 9)
    assert (agg.kill_recall.successes, agg.kill_recall.trials) == (8, 9)


def test_aggregate_requires_same_stages() -> None:
    truth = _truth()
    a = score_fates(_fates(dict(ORACLE)), truth)
    b = score_fates(_fates(dict(ORACLE)), truth, stages=[CAUSAL])
    with pytest.raises(ValueError, match="same stages"):
        aggregate_scores([a, b])
    with pytest.raises(ValueError):
        aggregate_scores([])
