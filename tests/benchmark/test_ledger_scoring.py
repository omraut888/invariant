"""Ground truth first, for the scorer: a ledger holding GroundTruth's own expected fates must
score perfectly on every preset and every stage prefix."""

from pathlib import Path

import pytest
from tests.fixtures.ledger import CLEAN_GIT, temp_settings, write_fates

from invariant.benchmark.config import (
    list_benchmark_presets,
    load_benchmark_config,
    load_benchmark_target_timing,
)
from invariant.benchmark.payload import synthetic_etf_payload, synthetic_tickers
from invariant.benchmark.synthetic import generate_market
from invariant.config.settings import load_config
from invariant.core.run_hash import RunContext
from invariant.data.snapshot import write_snapshot
from invariant.ledger.benchmark_score import score_run
from invariant.ledger.kill_ledger import STAGE_ORDER, KillLedger

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("preset", [None, *list_benchmark_presets()])
def test_oracle_ledger_scores_perfectly(tmp_path: Path, preset: str | None) -> None:
    cfg = load_benchmark_config(preset).model_copy(update={"n_obs": 300, "burn_in": 10})
    dataset = generate_market(cfg, 17)
    truth = dataset.ground_truth
    snap = write_snapshot([synthetic_etf_payload(dataset)], tmp_path / "snapshots")
    base = load_config()
    config = base.model_copy(
        update={
            "universe": base.universe.model_copy(
                update={"tickers": synthetic_tickers(truth.asset_names)}
            ),
            "pit": base.pit.model_copy(update={"targets": load_benchmark_target_timing()}),
        }
    )
    ctx = RunContext.create(
        config, snap.snapshot_id, settings=temp_settings(tmp_path), git_state=CLEAN_GIT
    )
    write_fates(KillLedger.open(ctx), {s.signal_id: s.expected_kill_stage for s in truth.signals})
    n_identifiable = sum(s.identifiable for s in truth.signals)
    for k in range(1, len(STAGE_ORDER) + 1):
        score = score_run(ctx.run_dir, truth, snap, stages=STAGE_ORDER[:k], write=False)
        m = score.matrix
        assert score.complete
        assert m.wrong_kill == 0 and m.missed_kill == 0
        assert m.total == n_identifiable
        assert m.kills_at_expected_stage == m.correct_kill
