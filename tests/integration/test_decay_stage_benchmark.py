"""Stage 1 and Stage 2 end to end on the planted-signal benchmark, through the validation
harness."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.ledger import CLEAN_GIT

from invariant.core.types import StageName, Verdict
from invariant.ledger.benchmark_score import Outcome
from invariant.stages.decay.validation import (
    DecayAcceptance,
    DecayValidationConfig,
    acceptance_pass_probability,
    load_decay_validation_config,
    run_benchmark_seed,
    run_validation,
    summarize_validation,
)


def _cfg(**update: object) -> DecayValidationConfig:
    base = {
        "root_seed": 11,
        "n_seeds": 2,
        "n_obs": 1500,
        "presets": ["base"],
        "confidence": 0.95,
        "n_boot": 99,
        "min_obs": 500,
        "acceptance": DecayAcceptance(
            preset="base",
            null_signals=("s_real",),
            null_kill_upper_max=0.075,
            decay_signals=("s_decay",),
            decay_kill_lower_min=0.52,
        ),
    }
    return DecayValidationConfig.model_validate({**base, **update})


def test_one_replication_through_both_stages(tmp_path: Path) -> None:
    rec = run_benchmark_seed("base", 0, _cfg(), git_state=CLEAN_GIT, work_dir=tmp_path)
    score = rec.score
    assert score.complete and score.stages == (StageName.CAUSAL, StageName.DECAY)
    by_id = {r.signal_id: r for r in score.signals}
    assert by_id["s_decay"].expected_stage is StageName.DECAY
    for sig in rec.signals:
        assert (sig.stage2_verdict is None) == (sig.stage1_verdict is Verdict.KILL)
        assert by_id[sig.signal_id].outcome is not Outcome.UNDECIDED
    assert not list(tmp_path.iterdir())


def test_parallel_identical_to_serial_and_summary(tmp_path: Path) -> None:
    cfg = _cfg()
    serial = run_validation(cfg, git_state=CLEAN_GIT, n_jobs=1)
    parallel = run_validation(
        cfg, git_state=CLEAN_GIT, n_jobs=2, checkpoint=tmp_path / "records.jsonl"
    )
    assert [r.model_copy(update={"seconds": 0.0}) for r in parallel] == [
        r.model_copy(update={"seconds": 0.0}) for r in serial
    ]
    report = summarize_validation(
        serial, cfg, alpha=0.05, git_sha=CLEAN_GIT.sha or "", git_dirty=False
    )
    assert [c.name for c in report.criteria] == ["complete", "null_kill_upper", "decay_kill_lower"]
    assert report.criteria[0].passed


def test_preregistered_config_loads() -> None:
    cfg = load_decay_validation_config()
    assert (cfg.root_seed, cfg.n_seeds, cfg.n_obs, cfg.min_obs) == (20261023, 1000, None, None)
    assert cfg.acceptance.decay_kill_lower_min == 0.52
    assert cfg.acceptance.null_kill_upper_max == 0.075


@pytest.mark.parametrize(
    ("reach", "rate", "thr", "side", "lo", "hi"),
    [
        (0.844, 0.044, 0.075, "upper_max", 0.95, 0.98),
        (0.844, 0.075, 0.075, "upper_max", 0.0, 0.05),
        (0.834, 0.622, 0.52, "lower_min", 0.999, 1.0),
        (0.834, 0.52, 0.52, "lower_min", 0.0, 0.05),
    ],
)
def test_preregistered_pass_probabilities(
    reach: float, rate: float, thr: float, side: str, lo: float, hi: float
) -> None:
    assert lo <= acceptance_pass_probability(1000, reach, rate, thr, side) <= hi
