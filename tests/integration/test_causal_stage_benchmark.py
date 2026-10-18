"""Stage 1 end to end on the planted-signal benchmark, through the validation harness."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.ledger import CLEAN_GIT

from invariant.core.types import KillReason, StageName, Verdict
from invariant.ledger.benchmark_score import Outcome
from invariant.stages.causal.validation import (
    AcceptanceCriteria,
    ValidationConfig,
    criterion_pass_probability,
    load_validation_config,
    run_benchmark_seed,
    run_validation,
    summarize_validation,
)


def _cfg(**update: object) -> ValidationConfig:
    base = {
        "root_seed": 11,
        "n_seeds": 2,
        "n_obs": 800,
        "presets": ["base", "hidden_confounder"],
        "confidence": 0.95,
        "n_boot": 99,
        "acceptance": AcceptanceCriteria(
            preset="base",
            null_signals=("s_conf", "s_noise"),
            null_pass_upper_max=0.075,
            true_signals=("s_real", "s_decay"),
            true_pass_lower_min=0.7,
        ),
    }
    return ValidationConfig.model_validate({**base, **update})


def test_one_replication_through_the_production_path(tmp_path: Path) -> None:
    rec = run_benchmark_seed("base", 0, _cfg(), git_state=CLEAN_GIT, work_dir=tmp_path)
    score = rec.score
    assert score.complete and not score.contains_placeholders
    assert score.stages == (StageName.CAUSAL,)
    assert [r.signal_id for r in score.signals] == ["s_real", "s_conf", "s_decay", "s_noise"]
    for row, sig in zip(score.signals, rec.signals, strict=True):
        assert row.observed_verdict is sig.verdict
        if sig.verdict is Verdict.KILL:
            assert sig.reason in {
                KillReason.CAUSAL_NO_SIGNIFICANT_LINK.value,
                KillReason.CAUSAL_NOT_SIGNIFICANT_AFTER_CORRECTION.value,
            }
        assert row.outcome is not Outcome.UNDECIDED
    assert rec.n_obs == 799 and rec.n_assets == 1
    assert not list(tmp_path.iterdir())  # the work directory is cleaned up


def test_hidden_confounder_marks_s_conf_unidentifiable(tmp_path: Path) -> None:
    rec = run_benchmark_seed("hidden_confounder", 0, _cfg(), git_state=CLEAN_GIT)
    assert rec.score.unidentifiable == ("s_conf",)
    conf = next(s for s in rec.signals if s.signal_id == "s_conf")
    assert conf.expected_t_hybrid > 0.5  # the identifiability limit, predicted by GroundTruth


def test_parallel_identical_to_serial_and_resumes(tmp_path: Path) -> None:
    cfg = _cfg()
    serial = run_validation(cfg, git_state=CLEAN_GIT, n_jobs=1)
    ckpt = tmp_path / "records.jsonl"
    parallel = run_validation(cfg, git_state=CLEAN_GIT, n_jobs=2, checkpoint=ckpt)
    strip = [r.model_copy(update={"seconds": 0.0}) for r in serial]
    assert [r.model_copy(update={"seconds": 0.0}) for r in parallel] == strip
    again = run_validation(cfg, git_state=CLEAN_GIT, n_jobs=1, checkpoint=ckpt)
    assert again == parallel  # everything reused from the checkpoint, seconds included
    report = summarize_validation(
        serial,
        cfg,
        alpha=0.05,
        tau_max=5,
        git_sha=CLEAN_GIT.sha,
        git_dirty=False,
        tigramite_version="x",
    )
    assert [p.preset for p in report.presets] == ["base", "hidden_confounder"]
    assert len(report.criteria) == 5 and report.criteria[0].passed
    base = report.presets[0]
    assert {s.signal_id for s in base.signals} == {"s_real", "s_conf", "s_decay", "s_noise"}
    assert all(s.pass_rate.trials == 2 for s in base.signals)


def test_validation_config_loads() -> None:
    cfg = load_validation_config()
    assert cfg.n_seeds == 500 and cfg.n_obs is None
    assert cfg.presets == ("base", "realistic", "hidden_confounder")
    assert cfg.acceptance.preset == "base"


@pytest.mark.parametrize(
    ("true_rate", "side", "threshold", "lo", "hi"),
    [
        (0.03, "upper_max", 0.075, 0.99, 1.0),
        (0.075, "upper_max", 0.075, 0.0, 0.05),
        (0.85, "lower_min", 0.70, 0.99, 1.0),
        (0.70, "lower_min", 0.70, 0.0, 0.05),
    ],
)
def test_criterion_pass_probability(
    true_rate: float, side: str, threshold: float, lo: float, hi: float
) -> None:
    p = criterion_pass_probability(500, true_rate, threshold, side)  # type: ignore[arg-type]
    assert lo <= p <= hi
