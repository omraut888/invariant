"""Every task kind of the memo validation (docs/adr/0017) on one tiny seed, so a smoke run
cannot miss a crashing code path. No rate is looked at here."""

from pathlib import Path

from invariant.core.run_hash import GitState
from invariant.stages.memo.corruption import CLASSES
from invariant.stages.memo.validation import (
    MemoValidationConfig,
    RunPreset,
    run_hand_sets,
    run_task,
    run_validation,
    summarize_validation,
)

GIT = GitState(sha="0" * 40, dirty=False)


def _cfg(**kw: object) -> MemoValidationConfig:
    base: dict[str, object] = {
        "root_seed": 1,
        "presets": [{"name": "tiny", "preset": "base", "n_obs": 450}],
        "n_seeds": 1,
        "max_seeds": 1,
        "coverage_min": 0,
        "instances_per_class": 1,
        "min_instances": 1,
        "hand_sets": ["tests/fixtures/memo_adversarial/set_a.yaml"],
    }
    return MemoValidationConfig.model_validate(base | kw)


def test_one_run_through_every_step(tmp_path: Path) -> None:
    cfg = _cfg()
    rec = run_task(RunPreset(name="tiny", preset="base", n_obs=450), 0, cfg, git_state=GIT)
    assert rec.template_accepted, rec.template_failures
    assert rec.deterministic and rec.ground_truth_independent
    assert [c.cls for c in rec.classes] == list(CLASSES)
    assert all(c.n == 1 for c in rec.classes)
    records = run_validation(cfg, git_state=GIT, checkpoint=tmp_path / "r.jsonl")
    assert records == [rec.model_copy(update={"seconds": records[0].seconds})]
    again = run_validation(cfg, git_state=GIT, checkpoint=tmp_path / "r.jsonl")
    assert again == records  # resumed from the checkpoint
    hand = run_hand_sets(cfg)
    report = summarize_validation(records, hand, cfg, git_sha="x", git_dirty=False)
    assert [g.name for g in report.gates] == ["G1", "G2", "G3", "G4"]
    assert report.n_runs == 1
