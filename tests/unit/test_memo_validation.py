"""The memo validation's coverage rule, bounds and gates (docs/adr/0017)."""

from pathlib import Path

import pytest

from invariant.core.run_hash import GitState
from invariant.stages.memo import validation as v
from invariant.stages.memo.corruption import CLASSES
from invariant.stages.memo.validation import (
    ClassRecord,
    HandEntryResult,
    RunRecord,
    load_memo_validation_config,
    summarize_validation,
)

GIT = GitState(sha="0" * 40, dirty=False)


def _rec(preset: str, i: int, cats: tuple[str, ...], *, misses: int = 0, n: int = 12) -> RunRecord:
    return RunRecord(
        preset=preset,
        index=i,
        run_id=f"{preset}{i}",
        categories=cats,
        recommendation="against trading",
        members=(),
        passed=(),
        passed_with_true_edge=(),
        template_accepted=True,
        template_failures=(),
        deterministic=True,
        ground_truth_independent=True,
        classes=tuple(
            ClassRecord(cls=c, n=n, misses=misses, channels={}, checks={}, missed=())
            for c in CLASSES
        ),
        seconds=1.0,
    )


def test_the_pre_registered_config_loads() -> None:
    cfg = load_memo_validation_config()
    assert cfg.root_seed == 20261105
    assert [p.name for p in cfg.presets] == [
        "base",
        "realistic",
        "hidden_confounder",
        "multi_asset",
        "volfrag",
        "base_short",
    ]
    assert len(cfg.presets) * cfg.n_seeds * cfg.instances_per_class >= cfg.min_instances
    assert set(v.COVERAGE_PRESETS) == set(v.CATEGORIES)
    names = {p.name for p in cfg.presets}
    assert all(set(ps) <= names for ps in v.COVERAGE_PRESETS.values())


def test_the_deviation_config_changes_only_the_seed_and_the_sets() -> None:
    """docs/adr/0017, deviation 1: a fresh root seed and set (c); everything else unchanged."""
    first = load_memo_validation_config()
    dev = load_memo_validation_config(Path("configs/memo_validation_deviation1.yaml"))
    assert dev.root_seed == 20261107 != first.root_seed
    assert dev.hand_sets == (*first.hand_sets, "tests/fixtures/memo_adversarial/set_c.yaml")
    same = {"root_seed", "hand_sets"}
    assert dev.model_dump(exclude=same) == first.model_dump(exclude=same)


def test_the_coverage_rule_adds_seeds_to_the_mapped_presets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_memo_validation_config().model_copy(
        update={"n_seeds": 2, "max_seeds": 5, "coverage_min": 3}
    )

    def fake(preset: v.RunPreset, index: int, *a: object, **k: object) -> RunRecord:
        cats = ["stage1_kill", "stage2_kill", "stage3_kill", "multi_member_pass", "candidate"]
        cats += ["against"] if preset.name == "realistic" else []
        cats += ["not_decided"] if preset.name == "base_short" and index >= 3 else []
        return _rec(preset.name, index, tuple(cats))

    monkeypatch.setattr(v, "run_task", fake)
    recs = v.run_validation(cfg, git_state=GIT)
    per = {p.name: sum(r.preset == p.name for r in recs) for p in cfg.presets}
    # against: realistic gives 1 per seed -> realistic and hidden_confounder grow to 3;
    # not_decided appears from base_short seed 3 on -> base_short grows to max_seeds 5.
    assert per == {
        "base": 2,
        "realistic": 3,
        "hidden_confounder": 3,
        "multi_asset": 2,
        "volfrag": 2,
        "base_short": 5,
    }


def test_the_zero_miss_bound() -> None:
    assert v._upper(500, 0) == pytest.approx(0.00597, abs=1e-5)
    assert v._upper(576, 0) < v._upper(500, 0) < v._upper(500, 1)


def test_gates() -> None:
    cfg = load_memo_validation_config()
    recs = [_rec("base", i, ("against",)) for i in range(48)]
    hand = [
        HandEntryResult(
            set="a", id="a1", cls="K1", expect="reject", accepted=False, checks=("V9",)
        ),
        HandEntryResult(set="b", id="b1", cls="benign", expect="accept", accepted=False, checks=()),
    ]
    rep = summarize_validation(recs, hand, cfg, git_sha="x", git_dirty=False)
    assert rep.accepted  # a benign rejection is reported, not gated
    assert rep.hand_sets[1].benign_rejected == ("b1",)
    missed = [h.model_copy(update={"accepted": True}) if h.id == "a1" else h for h in hand]
    assert not summarize_validation(recs, missed, cfg, git_sha="x", git_dirty=False).accepted
    few = [_rec("base", i, ("against",), n=10) for i in range(48)]
    g2 = summarize_validation(few, hand, cfg, git_sha="x", git_dirty=False).gates[1]
    assert not g2.passed  # 480 < 500 instances per class
    one = [*recs[:-1], _rec("base", 47, ("against",), misses=1)]
    assert not summarize_validation(one, hand, cfg, git_sha="x", git_dirty=False).gates[1].passed
    bad = [*recs[:-1], recs[-1].model_copy(update={"deterministic": False})]
    assert not summarize_validation(bad, hand, cfg, git_sha="x", git_dirty=False).gates[3].passed


KEYS = frozenset({"ledoit2008", "bollerslev1986"})


def _k7(*ops: dict[str, str]) -> dict[str, object]:
    return {"class": "K7", "expect": "reject", "ops": list(ops)}


def test_entailment_entries_are_defined_by_their_operations() -> None:
    claim = {"op": "insert_llm", "section": "stress", "text": "A claim ({cite:bollerslev1986})."}
    assert v.entailment_only(_k7(claim), KEYS)
    assert v.entailment_only(_k7(claim, {"op": "add_reference", "key": "bollerslev1986"}), KEYS)
    assert not v.entailment_only({**_k7(claim), "class": "K8"}, KEYS)
    assert not v.entailment_only({**_k7(claim), "expect": "accept"}, KEYS)
    assert not v.entailment_only(_k7(claim, {"op": "add_reference", "key": "ledoit2008"}), KEYS)
    assert not v.entailment_only(_k7(claim, {"op": "delete_reference", "key": "x"}), KEYS)
    assert not v.entailment_only(_k7({"op": "add_reference", "key": "ledoit2008"}), KEYS)
    for text in (
        "A claim ({cite:nobody2020}).",  # not a corpus key
        "A claim.",  # uncited
        "A claim ({cite:ledoit2008}). Studies agree.",  # a second, uncited sentence
    ):
        assert not v.entailment_only(_k7({**claim, "text": text}), KEYS)
    assert not v.entailment_only(_k7({**claim, "origin": "template"}), KEYS)


def test_the_entailment_class_in_the_hand_sets() -> None:
    """Classification only: the verifier is not run on set (b) here."""
    import yaml

    from invariant.stages.memo.literature import load_corpus

    keys = frozenset(load_corpus())
    root = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "memo_adversarial"
    found = {
        e["id"]
        for name in ("set_a.yaml", "set_b.yaml")
        for e in yaml.safe_load((root / name).read_text(encoding="utf-8"))["entries"]
        if v.entailment_only(e, keys)
    }
    assert found == {"b_k7_06", "b_k7_07"}


def test_an_entailment_miss_is_reported_not_gated() -> None:
    cfg = load_memo_validation_config()
    recs = [_rec("base", i, ("against",)) for i in range(48)]
    hand = [
        HandEntryResult(set="a", id="a1", cls="K1", expect="reject", accepted=False, checks=()),
        HandEntryResult(
            set="b", id="b1", cls="K7", expect="reject", accepted=True, checks=(), gated=False
        ),
    ]
    rep = summarize_validation(recs, hand, cfg, git_sha="x", git_dirty=False)
    assert rep.accepted
    assert rep.hand_sets[1].n_corruptions == 0
    assert rep.hand_sets[1].entailment_missed == ("b1",)
    gated = [hand[0], hand[1].model_copy(update={"gated": True})]
    assert not summarize_validation(recs, gated, cfg, git_sha="x", git_dirty=False).accepted
