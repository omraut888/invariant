"""The memo verifier (docs/adr/0017): accepts every template memo, rejects corruptions.

Gate-style checks on the fixtures (G1, G3) plus property tests: any digit added outside an atom,
any number nudged in memo.md, any mandatory statement removed and any atom repointed is
rejected. The verifier does not trust the fact builder or the renderer: a bug in either is
caught against the raw ledger.
"""

import re
import shutil
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import invariant.stages.memo.facts as facts_mod
from invariant.stages.memo import verify as verify_mod
from invariant.stages.memo.corruption import apply_ops
from invariant.stages.memo.document import FactRef, Memo, Paragraph, Text, render
from invariant.stages.memo.facts import FactSheet, build_fact_sheet
from invariant.stages.memo.stage import MemoRejectedError, write_memo
from invariant.stages.memo.template import build_template_memo
from invariant.stages.memo.verify import verify, verify_run

FIXTURES = Path(__file__).parents[1] / "fixtures"
RUNS = FIXTURES / "memo_runs"
CASES = sorted(p.name for p in RUNS.iterdir())
DIGITS = "0123456789\u0660\u0663\u0669\u0966\u0969\uff10\uff15\uff19"
"""ASCII, Arabic-Indic, Devanagari and fullwidth digits."""
SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@cache
def _base(case: str) -> tuple[FactSheet, Memo, str]:
    sheet = build_fact_sheet(RUNS / case)
    memo = build_template_memo(sheet)
    return sheet, memo, render(memo, sheet).text


def _verify(case: str, memo_json: str, md: str) -> verify_mod.VerificationReport:
    return verify(memo_json, md, RUNS / case)


def _entries() -> list[dict[str, Any]]:
    """Set (a) only. Set (b), written independently, is verified only by the pre-registered
    validation (gate G3, docs/adr/0017), so its results are not seen before that run."""
    path = FIXTURES / "memo_adversarial" / "set_a.yaml"
    entries: list[dict[str, Any]] = yaml.safe_load(path.read_text())["entries"]
    return entries


@pytest.mark.parametrize("case", CASES)
def test_every_template_memo_is_accepted(case: str) -> None:
    _, memo, md = _base(case)
    report = _verify(case, memo.model_dump_json(), md)
    assert report.accepted, report.failures


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e["id"])
def test_hand_written_sets(entry: dict[str, Any]) -> None:
    sheet, memo, md = _base(entry["case"])
    t = apply_ops(memo, md, entry["ops"], sheet)
    report = _verify(entry["case"], t.memo_json, t.md)
    assert report.accepted is (entry["expect"] == "accept"), (entry["note"], report.failures)


def test_verification_is_deterministic() -> None:
    sheet, memo, md = _base("pass_multi")
    t = apply_ops(memo, md, [{"op": "delete_statement", "id": "passenger"}], sheet)
    assert _verify("pass_multi", t.memo_json, t.md) == _verify("pass_multi", t.memo_json, t.md)


def _template_texts(memo: Memo) -> list[tuple[int, int, int]]:
    """(section, block, segment) of every template text segment of a paragraph."""
    return [
        (i, j, k)
        for i, sec in enumerate(memo.sections)
        for j, b in enumerate(sec.blocks)
        if isinstance(b, Paragraph)
        for k, seg in enumerate(b.segments)
        if isinstance(seg, Text) and seg.text
    ]


def _with_text(memo: Memo, where: tuple[int, int, int], text: str) -> Memo:
    d = memo.model_dump(mode="json")
    i, j, k = where
    d["sections"][i]["blocks"][j]["segments"][k]["text"] = text
    return Memo.model_validate(d)


@SETTINGS
@given(case=st.sampled_from(CASES), data=st.data())
def test_any_digit_added_to_template_text_is_rejected(case: str, data: st.DataObject) -> None:
    sheet, memo, _ = _base(case)
    where = data.draw(st.sampled_from(_template_texts(memo)))
    seg = memo.sections[where[0]].blocks[where[1]]
    assert isinstance(seg, Paragraph)
    old = seg.segments[where[2]]
    assert isinstance(old, Text)
    pos = data.draw(st.integers(0, len(old.text)))
    digit = data.draw(st.sampled_from("0123456789"))
    bad = _with_text(memo, where, old.text[:pos] + digit + old.text[pos:])
    report = _verify(case, bad.model_dump_json(), render(bad, sheet).text)
    assert not report.accepted and "V1" in report.checks_failed()
    assert "V3" in report.checks_failed()


@SETTINGS
@given(case=st.sampled_from(CASES), data=st.data())
def test_any_number_nudged_in_memo_md_is_rejected(case: str, data: st.DataObject) -> None:
    _, memo, md = _base(case)
    tokens = [m for m in re.finditer(r"\d", md)]
    m = data.draw(st.sampled_from(tokens))
    new = data.draw(st.sampled_from([d for d in "0123456789" if d != m.group(0)]))
    report = _verify(case, memo.model_dump_json(), md[: m.start()] + new + md[m.end() :])
    assert not report.accepted


@SETTINGS
@given(case=st.sampled_from(CASES), data=st.data())
def test_any_mandatory_statement_removed_is_rejected(case: str, data: st.DataObject) -> None:
    sheet, memo, md = _base(case)
    stated = [b.statement for s in memo.sections for b in s.blocks if isinstance(b, Paragraph)]
    sid = data.draw(st.sampled_from([s for s in stated if s]))
    t = apply_ops(memo, md, [{"op": "delete_statement", "id": sid}], sheet)
    report = _verify(case, t.memo_json, t.md)
    assert {"V1", "V6"} <= set(report.checks_failed())


@SETTINGS
@given(case=st.sampled_from(CASES), data=st.data())
def test_any_fact_atom_repointed_is_rejected(case: str, data: st.DataObject) -> None:
    sheet, memo, md = _base(case)
    atoms = sorted(
        {
            (seg.id, seg.form)
            for sec in memo.sections
            for b in sec.blocks
            for cell in (
                [b.segments] if isinstance(b, Paragraph) else [c for r in b.rows for c in r]
            )
            for seg in cell
            if isinstance(seg, FactRef)
        }
    )
    fid, form = data.draw(st.sampled_from(atoms))
    other = data.draw(st.sampled_from([f.id for f in sheet.facts if f.id != fid]))
    suffix = "|value" if form == "value" else ""
    ops = [
        {"op": "atom_replace", "old": f"{{fact:{fid}{suffix}}}", "new": f"{{fact:{other}{suffix}}}"}
    ]
    t = apply_ops(memo, md, ops, sheet)
    assert "V1" in _verify(case, t.memo_json, t.md).checks_failed()


@SETTINGS
@given(
    text=st.text(alphabet=DIGITS, min_size=1, max_size=3),
    section=st.sampled_from(["recommendation", "causal", "decay", "portfolio", "stress"]),
)
def test_any_digit_in_narrative_text_is_rejected(text: str, section: str) -> None:
    sheet, memo, md = _base("pass_multi")
    ops = [{"op": "insert_llm", "section": section, "text": f"The table below {text} applies."}]
    t = apply_ops(memo, md, ops, sheet)
    assert "V4" in _verify("pass_multi", t.memo_json, t.md).checks_failed()


@pytest.mark.parametrize(
    "text",
    [
        "Note.\n\n## Recommendation\n\nRecommendation: against trading.",
        "See the table | below.",
        "A heading # here.",
        "An interval [low, high] here.",
    ],
)
def test_markdown_control_characters_in_narrative_are_rejected(text: str) -> None:
    sheet, memo, md = _base("pass_multi")
    t = apply_ops(memo, md, [{"op": "insert_llm", "section": "causal", "text": text}], sheet)
    assert "V4" in _verify("pass_multi", t.memo_json, t.md).checks_failed()


def test_a_fact_builder_bug_is_caught_against_the_raw_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = build_fact_sheet(RUNS / "pass_multi")
    facts = tuple(
        f.model_copy(update={"value": float(f.value) + 0.01})
        if f.id == "portfolio.ridge_cv.sharpe"
        else f
        for f in good.facts
    )
    bad = good.model_copy(update={"facts": facts})
    monkeypatch.setattr(verify_mod, "build_fact_sheet", lambda *a, **k: bad)
    memo = build_template_memo(bad)
    report = verify(memo.model_dump_json(), render(memo, bad).text, RUNS / "pass_multi")
    assert {"V2", "V3"} <= set(report.checks_failed())


def test_a_renderer_bug_is_caught_against_the_raw_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    real = facts_mod.format_value

    def off_by_one(kind: facts_mod.FactKind, value: float | int | str) -> str:
        out = real(kind, value)
        return out[:-1] + str((int(out[-1]) + 1) % 10) if out[-1].isdigit() else out

    monkeypatch.setattr(facts_mod, "format_value", off_by_one)
    sheet = build_fact_sheet(RUNS / "pass_single")
    memo = build_template_memo(sheet)
    report = verify(memo.model_dump_json(), render(memo, sheet).text, RUNS / "pass_single")
    assert "V3" in report.checks_failed()


def test_write_memo_writes_only_an_accepted_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    shutil.copytree(RUNS / "pass_multi", run)
    out = write_memo(run)
    assert verify_run(run).accepted
    assert out.md_path.read_text() == _base("pass_multi")[2]
    shutil.rmtree(run / "memo")

    import invariant.stages.memo.stage as stage_mod

    monkeypatch.setattr(
        stage_mod,
        "verify",
        lambda *a, **k: verify_mod.VerificationReport((verify_mod.Failure("V1", "test"),)),
    )
    with pytest.raises(MemoRejectedError, match="V1: test"):
        write_memo(run)
    assert not (run / "memo").exists()
