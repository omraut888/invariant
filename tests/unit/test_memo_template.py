"""Stage 5 template memo (docs/adr/0017): atoms, mandatory statements, determinism."""

import re
from pathlib import Path

import pytest

from invariant.stages.memo.document import Memo, Paragraph, Table, Text, render
from invariant.stages.memo.facts import DataKind, Recommendation, build_fact_sheet
from invariant.stages.memo.literature import SUMMARY_MAX, load_corpus
from invariant.stages.memo.mandatory import MANDATORY, triggered
from invariant.stages.memo.template import build_template_memo, parse

FIXTURES = Path(__file__).parents[1] / "fixtures" / "memo_runs"
CASES = sorted(p.name for p in FIXTURES.iterdir())

EXPECTED = {
    "pass_single": (Recommendation.PAPER_TRADING_CANDIDATE, {"stress_description"}, {"passenger"}),
    "pass_multi": (Recommendation.PAPER_TRADING_CANDIDATE, {"passenger"}, {"collateral"}),
    "volfrag": (Recommendation.PAPER_TRADING_CANDIDATE, {"passenger"}, {"collateral"}),
    "portfolio_kill": (Recommendation.AGAINST, {"not_detectable", "gross_sharpe"}, {"passenger"}),
    "hidden_confounder": (Recommendation.AGAINST, {"collateral", "hidden_confounder"}, set()),
    "realistic": (Recommendation.AGAINST, {"not_detectable"}, {"portfolio_method", "no_half_life"}),
    "stage2_short": (Recommendation.NOT_DECIDED, {"not_decided", "no_half_life"}, {"decay_power"}),
    "not_decided": (Recommendation.NOT_DECIDED, {"not_decided"}, {"hidden_confounder"}),
}


def _memo(case: str) -> tuple[Memo, str]:
    sheet = build_fact_sheet(FIXTURES / case)
    memo = build_template_memo(sheet)
    return memo, render(memo, sheet).text


def test_every_case_is_expected() -> None:
    assert set(EXPECTED) == set(CASES)


@pytest.mark.parametrize("case", CASES)
def test_recommendation_and_mandatory_statements(case: str) -> None:
    rec, present, absent = EXPECTED[case]
    sheet = build_fact_sheet(FIXTURES / case)
    ids = {m.id for m in triggered(sheet)}
    assert sheet.recommendation is rec
    assert present <= ids and not absent & ids, ids
    assert {"data_source", "stage4_not_adopted"} <= ids
    memo = build_template_memo(sheet)
    stated = [
        b.statement
        for s in memo.sections
        for b in s.blocks
        if isinstance(b, Paragraph) and b.statement
    ]
    assert stated == [m.id for m in triggered(sheet)]


@pytest.mark.parametrize("case", CASES)
def test_every_digit_is_inside_an_atom(case: str) -> None:
    sheet = build_fact_sheet(FIXTURES / case)
    memo = build_template_memo(sheet)
    out = render(memo, sheet)
    covered = [False] * len(out.text)
    for sp in out.spans:
        if not isinstance(sp.segment, Text):
            covered[sp.start : sp.end] = [True] * (sp.end - sp.start)
    stray = [m.start() for m in re.finditer(r"\d", out.text) if not covered[m.start()]]
    assert not stray, [out.text[max(0, i - 30) : i + 10] for i in stray[:3]]
    for sec in memo.sections:
        assert not re.search(r"\d", sec.title)
        for b in sec.blocks:
            if isinstance(b, Table):
                assert not any(re.search(r"\d", h) for h in b.header)


@pytest.mark.parametrize("case", CASES)
def test_the_template_memo_is_deterministic_and_round_trips(case: str) -> None:
    memo, text = _memo(case)
    again, text2 = _memo(case)
    assert memo == again and text == text2
    assert Memo.model_validate_json(memo.model_dump_json()) == memo


def test_mandatory_templates_parse_for_every_fixture() -> None:
    for case in CASES:
        sheet = build_fact_sheet(FIXTURES / case)
        for m in MANDATORY:
            if m.trigger(sheet):
                segs = parse(m.text(sheet))
                assert all(not re.search(r"\d", s.text) for s in segs if isinstance(s, Text))


def test_real_data_frames_benchmark_rates_as_validation_rates() -> None:
    sheet = build_fact_sheet(FIXTURES / "pass_single").model_copy(
        update={"data_kind": DataKind.REAL}
    )
    text = next(m for m in MANDATORY if m.id == "data_source").text(sheet)
    assert "benchmark validation rate" in text and "not a property of this strategy" in text


def test_parse_rejects_malformed_atoms() -> None:
    assert parse("a {stage:causal} b")[1].type == "stage"
    with pytest.raises(ValueError, match="unparsed"):
        parse("a {fact:x b")
    with pytest.raises(ValueError):
        parse("{stage:nonsense}")


def test_literature_corpus() -> None:
    corpus = load_corpus()
    assert len(corpus) >= 20
    for e in corpus.values():
        assert len(e.summary) <= SUMMARY_MAX
        assert set(e.stages) <= {"causal", "decay", "portfolio", "adversarial", "benchmark"}
    keys = set()
    for case in CASES:
        memo, _ = _memo(case)
        keys |= {
            seg.key
            for s in memo.sections
            for b in s.blocks
            if isinstance(b, Paragraph)
            for seg in b.segments
            if seg.type in ("cite", "bibliography")
        }
    assert keys <= set(corpus)
