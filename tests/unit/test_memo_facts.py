"""Stage 5 fact sheet (docs/adr/0017): built from the ledger and stage results, cross-checked."""

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from invariant.core.types import KillReason, StageName
from invariant.ledger.kill_ledger import FateStatus, LedgerIntegrityError
from invariant.stages.memo.facts import (
    Fact,
    FactKind,
    MemoInputError,
    Recommendation,
    SignalFate,
    build_fact_sheet,
    format_value,
    recommend,
)
from invariant.stages.memo.reference import REFERENCE_FACTS, default_results_root

FIXTURES = Path(__file__).parents[1] / "fixtures" / "memo_runs"
CASES = sorted(p.name for p in FIXTURES.iterdir())


@pytest.fixture
def run_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "run"
    shutil.copytree(FIXTURES / "pass_multi", dest)
    return dest


@pytest.mark.parametrize(
    ("kind", "value", "text"),
    [
        (FactKind.P_VALUE, 0.0004, "< 0.001"),
        (FactKind.P_VALUE, 0.001, "0.001"),
        (FactKind.P_VALUE, 0.4964, "0.496"),
        (FactKind.SHARPE, -0.2604, "-0.26"),
        (FactKind.RATE, 0.73, "0.730"),
        (FactKind.COEFFICIENT, 0.0011834, "1.18e-03"),
        (FactKind.COUNT, 4738, "4738"),
        (FactKind.LEVEL, 0.05, "0.05"),
        (FactKind.TEXT, "SYN00", "SYN00"),
    ],
)
def test_format_value(kind: FactKind, value: float | int | str, text: str) -> None:
    assert format_value(kind, value) == text


def test_format_value_rejects_non_finite() -> None:
    with pytest.raises(MemoInputError, match="non-finite"):
        format_value(FactKind.SHARPE, float("nan"))


def test_fact_atoms_are_self_describing() -> None:
    f = Fact(
        id="causal.s_real.q",
        kind=FactKind.P_VALUE,
        subject="s_real",
        label="Stage one BH q-value",
        value=0.0001,
        sources=("ledger:s_real:causal:q",),
    )
    assert f.full_text() == "s_real's Stage one BH q-value < 0.001"
    g = Fact(
        id="portfolio.s_real.b",
        kind=FactKind.SHARPE,
        subject="s_real",
        label="marginal Sharpe contribution",
        value=0.5,
        ci=(0.1, 0.9),
        ci_confidence=0.95,
        sources=("a", "b", "c", "d"),
    )
    assert (
        g.full_text() == "s_real's marginal Sharpe contribution = 0.50 [0.10, 0.90], 95% interval"
    )


def test_fact_labels_carry_no_digits_and_sources_match_values() -> None:
    with pytest.raises(ValidationError, match="no digits"):
        Fact(
            id="x", kind=FactKind.COUNT, subject=None, label="Stage 1 rows", value=1, sources=("s",)
        )
    with pytest.raises(ValidationError, match="sources"):
        Fact(
            id="x",
            kind=FactKind.SHARPE,
            subject=None,
            label="a",
            value=1.0,
            ci=(0.0, 2.0),
            sources=("s",),
        )


def _fate(status: FateStatus, reason: KillReason | None) -> SignalFate:
    return SignalFate(
        signal_id="s", status=status, stage=StageName.CAUSAL, reason=reason, stages_decided=()
    )


def test_recommendation_rule() -> None:
    passed = _fate(FateStatus.PASSED, None)
    tested = _fate(FateStatus.KILLED, KillReason.CAUSAL_NO_SIGNIFICANT_LINK)
    general = _fate(FateStatus.KILLED, KillReason.INSUFFICIENT_DATA)
    assert recommend((passed, tested, general)) is Recommendation.PAPER_TRADING_CANDIDATE
    assert recommend((tested, general)) is Recommendation.NOT_DECIDED
    assert recommend((tested, tested)) is Recommendation.AGAINST


@pytest.mark.parametrize("case", CASES)
def test_every_fixture_builds_deterministically(case: str) -> None:
    a = build_fact_sheet(FIXTURES / case)
    b = build_fact_sheet(FIXTURES / case)
    assert a == b
    assert a.model_dump_json() == b.model_dump_json()


def test_facts_are_the_ledger_metrics() -> None:
    sheet = build_fact_sheet(FIXTURES / "pass_multi")
    ledger = [json.loads(x) for x in (FIXTURES / "pass_multi" / "kill_ledger.jsonl").open()]
    s1 = next(e for e in ledger if e["signal_id"] == "s_real" and e["stage"] == "causal")
    s3 = next(e for e in ledger if e["signal_id"] == "s_real" and e["stage"] == "portfolio")
    assert sheet.fact("causal.s_real.q").value == s1["metrics"]["q"]
    assert sheet.fact("portfolio.ridge_cv.sharpe").value == s3["metrics"]["ridge_cv_sharpe"]
    b = sheet.fact("portfolio.s_real.b")
    assert b.ci == (s3["metrics"]["b_ci_low"], s3["metrics"]["b_ci_high"])
    assert sheet.members == ("s_conf", "s_real")
    assert sheet.recommendation is Recommendation.PAPER_TRADING_CANDIDATE


def test_a_tampered_ledger_is_refused(run_copy: Path) -> None:
    path = run_copy / "kill_ledger.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"verdict":"pass"', '"verdict":"kill"', 1)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerIntegrityError):
        build_fact_sheet(run_copy)


def test_a_stage_result_that_disagrees_with_the_ledger_is_refused(run_copy: Path) -> None:
    path = run_copy / "portfolio_result.json"
    doc = json.loads(path.read_text())
    doc["evidence"]["arms"][3]["sharpe"] += 0.01
    path.write_text(json.dumps(doc))
    with pytest.raises(MemoInputError, match="disagree"):
        build_fact_sheet(run_copy)


def test_an_unfinished_run_is_refused(run_copy: Path) -> None:
    path = run_copy / "manifest.json"
    doc = json.loads(path.read_text())
    doc["status"] = "failed"
    path.write_text(json.dumps(doc))
    with pytest.raises(MemoInputError, match="not completed"):
        build_fact_sheet(run_copy)


def test_reference_facts_are_read_from_the_archived_reports() -> None:
    sheet = build_fact_sheet(FIXTURES / "pass_single")
    doc = json.loads(
        (default_results_root() / "0010-causal-stage-validation/report.json").read_text()
    )
    rate = doc["presets"][2]["signals"][1]["pass_rate"]
    f = sheet.fact("ref.hidden_confounder_stage1_pass")
    assert (f.value, f.ci, f.ci_confidence) == (
        rate["estimate"],
        (rate["ci_low"], rate["ci_high"]),
        rate["confidence"],
    )
    assert {r.id for r in REFERENCE_FACTS} <= {f.id for f in sheet.facts}


def test_reference_facts_frame_themselves_as_benchmark_rates() -> None:
    for f in build_fact_sheet(FIXTURES / "pass_single").facts:
        if f.id.startswith("ref."):
            assert "planted-signal benchmark" in f.label, f.id


def test_a_drifted_report_row_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "results"
    shutil.copytree(default_results_root(), root)
    path = root / "0010-causal-stage-validation/report.json"
    doc = json.loads(path.read_text())
    doc["presets"][1], doc["presets"][2] = doc["presets"][2], doc["presets"][1]
    path.write_text(json.dumps(doc))
    with pytest.raises(MemoInputError, match="hidden_confounder"):
        build_fact_sheet(FIXTURES / "pass_single", results_root=root)
