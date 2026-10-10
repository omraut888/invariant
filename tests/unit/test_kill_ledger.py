import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.fixtures.ledger import run_context, write_fates

from invariant.core.run_hash import RunContext, RunHashError, RunStateError
from invariant.core.types import KillReason, StageName, Verdict
from invariant.ledger.kill_ledger import (
    GENESIS_SHA256,
    HEAD_NAME,
    LEDGER_NAME,
    STAGE_ORDER,
    FateStatus,
    KillLedger,
    LedgerEntry,
    LedgerError,
    LedgerIntegrityError,
    check_stage_prefix,
)

CAUSAL, DECAY, DRO = StageName.CAUSAL, StageName.DECAY, StageName.DRO
FIXED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _ledger(tmp_path: Path) -> tuple[RunContext, KillLedger]:
    ctx = run_context(tmp_path)
    return ctx, KillLedger.open(ctx, now=lambda: FIXED)


def _kill(ledger: KillLedger, sid: str, stage: StageName = CAUSAL) -> LedgerEntry:
    return ledger.record(
        sid, stage, Verdict.KILL, reason=KillReason.INSUFFICIENT_DATA, metrics={"n": 10}
    )


def _lines(ctx: RunContext) -> list[str]:
    return (ctx.run_dir / LEDGER_NAME).read_text().splitlines()


def _rewrite(ctx: RunContext, lines: list[str], *, newline: bool = True) -> None:
    (ctx.run_dir / LEDGER_NAME).write_text("\n".join(lines) + ("\n" if newline else ""))


# --- recording and reading --------------------------------------------------------------------


def test_round_trip_and_chain(tmp_path: Path) -> None:
    ctx, ledger = _ledger(tmp_path)
    first = ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={"te": 0.25, "n": 100})
    second = _kill(ledger, "s_a", DECAY)
    assert first.seq == 0 and first.prev_sha256 == GENESIS_SHA256
    assert second.seq == 1 and second.prev_sha256 != GENESIS_SHA256
    assert first.run_id == ctx.run_id and first.attempt == 1
    assert first.recorded_at == FIXED
    reread = KillLedger.read(ctx.run_dir)
    assert reread.all_entries() == (first, second)


def test_stored_entry_is_canonical(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    entry = ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={"x": -0.0, "flag": True, "k": 3})
    assert entry.metrics == {"x": 0.0, "flag": True, "k": 3}
    assert type(entry.metrics["flag"]) is bool and type(entry.metrics["k"]) is int


def test_empty_ledger(tmp_path: Path) -> None:
    ctx, ledger = _ledger(tmp_path)
    assert ledger.entries() == ()
    assert KillLedger.read(ctx.run_dir).fates() == {}


def test_read_requires_a_run_manifest(tmp_path: Path) -> None:
    with pytest.raises(RunHashError, match="manifest"):
        KillLedger.read(tmp_path)


# --- decision rules ---------------------------------------------------------------------------


def test_kill_needs_reason_and_metrics(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    with pytest.raises(LedgerError, match="reason"):
        ledger.record("s_a", CAUSAL, Verdict.KILL, metrics={"n": 1})
    with pytest.raises(LedgerError, match="metrics"):
        ledger.record("s_a", CAUSAL, Verdict.KILL, reason=KillReason.INSUFFICIENT_DATA, metrics={})
    with pytest.raises(LedgerError, match="PASS has no reason"):
        ledger.record("s_a", CAUSAL, Verdict.PASS, reason=KillReason.INSUFFICIENT_DATA, metrics={})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_metrics_must_be_finite(tmp_path: Path, bad: float) -> None:
    _, ledger = _ledger(tmp_path)
    with pytest.raises(LedgerError, match="finite"):
        ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={"x": bad})


def test_invalid_signal_id_rejected(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    with pytest.raises(LedgerError, match="signal id"):
        ledger.record("Bad Id", CAUSAL, Verdict.PASS, metrics={})


def test_stage_order_duplicates_and_after_kill(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    with pytest.raises(LedgerError, match="must be at causal"):
        ledger.record("s_a", DECAY, Verdict.PASS, metrics={})
    ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={})
    with pytest.raises(LedgerError, match="must be at decay"):
        ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={})  # duplicate
    _kill(ledger, "s_a", DECAY)
    with pytest.raises(LedgerError, match="nothing may follow a kill"):
        ledger.record("s_a", DRO, Verdict.PASS, metrics={})
    assert len(ledger.entries()) == 2  # rejected decisions were not written


def test_all_stages_decided(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    write_fates(ledger, {"s_a": None})
    with pytest.raises(LedgerError, match="every stage"):
        ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={})


def test_placeholder_reason_requires_placeholder_flag(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    with pytest.raises(LedgerError, match="placeholder"):
        ledger.record("s_a", CAUSAL, Verdict.KILL, reason=KillReason.PLACEHOLDER, metrics={"x": 1})
    entry = ledger.record(
        "s_a",
        CAUSAL,
        Verdict.KILL,
        reason=KillReason.PLACEHOLDER,
        metrics={"x": 1},
        placeholder=True,
    )
    assert entry.placeholder


def test_placeholder_flag_is_detected_on_pass_entries(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={}, placeholder=True)
    assert ledger.contains_placeholders()
    _, other = _ledger(tmp_path / "other")
    other.record("s_a", CAUSAL, Verdict.PASS, metrics={})
    assert not other.contains_placeholders()


def test_placeholder_flag_is_hashed(tmp_path: Path) -> None:
    ctx, ledger = _ledger(tmp_path)
    ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={}, placeholder=True)
    ledger.record("s_b", CAUSAL, Verdict.PASS, metrics={})
    lines = _lines(ctx)
    lines[0] = lines[0].replace('"placeholder":true', '"placeholder":false')
    _rewrite(ctx, lines)
    with pytest.raises(LedgerIntegrityError, match="hash chain broken at line 1"):
        KillLedger.read(ctx.run_dir)


# --- run lifecycle ----------------------------------------------------------------------------


def test_closed_run_rejects_writes(tmp_path: Path) -> None:
    ctx, ledger = _ledger(tmp_path)
    ctx.mark_completed()
    with pytest.raises(RunStateError, match="closed"):
        ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={})


def test_read_only_ledger_rejects_writes(tmp_path: Path) -> None:
    ctx, _ = _ledger(tmp_path)
    with pytest.raises(RunStateError, match="read-only"):
        KillLedger.read(ctx.run_dir).record("s_a", CAUSAL, Verdict.PASS, metrics={})


def test_resumed_attempt_is_separate(tmp_path: Path) -> None:
    ctx, first = _ledger(tmp_path)
    _kill(first, "s_a")
    write_fates(first, {"s_b": None}, stop_before={"s_b": DRO})  # crashed mid-run
    ctx.mark_failed("boom")
    resumed = run_context(tmp_path)  # same inputs: attempt 2
    with pytest.raises(RunStateError, match="attempt 2"):
        first.record("s_c", CAUSAL, Verdict.PASS, metrics={})  # stale handle
    second = KillLedger.open(resumed)
    write_fates(second, {"s_a": None, "s_b": CAUSAL})  # s_a may be re-decided in attempt 2
    reread = KillLedger.read(ctx.run_dir)
    assert reread.attempts() == (1, 2)
    assert reread.what_killed("s_a") is None  # latest attempt by default
    assert reread.what_killed("s_a", attempt=1) is not None
    assert reread.fates()["s_b"].status is FateStatus.KILLED
    assert reread.fates(attempt=1)["s_b"].status is FateStatus.INCOMPLETE


# --- tamper evidence --------------------------------------------------------------------------


def _three(tmp_path: Path) -> RunContext:
    ctx, ledger = _ledger(tmp_path)
    write_fates(ledger, {"s_a": DECAY})
    _kill(ledger, "s_b")
    return ctx


def test_edited_line_detected(tmp_path: Path) -> None:
    ctx = _three(tmp_path)
    lines = _lines(ctx)
    lines[1] = lines[1].replace('"p_value":0.2', '"p_value":0.9')
    _rewrite(ctx, lines)
    with pytest.raises(LedgerIntegrityError, match="line 2"):
        KillLedger.read(ctx.run_dir)


def test_edited_last_line_detected(tmp_path: Path) -> None:
    # No successor line carries its hash; the edit must still be caught. Changing a kill to
    # a pass breaks the entry's own rules (a PASS with a reason).
    ctx = _three(tmp_path)
    lines = _lines(ctx)
    lines[-1] = lines[-1].replace('"verdict":"kill"', '"verdict":"pass"')
    _rewrite(ctx, lines)
    with pytest.raises(LedgerIntegrityError, match="line 2"):
        KillLedger.read(ctx.run_dir)


@pytest.mark.parametrize("drop", [0, 1])
def test_deleted_line_detected(tmp_path: Path, drop: int) -> None:
    ctx = _three(tmp_path)
    lines = _lines(ctx)
    del lines[drop]
    _rewrite(ctx, lines)
    with pytest.raises(LedgerIntegrityError):
        KillLedger.read(ctx.run_dir)


def test_reordered_lines_detected(tmp_path: Path) -> None:
    ctx = _three(tmp_path)
    lines = _lines(ctx)
    lines[1], lines[2] = lines[2], lines[1]
    _rewrite(ctx, lines)
    with pytest.raises(LedgerIntegrityError):
        KillLedger.read(ctx.run_dir)


def test_partial_last_line_detected(tmp_path: Path) -> None:
    ctx = _three(tmp_path)
    data = (ctx.run_dir / LEDGER_NAME).read_text()
    (ctx.run_dir / LEDGER_NAME).write_text(data[:-20])
    with pytest.raises(LedgerIntegrityError, match="partial last line"):
        KillLedger.read(ctx.run_dir)


def test_non_canonical_line_detected(tmp_path: Path) -> None:
    ctx = _three(tmp_path)
    lines = _lines(ctx)
    lines[0] = json.dumps(json.loads(lines[0]), indent=1).replace("\n", "")
    _rewrite(ctx, lines)
    with pytest.raises(LedgerIntegrityError, match="canonical"):
        KillLedger.read(ctx.run_dir)


def test_ledger_of_another_run_detected(tmp_path: Path) -> None:
    ctx = _three(tmp_path)
    other = run_context(tmp_path, snapshot_id="snap-2")
    (other.run_dir / LEDGER_NAME).write_bytes((ctx.run_dir / LEDGER_NAME).read_bytes())
    with pytest.raises(LedgerIntegrityError, match="belongs to run"):
        KillLedger.read(other.run_dir)


def test_entry_requires_aware_timestamp(tmp_path: Path) -> None:
    ctx = run_context(tmp_path)
    ledger = KillLedger.open(ctx, now=lambda: datetime(2026, 1, 1))  # naive
    with pytest.raises(LedgerError, match="timezone"):
        ledger.record("s_a", CAUSAL, Verdict.PASS, metrics={})
    aware = KillLedger.open(ctx, now=lambda: datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=5))
    assert aware.record("s_a", CAUSAL, Verdict.PASS, metrics={}).recorded_at.tzinfo is UTC


# --- queries ----------------------------------------------------------------------------------


def test_what_killed_and_counts(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    write_fates(ledger, {"s_a": DECAY, "s_b": CAUSAL, "s_c": None, "s_d": CAUSAL})
    killer = ledger.what_killed("s_a")
    assert killer is not None and killer.stage is DECAY
    assert killer.reason is KillReason.INSUFFICIENT_DATA and killer.metrics["p_value"] == 0.2
    assert ledger.what_killed("s_c") is None
    assert ledger.what_killed("unknown") is None
    counts = ledger.kill_counts_by_stage()
    assert list(counts) == list(STAGE_ORDER)
    assert counts == {CAUSAL: 2, DECAY: 1, DRO: 0, StageName.ADVERSARIAL: 0, StageName.MEMO: 0}


def test_fates_over_stage_prefix(tmp_path: Path) -> None:
    _, ledger = _ledger(tmp_path)
    write_fates(ledger, {"s_a": DECAY, "s_b": None}, stop_before={"s_b": DRO})
    full = ledger.fates()
    assert full["s_a"].status is FateStatus.KILLED and full["s_a"].stage is DECAY
    assert full["s_b"].status is FateStatus.INCOMPLETE and full["s_b"].stage is DECAY
    causal_only = ledger.fates([CAUSAL])
    assert causal_only["s_a"].status is FateStatus.PASSED  # its kill is outside the prefix
    assert causal_only["s_b"].status is FateStatus.PASSED


@pytest.mark.parametrize("stages", [[], [DECAY], [CAUSAL, DRO]])
def test_stage_prefix_validation(stages: list[StageName]) -> None:
    with pytest.raises(ValueError, match="prefix"):
        check_stage_prefix(stages)


_fate = st.one_of(st.none(), st.sampled_from(STAGE_ORDER))


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(st.dictionaries(st.sampled_from(["s_a", "s_b", "s_c", "s_d", "s_e"]), _fate, min_size=1))
def test_written_fates_are_read_back(
    tmp_path_factory: pytest.TempPathFactory, fates: dict[str, StageName | None]
) -> None:
    ctx = run_context(tmp_path_factory.mktemp("ledger"))
    write_fates(KillLedger.open(ctx), fates)
    reread = KillLedger.read(ctx.run_dir)
    got = reread.fates()
    for sid, kill_stage in fates.items():
        if kill_stage is None:
            assert got[sid].status is FateStatus.PASSED
        else:
            assert got[sid].status is FateStatus.KILLED and got[sid].stage is kill_stage
    assert sum(reread.kill_counts_by_stage().values()) == sum(s is not None for s in fates.values())


def test_valid_edit_of_last_line_detected_by_head(tmp_path: Path) -> None:
    # The edited line is still a valid entry and nothing after it carries its hash.
    ctx = _three(tmp_path)
    lines = _lines(ctx)
    lines[-1] = lines[-1].replace('"n":10', '"n":11')
    _rewrite(ctx, lines)
    with pytest.raises(LedgerIntegrityError, match="does not match the head"):
        KillLedger.read(ctx.run_dir)


def test_truncation_at_line_boundary_detected_by_head(tmp_path: Path) -> None:
    ctx = _three(tmp_path)
    _rewrite(ctx, _lines(ctx)[:-1])
    with pytest.raises(LedgerIntegrityError, match="head records 3"):
        KillLedger.read(ctx.run_dir)


def test_missing_head_detected(tmp_path: Path) -> None:
    ctx = _three(tmp_path)
    (ctx.run_dir / HEAD_NAME).unlink()
    with pytest.raises(LedgerIntegrityError, match="no head"):
        KillLedger.read(ctx.run_dir)
