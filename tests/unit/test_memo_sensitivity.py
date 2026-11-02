"""The lag-rule sensitivity run in the memo (docs/adr/0018, "Sensitivity run")."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from invariant.config.settings import PipelineConfig, Settings
from invariant.core.run_hash import GitState, RunContext, canonical_json
from invariant.core.types import KillReason, Verdict
from invariant.ledger.kill_ledger import HEAD_NAME, LEDGER_NAME, KillLedger
from invariant.stages.memo.corruption import apply_ops
from invariant.stages.memo.document import Paragraph, render
from invariant.stages.memo.facts import SENSITIVITY_DIR, MemoInputError, build_fact_sheet
from invariant.stages.memo.template import build_template_memo
from invariant.stages.memo.verify import verify

FIXTURE = Path(__file__).parents[1] / "fixtures" / "memo_runs" / "real_shaped"


def _primary(tmp: Path) -> Path:
    run = tmp / "primary"
    shutil.copytree(FIXTURE, run)
    for f in run.iterdir():
        f.chmod(0o644)
    return run


def _sensitivity(tmp: Path, primary: Path, *, flip: str | None, seed_offset: int = 0) -> Path:
    """A completed run on the primary's snapshot and seed whose ledger repeats the primary's,
    except that ``flip`` (if given) is killed at Stage 3."""
    manifest = json.loads((primary / "manifest.json").read_text())
    inputs = manifest["inputs"]
    config = PipelineConfig.model_validate(inputs["config"])
    # A different config (so a different run id) with the same snapshot and seed.
    config = config.model_copy(
        update={
            "seed": config.seed + seed_offset,
            "decay": config.decay.model_copy(update={"alpha": 0.049}),
        }
    )
    settings = Settings(_env_file=None, data_dir=tmp / "data", runs_dir=tmp / "runs")
    ctx = RunContext.create(
        config,
        inputs["snapshot_id"],
        settings=settings,
        # The fact sheet compares snapshot and seed; `invariant sensitivity` also checks code.
        git_state=GitState(sha=inputs["git"]["sha"], dirty=False),
    )
    ledger = KillLedger.open(ctx)
    for e in KillLedger.read(primary).entries():
        kill = e.signal_id == flip and e.stage.value == "portfolio"
        ledger.record(
            e.signal_id,
            e.stage,
            Verdict.KILL if kill else e.verdict,
            metrics=e.metrics,
            reason=KillReason.PORTFOLIO_NO_EDGE_OVER_EQUAL_WEIGHT if kill else e.reason,
        )
    ctx.mark_completed()
    dest = primary / SENSITIVITY_DIR
    dest.mkdir()
    for name in ("manifest.json", LEDGER_NAME, HEAD_NAME):
        shutil.copy2(ctx.run_dir / name, dest / name)
    return dest


def _memo(run: Path) -> tuple[str, str, list[str]]:
    sheet = build_fact_sheet(run)
    memo = build_template_memo(sheet)
    ids = [b.statement for s in memo.sections for b in s.blocks if isinstance(b, Paragraph)]
    return canonical_json(memo.model_dump(mode="json")), render(memo, sheet).text, ids  # type: ignore[return-value]


def test_without_a_sensitivity_run_neither_statement_appears() -> None:
    sheet = build_fact_sheet(FIXTURE)
    assert sheet.sensitivity_differs is None
    _, _, ids = _memo(FIXTURE)
    assert "lag_sensitivity" not in ids and "lag_sensitivity_agrees" not in ids


def test_an_agreeing_sensitivity_run_is_stated(tmp_path: Path) -> None:
    run = _primary(tmp_path)
    _sensitivity(tmp_path, run, flip=None)
    memo_json, md, ids = _memo(run)
    assert build_fact_sheet(run).sensitivity_differs == ()
    assert "lag_sensitivity_agrees" in ids and "lag_sensitivity" not in ids
    assert "no result here depends on that choice of posting lag" in md
    assert verify(memo_json, md, run).accepted


def test_a_differing_signal_is_named_and_verified(tmp_path: Path) -> None:
    run = _primary(tmp_path)
    _sensitivity(tmp_path, run, flip="sig_s_conf")
    memo_json, md, ids = _memo(run)
    assert build_fact_sheet(run).sensitivity_differs == ("sig_s_conf",)
    assert "lag_sensitivity" in ids and "lag_sensitivity_agrees" not in ids
    assert "For sig_s_conf, the outcome there differs" in md
    assert "depends on the posting-lag assumption" in md
    report = verify(memo_json, md, run)
    assert report.accepted, report.failures


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("For sig_s_conf, the outcome", "For sig_s_conf, sig_s_real, the outcome"),
        ("For sig_s_conf, the outcome", "For sig_s_real, the outcome"),
    ],
)
def test_misnaming_the_differing_signals_is_rejected(tmp_path: Path, old: str, new: str) -> None:
    run = _primary(tmp_path)
    _sensitivity(tmp_path, run, flip="sig_s_conf")
    sheet = build_fact_sheet(run)
    memo = build_template_memo(sheet)
    t = apply_ops(
        memo, render(memo, sheet).text, [{"op": "text_replace", "old": old, "new": new}], sheet
    )
    report = verify(t.memo_json, t.md, run)
    assert not report.accepted and "V6" in report.checks_failed()


def test_a_dropped_sensitivity_statement_is_rejected(tmp_path: Path) -> None:
    run = _primary(tmp_path)
    _sensitivity(tmp_path, run, flip="sig_s_conf")
    sheet = build_fact_sheet(run)
    memo = build_template_memo(sheet)
    t = apply_ops(
        memo, render(memo, sheet).text, [{"op": "delete_statement", "id": "lag_sensitivity"}], sheet
    )
    assert not verify(t.memo_json, t.md, run).accepted


def test_a_sensitivity_run_on_another_seed_is_refused(tmp_path: Path) -> None:
    run = _primary(tmp_path)
    _sensitivity(tmp_path, run, flip=None, seed_offset=1)
    with pytest.raises(MemoInputError, match="seed"):
        build_fact_sheet(run)
