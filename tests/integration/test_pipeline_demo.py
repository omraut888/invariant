"""The Stages 1-3 demo (scripts/pipeline_demo.py) on seed 0, the README's worked example: every
signal meets the fate GroundTruth expects, at the stage it expects, and the README's printed
numbers are the ones this run produces."""

from __future__ import annotations

import runpy
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import cast

import pytest
import structlog

from invariant.core.types import KillReason, StageName, Verdict
from invariant.ledger.benchmark_score import BENCHMARK_SCORE_NAME, BenchmarkScore
from invariant.ledger.kill_ledger import KillLedger

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pipeline_demo.py"


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)


def test_seed_0_kills_each_bad_signal_at_its_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    namespace = runpy.run_path(str(SCRIPT), run_name="pipeline_demo")
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])
    args = [
        "--seed",
        "0",
        "--data-dir",
        str(tmp_path / "data"),
        "--runs-dir",
        str(tmp_path / "runs"),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    (run_dir,) = (tmp_path / "runs").iterdir()
    score = BenchmarkScore.model_validate_json((run_dir / BENCHMARK_SCORE_NAME).read_text())
    assert score.complete and not score.contains_placeholders
    m = score.matrix
    assert (m.correct_kill, m.wrong_kill, m.correct_pass, m.missed_kill) == (3, 0, 1, 0)

    ledger = KillLedger.read(run_dir)
    kills = {sid: ledger.what_killed(sid) for sid in ledger.signal_ids()}
    assert kills["s_real"] is None
    s_decay = kills["s_decay"]
    assert s_decay is not None and s_decay.stage is StageName.DECAY
    assert s_decay.reason is KillReason.DECAY_EDGE_DECLINING
    for sid in ("s_conf", "s_noise"):
        entry = kills[sid]
        assert entry is not None and entry.stage is StageName.CAUSAL
    passed = [e for e in ledger.entries() if e.signal_id == "s_real"]
    assert [e.stage for e in passed] == [StageName.CAUSAL, StageName.DECAY, StageName.PORTFOLIO]
    assert all(e.verdict is Verdict.PASS for e in passed)
    # The numbers quoted in the README's example.
    assert "decline p 0.001 vs alpha 0.05" in out
    assert "coefficient before/after the break 0.00154 / -6.36e-05" in out
    assert "Ridge-CV Sharpe 1.1 (net 0.843) vs 1/N 0.183; LW p 0.001" in out
