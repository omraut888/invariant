"""The Phase 0 smoke script runs end to end offline, and a rerun resumes the same run."""

import json
import runpy
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import cast

import pytest
import structlog

from invariant.core.run_hash import RunManifest
from invariant.ledger.benchmark_score import BENCHMARK_SCORE_NAME, BenchmarkScore
from invariant.ledger.kill_ledger import KillLedger

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "phase0_smoke.py"


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    # The script configures structlog to write to the (captured) stdout of this test. Restore
    # the previous configuration so later tests do not log to a closed stream.
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)


def _main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(str(SCRIPT), run_name="phase0_smoke")
    return cast(Callable[[Sequence[str] | None], int], namespace["main"])


def test_smoke_runs_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main = _main()
    args = [
        "--n-obs",
        "300",
        "--data-dir",
        str(tmp_path / "data"),
        "--runs-dir",
        str(tmp_path / "runs"),
    ]
    assert main(args) == 0
    printed = capsys.readouterr().out
    manifest = RunManifest.model_validate(json.loads(printed[: printed.index("\n}\n") + 2]))
    assert manifest.status == "completed" and manifest.attempt == 1
    run_dir = tmp_path / "runs" / manifest.run_id
    score = BenchmarkScore.model_validate_json((run_dir / BENCHMARK_SCORE_NAME).read_text())
    assert score.contains_placeholders and score.complete
    assert all(e.placeholder for e in KillLedger.read(run_dir).all_entries())

    assert main(args) == 0  # same inputs: resumes as attempt 2, appends, scores attempt 2
    rerun = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    assert rerun.attempt == 2 and rerun.status == "completed"
    assert KillLedger.read(run_dir).attempts() == (1, 2)


@pytest.mark.parametrize(
    ("preset", "features"),
    [
        (None, "syn_s_real, syn_s_conf, syn_s_decay, syn_s_noise, syn_z"),
        ("hidden_confounder", "syn_s_real, syn_s_conf, syn_s_decay, syn_s_noise"),
    ],
)
def test_smoke_panel_has_observed_features(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], preset: str | None, features: str
) -> None:
    args = ["--n-obs", "300", "--data-dir", str(tmp_path / "d"), "--runs-dir", str(tmp_path / "r")]
    if preset is not None:
        args += ["--preset", preset]
    assert _main()(args) == 0
    assert f"observed features in panel: {features}\n" in capsys.readouterr().out
