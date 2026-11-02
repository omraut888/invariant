"""``invariant run`` end to end on a faked real-data snapshot (docs/adr/0018). Offline.

The panel is far below Stage 1's ``min_obs``, so every candidate is killed without a test
decision; the point is that the real-data path (signal panel, Stages 1-3, ``signals.json``,
memo and verifier) runs and the memo is accepted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.test_signals import FROZEN_IDS, _config, _snapshot

from invariant import cli
from invariant.config.settings import Settings
from invariant.stages.memo.facts import DataKind, Recommendation, build_fact_sheet


def test_invariant_run_writes_a_verified_real_data_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = _snapshot(tmp_path)
    config = _config().model_copy(update={"snapshot_id": snap.snapshot_id})
    settings = Settings(_env_file=None, data_dir=tmp_path, runs_dir=tmp_path / "runs")
    monkeypatch.setattr(cli, "load_config", lambda **_: config)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_snapshot", lambda sid: snap)
    assert cli.main(["run", "--config", "unused.yaml", "--allow-dirty"]) == 0
    (run,) = (tmp_path / "runs").iterdir()
    docs = json.loads((run / "signals.json").read_text())["signals"]
    assert [d["id"] for d in docs] == ["sig_" + i for i in FROZEN_IDS]
    sheet = build_fact_sheet(run)
    assert sheet.data_kind is DataKind.REAL
    assert sheet.recommendation is Recommendation.NOT_DECIDED  # below min_obs
    assert (run / "memo" / "memo.md").is_file()


def test_invariant_run_refuses_an_unpinned_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_config", lambda **_: _config())
    with pytest.raises(SystemExit, match="pinned snapshot_id"):
        cli.main(["run", "--config", "unused.yaml"])


def _max_lag(config: object) -> object:
    data = config.data  # type: ignore[attr-defined]
    market = data.fred_market.model_copy(update={"lag_rule": "max_lag"})
    return config.model_copy(update={"data": data.model_copy(update={"fred_market": market})})  # type: ignore[attr-defined]


def test_invariant_sensitivity_attaches_the_max_lag_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = _snapshot(tmp_path)
    primary_cfg = _config().model_copy(update={"snapshot_id": snap.snapshot_id})
    settings = Settings(_env_file=None, data_dir=tmp_path, runs_dir=tmp_path / "runs")
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_snapshot", lambda sid: snap)
    runs = {}
    for name, cfg in (("primary", primary_cfg), ("sensitivity", _max_lag(primary_cfg))):
        before = set((tmp_path / "runs").glob("*")) if (tmp_path / "runs").exists() else set()
        monkeypatch.setattr(cli, "load_config", lambda c=cfg, **_: c)
        assert cli.main(["run", "--config", "unused.yaml", "--allow-dirty"]) == 0
        (runs[name],) = set((tmp_path / "runs").glob("*")) - before
    with pytest.raises(SystemExit, match="lag rules"):
        cli.attach_sensitivity(runs["sensitivity"], runs["primary"])
    md = cli.attach_sensitivity(runs["primary"], runs["sensitivity"])
    assert "no result here depends on that choice of posting lag" in md.read_text()
    assert build_fact_sheet(runs["primary"]).sensitivity_differs == ()
    with pytest.raises(SystemExit, match="attached once"):
        cli.attach_sensitivity(runs["primary"], runs["sensitivity"])
