from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from tests.fixtures.causal import benchmark_panel, fast_causal

from invariant.config.settings import CausalConfig, load_config
from invariant.core.run_hash import RngFactory
from invariant.core.types import KillReason, Verdict
from invariant.data.pit import PitPanel
from invariant.stages.causal.multiple_testing import fdr_qvalues
from invariant.stages.causal.result import CausalResult, SignalCausalLink
from invariant.stages.causal.stage import (
    decide,
    ledger_metrics,
    run_causal_stage,
    select_inputs,
    signal_id_of,
    write_causal_result,
)

NO_LINK = KillReason.CAUSAL_NO_SIGNIFICANT_LINK
AFTER = KillReason.CAUSAL_NOT_SIGNIFICANT_AFTER_CORRECTION


@pytest.mark.parametrize(
    ("family_p", "q", "verdict", "reason"),
    [
        (0.01, 0.02, Verdict.PASS, None),
        (0.01, 0.05, Verdict.KILL, AFTER),  # a tie at alpha is a kill
        (0.04, 0.08, Verdict.KILL, AFTER),
        (0.05, 0.05, Verdict.KILL, NO_LINK),
        (0.30, 0.60, Verdict.KILL, NO_LINK),
    ],
)
def test_decide(family_p: float, q: float, verdict: Verdict, reason: KillReason | None) -> None:
    assert decide(family_p, q, 0.05) == (verdict, reason)


def _frame(n: int = 20) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "ret_AAA": rng.standard_normal(n),
            "syn_s_a": rng.standard_normal(n),
            "syn_z": rng.standard_normal(n),
            "macro_X": rng.standard_normal(n),
        },
        index=idx,
    )
    df.iloc[0, 0] = np.nan
    return df


def _cfg(**update: object) -> CausalConfig:
    base = load_config().causal
    return CausalConfig.model_validate({**base.model_dump(), **update})


def test_select_inputs_columns_and_trim() -> None:
    df = _frame()
    df["fwd_AAA"] = 1.0  # never selected, even if a target column reaches the features
    got = select_inputs(df, _cfg())
    assert got.return_columns == ("ret_AAA",)
    assert got.feature_columns == ("syn_s_a", "syn_z", "macro_X")
    assert got.signal_columns == got.feature_columns
    assert "fwd_AAA" not in got.frame.columns
    assert len(got.frame) == 19 and got.interior_missing == ()

    only = select_inputs(df, _cfg(signals=("s_a", "macro_x")))
    assert only.signal_columns == ("syn_s_a", "macro_X")
    assert "syn_z" in only.frame.columns  # still a conditioning variable
    assert [signal_id_of(c) for c in only.signal_columns] == ["s_a", "macro_x"]


def test_colliding_signal_ids_rejected() -> None:
    df = _frame()
    df["s_a"] = 0.0  # same id as syn_s_a
    with pytest.raises(ValueError, match="duplicate ids"):
        select_inputs(df, _cfg())


def test_select_inputs_errors_and_interior_missing() -> None:
    df = _frame()
    with pytest.raises(ValueError, match="not feature columns"):
        select_inputs(df, _cfg(signals=("nope",)))
    with pytest.raises(ValueError, match="no return"):
        select_inputs(df.drop(columns="ret_AAA"), _cfg())
    df.iloc[5, 1] = np.nan
    assert select_inputs(df, _cfg()).interior_missing == ("syn_s_a",)


def test_config_rejects_unvalidated_tests_and_bad_values() -> None:
    for bad in (
        {"ci_test": "parcorr"},
        {"ci_test": "parcorr_wls"},
        {"decision": "parents_r"},
        {"signals": ()},
        {"signals": ("a", "a")},
        {"min_obs": 50, "tau_max": 20},
    ):
        with pytest.raises(ValidationError):
            _cfg(**bad)


@pytest.fixture(scope="module")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[PitPanel, CausalResult]:
    bp = benchmark_panel(tmp_path_factory.mktemp("default"))
    return bp.panel, run_causal_stage(bp.panel, bp.config.causal, RngFactory(1))


def test_result_numbers_are_consistent(default_run: tuple[PitPanel, CausalResult]) -> None:
    panel, result = default_run
    cfg = result.config
    assert result.tested and result.n_obs == len(panel.features) - 1  # ret_* row 0 is missing
    assert result.n_rows_tested == result.n_obs - 2 * cfg.tau_max
    assert [s.signal_id for s in result.signals] == ["s_real", "s_conf", "s_decay", "s_noise"]
    assert "syn_z" in result.var_names and "syn_z" not in result.signal_columns
    fam = []
    for link in result.signals:
        ev = link.evidence
        assert ev is not None
        assert ev.n_family_tests == cfg.tau_max * len(result.return_columns)
        ps = [t.p for t in ev.tests]
        assert ev.family_p == pytest.approx(min(1.0, len(ps) * min(ps)))
        assert ev.mci_family_p == pytest.approx(min(1.0, len(ps) * min(t.p_mci for t in ev.tests)))
        best = min(ev.tests, key=lambda t: t.p)
        assert (ev.best_asset, ev.best_lag) == (best.asset, best.lag)
        assert ev.r_best_ci.estimate == pytest.approx(best.r, abs=1e-10)
        assert ev.r_best_ci.low <= ev.r_best_ci.estimate <= ev.r_best_ci.high
        assert ev.r_best_ci.n_rows == result.n_rows_tested
        assert (link.verdict, link.reason) == decide(ev.family_p, ev.q, cfg.alpha)
        fam.append(ev.family_p)
    np.testing.assert_allclose(
        [s.evidence.q for s in result.signals if s.evidence], fdr_qvalues(fam, "bh")
    )


def test_planted_signal_found_on_default(default_run: tuple[PitPanel, CausalResult]) -> None:
    # One seed, not a rate (the rates are in the M = 500 validation, ADR 0010): s_real's
    # expected hybrid t at T = 1500 is ~2.2, so only its direction is checked here.
    _, result = default_run
    real = next(s for s in result.signals if s.signal_id == "s_real")
    assert real.evidence is not None and real.evidence.r_best_ci.estimate > 0


def test_ledger_metrics_are_flat_and_finite(default_run: tuple[PitPanel, CausalResult]) -> None:
    _, result = default_run
    for link in result.signals:
        m = ledger_metrics(link, result)
        assert {"family_p", "q", "mci_family_p", "best_lag", "r_best_ci_low"} <= set(m)
        assert link.evidence is not None
        assert m["p.SYN00.lag1"] == next(t.p for t in link.evidence.tests if t.lag == 1)
        for v in m.values():
            assert isinstance(v, (bool, int, float, str))
            if isinstance(v, float):
                assert np.isfinite(v)


def test_deterministic_given_seed(
    tmp_path: Path, default_run: tuple[PitPanel, CausalResult]
) -> None:
    panel, result = default_run
    again = run_causal_stage(panel, result.config, RngFactory(1))
    assert again == result
    other = run_causal_stage(panel, result.config, RngFactory(2))
    a, b = other.signals[0].evidence, result.signals[0].evidence
    assert a is not None and b is not None
    assert a.family_p == b.family_p  # HC3 is analytic: only the bootstrap uses the seed
    assert a.r_best_ci != b.r_best_ci
    path = write_causal_result(tmp_path, result)
    assert CausalResult.model_validate_json(path.read_bytes()) == result


def test_insufficient_and_invalid_data(default_run: tuple[PitPanel, CausalResult]) -> None:
    panel, result = default_run
    short = run_causal_stage(
        panel, result.config.model_copy(update={"min_obs": 10_000}), RngFactory(1)
    )
    assert not short.tested
    assert all(
        s.reason is KillReason.INSUFFICIENT_DATA and s.evidence is None for s in short.signals
    )
    m = ledger_metrics(short.signals[0], short)
    assert m["n_obs"] == result.n_obs and "data_issue" in m

    feats = panel.features.copy()
    feats.loc[feats.index[100], "syn_s_real"] = np.nan
    holed = PitPanel(feats, panel.obs_dates, panel.targets, panel.config)
    bad = run_causal_stage(holed, result.config, RngFactory(1))
    assert all(s.reason is KillReason.DATA_VALIDATION_FAILED for s in bad.signals)
    assert bad.data_issue is not None and "syn_s_real" in bad.data_issue


def test_multi_asset_family_spans_assets(tmp_path: Path) -> None:
    bp = benchmark_panel(tmp_path, "multi_asset", n_obs=800)
    result = run_causal_stage(bp.panel, bp.config.causal, RngFactory(3))
    assert len(result.return_columns) == 5
    for link in result.signals:
        assert link.evidence is not None and link.evidence.n_family_tests == 25


def test_contract_validators() -> None:
    with pytest.raises(ValidationError):
        SignalCausalLink(
            signal_id="s", column="syn_s", verdict=Verdict.PASS, reason=NO_LINK, evidence=None
        )
    with pytest.raises(ValidationError):  # a causal.* kill must carry its evidence
        SignalCausalLink(
            signal_id="s", column="syn_s", verdict=Verdict.KILL, reason=NO_LINK, evidence=None
        )
    with pytest.raises(ValidationError):
        SignalCausalLink(
            signal_id="s", column="syn_s", verdict=Verdict.PASS, reason=None, evidence=None
        )


def test_fast_causal_keeps_base_values() -> None:
    base = load_config().causal
    fast = fast_causal(base)
    assert fast.bootstrap.n_boot == 199 and fast.ci_test == base.ci_test
