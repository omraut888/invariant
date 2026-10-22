from __future__ import annotations

import math

import pytest

from invariant.stages.causal.calibration import required_seeds
from invariant.stages.decay.halflife import Estimator
from invariant.stages.decay.study import (
    FitRecord,
    SeedRecord,
    SignalRecord,
    decay_scenarios,
    ksg_budget,
    load_decay_study_config,
    run_seed,
    summarize,
)


def test_preregistered_config_and_seed_count() -> None:
    cfg = load_decay_study_config()
    assert cfg.root_seed == 20261021
    assert cfg.n_gating == 6
    assert required_seeds(0.05, 0.075, cfg.n_gating, 0.90) == 1610
    names = {s.name for s in decay_scenarios()}
    gated = {*cfg.kill_gates, *cfg.estimator_gates, *cfg.coverage_gates, cfg.selection_scenario}
    assert gated <= names
    by = {s.name: s for s in decay_scenarios()}
    assert Estimator.KSG in by["d2_ident"].estimators
    assert by["d2_ident"].config.decay.half_life == 500.0
    assert by["k_rho99"].config.real.rho == 0.99


def _rec(name: str, i: int, arm: str, seconds: float = 1.0) -> SeedRecord:
    return SeedRecord.model_validate(
        {
            "scenario": name,
            "index": i,
            "arm": arm,
            "dataset_seed": i,
            "te_half_life": 125.0,
            "signals": {},
            "seconds": seconds,
        }
    )


def test_ksg_budget_rule_is_mechanical() -> None:
    recs = [_rec(s, i, "ksg", 100.0) for s in ("a", "b", "c") for i in range(8)]
    recs += [_rec("a", i, "main", 1e6) for i in range(8)]  # main-arm time is not counted
    within = ksg_budget(recs, budget_hours=6.0, workers=8, required_seeds=500)
    assert within.seconds_per_seed == 300.0
    assert within.projected_hours == pytest.approx(300 * 500 / 8 / 3600)
    assert within.full and within.ksg_n_seeds == 500
    over = ksg_budget(recs, budget_hours=6.0, workers=8, required_seeds=1610)
    assert not over.full
    assert over.ksg_n_seeds == math.floor(6 * 3600 * 8 / 300) == 576


def _sig(decline_p: float, fits: dict[str, tuple[float, float, float]]) -> SignalRecord:
    return SignalRecord(
        best_lag=1,
        n_conditions=3,
        family_p=0.001,
        stage1_pass=True,
        decline_p=decline_p,
        decline_stat=0.0,
        break_fraction=0.5,
        trend_p=0.5,
        fits={
            k: FitRecord(lam_hat=a, lam_low=b, lam_high=c, tau_hat=0)
            for k, a, b, c in ((k, *v) for k, v in fits.items())
        },
    )


def test_summarize_gates_and_simplicity_tie_break() -> None:
    cfg = load_decay_study_config().model_copy(update={"n_rmse_boot": 200})
    scenarios = decay_scenarios()
    lam = math.log(2) / 125.0
    recs: list[SeedRecord] = []
    for sc in scenarios:
        for i in range(400):
            real_fits = {e.value: (0.0, 0.0, 0.1) for e in sc.estimators if e is not Estimator.KSG}
            decay_fits = {
                e.value: (lam * (1.0 + 0.1 * ((i % 3) - 1)), lam / 2, lam * 2)
                for e in sc.estimators
                if e is not Estimator.KSG
            }
            recs.append(
                SeedRecord(
                    scenario=sc.name,
                    index=i,
                    arm="main",
                    dataset_seed=i,
                    te_half_life=125.0,
                    signals={
                        "s_real": _sig(0.5 if i % 50 else 0.01, real_fits),  # 2% false decay
                        "s_decay": _sig(0.01, decay_fits),
                    },
                    seconds=1.0,
                )
            )
    report = summarize(recs, scenarios, cfg, git_sha=None, git_dirty=None)
    assert report.kill_test_passes
    assert not report.ksg_eligible and report.ksg_n_seeds == 0
    assert report.admissible == {"ksg": False, "gaussian_te": True, "partial_corr": True}
    # identical errors: tied, so the simpler (c) wins
    assert report.selected is Estimator.PARTIAL_CORR


def test_run_seed_is_deterministic() -> None:
    cfg = load_decay_study_config().model_copy(
        update={
            "root_seed": 1,
            "n_obs": 1500,
            "decline": load_decay_study_config().decline.model_copy(update={"n_boot": 99}),
        }
    )
    sc = next(s for s in decay_scenarios() if s.name == "k_rho99")
    a, b = run_seed(sc, cfg, 0, "main"), run_seed(sc, cfg, 0, "main")
    assert a.model_dump(exclude={"seconds"}) == b.model_dump(exclude={"seconds"})
    assert set(a.signals["s_decay"].fits) == {"partial_corr", "gaussian_te"}
    other = run_seed(sc, cfg, 1, "main")
    assert other.dataset_seed != a.dataset_seed


@pytest.mark.parametrize("arm", ["main", "ksg"])
def test_every_arm_runs_end_to_end_on_a_tiny_seed(arm: str) -> None:
    """Every estimator arm, KSG included, through the study's own code path. The empty-Z KSG
    crash of the first full run (docs/adr/0011) was in an arm the fast suite never ran."""
    base = load_decay_study_config()
    cfg = base.model_copy(
        update={
            "root_seed": 1,
            "n_obs": 1000,
            "decline": base.decline.model_copy(update={"n_boot": 99}),
        }
    )
    sc = next(s for s in decay_scenarios() if s.name == "d2_ident")
    rec = run_seed(sc, cfg, 0, arm)  # type: ignore[arg-type]
    want = {"ksg"} if arm == "ksg" else {"partial_corr", "gaussian_te"}
    for sig in ("s_real", "s_decay"):
        assert set(rec.signals[sig].fits) == want
