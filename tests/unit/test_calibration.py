from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.ledger.benchmark_score import wilson_interval
from invariant.stages.causal.calibration import (
    STUDY2_GATES,
    BenchmarkData,
    CITestSpec,
    FamilyRole,
    LinkFamily,
    Scenario,
    SeedRecord,
    SizeStudyConfig,
    StudyReport,
    _max_passing_count,
    benchmark_scenarios,
    dataset_seed,
    load_size_study_config,
    null_config,
    pair_label,
    pass_probability,
    required_seeds,
    run_scenarios,
    summarize,
)
from invariant.stages.causal.ci_tests import CITestName
from invariant.stages.causal.pcmci import DecisionTest


def _cfg(**update: object) -> SizeStudyConfig:
    base = SizeStudyConfig(
        root_seed=7,
        n_obs=300,
        tau_max=2,
        pc_alpha=0.2,
        alpha=0.05,
        size_cutoff=0.075,
        all_pass_prob=0.9,
        tests=(
            CITestSpec(label="parcorr", test=CITestName.PARCORR),
            CITestSpec(label="wls10", test=CITestName.PARCORR_WLS, wls_window=10),
        ),
        selection_scenario="sc",
        selection_families=("alt",),
    )
    return base.model_copy(update=update)


# -- decision unit ----------------------------------------------------------------------------


def test_family_reject_is_bonferroni_over_lags() -> None:
    fam = LinkFamily(name="f", source="a", target="b", lags=(1, 2, 3, 4, 5), role=FamilyRole.NULL)
    assert fam.reject([0.5, 0.01, 0.5, 0.5, 0.5], 0.05)
    assert not fam.reject([0.5, 0.0101, 0.5, 0.5, 0.5], 0.05)
    one = LinkFamily(name="g", source="a", target="b", lags=(1,), role=FamilyRole.NULL)
    assert one.reject([0.05], 0.05) and not one.reject([0.0501], 0.05)
    with pytest.raises(ValueError, match="p-values"):
        fam.reject([0.1], 0.05)


# -- required M ---------------------------------------------------------------------------------


def test_required_seeds_for_study_rule() -> None:
    """The study's rule (cutoff 0.075, 7 gates, all-pass >= 0.9) needs M = 1654 (ADR 0009)."""
    m = required_seeds(0.05, 0.075, 7, 0.90)
    assert m == 1654
    per_family = 1 - 0.10 / 7
    assert all(pass_probability(k, 0.05, 0.075, 0.95) >= per_family for k in range(m, m + 201))
    # minimal: starting one earlier, the requirement fails somewhere in the window
    assert not all(
        pass_probability(k, 0.05, 0.075, 0.95) >= per_family for k in range(m - 1, m + 200)
    )


def test_rule_has_power_against_oversized_tests() -> None:
    m = required_seeds(0.05, 0.075, 7, 0.90)
    assert pass_probability(m, 0.10, 0.075, 0.95) < 1e-6


@given(st.integers(50, 3000), st.floats(0.01, 0.2), st.floats(0.01, 0.2))
def test_pass_probability_decreases_in_true_size(m: int, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert pass_probability(m, lo, 0.075, 0.95) >= pass_probability(m, hi, 0.075, 0.95) - 1e-12


@given(st.integers(20, 2000))
def test_max_passing_count_matches_wilson(m: int) -> None:
    k = _max_passing_count(m, 0.075, 0.95)
    if k >= 0:
        assert (wilson_interval(k, m).ci_high or 0.0) <= 0.075
    if k < m:
        assert (wilson_interval(k + 1, m).ci_high or 0.0) > 0.075


# -- seeds, scenarios ---------------------------------------------------------------------------


def test_dataset_seeds_deterministic_and_distinct() -> None:
    seeds = [dataset_seed(7, i) for i in range(200)]
    assert seeds == [dataset_seed(7, i) for i in range(200)]
    assert len(set(seeds)) == 200
    assert dataset_seed(8, 0) != dataset_seed(7, 0)


def test_null_config_removes_every_mean_effect() -> None:
    base = load_benchmark_config(None)
    cfg = null_config(base, garch=False)
    truth = generate_market(cfg.model_copy(update={"n_obs": 200, "burn_in": 10}), 1).ground_truth
    assert not any(s.has_causal_edge for s in truth.signals)
    assert cfg.garch.alpha == 0.0 and cfg.garch.beta == 0.0
    assert null_config(base, garch=True).garch == base.garch


def test_benchmark_scenarios_gating_families() -> None:
    gating = [
        f"{s.name}/{f.name}"
        for s in benchmark_scenarios(5)
        for f in s.families
        if f.role is FamilyRole.GATING_NULL
    ]
    assert sorted(gating) == sorted(
        [f"e2_garch_null/{s}->R" for s in ("s_real", "s_conf", "s_decay", "s_noise", "z")]
        + ["e3_default/s_conf->R"]
    )


def test_study_config_loads() -> None:
    cfg = load_size_study_config()
    assert cfg.size_cutoff == 0.075 and cfg.alpha == 0.05 and cfg.tau_max == 5


# -- parallel == serial ---------------------------------------------------------------------


def _tiny_scenario() -> Scenario:
    cfg = load_benchmark_config(None).model_copy(update={"burn_in": 20})
    return Scenario(
        "sc",
        "tiny",
        BenchmarkData(cfg),
        (
            LinkFamily(
                name="alt",
                source="s_real",
                target="asset_00",
                lags=(1, 2),
                role=FamilyRole.ALTERNATIVE,
            ),
            LinkFamily(
                name="null",
                source="s_noise",
                target="asset_00",
                lags=(1, 2),
                role=FamilyRole.GATING_NULL,
            ),
        ),
    )


def _strip(records: list[SeedRecord]) -> list[dict[str, object]]:
    return [r.model_dump(exclude={"seconds"}) for r in records]


def test_parallel_results_identical_to_serial(tmp_path: Path) -> None:
    sc, cfg = _tiny_scenario(), _cfg()
    serial = run_scenarios([sc], cfg, 4, n_jobs=1)
    parallel = run_scenarios([sc], cfg, 4, n_jobs=2, chunk_size=3)
    assert _strip(serial) == _strip(parallel)
    assert [r.index for r in serial] == [0, 1, 2, 3]

    ck = tmp_path / "records.jsonl"
    first = run_scenarios([sc], cfg, 2, n_jobs=1, checkpoint=ck)
    resumed = run_scenarios([sc], cfg, 4, n_jobs=2, checkpoint=ck)
    assert _strip(resumed) == _strip(serial)
    assert _strip(first) == _strip(serial[:2])
    assert len(ck.read_text().splitlines()) == 4


# -- summary rules ------------------------------------------------------------------------------


def _mci(label: str) -> str:
    return label if "|" in label else pair_label(label, DecisionTest.MCI)


def _records(
    n: int, null_reject: dict[str, int], alt_reject: dict[str, int], seconds: dict[str, float]
) -> list[SeedRecord]:
    """Hand-built records: ``lab`` rejects the null on the first ``null_reject[lab]`` seeds and
    the alternative on the first ``alt_reject[lab]`` seeds. A bare CI label means ``|mci``."""
    out = []
    for i in range(n):
        p_null = {_mci(lab): (0.001 if i < k else 0.9, 0.9) for lab, k in null_reject.items()}
        p_alt = {_mci(lab): (0.001 if i < k else 0.9, 0.9) for lab, k in alt_reject.items()}
        out.append(
            SeedRecord(
                scenario="sc",
                index=i,
                dataset_seed=i,
                p={"null": p_null, "alt": p_alt},
                seconds=seconds,
            )
        )
    return out


def _summary(records: list[SeedRecord]) -> StudyReport:
    return summarize(
        records, [_tiny_scenario()], _cfg(), tigramite_version="x", git_sha=None, git_dirty=None
    )


def test_oversized_test_is_inadmissible_and_never_selected() -> None:
    recs = _records(
        1000,
        {"parcorr": 150, "wls10": 40},
        {"parcorr": 900, "wls10": 500},
        {"parcorr": 1.0, "wls10": 2.0},
    )
    rep = _summary(recs)
    assert rep.admissible == {"parcorr|mci": False, "wls10|mci": True}
    assert rep.failed_gates["parcorr|mci"] == ("sc/null",)
    assert rep.selected == "wls10|mci"


def test_more_powerful_admissible_test_wins_over_cheaper() -> None:
    recs = _records(
        1000,
        {"parcorr": 40, "wls10": 40},
        {"parcorr": 500, "wls10": 700},
        {"parcorr": 1.0, "wls10": 2.0},
    )
    assert _summary(recs).selected == "wls10|mci"


def test_statistical_tie_goes_to_cheaper_test() -> None:
    recs = _records(
        1000,
        {"parcorr": 40, "wls10": 40},
        {"parcorr": 500, "wls10": 500},
        {"parcorr": 1.0, "wls10": 2.0},
    )
    # Shift wls10's alternative rejections by a few seeds: same rate, imperfectly paired.
    for r in recs[495:505]:
        r.p["alt"]["wls10|mci"] = (0.001, 0.9) if r.index >= 500 else (0.9, 0.9)
    rep = _summary(recs)
    assert rep.selected == "parcorr|mci"


def test_no_admissible_test_selects_nothing() -> None:
    recs = _records(
        1000,
        {"parcorr": 150, "wls10": 150},
        {"parcorr": 900, "wls10": 900},
        {"parcorr": 1.0, "wls10": 2.0},
    )
    rep = _summary(recs)
    assert rep.selected is None
    assert "no candidate" in rep.selection_reason


def test_rates_and_ks_reported() -> None:
    recs = _records(
        400,
        {"parcorr": 20, "wls10": 20},
        {"parcorr": 0, "wls10": 0},
        {"parcorr": 1.0, "wls10": 1.0},
    )
    rep = _summary(recs)
    null = [f for f in rep.families if f.family == "null"]
    assert {f.rate.successes for f in null} == {20}
    assert all(f.ks_uniform_p is None for f in null)  # two lags: no KS
    assert np.isclose(rep.per_family_pass_prob_at_alpha, pass_probability(400, 0.05, 0.075, 0.95))


# -- study 2: decision tests and simplicity tie-break -------------------------------------------


def _cfg2(**update: object) -> SizeStudyConfig:
    return _cfg(
        decisions=(DecisionTest.MCI, DecisionTest.PARENTS_R),
        tie_break="simplicity",
        decision_simplicity=(DecisionTest.PARENTS_R, DecisionTest.MCI),
        ci_simplicity=("parcorr", "wls10"),
        **update,
    )


def test_labels_are_ci_decision_pairs() -> None:
    assert _cfg2().labels == ("parcorr|mci", "parcorr|parents_r", "wls10|mci", "wls10|parents_r")
    with pytest.raises(ValueError, match="simplicity"):
        SizeStudyConfig.model_validate(
            {**_cfg2().model_dump(), "decision_simplicity": [DecisionTest.MCI]}
        )


def test_simplicity_tie_break_prefers_simpler_decision_test() -> None:
    labels = _cfg2().labels
    recs = _records(
        1000, dict.fromkeys(labels, 40), dict.fromkeys(labels, 500), {"parcorr": 2.0, "wls10": 1.0}
    )
    rep = summarize(
        recs, [_tiny_scenario()], _cfg2(), tigramite_version="x", git_sha=None, git_dirty=None
    )
    # All four tie; parents_r beats mci, then parcorr beats wls10, whatever the runtime.
    assert rep.selected == "parcorr|parents_r"


def test_inadmissible_simple_test_loses_to_admissible_complex_one() -> None:
    labels = _cfg2().labels
    null = dict.fromkeys(labels, 40) | {"parcorr|parents_r": 200, "wls10|parents_r": 200}
    recs = _records(1000, null, dict.fromkeys(labels, 500), {"parcorr": 1.0, "wls10": 1.0})
    rep = summarize(
        recs, [_tiny_scenario()], _cfg2(), tigramite_version="x", git_sha=None, git_dirty=None
    )
    assert rep.selected == "parcorr|mci"


def test_study2_gates() -> None:
    sc = benchmark_scenarios(5, STUDY2_GATES, names=("e2_garch_null", "e3_default"))
    gating = sorted(
        f"{s.name}/{f.name}" for s in sc for f in s.families if f.role is FamilyRole.GATING_NULL
    )
    assert gating == sorted(
        [f"e2_garch_null/{s}->R" for s in ("s_real", "s_conf", "s_decay", "s_noise")]
        + ["e3_default/s_conf->R", "e3_default/s_noise->R"]
    )
    with pytest.raises(ValueError, match="alternative"):
        benchmark_scenarios(5, {"e3_default": frozenset({"s_real->R"})})


def test_required_seeds_for_study2_rule() -> None:
    """Study 2 has 11 gates: M = 1785 (ADR 0009)."""
    assert required_seeds(0.05, 0.075, 11, 0.90) == 1785


def test_parallel_identical_to_serial_with_decision_tests() -> None:
    sc, cfg = _tiny_scenario(), _cfg2()
    serial = run_scenarios([sc], cfg, 3, n_jobs=1)
    parallel = run_scenarios([sc], cfg, 3, n_jobs=2, chunk_size=2)
    assert _strip(serial) == _strip(parallel)
    assert set(serial[0].p["alt"]) == set(cfg.labels)


def test_mci_decision_reproduces_study1_records() -> None:
    """Adding decision tests must not change the MCI p-values (study 2 reuses study 1's seeds)."""
    sc = _tiny_scenario()
    one = run_scenarios([sc], _cfg(), 2, n_jobs=1)
    two = run_scenarios([sc], _cfg2(), 2, n_jobs=1)
    for a, b in zip(one, two, strict=True):
        for fam in a.p:
            for lab, vals in a.p[fam].items():
                assert b.p[fam][lab] == vals


# -- study 3 ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _LengthCheck:
    """Builder that records the sample length it was asked for (as a 1-column feature)."""

    def __call__(self, seed: int, n_obs: int) -> tuple[np.ndarray, tuple[str, ...]]:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_obs, 2)), ("s_noise", "asset_00")


def test_scenario_n_obs_overrides_study_length() -> None:
    fam = LinkFamily(
        name="null", source="s_noise", target="asset_00", lags=(1,), role=FamilyRole.NULL
    )
    long = Scenario("long", "", _LengthCheck(), (fam,))
    short = Scenario("short", "", _LengthCheck(), (fam,), n_obs=250)
    cfg = _cfg(n_obs=400)
    for sc, expected in ((long, 400), (short, 250)):
        rec = run_scenarios([sc], cfg, 1, n_jobs=1)[0]
        data, _ = sc.build(rec.dataset_seed, sc.n_obs or cfg.n_obs)
        assert data.shape[0] == expected
    assert benchmark_scenarios(5, STUDY2_GATES, names=("e3_default",))[0].n_obs is None


def test_study3_scenarios_and_gates() -> None:
    from scripts.causal_ci_size_study import STUDY3_SHORT_T, scenarios_for

    cfg = load_size_study_config(Path("configs/causal_study3.yaml"))
    scenarios = scenarios_for(cfg)
    gates = [
        f"{s.name}/{f.name}"
        for s in scenarios
        for f in s.families
        if f.role is FamilyRole.GATING_NULL
    ]
    assert len(gates) == 11
    short = next(s for s in scenarios if s.name == "e6_stambaugh_short")
    assert short.n_obs == STUDY3_SHORT_T
    assert all(f.role is not FamilyRole.GATING_NULL for f in short.families)
    assert required_seeds(cfg.alpha, cfg.size_cutoff, len(gates), cfg.all_pass_prob) == 1785
    assert cfg.labels[:2] == ("parcorr|hybrid", "parcorr|mci")


def test_confirmation_study_config() -> None:
    from scripts.causal_ci_size_study import scenarios_for

    cfg = load_size_study_config(Path("configs/causal_study3_confirm.yaml"))
    study3 = load_size_study_config(Path("configs/causal_study3.yaml"))
    scenarios = scenarios_for(cfg)
    gates = [
        f"{s.name}/{f.name}"
        for s in scenarios
        for f in s.families
        if f.role is FamilyRole.GATING_NULL
    ]
    study3_gates = [
        f"{s.name}/{f.name}"
        for s in scenarios_for(study3)
        for f in s.families
        if f.role is FamilyRole.GATING_NULL
    ]
    assert gates == study3_gates
    assert required_seeds(cfg.alpha, cfg.size_cutoff, len(gates), cfg.all_pass_prob) == 1785
    assert cfg.labels == ("hc3|hybrid", "cwls20|hybrid")
    assert cfg.root_seed != study3.root_seed
    same = ("n_obs", "tau_max", "pc_alpha", "alpha", "size_cutoff", "all_pass_prob", "confidence")
    assert all(getattr(cfg, k) == getattr(study3, k) for k in same)
