from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.ledger.benchmark_score import wilson_interval
from invariant.stages.causal.calibration import (
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
    pass_probability,
    required_seeds,
    run_scenarios,
    summarize,
)
from invariant.stages.causal.ci_tests import CITestName


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


def _records(
    n: int, null_reject: dict[str, int], alt_reject: dict[str, int], seconds: dict[str, float]
) -> list[SeedRecord]:
    """Hand-built records: test ``lab`` rejects the null on the first ``null_reject[lab]`` seeds
    and the alternative on the first ``alt_reject[lab]`` seeds."""
    out = []
    for i in range(n):
        p_null = {lab: (0.001 if i < k else 0.9, 0.9) for lab, k in null_reject.items()}
        p_alt = {lab: (0.001 if i < k else 0.9, 0.9) for lab, k in alt_reject.items()}
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
    assert rep.admissible == {"parcorr": False, "wls10": True}
    assert rep.failed_gates["parcorr"] == ("sc/null",)
    assert rep.selected == "wls10"


def test_more_powerful_admissible_test_wins_over_cheaper() -> None:
    recs = _records(
        1000,
        {"parcorr": 40, "wls10": 40},
        {"parcorr": 500, "wls10": 700},
        {"parcorr": 1.0, "wls10": 2.0},
    )
    assert _summary(recs).selected == "wls10"


def test_statistical_tie_goes_to_cheaper_test() -> None:
    recs = _records(
        1000,
        {"parcorr": 40, "wls10": 40},
        {"parcorr": 500, "wls10": 500},
        {"parcorr": 1.0, "wls10": 2.0},
    )
    # Shift wls10's alternative rejections by a few seeds: same rate, imperfectly paired.
    for r in recs[495:505]:
        r.p["alt"]["wls10"] = (0.001, 0.9) if r.index >= 500 else (0.9, 0.9)
    rep = _summary(recs)
    assert rep.selected == "parcorr"


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
