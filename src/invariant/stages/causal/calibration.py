"""Monte Carlo size and power study used to choose Stage 1's CI test (docs/adr/0009).

Method
------
Each *scenario* is a data-generating process (DGP), and seed ``i`` gives one dataset from it.
PCMCI is run on that dataset once per candidate CI test. A *link family* is a set of lags of
one ``source -> target`` link. Its decision on a dataset is the Stage 1 decision unit:
**reject iff min over the family's lags of the MCI p-value <= alpha / n_lags** (Bonferroni,
valid under any dependence between the lags). A one-lag family is a fixed-lag test at alpha.
So every family gives one Bernoulli decision per seed, and seeds are independent. The rejection
rate across seeds therefore has an honest Wilson interval. Correlated lags are never pooled.

Families have a role. ``gating_null``: the null holds, and the test must keep its size there
to be admissible. ``null``: the null holds, and the rate is reported. ``alternative``: a true
edge, and the rate is power.

Admissibility rule (fixed before any result was seen): a test is admissible iff, for every
gating family, the upper bound of the 95% Wilson interval of its rejection rate is
<= ``size_cutoff``. ``required_seeds`` picks the number of seeds M so that a test whose true
size is exactly ``alpha`` on all ``n_gating`` families passes with probability >=
``all_pass_prob``. It uses a union bound, so this holds whatever the dependence between
families. Bonferroni families have true size <= alpha, so this is a lower bound.

Selection rule: among admissible tests, the one with the highest power on the selection
families wins (the mean rejection rate over the ``selection`` families of the
``selection_scenario``). A test whose paired power difference from the best has a 95% CI that
contains 0 counts as tied. Among tied tests, the one with the lowest median runtime wins.
Seeds are shared by every test, so the paired per-seed difference is used, not two
overlapping marginal CIs.

Reproducibility: seed ``i``'s randomness is derived only from ``(root_seed, i)`` through
``RngFactory``, so serial and parallel runs give identical records
(tests/unit/test_calibration.py). Every scenario uses the same dataset seed for seed ``i``
(common random numbers), so differences between scenarios are not draw noise.

Assumptions: seeds are i.i.d. replications of each DGP, so a scenario's result is a
statement about that DGP only. The benchmark makes signals independent of volatility, which
favours an OLS t-test. The vol-correlated null (E4, tests/fixtures/vol_null.py) exists to
cover the case where they are not.

Reference: Wilson (1927), "Probable inference, the law of succession, and statistical
inference", JASA 22(158).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import yaml
from joblib import Parallel, delayed, parallel_config
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.synthetic import SIGNAL_NODES, asset_names, generate_market
from invariant.config.settings import ConfigError, find_project_root
from invariant.core.logging import get_logger
from invariant.core.run_hash import RngFactory
from invariant.ledger.benchmark_score import RateEstimate, wilson_interval
from invariant.stages.causal.ci_tests import CITestName, make_ci_test
from invariant.stages.causal.pcmci import DecisionTest, run_pcmci

__all__ = [
    "DEFAULT_STUDY_CONFIG",
    "STUDY1_GATES",
    "STUDY2_GATES",
    "BenchmarkData",
    "CITestSpec",
    "FamilyRole",
    "FamilySummary",
    "LinkFamily",
    "PowerComparison",
    "Scenario",
    "SeedRecord",
    "SizeStudyConfig",
    "StudyReport",
    "benchmark_scenarios",
    "dataset_seed",
    "load_size_study_config",
    "null_config",
    "pair_label",
    "pass_probability",
    "required_seeds",
    "run_scenarios",
    "run_seed",
    "summarize",
    "write_report",
]

DEFAULT_STUDY_CONFIG: Final = Path("configs/causal_ci_study.yaml")
TARGET: Final = asset_names(1)[0]
_FloatArray = npt.NDArray[np.float64]
_Z95: Final = float(stats.norm.ppf(0.975))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FamilyRole(StrEnum):
    GATING_NULL = "gating_null"
    NULL = "null"
    ALTERNATIVE = "alternative"


class LinkFamily(_Frozen):
    """Lags of one ``source -> target`` link, decided as one Bonferroni test."""

    name: str
    source: str
    target: str
    lags: tuple[int, ...] = Field(min_length=1)
    role: FamilyRole

    def reject(self, p_values: Sequence[float], alpha: float) -> bool:
        if len(p_values) != len(self.lags):
            raise ValueError(f"{self.name}: need {len(self.lags)} p-values, got {len(p_values)}")
        return min(p_values) <= alpha / len(self.lags)


class CITestSpec(_Frozen):
    label: str
    test: CITestName
    wls_window: int = Field(default=10, ge=2)


class SizeStudyConfig(_Frozen):
    scenario_set: Literal["study1", "study2"] = "study1"
    """Which scenarios the script runs (ADR 0009); part of the config, so of the run key."""
    root_seed: int = Field(ge=0, lt=2**32)
    n_seeds: int | None = Field(default=None, ge=1)
    """``None``: use ``required_seeds`` for the configured rule."""
    n_obs: int = Field(ge=200)
    tau_max: int = Field(ge=1)
    pc_alpha: float = Field(gt=0.0, lt=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)
    size_cutoff: float = Field(gt=0.0, lt=1.0)
    all_pass_prob: float = Field(gt=0.0, lt=1.0)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    tests: tuple[CITestSpec, ...] = Field(min_length=1)
    decisions: tuple[DecisionTest, ...] = Field(default=(DecisionTest.MCI,), min_length=1)
    """Signal -> R decision tests (``pcmci`` module docstring). A candidate is a (CI test,
    decision test) pair, labelled ``pair_label(ci, decision)``."""
    selection_scenario: str
    selection_families: tuple[str, ...] = Field(min_length=1)
    tie_break: Literal["runtime", "simplicity"] = "runtime"
    """Among statistically tied candidates: lowest median runtime, or the first in
    ``(decision_simplicity, ci_simplicity)`` lexicographic order."""
    decision_simplicity: tuple[DecisionTest, ...] = ()
    ci_simplicity: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> SizeStudyConfig:
        labels = [t.label for t in self.tests]
        if len(set(labels)) != len(labels):
            raise ValueError(f"test labels must be unique: {labels}")
        if any("|" in label for label in labels):
            raise ValueError("test labels must not contain '|'")
        if len(set(self.decisions)) != len(self.decisions):
            raise ValueError("decisions must be unique")
        if self.size_cutoff <= self.alpha:
            raise ValueError("size_cutoff must exceed alpha")
        if self.tie_break == "simplicity" and (
            sorted(self.decision_simplicity) != sorted(self.decisions)
            or sorted(self.ci_simplicity) != sorted(labels)
        ):
            raise ValueError("simplicity orders must list every decision test and CI test once")
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(pair_label(t.label, d) for t in self.tests for d in self.decisions)


def pair_label(ci_label: str, decision: DecisionTest) -> str:
    return f"{ci_label}|{decision.value}"


def _split_label(label: str) -> tuple[str, DecisionTest]:
    ci, dec = label.split("|")
    return ci, DecisionTest(dec)


def load_size_study_config(path: Path | None = None) -> SizeStudyConfig:
    path = path if path is not None else DEFAULT_STUDY_CONFIG
    path = path if path.is_absolute() else find_project_root() / path
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"size study config not found: {path}") from None
    return SizeStudyConfig.model_validate(data)


# ---------------------------------------------------------------------------------------------
# Required number of seeds
# ---------------------------------------------------------------------------------------------


def _max_passing_count(m: int, cutoff: float, confidence: float) -> int:
    """Largest k with Wilson upper bound of k/m <= cutoff (-1 if none). Monotone in k."""
    lo, hi = -1, m
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ci_high = wilson_interval(mid, m, confidence).ci_high
        assert ci_high is not None
        if ci_high <= cutoff:
            lo = mid
        else:
            hi = mid - 1
    return lo


def pass_probability(m: int, true_size: float, cutoff: float, confidence: float) -> float:
    """P(a family with true rejection rate ``true_size`` passes the rule at ``m`` seeds)."""
    k = _max_passing_count(m, cutoff, confidence)
    return float(stats.binom.cdf(k, m, true_size)) if k >= 0 else 0.0


def required_seeds(
    alpha: float,
    cutoff: float,
    n_gating: int,
    all_pass_prob: float,
    confidence: float = 0.95,
    *,
    m_max: int = 20_000,
    window: int = 200,
) -> int:
    """Smallest M at which a true-``alpha`` test passes all gates.

    Per-family pass probability must be >= ``1 - (1 - all_pass_prob) / n_gating`` (union
    bound). The binomial pass probability is a sawtooth in M, so M is the smallest value from
    which the requirement holds over the next ``window`` seeds too.
    """
    if n_gating < 1:
        raise ValueError("n_gating must be >= 1")
    per_family = 1.0 - (1.0 - all_pass_prob) / n_gating
    streak = 0
    for m in range(1, m_max + 1):
        streak = streak + 1 if pass_probability(m, alpha, cutoff, confidence) >= per_family else 0
        if streak > window:
            return m - window
    raise ValueError(f"no M <= {m_max} satisfies the rule")


# ---------------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------------

DataBuilder = Callable[[int, int], tuple[_FloatArray, tuple[str, ...]]]
"""``(dataset_seed, n_obs) -> (T x N data, var_names)``. Must be picklable (module level)."""


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    build: DataBuilder
    families: tuple[LinkFamily, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.families]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate family names")


@dataclass(frozen=True)
class BenchmarkData:
    """Builder for a planted-signal benchmark DGP: observed features plus returns.

    Reads ``dataset.features`` and ``dataset.returns`` directly, not the PIT panel. The panel's
    ``syn_*`` columns equal ``features`` bit for bit (ADR 0008), and this study measures the CI
    test, not the data layer.
    """

    config: SyntheticMarketConfig

    def __call__(self, seed: int, n_obs: int) -> tuple[_FloatArray, tuple[str, ...]]:
        cfg = self.config.model_copy(update={"n_obs": n_obs})
        ds = generate_market(cfg, seed)
        frame = ds.features.join(ds.returns)
        return frame.to_numpy(dtype=np.float64), tuple(str(c) for c in frame.columns)


def null_config(base: SyntheticMarketConfig, *, garch: bool) -> SyntheticMarketConfig:
    update: dict[str, object] = {
        "factor": base.factor.model_copy(update={"ic": 0.0}),
        "real": base.real.model_copy(update={"ic": 0.0}),
        "decay": base.decay.model_copy(update={"ic": 0.0}),
    }
    if not garch:
        update["garch"] = base.garch.model_copy(update={"alpha": 0.0, "beta": 0.0})
    return SyntheticMarketConfig.model_validate(base.model_copy(update=update).model_dump())


def _fam(
    name: str, source: str, lags: Iterable[int], role: FamilyRole, target: str = TARGET
) -> LinkFamily:
    return LinkFamily(name=name, source=source, target=target, lags=tuple(lags), role=role)


def _global_null_families(tau_max: int) -> tuple[LinkFamily, ...]:
    lags = range(1, tau_max + 1)
    out: list[LinkFamily] = []
    for sig in (*SIGNAL_NODES, "z"):
        out.append(_fam(f"{sig}->R", sig, lags, FamilyRole.NULL))
        out.append(_fam(f"{sig}->R@1", sig, [1], FamilyRole.NULL))
        out.append(_fam(f"R->{sig}", TARGET, lags, FamilyRole.NULL, target=sig))
    out.append(_fam("R->R", TARGET, lags, FamilyRole.NULL))
    out.append(_fam("R->R@1", TARGET, [1], FamilyRole.NULL))
    return tuple(out)


def _planted_families(tau_max: int) -> tuple[LinkFamily, ...]:
    lags = range(1, tau_max + 1)
    alt, null = FamilyRole.ALTERNATIVE, FamilyRole.NULL
    return (
        _fam("s_conf->R", "s_conf", lags, null),
        _fam("s_conf->R@1", "s_conf", [1], null),
        _fam("s_noise->R", "s_noise", lags, null),
        _fam("s_noise->R@1", "s_noise", [1], null),
        _fam("s_real->R@2..", "s_real", range(2, tau_max + 1), null),
        _fam("R->R", TARGET, lags, null),
        _fam("R->R@1", TARGET, [1], null),
        _fam("s_real->R", "s_real", lags, alt),
        _fam("s_real->R@1", "s_real", [1], alt),
        _fam("s_decay->R", "s_decay", lags, alt),
        _fam("s_decay->R@1", "s_decay", [1], alt),
        _fam("z->R", "z", lags, alt),
    )


STUDY1_GATES: Final[dict[str, frozenset[str]]] = {
    "e2_garch_null": frozenset(f"{sig}->R" for sig in (*SIGNAL_NODES, "z")),
    "e3_default": frozenset({"s_conf->R"}),
}
"""Gating families of study 1 (CI test choice), as pre-registered in ADR 0009."""

STUDY2_GATES: Final[dict[str, frozenset[str]]] = {
    "e2_garch_null": frozenset(f"{sig}->R" for sig in SIGNAL_NODES),
    "e3_default": frozenset({"s_conf->R", "s_noise->R"}),
}
"""Benchmark gates of study 2 (decision test x CI test), ADR 0009. Z is not a candidate."""


def _gate(scenario: Scenario, gated: frozenset[str]) -> Scenario:
    names = {f.name for f in scenario.families}
    if not gated <= names:
        raise ValueError(f"{scenario.name}: unknown gated families {sorted(gated - names)}")
    fams = []
    for f in scenario.families:
        if f.name in gated:
            if f.role is FamilyRole.ALTERNATIVE:
                raise ValueError(f"{scenario.name}/{f.name}: an alternative cannot gate")
            f = f.model_copy(update={"role": FamilyRole.GATING_NULL})
        fams.append(f)
    return Scenario(scenario.name, scenario.description, scenario.build, tuple(fams))


def benchmark_scenarios(
    tau_max: int,
    gates: dict[str, frozenset[str]] = STUDY1_GATES,
    names: Sequence[str] | None = None,
) -> tuple[Scenario, ...]:
    """E1, E2, E3 (= E5 default) and E5 realistic. E4 and E6 live in tests/fixtures/.

    ``gates`` maps a scenario name to the families that gate in it; every other null family
    is reported only. ``names`` selects scenarios (default: all four).
    """
    base = load_benchmark_config(None)
    all_scenarios = (
        Scenario(
            "e1_iid_null",
            "Global null, i.i.d. Student-t(6) residuals (GARCH off). Control.",
            BenchmarkData(null_config(base, garch=False)),
            _global_null_families(tau_max),
        ),
        Scenario(
            "e2_garch_null",
            "Global null, GARCH(1,1)-t(6) residuals (all ICs 0).",
            BenchmarkData(null_config(base, garch=True)),
            _global_null_families(tau_max),
        ),
        Scenario(
            "e3_default",
            "Default preset, Z observed: conditional nulls and power.",
            BenchmarkData(base),
            _planted_families(tau_max),
        ),
        Scenario(
            "e5_realistic",
            "Realistic preset (IC 0.02): power.",
            BenchmarkData(load_benchmark_config("realistic")),
            _planted_families(tau_max),
        ),
    )
    unknown = set(gates) - {s.name for s in all_scenarios}
    if unknown:
        raise ValueError(f"gates name unknown scenarios: {sorted(unknown)}")
    chosen = [s for s in all_scenarios if names is None or s.name in names]
    return tuple(_gate(s, gates.get(s.name, frozenset())) for s in chosen)


# ---------------------------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------------------------


def _seed_factory(root_seed: int, index: int) -> RngFactory:
    return RngFactory(root_seed).child(f"seed/{index}")


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def dataset_seed(root_seed: int, index: int) -> int:
    """Dataset seed of replication ``index``: shared by all scenarios and tests."""
    return _u32(_seed_factory(root_seed, index), "dataset")


class SeedRecord(_Frozen):
    """One scenario's results on one seed.

    ``p[family][pair_label]``: p-values at the family's lags. ``seconds`` is keyed by CI test.
    """

    scenario: str
    index: int
    dataset_seed: int
    p: dict[str, dict[str, tuple[float, ...]]]
    seconds: dict[str, float]
    """Wall time per test. Not part of the reproducible result."""


def run_seed(scenario: Scenario, cfg: SizeStudyConfig, index: int) -> SeedRecord:
    """Pure in everything but ``seconds``: depends only on ``(scenario, cfg, index)``."""
    factory = _seed_factory(cfg.root_seed, index)
    ds_seed = _u32(factory, "dataset")
    data, names = scenario.build(ds_seed, cfg.n_obs)
    p: dict[str, dict[str, tuple[float, ...]]] = {f.name: {} for f in scenario.families}
    links = sorted({(f.source, f.target) for f in scenario.families})
    seconds: dict[str, float] = {}
    for spec in cfg.tests:
        test = make_ci_test(
            spec.test,
            seed=_u32(factory, f"ci_test/{scenario.name}/{spec.label}"),
            wls_window=spec.wls_window,
        )
        t0 = time.perf_counter()
        out = run_pcmci(
            data,
            names,
            test,
            tau_max=cfg.tau_max,
            pc_alpha=cfg.pc_alpha,
            decision_links=links,
            decisions=cfg.decisions,
        )
        seconds[spec.label] = time.perf_counter() - t0
        for fam in scenario.families:
            for decision in cfg.decisions:
                all_lags = out.decision_p[(decision, fam.source, fam.target)]
                vals = all_lags[[lag - 1 for lag in fam.lags]]
                label = pair_label(spec.label, decision)
                if not np.isfinite(vals).all():
                    raise RuntimeError(f"{scenario.name}/{fam.name}/{label}: non-finite p")
                p[fam.name][label] = tuple(float(v) for v in vals)
    return SeedRecord(
        scenario=scenario.name, index=index, dataset_seed=ds_seed, p=p, seconds=seconds
    )


def run_scenarios(
    scenarios: Sequence[Scenario],
    cfg: SizeStudyConfig,
    n_seeds: int,
    *,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
    chunk_size: int = 64,
) -> list[SeedRecord]:
    """Run every scenario on seeds ``0..n_seeds-1``, with ``n_jobs`` joblib (loky) workers.

    The result is ordered by ``(scenario, index)`` and does not depend on ``n_jobs``. With
    ``checkpoint``, finished records are appended to that JSONL file chunk by chunk, and
    records already there are reused, so an interrupted study resumes. BLAS is single-threaded
    in workers, so that ``n_jobs`` processes do not oversubscribe the cores.
    """
    done: dict[tuple[str, int], SeedRecord] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = SeedRecord.model_validate_json(line)
                done[(rec.scenario, rec.index)] = rec
    by_name = {s.name: s for s in scenarios}
    todo = [(s.name, i) for s in scenarios for i in range(n_seeds) if (s.name, i) not in done]
    log = get_logger(__name__)
    log.info("size_study_start", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            recs: list[SeedRecord] = Parallel()(
                delayed(run_seed)(by_name[name], cfg, i) for name, i in chunk
            )
            for rec in recs:
                done[(rec.scenario, rec.index)] = rec
            if checkpoint is not None:
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.writelines(r.model_dump_json() + "\n" for r in recs)
            log.info("size_study_progress", done=start + len(chunk), total=len(todo))
    return [done[(s.name, i)] for s in scenarios for i in range(n_seeds)]


# ---------------------------------------------------------------------------------------------
# Summary and decision
# ---------------------------------------------------------------------------------------------


class FamilySummary(_Frozen):
    scenario: str
    family: str
    role: FamilyRole
    lags: tuple[int, ...]
    test: str
    rate: RateEstimate
    ks_uniform_p: float | None
    """KS test of p-value uniformity (one-lag null families only)."""


class PowerComparison(_Frozen):
    test: str
    power: float
    diff_from_best: float
    diff_ci_low: float
    diff_ci_high: float


class StudyReport(_Frozen):
    config: SizeStudyConfig
    n_seeds: int
    n_gating: int
    per_family_pass_prob_at_alpha: float
    all_pass_prob_lower_bound: float
    max_passing_rejections: int
    tigramite_version: str
    git_sha: str | None
    git_dirty: bool | None
    families: tuple[FamilySummary, ...]
    admissible: dict[str, bool]
    failed_gates: dict[str, tuple[str, ...]]
    power: tuple[PowerComparison, ...]
    median_seconds: dict[str, float]
    selected: str | None
    selection_reason: str


def _decisions(
    records: Sequence[SeedRecord], scenario: Scenario, fam: LinkFamily, test: str, alpha: float
) -> list[bool]:
    return [fam.reject(r.p[fam.name][test], alpha) for r in records if r.scenario == scenario.name]


def summarize(
    records: Sequence[SeedRecord],
    scenarios: Sequence[Scenario],
    cfg: SizeStudyConfig,
    *,
    tigramite_version: str,
    git_sha: str | None,
    git_dirty: bool | None,
) -> StudyReport:
    """Rates, admissibility and the selected test (see module docstring for the rules)."""
    labels = list(cfg.labels)
    n_seeds = len({r.index for r in records})
    families: list[FamilySummary] = []
    failed: dict[str, list[str]] = {label: [] for label in labels}
    for sc in scenarios:
        for fam in sc.families:
            for label in labels:
                dec = _decisions(records, sc, fam, label, cfg.alpha)
                rate = wilson_interval(sum(dec), len(dec), cfg.confidence)
                ks = None
                if len(fam.lags) == 1 and fam.role is not FamilyRole.ALTERNATIVE:
                    ps = [r.p[fam.name][label][0] for r in records if r.scenario == sc.name]
                    ks = float(stats.kstest(ps, "uniform").pvalue)
                families.append(
                    FamilySummary(
                        scenario=sc.name,
                        family=fam.name,
                        role=fam.role,
                        lags=fam.lags,
                        test=label,
                        rate=rate,
                        ks_uniform_p=ks,
                    )
                )
                if fam.role is FamilyRole.GATING_NULL:
                    assert rate.ci_high is not None
                    if rate.ci_high > cfg.size_cutoff:
                        failed[label].append(f"{sc.name}/{fam.name}")
    n_gating = sum(f.role is FamilyRole.GATING_NULL for sc in scenarios for f in sc.families)
    admissible = {label: not failed[label] for label in labels}

    sel_sc = next(s for s in scenarios if s.name == cfg.selection_scenario)
    sel_fams = [f for f in sel_sc.families if f.name in cfg.selection_families]
    if len(sel_fams) != len(cfg.selection_families):
        raise ValueError("selection_families must name families of selection_scenario")
    # per-seed power score: mean rejection over the selection families
    score = {
        label: np.mean([_decisions(records, sel_sc, f, label, cfg.alpha) for f in sel_fams], axis=0)
        for label in labels
    }
    seconds = {
        label: statistics.median(r.seconds[_split_label(label)[0]] for r in records)
        for label in labels
    }
    candidates = [label for label in labels if admissible[label]]
    power: list[PowerComparison] = []
    selected: str | None = None
    if candidates:
        best = max(candidates, key=lambda lab: float(score[lab].mean()))
        tied = []
        for label in labels:
            d = score[best] - score[label]
            half = _Z95 * float(d.std(ddof=1)) / math.sqrt(len(d)) if len(d) > 1 else math.inf
            mean = float(d.mean())
            power.append(
                PowerComparison(
                    test=label,
                    power=float(score[label].mean()),
                    diff_from_best=mean,
                    diff_ci_low=mean - half,
                    diff_ci_high=mean + half,
                )
            )
            if label in candidates and mean - half <= 0.0:
                tied.append(label)
        if cfg.tie_break == "simplicity":

            def rank(lab: str) -> tuple[int, int]:
                ci, dec = _split_label(lab)
                return cfg.decision_simplicity.index(dec), cfg.ci_simplicity.index(ci)

            selected = min(tied, key=rank)
            how = "simplest"
        else:
            selected = min(tied, key=lambda lab: seconds[lab])
            how = "cheapest"
        reason = (
            f"admissible: {candidates}; highest power: {best}; "
            f"statistically tied with it: {tied}; {how} of those: {selected}"
        )
    else:
        reason = "no candidate is admissible; no test selected"
    per_family = pass_probability(n_seeds, cfg.alpha, cfg.size_cutoff, cfg.confidence)
    return StudyReport(
        config=cfg,
        n_seeds=n_seeds,
        n_gating=n_gating,
        per_family_pass_prob_at_alpha=per_family,
        all_pass_prob_lower_bound=max(0.0, 1.0 - n_gating * (1.0 - per_family)),
        max_passing_rejections=_max_passing_count(n_seeds, cfg.size_cutoff, cfg.confidence),
        tigramite_version=tigramite_version,
        git_sha=git_sha,
        git_dirty=git_dirty,
        families=tuple(families),
        admissible=admissible,
        failed_gates={k: tuple(v) for k, v in failed.items()},
        power=tuple(power),
        median_seconds=seconds,
        selected=selected,
        selection_reason=reason,
    )


def write_report(report: StudyReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
