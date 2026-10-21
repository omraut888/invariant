r"""Pre-registered Stage 2 study: the kill test's size and power, and the half-life estimators
(docs/adr/0011).

Each replication runs Stage 1's method on a benchmark dataset: PCMCI with ``hc3|hybrid``,
Bonferroni over lags, then BH over the four candidates. It takes each of s_real and s_decay at
its best lag with its conditioning set, exactly as Stage 1 hands them on. On that design it
runs:

- the **kill test** (``decline.decline_test``): one-sided sup-split, fixed-regressor wild
  bootstrap;
- the **half-life estimators** on 252-row non-overlapping windows: (c) ``partial_corr``, (b)
  ``gaussian_te`` and (a) ``ksg`` (``halflife`` module).

The KSG arm runs as separate tasks (``arm = "ksg"``), so it can be run on fewer seeds when the
compute-budget rule says so (``ksg_budget``).

Decision units: a kill-test rejection (``p < alpha``) is one Bernoulli per (scenario, signal,
seed). An estimator "claims decay" when its 95% interval excludes no decay (``lam_low > 0``).
It "covers" when the interval contains the true TE half-life ``GroundTruth.te_half_life_approx``.
Seeds are independent, so Wilson intervals are honest.

Rules (fixed in ADR 0011 before any full run):
- Kill test: the Wilson upper bound of the rejection rate on s_real is <= ``size_cutoff`` in
  every ``kill_gate`` scenario. Otherwise Stage 2 is not built.
- Estimator admissible: the false-decay Wilson upper bound on s_real is <= ``size_cutoff`` in
  every ``estimator_gate`` scenario, and the coverage Wilson lower bound on s_decay is >=
  ``coverage_min`` in every ``coverage_gate`` scenario.
- Selection: the lowest RMSE of ``lam_hat`` against the true TE decay rate in the
  ``selection_scenario``. A seed-bootstrap 95% CI of an RMSE difference from the best that
  contains 0 counts as tied. Ties go to the first in ``simplicity`` order. KSG is eligible only
  if it ran on every seed.

Reproducibility: as in ``causal.calibration``, seed i's randomness comes from
``RngFactory(root_seed).child("seed/i")``, and the dataset seed is shared by every scenario
(common random numbers).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml
from joblib import Parallel, delayed, parallel_config
from pydantic import BaseModel, ConfigDict, Field, model_validator

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.synthetic import SIGNAL_NODES, generate_market
from invariant.config.settings import ConfigError, deep_merge, find_project_root
from invariant.core.logging import get_logger
from invariant.core.run_hash import RngFactory
from invariant.core.types import Verdict
from invariant.ledger.benchmark_score import RateEstimate, wilson_interval
from invariant.stages.causal.ci_tests import CITestName, make_ci_test
from invariant.stages.causal.multiple_testing import bonferroni_family_p, fdr_qvalues
from invariant.stages.causal.pcmci import DecisionTest, lagged_design, run_pcmci
from invariant.stages.causal.stage import decide
from invariant.stages.decay.decline import decline_test
from invariant.stages.decay.halflife import (
    Estimator,
    fit_gaussian_te,
    fit_ksg,
    fit_partial_corr,
    make_grid,
)
from invariant.stages.decay.te import ksg_windows
from invariant.stages.decay.windows import window_bounds, window_partial_corr

__all__ = [
    "DEFAULT_STUDY_CONFIG",
    "STUDY_SIGNALS",
    "DecayStudyConfig",
    "DecayStudyReport",
    "KsgBudget",
    "Scenario",
    "SeedRecord",
    "decay_scenarios",
    "ksg_budget",
    "load_decay_study_config",
    "run_records",
    "run_seed",
    "summarize",
    "write_report",
]

DEFAULT_STUDY_CONFIG: Final = Path("configs/decay_study.yaml")
STUDY_SIGNALS: Final = ("s_real", "s_decay")
_LN2: Final = math.log(2.0)

Arm = Literal["main", "ksg"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DeclineSettings(_Frozen):
    alpha: float = Field(gt=0.0, lt=0.5)
    n_boot: int = Field(ge=99)
    trim: float = Field(gt=0.0, lt=0.5)
    step: int = Field(ge=1)


class GridSettings(_Frozen):
    tau_step: int = Field(ge=1)
    hl_min: float = Field(gt=0.0)
    hl_max: float = Field(gt=0.0)
    n_hl: int = Field(ge=2)


class DecayStudyConfig(_Frozen):
    root_seed: int = Field(ge=0, lt=2**32)
    n_seeds: int | None = Field(default=None, ge=1)
    """``None``: ``required_seeds`` over the gates (ADR 0011)."""
    ksg_n_seeds: int | None = Field(default=None, ge=1)
    """``None``: KSG runs on every seed. Otherwise the budget rule's ``M_ksg`` (ADR 0011)."""
    ksg_budget_hours: float = Field(gt=0.0)
    n_obs: int = Field(ge=1000)
    tau_max: int = Field(ge=1)
    pc_alpha: float = Field(gt=0.0, lt=1.0)
    causal_alpha: float = Field(gt=0.0, lt=0.5)
    decline: DeclineSettings
    window: int = Field(ge=20)
    grid: GridSettings
    size_cutoff: float = Field(gt=0.0, lt=1.0)
    all_pass_prob: float = Field(gt=0.0, lt=1.0)
    coverage_min: float = Field(gt=0.0, lt=1.0)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    kill_gates: tuple[str, ...] = Field(min_length=1)
    estimator_gates: tuple[str, ...] = Field(min_length=1)
    coverage_gates: tuple[str, ...] = Field(min_length=1)
    selection_scenario: str
    simplicity: tuple[Estimator, ...]
    n_rmse_boot: int = Field(ge=100)

    @model_validator(mode="after")
    def _check(self) -> DecayStudyConfig:
        if sorted(self.simplicity) != sorted(Estimator):
            raise ValueError("simplicity must list every estimator once")
        if self.size_cutoff <= self.decline.alpha:
            raise ValueError("size_cutoff must exceed the decline test's alpha")
        return self

    @property
    def n_gating(self) -> int:
        """Gates in the M rule: kill-test size gates plus estimator false-decay gates."""
        return len(self.kill_gates) + len(self.estimator_gates)


def load_decay_study_config(path: Path | None = None) -> DecayStudyConfig:
    path = path if path is not None else DEFAULT_STUDY_CONFIG
    path = path if path.is_absolute() else find_project_root() / path
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"decay study config not found: {path}") from None
    return DecayStudyConfig.model_validate(data)


# ---------------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    config: SyntheticMarketConfig
    estimators: tuple[Estimator, ...]
    """Half-life estimators run here. The kill test always runs on both signals."""


def _variant(base: SyntheticMarketConfig, override: dict[str, object]) -> SyntheticMarketConfig:
    return SyntheticMarketConfig.model_validate(deep_merge(base.model_dump(mode="json"), override))


ALL_ESTIMATORS: Final = (Estimator.PARTIAL_CORR, Estimator.GAUSSIAN_TE, Estimator.KSG)
CHEAP_ESTIMATORS: Final = (Estimator.PARTIAL_CORR, Estimator.GAUSSIAN_TE)
POWER_GRID: Final = (
    (0.3, 250.0),
    (0.6, 125.0),
    (0.6, 500.0),
    (0.8, 125.0),
    (0.8, 250.0),
    (0.8, 500.0),
)


def decay_scenarios() -> tuple[Scenario, ...]:
    """The pre-registered scenarios of ADR 0011. Common random numbers across all of them."""
    base = load_benchmark_config(None)
    out = [
        Scenario("d1_base", "Default preset (H 250, onset .6).", base, ALL_ESTIMATORS),
        Scenario(
            "d2_ident",
            "Identifiable: decay ic .25, H 500 (TE half-life 250).",
            _variant(base, {"decay": {"ic": 0.25, "half_life": 500.0}}),
            ALL_ESTIMATORS,
        ),
        Scenario(
            "d3_ident_fast",
            "Identifiable: decay ic .25, H 250 (TE half-life 125).",
            _variant(base, {"decay": {"ic": 0.25}}),
            ALL_ESTIMATORS,
        ),
        Scenario(
            "k_garch_hi",
            "Null for s_real: GARCH alpha .04, beta .95 (persistence .99).",
            _variant(base, {"garch": {"alpha": 0.04, "beta": 0.95}}),
            CHEAP_ESTIMATORS,
        ),
        Scenario(
            "k_rho99",
            "Null for s_real: s_real rho .99.",
            _variant(base, {"real": {"rho": 0.99}}),
            CHEAP_ESTIMATORS,
        ),
        Scenario(
            "k_realistic",
            "Null for s_real: realistic preset (ic .02).",
            load_benchmark_config("realistic"),
            CHEAP_ESTIMATORS,
        ),
        Scenario(
            "k_hidden",
            "Hidden confounder (reported): an omitted persistent parent of R.",
            load_benchmark_config("hidden_confounder"),
            CHEAP_ESTIMATORS,
        ),
    ]
    for frac, hl in POWER_GRID:
        out.append(
            Scenario(
                f"p_onset{frac:.1f}_h{hl:.0f}",
                f"Kill-test power: decay onset {frac}, coefficient half-life {hl:.0f}.",
                _variant(base, {"decay": {"start_frac": frac, "half_life": hl}}),
                (),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------------------------


def _seed_factory(root_seed: int, index: int) -> RngFactory:
    return RngFactory(root_seed).child(f"seed/{index}")


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


class FitRecord(_Frozen):
    lam_hat: float
    lam_low: float
    lam_high: float
    tau_hat: int


class SignalRecord(_Frozen):
    best_lag: int
    n_conditions: int
    family_p: float
    stage1_pass: bool
    decline_p: float | None = None
    decline_stat: float | None = None
    break_fraction: float | None = None
    trend_p: float | None = None
    fits: dict[str, FitRecord] = Field(default_factory=dict)


class SeedRecord(_Frozen):
    scenario: str
    index: int
    arm: Arm
    dataset_seed: int
    te_half_life: float | None
    """``GroundTruth.te_half_life_approx`` of s_decay."""
    signals: dict[str, SignalRecord]
    seconds: float
    """Wall time of this task. Not part of the reproducible result."""


def run_seed(scenario: Scenario, cfg: DecayStudyConfig, index: int, arm: Arm) -> SeedRecord:
    """Pure in everything but ``seconds``: depends only on ``(scenario, cfg, index, arm)``."""
    t0 = time.perf_counter()
    factory = _seed_factory(cfg.root_seed, index)
    ds_seed = _u32(factory, "dataset")
    ds = generate_market(scenario.config.model_copy(update={"n_obs": cfg.n_obs}), ds_seed)
    frame = ds.features.join(ds.returns)
    data = frame.to_numpy(dtype=np.float64)
    names = tuple(str(c) for c in frame.columns)
    target = str(ds.returns.columns[0])
    candidates = [s for s in SIGNAL_NODES if s in names]
    out = run_pcmci(
        data,
        names,
        make_ci_test(CITestName.PARCORR_HC3, seed=_u32(factory, "ci_test")),
        tau_max=cfg.tau_max,
        pc_alpha=cfg.pc_alpha,
        decision_links=[(s, target) for s in candidates],
        decisions=[DecisionTest.HYBRID],
    )
    fam_p = {
        s: bonferroni_family_p(out.decision_p[(DecisionTest.HYBRID, s, target)]) for s in candidates
    }
    q = dict(zip(candidates, fdr_qvalues([fam_p[s] for s in candidates], "bh"), strict=True))
    signals: dict[str, SignalRecord] = {}
    scen = factory.child(scenario.name)
    for sig in STUDY_SIGNALS:
        p_lags = out.decision_p[(DecisionTest.HYBRID, sig, target)]
        lag = int(np.argmin(p_lags)) + 1
        conds = out.conditions(DecisionTest.HYBRID, sig, target, lag)
        d = lagged_design(data, names, sig, target, lag, conds, tau_max=cfg.tau_max)
        verdict, _ = decide(fam_p[sig], float(q[sig]), cfg.causal_alpha)
        rec: dict[str, object] = {
            "best_lag": lag,
            "n_conditions": len(conds),
            "family_p": fam_p[sig],
            "stage1_pass": verdict is Verdict.PASS,
        }
        bounds = window_bounds(len(d.x), cfg.window)
        g = cfg.grid
        grid = make_grid(
            len(d.x), tau_step=g.tau_step, hl_min=g.hl_min, hl_max=g.hl_max, n_hl=g.n_hl
        )
        fits: dict[str, FitRecord] = {}
        if arm == "main":
            dt = decline_test(
                d.x,
                d.y,
                d.z,
                scen.generator(f"decline/{sig}"),
                n_boot=cfg.decline.n_boot,
                trim=cfg.decline.trim,
                step=cfg.decline.step,
            )
            rec |= {
                "decline_p": dt.p,
                "decline_stat": dt.statistic,
                "break_fraction": dt.break_fraction,
                "trend_p": dt.trend_p,
            }
            wpc = window_partial_corr(d.x, d.y, d.z, bounds)
            for est in scenario.estimators:
                if est is Estimator.PARTIAL_CORR:
                    fit = fit_partial_corr(wpc, grid)
                elif est is Estimator.GAUSSIAN_TE:
                    fit = fit_gaussian_te(wpc, grid)
                else:
                    continue
                fits[est.value] = FitRecord(
                    lam_hat=fit.lam_hat,
                    lam_low=fit.lam_low,
                    lam_high=fit.lam_high,
                    tau_hat=fit.tau_hat,
                )
        elif Estimator.KSG in scenario.estimators:
            state = scen.seed_sequence(f"ksg/{sig}").generate_state(len(bounds), np.uint32)
            seeds = [int(v) for v in state]
            fit = fit_ksg(ksg_windows(d.x, d.y, d.z, bounds, seeds), grid)
            fits[Estimator.KSG.value] = FitRecord(
                lam_hat=fit.lam_hat,
                lam_low=fit.lam_low,
                lam_high=fit.lam_high,
                tau_hat=fit.tau_hat,
            )
        signals[sig] = SignalRecord.model_validate(rec | {"fits": fits})
    decay_truth = ds.ground_truth.signal("s_decay").te_half_life_approx
    return SeedRecord(
        scenario=scenario.name,
        index=index,
        arm=arm,
        dataset_seed=ds_seed,
        te_half_life=decay_truth,
        signals=signals,
        seconds=time.perf_counter() - t0,
    )


def run_records(
    tasks: Sequence[tuple[Scenario, int, Arm]],
    cfg: DecayStudyConfig,
    *,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
    chunk_size: int = 64,
) -> list[SeedRecord]:
    """Run ``tasks`` with ``n_jobs`` loky workers, resuming from ``checkpoint`` (JSONL).

    The result follows the order of ``tasks`` and does not depend on ``n_jobs``. BLAS is
    single-threaded in the workers.
    """
    done: dict[tuple[str, int, str], SeedRecord] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = SeedRecord.model_validate_json(line)
                done[(rec.scenario, rec.index, rec.arm)] = rec
    todo = [t for t in tasks if (t[0].name, t[1], t[2]) not in done]
    log = get_logger(__name__)
    log.info("decay_study_start", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            recs: list[SeedRecord] = Parallel()(
                delayed(run_seed)(sc, cfg, i, arm) for sc, i, arm in chunk
            )
            for rec in recs:
                done[(rec.scenario, rec.index, rec.arm)] = rec
            if checkpoint is not None:
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.writelines(r.model_dump_json() + "\n" for r in recs)
            log.info("decay_study_progress", done=start + len(chunk), total=len(todo))
    return [done[(sc.name, i, arm)] for sc, i, arm in tasks]


# ---------------------------------------------------------------------------------------------
# KSG compute budget (ADR 0011)
# ---------------------------------------------------------------------------------------------


class KsgBudget(_Frozen):
    """The mechanical budget decision from a smoke run's KSG timings."""

    budget_hours: float
    workers: int
    smoke_seeds: int
    seconds_per_seed: float
    """Mean over smoke seeds of the summed KSG task wall time over all KSG scenarios."""
    required_seeds: int
    projected_hours: float
    full: bool
    ksg_n_seeds: int
    """``required_seeds`` if ``full``, else ``floor(budget * workers / seconds_per_seed)``."""


def ksg_budget(
    records: Sequence[SeedRecord], *, budget_hours: float, workers: int, required_seeds: int
) -> KsgBudget:
    ksg = [r for r in records if r.arm == "ksg"]
    if not ksg:
        raise ValueError("no KSG records to time")
    per_seed: dict[int, float] = {}
    for r in ksg:
        per_seed[r.index] = per_seed.get(r.index, 0.0) + r.seconds
    sec = float(np.mean(list(per_seed.values())))
    projected = sec * required_seeds / workers / 3600.0
    full = projected <= budget_hours
    m_ksg = required_seeds if full else math.floor(budget_hours * 3600.0 * workers / sec)
    return KsgBudget(
        budget_hours=budget_hours,
        workers=workers,
        smoke_seeds=len(per_seed),
        seconds_per_seed=sec,
        required_seeds=required_seeds,
        projected_hours=projected,
        full=full,
        ksg_n_seeds=min(m_ksg, required_seeds),
    )


# ---------------------------------------------------------------------------------------------
# Summary and decision
# ---------------------------------------------------------------------------------------------


class RejectionSummary(_Frozen):
    scenario: str
    signal: str
    gate: bool
    rate: RateEstimate
    rate_given_stage1_pass: RateEstimate
    trend_rate: RateEstimate
    """The trend-interaction diagnostic at the same alpha (never decides)."""


class EstimatorSummary(_Frozen):
    scenario: str
    signal: str
    estimator: Estimator
    n: int
    claims_decay: RateEstimate
    coverage: RateEstimate | None
    lam_true: float | None
    lam_bias: float | None
    lam_sd: float | None
    lam_rmse: float | None
    median_lam_hat: float | None
    """Median of ``lam_hat``; its TE half-life is ``ln2 / median`` (no decay when 0)."""
    no_decay_in_ci: float
    """Share of fits whose interval includes no decay."""


class RmseComparison(_Frozen):
    estimator: Estimator
    rmse: float
    diff_from_best: float
    diff_ci_low: float
    diff_ci_high: float


class DecayStudyReport(_Frozen):
    config: DecayStudyConfig
    n_seeds: int
    ksg_n_seeds: int
    ksg_eligible: bool
    git_sha: str | None
    git_dirty: bool | None
    rejections: tuple[RejectionSummary, ...]
    kill_failed_gates: tuple[str, ...]
    kill_test_passes: bool
    estimators: tuple[EstimatorSummary, ...]
    admissible: dict[str, bool]
    estimator_failed_gates: dict[str, tuple[str, ...]]
    rmse: tuple[RmseComparison, ...]
    selected: Estimator | None
    selection_reason: str
    median_seconds: dict[str, float]


def _rate(flags: Sequence[bool], confidence: float) -> RateEstimate:
    return wilson_interval(sum(flags), len(flags), confidence)


def _sq_errors(recs: Sequence[SeedRecord], est: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in recs:
        if r.te_half_life is None:
            continue
        fit = r.signals["s_decay"].fits.get(est)
        if fit is not None:
            out[r.index] = (fit.lam_hat - _LN2 / r.te_half_life) ** 2
    return out


def summarize(
    records: Sequence[SeedRecord],
    scenarios: Sequence[Scenario],
    cfg: DecayStudyConfig,
    *,
    git_sha: str | None,
    git_dirty: bool | None,
) -> DecayStudyReport:
    """Rates, gates, admissibility and the selected estimator (module docstring)."""
    conf, alpha = cfg.confidence, cfg.decline.alpha
    main = [r for r in records if r.arm == "main"]
    n_seeds = len({r.index for r in main})
    ksg_idx = {r.index for r in records if r.arm == "ksg"}
    ksg_eligible = n_seeds > 0 and len(ksg_idx) == n_seeds

    rejections: list[RejectionSummary] = []
    failed_kill: list[str] = []
    for sc in scenarios:
        recs = [r for r in main if r.scenario == sc.name]
        for sig in STUDY_SIGNALS:
            rows = [r.signals[sig] for r in recs]
            rej = [s.decline_p is not None and s.decline_p < alpha for s in rows]
            cond = [x for x, s in zip(rej, rows, strict=True) if s.stage1_pass]
            trend = [s.trend_p is not None and s.trend_p < alpha for s in rows]
            gate = sig == "s_real" and sc.name in cfg.kill_gates
            rate = _rate(rej, conf)
            rejections.append(
                RejectionSummary(
                    scenario=sc.name,
                    signal=sig,
                    gate=gate,
                    rate=rate,
                    rate_given_stage1_pass=_rate(cond, conf),
                    trend_rate=_rate(trend, conf),
                )
            )
            if gate and (rate.ci_high is None or rate.ci_high > cfg.size_cutoff):
                failed_kill.append(sc.name)

    est_rows: list[EstimatorSummary] = []
    failed_est: dict[str, list[str]] = {e.value: [] for e in Estimator}
    for sc in scenarios:
        for est in sc.estimators:
            arm: Arm = "ksg" if est is Estimator.KSG else "main"
            recs = [r for r in records if r.scenario == sc.name and r.arm == arm]
            for sig in STUDY_SIGNALS:
                fits = [(r, r.signals[sig].fits[est.value]) for r in recs]
                claims = _rate([f.lam_low > 0.0 for _, f in fits], conf)
                coverage = lam_true = bias = sd = rmse = None
                if sig == "s_decay" and fits and fits[0][0].te_half_life is not None:
                    hl = fits[0][0].te_half_life
                    assert hl is not None
                    lam_true = _LN2 / hl
                    lam = np.array([f.lam_hat for _, f in fits])
                    coverage = _rate([f.lam_low <= lam_true <= f.lam_high for _, f in fits], conf)
                    bias = float(lam.mean() - lam_true)
                    sd = float(lam.std(ddof=1)) if len(lam) > 1 else None
                    rmse = float(np.sqrt(np.mean((lam - lam_true) ** 2)))
                est_rows.append(
                    EstimatorSummary(
                        scenario=sc.name,
                        signal=sig,
                        estimator=est,
                        n=len(fits),
                        claims_decay=claims,
                        coverage=coverage,
                        lam_true=lam_true,
                        lam_bias=bias,
                        lam_sd=sd,
                        lam_rmse=rmse,
                        median_lam_hat=float(np.median([f.lam_hat for _, f in fits]))
                        if fits
                        else None,
                        no_decay_in_ci=float(np.mean([f.lam_low == 0.0 for _, f in fits]))
                        if fits
                        else 0.0,
                    )
                )
                too_many = claims.ci_high is None or claims.ci_high > cfg.size_cutoff
                if sig == "s_real" and sc.name in cfg.estimator_gates and too_many:
                    failed_est[est.value].append(f"{sc.name}/false_decay")
                low_cov = coverage is None or coverage.ci_low is None
                low_cov = low_cov or (
                    coverage is not None
                    and coverage.ci_low is not None
                    and coverage.ci_low < cfg.coverage_min
                )
                if sig == "s_decay" and sc.name in cfg.coverage_gates and low_cov:
                    failed_est[est.value].append(f"{sc.name}/coverage")

    eligible = [e for e in cfg.simplicity if e is not Estimator.KSG or ksg_eligible]
    admissible = {e.value: (e in eligible and not failed_est[e.value]) for e in Estimator}
    sel_recs = [r for r in records if r.scenario == cfg.selection_scenario]
    errs = {
        e: _sq_errors(
            [r for r in sel_recs if r.arm == ("ksg" if e is Estimator.KSG else "main")], e.value
        )
        for e in Estimator
    }
    candidates = [e for e in cfg.simplicity if admissible[e.value]]
    comparisons: list[RmseComparison] = []
    selected: Estimator | None = None
    if candidates:
        rmse_of = {e: math.sqrt(float(np.mean(list(errs[e].values())))) for e in candidates}
        best = min(candidates, key=lambda e: rmse_of[e])
        rng = RngFactory(cfg.root_seed).generator("rmse_bootstrap")
        idx = np.array(sorted(errs[best]))
        draws = rng.integers(0, len(idx), size=(cfg.n_rmse_boot, len(idx)))
        tied = []
        for e in candidates:
            a = np.array([errs[e][i] for i in idx])
            b = np.array([errs[best][i] for i in idx])
            diff_boot = np.sqrt(a[draws].mean(axis=1)) - np.sqrt(b[draws].mean(axis=1))
            lo, hi = np.quantile(diff_boot, [0.025, 0.975])
            diff = rmse_of[e] - rmse_of[best]
            comparisons.append(
                RmseComparison(
                    estimator=e,
                    rmse=rmse_of[e],
                    diff_from_best=diff,
                    diff_ci_low=float(lo),
                    diff_ci_high=float(hi),
                )
            )
            if lo <= 0.0:
                tied.append(e)
        selected = min(tied, key=cfg.simplicity.index)
        reason = (
            f"admissible: {[e.value for e in candidates]}; lowest RMSE: {best.value}; tied with "
            f"it: {[e.value for e in tied]}; simplest of those: {selected.value}"
        )
    else:
        reason = "no estimator is admissible; Stage 2 reports no half-life (returns to review)"
    by_arm: dict[str, list[float]] = {}
    for r in records:
        by_arm.setdefault(r.arm, []).append(r.seconds)
    return DecayStudyReport(
        config=cfg,
        n_seeds=n_seeds,
        ksg_n_seeds=len(ksg_idx),
        ksg_eligible=ksg_eligible,
        git_sha=git_sha,
        git_dirty=git_dirty,
        rejections=tuple(rejections),
        kill_failed_gates=tuple(failed_kill),
        kill_test_passes=not failed_kill,
        estimators=tuple(est_rows),
        admissible=admissible,
        estimator_failed_gates={k: tuple(v) for k, v in failed_est.items()},
        rmse=tuple(comparisons),
        selected=selected,
        selection_reason=reason,
        median_seconds={k: float(np.median(v)) for k, v in by_arm.items()},
    )


def write_report(report: DecayStudyReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
