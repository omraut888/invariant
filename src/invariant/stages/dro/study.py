r"""Pre-registered Stage 3 study: which portfolio arm the stage uses, and the size of its kill
test (docs/adr/0013).

Each replication generates a ``multi_asset`` benchmark market (five assets with a latent market
factor) and runs every arm walk-forward (``walkforward.run_walk_forward``) on the planted
signals ``(s_real, s_decay)``:
- 1/N;
- the candidates Markowitz, DRO-FC, Ridge-CV and DRO-CV.

It also solves the two oracles (``oracle.projection_oracle``) on the same dates.

Task kinds, one record each per ``(scenario, seed)``:
- ``main``: every arm and both oracles. For each candidate it runs the kill test (one-sided
  Ledoit-Wolf against 1/N) and computes the DSR across the four candidates, plus every
  reported metric.
- ``loso``: diagnostic B. The candidates on each leave-one-signal-out set; no tests.
- ``pipeline``: Stages 1 and 2 through the production path (``decay.validation``), then every
  arm on the signals they passed. Reported only.
- ``c_sens``: every arm and O1 at the sensitivity values of c. Reported only.

Decision units (seeds are independent, so Wilson intervals and seed bootstraps are honest):
- a kill-test rejection, ``p < lw_alpha``, is one Bernoulli per (scenario, candidate, seed);
- an arm's out-of-sample annualised Sharpe ratio is one draw per (scenario, arm, seed). An
  all-cash path, which earns the zero risk-free rate, has Sharpe ratio 0.

Rules (fixed in ADR 0013 before any full run):
- **Admissible**: a candidate is admissible iff its rejection rate has a Wilson upper bound
  <= ``size_cutoff`` in every ``size_gates`` scenario (nulls, where every arm's population
  Sharpe ratio is 0).
- **Selection**: the admissible candidate with the highest mean Sharpe ratio in
  ``selection_scenario``. O1 is shared by every arm within a seed, so this is also the lowest
  mean O1 regret. A candidate is tied with it when the paired seed-bootstrap 95% CI of the mean
  difference contains 0. The first tied candidate in ``simplicity`` order is selected. If no
  candidate is admissible, the stage is not built.
- **M**: ``required_seeds`` over the size gates for the null scenarios (``gate_seeds``). For
  the selection scenario, ``selection_seeds`` uses the pilot's largest paired-difference SD.
  Reported-only tasks run on the first ``n_report_seeds`` seeds (``plan_budget``).

Reproducibility: seed i's randomness comes from ``RngFactory(root_seed).child("seed/i")``,
and its dataset seed is shared by every scenario and task (common random numbers). It is
derived as in ``decay.validation``, so the ``pipeline`` task runs Stages 1-2 on the same
market as ``main``.

References: Ledoit & Wolf (2008); Bailey & Lopez de Prado (2014); Wilson (1927).
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import yaml
from joblib import Parallel, delayed, parallel_config
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import ConfigError, deep_merge, find_project_root
from invariant.core.logging import get_logger
from invariant.core.run_hash import GitState, RngFactory
from invariant.core.types import Verdict
from invariant.ledger.benchmark_score import RateEstimate, wilson_interval
from invariant.stages.causal.calibration import required_seeds
from invariant.stages.decay.validation import (
    DecayAcceptance,
    DecayValidationConfig,
    run_benchmark_seed,
)
from invariant.stages.dro.metrics import (
    TRADING_DAYS,
    deflated_sharpe_ratio,
    exposure_shares,
    sharpe,
    sharpe_difference_test,
    turnover,
)
from invariant.stages.dro.oracle import projection_oracle
from invariant.stages.dro.walkforward import (
    Arm,
    ArmPath,
    WalkForwardConfig,
    run_walk_forward,
    solve_path,
)

__all__ = [
    "CANDIDATES",
    "DEFAULT_STUDY_CONFIG",
    "REPORTED_KINDS",
    "BudgetPlan",
    "DroStudyConfig",
    "DroStudyReport",
    "Kind",
    "PilotReport",
    "Scenario",
    "Task",
    "TaskRecord",
    "dro_scenarios",
    "gate_seeds",
    "load_dro_study_config",
    "pilot_report",
    "plan_budget",
    "run_records",
    "run_task",
    "selection_seeds",
    "study_tasks",
    "summarize",
    "write_report",
]

DEFAULT_STUDY_CONFIG: Final = Path("configs/dro_study.yaml")
CANDIDATES: Final = (Arm.MARKOWITZ, Arm.DRO_FC, Arm.RIDGE_CV, Arm.DRO_CV)
_PRESET: Final = "multi_asset"
_OBSERVED: Final = ("z", "s_real", "s_conf", "s_decay", "s_noise")

Kind = Literal["main", "loso", "pipeline", "c_sens"]
_FloatArray = npt.NDArray[np.float64]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


class WalkSettings(_Frozen):
    window: int = Field(ge=50)
    refit_every: int = Field(ge=1)
    cv_every: int = Field(ge=1)
    cv_valid: int = Field(ge=10)
    fc_alpha: float = Field(gt=0.0, lt=1.0)
    fc_draws: int = Field(ge=100)
    ridge_grid: tuple[float, ...] = Field(min_length=1)
    eps_grid: tuple[float, ...] = Field(min_length=1)

    def config(self, c: float) -> WalkForwardConfig:
        return WalkForwardConfig(c=c, **self.model_dump())


class PipelineSettings(_Frozen):
    """Overrides for the ``pipeline`` task's Stages 1-2 (tests only; ``None`` when
    pre-registered)."""

    min_obs: int | None = Field(default=None, ge=100)


class DroStudyConfig(_Frozen):
    root_seed: int = Field(ge=0, lt=2**32)
    pilot_root_seed: int = Field(ge=0, lt=2**32)
    pilot_seeds: int = Field(ge=2)
    n_gate_seeds: int | None = Field(default=None, ge=1)
    """``None``: ``gate_seeds`` (ADR 0013)."""
    n_select_seeds: int | None = Field(default=None, ge=1)
    """``None`` until the pilot has run; then ``selection_seeds`` of the pilot, recorded."""
    n_report_seeds: int | None = Field(default=None, ge=1)
    """``None``: ``n_select_seeds``, unless the budget rule lowers it (ADR 0013)."""
    n_obs: int | None = Field(default=None, ge=200)
    """``None``: the preset's own n_obs (6000)."""
    c_annual: float = Field(ge=0.0)
    c_sensitivity: tuple[float, ...]
    signals: tuple[str, ...] = Field(min_length=1)
    walk: WalkSettings
    lw_alpha: float = Field(gt=0.0, lt=0.5)
    lw_n_boot: int = Field(ge=99)
    cost_bp: float = Field(ge=0.0)
    size_cutoff: float = Field(gt=0.0, lt=1.0)
    all_pass_prob: float = Field(gt=0.0, lt=1.0)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    size_gates: tuple[str, ...] = Field(min_length=1)
    selection_scenario: str
    reported_scenarios: tuple[str, ...]
    simplicity: tuple[Arm, ...]
    delta_sharpe: float = Field(gt=0.0)
    power: float = Field(gt=0.0, lt=1.0)
    sigma_confidence: float = Field(gt=0.0, lt=1.0)
    min_select_seeds: int = Field(ge=1)
    n_diff_boot: int = Field(ge=100)
    budget_hours: float = Field(gt=0.0)
    min_report_seeds: int = Field(ge=1)
    pipeline: PipelineSettings = PipelineSettings()

    @model_validator(mode="after")
    def _check(self) -> DroStudyConfig:
        names = {s.name for s in dro_scenarios()}
        used = (*self.size_gates, self.selection_scenario, *self.reported_scenarios)
        unknown = [s for s in used if s not in names]
        if unknown:
            raise ValueError(f"unknown scenarios {unknown}")
        if len(set(used)) != len(used):
            raise ValueError("a scenario may have only one role")
        if set(self.simplicity) != set(CANDIDATES) or len(self.simplicity) != len(CANDIDATES):
            raise ValueError(f"simplicity must order exactly {[a.value for a in CANDIDATES]}")
        return self

    @property
    def c(self) -> float:
        """Daily Sharpe hurdle ``c_annual / sqrt(252)``."""
        return self.c_annual / math.sqrt(TRADING_DAYS)

    @property
    def n_gating(self) -> int:
        return len(self.size_gates) * len(CANDIDATES)


def load_dro_study_config(path: Path | None = None) -> DroStudyConfig:
    p = path if path is not None else DEFAULT_STUDY_CONFIG
    p = p if p.is_absolute() else find_project_root() / p
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"DRO study config not found: {p}") from None
    return DroStudyConfig.model_validate(data)


# ---------------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    override: dict[str, object]

    def config(self, n_obs: int | None) -> SyntheticMarketConfig:
        base = load_benchmark_config(_PRESET).model_dump()
        data = deep_merge(base, self.override)
        if n_obs is not None:
            data["n_obs"] = n_obs
        return SyntheticMarketConfig.model_validate(data)


_NULL: Final[dict[str, object]] = {
    "mu": 0.0,
    "factor": {"ic": 0.0},
    "real": {"ic": 0.0},
    "decay": {"ic": 0.0},
}
_T5: Final[dict[str, object]] = {"garch": {"nu": 5.0}, "market": {"garch": {"nu": 5.0}}}


def dro_scenarios() -> tuple[Scenario, ...]:
    """The pre-registered scenarios (ADR 0013). Each is a deep merge onto ``multi_asset``."""
    return (
        Scenario("d1_base", "multi_asset as configured", {}),
        Scenario(
            "d2_realistic",
            "daily ICs .02 (factor, real, decay)",
            {"factor": {"ic": 0.02}, "real": {"ic": 0.02}, "decay": {"ic": 0.02}},
        ),
        Scenario("d3_null", "no drift, every IC 0: every arm's population Sharpe is 0", _NULL),
        Scenario("d4_heavy", "Student-t(5) innovations in both GARCH processes", _T5),
        Scenario("d5_null_heavy", "d3_null with Student-t(5) innovations", deep_merge(_NULL, _T5)),
    )


def _scenario(name: str) -> Scenario:
    return next(s for s in dro_scenarios() if s.name == name)


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


class ArmMetrics(_Frozen):
    arm: Arm
    sharpe: float
    """Annualised, gross. 0 for an all-cash path."""
    sharpe_net: float
    """Annualised, net of ``cost_bp`` per unit of one-way turnover."""
    mean: float
    sd: float
    invested: float
    cash: float
    partial: float
    turnover: float
    """Mean daily ``||w_t - w_{t-1}||_1``."""
    lw_delta: float | None = None
    """Non-annualised Sharpe difference against 1/N."""
    lw_p: float | None = None
    dsr: float | None = None
    chosen: tuple[float, ...] = ()
    fc_coverage_o1: float | None = None
    fc_coverage_o2: float | None = None
    fc_radius_mean: float | None = None


class OracleMetrics(_Frozen):
    info: Literal["o1", "o2"]
    sharpe: float
    invested: float
    cash: float
    turnover: float


class VariantResult(_Frozen):
    label: str
    signals: tuple[str, ...]
    c_annual: float
    arms: tuple[ArmMetrics, ...]
    oracles: tuple[OracleMetrics, ...] = ()

    def arm(self, arm: Arm) -> ArmMetrics:
        return next(m for m in self.arms if m.arm is arm)


class TaskRecord(_Frozen):
    scenario: str
    index: int
    kind: Kind
    dataset_seed: int
    variants: tuple[VariantResult, ...]
    passed: tuple[str, ...] | None = None
    """``pipeline``: the signals Stages 1-2 passed."""
    seconds: float
    """Wall time of this task. Not part of the reproducible result."""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.scenario, self.index, self.kind)


# ---------------------------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------------------------


def _seed_factory(root_seed: int, index: int) -> RngFactory:
    return RngFactory(root_seed).child(f"seed/{index}")


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def _sr(r: _FloatArray) -> float:
    s = sharpe(r)
    return 0.0 if math.isnan(s) else s


def _arm_metrics(path: ArmPath, cost_bp: float) -> ArmMetrics:
    to = turnover(path.weights)
    net = path.returns - cost_bp * 1e-4 * to
    shares = exposure_shares(path.weights)
    return ArmMetrics(
        arm=path.arm,
        sharpe=_sr(path.returns),
        sharpe_net=_sr(net),
        mean=float(path.returns.mean()),
        sd=float(path.returns.std(ddof=1)),
        invested=shares["invested"],
        cash=shares["cash"],
        partial=shares["partial"],
        turnover=float(to.mean()),
        chosen=tuple(path.chosen),
    )


def _oracle_metrics(info: Literal["o1", "o2"], w: _FloatArray, r: _FloatArray) -> OracleMetrics:
    shares = exposure_shares(w)
    return OracleMetrics(
        info=info,
        sharpe=_sr(r),
        invested=shares["invested"],
        cash=shares["cash"],
        turnover=float(turnover(w).mean()),
    )


def _run_variant(
    ds: SyntheticDataset,
    signals: Sequence[str],
    arms: Sequence[Arm],
    c_annual: float,
    cfg: DroStudyConfig,
    factory: RngFactory,
    *,
    label: str,
    oracles: Sequence[Literal["o1", "o2"]],
    tests: bool,
) -> VariantResult:
    c = c_annual / math.sqrt(TRADING_DAYS)
    factory = factory.child(label)
    x = ds.features.loc[:, list(signals)].to_numpy(dtype=np.float64)
    r = ds.returns.to_numpy(dtype=np.float64)
    paths = run_walk_forward(x, r, arms, cfg.walk.config(c), factory.child("wf"))
    dates = next(iter(paths.values())).dates
    metrics = {a: _arm_metrics(p, cfg.cost_bp) for a, p in paths.items()}

    truth = ds.ground_truth
    o1 = projection_oracle(truth, ds.features, list(signals))
    observed = [s for s in _OBSERVED if s in ds.features.columns]
    o2 = projection_oracle(truth, ds.features, observed) if "o2" in oracles else None
    oracle_out = []
    for info in oracles:
        fc = o1 if info == "o1" else o2
        assert fc is not None
        w, rets = solve_path(fc.mean, fc.cov, r, dates, c)
        oracle_out.append(_oracle_metrics(info, w, rets))

    if Arm.DRO_FC in paths:
        p = paths[Arm.DRO_FC]
        assert p.mu_hat is not None and p.radius is not None
        cov1 = np.linalg.norm(p.mu_hat - o1.mean[dates], axis=1) <= p.radius
        cov2 = None if o2 is None else np.linalg.norm(p.mu_hat - o2.mean[dates], axis=1)
        metrics[Arm.DRO_FC] = metrics[Arm.DRO_FC].model_copy(
            update={
                "fc_coverage_o1": float(cov1.mean()),
                "fc_coverage_o2": None if cov2 is None else float((cov2 <= p.radius).mean()),
                "fc_radius_mean": float(p.radius.mean()),
            }
        )

    if tests and Arm.EQUAL_WEIGHT in paths:
        lw = factory.child("lw")
        base = paths[Arm.EQUAL_WEIGHT].returns
        cands = [a for a in CANDIDATES if a in paths]
        daily = [sharpe(paths[a].returns, annualize=False) for a in cands]
        daily = [0.0 if math.isnan(s) else s for s in daily]
        var = float(np.var(daily, ddof=1)) if len(daily) > 1 else 0.0
        for a in cands:
            delta, p_val = sharpe_difference_test(
                paths[a].returns, base, lw.generator(a.value), n_boot=cfg.lw_n_boot
            )
            dsr = (
                deflated_sharpe_ratio(
                    paths[a].returns, n_trials=len(cands), trial_sharpe_variance=var
                )
                if len(cands) > 1
                else math.nan
            )
            metrics[a] = metrics[a].model_copy(
                update={
                    "lw_delta": None if math.isnan(delta) else delta,
                    "lw_p": p_val,
                    "dsr": None if math.isnan(dsr) else dsr,
                }
            )
    return VariantResult(
        label=label,
        signals=tuple(signals),
        c_annual=c_annual,
        arms=tuple(metrics[a] for a in arms),
        oracles=tuple(oracle_out),
    )


def _pipeline_passed(
    cfg: DroStudyConfig,
    index: int,
    git_state: GitState,
    patch: bytes | None,
    work_dir: Path | None,
) -> tuple[tuple[str, ...], int]:
    """The signals Stages 1-2 pass on seed ``index``'s selection-scenario market, and that
    market's dataset seed."""
    vcfg = DecayValidationConfig(
        root_seed=cfg.root_seed,
        n_seeds=index + 1,
        n_obs=cfg.n_obs,
        presets=(_PRESET,),
        confidence=cfg.confidence,
        n_boot=None,
        min_obs=cfg.pipeline.min_obs,
        acceptance=DecayAcceptance(
            preset=_PRESET,
            null_signals=("s_real",),
            null_kill_upper_max=0.075,
            decay_signals=("s_decay",),
            decay_kill_lower_min=0.5,
        ),
    )
    out = run_benchmark_seed(
        _PRESET, index, vcfg, git_state=git_state, patch=patch, work_dir=work_dir
    )
    passed = tuple(s.signal_id for s in out.signals if s.stage2_verdict is Verdict.PASS)
    return passed, out.dataset_seed


def run_task(
    scenario: str,
    index: int,
    kind: Kind,
    cfg: DroStudyConfig,
    *,
    root_seed: int,
    git_state: GitState,
    patch: bytes | None = None,
    work_dir: Path | None = None,
) -> TaskRecord:
    """Pure in everything but ``seconds``: depends only on its arguments' values."""
    t0 = time.perf_counter()
    factory = _seed_factory(root_seed, index)
    ds_seed = _u32(factory, "dataset")
    ds = generate_market(_scenario(scenario).config(cfg.n_obs), ds_seed)
    sub = factory.child(f"{scenario}/{kind}")
    all_arms = (Arm.EQUAL_WEIGHT, *CANDIDATES)
    sig = cfg.signals
    passed: tuple[str, ...] | None = None
    variants: list[VariantResult] = []
    if kind == "main":
        variants.append(
            _run_variant(
                ds,
                sig,
                all_arms,
                cfg.c_annual,
                cfg,
                sub,
                label="main",
                oracles=("o1", "o2"),
                tests=True,
            )
        )
    elif kind == "loso":
        for drop in sig:
            keep = [s for s in sig if s != drop]
            if keep:
                variants.append(
                    _run_variant(
                        ds,
                        keep,
                        CANDIDATES,
                        cfg.c_annual,
                        cfg,
                        sub,
                        label=f"drop:{drop}",
                        oracles=(),
                        tests=False,
                    )
                )
    elif kind == "pipeline":
        if root_seed != cfg.root_seed:  # the pilot: Stages 1-2 under the pilot's root seed
            cfg = cfg.model_copy(update={"root_seed": root_seed})
        if scenario != cfg.selection_scenario or _scenario(scenario).override:
            raise ValueError("the pipeline task runs on the unmodified multi_asset scenario")
        passed, pipe_seed = _pipeline_passed(cfg, index, git_state, patch, work_dir)
        if pipe_seed != ds_seed:
            raise RuntimeError(f"Stages 1-2 ran on dataset {pipe_seed}, not {ds_seed}")
        if passed:
            variants.append(
                _run_variant(
                    ds,
                    passed,
                    all_arms,
                    cfg.c_annual,
                    cfg,
                    sub,
                    label="pipeline",
                    oracles=("o1", "o2"),
                    tests=True,
                )
            )
    else:
        for c_annual in cfg.c_sensitivity:
            variants.append(
                _run_variant(
                    ds,
                    sig,
                    all_arms,
                    c_annual,
                    cfg,
                    sub,
                    label=f"c={c_annual:g}",
                    oracles=("o1",),
                    tests=True,
                )
            )
    return TaskRecord(
        scenario=scenario,
        index=index,
        kind=kind,
        dataset_seed=ds_seed,
        variants=tuple(variants),
        passed=passed,
        seconds=time.perf_counter() - t0,
    )


Task = tuple[str, int, Kind]


def run_records(
    tasks: Sequence[Task],
    cfg: DroStudyConfig,
    *,
    root_seed: int,
    git_state: GitState,
    patch: bytes | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
    chunk_size: int = 32,
) -> list[TaskRecord]:
    """Run ``tasks`` with ``n_jobs`` loky workers, resuming from ``checkpoint`` (JSONL).

    The result follows the order of ``tasks`` and does not depend on ``n_jobs``. BLAS is
    single-threaded in the workers.
    """
    done: dict[tuple[str, int, str], TaskRecord] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = TaskRecord.model_validate_json(line)
                done[rec.key] = rec
    todo = [t for t in tasks if t not in done]
    log = get_logger(__name__)
    log.info("dro_study_start", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            recs: list[TaskRecord] = Parallel()(
                delayed(run_task)(
                    sc, i, kind, cfg, root_seed=root_seed, git_state=git_state, patch=patch
                )
                for sc, i, kind in chunk
            )
            for rec in recs:
                done[rec.key] = rec
            if checkpoint is not None:
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.writelines(r.model_dump_json() + "\n" for r in recs)
            log.info("dro_study_progress", done=start + len(chunk), total=len(todo))
    return [done[t] for t in tasks]


# ---------------------------------------------------------------------------------------------
# M and the compute budget (ADR 0013)
# ---------------------------------------------------------------------------------------------


def gate_seeds(cfg: DroStudyConfig) -> int:
    """Seeds for the size-gate scenarios: ``required_seeds`` over every (gate, candidate)."""
    if cfg.n_gate_seeds is not None:
        return cfg.n_gate_seeds
    return required_seeds(cfg.lw_alpha, cfg.size_cutoff, cfg.n_gating, cfg.all_pass_prob)


def selection_seeds(
    sigma_hat: float,
    n_pilot: int,
    *,
    delta: float,
    power: float,
    sigma_confidence: float,
    floor: int,
    alpha: float = 0.05,
) -> int:
    """Seeds for which a paired mean difference of ``delta`` has a two-sided ``1 - alpha`` CI
    excluding 0 with probability ``power``: ``ceil(((z_{1-alpha/2} + z_power) sigma_U /
    delta)^2)``, at least ``floor``. ``sigma_U`` is the one-sided ``sigma_confidence`` upper
    bound for the SD from ``n_pilot`` pilot seeds (chi-square, ``n_pilot - 1`` df)."""
    df = n_pilot - 1
    sigma_u = sigma_hat * math.sqrt(df / float(stats.chi2.ppf(1.0 - sigma_confidence, df)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0)) + float(stats.norm.ppf(power))
    return max(floor, math.ceil((z * sigma_u / delta) ** 2))


class PilotReport(_Frozen):
    """What the pilot is used for: dispersion and timing. Never its rates or means."""

    n_seeds: int
    workers: int
    pair_sd: dict[str, float]
    """SD over pilot seeds of each candidate pair's Sharpe difference, selection scenario."""
    sigma_hat: float
    n_select_seeds: int
    seconds: dict[str, float]
    """Mean task wall time per ``scenario/kind``."""


def pilot_report(records: Sequence[TaskRecord], cfg: DroStudyConfig, workers: int) -> PilotReport:
    sel = sorted(
        (r for r in records if r.scenario == cfg.selection_scenario and r.kind == "main"),
        key=lambda r: r.index,
    )
    sr = {a: np.array([r.variants[0].arm(a).sharpe for r in sel]) for a in CANDIDATES}
    pair_sd = {
        f"{a.value}-{b.value}": float(np.std(sr[a] - sr[b], ddof=1))
        for a, b in itertools.combinations(CANDIDATES, 2)
    }
    sigma_hat = max(pair_sd.values())
    times: dict[str, list[float]] = {}
    for r in records:
        times.setdefault(f"{r.scenario}/{r.kind}", []).append(r.seconds)
    return PilotReport(
        n_seeds=len(sel),
        workers=workers,
        pair_sd=pair_sd,
        sigma_hat=sigma_hat,
        n_select_seeds=selection_seeds(
            sigma_hat,
            len(sel),
            delta=cfg.delta_sharpe,
            power=cfg.power,
            sigma_confidence=cfg.sigma_confidence,
            floor=cfg.min_select_seeds,
        ),
        seconds={k: float(np.mean(v)) for k, v in sorted(times.items())},
    )


REPORTED_KINDS: Final[tuple[Kind, ...]] = ("loso", "pipeline", "c_sens")


class BudgetPlan(_Frozen):
    """The mechanical budget decision (ADR 0013)."""

    budget_hours: float
    workers: int
    n_gate_seeds: int
    n_select_seeds: int
    n_report_seeds: int
    core_hours: float
    """Projected wall time of the gated and selection tasks."""
    projected_hours: float
    feasible: bool
    """False: even the core with ``min_report_seeds`` reported seeds exceeds the budget. The
    study is not run and the design goes back to review."""


def plan_budget(
    seconds: dict[str, float], cfg: DroStudyConfig, *, n_select: int, workers: int
) -> BudgetPlan:
    """Projected wall time = sum over tasks of the pilot's mean task seconds / ``workers``.

    The gated scenarios run at ``gate_seeds`` and the selection scenario at ``n_select``.
    Reported tasks (reported scenarios' ``main``, and ``loso``, ``pipeline`` and ``c_sens`` on
    the selection scenario) run on the first ``n_report`` seeds, ``n_report = n_select`` if that
    fits the budget. Otherwise it is the largest value that fits; if that is below
    ``min_report_seeds``, the plan is infeasible.
    """
    m_gate = gate_seeds(cfg)
    core = sum(seconds[f"{s}/main"] for s in cfg.size_gates) * m_gate
    core += seconds[f"{cfg.selection_scenario}/main"] * n_select
    per_report = sum(seconds[f"{s}/main"] for s in cfg.reported_scenarios)
    per_report += sum(seconds[f"{cfg.selection_scenario}/{k}"] for k in REPORTED_KINDS)
    budget = cfg.budget_hours * 3600.0 * workers
    if cfg.n_report_seeds is not None:
        m_rep = cfg.n_report_seeds
    else:
        m_rep = min(n_select, max(0, math.floor((budget - core) / per_report)))
    total = core + per_report * m_rep
    return BudgetPlan(
        budget_hours=cfg.budget_hours,
        workers=workers,
        n_gate_seeds=m_gate,
        n_select_seeds=n_select,
        n_report_seeds=m_rep,
        core_hours=core / 3600.0 / workers,
        projected_hours=total / 3600.0 / workers,
        feasible=m_rep >= min(cfg.min_report_seeds, n_select) and total <= budget,
    )


def study_tasks(cfg: DroStudyConfig, *, n_gate: int, n_select: int, n_report: int) -> list[Task]:
    tasks: list[Task] = []
    for s in cfg.size_gates:
        tasks += [(s, i, "main") for i in range(n_gate)]
    tasks += [(cfg.selection_scenario, i, "main") for i in range(n_select)]
    for s in cfg.reported_scenarios:
        tasks += [(s, i, "main") for i in range(n_report)]
    for k in REPORTED_KINDS:
        tasks += [(cfg.selection_scenario, i, k) for i in range(n_report)]
    return tasks


# ---------------------------------------------------------------------------------------------
# Summary and rules
# ---------------------------------------------------------------------------------------------


class ArmSummary(_Frozen):
    scenario: str
    variant: str
    arm: Arm
    n: int
    sharpe: float
    sharpe_net: float
    regret_o1: float | None
    regret_o2: float | None
    beats_equal_weight: RateEstimate | None
    """Kill-test rejection rate (``lw_p < lw_alpha``); None for 1/N and untested variants."""
    gate: bool
    invested: float
    cash: float
    partial: float
    turnover: float
    dsr_mean: float | None
    dsr_above_95: RateEstimate | None
    fc_coverage_o1: float | None
    fc_coverage_o2: float | None


class SelectionRow(_Frozen):
    arm: Arm
    admissible: bool
    mean_sharpe: float
    diff_from_best: float
    diff_ci_low: float
    diff_ci_high: float
    tied: bool


class LosoRow(_Frozen):
    arm: Arm
    dropped: str
    n: int
    marginal_sharpe: float
    """Mean of ``Sharpe(all signals) - Sharpe(without dropped)`` (diagnostic B)."""
    ci_low: float
    ci_high: float


class DroStudyReport(_Frozen):
    git_sha: str | None
    git_dirty: bool
    n_gate_seeds: int
    n_select_seeds: int
    n_report_seeds: int
    arms: tuple[ArmSummary, ...]
    failed_gates: tuple[str, ...]
    admissible: tuple[Arm, ...]
    selection: tuple[SelectionRow, ...]
    selected: Arm | None
    selection_reason: str
    loso: tuple[LosoRow, ...]
    pipeline_sets: dict[str, int]
    median_seconds: dict[str, float]


def _boot_mean_ci(d: _FloatArray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _mean_or_none(v: Sequence[float | None]) -> float | None:
    xs = [x for x in v if x is not None]
    return float(np.mean(xs)) if xs else None


def _summaries(recs: Sequence[TaskRecord], cfg: DroStudyConfig, gated: bool) -> list[ArmSummary]:
    out = []
    labels = sorted({v.label for r in recs for v in r.variants})
    for label in labels:
        vs = [v for r in recs for v in r.variants if v.label == label]
        for arm in [m.arm for m in vs[0].arms]:
            ms = [v.arm(arm) for v in vs]
            o1 = [next((o.sharpe for o in v.oracles if o.info == "o1"), None) for v in vs]
            o2 = [next((o.sharpe for o in v.oracles if o.info == "o2"), None) for v in vs]
            ps = [m.lw_p for m in ms if m.lw_p is not None]
            dsr = [m.dsr for m in ms if m.dsr is not None]
            out.append(
                ArmSummary(
                    scenario=recs[0].scenario,
                    variant=label,
                    arm=arm,
                    n=len(ms),
                    sharpe=float(np.mean([m.sharpe for m in ms])),
                    sharpe_net=float(np.mean([m.sharpe_net for m in ms])),
                    regret_o1=_mean_or_none(
                        [None if a is None else a - m.sharpe for a, m in zip(o1, ms, strict=True)]
                    ),
                    regret_o2=_mean_or_none(
                        [None if a is None else a - m.sharpe for a, m in zip(o2, ms, strict=True)]
                    ),
                    beats_equal_weight=wilson_interval(
                        sum(p < cfg.lw_alpha for p in ps), len(ps), cfg.confidence
                    )
                    if ps
                    else None,
                    gate=gated and arm in CANDIDATES,
                    invested=float(np.mean([m.invested for m in ms])),
                    cash=float(np.mean([m.cash for m in ms])),
                    partial=float(np.mean([m.partial for m in ms])),
                    turnover=float(np.mean([m.turnover for m in ms])),
                    dsr_mean=float(np.mean(dsr)) if dsr else None,
                    dsr_above_95=wilson_interval(
                        sum(d > 0.95 for d in dsr), len(dsr), cfg.confidence
                    )
                    if dsr
                    else None,
                    fc_coverage_o1=_mean_or_none([m.fc_coverage_o1 for m in ms]),
                    fc_coverage_o2=_mean_or_none([m.fc_coverage_o2 for m in ms]),
                )
            )
    return out


def summarize(
    records: Sequence[TaskRecord],
    cfg: DroStudyConfig,
    *,
    git_sha: str | None,
    git_dirty: bool,
) -> DroStudyReport:
    """Apply the pre-registered rules (module docstring) to ``records``."""
    by: dict[tuple[str, str], list[TaskRecord]] = {}
    for r in records:
        by.setdefault((r.scenario, r.kind), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r.index)

    arms: list[ArmSummary] = []
    for (scen, _kind), recs in sorted(by.items()):
        arms += _summaries(recs, cfg, gated=scen in cfg.size_gates)

    failed = []
    for s in arms:
        if s.gate and s.variant == "main":
            rate = s.beats_equal_weight
            assert rate is not None and rate.ci_high is not None
            if rate.ci_high > cfg.size_cutoff:
                failed.append(f"{s.scenario}/{s.arm.value}")
    admissible = tuple(
        a for a in cfg.simplicity if not any(f.endswith(f"/{a.value}") for f in failed)
    )

    sel_recs = by.get((cfg.selection_scenario, "main"), [])
    rng = RngFactory(cfg.root_seed).generator("selection_bootstrap")
    rows: list[SelectionRow] = []
    selected: Arm | None = None
    reason = "no admissible candidate: the stage is not built"
    if sel_recs:
        sr = {a: np.array([r.variants[0].arm(a).sharpe for r in sel_recs]) for a in CANDIDATES}
        pool = admissible or CANDIDATES
        best = max(pool, key=lambda a: (float(sr[a].mean()), -cfg.simplicity.index(a)))
        for a in cfg.simplicity:
            d = sr[best] - sr[a]
            lo, hi = (0.0, 0.0) if a is best else _boot_mean_ci(d, rng, cfg.n_diff_boot)
            rows.append(
                SelectionRow(
                    arm=a,
                    admissible=a in admissible,
                    mean_sharpe=float(sr[a].mean()),
                    diff_from_best=float(d.mean()),
                    diff_ci_low=lo,
                    diff_ci_high=hi,
                    tied=a in admissible and lo <= 0.0,
                )
            )
        if admissible:
            selected = next(row.arm for row in rows if row.tied)
            reason = (
                f"highest mean Sharpe: {best.value}; tied (CI of the difference contains 0): "
                f"{[row.arm.value for row in rows if row.tied]}; first in simplicity order"
            )

    loso: list[LosoRow] = []
    loso_recs = by.get((cfg.selection_scenario, "loso"), [])
    if loso_recs:
        full = {r.index: r.variants[0] for r in sel_recs}
        for a in CANDIDATES:
            for drop in cfg.signals:
                label = f"drop:{drop}"
                d = np.array(
                    [
                        full[r.index].arm(a).sharpe - v.arm(a).sharpe
                        for r in loso_recs
                        for v in r.variants
                        if v.label == label and r.index in full
                    ]
                )
                if len(d):
                    lo, hi = _boot_mean_ci(d, rng, cfg.n_diff_boot)
                    loso.append(
                        LosoRow(
                            arm=a,
                            dropped=drop,
                            n=len(d),
                            marginal_sharpe=float(d.mean()),
                            ci_low=lo,
                            ci_high=hi,
                        )
                    )

    sets: dict[str, int] = {}
    for r in by.get((cfg.selection_scenario, "pipeline"), []):
        key = ",".join(r.passed or ()) or "(none)"
        sets[key] = sets.get(key, 0) + 1

    times: dict[str, list[float]] = {}
    for r in records:
        times.setdefault(f"{r.scenario}/{r.kind}", []).append(r.seconds)
    n_rep = max(
        (len(v) for (s, k), v in by.items() if k != "main" or s in cfg.reported_scenarios),
        default=0,
    )
    return DroStudyReport(
        git_sha=git_sha,
        git_dirty=git_dirty,
        n_gate_seeds=max((len(by.get((s, "main"), [])) for s in cfg.size_gates), default=0),
        n_select_seeds=len(sel_recs),
        n_report_seeds=n_rep,
        arms=tuple(arms),
        failed_gates=tuple(failed),
        admissible=admissible,
        selection=tuple(rows),
        selected=selected,
        selection_reason=reason,
        loso=tuple(loso),
        pipeline_sets=sets,
        median_seconds={k: float(np.median(v)) for k, v in sorted(times.items())},
    )


def write_report(report: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
