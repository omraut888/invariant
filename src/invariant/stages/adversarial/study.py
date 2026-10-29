"""Stage 4 validation study on the planted-signal benchmark (docs/adr/0016, "Pre-registration").

Tasks, one replication each (seed i from ``RngFactory(root).child(f"seed/{i}")``):

- ``volfrag`` (gated). The `volfrag` market, with two forced member sets on the same market:
  - {s_volfrag} (S2: Stage 4 should kill);
  - {s_real} (S4: Stage 4 should pass).
  The {s_real} certification's region also gives S1: does it contain the state's population
  (pi_s, d_s)?
- ``null`` (gated, S3). `volfrag_null`, with {s_real, s_volfrag} forced (both zero edge, mu = 0):
  every state's population P&L mean is 0, so a kill is a false kill.
- ``noregime`` (gated, S5). `multi_asset` (unchanged), with {s_real} forced: Stage 4 should pass.
- ``regime`` (reported). `regime`, with {s_regime} forced: the latent-regime signal that ADR 0016
  found unidentifiable.
- ``pipeline`` (reported). `volfrag` through Stages 1-3 on the production path; Stage 4 then runs
  on the members that passed Stage 3.

The forced-member tasks run Stage 3's walk-forward (1/N and Ridge-CV, configs/base.yaml's
portfolio settings, execution lag 0) directly on the generated arrays, as ADR 0013 did, and
report Stage 3's kill-rule-A p-value on the same paths. Stage 4 is ``certify`` on Ridge-CV's and
1/N's P&L with the vol21, q = .70 state.
"""

from __future__ import annotations

import math
import statistics
import tempfile
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml
from joblib import Parallel, delayed, parallel_config
from pydantic import BaseModel, ConfigDict, Field

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.payload import synthetic_etf_payload, synthetic_feature_payload
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import (
    ConfigError,
    PipelineConfig,
    PortfolioConfig,
    Settings,
    find_project_root,
    load_config,
)
from invariant.core.logging import get_logger
from invariant.core.run_hash import GitState, RngFactory, RunContext
from invariant.core.types import Verdict
from invariant.data.pit import build_pit_panel
from invariant.data.snapshot import write_snapshot
from invariant.ledger.benchmark_score import RateEstimate, wilson_interval
from invariant.ledger.kill_ledger import KillLedger
from invariant.stages.adversarial.certify import CertificationConfig, certify
from invariant.stages.adversarial.region import chain_estimate
from invariant.stages.adversarial.state import volatility_state
from invariant.stages.causal.stage import (
    record_causal_ledger,
    run_causal_stage,
    write_causal_result,
)
from invariant.stages.causal.validation import pipeline_config
from invariant.stages.decay.stage import record_decay_ledger, run_decay_stage, write_decay_result
from invariant.stages.decay.validation import acceptance_pass_probability
from invariant.stages.dro.metrics import sharpe_difference_test
from invariant.stages.dro.walkforward import Arm, run_walk_forward
from invariant.stages.portfolio.stage import (
    record_portfolio_ledger,
    run_portfolio_stage,
    walk_config,
    write_portfolio_result,
)

__all__ = [
    "DEFAULT_STUDY_CONFIG",
    "Outcome",
    "StudyConfig",
    "StudyReport",
    "TaskRecord",
    "budget_plan",
    "load_study_config",
    "pilot_thresholds",
    "population_chain",
    "run_study",
    "run_task",
    "study_tasks",
    "summarize_study",
]

DEFAULT_STUDY_CONFIG: Final = Path("configs/adversarial_study.yaml")
Kind = Literal["volfrag", "null", "noregime", "regime", "pipeline"]
Task = tuple[Kind, int]
GATED: Final[tuple[Kind, ...]] = ("volfrag", "null", "noregime")
REPORTED: Final[tuple[Kind, ...]] = ("regime", "pipeline")
PRESET: Final[dict[Kind, str]] = {
    "volfrag": "volfrag",
    "null": "volfrag_null",
    "noregime": "multi_asset",
    "regime": "regime",
    "pipeline": "volfrag",
}


BUDGET_HOURS: Final = 24.0
MIN_REPORT: Final = 100


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Thresholds(_Frozen):
    """S2, S4 and S5 thresholds, set by ``pilot_thresholds`` and recorded before the full run."""

    s2_kill_lower_min: float
    s4_pass_lower_min: float
    s5_pass_lower_min: float


class StudyConfig(_Frozen):
    root_seed: int = Field(ge=0)
    pilot_root_seed: int = Field(ge=0)
    n_seeds: int = Field(ge=1)
    n_report_seeds: int = Field(ge=1)
    n_pilot: int = Field(ge=1)
    confidence: float = Field(gt=0.0, lt=1.0)
    size_upper_max: float = Field(gt=0.0, lt=1.0)
    pilot_margin: float = Field(ge=0.0, lt=1.0)
    s2_floor: float = Field(gt=0.0, lt=1.0)
    pass_floor: float = Field(gt=0.0, lt=1.0)
    population_n_obs: int = Field(ge=10_000)
    population_seed: int = Field(ge=0)
    causal_n_boot: int | None = None
    n_obs: int | None = None
    """None: each preset's own length (6000). Set only by the fast-suite smoke."""
    overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    """Pipeline-config sections to override (as in ADR 0014's harness), e.g. ``portfolio`` and
    ``decay`` lengths. Empty in the study; the fast-suite smoke uses it so one tiny seed reaches
    every code path, including the pipeline's Stage 4."""
    certification: CertificationConfig = CertificationConfig()
    thresholds: Thresholds | None = None
    """None until the pilot has run and its thresholds are recorded (ADR 0016, Pilot)."""


def load_study_config(path: Path | None = None) -> StudyConfig:
    resolved = path or DEFAULT_STUDY_CONFIG
    if not resolved.is_absolute():
        resolved = find_project_root() / resolved
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"study config not found: {resolved}") from None
    return StudyConfig.model_validate(data)


class Outcome(_Frozen):
    """Stage 3's p-value and Stage 4's decision for one member set."""

    members: tuple[str, ...]
    stage3_p: float
    kill: bool
    sr_h_star: float
    sr_h_star_upper: float
    sr_h_baseline: float
    sharpe_by_state: tuple[float, float]
    min_difference_to_equal_weight: float
    worst_share: float
    worst_duration: float
    worst_start: str
    block_length: int
    region_share: tuple[float, float]
    region_duration: tuple[float, float]
    contains_population: bool | None = None


class SignalOutcome(_Frozen):
    signal_id: str
    stage1: Verdict
    stage2: Verdict | None
    stage3: Verdict | None
    stage4: Verdict | None


class TaskRecord(_Frozen):
    kind: Kind
    index: int
    dataset_seed: int
    pipeline_seed: int
    outcomes: tuple[Outcome, ...]
    signals: tuple[SignalOutcome, ...] = ()
    seconds: float

    def outcome(self, members: tuple[str, ...]) -> Outcome | None:
        return next((o for o in self.outcomes if o.members == members), None)


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def _pipeline_base(cfg: StudyConfig) -> PipelineConfig:
    config = load_config()
    for section, values in cfg.overrides.items():
        current = getattr(config, section)
        updated = type(current).model_validate({**current.model_dump(), **values})
        config = config.model_copy(update={section: updated})
    return config


def _portfolio(cfg: StudyConfig) -> PortfolioConfig:
    return _pipeline_base(cfg).portfolio


def _dataset(kind: Kind, cfg: StudyConfig, seed: int) -> SyntheticDataset:
    bench = load_benchmark_config(PRESET[kind])
    if cfg.n_obs is not None:
        bench = bench.model_copy(update={"n_obs": cfg.n_obs})
    return generate_market(bench, seed)


@lru_cache(maxsize=4)
def population_chain(n_obs: int, seed: int, window: int = 1260) -> tuple[float, float]:
    """The vol21, q = .70 state's population (pi_s, d_s) on the `volfrag` market: the state
    computed on one ``n_obs``-day path (decision dates from ``window`` on) with its own
    quantile, by the same estimator as the region (``region.chain_estimate``)."""
    cfg = load_benchmark_config("volfrag").model_copy(update={"n_obs": n_obs})
    r = generate_market(cfg, seed).returns.to_numpy()
    dates = np.arange(window, len(r) - 1, dtype=np.int64)
    states, _ = volatility_state(r.mean(axis=1), dates)
    return chain_estimate(states[:-1], states[1:])


def _certify_members(
    dataset: SyntheticDataset,
    members: tuple[str, ...],
    cfg: StudyConfig,
    rng: RngFactory,
    population: tuple[float, float] | None = None,
) -> Outcome:
    portfolio = _portfolio(cfg)
    wf = walk_config(portfolio, 0)
    r = np.ascontiguousarray(dataset.returns.to_numpy())
    x = np.ascontiguousarray(dataset.features[list(members)].to_numpy())
    paths = run_walk_forward(x, r, [Arm.EQUAL_WEIGHT, Arm.RIDGE_CV], wf, rng.child("wf"))
    rc, ew = paths[Arm.RIDGE_CV], paths[Arm.EQUAL_WEIGHT]
    _, p3 = sharpe_difference_test(
        rc.returns, ew.returns, rng.generator("lw"), n_boot=portfolio.lw_n_boot
    )
    states, _ = volatility_state(r.mean(axis=1), rc.dates)
    res = certify(rc.returns, ew.returns, states, cfg.certification, rng.child("cert"))
    g = res.region
    return Outcome(
        members=members,
        stage3_p=float(p3),
        kill=res.kill,
        sr_h_star=res.sr_h_star,
        sr_h_star_upper=res.sr_h_star_upper,
        sr_h_baseline=res.sr_h_baseline,
        sharpe_by_state=res.sharpe_by_state,
        min_difference_to_equal_weight=res.min_difference_to_equal_weight,
        worst_share=res.worst_share,
        worst_duration=res.worst_duration,
        worst_start=res.worst_start.value,
        block_length=res.block_length,
        region_share=g.share_range,
        region_duration=(math.exp(g.log_duration_range[0]), math.exp(g.log_duration_range[1])),
        contains_population=None if population is None else g.contains(*population),
    )


def _pipeline(
    dataset: SyntheticDataset,
    seed: int,
    cfg: StudyConfig,
    git_state: GitState,
    patch: bytes | None,
    work_dir: Path | None,
) -> tuple[tuple[Outcome, ...], tuple[SignalOutcome, ...]]:
    config = pipeline_config(_pipeline_base(cfg), dataset, seed=seed, n_boot=cfg.causal_n_boot)
    dataset.ground_truth.require_target_timing(config.pit.targets)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        settings = Settings(_env_file=None, data_dir=root / "data", runs_dir=root / "runs")
        snapshot = write_snapshot(
            [synthetic_etf_payload(dataset), synthetic_feature_payload(dataset)],
            settings.data_dir / "snapshots",
        )
        panel = build_pit_panel(snapshot, config)
        ctx = RunContext.create(
            config, snapshot.snapshot_id, settings=settings, git_state=git_state, patch=patch
        )
        try:
            causal = run_causal_stage(panel, config.causal, ctx.rng)
            ledger = KillLedger.open(ctx)
            record_causal_ledger(ledger, causal)
            write_causal_result(ctx.run_dir, causal)
            decay = run_decay_stage(panel, causal, config.decay, ctx.rng)
            record_decay_ledger(ledger, decay)
            write_decay_result(ctx.run_dir, decay)
            portfolio = run_portfolio_stage(panel, decay, config.portfolio, ctx.rng)
            record_portfolio_ledger(ledger, portfolio)
            write_portfolio_result(ctx.run_dir, portfolio)
            ctx.mark_completed()
        except BaseException as exc:
            ctx.mark_failed(exc)
            raise
    passed = tuple(m.signal_id for m in portfolio.members if m.verdict is Verdict.PASS)
    outcomes: tuple[Outcome, ...] = ()
    if passed:
        outcomes = (_certify_members(dataset, passed, cfg, RngFactory(seed).child("stage4")),)
    stage4 = None if not outcomes else (Verdict.KILL if outcomes[0].kill else Verdict.PASS)
    s2 = {s.signal_id: s.verdict for s in decay.signals}
    s3 = {m.signal_id: m.verdict for m in portfolio.members}
    signals = tuple(
        SignalOutcome(
            signal_id=link.signal_id,
            stage1=link.verdict,
            stage2=s2.get(link.signal_id),
            stage3=s3.get(link.signal_id),
            stage4=stage4 if link.signal_id in passed else None,
        )
        for link in causal.signals
    )
    return outcomes, signals


def run_task(
    kind: Kind,
    index: int,
    cfg: StudyConfig,
    *,
    root_seed: int,
    git_state: GitState,
    patch: bytes | None = None,
    work_dir: Path | None = None,
) -> TaskRecord:
    """One replication (module docstring). Pure in all but ``seconds``."""
    t0 = time.perf_counter()
    factory = RngFactory(root_seed).child(f"seed/{index}")
    ds_seed, pl_seed = _u32(factory, "dataset"), _u32(factory, "pipeline")
    dataset = _dataset(kind, cfg, ds_seed)
    rng = RngFactory(pl_seed)
    signals: tuple[SignalOutcome, ...] = ()
    outcomes: tuple[Outcome, ...]
    if kind == "volfrag":
        pop = population_chain(cfg.population_n_obs, cfg.population_seed, _portfolio(cfg).window)
        outcomes = (
            _certify_members(dataset, ("s_volfrag",), cfg, rng.child("frag")),
            _certify_members(dataset, ("s_real",), cfg, rng.child("robust"), pop),
        )
    elif kind == "null":
        outcomes = (_certify_members(dataset, ("s_real", "s_volfrag"), cfg, rng.child("null")),)
    elif kind == "noregime":
        outcomes = (_certify_members(dataset, ("s_real",), cfg, rng.child("noregime")),)
    elif kind == "regime":
        outcomes = (_certify_members(dataset, ("s_regime",), cfg, rng.child("regime")),)
    else:
        outcomes, signals = _pipeline(dataset, pl_seed, cfg, git_state, patch, work_dir)
    return TaskRecord(
        kind=kind,
        index=index,
        dataset_seed=ds_seed,
        pipeline_seed=pl_seed,
        outcomes=outcomes,
        signals=signals,
        seconds=time.perf_counter() - t0,
    )


def study_tasks(n_seeds: int, n_report: int) -> list[Task]:
    tasks: list[Task] = [(k, i) for k in GATED for i in range(n_seeds)]
    tasks += [(k, i) for k in REPORTED for i in range(n_report)]
    return tasks


def run_study(
    cfg: StudyConfig,
    *,
    root_seed: int,
    n_seeds: int,
    n_report: int,
    git_state: GitState,
    patch: bytes | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
    chunk_size: int = 48,
) -> list[TaskRecord]:
    """Every task with joblib (loky) workers, in ``study_tasks`` order, independent of
    ``n_jobs``. Resumes from ``checkpoint`` (JSONL)."""
    tasks = study_tasks(n_seeds, n_report)
    done: dict[Task, TaskRecord] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = TaskRecord.model_validate_json(line)
                done[(rec.kind, rec.index)] = rec
    todo = [t for t in tasks if t not in done]
    log = get_logger(__name__)
    log.info("adversarial_study_start", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            recs: list[TaskRecord] = Parallel()(
                delayed(run_task)(k, i, cfg, root_seed=root_seed, git_state=git_state, patch=patch)
                for k, i in chunk
            )
            for rec in recs:
                done[(rec.kind, rec.index)] = rec
            if checkpoint is not None:
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.writelines(r.model_dump_json() + "\n" for r in recs)
            log.info("adversarial_study_progress", done=start + len(chunk), total=len(todo))
    return [done[t] for t in tasks]


# ---------------------------------------------------------------------------------------------
# Rates, pilot thresholds and acceptance
# ---------------------------------------------------------------------------------------------


def _rates(records: Sequence[TaskRecord], conf: float) -> dict[str, RateEstimate]:
    """The five gate rates (S1 miss, S2 kill, S3 kill, S4 pass, S5 pass) over the records."""

    def rate(flags: list[bool]) -> RateEstimate:
        return wilson_interval(sum(flags), len(flags), conf)

    vf = [r for r in records if r.kind == "volfrag"]
    frag = [o for r in vf if (o := r.outcome(("s_volfrag",))) is not None]
    robust = [o for r in vf if (o := r.outcome(("s_real",))) is not None]
    null = [o for r in records if r.kind == "null" for o in r.outcomes]
    noreg = [o for r in records if r.kind == "noregime" for o in r.outcomes]
    return {
        "S1_population_miss": rate([not bool(o.contains_population) for o in robust]),
        "S2_frag_kill": rate([o.kill for o in frag]),
        "S3_null_kill": rate([o.kill for o in null]),
        "S4_robust_pass": rate([not o.kill for o in robust]),
        "S5_noregime_pass": rate([not o.kill for o in noreg]),
    }


class PilotThreshold(_Frozen):
    gate: str
    successes: int
    trials: int
    wilson_lower: float
    threshold: float
    floor: float
    feasible: bool
    pass_probability_at_estimate: float
    pass_probability_at_wilson_lower: float
    pass_probability_at_threshold: float


def pilot_thresholds(records: Sequence[TaskRecord], cfg: StudyConfig) -> list[PilotThreshold]:
    """ADR 0016's rule: threshold = floor_.01(Wilson lower of the pilot rate - margin), checked
    against its floor, with exact pass probabilities at M = ``cfg.n_seeds``."""
    rates = _rates(records, cfg.confidence)
    out = []
    for gate, floor in (
        ("S2_frag_kill", cfg.s2_floor),
        ("S4_robust_pass", cfg.pass_floor),
        ("S5_noregime_pass", cfg.pass_floor),
    ):
        r = rates[gate]
        lower = r.ci_low or 0.0
        threshold = math.floor((lower - cfg.pilot_margin) * 100 + 1e-9) / 100

        def pp(rate: float, t: float = threshold) -> float:
            return acceptance_pass_probability(
                cfg.n_seeds, 1.0, min(max(rate, 0.0), 1.0), t, "lower_min", cfg.confidence
            )

        out.append(
            PilotThreshold(
                gate=gate,
                successes=r.successes,
                trials=r.trials,
                wilson_lower=lower,
                threshold=threshold,
                floor=floor,
                feasible=threshold >= floor,
                pass_probability_at_estimate=pp(r.successes / max(r.trials, 1)),
                pass_probability_at_wilson_lower=pp(lower),
                pass_probability_at_threshold=pp(threshold),
            )
        )
    return out


class Criterion(_Frozen):
    name: str
    rate: RateEstimate
    bound: float
    side: Literal["upper_max", "lower_min"]
    threshold: float
    passed: bool


class StudyReport(_Frozen):
    n_records: int
    criteria: tuple[Criterion, ...]
    accepted: bool
    reported: dict[str, float]


def summarize_study(records: Sequence[TaskRecord], cfg: StudyConfig) -> StudyReport:
    """Acceptance (S1-S5) and the reported quantities. Needs ``cfg.thresholds``."""
    if cfg.thresholds is None:
        raise ValueError("the S2/S4/S5 thresholds are set by the pilot and must be recorded first")
    rates = _rates(records, cfg.confidence)
    th = cfg.thresholds
    spec: list[tuple[str, Literal["upper_max", "lower_min"], float]] = [
        ("S1_population_miss", "upper_max", cfg.size_upper_max),
        ("S3_null_kill", "upper_max", cfg.size_upper_max),
        ("S2_frag_kill", "lower_min", th.s2_kill_lower_min),
        ("S4_robust_pass", "lower_min", th.s4_pass_lower_min),
        ("S5_noregime_pass", "lower_min", th.s5_pass_lower_min),
    ]
    criteria = []
    for name, side, threshold in spec:
        r = rates[name]
        bound = (r.ci_high or 1.0) if side == "upper_max" else (r.ci_low or 0.0)
        ok = bound <= threshold if side == "upper_max" else bound >= threshold
        criteria.append(
            Criterion(name=name, rate=r, bound=bound, side=side, threshold=threshold, passed=ok)
        )
    reported: dict[str, float] = {}
    regime = [o for r in records if r.kind == "regime" for o in r.outcomes]
    if regime:
        reported["regime_kill_rate"] = float(np.mean([o.kill for o in regime]))
    vf = [r for r in records if r.kind == "volfrag"]
    frag = [o for r in vf if (o := r.outcome(("s_volfrag",))) is not None]
    passed3 = [o for o in frag if o.stage3_p < load_config().portfolio.lw_alpha]
    if passed3:
        reported["frag_stage4_kill_given_stage3_pass"] = float(np.mean([o.kill for o in passed3]))
        reported["frag_stage3_pass_rate"] = len(passed3) / len(frag)
    pipe = [r for r in records if r.kind == "pipeline"]
    for sid in ("s_real", "s_volfrag", "s_conf", "s_noise", "s_decay"):
        reached = [
            s for r in pipe for s in r.signals if s.signal_id == sid and s.stage4 is not None
        ]
        if pipe:
            reported[f"pipeline_{sid}_reaches_stage4"] = len(reached) / len(pipe)
        if reached:
            reported[f"pipeline_{sid}_stage4_kill_given_reached"] = float(
                np.mean([s.stage4 is Verdict.KILL for s in reached])
            )
    reported["median_seconds"] = float(np.median([r.seconds for r in records])) if records else 0.0
    return StudyReport(
        n_records=len(records),
        criteria=tuple(criteria),
        accepted=all(c.passed for c in criteria),
        reported=reported,
    )


def budget_plan(
    records: Sequence[TaskRecord], cfg: StudyConfig, n_jobs: int
) -> dict[str, float | int | bool]:
    """ADR 0016's budget rule (mechanical; not revisited after results). The core is the gated
    tasks at M; the reported tasks run at ``n_report_seeds`` unless that overruns 24 h on
    ``n_jobs`` workers, then at the largest count that fits. The study is not run if the core
    alone overruns or the reported count would fall below 100. Timings are the pilot's means,
    measured under the same worker load."""
    sec = {k: statistics.mean(r.seconds for r in records if r.kind == k) for k in GATED + REPORTED}
    core = sum(sec[k] for k in GATED) * cfg.n_seeds / n_jobs / 3600
    per_report = sum(sec[k] for k in REPORTED) / n_jobs / 3600
    n_rep = cfg.n_report_seeds
    if core + per_report * n_rep > BUDGET_HOURS:
        n_rep = int((BUDGET_HOURS - core) / per_report) if per_report > 0 else 0
    return {
        **{f"seconds_{k}": v for k, v in sec.items()},
        "core_hours": core,
        "total_hours": core + per_report * n_rep,
        "n_report_seeds": n_rep,
        "feasible": core <= BUDGET_HOURS and n_rep >= MIN_REPORT,
    }
