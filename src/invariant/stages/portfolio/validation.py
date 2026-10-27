"""Monte Carlo validation of Stage 3 on the planted-signal benchmark (docs/adr/0014).

Method
------
Two task kinds, both through the production code path (snapshot -> PIT panel -> stage):

- **size** (gate V1): on a null scenario of ADR 0013 (every true mean edge and the drift set to
  0, so no portfolio truly beats 1/N), at ``size.n_obs`` rows, with ``size.members`` forced in
  as members (Stage 1 would rarely pass them on a null). A pass is a false pass; its rate is the
  kill test's size at the stage's minimum sample length.
- **pipeline** (gate V2, V3): ADR 0013's selection market (``multi_asset``), Stage 1 -> ledger
  -> Stage 2 -> ledger -> Stage 3 -> ledger, scored with ``score_run`` over (causal, decay,
  portfolio). V2 is ``power.signal``'s Stage 3 pass rate **given that it reached Stage 3**.

Seeds: replication ``i`` takes its dataset and pipeline seeds from
``RngFactory(root_seed).child("seed/i")`` as the earlier validations do, so the task kinds share
seeds (common random numbers) and serial and parallel runs are identical. Every benchmark task
checks the target timing against ``GroundTruth`` before any stage runs.

Reference: Wilson (1927), JASA 22(158).
"""

from __future__ import annotations

import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml
from joblib import Parallel, delayed, parallel_config
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from invariant.benchmark.payload import synthetic_etf_payload, synthetic_feature_payload
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import (
    ConfigError,
    PipelineConfig,
    Settings,
    find_project_root,
    load_config,
)
from invariant.core.logging import get_logger
from invariant.core.run_hash import GitState, RngFactory, RunContext
from invariant.core.types import StageName, Verdict
from invariant.data.pit import build_pit_panel
from invariant.data.snapshot import write_snapshot
from invariant.ledger.benchmark_score import (
    AggregateScore,
    BenchmarkScore,
    RateEstimate,
    aggregate_scores,
    score_run,
    wilson_interval,
)
from invariant.ledger.kill_ledger import KillLedger
from invariant.stages.causal.stage import (
    SYNTHETIC_PREFIX,
    record_causal_ledger,
    run_causal_stage,
    write_causal_result,
)
from invariant.stages.causal.validation import pipeline_config
from invariant.stages.decay.stage import record_decay_ledger, run_decay_stage, write_decay_result
from invariant.stages.dro.study import Scenario, dro_scenarios
from invariant.stages.dro.walkforward import Arm
from invariant.stages.portfolio.result import DECIDING_ARM, ArmReport, PortfolioResult
from invariant.stages.portfolio.stage import (
    ARMS,
    record_portfolio_ledger,
    run_portfolio_members,
    run_portfolio_stage,
    write_portfolio_result,
)

__all__ = [
    "DEFAULT_VALIDATION_CONFIG",
    "PortfolioValidationConfig",
    "PortfolioValidationReport",
    "TaskRecord",
    "load_portfolio_validation_config",
    "run_task",
    "run_validation",
    "summarize_validation",
    "validation_tasks",
]

DEFAULT_VALIDATION_CONFIG: Final = Path("configs/portfolio_stage_validation.yaml")
STAGES: Final = (StageName.CAUSAL, StageName.DECAY, StageName.PORTFOLIO)
PIPELINE_SCENARIO: Final = "d1_base"

Kind = Literal["size", "pipeline"]
Task = tuple[Kind, str, int]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SizeSettings(_Frozen):
    scenarios: tuple[str, ...] = Field(min_length=1)
    n_obs: StrictInt = Field(ge=200)
    members: tuple[str, ...] = Field(min_length=1)
    upper_max: float = Field(gt=0.0, lt=1.0)
    """V1: the false-pass rate's Wilson upper bound must be <= this, in every scenario."""


class PowerSettings(_Frozen):
    signal: str
    lower_min: float = Field(gt=0.0, lt=1.0)
    """V2: the signal's Stage 3 pass rate given it reached Stage 3 needs a Wilson lower bound
    >= this."""


class PortfolioValidationConfig(_Frozen):
    root_seed: StrictInt = Field(ge=0, lt=2**32)
    n_seeds: StrictInt = Field(ge=1)
    confidence: float = Field(gt=0.0, lt=1.0)
    n_boot: StrictInt | None = Field(ge=99)
    """Overrides Stage 1's ``causal.bootstrap.n_boot`` (descriptive interval only)."""
    pipeline_n_obs: StrictInt | None = Field(default=None, ge=200)
    """Overrides the pipeline market's n_obs (tests only; ``None`` pre-registered)."""
    overrides: dict[str, dict[str, object]] = Field(default_factory=dict)
    """Pipeline config section overrides, e.g. ``{"portfolio": {"min_obs": 300}}`` (tests only;
    empty pre-registered)."""
    size: SizeSettings
    power: PowerSettings

    @model_validator(mode="after")
    def _check(self) -> PortfolioValidationConfig:
        names = {s.name for s in dro_scenarios()}
        unknown = sorted(set(self.size.scenarios) - names)
        if unknown:
            raise ValueError(f"unknown scenarios: {unknown}")
        if PIPELINE_SCENARIO in self.size.scenarios:
            raise ValueError("the size gate runs on null scenarios, not the pipeline market")
        return self


def load_portfolio_validation_config(path: Path | None = None) -> PortfolioValidationConfig:
    p = path if path is not None else DEFAULT_VALIDATION_CONFIG
    p = p if p.is_absolute() else find_project_root() / p
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"validation config not found: {p}") from None
    return PortfolioValidationConfig.model_validate(data)


# ---------------------------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------------------------


class SignalOutcome(_Frozen):
    signal_id: str
    stage1: Verdict
    stage2: Verdict | None
    stage3: Verdict | None
    """``None`` when the signal never reached the stage."""
    stage3_reason: str | None


class MarginalOutcome(_Frozen):
    signal_id: str
    marginal_sharpe: float
    ci_low: float
    ci_high: float


class TaskRecord(_Frozen):
    kind: Kind
    scenario: str
    index: int
    dataset_seed: int
    pipeline_seed: int
    n_obs: int
    members: tuple[str, ...]
    verdict: Verdict | None
    """The portfolio's verdict; ``None`` when no signal reached Stage 3."""
    reason: str | None
    arms: tuple[ArmReport, ...]
    marginals: tuple[MarginalOutcome, ...]
    signals: tuple[SignalOutcome, ...]
    """Pipeline tasks only."""
    score: BenchmarkScore | None
    """Pipeline tasks only."""
    seconds: float
    """Wall time. Not part of the reproducible result."""

    def arm(self, arm: Arm) -> ArmReport | None:
        return next((a for a in self.arms if a.arm is arm), None)


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def _scenario(name: str) -> Scenario:
    return next(s for s in dro_scenarios() if s.name == name)


def _config(cfg: PortfolioValidationConfig, dataset: SyntheticDataset, seed: int) -> PipelineConfig:
    config = pipeline_config(load_config(), dataset, seed=seed, n_boot=cfg.n_boot)
    for section, values in cfg.overrides.items():
        current = getattr(config, section)
        config = config.model_copy(update={section: current.model_copy(update=values)})
    # The guard every benchmark task passes before any stage runs (ADR 0006).
    dataset.ground_truth.require_target_timing(config.pit.targets)
    return config


def _evidence(result: PortfolioResult) -> tuple[tuple[ArmReport, ...], tuple[MarginalOutcome, ...]]:
    arms = () if result.evidence is None else result.evidence.arms
    marginals = tuple(
        MarginalOutcome(
            signal_id=m.signal_id,
            marginal_sharpe=m.marginal_sharpe,
            ci_low=m.marginal_ci_low,
            ci_high=m.marginal_ci_high,
        )
        for m in result.members
        if m.marginal_sharpe is not None
        and m.marginal_ci_low is not None
        and m.marginal_ci_high is not None
    )
    return arms, marginals


def run_task(
    kind: Kind,
    scenario: str,
    index: int,
    cfg: PortfolioValidationConfig,
    *,
    git_state: GitState,
    patch: bytes | None = None,
    work_dir: Path | None = None,
) -> TaskRecord:
    """One replication (module docstring). Pure in all but ``seconds``."""
    t0 = time.perf_counter()
    factory = RngFactory(cfg.root_seed).child(f"seed/{index}")
    ds_seed, pl_seed = _u32(factory, "dataset"), _u32(factory, "pipeline")
    if kind == "size":
        if scenario not in cfg.size.scenarios:
            raise ValueError(f"{scenario} is not a size scenario")
        n_obs: int | None = cfg.size.n_obs
    else:
        if scenario != PIPELINE_SCENARIO:
            raise ValueError(f"the pipeline task runs on {PIPELINE_SCENARIO}")
        n_obs = cfg.pipeline_n_obs
    dataset = generate_market(_scenario(scenario).config(n_obs), ds_seed)
    config = _config(cfg, dataset, pl_seed)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        settings = Settings(_env_file=None, data_dir=root / "data", runs_dir=root / "runs")
        snapshot = write_snapshot(
            [synthetic_etf_payload(dataset), synthetic_feature_payload(dataset)],
            settings.data_dir / "snapshots",
        )
        panel = build_pit_panel(snapshot, config)
        if kind == "size":
            members = [(s, f"{SYNTHETIC_PREFIX}{s}") for s in cfg.size.members]
            result = run_portfolio_members(
                panel, members, config.portfolio, RngFactory(config.seed)
            )
            arms, marginals = _evidence(result)
            return TaskRecord(
                kind=kind,
                scenario=scenario,
                index=index,
                dataset_seed=ds_seed,
                pipeline_seed=pl_seed,
                n_obs=result.n_obs,
                members=tuple(m.signal_id for m in result.members),
                verdict=result.verdict,
                reason=None
                if not result.members or result.members[0].reason is None
                else result.members[0].reason.value,
                arms=arms,
                marginals=marginals,
                signals=(),
                score=None,
                seconds=time.perf_counter() - t0,
            )
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
            result = run_portfolio_stage(panel, decay, config.portfolio, ctx.rng)
            record_portfolio_ledger(ledger, result)
            write_portfolio_result(ctx.run_dir, result)
            ctx.mark_completed()
        except BaseException as exc:
            ctx.mark_failed(exc)
            raise
        score = score_run(ctx.run_dir, dataset.ground_truth, snapshot, stages=STAGES, write=False)

    s2 = {s.signal_id: s for s in decay.signals}
    s3 = {m.signal_id: m for m in result.members}
    signals = tuple(
        SignalOutcome(
            signal_id=link.signal_id,
            stage1=link.verdict,
            stage2=s2[link.signal_id].verdict if link.signal_id in s2 else None,
            stage3=s3[link.signal_id].verdict if link.signal_id in s3 else None,
            stage3_reason=None
            if link.signal_id not in s3 or s3[link.signal_id].reason is None
            else str(s3[link.signal_id].reason),
        )
        for link in causal.signals
    )
    arms, marginals = _evidence(result)
    first_reason = result.members[0].reason if result.members else None
    return TaskRecord(
        kind=kind,
        scenario=scenario,
        index=index,
        dataset_seed=ds_seed,
        pipeline_seed=pl_seed,
        n_obs=result.n_obs,
        members=tuple(m.signal_id for m in result.members),
        verdict=result.verdict,
        reason=None if first_reason is None else first_reason.value,
        arms=arms,
        marginals=marginals,
        signals=signals,
        score=score,
        seconds=time.perf_counter() - t0,
    )


def validation_tasks(cfg: PortfolioValidationConfig, n_seeds: int) -> list[Task]:
    tasks: list[Task] = [("size", s, i) for s in cfg.size.scenarios for i in range(n_seeds)]
    tasks += [("pipeline", PIPELINE_SCENARIO, i) for i in range(n_seeds)]
    return tasks


def run_validation(
    cfg: PortfolioValidationConfig,
    *,
    git_state: GitState,
    patch: bytes | None = None,
    n_seeds: int | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
    chunk_size: int = 48,
) -> list[TaskRecord]:
    """Every task on seeds ``0..M-1`` with joblib (loky) workers, in ``validation_tasks``
    order and independent of ``n_jobs``. Resumes from ``checkpoint`` (JSONL)."""
    m = n_seeds if n_seeds is not None else cfg.n_seeds
    tasks = validation_tasks(cfg, m)
    done: dict[Task, TaskRecord] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = TaskRecord.model_validate_json(line)
                done[(rec.kind, rec.scenario, rec.index)] = rec
    todo = [t for t in tasks if t not in done]
    log = get_logger(__name__)
    log.info("portfolio_validation_start", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            recs: list[TaskRecord] = Parallel()(
                delayed(run_task)(k, s, i, cfg, git_state=git_state, patch=patch)
                for k, s, i in chunk
            )
            for rec in recs:
                done[(rec.kind, rec.scenario, rec.index)] = rec
            if checkpoint is not None:
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.writelines(r.model_dump_json() + "\n" for r in recs)
            log.info("portfolio_validation_progress", done=start + len(chunk), total=len(todo))
    return [done[t] for t in tasks]


# ---------------------------------------------------------------------------------------------
# Summary and acceptance
# ---------------------------------------------------------------------------------------------


class ArmSummary(_Frozen):
    arm: Arm
    decides: bool
    n: int
    sharpe: float
    sharpe_net: float
    invested: float
    cash: float
    turnover: float
    beats_equal_weight: RateEstimate | None
    """Share of runs with LW p < lw_alpha (for Ridge-CV: the stage's pass rate)."""


class SignalSummary(_Frozen):
    signal_id: str
    reached_stage3: RateEstimate
    stage3_pass_given_reached: RateEstimate
    passes_all: RateEstimate


class MarginalSummary(_Frozen):
    signal_id: str
    n: int
    mean_marginal_sharpe: float
    ci_excludes_zero: RateEstimate


class ScenarioSummary(_Frozen):
    kind: Kind
    scenario: str
    n_runs: int
    n_ran: int
    """Runs in which the portfolio was built (some member reached Stage 3, enough data)."""
    stage_pass: RateEstimate
    arms: tuple[ArmSummary, ...]
    marginals: tuple[MarginalSummary, ...]
    signals: tuple[SignalSummary, ...]
    member_sets: dict[str, int]
    aggregate: AggregateScore | None
    median_seconds: float


class CriterionResult(_Frozen):
    name: str
    scope: str
    rate: RateEstimate | None
    bound: float | None
    threshold: float
    passed: bool


class PortfolioValidationReport(_Frozen):
    config: PortfolioValidationConfig
    n_seeds: int
    git_sha: str
    git_dirty: bool
    lw_alpha: float
    scenarios: tuple[ScenarioSummary, ...]
    criteria: tuple[CriterionResult, ...]
    accepted: bool


def _arm_summary(arm: Arm, recs: Sequence[TaskRecord], alpha: float, conf: float) -> ArmSummary:
    rs = [a for r in recs if (a := r.arm(arm)) is not None]
    ps = [a.lw_p for a in rs if a.lw_p is not None]
    return ArmSummary(
        arm=arm,
        decides=arm is DECIDING_ARM,
        n=len(rs),
        sharpe=float(np.mean([a.sharpe for a in rs])) if rs else float("nan"),
        sharpe_net=float(np.mean([a.sharpe_net for a in rs])) if rs else float("nan"),
        invested=float(np.mean([a.invested for a in rs])) if rs else float("nan"),
        cash=float(np.mean([a.cash for a in rs])) if rs else float("nan"),
        turnover=float(np.mean([a.turnover for a in rs])) if rs else float("nan"),
        beats_equal_weight=wilson_interval(sum(p < alpha for p in ps), len(ps), conf)
        if ps
        else None,
    )


def _summarize(
    kind: Kind, scenario: str, recs: Sequence[TaskRecord], alpha: float, conf: float
) -> ScenarioSummary:
    ran = [r for r in recs if r.arms]
    marg: dict[str, list[MarginalOutcome]] = {}
    for r in ran:
        for m in r.marginals:
            marg.setdefault(m.signal_id, []).append(m)
    sets: dict[str, int] = {}
    for r in recs:
        key = ",".join(r.members) or "(none)"
        sets[key] = sets.get(key, 0) + 1
    signals = []
    if kind == "pipeline":
        ids = sorted({s.signal_id for r in recs for s in r.signals})
        for sid in ids:
            rows = [next(s for s in r.signals if s.signal_id == sid) for r in recs]
            reached = [o for o in rows if o.stage3 is not None]
            signals.append(
                SignalSummary(
                    signal_id=sid,
                    reached_stage3=wilson_interval(len(reached), len(rows), conf),
                    stage3_pass_given_reached=wilson_interval(
                        sum(o.stage3 is Verdict.PASS for o in reached), len(reached), conf
                    ),
                    passes_all=wilson_interval(
                        sum(o.stage3 is Verdict.PASS for o in rows), len(rows), conf
                    ),
                )
            )
    scores = [r.score for r in recs if r.score is not None]
    return ScenarioSummary(
        kind=kind,
        scenario=scenario,
        n_runs=len(recs),
        n_ran=len(ran),
        stage_pass=wilson_interval(sum(r.verdict is Verdict.PASS for r in ran), len(ran), conf),
        arms=tuple(_arm_summary(a, ran, alpha, conf) for a in ARMS),
        marginals=tuple(
            MarginalSummary(
                signal_id=sid,
                n=len(ms),
                mean_marginal_sharpe=float(np.mean([m.marginal_sharpe for m in ms])),
                ci_excludes_zero=wilson_interval(
                    sum(m.ci_low > 0 or m.ci_high < 0 for m in ms), len(ms), conf
                ),
            )
            for sid, ms in sorted(marg.items())
        ),
        signals=tuple(signals),
        member_sets=dict(sorted(sets.items(), key=lambda kv: -kv[1])),
        aggregate=aggregate_scores(scores, conf) if scores else None,
        median_seconds=statistics.median(r.seconds for r in recs),
    )


def summarize_validation(
    records: Sequence[TaskRecord],
    cfg: PortfolioValidationConfig,
    *,
    n_seeds: int,
    lw_alpha: float,
    git_sha: str,
    git_dirty: bool,
) -> PortfolioValidationReport:
    """Per-scenario summaries and the pre-registered acceptance decision (ADR 0014)."""
    conf = cfg.confidence
    summaries = []
    criteria = []
    for s in cfg.size.scenarios:
        recs = [r for r in records if r.kind == "size" and r.scenario == s]
        summary = _summarize("size", s, recs, lw_alpha, conf)
        summaries.append(summary)
        rate = summary.stage_pass if summary.n_ran == len(recs) and recs else None
        bound = None if rate is None else rate.ci_high
        criteria.append(
            CriterionResult(
                name="V1_size",
                scope=s,
                rate=rate,
                bound=bound,
                threshold=cfg.size.upper_max,
                passed=bound is not None and bound <= cfg.size.upper_max,
            )
        )
    pipe = [r for r in records if r.kind == "pipeline"]
    summary = _summarize("pipeline", PIPELINE_SCENARIO, pipe, lw_alpha, conf)
    summaries.append(summary)
    power = next(
        (s.stage3_pass_given_reached for s in summary.signals if s.signal_id == cfg.power.signal),
        None,
    )
    low = None if power is None else power.ci_low
    criteria.append(
        CriterionResult(
            name="V2_power",
            scope=cfg.power.signal,
            rate=power,
            bound=low,
            threshold=cfg.power.lower_min,
            passed=low is not None and low >= cfg.power.lower_min,
        )
    )
    complete = bool(pipe) and all(r.score is not None and r.score.complete for r in pipe)
    expected = len(validation_tasks(cfg, n_seeds))
    criteria.append(
        CriterionResult(
            name="V3_complete",
            scope="all",
            rate=None,
            bound=None,
            threshold=float(expected),
            passed=complete and len(records) == expected,
        )
    )
    return PortfolioValidationReport(
        config=cfg,
        n_seeds=n_seeds,
        git_sha=git_sha,
        git_dirty=git_dirty,
        lw_alpha=lw_alpha,
        scenarios=tuple(summaries),
        criteria=tuple(criteria),
        accepted=all(c.passed for c in criteria),
    )
