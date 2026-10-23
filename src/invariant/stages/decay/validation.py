"""Monte Carlo validation of Stage 2 on the planted-signal benchmark (docs/adr/0012).

Method
------
Replication ``i`` of preset ``P`` runs the production path:
``generate_market`` -> snapshot -> PIT panel -> ``RunContext`` -> Stage 1 -> kill ledger ->
Stage 2 -> kill ledger -> ``score_run`` over (causal, decay). Stage 2 sees only the signals
Stage 1 passed, so its rates are reported **conditional on a Stage 1 pass**. Each such rate is
a binomial over the replications in which the signal passed Stage 1, and those are independent,
so the Wilson intervals are honest. The denominators are random, and the pre-registered pass
probabilities integrate over them (``acceptance_pass_probability``).

Seeds: replication ``i`` takes its dataset and pipeline seeds from
``RngFactory(root_seed).child("seed/i")``, as Stage 1's validation does. Presets share seeds
(common random numbers), and serial and parallel runs are identical.

Reference: Wilson (1927), JASA 22(158).
"""

from __future__ import annotations

import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import yaml
from joblib import Parallel, delayed, parallel_config
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from scipy import stats

from invariant.benchmark.payload import synthetic_etf_payload, synthetic_feature_payload
from invariant.benchmark.synthetic import generate_market
from invariant.config.settings import ConfigError, Settings, find_project_root, load_config
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
    record_causal_ledger,
    run_causal_stage,
    write_causal_result,
)
from invariant.stages.causal.validation import _market_config, pipeline_config
from invariant.stages.decay.stage import (
    record_decay_ledger,
    run_decay_stage,
    write_decay_result,
)

__all__ = [
    "DEFAULT_VALIDATION_CONFIG",
    "DecayAcceptance",
    "DecayValidationConfig",
    "DecayValidationReport",
    "SeedOutcome",
    "acceptance_pass_probability",
    "load_decay_validation_config",
    "run_benchmark_seed",
    "run_validation",
    "summarize_validation",
]

DEFAULT_VALIDATION_CONFIG: Final = Path("configs/decay_stage_validation.yaml")
STAGES: Final = (StageName.CAUSAL, StageName.DECAY)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DecayAcceptance(_Frozen):
    """Pre-registered: Stage 2 is accepted iff every criterion holds on ``preset``.

    - every replication completes, its ledger chain verifies, and no signal is undecided;
    - each ``null_signals`` Stage 2 kill rate, given a Stage 1 pass, has a Wilson upper bound
      <= ``null_kill_upper_max``;
    - each ``decay_signals`` Stage 2 kill rate, given a Stage 1 pass, has a Wilson lower bound
      >= ``decay_kill_lower_min``.
    """

    preset: str
    null_signals: tuple[str, ...] = Field(min_length=1)
    null_kill_upper_max: float = Field(gt=0.0, lt=1.0)
    decay_signals: tuple[str, ...] = Field(min_length=1)
    decay_kill_lower_min: float = Field(gt=0.0, lt=1.0)


class DecayValidationConfig(_Frozen):
    root_seed: StrictInt = Field(ge=0, lt=2**32)
    n_seeds: StrictInt = Field(ge=1)
    n_obs: StrictInt | None = Field(ge=200)
    presets: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(gt=0.0, lt=1.0)
    n_boot: StrictInt | None = Field(ge=99)
    """Overrides Stage 1's ``causal.bootstrap.n_boot`` (descriptive interval only)."""
    min_obs: StrictInt | None = Field(default=None, ge=100)
    """Overrides ``decay.min_obs`` (tests only; ``None`` in the pre-registered config)."""
    acceptance: DecayAcceptance

    @model_validator(mode="after")
    def _check(self) -> DecayValidationConfig:
        if len(set(self.presets)) != len(self.presets):
            raise ValueError(f"duplicate presets: {self.presets}")
        if self.acceptance.preset not in self.presets:
            raise ValueError(f"acceptance preset {self.acceptance.preset!r} is not run")
        return self


def load_decay_validation_config(path: Path | None = None) -> DecayValidationConfig:
    p = path if path is not None else DEFAULT_VALIDATION_CONFIG
    p = p if p.is_absolute() else find_project_root() / p
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"validation config not found: {p}") from None
    return DecayValidationConfig.model_validate(data)


# ---------------------------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------------------------


class SignalOutcome(_Frozen):
    signal_id: str
    stage1_verdict: Verdict
    stage2_verdict: Verdict | None
    """``None`` when Stage 1 killed the signal (it never reached Stage 2)."""
    stage2_reason: str | None
    decline_p: float | None
    break_fraction: float | None
    trend_p: float | None


class SeedOutcome(_Frozen):
    preset: str
    index: int
    dataset_seed: int
    pipeline_seed: int
    n_obs: int
    score: BenchmarkScore
    signals: tuple[SignalOutcome, ...]
    seconds: float
    """Wall time. Not part of the reproducible result."""


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def run_benchmark_seed(
    preset: str,
    index: int,
    cfg: DecayValidationConfig,
    *,
    git_state: GitState,
    patch: bytes | None = None,
    work_dir: Path | None = None,
) -> SeedOutcome:
    """One replication through the production path (module docstring). Pure in all but
    ``seconds``."""
    t0 = time.perf_counter()
    factory = RngFactory(cfg.root_seed).child(f"seed/{index}")
    ds_seed, pl_seed = _u32(factory, "dataset"), _u32(factory, "pipeline")
    dataset = generate_market(_market_config(preset, cfg.n_obs), ds_seed)
    truth = dataset.ground_truth
    config = pipeline_config(load_config(), dataset, seed=pl_seed, n_boot=cfg.n_boot)
    if cfg.min_obs is not None:
        config = config.model_copy(
            update={"decay": config.decay.model_copy(update={"min_obs": cfg.min_obs})}
        )
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        settings = Settings(_env_file=None, data_dir=root / "data", runs_dir=root / "runs")
        snapshot = write_snapshot(
            [synthetic_etf_payload(dataset), synthetic_feature_payload(dataset)],
            settings.data_dir / "snapshots",
        )
        ctx = RunContext.create(
            config, snapshot.snapshot_id, settings=settings, git_state=git_state, patch=patch
        )
        try:
            panel = build_pit_panel(snapshot, config)
            causal = run_causal_stage(panel, config.causal, ctx.rng)
            ledger = KillLedger.open(ctx)
            record_causal_ledger(ledger, causal)
            write_causal_result(ctx.run_dir, causal)
            decay = run_decay_stage(panel, causal, config.decay, ctx.rng)
            record_decay_ledger(ledger, decay)
            write_decay_result(ctx.run_dir, decay)
            ctx.mark_completed()
        except BaseException as exc:
            ctx.mark_failed(exc)
            raise
        score = score_run(ctx.run_dir, truth, snapshot, stages=STAGES, write=False)

    by_id = {s.signal_id: s for s in decay.signals}
    outcomes = []
    for link in causal.signals:
        d = by_id.get(link.signal_id)
        t = None if d is None else d.test
        outcomes.append(
            SignalOutcome(
                signal_id=link.signal_id,
                stage1_verdict=link.verdict,
                stage2_verdict=None if d is None else d.verdict,
                stage2_reason=None if d is None or d.reason is None else d.reason.value,
                decline_p=None if t is None else t.p,
                break_fraction=None if t is None else t.break_fraction,
                trend_p=None if t is None else t.trend_p,
            )
        )
    return SeedOutcome(
        preset=preset,
        index=index,
        dataset_seed=ds_seed,
        pipeline_seed=pl_seed,
        n_obs=causal.n_obs,
        score=score,
        signals=tuple(outcomes),
        seconds=time.perf_counter() - t0,
    )


def run_validation(
    cfg: DecayValidationConfig,
    *,
    git_state: GitState,
    patch: bytes | None = None,
    n_seeds: int | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
    chunk_size: int = 48,
) -> list[SeedOutcome]:
    """Every preset on seeds ``0..M-1`` with joblib (loky) workers; ordered by
    ``(preset, index)`` and independent of ``n_jobs``. Resumes from ``checkpoint`` (JSONL)."""
    m = n_seeds if n_seeds is not None else cfg.n_seeds
    done: dict[tuple[str, int], SeedOutcome] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = SeedOutcome.model_validate_json(line)
                done[(rec.preset, rec.index)] = rec
    todo = [(p, i) for p in cfg.presets for i in range(m) if (p, i) not in done]
    log = get_logger(__name__)
    log.info("decay_validation_start", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            recs: list[SeedOutcome] = Parallel()(
                delayed(run_benchmark_seed)(p, i, cfg, git_state=git_state, patch=patch)
                for p, i in chunk
            )
            for rec in recs:
                done[(rec.preset, rec.index)] = rec
            if checkpoint is not None:
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.writelines(r.model_dump_json() + "\n" for r in recs)
            log.info("decay_validation_progress", done=start + len(chunk), total=len(todo))
    return [done[(p, i)] for p in cfg.presets for i in range(m)]


# ---------------------------------------------------------------------------------------------
# Summary and acceptance
# ---------------------------------------------------------------------------------------------


class SignalSummary(_Frozen):
    signal_id: str
    stage1_pass_rate: RateEstimate
    stage2_kill_given_stage1_pass: RateEstimate
    """Share of Stage 1 passes that Stage 2 killed, for any reason."""
    reasons: dict[str, int]
    trend_kill_given_stage1_pass: RateEstimate
    """Diagnostic: the trend test at the same alpha (never decides)."""
    pipeline_pass_rate: RateEstimate
    """Passed both stages."""


class PresetSummary(_Frozen):
    preset: str
    n_runs: int
    n_incomplete: int
    aggregate: AggregateScore
    signals: tuple[SignalSummary, ...]
    median_seconds: float


class CriterionResult(_Frozen):
    name: str
    signal_id: str | None
    rate: RateEstimate | None
    bound: float | None
    threshold: float | None
    passed: bool


class DecayValidationReport(_Frozen):
    config: DecayValidationConfig
    n_seeds: int
    git_sha: str
    git_dirty: bool
    alpha: float
    presets: tuple[PresetSummary, ...]
    criteria: tuple[CriterionResult, ...]
    accepted: bool


def _summarize_signal(
    sid: str, records: Sequence[SeedOutcome], alpha: float, confidence: float
) -> SignalSummary:
    rows = [next(s for s in r.signals if s.signal_id == sid) for r in records]
    reached = [o for o in rows if o.stage1_verdict is Verdict.PASS]
    reasons: dict[str, int] = {}
    for o in reached:
        if o.stage2_reason is not None:
            reasons[o.stage2_reason] = reasons.get(o.stage2_reason, 0) + 1
    return SignalSummary(
        signal_id=sid,
        stage1_pass_rate=wilson_interval(len(reached), len(rows), confidence),
        stage2_kill_given_stage1_pass=wilson_interval(
            sum(o.stage2_verdict is Verdict.KILL for o in reached), len(reached), confidence
        ),
        reasons=dict(sorted(reasons.items())),
        trend_kill_given_stage1_pass=wilson_interval(
            sum(o.trend_p is not None and o.trend_p < alpha for o in reached),
            len(reached),
            confidence,
        ),
        pipeline_pass_rate=wilson_interval(
            sum(o.stage2_verdict is Verdict.PASS for o in rows), len(rows), confidence
        ),
    )


def _evaluate(acc: DecayAcceptance, summary: PresetSummary) -> tuple[CriterionResult, ...]:
    by_id = {s.signal_id: s for s in summary.signals}
    out = [
        CriterionResult(
            name="complete",
            signal_id=None,
            rate=None,
            bound=float(summary.n_incomplete),
            threshold=0.0,
            passed=summary.n_incomplete == 0 and summary.aggregate.n_incomplete_runs == 0,
        )
    ]
    for sid in acc.null_signals:
        rate = by_id[sid].stage2_kill_given_stage1_pass
        upper = rate.ci_high if rate.ci_high is not None else 1.0
        out.append(
            CriterionResult(
                name="null_kill_upper",
                signal_id=sid,
                rate=rate,
                bound=upper,
                threshold=acc.null_kill_upper_max,
                passed=upper <= acc.null_kill_upper_max,
            )
        )
    for sid in acc.decay_signals:
        rate = by_id[sid].stage2_kill_given_stage1_pass
        lower = rate.ci_low if rate.ci_low is not None else 0.0
        out.append(
            CriterionResult(
                name="decay_kill_lower",
                signal_id=sid,
                rate=rate,
                bound=lower,
                threshold=acc.decay_kill_lower_min,
                passed=lower >= acc.decay_kill_lower_min,
            )
        )
    return tuple(out)


def acceptance_pass_probability(
    m: int,
    reach_rate: float,
    kill_rate: float,
    threshold: float,
    side: str,
    confidence: float = 0.95,
) -> float:
    """P(criterion passes) when the denominator is Binomial(m, reach_rate) and the kill count
    given it is Binomial(n, kill_rate). ``side``: ``upper_max`` (Wilson upper <= threshold) or
    ``lower_min`` (Wilson lower >= threshold). Exact, by summing over n."""
    if side not in ("upper_max", "lower_min"):
        raise ValueError(f"unknown side {side!r}")
    total = 0.0
    p_n = stats.binom.pmf(np.arange(m + 1), m, reach_rate)
    for n in range(1, m + 1):
        if p_n[n] < 1e-12:
            continue
        ks = np.arange(n + 1)
        if side == "upper_max":
            ok = [(wilson_interval(int(k), n, confidence).ci_high or 1.0) <= threshold for k in ks]
        else:
            ok = [(wilson_interval(int(k), n, confidence).ci_low or 0.0) >= threshold for k in ks]
        pmf = stats.binom.pmf(ks, n, kill_rate)
        total += float(p_n[n]) * float(np.sum(pmf[np.asarray(ok, dtype=bool)]))
    return total


def summarize_validation(
    records: Sequence[SeedOutcome],
    cfg: DecayValidationConfig,
    *,
    alpha: float,
    git_sha: str,
    git_dirty: bool,
) -> DecayValidationReport:
    """Per-preset rates and the pre-registered acceptance decision (``DecayAcceptance``)."""
    presets = []
    for preset in cfg.presets:
        recs = [r for r in records if r.preset == preset]
        if not recs:
            raise ValueError(f"no records for preset {preset!r}")
        presets.append(
            PresetSummary(
                preset=preset,
                n_runs=len(recs),
                n_incomplete=sum(not r.score.complete for r in recs),
                aggregate=aggregate_scores([r.score for r in recs], cfg.confidence),
                signals=tuple(
                    _summarize_signal(s.signal_id, recs, alpha, cfg.confidence)
                    for s in recs[0].signals
                ),
                median_seconds=statistics.median(r.seconds for r in recs),
            )
        )
    accept_on = next(p for p in presets if p.preset == cfg.acceptance.preset)
    criteria = _evaluate(cfg.acceptance, accept_on)
    return DecayValidationReport(
        config=cfg,
        n_seeds=presets[0].n_runs,
        git_sha=git_sha,
        git_dirty=git_dirty,
        alpha=alpha,
        presets=tuple(presets),
        criteria=criteria,
        accepted=all(c.passed for c in criteria),
    )
