"""Monte Carlo validation of Stage 1 on the planted-signal benchmark (docs/adr/0010).

Method
------
Replication ``i`` of preset ``P`` runs the full production path on a fresh synthetic market:
``generate_market`` -> immutable snapshot -> point-in-time panel -> ``RunContext`` ->
``run_causal_stage`` -> kill ledger -> ``score_run`` restricted to Stage 1. Nothing is read
from the dataset directly, so the validation covers the data path as well as the method.
Replications are independent, and every rate gets a Wilson interval.

Seeds: replication ``i`` takes its dataset seed and its pipeline seed (which seeds the run's
``RngFactory``, and so Stage 1's streams) from ``RngFactory(root_seed).child("seed/i")``.
Every preset uses the same seeds for replication ``i`` (common random numbers), and serial and
parallel runs give identical records.

The acceptance criteria are fixed in the config (``AcceptanceCriteria``) and were committed
before the full run. ``criterion_pass_probability`` gives the probability that each criterion
passes for a given true rate, so the error rates of the rule itself are known before the run.

Limitations: each preset is one DGP, so its rates hold for that DGP only. The pooled kill
precision and recall have optimistic intervals because the signals of one run share a dataset
(``aggregate_scores``). Per-signal rates are exact binomial over independent replications.

Reference: Wilson (1927), JASA 22(158).
"""

from __future__ import annotations

import math
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
from scipy import stats

from invariant.benchmark.config import (
    SyntheticMarketConfig,
    load_benchmark_config,
    load_benchmark_target_timing,
)
from invariant.benchmark.payload import (
    synthetic_etf_payload,
    synthetic_feature_payload,
    synthetic_tickers,
)
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import (
    CausalConfig,
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
    record_causal_ledger,
    run_causal_stage,
    write_causal_result,
)

__all__ = [
    "DEFAULT_VALIDATION_CONFIG",
    "AcceptanceCriteria",
    "CriterionResult",
    "PresetSummary",
    "SeedOutcome",
    "SignalOutcome",
    "SignalSummary",
    "ValidationConfig",
    "ValidationReport",
    "criterion_pass_probability",
    "load_validation_config",
    "pipeline_config",
    "run_benchmark_seed",
    "run_validation",
    "summarize_validation",
]

DEFAULT_VALIDATION_CONFIG: Final = Path("configs/causal_stage_validation.yaml")
BASE_PRESET: Final = "base"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AcceptanceCriteria(_Frozen):
    """Pre-registered: Stage 1 is accepted iff every criterion holds on ``preset``.

    - every replication completes, its ledger chain verifies, and no signal is undecided;
    - each ``null_signals`` pass rate has a Wilson upper bound <= ``null_pass_upper_max``;
    - each ``true_signals`` pass rate has a Wilson lower bound >= ``true_pass_lower_min``.
    """

    preset: str
    null_signals: tuple[str, ...] = Field(min_length=1)
    null_pass_upper_max: float = Field(gt=0.0, lt=1.0)
    true_signals: tuple[str, ...] = Field(min_length=1)
    true_pass_lower_min: float = Field(gt=0.0, lt=1.0)


class ValidationConfig(_Frozen):
    root_seed: StrictInt = Field(ge=0, lt=2**32)
    n_seeds: StrictInt = Field(ge=1)
    n_obs: StrictInt | None = Field(ge=200)
    """``None``: each preset's own ``n_obs``."""
    presets: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(gt=0.0, lt=1.0)
    n_boot: StrictInt | None = Field(ge=99)
    """Overrides ``causal.bootstrap.n_boot`` (``None``: the pipeline config's)."""
    acceptance: AcceptanceCriteria

    @model_validator(mode="after")
    def _check(self) -> ValidationConfig:
        if len(set(self.presets)) != len(self.presets):
            raise ValueError(f"duplicate presets: {self.presets}")
        if self.acceptance.preset not in self.presets:
            raise ValueError(f"acceptance preset {self.acceptance.preset!r} is not run")
        return self


def load_validation_config(path: Path | None = None) -> ValidationConfig:
    p = path if path is not None else DEFAULT_VALIDATION_CONFIG
    p = p if p.is_absolute() else find_project_root() / p
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"validation config not found: {p}") from None
    return ValidationConfig.model_validate(data)


# ---------------------------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------------------------


class SignalOutcome(_Frozen):
    signal_id: str
    verdict: Verdict
    reason: str | None
    family_p: float | None
    q: float | None
    mci_family_p: float | None
    best_asset: str | None
    best_lag: int | None
    r_best: float | None
    r_best_ci_low: float | None
    r_best_ci_high: float | None
    r_lag1: float | None
    """Deciding test's partial correlation at lag 1 on the first asset."""
    expected_t_hybrid: float
    """GroundTruth's value for the first asset."""


class SeedOutcome(_Frozen):
    preset: str
    index: int
    dataset_seed: int
    pipeline_seed: int
    n_obs: int
    n_rows_tested: int | None
    first_asset: str
    n_assets: int
    score: BenchmarkScore
    signals: tuple[SignalOutcome, ...]
    seconds: float
    """Wall time of the replication. Not part of the reproducible result."""


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def _market_config(preset: str, n_obs: int | None) -> SyntheticMarketConfig:
    cfg = load_benchmark_config(None if preset == BASE_PRESET else preset)
    if n_obs is None:
        return cfg
    return SyntheticMarketConfig.model_validate({**cfg.model_dump(), "n_obs": n_obs})


def pipeline_config(
    base: PipelineConfig, dataset: SyntheticDataset, *, seed: int, n_boot: int | None
) -> PipelineConfig:
    """``base`` pointed at ``dataset``: its tickers, dates and target timing, the run seed, and
    the benchmark's candidate signals as Stage 1's signals (``z`` stays a conditioning node)."""
    truth = dataset.ground_truth
    boot = base.causal.bootstrap
    if n_boot is not None:
        boot = boot.model_copy(update={"n_boot": n_boot})
    causal = CausalConfig.model_validate(
        {
            **base.causal.model_dump(),
            "signals": tuple(s.signal_id for s in truth.signals),
            "bootstrap": boot.model_dump(),
        }
    )
    return base.model_copy(
        update={
            "universe": base.universe.model_copy(
                update={"tickers": synthetic_tickers(truth.asset_names)}
            ),
            "dates": base.dates.model_copy(
                update={"start": truth.first_date, "end": truth.last_date}
            ),
            "pit": base.pit.model_copy(update={"targets": load_benchmark_target_timing()}),
            "seed": seed,
            "causal": causal,
        }
    )


def run_benchmark_seed(
    preset: str,
    index: int,
    cfg: ValidationConfig,
    *,
    git_state: GitState,
    patch: bytes | None = None,
    work_dir: Path | None = None,
) -> SeedOutcome:
    """One replication through the production path (module docstring). Pure in all but
    ``seconds``. Snapshot and run directories live in ``work_dir`` (default: a temporary
    directory removed afterwards)."""
    t0 = time.perf_counter()
    factory = RngFactory(cfg.root_seed).child(f"seed/{index}")
    ds_seed, pl_seed = _u32(factory, "dataset"), _u32(factory, "pipeline")
    dataset = generate_market(_market_config(preset, cfg.n_obs), ds_seed)
    truth = dataset.ground_truth
    config = pipeline_config(load_config(), dataset, seed=pl_seed, n_boot=cfg.n_boot)
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
            result = run_causal_stage(panel, config.causal, ctx.rng)
            record_causal_ledger(KillLedger.open(ctx), result)
            write_causal_result(ctx.run_dir, result)
            ctx.mark_completed()
        except BaseException as exc:
            ctx.mark_failed(exc)
            raise
        score = score_run(ctx.run_dir, truth, snapshot, stages=(StageName.CAUSAL,), write=False)

    first = result.return_columns[0]
    outcomes = []
    for link in result.signals:
        ev = link.evidence
        lag1 = None if ev is None else next(t for t in ev.tests if t.asset == first and t.lag == 1)
        outcomes.append(
            SignalOutcome(
                signal_id=link.signal_id,
                verdict=link.verdict,
                reason=None if link.reason is None else link.reason.value,
                family_p=None if ev is None else ev.family_p,
                q=None if ev is None else ev.q,
                mci_family_p=None if ev is None else ev.mci_family_p,
                best_asset=None if ev is None else ev.best_asset,
                best_lag=None if ev is None else ev.best_lag,
                r_best=None if ev is None else ev.r_best_ci.estimate,
                r_best_ci_low=None if ev is None else ev.r_best_ci.low,
                r_best_ci_high=None if ev is None else ev.r_best_ci.high,
                r_lag1=None if lag1 is None else lag1.r,
                expected_t_hybrid=truth.signal(link.signal_id).expected_t_hybrid[0],
            )
        )
    return SeedOutcome(
        preset=preset,
        index=index,
        dataset_seed=ds_seed,
        pipeline_seed=pl_seed,
        n_obs=result.n_obs,
        n_rows_tested=result.n_rows_tested,
        first_asset=first,
        n_assets=len(result.return_columns),
        score=score,
        signals=tuple(outcomes),
        seconds=time.perf_counter() - t0,
    )


def run_validation(
    cfg: ValidationConfig,
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
    log.info("causal_validation_start", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
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
            log.info("causal_validation_progress", done=start + len(chunk), total=len(todo))
    return [done[(p, i)] for p in cfg.presets for i in range(m)]


# ---------------------------------------------------------------------------------------------
# Summary and acceptance
# ---------------------------------------------------------------------------------------------


class SignalSummary(_Frozen):
    signal_id: str
    identifiable: bool
    expected_verdict: Verdict
    pass_rate: RateEstimate
    reasons: dict[str, int]
    """Kill counts by reason code."""
    mci_family_rate: RateEstimate
    """Share of runs with MCI family p < alpha (Bonferroni only, no FDR). Diagnostic."""
    low_power_runs: int
    expected_t_hybrid_mean: float
    realized_t_lag1_mean: float | None
    """Mean of ``r_lag1 * sqrt(n_obs - 1)``, the realized analogue of ``expected_t_hybrid``."""
    realized_t_lag1_se: float | None
    predicted_power: float | None
    """Two-sided power at the Bonferroni threshold ``alpha / (2 K)`` for a normal statistic
    with mean ``expected_t_hybrid``: ignores PC1 selection, the other lags, and FDR."""
    bootstrap_coverage: RateEstimate | None
    """Share of runs whose interval covers the population lag-1 partial correlation, among
    runs whose best link is lag 1 of the first asset. Conditional on selection: a diagnostic."""


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


class ValidationReport(_Frozen):
    config: ValidationConfig
    n_seeds: int
    git_sha: str
    git_dirty: bool
    tigramite_version: str
    alpha: float
    presets: tuple[PresetSummary, ...]
    criteria: tuple[CriterionResult, ...]
    accepted: bool


def _two_sided_power(t: float, alpha: float, k: int) -> float:
    z = float(stats.norm.isf(alpha / (2.0 * k)))
    return float(stats.norm.sf(z - t) + stats.norm.cdf(-z - t))


def _summarize_signal(
    sid: str, records: Sequence[SeedOutcome], alpha: float, k: int, confidence: float
) -> SignalSummary:
    """``k``: tests in a signal's Bonferroni family (``tau_max`` x assets)."""
    rows = [(rec, next(s for s in rec.signals if s.signal_id == sid)) for rec in records]
    scored = [next(r for r in rec.score.signals if r.signal_id == sid) for rec in records]
    m = len(rows)
    reasons: dict[str, int] = {}
    for _, o in rows:
        if o.reason is not None:
            reasons[o.reason] = reasons.get(o.reason, 0) + 1
    t_exp = statistics.fmean(o.expected_t_hybrid for _, o in rows)
    realized = [o.r_lag1 * math.sqrt(rec.n_obs - 1) for rec, o in rows if o.r_lag1 is not None]
    covered = [
        o.r_best_ci_low <= o.expected_t_hybrid / math.sqrt(rec.n_obs - 1) <= o.r_best_ci_high
        for rec, o in rows
        if o.best_asset == rec.first_asset
        and o.best_lag == 1
        and o.r_best_ci_low is not None
        and o.r_best_ci_high is not None
    ]
    return SignalSummary(
        signal_id=sid,
        identifiable=scored[0].identifiable,
        expected_verdict=scored[0].expected_verdict,
        pass_rate=wilson_interval(sum(o.verdict is Verdict.PASS for _, o in rows), m, confidence),
        reasons=dict(sorted(reasons.items())),
        mci_family_rate=wilson_interval(
            sum(o.mci_family_p is not None and o.mci_family_p < alpha for _, o in rows),
            m,
            confidence,
        ),
        low_power_runs=sum(r.low_power for r in scored),
        expected_t_hybrid_mean=t_exp,
        realized_t_lag1_mean=statistics.fmean(realized) if realized else None,
        realized_t_lag1_se=(
            statistics.stdev(realized) / math.sqrt(len(realized)) if len(realized) > 1 else None
        ),
        predicted_power=_two_sided_power(t_exp, alpha, k),
        bootstrap_coverage=(
            wilson_interval(sum(covered), len(covered), confidence) if covered else None
        ),
    )


def _evaluate(acc: AcceptanceCriteria, summary: PresetSummary) -> tuple[CriterionResult, ...]:
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
        rate = by_id[sid].pass_rate
        upper = rate.ci_high if rate.ci_high is not None else 1.0
        out.append(
            CriterionResult(
                name="null_pass_upper",
                signal_id=sid,
                rate=rate,
                bound=upper,
                threshold=acc.null_pass_upper_max,
                passed=upper <= acc.null_pass_upper_max,
            )
        )
    for sid in acc.true_signals:
        rate = by_id[sid].pass_rate
        lower = rate.ci_low if rate.ci_low is not None else 0.0
        out.append(
            CriterionResult(
                name="true_pass_lower",
                signal_id=sid,
                rate=rate,
                bound=lower,
                threshold=acc.true_pass_lower_min,
                passed=lower >= acc.true_pass_lower_min,
            )
        )
    return tuple(out)


def criterion_pass_probability(
    m: int,
    true_rate: float,
    threshold: float,
    side: Literal["upper_max", "lower_min"],
    confidence: float = 0.95,
) -> float:
    """P(criterion passes) when the pass count is Binomial(m, true_rate): the Wilson upper
    bound <= threshold (``upper_max``) or the Wilson lower bound >= threshold (``lower_min``)."""
    ks = np.arange(m + 1)
    if side == "upper_max":
        ok = [(wilson_interval(int(k), m, confidence).ci_high or 1.0) <= threshold for k in ks]
    else:
        ok = [(wilson_interval(int(k), m, confidence).ci_low or 0.0) >= threshold for k in ks]
    pmf = stats.binom.pmf(ks, m, true_rate)
    return float(np.sum(pmf[np.asarray(ok, dtype=bool)]))


def summarize_validation(
    records: Sequence[SeedOutcome],
    cfg: ValidationConfig,
    *,
    alpha: float,
    tau_max: int,
    git_sha: str,
    git_dirty: bool,
    tigramite_version: str,
) -> ValidationReport:
    """Per-preset rates and the pre-registered acceptance decision (``AcceptanceCriteria``)."""
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
                    _summarize_signal(
                        s.signal_id, recs, alpha, tau_max * recs[0].n_assets, cfg.confidence
                    )
                    for s in recs[0].signals
                ),
                median_seconds=statistics.median(r.seconds for r in recs),
            )
        )
    accept_on = next(p for p in presets if p.preset == cfg.acceptance.preset)
    criteria = _evaluate(cfg.acceptance, accept_on)
    return ValidationReport(
        config=cfg,
        n_seeds=presets[0].n_runs,
        git_sha=git_sha,
        git_dirty=git_dirty,
        tigramite_version=tigramite_version,
        alpha=alpha,
        presets=tuple(presets),
        criteria=criteria,
        accepted=all(c.passed for c in criteria),
    )
