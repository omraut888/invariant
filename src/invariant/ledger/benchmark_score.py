"""Grade a run's kill ledger against the planted-signal benchmark's ``GroundTruth``.

Method
------
Each signal's fate over the evaluated stages (a prefix of ``StageName`` order) is compared with
its expected outcome. If the expected kill stage is not among the evaluated stages, the
expected outcome for that run is PASS. So a Phase 1 run that evaluates only ``causal`` is graded
on what ``causal`` should decide. Identifiable signals fall into one of four cells:

=================  ===================  ===================
                   observed KILL        observed PASS
=================  ===================  ===================
expected KILL      correct_kill         missed_kill
expected PASS      wrong_kill           correct_pass
=================  ===================  ===================

Every observed kill of an expected kill also records ``stage_match``: whether it happened at the
expected stage. Signals that ``GroundTruth`` marks ``identifiable=False`` are scored per signal
but kept out of the matrix, because no method can recover their status from the observed
variables (ADR 0003). A signal with no decision, or decisions that stop before the last
evaluated stage without a kill, is ``undecided``: never a pass. A score with any undecided
signal has ``complete=False``. Each row carries ``low_power`` (a true edge whose expected
t-stat under Stage 1's ``hybrid`` test, ``SignalTruth.expected_t_hybrid``, is below
``DETECTABILITY_T``), so a missed kill or wrong kill caused by lack of power can be told apart
from a wrong method.

``score_run`` refuses to grade a run that is not a run on this ``GroundTruth``'s dataset: the
run's snapshot must be the given synthetic snapshot whose ``dataset_sha256`` matches, and its
target timing must equal ``GroundTruth.target_timing`` (ADR 0006).

Aggregation
-----------
One run grades four signals: those are counts, not estimates. ``aggregate_scores`` pools runs
over independent seeds and reports each rate with a Wilson score interval (Wilson 1927, JASA
22(158); recommended over the Wald interval by Brown, Cai & DasGupta 2001, Statistical Science
16(2)). Per-signal rates are proportions over independent seeds, so the interval's i.i.d.
assumption holds. Pooled rates (precision, recall) mix signals from the same run, which share
one dataset and are not independent. Their intervals are therefore optimistic, and the field
docs say so.

See docs/adr/0007-kill-ledger-and-benchmark-scoring.md.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict
from scipy import stats

from invariant.benchmark.ground_truth import GroundTruth, SignalTruth
from invariant.benchmark.payload import SYNTHETIC_PROVIDER
from invariant.benchmark.synthetic import DETECTABILITY_T
from invariant.config.settings import TargetConfig
from invariant.core.run_hash import MANIFEST_NAME, RunManifest
from invariant.core.types import StageName, Verdict
from invariant.data.snapshot import Snapshot
from invariant.data.sources.base import SourceName
from invariant.ledger.kill_ledger import (
    STAGE_ORDER,
    Fate,
    FateStatus,
    KillLedger,
    check_stage_prefix,
)

__all__ = [
    "BENCHMARK_SCORE_NAME",
    "AggregateScore",
    "BenchmarkMismatchError",
    "BenchmarkScore",
    "ConfusionMatrix",
    "Outcome",
    "RateEstimate",
    "SignalScore",
    "aggregate_scores",
    "score_fates",
    "score_ledger",
    "score_run",
    "wilson_interval",
]

BENCHMARK_SCORE_NAME: Final = "benchmark_score.json"


class BenchmarkMismatchError(ValueError):
    """The run being scored was not produced from this ``GroundTruth``'s dataset."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Outcome(StrEnum):
    CORRECT_KILL = "correct_kill"
    WRONG_KILL = "wrong_kill"
    CORRECT_PASS = "correct_pass"
    MISSED_KILL = "missed_kill"
    UNDECIDED = "undecided"


class SignalScore(_Frozen):
    """One signal's expected and observed outcome over the evaluated stages."""

    signal_id: str
    identifiable: bool
    low_power: bool
    expected_verdict: Verdict
    expected_stage: StageName | None
    observed_verdict: Verdict | None
    observed_stage: StageName | None
    outcome: Outcome
    stage_match: bool | None
    """For a correct kill, whether it happened at the expected stage; otherwise ``None``."""


class ConfusionMatrix(_Frozen):
    """Counts over identifiable, decided signals."""

    correct_kill: int = 0
    wrong_kill: int = 0
    correct_pass: int = 0
    missed_kill: int = 0
    kills_at_expected_stage: int = 0
    """Correct kills whose stage matched; ``correct_kill - this`` were killed at another stage."""

    @property
    def total(self) -> int:
        return self.correct_kill + self.wrong_kill + self.correct_pass + self.missed_kill


class BenchmarkScore(_Frozen):
    run_id: str | None
    dataset_sha256: str
    stages: tuple[StageName, ...]
    matrix: ConfusionMatrix
    signals: tuple[SignalScore, ...]
    undecided: tuple[str, ...]
    unidentifiable: tuple[str, ...]
    complete: bool
    contains_placeholders: bool


def _expected(
    truth: SignalTruth, stages: tuple[StageName, ...]
) -> tuple[Verdict, StageName | None]:
    stage = truth.expected_kill_stage
    if truth.expected_verdict is Verdict.KILL and stage in stages:
        return Verdict.KILL, stage
    return Verdict.PASS, None


def _low_power(truth: SignalTruth) -> bool:
    # Keyed to the test Stage 1 decides with (GroundTruth v3), not the predictive regression.
    return truth.has_causal_edge and min(truth.expected_t_hybrid) < DETECTABILITY_T


def score_fates(
    fates: Mapping[str, Fate],
    truth: GroundTruth,
    *,
    stages: Sequence[StageName] = STAGE_ORDER,
    run_id: str | None = None,
    contains_placeholders: bool = False,
) -> BenchmarkScore:
    """Score observed ``fates`` (from ``KillLedger.fates(stages)``) against ``truth``."""
    evaluated = check_stage_prefix(stages)
    known = {s.signal_id for s in truth.signals}
    unknown = sorted(set(fates) - known)
    if unknown:
        raise BenchmarkMismatchError(f"ledger has signals not in GroundTruth: {unknown}")

    rows: list[SignalScore] = []
    counts = dict.fromkeys(Outcome, 0)
    stage_hits = 0
    for st in truth.signals:
        expected_verdict, expected_stage = _expected(st, evaluated)
        fate = fates.get(st.signal_id)
        observed_verdict: Verdict | None
        if fate is None or fate.status is FateStatus.INCOMPLETE:
            observed_verdict, observed_stage, outcome = None, None, Outcome.UNDECIDED
        elif fate.status is FateStatus.KILLED:
            observed_verdict, observed_stage = Verdict.KILL, fate.stage
            outcome = (
                Outcome.CORRECT_KILL if expected_verdict is Verdict.KILL else Outcome.WRONG_KILL
            )
        else:
            observed_verdict, observed_stage = Verdict.PASS, None
            outcome = (
                Outcome.MISSED_KILL if expected_verdict is Verdict.KILL else Outcome.CORRECT_PASS
            )
        stage_match = observed_stage == expected_stage if outcome is Outcome.CORRECT_KILL else None
        rows.append(
            SignalScore(
                signal_id=st.signal_id,
                identifiable=st.identifiable,
                low_power=_low_power(st),
                expected_verdict=expected_verdict,
                expected_stage=expected_stage,
                observed_verdict=observed_verdict,
                observed_stage=observed_stage,
                outcome=outcome,
                stage_match=stage_match,
            )
        )
        if st.identifiable and outcome is not Outcome.UNDECIDED:
            counts[outcome] += 1
            stage_hits += bool(stage_match)

    undecided = tuple(r.signal_id for r in rows if r.outcome is Outcome.UNDECIDED)
    return BenchmarkScore(
        run_id=run_id,
        dataset_sha256=truth.dataset_sha256,
        stages=evaluated,
        matrix=ConfusionMatrix(
            correct_kill=counts[Outcome.CORRECT_KILL],
            wrong_kill=counts[Outcome.WRONG_KILL],
            correct_pass=counts[Outcome.CORRECT_PASS],
            missed_kill=counts[Outcome.MISSED_KILL],
            kills_at_expected_stage=stage_hits,
        ),
        signals=tuple(rows),
        undecided=undecided,
        unidentifiable=tuple(r.signal_id for r in rows if not r.identifiable),
        complete=not undecided,
        contains_placeholders=contains_placeholders,
    )


def score_ledger(
    ledger: KillLedger,
    truth: GroundTruth,
    *,
    stages: Sequence[StageName] = STAGE_ORDER,
    attempt: int | None = None,
) -> BenchmarkScore:
    """Score one attempt (default: the latest) of ``ledger`` against ``truth``."""
    return score_fates(
        ledger.fates(stages, attempt),
        truth,
        stages=stages,
        run_id=ledger.run_id,
        contains_placeholders=ledger.contains_placeholders(attempt),
    )


def _run_target_timing(manifest: RunManifest) -> TargetConfig:
    pit = manifest.inputs.config.get("pit")
    targets = pit.get("targets") if isinstance(pit, dict) else None
    if not isinstance(targets, dict):
        raise BenchmarkMismatchError(f"run {manifest.run_id} config has no pit.targets")
    return TargetConfig.model_validate(targets)


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def score_run(
    run_dir: Path,
    truth: GroundTruth,
    snapshot: Snapshot,
    *,
    stages: Sequence[StageName] = STAGE_ORDER,
    write: bool = True,
) -> BenchmarkScore:
    """Check that ``run_dir`` is a run on ``truth``'s dataset, score it, write the score.

    Raises ``BenchmarkMismatchError`` if the run used another snapshot, the snapshot is not the
    synthetic dataset ``truth`` describes, or the run's target timing differs from
    ``truth.target_timing``. Writes ``<run_dir>/benchmark_score.json`` unless ``write=False``.
    """
    manifest = RunManifest.model_validate_json((run_dir / MANIFEST_NAME).read_bytes())
    if manifest.inputs.snapshot_id != snapshot.snapshot_id:
        raise BenchmarkMismatchError(
            f"run {manifest.run_id} used snapshot {manifest.inputs.snapshot_id}, "
            f"not {snapshot.snapshot_id}"
        )
    params = snapshot.manifest.source(SourceName.ETF).params
    if params.get("provider") != SYNTHETIC_PROVIDER:
        raise BenchmarkMismatchError(f"snapshot {snapshot.snapshot_id} is not synthetic data")
    if params.get("dataset_sha256") != truth.dataset_sha256:
        raise BenchmarkMismatchError(
            f"snapshot {snapshot.snapshot_id} holds dataset {params.get('dataset_sha256')}, "
            f"GroundTruth describes {truth.dataset_sha256}"
        )
    if any(e.source is SourceName.SYNTHETIC for e in snapshot.manifest.sources):
        features_sha = snapshot.manifest.source(SourceName.SYNTHETIC).params.get("dataset_sha256")
        if features_sha != truth.dataset_sha256:
            raise BenchmarkMismatchError(
                f"snapshot {snapshot.snapshot_id} holds features of dataset {features_sha}, "
                f"GroundTruth describes {truth.dataset_sha256}"
            )
    timing = _run_target_timing(manifest)
    if timing != truth.target_timing:
        raise BenchmarkMismatchError(
            f"run {manifest.run_id} built targets with {timing.model_dump()}, GroundTruth "
            f"expects {truth.target_timing.model_dump()}"
        )
    score = score_ledger(KillLedger.read(run_dir), truth, stages=stages)
    if write:
        text = json.dumps(score.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        _atomic_write_text(run_dir / BENCHMARK_SCORE_NAME, text)
    return score


# ---------------------------------------------------------------------------------------------
# Aggregation over seeds
# ---------------------------------------------------------------------------------------------


class RateEstimate(_Frozen):
    """``successes / trials`` with a Wilson score interval; ``None`` fields when ``trials == 0``."""

    successes: int
    trials: int
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> RateEstimate:
    """Wilson score interval for a binomial proportion (see module docstring)."""
    if not 0 <= successes <= trials:
        raise ValueError(f"need 0 <= successes <= trials, got {successes}/{trials}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if trials == 0:
        return RateEstimate(
            successes=0, trials=0, estimate=None, ci_low=None, ci_high=None, confidence=confidence
        )
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    n, p = float(trials), successes / trials
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return RateEstimate(
        successes=successes,
        trials=trials,
        estimate=p,
        ci_low=max(0.0, center - half),
        ci_high=min(1.0, center + half),
        confidence=confidence,
    )


class AggregateScore(_Frozen):
    n_runs: int
    n_incomplete_runs: int
    stages: tuple[StageName, ...]
    per_signal_correct: dict[str, RateEstimate]
    """Share of runs where the signal's outcome was correct (correct kill or correct pass).
    Independent across seeds, so the interval is valid. Undecided runs count as incorrect."""
    per_signal_stage_match: dict[str, RateEstimate]
    """Among runs with a correct kill of the signal, the share at the expected stage."""
    kill_precision: RateEstimate
    """Pooled correct_kill / (correct_kill + wrong_kill). Optimistic CI (signals in one run
    share a dataset)."""
    kill_recall: RateEstimate
    """Pooled correct_kill / (correct_kill + missed_kill). Optimistic CI, as above."""


def aggregate_scores(scores: Sequence[BenchmarkScore], confidence: float = 0.95) -> AggregateScore:
    """Pool benchmark scores from runs over independent seeds (see module docstring)."""
    if not scores:
        raise ValueError("need at least one score")
    stages = scores[0].stages
    if any(s.stages != stages for s in scores):
        raise ValueError("all scores must evaluate the same stages")
    signal_ids = sorted({r.signal_id for s in scores for r in s.signals if r.identifiable})
    correct = dict.fromkeys(signal_ids, 0)
    seen = dict.fromkeys(signal_ids, 0)
    hits = dict.fromkeys(signal_ids, 0)
    kills = dict.fromkeys(signal_ids, 0)
    for s in scores:
        for r in s.signals:
            if not r.identifiable:
                continue
            seen[r.signal_id] += 1
            correct[r.signal_id] += r.outcome in (Outcome.CORRECT_KILL, Outcome.CORRECT_PASS)
            if r.outcome is Outcome.CORRECT_KILL:
                kills[r.signal_id] += 1
                hits[r.signal_id] += bool(r.stage_match)
    ck = sum(s.matrix.correct_kill for s in scores)
    wk = sum(s.matrix.wrong_kill for s in scores)
    mk = sum(s.matrix.missed_kill for s in scores)
    return AggregateScore(
        n_runs=len(scores),
        n_incomplete_runs=sum(not s.complete for s in scores),
        stages=stages,
        per_signal_correct={
            sid: wilson_interval(correct[sid], seen[sid], confidence) for sid in signal_ids
        },
        per_signal_stage_match={
            sid: wilson_interval(hits[sid], kills[sid], confidence) for sid in signal_ids
        },
        kill_precision=wilson_interval(ck, ck + wk, confidence),
        kill_recall=wilson_interval(ck, ck + mk, confidence),
    )
