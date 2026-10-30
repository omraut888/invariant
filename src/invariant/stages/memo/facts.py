"""The memo's fact sheet: every number, verdict and reason a memo may state (docs/adr/0017).

Method
------
A fact sheet is built deterministically from one run directory and nothing else:
- the kill ledger, read through ``KillLedger.read`` (which verifies the hash chain);
- the stage results ``causal_result.json``, ``decay_result.json`` and ``portfolio_result.json``;
- the run manifest;
- benchmark validation rates from the archived reports in ``docs/results`` (``reference.py``).

The ledger is the decision record, so facts are read from its metrics. Each stage's verdict and
reason are cross-checked against that stage's result file, and the portfolio metrics against
``portfolio_result.json``. A disagreement raises ``MemoInputError``: two records of one run that
differ mean the memo cannot know which is true.

**GroundTruth is never read.** A memo states what the pipeline decided, not what the benchmark
planted (tests/leakage/test_memo_inputs.py).

Every fact has a stable id, a kind (which fixes its rendering, ``format_value``), a label (what
the number is, with no digits), the signal or object it is about, and ``sources``: where each of
its numbers came from (``ledger:<signal>:<stage>:<metric>``, ``result:<file>:<json pointer>``,
``manifest:<dotted path>``, ``reference:<report>:<json pointer>``, ``const:<name>``). The
verifier resolves sources by its own code path, independently of this module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from invariant.core.run_hash import MANIFEST_NAME
from invariant.core.types import KillReason, StageName
from invariant.ledger.kill_ledger import FateStatus, KillLedger, LedgerEntry, MetricValue
from invariant.stages.causal.stage import CAUSAL_RESULT_NAME
from invariant.stages.decay.stage import DECAY_RESULT_NAME
from invariant.stages.dro.walkforward import Arm
from invariant.stages.memo.reference import load_reference_facts
from invariant.stages.portfolio.stage import PORTFOLIO_RESULT_NAME

__all__ = [
    "ARM_LABELS",
    "MEMO_STAGES",
    "DataKind",
    "Fact",
    "FactKind",
    "FactSheet",
    "MemoInputError",
    "Recommendation",
    "SignalFate",
    "build_fact_sheet",
    "format_value",
]

MEMO_STAGES: Final = (StageName.CAUSAL, StageName.DECAY, StageName.PORTFOLIO)
"""The kill stages (Stage 4 is not one, ADR 0016)."""

ARM_LABELS: Final[Mapping[Arm, str]] = {
    Arm.EQUAL_WEIGHT: "equal weight",
    Arm.MARKOWITZ: "Markowitz",
    Arm.DRO_FC: "Wasserstein DRO (fixed radius)",
    Arm.RIDGE_CV: "Ridge-CV",
    Arm.DRO_CV: "Wasserstein DRO (CV radius)",
}
DSR_TRIALS: Final = 4
"""Trials behind the deflated Sharpe ratio: the four non-1/N arms (ADR 0014)."""


class MemoInputError(ValueError):
    """The run directory cannot support a memo (incomplete, inconsistent or missing)."""


class FactKind(StrEnum):
    P_VALUE = "p_value"
    SHARPE = "sharpe"
    PROBABILITY = "probability"
    CORRELATION = "correlation"
    COEFFICIENT = "coefficient"
    FRACTION = "fraction"
    RATE = "rate"
    COUNT = "count"
    LEVEL = "level"
    DATE = "date"
    TEXT = "text"


def format_value(kind: FactKind, value: float | int | str) -> str:
    """The canonical rendering of one value of ``kind``. Every number in a memo is rendered here."""
    if kind in (FactKind.TEXT, FactKind.DATE):
        return str(value)
    if kind is FactKind.COUNT:
        return str(int(value))
    v = float(value)
    if not math.isfinite(v):
        raise MemoInputError(f"non-finite {kind} value {v}")
    match kind:
        case FactKind.P_VALUE:
            return "< 0.001" if v < 0.001 else f"{v:.3f}"
        case FactKind.SHARPE:
            return f"{v:.2f}"
        case FactKind.PROBABILITY | FactKind.RATE | FactKind.CORRELATION:
            return f"{v:.3f}"
        case FactKind.COEFFICIENT:
            return f"{v:.2e}"
        case FactKind.FRACTION:
            return f"{v:.2f}"
        case FactKind.LEVEL:
            return f"{v:g}"
    raise AssertionError(kind)  # pragma: no cover


class Fact(BaseModel):
    """One statable fact. ``ci`` is an interval (``ci_confidence`` set) or a range (it is None)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    kind: FactKind
    subject: str | None
    label: str
    value: float | int | str
    ci: tuple[float, float] | None = None
    ci_confidence: float | None = None
    sources: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Fact:
        if any(ch.isdigit() for ch in self.label):
            raise ValueError(f"fact labels carry no digits: {self.label!r}")
        n = 1 + (2 if self.ci is not None else 0) + (1 if self.ci_confidence is not None else 0)
        if len(self.sources) != n:
            raise ValueError(f"{self.id}: {n} values need {n} sources, got {len(self.sources)}")
        if self.ci_confidence is not None and self.ci is None:
            raise ValueError(f"{self.id}: a confidence needs an interval")
        return self

    def value_text(self) -> str:
        """The value and interval without the label: ``0.072 [0.047, 0.098], 95% interval``."""
        out = format_value(self.kind, self.value)
        if self.ci is not None:
            lo, hi = (format_value(self.kind, x) for x in self.ci)
            out += f" [{lo}, {hi}]"
            out += (
                ", range" if self.ci_confidence is None else f", {self.ci_confidence:.0%} interval"
            )
        return out

    def full_text(self) -> str:
        """The self-describing atom: subject, label and value (docs/adr/0017, "atoms")."""
        head = f"{self.subject}'s {self.label}" if self.subject else self.label
        value = self.value_text()
        return f"{head} {value}" if value.startswith("<") else f"{head} = {value}"


class DataKind(StrEnum):
    SYNTHETIC = "synthetic"
    REAL = "real"


class Recommendation(StrEnum):
    """The memo's conclusion, a rule on the ledger's fates (docs/adr/0017); never the LLM's."""

    PAPER_TRADING_CANDIDATE = "candidate for paper trading"
    AGAINST = "against trading"
    NOT_DECIDED = "not decided"


class SignalFate(BaseModel):
    """One signal's outcome over the kill stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str
    status: FateStatus
    stage: StageName
    reason: KillReason | None
    stages_decided: tuple[StageName, ...]


class FactSheet(BaseModel):
    """Everything a memo may state about one run (module docstring)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    data_kind: DataKind
    fates: tuple[SignalFate, ...]
    members: tuple[str, ...]
    """Stage 3's members (empty when no signal reached Stage 3)."""
    portfolio_ran: bool
    """Stage 3 built a portfolio (its evidence exists)."""
    facts: tuple[Fact, ...]
    recommendation: Recommendation

    def fact(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(fact_id)

    def has(self, fact_id: str) -> bool:
        return any(f.id == fact_id for f in self.facts)

    def fate(self, signal_id: str) -> SignalFate:
        for f in self.fates:
            if f.signal_id == signal_id:
                return f
        raise KeyError(signal_id)

    def signal_ids(self) -> tuple[str, ...]:
        return tuple(f.signal_id for f in self.fates)


def recommend(fates: tuple[SignalFate, ...]) -> Recommendation:
    """Candidate if any signal passed every kill stage. Otherwise not decided if any signal was
    killed without a test decision (a ``general.*`` reason: it could have survived), and against
    if every signal was killed by a stage's test."""
    if any(f.status is FateStatus.PASSED for f in fates):
        return Recommendation.PAPER_TRADING_CANDIDATE
    if any(f.reason is not None and f.reason.stage is None for f in fates):
        return Recommendation.NOT_DECIDED
    return Recommendation.AGAINST


# --- building ---------------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise MemoInputError(f"missing {path.name} in {path.parent}")
    out = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(out, dict):
        raise MemoInputError(f"{path.name} is not a JSON object")
    return out


class _Builder:
    def __init__(self) -> None:
        self.facts: list[Fact] = []

    def add(
        self,
        fact_id: str,
        kind: FactKind,
        subject: str | None,
        label: str,
        value: float | int | str,
        source: str,
        *,
        ci: tuple[float, float] | None = None,
        ci_sources: tuple[str, str] | None = None,
        confidence: tuple[float, str] | None = None,
    ) -> None:
        sources = (source, *(ci_sources or ()), *((confidence[1],) if confidence else ()))
        self.facts.append(
            Fact(
                id=fact_id,
                kind=kind,
                subject=subject,
                label=label,
                value=value,
                ci=ci,
                ci_confidence=confidence[0] if confidence else None,
                sources=sources,
            )
        )

    def metric(
        self,
        entry: LedgerEntry,
        fact_id: str,
        kind: FactKind,
        subject: str | None,
        label: str,
        name: str,
        *,
        ci: tuple[str, str] | None = None,
        confidence: str | tuple[float, str] | None = None,
    ) -> None:
        """A fact read from one ledger metric (and, for an interval, two more). ``confidence``
        is a ledger metric name or a (value, source) pair read elsewhere."""
        m = entry.metrics
        if name not in m:
            return
        src = f"ledger:{entry.signal_id}:{entry.stage}:"
        interval = (_num(m[ci[0]]), _num(m[ci[1]])) if ci else None
        conf = (
            (_num(m[confidence]), src + confidence) if isinstance(confidence, str) else confidence
        )
        self.add(
            fact_id,
            kind,
            subject,
            label,
            _scalar(m[name]),
            src + name,
            ci=interval,
            ci_sources=(src + ci[0], src + ci[1]) if ci else None,
            confidence=conf,
        )


def _num(v: MetricValue) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise MemoInputError(f"expected a number, got {v!r}")
    return float(v)


def _scalar(v: MetricValue) -> float | int | str:
    if isinstance(v, bool):
        return str(v).lower()
    return v


def _check(label: str, ledger_value: object, result_value: object) -> None:
    if ledger_value != result_value:
        raise MemoInputError(
            f"ledger and stage result disagree on {label}: {ledger_value!r} vs {result_value!r}"
        )


def _signals_of(result: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    rows = result.get(key)
    if not isinstance(rows, list):
        raise MemoInputError(f"stage result has no {key!r} list")
    return {str(r["signal_id"]): r for r in rows if isinstance(r, dict)}


def build_fact_sheet(run_dir: Path, *, results_root: Path | None = None) -> FactSheet:
    """The fact sheet of the completed run in ``run_dir`` (module docstring).

    ``results_root`` is the directory holding the archived validation reports (default:
    ``<project>/docs/results``).
    """
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / MANIFEST_NAME)
    if manifest.get("status") != "completed":
        raise MemoInputError(f"run {run_dir.name} is not completed: {manifest.get('status')!r}")
    ledger = KillLedger.read(run_dir)  # verifies the hash chain
    entries = ledger.entries()
    if ledger.contains_placeholders():
        raise MemoInputError("the ledger contains placeholder entries")
    causal = _read_json(run_dir / CAUSAL_RESULT_NAME)
    fates_raw = ledger.fates(MEMO_STAGES)

    fates: list[SignalFate] = []
    for sid, fate in fates_raw.items():
        if fate.status is FateStatus.INCOMPLETE or fate.stage is None:
            raise MemoInputError(f"{sid} has no final verdict over Stages 1-3")
        decided = tuple(e.stage for e in entries if e.signal_id == sid and e.stage in MEMO_STAGES)
        fates.append(
            SignalFate(
                signal_id=sid,
                status=fate.status,
                stage=fate.stage,
                reason=fate.kill.reason if fate.kill else None,
                stages_decided=decided,
            )
        )
    by_stage: dict[StageName, dict[str, LedgerEntry]] = {s: {} for s in MEMO_STAGES}
    for e in entries:
        if e.stage in by_stage:
            by_stage[e.stage][e.signal_id] = e

    columns = causal.get("signal_columns")
    if not isinstance(columns, list) or not columns:
        raise MemoInputError("causal_result.json lists no signal columns")
    synthetic = [str(c).startswith("syn_") for c in columns]
    if any(synthetic) and not all(synthetic):
        raise MemoInputError("a run mixes synthetic and real signal columns")
    data_kind = DataKind.SYNTHETIC if all(synthetic) else DataKind.REAL

    b = _Builder()
    inputs = manifest["inputs"]
    assert isinstance(inputs, dict)
    git = inputs["git"]
    assert isinstance(git, dict)
    b.add("run.run_id", FactKind.TEXT, None, "run id", str(manifest["run_id"]), "manifest:run_id")
    b.add("run.seed", FactKind.COUNT, None, "run seed", int(inputs["seed"]), "manifest:inputs.seed")
    b.add(
        "run.snapshot_id",
        FactKind.TEXT,
        None,
        "data snapshot",
        str(inputs["snapshot_id"]),
        "manifest:inputs.snapshot_id",
    )
    b.add(
        "run.git_sha",
        FactKind.TEXT,
        None,
        "code commit",
        str(git["sha"]),
        "manifest:inputs.git.sha",
    )
    b.add(
        "run.n_obs",
        FactKind.COUNT,
        None,
        "panel rows",
        int(str(causal["n_obs"])),
        f"result:{CAUSAL_RESULT_NAME}:/n_obs",
    )
    for key, label in (("first_date", "first date"), ("last_date", "last date")):
        if causal.get(key) is not None:
            b.add(
                f"run.{key}",
                FactKind.DATE,
                None,
                f"sample {label}",
                str(causal[key]),
                f"result:{CAUSAL_RESULT_NAME}:/{key}",
            )

    # Stage 1
    causal_rows = _signals_of(causal, "signals")
    for sid, e in sorted(by_stage[StageName.CAUSAL].items()):
        row = causal_rows.get(sid)
        if row is None:
            raise MemoInputError(f"{sid} has a Stage 1 ledger entry but no Stage 1 result")
        _check(f"{sid} Stage 1 verdict", str(e.verdict), row["verdict"])
        _check(f"{sid} Stage 1 reason", e.reason and str(e.reason), row["reason"])
        ev = row.get("evidence")
        if isinstance(ev, dict):
            _check(f"{sid} family p", e.metrics.get("family_p"), ev["family_p"])
            _check(f"{sid} q", e.metrics.get("q"), ev["q"])
        p = f"causal.{sid}."
        b.metric(e, p + "family_p", FactKind.P_VALUE, sid, "Stage one family p-value", "family_p")
        b.metric(e, p + "q", FactKind.P_VALUE, sid, "Stage one BH q-value", "q")
        b.metric(e, p + "alpha", FactKind.LEVEL, None, "Stage one significance level", "alpha")
        b.metric(
            e,
            p + "n_family_tests",
            FactKind.COUNT,
            sid,
            "number of lag and asset tests",
            "n_family_tests",
        )
        b.metric(
            e,
            p + "n_fdr_signals",
            FactKind.COUNT,
            None,
            "number of candidate signals in the FDR family",
            "n_fdr_signals",
        )
        b.metric(e, p + "best_asset", FactKind.TEXT, sid, "best-link asset", "best_asset")
        b.metric(e, p + "best_lag", FactKind.COUNT, sid, "best-link lag in sessions", "best_lag")
        b.metric(
            e,
            p + "r_best",
            FactKind.CORRELATION,
            sid,
            "partial correlation at the best link (descriptive bootstrap interval)",
            "r_best",
            ci=("r_best_ci_low", "r_best_ci_high"),
            confidence="r_best_ci_confidence",
        )
        b.metric(e, p + "data_issue", FactKind.TEXT, sid, "Stage one data issue", "data_issue")

    # Stage 2
    if by_stage[StageName.DECAY]:
        decay = _read_json(run_dir / DECAY_RESULT_NAME)
        decay_rows = _signals_of(decay, "signals")
        for sid, e in sorted(by_stage[StageName.DECAY].items()):
            row = decay_rows.get(sid)
            if row is None:
                raise MemoInputError(f"{sid} has a Stage 2 ledger entry but no Stage 2 result")
            _check(f"{sid} Stage 2 verdict", str(e.verdict), row["verdict"])
            _check(f"{sid} Stage 2 reason", e.reason and str(e.reason), row["reason"])
            test = row.get("test")
            if isinstance(test, dict):
                _check(f"{sid} decline p", e.metrics.get("p"), test["p"])
            p = f"decay.{sid}."
            b.metric(e, p + "p", FactKind.P_VALUE, sid, "Stage two decline-test p-value", "p")
            b.metric(e, p + "alpha", FactKind.LEVEL, None, "Stage two significance level", "alpha")
            b.metric(
                e,
                p + "b_pre",
                FactKind.COEFFICIENT,
                sid,
                "coefficient before the estimated break",
                "b_pre",
            )
            b.metric(
                e,
                p + "b_post",
                FactKind.COEFFICIENT,
                sid,
                "coefficient after the estimated break",
                "b_post",
            )
            b.metric(
                e,
                p + "break_fraction",
                FactKind.FRACTION,
                sid,
                "estimated break, as a fraction of the sample",
                "break_fraction",
            )
            b.metric(
                e, p + "min_obs", FactKind.COUNT, None, "Stage two minimum design rows", "min_obs"
            )
            b.metric(e, p + "n_rows", FactKind.COUNT, sid, "Stage two design rows", "n_rows")
            b.metric(e, p + "data_issue", FactKind.TEXT, sid, "Stage two data issue", "data_issue")

    # Stage 3
    portfolio_entries = by_stage[StageName.PORTFOLIO]
    members: tuple[str, ...] = ()
    portfolio_ran = False
    if portfolio_entries:
        port = _read_json(run_dir / PORTFOLIO_RESULT_NAME)
        member_rows = _signals_of(port, "members")
        members = tuple(sorted(member_rows))
        _check("Stage 3 members", tuple(sorted(portfolio_entries)), members)
        first = portfolio_entries[members[0]]
        shared = {k: v for k, v in first.metrics.items() if not k.startswith("b_")}
        for sid, e in sorted(portfolio_entries.items()):
            row = member_rows[sid]
            _check(f"{sid} Stage 3 verdict", str(e.verdict), row["verdict"])
            _check(f"{sid} Stage 3 reason", e.reason and str(e.reason), row["reason"])
            _check(
                f"{sid} Stage 3 shared metrics",
                {k: v for k, v in e.metrics.items() if not k.startswith("b_")},
                shared,
            )
            _check(f"{sid} B", e.metrics.get("b_marginal_sharpe"), row.get("marginal_sharpe"))
            conf = port["config"]
            assert isinstance(conf, dict)
            if "b_marginal_sharpe" in e.metrics:
                b.add(
                    f"portfolio.{sid}.b",
                    FactKind.SHARPE,
                    sid,
                    "marginal Sharpe contribution (diagnostic B)",
                    _num(e.metrics["b_marginal_sharpe"]),
                    f"ledger:{sid}:portfolio:b_marginal_sharpe",
                    ci=(_num(e.metrics["b_ci_low"]), _num(e.metrics["b_ci_high"])),
                    ci_sources=(
                        f"ledger:{sid}:portfolio:b_ci_low",
                        f"ledger:{sid}:portfolio:b_ci_high",
                    ),
                    confidence=(
                        float(str(conf["confidence"])),
                        f"result:{PORTFOLIO_RESULT_NAME}:/config/confidence",
                    ),
                )
        evidence = port.get("evidence")
        portfolio_ran = isinstance(evidence, dict)
        if isinstance(evidence, dict):
            arms = evidence["arms"]
            assert isinstance(arms, list)
            for a in arms:
                _check(f"{a['arm']} Sharpe", first.metrics.get(f"{a['arm']}_sharpe"), a["sharpe"])
                _check(f"{a['arm']} LW p", first.metrics.get(f"{a['arm']}_lw_p"), a["lw_p"])
        e = first
        stress_conf: tuple[float, str] | None = None
        if isinstance(evidence, dict) and isinstance(evidence.get("stress"), dict):
            st = evidence["stress"]
            assert isinstance(st, dict)
            stress_conf = (
                float(str(st["confidence"])),
                f"result:{PORTFOLIO_RESULT_NAME}:/evidence/stress/confidence",
            )
        b.metric(
            e,
            "portfolio.n_members",
            FactKind.COUNT,
            "the portfolio",
            "number of members",
            "n_members",
        )
        b.metric(
            e,
            "portfolio.n_test_days",
            FactKind.COUNT,
            "the portfolio",
            "out-of-sample days",
            "n_test_days",
        )
        b.metric(
            e,
            "portfolio.lw_alpha",
            FactKind.LEVEL,
            None,
            "Stage three significance level",
            "lw_alpha",
        )
        b.metric(
            e,
            "portfolio.cost_bp",
            FactKind.LEVEL,
            None,
            "assumed cost in basis points per unit of turnover",
            "cost_bp",
        )
        b.metric(
            e, "portfolio.min_obs", FactKind.COUNT, None, "Stage three minimum rows", "min_obs"
        )
        b.metric(
            e,
            "portfolio.data_issue",
            FactKind.TEXT,
            "the portfolio",
            "Stage three data issue",
            "data_issue",
        )
        for arm, name in ARM_LABELS.items():
            k = arm.value
            subj = f"the {name} arm"
            b.metric(
                e,
                f"portfolio.{k}.sharpe",
                FactKind.SHARPE,
                subj,
                "gross annualised Sharpe ratio",
                f"{k}_sharpe",
            )
            b.metric(
                e,
                f"portfolio.{k}.sharpe_net",
                FactKind.SHARPE,
                subj,
                "annualised Sharpe ratio net of costs",
                f"{k}_sharpe_net",
            )
            b.metric(
                e,
                f"portfolio.{k}.dsr",
                FactKind.PROBABILITY,
                subj,
                "deflated Sharpe ratio (a probability)",
                f"{k}_dsr",
            )
            b.metric(
                e,
                f"portfolio.{k}.lw_p",
                FactKind.P_VALUE,
                subj,
                "one-sided Ledoit-Wolf p-value against equal weight",
                f"{k}_lw_p",
            )
            b.metric(
                e,
                f"portfolio.{k}.turnover",
                FactKind.FRACTION,
                subj,
                "mean daily one-way turnover",
                f"{k}_turnover",
            )
            b.metric(
                e,
                f"portfolio.{k}.invested",
                FactKind.FRACTION,
                subj,
                "mean invested share",
                f"{k}_invested",
            )
        b.metric(
            e,
            "stress.ridge_cv_high",
            FactKind.SHARPE,
            "the stress diagnostic",
            "Ridge-CV Sharpe ratio in the high-volatility state",
            "stress_ridge_cv_sharpe_high",
            ci=("stress_ridge_cv_ci_low", "stress_ridge_cv_ci_high"),
            confidence=stress_conf,
        )
        b.metric(
            e,
            "stress.ridge_cv_low",
            FactKind.SHARPE,
            "the stress diagnostic",
            "Ridge-CV Sharpe ratio outside the high-volatility state",
            "stress_ridge_cv_sharpe_low",
        )
        b.metric(
            e,
            "stress.equal_weight_high",
            FactKind.SHARPE,
            "the stress diagnostic",
            "equal-weight Sharpe ratio in the high-volatility state",
            "stress_equal_weight_sharpe_high",
            ci=("stress_equal_weight_ci_low", "stress_equal_weight_ci_high"),
            confidence=stress_conf,
        )
        b.metric(
            e,
            "stress.high_share",
            FactKind.FRACTION,
            "the stress diagnostic",
            "share of days in the high-volatility state",
            "stress_high_share",
        )
        b.metric(
            e,
            "stress.block_length",
            FactKind.COUNT,
            "the stress diagnostic",
            "bootstrap block length in days",
            "stress_block_length",
        )
        if portfolio_ran:
            b.add(
                "const.dsr_trials",
                FactKind.COUNT,
                None,
                "number of trials the deflated Sharpe ratio corrects for",
                DSR_TRIALS,
                "const:dsr_trials",
            )

    b.facts.extend(load_reference_facts(results_root))
    ids = [f.id for f in b.facts]
    if len(ids) != len(set(ids)):
        raise MemoInputError("duplicate fact ids")
    fates_t = tuple(sorted(fates, key=lambda f: f.signal_id))
    return FactSheet(
        run_id=str(manifest["run_id"]),
        data_kind=data_kind,
        fates=fates_t,
        members=members,
        portfolio_ran=portfolio_ran,
        facts=tuple(b.facts),
        recommendation=recommend(fates_t),
    )


def stage_number(stage: StageName) -> int:
    """1, 2 or 3 for the kill stages."""
    return MEMO_STAGES.index(stage) + 1
