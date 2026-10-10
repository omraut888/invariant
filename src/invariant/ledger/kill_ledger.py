"""The kill ledger: an append-only, tamper-evident record of every candidate's fate in a run.

Format
------
``runs/<run_id>/kill_ledger.jsonl`` holds one ``LedgerEntry`` per line, serialised as canonical
JSON (``invariant.core.run_hash.canonical_json``), so a given entry always has the same bytes.
An entry records one decision: ``(signal_id, stage, verdict)``, the ``KillReason`` for a kill,
the ``metrics`` behind the decision, a UTC timestamp and the run ``attempt`` it belongs to.
``placeholder`` marks entries not produced by a real stage (smoke runs, tests). It is part of the
hashed line, so it cannot be changed after the fact.

Integrity
---------
The ledger is a hash chain: ``prev_sha256`` of entry n is the SHA-256 of line n-1 (64 zeros for
the first line), and ``seq`` counts lines from 0. A chain alone cannot see a change to the
*last* line or a truncation at a line boundary, so ``kill_ledger.head.json`` records the entry
count and the last line's hash. It is replaced atomically after each append. Every read checks
the chain, ``seq``, the head, the run id and the per-attempt decision rules below. So an edited,
deleted, inserted or reordered line raises ``LedgerIntegrityError``, and so does a partial last
line or a line beyond the head (a crash mid-write). Nothing is repaired automatically. There is
no update or delete API.
Limitation: a chain proves internal consistency, not authenticity. Someone who rewrites the
whole file and recomputes every hash is not detected (that would need signing or an external
anchor).

Decision rules (per attempt, enforced on write and re-checked on read)
----------------------------------------------------------------------
A signal's decisions follow ``StageName`` order with no gaps, starting at ``causal``. Each
``(signal, stage)`` is decided at most once, and nothing follows a KILL. A KILL needs a
``reason`` and non-empty ``metrics``. A PASS has no reason. A stage-specific reason must belong
to the entry's stage. ``general.placeholder`` is only allowed when ``placeholder=True``.

Attempts
--------
A run that is resumed (``RunContext.create`` on existing inputs) gets a new ``attempt``. Its
entries are appended after the earlier attempt's, so the file stays append-only, and queries
default to the latest attempt so a crashed attempt's partial decisions are never mixed in.
Writing requires the run to be in status ``created`` with the attempt the ledger was opened
for. Limitation: single writer per run, as for ``RunContext`` (no inter-process lock).

See docs/adr/0007-kill-ledger-and-benchmark-scoring.md.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)
from structlog.typing import FilteringBoundLogger

from invariant.core.logging import get_logger
from invariant.core.run_hash import (
    MANIFEST_NAME,
    RunContext,
    RunHashError,
    RunManifest,
    RunStateError,
    RunStatus,
    canonical_json,
)
from invariant.core.types import KillReason, StageName, Verdict, make_signal_id

__all__ = [
    "GENESIS_SHA256",
    "HEAD_NAME",
    "LEDGER_NAME",
    "LEDGER_SCHEMA_VERSION",
    "STAGE_ORDER",
    "Fate",
    "FateStatus",
    "KillLedger",
    "LedgerEntry",
    "LedgerError",
    "LedgerIntegrityError",
    "MetricValue",
    "check_stage_prefix",
]

LEDGER_NAME: Final = "kill_ledger.jsonl"
HEAD_NAME: Final = "kill_ledger.head.json"
LEDGER_SCHEMA_VERSION: Final = 1
GENESIS_SHA256: Final = "0" * 64
STAGE_ORDER: Final[tuple[StageName, ...]] = tuple(StageName)

MetricValue = StrictBool | StrictInt | StrictFloat | str


def _log() -> FilteringBoundLogger:
    return get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class LedgerError(ValueError):
    """A decision violates the ledger's rules."""


class LedgerIntegrityError(LedgerError):
    """The ledger file on disk is not a valid, unbroken chain of entries."""


class LedgerEntry(BaseModel):
    """One decision about one signal at one stage (see module docstring)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_schema: int = LEDGER_SCHEMA_VERSION
    seq: int = Field(ge=0)
    run_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    attempt: int = Field(ge=1)
    signal_id: str
    stage: StageName
    verdict: Verdict
    reason: KillReason | None
    metrics: dict[str, MetricValue]
    placeholder: bool = False
    recorded_at: datetime
    prev_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("signal_id")
    @classmethod
    def _check_signal_id(cls, value: str) -> str:
        return make_signal_id(value)

    @field_validator("recorded_at")
    @classmethod
    def _check_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("metrics")
    @classmethod
    def _check_metrics(cls, value: dict[str, MetricValue]) -> dict[str, MetricValue]:
        bad = [k for k, v in value.items() if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            raise ValueError(f"metrics must be finite, got non-finite {bad}")
        if any(not k for k in value):
            raise ValueError("metric names must be non-empty")
        return value

    @model_validator(mode="after")
    def _check_decision(self) -> Self:
        if self.verdict is Verdict.KILL:
            if self.reason is None:
                raise ValueError("a KILL needs a reason")
            if not self.metrics:
                raise ValueError("a KILL needs the metrics behind the decision")
        elif self.reason is not None:
            raise ValueError("a PASS has no reason")
        if self.reason is not None and self.reason.stage not in (None, self.stage):
            raise ValueError(f"reason {self.reason} belongs to stage {self.reason.stage}")
        if self.reason is KillReason.PLACEHOLDER and not self.placeholder:
            raise ValueError("general.placeholder is only allowed on placeholder entries")
        return self

    def line(self) -> str:
        """The canonical serialisation written to disk (no trailing newline)."""
        return canonical_json(self.model_dump(mode="json"))


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Fates
# ---------------------------------------------------------------------------------------------


class FateStatus(StrEnum):
    KILLED = "killed"
    PASSED = "passed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class Fate:
    """Outcome of one signal over an evaluated stage prefix.

    ``KILLED``: killed at ``stage`` (``kill`` is the entry). ``PASSED``: passed every evaluated
    stage. ``INCOMPLETE``: neither; ``stage`` is the last stage decided, or ``None``.
    """

    signal_id: str
    status: FateStatus
    stage: StageName | None
    kill: LedgerEntry | None = None


def check_stage_prefix(stages: Sequence[StageName]) -> tuple[StageName, ...]:
    """Return ``stages`` as a tuple if it is a non-empty prefix of ``STAGE_ORDER``."""
    out = tuple(stages)
    if not out or out != STAGE_ORDER[: len(out)]:
        raise ValueError(
            f"stages must be a non-empty prefix of {[str(s) for s in STAGE_ORDER]}, got "
            f"{[str(s) for s in out]}"
        )
    return out


def _check_next(prior: Sequence[LedgerEntry], entry: LedgerEntry) -> None:
    if prior and prior[-1].verdict is Verdict.KILL:
        raise LedgerError(
            f"{entry.signal_id} was killed at {prior[-1].stage}; nothing may follow a kill"
        )
    if len(prior) == len(STAGE_ORDER):
        raise LedgerError(f"{entry.signal_id} already has a decision at every stage")
    expected = STAGE_ORDER[len(prior)]
    if entry.stage is not expected:
        raise LedgerError(
            f"{entry.signal_id}: next decision must be at {expected}, got {entry.stage}"
        )


# ---------------------------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------------------------


class _Head(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_entries: int = Field(ge=0)
    last_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _write_head(run_dir: Path, head: _Head) -> None:
    path = run_dir / HEAD_NAME
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=f".{HEAD_NAME}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(head.model_dump(mode="json")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _check_head(run_dir: Path, entries: list[LedgerEntry], last_sha: str) -> None:
    try:
        raw = (run_dir / HEAD_NAME).read_bytes()
    except FileNotFoundError:
        if entries:
            raise LedgerIntegrityError(f"{run_dir}: ledger has entries but no head") from None
        return
    try:
        head = _Head.model_validate_json(raw)
    except ValueError as exc:
        raise LedgerIntegrityError(f"{run_dir}: invalid ledger head: {exc}") from None
    if head.n_entries != len(entries):
        raise LedgerIntegrityError(
            f"{run_dir}: ledger has {len(entries)} entries, head records {head.n_entries} "
            "(truncated, or interrupted between append and head update)"
        )
    if head.last_sha256 != last_sha:
        raise LedgerIntegrityError(f"{run_dir}: last ledger line does not match the head")


def _read_entries(run_dir: Path, run_id: str) -> list[LedgerEntry]:
    path = run_dir / LEDGER_NAME
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    if not data:
        _check_head(run_dir, [], GENESIS_SHA256)
        return []
    if not data.endswith(b"\n"):
        raise LedgerIntegrityError(f"{path}: partial last line (interrupted write?)")
    entries: list[LedgerEntry] = []
    by_signal: dict[tuple[int, str], list[LedgerEntry]] = {}
    prev = GENESIS_SHA256
    for n, raw in enumerate(data.decode("utf-8").split("\n")[:-1]):
        try:
            entry = LedgerEntry.model_validate_json(raw)
        except ValueError as exc:
            raise LedgerIntegrityError(f"{path}: line {n} is not a valid entry: {exc}") from None
        if entry.line() != raw:
            raise LedgerIntegrityError(f"{path}: line {n} is not in canonical form")
        if entry.seq != n:
            raise LedgerIntegrityError(f"{path}: line {n} has seq {entry.seq}")
        if entry.prev_sha256 != prev:
            raise LedgerIntegrityError(f"{path}: hash chain broken at line {n}")
        if entry.run_id != run_id:
            raise LedgerIntegrityError(f"{path}: line {n} belongs to run {entry.run_id}")
        if entries and entry.attempt < entries[-1].attempt:
            raise LedgerIntegrityError(f"{path}: line {n} goes back to an earlier attempt")
        prior = by_signal.setdefault((entry.attempt, entry.signal_id), [])
        try:
            _check_next(prior, entry)
        except LedgerError as exc:
            raise LedgerIntegrityError(f"{path}: line {n}: {exc}") from None
        prior.append(entry)
        entries.append(entry)
        prev = _sha(raw)
    _check_head(run_dir, entries, prev)
    return entries


class KillLedger:
    """A run's kill ledger. Use ``open`` (to record) or ``read`` (read-only)."""

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        entries: list[LedgerEntry],
        *,
        ctx: RunContext | None = None,
        attempt: int | None = None,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self._entries = entries
        self._ctx = ctx
        self._attempt = attempt
        self._now = now

    @property
    def path(self) -> Path:
        return self.run_dir / LEDGER_NAME

    @classmethod
    def open(cls, ctx: RunContext, *, now: Callable[[], datetime] = _utcnow) -> KillLedger:
        """Open ``ctx``'s ledger for recording decisions of the run's current attempt."""
        attempt = ctx.manifest().attempt
        entries = _read_entries(ctx.run_dir, ctx.run_id)
        return cls(ctx.run_dir, ctx.run_id, entries, ctx=ctx, attempt=attempt, now=now)

    @classmethod
    def read(cls, run_dir: Path) -> KillLedger:
        """Read (and verify) the ledger of the run in ``run_dir``, read-only."""
        try:
            manifest = RunManifest.model_validate_json((run_dir / MANIFEST_NAME).read_bytes())
        except FileNotFoundError:
            raise RunHashError(f"{run_dir} has no run manifest") from None
        entries = _read_entries(run_dir, manifest.run_id)
        return cls(run_dir, manifest.run_id, entries)

    # --- writing ------------------------------------------------------------------------------

    def record(
        self,
        signal_id: str,
        stage: StageName,
        verdict: Verdict,
        *,
        metrics: Mapping[str, MetricValue],
        reason: KillReason | None = None,
        placeholder: bool = False,
    ) -> LedgerEntry:
        """Append one decision and return it as stored (see the module's decision rules)."""
        if self._ctx is None or self._attempt is None:
            raise RunStateError("this ledger was opened read-only")
        manifest = self._ctx.manifest()
        if manifest.status is not RunStatus.CREATED:
            raise RunStateError(f"run {self.run_id} is {manifest.status}; its ledger is closed")
        if manifest.attempt != self._attempt:
            raise RunStateError(
                f"run {self.run_id} is now at attempt {manifest.attempt}; this ledger was "
                f"opened for attempt {self._attempt}"
            )
        try:
            draft = LedgerEntry(
                seq=len(self._entries),
                run_id=self.run_id,
                attempt=self._attempt,
                signal_id=signal_id,
                stage=stage,
                verdict=verdict,
                reason=reason,
                metrics=dict(metrics),
                placeholder=placeholder,
                recorded_at=self._now(),
                prev_sha256=_sha(self._entries[-1].line()) if self._entries else GENESIS_SHA256,
            )
        except ValueError as exc:
            raise LedgerError(str(exc)) from exc
        prior = [
            e
            for e in self._entries
            if e.attempt == self._attempt and e.signal_id == draft.signal_id
        ]
        _check_next(prior, draft)
        line = draft.line()
        # Re-parse so the returned entry equals what a reader will see (canonicalisation maps
        # -0.0 to 0.0).
        stored = LedgerEntry.model_validate_json(line)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        _write_head(self.run_dir, _Head(n_entries=stored.seq + 1, last_sha256=_sha(line)))
        self._entries.append(stored)
        _log().info(
            "ledger_recorded",
            run_id=self.run_id,
            signal_id=stored.signal_id,
            stage=str(stage),
            verdict=str(verdict),
            reason=None if reason is None else str(reason),
            placeholder=placeholder,
        )
        return stored

    # --- queries ------------------------------------------------------------------------------

    def attempts(self) -> tuple[int, ...]:
        """Attempts that have entries, ascending."""
        return tuple(sorted({e.attempt for e in self._entries}))

    def latest_attempt(self) -> int | None:
        if self._attempt is not None:
            return self._attempt
        attempts = self.attempts()
        return attempts[-1] if attempts else None

    def all_entries(self) -> tuple[LedgerEntry, ...]:
        """Every entry of every attempt, in file order."""
        return tuple(self._entries)

    def entries(self, attempt: int | None = None) -> tuple[LedgerEntry, ...]:
        """Entries of ``attempt`` (default: the latest attempt), in file order."""
        which = self.latest_attempt() if attempt is None else attempt
        return tuple(e for e in self._entries if e.attempt == which)

    def signal_ids(self, attempt: int | None = None) -> tuple[str, ...]:
        return tuple(sorted({e.signal_id for e in self.entries(attempt)}))

    def what_killed(self, signal_id: str, attempt: int | None = None) -> LedgerEntry | None:
        """The KILL entry for ``signal_id``, or ``None`` if it was not killed."""
        return next(
            (
                e
                for e in self.entries(attempt)
                if e.signal_id == signal_id and e.verdict is Verdict.KILL
            ),
            None,
        )

    def kill_counts_by_stage(self, attempt: int | None = None) -> dict[StageName, int]:
        """Number of kills at each stage, every stage present (zeros included)."""
        counts = dict.fromkeys(STAGE_ORDER, 0)
        for e in self.entries(attempt):
            if e.verdict is Verdict.KILL:
                counts[e.stage] += 1
        return counts

    def fates(
        self, stages: Sequence[StageName] = STAGE_ORDER, attempt: int | None = None
    ) -> dict[str, Fate]:
        """Fate of every signal in the ledger over the evaluated stage prefix ``stages``."""
        evaluated = check_stage_prefix(stages)
        decisions: dict[str, list[LedgerEntry]] = {}
        for e in self.entries(attempt):
            if e.stage in evaluated:
                decisions.setdefault(e.signal_id, []).append(e)
        out: dict[str, Fate] = {}
        for sid in self.signal_ids(attempt):
            made = decisions.get(sid, [])
            if made and made[-1].verdict is Verdict.KILL:
                out[sid] = Fate(sid, FateStatus.KILLED, made[-1].stage, made[-1])
            elif len(made) == len(evaluated):
                out[sid] = Fate(sid, FateStatus.PASSED, made[-1].stage)
            else:
                out[sid] = Fate(sid, FateStatus.INCOMPLETE, made[-1].stage if made else None)
        return out

    def contains_placeholders(self, attempt: int | None = None) -> bool:
        return any(e.placeholder for e in self.entries(attempt))
