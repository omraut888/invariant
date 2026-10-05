"""Run identity, run manifests, and named RNG streams.

Run identity
------------
A run is keyed by ``run_id = sha256(canonical_json(RunInputs))[:16]`` where ``RunInputs`` holds
the data snapshot id, the canonicalized config, the git state, the package version, and the
seed. The full 64-hex digest is kept in the manifest; 16 hex chars (64 bits) keep directory names
short while making a collision negligible (birthday bound ~2^32 runs).

Canonicalization is *exact*: mapping keys are sorted, ``-0.0`` becomes ``0.0``, numpy scalars
become Python scalars, NaN/inf are rejected, and floats serialise with ``repr`` (the shortest
string that round-trips, identical on every IEEE-754 platform). Floats are deliberately **not**
rounded: rounding would let two different configs share a run_id and silently share a run
directory. ``bool`` and ``int`` stay distinct, as do ``1`` and ``1.0`` in raw mappings (typed
pydantic fields already coerce YAML ``1`` to ``1.0``). ``HASH_SCHEMA_VERSION`` is part of the
hashed payload and must be bumped whenever canonicalization changes.

A dirty working tree is flagged *and* contributes the SHA-256 of its patch (tracked changes plus
untracked, non-ignored files) to the hash, so two runs with different uncommitted edits never
share a run_id. The patch is saved next to the manifest as ``dirty.patch``. Such a run is not
reproducible from the commit SHA alone, and the manifest says so (``reproducible_from_sha``).

Environment versions (python, numpy, ...) are recorded in the manifest but not hashed: they are
pinned by ``uv.lock``, which the git SHA covers, and hashing them would give the same experiment
different ids on different machines. Limitation: NEP 19 permits numpy to change ``Generator``
distribution algorithms between versions, so an environment that drifts from the lockfile can
change results under the same run_id; the manifest's ``environment`` block is the evidence.

RNG streams
-----------
``RngFactory`` derives one ``np.random.Generator`` per *name* (e.g. a ``StageName``) from the
root seed with ``SeedSequence(entropy=seed, spawn_key=parent_key + words(sha256(name)))``. This is
the ``SeedSequence.spawn`` mechanism keyed by name rather than position: ``spawn(n)`` gives child
``i`` the key ``(i,)``, so inserting a stage would shift every later stage's stream. Keyed by
name, a stream depends only on (seed, path of names), never on which other streams exist.
SeedSequence hashes its whole entropy+spawn_key pool, so children are statistically independent
(O'Neill 2015, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for
Random Number Generation"; numpy NEP 19). Positional ``SeedSequence.spawn`` remains appropriate
*inside* a stage for homogeneous replicates (bootstrap draws), whose order is fixed.

Run lifecycle
-------------
The manifest carries ``status`` (created -> completed | failed), ``completed_at``, ``error`` and
``attempt``. These live outside ``inputs`` and so never affect the run_id. Re-creating a run with
the same inputs resumes it: the prior status is logged (with a warning if the prior attempt
failed or crashed, since stage artifacts may be partial), the status resets to ``created`` and
``attempt`` increments. Limitation: there is no inter-process lock; two processes running the
same run_id concurrently will race on the manifest.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from structlog.typing import FilteringBoundLogger

from invariant.config.settings import (
    PipelineConfig,
    Settings,
    find_project_root,
    load_settings,
)
from invariant.core.logging import get_logger

__all__ = [
    "HASH_SCHEMA_VERSION",
    "RUN_ID_LENGTH",
    "GitState",
    "RngFactory",
    "RngStreamReusedError",
    "RunContext",
    "RunHashError",
    "RunInputs",
    "RunManifest",
    "RunStateError",
    "RunStatus",
    "canonical_json",
    "canonicalize",
    "compute_digest",
    "compute_run_id",
    "read_git_state",
]

HASH_SCHEMA_VERSION = 1
RUN_ID_LENGTH = 16
MANIFEST_NAME = "manifest.json"
PATCH_NAME = "dirty.patch"
_PACKAGE = "invariant"
_ENV_PACKAGES = ("numpy", "pandas", "scipy", "pydantic", "pyarrow", "structlog", "pyyaml")
_SNAPSHOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _log() -> FilteringBoundLogger:
    # Resolved per call: binding at import time would freeze the structlog config in place
    # before configure_logging() runs.
    return get_logger(__name__)


class RunHashError(Exception):
    """Run identity could not be established, or an existing run directory is inconsistent."""


class RunStateError(Exception):
    """Illegal run lifecycle transition."""


class RngStreamReusedError(Exception):
    """The same RNG stream name was requested twice from one factory."""


# ---------------------------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------------------------


def canonicalize(obj: object, _path: str = "$") -> JsonValue:
    """Convert ``obj`` to a canonical JSON value (see module docstring for the rules).

    Raises ``TypeError`` naming the dotted path of the first unsupported value and ``ValueError``
    for non-finite floats or tz-aware datetimes.
    """
    if isinstance(obj, BaseModel):
        return canonicalize(obj.model_dump(mode="python"), _path)
    if isinstance(obj, Enum):
        return canonicalize(obj.value, _path)
    if isinstance(obj, np.generic):
        return canonicalize(obj.item(), _path)
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float {obj!r} at {_path} cannot be hashed")
        return 0.0 if obj == 0.0 else float(obj)
    if isinstance(obj, datetime):
        if obj.tzinfo is not None:
            raise ValueError(f"tz-aware datetime at {_path}; timestamps must be tz-naive")
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        out: dict[str, JsonValue] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"mapping key {key!r} at {_path} must be str")
            out[key] = canonicalize(value, f"{_path}.{key}")
        return out
    if isinstance(obj, list | tuple):
        return [canonicalize(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, Path):
        raise TypeError(f"Path at {_path} is machine-specific and cannot be hashed")
    if isinstance(obj, set | frozenset):
        raise TypeError(f"set at {_path} has no canonical order; use a sorted list")
    raise TypeError(f"unsupported type {type(obj).__name__} at {_path}")


def canonical_json(obj: object) -> str:
    """Serialise ``canonicalize(obj)`` with sorted keys and no insignificant whitespace."""
    return json.dumps(
        canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


# ---------------------------------------------------------------------------------------------
# Git state
# ---------------------------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GitState(_Frozen):
    """Commit SHA plus dirty flag; ``diff_sha256`` is the patch digest, set iff dirty."""

    sha: str
    dirty: bool
    diff_sha256: str | None = None

    @field_validator("sha")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        if not _SHA1_RE.fullmatch(value):
            raise ValueError(f"git sha must be 40 lowercase hex chars, got {value!r}")
        return value

    @field_validator("diff_sha256")
    @classmethod
    def _check_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError(f"diff_sha256 must be 64 lowercase hex chars, got {value!r}")
        return value

    @model_validator(mode="after")
    def _digest_iff_dirty(self) -> Self:
        if self.dirty != (self.diff_sha256 is not None):
            raise ValueError("diff_sha256 must be set exactly when the tree is dirty")
        return self


def _git(repo: Path, *args: str, ok_codes: tuple[int, ...] = (0,)) -> bytes:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        raise RunHashError("git executable not found; a run needs a commit SHA") from None
    if proc.returncode not in ok_codes:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise RunHashError(f"`git {' '.join(args)}` failed in {repo}: {stderr}")
    return proc.stdout


_DIFF_FLAGS = ("--binary", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/")


def read_git_state(repo_root: Path | None = None) -> tuple[GitState, bytes | None]:
    """Return the git state of ``repo_root`` (default: project root) and, if dirty, the patch.

    The patch is ``git diff HEAD`` plus a new-file diff for every untracked, non-ignored file, so
    it applies with ``git apply`` on top of the commit. Ignored paths (``runs/``, ``data/``) never
    make the tree dirty. Raises ``RunHashError`` if git is unavailable, ``repo_root`` is not a
    repository, or there is no commit.
    """
    repo = repo_root if repo_root is not None else find_project_root()
    sha = _git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not status:
        return GitState(sha=sha, dirty=False), None
    patch = _git(repo, "diff", "HEAD", *_DIFF_FLAGS)
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    for rel in sorted(p for p in untracked.decode().split("\0") if p):
        # --no-index exits 1 when the files differ, which they always do here.
        patch += _git(
            repo, "diff", "--no-index", *_DIFF_FLAGS, "--", os.devnull, rel, ok_codes=(0, 1)
        )
    digest = hashlib.sha256(patch).hexdigest()
    return GitState(sha=sha, dirty=True, diff_sha256=digest), patch


# ---------------------------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------------------------


class RunInputs(_Frozen):
    """Everything that determines a run's results; the hashed payload."""

    hash_schema: int = HASH_SCHEMA_VERSION
    snapshot_id: str
    config: dict[str, JsonValue]
    git: GitState
    package_version: str = Field(min_length=1)
    seed: int = Field(ge=0, lt=2**32)

    @field_validator("snapshot_id")
    @classmethod
    def _check_snapshot_id(cls, value: str) -> str:
        if not _SNAPSHOT_ID_RE.fullmatch(value):
            raise ValueError(
                f"invalid snapshot id {value!r}: must match {_SNAPSHOT_ID_RE.pattern}",
            )
        return value

    @field_validator("config", mode="before")
    @classmethod
    def _canonical_config(cls, value: object) -> JsonValue:
        return canonicalize(value)

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        snapshot_id: str,
        git: GitState,
        package_version: str,
    ) -> RunInputs:
        """Build inputs from a pipeline config; the seed is taken from ``config.seed``."""
        return cls(
            snapshot_id=snapshot_id,
            config=config.model_dump(mode="python"),
            git=git,
            package_version=package_version,
            seed=config.seed,
        )


def compute_digest(inputs: RunInputs) -> str:
    """Full SHA-256 hex digest of the canonical JSON of ``inputs``."""
    return hashlib.sha256(canonical_json(inputs).encode("utf-8")).hexdigest()


def compute_run_id(inputs: RunInputs) -> str:
    """The run id: the first ``RUN_ID_LENGTH`` hex chars of ``compute_digest(inputs)``."""
    return compute_digest(inputs)[:RUN_ID_LENGTH]


# ---------------------------------------------------------------------------------------------
# RNG streams
# ---------------------------------------------------------------------------------------------


def _name_key(name: str) -> tuple[int, ...]:
    if not isinstance(name, str) or not name:
        raise ValueError(f"RNG stream name must be a non-empty str, got {name!r}")
    words = np.frombuffer(hashlib.sha256(name.encode("utf-8")).digest(), dtype=">u4")
    return tuple(int(w) for w in words)


class RngFactory:
    """Derives independent, name-keyed ``np.random.Generator`` streams from one root seed.

    Each name may be claimed once per factory instance (by ``generator``, ``seed_sequence`` or
    ``child``); a second request raises ``RngStreamReusedError``, because two consumers silently
    sharing one stream would get perfectly correlated randomness. Separate factory instances
    with the same seed reproduce each other exactly. Generators use PCG64 explicitly so a change
    of numpy's ``default_rng`` bit generator cannot alter streams. See the module docstring for
    the derivation and its independence argument.
    """

    def __init__(self, seed: int, *, _spawn_key: tuple[int, ...] = ()) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")
        self._seed = seed
        self._spawn_key = _spawn_key
        self._issued: set[str] = set()

    @property
    def seed(self) -> int:
        return self._seed

    def _claim(self, name: str) -> tuple[int, ...]:
        key = _name_key(name)
        if name in self._issued:
            raise RngStreamReusedError(f"RNG stream {name!r} was already issued by this factory")
        self._issued.add(name)
        return self._spawn_key + key

    def seed_sequence(self, name: str) -> np.random.SeedSequence:
        """Claim ``name`` and return its ``SeedSequence`` (e.g. to ``spawn`` replicates)."""
        return np.random.SeedSequence(entropy=self._seed, spawn_key=self._claim(name))

    def generator(self, name: str) -> np.random.Generator:
        """Claim ``name`` and return a fresh PCG64 generator for its stream."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence(name)))

    def child(self, name: str) -> RngFactory:
        """Claim ``name`` and return a sub-factory whose streams are nested under it."""
        return RngFactory(self._seed, _spawn_key=self._claim(name))


# ---------------------------------------------------------------------------------------------
# Manifest and run context
# ---------------------------------------------------------------------------------------------


class RunStatus(StrEnum):
    CREATED = "created"
    COMPLETED = "completed"
    FAILED = "failed"


class RunManifest(_Frozen):
    """Contents of ``runs/<run_id>/manifest.json``. Only ``inputs`` determines the run id."""

    manifest_schema: int = 1
    run_id: str
    digest: str
    inputs: RunInputs
    reproducible_from_sha: bool
    status: RunStatus
    attempt: int = Field(ge=1)
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    environment: dict[str, str | None]


def _environment() -> dict[str, str | None]:
    env: dict[str, str | None] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for pkg in _ENV_PACKAGES:
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = None
    return env


def _now() -> datetime:
    return datetime.now(UTC)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    text = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    _atomic_write(path, text.encode("utf-8"))


def _read_manifest(path: Path) -> RunManifest:
    try:
        return RunManifest.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        raise RunHashError(f"run directory {path.parent} exists but has no manifest") from None
    except ValueError as exc:
        raise RunHashError(f"manifest {path} is invalid: {exc}") from exc


@dataclass(frozen=True)
class RunContext:
    """A run's identity, directory, and RNG factory. Construct via ``RunContext.create``."""

    run_id: str
    digest: str
    inputs: RunInputs
    run_dir: Path
    rng: RngFactory = field(compare=False)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    @classmethod
    def create(
        cls,
        config: PipelineConfig,
        snapshot_id: str,
        *,
        settings: Settings | None = None,
        repo_root: Path | None = None,
        git_state: GitState | None = None,
        patch: bytes | None = None,
    ) -> RunContext:
        """Compute the run id and create (or resume) ``<runs_dir>/<run_id>/manifest.json``.

        ``git_state``/``patch`` default to ``read_git_state(repo_root)``; pass them to inject a
        state in tests. Resuming with matching inputs resets the status to ``created`` and
        increments ``attempt``; a manifest whose inputs differ raises ``RunHashError``.
        """
        if git_state is None:
            git_state, patch = read_git_state(repo_root)
        if git_state.dirty and patch is None:
            raise RunHashError("a dirty git state requires its patch")
        inputs = RunInputs.from_config(
            config,
            snapshot_id,
            git_state,
            importlib.metadata.version(_PACKAGE),
        )
        digest = compute_digest(inputs)
        run_id = digest[:RUN_ID_LENGTH]
        runs_dir = (settings if settings is not None else load_settings()).runs_dir
        ctx = cls(
            run_id=run_id,
            digest=digest,
            inputs=inputs,
            run_dir=runs_dir / run_id,
            rng=RngFactory(inputs.seed),
        )
        if ctx.run_dir.exists():
            ctx._resume()
        else:
            ctx.run_dir.mkdir(parents=True)
            _write_manifest(
                ctx.manifest_path,
                RunManifest(
                    run_id=run_id,
                    digest=digest,
                    inputs=inputs,
                    reproducible_from_sha=not git_state.dirty,
                    status=RunStatus.CREATED,
                    attempt=1,
                    created_at=_now(),
                    environment=_environment(),
                ),
            )
            _log().info("run_created", run_id=run_id, run_dir=str(ctx.run_dir))
        if patch is not None and not (ctx.run_dir / PATCH_NAME).exists():
            _atomic_write(ctx.run_dir / PATCH_NAME, patch)
        return ctx

    def _resume(self) -> None:
        prior = _read_manifest(self.manifest_path)
        if prior.inputs != self.inputs or prior.digest != self.digest:
            raise RunHashError(
                f"manifest in {self.run_dir} does not match this run's inputs "
                "(hash collision or canonicalization bug); refusing to reuse the directory",
            )
        _log().info(
            "run_resumed",
            run_id=self.run_id,
            prior_status=str(prior.status),
            prior_attempt=prior.attempt,
        )
        environment = _environment()
        if environment != prior.environment:
            _log().warning(
                "run_environment_drift",
                run_id=self.run_id,
                prior=prior.environment,
                current=environment,
            )
        if prior.status is not RunStatus.COMPLETED:
            _log().warning(
                "run_resumed_from_incomplete",
                run_id=self.run_id,
                prior_status=str(prior.status),
                prior_error=prior.error,
                detail="stage artifacts in the run directory may be partial",
            )
        _write_manifest(
            self.manifest_path,
            prior.model_copy(
                update={
                    "status": RunStatus.CREATED,
                    "attempt": prior.attempt + 1,
                    "completed_at": None,
                    "error": None,
                    "environment": environment,
                },
            ),
        )

    def manifest(self) -> RunManifest:
        """Read the current manifest from disk."""
        return _read_manifest(self.manifest_path)

    def status(self) -> RunStatus:
        return self.manifest().status

    def _finish(self, status: RunStatus, error: str | None) -> None:
        current = self.manifest()
        if current.status is not RunStatus.CREATED:
            raise RunStateError(
                f"run {self.run_id} is already {current.status}; cannot mark it {status}",
            )
        _write_manifest(
            self.manifest_path,
            current.model_copy(
                update={"status": status, "completed_at": _now(), "error": error},
            ),
        )
        _log().info("run_finished", run_id=self.run_id, status=str(status), error=error)

    def mark_completed(self) -> None:
        """Transition ``created -> completed``."""
        self._finish(RunStatus.COMPLETED, None)

    def mark_failed(self, error: BaseException | str) -> None:
        """Transition ``created -> failed``, recording ``"<ExcType>: <message>"``."""
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self._finish(RunStatus.FAILED, message)
