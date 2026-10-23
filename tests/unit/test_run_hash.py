import itertools
import json
import os
import subprocess
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
import structlog
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from scipy import stats
from structlog.testing import capture_logs

from invariant.config.settings import (
    AlfredConfig,
    CausalBootstrapConfig,
    CausalConfig,
    DataConfig,
    DateRange,
    DecayConfig,
    FrenchConfig,
    HttpConfig,
    PipelineConfig,
    PitConfig,
    PublicationLags,
    Settings,
    TargetConfig,
    UniverseConfig,
)
from invariant.core.run_hash import (
    MANIFEST_NAME,
    PATCH_NAME,
    GitState,
    RngFactory,
    RngStreamReusedError,
    RunContext,
    RunHashError,
    RunInputs,
    RunStateError,
    RunStatus,
    canonical_json,
    canonicalize,
    compute_digest,
    compute_run_id,
    read_git_state,
)
from invariant.core.types import Frequency, StageName

SHA_A = "a" * 40
SHA_B = "b" * 40
DIGEST = "c" * 64
# Literal, not loaded from base.yaml, so the golden run id does not move with config edits.
DATA = DataConfig(
    calendar="XNYS",
    http=HttpConfig(timeout_s=60, max_retries=3, backoff_s=1.0),
    french=FrenchConfig(
        five_factor_url="https://example.test/ff5.zip",
        momentum_url="https://example.test/mom.zip",
    ),
    alfred=AlfredConfig(series=("UNRATE",), page_size=100_000),
)
PIT = PitConfig(
    targets=TargetConfig(execution_lag=1, horizon=1),
    publication_lag_sessions=PublicationLags(french=42),
)

CAUSAL = CausalConfig(
    ci_test="parcorr_hc3",
    decision="hybrid",
    tau_max=5,
    pc_alpha=0.2,
    alpha=0.05,
    fdr="bh",
    min_obs=500,
    signals=None,
    bootstrap=CausalBootstrapConfig(n_boot=999, block_length=None, confidence=0.95),
)

DECAY = DecayConfig(alpha=0.05, n_boot=999, trim=0.15, break_step=25, min_obs=5000)


def _config(**overrides: object) -> PipelineConfig:
    base: dict[str, object] = {
        "universe": UniverseConfig(tickers=("SPY", "TLT")),
        "dates": DateRange(start=date(2005, 1, 3), end=date(2025, 12, 31)),
        "frequency": Frequency.DAILY,
        "seed": 42,
        "data": DATA,
        "pit": PIT,
        "causal": CAUSAL,
        "decay": DECAY,
    }
    base.update(overrides)
    return PipelineConfig.model_validate(base)


def _inputs(**overrides: object) -> RunInputs:
    config = overrides.pop("config", _config())
    assert isinstance(config, PipelineConfig)
    fields: dict[str, object] = {
        "snapshot_id": "snap-2025-12-31",
        "git": GitState(sha=SHA_A, dirty=False),
        "package_version": "0.1.0",
    }
    fields.update(overrides)
    return RunInputs.from_config(config, **fields)  # type: ignore[arg-type]


# --- run identity -----------------------------------------------------------------------------


def test_same_inputs_same_run_id() -> None:
    assert compute_run_id(_inputs()) == compute_run_id(_inputs())
    assert len(compute_run_id(_inputs())) == 16
    assert compute_digest(_inputs()).startswith(compute_run_id(_inputs()))


def test_golden_run_id() -> None:
    # Pinned: if this changes without a change to the PipelineConfig schema, canonicalization
    # changed and HASH_SCHEMA_VERSION must be bumped. (Last re-pinned when `decay` was added.)
    assert compute_run_id(_inputs()) == "6b24e63ac1f49496"


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param({"config": _config(universe=UniverseConfig(tickers=("SPY",)))}, id="ticker"),
        pytest.param(
            {"config": _config(dates=DateRange(start=date(2005, 1, 4), end=date(2025, 12, 31)))},
            id="date",
        ),
        pytest.param({"config": _config(frequency=Frequency.WEEKLY)}, id="frequency"),
        pytest.param({"config": _config(seed=43)}, id="seed"),
        pytest.param(
            {
                "config": _config(
                    pit=PIT.model_copy(update={"targets": TargetConfig(execution_lag=0, horizon=1)})
                )
            },
            id="execution_lag",
        ),
        pytest.param({"snapshot_id": "snap-2025-12-30"}, id="snapshot"),
        pytest.param({"git": GitState(sha=SHA_B, dirty=False)}, id="git_sha"),
        pytest.param({"git": GitState(sha=SHA_A, dirty=True, diff_sha256=DIGEST)}, id="dirty"),
        pytest.param({"package_version": "0.1.1"}, id="package_version"),
    ],
)
def test_any_input_change_changes_run_id(variant: dict[str, object]) -> None:
    assert compute_run_id(_inputs(**variant)) != compute_run_id(_inputs())


def test_different_dirty_diffs_get_different_run_ids() -> None:
    a = _inputs(git=GitState(sha=SHA_A, dirty=True, diff_sha256="1" * 64))
    b = _inputs(git=GitState(sha=SHA_A, dirty=True, diff_sha256="2" * 64))
    assert compute_run_id(a) != compute_run_id(b)


def test_seed_comes_from_config() -> None:
    assert _inputs(config=_config(seed=7)).seed == 7


def test_git_state_digest_iff_dirty() -> None:
    with pytest.raises(ValidationError):
        GitState(sha=SHA_A, dirty=True)
    with pytest.raises(ValidationError):
        GitState(sha=SHA_A, dirty=False, diff_sha256=DIGEST)
    with pytest.raises(ValidationError):
        GitState(sha="abc", dirty=False)


def test_invalid_snapshot_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _inputs(snapshot_id="../escape")


# --- canonicalization -------------------------------------------------------------------------

_json_leaf = st.none() | st.booleans() | st.integers() | st.text(max_size=5)
_json_value = st.recursive(
    _json_leaf,
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=5), children, max_size=4)
    ),
    max_leaves=20,
)


def _reverse_keys(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _reverse_keys(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_reverse_keys(v) for v in obj]
    return obj


@given(st.dictionaries(st.text(max_size=5), _json_value, max_size=6))
def test_key_order_does_not_matter(mapping: dict[str, object]) -> None:
    assert canonical_json(mapping) == canonical_json(_reverse_keys(mapping))


@given(st.dictionaries(st.text(min_size=1, max_size=5), _json_value, min_size=1, max_size=6))
def test_key_order_does_not_change_run_id(raw: dict[str, object]) -> None:
    git = GitState(sha=SHA_A, dirty=False)
    a = RunInputs(snapshot_id="s", config=raw, git=git, package_version="1", seed=0)
    b = RunInputs(snapshot_id="s", config=_reverse_keys(raw), git=git, package_version="1", seed=0)
    assert compute_run_id(a) == compute_run_id(b)


def test_negative_zero_normalized() -> None:
    assert canonical_json({"x": -0.0}) == canonical_json({"x": 0.0})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize({"x": bad})


def test_floats_are_not_rounded() -> None:
    assert canonical_json(0.1) != canonical_json(0.1000000000000001)
    assert canonical_json(1e-9) != canonical_json(1.4e-9)


def test_numpy_scalars_match_python() -> None:
    assert canonical_json({"x": np.float64(0.1), "n": np.int64(3), "b": np.bool_(True)}) == (
        canonical_json({"x": 0.1, "n": 3, "b": True})
    )


def test_bool_and_int_distinct() -> None:
    assert canonical_json(True) != canonical_json(1)


def test_tuple_and_list_equivalent() -> None:
    assert canonical_json((1, 2)) == canonical_json([1, 2])


@pytest.mark.parametrize(
    ("bad", "exc"),
    [
        ({"x": {1, 2}}, TypeError),
        ({"x": Path("/tmp")}, TypeError),
        ({1: "a"}, TypeError),
        ({"x": datetime(2020, 1, 1, tzinfo=UTC)}, ValueError),
        ({"x": object()}, TypeError),
    ],
)
def test_unsupported_values_rejected(bad: object, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        canonicalize(bad)


def test_error_names_path() -> None:
    with pytest.raises(TypeError, match=r"\$\.a\.b\[1\]"):
        canonicalize({"a": {"b": [1, {2}]}})


# --- RNG streams ------------------------------------------------------------------------------


def test_streams_reproducible_across_factories() -> None:
    a = RngFactory(123).generator(StageName.CAUSAL).random(10)
    b = RngFactory(123).generator("causal").random(10)
    np.testing.assert_array_equal(a, b)


def test_different_names_and_seeds_differ() -> None:
    f = RngFactory(123)
    assert not np.array_equal(f.generator("causal").random(10), f.generator("decay").random(10))
    a = RngFactory(1).generator("causal").random(10)
    b = RngFactory(2).generator("causal").random(10)
    assert not np.array_equal(a, b)


def test_adding_a_stream_does_not_change_others() -> None:
    alone = RngFactory(9).generator("causal").random(100)
    f = RngFactory(9)
    f.generator("new_stage_inserted_first")
    f.generator("decay")
    np.testing.assert_array_equal(f.generator("causal").random(100), alone)


def test_child_factories_reproducible_and_distinct() -> None:
    a = RngFactory(5).child("causal").generator("sig-1").random(10)
    b = RngFactory(5).child("causal").generator("sig-1").random(10)
    np.testing.assert_array_equal(a, b)
    flat = RngFactory(5).generator("sig-1").random(10)
    assert not np.array_equal(a, flat)


def test_golden_stream() -> None:
    # Pinned: a change here silently changes every stochastic result in the system.
    draws = RngFactory(0).generator("causal").integers(0, 2**31, 5).tolist()
    assert draws == [1198434052, 424897039, 1580416821, 1505549008, 1636791995]


def test_reuse_guard() -> None:
    f = RngFactory(0)
    f.generator("causal")
    with pytest.raises(RngStreamReusedError):
        f.generator("causal")
    with pytest.raises(RngStreamReusedError):
        f.child("causal")


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_invalid_seed_rejected(seed: int) -> None:
    with pytest.raises(ValueError):
        RngFactory(seed)


def test_invalid_name_rejected() -> None:
    with pytest.raises(ValueError):
        RngFactory(0).generator("")


SEEDS = (0, 20260105, 2**32 - 1)
STAGE_PAIRS = list(itertools.combinations(list(StageName), 2))
N_DRAWS = 100_000
# Bonferroni over every (seed, pair) chi-square test.
ALPHA = 1e-4 / (len(SEEDS) * len(STAGE_PAIRS))


@pytest.mark.parametrize("seed", SEEDS)
def test_stage_streams_statistically_independent(seed: int) -> None:
    f = RngFactory(seed)
    draws = {stage: f.generator(stage).random(N_DRAWS) for stage in StageName}
    for a, b in STAGE_PAIRS:
        x, y = draws[a], draws[b]
        r = float(np.corrcoef(x, y)[0, 1])
        # Under independence r ~ N(0, 1/n): 5 sigma.
        assert abs(r) < 5 / np.sqrt(N_DRAWS), (a, b, r)
        joint, _, _ = np.histogram2d(x, y, bins=10, range=[[0, 1], [0, 1]])
        p = stats.chisquare(joint.ravel()).pvalue
        assert p > ALPHA, (a, b, p)


@pytest.mark.parametrize("seed", SEEDS)
def test_stage_streams_do_not_overlap(seed: int) -> None:
    f = RngFactory(seed)
    raw = [set(f.generator(s).bit_generator.random_raw(10_000).tolist()) for s in StageName]
    for a, b in itertools.combinations(raw, 2):
        assert not a & b


# --- git state --------------------------------------------------------------------------------

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=_GIT_ENV)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _run_git(path, "init", "-q")
    (path / "tracked.txt").write_text("v1\n")
    (path / ".gitignore").write_text("ignored/\n")
    _run_git(path, "add", ".")
    _run_git(path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    return path


def test_clean_tree(repo: Path) -> None:
    state, patch = read_git_state(repo)
    assert not state.dirty
    assert state.diff_sha256 is None
    assert patch is None
    assert len(state.sha) == 40


def test_modified_tracked_file_is_dirty(repo: Path) -> None:
    (repo / "tracked.txt").write_text("v2\n")
    state, patch = read_git_state(repo)
    assert state.dirty
    assert patch is not None
    assert b"+v2" in patch


def test_untracked_file_is_dirty_and_in_patch(repo: Path) -> None:
    (repo / "new.py").write_text("x = 1\n")
    state, patch = read_git_state(repo)
    assert state.dirty
    assert patch is not None
    assert b"new.py" in patch
    assert b"+x = 1" in patch


def test_different_edits_have_different_digests(repo: Path) -> None:
    (repo / "tracked.txt").write_text("v2\n")
    first, _ = read_git_state(repo)
    (repo / "tracked.txt").write_text("v3\n")
    second, _ = read_git_state(repo)
    assert first.diff_sha256 != second.diff_sha256


def test_ignored_files_do_not_dirty(repo: Path) -> None:
    (repo / "ignored").mkdir()
    (repo / "ignored" / "run.json").write_text("{}")
    assert not read_git_state(repo)[0].dirty


def test_patch_applies_to_commit(repo: Path, tmp_path: Path) -> None:
    (repo / "tracked.txt").write_text("v2\n")
    (repo / "new.py").write_text("x = 1\n")
    _, patch = read_git_state(repo)
    assert patch is not None
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True, env=_GIT_ENV)
    patch_file = tmp_path / "p.patch"
    patch_file.write_bytes(patch)
    _run_git(clone, "apply", str(patch_file))
    assert (clone / "tracked.txt").read_text() == "v2\n"
    assert (clone / "new.py").read_text() == "x = 1\n"


def test_non_repo_raises(tmp_path: Path) -> None:
    with pytest.raises(RunHashError):
        read_git_state(tmp_path)


# --- manifest and lifecycle -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(runs_dir=tmp_path / "runs")


CLEAN = GitState(sha=SHA_A, dirty=False)


def _create(settings: Settings, **kwargs: object) -> RunContext:
    params: dict[str, object] = {"settings": settings, "git_state": CLEAN}
    params.update(kwargs)
    return RunContext.create(_config(), "snap-1", **params)  # type: ignore[arg-type]


def _manifest_json(ctx: RunContext) -> dict[str, object]:
    data = json.loads(ctx.manifest_path.read_text())
    assert isinstance(data, dict)
    return data


def test_create_writes_manifest(settings: Settings) -> None:
    ctx = _create(settings)
    assert ctx.run_dir == settings.runs_dir / ctx.run_id
    data = _manifest_json(ctx)
    assert data["run_id"] == ctx.run_id
    assert data["digest"] == ctx.digest
    assert data["status"] == "created"
    assert data["attempt"] == 1
    assert data["reproducible_from_sha"] is True
    assert data["completed_at"] is None
    assert data["error"] is None
    env = data["environment"]
    assert isinstance(env, dict)
    assert env["numpy"] == np.__version__
    assert {"python", "pandas", "scipy", "pydantic", "pyarrow", "structlog", "pyyaml"} <= set(env)
    assert ctx.run_id == compute_run_id(RunInputs.model_validate(data["inputs"]))
    assert not (ctx.run_dir / PATCH_NAME).exists()
    assert not [p for p in ctx.run_dir.iterdir() if p.name != MANIFEST_NAME]  # no tmp leftovers


def test_run_rng_uses_config_seed(settings: Settings) -> None:
    ctx = _create(settings)
    np.testing.assert_array_equal(
        ctx.rng.generator("causal").random(5), RngFactory(42).generator("causal").random(5)
    )


def test_dirty_state_writes_patch(settings: Settings) -> None:
    dirty = GitState(sha=SHA_A, dirty=True, diff_sha256=DIGEST)
    ctx = _create(settings, git_state=dirty, patch=b"diff --git a/x b/x\n")
    assert (ctx.run_dir / PATCH_NAME).read_bytes() == b"diff --git a/x b/x\n"
    assert _manifest_json(ctx)["reproducible_from_sha"] is False
    assert ctx.run_id != _create(settings).run_id


def test_dirty_state_without_patch_rejected(settings: Settings) -> None:
    with pytest.raises(RunHashError):
        _create(settings, git_state=GitState(sha=SHA_A, dirty=True, diff_sha256=DIGEST))


def test_recreate_is_idempotent_and_keeps_created_at(settings: Settings) -> None:
    first = _create(settings)
    created_at = _manifest_json(first)["created_at"]
    first.mark_completed()
    second = _create(settings)
    assert second.run_id == first.run_id
    assert _manifest_json(second)["created_at"] == created_at


def test_tampered_manifest_raises(settings: Settings) -> None:
    ctx = _create(settings)
    data = _manifest_json(ctx)
    inputs = data["inputs"]
    assert isinstance(inputs, dict)
    inputs["snapshot_id"] = "other"
    ctx.manifest_path.write_text(json.dumps(data))
    with pytest.raises(RunHashError, match="does not match"):
        _create(settings)


def test_corrupt_or_missing_manifest_raises(settings: Settings) -> None:
    ctx = _create(settings)
    ctx.manifest_path.write_text("{not json")
    with pytest.raises(RunHashError, match="invalid"):
        _create(settings)
    ctx.manifest_path.unlink()
    with pytest.raises(RunHashError, match="no manifest"):
        _create(settings)


def test_mark_completed(settings: Settings) -> None:
    ctx = _create(settings)
    assert ctx.status() is RunStatus.CREATED
    ctx.mark_completed()
    m = ctx.manifest()
    assert m.status is RunStatus.COMPLETED
    assert m.completed_at is not None
    assert m.completed_at >= m.created_at
    assert m.error is None


def test_mark_failed(settings: Settings) -> None:
    ctx = _create(settings)
    ctx.mark_failed(ValueError("x"))
    m = ctx.manifest()
    assert m.status is RunStatus.FAILED
    assert m.error == "ValueError: x"
    assert m.completed_at is not None


@pytest.mark.parametrize("first", ["completed", "failed"])
@pytest.mark.parametrize("second", ["completed", "failed"])
def test_cannot_mark_ended_run(settings: Settings, first: str, second: str) -> None:
    ctx = _create(settings)

    def mark(status: str) -> None:
        if status == "completed":
            ctx.mark_completed()
        else:
            ctx.mark_failed("boom")

    mark(first)
    with pytest.raises(RunStateError):
        mark(second)
    assert ctx.status() == RunStatus(first)


@pytest.mark.parametrize("prior", ["created", "failed"])
def test_resume_incomplete_warns_and_resets(settings: Settings, prior: str) -> None:
    ctx = _create(settings)
    if prior == "failed":
        ctx.mark_failed("boom")
    with capture_logs() as logs:
        resumed = _create(settings)
    events = {e["event"]: e for e in logs}
    assert events["run_resumed"]["prior_status"] == prior
    warning = events["run_resumed_from_incomplete"]
    assert warning["log_level"] == "warning"
    assert "partial" in warning["detail"]
    m = resumed.manifest()
    assert m.status is RunStatus.CREATED
    assert m.attempt == 2
    assert m.error is None
    assert m.completed_at is None
    resumed.mark_completed()  # the new attempt can be marked again
    assert resumed.status() is RunStatus.COMPLETED


def test_resume_completed_does_not_warn(settings: Settings) -> None:
    _create(settings).mark_completed()
    with capture_logs() as logs:
        _create(settings)
    events = [e["event"] for e in logs]
    assert "run_resumed" in events
    assert "run_resumed_from_incomplete" not in events


def test_resume_completed_resets_and_can_be_marked_again(settings: Settings) -> None:
    first = _create(settings)
    first.mark_completed()
    created_at = first.manifest().created_at
    for attempt in (2, 3):
        resumed = _create(settings)
        assert resumed.run_id == first.run_id
        m = resumed.manifest()
        assert m.status is RunStatus.CREATED
        assert m.attempt == attempt
        assert m.completed_at is None
        assert m.error is None
        assert m.created_at == created_at
        resumed.mark_completed()
        assert resumed.status() is RunStatus.COMPLETED


def test_status_changes_do_not_change_run_id(settings: Settings) -> None:
    ctx = _create(settings)
    original_inputs = json.dumps(_manifest_json(ctx)["inputs"], sort_keys=True)

    def check() -> None:
        data = _manifest_json(ctx)
        assert json.dumps(data["inputs"], sort_keys=True) == original_inputs
        assert compute_run_id(RunInputs.model_validate(data["inputs"])) == ctx.run_id
        assert data["run_id"] == ctx.run_id

    ctx.mark_failed(RuntimeError("crash"))
    check()
    resumed = _create(settings)
    assert resumed.run_id == ctx.run_id
    check()
    resumed.mark_completed()
    check()
    assert _create(settings).run_id == ctx.run_id
    check()
