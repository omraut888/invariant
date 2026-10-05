# ADR 0002: Run identity, run manifests, and named RNG streams

Status: accepted (Phase 0)

## Context
Non-negotiable #5 requires every run to be keyed by a hash of (data snapshot id, config, git SHA,
seed), and all randomness to come from explicitly seeded generators. The run id names the run
directory, and every artifact, ledger row, and memo is later traced back through it. So two
different experiments must never share an id, and the same experiment must always get the same
id. Stages need independent random streams that stay stable as the pipeline grows.

## Decisions

**Hashed payload.** `RunInputs` contains `hash_schema`, `snapshot_id`, the canonical config,
`git` (sha, dirty, diff_sha256), `package_version`, and `seed`. The seed comes from
`PipelineConfig.seed`, so there is a single source of truth. `run_id` is the first 16 hex chars
(64 bits) of the SHA-256 of its canonical JSON, and the manifest keeps the full digest.
`hash_schema` is bumped whenever canonicalization changes. A golden-id test catches any accidental
change.

**Exact canonicalization, no float rounding.** Keys are sorted, `-0.0` becomes `0.0`, NaN and inf
are rejected, numpy scalars become Python scalars, and floats are written with `repr`, the shortest
round-trip form. Sets, `Path`s, non-str keys, and tz-aware datetimes are rejected. The request
asked for "normalized floats". Rounding was rejected because it lets two different configs collide
on one run_id, and they would then share a run directory.

**Dirty trees hash their diff.** A dirty flag alone would give two runs with different uncommitted
edits the same id. When the tree is dirty, the SHA-256 of a patch is also hashed. The patch is
`git diff HEAD` plus new-file diffs for untracked, non-ignored files. It is saved as
`runs/<id>/dirty.patch` and applies with `git apply` on top of the commit, and the manifest records
`reproducible_from_sha: false`. A missing git, a missing repository, or a missing commit is a hard
error, because a run without a SHA isn't reproducible.

**Environment recorded, not hashed.** Python, platform, and key package versions go in the manifest.
They are pinned by `uv.lock`, which the git SHA covers. Hashing them would give the same experiment
different ids on different machines.

**Name-keyed RNG streams.** A stream is
`SeedSequence(entropy=seed, spawn_key=parent_key + words(sha256(name)))` wrapped in an explicit
PCG64 `Generator`. Positional `SeedSequence.spawn` was rejected for stage streams because
inserting a stage shifts the indices, and so the streams, of every later stage. `RngFactory.child`
nests streams, e.g. stage, then signal. Positional `spawn` is still fine inside a stage for
homogeneous replicates. A factory refuses to issue the same name twice, which prevents silently
correlated streams.

**Run lifecycle.** The manifest carries `status` (created → completed | failed), `completed_at`,
`error`, and `attempt`, outside the hashed `inputs`. Every manifest write is atomic (tmp file +
`os.replace`). Re-creating a run with the same inputs resumes it:
- the prior status is logged, with a warning when the previous attempt failed or crashed, because
  stage artifacts may be partial
- the status resets to `created`
- `attempt` increments
- `created_at` is kept

A manifest whose inputs disagree with the recomputed ones is a hard error.

## Alternatives considered
- *Rounding floats to N significant digits*: causes collisions. Rejected.
- *Hashing only the dirty flag*: two different dirty trees would collide. Rejected.
- *Refusing dirty runs entirely*: too strict for research iteration. A dirty run is allowed, but
  it is flagged and its patch is saved.
- *Positional `spawn()` per stage*: not stable when stages are added. Rejected.
- *Hashing environment versions*: makes ids machine-specific. Rejected; the versions are recorded
  instead.
- *Full 64-hex run ids*: unwieldy directory names, and 64 bits is already far beyond any realistic
  run count.

## Consequences and known limitations
- NEP 19 lets numpy change `Generator` distribution algorithms between versions, so an environment
  that drifts from `uv.lock` can change results under the same run_id. A resume logs
  `run_environment_drift`, and the manifest is the evidence.
- No inter-process lock: two processes running the same run_id at the same time will race on the
  manifest.
- Until the snapshot layer exists, `snapshot_id` is a caller-supplied string. It should become a
  content hash of the snapshot.
- The golden run-id and golden-stream tests pin canonicalization and the RNG derivation. Changing
  either is a deliberate, versioned decision.
