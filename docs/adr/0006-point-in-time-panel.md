# ADR 0006: Point-in-time feature panel and target timing

Status: accepted (Phase 0)

## Context
Non-negotiable #1: a value used at session t may use only information available at t. ADR 0004
recorded what each source guarantees and bound `pit.py` to a next-session rule for ALFRED. This
ADR fixes how `invariant.data.pit` turns a snapshot into features and forward-return targets,
and how target timing is kept consistent between real data and the planted-signal benchmark.

## Decisions

**Information time per source.** Each raw value gets the first session it may be used. A
feature at t is a function of values whose information time is at or before t.

| Source | Information time | Feature |
|---|---|---|
| ETF adjusted close | the session itself (known at the close) | `ret_<T>`: log return, never a level |
| French factors | `publication_lag_sessions.french` sessions after the row's date (default **42**, about 2 months) | `ff_<factor>`, placed at that session, not carried forward |
| ALFRED vintage row | first session **strictly after** `realtime_start` (ADR 0004) | `macro_<S>`: newest observation in the vintages usable at t, plus `obs_dates` |

**ALFRED as-of rule.** A vintage row is usable on sessions `[first session > realtime_start,
first session > realtime_end + 1 day)`. The end is the session after the superseding vintage's
date, which is exactly when the next vintage becomes usable, so there is no gap and no overlap.
`realtime_end` of a still-current row is future information. It is used only through the test
`t < stop`, whose answer does not depend on it while `stop > t`. Withdrawn observations stop
being used once the withdrawal itself is usable.

**NaN vintage rows** (`.`) are never used as values and never resurrect an older value of the
same observation. If the newest observation is NaN, the feature is the newest observation that
has a value in the same vintages, and `obs_dates` shows the staleness.

**Panel columns come from configuration, not data.** Macro columns are the configured series,
so the column set cannot depend on which vintages exist yet. A series with nothing usable yet
is an all-NaN column.

**Targets.** `y_t = r_{t+1+e} + ... + r_{t+e+h}` with `e = pit.targets.execution_lag` and
`h = pit.targets.horizon`. NaN if the window has a NaN or runs past the sample. Both values are
**required config with no code default**:

- `configs/base.yaml`: `execution_lag: 1`. A feature computed from the close at t cannot be
  traded in the closing auction at t, so the position is entered at the close of t+1.
- `configs/benchmark.yaml`: `pit.targets.execution_lag: 0`, `horizon: 1`. The SCM plants
  `S_t -> R_{t+1}`, and every expected value in `GroundTruth` (marginal, partial and effective
  ICs, expected t-stats) is a moment of `(S_t, R_{t+1})`. With lag 1, a stage would be scored
  against `Corr(S_t, R_{t+2})`, roughly `rho` times the planted effect plus any decay, and could
  pass or fail for the wrong reason.

**Benchmark timing is enforced, not documented only.** `GroundTruth.target_timing` records the
timing its expected values assume (`EXPECTED_VALUES_TARGET_TIMING`, lag 0 and horizon 1;
`GROUND_TRUTH_SCHEMA_VERSION` 2). `invariant.benchmark.scoring.scoring_targets` is the only
entry point for benchmark targets. It raises `TargetTimingMismatchError` on any mismatch,
including horizon. `tests/benchmark/test_target_timing.py` fails if the benchmark YAML drifts
from `GroundTruth` or if the real-data timing is used to score the benchmark. It also checks
that the sample IC under the matching timing recovers the population IC.

**Leakage tests** (`tests/leakage/test_point_in_time.py`): a Hypothesis property perturbs every
input whose information time is after t and requires bit-identical features and observation
dates at sessions ≤ t. Macro inputs are generated as release histories, so changing the future
also moves `realtime_end` of past rows, adds and removes rows, and withdraws observations. The
property is itself tested: `hypothesis.find` must expose four deliberately leaky variants
(same-day vintages, final revised values, unlagged French factors, next-close returns). A real
ALFRED excerpt (PAYEMS 2024-01..03, all vintages, including the 2025 and 2026 benchmark
revisions) checks that the value used differs from the final revised value at every session
before the last revision is usable.

## Alternatives considered
- *"Latest vintage with `realtime_start <= t`"*, taken literally: this allows same-day use of a
  release with no time of day. Rejected by ADR 0004. Strictly-after is used.
- *Forward-filling French factors to the next available session*: this hides missing rows.
  Rejected for consistency with the ETF policy (nothing is filled).
- *A code default for `execution_lag`*: it would make it easy to score the benchmark with the
  real-data lag by accident. Rejected; it is required config.
- *Storing target timing in `SyntheticMarketConfig`*: timing does not change the generated
  data, and adding it would change dataset identity for no reason. Kept as a separate
  `pit.targets` section of `benchmark.yaml`.
- *Differenced macro features now*: a `.diff()` of the as-of series mixes vintages. Deferred to
  a follow-up that computes changes within one vintage.

## Consequences
- `PipelineConfig` gained a required `pit` section. Run ids change for the same experiment (the
  golden run id was re-pinned). Canonicalization is unchanged, so `HASH_SCHEMA_VERSION` stays 1.
- `GroundTruth` JSON gained `target_timing`, and the schema version moved to 2. The dataset
  content hash is unchanged.
- The first `french` lag sessions of the panel have no French factors. Rows dated before the
  panel are dropped rather than counted against a calendar the snapshot does not have.
- Macro features are levels only. INDPRO, PAYEMS and CPI levels are non-stationary. Stages must
  not treat them as stationary predictors until within-vintage changes exist.
