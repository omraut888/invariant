# ADR 0008: Synthetic feature source

Status: accepted (Phase 0). Supersedes the "new `SourceName.SYNTHETIC`" alternative rejected in
ADR 0007.

## Context
The Phase 0 smoke run wrote only the `etf` payload (prices) for a planted-signal dataset. The
PIT panel therefore held returns and nothing else: the candidate signals (and Z, when
observed) never reached the snapshot, so Stage 1 had nothing to run on. ADR 0007 had rejected a
dedicated synthetic source because, with prices only, it changed nothing. It now changes what
the panel contains.

The features must arrive with exactly the timing GroundTruth's expected values assume, and the
benchmark's answer key must not be able to leak in alongside them.

## Decisions

**A separate source, `SourceName.SYNTHETIC`.** Table `features`: a `date` column plus one
float64 column per observed feature, wide, named as GroundTruth names the nodes. Params:
`provider: synthetic`, `dataset_sha256`, `seed`, `columns`. The contract (table name, report,
reader) lives in `data/sources/synthetic.py`, so `invariant.data` never imports
`invariant.benchmark`. The builder is `benchmark.payload.synthetic_feature_payload`. It is never
fetched: `build_snapshot` defaults to `FETCHED_SOURCES` (`etf`, `french`, `alfred`) and refuses
anything else.

**An allowlist, not a denylist.** The payload accepts only columns that GroundTruth marks as
observed candidates or confounders. Z under `hidden_confounder` and every oracle-only latent
(`decay_multiplier`, which is the beta(t) path, `h_*`, `market`, `h_market`) are refused, even
if `dataset.features` were edited to hold them. A new latent added to the generator is excluded
by default.

**Lag 0, as a constant.** `syn_<name>` at session t is the value dated t
(`SYNTHETIC_FEATURE_LAG_SESSIONS = 0`), which is GroundTruth's `timing_convention`: a signal
dated t is known at the close of t and scored against `R_{t+1}`. It is not a YAML tunable,
because any other value scores stages against moments the answer key does not describe (the
same reasoning as `require_target_timing`, ADR 0006). Dates must be panel sessions. An
off-calendar date raises instead of being moved to the next session, and missing rows stay
missing.

**One dataset per snapshot.** When a snapshot has synthetic features, `build_pit_panel`
requires the ETF source to be synthetic with the same `dataset_sha256`, and `score_run` checks
the feature source's hash against GroundTruth as well as the price source's. Features of one
seed can never be paired with returns of another.

## Tests
- `tests/leakage/test_point_in_time.py`: the generic Hypothesis property now perturbs synthetic
  features dated after t, and a lag −1 variant is among the leaks the property must detect.
- `tests/leakage/test_synthetic_features.py`, end to end through real snapshots:
  - Perturbing features after t leaves panel rows at or before t bit-identical.
  - Perturbing the value dated t changes row t, so the lag is exactly 0.
  - `syn_*` equals `dataset.features` bit for bit.
  - No oracle latent reaches the snapshot or panel, checked by name (columns, params) and by
    value (catching renamed leaks), and the detector flags planted latents.
  - `hidden_confounder` has no `z` in any table, params or panel column.
- The smoke script asserts the panel holds exactly the observed features and no latent, for
  `base` and `hidden_confounder`.

## Limitations
- Lag 0 with `execution_lag = 0` means trading at the close the signal is observed at. That is
  the benchmark's convention and not a claim about real execution. Real signals will need their
  own information times.
- The value-level leak check compares against each latent column as stored. A leak through a
  non-trivial transform of a latent (e.g. `sqrt(h_t)`) would pass it. The allowlist is the
  primary guard; the detector is the backstop.

## Alternatives considered
- *Extra table in the `etf` payload*: no enum change, but it mixes prices and signals in one
  source and makes "is this snapshot synthetic" depend on table names. Rejected.
- *Long `(date, feature, value)` table*: the column set is fixed per dataset and checked against
  GroundTruth, so wide is simpler and exact. Rejected.
- *Configurable lag (`publication_lag_sessions.synthetic`)*: see above. Rejected.

## Consequences
- Benchmark snapshots now hold two sources, so their snapshot ids (and run hashes) differ from
  before. Existing local benchmark runs will not resume. Real-data snapshot ids are unchanged.
- `PitInputs` gained an optional `synthetic` field. `PitPanel.features` may hold `syn_*`
  columns. Stage 1 maps `syn_<id>` to signal `<id>` by dropping the prefix.
- The smoke script's expected confusion matrix now leaves out non-identifiable signals, as the
  scorer does. `--preset hidden_confounder` previously always failed on that.
