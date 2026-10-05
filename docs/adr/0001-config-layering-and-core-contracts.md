# ADR 0001: Config layering, path resolution, and core contracts

Status: accepted (Phase 0)

## Context
Every later component needs secrets and paths, a pipeline config that goes into the run hash,
shared enums for stages/verdicts/kill reasons, and one time-series index contract. Choices made
here are hard to change once runs and ledger rows exist.

## Decisions

**Settings.** `Settings` (pydantic-settings, frozen) reads `FRED_API_KEY`, `INVARIANT_DATA_DIR`,
`INVARIANT_RUNS_DIR` from the environment and `.env`. `FRED_API_KEY` is optional so the
benchmark and unit tests run without a key. The ALFRED source must raise when it is missing.

**Paths resolve against the project root, never the cwd.** `find_project_root()` walks up from
the settings module to the first directory containing `pyproject.toml`. The default `.env`,
`configs/base.yaml`, and any relative `data_dir`/`runs_dir` (from defaults, env vars, or `.env`)
are resolved against it. This needs an editable install. In a non-editable install the lookup
raises `ConfigError` instead of guessing.

**YAML layering.** `load_config(base, override)` merges the override onto the base with
`deep_merge`. Nested mappings merge recursively, while lists and scalars replace. Concatenating
lists would make it impossible to shrink the universe from an override. The merged mapping is
validated into a frozen `PipelineConfig` with `extra="forbid"` at every level, so a typo fails
the run instead of being silently ignored and still changing the run hash. `seed` is a
`StrictInt` in [0, 2^32), so YAML `"42"` or `true` is rejected.

**Universe and sample window.** SPY, the nine original sector SPDRs, TLT, IEF, and GLD, from
2005-01-03 to 2025-12-31. GLD (inception 2004-11-18) sets the start date of the balanced panel.
The end date is fixed because a moving "today" breaks reproducibility.

**Time-series index contract.** `validate_time_index` requires a non-empty, tz-naive, strictly
increasing `DatetimeIndex` with no NaT or duplicates. The daily check is a trading-day check:
weekdays at midnight, with consecutive gaps of at most `max_gap_bdays` (5) business days. Weekly
and monthly checks require exactly one observation per consecutive calendar period.

**KillReason.** This is one `StrEnum` with namespaced values `"<stage>.<code>"`, or
`"general.<code>"` for cross-cutting reasons. Enums with members can't be subclassed, so stages
add members to this enum. A unit test enforces the naming convention, and `.stage` recovers the
owning stage. Enum values are persisted in the ledger, so they are pinned by a test.

## Alternatives considered
- *Resolving against the cwd*: this is fragile, since notebooks/ and test tmpdirs would write
  data to the wrong place. Rejected.
- *An exact NYSE calendar check (`exchange_calendars`)*: more precise, but it needs a new
  dependency. Deferred to the data layer, where real trading calendars are needed anyway.
- *`pd.infer_freq`*: it returns `None` on any real daily series because of holidays. Rejected.
- *A KillReason enum per stage*: this needs a union type in the ledger and loses a single place
  to look up every reason. Rejected.
- *An unbalanced panel from 1999*: it gives more history, but PC causal discovery and DRO would
  have to handle missing assets for 2–6 years. Rejected for Phase 0. It can be revisited per stage
  with an override.

## Consequences and known limitations
- **Survivorship and hindsight bias.** The universe was chosen in 2026 from ETFs that still trade.
  Any result on it is conditional on that selection.
- **GICS 2018 reshuffle.** GOOGL, META, and others moved from XLK/XLY to XLC in Sep 2018, so XLK and
  XLY are not a constant exposure across the sample. This creates a structural break that stage 2
  (decay) and stage 4 (regimes) must not mistake for signal decay.
- The daily validator accepts a series that is missing an isolated trading day, or that has an
  observation on an exchange holiday. Exact calendar alignment is the data layer's job.
- Running from a non-editable install fails at settings load. This is intentional.
