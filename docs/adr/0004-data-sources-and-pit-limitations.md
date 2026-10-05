# ADR 0004: Data sources and their point-in-time limitations

Status: accepted (Phase 0)

## Context
Non-negotiable #1 says a value used at time t may use only information available at t. Phase 0
uses three free sources, and each one is point-in-time to a different degree. This ADR records
what each source does and does not guarantee, so later stages don't treat all three as equally
safe. It also fixes one rule that the point-in-time assembly (`pit.py`, next task) must follow.

All network access goes through narrow protocols in `data/sources/clients.py` (`HttpClient`,
`PriceClient`, `AlfredClient`). Parsing and validation are pure functions tested offline. Real
fetches live only in `tests/integration` and are marked `@pytest.mark.slow`.

## Decisions

**ETF prices (Yahoo via yfinance).** We fetch `Close`, `Adj Close` and `Volume` with
`auto_adjust=False`, `repair=False` and `actions=False`, and store them as a long table. Coverage is
checked against the *exact* XNYS session calendar from `exchange-calendars`. Comparing against the
union of the tickers' own dates would miss a session absent from every ticker. Missing sessions,
interior holes and off-calendar rows are listed per ticker in the snapshot manifest and logged.
Nothing is filled. A ticker with no data at all is an error. Log returns are computed only
between adjacent sessions, so a gap makes two returns NaN rather than one stretched return.
The session list itself is stored in the snapshot because calendars are code and change with
library updates. Note that exchange-calendars defaults to the last 20 years, so we always build
it with explicit bounds.

- *Not point-in-time in levels.* Yahoo back-adjusts history for every later split and dividend.
  An adjusted price level therefore contains future information. Returns between adjacent
  sessions don't, because the multiplicative adjustment factor cancels. Rule: never use adjusted
  price levels as features.
- Yahoo revises and rounds history silently. The snapshot is the only record of what we fetched.
- Survivorship and hindsight in the choice of universe (see ADR 0001).

**Fama-French 5 factors and momentum (Ken French Data Library).** We download the daily zips with
httpx (URLs verified on 2026-10-05 and kept in `configs/base.yaml`) and parse them strictly: the
header must match exactly, and the parser fails on any layout change. Values are converted from
percent to decimals, and the `-99.99`/`-999` sentinels become NaN and are counted. The two files
are outer-joined, and dates present in only one file are counted. The manifest records each
zip's SHA-256 and the CRSP build string ("202608"), which is the only identifier of the library
revision we fetched.

- *Not point-in-time.* Each CRSP update rebuilds the full history (data corrections, delisting
  returns, re-formed portfolios and breakpoints). Releases also lag the data by weeks, and only
  the current revision can be downloaded.
- Consequence: factor returns are fine for contemporaneous risk adjustment and attribution. Using
  them as predictors needs at least a publication lag of one month or more, plus an explicit
  caveat about revision noise. A backtest using them cannot be replicated exactly with a later
  download, only from the snapshot.
- Daily rows before 1952 include Saturday sessions, so the weekday-only daily check is not
  applied to this table.

**Macro series (ALFRED).** We call the FRED `series/observations` endpoint directly with httpx,
which replaces the earlier fredapi plan. fredapi dropped `realtime_end`, could not page past
100k rows, and used urllib without a timeout. We request the full real-time period
(`realtime_start=1776-07-04`, `realtime_end=9999-12-31`), which returns every vintage, and page
with `limit`/`offset`. After paging, the final row count must equal the `count` the API reported,
`count` must not change between pages, and no `(date, realtime_start)` row may repeat. If any of
these fails, the fetch raises. Each row stores `(series_id, observation_date, realtime_start,
realtime_end, value)`. FRED's `.` marker becomes a NaN row and is counted, not dropped. For each
observation, vintage periods must be disjoint. Starter series: INDPRO, UNRATE, CPIAUCSL, PAYEMS.
API keys never appear in error messages or logs, and the unredacted error is not kept as
exception context.

- **`realtime_start` has no time of day.** Most releases come out at 08:30 ET, but not all, and
  the API doesn't say which. **`pit.py` treats a vintage as usable from the next trading session
  after `realtime_start`, by default.** This is conservative. It costs at most one session of
  timeliness and cannot leak. Same-day use would need a per-release timestamp source and its own
  ADR.
- Observations dated before a series' first archived vintage (INDPRO 1927-01-26, UNRATE
  1960-03-15, CPIAUCSL 1972-07-21, PAYEMS 1955-05-06) carry that vintage date. That is later than
  the true publication date, so it's conservative.
- Real data has missing-value rows. For example, UNRATE has one `.` vintage around the 2025 US
  government shutdown and INDPRO has 600. They are kept and reported, never interpolated.

## Alternatives considered
- *yfinance `auto_adjust=True`*: this discards the unadjusted close, which we need to audit
  adjustments. Rejected.
- *Checking coverage against the union of returned dates*: this misses sessions absent from every
  ticker. Rejected in favour of the exchange calendar (new dependency: `exchange-calendars`).
- *fredapi*: see above. Removed from the dependencies.
- *Raising at FRED's 100k row limit instead of paging*: future series would fail for no good
  reason. Paging plus the exact count check is both safe and complete.
- *pandas-datareader for French data*: one more dependency over the same files, with a looser
  parser. Rejected.

## Consequences
- Every source's weaknesses are explicit in its module docstring and here. Later stages must cite
  them when using these data, especially French factors as predictors.
- `pit.py` is bound to the next-session rule above.
- `PipelineConfig` gained a required `data` section, so run ids change for the same experiment
  (the golden run id was re-pinned). Canonicalization did not change, so `HASH_SCHEMA_VERSION`
  stays 1.
