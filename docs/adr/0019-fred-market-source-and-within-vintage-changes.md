# ADR 0019: FRED market series, within-vintage macro changes, and the signal panel

Status: **proposed** (2026-10-31). Data methods for the first real-data run (ADR 0018). No
stage has run on real data.

## Context
ADR 0018 freezes ten common time-series signals. Building them point-in-time needs three
things the data layer did not have:

1. **Daily market series from FRED** (VIX, the 10y−3m spread, Moody's Baa and Aaa yields, the
   3-month bill). Measured against the FRED/ALFRED API on 2026-10-31, **every one of them has
   an ALFRED archive that begins after the 2005 sample start**:

   | series | FRED history | first ALFRED vintage |
   |---|---|---|
   | DTB3 | 1954 | 2005-06-28 |
   | VIXCLS | 1990 | 2010-11-22 |
   | T10Y3M | 1982 | 2014-01-27 |
   | DBAA, DAAA | 1986, 1983 | 2014-04-02 |

   Under ADR 0004's rule an observation is usable only from its first archived vintage, so these
   series would be unusable for 5 to 9 years of the sample. The ALFRED fetcher also cannot
   request them: FRED refuses a request spanning more than 2000 vintage dates.
2. **Changes of monthly macro series.** ADR 0006 deferred them: differencing the as-of level
   subtracts a value from an older vintage.
3. **Lookbacks without losing the panel's first year.** A 252-session window computed inside
   the panel leaves 5,028 complete rows of 5,283, 28 above Stage 2 and 3's `min_obs` of 5,000
   (and 4,986 with French factors as features).

## Decision

### 1. A `fred_market` source (`data/sources/fred_market.py`)
For each series and for observations dated from the warm-up start to the sample end, the source
stores the **current values**, the **archived vintages** (requested one calendar year of real
time at a time, merged back into ALFRED's row format) and a **per-observation audit**.

**Rule (user decision, 2026-10-31: "exact in archive").** Each series is split at its first
vintage V.
- **Observations dated on or after V** use their archived vintages under ADR 0004's rule
  exactly: usable from the session after FRED first posted a value, at the value in force.
- **Observations dated before V** have no vintage. They use the current value, usable
  `pre_archive_lag` sessions after their date, and are never superseded.
- `pre_archive_lag` = 1 + the largest **regular** posting lag in the archive, where the lag of
  an observation dated d first posted with a value on r is the number of sessions in (d, r].
  Two kinds of observation are not regular postings and are excluded: **backfilled** (first
  published as missing, `.`, and filled later) and **batch-posted** (first posted on a date that
  first-posted `BATCH_MIN` = 10 or more observations, an outage).
- A value FRED posted before its own date (a non-trading day filled with the previous close) has
  lag 0 and is flagged.

**How the rule was chosen.** I first proposed, and the user approved, a uniform lag of
1 + the archive's maximum posting lag. The full audit then showed that maximum is set by
one-off events, not by FRED's posting:

| series | lags ≤ 2 sessions | max lag | what sets the max |
|---|---|---|---|
| VIXCLS | 1.000 | 33 | 2021-04-16 published as missing, filled six weeks later |
| DBAA, DAAA | .962 | 111 | 2016-10-11 to 2017-03-21 absent from FRED, posted together on 2017-03-22 |
| DTB3 | .966 | 7 | weekly batches in 2005, when the archive began |
| T10Y3M | .998 | 6 | 2017-04-14 published as missing, filled later |

The uniform rule would have made `default_spread` five months stale and `vrp` 34 sessions stale
for the whole sample. Quantiles do not help (DBAA's 99th percentile is 81, the outage being 4% of
observations). The user chose the rule above. **The two exclusions were defined after these lags
were measured, and before any stage ran on real data.** Under it the outage and the backfills are
handled exactly where they are known: the last posted value is carried, which is what was
knowable.

**Resulting lags** (snapshot `930504ace5464752`, fetched 2026-10-31):

| series | first vintage | pre-archive lag | max lag | backfilled | batch-posted | revised (max) |
|---|---|---|---|---|---|---|
| VIXCLS | 2010-11-22 | 3 | 33 | 2 | 0 | 4 (0.75) |
| T10Y3M | 2014-01-27 | 5 | 6 | 1 | 0 | 2 (0.01) |
| DBAA | 2014-04-02 | 5 | 111 | 2 | 110 | 12 (0.21) |
| DAAA | 2014-04-02 | 5 | 111 | 2 | 110 | 13 (0.23) |
| DTB3 | 2005-06-28 | 8 | 7 | 2 | 0 | 0 |

No archived observation's current value differs from its last archived value.

**The alternative, kept as an option.** `data.fred_market.lag_rule: max_lag` applies the rule
first approved instead: current values throughout, each usable 1 + `max_posting_lag` sessions
after its date (T10Y3M 7, DTB3 8, VIXCLS 34, DBAA and DAAA 112 on this snapshot). It is
implemented as the pre-archive branch with no archive (`market_asof` with a first vintage of
9999-12-31). ADR 0018 runs it as a reported-only sensitivity run on the same snapshot.

### 2. Within-vintage changes (`pit.within_vintage_change`)
At session t the usable vintage rows (ADR 0006's as-of rule) are one vintage. The change is
f(x_t(o)) − f(x_t(o − k months)) with f = log or the identity, where o is the newest observation
such that both o and o − k have a value **in that vintage**. Both values come from the same
vintage (Croushore & Stark 2001). `obs_date` records o.

### 3. The signal panel (`data/signals.py`)
- Inputs are fetched from `signals.warmup_start` (2003-06-02, 401 sessions before the panel), so
  every lookback is complete at the first panel session; columns are then cut to `dates`.
- The panel holds only `ret_*` and `sig_*` columns: Stage 1 puts every non-return feature into
  its graph as a conditioning variable, so raw macro levels or French factors must not be there.
- Rolling windows are explicit sums over fixed past sessions, so a value at t depends only on
  its window, bit for bit; a missing return in a window makes the signal missing.
- Signal definitions are frozen in `REGISTRY` (ADR 0018).

### 4. Config and run identity
`data.fred_market` (with `lag_rule`), `signals` and `snapshot_id` are new optional config
sections, absent for the
benchmark. Adding them changed every run hash; the golden run id was re-pinned to
`f6b81b4c6108313f`. Canonicalization is unchanged, so `HASH_SCHEMA_VERSION` stays 1. The ETF
fetch starts at `PipelineConfig.fetch_start` (the warm-up start when signals are configured);
GLD's 371 sessions before its 2004-11-18 inception are reported as missing, all before the panel.

## Tests
- `tests/unit/test_fred_market.py`: chunk merging (a period split at a year boundary is one
  period; missing merges with missing; an open end), the audit (lags counted in sessions across a
  holiday, backfills, batches at and one below `BATCH_MIN`, early posts, revisions), the
  pre-archive lag, the client's one-year real-time windows, request spacing and key redaction.
- `tests/leakage/test_signal_inputs.py`:
  - `market_asof`: perturbing everything usable after t leaves values at ≤ t bit-identical,
    including the current (final) values of archived observations; it equals a brute-force
    oracle computed from release histories; `hypothesis.find` exposes three planted leaks (a
    pre-archive value one session early, same-day archived postings, final values from the
    first posting on).
  - `within_vintage_change`: the same future-perturbation property; equality with an oracle
    from release histories; `find` exposes two planted leaks (same-day releases, the base value
    from the final vintage).
  - Mixed-vintage differencing uses no future data, so no look-ahead property catches it. The
    oracle comparison does (`find` exposes it), and a hand-built revision shows the as-of
    difference log(104/101) where the within-vintage change is log(104/103).
  - Rolling windows ignore returns after t, a window reaching t+1 is caught, and every
    price- and market-based registry signal at t ignores returns after t.
- `tests/unit/test_signals.py`: exact values on hand-computable inputs, the frozen registry, and
  an end-to-end signal panel from a faked snapshot (columns, completeness at the panel start, a
  within-vintage change with revisions).

## Consequences and limitations
- **Before each archive begins** (DTB3 2005-06, VIX 2010-11, T10Y3M 2014-01, Moody's 2014-04),
  point-in-time correctness rests on the assumption that FRED posted as regularly as it did
  later; an outage before the archive would not be seen. Before the archive the current value is
  used, so a revision before the archive would be used in its revised form. Archived revisions
  were few and small (above).
- The exclusions are a judgement made after seeing the lags (above).
- Macro changes are step functions on the daily panel, changing only on release days, and were
  as much as 124 days stale around delayed releases. `default_spread` was 166 days stale during
  the Moody's outage, as it was in real time.
- The memo's real-data statement names these assumptions (ADR 0018).

## Alternatives considered
- **ALFRED's rule as is:** the market signals would be missing for 5 to 9 years. Rejected.
- **Current values with a uniform lag set by the archive maximum:** approved first, then replaced
  (above): it makes two signals months stale to cover one outage and one backfill.
- **A quantile lag:** leaks wherever posting was later than the quantile, and DBAA's outage
  still dominates its high quantiles. Rejected.
- **Another source for these series (e.g. Yahoo `^VIX`, `^TNX`):** no vintages at all, and a
  second price source to audit. Rejected.
