# ADR 0018: The first real-data run (signals, snapshot, report)

Status: **proposed**. Pre-registration (2026-10-31). Frozen and pushed before any stage runs on
real data. Results will go in a separate section at the end and are not edited into it. No
signal may be added, dropped or redefined after results are seen; any change is a new,
separately labelled run.

## Context
Stages 1–3 are accepted on the planted-signal benchmark (ADRs 0010, 0012, 0014), Stage 4 was
not adopted (ADR 0016), and the Stage 5 memo is accepted template-only (ADR 0017). This run takes
ETF returns, FRED market series and ALFRED macro vintages through Stages 1–3 and the memo. Real
data has no ground truth, so **the run is reported, not graded**: there is no pass rate to
compute and no gate on its outcome.

## Decisions (user, 2026-10-31)
- **A.** The candidate list below, ten signals. `vix` was dropped from the proposed eleven:
  `vrp` is built from VIX² and realised variance, so `vix`, `rv_spy` and `vrp` are
  near-redundant, and with PCMCI conditioning on two of them the third can be killed for
  collinearity rather than lack of evidence. This is the logic that dropped T10Y2Y.
- **B.** The `fred_market` source and its rule (ADR 0019), with DBAA and DAAA posting lags
  measured and the revision audit in the manifest. The lag rule was later changed by the user
  to "exact in archive" when the audit showed one-off outages (ADR 0019).
- **C.** No candidate built from Fama-French factors. With a two-month lag and full-history
  revisions they are weak predictors, and as features they cost 42 rows. They stay in the
  snapshot.
- **D.** Stage 1 keeps its validated `tau_min` = 1 (below).
- **E.** The memo states each signal's definition and citations (corpus entries checked against
  Crossref) and is revalidated on real-data-shaped runs on a fresh seed before this run (below).

## The candidates
Each is one common series at session t, tested against all 13 ETF returns. Definitions are frozen
in `invariant.data.signals.REGISTRY`; ledger ids are `sig_<id>`.

| # | id | definition at session t | source | motivated by |
|---|---|---|---|---|
| 1 | `tsmom_spy` | sum of SPY daily log returns over sessions t−251 to t−21 (12-1 month momentum) | ETF | Moskowitz, Ooi & Pedersen (2012) |
| 2 | `rv_spy` | ½·log((252/21)·Σ_{t−20..t} r²_SPY) | ETF | French, Schwert & Stambaugh (1987); Moreira & Muir (2017) |
| 3 | `vrp` | (VIX/100)²/12 − Σ_{t−20..t} r²_SPY | FRED market + ETF | Bollerslev, Tauchen & Zhou (2009) |
| 4 | `term_spread` | T10Y3M, the newest value usable at t | FRED market | Campbell (1987); Fama & French (1989) |
| 5 | `default_spread` | DBAA − DAAA, the newest values usable at t | FRED market | Keim & Stambaugh (1986); Fama & French (1989) |
| 6 | `rel_bill` | DTB3 minus its mean over the prior 252 sessions, each as usable at its session | FRED market | Ang & Bekaert (2007) |
| 7 | `ip_growth` | log INDPRO(o) − log INDPRO(o−1), both from the vintage usable at t | ALFRED | Fama (1990) |
| 8 | `payrolls_growth` | the same for PAYEMS | ALFRED | Flannery & Protopapadakis (2002) |
| 9 | `unrate_change` | UNRATE(o) − UNRATE(o−1), within one vintage | ALFRED | Boyd, Hu & Jagannathan (2005) |
| 10 | `inflation` | log CPIAUCSL(o) − log CPIAUCSL(o−12), within one vintage | ALFRED | Fama & Schwert (1977) |

Information times: ETF returns at the close of t; FRED market series under ADR 0019; ALFRED
vintages from the session after release (ADR 0004). The twelve papers are corpus entries
(`literature.yaml`, `stages: [signals]`); all twelve DOIs were checked against Crossref on
2026-10-31 (two that I had wrong, for Keim & Stambaugh and for Boyd et al., were corrected).

**Considered and dropped** (verified against the FRED/ALFRED API, 2026-10-31):

| series | reason |
|---|---|
| VIXCLS as a level (`vix`) | redundant with `vrp` and `rv_spy` (decision A) |
| ICE BofA HY and BBB OAS (BAMLH0A0HYM2, BAMLC0A4CBBB) | FRED history begins 2023-10-31 |
| NFCI | ALFRED archive begins 2011, and the index is revised |
| STLFSI4 | ALFRED archive begins 2022 |
| ICSA (initial claims) | ALFRED archive begins 2009; revised through seasonal factors |
| TEDRATE | discontinued 2022-01 |
| DTWEXBGS, DTWEXM (dollar) | DTWEXBGS begins 2006-01; DTWEXM discontinued 2019 |
| DCOILWTICO (oil) | posted in weekly batches; posting lag up to 48 sessions |
| DFF | revised; `rel_bill` covers the short rate |
| T10Y2Y, BAA10Y, AAA10Y | redundant with `term_spread` and `default_spread` |
| dividend yield, CAPE, cay | no data source in the project (backlog) |

Per-asset signals are out of scope (backlog).

## Data and settings
- **Snapshot `930504ace5464752`** (digest in its manifest; fetched 2026-10-31): ETF prices for
  the 13 tickers from 2003-06-02, French factors (not used), ALFRED vintages of INDPRO, UNRATE,
  CPIAUCSL and PAYEMS, and FRED market series VIXCLS, T10Y3M, DBAA, DAAA and DTB3 with their
  audit (ADR 0019). The snapshot is local (`data/` is not committed, and the sources' terms
  likely forbid republishing); its manifest records each file's SHA-256.
- **Config** `configs/real_0018.yaml` over `configs/base.yaml`: panel 2005-01-03 to 2025-12-31
  (5,283 sessions), execution lag 1, horizon 1, every stage setting as validated, run seed
  **20261112** (fresh), `snapshot_id: 930504ace5464752`.
- **`min_obs`, checked on the snapshot before any stage** (data only; no stage output was
  computed). Signal inputs start 401 sessions before the panel, so every lookback is complete at
  its first session. The panel has 5,282 complete rows (only the first session's return is
  missing) and no missing value after it.

  | stage | minimum | rows | margin |
  |---|---|---|---|
  | 1 (complete rows) | 500 | 5,282 | 4,782 |
  | 2 (design rows: Stage 1's rows less τ_max = 5) | 5,000 | 5,277 | 277 |
  | 3 (complete rows; the last two have no target) | 5,000 | 5,280 to 5,282 | ≥ 280 |

  If a minimum were not met, that stage kills every signal it sees with
  `general.insufficient_data`, the ledger records it, and the memo says "not decided". Neither
  the window nor any `min_obs` changes after the fact. Likewise a missing value inside the panel
  kills every signal with `general.data_validation_failed`; nothing is filled.
- **Stage 1's timing (decision D).** Stage 1 tests lags 1 to 5 of a signal against same-session
  returns. It tests causal evidence, and lag 1 is included even though a trade entered at the
  next session's close (execution lag 1) could not use it. Stage 3 enforces tradability: it
  forecasts from signals dated t and earns returns from t+2. The memo's real-data statement says
  this.

## Sensitivity run: the posting-lag assumption (pre-registered, reported only)
ADR 0019's primary rule rests on a judgement made after the lags were measured (excluding
backfills and batch posts from the pre-archive lag). Its alternative is run as well:
- **Config** `configs/real_0018_maxlag.yaml`: `configs/real_0018.yaml` with
  `data.fred_market.lag_rule: max_lag` and nothing else changed (same snapshot
  `930504ace5464752`, seed 20261112, code and stage settings; a test pins this). Under it every
  FRED market value is its current value, usable 1 + the archive's maximum posting lag sessions
  after its date, uniformly: **T10Y3M 7, DTB3 8, VIXCLS 34, DBAA and DAAA 112.** (T10Y3M's
  maximum lag is 6, a backfill included, so 7 is its value under this rule; the 5 quoted
  elsewhere is the primary rule's.) ETF and ALFRED inputs are identical in both runs; only
  `vrp`, `term_spread`, `default_spread` and `rel_bill` change.
- **Run** with `uv run invariant run --config configs/real_0018_maxlag.yaml` from the same
  worktree and pinned snapshot, then attached with `uv run invariant sensitivity --primary
  <primary run> --sensitivity <sensitivity run>`. The command checks that both runs are
  completed and identical but for the lag rule (snapshot, seed, code, config), copies the
  sensitivity run's manifest and hash-chained ledger into `<primary>/sensitivity/`, and rewrites
  the primary memo.
- **It is reported, not decisive.** The primary run's verdicts stand. A signal's outcome
  "differs" if its verdict or deciding stage differs between the runs.
- **If any outcome differs, the memo and the README name those signals and say that their
  result depends on the posting-lag assumption.** The memo's mandatory statement
  `lag_sensitivity` does this; `lag_sensitivity_agrees` states that no outcome differs. The
  verifier recomputes which one applies, and the exact set of signals named, from the two raw
  ledgers (V6). These statements are covered by unit tests
  (`tests/unit/test_memo_sensitivity.py`), not by the memo revalidation below, whose runs have
  no sensitivity run.
- The sensitivity run also writes its own memo, which carries neither statement.

## Limits stated before the run
- **Persistence.** Lag-1 autocorrelations on the panel: `term_spread` .9987, `inflation` .9985,
  `default_spread` .9984, `rel_bill` .9982, `tsmom_spy` .9937, `rv_spy` .9911, `ip_growth` .9642,
  `unrate_change` .9502, `payrolls_growth` .9478, `vrp` .9204. Stage 1's size was validated up to
  ρ = .99 at T = 6000 (ADR 0009, E6) and in the T(1−ρ) = 5 Stambaugh diagnostic, where the
  decision unit held (.029–.039) but the lag-1 test alone did not (.082). Here T(1−ρ) is about
  7 for `term_spread`, 8 to 10 for `inflation`, `default_spread` and `rel_bill`, and 33 and 47
  for `tsmom_spy` and `rv_spy`: closer to that diagnostic's regime than to the validated
  T = 6000, ρ = .99 case (60), but not below it.
- **Confounding.** Stage 1 conditions only on the observed variables (the ten signals and 13
  returns). A common cause outside them is not removed (ADR 0010).
- **Data.** Survivorship and hindsight in the ETF universe (ADR 0001, 0004); the FRED market
  series' pre-archive assumption (ADR 0019); macro values are vintages but the macro *calendar*
  (which months exist) is known.
- **Multiplicity.** Ten candidates are corrected for within Stage 1 (Bonferroni over 65 tests
  per signal, BH across signals) and the DSR covers Stage 3's four arms. The list itself was
  chosen from a literature that searched many predictors on overlapping data (Harvey, Liu & Zhu
  2016), which no correction here undoes.

## Expectation, stated in advance
**Most or all candidates will be killed, mostly at Stage 1.** The reasons:
- At realistic strength (IC .02), Stage 1 passed a true edge in .140 [.112, .173] of benchmark
  runs at T = 6000 (ADR 0010); at T ≈ 5,280 its power is lower.
- Stage 3's kill test has power at most .030 at realistic strength (ADR 0013).
- Out of sample, most of these predictors did not beat the historical mean (Welch & Goyal 2008),
  and published predictability shrinks after publication (McLean & Pontiff 2016); this sample
  starts after most of the motivating papers' samples end.

A signal that does pass every stage would be, by the memo's rule, a "candidate for paper
trading", and must be read with the known false-pass paths: confounding, persistence, and
passengers in a joint Stage 3 verdict (a signal with no edge passed all three stages in .015
[.010, .023] of benchmark runs, ADR 0014). This expectation is a statement, not a gate.

## Outputs (the report's contents, fixed now)
1. **The full ledger** (`kill_ledger.jsonl`, hash-chained), and a table of every signal: the
   stage that decided it, verdict, reason code and its numbers: Stage 1 family p, BH q, best
   asset and lag, partial correlation with its interval; Stage 2 decline p (if reached); Stage 3
   (if reached) every arm's gross and net Sharpe, LW p, DSR, and diagnostic B with its interval.
2. **The Stage 5 memo** (`memo.md`, `memo.json`), real-data variant, written only if the
   verifier accepts it: the signals table with each definition and its citations, and the
   real-data `data_source` statement (ADR 0004's limits, ADR 0019's assumption, Stage 1's
   timing, and that quoted rates are benchmark validation rates, not properties of this
   strategy).
3. **The stress diagnostic** (Stage 3's high-volatility Sharpe with its interval), **only if
   Stage 3 runs**, that is, if some signal passes Stages 1 and 2. If none does there is no
   portfolio and no stress diagnostic, and the report says so.
4. **The snapshot summary**: ETF coverage, the FRED market audit (ADR 0019's table), ALFRED
   missing values.
5. **The sensitivity run** (above): its ledger beside the primary's, signal by signal, and the
   memo's sensitivity statement.
6. **A README section** describing what survived and what was killed at which stage, with the
   numbers behind each decision, framed as a description of what this pipeline did on this
   data, **not a trading recommendation**, restating the expectation above, and naming any
   signal whose outcome depends on the posting-lag assumption.

## Memo revalidation on real-data-shaped runs (pre-registered 2026-10-31)
ADR 0017's validation runs were all synthetic, so the memo's real-data path (the signals table
with definitions and citations, the real-data `data_source` statement) has only unit tests.
- **Harness** as ADR 0017 (`scripts/memo_validation.py`), config
  `configs/memo_validation_real.yaml`: the run set, coverage rule, 12 corruptions per class per
  run and the gates of the accepted deviation-1 validation, with **every preset
  real-data-shaped** (`run_benchmark_pipeline(real_shaped=True)`: the panel's `syn_*` columns
  renamed `sig_*` before Stage 1, and a `signals.json` of stand-in definitions with digits and
  rotating source kinds) and a **fresh root seed 20261113**. A test pins that the config differs
  from the deviation-1 config only in the seed and the flag.
- **Gates** G1–G4 of ADR 0017, unchanged: G1 every template memo accepted; G2 at least 500
  instances and zero misses per class K1–K9; G3 zero misses on the gated entries of hand sets
  (a), (b) and (c), as a regression check (they are written against the synthetic fixtures); G4
  determinism and GroundTruth independence. Pass probabilities as ADR 0017: a correct verifier
  passes with probability 1; one with per-instance miss rate p passes G2 for a class with
  probability (1 − p)^576 (.750 at p = .0005, .056 at .005).
- **If any gate fails, the real-data run does not proceed**; the design returns to review.
- The arm runs on one tiny seed in the fast suite
  (`tests/integration/test_memo_validation_tasks.py`).

## Process
1. This ADR, ADR 0019 and the code are committed and pushed.
2. The memo revalidation runs from a git worktree at that commit. Its results come to the user
   before anything else.
3. If it passes, `uv run invariant run --config configs/real_0018.yaml` (the primary run), then
   `uv run invariant run --config configs/real_0018_maxlag.yaml` (the sensitivity run), then
   `uv run invariant sensitivity`, all from a git worktree at the same commit with a clean tree.
   `invariant run` refuses any snapshot but `930504ace5464752`.
4. Results come to the user before they are written into this ADR or the README.

## Alternatives considered
- **Per-asset signals** (each ETF's own momentum): a different hypothesis and many more tests;
  backlog.
- **Grading the run** against the literature's reported effects: they are in-sample estimates
  on other periods, not ground truth. Rejected.
- **A longer sample** (pre-2005) to raise power: GLD's inception binds the balanced panel
  (`configs/base.yaml`), and the base window is the one every stage was validated for.
