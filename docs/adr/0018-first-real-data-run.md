# ADR 0018: The first real-data run (signals, snapshot, report)

Status: **accepted** (2026-11-02): the memo revalidation passed every gate and both runs are
reported below. Pre-registration (2026-10-31). Frozen and pushed before any stage runs on
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

## Results, memo revalidation (recorded 2026-11-02, after the run)
Run from a worktree at the pushed pre-registration commit ae65781 (`git_dirty: false`), config
`configs/memo_validation_real.yaml`, root seed 20261113, median 32 s per run. No post-launch
fix. Report and hand-set results: `docs/results/0018-memo-revalidation/`.

48 real-data-shaped runs (8 per preset; the coverage rule added none). Categories: Stage 1 kill
48, Stage 2 kill 30, Stage 3 kill 7, multi-member pass 12; recommendations: candidate 22,
against 20, not decided 6.

| gate | result |
|---|---|
| G1 | **pass**: 48 of 48 template memos accepted |
| G2 | **pass**: K1–K9 each 0 misses in 576 (upper bound .0052) |
| G3 | **pass**: set (a) 0 of 96, set (b) 0 of 126, set (c) 0 of 146 gated entries missed |
| G4 | **pass**: 48 of 48 deterministic and GroundTruth-independent |

**The real-data memo path is revalidated; the run proceeded.** Reported, not gated: every benign
narrative sentence of sets (a) and (b) (25) is rejected, by design in template-only mode; set
(b)'s entailment entries 0 of 2 missed. The report's count of candidate memos with a passed
edgeless signal reads 22 of 22. That count is wrong: it is a harness bug in a reported-only
metric, recorded below; no gate reads it.

## Results, the real-data run (recorded 2026-11-02, after the run)
Both runs were made from the same worktree at ae65781 with a clean tree, on snapshot
`930504ace5464752`, seed 20261112, then attached with `invariant sensitivity`. No post-launch
fix; neither run was repeated.

| run | lag rule | run id | ledger head (SHA-256) | archive |
|---|---|---|---|---|
| primary | `exact_in_archive` | `f4e53b224b61ad35` | `ca27a4c9…` | `docs/results/0018-real-data-run-primary/` |
| sensitivity | `max_lag` | `75f1372b65ff89f7` | `053bcc61…` | `docs/results/0018-real-data-run-sensitivity/` |

Panel: 5,282 rows, 2005-01-04 to 2025-12-31; Stage 1 tested 5,272 rows (τ_max = 5). Both memos
were written and accepted by the verifier, and re-verified from the archived run directories.

**Every candidate was killed at Stage 1, `causal.no_significant_link`, in both runs.** Stages 2
and 3 saw no signal (`not_reached`: all ten), so there is no portfolio, no Sharpe, no DSR and no
stress diagnostic. The memo's recommendation is "against trading": no signal survived every
kill stage. This is the outcome stated in advance.

Stage 1 (output 1). Family p is Bonferroni over the signal's 65 tests (13 assets × 5 lags), q is
BH across the ten signals; the verdict is q against α = .05. Best link and r are descriptive only
(below).

| signal | family p | BH q | best link | partial r [95% interval] | sensitivity: family p, q, best link, r |
|---|---|---|---|---|---|
| `payrolls_growth` | .058 | .576 | XLI, lag 1 | −.050 [−.072, −.007] | .072, .716, XLI lag 1, −.048 |
| `unrate_change` | .403 | 1 | XLI, lag 1 | −.039 [−.064, −.016] | .467, 1, XLI lag 1, −.037 |
| `inflation` | .588 | 1 | XLK, lag 4 | −.048 [−.080, −.013] | .652, 1, XLV lag 1, −.031 |
| `ip_growth` | 1 | 1 | XLI, lag 3 | .040 [−.000, .070] | 1, 1, XLP lag 3, .039 |
| `rel_bill` | 1 | 1 | XLF, lag 3 | .041 [.005, .076] | 1, 1, XLF lag 3, .037 |
| `rv_spy` | 1 | 1 | XLK, lag 3 | .032 [.002, .062] | 1, 1, XLB lag 5, .027 |
| `term_spread` | 1 | 1 | GLD, lag 3 | −.026 [−.050, −.003] | 1, 1, GLD lag 3, +.044 |
| `vrp` | 1 | 1 | TLT, lag 3 | −.041 [−.077, −.011] | .507, 1, IEF lag 5, −.050 |
| `default_spread` | 1 | 1 | XLU, lag 3 | −.023 [−.056, .013] | 1, 1, GLD lag 5, +.019 |
| `tsmom_spy` | 1 | 1 | XLF, lag 4 | −.017 [−.041, .009] | 1, 1, XLU lag 2, +.014 |

**The bracketed intervals are not evidence.** Each is the interval of the best of 65 tests,
after selection, and Stage 1's bootstrap intervals already undercovered on the benchmark
(ADR 0010). That most exclude zero is what selection produces. For scale (an illustration, not
a test): the largest |r| of 65 independent null correlations at n = 5,272 has 5th, 50th and 95th
percentiles .028, .035 and .046; the ten best |r| here run from .017 to .050. The tests are
correlated, so this is only a guide; the family p is the calibrated statement.

**Sensitivity (output 5): no outcome differs.** Every signal has the same verdict and deciding
stage under both lag rules, so the memo carries `lag_sensitivity_agrees` and no signal's result
depends on the posting-lag assumption. Below the verdicts the runs differ even for the six
signals whose inputs are identical in both (ETF and ALFRED): Stage 1 conditions on every signal,
so changing four of them changes every conditioning set. `term_spread`'s best link is GLD at lag
3 in both, with r of opposite sign (−.026, +.044); `default_spread` and `tsmom_spy` also flip
sign at a different best link. This is another reason to read the best links as noise.

**What the null means.** It is partly a statement about power. At realistic strength (IC .02)
Stage 1 passed a true edge in .140 [.112, .173] of benchmark runs at T = 6000 (ADR 0010), and
this panel is shorter. Each signal's evidence is Bonferroni-corrected over its 65 tests, so a
single link must clear a high bar, and a weak edge shared by many assets gains nothing from
appearing in several of them (the backlog item below). A
kill here says the edge was not detectable by this test at this sample size, not that there is
no effect. The run does not reach Stages 2 and 3, so it says nothing about decay or
tradability.

**Snapshot (output 4)** (`docs/results/0018-snapshot/manifest.json`; derived statistics only, no
values). ETF: 12 tickers have all 5,684 sessions from 2003-06-02 with none missing; GLD begins
2004-11-18 (371 sessions before inception), before the panel start. FRED market: ADR 0019's
table (first vintages VIXCLS 2010-11-22, T10Y3M 2014-01-27, DBAA and DAAA 2014-04-02, DTB3
2005-06-28; pre-archive lags 3, 5, 5, 5, 8; DBAA and DAAA 110 batch-posted observations; revised
observations 4, 2, 12, 13, 0). ALFRED vintage rows with missing values: INDPRO 600, UNRATE 1,
CPIAUCSL 259, PAYEMS 0; the panel has no missing value after its first row.

## Harness fix: the edgeless-pass count (reported-only metric, 2026-11-02)
- **Cause.** `run_task` looked up each passed signal's planted edge by GroundTruth id
  (`s_real`) with a default of "no edge". In a real-data-shaped run the ledger names the signal
  `sig_s_real`, so every lookup missed, and every candidate memo was counted as having an
  edgeless pass (22 of 22).
- **Fix** (the commit that adds this section, after ae65781). The id mapping is one function,
  `benchmark.real_shaped_id`, used by the pipeline's rename and by the new
  `validation.passed_with_true_edge`, which raises on an id GroundTruth does not know instead of
  defaulting. Unit tests cover both id forms and the unknown-id error.
- **Effect.** The metric is reported, not gated: G1–G4, the verifier, the memo and both
  real-data runs are unaffected, and nothing was rerun. The count was recomputed from the
  archived `records.jsonl` with each run's GroundTruth regenerated from its seed: **2 of 22**
  candidate memos had a passed edgeless signal, both `hidden_confounder` runs in which `s_conf`
  passed (indices 1 and 2). One task (`hidden_confounder`, index 1) was rerun with the fix and
  reproduced its recorded run id `ad080fb9dfdab14c`, with flags (False, True) where the archived
  record has (False, False). The archived report is left as written.

## Backlog (added 2026-11-02)
- **A pooled cross-asset Stage 1 decision unit**: one test of a signal's link to all assets
  jointly, with more power for an effect common across assets than Bonferroni over 65 tests. It
  is to be chosen by its own pre-registered study (size under persistence, power on the
  benchmark, and the hidden-confounder pass rate), not adopted because this run found nothing.
- Per-asset signals (above).
