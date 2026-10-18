# ADR 0010: Stage 1, the causal screen

Status: accepted (2026-10-18). The M = 500 validation below was pre-registered and pushed before
its run (commit `ba43129`). Every acceptance criterion passed (final section).

## Context
ADR 0009 chose the method: PCMCI with `ParCorrHC3` as the CI test and `hybrid` as the decision
test, decided over a Bonferroni family of lags. This ADR turns that into a pipeline stage. It
covers what Stage 1 reads, how it decides, what it records, which of its assumptions market data
violates, and how it was validated on the planted benchmark before touching real data.

The spec (docs/invariant-spec.html) describes Stage 1 as PC (causal-learn) cross-checked with
NOTEARS, or a dowhy intervention test. **This stage deviates from the spec.** ADR 0009's three
pre-registered studies measured size and power for lagged, heteroskedastic financial series.
Neither NOTEARS (no lag structure, no finite-sample error control) nor an unvalidated PC variant
has that evidence. A second method would be added only after the same kind of study.

## Decision

### Inputs
- The PIT panel's **feature columns** (`syn_*`, `ff_*`, `macro_*`) and **returns** (`ret_*`).
  Forward targets live in `PitPanel.targets` and are never read. A `fwd_*` column that reached
  the features would be ignored too (tests/leakage/test_causal_inputs.py).
- Leading rows with any missing value are dropped (`ret_*` is missing on the first session). A
  missing value after that is `general.data_validation_failed` for every signal: dropping an
  interior row would make two non-adjacent sessions look one lag apart.
- Fewer than `min_obs` complete rows is `general.insufficient_data` for every signal.
  `min_obs = 500` is the shortest T at which the decision unit has been measured (ADR 0009's
  T = 500 Stambaugh diagnostic: .039 for `hc3|hybrid`).
- **Candidates**: `causal.signals`, or every feature when it is null. Features that are not
  candidates stay in the graph as conditioning variables. In the benchmark, `syn_z` is a
  conditioning variable: Z is a factor, not a candidate signal.
- **Signal ids**: the column name, lower-cased, without the `syn_` prefix (`syn_s_real` ->
  `s_real`, `macro_INDPRO` -> `macro_indpro`). Colliding ids are rejected.

### Method (`invariant.stages.causal.stage`)
1. One PCMCI run over all features and returns. Settings: tau_min 1, `tau_max` 5, `pc_alpha`
   0.2, CI test `ParCorrHC3` (analytic p-values).
2. For each candidate X, each return R^a and each lag in 1..tau_max, the `hybrid` test of
   `X_{t-lag} -> R^a_t`. The MCI test of the same link comes from the same run and is recorded
   as a diagnostic (ADR 0009, follow-up 1).
3. **Within a signal**: the family p-value is Bonferroni over its `tau_max x n_assets` tests,
   `min(1, K min p)`. Bonferroni is valid under any dependence between lags and assets.
4. **Across signals**: the family p-values are FDR-adjusted with BH (`causal.fdr: bh`), or with
   BY (`by`).
5. **Verdict**:
   - kill `causal.no_significant_link` if the family p is >= alpha (0.05);
   - otherwise kill `causal.not_significant_after_correction` if the q-value is >= alpha;
   - otherwise pass.
   A tie at alpha is a kill. The two reason codes separate "nothing there" from "something there
   that this many candidates cannot vouch for".
6. **Effect size**: a 95% circular block-bootstrap interval (Politis & Romano 1992; 999
   replicates, block `ceil(n^(1/3))`) of the partial correlation at the best (asset, lag). The
   conditioning set is held fixed. This interval is descriptive and never decides.

### Outputs
- `CausalResult` (frozen pydantic, written to `<run_dir>/causal_result.json`). It holds the
  config echo, the tigramite version, n_obs and rows tested, the PC1 parents of every variable,
  and the MCI graph at p <= alpha (uncorrected, descriptive). It also holds one
  `SignalCausalLink` per signal with:
  - per-(asset, lag) hybrid p and r, and MCI p and r;
  - the family p, the MCI family p and the q-value;
  - the best link with its conditioning set, and the bootstrap interval;
  - the verdict and reason.
- **One kill-ledger entry per signal**, pass or kill. Its metrics are flat, finite scalars:
  - `family_p`, `q`, `mci_family_p`, `n_family_tests`, `n_fdr_signals`, `best_asset`,
    `best_lag`, `r_best` and its CI;
  - `p.<asset>.lag<k>`, `r.<asset>.lag<k>` and `p_mci.<asset>.lag<k>`;
  - the config values the decision used.
- **RNG**: the run's `RngFactory` -> child `causal` -> stream `ci_test` (HC3 is analytic, so the
  seed is unused) and child `bootstrap` -> one generator per signal id. The p-values do not
  depend on the seed. The bootstrap intervals do, and are reproducible from it.
- **GroundTruth v3** adds `expected_t_hybrid`: the population partial correlation of `S_{t-1}`
  and `R_t` given the *oracle* hybrid set, times `sqrt(n_obs - 1)`. The oracle set is the true
  observed parents of R other than S, plus S's observed non-own parents shifted by one lag.
  `low_power` and the generator's `benchmark_low_power` warning are now keyed to it.
  - On the default preset it is 4.42 for s_real and 4.09 for s_decay (`expected_t`: 4.43,
    4.09). Hybrid keeps almost all of the predictive regression's power, unlike MCI (ADR 0009,
    finding 4).
  - It is 0 for s_conf and s_noise.
  - It is 3.47 for s_conf when Z is hidden: the bias no observed adjustment set removes.
  - A slow test checks it against a 200,000-step sample.

### Config
`causal:` in configs/base.yaml (`CausalConfig`, frozen, `extra="forbid"`).
- `ci_test` admits only `parcorr_hc3`. Any other test needs its own size study first.
- `decision` admits `hybrid | mci`. `parents_r` is excluded: it falsely passes S_conf at .446.
- Adding the block changed every run hash. The golden run id in tests/unit/test_run_hash.py
  was re-pinned.

## Assumptions and where they are violated
| Assumption | Violation | Consequence / what is measured |
|---|---|---|
| Causal sufficiency | A common cause missing from the panel (benchmark `hidden_confounder`; in markets, almost certain) | The confounded signal passes. The validation measures the rate on `hidden_confounder`. GroundTruth predicts it (`expected_t_hybrid` 3.47). Stage 1 cannot rule out an unobserved confounder, and the memo must say so. |
| Linear mean effects | Variance effects, threshold or sign-switching effects | ParCorr cannot see them. The benchmark's GARCH edges are, correctly, invisible. |
| Stationary causal structure | s_decay's coefficient decays; real markets have regimes | The test reports an average effect over the sample. A decaying edge can still pass (it should: Stage 2 owns decay). |
| Faithfulness | Exact cancellations | Not testable from data; assumed. |
| No contemporaneous effects | Same-day cross-asset effects | Lag 0 is never tested, so they are neither found nor used. |
| PC1 selection independent of the test | The same data both selects the conditioning set and tests on it | Post-selection inference (ADR 0009, study 1 finding 3: z -> R oversized for several CI tests). Not corrected; measured instead. `hc3|hybrid` passed all 11 null gates of study 3. |
| HC3 asymptotics | Small T, near-unit-root regressors (Stambaugh) | The decision unit was .039 at T = 500 with rho .99 (ADR 0009). Hence `min_obs = 500`. |
| BH under dependence | Every signal is tested against the same returns; two-sided correlated tests are not guaranteed PRDS | BH's FDR guarantee is not formally established here. BY is available (`fdr: by`) and valid under any dependence. The validation reports the realised false-pass rate. |
| Lag identification | PC1 misses the true lag at moderate power | `best_lag` is "where evidence is strongest", not the true horizon (ADR 0009 finding 5). Documented on the contract. |
| Block bootstrap | Persistent regressors (rho .9 to .99) have dependence beyond the block; the link and conditioning set are chosen on the same sample | The interval is somewhat narrow: 90% nominal gave .86 coverage on an AR(1) rho .9 regressor (tests/unit/test_bootstrap.py, slow). It is not selection-adjusted. It is descriptive only. |

**Look-ahead.** The verdict is a full-sample statement about the sample's structure, not a value
used for a trade at date t. The no-look-ahead rule therefore applies to the inputs, and it
holds for them:
- every panel value dated t is point-in-time (ADR 0006);
- every tested link runs from a feature dated t - lag (lag >= 1) to the return dated t;
- forward targets are never read.

Two canaries in tests/leakage/test_causal_inputs.py pin this. A feature that leaks the next
return is found at lag 1. A contemporaneous one is not. Out-of-sample evidence is Stage 2's job.

## Alternatives considered
- **Decide with MCI.** Admissible, but power .35 against .89 on the default preset (ADR 0009).
- **Bonferroni across signals.** Simpler, but its power falls as candidates are added. FDR is
  the standard for a screen whose survivors face four more stages.
- **BY by default.** Valid under any dependence, but costs a factor `sum 1/i` (2.08 at m = 4)
  in the threshold. BH is the default because the validation measures its realised false-pass
  rate directly, and BY is one config value away.
- **Selection-adjusted interval** (re-selecting the lag inside each replicate). It mixes
  different links across replicates and does not adjust for PC1 either. It was rejected in
  favour of a clearly labelled descriptive interval.

## Validation (pre-registered 2026-10-18, before the full run)
**Harness.** `invariant.stages.causal.validation` and `scripts/causal_stage_validation.py`,
with config `configs/causal_stage_validation.yaml`.
- Each replication runs the production path: `generate_market` -> snapshot -> PIT panel ->
  `RunContext` -> Stage 1 -> kill ledger (chain verified on read) -> `score_run(stages=(causal,))`.
- **M = 500**, root seed **20261019** (fresh), T = 6000. Presets: `base`, `realistic`,
  `hidden_confounder`, with common random numbers across presets.

**Reported per preset:**
- kill precision and recall (pooled; the intervals are optimistic, see `aggregate_scores`);
- per-signal pass rates with Wilson CIs, and kill-reason counts;
- MCI family rates as a diagnostic;
- mean realised lag-1 hybrid t against `expected_t_hybrid`;
- predicted power at the Bonferroni threshold;
- bootstrap coverage of the population lag-1 r when the best link is lag 1, as a diagnostic;
- runtime.

The `realistic` preset is a power result. On `hidden_confounder`, the s_conf pass rate is the
identifiability limit.

**Acceptance (base preset only; Stage 1 is done only if all hold):**
- **A1.** s_noise pass rate: Wilson upper bound <= .075.
- **A2.** s_conf pass rate: Wilson upper bound <= .075.
- **A3.** s_real and s_decay pass rates: Wilson lower bound >= .70. Both have true edges, and
  s_decay is killed at Stage 2, not here.
- **A4.** Every replication completes, every ledger chain verifies, and no signal is undecided.

**Prediction, and the error rates of the rule.** The prediction recomputes this exact decision
(Bonferroni family, then BH over the four signals) from study 3's stored per-seed `hc3|hybrid`
p-values on the same preset. That is prior data with root seed 20261014 (M = 1785). It predicts
these pass rates:

| | s_real | s_decay | s_conf | s_noise |
|---|---|---|---|---|
| predicted pass rate | .852 [.834, .867] | .839 [.821, .855] | .018 [.013, .025] | .020 [.014, .027] |
| criterion pass probability at that rate (M = 500) | 1.000 | 1.000 | 1.000 | 0.99999 |

The pass probability is `criterion_pass_probability`, i.e. the Binomial(500, rate) mass
satisfying the bound. At M = 500 the null criteria pass with at most 25 false passes (rate
.05), and A3 needs at least 371 passes (.742).

The rule's limits, stated before the run:
- A1/A2 is not a size certificate. A signal truly passing at .05 would meet it with probability
  only .55, and one at .04 with .89. The size certificate is ADR 0009's M = 1785 studies. A1/A2
  check that the production path reproduces their calibration.
- A3 has probability .68 of passing at a true rate of .75, and .02 at .70.
- The prediction differs from the run in small ways: the panel's `ret_*` drops the first
  session (5999 rows, not 6000), and returns pass through prices (differences ~1e-12).

If any criterion fails, Stage 1 is not done and the decision returns to review. No threshold
is changed after the run.

## Validation results (M = 500, commit `ba43129`, clean tree; report `runs/causal_stage_validation/76b2880f3c8dd866/report.json`)
tigramite 5.2.10.1. Every replication completed and every ledger chain verified (A4). Median
runtime is 1.8 to 2.4 s per replication, including the snapshot, the panel and the ledger.

**Pass rate by signal, with 95% Wilson CIs.** Expected verdicts at Stage 1: s_real pass, s_decay
pass, s_conf kill, s_noise kill.

| preset | s_real | s_decay | s_conf | s_noise | kill precision | kill recall |
|---|---|---|---|---|---|---|
| base | **.844** [.810, .873] | **.834** [.799, .864] | **.034** [.021, .054] | **.018** [.009, .034] | .858 | .974 |
| realistic | .140 [.112, .173] | .034 [.021, .054] | .014 [.007, .029] | .016 [.008, .031] | .519 | .985 |
| hidden_confounder | .890 [.860, .915] | .868 [.836, .895] | .730 [.689, .767] (not identifiable) | .022 [.012, .039] | .802 | .978 |

Precision and recall are pooled over the four signals of each run, so their intervals are
optimistic (`aggregate_scores`). In the realistic preset, precision is low because the true
edges are killed for lack of power, and Stage 1 counts those as wrong kills.

**Acceptance (base): accepted.**
- A1, s_noise: upper bound .034 <= .075.
- A2, s_conf: upper bound .054 <= .075.
- A3, s_real: lower bound .810 >= .70.
- A3, s_decay: lower bound .799 >= .70.
- A4: all 500 replications complete.

## Findings (validation)
1. **The production path reproduces the study.** The pass rates on the base preset match the
   prediction from study 3's stored p-values:
   - s_real .844 against .852, and s_decay .834 against .839;
   - s_noise .018 against .020;
   - s_conf .034 against .018. Its CI [.021, .054] only just overlaps the prediction's
     [.013, .025]. It is within the criterion, and still below alpha.

   Most kills of the true edges are `no_significant_link`: 53 of 78 for s_real and 52 of 83
   for s_decay. The BH step adds 25 and 31.
2. **Hybrid has nearly 2.5 times MCI's power on the planted edges.** On the base preset, the
   MCI family's rate is .364 for s_real and .318 for s_decay, against hybrid's .844 and .834.
   On the nulls, MCI's rates are .032 to .044.
3. **`expected_t_hybrid` overstates the realised t.**
   - Mean realised lag-1 hybrid t on the base preset: s_real 3.78 (SE .08) against an
     expected 4.42; s_decay 3.45 (SE .07) against 4.09. On hidden_confounder: s_real 3.98
     against 6.01.
   - The oracle set uses the true parents. PC1's estimated set differs: it adds a correlated
     non-parent of R (s_conf, which proxies the hidden Z), or a lag-2 copy of the signal
     itself. Either one removes part of the signal's own variation.
   - This explanation is a hypothesis and was not tested here.
   - Consequence: `expected_t_hybrid` is an upper bound for power, not a prediction. Its
     two-sided Bonferroni power (.968 for s_real) is above the observed .844. In the realistic
     preset, where power is low anyway, it matches better: .136 predicted, .140 observed.
4. **The identifiability limit is measured.** With Z hidden, Stage 1 passes the confounded
   s_conf in **73%** of runs [.689, .767]. Every observed adjustment set leaves the bias in
   place. This is the cost of violating causal sufficiency. It is the size of the risk that
   Stage 5's memo must state whenever a plausible confounder is unobserved.
5. **Realistic-preset power is low.** At IC .02, s_real passes 14% and s_decay 3.4%. A real
   edge that weak will usually be killed at Stage 1 at T = 6000. The memo should report a
   Stage 1 kill as "not detectable at this sample size", not as "no causal effect".
6. **The bootstrap interval undercovers.**
   - Coverage of the population lag-1 r by the nominal 95% interval, among runs whose best link
     is lag 1: s_real .858 [.819, .889] and s_decay .944 [.916, .963] on base; null signals .85
     to .90.
   - The likely causes are the persistence of the regressors (rho .9 to .99) and the
     estimated-set bias of finding 3.
   - The interval is descriptive, as designed. It must not be read as a calibrated 95% interval.

## Consequences
- Stage 1 is done on the planted benchmark. Real data comes next, with these documented limits:
  confounding by unobserved variables, low power for weak edges, a lag that is not identified,
  and an effect-size interval that runs narrow.
- Open question for review: `expected_t_hybrid` could be replaced, or accompanied, by a
  simulated power under estimated PC1 sets.
