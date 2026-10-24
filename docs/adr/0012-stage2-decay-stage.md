# ADR 0012: Stage 2, the decay stage

Status: **accepted (2026-10-23)**. The M = 1000 validation below was pre-registered and pushed
before its run (commit `90ce789`). Every acceptance criterion passed (final section).

## Context
ADR 0011 adopted the kill rule: the one-sided sup-split decline test (Andrews 1993) with a
fixed-regressor wild bootstrap (Hansen 2000), killing at p < .05. Its four false-decay gates
passed (.043–.051) and its power on s_decay at the default preset was .622 given a Stage 1
pass. No half-life estimator was admissible, so Stage 2 reports none. This ADR turns that into
a pipeline stage and pre-registers its validation through the production path, as ADR 0010
did for Stage 1.

**This stage deviates from the spec** (docs/invariant-spec.html, section 04). The spec decides
with a transfer-entropy half-life fitted to rolling windows. ADR 0011 showed that half-life to
be unidentifiable at realistic SNR (Cramér–Rao) and its profile-likelihood interval to be
anti-conservative (false-decay .10–.12). The deviation and its evidence are recorded there.

## Decision

### Inputs
- Stage 1's `CausalResult` and the same PIT panel. Stage 2 re-applies Stage 1's input
  selection (`causal.stage.select_inputs`), so its data matrix is the one Stage 1 tested.
- It raises `ValueError` when the columns, row count or dates differ from the `CausalResult`:
  the result would then belong to another panel.
- `fwd_*` is never read (tests/leakage/test_decay_inputs.py).
- Only signals Stage 1 **passed** reach Stage 2. The ones it killed are listed in
  `DecayResult.not_reached` and get no Stage 2 ledger entry (the ledger forbids entries after
  a kill).

### Method (`invariant.stages.decay.stage`)
1. For each signal that reached Stage 2, take Stage 1's best link: the asset, the lag and the
   conditioning set, all unchanged. Build the same design (`causal.pcmci.lagged_design`).
2. Run `decline.decline_test` with `n_boot` 999, `trim` .15 and `break_step` 25 (ADR 0011).
   The RNG is the run's `RngFactory` → child `decay` → one generator per signal id.
3. Verdict:
   - kill `general.insufficient_data` when the design has fewer than `min_obs` rows;
   - otherwise kill `decay.edge_declining` when p < α (.05);
   - otherwise pass.

   A tie at α passes, because the rule is p < α. There is no correction across signals
   (ADR 0011).
4. No half-life is reported. The ledger records `half_life: not_reported`.

### `min_obs` = 5000 design rows
The test's size and power were measured only at T = 6000 (ADR 0011). A shorter sample is
killed for insufficient data rather than decided by an unvalidated test. The real ETF panel
for 2005–2025 has about 5,280 sessions, so it clears this floor.

Alternatives considered:
- Stage 1's 500 rows: size unmeasured, and the power there is near zero.
- A size and power study at shorter T: deferred, and added to the ADR 0011 backlog.

### Outputs
- `DecayResult` (frozen pydantic, written to `<run_dir>/decay_result.json`). It holds the
  config, Stage 1's config, n_obs and dates, plus one `SignalDecay` per signal that reached
  Stage 2:
  - the link tested;
  - the verdict and reason;
  - the full `DeclineTest`: S, p, the break row and fraction, b_full, b_pre, b_post and their
    SEs, and the trend diagnostic.
- **One kill-ledger entry per signal that reached Stage 2.** Its metrics are flat, finite
  scalars: p, statistic, n_rows, break_row, break_fraction, b_full, b_pre, b_post, se_pre,
  se_post, trend_t and trend_p; the config values; the asset, lag and n_conditions; and
  `test: sup_split_wild_bootstrap`.
- `KillReason.DECAY_EDGE_DECLINING = "decay.edge_declining"`.

### Config
`decay:` in configs/base.yaml (`DecayConfig`: alpha, n_boot, trim, break_step, min_obs). Adding
it changed every run hash. The golden run id in tests/unit/test_run_hash.py was re-pinned to
`6b24e63ac1f49496`.

### Assumptions
See ADR 0011's table. In short:
- the residuals are an MDS given Stage 1's set;
- only the signal's coefficient changes (a strong decaying co-parent leaks slightly: .055 and
  .058);
- only the best link is tested;
- power is lowest for recent decay (.22–.44 at onset .8 T).

The verdict is a full-sample statement, and its inputs are point-in-time, so the look-ahead
rule holds as it does for Stage 1.

## Validation (pre-registered 2026-10-23, before the run)
**Harness.** `invariant.stages.decay.validation`, `scripts/decay_stage_validation.py`, config
`configs/decay_stage_validation.yaml`. Each replication:
- runs `generate_market` → snapshot → PIT panel → `RunContext` → Stage 1 → ledger → Stage 2
  → ledger (chain verified on read);
- is scored with `score_run(stages=(causal, decay))`.

**M = 1000**, root seed **20261023** (fresh), T = 6000. The presets are `base`, `realistic`
and `hidden_confounder`, with common random numbers.

M was raised from 500 before any run. At M = 500 only about 422 runs reach Stage 2, and the
null criterion would pass a correct stage only .73 of the time at the measured false-decay rate
of .044. At M = 1000 that is .965. The user chose M = 1000 with threshold .52 on 2026-10-23.

**The rates are conditional on a Stage 1 pass**, because Stage 2 only sees those runs. The
denominator is random, about Binomial(1000, Stage 1 pass rate).

**Acceptance (base preset; Stage 2 is done only if all hold):**
- **B1.** s_real Stage 2 kill rate given a Stage 1 pass (its false-decay rate): Wilson upper
  bound ≤ .075.
- **B2.** s_decay Stage 2 kill rate given a Stage 1 pass (its power): Wilson lower bound
  ≥ **.52**.
- **B3.** Every replication completes, every ledger chain verifies, and no signal is
  undecided.

**Prediction.** From ADR 0011 (study, root seed 20261021, M = 1610): s_real .044 [.034, .056]
and s_decay .622 [.596, .648], given a Stage 1 pass. The Stage 1 pass rates come from ADR 0010:
.844 for s_real and .834 for s_decay.

**Pass probabilities** (`acceptance_pass_probability`, exact: Binomial(1000, Stage 1 pass
rate) denominators, then a Binomial kill count):

| B1: true s_real false-decay rate | .030 | **.044** | .050 | .060 | .075 |
|---|---|---|---|---|---|
| P(B1 passes) | 1.000 | **.965** | .833 | .375 | .022 |

| B2: true s_decay kill rate | .648 | **.622** | .596 | .570 | .550 | .520 |
|---|---|---|---|---|---|---|
| P(B2 passes) | 1.000 | **1.000** | .993 | .826 | .411 | .025 |

So at the predicted rates, B1 and B2 both pass with probability ≥ .965 × 1.000. The rule fails a
stage whose false-decay rate is .075 (P(pass) .022) or whose power has dropped to .52 (.025).

Limits of the rule, stated before the run:
- **B1 is not a size certificate.** ADR 0011's M = 1610 study is the certificate. B1 checks
  that the production path reproduces it.
- **B2 cannot tell .57 from .62** (P(pass) .83 vs 1.00).
- **The study and this validation differ slightly.** The study read `dataset.features`, and
  this run goes through the PIT panel (5999 rows rather than 6000; ADR 0010).

**Reported, not gated:**
- the Stage 1 pass rates;
- the trend-test kill rate;
- the kill reasons;
- the share of runs in which each signal passes both stages;
- pooled kill precision and recall over (causal, decay), whose intervals are optimistic;
- the realistic and hidden_confounder presets;
- runtime.

If any criterion fails, Stage 2 is not done and the decision returns to review. No threshold
changes after the run.

## Validation results (M = 1000, commit `90ce789`, clean tree; report `runs/decay_stage_validation/41068846d9e5ae98/report.json`)
Every replication completed and every ledger chain verified (B3). The median runtime per
replication, both stages included, was 3.5–4.5 s.

**Base preset.** Wilson 95% CIs. "Given a Stage 1 pass" is the Stage 2 rate.

| signal | Stage 1 pass | Stage 2 kill, given a Stage 1 pass | trend diagnostic | passes both |
|---|---|---|---|---|
| s_real | .882 [.861, .901] | **.054 [.041, .071]** (n = 882) | .052 | .834 |
| s_decay | .830 [.805, .852] | **.663 [.630, .694]** (n = 830) | .661 | .280 |
| s_conf | .022 | .045 (n = 22) | .045 | .021 |
| s_noise | .030 | .000 (n = 30) | .067 | .030 |

Pooled over (causal, decay), with optimistic intervals: kill precision .941 and kill recall .890.

**Acceptance (base): accepted.**
- B1, s_real false-decay: upper bound **.0714** ≤ .075.
- B2, s_decay kill rate: lower bound **.630** ≥ .52.
- B3: all 1000 replications completed.

**Other presets (reported).**

| preset | s_real S2 kill given S1 pass | s_decay S2 kill given S1 pass | s_conf S2 kill given S1 pass |
|---|---|---|---|
| realistic | .077 [.041, .140] (n = 117) | .083 [.033, .196] (n = 48) | .083 (n = 12) |
| hidden_confounder | .054 [.041, .071] (n = 904) | .659 [.627, .690] (n = 851) | .045 [.032, .062] (n = 735) |

## Findings (validation)
1. **The production path reproduces the study, within noise. Both rates sit above their
   point predictions.**
   - s_real's false-decay rate is .054 against the study's .044 [.034, .056]. The intervals
     overlap, and both include .05. B1 passed with a margin of .004 on its upper bound, so it
     came close. It is consistent with a size at or slightly above nominal. This is not a
     size certificate: ADR 0011's gate (.043 [.035, .055] at M = 1610) is.
   - s_decay's power is .663 against .622 [.596, .648]. The intervals overlap.
   - The differences between the study and this run are the PIT panel path, fresh seeds, and
     a different set of Stage 1 passes (s_real .882 here against .844 in ADR 0010).
2. **Across both stages, s_decay is killed at Stage 2 in 55% of runs** (.830 × .663), and it
   passes both stages in 28%. Most of those 28% are runs where the decline was not detected.
   That is the power limit of ADR 0011, and the memo must report it for any signal that passes
   Stage 2.
3. **Stage 2 does not kill confounded signals, which is correct.** With Z hidden, s_conf
   passes Stage 1 in 73.5% of runs (the identifiability limit, ADR 0010). Stage 2 kills only
   4.5% of those, because s_conf's (spurious) edge does not decay. Stage 2 cannot remove that
   failure of causal sufficiency, and the memo must still state it.
4. **Realistic preset: few signals reach Stage 2, and the rates have wide intervals.**
   s_real's false-decay rate is .077 [.041, .140] on n = 117, which is uninformative about
   size.

## Consequences
- **Stage 2 is done on the planted benchmark**, with these documented limits:
  - no half-life is reported;
  - power is .6–.66 at default SNR and lowest for recent decay;
  - `min_obs` is 5000 design rows;
  - only the best link is tested;
  - the decline test cannot detect unobserved confounding (it is not its job).
- The ADR 0011 backlog holds the follow-ups: a calibrated half-life, decline-test size at
  shorter T, and the co-parent leak.
