# ADR 0012: Stage 2, the decay stage

Status: **pre-registered (2026-10-23)**. The stage is implemented. The validation below was
committed and pushed before its run. Results will be appended. Nothing above "Validation
results" changes after the run.

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

## Validation results
*Pending.*
