# ADR 0014: Stage 3, the portfolio construction stage (Ridge-CV, kill rule A)

Status: **pre-registered (2026-10-27); validation pending**. The validation below was fixed and
pushed before its run. Results will go in a separate section and are never edited into the
pre-registration.

## Context
ADR 0013's pre-registered study passed all eight size gates and selected **Ridge-CV**: ridge
regularization of the covariance, with its strength chosen by walk-forward CV. Wasserstein
DRO (DRO-FC, DRO-CV) was implemented, verified and evaluated there, and not selected. This
ADR turns that result into a pipeline stage and pre-registers its validation through the
production path, as ADRs 0010 and 0012 did for Stages 1 and 2.

**Naming.** Stage 3 is **portfolio construction**. A memo must say that Wasserstein DRO was
evaluated and ridge regularization selected, and must never call the portfolio
distributionally robust.

## Decisions (user, 2026-10-27)
- **The stage identifier is `portfolio`.** `StageName.DRO = "dro"` became
  `StageName.PORTFOLIO = "portfolio"`. This changes a public interface: ledger stage values and
  four tests. No Stage 3 ledger entry existed. The verified DRO code stays in
  `invariant.stages.dro`; the stage is `invariant.stages.portfolio`.
- **Diagnostic B with a single member** compares against Ridge-CV with no signal (an
  intercept-only forecast), so B is always defined when the stage runs.
- **Markowitz, DRO-FC and DRO-CV run beside Ridge-CV, reported only** (`decides: false`). They
  supply the DSR's trial variance (N = 4, as in ADR 0013) and the memo's DRO numbers on the
  same data.
- **The forecast timing comes from the PIT config's `targets.execution_lag`**, never
  hard-coded. The benchmark uses lag 0 (ADR 0013 and `GroundTruth`); real data uses lag 1.
  A benchmark panel with any other timing is refused.
- **A pre-push hook** runs ruff (check and format), mypy and the fast suite (see "Process").

## Decision

### Inputs
- Stage 2's `DecayResult` and the same PIT panel. Stage 1's input selection is reapplied and
  its row count and dates must match the `DecayResult`; otherwise `ValueError`.
- **Members**: the signals that passed Stage 2. Signals killed earlier are listed in
  `PortfolioResult.not_reached` and get no Stage 3 entry. No member: nothing runs.
- Data: the members' columns and every `ret_*` column, with leading incomplete rows trimmed.
  `fwd_*` is never read (tests/leakage/test_portfolio_inputs.py).

### Timing (`dro.walkforward`, `WalkForwardConfig.lag`)
With e = `pit.targets.execution_lag`, a decision at the close of t uses only rows <= t: refits
on pairs `(x[s], r[s+1+e])` with `s + 1 + e <= t`, and CV inside that window on the same
pairs. It is entered at the close of t + e and earns `r[t+1+e]`. At e = 0 this is exactly ADR
0013's timing; the default `lag = 0` leaves the study's code path unchanged. Checked
directly: rerunning ADR 0013's recorded task (d3_null, seed 0, main) with this code reproduces
its record exactly, apart from wall time.
- Only `horizon = 1` is supported (daily rebalancing, as validated); any other raises.
- A panel with `syn_*` members must use `GroundTruth`'s timing (execution lag 0, horizon 1),
  or the stage raises `TargetTimingMismatchError`, the error Stage 1's scoring guard uses. The
  validation harness also calls `GroundTruth.require_target_timing` before any stage, and
  `score_run` refuses a mismatched run.
- Tests: the timing test plants a lead of 1 + e and finds it only at the matching lag
  (e = 0, 1, 2); the look-ahead property test covers e = 0, 1, 2.

### Method (`invariant.stages.portfolio.stage`)
1. Walk-forward over 1/N, Markowitz, DRO-FC, Ridge-CV and DRO-CV with ADR 0013's settings:
   window 1260, refit every 21, CV every 252 on the last 252 rows, c = 0.5/√252, gross ≤ 1. A
   unit test pins `configs/base.yaml`'s `portfolio:` block to `configs/dro_study.yaml`'s values.
2. **Kill rule A.** Kill every member with `portfolio.no_edge_over_equal_weight` when the
   one-sided Ledoit–Wolf test of Ridge-CV's Sharpe against 1/N (999 replicates) gives
   **p ≥ .05**. A tie at α kills. General kills come first:
   - `general.data_validation_failed`: a missing value inside the sample;
   - `general.insufficient_data`: fewer than `min_obs` complete rows.
3. **Joint verdict.** Every member gets the portfolio's verdict. This answers ADR 0007's open
   question for Stage 3: a portfolio-level decision recorded on each member's ledger entry.
   On the benchmark, a Stage 3 kill of a true signal therefore scores as a wrong kill (a power
   miss), and a passing false positive from Stage 1 as a missed kill.
4. **Diagnostic B** (never decides): Ridge-CV's Sharpe minus Ridge-CV's Sharpe without the
   member (intercept-only with one member), with a paired circular-block percentile interval
   (block ⌈T^{1/3}⌉, 999 replicates).
5. **Reported** per arm: gross Sharpe and Sharpe net of 5 bp, mean, SD, exposure shares,
   turnover, LW Δ and p, DSR. Also Ridge-CV's chosen λ path and its final weights.

RNG: the run's `RngFactory` → child `portfolio` → `wf` (DRO-FC radius draws), `lw/<arm>`,
`b/<signal>`, and `drop/<signal>` for B's walk-forwards. Inputs are copied C-contiguous, so the
result depends only on the values, not the panel's memory layout (a leakage test caught a
1e-16 difference before this).

### `min_obs` = 5000 complete rows (not 6000 as first planned)
I had proposed 6000, the only length ADR 0013 ran. The real ETF panel (2005–2025) has about
5,280 sessions, so 6000 would kill every real run with insufficient data. Instead the floor is
5000 rows, and **gate V1 measures the kill test's size at exactly 5000 rows**. Power is only
measured at 6000 (V2); at 5000 it will be lower, and that is stated, not gated.

### Outputs
- `PortfolioResult` (frozen pydantic, `<run_dir>/portfolio_result.json`): the config, Stage
  2's config, n_obs and dates, the members with the joint verdict and B, and the
  `PortfolioEvidence`.
- **One ledger entry per member**, with flat finite metrics: every arm's Sharpe, net Sharpe,
  exposure, turnover, LW p and Δ, DSR; B and its interval and baseline; execution_lag,
  n_test_days, min_obs, n_members; `deciding_arm: ridge_cv`;
  `method: ridge_cv_selected_wasserstein_dro_evaluated`.

### Config
`portfolio:` in configs/base.yaml (`PortfolioConfig`). Adding it changed every run hash; the
golden run id in tests/unit/test_run_hash.py was re-pinned to `485aae913204902c`.

### Assumptions and limitations
- Linear forecasts from the members dated t. Stage 1's best lag is not used: it is not
  identified (ADR 0010), and ADR 0013 validated lag-0 features.
- Portfolio returns are `w · r` with log returns (first order); position drift between
  rebalances is ignored.
- The kill test's power at realistic signal strength is close to zero (ADR 0013, finding 2).
- Selection was on gross Sharpe; DRO-FC was better net of 5 bp (ADR 0013, finding 1; backlog).
- **LW replicates with zero variance.** When a bootstrap replicate of a nearly all-cash path
  has zero variance, its statistic is NaN and it counts as not exceeding the observed one,
  which lowers p slightly. This mostly affects DRO-FC on the nulls (99% cash). ADR 0013's size
  gates were measured with this behaviour (DRO-FC .035–.039), and V1 measures it for Ridge-CV.
  It is unchanged so that the stage is the code ADR 0013 evaluated.
- The verdict is a full-sample statement about the out-of-sample path; every weight dated t
  uses rows <= t (tests/leakage/test_dro_no_lookahead.py).

## Validation (pre-registered 2026-10-27, before the run)
**Harness.** `invariant.stages.portfolio.validation`, `scripts/portfolio_stage_validation.py`,
config `configs/portfolio_stage_validation.yaml`. Root seed **20261027** (fresh). Replication
i takes its dataset and pipeline seeds from `RngFactory(20261027).child("seed/i")`; task kinds
share seeds.
- **size** task: ADR 0013's null scenario (drift and every true edge 0) at `n_obs` 5001, which
  gives 5000 complete panel rows = `min_obs`. s_real and s_decay are forced in as members.
  Snapshot → PIT panel → Stage 3. A pass is a false pass. Scenarios d3_null (GARCH-normal) and
  d5_null_heavy (t(5) innovations).
- **pipeline** task: ADR 0013's selection market (`multi_asset`, T = 6000; 5999 panel rows).
  Snapshot → PIT panel → `RunContext` → Stage 1 → ledger → Stage 2 → ledger → Stage 3 → ledger,
  scored with `score_run(stages=(causal, decay, portfolio))`.

**M = 1273** = `required_seeds(.05, .075, n_gating = 2, all_pass = .90)` (the two V1 gates;
per-gate pass probability ≥ .95 by the union bound). The same M runs the pipeline task. 3 × 1273
= 3819 tasks.

**Acceptance (Stage 3 is done only if all hold):**
- **V1 (size, each of d3_null and d5_null_heavy).** Ridge-CV's false-pass rate (LW p < .05
  against 1/N), Wilson 95% upper bound ≤ **.075**. Every size run must build the portfolio.
- **V2 (power, pipeline).** s_real's Stage 3 pass rate given that it reached Stage 3, Wilson
  lower bound ≥ **.70**.
- **V3.** Every replication completes, every ledger chain verifies, and no signal is
  undecided over (causal, decay, portfolio).

**Predictions** (from ADR 0013, at 6000 rows): Ridge-CV's false-pass rate .045 [.036, .056]
(d3) and .040 [.031, .050] (d5); its pass rate on the pipeline's own signal sets .804
[.743, .854] (n = 194), with s_real reaching Stage 3 in 193 of 200 runs (.965).

**Pass probabilities.** V1 per gate (exact binomial at M = 1273):

| true false-pass rate | .030 | .040 | **.045** | .050 | .060 | .075 | .100 |
|---|---|---|---|---|---|---|---|
| P(gate passes) | 1.000 | 1.000 | **.996** | .959 | .559 | .025 | .000 |

Both gates at .045: .992; at .050: .920.

V2 (`acceptance_pass_probability`: Binomial(1273, .965) denominators, then a binomial pass
count):

| true pass rate given reached | .850 | **.804** | .780 | .760 | .743 | .720 | .700 |
|---|---|---|---|---|---|---|---|
| P(V2 passes) | 1.000 | **1.000** | 1.000 | .997 | .918 | .332 | .024 |

So at the predicted rates all criteria pass with probability ≥ .99. The rule fails a stage
whose size at 5000 rows is .075 (P(pass) .025 per gate) or whose power has fallen to .70
(.024). The .70 threshold passes with probability .918 even at the lower end of ADR 0013's
interval (.743).

**Limits of the rule, stated before the run:**
- V1 is the size certificate at **5000 rows**; ADR 0013's gates certify 6000. V1 adds the
  production path (PIT panel, `run_portfolio_members`) and the shorter sample.
- V2 cannot tell .74 from .80 well (P(pass) .92 against 1.00). It checks that the production
  path reproduces ADR 0013's pipeline power, not the power itself.
- The pipeline's member sets are Stage 1–2's (mostly {s_real}; ADR 0013 finding 6), so V2's
  rate mixes portfolios of one and two signals.

**Reported, not gated:**
- every arm's mean Sharpe (gross and net), exposure, turnover and beats-1/N rate, per task kind;
- B's mean and the share of runs with its interval excluding 0, per member;
- per signal over the pipeline: the rate of reaching Stage 3, passing given reached, and passing
  all three stages (including s_conf and s_noise, the Stage 1 false positives);
- the member-set distribution;
- pooled kill precision and recall over (causal, decay, portfolio), with optimistic intervals;
- runtime.

**Compute.** One task took 21 s (size) and 25 s (pipeline) on an idle machine (a smoke run on
root seed 1, which is not this run's; no rate was looked at). ADR 0013's d1 task took a median
42.6 s under 8-way load against about 41 s on one idle core, so load cost little there. Allowing
2× the idle times, the projection is about 3819 × 50 s / 8 ≈ 6.6 h.
No reported-only fallback is needed: every task is gated.

If any criterion fails, Stage 3 is not done and the decision returns to review. No threshold
changes after the run.

## Process
- **Pre-push hook.** `.pre-commit-config.yaml` adds `pre-push` hooks that run
  `ruff check .`, `ruff format --check .`, `mypy` and `pytest -m "not slow"` on the whole tree
  (`default_install_hook_types: [pre-commit, pre-push]`; install with
  `uv run pre-commit install`). This follows commit 9686660, which was pushed with a failing
  test (ADR 0013, Results).
- **A pre-existing flaky test, fixed (test only).** `test_direct_clarabel_matches_the_cvxpy_formulation`
  (ADR 0013's solver check) compared the two formulations to 1e-10 absolute. Both stop at
  Clarabel's default gap tolerance (1e-8, in units scaled by about 100), so each is within
  about 1e-10 of the optimum in caller units and they may differ by about 2e-10. Hypothesis
  found cases differing by 1.04e-10 (the value) and 1.6e-10 (the objective recomputed from w).
  Both bounds are now 5e-10, with those cases pinned as `@example`s. The solver code is
  unchanged.

## Alternatives considered
- **Per-signal kill from B** (kill a member whose marginal Sharpe is not positive): rejected
  for now. B was a diagnostic in ADR 0013, its interval's calibration is unmeasured, and with
  correlated members it is not a clean per-signal test.
- **Zero weight = kill** (ADR 0007's other suggestion): Ridge-CV's weights are on assets, not
  signals, so it does not apply.
- **min_obs 6000**: would never run on the real panel (above).
