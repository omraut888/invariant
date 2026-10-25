# ADR 0013: Stage 3 Wasserstein DRO portfolio construction (pre-registered method study)

Status: **pre-registered (2026-10-24); pilot recorded, full run pending**. The theory section records what was
transcribed and verified before any Stage 3 code. The study design, rules, c and the M formula
below were fixed and pushed before the pilot ran. The pilot's outputs (M, the budget plan)
are recorded afterwards, under "Pilot", and pushed before the full run. Results go in a separate
section and are never edited into the pre-registration.

## Decisions already taken (user, 2026-10-23)
- **Kill rule: A, with B as a diagnostic.**
  - The strategy is killed if the chosen arm does not beat 1/N out of sample. The test is a
    one-sided Ledoit & Wolf (2008) Sharpe-difference test at .05. Every member signal is killed,
    or passes, with it.
  - B, each signal's marginal value (leave-one-signal-out), is recorded in each member's
    ledger metrics and never kills.
- **Constraint set**, for every arm and the oracle: Σ|wᵢ| ≤ 1 (gross exposure ≤ 1), shorts and
  cash allowed.
- **Objective**: mean–standard deviation with a fixed coefficient c ≥ 0.
- **Arms**:
  - 1/N, Markowitz (ε = 0), Ridge-CV and DRO-CV;
  - DRO-FC, with a forecast-confidence radius. It competes and is not the default.
  - DRO-RWPI is **dropped** (below).
- **Must be pre-registered before the pilot**: how c is set and why. c must not be tuned on the
  pilot.
- **Must be reported for every arm**:
  - daily exposure (the share of days fully invested vs in cash);
  - turnover;
  - the 5 bp cost sensitivity;
  - for DRO-FC, the empirical coverage of its radius, ‖μ̂_t − μ_t‖₂ ≤ ε_t (reported, not
    gated).

## Theory: transcribed and verified

### Blanchet, Chen & Zhou (2018), arXiv:1802.04885v1
Transcribed from the arXiv v1 text. The arXiv page lists no journal.
- **Problem (2):** min over φ ∈ F_{δ,ᾱ}(n) of max over P ∈ U_δ(Pₙ) of φᵀVar_P(R)φ.
  - The ball is U_δ(Pₙ) = {P : D_c(P, Pₙ) ≤ δ}.
  - The feasible set is F = {φ : φᵀ1 = 1, min_{P∈U_δ} E_P(φᵀR) ≥ ᾱ}.
  - The cost is c(u, w) = ‖w − u‖_q², an order-2 Wasserstein discrepancy.
- **Prop. 1:** min over P ∈ U_δ of E_P(φᵀR) = E_Pₙ(φᵀR) − √δ‖φ‖_p, with 1/p + 1/q = 1.
- **Theorem 1:** (2) is equivalent to min over φ ∈ F of (√(φᵀVar_Pₙ(R)φ) + √δ‖φ‖_p)².
- **§3 (RWPI):** δ is the 1 − δ₀ quantile of a robust Wasserstein profile function. That
  function is built from the Lagrange conditions (11) of the budget-constrained problem (10),
  min over 1ᵀφ = 1 of φᵀE_P[RRᵀ]φ subject to φᵀE_P[R] = ρ.
  - Assumptions: A1 (a stationary, ergodic return series with a CLT), A2, and A3 (unique
    solution φ*).
  - Theorem 2: n·Rₙ ⇒ L₀, so δ = O(1/n).

### Nguyen, Shafiee, Filipović & Kuhn, "Mean-covariance robust risk measurement", arXiv:2112.09959
- **Theorem 5, eq. (8):** over a Gelbrich ball of radius ρ around (μ̂, Σ̂), with standard risk
  coefficient α ≥ 0, sup_Q R_Q(−wᵀξ) = −μ̂ᵀw + α√(wᵀΣ̂w) + ρ√(1+α²)‖w‖₂.
- The Gelbrich value equals the 2-Wasserstein value under their stated conditions (for
  example, a Gaussian nominal distribution with Σ̂ ≻ 0). In general it is an upper bound.

### Our case: empirical centre, cost ‖·‖₂², mean–std risk −E + c·Std
We derived that the worst case over the 2-Wasserstein ball of radius ε is **exactly**
−μ̂ᵀw + c·σ̂(w) + ε√(1+c²)‖w‖₂, where σ̂ is the empirical (1/n) standard deviation. The
argument:
- Moving each sample point by Δᵢ changes wᵀRᵢ by dᵢ = wᵀΔᵢ.
- Cauchy–Schwarz bounds (1/n)Σdᵢ² by ε²‖w‖₂², with equality for Δᵢ ∥ w.
- Writing d = t·1 + r·(standardised centred projection), the adversary maximises t + c·r
  subject to t² + r² ≤ ε²‖w‖², whose maximum is ε‖w‖√(1+c²).
- The bound is therefore attained. It also matches the Gelbrich upper bound above.

Numerical checks, in throwaway scripts; they become unit tests in step 3:

| check | result |
|---|---|
| explicit worst-case distribution, 200 instances | max gap 1.4e-17 |
| 10,000 random feasible perturbations | none exceed the bound |
| Gelbrich SDP, Clarabel, data in percent units, 20 instances | max relative gap 5.4e-7 |

In raw units (covariances around 1e-4), SCS was inaccurate, with gaps up to 7.6%. **Solve in
rescaled units.**

### Kannan, Bayraksan & Luedtke, "Residuals-based distributionally robust optimization with covariate information", arXiv:2012.01088v2 (Math. Programming 207, 2024)
Transcribed from the arXiv v2 text.
- **Model (§2.1):** Y = f*(X) + ε with f*(x) = E[Y | X = x]. The errors ε have mean zero and
  are **independent of X**, and f* is in a known model class F. The decision set Z is nonempty
  and compact.
- **Centre (§2.2):** the empirical-residuals distribution P̂ᴱᴿₙ(x) = (1/n)Σδ_{proj_Y(f̂ₙ(x) +
  ε̂ᵢ)}, with residuals ε̂ᵢ = yᵢ − f̂ₙ(xᵢ).
- **ER-DRO (7):** min over z ∈ Z of sup over Q ∈ P̂ₙ(x) of E_Q[c(z, Y)], with a p-Wasserstein
  ball of radius ζₙ(x) around P̂ᴱᴿₙ(x).
- **Assumption 1 (light tails):** E[exp(‖ε‖ᵃ)] < ∞ for some a > p. For p ≥ 2 this needs tails
  lighter than Gaussian.
- **Assumption 2:** finite-sample error bounds on f̂ₙ. OLS satisfies them with κ² = O(n⁻¹
  log α⁻¹).
- **Lemma 2:** Fournier–Guillin concentration for i.i.d. ε. They remark that it extends to time
  series.
- **Radius (9):** ζₙ(α, x) = 2κ_{p,n}(α/4, x) + κ⁽²⁾_{p,n}(α), the regression-error
  enlargement plus the usual concentration radius. They call it conservative, with unknown
  constants (Remark 3).
- **Theorem 7 (certificate):** under A1, A2 and i.i.d. errors, P{g(ẑ; x) ≤ v̂} ≥ 1 − α.
- **Theorems 9 and 10:** consistency, and a rate of Op(ζₙ). They also need A3–A6.
- **§6 (radius by CV), Algorithms 1–3:** K-fold CV with **random partitions**, which minimises
  the out-of-sample cost.

**What we take from it:** centring the ball on the forecast plus the training residuals, and
the idea of enlarging the radius for regression error. **What does not carry over:**

| KBL assumption | Our setting | Consequence |
|---|---|---|
| Expected cost E_Q[c(z, Y)] | mean–std, a nonlinear functional of Q | Theorems 7, 9 and 10 don't cover our objective. We use their centring with our exact mean–std closed form. |
| ε independent of X | GARCH residuals do not depend on the signals (ADR 0003), but their distribution varies in time | Holds marginally at each date, not across time. |
| i.i.d. errors (Lemma 2) | GARCH volatility clustering; the projection residual contains the omitted, persistent Z (φ = .99) | Violated. Serial dependence. |
| Assumption 1 (light tails, a > p = 2) | Student-t(6) residuals | **Violated outright**: E[exp(‖ε‖ᵃ)] = ∞ for every a > 0. |
| A fixed f* in F | s_decay's coefficient varies in time | Misspecified. Their Remark 1 covers convergence to the best approximation. A rolling window tracks the change partly. |
| Random K-fold CV (§6) | Serially dependent time series | **Not valid here.** Our CV must be walk-forward (blocked, past only). |
| Compact Z, convex support | Σ\|wᵢ\| ≤ 1; Y = ℝᴺ | Holds. proj_Y is the identity. |

### RWPI does not carry over (DRO-RWPI dropped)
- BCZ's δ comes from the smooth Lagrange conditions of the budget-constrained problem (10) and
  needs a unique solution (A3). Our problem has no budget, an L1 constraint and a homogeneous
  objective. Its optimum is either at the gross limit or exactly w = 0, so the optimality
  conditions are non-smooth and the confidence-region idea degenerates at cash.
- RWPI calibrates the sampling error of the empirical distribution of R. Our ball is centred
  on forecast residuals, and most of the forecast error comes from the estimated coefficients
  B̂. That shifts the centre in a way RWPI does not account for.
- The constants of Theorem 2 cannot be read reliably from the extracted text.
- Adapting RWPI would be a new derivation, not a transcription. **The arm is dropped before
  pre-registration, as agreed.**

### Esfahani & Kuhn (2018)
**Not cited as the basis for anything.** Its theorems have not been transcribed.

## Pre-registration (fixed 2026-10-24, before the pilot)

### Approved departures (user, 2026-10-24)
Each was reported with the step-3 code and approved before the pilot. Each departs from a
transcribed method, and each reason is recorded here.

| # | departure | reason |
|---|---|---|
| 1 | Ledoit–Wolf test is **one-sided** (`alternative="greater"`), p = (#{(Δ* − Δ)/s* ≥ Δ/s} + 1)/(M + 1) | The kill rule asks a one-sided question: does the arm beat 1/N? |
| 2 | **Fixed block size** ⌈T^{1/3}⌉ (17 at T = 4739) in place of LW's Algorithm 3.1, and the **block-sum** Ψ estimator for the original data in place of a prewhitened QS HAC | Accepted because the test's size is **gated empirically** on the null scenarios d3 and d5 below, rather than assumed. |
| 3 | **Walk-forward CV** folds (fit on the first 1008 rows of the window, validate on the last 252), ties to the smallest grid value, in place of KBL's random K-fold (their §6) | Random K-folds leak under serial dependence: validation rows would sit between training rows. |
| 4 | DRO-FC's radius uses the **Newey–West rule-of-thumb bandwidth** ⌊4(n/100)^{2/9}⌋ (7 lags at n = 1260) | Kept. Stage 1 already showed this bandwidth struggles with the φ = .99 confounder, so the radius's coverage against the true conditional mean is **reported** per scenario (O1 and O2 means), not assumed. |
| 5 | **c = 0.5/√252** (daily) | Below. |

### The value of c (fixed now, before the pilot; never tuned on it)
- c is the **investor-stated annualised Sharpe hurdle of 0.5**. Under gross ≤ 1, the robust
  objective is positively homogeneous, so a portfolio is held only when some fully invested
  direction's forecast Sharpe exceeds c, plus the DRO penalty. c is therefore a statement of
  what is worth trading. It is not a risk-aversion parameter to be fitted.
- 0.5 is anchored near 1/N's population Sharpe on multi_asset, about 0.45 from the configured
  drift and volatilities. A strategy that cannot clear the hurdle 1/N roughly clears without
  forecasting has no reason to take risk.
- The value was chosen from those two facts, before any Stage 3 run. **0.25 and 1.0 are reported
  only** (task `c_sens`) and never feed a decision.

### Implementation change made before the pilot (no study run had used the old code)
- `MeanStdProblem` now passes the same SOCP to **Clarabel directly**, instead of through cvxpy.
  Profiling showed cvxpy's canonicalisation took about 3.7 ms per solve, against 0.6 ms for
  Clarabel's own solve. Per-seed time on one core fell from about 240 s to about 41 s.
- The formulation is unchanged. A Hypothesis test checks the solution against the cvxpy
  formulation (60 random instances, values to 1e-6 relative).
- Cash detection is now scale-free: a value ≤ 1e-9 or a gross exposure ≤ 1e-6 means cash.
- Dependencies: `clarabel` is a runtime dependency, added directly because it was previously
  transitive. cvxpy moved to the dev group, where it is used only as the test reference and
  for the Gelbrich SDP check.

### Scenarios
Every scenario is a deep merge onto the `multi_asset` preset: 5 assets, n_obs = 6000, a
latent GARCH-t market factor, and loading dispersion .25.

| name | role | change |
|---|---|---|
| d1_base | **selection** | none |
| d2_realistic | reported | daily ICs .02 (factor, real, decay) |
| d3_null | **size gate** | μ = 0 and every IC 0 |
| d4_heavy | reported | Student-t(5) innovations in both GARCH processes (fourth-moment condition .989 < 1) |
| d5_null_heavy | **size gate** | d3 with t(5) innovations |

- **The nulls set μ = 0 as well as the ICs.** With drift, portfolios that tilt towards
  low-beta assets have a higher population Sharpe than 1/N, so H0 would not hold. With no
  drift and no signal, E[r_{t+1} | F_t] = 0 and every arm's w_t is F_t-measurable. Every arm
  then has population Sharpe 0, which is H0's boundary, where the test's size is its level.
- Heavy tails make t(4.5) invalid at the configured GARCH (fourth-moment condition 1.011), so
  t(5) is the heaviest valid choice.

### Arms, signals and walk-forward (all fixed)
- **Arms**:
  - 1/N (the kill-test baseline);
  - the candidates, in simplicity order: **Markowitz < DRO-FC < Ridge-CV < DRO-CV**.
- **Why that order:** Markowitz has no tuning. DRO-FC has a closed-form radius and no tuning
  loop. Ridge-CV and DRO-CV each choose from an 8-value grid by walk-forward CV. Ridge comes
  first because its effect, shrinking the covariance, is the more conventional.
- **Signals:** the planted true causes, {s_real, s_decay}. The selection does not depend on
  Stage 1–2 randomness. The pipeline's own passed set is reported on d1 (task `pipeline`:
  Stages 1–2 through the production path, on the same market).
- **Walk-forward settings:**
  - window 1260;
  - refit every 21 rows;
  - CV every 252 rows, on the last 252 rows of the window;
  - DRO-FC α = .05, with 4000 seeded draws;
  - ridge grid {0, .01, .03, .1, .3, 1, 3, 10} × tr(Σ̂)/N;
  - ε grid {0, .0025, .005, .01, .02, .05, .1, .2} × √(tr(Σ̂)/N).
- **Test dates:** 1260 … 5998, so T = 4739 out-of-sample days.
- **Oracles:**
  - O1 is the exact projection on the arm's signals;
  - O2 uses every observed state node;
  - both are solved with the same c and constraint.
- **An all-cash path has Sharpe ratio 0**, since it earns the zero risk-free rate.

### Kill rule (Stage 3, as built after this study)
**A.** The strategy is killed if the selected arm does not beat 1/N out of sample. The test is
a one-sided Ledoit–Wolf at α = .05, with B = 999 and block ⌈T^{1/3}⌉. An arm that stays in cash
throughout has p = 1. Every member signal is killed, or passes, with it.
**B (diagnostic, never kills).** Each signal's marginal value is Sharpe(all signals) −
Sharpe(without it), and it goes in each member's ledger metrics. The study reports B for every
candidate on d1 (task `loso`), with seed-bootstrap CIs.

### Gates (decide admissibility)
For each candidate and each null scenario (d3_null, d5_null_heavy): the kill-test rejection
rate has a **Wilson 95% upper bound ≤ .075**. That makes 8 gates. A candidate that fails
either of its gates is inadmissible.

### Selection rule and tie-break
- Among admissible candidates, the one with the **highest mean annualised out-of-sample
  Sharpe on d1_base** is best (gross of costs). O1 is shared by all arms within a seed, so this
  is the same as the lowest mean O1 regret.
- A candidate is **tied** with the best when the paired seed-bootstrap 95% CI (2000
  resamples) of the mean Sharpe difference contains 0.
- The **first tied candidate in simplicity order is selected.**
- If Markowitz is selected, Stage 3 uses Markowitz, and the memo states that Wasserstein
  robustness did not earn its place on the benchmark.
- If no candidate is admissible, the stage is not built and the design goes back to review.

### M
- **Gated scenarios (d3, d5):**
  - M_gate = `required_seeds(0.05, 0.075, 8, 0.90)` = **1683**.
  - Pass probability per gate: **.990** at true size .05, **.025** at .075, and **0.000** at .10.
    It is 1.000 at .03.
  - All 8 pass with probability ≥ .918 at .05 (union bound; .921 if the gates were
    independent).
- **Selection scenario (d1):**
  - M_sel = max(200, ⌈((z.975 + z.90) σ_U / 0.10)²⌉).
  - σ̂ is the largest SD, over the 20 pilot seeds, of any candidate pair's per-seed Sharpe
    difference.
  - σ_U = σ̂ √(19/χ²_{.05,19}) = 1.370 σ̂ is its one-sided 95% upper bound.
  - With that σ_U, a true mean Sharpe difference of **0.10 (annualised)** gives a CI that
    excludes 0 with probability ≥ .90. A truly equal pair is tied with probability about .95.
  - Examples: σ̂ = .2 → 79 (raised to the 200 floor); .3 → 178 (floor, 200); .4 → 316.
  - The floor keeps the reported rates' Wilson half-widths at ≤ .07.
- **Reported tasks** run on the first M_rep = M_sel seeds, unless the budget rule lowers M_rep:
  - main on d2 and d4;
  - loso, pipeline and c_sens on d1.

### Pilot
- 20 seeds from **root seed 20261024**, separate from the study's **20261025**. Neither root
  seed is used by any Stage 1 or 2 study or validation.
- The pilot runs every scenario and every task kind.
- It is used **only** for σ̂ (above) and the per-task timings (below). Its rates and means are
  neither printed nor used (`scripts/dro_study.py --pilot` prints only the SDs, M_sel and the
  budget plan).

### Compute budget (mechanical; not revisited after results)
- **Budget:** 24 h wall clock on all 8 cores of the run machine.
- **Projection:** Σ over tasks of (the pilot's mean seconds for that scenario and kind) / 8.
- The core is the gated scenarios at M_gate plus d1 at M_sel.
  - If core + the reported tasks at M_sel fit, everything runs at those sizes.
  - Otherwise M_rep is the largest value that fits.
  - If M_rep would fall below 100, or the core alone exceeds the budget, **the study is not
    run** and the design goes back to the user. The gates and M_gate are never weakened to fit.

### Reported for every arm (never gated)
For every arm, by scenario:
- mean Sharpe, gross and **net of 5 bp** per unit of one-way turnover;
- O1 and O2 regret;
- the kill-test rejection rate against 1/N (power on d1, d2 and d4);
- **daily exposure** (share of days fully invested, in cash, or partial) and mean daily
  **turnover**;
- the DSR (Bailey & López de Prado 2014). It uses N = 4 candidate trials and the trials'
  Sharpe variance within the seed (ddof 1), and is reported as a mean and as the share of
  seeds with DSR > .95;
- for DRO-FC, the radius's **empirical coverage**: the share of days with ‖μ̂_t − μ_t‖₂ ≤ ε_t,
  against O1's mean and against O2's mean.

Also reported:
- the CV arms' chosen grid values;
- B per signal;
- the distribution of the pipeline's passed signal sets, and the arms on them;
- the c = 0.25 and c = 1.0 sensitivities.

### Fast-suite coverage
`tests/integration/test_dro_study_tasks.py` runs every scenario and every task kind on one
tiny seed (n_obs 2500, window 300):
- the pipeline path, with Stage 1 passing s_real so the arms run on it;
- the summary rules;
- the pilot report;
- the budget plan.

`tests/unit/test_dro_study.py` tests:
- the M formula;
- the budget plan;
- the gates;
- the tie-break;
- the scenario configs.

### Limitations stated in advance
- Selection is on one scenario (d1). d2 and d4 are reported, so a ranking that is fragile to
  SNR or tails will be visible, but it will not change the choice.
- The selection uses the true signal set. The pipeline's own set, which can include a
  false positive or miss s_decay, is reported, not selected on.
- The size gates hold at H0's boundary with zero-mean returns. Under drift the null is
  composite. Arms that sit in cash have p = 1, which makes the test conservative for them, and
  exposure is reported so that this is visible.
- The DSR's N = 4 ignores the CV grids. Their choices are made walk-forward, inside the arm,
  and are not seen out of sample.
- Gross selection: turnover costs are reported (5 bp), not optimised.

### Run discipline
- The pilot and the full run execute from a git worktree at the pushed commit.
- Post-launch bug fixes are recorded here with their cause, fix commit, and resume or restart
  decision.

## Pilot (recorded 2026-10-24, after the pre-registration commit 5b5c19e and before the full run)
The pilot ran from a worktree at 5b5c19e: 20 seeds from root 20261024, every scenario and task
kind, on 8 workers. Only the quantities below were printed. Its rates and means were not looked
at or used.

- **σ̂ by candidate pair** (SD of the per-seed difference in annualised Sharpe, d1_base):

  | pair | σ̂ |
  |---|---|
  | Markowitz – DRO-FC | **.2323** (the largest) |
  | DRO-FC – DRO-CV | .2263 |
  | DRO-FC – Ridge-CV | .2012 |
  | Markowitz – DRO-CV | .1246 |
  | Markowitz – Ridge-CV | .1051 |
  | Ridge-CV – DRO-CV | .0676 |

- **M_sel:**
  - σ_U = 1.370 × .2323 = .318, so the formula gives ⌈(3.2415 × .318 / .10)²⌉ = 107.
  - The floor applies: **M_sel = 200**.
- **Mean task seconds** (wall time under 8-way load):

  | scenario / task | seconds |
  |---|---|
  | d1 main | 148.5 |
  | d1 loso | 211.4 |
  | d1 pipeline | 126.2 |
  | d1 c_sens | 203.6 |
  | d2 main | 167.7 |
  | d3 main | 119.8 |
  | d4 main | 134.8 |
  | d5 main | 119.3 |

- **Budget plan:**
  - the core (d3 and d5 at 1683, d1 at 200) projects to 15.0 h;
  - with the reported tasks at 200 seeds, 20.9 h, within the 24 h budget;
  - feasible, so **M_rep = 200** (no reduction).
- **Recorded in `configs/dro_study.yaml`:** `n_select_seeds: 200`, `n_report_seeds: 200`.
- **Full run:** 2 × 1683 + 200 + 2 × 200 + 3 × 200 = **4566 tasks**, root seed 20261025.
