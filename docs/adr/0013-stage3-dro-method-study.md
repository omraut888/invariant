# ADR 0013: Stage 3 Wasserstein DRO portfolio construction (pre-registered method study)

Status: **draft**. This is not yet a pre-registration. The theory section below records what
was transcribed and verified before any Stage 3 code. The study design, rules, the value of c
and M are added before the pilot, and the file is committed and pushed before any full run.

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

## Pending (before pre-registration)
- The value of c, and its justification.
- Transcriptions of Ledoit & Wolf (2008) and Bailey & López de Prado (2014), before either is
  used.
- The scenarios, gates, selection rule and tie-break.
- The pilot-based formula for M and the compute budget.
