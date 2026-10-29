# ADR 0015: Benchmark v4: a regime-fragile signal (s_regime) on a Markov-switching GARCH market

Status: **accepted (2026-10-29)**. Calibration values reviewed by the user at Stage 4 stop point 1.

## Context
Stage 4 (adversarial regime certification) needs a ground truth to be graded against
(non-negotiable 2). The v3 benchmark has no regime-fragile signal: no edge in it depends on
the market state. v4 adds one, with three expected outcomes:
- Stages 1–3 may pass it, because its edge averaged over regimes is real;
- Stage 4 should kill a portfolio built on it;
- s_real, whose edge holds in every regime, should pass Stage 4.

Every existing preset, and every result validated on it, must be unchanged.

## Decision

### Opt-in regime block (existing presets unchanged)
`SyntheticMarketConfig.regime` defaults to `None`. When it is `None`:
- no new RNG stream is drawn;
- no column or node is added;
- the simulation runs the pre-v4 code path;
- the field is excluded from dumps (pydantic `exclude_if`), so config hashes don't move.

`RealizedLoadings.beta_regime` and `GroundTruth.regime` are excluded the same way.
`tests/unit/test_existing_presets_pinned.py` was committed on the unchanged generator (1bb5f14)
before any of this. It pins, for every pre-v4 preset on two seeds:
- the dataset SHA-256 (the snapshot id);
- the config dump;
- the answer key (signals, edges, nodes, Σ, loadings).

The pins hold, and the existing golden values (`test_synthetic.py`, `test_run_hash.py`) are
untouched. `GROUND_TRUTH_SCHEMA_VERSION` is now 4.

### The model (`benchmark/synthetic.py`, `RegimeParams`)
- **Chain.** A latent chain ξ_t ∈ {calm, stress}, with ξ_0 drawn from its stationary
  distribution. It is parameterised by the stress share π_s and the expected stress spell d_s:
  - P(stress → calm) = 1/d_s;
  - P(calm → stress) = (1/d_s)·π_s/(1 − π_s).
- **Markov-switching-intercept GARCH.** h_t = ω_{ξ_t} + α a²_{t−1} + β h_{t−1}, with
  ω_k = (r_k σ̄)²(1 − α − β). The residuals and the market factor have separate ratios r_k
  (`resid_vol_ratio`, `market_vol_ratio`), so cross-asset correlation rises in stress.
  - It is a single recursion, so there is no path dependence (the problem Gray 1996 and Haas,
    Mittnik & Paolella 2004 address when α and β also switch).
  - The fourth-moment condition is unchanged, because ω is bounded.
  - E[h | ξ = k] is solved exactly from m = ω∘π + (α + β)Pᵀm and stored in `RegimeTruth`. It
    is not (r_k σ̄)², because h remembers the previous regime: at the chosen values,
    E[h | stress] = 2.04e-4, against 2.56e-4 for (r_k σ̄)².
- **s_regime.** A unit-variance AR(1) that is independent of everything. Its term in R_t is
  c(ξ_t)·β_i·S_{t−1}, with c = 1 in calm and c = `stress_ratio` in stress. It is appended to
  the state as index 5.
- **ICs.** Every `ic` is scaled by the stationary-mixture sd of the non-signal return, so a
  nominal ic means the same as in v3. Population ICs are exact: S_regime's term splits into
  E[c]βS, plus (c − E[c])βS, which is uncorrelated with the state and adds Var(c)β² of noise.
  `RegimeTruth.regime_ic` holds the exact per-regime ICs of s_real and s_regime.
- **Graph.** New mechanisms:
  - `linear_regime_switching` (s_regime → asset);
  - `markov` (the chain's own lag-1 edge).

  The latent `regime` node enters each target's variance at lag 0, and a `regime` latent column
  is added (oracle only).
- **Streams:** `state/regime`, `state/s_regime`, `loadings/regime`.
- **Verdicts.** s_regime has `has_causal_edge` true and is expected KILL at `adversarial`. A kill
  at Stages 1–3 already scores as `CORRECT_KILL` with `stage_match = False`
  (`ledger/benchmark_score.py`), so no new field was needed.

**Presets.**
- `regime`: `multi_asset` plus the block below.
- `regime_null`: the same market with μ = 0 and every IC 0, for Stage 4's size scenario.

### Calibration (`scripts/regime_calibration.py`)
The calibration was fixed before any Stage 4 code ran, and uses only population quantities plus
a regime-blind projection oracle (`benchmark/regime_oracle.py`), never a stage's output.
- **Oracle.** It forecasts R_{t+1} by its population projection on one signal, so the slope is
  averaged over regimes, which is all a regime-blind linear strategy can learn. Its weights solve
  Stage 3's mean–std problem (c = 0.5/√252, gross ≤ 1, `MeanStdProblem`). Moments are estimated
  on a 10⁶-day training path, and Sharpe is measured on a separate 10⁶-day test path, by
  return-date regime.
- **Draws:** 5 loading and path draws from root seed **20261031**, used by no other study.
- **Held fixed:** everything of `multi_asset`; π_s = .2; d_s = 120 days; vol ratios 2 (residual)
  and 2.5 (market); ρ_regime = 0 (see below).

**Candidates:** ic ∈ {.10, .12, .15, .20} × stress_ratio ∈ {−1, −1.5, −2}.

**Criteria:**
- **C1:** s_regime's stationary-average `expected_t_hybrid` ≥ 4 on every asset of every draw.
  s_real's is about 4.4, so Stages 1–3 see s_regime's average edge as they see s_real's.
- **C2:** the oracle's stress-regime Sharpe is ≤ −1.0 for s_regime and ≥ +0.3 for s_real (means
  over draws).
- **C3:** the maximum population |Corr(R_t, R_{t−k})| over k = 1..20 is ≤ `multi_asset`'s, on
  every asset of every draw. The comparison is paired (same market betas and loading draws).
  This ACF is exact (`population_return_acf`) and checked against a 400k-day simulation.

**Rule:** among passing candidates, the smallest ic, then the smallest |stress_ratio|. That is,
the weakest planted signal that meets every criterion.

**Result** (`docs/results/0015-regime-calibration/calibration.json`). Only three candidates pass
C1 once loading dispersion is included:

| ic | stress_ratio | C1 min t | C3 ACF ratio | C2 s_regime stress SR | s_regime calm / overall SR | C2 s_real stress SR | pass |
|---|---|---|---|---|---|---|---|
| **.15** | **−1.0** | **4.43** | **.991** | **−1.68 (SE .13)** | **+3.14 / +1.60** | **+.95 (SE .07)** | **chosen** |
| .20 | −1.0 | 5.89 | .984 | −2.23 | +4.05 / +2.05 | +.94 | pass |
| .20 | −1.5 | 4.90 | .980 | −3.36 | +4.10 / +1.73 | +.93 | pass |
| .15 | −1.5 | 3.69 | .989 | – | – | – | C1 fails |
| ≤ .12 | any | ≤ 3.55 | ≤ .996 | – | – | – | C1 fails |

1/N's overall oracle-day Sharpe is +.31. So s_regime's regime-blind portfolio beats 1/N by a
wide margin, and Stage 3 has every reason to pass it.

**`regime` preset:** π_s .2, d_s 120, residual vol ratio 2, market vol ratio 2.5, s_regime
{ρ 0, ic .15, stress_ratio −1.0}. That gives a calm-regime IC of .176 and a stress-regime IC
of −.10 (nominal asset).

### Two changes from the plan, and why

**C3 was added.**
- **The finding.** With the planned s_regime (ρ = .9, ic .10, stress_ratio −1.5), the regime
  preset failed `no_linear_autocorrelation` on 3 of 25 asset series (seeds 0–4).
  - A persistent signal with a strong edge makes returns autocorrelated:
    Cov(R_t, R_{t−k}) = coef′AᵏΣcoef + β²ρᵏE[c_t c_{t−k}].
  - The population lag-1 ACF was about twice `multi_asset`'s.
  - C1 and C2 alone could not see this.
- **What the scan showed.** No persistent candidate (ρ ≥ .3) met C1 and C3 together, even with
  s_decay's edge removed to free up predictability.
- **Why not loosen the check instead.** Loosening the stylized-facts check, or tuning the seed,
  would hide a realism defect. C3 ties the new preset's realism to an already-accepted preset
  instead.

**s_regime has ρ = 0.** With ρ = 0, S_regime contributes nothing to the return ACF at any lag
(ρᵏ = 0), and the added predictable variance slightly lowers the others' share. So C3 holds
(.991), s_decay keeps its edge, and C1 and C2 are met.

### Validation (`tests/benchmark/test_regime_benchmark.py`, plus extensions)
Tolerances are 4 batch-means SEs (100 contiguous batches) computed in the test, because the chain
and h are strongly autocorrelated.

**Chain:**
- P is stochastic and π is stationary;
- an impossible chain is rejected;
- occupancy and mean spell length match on 2·10⁶ steps.

**MS-GARCH:** E[h | ξ = k] matches simulation, and the π-weighted values equal the mixture
variance.

**s_regime** (400k-day path):
- its marginal IC and both per-regime ICs match the GroundTruth, and the calm IC > 0 > the
  stress IC;
- the population return ACF matches simulation at lags 1–3 for `multi_asset` and `regime`.

**Graph:**
- the intervention test (`test_planted_signal.py`) now covers `regime`, including the s_regime
  switching edge;
- the causality leakage test (`test_synthetic_causality.py`) covers `regime`, redrawing the
  chain's uniforms and s_regime's innovations after the cut.

**Common random numbers:** changing `stress_ratio` leaves the features, the chain and the
latents bit-identical.

**Answer key:** the verdicts and a JSON round-trip check; `regime_null` expects every signal
killed at Stage 1.

**Stylized facts:**
- all three hold on every asset of `regime` on seed 0, the claim ADR 0003 makes for
  `multi_asset`;
- paired over seeds 0–4, `regime` fails no more checks than `multi_asset`: **1 of 75 against 6
  of 75**.

### Correction to ADR 0003
ADR 0003 says the three stylized facts hold "on every asset in multi-asset mode". The test behind
that claim (`test_stylized_facts_hold_multi_asset`) uses **seed 0 only**. On seeds 0–4,
`multi_asset` fails **6 of 75** asset-checks:
- 5 are `volatility_clustering`, mean |r| ACF .042–.050 against a threshold of .05;
- 1 is `no_linear_autocorrelation`, robust t 3.53 against 3.48.

So at T = 6000 the checker is borderline on some assets. The claim holds for seed 0, not for
every seed. No Stage 1–3 result depends on this claim. ADR 0003 carries a pointer here.

## Limitations
- **s_regime is serially uncorrelated (ρ = 0), and real regime-dependent signals are usually
  persistent** (value, momentum, carry, macro signals). It is closer to a fast signal such as
  order flow or overnight news. This is a modelling concession to C3: at this sample size a
  persistent signal strong enough to be detected and to reverse visibly autocorrelates returns
  beyond the accepted preset. Consequences:
  - Stage 4 is validated on a non-persistent fragile signal only;
  - a Ridge-CV portfolio on s_regime turns over heavily (the kill rule is on gross Sharpe);
  - Stage 2's decay test sees a signal with no autocorrelation of its own.

  Whether Stage 4 behaves the same on a persistent fragile signal is untested.
- **The edge depends on the regime through a sign flip at the same magnitude** (−1.0).
  Gradual erosion, or fragility to a regime other than high volatility, is not planted.
- **Regimes are volatility regimes only**: drift μ is the same in calm and stress. Real stress
  usually has negative drift, so 1/N is a harder benchmark here than in real stress.
- **Two regimes, exogenous to the signals.** Nothing feeds back from returns to the chain.
- **Calibration targets are oracle quantities.** Stage 3's Ridge-CV learns the average slope with
  error, so its stress Sharpe will be less extreme than the oracle's.
- **High planted ICs** (calm .176, stress −.10). Like the v3 defaults, the benchmark is
  deliberately detectable. A realistic-strength variant is not defined; Stage 3 already loses
  all power at realistic ICs (ADR 0013, finding 2).

## Alternatives considered
- **Persistent s_regime with a looser realism check**: rejected (C3 above).
- **Dropping s_decay from the regime preset to free up predictability**: insufficient for
  persistent candidates, and unnecessary at ρ = 0. s_decay is kept, so the pipeline on `regime`
  still exercises Stage 2.
- **Regime-specific α and β** (Haas, Mittnik & Paolella 2004 in full): it adds path-dependence
  questions to the *benchmark* with no benefit to the Stage 4 test. Only the intercept switches.
- **Making every preset regime-aware by default**: it would move every pinned output.
  Opt-in instead.
