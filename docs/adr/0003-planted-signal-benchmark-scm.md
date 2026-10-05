# ADR 0003: Planted-signal benchmark — SCM design and limitations

Status: accepted (Phase 0)

## Context
Non-negotiable #2: every stage must recover a planted truth before it touches real data. The
benchmark is therefore the reference every later stage is scored against. A flaw in it (a leak,
a mislabelled edge, an unrealistically easy signal) would certify broken stages. It has to:
plant one signal per kill path, be realistic enough that passing it means something, export an
exact answer key, and be honest about what no method could recover from it.

## Decisions

### Timing
A value dated *t* is known at the close of *t*. `returns[t]` is the log return close(t−1) →
close(t). A signal dated *t* is scored against `returns[t+1]`. All mean edges have lag ≥ 1
(fully lagged SCM, the PCMCI setting with τ_min = 1); the only lag-0 edge is the latent market
factor. Dates are consecutive weekdays (`pd.bdate_range`), with no holiday calendar.

### Structural equations
State `X_t = (Z, S_real, S_conf, S_decay, S_noise)` is a Gaussian VAR(1):

```
Z_t   = φ Z_{t-1} + √(1-φ²) η_t
S^k_t = ρ_k S^k_{t-1} + γ_k Z_{t-1} + σ_k u^k_t
        γ_k = κ_k (1 − ρ_k φ)/φ,   σ_k² = 1 − ρ_k² − γ_k² − 2 ρ_k γ_k κ_k
        ⇒ Var(S^k) = 1, Corr(S^k_t, Z_t) = κ_k   (κ_conf = 0.8, κ_real = 0.3, else 0)

R_{i,t} = μ + b_i M_t + βz_i Z_{t-1} + βr_i S_real_{t-1} + βd_i m(t−1) S_decay_{t-1} + a_{i,t}
a_{i,t} = √h_{i,t} ε_{i,t},  ε ~ unit-variance Student-t(ν)
h_{i,t} = ω + α a²_{i,t-1} + β h_{i,t-1},  ω = σ̄²(1 − α − β)
m(s)    = 1 (s < t0),  2^{−(s−t0)/H} (s ≥ t0)       — indexed by the signal's date
```

`M_t` (multi-asset only) is a latent GARCH-t factor with betas b_i ~ U[0.7, 1.3]. Loadings are
`β = ic · sd_i · exp(δξ − δ²/2)`, where `sd_i = √(σ̄² + b_i² σ̄_M²)` is the sd of the asset's
non-signal return. `ic` therefore means the same thing in single- and multi-asset mode. (With
`sd_i = σ̄` instead, the market noise cut multi-asset expected t-stats roughly in half.) The
lognormal multiplier keeps signs and has mean 1, and it is exactly 1 when δ = 0.

Defaults (`configs/benchmark.yaml`): T = 6000, burn-in 1000, φ = 0.99, ρ = 0.9 (noise 0.95),
ic_z = ic_real = 0.06, ic_decay = 0.08, t0 = 0.6T, H = 250, ν = 6, α = 0.06, β = 0.92,
σ̄ = 1%/day. Presets: `realistic` (all ic = 0.02), `hidden_confounder`, `multi_asset`.

### Design choices that matter for validity
- **GARCH acts on the residual, not on R.** Volatility never depends on the signals, and
  signal innovations are independent of return innovations. That rules out Stambaugh (1999)
  predictive-regression bias by construction.
- **The unit-variance parameterisation is exact** (property-tested against the Lyapunov
  solution). Population moments come from `Σ = AΣAᵀ + BBᵀ`, so every population IC and partial
  correlation in the ground truth is exact rather than simulated.
- **Fourth-moment validation.** Configs must satisfy β² + 2αβ + kα² < 1, with
  k = 3(ν−2)/(ν−4). Otherwise excess kurtosis is infinite, and checking it is meaningless. At the
  defaults the condition is 0.978 and population residual excess kurtosis is 8.0.
- **Stationary initialisation.** `X_0 ~ N(0, Σ)` and `h_0 = σ̄²`, so burn-in is a safety margin,
  not a requirement (verified across 3000 seeds).
- **Common random numbers.** Each component draws from its own named `RngFactory` stream.
  Changing one `ic` leaves every other path bit-identical, so difficulty sweeps are paired
  comparisons. Adding assets leaves existing assets unchanged. Hidden-Z mode simulates
  identical paths and only drops the column.
- **Causal generator.** A hypothesis test redraws all innovations after a random cut point and
  requires every path up to the cut to be bit-identical (`tests/leakage/`).

### Ground truth semantics
`GroundTruth` (frozen, exact JSON round-trip) holds the nodes, typed lagged edges, the per-lag
adjacency, the realized loadings, Σ, the full config, the seed, and the dataset SHA-256 (usable
as `snapshot_id`). Edge mechanisms:
- `linear`: a constant linear effect.
- `linear_time_varying`: the S_decay edge; the coefficient stored is the initial one.
- `variance`: the GARCH R→R edge. It has a zero mean effect, so a linear CI test should not
  find it.

Per signal, the ground truth records:
- `expected_verdict` and `expected_kill_stage`: the **oracle** outcomes.
- `identifiable`: false for S_conf when Z is hidden.
- `marginal_ic`, `partial_ic` (given the other *observed* parents), `effective_partial_ic`
  (S_decay's coefficient replaced by its sample mean), and `expected_t = effective_partial_ic ·
  √(T−1)`. These are what let a stage test separate "wrong" from "underpowered".
- For S_decay: `coef_half_life = H` and `te_half_life_approx = H/2`. At low SNR,
  TE ≈ β²Var(S)/2σ², so transfer entropy halves twice as fast as the coefficient. Stage 2 must
  say which half-life it estimates.

Verdicts are derived from the config, not hard-coded. Setting `real.ic = 0` turns S_real into a
null with expected kill at Stage 1.

### Validation (all on fixed seeds, tolerances ≥ 4 SE)
- **Graph = simulator, by intervention.** Shock each node's innovation at step s. The nodes
  that move at s and s+1 must be exactly the lag-0 closure and the lag-1 children that the ground
  truth predicts. This is checked in single- and multi-asset mode.
- **Sample moments.** On a 200k-step path, the sample covariance matches Σ and the sample
  marginal ICs match the ground truth.
- **Decay.** In a high-SNR config (ic 0.5, H = 1000, 500-day windows, 20 seeds), the windowed
  slope ratio is 0.541 ± 0.025 at one half-life (truth 0.500) and 0.266 ± 0.022 at two
  (truth 0.250). At the default IC, a 500-day slope has relative SE ≈ 0.9, so a rolling
  regression cannot test this. That is why the test uses a high-SNR config.
- **Stylized facts.** All three hold on 5 seeds in single-asset mode and on every asset in
  multi-asset mode.
- **The S_conf trap.** At T = 20000, S_conf's marginal HAC t-stat is 7.6–11.2. Its partial
  |t| given {Z, S_real, S_decay, 5 R lags} is ≤ 1.6. With Z hidden, its partial t stays at
  5.0–8.7, matching the ground-truth bias.

### Monte Carlo: test size and power (`tests/benchmark/test_monte_carlo.py`)
Default config, T = 6000, two-sided α = 5%. The 200-seed run is the CI test. The 2000-seed run
(`INVARIANT_MC_SEEDS=2000`) is reported because 200 seeds resolve a size only to about ±3.5
points.

| Test (null true unless noted) | 200 seeds | 2000 seeds [95% Wilson CI] |
|---|---|---|
| R_{t+1} on S_noise_t, naive OLS t | 0.070 | **0.079** [0.068, 0.092] |
| R_{t+1} on S_noise_t, Newey–West t (9 lags) | 0.070 | **0.071** [0.061, 0.083] |
| S_conf partial (parents + R lags), OLS | 0.085 | 0.053 [0.044, 0.064] |
| S_conf partial, HAC | 0.070 | 0.053 [0.044, 0.064] |
| S_conf marginal, HAC (trap; power) | 1.000 | 0.989 |
| S_real partial, HAC (power; predicted 0.993) | 0.990 | 0.990 |
| S_decay partial, HAC (power; predicted 0.983) | 0.980 | 0.982 |

**Finding on S_noise.** A naive t-test of returns on the persistent but unrelated S_noise rejects
at about 8%, not 5%. The over-rejection is mild but statistically clear at 2000 seeds.

The cause is not Granger–Newbold in the usual sense, because returns are near white noise. In
the univariate regression, the error contains the omitted persistent predictable part of returns
(βz Z + βr S_real + βd S_decay, with AR coefficients of 0.9–0.99). A persistent regressor times a
persistent error is a miniature Granger–Newbold setting. The exact asymptotic variance of the
t-stat is

  `Var(t) = [Var(e) + 2 Σ_k ρ^k cᵀA^kΣc] / Var(e)`, with `Σ_k (ρA)^k = (I − ρA)^{-1} − I`.

That gives Var(t) = 1.28 and a predicted size of **8.3%**, inside the empirical CI. Newey–West
with the rule-of-thumb bandwidth (9 lags) only partly corrects it (7.1%), because Z's
persistence (φ = 0.99, half-life ~69 days) needs a far longer bandwidth. Once the true parents
are conditioned on, the error is white and the size is correct (5.3%).

Implications:
1. S_noise is a weak spurious-regression trap, and the ground truth does not claim it is a
   strong one.
2. The distortion grows with the share of predictable, persistent variance in returns. A stage
   that screens signals univariately will over-admit nulls, so FDR control has to account for it.
3. Default-bandwidth HAC is not a fix when the omitted component is highly persistent.

At 200 seeds, the 8.5% size of the OLS partial test was noise. At 2000 seeds it is 5.3%.

## Alternatives considered
- *Contemporaneous Z→S_conf edge*: equivalent in substance, but it adds lag-0 orientation
  problems that PCMCI (as opposed to PCMCI+) does not address. Rejected in favour of a fully
  lagged graph.
- *GARCH on total returns*: volatility would then depend on the signals, and the R→R edge would
  carry mean information through the signal terms. Rejected.
- *Asserting that S_noise is a strong trap*: unsupported by the theory. It is measured instead
  (see the table above).
- *Rolling-regression decay test at the default SNR*: uninformative (relative SE ≈ 0.9). A
  deterministic path check plus a high-SNR statistical check is used instead.
- *statsmodels for OLS/HAC*: a heavy dependency for one estimator. `benchmark/diagnostics.py` is
  about 100 lines, unit-tested against scipy and White HC0, and cross-checked to
  rtol 1e-8 against statsmodels (a dev-only dependency; a test forbids importing it from `src/`).
- *Per-asset cross-sectional signals*: deferred. The common time-series signals with per-asset
  loadings exercise DRO without expanding the scope of the SCM.
- *Loadings of (1 + δξ)*: they can flip sign. The mean-one lognormal multiplier is used instead.

## Consequences and known limitations
- **Easier than reality in these ways:** linear mean effects; Gaussian signal innovations;
  independent signal and return innovations (no Stambaugh bias); symmetric GARCH (no leverage
  effect); no jumps, regime switches or structural breaks other than the planted decay; constant
  coefficients otherwise; no transaction costs; no holiday calendar.
- **Assumption violations by design:** S_decay makes the conditional mean non-stationary,
  violating PCMCI's stationarity assumption. GARCH violates the i.i.d. residual assumption
  behind the classical ParCorr t-test, though at these settings the effect on the partial test's
  size is not detectable (5.3%).
- **Hidden Z:** no observational method can kill S_conf, because no adjustment set exists.
  Stage tests must report this as a documented false positive, not as a failure.
- **Power, not correctness, at realistic ICs:** at ic = 0.02 and T = 6000, expected t ≈ 1.5 for
  S_real. Failing to detect it there is the honest outcome. The generator logs
  `benchmark_low_power` whenever an edge's expected t is below 3.
- **Approximations in the ground truth:** `expected_t` uses a stationary, low-SNR
  approximation. For S_decay it is close (predicted power 0.983 vs 0.982 empirical).
  `te_half_life_approx = H/2` holds only at low SNR.
- **Pinned outputs:** a golden-hash test pins the generator. Changing stream names, recursions or
  the content-hash encoding is a deliberate, versioned decision (bump
  `GROUND_TRUTH_SCHEMA_VERSION` when the schema changes).
- **Stylized-facts caveat:** at ν = 6 the 8th moment is infinite, so sample kurtosis has no CLT.
  The kurtosis threshold is loose (> 1), and no p-value is reported for it.
