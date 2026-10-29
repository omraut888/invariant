# INVARIANT: design overview

INVARIANT takes candidate trading signals and passes each one through five stages. Each stage
targets one way a signal can look good and still be worthless, and each can kill the signal.
Every decision, pass or kill, goes into a tamper-evident ledger with the numbers behind it.
Every stage is validated on a planted-signal benchmark, whose ground truth is known, before it
is used on real data.

The decisions behind this design are in the [architecture decision records](adr/). This page
is a map of them. An early design spec existed in the scaffold commit (`707e38b`); it has been
superseded by the ADRs, and where they differ the ADRs are authoritative.

## The five stages

| # | Stage | Failure mode it targets | Status |
|---|---|---|---|
| 1 | Causal screen | **Spurious signal**: the feature predicts returns only because a third variable drives both | built, accepted on the benchmark |
| 2 | Decay test | **Silent decay**: a real edge that is being arbitraged away | built, accepted on the benchmark |
| 3 | Portfolio construction | **Estimation error**: weights that look optimal in sample and fail out of sample | built, accepted on the benchmark |
| 4 | Adversarial regime certification | **Regime fragility**: an edge that holds only in the regime it was found in | design |
| 5 | Research memo | **Unsupported claims**: a recommendation stronger than the evidence | design |

### 1. Causal screen ([ADR 0009](adr/0009-stage1-ci-test-and-multiple-testing.md), [0010](adr/0010-stage1-causal-stage.md))
- **Method.** PCMCI over the signals, the asset returns and the observed conditioning variables.
  Each signal's link to returns is tested with an HC3 partial-correlation test (the `hybrid`
  decision), with Bonferroni over lags and Benjamini–Hochberg across signals.
- **Kills** a signal with no significant conditional link to any return.
- **Cannot detect** a confounder that is missing from the data. On the benchmark's
  `hidden_confounder` preset the confounded signal passes in most runs, and no observational
  adjustment can fix that. At realistic signal strength the test has little power, so a kill
  means "not detectable at this sample size", not "no effect". The best lag is not identified.

### 2. Decay test ([ADR 0011](adr/0011-stage2-decay-kill-test-and-estimators.md), [0012](adr/0012-stage2-decay-stage.md))
- **Method.** A one-sided sup-split test (Andrews 1993) of whether the coefficient on Stage 1's
  best link declined, with a fixed-regressor wild-bootstrap p-value (Hansen 2000).
- **Kills** a signal whose edge declined (p < .05).
- **Does not report a half-life.** At the benchmark's signal-to-noise ratio, the Cramér–Rao
  bound makes the half-life unidentifiable, and the candidate estimators' intervals claimed
  decay when there was none.
- **Cannot detect** a decay that started recently with much power, or a spurious edge that does
  not decay.

### 3. Portfolio construction ([ADR 0013](adr/0013-stage3-dro-method-study.md), [0014](adr/0014-stage3-portfolio-stage.md))
- **Method.** Mean–std portfolios on linear forecasts from the surviving signals, gross
  exposure ≤ 1, refit walk-forward. A pre-registered study compared Markowitz, two Wasserstein
  DRO variants and a ridge shrinkage of the covariance. **Ridge-CV was selected**; Wasserstein
  DRO was implemented, verified and evaluated, and not selected.
- **Kills** the portfolio (all its signals together) unless it beats 1/N on a one-sided
  Ledoit–Wolf Sharpe-difference test. A leave-one-out marginal Sharpe per signal (diagnostic
  B) is reported only.
- **Cannot detect** a passenger: a worthless signal in a portfolio with a real one passes with
  it. The test is on gross Sharpe, so a pass does not imply a positive net return after costs.

### 4. Adversarial regime certification (design)
The intent is to search for the worst plausible market path for the surviving portfolio, under
a realism constraint that stops the search from winning with implausible paths. The method
(generator, adversary, realism test) will be chosen by a pre-registered study, as Stages 1–3
were, and validated on the benchmark before it is used.

### 5. Research memo (design)
A memo per strategy that argues for or against it, with citations, built only from the ledger
and the stage results. It must state each stage's limits as they apply: for example, that
Stage 1 cannot rule out an unobserved confounder, and that Stage 3 used ridge regularization
(Wasserstein DRO was evaluated, not selected).

## Foundations

| Component | Where | ADR |
|---|---|---|
| Settings (`.env`) and layered YAML config | `config/` | [0001](adr/0001-config-layering-and-core-contracts.md) |
| Run identity (hash of snapshot, config, git SHA, seed), manifests, named RNG streams | `core/run_hash.py` | [0002](adr/0002-run-identity-and-rng-streams.md) |
| Planted-signal benchmark (SCM generator, GroundTruth, stylized-facts check) | `benchmark/` | [0003](adr/0003-planted-signal-benchmark-scm.md) |
| Data sources: ETF prices, Fama-French factors, ALFRED vintages | `data/sources/` | [0004](adr/0004-data-sources-and-pit-limitations.md) |
| Immutable, content-addressed snapshots | `data/snapshot.py` | [0005](adr/0005-immutable-snapshots.md) |
| Point-in-time panel and forward-return targets | `data/pit.py` | [0006](adr/0006-point-in-time-panel.md) |
| Kill ledger and benchmark scorer | `ledger/` | [0007](adr/0007-kill-ledger-and-benchmark-scoring.md) |
| Synthetic feature source (observed signals, lag 0) | `data/sources/synthetic.py`, `benchmark/payload.py` | [0008](adr/0008-synthetic-feature-source.md) |

Paths are under `src/invariant/`.

**The benchmark.** A structural causal model with four candidate signals whose truth is
known: `s_real` (a constant true edge), `s_decay` (a true edge that decays), `s_conf`
(confounded by a common cause `z`, no edge) and `s_noise` (no edge). Presets vary the signal
strength, the volatility model, whether `z` is observed, and the number of assets.

**Guarantees enforced by tests.**
- No look-ahead in any data transform (`tests/leakage/`). The leakage tests are themselves
  checked against deliberately leaky variants.
- Macro data uses ALFRED vintages from the session after release, never final revised values.
- Benchmark runs are scored only with the target timing GroundTruth assumes, on the dataset it
  describes.
- All randomness flows from explicitly seeded generators; a run is keyed by its snapshot,
  config, git SHA and seed.
- The ledger is append-only and hash-chained.

**Data limitations.**
- The ETF universe has survivorship and hindsight bias.
- Yahoo adjusted price *levels* are not point-in-time; only returns are used.
- Fama-French factors are revised with every release; they are used only with a 42-session
  lag and a revision caveat.
- Macro features are levels only for now.
