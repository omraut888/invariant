# INVARIANT: design overview

INVARIANT takes candidate trading signals and passes each one through five stages. Each stage
targets one way a signal can look good and still be worthless. Stages 1–3 can kill the signal;
Stage 4 was evaluated and not adopted as a kill stage (below), and Stage 5 writes a
template-only memo.
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
| 4 | Adversarial regime certification | **Regime fragility**: an edge that holds only in the regime it was found in | evaluated, **not adopted** as a kill stage; reported-only stress diagnostic |
| 5 | Research memo | **Unsupported claims**: a recommendation stronger than the evidence | built, accepted on the benchmark (template only; LLM narrative cut) |

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

### 4. Adversarial regime certification ([ADR 0015](adr/0015-benchmark-v4-regime-signal.md), [0016](adr/0016-stage4-adversarial-study.md)): evaluated, not adopted
- **Intent.** Kill a strategy that is profitable on average but fails under the worst plausible
  regime mix.
- **Why it was not adopted.** The benchmark gained regime-fragile signals (ADR 0015), and the
  design was taken as far as a population feasibility check (ADR 0016). Four findings ended it:
  - latent regimes are not identifiable from returns at this sample size;
  - an observable high-volatility state is only a noisy proxy for them;
  - three bootstrap path generators failed to reproduce the stylized facts;
  - the plausible set for the state chain is too narrow: even with perfectly known moments, the
    worst case of a signal whose edge reverses in high volatility stays at Sharpe +0.90.

  At ~24 years of daily data, plausible variation in high-volatility exposure is too small to
  overturn an average-profitable strategy, and a plain "loses money in stress" rule would
  penalise ordinary market exposure, not fragility.
- **What remains.** A reported-only **stress diagnostic** in Stage 3's ledger metrics: Ridge-CV's
  and 1/N's Sharpe in the vol21/q=.70 high-volatility state, with a block-bootstrap interval. It
  never decides.

### 5. Research memo ([ADR 0017](adr/0017-stage5-research-memo.md))
- **Method.** A fact sheet is built from the run directory only (the hash-chained ledger, the
  three stage results, the manifest, and archived benchmark validation rates); GroundTruth is
  never read. The template memo is a document of segments in which every digit lies inside an
  atom rendering one fact with its subject and label. The recommendation follows a fixed rule
  ("candidate for paper trading", "not decided", "against trading"), and the limits section
  holds exactly the mandatory statements whose triggers fire: for example, that Stage 1
  cannot rule out an unobserved confounder, that Stage 3 used ridge regularization (Wasserstein
  DRO was evaluated, not selected), and that Stage 4 was evaluated and not adopted, so the
  stress diagnostic is a description, not a certification. Citations come from a curated
  26-entry corpus.
- **Verifier.** Checks V1–V10 rebuild the fact sheet, re-read the raw files by their own code
  path, and reject any memo that is not exactly the run's template memo: structure and every
  block equal to the re-render (V1), atoms and numbers equal to their raw sources (V2, V3),
  verdicts and the recommendation recomputed from the raw ledger (V5), the triggered statement
  set (V6), corpus citations (V7), `memo.md` equal to the rendering (V9), and plain characters
  (V10: NFKC, no format characters, no non-ASCII letters). A rejected memo is never patched.
- **The LLM narrative was cut.** Its prose was to be policed by word lists. An independent red
  team's hand-written set got 7 of 126 corruptions past them (claims in new wording, Unicode
  tricks) and saw 8 of 13 harmless sentences rejected; word-list verification cannot police
  free prose, so the verifier rejects any narrative paragraph. A narrative that only chooses
  among verified atoms is in the backlog.
- **Validated** on fresh seeds: 48 of 48 template memos accepted; 0 misses in 576 generated
  corruptions of each of 9 classes; 0 misses on three hand-written sets (368 corruptions, the
  last written by a new independent agent after the change).
- **Cannot check** that a cited paper supports the claim it is attached to. Most template
  tampering is caught by the re-render comparison alone, so a bug shared by the template and
  that re-render would pass; a raw-path second check is in the backlog.

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
| Benchmark v4: regime-switching GARCH, s_regime, s_volfrag | `benchmark/synthetic.py`, `benchmark/regime_oracle.py` | [0015](adr/0015-benchmark-v4-regime-signal.md) |

Paths are under `src/invariant/`.

**The benchmark.** A structural causal model with four candidate signals whose truth is
known: `s_real` (a constant true edge), `s_decay` (a true edge that decays), `s_conf`
(confounded by a common cause `z`, no edge) and `s_noise` (no edge). Presets vary the signal
strength, the volatility model, whether `z` is observed, and the number of assets. The opt-in
v4 presets ([ADR 0015](adr/0015-benchmark-v4-regime-signal.md)) add a fifth signal:
- `s_regime`, whose edge reverses in a latent stress regime;
- or `s_volfrag`, whose edge reverses when market volatility is high.

Both are reported only, the evidence behind Stage 4's rejection. Every pre-v4 preset is
byte-identical (pinned by a test).

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
