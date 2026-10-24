# INVARIANT

An autonomous quant research pipeline. A candidate signal must survive five stages, any of which
can kill it: causal discovery, decay validation, distributionally robust portfolio
construction, adversarial regime certification, and a citation-backed research memo. Every
stage is validated on a planted-signal benchmark before it touches real data. Every rejection
is recorded with the numbers behind it.

Full spec: [`docs/invariant-spec.html`](docs/invariant-spec.html). Working rules: [`CLAUDE.md`](CLAUDE.md).

## Setup

```sh
uv sync                      # Python 3.11, locked dependencies
cp .env.example .env         # FRED_API_KEY is needed only to fetch ALFRED data
```

## Commands

```sh
uv run pytest -m "not slow"                       # fast suite (offline)
uv run pytest                                     # everything, including real network fetches
uv run ruff check . --fix && uv run ruff format .
uv run mypy                                       # strict, over src/, tests/ and scripts/
uv run python scripts/phase0_smoke.py --n-obs 600 # end-to-end Phase 0 smoke run
```

## Results so far

Both stages are accepted on the planted-signal benchmark (default preset, T = 6000, Wilson 95%
CIs). Real data has not been run yet.

**Stage 1, causal screen** ([ADR 0009](docs/adr/0009-stage1-ci-test-and-multiple-testing.md),
[0010](docs/adr/0010-stage1-causal-stage.md)). PCMCI with an HC3 partial-correlation test and
the `hybrid` decision test, Bonferroni over lags, then BH across signals. M = 500 runs:

| signal | truth | Stage 1 pass rate |
|---|---|---|
| s_real | true edge | .844 [.810, .873] |
| s_decay | true edge that decays | .834 [.799, .864] |
| s_conf | confounded, no edge | .034 [.021, .054] |
| s_noise | no edge | .018 [.009, .034] |

Caveats:
- At a realistic IC (.02), s_real passes only 14%. That is a power limit, so a Stage 1 kill
  means "not detectable at this sample size", not "no effect".
- The best lag is not identified.
- The effect-size bootstrap interval undercovers (as low as .86 for a nominal .95, on s_real).

**Stage 2, decay** ([ADR 0011](docs/adr/0011-stage2-decay-kill-test-and-estimators.md),
[0012](docs/adr/0012-stage2-decay-stage.md)). A one-sided sup-split test of whether the edge
declined, with a wild-bootstrap p-value; it kills at p < .05. M = 1000 runs, rates among
signals that passed Stage 1:
- s_real (no decay) is killed at **.054 [.041, .071]**. That is the false-decay rate.
- s_decay is killed at **.663 [.630, .694]**. That is the power.

Caveats:
- No half-life is reported. It is not identifiable at the benchmark's SNR, and the candidate
  estimators' intervals claimed decay 10–12% of the time when there was none.
- Power is lowest for a decay that started recently (.22–.44 when it starts at 0.8 T).
- s_decay still passes both stages in 28% of runs.
- Stage 2 needs at least 5000 rows, the only length at which it was validated.

**Structural limit: no stage fixes a hidden confounder.** When the common cause Z is
unobserved, Stage 1 passes the confounded s_conf in 73% of runs. Stage 2 kills only 4.5% of
those, because a spurious edge need not decay. So s_conf passes both stages in 70% of runs. No
observational adjustment can remove this, and the memo must state it whenever a plausible
confounder is missing from the data.

## Phase 0: foundations (done)

These components are built, tested and documented:

| Component | Where | ADR |
|---|---|---|
| Settings (`.env`) and layered YAML config | `config/` | [0001](docs/adr/0001-config-layering-and-core-contracts.md) |
| Run identity (hash of snapshot, config, git SHA, seed), manifests, named RNG streams | `core/run_hash.py` | [0002](docs/adr/0002-run-identity-and-rng-streams.md) |
| Planted-signal benchmark (SCM generator, GroundTruth, stylized-facts check) | `benchmark/` | [0003](docs/adr/0003-planted-signal-benchmark-scm.md) |
| Data sources: ETF prices, Fama-French factors, ALFRED vintages | `data/sources/` | [0004](docs/adr/0004-data-sources-and-pit-limitations.md) |
| Immutable, content-addressed snapshots | `data/snapshot.py` | [0005](docs/adr/0005-immutable-snapshots.md) |
| Point-in-time panel and forward-return targets | `data/pit.py` | [0006](docs/adr/0006-point-in-time-panel.md) |
| Kill ledger and benchmark scorer | `ledger/` | [0007](docs/adr/0007-kill-ledger-and-benchmark-scoring.md) |
| Synthetic feature source (observed signals, lag 0) | `data/sources/synthetic.py`, `benchmark/payload.py` | [0008](docs/adr/0008-synthetic-feature-source.md) |

**Smoke run.** `scripts/phase0_smoke.py` does the following:
1. Generates a synthetic market and writes it as a snapshot (prices and observed features).
2. Creates (or resumes) a run keyed by that snapshot.
3. Builds the point-in-time panel and checks it holds exactly the observed features, with no
   oracle-only latent.
4. Records **placeholder** ledger entries: a deliberately mixed set with a correct pass, a correct
   kill, a kill at the wrong stage and a missed kill.
5. Scores them against GroundTruth, and exits non-zero unless the known result comes back.
6. Prints the run manifest.

Its output says nothing about any method; it only proves the plumbing works.

**Guarantees enforced by tests.**
- No look-ahead in any data transform (`tests/leakage/`). The tests are themselves checked
  against deliberately leaky variants.
- Macro data uses ALFRED vintages from the session after release, never final revised values.
- Benchmark runs are scored only with the target timing GroundTruth assumes, on the dataset it
  describes.
- The ledger is append-only and tamper-evident.

**Known limitations** (details in the ADRs):
- The ETF universe has survivorship and hindsight bias.
- Yahoo adjusted price *levels* are not point-in-time; only returns are used.
- Fama-French factors are revised with every release; they are used only with a 42-session lag
  and a revision caveat.
- Macro features are levels only for now; within-vintage changes come next.
- Per-signal verdicts for the DRO and adversarial stages are an open design question (ADR 0007).

**Next: Stage 3.** Wasserstein distributionally robust portfolio construction.
