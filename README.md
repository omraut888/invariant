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

## Phase 0 status: foundations

No stage logic exists yet. These components are built, tested and documented:

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

**Next: Phase 1.** The causal discovery stage, validated on the benchmark through `score_run`
with `stages=[causal]`, with surrogate significance tests and FDR control across candidates.
