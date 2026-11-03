# Changelog

All notable changes to this project are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-03

The first release. Every number below links back to an archived report in `docs/results/`
through the README; the ADRs in `docs/adr/` are authoritative.

### Stages
| # | Stage | Status | ADRs |
|---|---|---|---|
| 1 | Causal screen: PCMCI with an HC3 partial-correlation test (`hybrid` decision), Bonferroni over each signal's lags and assets, BH across signals | adopted as a kill stage | 0009, 0010 |
| 2 | Decay test: one-sided sup-split decline test with a wild-bootstrap p-value; no half-life reported (not identifiable) | adopted as a kill stage | 0011, 0012 |
| 3 | Portfolio construction: Ridge-CV mean–std portfolio, walk-forward, killed unless a one-sided Ledoit–Wolf test beats 1/N; Wasserstein DRO implemented, verified and evaluated, not selected | adopted as a kill stage | 0013, 0014 |
| 4 | Adversarial regime certification | evaluated, **rejected as a kill stage**; a reported-only stress diagnostic (high-volatility Sharpe with an interval) is kept in Stage 3 | 0015, 0016 |
| 5 | Research memo: template only, built from the run's ledger and results; the LLM narrative was cut | accepted | 0017 |

### Benchmark
- Planted-signal structural causal model with known ground truth (`s_real`, `s_decay`,
  `s_conf`, `s_noise`), presets for realistic strength, a hidden confounder and multiple
  assets, and opt-in v4 presets with regime-fragile signals (`s_regime`, `s_volfrag`,
  reported only) (ADRs 0003, 0008, 0015).
- Every stage was accepted on it by a pre-registered validation before touching real data.

### Real data
- Sources: ETF returns, Fama-French factors (fetched, not used as candidates), ALFRED macro
  vintages, and daily FRED market series with a posting-lag audit and an "exact in archive"
  rule (ADRs 0004, 0019). Immutable, content-addressed snapshots (ADR 0005) and a
  point-in-time panel (ADR 0006).
- **First real-data run** (ADR 0018): ten pre-registered signals against 13 ETFs, 2005–2025.
  All ten were killed at Stage 1 (smallest BH q .576), and a pre-registered posting-lag
  sensitivity run gave the same outcome for every signal. Ledgers, memos, the memo
  revalidation report and the snapshot manifest are archived in `docs/results/0018-*`.

### Verifier
- Checks V1–V10 rebuild the fact sheet from raw files by their own code path and accept a
  memo only if it is exactly the run's template memo (including a character check, V10).
- Validated on fresh seeds: 48 of 48 template memos accepted; 0 misses in 576 generated
  corruptions of each of 9 classes; 0 misses on three hand-written sets (368 entries).
  Revalidated on real-data-shaped runs before the real-data run.

### Reproducibility and tooling
- Each run is keyed by a hash of snapshot id, config, git SHA, package version and seed, and
  has a manifest and an append-only, hash-chained kill ledger (ADRs 0002, 0007). All
  randomness flows from named, seeded generators.
- CLI: `invariant snapshot`, `invariant run`, `invariant sensitivity`.
- `scripts/pipeline_demo.py --seed 0` runs Stages 1–3 on one benchmark market.
- Fast test suite with leakage tests for every data transform; mypy strict and ruff, enforced
  by a pre-push hook.
- One suite-wide Hypothesis profile (`tests/conftest.py`) with no per-example deadline: the
  property tests check correctness, not speed, and the default 200 ms deadline produced
  spurious failures on a loaded machine. The 30 per-test `deadline=None` overrides it made
  redundant were removed.

### Known limitations
- Every rate in the README except the real-data run is measured on synthetic markets, which
  are stationary, linear and have a known candidate set.
- Stage 1 has little power at realistic strength (.140 at IC .02, ADR 0010), so the real-data
  null is weak evidence, not proof of no effect; it cannot remove a confounder missing from
  the data (a confounded signal passes Stages 1–2 in .702 of hidden-confounder runs).
- Stage 2 does not report a half-life and has least power for a recent decay.
- Stage 3 decides a portfolio's signals jointly, so passengers pass with a real signal, and
  its test is on gross Sharpe; net of 5 bp the method ranking flips (ADR 0013).
- The memo verifier checks that a citation is in the corpus, not that the paper supports the
  claim, and relies mostly on one re-render comparison.
- Data: survivorship and hindsight in the ETF universe; Yahoo price levels are not
  point-in-time; daily FRED series have no vintage archive before 2005–2014 (by series), so
  earlier availability is an assumption; Fama-French factors are revised with every release.
- The real-data snapshot is not distributed (source terms), so the real-data run can be
  inspected through its archived ledgers but not rerun from a clone.
- Run ids include the package version, so runs made under 0.1.0 reproduce at their recorded
  git SHA, not under 1.0.0.

[1.0.0]: https://github.com/omraut888/invariant/releases/tag/v1.0.0
