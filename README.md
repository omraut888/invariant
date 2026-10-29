# INVARIANT

**Status:** Stages 1–3 (causal screen, decay test, portfolio construction) are complete and
validated on a planted-signal benchmark. Stage 4 (adversarial stress) is targeted for early
November 2026, and Stage 5 (research memo) and the first real-data run for mid-November 2026.
No result below comes from real market data yet.

INVARIANT is a research pipeline for trading signals that tries to kill every candidate and
reports what survives. Most backtests show the signals that worked. INVARIANT puts each
candidate through five stages, each built to catch one specific way a signal can look good
and be worthless. Every pass and every kill is written to a tamper-evident ledger with the
numbers behind it. Each stage was validated on synthetic markets with planted signals, where
the right answer is known, and each method was chosen by a study whose rules were committed
before it ran. Where a method failed, or a stage has a blind spot, this README says so.

**At a glance** (synthetic benchmark, 95% intervals):
- **Kills are mostly right, not always.** Pooled over Stages 1–3, kill precision is .927
  [.918, .935] and recall .917 [.908, .925]
  ([report](docs/results/0014-portfolio-stage-validation/report.json)).
- **Ridge regularization beat Wasserstein DRO** in a pre-registered study of portfolio
  methods, on gross Sharpe: 1.560 against 1.525
  ([report](docs/results/0013-dro-method-study/report.json)).
- **At realistic signal strength the pipeline kills the strategy.** With ICs of .02, no
  portfolio method beats 1/N in more than .030 of runs
  ([report](docs/results/0013-dro-method-study/report.json)), and Stage 1 detects the true
  edge only .140 of the time ([report](docs/results/0010-causal-stage-validation/report.json)).
- **A hidden confounder gets through.** When the common cause is not in the data, the
  confounded signal passes Stages 1–2 in .702 [.673, .730] of runs
  ([report](docs/results/0012-decay-stage-validation/report.json)). No observational stage
  can fix that.

```
 candidate signals
        │
        ▼
 1  Causal screen ─────────── spurious signal: a third variable drives both feature and return
        │
        ▼
 2  Decay test ────────────── silent decay: a real edge that is being arbitraged away
        │
        ▼
 3  Portfolio construction ── estimation error: weights that only work in sample
        │
        ▼
 4  Adversarial stress ─────── regime fragility: an edge that holds in one regime only   (in progress)
        │
        ▼
 5  Research memo ─────────── unsupported claims: a case stronger than the evidence     (in progress)
        │
        ▼
 survivors, plus a ledger entry for every kill
```

More detail: [docs/design.md](docs/design.md) and the decision records in
[docs/adr/](docs/adr/).

## Results on the benchmark

The benchmark is a structural causal model with four candidate signals whose truth is known:
`s_real` (a constant true edge), `s_decay` (a true edge that decays), `s_conf` (correlated with
returns only through a common cause `z`) and `s_noise` (nothing). Default preset: T = 6000
daily observations. Each rate is over M independent synthetic markets, with a Wilson 95%
interval. Each number links to the archived report it comes from.

### Stage 1, causal screen
PCMCI with an HC3 partial-correlation test, Bonferroni over lags and Benjamini–Hochberg across
signals ([ADR 0009](docs/adr/0009-stage1-ci-test-and-multiple-testing.md),
[0010](docs/adr/0010-stage1-causal-stage.md)). Pass rates over M = 500
([report](docs/results/0010-causal-stage-validation/report.json)):

| signal | truth | passes Stage 1 |
|---|---|---|
| s_real | true edge | .844 [.810, .873] |
| s_decay | true edge that decays | .834 [.799, .864] |
| s_conf | confounded, no edge | .034 [.021, .054] |
| s_noise | no edge | .018 [.009, .034] |

- **At realistic strength it has little power.** With an IC of .02, s_real passes only
  .140 [.112, .173] of the time
  ([report](docs/results/0010-causal-stage-validation/report.json)). A Stage 1 kill means "not
  detectable at this sample size", not "no effect".
- **It cannot handle a hidden confounder.** When `z` is not in the data, s_conf passes Stage 1
  in .730 [.689, .767] of runs
  ([report](docs/results/0010-causal-stage-validation/report.json)). Stage 2 kills only .045
  [.032, .062] of those, because a spurious edge need not decay, so s_conf passes both stages in
  .702 [.673, .730] of runs ([report](docs/results/0012-decay-stage-validation/report.json)). No
  observational method can remove this. The memo will have to say so whenever a plausible
  confounder is missing from the data.
- The best lag is not identified.

### Stage 2, decay test
A one-sided sup-split test of whether the edge declined, with a wild-bootstrap p-value; it
kills at p < .05 ([ADR 0011](docs/adr/0011-stage2-decay-kill-test-and-estimators.md),
[0012](docs/adr/0012-stage2-decay-stage.md)). M = 1000, rates among signals that passed
Stage 1 ([report](docs/results/0012-decay-stage-validation/report.json)):
- s_real (no decay) is killed in .054 [.041, .071] of runs. That is the false-decay rate.
- s_decay is killed in .663 [.630, .694] of runs. That is the power.

Caveats:
- **No half-life is reported.** The project plan called for one, but at the benchmark's
  signal-to-noise ratio it is not identifiable, and every candidate estimator's interval
  claimed decay in 10–12% of runs with no decay
  ([report](docs/results/0011-decay-study/report.json)).
- Power is lowest for a decay that started recently.
- s_decay still passes both stages in 28% of runs.

### Stage 3, portfolio construction
**Ridge regularization beat Wasserstein DRO.** A pre-registered study compared Markowitz, two
Wasserstein distributionally robust (DRO) variants and a cross-validated ridge shrinkage of
the covariance (Ridge-CV). All four were mean–std portfolios with gross exposure ≤ 1
([ADR 0013](docs/adr/0013-stage3-dro-method-study.md),
[report](docs/results/0013-dro-method-study/report.json)).
- Every arm met the false-pass gate on two null markets.
- Ridge-CV had the best Sharpe: 1.560, against 1.525 for the better DRO arm. The interval of the
  difference excludes 0: .034 [.021, .047]. Ridge-CV was selected. Wasserstein DRO was
  implemented, verified against exact solutions and evaluated, and it was not selected.
- **After costs the ranking flips.** Net of 5 bp the DRO forecast-confidence arm is best
  (1.303 against 1.226), because it trades far less. Selection was pre-registered on gross
  Sharpe, which in hindsight was the wrong metric. A fresh-seed, net-of-cost confirmation study
  is in the backlog.
- **At realistic strength every arm fails.** With ICs of .02, no arm beats 1/N in more than
  .030 [.014, .064] of runs, so the stage kills the strategy. That is a power limit of the
  signals, not a defect of the test.

The stage itself ([ADR 0014](docs/adr/0014-stage3-portfolio-stage.md)) runs Ridge-CV
walk-forward. It kills the portfolio unless a one-sided Ledoit–Wolf test shows it beating 1/N
(p < .05). On fresh seeds, M = 1273
([report](docs/results/0014-portfolio-stage-validation/report.json)):
- False-pass rate on two null markets at 5000 rows: .049 [.039, .063] and .043 [.033, .056].
- s_real passes in .828 [.806, .848] of the runs where it reaches Stage 3.

Caveats:
- **Passengers get through.** Stage 3 passes or kills a portfolio's signals together, so a
  worthless signal that reaches the stage with s_real passes too. s_noise passes all three
  stages in .015 [.010, .023] of runs and s_conf in .023 [.016, .033]
  ([report](docs/results/0014-portfolio-stage-validation/report.json)). A per-signal marginal
  Sharpe, reported but not used to decide, does separate them: its interval excludes 0 for
  s_noise in 4% of those runs, against 88% for s_real. A per-signal rule built on it is in the
  backlog.
- **The test is on gross Sharpe.** On the null markets Ridge-CV loses about .31 of Sharpe to
  costs, so a false pass would lose money.

### One run: what it killed and why
`scripts/pipeline_demo.py --seed 0` runs Stages 1–3 on one benchmark market (T = 6000). Seed 0
was fixed before the demo's first run, not picked afterwards, and a test pins this output. On
this seed every signal meets its expected fate (3 correct kills, 1 correct pass). Across seeds
the pipeline is not always right: pooled over the three stages, kill precision is .927
[.918, .935] and recall .917 [.908, .925] (M = 1273,
[ADR 0014 report](docs/results/0014-portfolio-stage-validation/report.json); the intervals
are optimistic, because the four signals in a run share one dataset and are not independent).

```
s_real: should pass every stage; passed causal, decay, portfolio
  causal    pass  BH q 3.23e-06 vs alpha 0.05; partial r 0.0719 [0.0471, 0.0982] at lag 1
  decay     pass  decline p 0.496 vs alpha 0.05; coefficient before/after the break 0.00118 / 0.000747 (break at 0.192 T)
  portfolio pass  Ridge-CV Sharpe 1.1 (net 0.843) vs 1/N 0.183; LW p 0.001 vs 0.05; 1 member(s); B 0.496 [0.101, 0.889] vs intercept_only
s_conf: should be killed at causal; KILLED at causal (causal.no_significant_link)
  causal    kill  BH q 0.107 vs alpha 0.05; partial r 0.0388 [0.0162, 0.0651] at lag 1
s_decay: should be killed at decay; KILLED at decay (decay.edge_declining)
  causal    pass  BH q 0.00141 vs alpha 0.05; partial r 0.0537 [0.0284, 0.0778] at lag 1
  decay     kill  decline p 0.001 vs alpha 0.05; coefficient before/after the break 0.00154 / -6.36e-05 (break at 0.396 T)
s_noise: should be killed at causal; KILLED at causal (causal.no_significant_link)
  causal    kill  BH q 1 vs alpha 0.05; partial r -0.0199 [-0.0459, 0.00632] at lag 5
```

s_decay is a real edge, so Stage 1 passes it (q = .0014). Stage 2 then finds that its
coefficient fell from .00154 to about zero, with p = .001, and kills it as
`decay.edge_declining`. The break point is descriptive only; the test does not estimate when
the decay began.

Two more things in this run. s_conf's partial-correlation interval excludes 0, but the kill
comes from the pre-registered test, and Stage 1's bootstrap intervals are known to undercover
(ADR 0010). s_real's Stage 3 net Sharpe (0.843) is what remains after 5 bp costs.

## How the results were produced

Every method choice and every acceptance decision went through the same trail:
1. An ADR states the rule, the gates, how the number of seeds M was derived (with the
   probability of passing at the expected rate and at the failure rate), and the tie-break.
   It is committed and pushed before any full run.
2. The run executes from a git worktree at that commit, so later edits cannot change the code
   being run. Its report records the git SHA, and each run is keyed by a hash of the data
   snapshot, config, SHA and seed.
3. Results go in a separate section of the ADR; the pre-registration text is not edited.
   Departures from a rule, and bugs fixed after launch, are recorded with their reasons.
4. The reports behind every ADR are archived in [docs/results/](docs/results/).

Other rules: no look-ahead (every data transform has a leakage test), macro data uses ALFRED
vintages rather than revised values, all randomness is explicitly seeded, every estimate has an
interval, and mypy (strict), ruff and the fast test suite must pass before any push (a
pre-push hook enforces this).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). From a fresh clone:

```sh
git clone https://github.com/omraut888/invariant.git && cd invariant
uv sync --locked
uv run python scripts/pipeline_demo.py --seed 0    # the run above, about 30 s
```

Then:

```sh
uv run pytest -m "not slow"                        # fast suite, offline
uv run python -m scripts.portfolio_stage_validation --n-jobs 8   # ADR 0014 validation, ~14 h
```

Each stage's study and validation has a script in `scripts/` and a config in `configs/`.
Benchmark runs need no API key. `FRED_API_KEY` in `.env` (see `.env.example`) is needed only to
fetch real ALFRED data. Working rules for contributors are in [CLAUDE.md](CLAUDE.md).

## Roadmap

- **Stage 4, adversarial stress** (early November 2026): search for the worst plausible market
  path under a realism constraint. The method will be chosen by a pre-registered study.
- **Stage 5, research memo** (mid-November 2026): a citation-backed memo for or against each
  strategy, built from the ledger, that states each stage's limits.
- **Real data** (mid-November 2026): ETFs, Fama-French factors and ALFRED macro vintages
  through all five stages.

**Backlog**, each to be decided by its own pre-registered study:
- A net-of-cost confirmation of the Stage 3 method, at a cost level justified in advance
  (ADR 0013).
- A per-signal Stage 3 rule built on the marginal-Sharpe diagnostic, to stop passengers
  (ADR 0014).
- A calibrated confidence radius for the DRO forecast arm, before it is reconsidered
  (ADR 0013).
- A calibrated half-life interval (ADR 0011).
- The decline test's size and power at T < 6000, which would allow a lower `min_obs`
  (ADR 0011).
- The decline test's small excess size when a strong decaying co-parent is present (ADR 0011).

## Limitations

- Everything above is measured on synthetic markets. Real markets violate the benchmark's
  assumptions (stationarity, linear effects, a known set of candidates).
- The ETF universe has survivorship and hindsight bias. Yahoo adjusted price levels are not
  point-in-time, so only returns are used. Fama-French factors are revised with every release
  and are used with a 42-session lag.

## License

[MIT](LICENSE)
