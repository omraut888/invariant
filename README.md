# INVARIANT

**Status (v1.0.0):**
- **Stages 1–3** (causal screen, decay test, portfolio construction) are adopted as kill stages,
  each accepted on a planted-signal benchmark before use on real data.
- **Stage 4** (adversarial stress) was evaluated and **rejected as a kill stage**
  ([ADR 0016](docs/adr/0016-stage4-adversarial-study.md)); a reported-only stress diagnostic
  is kept in Stage 3.
- **Stage 5** (research memo) is accepted, **template only**: a memo built from the ledger and
  stage results, and a verifier that rejects any memo that is not exactly that memo
  ([ADR 0017](docs/adr/0017-stage5-research-memo.md)). The planned LLM narrative was cut,
  because word-list verification could not police free prose.
- **The first real-data run is reported**
  ([ADR 0018](docs/adr/0018-first-real-data-run.md)): all ten pre-registered candidates were
  killed at Stage 1 ([below](#results-on-real-data)). Every other result in this README is from
  the synthetic benchmark.

INVARIANT is a research pipeline for trading signals that tries to kill every candidate and
reports what survives. Most backtests show the signals that worked. INVARIANT puts each
candidate through its stages, each built to catch one specific way a signal can look good
and be worthless. Every pass and every kill is written to a tamper-evident ledger with the
numbers behind it. Each stage was validated on synthetic markets with planted signals, where
the right answer is known, and each method was chosen by a study whose rules were committed
before it ran. Where a method failed, or a stage has a blind spot, this README says so.

**At a glance** (synthetic benchmark, 95% intervals):
- **Kills are mostly right, not always.** Pooled over Stages 1–3, kill precision is .927
  [.918, .935] and recall .917 [.908, .925]
  ([report](docs/results/0014-portfolio-stage-validation/report.json)).
- **Ridge regularization beat Wasserstein DRO** in a pre-registered study of portfolio
  methods, on gross Sharpe: 1.560 against 1.526 for the better DRO arm
  ([report](docs/results/0013-dro-method-study/report.json)).
- **At realistic signal strength the pipeline kills the strategy.** With ICs of .02, no
  portfolio method beats 1/N in more than .030 of runs
  ([report](docs/results/0013-dro-method-study/report.json)), and Stage 1 detects the true
  edge only .140 of the time ([report](docs/results/0010-causal-stage-validation/report.json)).
- **A hidden confounder gets through.** When the common cause is not in the data, the
  confounded signal passes Stages 1–2 in .702 [.673, .730] of runs
  ([report](docs/results/0012-decay-stage-validation/report.json)). No observational stage
  can fix that.
- **Free prose could not be verified, so the memo has none.** A word-list verifier for an LLM
  narrative missed 7 of 126 corruptions written by an independent agent, and rejected 8 of its
  13 harmless sentences ([report](docs/results/0017-memo-validation-run1/report.json)). The
  template-only verifier then missed none of 368 hand-written and 5184 generated corruptions,
  and accepted all 48 template memos
  ([report](docs/results/0017-memo-validation-deviation1/report.json)).
- **On real data, nothing survived Stage 1.** Ten macro and market signals from the
  literature, 2005–2025, against 13 ETFs: the smallest BH q was .576
  ([ledger](docs/results/0018-real-data-run-primary/kill_ledger.jsonl)). At realistic
  strength Stage 1 has power of about .14, so this is a weak negative, not proof of no effect.

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
 4  Adversarial stress ─────── regime fragility: evaluated, not adopted (diagnostic only)
        │
        ▼
 5  Research memo ─────────── unsupported claims: a case stronger than the evidence
        │
        ▼
 survivors, plus a ledger entry for every kill
```

More detail: [docs/design.md](docs/design.md) and the decision records in
[docs/adr/](docs/adr/).

## Results on real data

**This describes what the pipeline did on one dataset. It is not a trading recommendation, and a
kill is not evidence that a signal has no effect.**

The first real-data run ([ADR 0018](docs/adr/0018-first-real-data-run.md)) fixed ten candidate
signals and every setting before any stage saw market data. Each signal is one common series
(12-1 momentum and realised volatility of SPY, the variance risk premium, the term and default
spreads, the relative bill rate, and four ALFRED macro series taken as their real-time
vintages), tested against daily returns of 13 ETFs (SPY, nine sector SPDRs, TLT, IEF, GLD),
2005-01-04 to 2025-12-31 (5,282 sessions). Snapshot `930504ace5464752`, seed 20261112, commit
ae65781. Before the run, the memo's real-data path was revalidated on fresh seeds and passed
every gate ([report](docs/results/0018-memo-revalidation/report.json)).

**What was killed, and where.** All ten were killed at Stage 1 (causal screen) as
`causal.no_significant_link`. None reached Stage 2 or 3, so there is no portfolio, Sharpe, DSR
or stress diagnostic, and the memo's recommendation is "against trading". This is the outcome
the ADR predicted before the run.

| signal | family p (Bonferroni, 65 tests) | BH q (10 signals) | best link | partial r [interval] ⚠ |
|---|---|---|---|---|
| `payrolls_growth` | .058 | .576 | XLI, lag 1 | −.050 [−.072, −.007] |
| `unrate_change` | .403 | 1 | XLI, lag 1 | −.039 [−.064, −.016] |
| `inflation` | .588 | 1 | XLK, lag 4 | −.048 [−.080, −.013] |
| `ip_growth` | 1 | 1 | XLI, lag 3 | .040 [−.000, .070] |
| `rel_bill` | 1 | 1 | XLF, lag 3 | .041 [.005, .076] |
| `rv_spy` | 1 | 1 | XLK, lag 3 | .032 [.002, .062] |
| `term_spread` | 1 | 1 | GLD, lag 3 | −.026 [−.050, −.003] |
| `vrp` | 1 | 1 | TLT, lag 3 | −.041 [−.077, −.011] |
| `default_spread` | 1 | 1 | XLU, lag 3 | −.023 [−.056, .013] |
| `tsmom_spy` | 1 | 1 | XLF, lag 4 | −.017 [−.041, .009] |

> **⚠ The bracketed intervals are not evidence of an effect.** Each is for the best of a
> signal's 65 tests (13 assets × 5 lags), chosen after the fact, and Stage 1's bootstrap
> intervals already ran narrow on the benchmark (ADR 0010). That most exclude zero is what
> selecting the largest of 65 correlations produces: under independence, the largest |r| of 65
> null correlations at this sample size has a median of .035 and a 95th percentile of .046.
> The family p and BH q are the calibrated numbers, and the verdict uses only q. In the
> sensitivity run below, `term_spread`'s best link is the same (GLD, lag 3) but r changes sign.

**The null is partly a statement about power.** At realistic signal strength (IC .02), Stage 1
detected a true edge in only .140 [.112, .173] of benchmark runs at T = 6000
([report](docs/results/0010-causal-stage-validation/report.json)), and this panel is shorter.
Each signal's evidence is Bonferroni-corrected over its 65 tests, so a weak edge shared by many
assets is not helped by appearing in several of them. A pooled cross-asset test is in the
backlog. Stage 1 also conditions only on these 23 observed series, so it cannot rule out
confounding by anything else.

**Sensitivity: the posting-lag assumption does not change any outcome.** Daily FRED series have
no vintage archive before 2005–2014 (by series), so their pre-archive availability is an assumption
([ADR 0019](docs/adr/0019-fred-market-source-and-within-vintage-changes.md)). A pre-registered
second run used the most conservative alternative (each value usable only after the longest
delay the archive shows: up to 112 sessions for the credit spreads). Every signal got the same
verdict at the same stage (smallest q .716,
[ledger](docs/results/0018-real-data-run-sensitivity/kill_ledger.jsonl)).

Archived: both runs' ledgers, stage results and memos
([primary memo](docs/results/0018-real-data-run-primary/memo/memo.md)), the revalidation report,
and the snapshot's manifest (row counts, coverage and the FRED posting audit; no market values,
which the sources' terms do not let us republish).

## Results on the benchmark

The benchmark is a structural causal model with four candidate signals whose truth is known:
`s_real` (a constant true edge), `s_decay` (a true edge that decays), `s_conf` (correlated with
returns only through a common cause `z`) and `s_noise` (nothing). Default preset: T = 6000
daily observations. Each rate is over M independent synthetic markets, with a Wilson 95%
interval. Each number links to the archived report it comes from.

### Stage 1, causal screen
PCMCI with an HC3 partial-correlation test, Bonferroni over lags and assets and
Benjamini–Hochberg across signals ([ADR 0009](docs/adr/0009-stage1-ci-test-and-multiple-testing.md),
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
  ([report](docs/results/0010-causal-stage-validation/report.json)). In the Stage 2 validation
  (fresh seeds) it passed Stage 1 in .735 and Stage 2 killed only .045 [.032, .062] of those,
  because a spurious edge need not decay, so s_conf passes both stages in .702 [.673, .730] of
  runs ([report](docs/results/0012-decay-stage-validation/report.json)). No
  observational method can remove this. The memo says so whenever a signal passes Stage 1.
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
  signal-to-noise ratio it is not identifiable, and in the study's three main scenarios
  every candidate estimator's interval claimed decay in 9–12% of runs with no decay
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
- Ridge-CV had the best Sharpe: 1.560, against 1.526 for the better DRO arm (DRO-CV). The
  interval of the difference excludes 0: .034 [.021, .047]. Ridge-CV was selected. Wasserstein DRO was
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
- **The test is on gross Sharpe.** On the null markets Ridge-CV's gross Sharpe is about 0 and
  its net Sharpe −.31 to −.32, so a false pass would lose money.

### Stage 5, research memo
[ADR 0017](docs/adr/0017-stage5-research-memo.md). The memo is built from one run's ledger,
stage results and manifest; it never reads the benchmark's ground truth (a leakage test pins
this).
- **Every number comes from the run.** The memo is a document of segments, and every digit lies
  inside an atom that renders one fact-sheet value with its subject and label. The verifier
  rebuilds the fact sheet and reads the raw ledger and result files by its own code path.
- **The recommendation follows a fixed rule.** "Candidate for paper trading" if any signal
  passed every kill stage; "not decided" if a signal was killed without a test decision;
  otherwise "against trading". The memo states every caveat whose trigger fires (a hidden
  confounder can pass Stage 1; DRO was evaluated and ridge selected; Stage 4 was not adopted;
  the stress diagnostic is a description, not a certification; passengers; ...).
- **Template only.** The verifier accepts a memo only if it is exactly the run's template
  memo: every block equals the re-render, `memo.md` equals the rendering of `memo.json`, and
  no text may hold a zero-width or other format character, or a non-ASCII letter that could
  pass for an ASCII one.
- **The LLM narrative was cut.** The planned narrative was to be checked by word lists (no
  numbers, verdict words, banned claims or uncited literature cues in prose). A set of
  adversarial edits written by an independent agent, which never saw the verifier, got 7 of
  126 past it: claims in new wording ("for practical purposes as good as clearing the bar")
  and Unicode tricks. It also rejected 8 of 13 harmless sentences
  ([run 1 report](docs/results/0017-memo-validation-run1/report.json)). Adding the missed
  words would only have fitted that one set, so narrative paragraphs are now rejected outright.
- **Validation.** On fresh seeds, the template-only verifier accepted all 48 template memos,
  missed none of 576 generated corruptions in each of 9 classes (95% upper bound on the miss
  rate .0052), and missed none of three hand-written sets (96, 126 and 146 entries; the third
  written afterwards by a new independent agent)
  ([report](docs/results/0017-memo-validation-deviation1/report.json)).
- **What it cannot do.** It checks that a cited paper is in the corpus, not that it supports
  the claim. Most template tampering is caught by one check (block equals re-render), so a bug
  shared by the template and the verifier's re-render would pass. "-0.00" and "0.00" compare
  equal at display precision.

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
uv run python -m scripts.memo_validation --n-jobs 8 \
    --config configs/memo_validation_deviation1.yaml               # ADR 0017 validation, ~20 min
```

Each stage's study and validation has a script in `scripts/` and a config in `configs/`.
Benchmark runs need no API key. `FRED_API_KEY` in `.env` (see `.env.example`) is needed only to
fetch real FRED and ALFRED data (`invariant snapshot`). The real-data run
(`invariant run --config configs/real_0018.yaml`) refuses any snapshot but the pinned
`930504ace5464752`, which is not distributed, so it cannot be rerun from a clone; its ledgers
and memos are archived in [docs/results/](docs/results/). Working rules for contributors are in [CLAUDE.md](CLAUDE.md).

## What v1.0 contains, and what it does not

- **Stages 1–3: adopted as kill stages** ([ADRs 0009–0014](docs/adr/)).
- **Stage 4, adversarial stress: evaluated, rejected as a kill stage**
  ([ADR 0016](docs/adr/0016-stage4-adversarial-study.md)).
  - **What was tried.** Two regime-fragile signals were planted
    ([ADR 0015](docs/adr/0015-benchmark-v4-regime-signal.md)), and the design was taken as far
    as a population feasibility check.
  - **Why it failed.**
    - Latent regimes are not identifiable from returns at this sample size.
    - Bootstrap path generators failed the stylized-facts check three times.
    - The plausible set for high-volatility exposure is too narrow to overturn an
      average-profitable strategy: the worst-case Sharpe stays at +0.90 even with perfectly
      known moments ([report](docs/results/0016-feasibility/feasibility.json)).
    - A plain "loses money in stress" rule would penalise ordinary market exposure, not
      fragility.
  - **What remains.** Stage 3 reports a stress diagnostic: each portfolio's Sharpe in
    high-volatility periods, with an interval. It never decides.
- **Stage 5, research memo: accepted, template only**
  ([ADR 0017](docs/adr/0017-stage5-research-memo.md)). The LLM narrative was cut after an
  independent red team showed that word-list verification cannot police free prose.
- **Real data: first run reported** ([ADR 0018](docs/adr/0018-first-real-data-run.md),
  [0019](docs/adr/0019-fred-market-source-and-within-vintage-changes.md)). Ten candidates, all
  killed at Stage 1; the posting-lag sensitivity run agrees.

There are no target dates. Remaining work is the backlog below.

**Backlog**, each to be decided by its own pre-registered study:
- A pooled cross-asset Stage 1 decision unit: one test of a signal against all assets jointly,
  with more power for an effect common across assets (ADR 0018).
- Per-signal-per-asset candidates, such as each ETF's own momentum (ADR 0018).
- A net-of-cost confirmation of the Stage 3 method, at a cost level justified in advance
  (ADR 0013).
- A per-signal Stage 3 rule built on the marginal-Sharpe diagnostic, to stop passengers
  (ADR 0014).
- A calibrated confidence radius for the DRO forecast arm, before it is reconsidered
  (ADR 0013).
- A calibrated half-life interval (ADR 0011).
- A structured LLM narrative that chooses and orders verified atoms and fixed sentences and
  writes no free text, with its own validation (ADR 0017).
- An independent second check on the memo's template text, which the re-render comparison
  alone now covers, and a sign check for values displayed as zero (ADR 0017).
- The decline test's size and power at T < 6000, which would allow a lower `min_obs`
  (ADR 0011).
- The decline test's small excess size when a strong decaying co-parent is present (ADR 0011).

## Limitations

- Every rate above is measured on synthetic markets. Real markets violate the benchmark's
  assumptions (stationarity, linear effects, a known set of candidates), and the real-data run
  has no ground truth: it is reported, not graded.
- Daily FRED series have no vintage archive before 2005–2014 (by series); their earlier availability is an
  assumption, checked by a sensitivity run (ADR 0019).
- The candidate list comes from a literature that searched many predictors on overlapping data;
  no correction here undoes that (Harvey, Liu & Zhu 2016).
- The ETF universe has survivorship and hindsight bias. Yahoo adjusted price levels are not
  point-in-time, so only returns are used. Fama-French factors are revised with every release
  and are used with a 42-session lag.

## License

[MIT](LICENSE)
