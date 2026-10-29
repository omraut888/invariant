# ADR 0016: Stage 4, adversarial regime certification (study)

Status: **draft**. Only the labeler validation targets below are fixed. The study's
pre-registration is written later, reviewed by the user (Stage 4 stop point 2), and then pushed.

## Labeler validation targets (fixed 2026-10-29, before the MS-GARCH labeler ran)

**Background.** A Gaussian 2-state Markov-switching labeler (Hamilton 1989, EM) was tried first,
on 20 dev seeds from root **20261101**, equal-weight daily returns. It fit cleanly (20 of 20
converged) and agreed with statsmodels, but:
- median daily accuracy against the true regime was .83 (.73–.91);
- its fitted stress spells were too short: median 36 days, against an in-sample mean true spell
  of 92;
- on `multi_asset`, which has no regimes, it "found" stress regimes (median 19-day spells, .22
  share) by mistaking GARCH volatility clusters for regime changes.

Labelling 5–10-day sums fixed the spell length (80–99 days) but not accuracy (.80–.82), and it
invented long regimes on `multi_asset`. The user chose a Markov-switching GARCH labeler instead
(below).

**Labeler.** A Markov-switching GARCH(1,1) with Student-t errors (Haas, Mittnik & Paolella 2004):
- each regime has its own GARCH recursion, run in parallel on the same observed shocks;
- the number of regimes, 1 or 2, is chosen by BIC;
- with one regime, every day is labelled calm, the adversary has nothing to move, and Stage 4
  passes. That is how scenario S5 (`multi_asset`) is meant to pass.

**Targets.** These are evaluated once on the same 20 dev seeds per preset (root 20261101,
`RngFactory(20261101).child(f"dev/{i}")`, i = 0..19), equal-weight daily return, hard labels by
smoothed stress probability > .5:

| # | preset | target |
|---|---|---|
| L1 | `regime` | median over seeds of daily label accuracy against the true regime **≥ .90** |
| L2 | `regime` | median over seeds of (fitted expected stress spell, 1/P(stress→calm)) / (in-sample mean true stress spell) **within [0.7, 1.3]** |
| L3 | `multi_asset` | BIC selects **one regime on all 20 seeds** (no invented regimes) |

Reported, not a target: the number of `regime` seeds on which BIC selects two regimes. A seed on
which it selects one is scored as all calm, so it counts against L1.

**Rule.** If any target is missed, I stop and show the user. There is no iteration on these dev
seeds, and no change to the targets. The work is time-boxed to about a day.

## Result: the latent regime is not identifiable from the equal-weight return (2026-10-29)
One run, labeler code `d36c6b9`, targets `41b71c0`. Record:
[`docs/results/0016-labeler-dev/labeler_dev.json`](../results/0016-labeler-dev/labeler_dev.json).

| target | result | met |
|---|---|---|
| L1 median accuracy ≥ .90 | .851 (every day labelled calm) | no |
| L2 spell ratio in [0.7, 1.3] | no two-regime model chosen | no |
| L3 one regime on all 20 `multi_asset` seeds | 20 of 20 | yes |
| reported: two regimes chosen on `regime` | **0 of 20** | – |

**This is a finding about identification, not a fitting failure.**
- **The fits are fine.** The two-regime MS-GARCH-t fits converged on all 40 seeds, and on
  `regime` seeds their stress share and spell were roughly right: median .20 and 83 days,
  against a true in-sample .15 and 92.
- **Two regimes barely improve the fit.**
  - On `regime` seeds, two regimes improve the log-likelihood over one GARCH-t by a median of
    **6.1** units (range 1.1–14.9).
  - On `multi_asset` seeds, which have no regimes, by **3.3** (0.0–6.6).
  - BIC's penalty for the five extra parameters at T = 6000 is **21.75** units.
- **No threshold separates them.** The BIC gaps overlap: 13.8–41.2 on `regime` against 30.3–43.4
  on `multi_asset`. Only 8 of 20 `regime` seeds have a gain above the largest no-regime gain, so a
  likelihood threshold chosen after the fact would detect the planted regimes about 40% of the time.
- **Why.** A persistent GARCH-t (β = .92) absorbs a 2–2.5× switch in the variance intercept
  almost entirely. The same happened on independent synthetic data in the unit tests
  (`test_optimum_is_at_least_as_likely_as_the_truth`).
- **Implication.** A Stage 4 built on a latent-regime labeler would label every day calm on this
  benchmark. The adversary would have nothing to move, and Stage 4 would pass every strategy, so
  its power gate would fail by construction.
- **Likely also on real data.** About 20 years of daily data probably cannot separate regimes of
  this kind from GARCH clustering. Stage 4 must not claim to detect latent regimes.

The MS-GARCH labeler (`stages/adversarial/regimes.py`) stays in the codebase, tested, as an
evaluated alternative. Stage 4 does not use it.

## Decision: an observable volatility state (user, 2026-10-29)
Fixed before any run that uses it.

**Definition.**
- `v_t` = the sample sd of the equal-weight daily return over days t−20..t (trailing, 21 days).
- Day t is in the **high-volatility state** iff `v_t` exceeds the **q = .70** quantile of `v` over
  the Stage 3 out-of-sample days.
- The P&L earned on day t+1 (weights dated t, return dated t+1) is labelled with the state of day
  **t**. The label is therefore fixed before the return it labels, so no day is selected on its own
  outcome. Using `v_{t+1}` would include r_{t+1}, select on |r_{t+1}|, and distort the
  per-state P&L moments.

**Why q = .70** (argued without reference to the benchmark's planted stress share).
- Stage 4's kill rests on the strategy's Sharpe in the high-volatility state. That estimate is only
  decision-relevant if its standard error is no larger than the Sharpe hurdle Stage 3 already
  uses: c = 0.5 annualised (ADR 0013).
- At Sharpe ≈ 0, the annualised SE is about √(252/n_s), so n_s ≥ 1008 days are needed in the state.
- The smallest panel Stage 3 admits (`min_obs` = 5000 rows; window 1260) has 3740 out-of-sample
  days. That needs a state share of at least 1008/3740 = .27, which on a grid of .05 gives a top
  share of **.30**.
- On the benchmark (4739 out-of-sample days), n_s ≈ 1422 and the SE is ≈ .42. On the real ETF
  panel (about 4020 out-of-sample days), n_s ≈ 1206 and the SE is ≈ .46.
- **Stated explicitly:** a .30 share is above the benchmark's planted stress share (.20; the
  in-sample median on the dev seeds is .15). So the high-volatility state necessarily includes
  calm-regime days, and the labels are noisy against the truth by design. This dilutes s_regime's
  fragility in the state, and the pilot and the study measure the resulting power rather than
  assuming it.

**Consequences.**
- The generator, the realism gate and the adversary are unchanged, with "regime" replaced by this
  state.
  - θ̂ is the empirical transition frequency of the binary state sequence.
  - The adversary moves the state's share and its expected spell length.
- **S5 changes.** No "one regime, so pass" path exists. `multi_asset` with {s_real} is graded like
  any robust strategy (pass rate, threshold from the pilot rule).
- **Stage 4's claim** becomes: "survives the worst plausible mix of high- and low-volatility
  periods". It does not claim to recover a latent regime.
