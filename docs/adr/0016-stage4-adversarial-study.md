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
