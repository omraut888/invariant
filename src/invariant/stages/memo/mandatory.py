"""Mandatory statements: the caveats a memo must carry whenever they apply (docs/adr/0017).

Each statement has an id, a **trigger** (a predicate on the fact sheet) and a template. The
template memo includes exactly the statements whose trigger fires, in registry order, and the
verifier requires exactly that set: a missing caveat is an overclaim, and a caveat whose trigger
did not fire is a false statement about the run.

Templates use the atom syntax of ``template.parse``: ``{fact:<id>}``, ``{fate:<signal>}``,
``{stage:<stage>}``, ``{cite:<key>}``, ``{adr:<n>}``, ``{term:<t>}``. Their plain text carries
no digits (tested).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from invariant.core.types import KillReason, StageName, Verdict
from invariant.ledger.kill_ledger import FateStatus
from invariant.stages.memo.facts import DataKind, FactSheet

__all__ = ["MANDATORY", "Mandatory", "triggered"]


@dataclass(frozen=True)
class Mandatory:
    id: str
    trigger: Callable[[FactSheet], bool]
    text: Callable[[FactSheet], str]


def _passed_stage(sheet: FactSheet, stage: StageName) -> bool:
    return any(
        stage in f.stages_decided and not (f.stage is stage and f.status is FateStatus.KILLED)
        for f in sheet.fates
    )


def _reached(sheet: FactSheet, stage: StageName) -> bool:
    return any(stage in f.stages_decided for f in sheet.fates)


def _reasons(sheet: FactSheet) -> set[KillReason]:
    return {f.reason for f in sheet.fates if f.reason is not None}


def _portfolio_verdict(sheet: FactSheet) -> Verdict | None:
    if not sheet.members:
        return None
    f = sheet.fate(sheet.members[0])
    return Verdict.KILL if f.status is FateStatus.KILLED else Verdict.PASS


_UNDETECTABLE: Final = {
    KillReason.CAUSAL_NO_SIGNIFICANT_LINK,
    KillReason.CAUSAL_NOT_SIGNIFICANT_AFTER_CORRECTION,
    KillReason.PORTFOLIO_NO_EDGE_OVER_EQUAL_WEIGHT,
}


def _data(sheet: FactSheet) -> str:
    if sheet.data_kind is DataKind.SYNTHETIC:
        return (
            "This run used the synthetic planted-signal benchmark ({adr:3}), not market data. "
            "One run is an illustration of how the pipeline decides, not evidence about any "
            "strategy. The rates at which each stage decides correctly were measured over many "
            "runs and are in the validation ADRs."
        )
    return (
        "This run used market data, with known limits ({adr:4}): the ETF universe has "
        "survivorship and hindsight bias, price levels are not point-in-time (only returns are "
        "used), and the Fama-French factors are revised with every release and used with a "
        "lag. Every rate this memo quotes from the planted-signal benchmark is a benchmark "
        "validation rate, measured on synthetic data with a known answer. It describes how the "
        "pipeline's tests behave, and is not a property of this strategy."
    )


def _undetectable(sheet: FactSheet) -> str:
    reasons = _reasons(sheet)
    out = []
    if reasons & {
        KillReason.CAUSAL_NO_SIGNIFICANT_LINK,
        KillReason.CAUSAL_NOT_SIGNIFICANT_AFTER_CORRECTION,
    }:
        out.append(
            "A kill at {stage:causal} means the edge was not detectable at this sample size, "
            "not that there is no effect. The test has little power against weak edges. "
            "{fact:ref.realistic_stage1_power}. {fact:ref.base_stage1_power}."
        )
    if KillReason.PORTFOLIO_NO_EDGE_OVER_EQUAL_WEIGHT in reasons:
        out.append(
            "A kill at {stage:portfolio} means the portfolio's advantage over {term:one_over_n} "
            "was not detectable at this sample size, not that it has none. Even at the "
            "benchmark's default signal strength the kill test passes a portfolio with a true "
            "edge only part of the time. {fact:ref.stage3_power}. At realistic strength its "
            "power is close to zero ({adr:13})."
        )
    return " ".join(out)


def _gross(sheet: FactSheet) -> str:
    head = (
        "The kill test of {stage:portfolio} compares gross Sharpe ratios, so a pass does not "
        "imply a positive return after costs ({adr:14}). "
    )
    if not sheet.has("portfolio.ridge_cv.sharpe_net"):
        return head.rstrip()
    net = float(sheet.fact("portfolio.ridge_cv.sharpe_net").value)
    if net <= 0:
        return head + (
            "Net of costs this portfolio does not earn a positive Sharpe ratio: Ridge-CV's "
            "net-of-cost Sharpe ratio is {fact:portfolio.ridge_cv.sharpe_net|value}, at an "
            "assumed {fact:portfolio.cost_bp|value} basis points per unit of turnover."
        )
    return head + (
        "For this portfolio, Ridge-CV's net-of-cost Sharpe ratio is "
        "{fact:portfolio.ridge_cv.sharpe_net|value}, at an assumed "
        "{fact:portfolio.cost_bp|value} basis points per unit of turnover; real costs may be "
        "higher."
    )


def _b_list(sheet: FactSheet) -> str:
    facts = [f"{{fact:portfolio.{m}.b}}" for m in sheet.members if sheet.has(f"portfolio.{m}.b")]
    return "; ".join(facts)


def _b_zero(sheet: FactSheet) -> list[str]:
    out = []
    for m in sheet.members:
        if sheet.has(f"portfolio.{m}.b"):
            ci = sheet.fact(f"portfolio.{m}.b").ci
            if ci is not None and ci[0] <= 0.0 <= ci[1]:
                out.append(m)
    return out


def _passenger(sheet: FactSheet) -> str:
    text = (
        "{stage:portfolio} decides once for the whole portfolio, so every member shares its "
        "verdict, and a member with no edge can pass as a passenger of one that has it. "
        "{fact:ref.noise_passenger}. Diagnostic B, each member's marginal "
        "contribution, is reported and never decides: " + _b_list(sheet) + "."
    )
    zero = _b_zero(sheet)
    if zero:
        text += (
            " The B interval includes zero for "
            + ", ".join(zero)
            + ", so B does not separate "
            + ("it" if len(zero) == 1 else "them")
            + " from a passenger."
        )
    return text


def _collateral(sheet: FactSheet) -> str:
    text = (
        "{stage:portfolio} killed every member together, because it decides once for the whole "
        "portfolio. A member with a real edge may be among them."
    )
    if _b_list(sheet):
        text += (
            " Diagnostic B, each member's marginal contribution, never decides: "
            + _b_list(sheet)
            + "."
        )
    return text


MANDATORY: Final[tuple[Mandatory, ...]] = (
    Mandatory("data_source", lambda s: True, _data),
    Mandatory(
        "hidden_confounder",
        lambda s: _passed_stage(s, StageName.CAUSAL),
        lambda s: (
            "{stage:causal} tests conditional dependence given the observed variables only. It "
            "cannot rule out a common cause missing from the data, and no observational "
            "adjustment removes that bias ({adr:10}). "
            "{fact:ref.hidden_confounder_stage1_pass}."
        ),
    ),
    Mandatory("not_detectable", lambda s: bool(_reasons(s) & _UNDETECTABLE), _undetectable),
    Mandatory(
        "not_decided",
        lambda s: any(r.stage is None for r in _reasons(s)),
        lambda s: (
            "Some signals were killed without a test decision, because the data were "
            "insufficient or invalid for the stage. Such a kill is evidence neither for nor "
            "against an edge."
        ),
    ),
    Mandatory(
        "stage1_descriptive",
        lambda s: any(f.id.startswith("causal.") and f.id.endswith(".r_best") for f in s.facts),
        lambda s: (
            "At {stage:causal}, the best lag is where the evidence was strongest, not an "
            "identified horizon, and the partial-correlation intervals are descriptive: they are "
            "not adjusted for the selection of the link and ran narrow on the benchmark ({adr:10})."
        ),
    ),
    Mandatory(
        "no_half_life",
        lambda s: _reached(s, StageName.DECAY),
        lambda s: (
            "{stage:decay} reports no half-life. At this signal-to-noise ratio a half-life is "
            "not identifiable, and the candidate estimators' intervals claimed decay where there "
            "was none ({adr:11}). The stage tests only whether the edge declined."
        ),
    ),
    Mandatory(
        "decay_power",
        lambda s: _passed_stage(s, StageName.DECAY),
        lambda s: (
            "A pass at {stage:decay} does not show that an edge is stable. The test misses "
            "many real declines and is weakest when the decline began recently. "
            "{fact:ref.stage2_power}."
        ),
    ),
    Mandatory(
        "portfolio_method",
        lambda s: s.portfolio_ran,
        lambda s: (
            "{stage:portfolio} used ridge regularization of the covariance, with its strength "
            "chosen by walk-forward cross-validation (Ridge-CV). Wasserstein distributionally "
            "robust optimization was implemented, verified and evaluated in a pre-registered "
            "study and was not selected ({adr:13}), so this portfolio is not distributionally "
            "robust. The DRO arms are reported for comparison only and do not decide."
        ),
    ),
    Mandatory("gross_sharpe", lambda s: s.portfolio_ran, _gross),
    Mandatory(
        "deflated_sharpe",
        lambda s: s.has("portfolio.ridge_cv.dsr"),
        lambda s: (
            "The Sharpe ratios are not adjusted for the arms that were tried. The deflated "
            "Sharpe ratio corrects for the {fact:const.dsr_trials|value} arms tried "
            "({cite:bailey2014}); Ridge-CV's is {fact:portfolio.ridge_cv.dsr|value}, a "
            "probability."
        ),
    ),
    Mandatory(
        "passenger",
        lambda s: _portfolio_verdict(s) is Verdict.PASS and len(s.members) >= 2,
        _passenger,
    ),
    Mandatory(
        "collateral",
        lambda s: _portfolio_verdict(s) is Verdict.KILL and len(s.members) >= 2,
        _collateral,
    ),
    Mandatory(
        "stage4_not_adopted",
        lambda s: True,
        lambda s: (
            "{stage:adversarial} was evaluated and not adopted as a kill stage ({adr:16}). At "
            "about twenty-four years of daily data, the plausible variation in a strategy's "
            "exposure to high-volatility periods is too small to overturn a strategy that is "
            "profitable on average, and a rule that killed for losing money in stress would "
            "penalise ordinary market exposure rather than fragility. "
            "{fact:ref.stage4_worst_case}."
        ),
    ),
    Mandatory(
        "stress_description",
        lambda s: s.has("stress.ridge_cv_high"),
        lambda s: (
            "The stress diagnostic describes the portfolio's performance in a high-volatility "
            "state. It is a description, not a certification, and it never decides. Ridge-CV's "
            "Sharpe ratio in that state is {fact:stress.ridge_cv_high|value} and equal "
            "weight's is {fact:stress.equal_weight_high|value}; "
            "{fact:stress.high_share|value} of days fall in it. The state is an observable "
            "proxy built from trailing volatility, not a recovered regime. "
            "{fact:ref.stage4_state_precision}."
        ),
    ),
)


def triggered(sheet: FactSheet) -> tuple[Mandatory, ...]:
    """The statements ``sheet`` requires, in registry order."""
    return tuple(m for m in MANDATORY if m.trigger(sheet))
