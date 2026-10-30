"""Stage 3 result contracts. Frozen, JSON-serialisable, and the only thing Stage 3 hands on.

Stage 3 builds **one** portfolio from the signals that passed Stage 2 and decides once for all
of them (kill rule A, ADR 0013): every member gets the portfolio's verdict. That settles the
per-signal question ADR 0007 left open (ADR 0014). Signals killed earlier never reach Stage 3.

The deciding arm is Ridge-CV. Markowitz and the Wasserstein DRO arms (DRO-FC, DRO-CV) are
evaluated on the same data and reported with ``decides=False``: Wasserstein DRO was implemented,
verified and evaluated, and ridge regularization was selected (ADR 0013).
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from invariant.config.settings import DecayConfig, PortfolioConfig
from invariant.core.types import KillReason, StageName, Verdict, make_signal_id
from invariant.stages.adversarial.diagnostic import StressDiagnostic
from invariant.stages.dro.walkforward import Arm

__all__ = [
    "DECIDING_ARM",
    "ArmReport",
    "MemberSignal",
    "PortfolioEvidence",
    "PortfolioResult",
]

DECIDING_ARM = Arm.RIDGE_CV
"""Selected by ADR 0013's pre-registered study."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArmReport(_Frozen):
    """One arm's out-of-sample path summary. Sharpe ratios are annualised; an all-cash path has
    Sharpe 0. ``lw_*`` and ``dsr`` are ``None`` for 1/N, the test's baseline."""

    arm: Arm
    decides: bool
    sharpe: float
    sharpe_net: float
    """Net of ``cost_bp`` per unit of one-way turnover (reported only)."""
    mean: float
    sd: float
    invested: float
    cash: float
    partial: float
    turnover: float
    lw_delta: float | None = None
    """Non-annualised Sharpe difference against 1/N."""
    lw_p: float | None = None
    """One-sided Ledoit-Wolf p-value of beating 1/N."""
    dsr: float | None = None
    """Deflated Sharpe Ratio over the four candidate arms (N = 4)."""


class PortfolioEvidence(_Frozen):
    """Everything measured on the out-of-sample path."""

    execution_lag: int = Field(ge=0)
    n_rows: int = Field(ge=0)
    """Complete panel rows used."""
    n_test_days: int = Field(ge=1)
    first_decision: date
    last_decision: date
    arms: tuple[ArmReport, ...]
    ridge_lambdas: tuple[float, ...]
    """Ridge-CV's grid value at each CV date (in units of tr(S)/N)."""
    final_weights: tuple[tuple[str, float], ...]
    """Ridge-CV's weights at the last decision date, by return column."""
    stress: StressDiagnostic | None = None
    """Reported only, never decides: Sharpe in the high-volatility state (ADR 0016)."""

    def arm(self, arm: Arm) -> ArmReport:
        return next(a for a in self.arms if a.arm is arm)

    @property
    def deciding(self) -> ArmReport:
        return self.arm(DECIDING_ARM)


class MemberSignal(_Frozen):
    """One signal in the portfolio, with the joint verdict and diagnostic B.

    B is Ridge-CV's Sharpe with every member minus its Sharpe without this one, with a paired
    block-bootstrap interval. With a single member the comparison is Ridge-CV with no signal (an
    intercept-only forecast), so B is always defined when the stage ran. B never decides.
    """

    signal_id: str
    column: str
    verdict: Verdict
    reason: KillReason | None
    marginal_sharpe: float | None = None
    marginal_ci_low: float | None = None
    marginal_ci_high: float | None = None
    baseline: Literal["other_members", "intercept_only"] | None = None
    data_issue: str | None = None

    @field_validator("signal_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return make_signal_id(value)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.verdict is Verdict.KILL) != (self.reason is not None):
            raise ValueError("a kill needs a reason and a pass must not have one")
        if self.reason is not None and self.reason.stage not in (None, StageName.PORTFOLIO):
            raise ValueError(f"reason {self.reason} does not belong to Stage 3")
        return self


class PortfolioResult(_Frozen):
    """Everything Stage 3 decided and measured on one panel."""

    config: PortfolioConfig
    decay_config: DecayConfig | None
    """Stage 2's config; ``None`` when members were given directly (the size validation)."""
    n_obs: int = Field(ge=0)
    first_date: date | None
    last_date: date | None
    members: tuple[MemberSignal, ...]
    evidence: PortfolioEvidence | None
    """``None`` when the stage could not run (no member, or a ``general.*`` kill)."""
    not_reached: tuple[str, ...]
    """Signal ids killed before Stage 3."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        verdicts = {(m.verdict, m.reason) for m in self.members}
        if len(verdicts) > 1:
            raise ValueError("every member must share the portfolio's verdict")
        if self.evidence is None and self.members:
            reason = self.members[0].reason
            if reason is None or reason.stage is not None:
                raise ValueError("only a general.* kill may come without evidence")
        return self

    @property
    def verdict(self) -> Verdict | None:
        """The portfolio's verdict, or ``None`` when no signal reached Stage 3."""
        return self.members[0].verdict if self.members else None
