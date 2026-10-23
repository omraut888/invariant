"""Stage 2 result contracts. Frozen, JSON-serialisable, and the only thing Stage 2 hands on.

One ``SignalDecay`` per signal that Stage 1 passed. Signals Stage 1 killed never reach Stage 2:
their fate is already in the kill ledger, and no entry may follow a kill. Stage 2 reports no
half-life, because no estimator was admissible in docs/adr/0011.
"""

from __future__ import annotations

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from invariant.config.settings import CausalConfig, DecayConfig
from invariant.core.types import KillReason, StageName, Verdict, make_signal_id
from invariant.stages.decay.decline import DeclineTest

__all__ = ["DecayResult", "SignalDecay"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SignalDecay(_Frozen):
    """Stage 2's decision on one signal Stage 1 passed.

    ``asset``, ``lag`` and ``conditions`` are Stage 1's best link, tested unchanged. ``test``
    is ``None`` only when the stage could not test, and then the reason is a ``general.*``
    one.
    """

    signal_id: str
    column: str
    asset: str
    lag: int = Field(ge=1)
    conditions: tuple[tuple[str, int], ...]
    verdict: Verdict
    reason: KillReason | None
    test: DeclineTest | None
    data_issue: str | None = None

    @field_validator("signal_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return make_signal_id(value)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.verdict is Verdict.KILL) != (self.reason is not None):
            raise ValueError("a kill needs a reason and a pass must not have one")
        if self.reason is not None and self.reason.stage not in (None, StageName.DECAY):
            raise ValueError(f"reason {self.reason} does not belong to Stage 2")
        if self.test is None and (self.reason is None or self.reason.stage is not None):
            raise ValueError("only a general.* kill may come without a test")
        return self


class DecayResult(_Frozen):
    """Everything Stage 2 decided and measured on one panel."""

    config: DecayConfig
    causal_config: CausalConfig
    """Stage 1's config, whose ``tau_max`` built the designs."""
    n_obs: int = Field(ge=0)
    first_date: date | None
    last_date: date | None
    signals: tuple[SignalDecay, ...]
    not_reached: tuple[str, ...]
    """Signal ids Stage 1 killed: they never reach Stage 2."""
