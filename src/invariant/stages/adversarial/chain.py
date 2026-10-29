"""The two-state chain Stage 4's adversary moves (docs/adr/0016).

States are 0 = low and 1 = high volatility (``state.volatility_state``). A chain is parameterised
by its stationary high share pi_s and its mean high spell d_s (days):
P(high -> low) = 1/d_s and P(low -> high) = (1/d_s) pi_s / (1 - pi_s).

**Markov approximation.** The observed state is a thresholded 21-day trailing volatility, so its
spells are not geometric and the sequence is not a first-order Markov chain. Stage 4 uses the
first-order chain with the same (pi_s, d_s) only to compute the expected share of a horizon
spent in each state (``objective.horizon_weights``). The approximation therefore enters through
how quickly a start in one state mixes toward pi_s. It does not enter the per-state P&L moments,
which are empirical.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["Start", "StateChain", "state_path"]

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class Start(StrEnum):
    """The state of the day before the horizon (its first day is drawn from that row of P)."""

    LOW = "low"
    HIGH = "high"
    STATIONARY = "stationary"


class StateChain(BaseModel):
    """Two-state chain (0 = low, 1 = high) by high share and mean high spell (days)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    share: float = Field(gt=0.0, lt=1.0)
    duration: float = Field(ge=1.0)

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if self.low_to_high > 1.0:
            raise ValueError(
                f"share {self.share} with duration {self.duration} needs P(low -> high) "
                f"{self.low_to_high:.3g} > 1",
            )
        return self

    @property
    def high_to_low(self) -> float:
        return 1.0 / self.duration

    @property
    def low_to_high(self) -> float:
        return self.high_to_low * self.share / (1.0 - self.share)

    @property
    def transition(self) -> FloatArray:
        q, p = self.low_to_high, self.high_to_low
        return np.array([[1.0 - q, q], [p, 1.0 - p]])

    @property
    def stationary(self) -> FloatArray:
        return np.array([1.0 - self.share, self.share])

    @classmethod
    def is_valid(cls, share: float, duration: float) -> bool:
        return 0.0 < share < 1.0 and duration >= 1.0 and share / (1.0 - share) <= duration

    @classmethod
    def from_states(cls, states: IntArray) -> StateChain:
        """The empirical chain of a 0/1 sequence: its high share and its mean high run (raised,
        if needed, to the smallest duration that is valid for that share)."""
        x = np.asarray(states, dtype=np.int64)
        if not 0 < x.sum() < len(x):
            raise ValueError("the sequence must contain both states")
        starts = np.flatnonzero(np.diff(np.concatenate([[0], x])) == 1)
        ends = np.flatnonzero(np.diff(np.concatenate([x, [0]])) == -1)
        share = float(x.mean())
        duration = max(float((ends - starts + 1).mean()), share / (1.0 - share), 1.0)
        return cls(share=share, duration=duration)


def state_path(chain: StateChain, start: Start, length: int, gen: np.random.Generator) -> IntArray:
    """z_1..z_length from ``chain``; z_1 is drawn from the row of P of ``start`` (or from pi)."""
    u = gen.random(length)
    out = np.empty(length, dtype=np.int64)
    if start is Start.STATIONARY:
        prob_high = chain.share
    else:
        prob_high = float(chain.transition[1 if start is Start.HIGH else 0, 1])
    state = int(u[0] < prob_high)
    out[0] = state
    q, p = chain.low_to_high, chain.high_to_low
    for t in range(1, length):
        state = int(u[t] < q) if state == 0 else int(u[t] >= p)
        out[t] = state
    return out
