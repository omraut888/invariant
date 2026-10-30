"""Reported-only stress diagnostic attached to Stage 3 (docs/adr/0016, "Consequences").

Method
------
- **State.** Each out-of-sample P&L day of Stage 3's walk-forward is labelled with the vol21,
  q = .70 volatility state of its decision date (``state.volatility_state``).
- **Measure.** The diagnostic is the annualised Sharpe ratio of Ridge-CV's, and of 1/N's, P&L on
  the high-volatility days.
- **Interval.** A percentile interval from a moving-block bootstrap of the joint daily series
  (P&L, state). The block length is Stage 4's Politis-White rule
  (``certify.certification_block_length``); each replicate recomputes the high-state Sharpe
  ratios.

**It never decides.** ADR 0016 found that a kill rule built on stress performance cannot be
made decision-relevant at about 24 years of daily data: the plausible variation in
high-volatility exposure is too small to overturn an average-positive strategy, and a raw
"loses money in stress" rule would penalise ordinary market exposure rather than fragility.
The diagnostic describes; the memo must not present it as a certification.

Assumptions: the high-volatility state is an observable proxy, not a recovered regime (on the
benchmark only 39% of its days fall in the true stress regime, ADR 0016). The interval assumes
weak dependence.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from invariant.stages.adversarial.certify import certification_block_length
from invariant.stages.adversarial.objective import TRADING_DAYS
from invariant.stages.adversarial.region import moving_block_indices
from invariant.stages.adversarial.state import volatility_state

__all__ = ["StressDiagnostic", "stress_diagnostic"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class StressDiagnostic(BaseModel):
    """High-volatility-state Sharpe ratios (annualised) of Ridge-CV and 1/N, with intervals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    high_share: float
    vol_threshold: float
    block_length: int = Field(ge=2)
    n_boot: int
    confidence: float
    ridge_sharpe_high: float
    ridge_ci: tuple[float, float]
    ridge_sharpe_low: float
    equal_weight_sharpe_high: float
    equal_weight_ci: tuple[float, float]


def _sharpe(x: FloatArray) -> float:
    if len(x) < 2:
        return math.nan
    sd = float(x.std(ddof=1))
    return 0.0 if sd == 0.0 else float(x.mean()) / sd * math.sqrt(TRADING_DAYS)


def stress_diagnostic(
    ridge: FloatArray,
    equal_weight: FloatArray,
    returns: FloatArray,
    dates: IntArray,
    *,
    n_boot: int,
    confidence: float,
    gen: np.random.Generator,
) -> StressDiagnostic:
    """The diagnostic (module docstring). ``ridge`` and ``equal_weight`` are the walk-forward's
    P&L rows, earned on ``dates + 1 + lag``; ``returns`` is the (T, N) panel the state is built
    from; ``dates`` are the decision dates."""
    states, threshold = volatility_state(np.asarray(returns).mean(axis=1), dates)
    block = certification_block_length(ridge, states)
    hi = states == 1
    boot_r, boot_e = [], []
    for _ in range(n_boot):
        idx = moving_block_indices(len(states), block, gen)
        sel = idx[hi[idx]]
        boot_r.append(_sharpe(ridge[sel]))
        boot_e.append(_sharpe(equal_weight[sel]))
    tail = (1.0 - confidence) / 2.0

    def ci(values: list[float]) -> tuple[float, float]:
        v = np.asarray(values)
        v = v[np.isfinite(v)]
        return float(np.quantile(v, tail)), float(np.quantile(v, 1.0 - tail))

    return StressDiagnostic(
        high_share=float(hi.mean()),
        vol_threshold=threshold,
        block_length=block,
        n_boot=n_boot,
        confidence=confidence,
        ridge_sharpe_high=_sharpe(ridge[hi]),
        ridge_ci=ci(boot_r),
        ridge_sharpe_low=_sharpe(ridge[~hi]),
        equal_weight_sharpe_high=_sharpe(equal_weight[hi]),
        equal_weight_ci=ci(boot_e),
    )
