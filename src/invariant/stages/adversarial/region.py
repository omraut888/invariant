"""Block-bootstrap confidence region for the volatility state's chain (docs/adr/0016).

Method
------
For the observed 0/1 state sequence x_1..x_n (``state.volatility_state``):
- **Estimates.** pi_hat = mean(x_t) over t < n, and d_hat = 1 / p_hat with
  p_hat = sum_t x_t (1 - x_{t+1}) / sum_t x_t, the high-to-low transition frequency. Both are smooth
  functions of means of the pairs (x_t, x_{t+1}), t = 1..n-1.
- **Replicates.** The moving-block bootstrap of those pairs (Kuensch 1989; blocks of L pairs,
  starts uniform) gives valid replicates of smooth functions of means under weak dependence.
  Because pairs are resampled whole, a block join never creates a spurious transition.
- **Region.** In z = (pi, log d), with the replicates' covariance S, the region is
  {z : (z - z_hat)' S^{-1} (z - z_hat) <= q}, where q is the
  ceil((B + 1) level)-th smallest replicate distance. It is the bootstrap's own Mahalanobis
  contour, calibrated by the replicates, not by a chi-square approximation.

Assumptions and limitations:
- The sequence is stationary and weakly dependent.
- The region is elliptical in (pi, log d), which suits approximately normal replicates.
- pi_hat equals 1 - q, the state's quantile, by construction in the sample, so the region's
  width in pi reflects sampling variability given the fixed threshold.
- A replicate with no high-to-low transition (d infinite) is dropped and counted.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from invariant.stages.adversarial.chain import StateChain

__all__ = ["ChainRegion", "chain_estimate", "fit_region", "moving_block_indices"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def chain_estimate(first: IntArray, second: IntArray) -> tuple[float, float]:
    """``(pi, d)`` from pairs ``(x_t, x_{t+1})``; d is ``inf`` with no high-to-low transition."""
    high = float(first.sum())
    if high == 0.0:
        return 0.0, math.inf
    exits = float((first * (1 - second)).sum())
    return high / len(first), (math.inf if exits == 0.0 else high / exits)


def moving_block_indices(n: int, block: int, gen: np.random.Generator) -> IntArray:
    """``n`` indices in [0, n): concatenated moving blocks of ``block`` with uniform starts."""
    if not 1 <= block <= n:
        raise ValueError("block must be in [1, n]")
    n_blocks = math.ceil(n / block)
    starts = gen.integers(0, n - block + 1, size=n_blocks)
    return np.asarray((starts[:, None] + np.arange(block)[None, :]).ravel()[:n], dtype=np.int64)


class ChainRegion(BaseModel):
    """A fitted region (module docstring). ``cov`` and ``center`` are in (pi, log d)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    share_hat: float
    duration_hat: float
    center: tuple[float, float]
    cov: tuple[tuple[float, float], tuple[float, float]]
    radius2: float
    level: float
    block_length: int
    n_boot: int
    n_dropped: int
    share_range: tuple[float, float]
    log_duration_range: tuple[float, float]

    def distance2(self, share: float, duration: float) -> float:
        z = np.array([share, math.log(duration)]) - np.array(self.center)
        return float(z @ np.linalg.solve(np.array(self.cov), z))

    def contains(self, share: float, duration: float) -> bool:
        return math.isfinite(duration) and self.distance2(share, duration) <= self.radius2

    def grid(self, n: int) -> list[StateChain]:
        """Valid chains on an n x n grid over the replicates' range that lie in the region, plus
        the point estimate."""
        out = [StateChain(share=self.share_hat, duration=self.duration_hat)]
        for s in np.linspace(*self.share_range, n):
            for ld in np.linspace(*self.log_duration_range, n):
                share, dur = float(s), math.exp(float(ld))
                if StateChain.is_valid(share, dur) and self.contains(share, dur):
                    out.append(StateChain(share=share, duration=dur))
        return out


def fit_region(
    states: IntArray,
    block: int,
    n_boot: int,
    level: float,
    gen: np.random.Generator,
) -> ChainRegion:
    """The ``level`` confidence region for (pi_s, d_s) of ``states`` (module docstring)."""
    x = np.asarray(states, dtype=np.int64)
    if x.ndim != 1 or not np.isin(x, (0, 1)).all():
        raise ValueError("states must be a 1-d sequence of 0 and 1")
    first, second = x[:-1], x[1:]
    share, duration = chain_estimate(first, second)
    if not (0.0 < share < 1.0 and math.isfinite(duration)):
        raise ValueError("the state sequence needs both states and a high-to-low transition")
    m = len(first)
    reps: list[tuple[float, float]] = []
    dropped = 0
    for _ in range(n_boot):
        idx = moving_block_indices(m, block, gen)
        s, d = chain_estimate(first[idx], second[idx])
        if 0.0 < s < 1.0 and math.isfinite(d):
            reps.append((s, math.log(d)))
        else:
            dropped += 1
    z = np.array(reps)
    if len(z) < 20:
        raise ValueError("too few usable bootstrap replicates for a region")
    center = np.array([share, math.log(duration)])
    cov = np.cov(z, rowvar=False)
    diff = z - center
    dist = np.einsum("bi,bi->b", diff @ np.linalg.inv(cov), diff)
    rank = min(math.ceil((len(dist) + 1) * level), len(dist))
    radius2 = float(np.sort(dist)[rank - 1])
    inside = z[dist <= radius2]
    return ChainRegion(
        share_hat=share,
        duration_hat=duration,
        center=(float(center[0]), float(center[1])),
        cov=((float(cov[0, 0]), float(cov[0, 1])), (float(cov[1, 0]), float(cov[1, 1]))),
        radius2=radius2,
        level=level,
        block_length=block,
        n_boot=n_boot,
        n_dropped=dropped,
        share_range=(float(inside[:, 0].min()), float(inside[:, 0].max())),
        log_duration_range=(float(inside[:, 1].min()), float(inside[:, 1].max())),
    )
