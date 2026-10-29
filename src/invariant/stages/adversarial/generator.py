"""Block-type Markov-chain bootstrap generator for Stage 4 (docs/adr/0016).

Method
------
The observed out-of-sample P&L days carry Stage 4's day-level volatility state
(``state.volatility_state``). Blocks are L consecutive observed days. L is chosen by the
Politis-White selector on |r| of the equal-weight return (``blocklength.stage4_block_length``),
so each block keeps the observed volatility clustering inside it.
- **Pool.** Every moving block (Kuensch 1989, "The jackknife and the bootstrap for general
  stationary observations", Ann. Statist. 17). A block's **type** is high if more than half of
  its days are in the high-volatility state, otherwise low (a tie is low).
- **Tiling.** The non-overlapping tiling of the observed path into blocks of L gives the fitted
  chain theta_hat (``BlockPool.chain_hat``): the share of high tiles and their mean run, in
  blocks.
- **A generated path.**
  1. A type sequence over ceil(length / L) slots, from a two-state chain (``StateChain``, in block
     units) that starts from a given type or from its stationary distribution.
  2. For each slot, a uniformly drawn pool block of that type that does not overlap the blocks
     of the previous m - 1 slots, with m = ceil(63 / L) + 1 (rejection sampling). Blocks d slots
     apart can share a day only at positions more than (d - 1) L apart. So no observed day
     appears twice within 63 positions: the longest lag any G0 check (20) or realism feature
     (three 21-day windows) uses. Strict no-overlap within a whole path is infeasible: a path
     needs more disjoint blocks than the observed path contains (ADR 0016).
  3. The concatenation, truncated to ``length``.

Every generated day is an *observed row index*, so it carries that day's whole row jointly: the
signals, every asset's return, the strategy's and 1/N's P&L, and its own day-level state. The
adversary moves only the chain (the type share and mean run, in blocks) and the starting type.

Assumptions and limitations:
- Dependence across block joins is lost.
- A high-type block contains low-state days, and the reverse, so the adversary's lever is
  diluted.
- Spells are controlled only in whole blocks, which is coarse for a horizon of a few blocks
  (ADR 0016).
- The strategy's P&L is the walk-forward's realised P&L (a fixed policy; no refitting on
  generated history).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["NO_REPEAT_WITHIN", "BlockPool", "Start", "StateChain", "no_overlap_slots", "state_path"]

NO_REPEAT_WITHIN = 63
"""No observed day may appear twice within this many positions of a generated path."""
_MAX_REJECTIONS = 10_000

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class Start(StrEnum):
    """The type of the slot before a generated path (its first slot is drawn from that row of P)."""

    LOW = "low"
    HIGH = "high"
    STATIONARY = "stationary"


class StateChain(BaseModel):
    """Two-state chain (0 = low, 1 = high) by high share and mean high run (in chain steps)."""

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


def no_overlap_slots(block_length: int) -> int:
    """m = ceil(NO_REPEAT_WITHIN / L) + 1: a block may not overlap the previous m - 1 slots."""
    return math.ceil(NO_REPEAT_WITHIN / block_length) + 1


@dataclass(frozen=True)
class BlockPool:
    """Moving blocks of ``length`` observed days, by type (module docstring)."""

    length: int
    n_days: int
    starts: tuple[IntArray, IntArray]
    tile_types: IntArray
    coverage: tuple[FloatArray, FloatArray]
    """``coverage[k][d]``: how many type-k pool blocks contain day d (weights for type moments)."""

    @classmethod
    def from_states(cls, states: IntArray, length: int) -> BlockPool:
        x = np.asarray(states, dtype=np.int64)
        n = len(x)
        if x.ndim != 1 or not np.isin(x, (0, 1)).all():
            raise ValueError("states must be a 1-d sequence of 0 and 1")
        if not 2 <= length <= n // 2:
            raise ValueError(f"block length {length} must be in [2, n/2] for n = {n}")
        cs = np.concatenate([[0], np.cumsum(x)])
        high_share = (cs[length:] - cs[:-length]) / length
        types = (high_share > 0.5).astype(np.int64)
        tiles = types[::length][: n // length]
        if not 0 < tiles.sum() < len(tiles):
            raise ValueError("the tiling needs both block types")
        starts = (np.flatnonzero(types == 0), np.flatnonzero(types == 1))
        coverage = []
        for k in (0, 1):
            diff = np.zeros(n + 1)
            np.add.at(diff, starts[k], 1.0)
            np.add.at(diff, starts[k] + length, -1.0)
            coverage.append(np.cumsum(diff)[:n])
        return cls(
            length=length,
            n_days=n,
            starts=(starts[0].astype(np.int64), starts[1].astype(np.int64)),
            tile_types=tiles.astype(np.int64),
            coverage=(coverage[0], coverage[1]),
        )

    @property
    def chain_hat(self) -> StateChain:
        return StateChain.from_states(self.tile_types)

    def _blocks(self, types: IntArray, gen: np.random.Generator) -> IntArray:
        picks = np.empty(len(types), dtype=np.int64)
        for k in (0, 1):
            sel = types == k
            picks[sel] = self.starts[k][gen.integers(len(self.starts[k]), size=int(sel.sum()))]
        return np.asarray((picks[:, None] + np.arange(self.length)[None, :]).ravel())

    def generate(
        self,
        chain: StateChain,
        start: Start,
        length: int,
        gen: np.random.Generator,
    ) -> IntArray:
        """Observed row indices of one generated path of ``length`` days (local no-overlap)."""
        n_slots = math.ceil(length / self.length)
        types = state_path(chain, start, n_slots, gen)
        back = no_overlap_slots(self.length) - 1
        picks: list[int] = []
        for j, k in enumerate(types.tolist()):
            pool = self.starts[k]
            recent = picks[max(0, j - back) : j]
            for _ in range(_MAX_REJECTIONS):
                s = int(pool[gen.integers(len(pool))])
                if all(abs(s - r) >= self.length for r in recent):
                    break
            else:
                raise RuntimeError("no non-overlapping block found; the type pool is too small")
            picks.append(s)
        starts = np.array(picks, dtype=np.int64)
        return np.asarray((starts[:, None] + np.arange(self.length)[None, :]).ravel()[:length])

    def resample(self, gen: np.random.Generator) -> tuple[IntArray, IntArray]:
        """One type-stratified replicate: as many type-k pool blocks as the tiling has type-k
        tiles, returned as the day indices of each type (Stage 4's interval). Not a path: only
        per-type means and sds are computed from it, so block adjacency is irrelevant and the
        local no-overlap rule does not apply."""
        out = []
        for k in (0, 1):
            n_k = int((self.tile_types == k).sum())
            out.append(self._blocks(np.full(n_k, k, dtype=np.int64), gen))
        return out[0], out[1]

    def type_moments(self, pnl: FloatArray) -> tuple[FloatArray, FloatArray]:
        """P&L mean and sd (ddof 0) per block type, over the days of all pool blocks of that type
        (coverage-weighted: the day distribution of a uniformly drawn type-k block)."""
        mu, sd = np.empty(2), np.empty(2)
        for k in (0, 1):
            w = self.coverage[k]
            mu[k] = float(w @ pnl) / float(w.sum())
            sd[k] = math.sqrt(max(float(w @ (pnl - mu[k]) ** 2) / float(w.sum()), 0.0))
        return mu, sd
