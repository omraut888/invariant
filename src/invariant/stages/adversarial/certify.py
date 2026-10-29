"""Stage 4's certification of a strategy's out-of-sample P&L (docs/adr/0016).

Method
------
Inputs: the strategy's and 1/N's realised daily P&L from Stage 3's walk-forward, the asset
returns on those days, and the observable volatility state of each P&L day
(``state.volatility_state``).

0. **Blocks.** L = ``blocklength.stage4_block_length`` of the equal-weight return on the P&L
   days; the pool of moving blocks is typed by majority state (``generator.BlockPool``).
1. **Realism gate** (``realism.py``). The fitted chain theta_hat is the tiling's high-type share
   and mean high run (in blocks). The null is ``n_null`` generated paths at theta_hat; the
   threshold is their (1 - alpha) order statistic.
2. **Plausible set.** Every grid point (high-type share x mean high run in blocks, valid chains
   only) whose MMD^2 is within the threshold, plus theta_hat. Grid points share one random
   stream (common random numbers), so the boundary is smooth.
3. **Adversary.** An exhaustive grid over the plausible set and the starting state
   {low, high, stationary}. SR_H* is the minimum horizon Sharpe ratio (``objective.sr_h``). With
   two states this is a corner of the plausible set whenever the strategy does worse in one
   state; the grid confirms that rather than discovering it.
4. **Interval.** ``n_boot`` type-stratified replicates of the observed P&L
   (``BlockPool.resample``). Each replicate re-estimates the per-type moments and
   **re-minimises over the plausible set**, so the adversary's selection is inside the interval.
   The plausible set itself is held fixed.
5. **Kill rule A** (user, 2026-10-29): kill iff the ``upper_level`` (one-sided 95%) bootstrap
   quantile of SR_H* is below 0.

Reported only:
- SR_H of 1/N at the worst point, and the strategy's minimum difference to 1/N over the set;
- the distribution of realised H-day Sharpe ratios on paths generated at the worst point;
- a GPD tail fit (peaks over a threshold; Pickands 1975, Balkema & de Haan 1974) of daily losses
  on those paths, with ES_99 and bootstrap intervals over paths.

Assumptions and limitations:
- the fixed-policy approximation (the P&L does not refit on generated history);
- the plausible set is conditional on one observed path;
- the GPD fit ignores the clustering of extremes (no declustering), so its intervals are optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from invariant.core.run_hash import RngFactory
from invariant.stages.adversarial.blocklength import stage4_block_length
from invariant.stages.adversarial.generator import BlockPool, Start, StateChain
from invariant.stages.adversarial.objective import TRADING_DAYS, horizon_weights, sr_h
from invariant.stages.adversarial.realism import RealismReference, null_threshold

__all__ = [
    "CertificationConfig",
    "CertificationResult",
    "GridPoint",
    "RealismGate",
    "certify",
    "fit_realism",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CertificationConfig(_Frozen):
    horizon: int = Field(default=252, ge=1)
    path_multiple: int = Field(default=4, ge=1)
    n_null: int = Field(default=199, ge=19)
    realism_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    share_low: float = 0.02
    share_high: float = 0.80
    n_share: int = Field(default=16, ge=2)
    duration_high_days: float = 1000.0
    """The spell axis runs from 1 block to this many days, in blocks, log-spaced."""
    n_duration: int = Field(default=16, ge=2)
    n_boot: int = Field(default=999, ge=19)
    upper_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    block_length: int | None = None
    """None: ADR 0016's Politis-White rule (``stage4_block_length``)."""
    n_report_paths: int = Field(default=500, ge=10)
    gpd_tail: float = Field(default=0.95, gt=0.5, lt=1.0)
    es_level: float = Field(default=0.99, gt=0.5, lt=1.0)
    gpd_n_boot: int = Field(default=199, ge=0)

    def grid(self, block_length: int) -> list[StateChain]:
        shares = np.linspace(self.share_low, self.share_high, self.n_share)
        top = max(self.duration_high_days / block_length, 1.0 + 1e-9)
        durations = np.geomspace(1.0, top, self.n_duration)
        return [
            StateChain(share=float(s), duration=float(d))
            for s in shares
            for d in durations
            if StateChain.is_valid(float(s), float(d))
        ]


class GridPoint(_Frozen):
    share: float
    duration: float  # in blocks
    mmd2: float
    plausible: bool


class GpdFit(_Frozen):
    threshold: float
    exceed_share: float
    xi: float
    sigma: float
    es: float
    es_ci_low: float | None
    es_ci_high: float | None
    xi_ci_low: float | None
    xi_ci_high: float | None


class CertificationResult(_Frozen):
    """Everything Stage 4 decided on and reported (module docstring). Sharpe ratios annualised."""

    n_days: int
    high_share_observed: float
    block_length: int
    n_high_tiles: int
    n_tiles: int
    chain_hat: StateChain
    mu: tuple[float, float]
    sd: tuple[float, float]
    sharpe_by_type: tuple[float, float]
    sharpe_by_state: tuple[float, float]
    equal_weight_sharpe_by_state: tuple[float, float]
    bandwidth: float
    null_threshold: float
    grid: tuple[GridPoint, ...]
    n_plausible: int
    worst_share: float
    worst_duration: float  # in blocks
    worst_start: Start
    sr_h_star: float
    sr_h_star_upper: float
    sr_h_star_ci: tuple[float, float]
    kill: bool
    sr_h_baseline: float
    sr_h_equal_weight_at_worst: float
    min_difference_to_equal_weight: float
    path_sharpe_quantiles: tuple[float, ...]
    gpd: GpdFit | None


def _sharpe(x: FloatArray) -> float:
    sd = float(x.std(ddof=1))
    return 0.0 if sd == 0.0 else float(x.mean()) / sd * math.sqrt(TRADING_DAYS)


@dataclass(frozen=True)
class RealismGate:
    """A fitted realism gate: the observed reference, the generator inputs and the threshold."""

    reference: RealismReference
    pool: BlockPool
    returns: FloatArray
    length: int
    threshold: float

    @property
    def chain_hat(self) -> StateChain:
        return self.pool.chain_hat

    def statistic(self, chain: StateChain, gen: np.random.Generator) -> float:
        idx = self.pool.generate(chain, Start.STATIONARY, self.length, gen)
        return self.reference.statistic(self.returns[idx])


def fit_realism(
    returns: FloatArray,
    states: IntArray,
    config: CertificationConfig,
    rng: RngFactory,
) -> RealismGate:
    """Fit theta_hat and calibrate the realism threshold on the observed rows (``null/b``)."""
    block = config.block_length or stage4_block_length(returns.mean(axis=1))
    pool = BlockPool.from_states(states, block)
    ref = RealismReference.from_returns(returns)
    length = config.path_multiple * len(states)
    partial = RealismGate(ref, pool, returns, length, math.inf)
    chain_hat = pool.chain_hat
    null = np.array(
        [partial.statistic(chain_hat, rng.generator(f"null/{b}")) for b in range(config.n_null)]
    )
    threshold = null_threshold(null, config.realism_alpha)
    return RealismGate(ref, pool, returns, length, threshold)


def _gpd(losses: FloatArray, tail: float, level: float) -> tuple[float, float, float, float, float]:
    """``(u, zeta, xi, sigma, ES_level)`` of a peaks-over-threshold GPD fit (loc 0)."""
    u = float(np.quantile(losses, tail))
    excess = losses[losses > u] - u
    xi, _, sigma = stats.genpareto.fit(excess, floc=0.0)
    zeta = len(excess) / len(losses)
    p = 1.0 - level
    if abs(xi) < 1e-9:
        var = u + sigma * math.log(zeta / p)
    else:
        var = u + sigma / xi * ((p / zeta) ** (-xi) - 1.0)
    es = (var + sigma - xi * u) / (1.0 - xi) if xi < 1.0 else math.inf
    return u, zeta, float(xi), float(sigma), float(es)


def certify(
    pnl: FloatArray,
    pnl_equal_weight: FloatArray,
    returns: FloatArray,
    states: IntArray,
    config: CertificationConfig,
    rng: RngFactory,
) -> CertificationResult:
    """Stage 4 on aligned daily arrays (module docstring). Streams: ``realism/null/b``,
    ``realism/grid``, ``boot``, ``paths``, ``gpd``."""
    pnl = np.ascontiguousarray(pnl, dtype=np.float64)
    pnl_ew = np.ascontiguousarray(pnl_equal_weight, dtype=np.float64)
    returns = np.ascontiguousarray(returns, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    n = len(pnl)
    if not (len(pnl_ew) == len(states) == returns.shape[0] == n):
        raise ValueError("pnl, pnl_equal_weight, returns and states must be aligned")
    if not (np.all(np.isfinite(pnl)) and np.all(np.isfinite(pnl_ew))):
        raise ValueError("P&L must be finite")

    realism_rng = rng.child("realism")
    gate = fit_realism(returns, states, config, realism_rng)
    grid: list[GridPoint] = []
    plausible: list[StateChain] = [gate.chain_hat]
    # Common random numbers: every grid point gets a fresh generator from one SeedSequence, so
    # all points consume the same uniforms (explicitly; the factory forbids reissuing a name).
    grid_seed = realism_rng.seed_sequence("grid")
    pool = gate.pool
    block = pool.length
    for chain in config.grid(block):
        stat = gate.statistic(chain, np.random.Generator(np.random.PCG64(grid_seed)))
        ok = stat <= gate.threshold
        grid.append(GridPoint(share=chain.share, duration=chain.duration, mmd2=stat, plausible=ok))
        if ok:
            plausible.append(chain)

    candidates = [(c, s) for c in plausible for s in Start]
    weights = np.array([horizon_weights(c, s, config.horizon, block) for c, s in candidates])

    def min_sr(mu: FloatArray, sd: FloatArray) -> tuple[float, int]:
        mean = weights @ mu
        var = weights @ (sd * sd + mu * mu) - mean * mean
        sr = np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-300)), 0.0) * math.sqrt(
            TRADING_DAYS
        )
        k = int(np.argmin(sr))
        return float(sr[k]), k

    mu, sd = pool.type_moments(pnl)
    sr_star, k_star = min_sr(mu, sd)
    worst_chain, worst_start = candidates[k_star]

    boot_gen = rng.generator("boot")
    boot = np.empty(config.n_boot)
    for b in range(config.n_boot):
        i0, i1 = pool.resample(boot_gen)
        mu_b = np.array([pnl[i0].mean(), pnl[i1].mean()])
        sd_b = np.array([pnl[i0].std(), pnl[i1].std()])
        boot[b] = min_sr(mu_b, sd_b)[0]
    upper = float(np.quantile(boot, config.upper_level))
    lo_q = 1.0 - config.upper_level

    mu_ew, sd_ew = pool.type_moments(pnl_ew)
    diffs = [
        sr_h(mu, sd, c, s, config.horizon, block) - sr_h(mu_ew, sd_ew, c, s, config.horizon, block)
        for c, s in candidates
    ]

    path_gen = rng.generator("paths")
    paths = [
        pool.generate(worst_chain, worst_start, config.horizon, path_gen)
        for _ in range(config.n_report_paths)
    ]
    path_sr = np.array([_sharpe(pnl[idx]) for idx in paths])
    gpd = _gpd_report(pnl, paths, config, rng.generator("gpd"))

    return CertificationResult(
        n_days=n,
        high_share_observed=float(states.mean()),
        block_length=block,
        n_high_tiles=int(pool.tile_types.sum()),
        n_tiles=len(pool.tile_types),
        chain_hat=gate.chain_hat,
        mu=(float(mu[0]), float(mu[1])),
        sd=(float(sd[0]), float(sd[1])),
        sharpe_by_type=(
            float(mu[0] / sd[0] * math.sqrt(TRADING_DAYS)) if sd[0] > 0 else 0.0,
            float(mu[1] / sd[1] * math.sqrt(TRADING_DAYS)) if sd[1] > 0 else 0.0,
        ),
        sharpe_by_state=(_sharpe(pnl[states == 0]), _sharpe(pnl[states == 1])),
        equal_weight_sharpe_by_state=(_sharpe(pnl_ew[states == 0]), _sharpe(pnl_ew[states == 1])),
        bandwidth=gate.reference.bandwidth,
        null_threshold=gate.threshold,
        grid=tuple(grid),
        n_plausible=len(plausible),
        worst_share=worst_chain.share,
        worst_duration=worst_chain.duration,
        worst_start=worst_start,
        sr_h_star=sr_star,
        sr_h_star_upper=upper,
        sr_h_star_ci=(float(np.quantile(boot, lo_q)), upper),
        kill=upper < 0.0,
        sr_h_baseline=sr_h(mu, sd, gate.chain_hat, Start.STATIONARY, config.horizon, block),
        sr_h_equal_weight_at_worst=sr_h(
            mu_ew, sd_ew, worst_chain, worst_start, config.horizon, block
        ),
        min_difference_to_equal_weight=float(min(diffs)),
        path_sharpe_quantiles=tuple(
            float(v) for v in np.quantile(path_sr, (0.05, 0.25, 0.5, 0.75, 0.95))
        ),
        gpd=gpd,
    )


def _gpd_report(
    pnl: FloatArray,
    paths: list[IntArray],
    config: CertificationConfig,
    gen: np.random.Generator,
) -> GpdFit | None:
    """GPD tail of pooled daily losses on the reported paths; percentile CIs over paths."""
    losses = -np.concatenate([pnl[idx] for idx in paths])
    if np.count_nonzero(losses) < 100:  # e.g. a strategy that is almost always in cash
        return None
    u, zeta, xi, sigma, es = _gpd(losses, config.gpd_tail, config.es_level)
    es_b, xi_b = [], []
    for _ in range(config.gpd_n_boot):
        pick = gen.integers(len(paths), size=len(paths))
        res = _gpd(-np.concatenate([pnl[paths[j]] for j in pick]), config.gpd_tail, config.es_level)
        xi_b.append(res[2])
        es_b.append(res[4])
    have_ci = config.gpd_n_boot > 0
    return GpdFit(
        threshold=u,
        exceed_share=zeta,
        xi=xi,
        sigma=sigma,
        es=es,
        es_ci_low=float(np.quantile(es_b, 0.025)) if have_ci else None,
        es_ci_high=float(np.quantile(es_b, 0.975)) if have_ci else None,
        xi_ci_low=float(np.quantile(xi_b, 0.025)) if have_ci else None,
        xi_ci_high=float(np.quantile(xi_b, 0.975)) if have_ci else None,
    )
