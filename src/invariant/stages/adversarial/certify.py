"""Stage 4's certification of a strategy's out-of-sample P&L (docs/adr/0016, option 4).

Method
------
Inputs: the strategy's and 1/N's realised daily P&L from Stage 3's walk-forward, aligned with the
observable volatility state of each P&L day (``state.volatility_state``: vol21, q = .70).

1. **Block length.** L = max(2, round(b_CB)), taking b_CB as the larger of the Politis & White
   (2004) / Patton, Politis & White (2009) circular-block estimates on the state indicator and on
   |strategy P&L|. Both series' dependence must survive the bootstrap (``blocklength.py``).
2. **Confidence region** for the state's chain (pi_s, d_s): the bootstrap's own 95% Mahalanobis
   contour in (pi, log d) from a moving-block bootstrap of the state pairs (``region.py``).
3. **Adversary.** An exhaustive grid over the chains in the region (an n x n grid over the
   replicates' range, plus the point estimate) and over the starting state
   {low, high, stationary}. SR_H* is the minimum horizon Sharpe ratio (``objective.sr_h``, day
   level, H = 252). With two states this is a corner of the region whenever the strategy does
   worse in one state; the grid confirms that rather than discovering it.
4. **Interval.** ``n_boot`` moving-block replicates of the joint daily series (P&L, state). Each
   re-estimates the per-state moments and **re-minimises over the region's chains**, so the
   adversary's selection is inside the interval. The region itself is held fixed.
5. **Kill rule A** (user, 2026-10-29): kill iff the ``upper_level`` (one-sided 95%) bootstrap
   quantile of SR_H* is below 0.

Reported only:
- SR_H of 1/N at the worst point, and the strategy's minimum difference to 1/N over the region;
- the bootstrap distribution of SR_H* (quantiles), the worst-case Sharpe distribution;
- a GPD peaks-over-threshold tail fit (Pickands 1975; Balkema & de Haan 1974) of the observed
  daily losses, overall and in the high-volatility state, with ES_99 and moving-block bootstrap
  intervals.

Assumptions and limitations:
- **Markov approximation**: horizon weights use a first-order chain matched to (pi_s, d_s),
  while the observed state is not first-order Markov (``chain.py``).
- **Fixed policy**: the per-state P&L is the walk-forward's realised P&L; nothing refits under a
  changed state mix.
- The region is conditional on one observed path.
- The GPD fit ignores the clustering of extremes, so its intervals are optimistic.
- No paths are generated. The earlier generators failed G0 (ADR 0016), and paths were only ever
  needed for the realism gate.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from invariant.core.run_hash import RngFactory
from invariant.stages.adversarial.blocklength import optimal_block_length
from invariant.stages.adversarial.chain import Start, StateChain
from invariant.stages.adversarial.objective import (
    TRADING_DAYS,
    horizon_weights,
    sr_h,
    state_moments,
)
from invariant.stages.adversarial.region import ChainRegion, fit_region, moving_block_indices

__all__ = [
    "CertificationConfig",
    "CertificationResult",
    "GpdFit",
    "certification_block_length",
    "certify",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CertificationConfig(_Frozen):
    horizon: int = Field(default=252, ge=1)
    region_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    n_region_boot: int = Field(default=999, ge=19)
    n_grid: int = Field(default=25, ge=2)
    n_boot: int = Field(default=999, ge=19)
    upper_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    block_length: int | None = None
    """None: the Politis-White rule (``certification_block_length``)."""
    gpd_tail: float = Field(default=0.95, gt=0.5, lt=1.0)
    es_level: float = Field(default=0.99, gt=0.5, lt=1.0)
    gpd_n_boot: int = Field(default=199, ge=0)


class GpdFit(_Frozen):
    threshold: float
    exceed_share: float
    xi: float
    sigma: float
    es: float
    es_ci: tuple[float, float] | None
    xi_ci: tuple[float, float] | None


class CertificationResult(_Frozen):
    """Everything Stage 4 decided on and reported (module docstring). Sharpe ratios annualised."""

    n_days: int
    block_length: int
    region: ChainRegion
    n_candidates: int
    mu: tuple[float, float]
    sd: tuple[float, float]
    sharpe_by_state: tuple[float, float]
    equal_weight_sharpe_by_state: tuple[float, float]
    worst_share: float
    worst_duration: float
    worst_start: Start
    sr_h_star: float
    sr_h_star_upper: float
    sr_h_star_quantiles: tuple[float, ...]
    kill: bool
    sr_h_baseline: float
    sr_h_equal_weight_at_worst: float
    min_difference_to_equal_weight: float
    gpd_all: GpdFit | None
    gpd_high: GpdFit | None


def certification_block_length(pnl: FloatArray, states: IntArray) -> int:
    """max(2, round(max b_CB)) over the state indicator and |pnl| (Politis-White, ADR 0016)."""
    b = max(
        optimal_block_length(np.asarray(states, dtype=np.float64)).circular,
        optimal_block_length(np.abs(np.asarray(pnl, dtype=np.float64))).circular,
    )
    return max(2, round(b))


def _sharpe(x: FloatArray) -> float:
    sd = float(x.std(ddof=1))
    return 0.0 if sd == 0.0 else float(x.mean()) / sd * math.sqrt(TRADING_DAYS)


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


def _gpd_report(
    losses: FloatArray,
    block: int,
    config: CertificationConfig,
    gen: np.random.Generator,
) -> GpdFit | None:
    if np.count_nonzero(losses) < 200:  # e.g. a strategy that is almost always in cash
        return None
    u, zeta, xi, sigma, es = _gpd(losses, config.gpd_tail, config.es_level)
    es_b, xi_b = [], []
    for _ in range(config.gpd_n_boot):
        idx = moving_block_indices(len(losses), min(block, len(losses)), gen)
        res = _gpd(losses[idx], config.gpd_tail, config.es_level)
        xi_b.append(res[2])
        es_b.append(res[4])
    have = config.gpd_n_boot > 0
    return GpdFit(
        threshold=u,
        exceed_share=zeta,
        xi=xi,
        sigma=sigma,
        es=es,
        es_ci=(float(np.quantile(es_b, 0.025)), float(np.quantile(es_b, 0.975))) if have else None,
        xi_ci=(float(np.quantile(xi_b, 0.025)), float(np.quantile(xi_b, 0.975))) if have else None,
    )


def certify(
    pnl: FloatArray,
    pnl_equal_weight: FloatArray,
    states: IntArray,
    config: CertificationConfig,
    rng: RngFactory,
) -> CertificationResult:
    """Stage 4 on aligned daily arrays (module docstring). Streams: ``region``, ``boot``,
    ``gpd_all``, ``gpd_high``."""
    pnl = np.ascontiguousarray(pnl, dtype=np.float64)
    pnl_ew = np.ascontiguousarray(pnl_equal_weight, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    n = len(pnl)
    if not len(pnl_ew) == len(states) == n:
        raise ValueError("pnl, pnl_equal_weight and states must be aligned")
    if not (np.all(np.isfinite(pnl)) and np.all(np.isfinite(pnl_ew))):
        raise ValueError("P&L must be finite")

    block = config.block_length or certification_block_length(pnl, states)
    region = fit_region(
        states, block, config.n_region_boot, config.region_level, rng.generator("region")
    )
    chains = region.grid(config.n_grid)
    candidates = [(c, s) for c in chains for s in Start]
    weights = np.array([horizon_weights(c, s, config.horizon) for c, s in candidates])

    def min_sr(mu: FloatArray, sd: FloatArray) -> tuple[float, int]:
        mean = weights @ mu
        var = weights @ (sd * sd + mu * mu) - mean * mean
        sr = np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-300)), 0.0)
        k = int(np.argmin(sr))
        return float(sr[k]) * math.sqrt(TRADING_DAYS), k

    mu, sd = state_moments(pnl, states)
    sr_star, k_star = min_sr(mu, sd)
    worst_chain, worst_start = candidates[k_star]

    boot_gen = rng.generator("boot")
    boot = np.empty(config.n_boot)
    for b in range(config.n_boot):
        idx = moving_block_indices(n, block, boot_gen)
        s_b, p_b = states[idx], pnl[idx]
        if not 1 < s_b.sum() < n - 1:  # a replicate with (almost) one state: no moments
            boot[b] = sr_star
            continue
        boot[b] = min_sr(*state_moments(p_b, s_b))[0]
    upper = float(np.quantile(boot, config.upper_level))

    mu_ew, sd_ew = state_moments(pnl_ew, states)
    diffs = [
        sr_h(mu, sd, c, s, config.horizon) - sr_h(mu_ew, sd_ew, c, s, config.horizon)
        for c, s in candidates
    ]
    hat = StateChain(share=region.share_hat, duration=region.duration_hat)
    return CertificationResult(
        n_days=n,
        block_length=block,
        region=region,
        n_candidates=len(candidates),
        mu=(float(mu[0]), float(mu[1])),
        sd=(float(sd[0]), float(sd[1])),
        sharpe_by_state=(_sharpe(pnl[states == 0]), _sharpe(pnl[states == 1])),
        equal_weight_sharpe_by_state=(_sharpe(pnl_ew[states == 0]), _sharpe(pnl_ew[states == 1])),
        worst_share=worst_chain.share,
        worst_duration=worst_chain.duration,
        worst_start=worst_start,
        sr_h_star=sr_star,
        sr_h_star_upper=upper,
        sr_h_star_quantiles=tuple(
            float(v) for v in np.quantile(boot, (0.05, 0.25, 0.5, 0.75, 0.95))
        ),
        kill=upper < 0.0,
        sr_h_baseline=sr_h(mu, sd, hat, Start.STATIONARY, config.horizon),
        sr_h_equal_weight_at_worst=sr_h(mu_ew, sd_ew, worst_chain, worst_start, config.horizon),
        min_difference_to_equal_weight=float(min(diffs)),
        gpd_all=_gpd_report(-pnl, block, config, rng.generator("gpd_all")),
        gpd_high=_gpd_report(-pnl[states == 1], block, config, rng.generator("gpd_high")),
    )
