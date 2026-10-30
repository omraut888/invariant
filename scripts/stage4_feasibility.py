"""Stage 4 feasibility check on population quantities (ADR 0016; no pilot or study seed).

On one long `volfrag` path (seed 20261104, used by nothing else), Stage 3's walk-forward
(Ridge-CV and 1/N, configs/base.yaml, lag 0) gives the per-state daily P&L moments of
{s_volfrag} and of {s_real} under the vol21, q = .70 state: population moments, up to Monte
Carlo error.

The path is cut into non-overlapping windows of 4739 days, the study's out-of-sample length.
Each window's state is computed with that window's own quantile, as a real run would, and its
95% region is fitted exactly as ``certify`` does (block length, 999 replicates, the grid). The
adversary is then evaluated with the **population** moments. The result is SR_H* without
estimation error in the moments, over regions of the width a real run gets: what the design
would decide with perfect knowledge of the strategy's per-state P&L.

Usage: uv run python scripts/stage4_feasibility.py [--n-obs N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
from joblib import Parallel, delayed

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.config.settings import load_config
from invariant.core.run_hash import RngFactory
from invariant.stages.adversarial.certify import CertificationConfig, certification_block_length
from invariant.stages.adversarial.chain import Start, StateChain
from invariant.stages.adversarial.objective import (
    TRADING_DAYS,
    horizon_weights,
    mixture_sharpe,
    state_moments,
)
from invariant.stages.adversarial.region import fit_region
from invariant.stages.adversarial.state import volatility_state
from invariant.stages.dro.walkforward import Arm, run_walk_forward
from invariant.stages.portfolio.stage import walk_config

SEED = 20261104
WINDOW_DAYS = 4739


class Window(TypedDict):
    window: int
    block_length: int
    region_share: tuple[float, float]
    region_duration: tuple[float, float]
    sr_h_star_population: float
    worst_share: float
    worst_duration: float
    worst_start: str
    horizon_high_share: float


def _walk(n_obs: int, member: str) -> dict[str, list[float] | list[int]]:
    d = generate_market(load_benchmark_config("volfrag").model_copy(update={"n_obs": n_obs}), SEED)
    r = np.ascontiguousarray(d.returns.to_numpy())
    x = np.ascontiguousarray(d.features[[member]].to_numpy())
    wf = walk_config(load_config().portfolio, 0)
    paths = run_walk_forward(
        x, r, [Arm.EQUAL_WEIGHT, Arm.RIDGE_CV], wf, RngFactory(SEED).child(member)
    )
    rc = paths[Arm.RIDGE_CV]
    return {
        "dates": rc.dates.tolist(),
        "pnl": rc.returns.tolist(),
        "pnl_ew": paths[Arm.EQUAL_WEIGHT].returns.tolist(),
        "ew_return": r.mean(axis=1).tolist(),
    }


def _adversary(
    mu: np.ndarray, sd: np.ndarray, region_chains: list[StateChain], horizon: int
) -> tuple[float, float, float, str, float]:
    best = (math.inf, 0.0, 0.0, "", 0.0)
    for c in region_chains:
        for s in Start:
            w = horizon_weights(c, s, horizon)
            sr = mixture_sharpe(mu, sd, w)
            if sr < best[0]:
                best = (sr, c.share, c.duration, s.value, float(w[1]))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=60_000)
    parser.add_argument("--out", type=Path, default=Path("docs/results/0016-feasibility"))
    args = parser.parse_args()
    cfg = CertificationConfig()
    members = ("s_volfrag", "s_real")
    walks = dict(
        zip(
            members, Parallel(n_jobs=2)(delayed(_walk)(args.n_obs, m) for m in members), strict=True
        )
    )
    report: dict[str, object] = {"seed": SEED, "n_obs": args.n_obs, "window_days": WINDOW_DAYS}
    for m in members:
        w = walks[m]
        dates = np.array(w["dates"], dtype=np.int64)
        pnl, ew = np.array(w["pnl"]), np.array(w["ew_return"])
        states, _ = volatility_state(ew, dates)
        mu, sd = state_moments(pnl, states)
        per_state = [float(mu[k] / sd[k] * math.sqrt(TRADING_DAYS)) for k in (0, 1)]
        windows: list[Window] = []
        for k in range(len(dates) // WINDOW_DAYS):
            sl = slice(k * WINDOW_DAYS, (k + 1) * WINDOW_DAYS)
            st_w, _ = volatility_state(ew, dates[sl])
            block = certification_block_length(pnl[sl], st_w)
            region = fit_region(
                st_w, block, cfg.n_region_boot, cfg.region_level, np.random.default_rng(1000 + k)
            )
            sr, share, dur, start, high_w = _adversary(mu, sd, region.grid(cfg.n_grid), cfg.horizon)
            windows.append(
                Window(
                    window=k,
                    block_length=block,
                    region_share=region.share_range,
                    region_duration=(
                        math.exp(region.log_duration_range[0]),
                        math.exp(region.log_duration_range[1]),
                    ),
                    sr_h_star_population=sr,
                    worst_share=share,
                    worst_duration=dur,
                    worst_start=start,
                    horizon_high_share=high_w,
                )
            )
        srs = [x["sr_h_star_population"] for x in windows]
        highs = [x["horizon_high_share"] for x in windows]
        report[m] = {
            "n_days": len(pnl),
            "per_state_sharpe": per_state,
            "per_state_mu": mu.tolist(),
            "per_state_sd": sd.tolist(),
            "stationary_sharpe": mixture_sharpe(mu, sd, np.array([0.7, 0.3])),
            "sr_h_star_median": float(np.median(srs)),
            "sr_h_star_range": (float(min(srs)), float(max(srs))),
            "horizon_high_share_median": float(np.median(highs)),
            "windows": windows,
        }
        high_median = float(np.median(highs))
        print(
            f"{m}: per-state SR low {per_state[0]:+.2f} high {per_state[1]:+.2f}; "
            f"SR_H* at population moments median {np.median(srs):+.2f} "
            f"[{min(srs):+.2f}, {max(srs):+.2f}] over {len(windows)} windows; "
            f"worst horizon high share median {high_median:.2f}"
        )
        for x in windows:
            rs, rd = x["region_share"], x["region_duration"]
            print(
                f"   w{x['window']:2d} L {x['block_length']:3d} "
                f"region pi {rs[0]:.2f}-{rs[1]:.2f} d {rd[0]:.1f}-{rd[1]:.1f} | "
                f"worst pi {x['worst_share']:.2f} d {x['worst_duration']:.1f} "
                f"start {x['worst_start']} -> high share {x['horizon_high_share']:.2f}, "
                f"SR_H* {x['sr_h_star_population']:+.2f}"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
