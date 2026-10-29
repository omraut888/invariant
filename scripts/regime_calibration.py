"""Population calibration of the v4 `regime` preset (ADR 0015).

Fixed before any Stage 4 code runs; uses only population quantities and the regime-blind
projection oracle, never a stage's output. Held fixed: everything of `multi_asset`, the chain
(stress share .2, expected stress spell 120 days), the vol ratios (residual 2, market 2.5), and
S_regime's persistence rho = 0 (a persistent S_regime adds return autocorrelation: C3).

Candidates: ic in {.10, .12, .15, .20} x stress_ratio in {-1, -1.5, -2}. Criteria:
- C1: S_regime's stationary-average expected_t_hybrid >= 4 on every asset (min over assets),
  for each of the 5 calibration loading draws (s_real's is about 4.4);
- C2: oracle single-signal stress-regime Sharpe <= -1.0 for S_regime and >= +0.3 for s_real
  (mean over 5 seed pairs, 10^6-day paths);
- C3: max population |ACF(R)| over lags 1..20 <= multi_asset's, every asset and draw (same
  market betas and loadings, so the comparison is paired).
Rule: the passing candidate with the smallest ic, then the smallest |stress_ratio| (the weakest
planted signal that meets every criterion).

Usage: uv run python scripts/regime_calibration.py [--out docs/results/0015-regime-calibration]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.regime_oracle import oracle_regime_sharpe
from invariant.benchmark.synthetic import (
    REGIME_SIGNAL,
    draw_loadings,
    population_moments,
    population_return_acf,
    state_nodes,
)
from invariant.core.run_hash import RngFactory

ROOT_SEED = 20261031
N_DRAWS = 5
ORACLE_N_OBS = 1_000_000
C = 0.5 / math.sqrt(252.0)
ICS = (0.10, 0.12, 0.15, 0.20)
RATIOS = (-1.0, -1.5, -2.0)


def _seed(kind: str, i: int) -> int:
    return int(RngFactory(ROOT_SEED).child(f"{kind}/{i}").generator("seed").integers(2**32))


def _candidate(ic: float, ratio: float) -> SyntheticMarketConfig:
    base = load_benchmark_config("regime")
    assert base.regime is not None
    signal = base.regime.signal.model_copy(update={"rho": 0.0, "ic": ic, "stress_ratio": ratio})
    return base.model_copy(update={"regime": base.regime.model_copy(update={"signal": signal})})


def _c1_c3(cfg: SyntheticMarketConfig) -> dict[str, float]:
    ref = load_benchmark_config("multi_asset")
    k = state_nodes(cfg).index(REGIME_SIGNAL)
    root_n = math.sqrt(cfg.n_obs - 1)
    t_min, acf_ratio_max = math.inf, 0.0
    for i in range(N_DRAWS):
        seed = _seed("loadings", i)
        loadings = draw_loadings(cfg, RngFactory(seed))
        ref_loadings = draw_loadings(ref, RngFactory(seed))
        for m in population_moments(cfg, loadings):
            t_min = min(t_min, float(m.effective_hybrid_ic[k]) * root_n)
        for a in range(cfg.n_assets):
            own = float(np.abs(population_return_acf(cfg, loadings, a, 20)).max())
            other = float(np.abs(population_return_acf(ref, ref_loadings, a, 20)).max())
            acf_ratio_max = max(acf_ratio_max, own / other)
    return {"c1_min_expected_t_hybrid": t_min, "c3_max_acf_ratio": acf_ratio_max}


def _oracle(ic: float, ratio: float, member: str, i: int) -> dict[str, float | str | int]:
    cfg = _candidate(ic, ratio).model_copy(update={"n_obs": ORACLE_N_OBS})
    o = oracle_regime_sharpe(
        cfg, member, train_seed=_seed("train", i), test_seed=_seed("test", i), c=C
    )
    return {"ic": ic, "ratio": ratio, "member": member, "draw": i, **vars(o)}


def _line(r: Mapping[str, object]) -> str:
    g = REGIME_SIGNAL
    out = (
        f"ic {r['ic']:.2f} ratio {r['ratio']:+.1f} | C1 t_min {r['c1_min_expected_t_hybrid']:.2f}"
        f" C3 acf {r['c3_max_acf_ratio']:.3f}"
    )
    if f"{g}_stress" in r:
        out += (
            f" | C2 regime stress {r[f'{g}_stress']:+.2f} (calm {r[f'{g}_calm']:+.2f},"
            f" stat {r[f'{g}_stationary']:+.2f}) real stress {r['s_real_stress']:+.2f}"
            f" | 1/N stat {r['s_real_equal_weight_stationary']:+.2f}"
        )
    return out + ("  PASS" if r["passes"] else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/results/0015-regime-calibration"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    rows = []
    for ic, ratio in itertools.product(ICS, RATIOS):
        rows.append({"ic": ic, "ratio": ratio, **_c1_c3(_candidate(ic, ratio))})
    jobs = [
        (r["ic"], r["ratio"], m, i)
        for r in rows
        if r["c1_min_expected_t_hybrid"] >= 4.0 and r["c3_max_acf_ratio"] <= 1.0
        for m in (REGIME_SIGNAL, "s_real")
        for i in range(N_DRAWS)
    ]
    oracle = Parallel(n_jobs=args.jobs)(delayed(_oracle)(*j) for j in jobs)
    for r in rows:
        for m in (REGIME_SIGNAL, "s_real"):
            mine = [
                o for o in oracle if (o["ic"], o["ratio"], o["member"]) == (r["ic"], r["ratio"], m)
            ]
            if mine:
                for key in ("calm", "stress", "stationary", "equal_weight_stationary"):
                    vals = np.array([float(o[key]) for o in mine])
                    r[f"{m}_{key}"] = float(vals.mean())
                    r[f"{m}_{key}_se"] = float(vals.std(ddof=1) / math.sqrt(len(vals)))
        r["passes"] = bool(
            r["c1_min_expected_t_hybrid"] >= 4.0
            and r["c3_max_acf_ratio"] <= 1.0
            and r.get(f"{REGIME_SIGNAL}_stress", math.inf) <= -1.0
            and r.get("s_real_stress", -math.inf) >= 0.3
        )
    passing = sorted((r for r in rows if r["passes"]), key=lambda r: (r["ic"], abs(r["ratio"])))
    chosen = passing[0] if passing else None
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "root_seed": ROOT_SEED,
        "oracle_n_obs": ORACLE_N_OBS,
        "n_draws": N_DRAWS,
        "c": C,
        "candidates": rows,
        "oracle_runs": oracle,
        "chosen": chosen,
    }
    (args.out / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        print(_line(r))
    print("chosen:", None if chosen is None else (chosen["ic"], chosen["ratio"]))


if __name__ == "__main__":
    main()
