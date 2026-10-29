"""Population calibration of the v4.1 `volfrag` preset (ADR 0015 amendment).

The criteria and rule below were committed in f90d54b, before this script or s_volfrag existed.
Population and oracle quantities only; no stage output. Same draws as the regime calibration
(root seed 20261031, 5 loading and path draws, 10^6-day oracle paths).

Held fixed: the multi_asset market, rho = 0, and Stage 4's volatility state (vol21, q = .70,
``stages/adversarial/state.py``), which is never re-tuned to the signal.

Candidates: ic in {.10, .12, .15, .20} x stress_ratio in {-1, -1.5, -2} x h_quantile in
{.70, .80, .90}. Criteria:
- C1: s_volfrag's stationary-average expected_t_hybrid >= 4 on every asset of every draw;
- C2': s_real's oracle Sharpe in the high-volatility state >= +0.3;
- C3: max population |ACF(R)| over lags 1..20 <= multi_asset's, paired;
- C4: s_volfrag's oracle Sharpe in the high-volatility state <= -1.0 (mean over draws).
Rule: the passing candidate with the smallest ic, then the smallest |stress_ratio|, then the
highest h_quantile.

Usage: uv run python scripts/volfrag_calibration.py [--out docs/results/0015-volfrag-calibration]
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

from invariant.benchmark.config import SyntheticMarketConfig, VolFragParams, load_benchmark_config
from invariant.benchmark.regime_oracle import oracle_state_sharpe
from invariant.benchmark.synthetic import (
    VOLFRAG_SIGNAL,
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
QUANTILES = (0.70, 0.80, 0.90)


def _seed(kind: str, i: int) -> int:
    return int(RngFactory(ROOT_SEED).child(f"{kind}/{i}").generator("seed").integers(2**32))


def _candidate(ic: float, ratio: float, q: float) -> SyntheticMarketConfig:
    base = load_benchmark_config("volfrag")
    return base.model_copy(
        update={"volfrag": VolFragParams(ic=ic, stress_ratio=ratio, h_quantile=q)}
    )


def _c1_c3(cfg: SyntheticMarketConfig) -> dict[str, float]:
    ref = load_benchmark_config("multi_asset")
    k = state_nodes(cfg).index(VOLFRAG_SIGNAL)
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


def _oracle(ic: float, ratio: float, q: float, member: str, i: int) -> dict[str, object]:
    cfg = _candidate(ic, ratio, q).model_copy(update={"n_obs": ORACLE_N_OBS})
    o = oracle_state_sharpe(
        cfg, member, train_seed=_seed("train", i), test_seed=_seed("test", i), c=C
    )
    return {"ic": ic, "ratio": ratio, "q": q, "member": member, "draw": i, **vars(o)}


def _line(r: Mapping[str, object]) -> str:
    out = (
        f"ic {r['ic']:.2f} ratio {r['ratio']:+.1f} q {r['q']:.2f} | C1 t_min "
        f"{r['c1_min_expected_t_hybrid']:.2f} C3 acf {r['c3_max_acf_ratio']:.3f}"
    )
    if f"{VOLFRAG_SIGNAL}_high" in r:
        out += (
            f" | C4 volfrag high {r[f'{VOLFRAG_SIGNAL}_high']:+.2f}"
            f" (low {r[f'{VOLFRAG_SIGNAL}_low']:+.2f}) C2' real high {r['s_real_high']:+.2f}"
            f" | 1/N high {r['s_real_equal_weight_high']:+.2f}"
        )
    return out + ("  PASS" if r["passes"] else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/results/0015-volfrag-calibration"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for ic, ratio, q in itertools.product(ICS, RATIOS, QUANTILES):
        rows.append({"ic": ic, "ratio": ratio, "q": q, **_c1_c3(_candidate(ic, ratio, q))})
    jobs = [
        (r["ic"], r["ratio"], r["q"], m, i)
        for r in rows
        if float(r["c1_min_expected_t_hybrid"]) >= 4.0  # type: ignore[arg-type]
        and float(r["c3_max_acf_ratio"]) <= 1.0  # type: ignore[arg-type]
        for m in (VOLFRAG_SIGNAL, "s_real")
        for i in range(N_DRAWS)
    ]
    oracle = Parallel(n_jobs=args.jobs)(delayed(_oracle)(*j) for j in jobs)
    for r in rows:
        key = (r["ic"], r["ratio"], r["q"])
        for m in (VOLFRAG_SIGNAL, "s_real"):
            mine = [o for o in oracle if (o["ic"], o["ratio"], o["q"], o["member"]) == (*key, m)]
            if mine:
                for field in ("high", "low", "equal_weight_high"):
                    vals = np.array([float(o[field]) for o in mine])
                    r[f"{m}_{field}"] = float(vals.mean())
                    r[f"{m}_{field}_se"] = float(vals.std(ddof=1) / math.sqrt(len(vals)))
        r["passes"] = bool(
            float(r["c1_min_expected_t_hybrid"]) >= 4.0  # type: ignore[arg-type]
            and float(r["c3_max_acf_ratio"]) <= 1.0  # type: ignore[arg-type]
            and float(r.get(f"{VOLFRAG_SIGNAL}_high", math.inf)) <= -1.0  # type: ignore[arg-type]
            and float(r.get("s_real_high", -math.inf)) >= 0.3  # type: ignore[arg-type]
        )
    passing = sorted(
        (r for r in rows if r["passes"]),
        key=lambda r: (r["ic"], abs(float(r["ratio"])), -float(r["q"])),  # type: ignore[arg-type]
    )
    chosen = passing[0] if passing else None
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "root_seed": ROOT_SEED,
        "oracle_n_obs": ORACLE_N_OBS,
        "n_draws": N_DRAWS,
        "c": C,
        "vol_state": {"window": 21, "quantile": 0.70},
        "candidates": rows,
        "oracle_runs": oracle,
        "chosen": chosen,
    }
    (args.out / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        print(_line(r))
    print("chosen:", None if chosen is None else (chosen["ic"], chosen["ratio"], chosen["q"]))


if __name__ == "__main__":
    main()
