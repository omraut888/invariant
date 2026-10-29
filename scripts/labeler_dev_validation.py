"""One-shot evaluation of the Stage 4 MS-GARCH labeller against ADR 0016's targets L1-L3.

The targets were fixed in commit 41b71c0, before this labeller existed. This script is run once.
If a target is missed, the work stops for review; there is no iteration on these dev seeds.

Dev seeds: RngFactory(20261101).child(f"dev/{i}"), i = 0..19, presets `regime` and
`multi_asset`. The series labelled is the equal-weight daily return.

Usage: uv run python scripts/labeler_dev_validation.py [--out docs/results/0016-labeler-dev]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import REGIME_NODE, generate_market
from invariant.core.run_hash import RngFactory
from invariant.stages.adversarial.regimes import label_regimes, select_regimes

ROOT_SEED = 20261101
N_SEEDS = 20


def _spells(x: np.ndarray) -> list[int]:
    out: list[int] = []
    run = 0
    for v in x.tolist():
        if v == 1:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def _one(preset: str, i: int) -> dict[str, object]:
    seed = int(RngFactory(ROOT_SEED).child(f"dev/{i}").generator("seed").integers(2**32))
    d = generate_market(load_benchmark_config(preset), seed)
    r = d.returns.mean(axis=1).to_numpy()
    sel = select_regimes(r)
    fit = sel.chosen
    lab = label_regimes(r, fit)
    row: dict[str, object] = {
        "preset": preset,
        "i": i,
        "seed": seed,
        "n_regimes": fit.n_regimes,
        "bic_one": sel.one.bic,
        "bic_two": sel.two.bic,
        "converged_one": sel.one.converged,
        "converged_two": sel.two.converged,
        "two_stress_share": sel.two.stress_share,
        "two_stress_duration": sel.two.stress_duration,
        "label_share": float(lab.mean()),
    }
    if REGIME_NODE in d.latents:
        xi = d.latents[REGIME_NODE].to_numpy().astype(np.int64)
        true_spell = float(np.mean(_spells(xi)))
        row |= {
            "accuracy": float((lab == xi).mean()),
            "tpr": float(lab[xi == 1].mean()),
            "tnr": float(1.0 - lab[xi == 0].mean()),
            "true_share": float(xi.mean()),
            "true_mean_spell": true_spell,
            "spell_ratio": (fit.stress_duration / true_spell) if fit.n_regimes == 2 else 0.0,
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/results/0016-labeler-dev"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(_one)(p, i) for p in ("regime", "multi_asset") for i in range(N_SEEDS)
    )
    reg = [r for r in rows if r["preset"] == "regime"]
    ma = [r for r in rows if r["preset"] == "multi_asset"]
    l1 = float(np.median([float(r["accuracy"]) for r in reg]))
    l2 = float(np.median([float(r["spell_ratio"]) for r in reg]))
    l3 = sum(int(r["n_regimes"] == 1) for r in ma)
    targets = {
        "L1_median_accuracy": {"value": l1, "target": ">= 0.90", "met": l1 >= 0.90},
        "L2_median_spell_ratio": {"value": l2, "target": "in [0.7, 1.3]", "met": 0.7 <= l2 <= 1.3},
        "L3_multi_asset_one_regime": {"value": l3, "target": "20 of 20", "met": l3 == N_SEEDS},
        "reported_regime_two_regimes": sum(int(r["n_regimes"] == 2) for r in reg),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labeler_dev.json").write_text(
        json.dumps({"root_seed": ROOT_SEED, "targets": targets, "rows": rows}, indent=2) + "\n"
    )
    print(json.dumps(targets, indent=2))


if __name__ == "__main__":
    main()
