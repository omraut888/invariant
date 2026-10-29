"""Report (not gate) how well Stage 4's observable volatility state overlaps the benchmark's true
regime on the labeller dev seeds (ADR 0016). The state definition (q = .70) was fixed in 73367a7
before this ran.

Decision dates are Stage 3's: t = 1260 .. n_obs - 2. A P&L row is earned on t + 1 and is labelled
with the state of day t, so the truth it is compared with is the regime of day t + 1.

Usage: uv run python scripts/vol_state_dev_overlap.py [--out docs/results/0016-vol-state-dev]
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
from invariant.stages.adversarial.state import volatility_state

ROOT_SEED = 20261101
N_SEEDS = 20
WINDOW = 1260


def _mean_spell(x: np.ndarray) -> float:
    starts = np.flatnonzero(np.diff(np.concatenate([[0], x])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([x, [0]])) == -1)
    return float((ends - starts + 1).mean()) if len(starts) else 0.0


def _one(i: int) -> dict[str, float]:
    seed = int(RngFactory(ROOT_SEED).child(f"dev/{i}").generator("seed").integers(2**32))
    d = generate_market(load_benchmark_config("regime"), seed)
    ew = d.returns.mean(axis=1).to_numpy()
    dates = np.arange(WINDOW, len(ew) - 1, dtype=np.int64)
    state, _ = volatility_state(ew, dates)
    truth = d.latents[REGIME_NODE].to_numpy().astype(np.int64)[dates + 1]
    return {
        "i": i,
        "accuracy": float((state == truth).mean()),
        "tpr": float(state[truth == 1].mean()),
        "tnr": float(1.0 - state[truth == 0].mean()),
        "precision": float(truth[state == 1].mean()),
        "true_share": float(truth.mean()),
        "state_mean_spell": _mean_spell(state),
        "true_mean_spell": _mean_spell(truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/results/0016-vol-state-dev"))
    args = parser.parse_args()
    rows = Parallel(n_jobs=8)(delayed(_one)(i) for i in range(N_SEEDS))
    summary = {
        k: {
            "median": float(np.median([r[k] for r in rows])),
            "min": float(min(r[k] for r in rows)),
            "max": float(max(r[k] for r in rows)),
        }
        for k in rows[0]
        if k != "i"
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overlap.json").write_text(
        json.dumps({"root_seed": ROOT_SEED, "summary": summary, "rows": rows}, indent=2) + "\n"
    )
    for k, v in summary.items():
        print(f"{k:18s} median {v['median']:.3f}  [{v['min']:.3f}, {v['max']:.3f}]")


if __name__ == "__main__":
    main()
