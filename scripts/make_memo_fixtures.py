"""Build the run-directory fixtures the memo unit tests read (tests/fixtures/memo_runs/).

Each case runs Stages 1-3 on one planted-signal dataset through the production path
(``stages.memo.benchmark``) and copies the run's manifest, ledger and stage results. Seeds are
searched in order until each case's outcome appears; these are test fixtures, not evidence (the
memo's validation, docs/adr/0017, uses its own pre-registered root seed).

Usage: uv run python scripts/make_memo_fixtures.py [--max-seed N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from joblib import Parallel, delayed

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.config.settings import find_project_root
from invariant.core.logging import LogMode, configure_logging
from invariant.core.run_hash import read_git_state
from invariant.stages.dro.study import dro_scenarios
from invariant.stages.memo.benchmark import run_benchmark_pipeline

FILES = (
    "manifest.json",
    "kill_ledger.jsonl",
    "kill_ledger.head.json",
    "causal_result.json",
    "decay_result.json",
    "portfolio_result.json",
)
OUT = find_project_root() / "tests" / "fixtures" / "memo_runs"


def _decay_general_kill(run: Path) -> bool:
    lines = (run / "kill_ledger.jsonl").read_text().splitlines()
    return any(
        json.loads(x)["stage"] == "decay" and json.loads(x)["reason"] == "general.insufficient_data"
        for x in lines
    )


def _members(run: Path) -> tuple[list[str], str | None]:
    p = run / "portfolio_result.json"
    if not p.is_file():
        return [], None
    doc = json.loads(p.read_text())
    m = doc["members"]
    return [x["signal_id"] for x in m], (m[0]["verdict"] if m else None)


def _case(preset: str, n_obs: int | None, seed: int, tmp: str) -> tuple[str, int, str]:
    if preset == "d1_base":
        cfg = next(s for s in dro_scenarios() if s.name == "d1_base").config(n_obs)
    else:
        cfg = load_benchmark_config(None if preset == "base" else preset)
        if n_obs is not None:
            cfg = cfg.model_copy(update={"n_obs": n_obs})
    git, patch = read_git_state()
    run = run_benchmark_pipeline(
        generate_market(cfg, seed),
        seed,
        Path(tmp) / f"{preset}-{n_obs}-{seed}",
        git_state=git,
        patch=patch,
    )
    return preset, seed, str(run)


Want = Callable[[list[str], str | None, Path], bool]
CASES: dict[str, tuple[str, int | None, Want]] = {
    "pass_single": ("d1_base", None, lambda m, v, r: len(m) == 1 and v == "pass"),
    "pass_multi": ("d1_base", None, lambda m, v, r: len(m) >= 2 and v == "pass"),
    "portfolio_kill": ("d1_base", None, lambda m, v, r: v == "kill"),
    "realistic": ("realistic", None, lambda m, v, r: True),
    "hidden_confounder": ("hidden_confounder", None, lambda m, v, r: True),
    "volfrag": ("volfrag", None, lambda m, v, r: v is not None),
    "not_decided": ("base", 450, lambda m, v, r: True),
    "stage2_short": ("base", 4000, lambda m, v, r: _decay_general_kill(r)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seed", type=int, default=24)
    parser.add_argument("--only", nargs="*", default=None, help="rebuild only these cases")
    args = parser.parse_args()
    cases = {k: v for k, v in CASES.items() if args.only is None or k in args.only}
    configure_logging(LogMode.DEV, level="ERROR")
    found: dict[str, Path] = {}
    with tempfile.TemporaryDirectory() as tmp:
        jobs = sorted({(p, n) for p, n, _ in cases.values()})
        for seed in range(args.max_seed):
            todo = [
                (p, n)
                for p, n in jobs
                if any(c[0] == p and c[1] == n and k not in found for k, c in cases.items())
            ]
            if not todo:
                break
            runs = Parallel(n_jobs=-1)(delayed(_case)(p, n, seed, tmp) for p, n in todo)
            for (p, n), (_, _, run) in zip(todo, runs, strict=True):
                members, verdict = _members(Path(run))
                for name, (cp, cn, want) in cases.items():
                    if (
                        name not in found
                        and cp == p
                        and cn == n
                        and want(members, verdict, Path(run))
                    ):
                        found[name] = Path(run)
                        print(f"{name}: {p} n_obs={n} seed {seed} members={members} {verdict}")
        missing = set(cases) - set(found)
        if missing:
            raise SystemExit(f"no seed below {args.max_seed} produced {sorted(missing)}")
        for name, run in found.items():
            dest = OUT / name
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            for f in FILES:
                if (run / f).is_file():
                    shutil.copy2(run / f, dest / f)


if __name__ == "__main__":
    main()
