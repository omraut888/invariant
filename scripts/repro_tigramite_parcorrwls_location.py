"""Minimal reproduction: tigramite's ParCorrWLS p-values depend on the location of X.

Uses only numpy and tigramite's public API (no INVARIANT code). X and Y are i.i.d. N(0, 1) and
independent, except in the power check, where Y_t = 0.15 X_{t-1} + noise. Adding a constant to X
must not change a partial-correlation test, and it does not change ParCorr. It changes
ParCorrWLS (tigramite 5.2.10.1, expert_knowledge="time-dependent heteroskedasticity"):
the size collapses toward 0 and so does the power.

Cause (docs/adr/0009): ParCorrWLS._get_single_residuals defaults to standardize=False, while its
own docstring says "default: True ... Must be used for partial correlation". The regressions
have no intercept. With an empty Z the time-dependent std is a window mean of |level|.

    uv run python scripts/repro_tigramite_parcorrwls_location.py
"""

from __future__ import annotations

import importlib.metadata

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS

T, REPS, ALPHA, SHIFT = 1000, 400, 0.05, 10.0


def p_value(test_cls: type, data: np.ndarray, z: list[tuple[int, int]]) -> float:
    test = test_cls(significance="analytic")
    test.set_dataframe(pp.DataFrame(data))
    _, p = test.run_test(X=[(0, -1)], Y=[(1, 0)], Z=z, tau_max=2)
    return float(p)


def main() -> None:
    print(f"tigramite {importlib.metadata.version('tigramite')}, T={T}, reps={REPS}")
    rng = np.random.Generator(np.random.PCG64(20261016))
    for effect in (0.0, 0.15):
        for z_label, z in (("Z = {}", []), ("Z = {X_{t-2}}", [(0, -2)])):
            rej: dict[tuple[str, float], int] = {}
            for _ in range(REPS):
                x = rng.standard_normal(T)
                y = rng.standard_normal(T)
                y[1:] += effect * x[:-1]
                for shift in (0.0, SHIFT):
                    data = np.column_stack([x + shift, y])
                    for name, cls in (("ParCorr", ParCorr), ("ParCorrWLS", ParCorrWLS)):
                        key = (name, shift)
                        rej[key] = rej.get(key, 0) + (p_value(cls, data, z) <= ALPHA)
            what = "size" if effect == 0 else f"power (effect {effect})"
            for name in ("ParCorr", "ParCorrWLS"):
                print(
                    f"{what:20s} {z_label:15s} {name:10s} "
                    f"mean(X)=0: {rej[(name, 0.0)] / REPS:.3f}   "
                    f"mean(X)={SHIFT:g}: {rej[(name, SHIFT)] / REPS:.3f}"
                )


if __name__ == "__main__":
    main()
