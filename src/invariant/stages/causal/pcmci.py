"""Typed wrapper around tigramite's PCMCI, so its untyped arrays stop at this module.

Method
------
PCMCI (Runge et al. 2019): (1) PC1 condition selection finds a superset of each variable's
lagged parents at significance ``pc_alpha``. (2) The momentary conditional independence (MCI)
test of ``X_{t-tau} -> Y_t`` conditions on the selected parents of both ``Y_t`` and
``X_{t-tau}``. Only lagged links are tested (``tau_min = 1``): every planted mean effect has
lag >= 1, and a lag-0 link has no direction under PCMCI.

Assumptions (Runge et al. 2019, section "Assumptions"): causal sufficiency (no unobserved
common cause of two observed variables), the causal Markov condition, faithfulness, no
contemporaneous effects, and stationarity of the causal structure. The CI test adds its own
assumptions (``ci_tests``). Where the benchmark violates them is documented in
docs/adr/0009-stage1-ci-test-and-multiple-testing.md.

``p[i, j, tau]`` and ``val[i, j, tau]`` are the MCI p-value and test statistic (the partial
correlation for ParCorr-type tests) of ``var_names[i]`` at lag ``tau`` -> ``var_names[j]``.
Index ``tau = 0`` is unused (NaN).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tigramite import data_processing as pp
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.pcmci import PCMCI

__all__ = ["PcmciOutput", "run_pcmci"]

_FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PcmciOutput:
    var_names: tuple[str, ...]
    tau_max: int
    n_obs: int
    p: _FloatArray  # (N, N, tau_max + 1); [:, :, 0] is NaN
    val: _FloatArray  # same shape

    def index(self, name: str) -> int:
        return self.var_names.index(name)

    def link_p(self, source: str, target: str, lags: Sequence[int]) -> _FloatArray:
        """MCI p-values of ``source_{t-lag} -> target_t`` for each lag in ``lags``."""
        return self._link(self.p, source, target, lags)

    def link_val(self, source: str, target: str, lags: Sequence[int]) -> _FloatArray:
        return self._link(self.val, source, target, lags)

    def _link(self, arr: _FloatArray, source: str, target: str, lags: Sequence[int]) -> _FloatArray:
        bad = [lag for lag in lags if not 1 <= lag <= self.tau_max]
        if bad:
            raise ValueError(f"lags must be in [1, {self.tau_max}], got {bad}")
        return np.asarray(arr[self.index(source), self.index(target), list(lags)], dtype=float)


def run_pcmci(
    data: _FloatArray,
    var_names: Sequence[str],
    ci_test: CondIndTest,
    *,
    tau_max: int,
    pc_alpha: float,
) -> PcmciOutput:
    """Run PCMCI on ``data`` (``T x N``, time in rows, no NaN). Pure given ``ci_test``'s seed."""
    if data.ndim != 2 or data.shape[1] != len(var_names):
        raise ValueError(f"data shape {data.shape} does not match {len(var_names)} var_names")
    if len(set(var_names)) != len(var_names):
        raise ValueError(f"var_names must be unique: {list(var_names)}")
    if not np.isfinite(data).all():
        raise ValueError("data must be finite; drop or mask missing rows before PCMCI")
    if tau_max < 1:
        raise ValueError(f"tau_max must be >= 1, got {tau_max}")
    if not 0.0 < pc_alpha < 1.0:
        raise ValueError(f"pc_alpha must be in (0, 1), got {pc_alpha}")
    frame = pp.DataFrame(np.ascontiguousarray(data, dtype=float), var_names=list(var_names))
    pcmci = PCMCI(dataframe=frame, cond_ind_test=ci_test, verbosity=0)
    result = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    p = np.array(result["p_matrix"], dtype=float)
    val = np.array(result["val_matrix"], dtype=float)
    p[:, :, 0] = np.nan
    val[:, :, 0] = np.nan
    return PcmciOutput(
        var_names=tuple(var_names), tau_max=tau_max, n_obs=data.shape[0], p=p, val=val
    )
