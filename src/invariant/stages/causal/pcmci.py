r"""Typed wrapper around tigramite's PCMCI, so its untyped arrays stop at this module.

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

Decision tests
--------------
For the ``decision_links`` requested, ``run_pcmci`` also tests ``X_{t-tau} -> Y_t`` for every
lag in [1, tau_max] against three conditioning sets. All three come from the same PC1 parents
P(.) and are tested with the same CI test:

- ``mci``: ``P(Y) \ {X_{t-tau}}`` plus ``P(X)`` shifted by tau (tigramite's MCI set, built
  in tigramite's order; it reproduces ``p`` exactly, tests/unit/test_pcmci.py).
- ``parents_r``: ``P(Y) \ {X_{t-tau}}`` only. This is a predictive regression on the selected
  parents. It has full power for a persistent X, but it is exposed to confounding through a
  parent of X that PC1 did not select for Y, and to Stambaugh (1999) bias.
- ``hybrid``: ``parents_r`` plus the shifted parents of X *other than X's own lags*. It
  conditions on X's other causes (e.g. a confounder), but not on X's own past, which is what
  costs MCI its power on persistent signals (ADR 0009, finding 4).

Reference: Stambaugh (1999), "Predictive regressions", Journal of Financial Economics 54(3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from tigramite import data_processing as pp
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.pcmci import PCMCI

__all__ = ["DecisionTest", "PcmciOutput", "conditioning_set", "run_pcmci"]

_FloatArray = npt.NDArray[np.float64]
_Node = tuple[int, int]  # tigramite form: (variable index, -lag)


class DecisionTest(StrEnum):
    MCI = "mci"
    PARENTS_R = "parents_r"
    HYBRID = "hybrid"


def conditioning_set(
    decision: DecisionTest,
    parents: dict[int, list[_Node]],
    source: int,
    target: int,
    lag: int,
) -> list[_Node]:
    """Conditions for ``source_{t-lag} -> target_t`` (see module docstring)."""
    x = (source, -lag)
    z = [node for node in parents[target] if node != x]
    if decision is DecisionTest.PARENTS_R:
        return z
    shifted = [(k, -lag + k_tau) for k, k_tau in parents[source]]
    if decision is DecisionTest.HYBRID:
        shifted = [node for node in shifted if node[0] != source]
    z += [node for node in shifted if node not in z]
    return z


@dataclass(frozen=True)
class PcmciOutput:
    var_names: tuple[str, ...]
    tau_max: int
    n_obs: int
    p: _FloatArray  # (N, N, tau_max + 1); [:, :, 0] is NaN
    val: _FloatArray  # same shape
    parents: dict[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)
    """PC1 parents per variable as ``(name, lag)``, lag >= 1, in tigramite's order."""
    decision_p: dict[tuple[DecisionTest, str, str], _FloatArray] = field(default_factory=dict)
    """``(decision, source, target)`` -> p-values at lags 1..tau_max."""
    decision_val: dict[tuple[DecisionTest, str, str], _FloatArray] = field(default_factory=dict)

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
    decision_links: Sequence[tuple[str, str]] = (),
    decisions: Sequence[DecisionTest] = (),
) -> PcmciOutput:
    """Run PCMCI on ``data`` (``T x N``, time in rows, no NaN). Pure given ``ci_test``'s seed.

    For every ``(source, target)`` in ``decision_links`` and every test in ``decisions``, the
    decision-test p-values at lags 1..tau_max are added (see module docstring).
    """
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
    names = tuple(var_names)
    raw_parents: dict[int, list[_Node]] = {
        int(j): [(int(i), int(t)) for i, t in nodes] for j, nodes in pcmci.all_parents.items()
    }
    decision_p: dict[tuple[DecisionTest, str, str], _FloatArray] = {}
    decision_val: dict[tuple[DecisionTest, str, str], _FloatArray] = {}
    test = pcmci.cond_ind_test  # PCMCI's copy, bound to the dataframe
    for source, target in decision_links:
        i, j = names.index(source), names.index(target)
        for decision in decisions:
            ps, vs = np.empty(tau_max), np.empty(tau_max)
            for lag in range(1, tau_max + 1):
                z = conditioning_set(decision, raw_parents, i, j, lag)
                # Without alpha_or_thres, run_test returns (val, pval); alpha never alters pval.
                v, pv = test.run_test([(i, -lag)], [(j, 0)], Z=z, tau_max=tau_max)
                ps[lag - 1], vs[lag - 1] = float(pv), float(v)
            decision_p[(decision, source, target)] = ps
            decision_val[(decision, source, target)] = vs
    return PcmciOutput(
        var_names=names,
        tau_max=tau_max,
        n_obs=data.shape[0],
        p=p,
        val=val,
        parents={names[j]: tuple((names[i], -t) for i, t in raw_parents[j]) for j in raw_parents},
        decision_p=decision_p,
        decision_val=decision_val,
    )
