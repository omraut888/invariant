r"""Multiple-testing corrections for Stage 1: Bonferroni within a signal, FDR across signals.

Method
------
Within a signal, every ``(asset, lag)`` test of ``X_{t-lag} -> R^a_t`` is one hypothesis of the
signal's family, and the family is decided by Bonferroni: ``family p = min(1, K min_k p_k)``
over its ``K`` tests. Bonferroni needs no assumption about the dependence between the tests,
which is strong here (adjacent lags of a persistent signal, correlated assets).

Across signals, the family p-values are adjusted for the false discovery rate:

- ``bh``: Benjamini & Hochberg (1995). It controls the FDR at level ``q`` when the p-values are
  independent or positively regression dependent on a subset (PRDS, Benjamini & Yekutieli 2001).
- ``by``: Benjamini & Yekutieli (2001), BH with ``q / sum_{i<=m} 1/i``. It controls the FDR under
  any dependence.

Assumption violated by Stage 1, stated rather than assumed away: every signal is tested against
the same returns, so the family p-values are dependent, and two-sided tests of correlated
statistics are not guaranteed PRDS. BH is therefore not guaranteed to control the FDR. BY is,
at a cost in power. docs/adr/0010 records which one runs and why.

The adjusted values are the step-up "q-values": ``q_(i) = min_{j >= i} min(1, m c p_(j) / j)``,
with ``c = 1`` (BH) or ``c = sum_{k<=m} 1/k`` (BY). A hypothesis is rejected at level alpha iff
its q-value is ``<= alpha``. These equal statsmodels' ``multipletests(method="fdr_bh" /
"fdr_by")`` adjusted p-values (tests/unit/test_multiple_testing.py).

References: Benjamini & Hochberg (1995), JRSS B 57(1); Benjamini & Yekutieli (2001), Annals of
Statistics 29(4).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["FdrMethod", "bonferroni_family_p", "fdr_qvalues"]

FdrMethod = Literal["bh", "by"]
_FloatArray = npt.NDArray[np.float64]


def _check_p(p: _FloatArray) -> None:
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p-values must be a non-empty 1-d array")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be finite and in [0, 1]")


def bonferroni_family_p(p_values: Sequence[float] | _FloatArray) -> float:
    """Bonferroni p-value of a family: ``min(1, K * min p)`` over its ``K`` tests."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    _check_p(p)
    return float(min(1.0, p.size * p.min()))


def fdr_qvalues(p_values: Sequence[float] | _FloatArray, method: FdrMethod) -> _FloatArray:
    """BH or BY step-up adjusted p-values, in the input order (see module docstring)."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    _check_p(p)
    m = p.size
    c = 1.0 if method == "bh" else float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p, kind="stable")
    ranked = p[order] * (m * c) / np.arange(1, m + 1)
    stepped = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(stepped, 1.0)
    return q
