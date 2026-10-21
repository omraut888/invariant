r"""Transfer entropy of a Stage 1 link, per window: Gaussian (closed form) and KSG.

Definitions
-----------
For the link ``X_{t-lag} -> R_t`` with Stage 1's conditioning set ``Z``, the transfer entropy
is the conditional mutual information

    TE = I(R_t ; X_{t-lag} | Z)   (nats).

The conditioning set holds R's own past and the other parents, so this is Schreiber's (2000) TE
with a conditioning set chosen by causal discovery, and not a fixed embedding.

- **Gaussian TE**: ``-1/2 log(1 - r^2)``, where r is the partial correlation of x and y given
  z. For jointly Gaussian variables this is the exact CMI, and it equals half the Granger
  causality log variance ratio (Barnett, Barrett & Seth 2009). It is a function of ``|r|``, so
  it carries the same information as the partial correlation, minus the sign.
- **KSG TE**: the Frenzel-Pompe (2007) k-nearest-neighbour CMI estimator (the conditional
  form of Kraskov, Stögbauer & Grassberger 2004). It uses tigramite's ``CMIknn`` with its
  defaults: ``knn = 0.2 T``, rank transform, 5 shuffle neighbours, 500 surrogates. Its
  significance comes from the local-permutation surrogate test of Runge (2018): Y is permuted
  only among samples whose Z values are near neighbours, so the surrogates keep the X-Z and
  Y-Z dependence and break only the X-Y link given Z. The surrogate distribution's mean
  estimates the estimator's finite-sample floor under conditional independence. Its standard
  deviation sets the window's noise scale in the half-life fit.

Assumptions and limitations
---------------------------
- Low-SNR link between TE and the coefficient: ``TE ≈ beta^2 Var(x|z) / (2 sigma^2)``, so
  TE halves twice as fast as the coefficient. At r0 = .25 the exact Gaussian TE half-life
  is 2.3% shorter than ``H/2`` (docs/adr/0011).
- KSG has a strong finite-sample bias at dimension ``2 + dim Z`` and window lengths of a few
  hundred rows. The surrogate mean measures that bias under the null. It is not removed under
  the alternative.
- The surrogate SD is the null SD. Under a strong link the estimator's SD is larger, and the
  study measures what this does to CI coverage.

References: Schreiber (2000), PRL 85(2); Kraskov, Stögbauer & Grassberger (2004), PRE 69;
Frenzel & Pompe (2007), PRL 99; Barnett, Barrett & Seth (2009), PRL 103; Runge (2018),
"Conditional independence testing based on a nearest-neighbor estimator of conditional mutual
information", AISTATS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tigramite.independence_tests.cmiknn import CMIknn

__all__ = ["KsgWindows", "gaussian_te", "ksg_windows"]

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]


def gaussian_te(r: _FloatArray) -> _FloatArray:
    """Gaussian TE ``-1/2 log(1 - r^2)`` in nats, elementwise."""
    r = np.asarray(r, dtype=np.float64)
    return np.asarray(-0.5 * np.log1p(-(r**2)), dtype=np.float64)


@dataclass(frozen=True)
class KsgWindows:
    """Per-window KSG CMI, with the mean and SD of its surrogate null and the surrogate p."""

    bounds: _IntArray
    value: _FloatArray
    null_mean: _FloatArray
    null_sd: _FloatArray
    p: _FloatArray


def ksg_windows(
    x: _FloatArray,
    y: _FloatArray,
    z: _FloatArray,
    bounds: _IntArray,
    seeds: Sequence[int],
    *,
    workers: int = 1,
) -> KsgWindows:
    """KSG CMI ``I(y; x | z)`` and its local-permutation surrogate test on every window.

    ``seeds[j]`` seeds window j's ``CMIknn`` (tie-breaking noise and permutations), so the
    result is deterministic. ``workers`` is passed to scipy's KD-tree. Use 1 inside a process
    pool.
    """
    if len(seeds) != len(bounds):
        raise ValueError("one seed per window is required")
    k = z.shape[1]
    xyz = np.array([0, 1, *([2] * k)], dtype=np.int64)
    out = np.empty((4, len(bounds)))
    for j, (a, b) in enumerate(bounds):
        array = np.vstack([x[a:b], y[a:b], z[a:b].T]).astype(np.float64)
        test = CMIknn(seed=seeds[j], workers=workers)
        value = float(test.get_dependence_measure(array.copy(), xyz))
        p, null = test.get_shuffle_significance(array.copy(), xyz, value, return_null_dist=True)
        out[:, j] = (value, float(np.mean(null)), float(np.std(null, ddof=1)), float(p))
    return KsgWindows(bounds=bounds, value=out[0], null_mean=out[1], null_sd=out[2], p=out[3])
