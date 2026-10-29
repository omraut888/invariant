"""Realism gate for Stage 4: an MMD two-sample test of generated against observed paths
(docs/adr/0016).

Method
------
**Features.** Non-overlapping 21-day windows of the asset returns. Per window w:
- log realised sd of the equal-weight return, at w, w-1 and w-2 (so state persistence shows);
- the mean pairwise correlation of the assets;
- the equal-weight window return.

These target exactly what the adversary moves: how often, and for how long, the market is in
the high-volatility state. Signals are left out because the adversary cannot change them. The
features are standardised by the observed windows' means and sds.

**Statistic.** Unbiased MMD^2 (Gretton, Borgwardt, Rasch, Schoelkopf & Smola 2012, "A kernel
two-sample test", JMLR 13, eq. 3), with a Gaussian kernel whose bandwidth is the median pairwise
distance of the observed features (the median heuristic, fixed per run).

**Null and threshold.**
- Each generated path is ``path_multiple`` times the observed length.
- The null distribution is MMD^2(generated at theta_hat, observed) over ``n_null``
  independent paths at the fitted chain theta_hat: observed against resampled observed, by the
  same generator.
- theta is plausible iff MMD^2(theta) does not exceed the null's empirical (1 - alpha) quantile
  (order statistic ceil((n_null + 1)(1 - alpha)), the 190th of 199 at alpha = .05).
- theta_hat is plausible by definition.

Assumptions and limitations:
- Windows are treated as exchangeable within a path. Adjacent windows are dependent through the
  lag features, so the test's calibration comes from the generator-based null, not from MMD
  asymptotics.
- The test's power against a given theta depends on the feature choice.
- "Plausible" means "not distinguishable from the observed path at this length". With long
  samples the set shrinks toward theta_hat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["RealismReference", "mmd2_unbiased", "null_threshold", "window_features"]

FloatArray = npt.NDArray[np.float64]

WINDOW = 21
N_FEATURES = 5


def window_features(returns: FloatArray, window: int = WINDOW) -> FloatArray:
    """Per-window features (module docstring), shape (n_windows - 2, 5). ``returns`` is (T, N)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("returns must be (T, N) with N >= 2")
    n_win = r.shape[0] // window
    if n_win < 3:
        raise ValueError("need at least 3 full windows")
    blocks = r[: n_win * window].reshape(n_win, window, r.shape[1])
    ew = blocks.mean(axis=2)
    log_sd = np.log(ew.std(axis=1, ddof=1))
    centred = blocks - blocks.mean(axis=1, keepdims=True)
    cov = np.einsum("wti,wtj->wij", centred, centred)
    sd = np.sqrt(np.einsum("wii->wi", cov))
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    n = r.shape[1]
    mean_corr = (corr.sum(axis=(1, 2)) - n) / (n * (n - 1))
    ew_sum = ew.sum(axis=1)
    return np.column_stack(
        [log_sd[2:], log_sd[1:-1], log_sd[:-2], mean_corr[2:], ew_sum[2:]],
    )


def _sq_dists(a: FloatArray, b: FloatArray) -> FloatArray:
    d = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * a @ b.T
    return np.asarray(np.maximum(d, 0.0), dtype=np.float64)


def mmd2_unbiased(x: FloatArray, y: FloatArray, bandwidth: float) -> float:
    """Unbiased MMD^2 between samples ``x`` (m, d) and ``y`` (n, d), Gaussian kernel."""
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        raise ValueError("each sample needs at least 2 points")
    g = -1.0 / (2.0 * bandwidth * bandwidth)
    kxx = np.exp(g * _sq_dists(x, x))
    kyy = np.exp(g * _sq_dists(y, y))
    kxy = np.exp(g * _sq_dists(x, y))
    sxx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(sxx + syy - 2.0 * kxy.mean())


@dataclass(frozen=True)
class RealismReference:
    """The observed features, standardised, with the fixed bandwidth."""

    features: FloatArray
    mean: FloatArray
    sd: FloatArray
    bandwidth: float

    @classmethod
    def from_returns(cls, returns: FloatArray) -> RealismReference:
        f = window_features(returns)
        mean, sd = f.mean(axis=0), f.std(axis=0, ddof=1)
        if not np.all(sd > 0):
            raise ValueError("a realism feature is constant on the observed path")
        z = (f - mean) / sd
        d = np.sqrt(_sq_dists(z, z))
        bandwidth = float(np.median(d[np.triu_indices(len(z), k=1)]))
        return cls(features=z, mean=mean, sd=sd, bandwidth=bandwidth)

    def statistic(self, generated_returns: FloatArray) -> float:
        """MMD^2 of a generated path's standardised features against the observed ones."""
        g = (window_features(generated_returns) - self.mean) / self.sd
        return mmd2_unbiased(g, self.features, self.bandwidth)


def null_threshold(null: FloatArray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th order statistic of the null sample (1-based)."""
    x = np.sort(np.asarray(null, dtype=np.float64))
    rank = math.ceil((len(x) + 1) * (1.0 - alpha))
    if not 1 <= rank <= len(x):
        raise ValueError(f"{len(x)} null replicates cannot give a level-{alpha} threshold")
    return float(x[rank - 1])
