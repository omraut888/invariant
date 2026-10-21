"""Pre-study numbers in docs/adr/0011 (findings 1-4), reproducible from fixed seeds.

1. Cramer-Rao bound of the decay rate in beta(t) = beta0 * 2^{-(t - tau)/H} (oracle
   conditioning, unit-variance x and noise, r0 = partial correlation before the onset).
2. Noise scale of short windows against the planted TE.
3. Power of the "recent window insignificant" rule's false-decay on s_real.
4. Simulated power of the one-sided sup-split and linear-trend decline tests (Gaussian, i.i.d.
   x, the critical value of sup-split simulated under a constant coefficient).

    uv run python -m scripts.decay_prestudy_analysis
"""

from __future__ import annotations

import numpy as np
from scipy import stats

T = 6000


def path(r0: float, onset: float, hl: float) -> np.ndarray:
    t = np.arange(T)
    t0 = int(onset * T)
    return r0 * np.where(t >= t0, np.exp2(-np.maximum(t - t0, 0) / hl), 1.0)


def crb_rel_se(r0: float, onset: float, hl: float, *, tau_known: bool) -> float:
    t = np.arange(T)
    t0 = int(onset * T)
    lam = np.log(2) / hl
    u = np.maximum(t - t0, 0)
    post = t >= t0
    b = path(r0, onset, hl)
    cols = [b / r0, -u * b] + ([] if tau_known else [np.where(post, lam * b, 0.0)])
    g = np.column_stack(cols)
    return float(np.sqrt(np.linalg.inv(g.T @ g)[1, 1]) / lam)


def power_grid(reps: int = 1500, seed: int = 2) -> None:
    rng = np.random.default_rng(seed)
    ks = np.arange(int(0.15 * T), int(0.85 * T) + 1, 25)
    tau = (np.arange(T) - T / 2) / T

    def draw(b: np.ndarray) -> tuple[float, float]:
        x = rng.standard_normal(T)
        y = b * x + rng.standard_normal(T)
        sxy, sxx = np.cumsum(x * y), np.cumsum(x * x)
        a, c = sxx[ks - 1], sxx[-1] - sxx[ks - 1]
        z = (sxy[ks - 1] / a - (sxy[-1] - sxy[ks - 1]) / c) / np.sqrt(1 / a + 1 / c)
        d = np.column_stack([x, x * tau])
        coef = np.linalg.solve(d.T @ d, d.T @ y)
        return float(z.max()), float(-coef[1] / np.sqrt(np.linalg.inv(d.T @ d)[1, 1]))

    null = np.array([draw(np.full(T, 0.06)) for _ in range(reps)])
    crit = float(np.quantile(null[:, 0], 0.95))
    print(f"sup-split 5% critical value {crit:.2f}; trend size {(null[:, 1] > 1.645).mean():.3f}")
    for onset, hl in (
        (0.3, 250),
        (0.6, 125),
        (0.6, 250),
        (0.6, 500),
        (0.8, 125),
        (0.8, 250),
        (0.8, 500),
    ):
        alt = np.array([draw(path(0.08, onset, hl)) for _ in range(reps)])
        print(
            f"onset {onset} H {hl}: sup-split {(alt[:, 0] > crit).mean():.2f}, "
            f"trend {(alt[:, 1] > 1.645).mean():.2f}"
        )


def main() -> None:
    for known in (True, False):
        rel = crb_rel_se(0.08, 0.6, 250.0, tau_known=known)
        lam_te = 2 * np.log(2) / 250.0
        se_hl = 0.5 * np.log(2) / (np.log(2) / 250.0) ** 2 * rel * np.log(2) / 250.0
        print(
            f"CRB r0 .08 H 250, tau known={known}: relSE(lambda) {rel:.2f}, "
            f"SE(TE half-life) ~ {se_hl:.0f} days (lambda_TE {lam_te:.5f})"
        )
    for r0, hl in ((0.25, 500.0), (0.25, 250.0)):
        print(f"CRB r0 {r0} H {hl:.0f}: relSE {crb_rel_se(r0, 0.6, hl, tau_known=False):.2f}")
        mh = np.sqrt((1 - np.sqrt(1 - r0**2)) / r0**2)
        print(f"  exact TE half-life {-np.log2(mh) * hl:.1f} vs H/2 {hl / 2:.0f}")
    for w in (60, 120, 252, 500):
        print(
            f"window {w}: SE(r) {1 / np.sqrt(w):.3f}, Gaussian TE floor {1 / (2 * w):.4f} "
            f"vs TE0 {0.08**2 / 2:.4f}"
        )
    for w in (250, 500, 1000):
        zt = 0.06 * np.sqrt(w)
        p = stats.norm.cdf(1.96 - zt) - stats.norm.cdf(-1.96 - zt)
        print(f"s_real recent window {w}: P(not significant) {p:.2f}")
    power_grid()


if __name__ == "__main__":
    main()
