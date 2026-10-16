"""E6: a Stambaugh-style null for the Stage 1 decision-test study (test-only DGP, ADR 0009).

    x_t = rho x_{t-1} + u_t,       R_t = mu + e_t,      (u_t, e_t) ~ N(0, [[s_u^2, c], [c, s_e^2]])

with ``rho = 0.99`` and ``corr(u_t, e_t) = -0.8``. x has no effect on returns, so the null of no
link x -> R holds at every lag. But x's innovations are contemporaneously correlated with
return innovations, as for valuation ratios, where a price shock moves both. Then the OLS
slope of ``R_t`` on ``x_{t-1}`` inherits the downward small-sample bias of the estimated rho,
and its t-test is oversized (Stambaugh 1999).

Innovations are Gaussian and homoskedastic, so E6 isolates the Stambaugh effect from the
GARCH effect that E4 covers. ``x_0`` is drawn from the stationary distribution, so no
burn-in is needed. ``x_t`` is a function of innovations up to t, so it is known at the close
of t.

Reference: Stambaugh (1999), "Predictive regressions", Journal of Financial Economics 54(3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

from invariant.stages.causal.calibration import TARGET, FamilyRole, LinkFamily, Scenario

FloatArray = npt.NDArray[np.float64]
SIGNAL = "s_stam"


@dataclass(frozen=True)
class StambaughData:
    rho: float = 0.99
    corr: float = -0.8
    sigma_u: float = 0.05
    sigma_e: float = 0.01
    mu: float = 0.0003

    def __call__(self, seed: int, n_obs: int) -> tuple[FloatArray, tuple[str, ...]]:
        gen = np.random.Generator(np.random.PCG64(seed))
        z = gen.standard_normal((n_obs, 2))
        u = self.sigma_u * z[:, 0]
        e = self.sigma_e * (self.corr * z[:, 0] + math.sqrt(1.0 - self.corr**2) * z[:, 1])
        x0 = gen.standard_normal() * self.sigma_u / math.sqrt(1.0 - self.rho**2)
        x_in = u.copy()
        x_in[0] = x0  # x_0 drawn from the stationary law (u_0 unused, like the benchmark)
        x = np.asarray(sp_signal.lfilter([1.0], [1.0, -self.rho], x_in), dtype=np.float64)
        return np.column_stack([x, self.mu + e]), (SIGNAL, TARGET)


def stambaugh_scenario(
    tau_max: int, *, n_obs: int | None = None, gating: bool = True, name: str = "e6_stambaugh"
) -> Scenario:
    """E6. ``n_obs`` overrides the study's T; a non-gating short-T run is a diagnostic."""
    lags = tuple(range(1, tau_max + 1))
    role = FamilyRole.GATING_NULL if gating else FamilyRole.NULL
    return Scenario(
        name,
        "Null; persistent signal (rho .99) with innovations correlated -0.8 with returns"
        + (f"; T = {n_obs}." if n_obs else "."),
        StambaughData(),
        (
            LinkFamily(name=f"{SIGNAL}->R", source=SIGNAL, target=TARGET, lags=lags, role=role),
            LinkFamily(
                name=f"{SIGNAL}->R@1",
                source=SIGNAL,
                target=TARGET,
                lags=(1,),
                role=FamilyRole.NULL,
            ),
        ),
        n_obs,
    )
