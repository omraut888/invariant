"""E4: a vol-correlated null for the Stage 1 CI-test size study (test-only DGP, ADR 0009).

Returns are the benchmark's global-null GARCH(1,1)-t returns, ``R_t = mu + a_t``, which have no
mean effect from anything. The signal is a volatility forecast: ``S_t = log h_{t+1}``, where
``h_{t+1} = omega + alpha a_t^2 + beta h_t`` is the conditional variance of ``a_{t+1}``. It is
a function of returns up to t only, so it is known at the close of t
(tests/leakage/test_vol_null_timing.py).

Under this DGP ``Cov(S_{t-1}, R_t) = 0``, so the null of no link S -> R holds at every lag.
But ``S_{t-1}^2`` is correlated with ``Var(R_t | past) = h_t``, which is the case where the OLS
t-test's homoskedastic standard error is wrong (White 1980). Real volatility-timing signals
(VIX, realized vol, momentum in crashes) look like this. The planted-signal benchmark does not,
because its signals are independent of volatility by construction.

No standardization is applied: partial correlation is scale-invariant, and a full-sample
z-score would be a (harmless, but avoidable) look-ahead in the scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.stages.causal.calibration import (
    TARGET,
    FamilyRole,
    LinkFamily,
    Scenario,
    null_config,
)

FloatArray = npt.NDArray[np.float64]
SIGNAL = "s_vol"


@dataclass(frozen=True)
class VolNullData:
    config: SyntheticMarketConfig

    def __call__(self, seed: int, n_obs: int) -> tuple[FloatArray, tuple[str, ...]]:
        # One extra simulated day: S_t needs h_{t+1}.
        ds = generate_market(self.config.model_copy(update={"n_obs": n_obs + 1}), seed)
        h = ds.latents[f"h_{TARGET}"].to_numpy(dtype=np.float64)
        r = ds.returns[TARGET].to_numpy(dtype=np.float64)
        data = np.column_stack([np.log(h[1:]), r[:-1]])
        return data, (SIGNAL, TARGET)


def vol_null_config() -> SyntheticMarketConfig:
    return null_config(load_benchmark_config(None), garch=True)


def vol_null_scenario(tau_max: int) -> Scenario:
    lags = tuple(range(1, tau_max + 1))
    return Scenario(
        "e4_vol_null",
        "Global null; signal = log conditional variance forecast (vol-correlated null).",
        VolNullData(vol_null_config()),
        (
            LinkFamily(
                name=f"{SIGNAL}->R",
                source=SIGNAL,
                target=TARGET,
                lags=lags,
                role=FamilyRole.GATING_NULL,
            ),
            LinkFamily(
                name=f"{SIGNAL}->R@1", source=SIGNAL, target=TARGET, lags=(1,), role=FamilyRole.NULL
            ),
            LinkFamily(name="R->R", source=TARGET, target=TARGET, lags=lags, role=FamilyRole.NULL),
            LinkFamily(
                name="R->R@1", source=TARGET, target=TARGET, lags=(1,), role=FamilyRole.NULL
            ),
        ),
    )
