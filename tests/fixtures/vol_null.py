"""E4: vol-correlated nulls for the Stage 1 size studies (test-only DGPs, ADR 0009).

Returns are the benchmark's global-null GARCH(1,1)-t returns, ``R_t = mu + a_t``. Nothing has
a mean effect on them. The signal is a volatility measure known at the close of t:

- ``log_h`` (E4, and the persistence variants): ``S_t = log h_{t+1}``, where ``h_{t+1} = omega
  + alpha a_t^2 + beta h_t`` is the conditional variance of ``a_{t+1}``. It is a function of
  returns up to t (tests/leakage/test_vol_null_timing.py).
- ``rv22`` (E4-rv): ``S_t = log(mean(R_{t-21}^2, ..., R_t^2))``, a 22-day trailing realized
  variance. This is what a practitioner would actually compute.

Under these DGPs ``Cov(S_{t-1}, R_t) = 0``, so the null of no link S -> R holds at every lag.
But ``S_{t-1}^2`` is correlated with ``Var(R_t | past)``, which is the case where the OLS
t-test's homoskedastic standard error is wrong (White 1980). The planted-signal benchmark's
signals are independent of volatility by construction, so it never exercises this.

No standardization is applied: partial correlation is scale-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
RV_WINDOW = 22
SignalKind = Literal["log_h", "rv22"]


@dataclass(frozen=True)
class VolNullData:
    config: SyntheticMarketConfig
    signal: SignalKind = "log_h"

    def __call__(self, seed: int, n_obs: int) -> tuple[FloatArray, tuple[str, ...]]:
        if self.signal == "log_h":
            # One extra simulated day: S_t needs h_{t+1}.
            ds = generate_market(self.config.model_copy(update={"n_obs": n_obs + 1}), seed)
            h = ds.latents[f"h_{TARGET}"].to_numpy(dtype=np.float64)
            r = ds.returns[TARGET].to_numpy(dtype=np.float64)
            return np.column_stack([np.log(h[1:]), r[:-1]]), (SIGNAL, TARGET)
        # rv22: RV_WINDOW - 1 extra leading days so the first row has a full window.
        extra = RV_WINDOW - 1
        ds = generate_market(self.config.model_copy(update={"n_obs": n_obs + extra}), seed)
        r = ds.returns[TARGET].to_numpy(dtype=np.float64)
        csum = np.concatenate(([0.0], np.cumsum(r**2)))
        rv = (csum[RV_WINDOW:] - csum[:-RV_WINDOW]) / RV_WINDOW  # rv[k] covers r[k : k + 22]
        return np.column_stack([np.log(rv), r[extra:]]), (SIGNAL, TARGET)


def vol_null_config(alpha: float | None = None, beta: float | None = None) -> SyntheticMarketConfig:
    """Global-null GARCH config; ``alpha``/``beta`` override the benchmark's GARCH (validated)."""
    cfg = null_config(load_benchmark_config(None), garch=True)
    update = {k: v for k, v in (("alpha", alpha), ("beta", beta)) if v is not None}
    if not update:
        return cfg
    dumped = cfg.model_dump()
    dumped["garch"] = {**dumped["garch"], **update}
    return SyntheticMarketConfig.model_validate(dumped)


def _families(tau_max: int) -> tuple[LinkFamily, ...]:
    lags = tuple(range(1, tau_max + 1))
    null, gate = FamilyRole.NULL, FamilyRole.GATING_NULL
    return (
        LinkFamily(name=f"{SIGNAL}->R", source=SIGNAL, target=TARGET, lags=lags, role=gate),
        LinkFamily(name=f"{SIGNAL}->R@1", source=SIGNAL, target=TARGET, lags=(1,), role=null),
        LinkFamily(name="R->R", source=TARGET, target=TARGET, lags=lags, role=null),
        LinkFamily(name="R->R@1", source=TARGET, target=TARGET, lags=(1,), role=null),
    )


def vol_null_scenario(tau_max: int) -> Scenario:
    """E4 as run in study 1: benchmark GARCH (alpha .06, beta .92), ``log_h`` signal."""
    return Scenario(
        "e4_vol_null",
        "Global null; signal = log conditional variance forecast (vol-correlated null).",
        VolNullData(vol_null_config()),
        _families(tau_max),
    )


def vol_null_variants(tau_max: int) -> tuple[Scenario, ...]:
    """E4 plus the robustness variants of study 2 (ADR 0009)."""
    return (
        vol_null_scenario(tau_max),
        Scenario(
            "e4_low_persist",
            "As E4 with GARCH alpha .10, beta .80 (persistence .90).",
            VolNullData(vol_null_config(alpha=0.10, beta=0.80)),
            _families(tau_max),
        ),
        Scenario(
            "e4_high_persist",
            "As E4 with GARCH alpha .04, beta .95 (persistence .99).",
            VolNullData(vol_null_config(alpha=0.04, beta=0.95)),
            _families(tau_max),
        ),
        Scenario(
            "e4_rv22",
            "As E4 but the signal is 22-day trailing log realized variance.",
            VolNullData(vol_null_config(), signal="rv22"),
            _families(tau_max),
        ),
    )
