r"""Oracle forecasts on the planted-signal benchmark: exact population projections (docs/adr/0013).

The benchmark's returns are (ADR 0003)

    R_{i,t+1} = mu + b_i M_{t+1} + c_{i,t}' X_t + a_{i,t+1},
    c_{i,t} = (beta_z_i, beta_real_i, 0, beta_decay_i * m(t), 0)
              over the state (Z, S_real, S_conf, S_decay, S_noise),

with X a stationary Gaussian VAR(1) with covariance Sigma, a latent GARCH-t market factor M and
independent GARCH-t residuals a. ``m(t)`` is the decay multiplier at the signal's date. For an
information set S of state nodes, the population linear projection of R_{t+1} on X_{S,t} is

    E[R_{t+1} | X_{S,t}] = mu + C_t Sigma_{.S} Sigma_SS^{-1} X_{S,t},

which is exact because X is Gaussian and M and a are independent of it with mean zero. Its
residual covariance is

    C_t (Sigma - Sigma_{.S} Sigma_SS^{-1} Sigma_{S.}) C_t' + b b' sigma_M^2 + diag(sigma_bar^2),

using the *unconditional* GARCH variances.

- **O1** uses the arm's own information set, e.g. the signals Stage 3 forecasts with. It is the
  best any linear forecast on those signals can do, so regret against it isolates estimation and
  optimisation error.
- **O2** uses every observed state node with its true coefficients. Against it, regret also
  includes the signals the pipeline left out.

Limitations: the oracle covariance ignores GARCH's time variation (no method here models it
either), and the projection is the best *linear* predictor, which for this SCM is also the
conditional mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from invariant.benchmark.ground_truth import GroundTruth
from invariant.benchmark.synthetic import STATE_NODES, decay_multiplier

__all__ = ["OracleForecasts", "projection_oracle"]

_FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OracleForecasts:
    """``mean[t]`` forecasts ``returns[t + 1]`` from features dated t. ``cov[t]`` is its residual
    covariance (time-varying only through the decay multiplier)."""

    info: tuple[str, ...]
    mean: _FloatArray  # (T, N)
    cov: _FloatArray  # (T, N, N)


def projection_oracle(
    truth: GroundTruth, features: pd.DataFrame, info: Sequence[str]
) -> OracleForecasts:
    """Exact population projection of ``R_{t+1}`` on the state nodes ``info`` dated t.

    ``features`` holds the dataset's observed state nodes by name (``SyntheticDataset.features``)
    on the dataset's output dates. ``info`` must be observed nodes.
    """
    cfg = truth.config
    names = list(truth.state_order)
    if names != list(STATE_NODES):
        raise ValueError(f"unexpected state order {names}")
    missing = [s for s in info if s not in features.columns]
    if missing or not info:
        raise ValueError(f"information set must be non-empty observed nodes, missing {missing}")
    sigma = np.array(truth.stationary_covariance)
    idx = [names.index(s) for s in info]
    s_ss = sigma[np.ix_(idx, idx)]
    s_xs = sigma[:, idx]
    proj = s_xs @ np.linalg.inv(s_ss)  # (5, |S|): E[X | X_S] = proj X_S
    resid_state = sigma - proj @ s_xs.T

    load = truth.loadings
    n_t = len(features)
    mult = decay_multiplier(cfg)[cfg.burn_in : cfg.burn_in + n_t]
    base = np.column_stack(
        [
            np.array(load.beta_z),
            np.array(load.beta_real),
            np.zeros(len(load.beta_z)),
            np.zeros(len(load.beta_z)),
            np.zeros(len(load.beta_z)),
        ]
    )  # (N, 5)
    decay_col = np.array(load.beta_decay_initial)
    coef = np.repeat(base[None], n_t, axis=0)
    coef[:, :, names.index("s_decay")] = decay_col[None, :] * mult[:, None]  # (T, N, 5)

    x_s = features.loc[:, list(info)].to_numpy(dtype=np.float64)  # (T, |S|)
    mean = cfg.mu + np.einsum("tnk,ks,ts->tn", coef, proj, x_s)
    b = np.array(load.market_beta)
    market_var = cfg.market.garch.sigma_bar**2 if cfg.market is not None else 0.0
    noise = np.outer(b, b) * market_var + np.eye(len(b)) * cfg.garch.sigma_bar**2
    cov = np.einsum("tnk,kl,tml->tnm", coef, resid_state, coef) + noise[None]
    return OracleForecasts(info=tuple(info), mean=mean, cov=cov)
