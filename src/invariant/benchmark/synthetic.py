"""Seeded synthetic market drawn from a known lagged structural causal model (SCM).

Every pipeline stage is validated against this market before it touches real data. The model
and its limitations are documented in docs/adr/0003-planted-signal-benchmark-scm.md.

Model
-----
State ``X_t = (Z, S_real, S_conf, S_decay, S_noise)_t`` is a Gaussian VAR(1), ``X_t = A X_{t-1} +
B nu_t``:

    Z_t   = phi Z_{t-1} + sqrt(1 - phi^2) eta_t
    S^k_t = rho_k S^k_{t-1} + gamma_k Z_{t-1} + sigma_k u^k_t

with ``gamma_k, sigma_k`` from ``config.signal_coefficients`` so that ``Var(S^k) = 1`` and
``Corr(S^k_t, Z_t) = z_corr_k`` (zero for S_decay and S_noise). Returns for asset i:

    R_{i,t}  = mu + b_i M_t + bz_i Z_{t-1} + br_i S_real_{t-1} + bd_i m(t-1) S_decay_{t-1} + a_{i,t}
    a_{i,t}  = sqrt(h_{i,t}) eps_{i,t},  eps ~ Student-t(nu) scaled to unit variance
    h_{i,t}  = omega + alpha a_{i,t-1}^2 + beta h_{i,t-1}
    m(s)     = 1 for s < t0, 2^{-(s - t0) / H} for s >= t0     (s = signal date)

``M_t`` is a latent GARCH-t market factor (multi-asset mode). Loadings are ``ic * sd_i``
times a per-asset dispersion multiplier (``sd_i`` = unconditional sd of the asset's
non-signal return). GARCH acts on the residual ``a``, never on the signal
part, so volatility is independent of the signals; signal innovations are independent of return
innovations, so there is no Stambaugh (1999) predictive-regression bias by construction.

Initialisation: ``X_0 ~ N(0, Sigma)`` with ``Sigma`` the stationary covariance (discrete Lyapunov
equation ``Sigma = A Sigma A' + B B'``) and ``h_0 = sigma_bar^2``, the unconditional variance, so
burn-in is a safety margin rather than a requirement. The first simulated return (inside
burn-in) has no lagged state and omits the signal terms.

Assumptions and limitations (see the ADR for the full list): linear mean effects, Gaussian
signal innovations, symmetric GARCH (no leverage effect), no jumps or regime switches, constant
coefficients except S_decay, a weekday calendar without holidays. S_decay makes the return
process non-stationary in its conditional mean by design, which violates the stationarity
assumption of PCMCI-type causal discovery.

Randomness
----------
Every component draws from its own named ``RngFactory`` stream (``state/z``, ``state/s_real``, ...,
``eps/asset_00``, ``eps/market``, ``loadings/signal``, ``loadings/market``). Changing one
parameter therefore leaves the draws of every other component unchanged (common random numbers
across difficulty sweeps), and adding assets leaves existing assets' residual paths unchanged.

References: Bollerslev (1986) "Generalized autoregressive conditional heteroskedasticity";
Bollerslev (1987) "A conditionally heteroskedastic time series model for speculative prices and
rates of return" (GARCH-t); Stambaugh (1999) "Predictive regressions"; Runge et al. (2019)
"Detecting and quantifying causal associations in large nonlinear time series datasets" (the
lagged-SCM / PCMCI setting this benchmark targets).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import linalg, signal

from invariant.benchmark.config import (
    GarchParams,
    RegimeParams,
    SyntheticMarketConfig,
    VolFragParams,
    signal_coefficients,
)
from invariant.benchmark.ground_truth import (
    EXPECTED_VALUES_TARGET_TIMING,
    CausalEdge,
    GroundTruth,
    Mechanism,
    Node,
    NodeRole,
    RealizedLoadings,
    RegimeIC,
    RegimeTruth,
    SignalTruth,
    VolFragTruth,
)
from invariant.core.logging import get_logger
from invariant.core.run_hash import RngFactory
from invariant.core.types import Frequency, StageName, Verdict, validate_time_index

__all__ = [
    "DETECTABILITY_T",
    "REGIME_NODE",
    "REGIME_SIGNAL",
    "SIGNAL_NODES",
    "STATE_NODES",
    "VOLFRAG_SIGNAL",
    "Innovations",
    "SimulatedPaths",
    "SyntheticDataset",
    "asset_names",
    "build_ground_truth",
    "decay_multiplier",
    "draw_innovations",
    "draw_loadings",
    "generate_market",
    "population_moments",
    "population_return_acf",
    "regime_conditional_variance",
    "regime_path",
    "regime_truth",
    "signal_nodes",
    "simulate",
    "state_nodes",
    "stationary_covariance",
    "transition_matrices",
    "volfrag_threshold",
    "volfrag_truth",
]

FloatArray = npt.NDArray[np.float64]

STATE_NODES: tuple[str, ...] = ("z", "s_real", "s_conf", "s_decay", "s_noise")
SIGNAL_NODES: tuple[str, ...] = STATE_NODES[1:]
_Z, _REAL, _CONF, _DECAY, _NOISE = range(5)
MARKET_NODE = "market"
REGIME_SIGNAL = "s_regime"
"""v4 (ADR 0015): appended to the state as index 5 when ``config.regime`` is set."""
REGIME_NODE = "regime"
VOLFRAG_SIGNAL = "s_volfrag"
"""v4.1 (ADR 0015 amendment): appended to the state as index 5 when ``config.volfrag`` is set."""
VOLFRAG_MC_SEED = 20261102
VOLFRAG_N_MC = 2_000_000
_REGIME = 5  # index of the v4 extra signal (s_regime or s_volfrag)
_CALM, _STRESS = 0, 1

DETECTABILITY_T = 3.0
"""A true signal whose expected Stage-1 t-stat is below this triggers a low-power warning."""


def asset_names(n_assets: int) -> tuple[str, ...]:
    return tuple(f"asset_{i:02d}" for i in range(n_assets))


def _extra_signal(config: SyntheticMarketConfig) -> str | None:
    if config.regime is not None:
        return REGIME_SIGNAL
    if config.volfrag is not None:
        return VOLFRAG_SIGNAL
    return None


def state_nodes(config: SyntheticMarketConfig) -> tuple[str, ...]:
    """``STATE_NODES``, plus ``s_regime`` or ``s_volfrag`` when the config has a v4 block."""
    extra = _extra_signal(config)
    return STATE_NODES + ((extra,) if extra is not None else ())


def signal_nodes(config: SyntheticMarketConfig) -> tuple[str, ...]:
    return state_nodes(config)[1:]


# ---------------------------------------------------------------------------------------------
# Population structure
# ---------------------------------------------------------------------------------------------


def transition_matrices(config: SyntheticMarketConfig) -> tuple[FloatArray, FloatArray]:
    """``(A, B)`` of the state VAR(1) ``X_t = A X_{t-1} + B nu_t``, ``nu ~ N(0, I)``.

    5 x 5 over ``STATE_NODES``; 6 x 6 with ``s_regime`` (an independent AR(1)) when a regime is
    configured.
    """
    phi = config.factor.phi
    a = np.zeros((5, 5))
    b = np.zeros((5, 5))
    a[_Z, _Z] = phi
    b[_Z, _Z] = math.sqrt(1.0 - phi**2)
    specs = (
        (_REAL, config.real.rho, config.real.z_corr),
        (_CONF, config.conf.rho, config.conf.z_corr),
        (_DECAY, config.decay.rho, 0.0),
        (_NOISE, config.noise.rho, 0.0),
    )
    for k, rho, z_corr in specs:
        gamma, sigma2 = signal_coefficients(rho, z_corr, phi)
        a[k, k] = rho
        a[k, _Z] = gamma
        b[k, k] = math.sqrt(sigma2)
    if config.regime is None and config.volfrag is None:
        return a, b
    rho_r = config.regime.signal.rho if config.regime is not None else 0.0
    a6, b6 = np.zeros((6, 6)), np.zeros((6, 6))
    a6[:5, :5], b6[:5, :5] = a, b
    a6[_REGIME, _REGIME] = rho_r
    b6[_REGIME, _REGIME] = math.sqrt(1.0 - rho_r**2)
    return a6, b6


def stationary_covariance(config: SyntheticMarketConfig) -> FloatArray:
    """Stationary covariance of ``X``: solves ``Sigma = A Sigma A' + B B'``."""
    a, b = transition_matrices(config)
    sigma = np.asarray(linalg.solve_discrete_lyapunov(a, b @ b.T), dtype=np.float64)
    return (sigma + sigma.T) / 2.0


def decay_multiplier(config: SyntheticMarketConfig) -> FloatArray:
    """``m(s)`` for every simulated step s (burn-in included), indexed by signal date."""
    j = np.arange(config.n_total, dtype=np.float64) - config.burn_in
    elapsed = np.maximum(j - config.decay_start_index, 0.0)
    return np.asarray(np.exp2(-elapsed / config.decay.half_life), dtype=np.float64)


def regime_path(regime: RegimeParams, u: FloatArray) -> npt.NDArray[np.int64]:
    """Markov chain path (0 = calm, 1 = stress) driven by uniforms ``u``.

    ``xi_0`` is drawn from the stationary distribution (stress iff ``u_0 < pi_s``). Afterwards
    calm moves to stress iff ``u_s < P(calm -> stress)``, and stress moves to calm iff
    ``u_s < P(stress -> calm)``. Causal: ``xi_s`` depends only on ``u_0..u_s``.
    """
    q, p = regime.calm_to_stress, regime.stress_to_calm
    out = np.empty(len(u), dtype=np.int64)
    state = int(u[0] < regime.stress_share)
    for s, us in enumerate(u.tolist()):
        if s > 0:
            state = int(us < q) if state == _CALM else int(us >= p)
        out[s] = state
    return out


def regime_conditional_variance(
    regime: RegimeParams,
    garch: GarchParams,
    vol_ratio: float,
) -> tuple[float, float]:
    """Exact ``E[h_t | xi_t = k]`` for the Markov-switching-intercept GARCH, k = (calm, stress).

    With ``m_k = E[h_t 1{xi_t = k}]``, ``h_t = omega_{xi_t} + alpha a_{t-1}^2 + beta h_{t-1}``,
    ``E[a_{t-1}^2 | F] = h_{t-1}`` and an exogenous chain,
    ``m_k = omega_k pi_k + (alpha + beta) sum_j m_j P_jk``: a 2 x 2 linear system. The values
    sum (over pi) to the unconditional variance ``sum_k pi_k omega_k / (1 - alpha - beta)``.
    """
    p_mat = np.array(regime.transition_matrix)
    pi = np.array(regime.stationary)
    omega = garch.omega * np.array([1.0, vol_ratio**2])
    m = np.linalg.solve(np.eye(2) - (garch.alpha + garch.beta) * p_mat.T, omega * pi)
    cond = m / pi
    return float(cond[0]), float(cond[1])


@lru_cache(maxsize=8)
def volfrag_threshold(garch: GarchParams, h_quantile: float) -> tuple[float, float, float]:
    """``(tau, E[h | h <= tau], E[h | h > tau])`` for a GARCH-t's conditional variance h.

    ``tau`` is the population ``h_quantile``-quantile of h. There is no closed form, so it is
    the empirical quantile of a fixed-seed (``VOLFRAG_MC_SEED``) ``VOLFRAG_N_MC``-step
    simulation started at the unconditional variance (the first 10,000 steps are dropped).
    """
    gen = np.random.default_rng(VOLFRAG_MC_SEED)
    eps = _standardized_t(gen, garch.nu, VOLFRAG_N_MC + 10_000)
    h, _ = _garch_path(eps, garch)
    h = h[10_000:]
    tau = float(np.quantile(h, h_quantile))
    return tau, float(h[h <= tau].mean()), float(h[h > tau].mean())


def _unconditional_variance(regime: RegimeParams, garch: GarchParams, vol_ratio: float) -> float:
    pi_c, pi_s = regime.stationary
    return garch.sigma_bar**2 * (pi_c + pi_s * vol_ratio**2)


# ---------------------------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------------------------


def draw_loadings(config: SyntheticMarketConfig, rng: RngFactory) -> RealizedLoadings:
    """Per-asset loadings ``ic * sd_i * exp(delta xi - delta^2 / 2)``.

    ``sd_i = sqrt(sigma_bar^2 + b_i^2 sigma_bar_M^2)`` is the unconditional sd of the asset's
    non-signal return (residual plus market), so a nominal ``ic`` means the same in single- and
    multi-asset mode. With a regime block the variances are the stationary regime mixtures, and
    ``S_regime``'s calm loading is drawn from its own stream (``loadings/regime``).
    """
    n = config.n_assets
    delta = config.loading_dispersion
    regime = config.regime
    if config.market is None:
        market_beta = np.zeros(n)
        market_var = 0.0
    else:
        m = config.market
        market_beta = rng.generator("loadings/market").uniform(m.beta_low, m.beta_high, n)
        market_var = m.garch.sigma_bar**2
        if regime is not None:
            market_var = _unconditional_variance(regime, m.garch, regime.market_vol_ratio)
    resid_var = config.garch.sigma_bar**2
    if regime is not None:
        resid_var = _unconditional_variance(regime, config.garch, regime.resid_vol_ratio)
    noise_sd = np.sqrt(resid_var + market_beta**2 * market_var)
    xi = rng.generator("loadings/signal").standard_normal((n, 3))
    mult = np.exp(delta * xi - delta**2 / 2.0)
    nominal_ic = np.array([config.factor.ic, config.real.ic, config.decay.ic])
    betas = mult * nominal_ic * noise_sd[:, None]
    beta_regime = beta_volfrag = None
    if regime is not None:
        xi_r = rng.generator("loadings/regime").standard_normal(n)
        mult_r = np.exp(delta * xi_r - delta**2 / 2.0)
        beta_regime = tuple(float(v) for v in mult_r * regime.signal.ic * noise_sd)
    if config.volfrag is not None:
        xi_v = rng.generator("loadings/volfrag").standard_normal(n)
        mult_v = np.exp(delta * xi_v - delta**2 / 2.0)
        beta_volfrag = tuple(float(v) for v in mult_v * config.volfrag.ic * noise_sd)
    return RealizedLoadings(
        beta_z=tuple(float(v) for v in betas[:, 0]),
        beta_real=tuple(float(v) for v in betas[:, 1]),
        beta_decay_initial=tuple(float(v) for v in betas[:, 2]),
        market_beta=tuple(float(v) for v in market_beta),
        beta_regime=beta_regime,
        beta_volfrag=beta_volfrag,
    )


@dataclass(frozen=True)
class Innovations:
    """All random inputs of one simulation. ``simulate`` is a pure function of these.

    ``state[s]`` holds the standard-normal innovations ``(eta, u_real, u_conf, u_decay, u_noise)``
    of step s (row 0 unused: the initial state is ``x0``). ``eps`` / ``eps_market`` are
    unit-variance Student-t residual innovations. With a regime block, ``regime_u`` holds the
    chain's uniforms and ``s_regime`` the standard-normal innovations of ``S_regime`` (row 0 is
    its stationary initial value).
    """

    x0: FloatArray
    state: FloatArray
    eps: FloatArray
    eps_market: FloatArray | None
    regime_u: FloatArray | None = None
    s_regime: FloatArray | None = None
    s_volfrag: FloatArray | None = None


def _standardized_t(gen: np.random.Generator, nu: float, n: int) -> FloatArray:
    return np.asarray(gen.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu), dtype=np.float64)


def draw_innovations(config: SyntheticMarketConfig, rng: RngFactory) -> Innovations:
    """Draw every innovation from its own named stream of ``rng`` (see module docstring)."""
    n = config.n_total
    chol = np.linalg.cholesky(stationary_covariance(config))[:5, :5]
    x0 = chol @ rng.generator("state/x0").standard_normal(5)
    state = np.column_stack(
        [rng.generator(f"state/{name}").standard_normal(n) for name in STATE_NODES],
    )
    nu = config.garch.nu
    eps = np.column_stack(
        [
            _standardized_t(rng.generator(f"eps/{name}"), nu, n)
            for name in asset_names(config.n_assets)
        ],
    )
    eps_market = None
    if config.market is not None:
        eps_market = _standardized_t(rng.generator("eps/market"), config.market.garch.nu, n)
    if config.volfrag is not None:
        return Innovations(
            x0=x0,
            state=state,
            eps=eps,
            eps_market=eps_market,
            s_volfrag=rng.generator(f"state/{VOLFRAG_SIGNAL}").standard_normal(n),
        )
    if config.regime is None:
        return Innovations(x0=x0, state=state, eps=eps, eps_market=eps_market)
    # chol is 6 x 6 here but s_regime is independent with unit variance, so x0 drawn above from
    # the first five draws is unaffected; s_regime's initial value is row 0 of its own stream.
    return Innovations(
        x0=x0,
        state=state,
        eps=eps,
        eps_market=eps_market,
        regime_u=rng.generator("state/regime").random(n),
        s_regime=rng.generator(f"state/{REGIME_SIGNAL}").standard_normal(n),
    )


# ---------------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatedPaths:
    """Full simulated paths, burn-in included. Row s is step s."""

    state: FloatArray  # (n_total, 5 or 6), columns state_nodes(config)
    returns: FloatArray  # (n_total, n_assets)
    h: FloatArray  # (n_total, n_assets) conditional variance of the residual
    market: FloatArray  # (n_total,), zeros without a market factor
    h_market: FloatArray  # (n_total,), zeros without a market factor
    decay_multiplier: FloatArray  # (n_total,), indexed by signal date
    regime: npt.NDArray[np.int64] | None = None  # (n_total,), 0 calm / 1 stress; v4 only


def _garch_path(eps: FloatArray, garch: GarchParams) -> tuple[FloatArray, FloatArray]:
    """``(h, a)`` of a GARCH(1,1) driven by ``eps``, with ``h_0`` = unconditional variance."""
    omega, alpha, beta = garch.omega, garch.alpha, garch.beta
    h_val = garch.sigma_bar**2
    a_val = 0.0
    h_out: list[float] = []
    a_out: list[float] = []
    for s, e in enumerate(eps.tolist()):
        if s > 0:
            h_val = omega + alpha * a_val * a_val + beta * h_val
        a_val = math.sqrt(h_val) * e
        h_out.append(h_val)
        a_out.append(a_val)
    return np.array(h_out), np.array(a_out)


def _ms_garch_path(
    eps: FloatArray,
    garch: GarchParams,
    scale: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """``(h, a)`` of a GARCH(1,1) whose intercept switches with the regime (v4).

    ``h_s = omega scale_s^2 + alpha a_{s-1}^2 + beta h_{s-1}``: in regime k the unconditional
    variance would be ``(scale_k sigma_bar)^2``. One recursion, so there is no path dependence
    (the problem Gray 1996 and Haas, Mittnik & Paolella 2004 solve for regime-specific alpha,
    beta). ``h_0 = (scale_0 sigma_bar)^2``; burn-in absorbs the start.
    """
    omega, alpha, beta = garch.omega, garch.alpha, garch.beta
    omegas = (omega * scale * scale).tolist()
    h_val = garch.sigma_bar**2 * float(scale[0]) ** 2
    a_val = 0.0
    h_out: list[float] = []
    a_out: list[float] = []
    for s, e in enumerate(eps.tolist()):
        if s > 0:
            h_val = omegas[s] + alpha * a_val * a_val + beta * h_val
        a_val = math.sqrt(h_val) * e
        h_out.append(h_val)
        a_out.append(a_val)
    return np.array(h_out), np.array(a_out)


def simulate(
    config: SyntheticMarketConfig,
    innovations: Innovations,
    loadings: RealizedLoadings,
) -> SimulatedPaths:
    """Run the SCM recursions. Pure and causal: step s uses only innovations at steps <= s."""
    n = config.n_total
    if innovations.state.shape != (n, 5) or innovations.eps.shape != (n, config.n_assets):
        raise ValueError("innovations do not match the config's dimensions")
    if config.regime is not None or config.volfrag is not None:
        return _simulate_v4(config, innovations, loadings)
    a_mat, b_mat = transition_matrices(config)
    state = np.empty((n, 5))
    # Z: AR(1) via lfilter; feeding x0 as the step-0 input yields state[0] = x0 exactly.
    z_in = b_mat[_Z, _Z] * innovations.state[:, _Z]
    z_in[0] = innovations.x0[_Z]
    state[:, _Z] = signal.lfilter([1.0], [1.0, -a_mat[_Z, _Z]], z_in)
    for k in range(1, 5):
        s_in = np.empty(n)
        s_in[0] = innovations.x0[k]
        s_in[1:] = a_mat[k, _Z] * state[:-1, _Z] + b_mat[k, k] * innovations.state[1:, k]
        state[:, k] = signal.lfilter([1.0], [1.0, -a_mat[k, k]], s_in)

    h = np.empty((n, config.n_assets))
    resid = np.empty((n, config.n_assets))
    for i in range(config.n_assets):
        h[:, i], resid[:, i] = _garch_path(innovations.eps[:, i], config.garch)

    market = np.zeros(n)
    h_market = np.zeros(n)
    if config.market is not None:
        if innovations.eps_market is None or innovations.eps_market.shape != (n,):
            raise ValueError("market factor configured but eps_market missing or misshapen")
        h_market, market = _garch_path(innovations.eps_market, config.market.garch)

    mult = decay_multiplier(config)
    beta_z = np.array(loadings.beta_z)
    beta_real = np.array(loadings.beta_real)
    beta_decay = np.array(loadings.beta_decay_initial)
    mean = np.zeros((n, config.n_assets))
    lagged = state[:-1]
    mean[1:] = (
        lagged[:, [_Z]] * beta_z
        + lagged[:, [_REAL]] * beta_real
        + (lagged[:, _DECAY] * mult[:-1])[:, None] * beta_decay
    )
    returns = config.mu + market[:, None] * np.array(loadings.market_beta) + mean + resid
    return SimulatedPaths(
        state=state,
        returns=returns,
        h=h,
        market=market,
        h_market=h_market,
        decay_multiplier=mult,
    )


def _simulate_v4(
    config: SyntheticMarketConfig,
    innovations: Innovations,
    loadings: RealizedLoadings,
) -> SimulatedPaths:
    """``simulate`` for a v4 config: the v3 recursions plus the extra signal (index 5).

    ``regime``: the chain, the switching GARCH intercepts and ``c(xi_t) beta S_regime_{t-1}``.
    ``volfrag``: plain GARCH, and ``c(h_{M,t}) beta S_volfrag_{t-1}`` with ``c`` flipped when the
    market's conditional variance exceeds ``volfrag_threshold``.
    """
    regime, volfrag = config.regime, config.volfrag
    if regime is not None:
        extra_innov, beta_extra = innovations.s_regime, loadings.beta_regime
        if innovations.regime_u is None:
            raise ValueError("regime configured but its innovations are missing")
    else:
        assert volfrag is not None  # dispatched by simulate
        extra_innov, beta_extra = innovations.s_volfrag, loadings.beta_volfrag
    if extra_innov is None or beta_extra is None:
        raise ValueError("v4 signal configured but its innovations or loadings are missing")
    n = config.n_total
    a_mat, b_mat = transition_matrices(config)
    state = np.empty((n, 6))
    z_in = b_mat[_Z, _Z] * innovations.state[:, _Z]
    z_in[0] = innovations.x0[_Z]
    state[:, _Z] = signal.lfilter([1.0], [1.0, -a_mat[_Z, _Z]], z_in)
    for k in range(1, 5):
        s_in = np.empty(n)
        s_in[0] = innovations.x0[k]
        s_in[1:] = a_mat[k, _Z] * state[:-1, _Z] + b_mat[k, k] * innovations.state[1:, k]
        state[:, k] = signal.lfilter([1.0], [1.0, -a_mat[k, k]], s_in)
    r_in = b_mat[_REGIME, _REGIME] * extra_innov
    r_in[0] = extra_innov[0]
    state[:, _REGIME] = signal.lfilter([1.0], [1.0, -a_mat[_REGIME, _REGIME]], r_in)

    xi = None
    resid_scale = market_scale = np.ones(n)
    if regime is not None:
        assert innovations.regime_u is not None
        xi = regime_path(regime, innovations.regime_u)
        stress = xi == _STRESS
        resid_scale = np.where(stress, regime.resid_vol_ratio, 1.0)
        market_scale = np.where(stress, regime.market_vol_ratio, 1.0)
    h = np.empty((n, config.n_assets))
    resid = np.empty((n, config.n_assets))
    for i in range(config.n_assets):
        if regime is not None:
            h[:, i], resid[:, i] = _ms_garch_path(innovations.eps[:, i], config.garch, resid_scale)
        else:
            h[:, i], resid[:, i] = _garch_path(innovations.eps[:, i], config.garch)
    market = np.zeros(n)
    h_market = np.zeros(n)
    if config.market is not None:
        if innovations.eps_market is None or innovations.eps_market.shape != (n,):
            raise ValueError("market factor configured but eps_market missing or misshapen")
        if regime is not None:
            h_market, market = _ms_garch_path(
                innovations.eps_market, config.market.garch, market_scale
            )
        else:
            h_market, market = _garch_path(innovations.eps_market, config.market.garch)

    if regime is not None:
        assert xi is not None
        c_path = np.where(xi == _STRESS, regime.signal.stress_ratio, 1.0)
    else:
        assert volfrag is not None and config.market is not None
        tau, _, _ = volfrag_threshold(config.market.garch, volfrag.h_quantile)
        c_path = np.where(h_market > tau, volfrag.stress_ratio, 1.0)
    mult = decay_multiplier(config)
    lagged = state[:-1]
    mean = np.zeros((n, config.n_assets))
    mean[1:] = (
        lagged[:, [_Z]] * np.array(loadings.beta_z)
        + lagged[:, [_REAL]] * np.array(loadings.beta_real)
        + (lagged[:, _DECAY] * mult[:-1])[:, None] * np.array(loadings.beta_decay_initial)
        + (lagged[:, _REGIME] * c_path[1:])[:, None] * np.array(beta_extra)
    )
    returns = config.mu + market[:, None] * np.array(loadings.market_beta) + mean + resid
    return SimulatedPaths(
        state=state,
        returns=returns,
        h=h,
        market=market,
        h_market=h_market,
        decay_multiplier=mult,
        regime=xi,
    )


# ---------------------------------------------------------------------------------------------
# Population moments and ground truth
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _AssetMoments:
    marginal_ic: FloatArray  # (d,) Corr(X_k,t-1, R_t), initial coefficients
    partial_ic: FloatArray  # (d,) given other observed parents, initial coefficients
    effective_partial_ic: FloatArray  # (d,) with S_decay at its average coefficient
    effective_hybrid_ic: FloatArray  # (d,) given Stage 1's oracle hybrid set, S_decay averaged


def _observed_parents(config: SyntheticMarketConfig) -> list[int]:
    parents = []
    if config.observe_confounder and config.factor.ic > 0:
        parents.append(_Z)
    if config.real.ic > 0:
        parents.append(_REAL)
    if config.decay.ic > 0:
        parents.append(_DECAY)
    if config.regime is not None and config.regime.signal.ic > 0:
        parents.append(_REGIME)
    if config.volfrag is not None and config.volfrag.ic > 0:
        parents.append(_REGIME)
    return parents


def _partial_corr(cov: FloatArray, i: int, j: int, cond: list[int]) -> float:
    idx = [i, j, *cond]
    precision = np.linalg.inv(cov[np.ix_(idx, idx)])
    return float(-precision[0, 1] / math.sqrt(precision[0, 0] * precision[1, 1]))


def _noise_var(config: SyntheticMarketConfig, market_beta: float) -> float:
    """Unconditional variance of the non-signal return (residual plus market)."""
    regime = config.regime
    if regime is None:
        var_resid = config.garch.sigma_bar**2
        if config.market is not None:
            var_resid += market_beta**2 * config.market.garch.sigma_bar**2
        return var_resid
    var_resid = _unconditional_variance(regime, config.garch, regime.resid_vol_ratio)
    if config.market is not None:
        var_resid += market_beta**2 * _unconditional_variance(
            regime, config.market.garch, regime.market_vol_ratio
        )
    return var_resid


def _joint_cov(sigma: FloatArray, coef: FloatArray, var_resid: float) -> FloatArray:
    """Stationary covariance of ``(X_{t-1}, R_t)`` for loading vector ``coef`` on ``X_{t-1}``."""
    d = sigma.shape[0]
    joint = np.zeros((d + 1, d + 1))
    joint[:d, :d] = sigma
    cov_xr = sigma @ coef
    joint[:d, d] = cov_xr
    joint[d, :d] = cov_xr
    joint[d, d] = float(coef @ sigma @ coef) + var_resid
    return joint


def _joint_cov_two_lags(
    config: SyntheticMarketConfig,
    sigma: FloatArray,
    coef: FloatArray,
    var_resid: float,
) -> FloatArray:
    """Stationary covariance of ``(X_{t-1}, X_{t-2}, R_t)`` (indices 0..d-1, d..2d-1, 2d).

    ``X_{t-1} = A X_{t-2} + B e_{t-1}`` gives ``Cov(X_{t-1}, X_{t-2}) = A Sigma`` and
    ``Cov(X_{t-2}, R_t) = (A Sigma)' coef``.
    """
    a, _ = transition_matrices(config)
    d = sigma.shape[0]
    one_lag = _joint_cov(sigma, coef, var_resid)
    cross = a @ sigma
    joint = np.zeros((2 * d + 1, 2 * d + 1))
    joint[:d, :d] = sigma
    joint[d : 2 * d, d : 2 * d] = sigma
    joint[:d, d : 2 * d] = cross
    joint[d : 2 * d, :d] = cross.T
    joint[:d, 2 * d] = joint[2 * d, :d] = one_lag[:d, d]
    joint[d : 2 * d, 2 * d] = joint[2 * d, d : 2 * d] = cross.T @ coef
    joint[2 * d, 2 * d] = one_lag[d, d]
    return joint


def hybrid_oracle_set(config: SyntheticMarketConfig, k: int) -> list[int]:
    """Stage 1's ``hybrid`` conditioning set for ``X^k_{t-1} -> R_t`` with the true parents in
    place of PC1's, as indices into ``(X_{t-1}, X_{t-2})``: the observed parents of ``R`` other
    than ``X^k_{t-1}``, plus the observed parents of ``X^k`` other than its own lag, shifted by
    one lag. In this SCM the only such parent is ``Z_{t-1}`` (so ``Z_{t-2}``), for a signal
    that loads on Z, when Z is observed.
    """
    a, _ = transition_matrices(config)
    d = a.shape[0]
    cond = [p for p in _observed_parents(config) if p != k]
    if k != _Z and config.observe_confounder and a[k, _Z] != 0.0:
        cond.append(d + _Z)
    return cond


def _mean_multiplier(spec: RegimeParams | VolFragParams) -> tuple[float, float]:
    """``(E[c], Var(c))``: under the stationary chain, or over the volatility state."""
    if isinstance(spec, RegimeParams):
        pi = np.array(spec.stationary)
    else:
        pi = np.array([spec.h_quantile, spec.high_share])
    c = np.array(spec.loading_multipliers)
    mean = float(pi @ c)
    return mean, float(pi @ (c - mean) ** 2)


def _extra_loading(
    config: SyntheticMarketConfig,
    loadings: RealizedLoadings,
    asset: int,
) -> tuple[float, RegimeParams | VolFragParams] | None:
    """The v4 signal's calm/low-volatility loading on ``asset`` and its multiplier spec."""
    if config.regime is not None:
        assert loadings.beta_regime is not None  # drawn whenever regime is set
        return loadings.beta_regime[asset], config.regime
    if config.volfrag is not None:
        assert loadings.beta_volfrag is not None
        return loadings.beta_volfrag[asset], config.volfrag
    return None


def population_moments(
    config: SyntheticMarketConfig,
    loadings: RealizedLoadings,
) -> list[_AssetMoments]:
    """Exact population ICs per asset for every state variable (see ``SignalTruth``).

    The effective coefficient of S_decay is its average multiplier over the ``n_obs - 1`` signal
    dates that have a next-day return; treating it as constant is a stationary approximation of
    the full-sample regression slope. Return lags are not conditioned on: given the parents,
    they carry no linear information about ``R_t`` (GARCH affects only the variance).

    With a regime (v4), ``S_regime``'s term ``c(xi_t) beta S_{t-1}`` is split into its linear
    part ``E[c] beta S_{t-1}`` and ``(c - E[c]) beta S_{t-1}``, which is uncorrelated with every
    state variable (the chain is independent of the state), so it adds ``Var(c) beta^2`` to the
    noise variance. Second moments, and hence every IC here, are exact; per-regime ICs are in
    ``RegimeTruth``.
    """
    sigma = stationary_covariance(config)
    d = sigma.shape[0]
    mult = decay_multiplier(config)[config.burn_in : config.burn_in + config.n_obs - 1]
    decay_scale = float(mult.mean())
    parents = _observed_parents(config)
    out = []
    for i in range(config.n_assets):
        coef = np.zeros(d)
        coef[_Z] = loadings.beta_z[i]
        coef[_REAL] = loadings.beta_real[i]
        coef[_DECAY] = loadings.beta_decay_initial[i]
        mb = loadings.market_beta[i]
        var_resid = _noise_var(config, mb)
        extra = _extra_loading(config, loadings, i)
        if extra is not None:
            beta_x, spec = extra
            mean_c, var_c = _mean_multiplier(spec)
            coef[_REGIME] = beta_x * mean_c
            var_resid += var_c * beta_x**2
        coef_eff = coef.copy()
        coef_eff[_DECAY] *= decay_scale
        joint = _joint_cov(sigma, coef, var_resid)
        joint_eff = _joint_cov(sigma, coef_eff, var_resid)
        marginal = joint[:d, d] / math.sqrt(joint[d, d])
        partial = np.array(
            [_partial_corr(joint, k, d, [p for p in parents if p != k]) for k in range(d)]
        )
        partial_eff = np.array(
            [_partial_corr(joint_eff, k, d, [p for p in parents if p != k]) for k in range(d)],
        )
        joint2_eff = _joint_cov_two_lags(config, sigma, coef_eff, var_resid)
        hybrid_eff = np.array(
            [_partial_corr(joint2_eff, k, 2 * d, hybrid_oracle_set(config, k)) for k in range(d)],
        )
        out.append(_AssetMoments(marginal, partial, partial_eff, hybrid_eff))
    return out


def population_return_acf(
    config: SyntheticMarketConfig,
    loadings: RealizedLoadings,
    asset: int,
    max_lag: int,
) -> FloatArray:
    """Exact ``Corr(R_t, R_{t-k})`` for k = 1..max_lag (S_decay at its average coefficient).

    Only the mean terms are autocorrelated (residual and market shocks are martingale
    differences independent of the state), so ``Cov(R_t, R_{t-k}) = coef' A^k Sigma coef``,
    plus, with a regime, ``beta_r^2 rho_r^k E[c(xi_t) c(xi_{t-k})]`` for S_regime's switching
    term, with ``E[c_t c_{t-k}] = sum_ij pi_i c_i (P^k)_ij c_j``. A persistent predictable mean
    is what makes the benchmark's returns autocorrelated, so this bounds how strong planted
    edges can be before the no-linear-autocorrelation stylized fact fails (ADR 0015).
    """
    sigma = stationary_covariance(config)
    a_mat, _ = transition_matrices(config)
    d = sigma.shape[0]
    mult = decay_multiplier(config)[config.burn_in : config.burn_in + config.n_obs - 1]
    coef = np.zeros(d)
    coef[_Z] = loadings.beta_z[asset]
    coef[_REAL] = loadings.beta_real[asset]
    coef[_DECAY] = loadings.beta_decay_initial[asset] * float(mult.mean())
    var_noise = _noise_var(config, loadings.market_beta[asset])
    if config.volfrag is not None:
        # rho = 0: S_volfrag's term is serially uncorrelated and adds only E[c^2] beta^2.
        assert loadings.beta_volfrag is not None
        pi_v = np.array([config.volfrag.h_quantile, config.volfrag.high_share])
        c_v = np.array(config.volfrag.loading_multipliers)
        var_noise += float(pi_v @ c_v**2) * loadings.beta_volfrag[asset] ** 2
    regime = config.regime
    if regime is not None:
        assert loadings.beta_regime is not None
        beta_r = loadings.beta_regime[asset]
        pi = np.array(regime.stationary)
        c = np.array(regime.loading_multipliers)
        p_mat = np.array(regime.transition_matrix)
        var_noise += float(pi @ c**2) * beta_r**2
    var_r = float(coef @ sigma @ coef) + var_noise
    out = np.empty(max_lag)
    a_k = np.eye(d)
    p_k = np.eye(2)
    for k in range(1, max_lag + 1):
        a_k = a_k @ a_mat
        cov = float(coef @ a_k @ sigma @ coef)
        if regime is not None:
            p_k = p_k @ p_mat
            rho_r = regime.signal.rho
            cov += beta_r**2 * rho_r**k * float((pi * c) @ p_k @ c)
        out[k - 1] = cov / var_r
    return out


def regime_truth(config: SyntheticMarketConfig, loadings: RealizedLoadings) -> RegimeTruth:
    """The ``RegimeTruth`` of a v4 config, with exact per-regime ICs.

    ``Corr(S_{t-1}, R_t | xi_t = k)``: the state is independent of ``xi_t``, so the covariance is
    ``(Sigma coef_k)_S`` and ``Var(R_t | xi_t = k) = coef_k' Sigma coef_k + E[h | k] +
    b^2 E[h_M | k]``, where ``coef_k`` has ``S_regime``'s loading times ``c_k`` and S_decay at its
    average coefficient.
    """
    regime = config.regime
    if regime is None or loadings.beta_regime is None:
        raise ValueError("regime_truth needs a config with a regime block")
    sigma = stationary_covariance(config)
    d = sigma.shape[0]
    resid_v = regime_conditional_variance(regime, config.garch, regime.resid_vol_ratio)
    market_v = (0.0, 0.0)
    if config.market is not None:
        market_v = regime_conditional_variance(regime, config.market.garch, regime.market_vol_ratio)
    mult = decay_multiplier(config)[config.burn_in : config.burn_in + config.n_obs - 1]
    decay_scale = float(mult.mean())
    ics: dict[str, tuple[list[float], list[float]]] = {
        "s_real": ([], []),
        REGIME_SIGNAL: ([], []),
    }
    for i in range(config.n_assets):
        for k, c_k in enumerate(regime.loading_multipliers):
            coef = np.zeros(d)
            coef[_Z] = loadings.beta_z[i]
            coef[_REAL] = loadings.beta_real[i]
            coef[_DECAY] = loadings.beta_decay_initial[i] * decay_scale
            coef[_REGIME] = loadings.beta_regime[i] * c_k
            var_r = (
                float(coef @ sigma @ coef) + resid_v[k] + loadings.market_beta[i] ** 2 * market_v[k]
            )
            cov = sigma @ coef
            for sid, idx in (("s_real", _REAL), (REGIME_SIGNAL, _REGIME)):
                ics[sid][k].append(float(cov[idx]) / math.sqrt(sigma[idx, idx] * var_r))
    return RegimeTruth(
        transition_matrix=regime.transition_matrix,
        stationary=regime.stationary,
        expected_durations=(regime.calm_duration, regime.stress_duration),
        resid_vol_ratio=regime.resid_vol_ratio,
        market_vol_ratio=regime.market_vol_ratio,
        loading_multipliers=regime.loading_multipliers,
        resid_variance=resid_v,
        market_variance=market_v,
        regime_ic=tuple(
            RegimeIC(signal_id=sid, calm=tuple(calm), stress=tuple(stress))
            for sid, (calm, stress) in ics.items()
        ),
    )


def volfrag_truth(config: SyntheticMarketConfig, loadings: RealizedLoadings) -> VolFragTruth:
    """The ``VolFragTruth`` of a v4.1 config, with per-state ICs of s_real and s_volfrag.

    ``Corr(S_{t-1}, R_t | state_t)``: the state (h_M vs tau) is independent of the signals, so the
    covariance is ``(Sigma coef_k)_S`` and ``Var(R_t | state) = coef_k' Sigma coef_k +
    sigma_bar^2 + b^2 E[h_M | state]`` (the residual GARCH is independent of the market's).
    """
    volfrag = config.volfrag
    if volfrag is None or config.market is None or loadings.beta_volfrag is None:
        raise ValueError("volfrag_truth needs a config with a volfrag block")
    tau, h_low, h_high = volfrag_threshold(config.market.garch, volfrag.h_quantile)
    sigma = stationary_covariance(config)
    d = sigma.shape[0]
    mult = decay_multiplier(config)[config.burn_in : config.burn_in + config.n_obs - 1]
    decay_scale = float(mult.mean())
    ics: dict[str, tuple[list[float], list[float]]] = {"s_real": ([], []), VOLFRAG_SIGNAL: ([], [])}
    for i in range(config.n_assets):
        pairs = zip(volfrag.loading_multipliers, (h_low, h_high), strict=True)
        for k, (c_k, h_m) in enumerate(pairs):
            coef = np.zeros(d)
            coef[_Z] = loadings.beta_z[i]
            coef[_REAL] = loadings.beta_real[i]
            coef[_DECAY] = loadings.beta_decay_initial[i] * decay_scale
            coef[_REGIME] = loadings.beta_volfrag[i] * c_k
            var_r = (
                float(coef @ sigma @ coef)
                + config.garch.sigma_bar**2
                + loadings.market_beta[i] ** 2 * h_m
            )
            cov = sigma @ coef
            for sid, idx in (("s_real", _REAL), (VOLFRAG_SIGNAL, _REGIME)):
                ics[sid][k].append(float(cov[idx]) / math.sqrt(sigma[idx, idx] * var_r))
    return VolFragTruth(
        h_quantile=volfrag.h_quantile,
        tau=tau,
        n_mc=VOLFRAG_N_MC,
        mc_seed=VOLFRAG_MC_SEED,
        loading_multipliers=volfrag.loading_multipliers,
        market_variance=(h_low, h_high),
        state_ic=tuple(
            RegimeIC(signal_id=sid, calm=tuple(low), stress=tuple(high))
            for sid, (low, high) in ics.items()
        ),
    )


def _edges(config: SyntheticMarketConfig, loadings: RealizedLoadings) -> list[CausalEdge]:
    a_mat, _ = transition_matrices(config)
    lin, tv, var = Mechanism.LINEAR, Mechanism.LINEAR_TIME_VARYING, Mechanism.VARIANCE
    edges: list[CausalEdge] = []
    for k, name in enumerate(state_nodes(config)):
        if a_mat[k, k] != 0.0:
            edges.append(
                CausalEdge(source=name, target=name, lag=1, mechanism=lin, coefficient=a_mat[k, k])
            )
    for k in (_REAL, _CONF):
        if a_mat[k, _Z] != 0.0:
            edges.append(
                CausalEdge(
                    source="z",
                    target=STATE_NODES[k],
                    lag=1,
                    mechanism=lin,
                    coefficient=a_mat[k, _Z],
                ),
            )
    for i, asset in enumerate(asset_names(config.n_assets)):
        for source, coef, mech in (
            ("z", loadings.beta_z[i], lin),
            ("s_real", loadings.beta_real[i], lin),
            ("s_decay", loadings.beta_decay_initial[i], tv),
        ):
            if coef != 0.0:
                edges.append(
                    CausalEdge(source=source, target=asset, lag=1, mechanism=mech, coefficient=coef)
                )
        if config.garch.alpha > 0:
            edges.append(
                CausalEdge(
                    source=asset, target=asset, lag=1, mechanism=var, coefficient=config.garch.alpha
                ),
            )
        if config.market is not None and loadings.market_beta[i] != 0.0:
            edges.append(
                CausalEdge(
                    source=MARKET_NODE,
                    target=asset,
                    lag=0,
                    mechanism=lin,
                    coefficient=loadings.market_beta[i],
                ),
            )
    if config.market is not None and config.market.garch.alpha > 0:
        edges.append(
            CausalEdge(
                source=MARKET_NODE,
                target=MARKET_NODE,
                lag=1,
                mechanism=var,
                coefficient=config.market.garch.alpha,
            ),
        )
    if config.regime is not None:
        edges += _regime_edges(config, loadings)
    if config.volfrag is not None:
        assert loadings.beta_volfrag is not None
        # c(h_{M,t}) also depends on the market's past shocks, through h_M; the market's lag-1
        # variance edge and its lag-0 edge to every asset already cover that support.
        edges += [
            CausalEdge(
                source=VOLFRAG_SIGNAL,
                target=asset,
                lag=1,
                mechanism=Mechanism.LINEAR_VOL_SWITCHING,
                coefficient=loadings.beta_volfrag[i],
            )
            for i, asset in enumerate(asset_names(config.n_assets))
            if loadings.beta_volfrag[i] != 0.0
        ]
    return edges


def _regime_edges(config: SyntheticMarketConfig, loadings: RealizedLoadings) -> list[CausalEdge]:
    """v4 edges: S_regime's switching loading, the regime's variance edges and its chain."""
    regime = config.regime
    assert regime is not None and loadings.beta_regime is not None
    var = Mechanism.VARIANCE
    edges: list[CausalEdge] = []
    for i, asset in enumerate(asset_names(config.n_assets)):
        if loadings.beta_regime[i] != 0.0:
            edges.append(
                CausalEdge(
                    source=REGIME_SIGNAL,
                    target=asset,
                    lag=1,
                    mechanism=Mechanism.LINEAR_REGIME_SWITCHING,
                    coefficient=loadings.beta_regime[i],
                ),
            )
        # The regime also scales S_regime's loading (an interaction), which the variance edge
        # does not express; RegimeTruth.loading_multipliers records it.
        edges.append(
            CausalEdge(
                source=REGIME_NODE,
                target=asset,
                lag=0,
                mechanism=var,
                coefficient=regime.resid_vol_ratio,
            ),
        )
    if config.market is not None:
        edges.append(
            CausalEdge(
                source=REGIME_NODE,
                target=MARKET_NODE,
                lag=0,
                mechanism=var,
                coefficient=regime.market_vol_ratio,
            ),
        )
    edges.append(
        CausalEdge(
            source=REGIME_NODE,
            target=REGIME_NODE,
            lag=1,
            mechanism=Mechanism.MARKOV,
            coefficient=regime.transition_matrix[1][1],
        ),
    )
    return edges


def _nodes(config: SyntheticMarketConfig) -> list[Node]:
    nodes = [Node(name="z", role=NodeRole.CONFOUNDER, observed=config.observe_confounder)]
    nodes += [Node(name=s, role=NodeRole.CANDIDATE, observed=True) for s in signal_nodes(config)]
    nodes += [
        Node(name=a, role=NodeRole.TARGET, observed=True) for a in asset_names(config.n_assets)
    ]
    if config.market is not None:
        nodes.append(Node(name=MARKET_NODE, role=NodeRole.LATENT, observed=False))
    if config.regime is not None:
        nodes.append(Node(name=REGIME_NODE, role=NodeRole.LATENT, observed=False))
    return nodes


def _signal_truths(
    config: SyntheticMarketConfig,
    loadings: RealizedLoadings,
    index: pd.DatetimeIndex,
) -> list[SignalTruth]:
    moments = population_moments(config, loadings)
    root_n = math.sqrt(config.n_obs - 1)
    has_edge = {
        "s_real": config.real.ic > 0,
        "s_conf": False,
        "s_decay": config.decay.ic > 0,
        "s_noise": False,
        REGIME_SIGNAL: config.regime is not None and config.regime.signal.ic > 0,
        VOLFRAG_SIGNAL: config.volfrag is not None and config.volfrag.ic > 0,
    }
    confounded = config.factor.ic > 0 and config.conf.z_corr != 0.0
    notes = {
        "s_real": "True lag-1 cause with constant coefficient; correlated with Z by design.",
        "s_conf": "No edge to returns; correlated with forward returns only through Z.",
        "s_decay": "True lag-1 cause whose coefficient decays exponentially after decay_start.",
        "s_noise": "Persistent AR(1) independent of every other variable.",
        REGIME_SIGNAL: (
            "True lag-1 cause whose loading depends on the latent regime: calm value in calm, "
            "times stress_ratio in stress (reversed when negative). Its stationary-average edge "
            "is what Stages 1-3 see. Reported only: at this strength the latent regime is not "
            "identifiable from returns (ADR 0016), so no observable-state Stage 4 can recover it."
        ),
        VOLFRAG_SIGNAL: (
            "True lag-1 cause whose loading reverses when the market's conditional variance "
            "exceeds its h_quantile (momentum-crash pattern). Reported only: Stage 4 was evaluated "
            "and not adopted as a kill stage (ADR 0016); within Stages 1-3 its oracle verdict is "
            "pass (a true, average-positive edge)."
        ),
    }
    out = []
    for k, sid in enumerate(signal_nodes(config), start=1):
        identifiable = not (sid == "s_conf" and confounded and not config.observe_confounder)
        if sid in (REGIME_SIGNAL, VOLFRAG_SIGNAL) and has_edge[sid]:
            # ADR 0016: Stage 4 was not adopted as a kill stage, so neither v4 signal's expected
            # adversarial kill is gated; both are reported and excluded from the kill counts.
            identifiable = False
        note = notes[sid]
        if not identifiable and sid == "s_conf":
            note += (
                " Z is hidden: no observed adjustment set exists, so this non-edge cannot be "
                "distinguished from a direct effect (causal sufficiency violated)."
            )
        if not has_edge[sid]:
            verdict, stage = Verdict.KILL, StageName.CAUSAL
        elif sid == "s_decay":
            verdict, stage = Verdict.KILL, StageName.DECAY
        elif sid in (REGIME_SIGNAL, VOLFRAG_SIGNAL):
            verdict, stage = Verdict.KILL, StageName.ADVERSARIAL
        else:
            verdict, stage = Verdict.PASS, None
        decay_kwargs: dict[str, object] = {}
        if sid == "s_decay" and has_edge[sid]:
            mult = decay_multiplier(config)
            decay_kwargs = {
                "coef_half_life": config.decay.half_life,
                # TE ~ beta^2 Var(S) / (2 sigma^2) at low SNR, so it halves twice as fast.
                "te_half_life_approx": config.decay.half_life / 2.0,
                "decay_start": index[config.decay_start_index].date(),
                "final_coef_fraction": float(mult[-1]),
            }
        out.append(
            SignalTruth.model_validate(
                {
                    "signal_id": sid,
                    "has_causal_edge": has_edge[sid],
                    "expected_verdict": verdict,
                    "expected_kill_stage": stage,
                    "identifiable": identifiable,
                    "marginal_ic": tuple(float(m.marginal_ic[k]) for m in moments),
                    "partial_ic": tuple(float(m.partial_ic[k]) for m in moments),
                    "effective_partial_ic": tuple(
                        float(m.effective_partial_ic[k]) for m in moments
                    ),
                    "expected_t": tuple(float(m.effective_partial_ic[k]) * root_n for m in moments),
                    "expected_t_hybrid": tuple(
                        float(m.effective_hybrid_ic[k]) * root_n for m in moments
                    ),
                    "note": note,
                    **decay_kwargs,
                },
            ),
        )
    return out


def build_ground_truth(
    config: SyntheticMarketConfig,
    seed: int,
    loadings: RealizedLoadings,
    index: pd.DatetimeIndex,
    dataset_sha256: str,
) -> GroundTruth:
    """Assemble the ``GroundTruth`` for a dataset generated from ``config`` and ``seed``."""
    sigma = stationary_covariance(config)
    return GroundTruth(
        seed=seed,
        config=config,
        target_timing=EXPECTED_VALUES_TARGET_TIMING,
        nodes=tuple(_nodes(config)),
        edges=tuple(_edges(config, loadings)),
        signals=tuple(_signal_truths(config, loadings, index)),
        asset_names=asset_names(config.n_assets),
        loadings=loadings,
        state_order=state_nodes(config),
        stationary_covariance=tuple(tuple(float(v) for v in row) for row in sigma),
        first_date=index[0].date(),
        last_date=index[-1].date(),
        dataset_sha256=dataset_sha256,
        regime=regime_truth(config, loadings) if config.regime is not None else None,
        volfrag=volfrag_truth(config, loadings) if config.volfrag is not None else None,
    )


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------


def _content_hash(frames: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for name in sorted(frames):
        frame = frames[name]
        digest.update(name.encode())
        digest.update("\x1f".join(map(str, frame.columns)).encode())
        digest.update(frame.index.to_numpy(dtype="datetime64[ns]").view(np.int64).tobytes())
        digest.update(np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SyntheticDataset:
    """A generated market and its answer key.

    ``features``: candidate signals (and ``z`` when observed), dated by when they are known.
    ``returns``: one column per asset. ``latents``: oracle-only quantities (``z``,
    ``decay_multiplier``, ``h_<asset>``, and ``market``/``h_market`` in multi-asset mode). Stages
    must never read ``latents``; tests may. The frames are not deep-frozen; ``verify`` checks
    them against the hash recorded in ``ground_truth``.
    """

    features: pd.DataFrame
    returns: pd.DataFrame
    latents: pd.DataFrame
    ground_truth: GroundTruth

    def content_hash(self) -> str:
        return _content_hash(
            {"features": self.features, "returns": self.returns, "latents": self.latents},
        )

    def verify(self) -> None:
        """Raise ``ValueError`` if the frames no longer match ``ground_truth.dataset_sha256``."""
        actual = self.content_hash()
        if actual != self.ground_truth.dataset_sha256:
            raise ValueError(
                f"dataset content hash {actual} != recorded {self.ground_truth.dataset_sha256}",
            )


def generate_market(config: SyntheticMarketConfig, seed: int) -> SyntheticDataset:
    """Generate a synthetic market and its ``GroundTruth`` from ``config`` and ``seed``.

    Takes the integer seed rather than an ``RngFactory`` so the dataset is reproducible from
    ``(config, seed)`` alone, both of which are recorded in the ground truth.
    """
    rng = RngFactory(seed)
    loadings = draw_loadings(config, rng)
    paths = simulate(config, draw_innovations(config, rng), loadings)
    keep = slice(config.burn_in, None)
    index = pd.bdate_range(config.start_date, periods=config.n_obs)
    assets = asset_names(config.n_assets)

    nodes = state_nodes(config)
    feature_cols = list(nodes[1:]) + (["z"] if config.observe_confounder else [])
    feature_idx = [nodes.index(c) for c in feature_cols]
    features = pd.DataFrame(paths.state[keep][:, feature_idx], index=index, columns=feature_cols)
    returns = pd.DataFrame(paths.returns[keep], index=index, columns=list(assets))
    latent_data: dict[str, FloatArray] = {
        "z": paths.state[keep, _Z],
        "decay_multiplier": paths.decay_multiplier[keep],
    }
    for i, asset in enumerate(assets):
        latent_data[f"h_{asset}"] = paths.h[keep, i]
    if config.market is not None:
        latent_data[MARKET_NODE] = paths.market[keep]
        latent_data["h_market"] = paths.h_market[keep]
    if paths.regime is not None:
        latent_data[REGIME_NODE] = paths.regime[keep].astype(np.float64)
    latents = pd.DataFrame(latent_data, index=index)
    for frame in (features, returns, latents):
        validate_time_index(frame, Frequency.DAILY)

    sha = _content_hash({"features": features, "returns": returns, "latents": latents})
    truth = build_ground_truth(config, seed, loadings, index, sha)
    log = get_logger(__name__)
    for s in truth.signals:
        if s.has_causal_edge and min(s.expected_t_hybrid) < DETECTABILITY_T:
            log.warning(
                "benchmark_low_power",
                signal=s.signal_id,
                expected_t_hybrid=min(s.expected_t_hybrid),
                threshold=DETECTABILITY_T,
                detail="a correct Stage 1 may fail to detect this true edge at this sample size",
            )
    log.info("benchmark_generated", seed=seed, snapshot_id=truth.snapshot_id, n_obs=config.n_obs)
    return SyntheticDataset(features=features, returns=returns, latents=latents, ground_truth=truth)
