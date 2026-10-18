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

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import linalg, signal

from invariant.benchmark.config import GarchParams, SyntheticMarketConfig, signal_coefficients
from invariant.benchmark.ground_truth import (
    EXPECTED_VALUES_TARGET_TIMING,
    CausalEdge,
    GroundTruth,
    Mechanism,
    Node,
    NodeRole,
    RealizedLoadings,
    SignalTruth,
)
from invariant.core.logging import get_logger
from invariant.core.run_hash import RngFactory
from invariant.core.types import Frequency, StageName, Verdict, validate_time_index

__all__ = [
    "DETECTABILITY_T",
    "SIGNAL_NODES",
    "STATE_NODES",
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
    "simulate",
    "stationary_covariance",
    "transition_matrices",
]

FloatArray = npt.NDArray[np.float64]

STATE_NODES: tuple[str, ...] = ("z", "s_real", "s_conf", "s_decay", "s_noise")
SIGNAL_NODES: tuple[str, ...] = STATE_NODES[1:]
_Z, _REAL, _CONF, _DECAY, _NOISE = range(5)
MARKET_NODE = "market"

DETECTABILITY_T = 3.0
"""A true signal whose expected Stage-1 t-stat is below this triggers a low-power warning."""


def asset_names(n_assets: int) -> tuple[str, ...]:
    return tuple(f"asset_{i:02d}" for i in range(n_assets))


# ---------------------------------------------------------------------------------------------
# Population structure
# ---------------------------------------------------------------------------------------------


def transition_matrices(config: SyntheticMarketConfig) -> tuple[FloatArray, FloatArray]:
    """``(A, B)`` of the state VAR(1) ``X_t = A X_{t-1} + B nu_t``, ``nu ~ N(0, I)``."""
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
    return a, b


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


# ---------------------------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------------------------


def draw_loadings(config: SyntheticMarketConfig, rng: RngFactory) -> RealizedLoadings:
    """Per-asset loadings ``ic * sd_i * exp(delta xi - delta^2 / 2)``.

    ``sd_i = sqrt(sigma_bar^2 + b_i^2 sigma_bar_M^2)`` is the unconditional sd of the asset's
    non-signal return (residual plus market), so a nominal ``ic`` means the same in single- and
    multi-asset mode.
    """
    n = config.n_assets
    delta = config.loading_dispersion
    if config.market is None:
        market_beta = np.zeros(n)
        market_var = 0.0
    else:
        m = config.market
        market_beta = rng.generator("loadings/market").uniform(m.beta_low, m.beta_high, n)
        market_var = m.garch.sigma_bar**2
    noise_sd = np.sqrt(config.garch.sigma_bar**2 + market_beta**2 * market_var)
    xi = rng.generator("loadings/signal").standard_normal((n, 3))
    mult = np.exp(delta * xi - delta**2 / 2.0)
    nominal_ic = np.array([config.factor.ic, config.real.ic, config.decay.ic])
    betas = mult * nominal_ic * noise_sd[:, None]
    return RealizedLoadings(
        beta_z=tuple(float(v) for v in betas[:, 0]),
        beta_real=tuple(float(v) for v in betas[:, 1]),
        beta_decay_initial=tuple(float(v) for v in betas[:, 2]),
        market_beta=tuple(float(v) for v in market_beta),
    )


@dataclass(frozen=True)
class Innovations:
    """All random inputs of one simulation. ``simulate`` is a pure function of these.

    ``state[s]`` holds the standard-normal innovations ``(eta, u_real, u_conf, u_decay, u_noise)``
    of step s (row 0 unused: the initial state is ``x0``). ``eps`` / ``eps_market`` are
    unit-variance Student-t residual innovations.
    """

    x0: FloatArray
    state: FloatArray
    eps: FloatArray
    eps_market: FloatArray | None


def _standardized_t(gen: np.random.Generator, nu: float, n: int) -> FloatArray:
    return np.asarray(gen.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu), dtype=np.float64)


def draw_innovations(config: SyntheticMarketConfig, rng: RngFactory) -> Innovations:
    """Draw every innovation from its own named stream of ``rng`` (see module docstring)."""
    n = config.n_total
    chol = np.linalg.cholesky(stationary_covariance(config))
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
    return Innovations(x0=x0, state=state, eps=eps, eps_market=eps_market)


# ---------------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatedPaths:
    """Full simulated paths, burn-in included. Row s is step s."""

    state: FloatArray  # (n_total, 5), columns STATE_NODES
    returns: FloatArray  # (n_total, n_assets)
    h: FloatArray  # (n_total, n_assets) conditional variance of the residual
    market: FloatArray  # (n_total,), zeros without a market factor
    h_market: FloatArray  # (n_total,), zeros without a market factor
    decay_multiplier: FloatArray  # (n_total,), indexed by signal date


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


def simulate(
    config: SyntheticMarketConfig,
    innovations: Innovations,
    loadings: RealizedLoadings,
) -> SimulatedPaths:
    """Run the SCM recursions. Pure and causal: step s uses only innovations at steps <= s."""
    n = config.n_total
    if innovations.state.shape != (n, 5) or innovations.eps.shape != (n, config.n_assets):
        raise ValueError("innovations do not match the config's dimensions")
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


# ---------------------------------------------------------------------------------------------
# Population moments and ground truth
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _AssetMoments:
    marginal_ic: FloatArray  # (5,) Corr(X_k,t-1, R_t), initial coefficients
    partial_ic: FloatArray  # (5,) given other observed parents, initial coefficients
    effective_partial_ic: FloatArray  # (5,) with S_decay at its average coefficient
    effective_hybrid_ic: FloatArray  # (5,) given Stage 1's oracle hybrid set, S_decay averaged


def _observed_parents(config: SyntheticMarketConfig) -> list[int]:
    parents = []
    if config.observe_confounder and config.factor.ic > 0:
        parents.append(_Z)
    if config.real.ic > 0:
        parents.append(_REAL)
    if config.decay.ic > 0:
        parents.append(_DECAY)
    return parents


def _partial_corr(cov: FloatArray, i: int, j: int, cond: list[int]) -> float:
    idx = [i, j, *cond]
    precision = np.linalg.inv(cov[np.ix_(idx, idx)])
    return float(-precision[0, 1] / math.sqrt(precision[0, 0] * precision[1, 1]))


def _joint_cov(
    config: SyntheticMarketConfig,
    sigma: FloatArray,
    coef: FloatArray,
    market_beta: float,
) -> FloatArray:
    """Stationary covariance of ``(X_{t-1}, R_t)`` for loading vector ``coef`` on ``X_{t-1}``."""
    var_resid = config.garch.sigma_bar**2
    if config.market is not None:
        var_resid += market_beta**2 * config.market.garch.sigma_bar**2
    joint = np.zeros((6, 6))
    joint[:5, :5] = sigma
    cov_xr = sigma @ coef
    joint[:5, 5] = cov_xr
    joint[5, :5] = cov_xr
    joint[5, 5] = float(coef @ sigma @ coef) + var_resid
    return joint


def _joint_cov_two_lags(
    config: SyntheticMarketConfig,
    sigma: FloatArray,
    coef: FloatArray,
    market_beta: float,
) -> FloatArray:
    """Stationary covariance of ``(X_{t-1}, X_{t-2}, R_t)`` (indices 0-4, 5-9, 10).

    ``X_{t-1} = A X_{t-2} + B e_{t-1}`` gives ``Cov(X_{t-1}, X_{t-2}) = A Sigma`` and
    ``Cov(X_{t-2}, R_t) = (A Sigma)' coef``.
    """
    a, _ = transition_matrices(config)
    one_lag = _joint_cov(config, sigma, coef, market_beta)
    cross = a @ sigma
    joint = np.zeros((11, 11))
    joint[:5, :5] = sigma
    joint[5:10, 5:10] = sigma
    joint[:5, 5:10] = cross
    joint[5:10, :5] = cross.T
    joint[:5, 10] = joint[10, :5] = one_lag[:5, 5]
    joint[5:10, 10] = joint[10, 5:10] = cross.T @ coef
    joint[10, 10] = one_lag[5, 5]
    return joint


def hybrid_oracle_set(config: SyntheticMarketConfig, k: int) -> list[int]:
    """Stage 1's ``hybrid`` conditioning set for ``X^k_{t-1} -> R_t`` with the true parents in
    place of PC1's, as indices into ``(X_{t-1}, X_{t-2})``: the observed parents of ``R`` other
    than ``X^k_{t-1}``, plus the observed parents of ``X^k`` other than its own lag, shifted by
    one lag. In this SCM the only such parent is ``Z_{t-1}`` (so ``Z_{t-2}``), for a signal
    that loads on Z, when Z is observed.
    """
    a, _ = transition_matrices(config)
    cond = [p for p in _observed_parents(config) if p != k]
    if k != _Z and config.observe_confounder and a[k, _Z] != 0.0:
        cond.append(5 + _Z)
    return cond


def population_moments(
    config: SyntheticMarketConfig,
    loadings: RealizedLoadings,
) -> list[_AssetMoments]:
    """Exact population ICs per asset for every state variable (see ``SignalTruth``).

    The effective coefficient of S_decay is its average multiplier over the ``n_obs - 1`` signal
    dates that have a next-day return; treating it as constant is a stationary approximation of
    the full-sample regression slope. Return lags are not conditioned on: given the parents,
    they carry no linear information about ``R_t`` (GARCH affects only the variance).
    """
    sigma = stationary_covariance(config)
    mult = decay_multiplier(config)[config.burn_in : config.burn_in + config.n_obs - 1]
    decay_scale = float(mult.mean())
    parents = _observed_parents(config)
    out = []
    for i in range(config.n_assets):
        coef = np.zeros(5)
        coef[_Z] = loadings.beta_z[i]
        coef[_REAL] = loadings.beta_real[i]
        coef[_DECAY] = loadings.beta_decay_initial[i]
        coef_eff = coef.copy()
        coef_eff[_DECAY] *= decay_scale
        mb = loadings.market_beta[i]
        joint = _joint_cov(config, sigma, coef, mb)
        joint_eff = _joint_cov(config, sigma, coef_eff, mb)
        marginal = joint[:5, 5] / math.sqrt(joint[5, 5])
        partial = np.array(
            [_partial_corr(joint, k, 5, [p for p in parents if p != k]) for k in range(5)]
        )
        partial_eff = np.array(
            [_partial_corr(joint_eff, k, 5, [p for p in parents if p != k]) for k in range(5)],
        )
        joint2_eff = _joint_cov_two_lags(config, sigma, coef_eff, mb)
        hybrid_eff = np.array(
            [_partial_corr(joint2_eff, k, 10, hybrid_oracle_set(config, k)) for k in range(5)],
        )
        out.append(_AssetMoments(marginal, partial, partial_eff, hybrid_eff))
    return out


def _edges(config: SyntheticMarketConfig, loadings: RealizedLoadings) -> list[CausalEdge]:
    a_mat, _ = transition_matrices(config)
    lin, tv, var = Mechanism.LINEAR, Mechanism.LINEAR_TIME_VARYING, Mechanism.VARIANCE
    edges: list[CausalEdge] = []
    for k, name in enumerate(STATE_NODES):
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
    return edges


def _nodes(config: SyntheticMarketConfig) -> list[Node]:
    nodes = [Node(name="z", role=NodeRole.CONFOUNDER, observed=config.observe_confounder)]
    nodes += [Node(name=s, role=NodeRole.CANDIDATE, observed=True) for s in SIGNAL_NODES]
    nodes += [
        Node(name=a, role=NodeRole.TARGET, observed=True) for a in asset_names(config.n_assets)
    ]
    if config.market is not None:
        nodes.append(Node(name=MARKET_NODE, role=NodeRole.LATENT, observed=False))
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
    }
    confounded = config.factor.ic > 0 and config.conf.z_corr != 0.0
    notes = {
        "s_real": "True lag-1 cause with constant coefficient; correlated with Z by design.",
        "s_conf": "No edge to returns; correlated with forward returns only through Z.",
        "s_decay": "True lag-1 cause whose coefficient decays exponentially after decay_start.",
        "s_noise": "Persistent AR(1) independent of every other variable.",
    }
    out = []
    for k, sid in enumerate(SIGNAL_NODES, start=1):
        identifiable = not (sid == "s_conf" and confounded and not config.observe_confounder)
        note = notes[sid]
        if not identifiable:
            note += (
                " Z is hidden: no observed adjustment set exists, so this non-edge cannot be "
                "distinguished from a direct effect (causal sufficiency violated)."
            )
        if not has_edge[sid]:
            verdict, stage = Verdict.KILL, StageName.CAUSAL
        elif sid == "s_decay":
            verdict, stage = Verdict.KILL, StageName.DECAY
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
        state_order=STATE_NODES,
        stationary_covariance=tuple(tuple(float(v) for v in row) for row in sigma),
        first_date=index[0].date(),
        last_date=index[-1].date(),
        dataset_sha256=dataset_sha256,
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

    feature_cols = list(SIGNAL_NODES) + (["z"] if config.observe_confounder else [])
    feature_idx = [STATE_NODES.index(c) for c in feature_cols]
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
