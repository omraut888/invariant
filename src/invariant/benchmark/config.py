"""Configuration of the planted-signal synthetic market.

The structural causal model these parameters drive is specified in ``synthetic.py`` and in
docs/adr/0003-planted-signal-benchmark-scm.md. This module only holds the typed, frozen,
validated parameters and the YAML loader.

Signal strength is given as a *nominal* information coefficient ``ic``: the return loading is
``beta = ic * sd_i``, where ``sd_i = sqrt(garch.sigma_bar^2 + b_i^2 market.garch.sigma_bar^2)`` is
the unconditional sd of asset i's non-signal return, and every state variable has unit variance.
So ``ic`` is the correlation the signal would have with returns if it were the only predictor.
The exact population ICs (which also account for the other predictors and for correlation
through Z) are computed and exported in ``GroundTruth``.

Validators reject parameter sets for which the model is not what it claims to be: a
non-stationary or ill-posed signal process (non-positive innovation variance), a GARCH process
without a finite unconditional variance, or one without a finite fourth moment (for which
excess kurtosis, one of the stylized facts we check, is infinite).

YAML layout (``configs/benchmark.yaml``): a ``base`` mapping holding a full config and a
``presets`` mapping of named overrides, each deep-merged onto ``base`` with the same rules as the
pipeline config (``invariant.config.settings.deep_merge``).
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from invariant.config.settings import ConfigError, deep_merge, find_project_root

__all__ = [
    "DEFAULT_BENCHMARK_CONFIG",
    "ConfoundedSignalParams",
    "DecaySignalParams",
    "FactorParams",
    "GarchParams",
    "MarketParams",
    "NoiseSignalParams",
    "RealSignalParams",
    "SyntheticMarketConfig",
    "list_benchmark_presets",
    "load_benchmark_config",
    "signal_coefficients",
]

DEFAULT_BENCHMARK_CONFIG = Path("configs/benchmark.yaml")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def signal_coefficients(rho: float, z_corr: float, phi: float) -> tuple[float, float]:
    """Return ``(gamma, sigma2)`` for ``S_t = rho S_{t-1} + gamma Z_{t-1} + sqrt(sigma2) u_t``.

    Chosen so that, with ``Z`` a unit-variance AR(1) with coefficient ``phi``, the stationary
    ``Var(S) = 1`` and ``Corr(S_t, Z_t) = z_corr``:

        gamma  = z_corr (1 - rho phi) / phi
        sigma2 = 1 - rho^2 - gamma^2 - 2 rho gamma z_corr

    Derivation: stationarity of ``Cov(S_t, Z_t) = rho phi C + gamma phi`` gives ``C = z_corr``;
    stationarity of ``Var(S_t) = rho^2 + gamma^2 + 2 rho gamma C + sigma2`` gives ``sigma2``.
    ``sigma2 <= 0`` means no such process exists; callers must reject it.
    """
    gamma = z_corr * (1.0 - rho * phi) / phi
    sigma2 = 1.0 - rho**2 - gamma**2 - 2.0 * rho * gamma * z_corr
    return gamma, sigma2


class FactorParams(_Frozen):
    """Persistent macro factor ``Z_t = phi Z_{t-1} + sqrt(1 - phi^2) eta_t`` (unit variance).

    ``ic`` is the nominal IC of ``Z_{t-1}`` for ``R_t``. ``phi > 0`` is required because the
    signal loadings on ``Z`` are parameterised through ``Corr(S, Z)`` (see
    ``signal_coefficients``), which divides by ``phi``.
    """

    phi: float = Field(gt=0.0, lt=1.0)
    ic: float = Field(ge=0.0, lt=1.0)


class RealSignalParams(_Frozen):
    """``S_real``: true lag-1 cause of returns; optionally correlated with ``Z``."""

    rho: float = Field(ge=0.0, lt=1.0)
    z_corr: float = Field(gt=-1.0, lt=1.0)
    ic: float = Field(ge=0.0, lt=1.0)


class ConfoundedSignalParams(_Frozen):
    """``S_conf``: driven by ``Z``, no edge to returns."""

    rho: float = Field(ge=0.0, lt=1.0)
    z_corr: float = Field(gt=-1.0, lt=1.0)


class DecaySignalParams(_Frozen):
    """``S_decay``: true cause whose coefficient halves every ``half_life`` trading days.

    The decay starts at output-sample index ``floor(start_frac * n_obs)``.
    """

    rho: float = Field(ge=0.0, lt=1.0)
    ic: float = Field(ge=0.0, lt=1.0)
    start_frac: float = Field(gt=0.0, lt=1.0)
    half_life: float = Field(gt=0.0)


class NoiseSignalParams(_Frozen):
    """``S_noise``: persistent AR(1), independent of everything."""

    rho: float = Field(ge=0.0, lt=1.0)


class GarchParams(_Frozen):
    """GARCH(1,1) with unit-variance Student-t innovations (Bollerslev 1986, 1987).

    ``h_t = omega + alpha a_{t-1}^2 + beta h_{t-1}`` with ``omega = sigma_bar^2 (1 - alpha -
    beta)``, so ``sigma_bar`` is the unconditional standard deviation of ``a_t``.

    The fourth moment of ``a_t`` is finite iff ``beta^2 + 2 alpha beta + k alpha^2 < 1``, where
    ``k = 3 (nu - 2) / (nu - 4)`` is the kurtosis of the standardized t innovation (He &
    Terasvirta 1999, "Properties of moments of a family of GARCH processes", J. Econometrics).
    This is enforced: otherwise population excess kurtosis is infinite.
    """

    nu: float = Field(gt=4.0)
    alpha: float = Field(ge=0.0, lt=1.0)
    beta: float = Field(ge=0.0, lt=1.0)
    sigma_bar: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_moments(self) -> Self:
        if self.alpha + self.beta >= 1.0:
            raise ValueError(
                f"alpha + beta = {self.alpha + self.beta} must be < 1 for a finite "
                "unconditional variance",
            )
        if self.fourth_moment_condition >= 1.0:
            raise ValueError(
                f"beta^2 + 2 alpha beta + k alpha^2 = {self.fourth_moment_condition:.4f} must be "
                f"< 1 (k = {self.innovation_kurtosis:.3f} for nu = {self.nu}); otherwise the "
                "fourth moment of returns is infinite",
            )
        return self

    @property
    def omega(self) -> float:
        return self.sigma_bar**2 * (1.0 - self.alpha - self.beta)

    @property
    def innovation_kurtosis(self) -> float:
        """Kurtosis (not excess) of the unit-variance Student-t innovation."""
        return 3.0 * (self.nu - 2.0) / (self.nu - 4.0)

    @property
    def fourth_moment_condition(self) -> float:
        k = self.innovation_kurtosis
        return self.beta**2 + 2.0 * self.alpha * self.beta + k * self.alpha**2

    @property
    def residual_kurtosis(self) -> float:
        """Population kurtosis (not excess) of ``a_t``.

        ``k (1 - (alpha + beta)^2) / (1 - beta^2 - 2 alpha beta - k alpha^2)``.
        """
        k = self.innovation_kurtosis
        persistence = self.alpha + self.beta
        return k * (1.0 - persistence**2) / (1.0 - self.fourth_moment_condition)


class MarketParams(_Frozen):
    """Latent common market factor ``M_t`` (its own GARCH-t) with betas ~ U[beta_low, beta_high]."""

    garch: GarchParams
    beta_low: float
    beta_high: float

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.beta_low > self.beta_high:
            raise ValueError(f"beta_low {self.beta_low} must be <= beta_high {self.beta_high}")
        return self


class SyntheticMarketConfig(_Frozen):
    """All generation parameters of the planted-signal market. Hashed into ``GroundTruth``.

    ``n_obs`` counts output observations; ``burn_in`` extra steps are simulated first and
    discarded. The state starts at its stationary distribution, so ``burn_in`` is a safety margin,
    not a requirement; it must be >= 1 because the first simulated return has no lagged state.
    ``loading_dispersion`` (delta) scales per-asset loadings by ``exp(delta xi - delta^2 / 2)``,
    ``xi ~ N(0, 1)``: mean-one, sign-preserving, exactly 1 when ``delta = 0``.
    """

    n_obs: StrictInt = Field(ge=50)
    burn_in: StrictInt = Field(ge=1)
    start_date: date
    n_assets: StrictInt = Field(ge=1, le=100)
    observe_confounder: bool = True
    mu: float = 0.0
    factor: FactorParams
    real: RealSignalParams
    conf: ConfoundedSignalParams
    decay: DecaySignalParams
    noise: NoiseSignalParams
    garch: GarchParams
    market: MarketParams | None = None
    loading_dispersion: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        if self.start_date.weekday() >= 5:
            raise ValueError(f"start_date {self.start_date} must be a weekday")
        for name, rho, z_corr in (
            ("real", self.real.rho, self.real.z_corr),
            ("conf", self.conf.rho, self.conf.z_corr),
        ):
            _, sigma2 = signal_coefficients(rho, z_corr, self.factor.phi)
            if sigma2 <= 1e-8:
                raise ValueError(
                    f"{name}: rho={rho}, z_corr={z_corr}, phi={self.factor.phi} imply innovation "
                    f"variance {sigma2:.3g} <= 0; no unit-variance process has this persistence "
                    "and correlation with Z",
                )
        start = self.decay_start_index
        if not 1 <= start < self.n_obs - 1:
            raise ValueError(
                f"decay start index {start} must lie strictly inside the sample "
                f"[1, {self.n_obs - 1})",
            )
        return self

    @property
    def n_total(self) -> int:
        """Simulated steps including burn-in."""
        return self.burn_in + self.n_obs

    @property
    def decay_start_index(self) -> int:
        """Output-sample index t0 of the first signal date with a decayed coefficient."""
        return math.floor(self.decay.start_frac * self.n_obs)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else find_project_root() / path


def _read_benchmark_yaml(path: Path | None) -> tuple[dict[str, object], dict[str, object]]:
    resolved = _resolve(path or DEFAULT_BENCHMARK_CONFIG)
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"benchmark config not found: {resolved}") from None
    if not isinstance(data, dict) or not isinstance(data.get("base"), dict):
        raise ConfigError(f"{resolved} must be a mapping with a 'base' mapping")
    presets = data.get("presets", {})
    if not isinstance(presets, dict) or not all(isinstance(v, dict) for v in presets.values()):
        raise ConfigError(f"{resolved}: 'presets' must map names to override mappings")
    extra = set(data) - {"base", "presets"}
    if extra:
        raise ConfigError(f"{resolved}: unknown top-level keys {sorted(extra)}")
    return data["base"], presets


def list_benchmark_presets(path: Path | None = None) -> tuple[str, ...]:
    """Names of the presets defined in the benchmark YAML, sorted."""
    _, presets = _read_benchmark_yaml(path)
    return tuple(sorted(presets))


def load_benchmark_config(
    preset: str | None = None,
    path: Path | None = None,
) -> SyntheticMarketConfig:
    """Load ``base`` from ``path`` (default ``configs/benchmark.yaml``), apply ``preset``, validate.

    Relative paths resolve against the project root. An unknown preset raises ``ConfigError``.
    """
    base, presets = _read_benchmark_yaml(path)
    merged: dict[str, object] = base
    if preset is not None:
        if preset not in presets:
            raise ConfigError(f"unknown benchmark preset {preset!r}; known: {sorted(presets)}")
        override = presets[preset]
        assert isinstance(override, dict)  # checked in _read_benchmark_yaml
        merged = deep_merge(base, override)
    return SyntheticMarketConfig.model_validate(merged)
