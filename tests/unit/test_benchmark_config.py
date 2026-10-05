from datetime import date
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError
from scipy import linalg

from invariant.benchmark.config import (
    GarchParams,
    SyntheticMarketConfig,
    list_benchmark_presets,
    load_benchmark_config,
    signal_coefficients,
)
from invariant.config.settings import ConfigError


def _base() -> SyntheticMarketConfig:
    return load_benchmark_config()


def _with(**updates: object) -> SyntheticMarketConfig:
    data = _base().model_dump()
    for dotted, value in updates.items():
        node = data
        *path, last = dotted.split("__")
        for key in path:
            node = node[key]
        node[last] = value
    return SyntheticMarketConfig.model_validate(data)


# --- loading ---------------------------------------------------------------------------------


def test_base_config_loads_with_documented_defaults() -> None:
    cfg = _base()
    assert cfg.n_obs == 6000
    assert cfg.real.z_corr == 0.3
    assert (cfg.factor.ic, cfg.real.ic, cfg.decay.ic) == (0.06, 0.06, 0.08)
    assert cfg.market is None
    assert cfg.observe_confounder


def test_presets_are_listed_and_load() -> None:
    names = list_benchmark_presets()
    assert {"realistic", "hidden_confounder", "multi_asset"} <= set(names)
    for name in names:
        load_benchmark_config(name)


def test_realistic_preset_is_a_named_sweep_point_at_ic_002() -> None:
    cfg = load_benchmark_config("realistic")
    assert (cfg.factor.ic, cfg.real.ic, cfg.decay.ic) == (0.02, 0.02, 0.02)
    # Everything else is inherited from base.
    assert (
        cfg.model_copy(
            update={"factor": _base().factor, "real": _base().real, "decay": _base().decay}
        )
        == _base()
    )


def test_multi_asset_preset_deep_merges_garch() -> None:
    cfg = load_benchmark_config("multi_asset")
    assert cfg.n_assets == 5
    assert cfg.market is not None
    assert cfg.garch.sigma_bar == 0.008
    assert cfg.garch.nu == _base().garch.nu  # untouched keys survive the merge


def test_unknown_preset_raises() -> None:
    with pytest.raises(ConfigError, match="unknown benchmark preset"):
        load_benchmark_config("nope")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_benchmark_config(path=tmp_path / "missing.yaml")


def test_malformed_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "b.yaml"
    bad.write_text("presets: {}\n")
    with pytest.raises(ConfigError, match="'base'"):
        load_benchmark_config(path=bad)
    bad.write_text("base: {}\nextra: 1\n")
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_benchmark_config(path=bad)


# --- validators -------------------------------------------------------------------------------


def test_extra_keys_rejected() -> None:
    data = _base().model_dump()
    data["real"]["typo"] = 1
    with pytest.raises(ValidationError):
        SyntheticMarketConfig.model_validate(data)


def test_garch_infinite_fourth_moment_rejected() -> None:
    # nu=5, alpha=.08, beta=.90: 0.81 + 0.144 + 9 * 0.0064 = 1.0116 >= 1.
    with pytest.raises(ValidationError, match="fourth moment"):
        GarchParams(nu=5, alpha=0.08, beta=0.90, sigma_bar=0.01)


def test_garch_nonstationary_rejected() -> None:
    with pytest.raises(ValidationError, match="alpha \\+ beta"):
        GarchParams(nu=30, alpha=0.1, beta=0.9, sigma_bar=0.01)


def test_garch_nu_must_exceed_four() -> None:
    with pytest.raises(ValidationError):
        GarchParams(nu=4, alpha=0.05, beta=0.9, sigma_bar=0.01)


def test_default_garch_moments() -> None:
    g = _base().garch
    assert g.innovation_kurtosis == pytest.approx(6.0)
    assert g.fourth_moment_condition == pytest.approx(0.9784)
    assert g.residual_kurtosis - 3.0 == pytest.approx(6 * 0.0396 / 0.0216 - 3.0)
    assert g.omega == pytest.approx(1e-4 * 0.02)


def test_impossible_signal_persistence_rejected() -> None:
    # rho=.99 with Corr(S, Z)=.9 and phi=.99 has no unit-variance solution.
    with pytest.raises(ValidationError, match="innovation"):
        _with(conf__rho=0.99, conf__z_corr=0.9)


def test_weekend_start_rejected() -> None:
    with pytest.raises(ValidationError, match="weekday"):
        _with(start_date=date(2000, 1, 1))


def test_decay_start_must_be_inside_sample() -> None:
    with pytest.raises(ValidationError, match="decay start"):
        _with(n_obs=50, decay__start_frac=0.001)


def test_burn_in_at_least_one() -> None:
    with pytest.raises(ValidationError):
        _with(burn_in=0)


def test_market_beta_range_ordered() -> None:
    data = load_benchmark_config("multi_asset").model_dump()
    data["market"]["beta_low"] = 2.0
    with pytest.raises(ValidationError, match="beta_low"):
        SyntheticMarketConfig.model_validate(data)


def test_decay_start_index() -> None:
    assert _base().decay_start_index == 3600
    assert _base().n_total == 7000


# --- signal_coefficients: the unit-variance parameterisation is exact -------------------------


@given(
    phi=st.floats(0.05, 0.995),
    rho=st.floats(0.0, 0.98),
    z_corr=st.floats(-0.95, 0.95),
)
def test_signal_coefficients_give_unit_variance_and_target_corr(
    phi: float, rho: float, z_corr: float
) -> None:
    gamma, sigma2 = signal_coefficients(rho, z_corr, phi)
    assume(sigma2 > 1e-3)
    a = np.array([[phi, 0.0], [gamma, rho]])
    b = np.diag([np.sqrt(1 - phi**2), np.sqrt(sigma2)])
    cov = linalg.solve_discrete_lyapunov(a, b @ b.T)
    np.testing.assert_allclose(np.diag(cov), [1.0, 1.0], rtol=1e-7, atol=1e-9)
    assert cov[0, 1] == pytest.approx(z_corr, abs=1e-7)


def test_zero_z_corr_gives_plain_ar1() -> None:
    gamma, sigma2 = signal_coefficients(0.9, 0.0, 0.99)
    assert gamma == 0.0
    assert sigma2 == pytest.approx(1 - 0.81)
