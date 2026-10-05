import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.synthetic import (
    STATE_NODES,
    SyntheticDataset,
    decay_multiplier,
    draw_innovations,
    draw_loadings,
    generate_market,
    simulate,
    stationary_covariance,
    transition_matrices,
)
from invariant.core.run_hash import RngFactory
from invariant.core.types import Frequency, validate_time_index


def _small(preset: str | None = None, **updates: object) -> SyntheticMarketConfig:
    cfg = load_benchmark_config(preset)
    return cfg.model_copy(update={"n_obs": 400, "burn_in": 50, **updates})


def _gen(cfg: SyntheticMarketConfig, seed: int = 7) -> SyntheticDataset:
    return generate_market(cfg, seed)


# --- determinism and reproducibility ----------------------------------------------------------


def test_same_seed_is_bit_identical() -> None:
    a, b = _gen(_small()), _gen(_small())
    pd.testing.assert_frame_equal(a.features, b.features, check_exact=True)
    pd.testing.assert_frame_equal(a.returns, b.returns, check_exact=True)
    pd.testing.assert_frame_equal(a.latents, b.latents, check_exact=True)
    assert a.ground_truth == b.ground_truth
    assert a.ground_truth.dataset_sha256 == b.ground_truth.dataset_sha256


def test_different_seed_differs() -> None:
    a, b = _gen(_small(), 1), _gen(_small(), 2)
    assert not np.allclose(a.returns.to_numpy(), b.returns.to_numpy())
    assert a.ground_truth.snapshot_id != b.ground_truth.snapshot_id


def test_golden_values_pin_the_generator() -> None:
    # Pins the stream derivation + recursions. Changing these is a deliberate, versioned decision.
    d = _gen(_small(), 20260105)
    assert d.returns.iloc[-1, 0] == GOLDEN_LAST_RETURN
    assert d.ground_truth.dataset_sha256[:16] == GOLDEN_SHA_PREFIX


GOLDEN_LAST_RETURN = 0.011999587638460206
GOLDEN_SHA_PREFIX = "7e96e91da373bbea"


def test_verify_detects_tampering() -> None:
    d = _gen(_small())
    d.verify()
    values = d.returns.to_numpy(copy=True)
    d.returns.iloc[5, 0] = values[5, 0] + 1e-9
    with pytest.raises(ValueError, match="content hash"):
        d.verify()


# --- common random numbers --------------------------------------------------------------------


def test_changing_one_ic_leaves_other_paths_unchanged() -> None:
    base = _small()
    other = base.model_copy(update={"real": base.real.model_copy(update={"ic": 0.2})})
    a, b = _gen(base), _gen(other)
    pd.testing.assert_frame_equal(a.features, b.features, check_exact=True)
    pd.testing.assert_series_equal(
        a.latents["h_asset_00"], b.latents["h_asset_00"], check_exact=True
    )
    assert not np.allclose(a.returns.to_numpy(), b.returns.to_numpy())


def test_adding_assets_keeps_existing_asset_paths() -> None:
    cfg = _small("multi_asset")
    a = _gen(cfg)
    b = _gen(cfg.model_copy(update={"n_assets": 7}))
    pd.testing.assert_frame_equal(a.returns, b.returns[a.returns.columns], check_exact=True)
    assert a.ground_truth.loadings.beta_real == b.ground_truth.loadings.beta_real[:5]


def test_hidden_confounder_only_changes_observed_columns() -> None:
    a = _gen(_small())
    b = _gen(_small("hidden_confounder"))
    assert "z" in a.features.columns
    assert "z" not in b.features.columns
    pd.testing.assert_frame_equal(a.features.drop(columns="z"), b.features, check_exact=True)
    pd.testing.assert_frame_equal(a.returns, b.returns, check_exact=True)
    pd.testing.assert_series_equal(a.features["z"], b.latents["z"], check_names=False)


# --- output contract --------------------------------------------------------------------------


def test_frames_satisfy_time_index_contract_and_shapes() -> None:
    d = _gen(_small("multi_asset"))
    for frame in (d.features, d.returns, d.latents):
        validate_time_index(frame, Frequency.DAILY)
        assert len(frame) == 400
        assert np.isfinite(frame.to_numpy()).all()
    assert list(d.features.columns) == ["s_real", "s_conf", "s_decay", "s_noise", "z"]
    assert list(d.returns.columns) == [f"asset_{i:02d}" for i in range(5)]
    assert {"market", "h_market", "decay_multiplier", "z"} <= set(d.latents.columns)
    assert d.returns.index[0] == pd.Timestamp("2000-01-03")
    assert (d.latents.filter(like="h_") > 0).all().all()


def test_ground_truth_dates_and_snapshot() -> None:
    d = _gen(_small())
    gt = d.ground_truth
    assert gt.first_date == d.returns.index[0].date()
    assert gt.last_date == d.returns.index[-1].date()
    assert gt.snapshot_id == f"synthetic-{d.content_hash()[:16]}"


def test_low_power_warning_on_realistic_preset() -> None:
    with capture_logs() as logs:
        _gen(load_benchmark_config("realistic").model_copy(update={"n_obs": 1000}))
    warned = {e["signal"] for e in logs if e["event"] == "benchmark_low_power"}
    assert warned == {"s_real", "s_decay"}


# --- decay schedule ---------------------------------------------------------------------------


def test_decay_multiplier_halves_exactly_at_half_life() -> None:
    cfg = load_benchmark_config()
    m = decay_multiplier(cfg)[cfg.burn_in :]
    t0, h = cfg.decay_start_index, int(cfg.decay.half_life)
    assert np.all(m[:t0] == 1.0)
    assert m[t0] == 1.0
    assert m[t0 + h] == pytest.approx(0.5, rel=1e-12)
    assert m[t0 + 2 * h] == pytest.approx(0.25, rel=1e-12)
    assert np.all(np.diff(m[t0:]) < 0)


def test_decay_multiplier_exported_in_latents() -> None:
    cfg = _small()
    d = _gen(cfg)
    np.testing.assert_array_equal(
        d.latents["decay_multiplier"].to_numpy(), decay_multiplier(cfg)[cfg.burn_in :]
    )
    s = d.ground_truth.signal("s_decay")
    assert s.decay_start == d.returns.index[cfg.decay_start_index].date()


# --- population structure ---------------------------------------------------------------------


def test_stationary_covariance_has_unit_diagonal_and_target_z_corr() -> None:
    cfg = load_benchmark_config()
    sigma = stationary_covariance(cfg)
    np.testing.assert_allclose(np.diag(sigma), 1.0, atol=1e-10)
    z = STATE_NODES.index("z")
    assert sigma[z, STATE_NODES.index("s_real")] == pytest.approx(cfg.real.z_corr)
    assert sigma[z, STATE_NODES.index("s_conf")] == pytest.approx(cfg.conf.z_corr)
    assert sigma[z, STATE_NODES.index("s_noise")] == pytest.approx(0.0, abs=1e-12)
    assert sigma[z, STATE_NODES.index("s_decay")] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(sigma) > 0, True)


def test_transition_matrix_is_stable() -> None:
    a, _ = transition_matrices(load_benchmark_config())
    assert np.max(np.abs(np.linalg.eigvals(a))) < 1.0


def test_initial_state_is_drawn_from_stationary_distribution() -> None:
    # Across seeds, the step-0 state must have covariance Sigma (no burn-in needed).
    cfg = _small()
    sigma = stationary_covariance(cfg)
    x0 = np.array([draw_innovations(cfg, RngFactory(s)).x0 for s in range(3000)])
    cov = np.cov(x0, rowvar=False)
    # SE of a sample covariance entry at n=3000 is <= sqrt(2/3000) ~ 0.026.
    np.testing.assert_allclose(cov, sigma, atol=0.1)


def test_garch_starts_at_unconditional_variance() -> None:
    cfg = _small()
    rng = RngFactory(3)
    loadings = draw_loadings(cfg, rng)
    paths = simulate(cfg, draw_innovations(cfg, rng), loadings)
    assert paths.h[0, 0] == cfg.garch.sigma_bar**2


def test_simulate_rejects_mismatched_innovations() -> None:
    cfg = _small()
    rng = RngFactory(3)
    loadings = draw_loadings(cfg, rng)
    innov = draw_innovations(cfg, rng)
    with pytest.raises(ValueError, match="dimensions"):
        simulate(cfg.model_copy(update={"n_obs": 401}), innov, loadings)
    with pytest.raises(ValueError, match="eps_market"):
        simulate(_small("multi_asset").model_copy(update={"n_assets": 1}), innov, loadings)


def test_loading_dispersion_is_sign_preserving_and_zero_means_nominal() -> None:
    cfg = _small()
    flat = draw_loadings(cfg, RngFactory(1))
    assert flat.beta_real == (cfg.real.ic * cfg.garch.sigma_bar,)
    disp = draw_loadings(_small("multi_asset"), RngFactory(1))
    assert all(b > 0 for b in disp.beta_real + disp.beta_z + disp.beta_decay_initial)
    assert len(set(disp.beta_real)) == 5
