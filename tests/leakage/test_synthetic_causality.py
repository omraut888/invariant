"""The synthetic generator is causal: no value dated t depends on randomness after t.

If it were not, a stage could "detect" planted signals through look-ahead built into the
benchmark itself. The test redraws every innovation after a cut point and requires every path
up to and including the cut to be bit-identical.
"""

from dataclasses import fields, replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import (
    Innovations,
    SimulatedPaths,
    draw_innovations,
    draw_loadings,
    generate_market,
    simulate,
)
from invariant.core.run_hash import RngFactory

pytestmark = pytest.mark.leakage

CFG = load_benchmark_config("multi_asset").model_copy(update={"n_obs": 120, "burn_in": 5})


def _redraw_after(innov: Innovations, cut: int, seed: int) -> Innovations:
    gen = np.random.default_rng(seed)
    state = innov.state.copy()
    state[cut + 1 :] = gen.standard_normal(state[cut + 1 :].shape)
    eps = innov.eps.copy()
    eps[cut + 1 :] = gen.standard_t(6, eps[cut + 1 :].shape)
    assert innov.eps_market is not None
    eps_m = innov.eps_market.copy()
    eps_m[cut + 1 :] = gen.standard_t(6, eps_m[cut + 1 :].shape)
    return replace(innov, state=state, eps=eps, eps_market=eps_m)


@settings(max_examples=40, deadline=None)
@given(cut=st.integers(0, CFG.n_total - 2), seed=st.integers(0, 2**32 - 1))
def test_paths_up_to_t_ignore_innovations_after_t(cut: int, seed: int) -> None:
    rng = RngFactory(17)
    loadings = draw_loadings(CFG, rng)
    innov = draw_innovations(CFG, rng)
    a = simulate(CFG, innov, loadings)
    b = simulate(CFG, _redraw_after(innov, cut, seed), loadings)
    for f in fields(SimulatedPaths):
        before, after = getattr(a, f.name), getattr(b, f.name)
        np.testing.assert_array_equal(before[: cut + 1], after[: cut + 1], err_msg=f.name)
    # The redraw actually changed the future (otherwise the test is vacuous).
    assert not np.array_equal(a.returns[cut + 1 :], b.returns[cut + 1 :])


def test_features_dated_t_are_uncorrelated_with_future_residual_shocks() -> None:
    # Population check of the timing convention on a long sample: a feature dated t must carry
    # no information about the residual innovation of R_{t+1} (only about its mean).
    cfg = load_benchmark_config().model_copy(update={"n_obs": 50_000})
    d = generate_market(cfg, 3)
    r_next = d.returns["asset_00"].to_numpy()[1:]
    f = d.features.to_numpy()[:-1]
    gt = d.ground_truth
    loadings = gt.loadings
    mean_next = (
        loadings.beta_z[0] * d.latents["z"].to_numpy()[:-1]
        + loadings.beta_real[0] * d.features["s_real"].to_numpy()[:-1]
        + loadings.beta_decay_initial[0]
        * d.latents["decay_multiplier"].to_numpy()[:-1]
        * d.features["s_decay"].to_numpy()[:-1]
    )
    shock = r_next - cfg.mu - mean_next
    tol = 4 * 1.5 / np.sqrt(len(shock))  # SE ~ 1/sqrt(n), x1.5 for GARCH, 4 SE
    for k in range(f.shape[1]):
        assert abs(np.corrcoef(f[:, k], shock)[0, 1]) < tol, d.features.columns[k]
