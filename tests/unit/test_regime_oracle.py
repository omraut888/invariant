import math

import numpy as np
import pytest

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.regime_oracle import _projection, oracle_regime_sharpe

C = 0.5 / math.sqrt(252.0)


def test_projection_recovers_a_planted_linear_slope() -> None:
    gen = np.random.default_rng(0)
    s = gen.standard_normal(50_000)
    r = 0.1 + np.outer(s, [0.5, -0.2]) + 0.01 * gen.standard_normal((50_000, 2))
    a, b, cov = _projection(s, r)
    np.testing.assert_allclose(a, [0.1, 0.1], atol=1e-3)
    np.testing.assert_allclose(b, [0.5, -0.2], atol=1e-3)
    np.testing.assert_allclose(np.diag(cov), [1e-4, 1e-4], rtol=0.05)


def test_oracle_sees_the_planted_reversal() -> None:
    cfg = load_benchmark_config("regime").model_copy(update={"n_obs": 60_000})
    reg = oracle_regime_sharpe(cfg, "s_regime", train_seed=1, test_seed=2, c=C)
    real = oracle_regime_sharpe(cfg, "s_real", train_seed=1, test_seed=2, c=C)
    assert reg.calm > 0 > reg.stress
    assert real.stress > 0
    assert 0 < reg.stress_share_test < 1


def test_oracle_needs_a_regime() -> None:
    with pytest.raises(ValueError, match="regime block"):
        oracle_regime_sharpe(load_benchmark_config(), "s_real", train_seed=1, test_seed=2, c=C)
