import numpy as np
import pytest

from invariant.stages.causal.ci_tests import CITestName, make_ci_test
from invariant.stages.causal.pcmci import run_pcmci


def _var(seed: int, t: int = 2000) -> np.ndarray:
    """x -> y at lag 2 with coefficient 0.5; w independent."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((t, 3))
    x, y, w = np.zeros(t), np.zeros(t), np.zeros(t)
    for s in range(2, t):
        x[s] = 0.5 * x[s - 1] + e[s, 0]
        y[s] = 0.3 * y[s - 1] + 0.5 * x[s - 2] + e[s, 1]
        w[s] = 0.5 * w[s - 1] + e[s, 2]
    return np.column_stack([x, y, w])


def test_recovers_planted_lagged_link() -> None:
    out = run_pcmci(
        _var(0), ["x", "y", "w"], make_ci_test(CITestName.PARCORR, seed=0), tau_max=3, pc_alpha=0.2
    )
    assert out.p.shape == (3, 3, 4) and out.val.shape == (3, 3, 4)
    assert np.isnan(out.p[:, :, 0]).all()
    assert out.link_p("x", "y", [2])[0] < 1e-10
    assert out.link_val("x", "y", [2])[0] > 0.3
    assert out.link_p("w", "y", [1, 2, 3]).min() > 1e-3
    assert out.n_obs == 2000


@pytest.mark.parametrize(
    ("data", "names", "kwargs", "match"),
    [
        (np.zeros((10, 2)), ["a"], {}, "does not match"),
        (np.zeros((10, 2)), ["a", "a"], {}, "unique"),
        (np.full((10, 2), np.nan), ["a", "b"], {}, "finite"),
        (np.zeros((10, 2)), ["a", "b"], {"tau_max": 0}, "tau_max"),
        (np.zeros((10, 2)), ["a", "b"], {"pc_alpha": 1.0}, "pc_alpha"),
    ],
)
def test_input_validation(
    data: np.ndarray, names: list[str], kwargs: dict[str, float], match: str
) -> None:
    args: dict[str, float] = {"tau_max": 2, "pc_alpha": 0.2, **kwargs}
    with pytest.raises(ValueError, match=match):
        run_pcmci(
            data,
            names,
            make_ci_test(CITestName.PARCORR, seed=0),
            tau_max=int(args["tau_max"]),
            pc_alpha=args["pc_alpha"],
        )


def test_link_lags_validated() -> None:
    out = run_pcmci(
        _var(1, 300),
        ["x", "y", "w"],
        make_ci_test(CITestName.PARCORR, seed=0),
        tau_max=2,
        pc_alpha=0.2,
    )
    with pytest.raises(ValueError, match="lags"):
        out.link_p("x", "y", [0])
    with pytest.raises(ValueError, match="lags"):
        out.link_p("x", "y", [3])
