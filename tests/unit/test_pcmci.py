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


# -- decision tests ---------------------------------------------------------------------------

_PARENTS: dict[int, list[tuple[int, int]]] = {
    0: [(0, -1), (2, -1)],  # X: own lag and a "confounder" 2
    1: [(1, -2), (0, -1), (2, -1)],  # Y
    2: [(2, -1)],
}


def test_conditioning_sets() -> None:
    from invariant.stages.causal.pcmci import DecisionTest, conditioning_set

    # X_{t-1} -> Y_t: drop X_{t-1} from P(Y); shift P(X) by 1.
    assert conditioning_set(DecisionTest.PARENTS_R, _PARENTS, 0, 1, 1) == [(1, -2), (2, -1)]
    assert conditioning_set(DecisionTest.MCI, _PARENTS, 0, 1, 1) == [
        (1, -2),
        (2, -1),
        (0, -2),
        (2, -2),
    ]
    # hybrid: P(X) minus X's own lags.
    assert conditioning_set(DecisionTest.HYBRID, _PARENTS, 0, 1, 1) == [(1, -2), (2, -1), (2, -2)]
    # A lag that is not a parent keeps P(Y) whole.
    assert conditioning_set(DecisionTest.PARENTS_R, _PARENTS, 0, 1, 3) == _PARENTS[1]


@pytest.mark.parametrize("name", list(CITestName))
def test_mci_decision_reproduces_tigramite_p_matrix(name: CITestName) -> None:
    from invariant.stages.causal.pcmci import DecisionTest

    data = _var(2, 1500)
    out = run_pcmci(
        data,
        ["x", "y", "w"],
        make_ci_test(name, seed=0),
        tau_max=3,
        pc_alpha=0.2,
        decision_links=[("x", "y"), ("w", "y"), ("y", "y")],
        decisions=list(DecisionTest),
    )
    for src in ("x", "w", "y"):
        np.testing.assert_array_equal(
            out.decision_p[(DecisionTest.MCI, src, "y")], out.link_p(src, "y", [1, 2, 3])
        )
        np.testing.assert_array_equal(
            out.decision_val[(DecisionTest.MCI, src, "y")], out.link_val(src, "y", [1, 2, 3])
        )
    assert ("x", 2) in out.parents["y"]


def test_parents_r_regains_power_on_a_persistent_cause() -> None:
    """MCI tests only X's innovation; parents_r uses all of X's variation (ADR 0009, finding 4)."""
    from invariant.stages.causal.pcmci import DecisionTest

    rng = np.random.default_rng(5)
    t = 20_000  # se of a partial correlation ~ 0.007
    e = rng.standard_normal((t, 2))
    x, y = np.zeros(t), np.zeros(t)
    for s in range(1, t):
        x[s] = 0.95 * x[s - 1] + e[s, 0]
        y[s] = 0.1 * x[s - 1] + e[s, 1]
    out = run_pcmci(
        np.column_stack([x, y]),
        ["x", "y"],
        make_ci_test(CITestName.PARCORR, seed=0),
        tau_max=1,
        pc_alpha=0.2,
        decision_links=[("x", "y")],
        decisions=list(DecisionTest),
    )
    mci = abs(out.decision_val[(DecisionTest.MCI, "x", "y")][0])
    pr = abs(out.decision_val[(DecisionTest.PARENTS_R, "x", "y")][0])
    # Population (simulated at T = 400k): parents_r 0.297, MCI 0.101 = 0.1 sd(u) / sd(y).
    assert pr == pytest.approx(0.297, abs=0.03)
    assert mci == pytest.approx(0.101, abs=0.03)


@pytest.mark.parametrize("decision", ["hybrid", "mci"])
def test_lagged_design_reproduces_decision_statistic(decision: str) -> None:
    from invariant.stages.causal.bootstrap import partial_corr
    from invariant.stages.causal.ci_tests import hc3_p_value
    from invariant.stages.causal.pcmci import DecisionTest, lagged_design

    dt = DecisionTest(decision)
    data = _var(4, 1200)
    data[:, 2] += 0.6 * np.roll(data[:, 0], 1)  # give w a parent so the sets are non-trivial
    names = ["x", "y", "w"]
    out = run_pcmci(
        data,
        names,
        make_ci_test(CITestName.PARCORR_HC3, seed=0),
        tau_max=3,
        pc_alpha=0.2,
        decision_links=[("x", "y"), ("w", "y")],
        decisions=[dt],
    )
    for src in ("x", "w"):
        for lag in (1, 2, 3):
            cond = out.conditions(dt, src, "y", lag)
            d = lagged_design(data, names, src, "y", lag, cond, tau_max=3)
            assert d.x.shape == (1200 - 6,)
            r = partial_corr(d.x, d.y, d.z)
            assert r == pytest.approx(out.decision_val[(dt, src, "y")][lag - 1], abs=1e-10)
            p = hc3_p_value(np.vstack([d.x, d.y, d.z.T]))
            assert p == pytest.approx(out.decision_p[(dt, src, "y")][lag - 1], rel=1e-8)
