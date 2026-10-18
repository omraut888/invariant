from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statsmodels.stats.multitest import multipletests

from invariant.stages.causal.multiple_testing import (
    FdrMethod,
    bonferroni_family_p,
    fdr_qvalues,
)

_P = st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=30)


def test_bonferroni_family_p() -> None:
    assert bonferroni_family_p([0.01, 0.2, 0.5]) == pytest.approx(0.03)
    assert bonferroni_family_p([0.4, 0.9]) == 0.8
    assert bonferroni_family_p([0.6, 0.7]) == 1.0


@pytest.mark.parametrize("bad", [[], [np.nan], [-0.1], [1.1]])
def test_invalid_p_values_rejected(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        bonferroni_family_p(bad)
    with pytest.raises(ValueError):
        fdr_qvalues(bad, "bh")


@given(_P)
def test_qvalues_match_statsmodels(p: list[float]) -> None:
    methods: tuple[tuple[FdrMethod, str], ...] = (("bh", "fdr_bh"), ("by", "fdr_by"))
    for method, sm in methods:
        want = np.asarray(multipletests(p, method=sm)[1], dtype=float)
        np.testing.assert_allclose(fdr_qvalues(p, method), want, rtol=1e-12, atol=1e-15)


@given(_P)
def test_qvalue_invariants(p: list[float]) -> None:
    arr = np.asarray(p)
    bh, by = fdr_qvalues(arr, "bh"), fdr_qvalues(arr, "by")
    assert ((bh >= arr - 1e-15) & (bh <= 1.0)).all()
    assert (by >= bh - 1e-15).all()
    order = np.argsort(arr, kind="stable")
    assert (np.diff(bh[order]) >= -1e-15).all()  # monotone in p
    perm = np.random.default_rng(0).permutation(arr.size)
    np.testing.assert_array_equal(fdr_qvalues(arr[perm], "bh"), bh[perm])


@given(_P, st.floats(0.001, 0.2))
def test_bh_rejections_equal_step_up_rule(p: list[float], alpha: float) -> None:
    arr = np.sort(np.asarray(p))
    m = arr.size
    passing = [k for k in range(1, m + 1) if arr[k - 1] <= alpha * k / m]
    n_reject = max(passing, default=0)
    q = fdr_qvalues(arr, "bh")
    # Exact boundary ties can differ by rounding; the rule is checked away from them.
    if not np.any(np.isclose(arr * m / np.arange(1, m + 1), alpha, rtol=1e-9)):
        assert int((q <= alpha).sum()) == n_reject


def test_single_hypothesis_is_unadjusted() -> None:
    assert fdr_qvalues([0.03], "bh")[0] == 0.03
    assert fdr_qvalues([0.03], "by")[0] == 0.03
