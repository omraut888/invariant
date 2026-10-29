"""Politis-White (2004) / Patton-Politis-White (2009) block lengths (ADR 0016).

``arch.bootstrap.optimal_block_length`` (dev-only reference) differs in two conventions: it sets
m_hat to the first negligible lag (the paper: the lag *before* it), and it normalises the
autocorrelations by overlapping-segment variances. So the two agree exactly when the bandwidth
coincides (i.i.d. input), and closely otherwise. The closed-form AR(1) value checks the level.
"""

import math

import numpy as np
import pytest
from arch.bootstrap import optimal_block_length as arch_block_length

from invariant.stages.adversarial.blocklength import optimal_block_length, stage4_block_length


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    e = np.random.default_rng(seed).standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("seed", range(3))
def test_matches_arch_exactly_on_iid_input(seed: int) -> None:
    x = np.abs(np.random.default_rng(seed).standard_normal(5000))
    ref = arch_block_length(x).iloc[0]
    ours = optimal_block_length(x)
    assert ours.stationary == pytest.approx(ref["stationary"], rel=1e-10)
    assert ours.circular == pytest.approx(ref["circular"], rel=1e-10)


def test_close_to_arch_and_to_the_ar1_closed_form() -> None:
    phi, n = 0.7, 5000
    # b_CB = (3/2 (G/g)^2)^(1/3) n^(1/3) with G/g = 2 phi / (1 - phi^2) for an AR(1).
    theory = (1.5 * (2 * phi / (1 - phi**2)) ** 2) ** (1 / 3) * n ** (1 / 3)
    ours = [optimal_block_length(_ar1(phi, n, s)).circular for s in range(20)]
    refs = [arch_block_length(_ar1(phi, n, s)).iloc[0]["circular"] for s in range(20)]
    assert np.mean(ours) == pytest.approx(theory, rel=0.10)
    assert np.mean(ours) == pytest.approx(np.mean(refs), rel=0.10)


def test_cap_and_stage4_rule() -> None:
    x = np.cumsum(np.random.default_rng(1).standard_normal(400))  # a random walk
    b = optimal_block_length(x)
    assert b.circular <= math.ceil(min(3 * math.sqrt(400), 400 / 3))
    assert stage4_block_length(np.random.default_rng(2).standard_normal(3000)) >= 2


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        optimal_block_length(np.full(50, np.nan))
    with pytest.raises(ValueError, match="constant"):
        optimal_block_length(np.ones(50))
