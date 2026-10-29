"""Paired stylized-facts comparison of two benchmark presets (ADR 0015).

Same seeds 0-4 and the same assets, so each (seed, asset, check) is a matched pair of pass/fail
outcomes. "Preset A is not significantly worse than B" is the exact one-sided McNemar test: of
the discordant pairs, b fail only under A and c fail only under B; under H0 (no difference),
b ~ Binomial(b + c, 1/2), and the p-value is P(X >= b). It passes at alpha = .05 iff p >= .05.
"""

import numpy as np
from scipy import stats

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.stylized_facts import check_stylized_facts
from invariant.benchmark.synthetic import generate_market

SEEDS = range(5)


def failed_checks(preset: str) -> np.ndarray:
    """Boolean vector of failures over (seed, asset, check), in a fixed order."""
    out: list[bool] = []
    for seed in SEEDS:
        d = generate_market(load_benchmark_config(preset), seed)
        for asset in d.returns.columns:
            out += [not c.passed for c in check_stylized_facts(d.returns[asset]).checks]
    return np.array(out)


def mcnemar_worse(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """``(b_only_a, c_only_b, one-sided exact p)`` for "a fails more often than b"."""
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5, alternative="greater").pvalue)
    return only_a, only_b, p
