"""G0 (ADR 0016): at the fitted chain theta_hat, generated paths keep the stylized facts.

Re-specified (ADR 0016 deviation): the observed series themselves fail the fixed thresholds on
6 of 75 checks. So G0 is the exact one-sided McNemar test that generated paths are not
significantly worse than the observed series. Each (seed, asset, check) outcome is paired:
`volfrag` seeds 0-4, every asset, one path at theta_hat of the observed length.
"""

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.paired_facts import mcnemar_worse

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.stylized_facts import check_stylized_facts
from invariant.benchmark.synthetic import generate_market
from invariant.stages.adversarial.blocklength import stage4_block_length
from invariant.stages.adversarial.generator import BlockPool, Start
from invariant.stages.adversarial.state import volatility_state

pytestmark = pytest.mark.benchmark


def _outcomes() -> tuple[np.ndarray, np.ndarray]:
    gen_fail: list[bool] = []
    obs_fail: list[bool] = []
    for seed in range(5):
        d = generate_market(load_benchmark_config("volfrag"), seed)
        r = d.returns.to_numpy()
        dates = np.arange(1260, len(r) - 1, dtype=np.int64)
        states, _ = volatility_state(r.mean(axis=1), dates)
        rows = r[dates + 1]
        pool = BlockPool.from_states(states, stage4_block_length(rows.mean(axis=1)))
        idx = pool.generate(
            pool.chain_hat, Start.STATIONARY, len(rows), np.random.default_rng(1000 + seed)
        )
        index = pd.bdate_range("2000-01-03", periods=len(idx))
        for i in range(rows.shape[1]):
            gen = check_stylized_facts(pd.Series(rows[idx, i], index=index))
            obs = check_stylized_facts(pd.Series(rows[:, i], index=index))
            gen_fail += [not c.passed for c in gen.checks]
            obs_fail += [not c.passed for c in obs.checks]
    return np.array(gen_fail), np.array(obs_fail)


def test_generated_paths_are_not_significantly_worse_than_observed() -> None:
    gen_fail, obs_fail = _outcomes()
    only_gen, only_obs, p = mcnemar_worse(gen_fail, obs_fail)
    assert p >= 0.05, (int(gen_fail.sum()), int(obs_fail.sum()), only_gen, only_obs, p)
