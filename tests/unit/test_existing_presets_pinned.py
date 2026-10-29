"""Pins every pre-v4 benchmark preset's data, config dump and answer key (ADR 0015).

Benchmark v4 adds an opt-in regime block. These values were computed on the generator before
that change (commit 140804f); every validated Stage 1-3 result depends on them, so they must not
move. A failure here means an existing preset changed, which ADR 0015 rules out.
"""

import hashlib

import pytest

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.core.run_hash import canonical_json

# (preset, seed) -> (dataset sha256[:16], config dump [:16], answer key [:16])
PINNED: dict[tuple[str | None, int], tuple[str, str, str]] = {
    (None, 0): ("5883fe692abdff28", "fae6bf0461cd09af", "460127bf6e193dc9"),
    (None, 20261029): ("97fbd4d351cdf12a", "fae6bf0461cd09af", "460127bf6e193dc9"),
    ("hidden_confounder", 0): ("cdbaf055bcef1395", "007e2a863b8ed61a", "3751dd92b2855401"),
    ("hidden_confounder", 20261029): (
        "b1f0dc0b8492d03f",
        "007e2a863b8ed61a",
        "3751dd92b2855401",
    ),
    ("multi_asset", 0): ("ff22cb626602e676", "389553a487b7f7ef", "92b9ace57cb1d62e"),
    ("multi_asset", 20261029): ("d166b7ded43767ed", "389553a487b7f7ef", "ec646cdbb9008063"),
    ("realistic", 0): ("4efb8b18c9f5048c", "a1cdffaf251c11d0", "775eab8583b72cb5"),
    ("realistic", 20261029): ("f1392904aab8bfbc", "a1cdffaf251c11d0", "775eab8583b72cb5"),
}


def _short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize(("preset", "seed"), list(PINNED))
def test_existing_preset_is_unchanged(preset: str | None, seed: int) -> None:
    cfg = load_benchmark_config(preset)
    truth = generate_market(cfg, seed).ground_truth
    answer_key = canonical_json(
        [truth.signals, truth.edges, truth.nodes, truth.stationary_covariance, truth.loadings],
    )
    actual = (truth.dataset_sha256[:16], _short(canonical_json(cfg)), _short(answer_key))
    assert actual == PINNED[(preset, seed)]
