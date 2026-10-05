import copy
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from invariant.config.settings import (
    ConfigError,
    DateRange,
    PipelineConfig,
    UniverseConfig,
    deep_merge,
    load_config,
)
from invariant.core.types import Frequency

EXPECTED_TICKERS = (
    "SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "TLT", "IEF", "GLD",
)  # fmt: skip


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_base_config_loads() -> None:
    cfg = load_config()
    assert cfg.universe.tickers == EXPECTED_TICKERS
    assert cfg.dates == DateRange(start=date(2005, 1, 3), end=date(2025, 12, 31))
    assert cfg.frequency is Frequency.DAILY
    assert cfg.seed == 20260105


def test_override_merges_on_top(tmp_path: Path) -> None:
    override = _write(tmp_path / "o.yaml", "dates:\n  end: 2015-12-31\nseed: 7\n")
    cfg = load_config(override_path=override)
    assert cfg.dates.start == date(2005, 1, 3)  # kept from base
    assert cfg.dates.end == date(2015, 12, 31)
    assert cfg.seed == 7
    assert cfg.universe.tickers == EXPECTED_TICKERS


def test_override_list_replaces(tmp_path: Path) -> None:
    override = _write(tmp_path / "o.yaml", "universe:\n  tickers: [SPY, TLT]\n")
    assert load_config(override_path=override).universe.tickers == ("SPY", "TLT")


def test_unknown_key_rejected(tmp_path: Path) -> None:
    override = _write(tmp_path / "o.yaml", "sede: 3\n")
    with pytest.raises(ValidationError, match="sede"):
        load_config(override_path=override)


def test_unknown_nested_key_rejected(tmp_path: Path) -> None:
    override = _write(tmp_path / "o.yaml", "dates:\n  ends: 2015-01-01\n")
    with pytest.raises(ValidationError, match="ends"):
        load_config(override_path=override)


@pytest.mark.parametrize("end", ["2005-01-03", "2004-12-31"])
def test_start_not_before_end_rejected(tmp_path: Path, end: str) -> None:
    override = _write(tmp_path / "o.yaml", f"dates:\n  end: {end}\n")
    with pytest.raises(ValidationError, match="must be before"):
        load_config(override_path=override)


@pytest.mark.parametrize(
    ("tickers", "match"),
    [
        (("SPY", "SPY"), "duplicate"),
        (("spy",), "uppercase"),
        (("SP Y",), "uppercase"),
        ((), "at least 1"),
    ],
)
def test_bad_tickers_rejected(tickers: tuple[str, ...], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        UniverseConfig(tickers=tickers)


@pytest.mark.parametrize("seed", ["-1", "4294967296", "'42'", "true", "1.5"])
def test_bad_seed_rejected(tmp_path: Path, seed: str) -> None:
    override = _write(tmp_path / "o.yaml", f"seed: {seed}\n")
    with pytest.raises(ValidationError):
        load_config(override_path=override)


def test_bad_frequency_rejected(tmp_path: Path) -> None:
    override = _write(tmp_path / "o.yaml", "frequency: hourly\n")
    with pytest.raises(ValidationError):
        load_config(override_path=override)


@pytest.mark.parametrize("text", ["", "# only a comment\n", "- a\n- b\n", "42\n"])
def test_non_mapping_yaml_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path / "base.yaml", text))


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_config_frozen() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.seed = 1  # type: ignore[misc]
    assert isinstance(cfg, PipelineConfig)


# --- deep_merge properties -------------------------------------------------------------------

_keys = st.text(alphabet="abcde", min_size=1, max_size=3)
_leaves = st.one_of(
    st.none(), st.integers(), st.text(max_size=3), st.lists(st.integers(), max_size=3)
)
_trees = st.recursive(
    _leaves,
    lambda children: st.dictionaries(_keys, children, max_size=4),
    max_leaves=12,
)
_mappings = st.dictionaries(_keys, _trees, max_size=4)


@given(_mappings)
def test_merge_with_empty_is_identity(a: dict[str, object]) -> None:
    assert deep_merge(a, {}) == a
    assert deep_merge({}, a) == a


@given(_mappings, _mappings)
def test_merge_does_not_mutate_inputs(a: dict[str, object], b: dict[str, object]) -> None:
    a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
    result = deep_merge(a, b)
    assert (a, b) == (a0, b0)
    # The result shares no mutable objects with the inputs.
    _mutate_all(result)
    assert (a, b) == (a0, b0)


@given(_mappings, _mappings)
def test_override_wins_and_base_keys_kept(a: dict[str, object], b: dict[str, object]) -> None:
    result = deep_merge(a, b)
    assert set(result) == set(a) | set(b)
    for key, value in b.items():
        if not (isinstance(value, dict) and isinstance(a.get(key), dict)):
            assert result[key] == value
    for key in set(a) - set(b):
        assert result[key] == a[key]


@given(_mappings, _mappings)
def test_merge_idempotent(a: dict[str, object], b: dict[str, object]) -> None:
    once = deep_merge(a, b)
    assert deep_merge(once, b) == once


def _mutate_all(obj: object) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            _mutate_all(value)
        obj["__mutated__"] = True
    elif isinstance(obj, list):
        obj.append("__mutated__")
