"""The suite-wide Hypothesis profile (tests/conftest.py)."""

from hypothesis import settings


def test_the_invariant_profile_is_loaded_with_no_deadline() -> None:
    assert settings().deadline is None


def test_per_test_settings_inherit_the_profile() -> None:
    """A per-test ``@settings(max_examples=...)`` builds on the loaded profile."""
    assert settings(max_examples=5).deadline is None
