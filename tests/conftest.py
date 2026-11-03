"""Suite-wide test configuration.

Hypothesis: every property test here checks correctness, not speed, so the suite runs under one
profile with no per-example deadline. Hypothesis's default 200 ms deadline turned ordinary
examples into failures on a loaded machine (two suites sharing the CPU). Per-test ``@settings``
inherit this profile, so they set only what differs (``max_examples``, health checks).
"""

from hypothesis import settings

settings.register_profile("invariant", deadline=None)
settings.load_profile("invariant")
