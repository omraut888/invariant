"""structlog configuration.

Two modes:

- ``LogMode.DEV``: human-readable console output for interactive work.
- ``LogMode.RUN``: one JSON object per line, for pipeline runs whose logs are archived next to
  the run artefacts and parsed later. Keys are sorted so identical events serialise identically.

The caller picks the mode explicitly; nothing is inferred from the TTY. ``run_id`` is bound via
contextvars so every event in the run carries it.
"""

from __future__ import annotations

import logging
import sys
from enum import StrEnum
from typing import TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor

__all__ = ["LogMode", "configure_logging", "get_logger"]


class LogMode(StrEnum):
    DEV = "dev"
    RUN = "run"


def _level_number(level: str) -> int:
    mapping = logging.getLevelNamesMapping()
    try:
        return mapping[level.upper()]
    except KeyError:
        raise ValueError(f"unknown log level {level!r}") from None


def configure_logging(
    mode: LogMode,
    *,
    level: str = "INFO",
    stream: TextIO | None = None,
    run_id: str | None = None,
) -> None:
    """Configure structlog globally. Safe to call repeatedly; the latest call wins.

    ``stream`` defaults to ``sys.stderr`` (resolved at call time so test capture works).
    """
    out = stream if stream is not None else sys.stderr
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    processors: list[Processor]
    if mode is LogMode.RUN:
        processors = [
            *shared,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=True, default=str),
        ]
    else:
        is_tty = getattr(out, "isatty", lambda: False)()
        processors = [*shared, structlog.dev.ConsoleRenderer(colors=bool(is_tty))]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_level_number(level)),
        logger_factory=structlog.PrintLoggerFactory(file=out),
        cache_logger_on_first_use=False,
    )
    structlog.contextvars.clear_contextvars()
    if run_id is not None:
        structlog.contextvars.bind_contextvars(run_id=run_id)


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a logger whose events carry ``logger=name``."""
    # cast: structlog.get_logger is typed as returning Any; with our configure() it is a
    # FilteringBoundLogger.
    return cast(FilteringBoundLogger, structlog.get_logger()).bind(logger=name)
