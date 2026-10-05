import io
import json
from collections.abc import Iterator

import pytest
import structlog

from invariant.core.logging import LogMode, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def _lines(buf: io.StringIO) -> list[str]:
    return [line for line in buf.getvalue().splitlines() if line]


def test_run_mode_emits_json_lines() -> None:
    buf = io.StringIO()
    configure_logging(LogMode.RUN, stream=buf, run_id="abc123")
    log = get_logger("test")
    log.info("first", n=1)
    log.warning("second")

    records = [json.loads(line) for line in _lines(buf)]
    assert [r["event"] for r in records] == ["first", "second"]
    assert [r["level"] for r in records] == ["info", "warning"]
    for r in records:
        assert r["run_id"] == "abc123"
        assert r["logger"] == "test"
        assert r["timestamp"].endswith("Z")
    assert records[0]["n"] == 1


def test_run_mode_serialises_exceptions_and_unknown_types() -> None:
    buf = io.StringIO()
    configure_logging(LogMode.RUN, stream=buf)
    log = get_logger("test")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed", obj=object())
    (record,) = [json.loads(line) for line in _lines(buf)]
    assert "RuntimeError: boom" in record["exception"]
    assert "run_id" not in record


def test_dev_mode_is_human_readable() -> None:
    buf = io.StringIO()
    configure_logging(LogMode.DEV, stream=buf)
    get_logger("test").info("hello world", key="value")
    out = buf.getvalue()
    assert "hello world" in out
    assert "key=value" in out
    assert "\x1b[" not in out  # no ANSI colours on a non-TTY stream
    with pytest.raises(json.JSONDecodeError):
        json.loads(_lines(buf)[0])


def test_level_filter() -> None:
    buf = io.StringIO()
    configure_logging(LogMode.RUN, stream=buf, level="warning")
    log = get_logger("test")
    log.debug("d")
    log.info("i")
    log.warning("w")
    assert [json.loads(line)["event"] for line in _lines(buf)] == ["w"]


def test_unknown_level_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(LogMode.DEV, level="LOUD")


def test_reconfigure_switches_mode_and_run_id() -> None:
    dev_buf, run_buf = io.StringIO(), io.StringIO()
    configure_logging(LogMode.DEV, stream=dev_buf, run_id="old")
    configure_logging(LogMode.RUN, stream=run_buf, run_id="new")
    get_logger("test").info("after")
    assert dev_buf.getvalue() == ""
    (record,) = [json.loads(line) for line in _lines(run_buf)]
    assert record["run_id"] == "new"
