import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from invariant.config.settings import HttpConfig
from invariant.data.sources.clients import (
    ALFRED_REALTIME_END,
    ALFRED_REALTIME_START,
    AlfredFetchError,
    FredAlfredClient,
    HttpError,
    HttpxClient,
)

CONFIG = HttpConfig(timeout_s=5, max_retries=2, backoff_s=1.0)
KEY = "s3cr3tkey0123456789"
URL = "https://api.example.test/fred/series/observations"

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, sleeps: list[float] | None = None) -> HttpxClient:
    record = sleeps if sleeps is not None else []
    return HttpxClient(CONFIG, transport=httpx.MockTransport(handler), sleep=record.append)


# --- HttpxClient ------------------------------------------------------------------------------


def test_success_returns_body() -> None:
    client = _client(lambda req: httpx.Response(200, content=b"payload"))
    assert client.get_bytes("https://x.test/a", {"q": "1"}) == b"payload"


def test_retries_5xx_with_exponential_backoff() -> None:
    responses = iter([503, 502, 200])
    sleeps: list[float] = []
    client = _client(lambda req: httpx.Response(next(responses), content=b"ok"), sleeps)
    assert client.get_bytes("https://x.test/a") == b"ok"
    assert sleeps == [1.0, 2.0]


def test_retries_transport_errors() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, content=b"ok")

    assert _client(handler).get_bytes("https://x.test/a") == b"ok"
    assert calls["n"] == 2


def test_client_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="nope")

    with pytest.raises(HttpError, match="HTTP 404") as info:
        _client(handler).get_bytes("https://x.test/a")
    assert info.value.status == 404
    assert calls["n"] == 1


def test_exhausted_retries_raise_without_query_string() -> None:
    sleeps: list[float] = []
    client = _client(lambda req: httpx.Response(500, text="boom"), sleeps)
    with pytest.raises(HttpError, match="3 attempt") as info:
        client.get_bytes("https://x.test/a", {"api_key": KEY})
    assert KEY not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__context__ is None
    assert sleeps == [1.0, 2.0]


def test_transport_error_exhaustion_does_not_chain_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(HttpError, match="ConnectError") as info:
        _client(handler).get_bytes("https://x.test/a", {"api_key": KEY})
    assert info.value.status is None
    assert KEY not in str(info.value)
    assert info.value.__context__ is None  # the httpx exception's URL carries the key


# --- FredAlfredClient -------------------------------------------------------------------------


def _obs(i: int) -> dict[str, str]:
    return {
        "realtime_start": f"2020-01-{i + 1:02d}",
        "realtime_end": ALFRED_REALTIME_END,
        "date": "2019-12-01",
        "value": str(1.0 + i),
    }


def _fred_server(
    rows: list[dict[str, str]],
    *,
    count: int | None = None,
    counts: list[int] | None = None,
    requests: list[httpx.Request] | None = None,
) -> Handler:
    """Paginating fake of series/observations: honours limit/offset, reports ``count``."""
    seen = requests if requests is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        offset = int(request.url.params["offset"])
        limit = int(request.url.params["limit"])
        reported = counts[len(seen) - 1] if counts else (len(rows) if count is None else count)
        body = {
            "count": reported,
            "offset": offset,
            "limit": limit,
            "observations": rows[offset : offset + limit],
        }
        return httpx.Response(200, content=json.dumps(body).encode())

    return handler


def _alfred(handler: Handler, page_size: int = 2) -> FredAlfredClient:
    return FredAlfredClient(_client(handler), SecretStr(KEY), page_size=page_size, url=URL)


def test_multi_page_fetch_retrieves_every_row() -> None:
    rows = [_obs(i) for i in range(5)]
    requests: list[httpx.Request] = []
    raw = _alfred(_fred_server(rows, requests=requests)).all_releases("UNRATE")
    assert raw.count == 5
    assert raw.pages == 3
    assert [dict(r) for r in raw.rows] == rows
    assert [int(r.url.params["offset"]) for r in requests] == [0, 2, 4]
    params = requests[0].url.params
    assert params["series_id"] == "UNRATE"
    assert params["realtime_start"] == ALFRED_REALTIME_START == "1776-07-04"
    assert params["realtime_end"] == ALFRED_REALTIME_END == "9999-12-31"
    assert params["file_type"] == "json"
    assert params["limit"] == "2"
    assert params["api_key"] == KEY


def test_single_page_when_count_fits() -> None:
    rows = [_obs(i) for i in range(3)]
    raw = _alfred(_fred_server(rows), page_size=100_000).all_releases("UNRATE")
    assert raw.pages == 1
    assert len(raw.rows) == 3


def test_row_count_short_of_api_count_raises() -> None:
    rows = [_obs(i) for i in range(3)]
    with pytest.raises(AlfredFetchError, match="retrieved 3 rows but the API reported 4"):
        _alfred(_fred_server(rows, count=4)).all_releases("UNRATE")


def test_count_changing_between_pages_raises() -> None:
    rows = [_obs(i) for i in range(4)]
    with pytest.raises(AlfredFetchError, match="count changed from 4 to 5"):
        _alfred(_fred_server(rows, counts=[4, 5])).all_releases("UNRATE")


def test_duplicate_rows_across_pages_raise() -> None:
    rows = [_obs(0), _obs(1), _obs(1), _obs(2)]  # unstable ordering repeats a row
    with pytest.raises(AlfredFetchError, match="duplicate"):
        _alfred(_fred_server(rows)).all_releases("UNRATE")


def test_malformed_response_raises() -> None:
    handler: Handler = lambda req: httpx.Response(200, content=b'{"observations": []}')  # noqa: E731
    with pytest.raises(AlfredFetchError, match="malformed"):
        _alfred(handler).all_releases("UNRATE")


def test_malformed_observation_raises() -> None:
    body = {"count": 1, "observations": [{"date": "2020-01-01", "value": 1.0}]}
    handler: Handler = lambda req: httpx.Response(200, content=json.dumps(body).encode())  # noqa: E731
    with pytest.raises(AlfredFetchError, match="malformed observation"):
        _alfred(handler).all_releases("UNRATE")


def test_api_error_message_is_redacted() -> None:
    handler: Handler = lambda req: httpx.Response(  # noqa: E731
        400, text=f"Bad Request. api_key {KEY} is not registered."
    )
    with pytest.raises(HttpError) as info:
        _alfred(handler).all_releases("UNRATE")
    assert KEY not in str(info.value)
    assert "***" in str(info.value)
    assert info.value.__context__ is None


def test_invalid_construction() -> None:
    http = _client(lambda req: httpx.Response(200))
    with pytest.raises(ValueError, match="empty"):
        FredAlfredClient(http, SecretStr(""))
    with pytest.raises(ValueError, match="page_size"):
        FredAlfredClient(http, SecretStr(KEY), page_size=100_001)
