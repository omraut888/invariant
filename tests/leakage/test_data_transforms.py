"""Look-ahead guards for every data-layer transform.

Each transform must produce, for time t, a value that depends only on inputs at or before t
(or, for row-wise parsers, only on the row itself). The tests perturb "the future" (or other
rows) and require the past (or this row) to be bit-identical.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.fixtures.fakes import alfred_raw, alfred_row, price_frame

from invariant.data.sources.alfred import parse_alfred_rows
from invariant.data.sources.etf import adj_close_panel, log_returns, parse_price_frame
from invariant.data.sources.french import FIVE_FACTOR_HEADER, parse_french_csv

pytestmark = pytest.mark.leakage

N = 12
INDEX = pd.DatetimeIndex(
    pd.bdate_range("2025-03-03", periods=N).astype("datetime64[us]"), name="date"
)
price = st.floats(min_value=0.5, max_value=500.0)
maybe_price = st.one_of(price, st.just(float("nan")))


@settings(max_examples=60)
@given(
    st.lists(maybe_price, min_size=N, max_size=N),
    st.integers(min_value=0, max_value=N - 1),
    st.lists(maybe_price, min_size=N, max_size=N),
)
def test_log_returns_ignore_the_future(base: list[float], cut: int, other: list[float]) -> None:
    a = pd.DataFrame({"X": base}, index=INDEX)
    b = a.copy()
    b.iloc[cut + 1 :, 0] = other[cut + 1 :]  # rewrite everything after `cut`
    ra, rb = log_returns(a), log_returns(b)
    pd.testing.assert_frame_equal(ra.iloc[: cut + 1], rb.iloc[: cut + 1])


@settings(max_examples=40)
@given(
    st.lists(price, min_size=N, max_size=N),
    st.integers(min_value=0, max_value=N - 2),
    st.lists(maybe_price, min_size=N, max_size=N),
)
def test_adj_close_panel_ignores_the_future(
    base: list[float], cut: int, other: list[float]
) -> None:
    dates = [d.strftime("%Y-%m-%d") for d in INDEX]
    future = [*base[: cut + 1], *other[cut + 1 :]]
    pa = adj_close_panel(parse_price_frame(price_frame(dates, {"X": base}), ["X"]), INDEX, ["X"])
    pb = adj_close_panel(parse_price_frame(price_frame(dates, {"X": future}), ["X"]), INDEX, ["X"])
    pd.testing.assert_frame_equal(pa.iloc[: cut + 1], pb.iloc[: cut + 1])


def _ff_text(rows: list[tuple[str, list[float]]]) -> str:
    body = "\n".join(f"{d}," + ",".join(f"{v:.2f}" for v in vals) for d, vals in rows)
    return "header\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n" + body + "\n\nCopyright\n"


@settings(max_examples=40)
@given(
    st.lists(
        st.lists(st.floats(min_value=-20, max_value=20), min_size=6, max_size=6),
        min_size=4,
        max_size=4,
    ),
    st.integers(min_value=0, max_value=3),
    st.lists(st.floats(min_value=-20, max_value=20), min_size=6, max_size=6),
)
def test_french_parse_is_row_wise(
    values: list[list[float]], changed: int, replacement: list[float]
) -> None:
    dates = ["20250303", "20250304", "20250305", "20250306"]
    a = parse_french_csv(_ff_text(list(zip(dates, values, strict=True))), FIVE_FACTOR_HEADER)
    edited = [list(v) for v in values]
    edited[changed] = replacement
    b = parse_french_csv(_ff_text(list(zip(dates, edited, strict=True))), FIVE_FACTOR_HEADER)
    keep = [i for i in range(4) if i != changed]
    pd.testing.assert_frame_equal(a.frame.iloc[keep], b.frame.iloc[keep])


def test_alfred_parse_preserves_every_vintage_and_publication_date() -> None:
    rows = [
        alfred_row("2025-01-01", "2025-02-07", "2025-03-06", "143.0"),
        alfred_row("2025-01-01", "2025-03-07", "2025-04-03", "143.2"),
        alfred_row("2025-01-01", "2025-04-04", "9999-12-31", "143.4"),
        alfred_row("2025-02-01", "2025-03-07", "9999-12-31", "."),
    ]
    frame = parse_alfred_rows(alfred_raw("PAYEMS", list(reversed(rows))))
    got = sorted(
        zip(
            frame["observation_date"].dt.strftime("%Y-%m-%d"),
            frame["realtime_start"].dt.strftime("%Y-%m-%d"),
            frame["realtime_end"].dt.strftime("%Y-%m-%d"),
            strict=True,
        )
    )
    want = sorted((r["date"], r["realtime_start"], r["realtime_end"]) for r in rows)
    assert got == want  # nothing collapsed to final values, no date moved earlier
    jan = frame[frame["observation_date"] == pd.Timestamp("2025-01-01")]
    assert list(jan["value"]) == [143.0, 143.2, 143.4]
    assert np.isnan(frame["value"].iloc[-1])
