"""The corruption operations and the hand-written adversarial sets apply (docs/adr/0017).

These tests check the sets themselves: every entry applies to its fixture memo and changes it.
Whether the verifier rejects them is tested in ``test_memo_verify.py``.
"""

from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from invariant.stages.memo.corruption import CorruptionError, apply_ops
from invariant.stages.memo.document import Memo, render
from invariant.stages.memo.facts import FactSheet, build_fact_sheet
from invariant.stages.memo.template import build_template_memo

FIXTURES = Path(__file__).parents[1] / "fixtures"
SETS = sorted((FIXTURES / "memo_adversarial").glob("set_*.yaml"))
CLASSES = {f"K{i}" for i in range(1, 10)} | {"S", "benign"}


@cache
def _base(case: str) -> tuple[FactSheet, Memo, str]:
    sheet = build_fact_sheet(FIXTURES / "memo_runs" / case)
    memo = build_template_memo(sheet)
    return sheet, memo, render(memo, sheet).text


def _entries() -> list[dict[str, Any]]:
    out = []
    for path in SETS:
        doc = yaml.safe_load(path.read_text())
        out.extend({**e, "set": doc["set"]} for e in doc["entries"])
    return out


def test_sets_are_well_formed() -> None:
    entries = _entries()
    assert SETS and entries
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    for e in entries:
        assert e["class"] in CLASSES, e["id"]
        assert e["expect"] == ("accept" if e["class"] == "benign" else "reject"), e["id"]
        assert e["note"] and e["ops"], e["id"]
    assert {e["class"] for e in entries} == CLASSES


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e["id"])
def test_every_entry_applies_and_changes_the_memo(entry: dict[str, Any]) -> None:
    sheet, memo, md = _base(entry["case"])
    out = apply_ops(memo, md, entry["ops"], sheet)
    assert out.memo_json != memo.model_dump_json() or out.md != md


def test_an_op_that_does_not_apply_is_an_error() -> None:
    sheet, memo, md = _base("pass_multi")
    with pytest.raises(CorruptionError, match="not in memo"):
        apply_ops(memo, md, [{"op": "md_replace", "old": "no such text", "new": "x"}], sheet)
    with pytest.raises(CorruptionError, match="no statement"):
        apply_ops(memo, md, [{"op": "delete_statement", "id": "collateral"}], sheet)
    with pytest.raises(CorruptionError, match="does not render"):
        apply_ops(
            memo, md, [{"op": "insert_llm", "section": "causal", "text": "{cite:nope2020}"}], sheet
        )
