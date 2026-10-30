"""Stage 5 entry point: build a run's memo and write it next to the run (docs/adr/0017).

Writes ``<run_dir>/memo/memo.json`` (the document) and ``memo.md`` (its rendering). The memo is
written only after the verifier accepts it (verify.py, next).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from invariant.core.run_hash import canonical_json
from invariant.stages.memo.document import Memo, render
from invariant.stages.memo.facts import FactSheet, build_fact_sheet
from invariant.stages.memo.template import build_template_memo

__all__ = ["MEMO_DIR", "MemoOutput", "write_memo"]

MEMO_DIR: Final = "memo"


@dataclass(frozen=True)
class MemoOutput:
    sheet: FactSheet
    memo: Memo
    json_path: Path
    md_path: Path


def write_memo(run_dir: Path, *, results_root: Path | None = None) -> MemoOutput:
    """Build the template memo of the completed run in ``run_dir`` and write it."""
    sheet = build_fact_sheet(run_dir, results_root=results_root)
    memo = build_template_memo(sheet)
    out = run_dir / MEMO_DIR
    out.mkdir(exist_ok=True)
    json_path, md_path = out / "memo.json", out / "memo.md"
    json_path.write_text(canonical_json(memo.model_dump(mode="json")) + "\n", encoding="utf-8")
    md_path.write_text(render(memo, sheet).text, encoding="utf-8")
    return MemoOutput(sheet, memo, json_path, md_path)
