"""Stage 5 entry point: build a run's memo and write it next to the run (docs/adr/0017).

Writes ``<run_dir>/memo/memo.json`` (the document) and ``memo.md`` (its rendering). The memo is
written only after the verifier (``verify.py``) accepts the exact bytes that are written; a
rejected memo is not written, and nothing is patched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from invariant.core.run_hash import canonical_json
from invariant.stages.memo.document import Memo, render
from invariant.stages.memo.facts import FactSheet, build_fact_sheet
from invariant.stages.memo.template import build_template_memo
from invariant.stages.memo.verify import VerificationReport, verify

__all__ = ["MEMO_DIR", "MemoOutput", "MemoRejectedError", "write_memo"]

MEMO_DIR: Final = "memo"


class MemoRejectedError(RuntimeError):
    """The verifier rejected the memo; it was not written."""

    def __init__(self, report: VerificationReport) -> None:
        self.report = report
        lines = "; ".join(f"{f.check}: {f.detail}" for f in report.failures[:5])
        super().__init__(f"memo rejected ({len(report.failures)} failures): {lines}")


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
    memo_json = canonical_json(memo.model_dump(mode="json")) + "\n"
    md = render(memo, sheet).text
    report = verify(memo_json, md, run_dir, results_root=results_root)
    if not report.accepted:
        raise MemoRejectedError(report)
    out = run_dir / MEMO_DIR
    out.mkdir(exist_ok=True)
    json_path, md_path = out / "memo.json", out / "memo.md"
    json_path.write_text(memo_json, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    return MemoOutput(sheet, memo, json_path, md_path)
