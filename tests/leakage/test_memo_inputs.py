"""The memo never sees the benchmark's answer key (docs/adr/0017).

A memo states what the pipeline decided. If the memo code could read GroundTruth, a benchmark
memo could state the planted truth and look better than the pipeline is.
"""

import ast
import json
import shutil
from pathlib import Path

from invariant.stages.memo.document import render
from invariant.stages.memo.facts import build_fact_sheet
from invariant.stages.memo.template import build_template_memo

MEMO_SRC = Path(__file__).parents[2] / "src" / "invariant" / "stages" / "memo"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "memo_runs"
HARNESS = {"benchmark.py", "validation.py"}
"""Modules that generate benchmark runs; they never build or verify a memo's content."""


def test_memo_code_does_not_import_the_benchmark() -> None:
    for path in MEMO_SRC.glob("*.py"):
        if path.name in HARNESS:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(n.startswith("invariant.benchmark") for n in names), (path.name, names)


def test_a_ground_truth_file_in_the_run_dir_changes_nothing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(FIXTURES / "pass_multi", run)
    before = build_fact_sheet(run)
    (run / "ground_truth.json").write_text(json.dumps({"s_conf": "confounded: kill at causal"}))
    (run / "benchmark_score.json").write_text(json.dumps({"wrong_kill": 99}))
    after = build_fact_sheet(run)
    assert before == after
    assert (
        render(build_template_memo(before), before).text
        == render(build_template_memo(after), after).text
    )
