"""Benchmark validation rates a memo quotes, loaded from the archived reports (docs/adr/0017).

A memo's limits are only as concrete as the evidence behind them. "Stage 1 cannot rule out a
hidden confounder" is stated together with how often a confounded signal passed Stage 1 on the
planted-signal benchmark. Those numbers live in ``docs/results/<adr>/*.json``, the reports the
validation ADRs archived. They are read from there, never typed into templates.

Each rate is labelled as a **benchmark validation rate**, the label says on which preset, and
it is never a property of the strategy in the run. On real data the mandatory statements say
so again (``mandatory.py``).

Every entry names the report, a JSON pointer to a rate object (``estimate``, ``ci_low``,
``ci_high``, ``confidence``) or a plain number, and ``checks``: pointers whose values must equal
the stated preset, scenario or signal, so an index that drifts in a re-archived report fails
loudly instead of quoting the wrong row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from invariant.config.settings import find_project_root

if TYPE_CHECKING:
    from invariant.stages.memo.facts import Fact

__all__ = ["REFERENCE_FACTS", "ReferenceFact", "load_reference_facts", "resolve_pointer"]


@dataclass(frozen=True)
class ReferenceFact:
    id: str
    report: str
    """Path under the results root."""
    pointer: str
    label: str
    kind: str
    interval: bool
    """``pointer`` names a rate object with an interval (else a plain number)."""
    checks: tuple[tuple[str, str], ...] = field(default=())
    range_pointers: tuple[str, str] | None = None
    """A plain number with a reported range (low, high), not a confidence interval."""


_S1 = "0010-causal-stage-validation/report.json"
_S2 = "0012-decay-stage-validation/report.json"
_S3 = "0014-portfolio-stage-validation/report.json"
_S4F = "0016-feasibility/feasibility.json"
_S4O = "0016-vol-state-dev/overlap.json"

REFERENCE_FACTS: Final[tuple[ReferenceFact, ...]] = (
    ReferenceFact(
        "ref.hidden_confounder_stage1_pass",
        _S1,
        "/presets/2/signals/1/pass_rate",
        "benchmark validation rate (planted-signal benchmark, hidden_confounder preset): how "
        "often a confounded signal with its common cause unobserved passed Stage one",
        "rate",
        True,
        (("/presets/2/preset", "hidden_confounder"), ("/presets/2/signals/1/signal_id", "s_conf")),
    ),
    ReferenceFact(
        "ref.realistic_stage1_power",
        _S1,
        "/presets/1/signals/0/pass_rate",
        "benchmark validation rate (planted-signal benchmark, realistic preset): how often a "
        "true edge of realistic strength passed Stage one",
        "rate",
        True,
        (("/presets/1/preset", "realistic"), ("/presets/1/signals/0/signal_id", "s_real")),
    ),
    ReferenceFact(
        "ref.base_stage1_power",
        _S1,
        "/presets/0/signals/0/pass_rate",
        "benchmark validation rate (planted-signal benchmark, base preset): how often a true "
        "edge of default strength passed Stage one",
        "rate",
        True,
        (("/presets/0/preset", "base"), ("/presets/0/signals/0/signal_id", "s_real")),
    ),
    ReferenceFact(
        "ref.stage2_power",
        _S2,
        "/presets/0/signals/2/stage2_kill_given_stage1_pass",
        "benchmark validation rate (planted-signal benchmark, base preset): how often Stage two "
        "detected a planted decay, given a Stage one pass",
        "rate",
        True,
        (("/presets/0/preset", "base"), ("/presets/0/signals/2/signal_id", "s_decay")),
    ),
    ReferenceFact(
        "ref.stage3_power",
        _S3,
        "/scenarios/2/signals/3/stage3_pass_given_reached",
        "benchmark validation rate (planted-signal benchmark, multi-asset pipeline): how often "
        "a portfolio with a true edge passed Stage three, given that it reached Stage three",
        "rate",
        True,
        (("/scenarios/2/scenario", "d1_base"), ("/scenarios/2/signals/3/signal_id", "s_real")),
    ),
    ReferenceFact(
        "ref.stage3_size",
        _S3,
        "/scenarios/0/stage_pass",
        "benchmark validation rate (planted-signal benchmark, no-edge market at the minimum "
        "sample): how often the Stage three kill test passed a portfolio with no edge",
        "rate",
        True,
        (("/scenarios/0/scenario", "d3_null"),),
    ),
    ReferenceFact(
        "ref.noise_passenger",
        _S3,
        "/scenarios/2/signals/2/passes_all",
        "benchmark validation rate (planted-signal benchmark, multi-asset pipeline): how often "
        "a signal with no edge passed all three kill stages, almost always as a passenger of a "
        "true one",
        "rate",
        True,
        (("/scenarios/2/scenario", "d1_base"), ("/scenarios/2/signals/2/signal_id", "s_noise")),
    ),
    ReferenceFact(
        "ref.stage4_worst_case",
        _S4F,
        "/s_volfrag/sr_h_star_median",
        "Stage four feasibility check (planted-signal benchmark, volfrag preset, population "
        "moments): the worst plausible horizon Sharpe ratio of a signal whose edge reverses in "
        "high volatility, median over windows",
        "sharpe",
        False,
        range_pointers=("/s_volfrag/sr_h_star_range/0", "/s_volfrag/sr_h_star_range/1"),
    ),
    ReferenceFact(
        "ref.stage4_state_precision",
        _S4O,
        "/summary/precision/median",
        "Stage four development check (planted-signal benchmark, regime preset): share of "
        "high-volatility-state days that fell in the true stress regime, median over seeds",
        "fraction",
        False,
    ),
)


def resolve_pointer(doc: object, pointer: str) -> object:
    """RFC 6901 JSON pointer lookup (no escapes needed for these reports)."""
    node = doc
    for part in pointer.strip("/").split("/"):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node[part]
        else:
            raise KeyError(pointer)
    return node


def default_results_root() -> Path:
    return find_project_root() / "docs" / "results"


def load_reference_facts(results_root: Path | None = None) -> list[Fact]:
    """Every ``REFERENCE_FACTS`` entry as a ``Fact``, read from the archived reports."""
    from invariant.stages.memo.facts import Fact, FactKind, MemoInputError

    root = results_root if results_root is not None else default_results_root()
    cache: dict[str, object] = {}
    out: list[Fact] = []
    for ref in REFERENCE_FACTS:
        if ref.report not in cache:
            path = root / ref.report
            if not path.is_file():
                raise MemoInputError(f"archived report {path} is missing")
            cache[ref.report] = json.loads(path.read_text(encoding="utf-8"))
        doc = cache[ref.report]
        for pointer, expected in ref.checks:
            if resolve_pointer(doc, pointer) != expected:
                raise MemoInputError(f"{ref.report}{pointer} is not {expected!r}")
        src = f"reference:{ref.report}:"
        node = resolve_pointer(doc, ref.pointer)
        if ref.interval:
            assert isinstance(node, dict)
            out.append(
                Fact(
                    id=ref.id,
                    kind=FactKind(ref.kind),
                    subject=None,
                    label=ref.label,
                    value=float(node["estimate"]),
                    ci=(float(node["ci_low"]), float(node["ci_high"])),
                    ci_confidence=float(node["confidence"]),
                    sources=tuple(
                        src + ref.pointer + "/" + k
                        for k in ("estimate", "ci_low", "ci_high", "confidence")
                    ),
                )
            )
        else:
            rng = (
                tuple(float(str(resolve_pointer(doc, p))) for p in ref.range_pointers)
                if ref.range_pointers
                else None
            )
            out.append(
                Fact(
                    id=ref.id,
                    kind=FactKind(ref.kind),
                    subject=None,
                    label=ref.label,
                    value=float(str(node)),
                    ci=(rng[0], rng[1]) if rng else None,
                    sources=(
                        src + ref.pointer,
                        *(src + p for p in (ref.range_pointers or ())),
                    ),
                )
            )
    return out
