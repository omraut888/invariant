"""The memo document: typed segments, blocks and sections, and its Markdown rendering.

A memo is data (``memo.json``) before it is text (``memo.md``). Its paragraphs are sequences of
**segments**: plain text, or an **atom** that renders one item of the fact sheet in a fixed,
self-describing form (docs/adr/0017):
- ``fact``: a fact with its subject and label ("s_real's Stage one BH q-value < 0.001"), or in
  a table cell only its value;
- ``fate``: a signal's outcome over the kill stages, from the ledger;
- ``stage``, ``adr``, ``cite``, ``term``: a stage name, an ADR number, an author-year citation
  and a fixed term such as "1/N";
- ``recommendation``: the rule-based conclusion.

Every digit in a memo lies inside an atom. Plain text carries none, so no number can enter a
memo except through the fact sheet. ``render`` returns the Markdown with the span of every atom,
which the verifier uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from invariant.core.types import KillReason, StageName
from invariant.ledger.kill_ledger import FateStatus
from invariant.stages.memo.facts import MEMO_STAGES, FactSheet, Recommendation
from invariant.stages.memo.literature import cite_text, load_corpus, reference_text

__all__ = [
    "MEMO_SCHEMA_VERSION",
    "STAGE_TITLES",
    "TERMS",
    "AdrRef",
    "Atom",
    "BibRef",
    "Block",
    "CiteRef",
    "FactRef",
    "FateRef",
    "Memo",
    "Paragraph",
    "RecRef",
    "Rendered",
    "Section",
    "Segment",
    "StageRef",
    "Table",
    "TermRef",
    "Text",
    "fate_text",
    "render",
    "render_segment",
]

MEMO_SCHEMA_VERSION: Final = 1

STAGE_TITLES: Final = {
    StageName.CAUSAL: "causal screen",
    StageName.DECAY: "decay test",
    StageName.PORTFOLIO: "portfolio construction",
    StageName.ADVERSARIAL: "adversarial regime certification",
    StageName.MEMO: "research memo",
}
TERMS: Final = {"one_over_n": "1/N", "hc3": "HC3"}


class _F(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Text(_F):
    type: Literal["text"] = "text"
    text: str


class FactRef(_F):
    type: Literal["fact"] = "fact"
    id: str
    form: Literal["full", "value"] = "full"


class FateRef(_F):
    type: Literal["fate"] = "fate"
    signal: str


class StageRef(_F):
    type: Literal["stage"] = "stage"
    stage: StageName


class CiteRef(_F):
    type: Literal["cite"] = "cite"
    key: str


class AdrRef(_F):
    type: Literal["adr"] = "adr"
    number: int = Field(ge=1, le=9999)


class TermRef(_F):
    type: Literal["term"] = "term"
    term: str


class RecRef(_F):
    type: Literal["recommendation"] = "recommendation"


class BibRef(_F):
    """A bibliography line (``literature.reference_text``)."""

    type: Literal["bibliography"] = "bibliography"
    key: str


Atom = FactRef | FateRef | StageRef | CiteRef | AdrRef | TermRef | RecRef | BibRef
Segment = Annotated[
    Text | FactRef | FateRef | StageRef | CiteRef | AdrRef | TermRef | RecRef | BibRef,
    Field(discriminator="type"),
]
Origin = Literal["template", "llm"]


class Paragraph(_F):
    kind: Literal["paragraph"] = "paragraph"
    origin: Origin = "template"
    statement: str | None = None
    """The mandatory statement this paragraph is (``mandatory.py``), if any."""
    segments: tuple[Segment, ...]


class Table(_F):
    kind: Literal["table"] = "table"
    header: tuple[str, ...]
    rows: tuple[tuple[tuple[Segment, ...], ...], ...]


Block = Annotated[Paragraph | Table, Field(discriminator="kind")]


class Section(_F):
    id: str
    title: str
    blocks: tuple[Block, ...]


class Memo(_F):
    memo_schema: int = MEMO_SCHEMA_VERSION
    run_id: str
    recommendation: Recommendation
    narrative: Literal["none", "llm"]
    narrative_model: str | None = None
    title: str
    sections: tuple[Section, ...]


# --- rendering --------------------------------------------------------------------------------


def stage_label(stage: StageName) -> str:
    return f"Stage {tuple(StageName).index(stage) + 1} ({STAGE_TITLES[stage]})"


def fate_text(sheet: FactSheet, signal_id: str) -> str:
    """A signal's outcome over the kill stages, in words, from the ledger's fate."""
    f = sheet.fate(signal_id)
    if f.status is FateStatus.PASSED:
        return f"{signal_id} passed all three kill stages"
    reason = f.reason
    assert reason is not None
    how = "without a test decision" if reason.stage is None else "by its test"
    return f"{signal_id} was killed at {stage_label(f.stage)} {how} ({reason})"


def recommendation_text(rec: Recommendation) -> str:
    return f"Recommendation: {rec}"


def render_segment(seg: Segment, sheet: FactSheet) -> str:
    match seg:
        case Text():
            return seg.text
        case FactRef():
            fact = sheet.fact(seg.id)
            return fact.full_text() if seg.form == "full" else fact.value_text()
        case FateRef():
            return fate_text(sheet, seg.signal)
        case StageRef():
            return stage_label(seg.stage)
        case CiteRef():
            return cite_text(load_corpus()[seg.key])
        case AdrRef():
            return f"ADR {seg.number:04d}"
        case TermRef():
            return TERMS[seg.term]
        case RecRef():
            return recommendation_text(sheet.recommendation)
        case BibRef():
            return reference_text(load_corpus()[seg.key])
    raise AssertionError(seg)  # pragma: no cover


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    segment: Segment
    origin: Origin
    section: str
    statement: str | None


@dataclass(frozen=True)
class Rendered:
    text: str
    spans: tuple[Span, ...]
    """Every segment (atoms and text) with its character range in ``text``."""


class _Out:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.n = 0
        self.spans: list[Span] = []

    def raw(self, s: str) -> None:
        self.parts.append(s)
        self.n += len(s)

    def seg(
        self, seg: Segment, sheet: FactSheet, origin: Origin, sec: str, stmt: str | None
    ) -> None:
        s = render_segment(seg, sheet)
        self.spans.append(Span(self.n, self.n + len(s), seg, origin, sec, stmt))
        self.raw(s)


def render(memo: Memo, sheet: FactSheet) -> Rendered:
    """``memo`` as Markdown. Narrative (LLM) paragraphs are block quotes."""
    out = _Out()
    out.raw(f"# {memo.title}\n\n")
    for sec in memo.sections:
        out.raw(f"## {sec.title}\n\n")
        for block in sec.blocks:
            if isinstance(block, Paragraph):
                if block.origin == "llm":
                    out.raw("> ")
                for seg in block.segments:
                    out.seg(seg, sheet, block.origin, sec.id, block.statement)
                out.raw("\n\n")
            else:
                out.raw("| " + " | ".join(block.header) + " |\n")
                out.raw("|" + "---|" * len(block.header) + "\n")
                for row in block.rows:
                    out.raw("|")
                    for cell in row:
                        out.raw(" ")
                        for seg in cell:
                            out.seg(seg, sheet, "template", sec.id, None)
                        out.raw(" |")
                    out.raw("\n")
                out.raw("\n")
    return Rendered("".join(out.parts).rstrip("\n") + "\n", tuple(out.spans))


def kill_stage_refs() -> tuple[StageRef, ...]:
    return tuple(StageRef(stage=s) for s in MEMO_STAGES)


def is_general(reason: KillReason | None) -> bool:
    return reason is not None and reason.stage is None
