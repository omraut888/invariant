"""Corrupted memos for the verifier's validation (docs/adr/0017).

A corruption takes an accepted memo (``memo.json`` as a ``Memo`` and ``memo.md`` as text) and
returns a tampered pair the verifier must reject. This module holds the **edit operations** the
hand-written adversarial sets are written in (``tests/fixtures/memo_adversarial/``). Each set
entry is a list of operations applied in order; unless an operation says ``md: keep``, the
Markdown is re-rendered from the edited memo, so the tamperer controls both files and the
corruption is not caught merely because ``memo.md`` and ``memo.json`` disagree.

``generate`` draws seeded random corruptions of the classes K1-K9 for the validation gate G2
(its channels are listed in ``_channel``). Its narrative channels draw sentences from fixed
banks, so G2 measures the verifier's mechanism on those banks; the hand-written sets measure
how far its lexicons generalise.

Operations (``op`` and its arguments):
- ``md_replace`` ``old``, ``new``: edit ``memo.md`` only (``old`` must occur);
- ``text_replace`` ``old``, ``new``: edit the one template text segment containing ``old``;
- ``atom_replace`` ``old``, ``new``: replace the first segment equal to atom ``old`` by ``new``;
- ``delete_atom`` ``atom``: remove the first segment equal to ``atom``;
- ``insert_llm`` ``section``, ``text`` [, ``origin``, ``statement``, ``count``]: append a
  narrative paragraph (parsed atom syntax) to a section, and mark the memo as narrated;
- ``delete_statement`` / ``add_statement`` ``id``: drop a mandatory paragraph, or append the
  registry's text for ``id`` to the limits section;
- ``set_statement`` ``old``, ``new`` and ``set_origin`` ``statement``, ``origin``: relabel a
  mandatory paragraph;
- ``set_field`` ``field``, ``value``: a top-level memo field (``recommendation``, ``run_id``,
  ``narrative``, ``title``);
- ``set_title`` ``section``, ``title``;
- ``delete_reference`` / ``add_reference`` ``key``: edit the bibliography.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeVar

import numpy as np

from invariant.core.run_hash import canonical_json
from invariant.core.types import KillReason
from invariant.ledger.kill_ledger import FateStatus
from invariant.stages.memo.document import (
    CiteRef,
    FactRef,
    Memo,
    Paragraph,
    Span,
    Text,
    fate_text,
    recommendation_text,
    render,
    stage_label,
)
from invariant.stages.memo.facts import MEMO_STAGES, FactSheet, Recommendation
from invariant.stages.memo.literature import load_corpus
from invariant.stages.memo.mandatory import MANDATORY
from invariant.stages.memo.template import parse

__all__ = [
    "CLASSES",
    "FAKE_NARRATIVE_MODEL",
    "CorruptionError",
    "Instance",
    "Tampered",
    "apply_ops",
    "generate",
]

FAKE_NARRATIVE_MODEL: Final = "fake-narrative"

# A memo is edited as its JSON dump: plain dicts and lists, which is what the operations move.
_Doc = dict[str, Any]  # Any: pydantic's JSON dump of a Memo


class CorruptionError(ValueError):
    """An operation does not apply to this memo (the set entry is wrong, not the verifier)."""


@dataclass(frozen=True)
class Tampered:
    memo_json: str
    """The edited memo as JSON text (it may no longer validate as a ``Memo``)."""
    md: str


def _segment(atom: str) -> dict[str, Any]:
    segs = parse(atom)
    if len(segs) != 1:
        raise CorruptionError(f"{atom!r} is not a single atom")
    return segs[0].model_dump(mode="json")


def _cells(doc: _Doc) -> list[list[dict[str, Any]]]:
    """Every segment list of the memo: paragraphs and table cells, in order."""
    out: list[list[dict[str, Any]]] = []
    for sec in doc["sections"]:
        for block in sec["blocks"]:
            if block["kind"] == "paragraph":
                out.append(block["segments"])
            else:
                out.extend(cell for row in block["rows"] for cell in row)
    return out


def _section(doc: _Doc, sid: str) -> dict[str, Any]:
    for sec in doc["sections"]:
        if sec["id"] == sid:
            return sec  # type: ignore[no-any-return]
    raise CorruptionError(f"no section {sid!r}")


def _statement(doc: _Doc, sid: str) -> tuple[list[dict[str, Any]], int]:
    for sec in doc["sections"]:
        for i, block in enumerate(sec["blocks"]):
            if block["kind"] == "paragraph" and block.get("statement") == sid:
                return sec["blocks"], i
    raise CorruptionError(f"no statement {sid!r}")


def _apply(doc: _Doc, md: str, op: Mapping[str, Any], sheet: FactSheet) -> tuple[_Doc, str]:
    kind = op["op"]
    match kind:
        case "md_replace":
            if op["old"] not in md:
                raise CorruptionError(f"md_replace: {op['old']!r} not in memo.md")
            return doc, md.replace(op["old"], op["new"], 1)
        case "text_replace":
            hits = [
                s
                for cell in _cells(doc)
                for s in cell
                if s["type"] == "text" and op["old"] in s["text"]
            ]
            if len(hits) != 1:
                raise CorruptionError(f"text_replace: {op['old']!r} in {len(hits)} segments")
            hits[0]["text"] = hits[0]["text"].replace(op["old"], op["new"], 1)
        case "atom_replace" | "delete_atom":
            target = _segment(op["old"] if kind == "atom_replace" else op["atom"])
            for cell in _cells(doc):
                for i, s in enumerate(cell):
                    if s == target:
                        if kind == "atom_replace":
                            cell[i] = _segment(op["new"])
                        else:
                            del cell[i]
                        break
                else:
                    continue
                break
            else:
                raise CorruptionError(f"{kind}: no segment {target}")
        case "insert_llm":
            origin = op.get("origin", "llm")
            para = {
                "kind": "paragraph",
                "origin": origin,
                "statement": op.get("statement"),
                "segments": [s.model_dump(mode="json") for s in parse(op["text"])],
            }
            for _ in range(int(op.get("count", 1))):
                _section(doc, op["section"])["blocks"].append(dict(para))
            if origin == "llm":
                doc["narrative"], doc["narrative_model"] = "llm", FAKE_NARRATIVE_MODEL
        case "delete_statement":
            blocks, i = _statement(doc, op["id"])
            del blocks[i]
        case "add_statement":
            m = next((m for m in MANDATORY if m.id == op["id"]), None)
            if m is None:
                raise CorruptionError(f"no mandatory statement {op['id']!r}")
            segs = [s.model_dump(mode="json") for s in parse(m.text(sheet))]
            _section(doc, "limits")["blocks"].append(
                {"kind": "paragraph", "origin": "template", "statement": m.id, "segments": segs}
            )
        case "set_statement":
            blocks, i = _statement(doc, op["old"])
            blocks[i]["statement"] = op["new"]
        case "set_origin":
            blocks, i = _statement(doc, op["statement"])
            blocks[i]["origin"] = op["origin"]
        case "set_field":
            if op["field"] not in doc:
                raise CorruptionError(f"no memo field {op['field']!r}")
            doc[op["field"]] = op["value"]
        case "set_title":
            _section(doc, op["section"])["title"] = op["title"]
        case "delete_reference" | "add_reference":
            blocks = _section(doc, "references")["blocks"]
            bib = {"type": "bibliography", "key": op["key"]}
            idx = [i for i, b in enumerate(blocks) if b["segments"] == [bib]]
            if kind == "delete_reference":
                if not idx:
                    raise CorruptionError(f"no reference {op['key']!r}")
                del blocks[idx[0]]
            else:
                blocks.append(
                    {
                        "kind": "paragraph",
                        "origin": "template",
                        "statement": None,
                        "segments": [bib],
                    }
                )
        case _:
            raise CorruptionError(f"unknown operation {kind!r}")
    if op.get("md", "rerender") == "keep":
        return doc, md
    try:
        return doc, render(Memo.model_validate(doc), sheet).text
    except (KeyError, ValueError) as exc:
        raise CorruptionError(f"{kind}: the edited memo does not render ({exc})") from exc


def apply_ops(memo: Memo, md: str, ops: Sequence[Mapping[str, Any]], sheet: FactSheet) -> Tampered:
    """Apply ``ops`` in order (module docstring) and return the tampered files."""
    doc: _Doc = memo.model_dump(mode="json")
    for op in ops:
        doc, md = _apply(doc, md, op, sheet)
    if doc == memo.model_dump(mode="json") and md == render(memo, sheet).text:
        raise CorruptionError("the operations changed nothing")
    return Tampered(canonical_json(doc), md)


# --- seeded generators of the corruption classes (validation gate G2) -------------------------

CLASSES: Final = ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9")
"""The corruption classes of docs/adr/0017. Each generator picks one of its channels at random;
a channel that cannot apply to this memo is skipped and another drawn."""

_NUMERIC: Final = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")
_FIXED_NUMBERS: Final = re.compile(r"Stage \d\b|ADR \d{4}\b|\b\d{1,3}% interval\b")
_NUMBER_WORDS: Final = (
    "two",
    "three",
    "seven",
    "ten",
    "twenty",
    "a hundred",
    "a dozen",
    "half",
    "a third",
    "a quarter",
    "twice",
    "three times",
    "tenfold",
    "ninety percent",
    "a few",
    "several",
    "most",
    "the majority of",
)
_NARRATIVE_NUMBER: Final = (
    "The edge is worth about {n} Sharpe points.",
    "Roughly {n} of the candidates showed a link.",
    "The portfolio held {n} positions on average.",
    "Costs would take about {n} of the return.",
    "The sample covers {n} years of daily data.",
    "In high volatility the Sharpe ratio fell by {n}.",
)
_VERDICT_SUBJECTS: Final = (
    "The strongest candidate",
    "The edge",
    "The portfolio",
    "Its link to returns",
    "The second candidate",
    "The decaying candidate",
    "Every candidate",
    "{sid}",
)
_VERDICT_PHRASES: Final = (
    "passed the decay test",
    "was killed by the causal screen",
    "held up out of sample",
    "survived every stage",
    "failed the portfolio test",
    "was rejected for decline",
    "did not decline",
    "is confirmed",
    "was validated on fresh data",
    "beat equal weight",
    "is a real edge",
    "was ruled out",
    "is significant",
)
_RECOMMENDATION_PHRASES: Final = (
    "We recommend trading this strategy.",
    "The strategy should be traded with real capital.",
    "This is an attractive strategy to allocate to.",
    "The evidence argues against putting money behind it.",
    "The strategy is worth deploying now.",
    "On balance the case for trading it is strong.",
)
_LITERATURE_CLAIMS: Final = (
    "such links weaken once they are published",
    "shrinkage stabilises portfolio weights",
    "momentum strategies crash after market rebounds",
    "breaks are hardest to detect near the ends of a sample",
    "confounding is common in return prediction",
)
_LITERATURE_LEADS: Final = (
    "{surname} showed that {claim}.",
    "{surname} found that {claim}.",
    "As {surname} argued, {claim}.",
    "The literature finds that {claim}.",
    "Studies show that {claim}.",
    "It is well known that {claim}.",
    "Prior work suggests that {claim}.",
    "Research has documented that {claim}.",
)
_BANNED_SENTENCES: Final = (
    "The portfolio is distributionally robust.",
    "This is a DRO portfolio.",
    "The strategy is certified for high volatility.",
    "The portfolio was stress-tested against market crashes.",
    "Its edge over equal weight is guaranteed.",
    "The edge is proven on this sample.",
    "The signal has a causal effect on returns.",
    "The edge has a long half-life.",
    "The strategy is robust to regimes.",
    "The portfolio is regime-proof.",
)
_FAKE_KEYS: Final = ("smith2020", "jones2015a", "fama1970", "lopez2018", "doe2024")


_T = TypeVar("_T")


def _choice(rng: np.random.Generator, seq: Sequence[_T]) -> _T:
    return seq[int(rng.integers(len(seq)))]


def _number_text(rng: np.random.Generator) -> str:
    match int(rng.integers(4)):
        case 0:
            return f"{rng.uniform(-3, 3):.{int(rng.integers(1, 4))}f}"
        case 1:
            return str(int(rng.integers(2, 5000)))
        case 2:
            return f"{int(rng.integers(1, 100))}%"
    return str(_choice(rng, _NUMBER_WORDS))


def _fact_atoms(memo: Memo) -> list[tuple[str, str]]:
    """(id, form) of every fact atom of the template blocks."""
    out: dict[tuple[str, str], None] = {}
    for sec in memo.sections:
        for block in sec.blocks:
            cells = (
                [block.segments]
                if isinstance(block, Paragraph)
                else [c for r in block.rows for c in r]
            )
            for cell in cells:
                for seg in cell:
                    if isinstance(seg, FactRef):
                        out[(seg.id, seg.form)] = None
    return list(out)


def _atom_text(sheet: FactSheet, fid: str, form: str) -> str:
    f = sheet.fact(fid)
    return f.value_text() if form == "value" else f.full_text()


def _atom(fid: str, form: str) -> str:
    return f"{{fact:{fid}|value}}" if form == "value" else f"{{fact:{fid}}}"


def _md_number_edit(md: str, spans: Sequence[Span], rng: np.random.Generator) -> str:
    """K1 in memo.md: one number inside a fact atom nudged, sign-flipped or swapped."""
    tokens: list[tuple[int, int, str]] = []
    for sp in spans:
        if not isinstance(sp.segment, FactRef):
            continue
        text = md[sp.start : sp.end]
        masked = _FIXED_NUMBERS.sub(lambda m: "#" * len(m.group(0)), text)
        tokens += [
            (sp.start + m.start(), sp.start + m.end(), m.group(0))
            for m in _NUMERIC.finditer(masked)
        ]
    if not tokens:
        raise CorruptionError("no number in a fact atom")
    start, end, tok = _choice(rng, tokens)
    match int(rng.integers(3)):
        case 0:
            last = tok[-1]
            new = tok[:-1] + _choice(rng, [d for d in "0123456789" if d != last])
        case 1:
            new = tok[1:] if tok.startswith("-") else "-" + tok
        case _:
            others = [t for t in tokens if t[2] != tok]
            if not others:
                raise CorruptionError("no other number to swap in")
            new = str(_choice(rng, others)[2])
    return md[:start] + new + md[end:]


def _channel(
    cls: str, memo: Memo, md: str, sheet: FactSheet, rng: np.random.Generator
) -> tuple[str, list[dict[str, Any]] | str]:
    """One random channel of ``cls``: a name and either ops or an edited memo.md."""
    narrative = ("recommendation", "causal", "decay", "portfolio", "stress")
    present = [s.id for s in memo.sections]
    narr = [s for s in narrative if s in present]
    stated = [
        b.statement
        for s in memo.sections
        for b in s.blocks
        if isinstance(b, Paragraph) and b.statement
    ]
    atoms = _fact_atoms(memo)
    signals = list(sheet.signal_ids())
    match cls:
        case "K1":
            if rng.random() < 0.5:
                return "md_number", _md_number_edit(md, render(memo, sheet).spans, rng)
            fid, form = _choice(rng, atoms)
            text = _atom_text(sheet, fid, form)
            same = [a for a in atoms if a[1] == form and _atom_text(sheet, a[0], form) != text]
            if not same:
                raise CorruptionError("no atom of the same form renders differently")
            other = _choice(rng, same)[0]
            return "swap_atoms", [
                {"op": "atom_replace", "old": _atom(fid, form), "new": _atom(other, form)}
            ]
        case "K2":
            k = int(rng.integers(3))
            if k == 0:
                text = str(_choice(rng, _NARRATIVE_NUMBER)).format(n=_number_text(rng))
                return "narrative_number", [
                    {"op": "insert_llm", "section": _choice(rng, narr), "text": text}
                ]
            if k == 1:
                texts = [
                    seg.text
                    for s in memo.sections
                    for b in s.blocks
                    if isinstance(b, Paragraph)
                    for seg in b.segments
                    if isinstance(seg, Text) and " " in seg.text
                ]
                old = _choice(rng, texts)
                spaces = [i for i, ch in enumerate(old) if ch == " "]
                i = _choice(rng, spaces)
                new = old[:i] + " " + _number_text(rng) + old[i:]
                return "template_number", [{"op": "text_replace", "old": old, "new": new}]
            spaces = [i for i, ch in enumerate(md) if ch == " "]
            i = _choice(rng, spaces)
            return "md_number", md[:i] + f" (n = {int(rng.integers(2, 9999))})" + md[i:]
        case "K3":
            fid, form = _choice(rng, atoms)
            other = _choice(rng, [f.id for f in sheet.facts if f.id != fid])
            keep = bool(rng.random() < 0.5)
            return "repoint_" + ("keep" if keep else "rerender"), [
                {
                    "op": "atom_replace",
                    "old": _atom(fid, form),
                    "new": _atom(other, form),
                    "md": "keep" if keep else "rerender",
                }
            ]
        case "K4":
            k = int(rng.integers(3))
            if k == 0:
                sid = _choice(rng, signals)
                others = [s for s in signals if s != sid]
                if not others:
                    raise CorruptionError("one signal only")
                return "swap_fate", [
                    {
                        "op": "atom_replace",
                        "old": f"{{fate:{sid}}}",
                        "new": f"{{fate:{_choice(rng, others)}}}",
                    }
                ]
            if k == 1:
                sid = _choice(rng, signals)
                text = fate_text(sheet, sid)
                if text not in md:
                    raise CorruptionError("fate text not in memo.md")
                f = sheet.fate(sid)
                if f.status is FateStatus.PASSED:
                    stage = _choice(rng, list(MEMO_STAGES))
                    reasons = [r for r in KillReason if r.stage is stage]
                    alt = (
                        f"{sid} was killed at {stage_label(stage)} by its test "
                        f"({_choice(rng, reasons)})"
                    )
                else:
                    alt = f"{sid} passed all three kill stages"
                return "md_fate", md.replace(text, alt, 1)
            subject = str(_choice(rng, _VERDICT_SUBJECTS)).format(sid=_choice(rng, signals))
            text = f"{subject} {_choice(rng, _VERDICT_PHRASES)}."
            return "narrative_verdict", [
                {"op": "insert_llm", "section": _choice(rng, narr), "text": text}
            ]
        case "K5":
            k = int(rng.integers(3))
            recs = [r for r in Recommendation if r is not sheet.recommendation]
            if k == 0:
                return "field", [
                    {
                        "op": "set_field",
                        "field": "recommendation",
                        "value": str(_choice(rng, recs)),
                    }
                ]
            if k == 1:
                old = recommendation_text(sheet.recommendation)
                return "md_line", md.replace(old, recommendation_text(_choice(rng, recs)), 1)
            return "narrative", [
                {
                    "op": "insert_llm",
                    "section": "recommendation",
                    "text": str(_choice(rng, _RECOMMENDATION_PHRASES)),
                }
            ]
        case "K6":
            k = int(rng.integers(5))
            sid = _choice(rng, stated)
            if k == 0:
                return "delete", [{"op": "delete_statement", "id": sid}]
            if k == 1:
                absent = [m.id for m in MANDATORY if m.id not in stated]
                rng.shuffle(absent)
                for a in absent:
                    try:
                        apply_ops(memo, md, [{"op": "add_statement", "id": a}], sheet)
                    except (CorruptionError, KeyError, ValueError):
                        continue
                    return "spurious", [{"op": "add_statement", "id": a}]
                raise CorruptionError("no spurious statement renders")
            if k == 2:
                other = _choice(rng, [m.id for m in MANDATORY if m.id != sid])
                return "relabel", [{"op": "set_statement", "old": sid, "new": other}]
            if k == 3:
                return "to_narrative", [{"op": "set_origin", "statement": sid, "origin": "llm"}]
            texts = [
                seg.text
                for s in memo.sections
                for b in s.blocks
                if isinstance(b, Paragraph) and b.statement == sid
                for seg in b.segments
                if isinstance(seg, Text) and seg.text.strip()
            ]
            old = _choice(rng, texts)
            for neg in (" not ", " no ", " never ", " cannot "):
                if neg in old:
                    return "reword", [
                        {"op": "text_replace", "old": old, "new": old.replace(neg, " ", 1)}
                    ]
            words = old.split(" ")
            if len(words) < 3:
                raise CorruptionError("statement text too short to reword")
            i = int(rng.integers(len(words)))
            return "reword", [
                {"op": "text_replace", "old": old, "new": " ".join(words[:i] + words[i + 1 :])}
            ]
        case "K7":
            cites = [
                seg.key
                for s in memo.sections
                for b in s.blocks
                if isinstance(b, Paragraph)
                for seg in b.segments
                if isinstance(seg, CiteRef)
            ]
            k = int(rng.integers(4))
            if k == 0 and cites:
                return "drop_cite", [
                    {"op": "delete_atom", "atom": f"{{cite:{_choice(rng, cites)}}}"}
                ]
            if k == 1 and cites:
                return "fake_key", [
                    {
                        "op": "atom_replace",
                        "old": f"{{cite:{_choice(rng, cites)}}}",
                        "new": f"{{cite:{_choice(rng, _FAKE_KEYS)}}}",
                        "md": "keep",
                    }
                ]
            if k == 2:
                surname = _choice(rng, [a for e in load_corpus().values() for a in e.authors])
                text = str(_choice(rng, _LITERATURE_LEADS)).format(
                    surname=surname, claim=_choice(rng, _LITERATURE_CLAIMS)
                )
                return "uncited", [
                    {"op": "insert_llm", "section": _choice(rng, narr), "text": text}
                ]
            uncited = sorted(set(load_corpus()) - set(cites))
            if rng.random() < 0.5 and cites:
                return "bib_drop", [
                    {"op": "delete_reference", "key": _choice(rng, sorted(set(cites)))}
                ]
            return "bib_add", [{"op": "add_reference", "key": _choice(rng, uncited)}]
        case "K8":
            return "banned", [
                {
                    "op": "insert_llm",
                    "section": _choice(rng, narr),
                    "text": str(_choice(rng, _BANNED_SENTENCES)),
                }
            ]
        case "K9":
            i = int(rng.integers(len(md)))
            k = int(rng.integers(4))
            if k == 0:
                ch = _choice(
                    rng, [c for c in "abcdefghijklmnopqrstuvwxyz0123456789 .,;|#-" if c != md[i]]
                )
                return "substitute", md[:i] + ch + md[i + 1 :]
            if k == 1:
                return "insert", md[:i] + _choice(rng, list("abcxyz01 .,;")) + md[i:]
            if k == 2:
                return "delete", md[:i] + md[i + 1 :]
            lines = md.split("\n")
            j = int(rng.integers(len(lines)))
            return "duplicate_line", "\n".join(lines[: j + 1] + lines[j:])
    raise CorruptionError(f"unknown class {cls!r}")


@dataclass(frozen=True)
class Instance:
    cls: str
    channel: str
    tampered: Tampered


def generate(
    cls: str, memo: Memo, md: str, sheet: FactSheet, rng: np.random.Generator, *, tries: int = 50
) -> Instance:
    """One random corruption of class ``cls`` (``CLASSES``) of an accepted memo."""
    for _ in range(tries):
        try:
            channel, edit = _channel(cls, memo, md, sheet, rng)
            if isinstance(edit, str):
                if edit == md:
                    continue
                t = Tampered(canonical_json(memo.model_dump(mode="json")), edit)
            else:
                t = apply_ops(memo, md, edit, sheet)
        except CorruptionError:
            continue
        return Instance(cls, channel, t)
    raise CorruptionError(f"no {cls} corruption applies to this memo after {tries} tries")
