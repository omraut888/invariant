"""Corrupted memos for the verifier's validation (docs/adr/0017).

A corruption takes an accepted memo (``memo.json`` as a ``Memo`` and ``memo.md`` as text) and
returns a tampered pair the verifier must reject. This module holds the **edit operations** the
hand-written adversarial sets are written in (``tests/fixtures/memo_adversarial/``). Each set
entry is a list of operations applied in order; unless an operation says ``md: keep``, the
Markdown is re-rendered from the edited memo, so the tamperer controls both files and the
corruption is not caught merely because ``memo.md`` and ``memo.json`` disagree.

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from invariant.core.run_hash import canonical_json
from invariant.stages.memo.document import Memo, render
from invariant.stages.memo.facts import FactSheet
from invariant.stages.memo.mandatory import MANDATORY
from invariant.stages.memo.template import parse

__all__ = ["FAKE_NARRATIVE_MODEL", "CorruptionError", "Tampered", "apply_ops"]

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
