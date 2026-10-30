"""The memo verifier: a memo is accepted only if every check passes (docs/adr/0017).

A memo that fails is **rejected, not patched**: the verifier reports every failure and changes
nothing. Its input is the two files a memo is published as (``memo.json`` and ``memo.md``) and
the run directory; it rebuilds the fact sheet itself and, for the numbers, verdicts and
triggers, reads the raw ledger lines and result files by its own code path, so a bug shared by
the fact builder and the renderer is still caught.

Checks
------
- **V1 structure.** The memo parses; its run id, title, recommendation and sections are the
  template memo's; every template block (paragraph or table) equals the verifier's re-render;
  narrative (LLM) paragraphs only end the narrative sections, at most ``MAX_NARRATIVE`` per
  section, carry no statement id, and use only text, full-form fact, fate and cite segments;
  the ``narrative`` flag says whether there are any; the bibliography is exactly the cited keys.
- **V2 atoms.** Every atom names something that exists (fact, signal, corpus key, term), and
  each fact's value equals its sources, resolved from the raw files.
- **V3 numbers.** Every digit of ``memo.md`` lies inside an atom, and every number in an atom
  equals, at its displayed precision, the raw value its source names.
- **V4 narrative text.** Narrative text segments carry no digit, number or quantity word,
  frequency or probability word, signal id, stage name, verdict word, causal or performance
  claim, recommendation word or Markdown control character. Verdicts and the recommendation
  live only in atoms and template blocks; anything claim-like the verifier cannot check is
  rejected (a deliberate trade of false rejections of harmless prose for catch rate).
- **V5 verdicts.** Every fate atom, the recommendation and every clause of ``memo.md`` that
  joins a signal id to a verdict word agree with the fates read from the raw ledger. Such a
  clause must have a canonical form; one that does not is a failure.
- **V6 mandatory statements.** The statement ids present, in order, are exactly the triggered
  set, which the verifier also recomputes from the raw ledger; each is a template paragraph of
  the limits section (its exact text is V1).
- **V7 literature.** Every cite key is in the corpus, and a narrative sentence with a literature
  cue (an author's surname, "et al", "literature", "studies", "research", "found", "shown",
  "according to", "well known", ...) carries a cite atom.
- **V8 banned phrases** in narrative text ("distributionally robust", "DRO portfolio",
  "certif*", "stress-test*", "guarantee*", "prove*", "causal effect", "half-life", "robust*",
  "-proof", ...).
- **V9 rendering.** ``memo.md`` equals ``render(memo.json)``.

Limits
------
The verifier checks that a literature claim is cited, not that the cited paper supports it;
entailment is semantic. It checks narrative prose by lexicon, so a claim phrased in words it
does not list can pass V4; the adversarial sets measure that (docs/adr/0017).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from invariant.ledger.kill_ledger import LEDGER_NAME
from invariant.stages.memo.document import (
    TERMS,
    BibRef,
    CiteRef,
    FactRef,
    FateRef,
    Memo,
    Paragraph,
    Section,
    Segment,
    Table,
    TermRef,
    Text,
    render,
)
from invariant.stages.memo.facts import (
    DSR_TRIALS,
    MEMO_STAGES,
    Fact,
    FactKind,
    FactSheet,
    MemoInputError,
    Recommendation,
    build_fact_sheet,
)
from invariant.stages.memo.literature import load_corpus
from invariant.stages.memo.mandatory import triggered
from invariant.stages.memo.reference import default_results_root
from invariant.stages.memo.template import (
    NARRATIVE_SECTIONS,
    build_template_memo,
    references_section,
)

__all__ = ["MAX_NARRATIVE", "Failure", "VerificationReport", "verify", "verify_run"]

MAX_NARRATIVE: Final = 2
"""Narrative paragraphs a section may end with."""


@dataclass(frozen=True)
class Failure:
    check: str
    """``V1`` ... ``V9``."""
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    failures: tuple[Failure, ...]

    @property
    def accepted(self) -> bool:
        return not self.failures

    def checks_failed(self) -> tuple[str, ...]:
        return tuple(sorted({f.check for f in self.failures}))


@dataclass
class _Log:
    failures: list[Failure] = field(default_factory=list)

    def fail(self, check: str, detail: str) -> None:
        self.failures.append(Failure(check, detail))


# --- the raw run, read by the verifier's own code path ---------------------------------------


@dataclass(frozen=True)
class _RawFate:
    passed: bool
    stage: str | None
    reason: str | None


class _Raw:
    """The run directory's ledger and result files as plain JSON, independent of ``facts.py``."""

    def __init__(self, run_dir: Path, results_root: Path) -> None:
        self.run_dir = run_dir
        self.results_root = results_root
        lines = (run_dir / LEDGER_NAME).read_text(encoding="utf-8").splitlines()
        entries = [json.loads(x) for x in lines if x.strip()]
        last = max((int(e["attempt"]) for e in entries), default=1)
        self.entries: list[dict[str, object]] = [e for e in entries if e["attempt"] == last]
        self._json: dict[Path, object] = {}

    def doc(self, path: Path) -> object:
        if path not in self._json:
            self._json[path] = json.loads(path.read_text(encoding="utf-8"))
        return self._json[path]

    def signals(self) -> list[str]:
        return list(dict.fromkeys(str(e["signal_id"]) for e in self.entries))

    def entry(self, sid: str, stage: str) -> dict[str, object] | None:
        hits = [e for e in self.entries if e["signal_id"] == sid and e["stage"] == stage]
        return hits[-1] if hits else None

    def stage_entries(self, stage: str) -> list[dict[str, object]]:
        return [e for e in self.entries if e["stage"] == stage]

    def fate(self, sid: str) -> _RawFate:
        stages = [str(s) for s in MEMO_STAGES]
        made = [e for e in self.entries if e["signal_id"] == sid and e["stage"] in stages]
        if made and made[-1]["verdict"] == "kill":
            reason = made[-1]["reason"]
            return _RawFate(False, str(made[-1]["stage"]), None if reason is None else str(reason))
        if len(made) == len(stages):
            return _RawFate(True, None, None)
        raise MemoInputError(f"{sid} has no final verdict over Stages 1-3 in the raw ledger")

    def recommendation(self) -> Recommendation:
        fates = [self.fate(s) for s in self.signals()]
        if any(f.passed for f in fates):
            return Recommendation.PAPER_TRADING_CANDIDATE
        if any(f.reason is not None and f.reason.startswith("general.") for f in fates):
            return Recommendation.NOT_DECIDED
        return Recommendation.AGAINST

    def resolve(self, source: str) -> object:
        """The raw value a fact source names (``facts.py`` module docstring)."""
        kind, _, rest = source.partition(":")
        match kind:
            case "ledger":
                sid, stage, metric = rest.split(":", 2)
                e = self.entry(sid, stage)
                if e is None:
                    raise KeyError(source)
                metrics = e["metrics"]
                assert isinstance(metrics, dict)
                return metrics[metric]
            case "result":
                name, _, pointer = rest.partition(":")
                return _pointer(self.doc(self.run_dir / name), pointer)
            case "manifest":
                node = self.doc(self.run_dir / "manifest.json")
                for part in rest.split("."):
                    assert isinstance(node, dict)
                    node = node[part]
                return node
            case "reference":
                report, _, pointer = rest.rpartition(":")
                return _pointer(self.doc(self.results_root / report), pointer)
            case "const":
                if rest == "dsr_trials":
                    return DSR_TRIALS
        raise KeyError(source)


def _pointer(doc: object, pointer: str) -> object:
    node = doc
    for part in pointer.strip("/").split("/"):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node[part]
        else:
            raise KeyError(pointer)
    return node


# --- lexicons (V4, V7, V8) -------------------------------------------------------------------


def _words(*alternatives: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


_NUMBER_WORDS: Final = _words(
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve",
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen",
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety",
    r"hundreds?|thousands?|millions?|billions?|dozens?|score",
    r"half|halves|halve[ds]?|halving|thirds?|quarters?|fifths?|sixths?|tenths?",
    r"first|second|fourth|fifth|sixth|seventh|eighth|ninth|tenth",
    r"twice|thrice|double[ds]?|doubling|triple[ds]?|tripling|quadruple[ds]?",
    r"\w+fold|percent|percentages?|per\s+cent|pct|basis\s+points?|bps?",
    r"single|couple|few|fewer|several|many|most|mostly|majority|minority|numerous|handful",
    r"multiple|plenty|bulk",
)
_FREQUENCY_WORDS: Final = _words(
    r"always|never|often|rarely|seldom|usually|frequently|sometimes|occasionally|typically",
    r"likely|unlikely|likelihood|probably|probable|improbable|probability|chances?|odds",
    r"certainly|certain|surely|almost|nearly|roughly|approximately|virtually|mostly",
)
_STAGE_WORDS: Final = _words(
    r"stages?|causal\s+screen|decay\s+test|portfolio\s+construction|kill\s+test",
    r"adversarial|certification|research\s+memo",
)
_VERDICT_WORDS: Final = _words(
    r"pass|passes|passed|passing|kill|kills|killed|killing",
    r"surviv\w*|fail\w*|reject\w*|confirm\w*|validat\w*|verif\w*|accept\w*",
    r"held\s+up|holds?\s+up|holding\s+up|ruled?\s+out|rules\s+out|ruling\s+out",
    r"significant|significantly|insignificant|detect\w*|establish\w*|demonstrat\w*",
    r"succeed\w*|success\w*|winners?|losers?|outperform\w*|underperform\w*|beat|beats|beaten",
    r"better|worse|best|worst|superior|inferior|stable|unstable|stability|persist\w*",
    r"passengers?|collateral|confound\w*|spurious|genuine|real\s+edge|true\s+edge",
)
_CAUSAL_WORDS: Final = _words(
    r"effects?|causal\w*|causes?|caused|causing|causation|drives?|driven|driving",
)
_PERFORMANCE_WORDS: Final = _words(
    r"profit\w*|earn\w*|lose|loses|lost|losing|loss|losses|gains?|gained|returns?\s+of",
    r"alpha|outperformance|excess\s+returns?",
)
_RECOMMENDATION_WORDS: Final = _words(
    r"recommend\w*|candidates?\s+for|(?<!paper\s)trade[ds]?|(?<!paper\s)trading|invest\w*",
    r"allocat\w*|capital|money|buy|buying|sell|selling|attractive|worth|should|ought",
    r"deploy\w*|go\s+live|case\s+for|case\s+against|undecided|not\s+decided",
)
_BANNED: Final = _words(
    r"distributionally\s+robust|dro\s+portfolios?|certif\w*|stress[-\s]tested|stress[-\s]tests?",
    r"stress[-\s]testing|guarantee\w*|prove[dns]?|proving|proof|causal\s+effects?",
    r"half[-\s]li(?:fe|ves)|robust\w*|\w+-proof|immune|immunity|resilien\w*|bullet\s*proof",
    r"safe|safety|risk-free|riskless",
)
_CONTROL: Final = re.compile(r"[\d%±=<>~^|#`\[\]{}\\*_\n\r\t]")
"""Digits, number symbols and Markdown control characters."""
_SIGNAL_LIKE: Final = re.compile(r"\b(?:s|syn)_\w+", re.IGNORECASE)

_CITE_CUES: Final = _words(
    r"et\s+al|literature|stud(?:y|ies|ied)|research\w*|papers?(?!\s+trad)|articles?|authors?",
    r"(?:prior|earlier|previous|existing|related|seminal|classic)\s+work",
    r"work\s+(?:of|by)|well[-\s]known|widely|known\s+(?:to|that)|shown|showed|shows?\s+that",
    r"found|finds?\s+that|according\s+to|evidence\s+from|documented|reported|argue[ds]?",
    r"suggest\w*|theory|theoretical|economists?|academics?|researchers?|scholars?",
)


def _surnames() -> re.Pattern[str]:
    names = {a for e in load_corpus().values() for a in e.authors}
    parts = {p for n in names for p in re.split(r"[\s-]+", n) if len(p) > 1 and p[0].isupper()}
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, parts))) + r")\b")


# --- the checks -------------------------------------------------------------------------------


_NARRATIVE_SEGMENTS: Final = (Text, FactRef, FateRef, CiteRef)


def _cells(block: Paragraph | Table) -> Iterator[tuple[Segment, ...]]:
    if isinstance(block, Paragraph):
        yield block.segments
    else:
        for row in block.rows:
            yield from row


def _segments(sections: Iterable[Section]) -> Iterator[Segment]:
    for sec in sections:
        for block in sec.blocks:
            for cell in _cells(block):
                yield from cell


def _is_narrative(block: Paragraph | Table) -> bool:
    return isinstance(block, Paragraph) and block.origin == "llm"


def _v1_structure(memo: Memo, sheet: FactSheet, log: _Log) -> None:
    expected = build_template_memo(sheet)
    for name in ("memo_schema", "run_id", "title"):
        if getattr(memo, name) != getattr(expected, name):
            log.fail("V1", f"memo {name} is {getattr(memo, name)!r}, not the template's")
    ids = [s.id for s in memo.sections]
    if ids != [s.id for s in expected.sections]:
        log.fail("V1", f"sections {ids} are not the template's {[s.id for s in expected.sections]}")
        return
    narrated = False
    for sec, exp in zip(memo.sections, expected.sections, strict=True):
        if sec.title != exp.title:
            log.fail("V1", f"section {sec.id} has title {sec.title!r}")
        template = [b for b in sec.blocks if not _is_narrative(b)]
        narrative = [b for b in sec.blocks if _is_narrative(b)]
        if sec.id == "references":
            want = references_section([s for s in memo.sections if s.id != "references"])
            if template != list(want.blocks):
                log.fail("V1", "the bibliography is not exactly the cited corpus entries")
        elif template != list(exp.blocks):
            log.fail("V1", f"section {sec.id}: template blocks differ from the re-render")
        if not narrative:
            continue
        narrated = True
        if sec.id not in NARRATIVE_SECTIONS:
            log.fail("V1", f"section {sec.id} may not carry narrative paragraphs")
        if len(narrative) > MAX_NARRATIVE:
            log.fail("V1", f"section {sec.id} has {len(narrative)} narrative paragraphs")
        if sec.blocks[len(template) :] != tuple(narrative):
            log.fail("V1", f"section {sec.id}: narrative paragraphs must end the section")
        for b in narrative:
            assert isinstance(b, Paragraph)
            if b.statement is not None:
                log.fail("V1", f"a narrative paragraph claims statement {b.statement!r}")
            for seg in b.segments:
                if not isinstance(seg, _NARRATIVE_SEGMENTS):
                    log.fail("V1", f"narrative segment of type {seg.type!r} in {sec.id}")
                elif isinstance(seg, FactRef) and seg.form != "full":
                    log.fail("V1", f"narrative fact atom {seg.id} is not in its full form")
    if narrated != (memo.narrative == "llm") or (memo.narrative == "llm") != bool(
        memo.narrative_model
    ):
        log.fail("V1", f"narrative flag {memo.narrative!r} does not match the paragraphs")


def _v2_atoms(memo: Memo, sheet: FactSheet, raw: _Raw, log: _Log) -> bool:
    """Every atom resolves; returns whether the memo can be rendered."""
    ok = True
    corpus = load_corpus()
    signals = set(sheet.signal_ids())
    checked: set[str] = set()
    for seg in _segments(memo.sections):
        if isinstance(seg, FactRef):
            if not sheet.has(seg.id):
                log.fail("V2", f"fact atom {seg.id!r} names no fact")
                ok = False
            elif seg.id not in checked:
                checked.add(seg.id)
                _check_sources(sheet.fact(seg.id), raw, log)
        elif isinstance(seg, FateRef) and seg.signal not in signals:
            log.fail("V2", f"fate atom names unknown signal {seg.signal!r}")
            ok = False
        elif isinstance(seg, CiteRef | BibRef) and seg.key not in corpus:
            log.fail("V2", f"citation key {seg.key!r} is not in the corpus")
            ok = False
        elif isinstance(seg, TermRef) and seg.term not in TERMS:
            log.fail("V2", f"unknown term {seg.term!r}")
            ok = False
    return ok


def _raw_scalar(v: object) -> object:
    return str(v).lower() if isinstance(v, bool) else v


def _check_sources(fact: Fact, raw: _Raw, log: _Log) -> None:
    try:
        values = [_raw_scalar(raw.resolve(s)) for s in fact.sources]
    except (KeyError, IndexError, ValueError, OSError) as exc:
        log.fail("V2", f"{fact.id}: a source does not resolve ({exc!r})")
        return
    stated: list[object] = [fact.value]
    if fact.ci is not None:
        stated += list(fact.ci)
    if fact.ci_confidence is not None:
        stated.append(fact.ci_confidence)
    for src, got, want in zip(fact.sources, values, stated, strict=True):
        same = (
            math.isclose(float(got), float(want), rel_tol=0, abs_tol=0)
            if isinstance(got, int | float) and isinstance(want, int | float)
            else str(got) == str(want)
        )
        if not same:
            log.fail("V2", f"{fact.id}: {src} is {got!r}, the fact says {want!r}")


_NUMBER: Final = re.compile(r"(<\s*)?(?<![\w.])(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?![\w])")
_CONSTANT_NUMBERS: Final = re.compile(r"Stage \d\b|ADR \d{4}\b|\b\d{1,3}% interval\b")


def _displayed_close(token: str, value: float) -> bool:
    """``token`` is ``value`` rounded to the token's displayed precision."""
    mantissa, _, exp = token.partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    unit = 10.0 ** (-decimals) * (10.0 ** int(exp) if exp else 1.0)
    return abs(float(token) - value) <= unit / 2 * (1 + 1e-9) + 1e-15


def _v3_atom_numbers(fact: Fact, text: str, raw: _Raw, log: _Log) -> None:
    try:
        values = [_raw_scalar(raw.resolve(s)) for s in fact.sources]
    except (KeyError, IndexError, ValueError, OSError):
        return  # reported by V2
    if fact.kind in (FactKind.TEXT, FactKind.DATE):
        if str(values[0]) not in text:
            log.fail("V3", f"{fact.id}: {values[0]!r} is not in its atom text {text!r}")
        return
    if fact.ci_confidence is not None:
        conf = values[-1]
        assert isinstance(conf, int | float)
        if f"{round(float(conf) * 100)}% interval" not in text:
            log.fail("V3", f"{fact.id}: the interval's confidence is not stated as {conf}")
    tokens = _NUMBER.findall(_CONSTANT_NUMBERS.sub(" ", text))
    want = values[:3] if fact.ci is not None else values[:1]
    if len(tokens) != len(want):
        log.fail("V3", f"{fact.id}: {len(tokens)} numbers in {text!r}, expected {len(want)}")
        return
    for (lt, tok), v in zip(tokens, want, strict=True):
        if not isinstance(v, int | float):
            log.fail("V3", f"{fact.id}: raw value {v!r} is not a number")
            continue
        if fact.kind is FactKind.COUNT:
            good = not lt and "." not in tok and int(tok) == v
        elif lt:
            good = fact.kind is FactKind.P_VALUE and float(v) < float(tok)
        else:
            good = _displayed_close(tok, float(v))
        if not good:
            log.fail("V3", f"{fact.id}: {(lt + tok).strip()!r} does not show raw value {v!r}")


def _v3_numbers(memo: Memo, sheet: FactSheet, md: str, raw: _Raw, log: _Log) -> None:
    out = render(memo, sheet)
    covered = bytearray(len(out.text))
    for sp in out.spans:
        if not isinstance(sp.segment, Text):
            covered[sp.start : sp.end] = b"\x01" * (sp.end - sp.start)
        if isinstance(sp.segment, FactRef):
            _v3_atom_numbers(sheet.fact(sp.segment.id), out.text[sp.start : sp.end], raw, log)
    stray = [m.start() for m in re.finditer(r"\d", out.text) if not covered[m.start()]]
    for i in stray[:5]:
        log.fail("V3", f"a digit outside any atom: {out.text[max(0, i - 30) : i + 10]!r}")
    if md == out.text:
        return
    # memo.md differs from the rendering (V9); its digits are checked against no atom.
    if re.search(r"\d", md) and not stray:
        extra = sorted(set(_NUMBER.findall(md)) - set(_NUMBER.findall(out.text)))
        for lt, tok in extra[:5]:
            log.fail("V3", f"memo.md states {(lt + tok).strip()!r}, which no atom renders")


def _v4_v7_v8_narrative(memo: Memo, sheet: FactSheet, raw: _Raw, log: _Log) -> None:
    signals = set(raw.signals()) | set(sheet.signal_ids())
    signal_re = (
        re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, sorted(signals))) + r")(?![\w])")
        if signals
        else None
    )
    surnames = _surnames()
    for sec in memo.sections:
        for block in sec.blocks:
            if not _is_narrative(block):
                continue
            assert isinstance(block, Paragraph)
            texts = [s.text for s in block.segments if isinstance(s, Text)]
            joined = " \x00 ".join(
                s.text if isinstance(s, Text) else "\x00" for s in block.segments
            )
            for t in texts:
                if m := _CONTROL.search(t):
                    log.fail("V4", f"{sec.id}: narrative character {m.group(0)!r} in {t!r}")
                if any(ch.isdigit() for ch in t):
                    log.fail("V4", f"{sec.id}: a digit in narrative text {t!r}")
            for name, pattern in (
                ("number or quantity word", _NUMBER_WORDS),
                ("frequency or probability word", _FREQUENCY_WORDS),
                ("stage name", _STAGE_WORDS),
                ("verdict word", _VERDICT_WORDS),
                ("causal claim", _CAUSAL_WORDS),
                ("performance claim", _PERFORMANCE_WORDS),
                ("recommendation word", _RECOMMENDATION_WORDS),
            ):
                if m := pattern.search(joined):
                    log.fail("V4", f"{sec.id}: {name} {m.group(0)!r} in narrative text")
            if (signal_re and (m := signal_re.search(joined))) or (
                m := _SIGNAL_LIKE.search(joined)
            ):
                log.fail("V4", f"{sec.id}: signal id {m.group(0)!r} in narrative text")
            if m := _BANNED.search(joined):
                log.fail("V8", f"{sec.id}: banned phrase {m.group(0)!r} in narrative text")
            for sentence in _sentences(block.segments):
                text = " ".join(s.text for s in sentence if isinstance(s, Text))
                cue = _CITE_CUES.search(text) or surnames.search(text)
                if cue and not any(isinstance(s, CiteRef) for s in sentence):
                    log.fail("V7", f"{sec.id}: literature cue {cue.group(0)!r} without a citation")


def _sentences(segments: Sequence[Segment]) -> list[list[Segment]]:
    out: list[list[Segment]] = [[]]
    for seg in segments:
        if not isinstance(seg, Text):
            out[-1].append(seg)
            continue
        parts = re.split(r"(?<=[.!?])\s+", seg.text)
        for i, part in enumerate(parts):
            if i:
                out.append([])
            if part:
                out[-1].append(Text(text=part))
    return [s for s in out if s]


_CLAUSE_SPLIT: Final = re.compile(r"(?<=[.;:!?])\s+|\n|\|")
_CLAUSE_VERDICT: Final = _words(
    r"pass|passes|passed|passing|kill|kills|killed|killing|surviv\w*|fail\w*|reject\w*",
    r"held\s+up|holds?\s+up|ruled?\s+out|dropped|eliminated",
)
_PASSED: Final = re.compile(r"^(?P<sid>\S+) passed all three kill stages$")
_KILLED: Final = re.compile(
    r"^(?P<sid>\S+) was killed at Stage (?P<n>[1-3]) \([a-z ]+\) "
    r"(?P<how>by its test|without a test decision) \((?P<reason>[a-z_.]+)\)$"
)


def _v5_verdicts(memo: Memo, sheet: FactSheet, md: str, raw: _Raw, log: _Log) -> None:
    want = raw.recommendation()
    if memo.recommendation is not want or sheet.recommendation is not want:
        log.fail("V5", f"recommendation {memo.recommendation!r}; the raw ledger gives {want!r}")
    stages = [str(s) for s in MEMO_STAGES]
    signals = raw.signals()
    sid_re = re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, signals)) + r")(?![\w])")
    for clause in _CLAUSE_SPLIT.split(md):
        clause = re.sub(r"^>\s*", "", clause.strip()).rstrip(".;:!?").strip()
        ids = sid_re.findall(clause)
        if not ids or not _CLAUSE_VERDICT.search(clause):
            continue
        m = _PASSED.match(clause) or _KILLED.match(clause)
        if m is None or len(ids) != 1:
            log.fail("V5", f"a verdict clause not in canonical form: {clause[:120]!r}")
            continue
        fate = raw.fate(m.group("sid"))
        if "n" not in m.groupdict():
            ok = fate.passed
        else:
            ok = (
                not fate.passed
                and fate.stage == stages[int(m.group("n")) - 1]
                and fate.reason == m.group("reason")
                and (m.group("how") == "without a test decision")
                == bool(fate.reason and fate.reason.startswith("general."))
            )
        if not ok:
            log.fail("V5", f"{clause!r} disagrees with the raw ledger ({fate})")


def _raw_triggers(raw: _Raw) -> set[str]:
    """The mandatory statements' triggers, recomputed from the raw ledger (``mandatory.py``)."""
    out = {"data_source", "stage4_not_adopted"}
    reasons = {
        str(e["reason"]) for e in raw.entries if e["verdict"] == "kill" and e["reason"] is not None
    }
    causal = raw.stage_entries("causal")
    decay = raw.stage_entries("decay")
    port = raw.stage_entries("portfolio")
    if any(e["verdict"] == "pass" for e in causal):
        out.add("hidden_confounder")
    if reasons & {
        "causal.no_significant_link",
        "causal.not_significant_after_correction",
        "portfolio.no_edge_over_equal_weight",
    }:
        out.add("not_detectable")
    if any(r.startswith("general.") for r in reasons):
        out.add("not_decided")
    if any("r_best" in _metrics(e) for e in causal):
        out.add("stage1_descriptive")
    if decay:
        out.add("no_half_life")
    if any(e["verdict"] == "pass" for e in decay):
        out.add("decay_power")
    result = raw.run_dir / "portfolio_result.json"
    doc = raw.doc(result) if port and result.is_file() else None
    if isinstance(doc, dict) and isinstance(doc.get("evidence"), dict):
        out |= {"portfolio_method", "gross_sharpe"}
    if port:
        if "ridge_cv_dsr" in _metrics(port[0]):
            out.add("deflated_sharpe")
        if "stress_ridge_cv_sharpe_high" in _metrics(port[0]):
            out.add("stress_description")
        if len({e["signal_id"] for e in port}) >= 2:
            out.add("passenger" if port[0]["verdict"] == "pass" else "collateral")
    return out


def _metrics(entry: dict[str, object]) -> dict[str, object]:
    m = entry["metrics"]
    assert isinstance(m, dict)
    return m


def _v6_mandatory(memo: Memo, sheet: FactSheet, raw: _Raw, log: _Log) -> None:
    want = [m.id for m in triggered(sheet)]
    if set(want) != _raw_triggers(raw):
        log.fail("V6", f"triggered statements {sorted(want)} disagree with the raw ledger")
    found: list[str] = []
    for sec in memo.sections:
        for block in sec.blocks:
            if isinstance(block, Paragraph) and block.statement is not None:
                found.append(block.statement)
                if sec.id != "limits" or block.origin != "template":
                    log.fail("V6", f"statement {block.statement!r} is not a template limit")
    if found != want:
        log.fail("V6", f"statements {found} are not the triggered set {want}")


def verify(
    memo_json: str, memo_md: str, run_dir: Path, *, results_root: Path | None = None
) -> VerificationReport:
    """Verify a memo's two files against its run directory (module docstring)."""
    log = _Log()
    root = results_root if results_root is not None else default_results_root()
    try:
        memo = Memo.model_validate_json(memo_json)
    except ValidationError as exc:
        return VerificationReport((Failure("V1", f"memo.json is not a memo: {exc}"),))
    try:
        sheet = build_fact_sheet(run_dir, results_root=root)
        raw = _Raw(Path(run_dir), root)
    except (MemoInputError, OSError, ValueError) as exc:
        return VerificationReport((Failure("V1", f"the run cannot support a memo: {exc}"),))
    if memo.run_id != sheet.run_id:
        log.fail("V1", f"memo run id {memo.run_id!r} is not the run's {sheet.run_id!r}")
    _v1_structure(memo, sheet, log)
    renders = _v2_atoms(memo, sheet, raw, log)
    if renders:
        _v3_numbers(memo, sheet, memo_md, raw, log)
        if memo_md != render(memo, sheet).text:
            log.fail("V9", "memo.md is not the rendering of memo.json")
    else:
        log.fail("V9", "memo.json does not render")
    _v4_v7_v8_narrative(memo, sheet, raw, log)
    _v5_verdicts(memo, sheet, memo_md, raw, log)
    _v6_mandatory(memo, sheet, raw, log)
    return VerificationReport(tuple(log.failures))


def verify_run(run_dir: Path, *, results_root: Path | None = None) -> VerificationReport:
    """Verify ``<run_dir>/memo/memo.json`` and ``memo.md``."""
    d = Path(run_dir) / "memo"
    return verify(
        (d / "memo.json").read_text(encoding="utf-8"),
        (d / "memo.md").read_text(encoding="utf-8"),
        run_dir,
        results_root=results_root,
    )
