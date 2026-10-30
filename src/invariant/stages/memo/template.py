"""The template memo: the deterministic memo built from the fact sheet alone (docs/adr/0017).

The template memo is the product. It states every verdict, the numbers behind it, the mandatory
statements that apply, the methods with their citations, and how to reproduce the run. An
optional LLM narrative (``narrative.py``) may add prose to it, but never replaces any of it.

Templates are written in a small atom syntax, parsed by ``parse``:
``{fact:<id>}``, ``{fact:<id>|value}``, ``{fate:<signal>}``, ``{stage:<stage>}``,
``{cite:<key>}``, ``{adr:<n>}``, ``{term:<t>}`` and ``{recommendation}``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from invariant.core.types import StageName
from invariant.ledger.kill_ledger import FateStatus
from invariant.stages.dro.walkforward import Arm
from invariant.stages.memo.document import (
    AdrRef,
    BibRef,
    Block,
    CiteRef,
    FactRef,
    FateRef,
    Memo,
    Paragraph,
    RecRef,
    Section,
    Segment,
    StageRef,
    Table,
    TermRef,
    Text,
)
from invariant.stages.memo.facts import ARM_LABELS, FactSheet, Recommendation
from invariant.stages.memo.mandatory import triggered

__all__ = ["NARRATIVE_SECTIONS", "build_template_memo", "cited_keys", "parse", "references_section"]

_ATOM: Final = re.compile(
    r"\{(fact|fate|stage|cite|adr|term|recommendation)(?::([^}|]+))?(?:\|(value))?\}"
)

NARRATIVE_SECTIONS: Final = ("recommendation", "causal", "decay", "portfolio", "stress")
"""Sections an LLM narrative may add paragraphs to, at their end."""


def parse(template: str) -> tuple[Segment, ...]:
    """Template text to segments (module docstring). Unknown atoms raise ``ValueError``."""
    out: list[Segment] = []
    pos = 0
    for m in _ATOM.finditer(template):
        if m.start() > pos:
            out.append(Text(text=template[pos : m.start()]))
        kind, arg, form = m.group(1), m.group(2), m.group(3)
        if kind != "recommendation" and not arg:
            raise ValueError(f"atom {m.group(0)!r} needs an argument")
        match kind:
            case "fact":
                out.append(FactRef(id=str(arg), form="value" if form else "full"))
            case "fate":
                out.append(FateRef(signal=str(arg)))
            case "stage":
                out.append(StageRef(stage=StageName(arg)))
            case "cite":
                out.append(CiteRef(key=str(arg)))
            case "adr":
                out.append(AdrRef(number=int(str(arg))))
            case "term":
                out.append(TermRef(term=str(arg)))
            case "recommendation":
                out.append(RecRef())
        pos = m.end()
    if pos < len(template):
        out.append(Text(text=template[pos:]))
    if any("{" in s.text or "}" in s.text for s in out if isinstance(s, Text)):
        raise ValueError(f"unparsed atom in template: {template!r}")
    return tuple(out)


def _p(template: str, statement: str | None = None) -> Paragraph:
    return Paragraph(segments=parse(template), statement=statement)


def _cell(template: str) -> tuple[Segment, ...]:
    return parse(template)


def _v(sheet: FactSheet, fact_id: str, missing: str = "not reported") -> str:
    return f"{{fact:{fact_id}|value}}" if sheet.has(fact_id) else missing


def _recommendation(sheet: FactSheet) -> Section:
    passed = [f.signal_id for f in sheet.fates if f.status is FateStatus.PASSED]
    match sheet.recommendation:
        case Recommendation.PAPER_TRADING_CANDIDATE:
            why = (
                "At least one signal survived every kill stage: "
                + "; ".join(f"{{fate:{s}}}" for s in passed)
                + ". That makes the strategy a candidate for paper trading, not a recommendation "
                "to trade. The evidence is one sample and a test on gross Sharpe ratios, with "
                "the limits stated below."
            )
        case Recommendation.AGAINST:
            why = (
                "No signal survived every kill stage, so there is no strategy to trade. Each "
                "signal's outcome, and the numbers behind it, follow."
            )
        case Recommendation.NOT_DECIDED:
            why = (
                "No signal survived every kill stage, but at least one was killed without a test "
                "decision, because the data were insufficient or invalid for the stage. That "
                "signal was not ruled out, so the run supports no conclusion either way."
            )
    blocks: list[Block] = [_p("{recommendation}."), _p(why)]
    return Section(id="recommendation", title="Recommendation", blocks=tuple(blocks))


def _signals(sheet: FactSheet) -> Section:
    rows = tuple((_cell(s), _cell(f"{{fate:{s}}}")) for s in sheet.signal_ids())
    return Section(
        id="signals",
        title="Signals",
        blocks=(Table(header=("Signal", "Outcome over the kill stages"), rows=rows),),
    )


def _causal(sheet: FactSheet) -> Section | None:
    sids = [s for s in sheet.signal_ids() if StageName.CAUSAL in sheet.fate(s).stages_decided]
    if not sids:
        return None
    rows = []
    for s in sids:
        p = f"causal.{s}."
        if sheet.has(p + "family_p"):
            rows.append(
                (
                    _cell(s),
                    _cell(_v(sheet, p + "family_p")),
                    _cell(_v(sheet, p + "q")),
                    _cell(f"{_v(sheet, p + 'best_asset')}, lag {_v(sheet, p + 'best_lag')}"),
                    _cell(_v(sheet, p + "r_best")),
                )
            )
        else:
            issue = _v(sheet, p + "data_issue", "no evidence")
            rows.append((_cell(s), _cell("not tested"), _cell(issue), _cell(""), _cell("")))
    intro = (
        "Each signal's lagged link to the asset returns, given the observed variables. The "
        "verdict uses the BH q-value at the {fact:causal." + sids[0] + ".alpha|value} level."
    )
    blocks: tuple[Block, ...] = (
        _p(intro),
        Table(
            header=("Signal", "Family p", "BH q", "Best link", "Partial correlation"),
            rows=tuple(rows),
        ),
    )
    return Section(id="causal", title="Causal screen", blocks=blocks)


def _decay(sheet: FactSheet) -> Section | None:
    sids = [s for s in sheet.signal_ids() if StageName.DECAY in sheet.fate(s).stages_decided]
    if not sids:
        return None
    rows = []
    for s in sids:
        p = f"decay.{s}."
        if sheet.has(p + "p"):
            rows.append(
                (
                    _cell(s),
                    _cell(_v(sheet, p + "p")),
                    _cell(_v(sheet, p + "b_pre")),
                    _cell(_v(sheet, p + "b_post")),
                    _cell(_v(sheet, p + "break_fraction")),
                )
            )
        else:
            issue = _v(sheet, p + "data_issue", "no evidence")
            rows.append((_cell(s), _cell("not tested"), _cell(issue), _cell(""), _cell("")))
    intro = (
        "Whether the coefficient on each signal's best link declined over the sample. A "
        "decline is a kill at the {fact:decay." + sids[0] + ".alpha|value} level."
    )
    blocks: tuple[Block, ...] = (
        _p(intro),
        Table(
            header=("Signal", "Decline p", "Coefficient before", "Coefficient after", "Break at"),
            rows=tuple(rows),
        ),
    )
    return Section(id="decay", title="Decay test", blocks=blocks)


def _portfolio(sheet: FactSheet) -> Section | None:
    if not sheet.members:
        return None
    blocks: list[Block] = []
    if not sheet.portfolio_ran:
        blocks.append(
            _p(
                "The portfolio was not built: "
                + _v(sheet, "portfolio.data_issue", "no evidence")
                + "."
            )
        )
        return Section(id="portfolio", title="Portfolio construction", blocks=tuple(blocks))
    blocks.append(
        _p(
            "Members: " + ", ".join(sheet.members) + ". The deciding arm is Ridge-CV, tested "
            "against {term:one_over_n} at the {fact:portfolio.lw_alpha|value} level, over "
            "{fact:portfolio.n_test_days|value} out-of-sample days."
        )
    )
    rows = []
    for arm, name in ARM_LABELS.items():
        k = f"portfolio.{arm.value}."
        rows.append(
            (
                _cell(name),
                _cell(_v(sheet, k + "sharpe")),
                _cell(_v(sheet, k + "sharpe_net")),
                _cell(_v(sheet, k + "dsr", "")),
                _cell(_v(sheet, k + "lw_p", "")),
                _cell(_v(sheet, k + "turnover")),
                _cell("yes" if arm is Arm.RIDGE_CV else "no"),
            )
        )
    blocks.append(
        Table(
            header=(
                "Arm",
                "Gross Sharpe",
                "Net Sharpe",
                "Deflated Sharpe",
                "LW p vs equal weight",
                "Turnover",
                "Decides",
            ),
            rows=tuple(rows),
        )
    )
    b_rows = tuple((_cell(m), _cell(_v(sheet, f"portfolio.{m}.b"))) for m in sheet.members)
    blocks.append(Table(header=("Member", "Diagnostic B (never decides)"), rows=b_rows))
    return Section(id="portfolio", title="Portfolio construction", blocks=tuple(blocks))


def _stress(sheet: FactSheet) -> Section | None:
    if not sheet.has("stress.ridge_cv_high"):
        return None
    rows = (
        (
            _cell("Ridge-CV"),
            _cell(_v(sheet, "stress.ridge_cv_high")),
            _cell(_v(sheet, "stress.ridge_cv_low")),
        ),
        (_cell("equal weight"), _cell(_v(sheet, "stress.equal_weight_high")), _cell("")),
    )
    blocks: tuple[Block, ...] = (
        _p(
            "Reported only; it never decides. Days are labelled by trailing volatility, and "
            "{fact:stress.high_share|value} of them fall in the high-volatility state."
        ),
        Table(header=("Arm", "Sharpe, high-volatility state", "Sharpe, other days"), rows=rows),
    )
    return Section(id="stress", title="Stress diagnostic", blocks=blocks)


def _limits(sheet: FactSheet) -> Section:
    blocks = tuple(_p(m.text(sheet), statement=m.id) for m in triggered(sheet))
    return Section(id="limits", title="Limits that apply to this run", blocks=blocks)


def _methods(sheet: FactSheet) -> Section:
    text: list[str] = []
    reached = {s for f in sheet.fates for s in f.stages_decided}
    if StageName.CAUSAL in reached:
        text.append(
            "{stage:causal}: PCMCI ({cite:runge2019}) with an {term:hc3} partial-correlation test "
            "({cite:mackinnon1985}), Bonferroni over each signal's lags and assets, and the "
            "Benjamini-Hochberg procedure across signals ({cite:benjamini1995}) ({adr:9}, "
            "{adr:10}). Persistent predictors distort such tests in small samples "
            "({cite:stambaugh1999}), which is why the stage requires a minimum sample."
        )
    if StageName.DECAY in reached:
        text.append(
            "{stage:decay}: a one-sided sup-split test of a decline in the coefficient on the "
            "best link ({cite:andrews1993}), with a fixed-regressor wild-bootstrap p-value "
            "({cite:hansen2000}) ({adr:11}, {adr:12})."
        )
    if sheet.portfolio_ran:
        text.append(
            "{stage:portfolio}: mean-standard-deviation portfolios on linear forecasts from the "
            "surviving signals, with gross exposure capped and refit walk-forward. The kill "
            "test is the one-sided Ledoit-Wolf test of the Sharpe-ratio difference against "
            "{term:one_over_n} ({cite:ledoit2008}). The Wasserstein DRO arms follow "
            "({cite:esfahani2018}) ({adr:13}, {adr:14})."
        )
    if sheet.has("stress.ridge_cv_high"):
        text.append(
            "Stress diagnostic: each day is labelled by trailing volatility, and Sharpe ratios "
            "in the high state get a moving-block bootstrap interval ({cite:kunsch1989}) with the "
            "block length of ({cite:politis2004}; {cite:patton2009}) ({adr:16})."
        )
    text.append("Benchmark validation rates carry Wilson intervals ({cite:wilson1927}).")
    return Section(id="methods", title="Methods", blocks=tuple(_p(t) for t in text))


def _reproducibility(sheet: FactSheet) -> Section:
    dates = (
        ", from {fact:run.first_date|value} to {fact:run.last_date|value}"
        if sheet.has("run.first_date") and sheet.has("run.last_date")
        else ""
    )
    return Section(
        id="reproducibility",
        title="Reproducibility",
        blocks=(
            _p(
                "{fact:run.run_id}; {fact:run.snapshot_id}; {fact:run.git_sha}; {fact:run.seed}; "
                "{fact:run.n_obs}" + dates + ". A run is keyed by a hash of its data snapshot, "
                "config, code commit and seed ({adr:2}); every decision is in its hash-chained "
                "kill ledger ({adr:7})."
            ),
        ),
    )


def cited_keys(sections: Iterable[Section]) -> tuple[str, ...]:
    """Citation keys in order of first appearance."""
    seen: dict[str, None] = {}
    for sec in sections:
        for block in sec.blocks:
            cells: list[tuple[Segment, ...]] = (
                [block.segments]
                if isinstance(block, Paragraph)
                else [c for r in block.rows for c in r]
            )
            for cell in cells:
                for seg in cell:
                    if isinstance(seg, CiteRef):
                        seen.setdefault(seg.key, None)
    return tuple(seen)


def references_section(sections: Iterable[Section]) -> Section:
    blocks = tuple(Paragraph(segments=(BibRef(key=k),)) for k in sorted(cited_keys(sections)))
    return Section(id="references", title="References", blocks=blocks)


def build_template_memo(sheet: FactSheet) -> Memo:
    """The template memo for ``sheet`` (module docstring). Deterministic."""
    parts = [
        _recommendation(sheet),
        _signals(sheet),
        _causal(sheet),
        _decay(sheet),
        _portfolio(sheet),
        _stress(sheet),
        _limits(sheet),
        _methods(sheet),
        _reproducibility(sheet),
    ]
    sections = [s for s in parts if s is not None]
    sections.append(references_section(sections))
    return Memo(
        run_id=sheet.run_id,
        recommendation=sheet.recommendation,
        narrative="none",
        title="Research memo",
        sections=tuple(sections),
    )
