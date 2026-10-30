"""The curated literature corpus (``literature.yaml``, docs/adr/0017).

A memo cites only entries of this corpus. Each entry is bibliographic metadata and a one-line
summary in our own words; no text from the papers is stored. The DOIs were checked against
Crossref when the corpus was written (``scripts/check_literature_dois.py``).
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LiteratureEntry", "cite_text", "load_corpus", "reference_text"]

SUMMARY_MAX = 200


class LiteratureEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(pattern=r"^[a-z]+[0-9]{4}[a-z]?$")
    authors: tuple[str, ...] = Field(min_length=1)
    year: int = Field(ge=1800, le=2100)
    title: str
    venue: str
    doi: str | None
    summary: str = Field(max_length=SUMMARY_MAX)
    stages: tuple[str, ...]


@cache
def load_corpus() -> dict[str, LiteratureEntry]:
    raw = yaml.safe_load(files("invariant.stages.memo").joinpath("literature.yaml").read_text())
    entries = [LiteratureEntry.model_validate(e) for e in raw["entries"]]
    out = {e.key: e for e in entries}
    if len(out) != len(entries):
        raise ValueError("duplicate literature keys")
    return out


def cite_text(entry: LiteratureEntry) -> str:
    """Author-year form: ``Andrews 1993``, ``Ledoit & Wolf 2008``, ``Runge et al. 2019``."""
    a = entry.authors
    names = a[0] if len(a) == 1 else f"{a[0]} & {a[1]}" if len(a) == 2 else f"{a[0]} et al."
    return f"{names} {entry.year}"


def reference_text(entry: LiteratureEntry) -> str:
    """One bibliography line, with our summary."""
    doi = f" https://doi.org/{entry.doi}" if entry.doi else ""
    return (
        f"{', '.join(entry.authors)} ({entry.year}). {entry.title}. {entry.venue}.{doi} "
        f"Summary: {entry.summary}"
    )
