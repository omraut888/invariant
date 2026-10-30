"""Check every DOI in the memo's literature corpus against Crossref (dev only; network).

Prints, per entry, whether the DOI resolves and whether Crossref's year, title and first author
agree with the corpus. Not part of the test suite: it calls a public API.

Usage: uv run python scripts/check_literature_dois.py
"""

from __future__ import annotations

import time

import httpx

from invariant.stages.memo.literature import load_corpus


def main() -> None:
    for e in load_corpus().values():
        if not e.doi:
            print(f"no DOI   {e.key}")
            continue
        r = httpx.get(f"https://api.crossref.org/works/{e.doi}", timeout=20)
        if r.status_code != 200:
            print(f"MISSING  {e.key} {e.doi} ({r.status_code})")
            continue
        m = r.json()["message"]
        year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        first = (m.get("author") or [{}])[0].get("family", "")
        title = (m.get("title") or [""])[0]
        ok = "ok" if year == e.year and first.lower() in e.authors[0].lower() else "CHECK"
        print(f"{ok:8s} {e.key}: crossref {year} {first}: {title[:70]}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
