"""Build the commit subject for a data run, by reading the payload properly.

The data loop's commit subject is part of the audit trail and is what the
owner sees in the GitHub notification email, so "top:" has to name the actual
top-ranked stocks.

It used to be scraped out of ``dashboard_data.js`` with a regex for
``"ticker": "XXX"``, taking the first five matches. That worked only by
coincidence: the lowercase key belonged to the model-portfolio holdings, which
happened to be serialised in rank order. When the portfolio surface was removed
on 2026-08-26 the same regex started matching the first stock's **sector
peers** instead, and the 19:38 run committed ``top: MAA DOC KIM REG UDR`` when
the real top five were ``HST EXPE APA EIX CF``. Nothing failed; the commit log
simply began lying.

So this reads the ranking by rank, from ``table_data``, and does not depend on
key ordering, key casing, or which keys happen to exist elsewhere in the
payload.

Usage:
    python scripts/commit_subject.py <date> [path/to/dashboard_data.js]

Prints one line to stdout. Never fails the caller: on any problem it falls back
to a subject that is merely uninformative rather than wrong.
"""

import json
import re
import sys
from pathlib import Path

TOP_N = 5


def _load(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    match = re.search(r"window\.SCREENER_DATA\s*=\s*", raw)
    if not match:
        raise ValueError("not a dashboard_data.js payload")
    return json.loads(raw[match.end():].rstrip().rstrip(";"))


def build_subject(date: str, payload: dict) -> str:
    subject = "data: screener run %s" % date

    rows = payload.get("table_data") or []
    scored = (payload.get("kpis") or {}).get("universe_size")
    if scored is None and rows:
        scored = len(rows)
    if scored:
        subject += " - %s scored" % scored

    ranked = [r for r in rows if r.get("Rank") is not None]
    ranked.sort(key=lambda r: r["Rank"])
    top = [str(r["Ticker"]) for r in ranked[:TOP_N] if r.get("Ticker")]
    if top:
        subject += ", top: %s" % " ".join(top)
    return subject


def main(argv: list[str]) -> int:
    date = argv[1] if len(argv) > 1 else ""
    path = Path(argv[2]) if len(argv) > 2 else Path("dashboard_data.js")
    try:
        print(build_subject(date, _load(path)))
    except Exception:
        # An unhelpful subject is fine; a wrong one is not, and a crash here
        # must never stop a healthy run from publishing.
        print("data: screener run %s" % date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
