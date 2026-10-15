"""GM's own adjusted automotive free cash flow, from its 10-K, against the capex proposal's number.

For the review of the open capex proposal. The proposal would publish GM's free cash flow as
consolidated operating cash flow minus industrial capex. GM publishes **automotive** operating
cash flow minus capital expenditures. Whether those two are close is the question the decision
turns on, and it is answerable only from GM's own reconciliation, which is in the 10-K's
liquidity section (the segment columns are not in `companyfacts`).

Prints every passage in the newest 10-K that mentions automotive operating cash flow or adjusted
automotive free cash flow.

    python research/measurements/2026-10-15-gm-automotive-ocf.py [TICKER] [PATTERN]
"""
from __future__ import annotations

import re
import sys
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TICKER = (sys.argv[1] if len(sys.argv) > 1 else "GM").upper()
PATTERN = re.compile(sys.argv[2] if len(sys.argv) > 2
                     else r"adjusted automotive free cash flow|automotive net cash provided|"
                          r"net automotive cash provided", re.I)


def text_of(html: str) -> str:
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", unescape(t))


def main() -> None:
    from insider_activity import Edgar, ticker_map, user_agent
    edgar = Edgar(user_agent())
    cik = int(ticker_map(edgar)[TICKER])
    sub = edgar.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    r = sub["filings"]["recent"]
    rows = [(r["filingDate"][i], r["accessionNumber"][i], r["primaryDocument"][i])
            for i, f in enumerate(r["form"]) if f == "10-K"]
    if not rows:
        raise SystemExit(f"no 10-K in {TICKER}'s recent filings")
    date, acc, doc = max(rows)
    acc = acc.replace("-", "")
    print(f"{TICKER} 10-K filed {date}: {doc}\n")
    body = text_of(edgar.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}").text)
    hits = list(PATTERN.finditer(body))
    print(f"{len(hits)} passage(s)\n")
    for m in hits:
        print("..." + body[max(0, m.start() - 500): m.end() + 700].strip() + "...\n")


if __name__ == "__main__":
    main()
