"""Second reading of test B in ``research/2026-10-22-free-data-that-adds-evidence.md``.

The note claims FINRA's biweekly equity short-interest files carry exchange-listed rows
(AAPL, JPM) with a ``daysToCoverQuantity`` field back to at least 2017-12-29 - against
FINRA's own page, which says exchange-listed data begins June 2021 - and that this
un-benches ``short_interest_ratio`` (13.5 of Revisions' 100 points, 1.35 composite points)
for backtest v2.

A registration must not rest on a claim nobody re-ran. This probes the files themselves:
for each settlement date, is the file served, does it carry a days-to-cover column, how
many rows does it hold, what market-class codes appear, and are the S&P 500 names there?

Read-only. Writes its result next to itself as JSON. Nothing here is decision-grade for
any weight (CLAUDE.md rule 5); it decides only whether the proposed test can be built.

    python -I -X utf8 research/measurements/2026-10-23-finra-short-interest-probe.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).with_suffix(".json")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")

# The path the note probed, verbatim.
URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{date}.csv"

# Settlement dates: the six the note reports as present, the ones it reports as 403,
# FINRA's own stated start for exchange-listed data, and a recent one.
DATES = [
    "20141231", "20151231", "20161230",          # note: "earlier dates tried returned 403"
    "20171229", "20180629", "20181231",          # note: present
    "20190115", "20191231", "20210528",          # note: present
    "20210615", "20230630", "20250630",          # after FINRA's stated June-2021 start
]

# Large exchange-listed S&P 500 members the test would need.
WANT = ["AAPL", "JPM", "MSFT", "XOM", "JNJ", "KO", "GE"]


def fetch(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:                                    # noqa: BLE001
        return -1, str(e).encode()


def probe(date: str) -> dict:
    status, body = fetch(URL.format(date=date))
    out: dict = {"settlement_date": date, "http_status": status, "bytes": len(body)}
    if status != 200 or not body:
        return out
    text = body.decode("utf-8-sig", errors="replace")
    # FINRA serves these pipe-delimited despite the .csv extension - sniff rather than assume.
    first = text.split("\n", 1)[0]
    delim = "|" if first.count("|") > first.count(",") else ","
    out["delimiter"] = delim
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    out["rows"] = len(rows)
    out["columns"] = list(rows[0].keys()) if rows else []
    # Which column, if any, is days to cover?
    dtc_col = next((c for c in out["columns"] if "daystocover" in c.lower().replace("_", "")), None)
    out["days_to_cover_column"] = dtc_col
    # Market class codes tell exchange-listed from OTC.
    cls_col = next((c for c in out["columns"] if c.lower().replace("_", "") in
                    ("marketclasscode", "marketclass")), None)
    out["market_class_column"] = cls_col
    if cls_col:
        counts: dict[str, int] = {}
        for r in rows:
            counts[(r.get(cls_col) or "").strip()] = counts.get((r.get(cls_col) or "").strip(), 0) + 1
        out["market_class_counts"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    sym_col = next((c for c in out["columns"] if c.lower().replace("_", "") in
                    ("symbolcode", "symbol")), None)
    out["symbol_column"] = sym_col
    if sym_col:
        index = {}
        for r in rows:
            index[(r.get(sym_col) or "").strip().upper()] = r
        found = {}
        for t in WANT:
            r = index.get(t)
            if r is None:
                found[t] = None
                continue
            found[t] = {
                "market_class": (r.get(cls_col) or "").strip() if cls_col else None,
                "days_to_cover": (r.get(dtc_col) or "").strip() if dtc_col else None,
                "short_position": (r.get("currentShortPositionQuantity") or "").strip(),
                "avg_daily_volume": (r.get("averageDailyVolumeQuantity") or "").strip(),
            }
        out["wanted_symbols"] = found
        out["wanted_found"] = sum(1 for v in found.values() if v)
        # How many rows carry a usable (non-empty, numeric, > 0) days to cover?
        if dtc_col:
            usable = 0
            for r in rows:
                v = (r.get(dtc_col) or "").strip()
                try:
                    if float(v) > 0:
                        usable += 1
                except ValueError:
                    pass
            out["rows_with_usable_days_to_cover"] = usable
    return out


def main() -> int:
    results = []
    for d in DATES:
        r = probe(d)
        results.append(r)
        print(f"{d}  http={r['http_status']:>4}  rows={r.get('rows','-'):>8}  "
              f"dtc_col={r.get('days_to_cover_column')}  "
              f"wanted_found={r.get('wanted_found','-')}  "
              f"classes={list((r.get('market_class_counts') or {}).items())[:4]}")
        time.sleep(1.0)
    OUT.write_text(json.dumps({"url_template": URL, "wanted": WANT, "probes": results},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
