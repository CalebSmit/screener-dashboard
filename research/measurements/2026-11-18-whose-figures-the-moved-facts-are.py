"""Are the facts the SEC filed under a co-registrant the listed parent's own figures? - 2026-11-18

`2026-11-18-where-the-missing-filings-went.py` found that 19 companies' 10-Qs since May sit in a
subsidiary's or operating partnership's `companyfacts` record. Before anything reads them as the
parent's, that has to be shown, and a filing carries its own test: every 10-Q repeats earlier
figures as comparatives (the prior year-end balance sheet, the same quarter a year before). Those
same (concept, unit, start, end) facts are already in the parent's own record from the filings
the SEC *did* attribute to it.

For each moved accession this prints how many of its facts overlap the parent's own record on
(concept, unit, start, end), how many of those are equal, and the same comparison against the
co-registrant's *own earlier* record (its facts from accessions before the move) - if the moved
facts were the subsidiary's, that second number would be the high one.

Also asks the frames API whether Citigroup's 2026 10-Qs are anywhere in the SEC's XBRL data.

Network: one companyfacts request per registrant, kept in the gitignored `logs/cf_cache/`.

    python research/measurements/2026-11-18-whose-figures-the-moved-facts-are.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from insider_activity import Edgar, ticker_map, user_agent  # noqa: E402

CACHE = ROOT / "logs" / "cf_cache"
# ticker -> the co-registrants whose record held its accessions (printed by the first script)
MOVED = {"AEP": [81027, 92487], "CNP": [1042773], "D": [103682], "DLR": [1494877], "DTE": [28385],
         "DUK": [17797], "ETR": [7323], "EVRG": [54507], "EXC": [27879, 78100], "FE": [2152026],
         "FRT": [1901876], "LNT": [52485, 107832], "MAA": [1581776], "NEE": [37634], "PLD": [1045610],
         "PPL": [60549, 317187], "REG": [1066247], "SO": [66904, 1004155], "SRE": [86521, 92108]}
FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")
MOVE_FROM = "2026-05-15"


def record(edgar: Edgar, cik: int) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{cik}.json"
    if not p.exists():
        p.write_text(edgar.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json").text,
                     encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def flat(j: dict) -> list[tuple]:
    out = []
    for ns, concepts in (j.get("facts") or {}).items():
        for c, body in concepts.items():
            for unit, facts in (body.get("units") or {}).items():
                for f in facts:
                    if f.get("form") in FORMS:
                        out.append(((ns, c, unit, f.get("start"), f.get("end")), f.get("val"),
                                    f.get("accn"), f.get("filed")))
    return out


def agree(moved: dict, reference: dict) -> tuple[int, int]:
    both = [k for k in moved if k in reference]
    return len(both), sum(1 for k in both if moved[k] == reference[k])


def main() -> int:
    edgar = Edgar(user_agent())
    cmap = ticker_map(edgar)
    print("ticker  co-registrant  moved accession       facts  vs parent's own record   vs co-registrant's earlier record")
    tot = [0, 0, 0, 0]
    for t, cos in MOVED.items():
        own = flat(record(edgar, cmap[t]))
        # the parent's own record: the newest filed value per key
        parent = {}
        for k, v, _a, filed in sorted(own, key=lambda r: r[3] or ""):
            parent[k] = v
        for co in cos:
            rows = flat(record(edgar, co))
            earlier = {}
            for k, v, _a, filed in sorted(rows, key=lambda r: r[3] or ""):
                if (filed or "") < MOVE_FROM:
                    earlier[k] = v
            for accn in sorted({a for _k, _v, a, filed in rows if (filed or "") >= MOVE_FROM}):
                moved = {k: v for k, v, a, _f in rows if a == accn}
                n_p, eq_p = agree(moved, parent)
                n_e, eq_e = agree(moved, earlier)
                for i, x in enumerate((n_p, eq_p, n_e, eq_e)):
                    tot[i] += x
                print(f"{t:6}  {co:>12}  {accn}  {len(moved):5}  {eq_p:4} of {n_p:4} equal"
                      f"{'':8}{eq_e:4} of {n_e:4} equal")
    print(f"\nall moved accessions: {tot[1]} of {tot[0]} overlapping facts equal the parent's own record "
          f"({100 * tot[1] / max(tot[0], 1):.1f}%); {tot[3]} of {tot[2]} equal the co-registrant's own earlier "
          f"record ({100 * tot[3] / max(tot[2], 1):.1f}%)")

    # The baseline that number needs: how often does a filing the SEC did NOT move agree with the
    # same company's earlier filings? (Comparatives are restated and re-rounded in the ordinary
    # course.) Each parent's newest accession in its own record against everything filed before it.
    base = [0, 0]
    print("\nbaseline - each parent's last un-moved filing against its own earlier record:")
    for t in MOVED:
        own = flat(record(edgar, cmap[t]))
        last_filed, last_accn = max((f or "", a) for _k, _v, a, f in own)
        earlier = {}
        for k, v, _a, filed in sorted(own, key=lambda r: r[3] or ""):
            if (filed or "") < last_filed:
                earlier[k] = v
        n, eq = agree({k: v for k, v, a, _f in own if a == last_accn}, earlier)
        base[0] += n
        base[1] += eq
        print(f"  {t:5} {last_accn} filed {last_filed}: {eq} of {n} equal")
    print(f"baseline: {base[1]} of {base[0]} ({100 * base[1] / max(base[0], 1):.1f}%)")

    # Citigroup: one registrant, three XBRL 10-Qs in 2026, none in its companyfacts record.
    cik = cmap["C"]
    for frame in ("CY2026Q1I", "CY2026Q2I", "CY2026Q3I", "CY2025Q4I"):
        try:
            j = edgar.get(f"https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/{frame}.json").json()
        except Exception as e:  # noqa: BLE001
            print(f"frames Assets {frame}: {type(e).__name__}")
            continue
        hit = [d for d in j.get("data", []) if d.get("cik") == cik]
        print(f"frames us-gaap/Assets {frame}: {len(j.get('data', []))} filers; Citigroup: "
              f"{[(d['accn'], d['val']) for d in hit] or 'absent'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
