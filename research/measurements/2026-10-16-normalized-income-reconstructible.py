"""Can "Normalized Income" be reconstructed from figures the page could show? - 2026-10-16

The open proposal in ``research/2026-10-15-earnings-definition.md`` would score earnings yield on
the source's "Normalized Income" instead of net income as reported. CLAUDE.md row 0.8c says a
scoring change ships with its frontend and that an equation is shown on the metric's row **only if
it reproduces**; row 0.8 says the page shows the engine's numbers and never a re-derivation. So
before the proposal is judged on its evidence, one thing has to be true of its input: a reader must
be able to get from figures on the page to the number the screener scored.

The source serves five rows that bear on it, and the documented construction is

    Normalized Income = Net Income - (Total Unusual Items - Tax Effect Of Unusual Items)
    Tax Effect Of Unusual Items = Total Unusual Items x Tax Rate For Calcs
    Tax Rate For Calcs ?= Tax Provision / Pretax Income          (the company's own effective rate)

This checks each link, per company, at both the latest fiscal year and the trailing-twelve-month
window the screener actually scores, reading every row **from the same column** (see
``2026-10-16-normalized-income-same-column.py`` for why that matters). Three things decide whether
0.8c can be met:

* **is the adjustment reproducible** from served rows at all;
* **is it served** for the period the screener scores - a row the engine cannot sum over four
  quarters is a row the page cannot show;
* **is the rate the company's own**, so a reader can tie it to a filing.

Offline; reads the scan's JSON only.

Usage: python research/measurements/2026-10-16-normalized-income-reconstructible.py
"""
import json
from collections import Counter
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
SCAN = HERE / "2026-10-16-normalized-income-same-column.json"
OUT = Path(__file__).with_suffix(".json")

REL = 1e-3        # "the same figure": within 0.1% of the larger magnitude
NI, NICS = "Net Income", "Net Income Common Stockholders"
NO, UI = "Normalized Income", "Total Unusual Items"
TE, TR = "Tax Effect Of Unusual Items", "Tax Rate For Calcs"
TP, PI = "Tax Provision", "Pretax Income"


def close(a, b, rel=REL):
    if a is None or b is None:
        return False
    return abs(a - b) / max(abs(a), abs(b), 1.0) <= rel


def analyse(period, data):
    kinds, gaps, rate_gaps, cases = Counter(), [], [], []
    quarters_short = Counter()
    for t, rows in data.items():
        p = rows.get(period)
        if not p:
            kinds["the period itself could not be formed"] += 1
            continue
        v = p["vals"]
        if period == "ttm":
            for row in (NO, UI, TE):
                n = p["quarters_populated"].get(row, 0)
                if n < 4:
                    quarters_short[f"{row}: {n} of 4 quarters"] += 1
        ni, nics, no, ui, te = v.get(NI), v.get(NICS), v.get(NO), v.get(UI), v.get(TE)
        if no is None:
            kinds["no Normalized Income for the period - nothing to score"] += 1
            continue
        if ni is None:
            kinds["no Net Income for the period"] += 1
            continue
        if close(no, ni):
            kinds["no adjustment: normalized equals net"] += 1
            continue
        # the source adjusted. Can the adjustment be rebuilt from served rows?
        if ui is None:
            kinds["ADJUSTED, but no unusual-items row to explain it"] += 1
            cases.append({"ticker": t, "why": "no unusual-items row", "net": ni,
                          "normalized": no, "unusual": None, "tax_effect": te})
            continue
        adj = ni - no                       # what the source took out, after tax
        for base, label in ((ni, "Net Income"), (nics, "Net Income Common Stockholders")):
            if base is None:
                continue
            if te is not None and close(base - (ui - te), no):
                kinds[f"rebuilt from served rows (base: {label})"] += 1
                break
        else:
            gap = abs(adj - (ui - te)) if te is not None else None
            kinds["ADJUSTED, and the served rows do not rebuild it"] += 1
            if gap is not None:
                gaps.append(gap)
            cases.append({"ticker": t, "why": "served rows do not rebuild it", "net": ni,
                          "nics": nics, "normalized": no, "unusual": ui, "tax_effect": te,
                          "took_out": adj, "unusual_less_tax_effect": (ui - te) if te is not None else None,
                          "gap": gap})
        # separately: is the rate the source used the company's own effective rate?
        tr, tp, pi = v.get(TR), v.get(TP), v.get(PI)
        if tr is not None and tp is not None and pi and abs(pi) > 1e6:
            rate_gaps.append(abs(tr - tp / pi))
    rebuilt = sum(n for k, n in kinds.items() if k.startswith("rebuilt"))
    adjusted = rebuilt + kinds["ADJUSTED, and the served rows do not rebuild it"] + \
        kinds["ADJUSTED, but no unusual-items row to explain it"]
    return {
        "companies": len(data),
        "kinds": dict(kinds.most_common()),
        "adjusted_by_the_source": adjusted,
        "rebuilt_from_served_rows": rebuilt,
        "share_rebuilt": round(rebuilt / adjusted, 4) if adjusted else None,
        "median_unexplained_gap_usd": round(median(gaps), 0) if gaps else None,
        "served_rate_vs_tax_provision_over_pretax": {
            "checked": len(rate_gaps),
            "within_1pp": sum(1 for g in rate_gaps if g <= 0.01),
            "median_gap_pp": round(median(rate_gaps) * 100, 2) if rate_gaps else None,
        },
        "quarters_short": dict(quarters_short.most_common()) if period == "ttm" else None,
        "cases": sorted(cases, key=lambda c: -abs(c["net"]))[:15],
    }


scan = json.loads(SCAN.read_text(encoding="utf-8"))
result = {"source": SCAN.name, "scanned": scan["scanned"], "failed": len(scan["failed"]),
          "annual": analyse("annual", scan["rows"]), "ttm": analyse("ttm", scan["rows"])}
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

for period in ("annual", "ttm"):
    r = result[period]
    print(f"--- {period} ---")
    for k, n in r["kinds"].items():
        print(f"   {n:4d}  {k}")
    print(f"   adjusted by the source: {r['adjusted_by_the_source']}; "
          f"rebuilt from served rows: {r['rebuilt_from_served_rows']} ({r['share_rebuilt']})")
    if r["median_unexplained_gap_usd"]:
        print(f"   median unexplained gap: ${r['median_unexplained_gap_usd']:,.0f}")
    rr = r["served_rate_vs_tax_provision_over_pretax"]
    print(f"   'Tax Rate For Calcs' vs Tax Provision / Pretax Income: {rr['within_1pp']} of "
          f"{rr['checked']} within 1pp (median gap {rr['median_gap_pp']}pp)")
    if r["quarters_short"]:
        print("   rows short of four quarters:")
        for k, n in list(r["quarters_short"].items())[:6]:
            print(f"      {n:4d}  {k}")
print(f"\nwrote {OUT.name}")
