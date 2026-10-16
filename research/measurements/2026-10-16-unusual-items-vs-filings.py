"""The source's "unusual items" against the companies' own filings - 2026-10-16

``research/2026-10-15-earnings-definition.md`` section 6 names the test that would make its
reviewer refuse the proposal: *"evidence that the source's 'unusual items' are inconsistent across
companies or misclassify recurring costs at scale (a sample of 20 against the companies' own
special-item disclosures would test it)"*. This runs that test on every constituent instead of
twenty, from the weekly ``companyfacts`` cache the run already keeps - offline, read-only.

Two questions, both about the latest fiscal year the source serves (so the two sides describe the
same period):

**A. Is the adjustment reproducible?** The source's construction is
``Normalized Income = Net Income - Total Unusual Items x (1 - tax rate)``, and the rate is the
source's own, unpublished. If it is the company's **effective tax rate** - a figure in every
filing (``IncomeTaxExpenseBenefit`` / pre-tax income) - then a reader can rebuild the number the
screener scored from three published figures, and CLAUDE.md row 0.8c's equation line is possible.
If it is not, the page would print an input whose arithmetic nothing on it reproduces.

**B. Do the "unusual" items recur?** Rouen, So & Wang (2021) note that restructuring is the most
common non-core item and warn about charges that repeat year after year; MSCI excludes
non-recurring items but keeps oil-and-gas property impairments *because* they recur there. For each
company this counts the fiscal years, of the last four it filed, carrying a tagged restructuring or
impairment charge - from the companies' own tags, not the source's classification.

Usage: python research/measurements/2026-10-16-unusual-items-vs-filings.py   (offline; JSON beside it)
"""
import json
from collections import Counter
from pathlib import Path
from statistics import median

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCAN = HERE / "2026-10-15-earnings-rows-scan.json"
FACTS = ROOT / "data" / "sec" / "pit" / "facts.parquet"
OUT = Path(__file__).with_suffix(".json")

PRETAX = ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments")
TAX = "IncomeTaxExpenseBenefit"
# The companies' own special-item tags. Charges are filed positive (an expense); the overlapping
# "restructuring and impairment" roll-ups are taken in priority order so nothing is counted twice.
SPECIAL_GROUPS = [
    ("restructuring", ("RestructuringCostsAndAssetImpairmentCharges",
                       "RestructuringSettlementAndImpairmentProvisions",
                       "RestructuringCharges")),
    ("goodwill", ("GoodwillAndIntangibleAssetImpairment", "GoodwillImpairmentLoss")),
    ("intangibles", ("ImpairmentOfIntangibleAssetsExcludingGoodwill",
                     "ImpairmentOfIntangibleAssetsIndefinitelivedExcludingGoodwill")),
    ("assets", ("AssetImpairmentCharges", "OtherAssetImpairmentCharges",
                "ImpairmentOfLongLivedAssetsHeldForUse")),
]
RATE_TOL = 0.05       # the implied rate "is" the filed effective rate within 5 percentage points
MATERIAL = 0.05       # a special item "material" at 5% of the year's pre-tax income


def first_filed(rows):
    if rows.empty:
        return None
    return float(rows.sort_values("filed").iloc[0]["val"])


def fy_value(fy, concepts, end):
    """The fiscal-year value for the year ending `end`, as first filed, first concept that has it."""
    for c in concepts:
        rows = fy[(fy["concept"] == c) & (fy["end"] >= end - pd.Timedelta(days=10)) &
                  (fy["end"] <= end + pd.Timedelta(days=10))]
        v = first_filed(rows)
        if v is not None:
            return v, c
    return None, None


def main():
    scan = json.loads(SCAN.read_text(encoding="utf-8"))["rows"]
    df = pd.read_parquet(FACTS)
    df["end"] = pd.to_datetime(df["end"])
    df["start"] = pd.to_datetime(df["start"])
    fy_all = df[(df["fp"] == "FY") & (df["form"].isin(("10-K", "10-K/A")))].copy()
    # a fiscal-year flow spans about a year; an instant has no start
    span = (fy_all["end"] - fy_all["start"]).dt.days
    fy_all = fy_all[span.between(330, 400) | fy_all["start"].isna()]

    reconcile, recurrence, missing = [], [], Counter()
    for t, rows in scan.items():
        fy = fy_all[fy_all["ticker"] == t]
        ann = rows["annual"]
        ni = ann.get("Net Income")
        no = ann.get("Normalized Income")
        ui = ann.get("Total Unusual Items")

        # --- A. reproducible from the filed effective tax rate? ---
        # The 2026-10-15 scan records each row's latest *populated* year, which is not the same
        # year for every row (UPS's "Normalized Income" is blank for fiscal 2025, BAC's "Total
        # Unusual Items" is), so the three must be required to share a fiscal-year end or the
        # comparison is between periods. Found 2026-10-16 while reviewing the proposal.
        if ni and no and ui and abs(float(ui["val"])) >= 1.0 and ni["end"] == no["end"] == ui["end"]:
            niv, nov, uiv = float(ni["val"]), float(no["val"]), float(ui["val"])
            end = pd.Timestamp(ni["end"])
            if abs(niv - nov) / max(abs(niv), abs(nov), 1.0) > 1e-4 and not fy.empty:
                implied = 1.0 - (niv - nov) / uiv
                pretax, pc = fy_value(fy, PRETAX, end)
                tax, _ = fy_value(fy, (TAX,), end)
                if pretax is None or tax is None or abs(pretax) < 1e6:
                    missing["no filed tax rate for the year"] += 1
                else:
                    eff = tax / pretax
                    reconcile.append({"ticker": t, "net": niv, "normalized": nov, "unusual": uiv,
                                      "implied_rate": round(implied, 4),
                                      "filed_effective_rate": round(eff, 4),
                                      "gap": round(implied - eff, 4),
                                      "pretax": pretax, "tax": tax, "pretax_concept": pc,
                                      "fy_end": str(end.date())})
            elif fy.empty:
                missing["no fiscal-year filings cached"] += 1
        elif ni and no and ui and abs(float(ui["val"])) >= 1.0:
            missing["the three rows are not all from the same fiscal year"] += 1

        # --- B. do the filed special items recur? ---
        if fy.empty:
            continue
        years = sorted({d.year for d in fy[fy["concept"] == TAX]["end"]})[-4:]
        if len(years) < 3:
            missing["under three fiscal years cached"] += 1
            continue
        hits, material_hits, detail = 0, 0, {}
        for y in years:
            ends = sorted({d for d in fy["end"] if d.year == y})
            if not ends:
                continue
            end = max(ends)
            total = 0.0
            for _, concepts in SPECIAL_GROUPS:
                v, _ = fy_value(fy, concepts, end)
                if v:
                    total += abs(v)
            pretax, _ = fy_value(fy, PRETAX, end)
            detail[y] = {"special": total, "pretax": pretax}
            if total > 0:
                hits += 1
                if pretax and abs(pretax) > 1e6 and total / abs(pretax) >= MATERIAL:
                    material_hits += 1
        recurrence.append({"ticker": t, "years": len(years), "years_with_a_special_item": hits,
                           "years_with_a_material_one": material_hits, "detail": detail})

    # --- summarise ---
    gaps = [abs(r["gap"]) for r in reconcile]
    within = [r for r in reconcile if abs(r["gap"]) <= RATE_TOL]
    full4 = [r for r in recurrence if r["years"] == 4]
    summary = {
        "A_reproducible_from_the_filed_tax_rate": {
            "companies_the_source_adjusted_and_a_rate_could_be_checked": len(reconcile),
            "skipped": dict(missing),
            f"implied_rate_within_{int(RATE_TOL * 100)}pp_of_the_filed_effective_rate": len(within),
            "share_within": round(len(within) / len(reconcile), 4) if reconcile else None,
            "median_absolute_gap_pp": round(median(gaps) * 100, 2) if gaps else None,
            "gap_over_20pp": sum(1 for g in gaps if g > 0.20),
            "gap_over_50pp": sum(1 for g in gaps if g > 0.50),
            "worst_by_net_income": sorted(reconcile, key=lambda r: -abs(r["net"]))[:15],
        },
        "B_do_the_filed_special_items_recur": {
            "companies_with_four_fiscal_years": len(full4),
            "distribution_years_with_a_special_item": dict(
                sorted(Counter(r["years_with_a_special_item"] for r in full4).items())),
            "distribution_years_with_a_material_one": dict(
                sorted(Counter(r["years_with_a_material_one"] for r in full4).items())),
            "all_four_years": sorted(r["ticker"] for r in full4 if r["years_with_a_special_item"] == 4),
            "all_four_years_material": sorted(r["ticker"] for r in full4
                                              if r["years_with_a_material_one"] == 4),
        },
    }
    OUT.write_text(json.dumps({"summary": summary, "reconcile": reconcile,
                               "recurrence": recurrence}, indent=1), encoding="utf-8")

    a = summary["A_reproducible_from_the_filed_tax_rate"]
    print("A. Can a reader rebuild Normalized Income from net income, unusual items and the")
    print("   company's own effective tax rate?")
    print(f"   checkable for {a['companies_the_source_adjusted_and_a_rate_could_be_checked']} companies; "
          f"skipped {a['skipped']}")
    print(f"   within {int(RATE_TOL * 100)}pp: {a[f'implied_rate_within_{int(RATE_TOL * 100)}pp_of_the_filed_effective_rate']} "
          f"({a['share_within']}); median absolute gap {a['median_absolute_gap_pp']}pp; "
          f"over 20pp {a['gap_over_20pp']}; over 50pp {a['gap_over_50pp']}")
    b = summary["B_do_the_filed_special_items_recur"]
    print("\nB. Of the companies with four fiscal years filed, how many years carried a")
    print("   restructuring or impairment charge the company itself tagged?")
    print(f"   n = {b['companies_with_four_fiscal_years']}")
    print(f"   any charge:      {b['distribution_years_with_a_special_item']}")
    print(f"   >= 5% of pretax: {b['distribution_years_with_a_material_one']}")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
