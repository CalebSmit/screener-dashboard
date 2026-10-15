"""Reviewing the open capex proposal: what does removing a captive's lease purchases leave behind?

For the review of `research/2026-10-14-capex-definition.md` section 4 (proposal: FCF capex =
Yahoo's capex minus a captive finance arm's purchases of assets to lease out, for GM, DE, CAT,
PCAR). The note measured the numerator's own figure. This measures the thing it did not: what
*else* the lease book puts into the ratio, on both sides.

A leasing operation's cash cycle has three legs:

* buy the vehicle / machine - investing outflow (the tag the proposal removes);
* collect rentals over the lease - **operating** inflow, inside the operating cash flow the
  screener's FCF starts from;
* sell it at lease end - investing inflow (`ProceedsFrom...`), which `OCF - capex` never counted.

So the question a reviewer has to answer is not "is a lease fleet industrial capex?" (no source
says it is) but "does `OCF - PP&E` describe the industrial business?" - because that numerator
keeps the rentals and drops the purchases.

Reads: each company's own `companyfacts` (one SEC request each, the weekly cache does not carry
the lease tags until its next refresh) and the published payload for the run's EV, cash-flow
quarter and sector percentile. No return is read (rule 4); nothing is written outside --out.

    python research/measurements/2026-10-15-capex-proposal-review.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sec_crosscheck as X  # noqa: E402

CAPTIVES = ["GM", "DE", "CAT", "PCAR", "F"]

# Cash-flow-statement tags of the lease book, as these filers tag them.
LEASE_OUT = ("PaymentsToAcquireLeasesHeldForInvestment", "PaymentsToAcquireEquipmentOnLease",
             "PaymentsToAcquireLoansAndLeasesHeldForInvestment")
LEASE_IN = ("ProceedsFromSaleOfLeasesHeldForInvestment", "ProceedsFromSaleOfEquipmentOnLease",
            "ProceedsFromSaleAndCollectionOfLoansReceivable",
            "ProceedsFromSaleOfPropertyHeldForSale")
PPE = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
OCF = ("NetCashProvidedByUsedInOperatingActivities",)
# Depreciation of assets on operating lease: the non-cash charge whose add-back is how the
# fleet's cost passes through operating cash flow.
LEASE_DEP = ("DepreciationOfLeasedAssets", "OperatingLeasesIncomeStatementDepreciationExpenseOnPropertySubjectToOrAvailableForOperatingLease",
             "CostOfGoodsAndServicesSoldDepreciationAndAmortization")


def fetch_facts(tickers):
    """One `companyfacts` request per company -> the same long frame the weekly cache holds."""
    from insider_activity import Edgar, ticker_map, user_agent
    ua = user_agent()
    if not ua:
        raise SystemExit("no SEC identity configured (data/sec/user_agent.txt)")
    edgar = Edgar(ua)
    cmap = ticker_map(edgar)
    rows = []
    for t in tickers:
        cik = cmap.get(t.upper())
        j = edgar.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json").json()
        gaap = (j.get("facts") or {}).get("us-gaap") or {}
        for concept, body in gaap.items():
            for f in ((body.get("units") or {}).get("USD") or []):
                if f.get("form") in X.FORMS:
                    rows.append((t, concept, f.get("start"), f.get("end"), f.get("filed"),
                                 f.get("form"), f.get("val")))
    df = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "form", "val"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--facts", default=None, help="a saved frame from a previous run (no network)")
    args = ap.parse_args()

    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    detail = data["stock_detail"]

    if args.facts and Path(args.facts).exists():
        facts = pd.read_parquet(args.facts)
    else:
        facts = fetch_facts(CAPTIVES)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            facts.to_parquet(Path(args.out) / "captive_facts.parquet")

    print(f"run {data['kpis'].get('run_timestamp')}; {len(facts):,} USD facts for {facts['ticker'].nunique()} companies\n")

    for t in CAPTIVES:
        det = detail.get(t)
        if det is None:
            print(f"{t}: not in the run")
            continue
        inp, asof = det.get("inp") or {}, det.get("asof") or {}
        end = pd.Timestamp(asof["cf"])
        f = facts[facts["ticker"] == t]
        ocf, _, _ = X.sec_value(f, OCF, "flow", end)
        ppe, ppe_c, _ = X.sec_value(f, PPE, "flow", end)
        out, out_c, _ = X.sec_value(f, LEASE_OUT, "flow", end)
        inn, in_c, _ = X.sec_value(f, LEASE_IN, "flow", end)
        dep, dep_c, _ = X.sec_value(f, LEASE_DEP, "flow", end)
        ev = inp.get("ev_used")
        y_ocf, y_capex = inp.get("operatingCashFlow"), inp.get("capex")
        print(f"=== {t}  ({det.get('sector')}), cash flow to {end.date()}, EV ${ev/1e9:,.1f}B")
        print(f"  Yahoo: OCF ${y_ocf/1e9:,.2f}B  capex ${abs(y_capex)/1e9:,.2f}B  -> FCF "
              f"${(y_ocf - abs(y_capex))/1e9:,.2f}B  yield {(y_ocf - abs(y_capex))/ev:+.2%}")
        def b(v):
            return "n/a" if v is None else f"${v/1e9:,.2f}B"
        print(f"  filed: OCF {b(ocf)}  PP&E {b(ppe)} [{ppe_c}]")
        print(f"         lease purchases {b(out)} [{out_c}]")
        print(f"         lease proceeds  {b(inn)} [{in_c}]")
        print(f"         lease depreciation {b(dep)} [{dep_c}]")
        if None not in (ocf, ppe) and ev:
            variants = {
                "a. consolidated (today)": y_ocf - abs(y_capex),
                "b. proposal: OCF - filed PP&E": ocf - ppe,
            }
            if out is not None:
                variants["c. proposal, Yahoo capex - lease purchases"] = y_ocf - (abs(y_capex) - out)
            if out is not None and inn is not None:
                variants["d. lease book netted (purchases - proceeds)"] = ocf - ppe - (out - inn)
            for name, v in variants.items():
                print(f"    {name:48} FCF {v/1e9:>8,.2f}B  yield {v/ev:+7.2%}")
        print()

    # What a lease book's rentals are worth inside operating cash flow, for the one company that
    # lets it be bounded from filed figures: lease depreciation is the fleet's cost passing
    # through, so rentals >= depreciation for a book that covers its own cost.
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
