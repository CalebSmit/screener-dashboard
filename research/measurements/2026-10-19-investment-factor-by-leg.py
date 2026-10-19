"""The investment factor among big companies, split into the leg a long-only screener keeps.

Written for research/2026-10-19-revenue-growth-vs-asset-growth.md, which needed to know what
the Investment category's 5 points actually stand on for THIS universe: 500 large companies,
long only, no shorts.

The methodology page's Investment section says CMA "captures the historical tendency for
companies that invest conservatively to outperform those that aggressively expand their asset
base" and gives no figure and no large-company caveat - where the Size and Momentum sections
both give one. This script produces the figures that section is missing, from the lab's own data
(the Kenneth R. French library's 2x3 size-by-investment portfolios, via factor_lab's experiment
8 split), so the note can quote a measured number rather than a characterisation.

The split is the one experiment 8 already publishes and tests month by month:

    big   = the factor among big companies        (conservative big minus aggressive big)
    long  = conservative big companies MINUS the market   <- the half a long-only book keeps
    short = the market MINUS aggressive big companies     <- the half it does not
    small = the factor among small companies      (for the size conditioning)

and long + short = big by construction.

All four factors are printed, not only investment, because the cross-factor comparison is what
makes investment's number readable: a figure for one factor in isolation cannot say whether
t 1.46 is poor for this data or ordinary for it.

This is a factor-portfolio return from a public library, not a backtest of this screener's
ranking and not this system's own IC history. It is the same class of evidence the lab
publishes and the methodology page already cites for Size and Momentum - published-research
replication, not a measured result about the screener (CLAUDE.md rules 4 and 5; plan/factor-lab.md,
"What a lab result is evidence for").

Needs the library files: python factor_lab.py --fetch (they live in the gitignored data/lab/).

Run from the repo root:
    python research/measurements/2026-10-19-investment-factor-by-leg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
sys.path.insert(0, str(ROOT))

import factor_lab as fl  # noqa: E402  (after the path insert)

# Periods. 2000 and 2014 are the two cut points the lab and the methodology page already use
# elsewhere (the Momentum section's "flat since 2000"; the page's limitation 11 on 2014), so
# they are not chosen here to suit an answer.
PERIODS = {"1963-2026": None, "since 2000": 200001, "since 2014": 201401}
LEGS = ("big", "long", "short", "small")


def stat(values: list, mask: np.ndarray) -> dict | None:
    v = np.asarray(values, dtype=float)[mask]
    ok = ~np.isnan(v)
    if ok.sum() < 12:
        return None
    mean = float(np.nanmean(v))
    se = float(np.nanstd(v, ddof=1) / np.sqrt(ok.sum()))
    return {"mean_pct_per_month": round(mean, 4), "t": round(mean / se, 3),
            "months": int(ok.sum())}


def main() -> None:
    payload = fl.build_payload()
    split = payload["split"]
    months = np.array([int(m) for m in split["months"]])
    res: dict = {
        "source": "Kenneth R. French data library, 2x3 size-by-investment portfolios, "
                  "via factor_lab experiment 8 (big-and-small)",
        "span": [int(months[0]), int(months[-1])],
        "n_months": int(len(months)),
        "legs": {"big": "conservative big minus aggressive big",
                 "long": "conservative big minus the market (the long-only half)",
                 "short": "the market minus aggressive big",
                 "small": "the factor among small companies"},
        "factors": {},
    }
    print(f"source: French library 2x3 size x investment, via factor_lab experiment 8")
    print(f"span: {months[0]} .. {months[-1]}  ({len(months)} months)")
    print("\nmonthly average and t-statistic, by factor, leg and period\n")
    head = f"{'factor':14s} {'leg':6s} " + " ".join(f"{p:>23s}" for p in PERIODS)
    print(head)
    print("-" * len(head))
    for key in payload["split_keys"]:
        res["factors"][key] = {}
        for leg in LEGS:
            cells = []
            res["factors"][key][leg] = {}
            for label, start in PERIODS.items():
                mask = np.ones(len(months), bool) if start is None else months >= start
                s = stat(split[key][leg], mask)
                res["factors"][key][leg][label] = s
                cells.append("n/a" if s is None else
                             f"{s['mean_pct_per_month']:+.3f} (t {s['t']:+.2f}, n{s['months']})")
            print(f"{key:14s} {leg:6s} " + " ".join(f"{c:>23s}" for c in cells))
        print()

    # The two readings the note turns on, stated as numbers rather than left to the reader.
    inv = res["factors"]["investment"]
    res["findings"] = {
        "big_company_factor_not_distinguishable_full_sample":
            abs(inv["big"]["1963-2026"]["t"]) < 2,
        "long_leg_distinguishable_full_sample":
            abs(inv["long"]["1963-2026"]["t"]) >= 2,
        "small_over_big_mean_ratio": round(
            inv["small"]["1963-2026"]["mean_pct_per_month"]
            / inv["big"]["1963-2026"]["mean_pct_per_month"], 2),
        "long_legs_clearing_t2_full_sample": sorted(
            k for k in res["factors"]
            if abs(res["factors"][k]["long"]["1963-2026"]["t"]) >= 2),
        "big_legs_clearing_t2_full_sample": sorted(
            k for k in res["factors"]
            if abs(res["factors"][k]["big"]["1963-2026"]["t"]) >= 2),
    }
    f = res["findings"]
    print("--- what the note reads off this ---")
    print(f"investment among big companies, full sample: "
          f"{inv['big']['1963-2026']['mean_pct_per_month']:+.3f}%/month, "
          f"t {inv['big']['1963-2026']['t']:+.2f} -> "
          f"{'NOT distinguishable' if f['big_company_factor_not_distinguishable_full_sample'] else 'distinguishable'}")
    print(f"the long-only half of it:                   "
          f"{inv['long']['1963-2026']['mean_pct_per_month']:+.3f}%/month, "
          f"t {inv['long']['1963-2026']['t']:+.2f} -> "
          f"{'distinguishable' if f['long_leg_distinguishable_full_sample'] else 'NOT distinguishable'}")
    print(f"small companies earn {f['small_over_big_mean_ratio']}x the big-company figure")
    print(f"long legs clearing |t| >= 2 over the full sample: "
          f"{', '.join(f['long_legs_clearing_t2_full_sample'])}")
    print(f"long-SHORT versions clearing it:                  "
          f"{', '.join(f['big_legs_clearing_t2_full_sample']) or 'none'}")

    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
