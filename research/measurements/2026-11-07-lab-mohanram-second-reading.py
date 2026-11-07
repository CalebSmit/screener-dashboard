"""Experiment 19's second reading, measured (OWNER_FOCUS 2026-10-20 item 2).

`research/2026-11-06-lab-mohanram.md` section 7 leaves a second reader three things. This script
is two of them, and deliberately **does not import `factor_lab`** - that is the point of a second
implementation:

  1. Recompute every measured figure in the note's section 5 table, and the three later windows,
     straight from the Open Source Asset Pricing portfolio files.
  2. Test the note's section 4 claim - that `SignalDoc.csv`'s `Return` column is the *paper's*
     figure as the library transcribed it, not a measured one - on nine signals rather than the
     one (`PS`) the note checked.

The third (re-reading Table 4 Panel A from the published article) needs a copy of the version of
record and is done by hand; see the note's "Second reading" section for what was obtained.

Run:  python research/measurements/2026-11-07-lab-mohanram-second-reading.py
Reads the gitignored `data/lab/osap/` files only. Writes nothing.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

OSAP = pathlib.Path(__file__).resolve().parents[2] / "data" / "lab" / "osap"
EW = OSAP / "PredictorPortsFull.parquet"
VW = OSAP / "PredictorAltPorts_LiqScreen_VWforce.parquet"
PORTS = ["01", "02", "03", "04", "05", "06"]

#: The note's section 5 table and its section 5 reading 3, as published. A second reading has to
#: say what it was checking against, or "it reproduces" means nothing.
PUBLISHED = {
    "paper": {"01": 3.22, "02": 5.40, "03": 6.48, "04": 12.75, "05": 7.58, "06": 16.48,
              "spread": 13.26, "t": 4.96},
    "after": {"spread": 5.04, "t": 2.32},
    "since2007": {"spread": 6.94, "t": 2.72},
    "after_vw": {"spread": 3.67, "t": 1.52},
    "since2007_vw": {"spread": 6.01, "t": 2.21},
}
#: The paper's own Table 4 Panel A, re-read from the PDF for this second reading.
PAPER = {"01": -4.0, "02": 5.1, "03": 7.8, "04": 11.6, "05": 16.5, "06": 17.4, "spread": 21.4}
TOLERANCE = 4.0


def wide(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    ms = df[df["signalname"] == "MS"].copy()
    ms["date"] = pd.to_datetime(ms["date"])
    ms["ym"] = ms["date"].dt.to_period("M")
    return ms.pivot_table(index="ym", columns="port", values="ret", aggfunc="first")


def window(w: pd.DataFrame, lo: str | None, hi: str | None) -> pd.DataFrame:
    if lo:
        w = w[w.index >= pd.Period(lo, "M")]
    if hi:
        w = w[w.index <= pd.Period(hi, "M")]
    return w


def spread_of(w: pd.DataFrame) -> tuple[float, float, int]:
    s = (w["06"] - w["01"]).dropna()
    t = s.mean() / (s.std(ddof=1) / len(s) ** 0.5)
    return s.mean() * 12, t, len(s)


def agree(got: float, published: float, tol: float = 0.02) -> str:
    return "ok" if abs(got - published) <= tol else f"DIFFERS (published {published})"


def main() -> int:
    if not EW.exists():
        print(f"MISSING DATA: {EW} - the lab's library is gitignored; fetch it with "
              "`python factor_lab.py --fetch` first")
        return 3

    bad = 0
    print("=== 1. the note's section 5 table, recomputed ===")
    w = window(wide(EW), "1978-01", "2001-12")
    print(f"  {len(w)} months {w.index.min()}..{w.index.max()}, equally weighted")
    print(f"  {'group':8} {'paper':>7} {'here':>8} {'diff':>7} {'within 4.0':>11}  vs the note")
    n_within = 0
    for p in PORTS:
        here = w[p].mean() * 12
        diff = here - PAPER[p]
        within = abs(diff) <= TOLERANCE
        n_within += within
        note = agree(here, PUBLISHED["paper"][p])
        bad += note != "ok"
        print(f"  {p:8} {PAPER[p]:>7.1f} {here:>8.2f} {diff:>+7.2f} {str(within):>11}  {note}")
    sp, t, n = spread_of(w)
    within_sp = abs(sp - PAPER["spread"]) <= TOLERANCE
    n_within += within_sp
    for got, key in ((sp, "spread"), (t, "t")):
        note = agree(got, PUBLISHED["paper"][key])
        bad += note != "ok"
        print(f"  {key:8} {PAPER.get(key, float('nan')):>7} {got:>8.2f} "
              f"{'':>7} {str(within_sp) if key == 'spread' else '':>11}  {note}")
    print(f"  within tolerance: {n_within} of 7 - the note says 4 of 7: "
          f"{'ok' if n_within == 4 else 'DIFFERS'}")
    bad += n_within != 4

    print("\n=== 2. after the paper, both weightings ===")
    for label, path, lo, key in (("ew 2002-", EW, "2002-01", "after"),
                                 ("ew 2007-", EW, "2007-01", "since2007"),
                                 ("vw 2002-", VW, "2002-01", "after_vw"),
                                 ("vw 2007-", VW, "2007-01", "since2007_vw")):
        sp, t, n = spread_of(window(wide(path), lo, None))
        a, b = agree(sp, PUBLISHED[key]["spread"]), agree(t, PUBLISHED[key]["t"])
        bad += (a != "ok") + (b != "ok")
        print(f"  {label}: spread {sp:+.2f}% a year ({a}), t {t:.2f} ({b}), {n} months")

    print("\n=== 3. the strongest group was the calmest? ===")
    sds = {p: w[p].std(ddof=1) for p in PORTS}
    print("  " + "  ".join(f"{p}:{v:.2f}" for p, v in sds.items()))
    calmest = min(sds, key=sds.get)
    print(f"  calmest group: {calmest} - {'ok' if calmest == '06' else 'DIFFERS'}")
    bad += calmest != "06"

    print("\n=== 4. is SignalDoc's Return the paper's figure or a measured one? ===")
    doc = pd.read_csv(OSAP / "SignalDoc.csv", encoding="latin-1")
    ports = pd.read_parquet(EW)
    ports["date"] = pd.to_datetime(ports["date"])
    ls = ports[ports["port"] == "LS"]
    rows = doc[doc["Acronym"].isin(["MS", "PS", "Accruals", "AssetGrowth", "Mom12m", "GP",
                                    "NetEquityFinance"])]
    print(f"  {'signal':18} {'doc':>8} {'measured':>9} {'ratio':>6}  window")
    n_differ = 0
    for _, r in rows.iterrows():
        g = ls[ls["signalname"] == r["Acronym"]]
        lo, hi = r["SampleStartYear"], r["SampleEndYear"]
        if pd.notna(lo) and pd.notna(hi):
            g = g[(g["date"].dt.year >= int(lo)) & (g["date"].dt.year <= int(hi))]
        if g.empty or pd.isna(r["Return"]):
            continue
        m, d = g["ret"].mean(), float(r["Return"])
        n_differ += abs(m - d) > 0.05
        print(f"  {r['Acronym']:18} {d:>8.4f} {m:>9.4f} {d / m:>6.2f}  "
              f"{int(lo)}-{int(hi)}")
    print(f"  entries differing from the library's own measured long-short: {n_differ} of "
          f"{len(rows)} - the note's reading holds" if n_differ >= 5 else
          f"  only {n_differ} differ - the note's reading does NOT hold")
    bad += n_differ < 5

    print(f"\n{'ALL CHECKS REPRODUCE' if bad == 0 else f'{bad} CHECK(S) DID NOT REPRODUCE'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
