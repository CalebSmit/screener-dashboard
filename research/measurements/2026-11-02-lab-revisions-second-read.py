"""Second reading of Factor Lab experiment 18 (analyst revisions, Chan, Jegadeesh & Lakonishok 1996).

An independent recomputation. It does NOT import factor_lab: it reads Open Source Asset Pricing's
REV6 decile returns straight from the files in data/lab/osap/ with pandas, applies the construction
the record states (equal-weighted deciles, 197701-199212, six times the average month against the
paper's six-month returns; the spread is decile 10 minus decile 1 month by month, t = mean / (sd(n-1)
/ sqrt(n))), and prints every figure the research note and the lab page publish, next to the value
the published payload (lab_data.js) carries where it carries one.

    python research/measurements/2026-11-02-lab-revisions-second-read.py

Reads data/lab/osap/PredictorAltPorts_DecilesEW.zip, PredictorAltPorts_DecilesVW.parquet and
PS_OriginalCuts_VW.csv (gitignored; `python factor_lab.py --fetch` downloads them) and lab_data.js.
"""
from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OSAP = ROOT / "data" / "lab" / "osap"
PORTS = [f"{i:02d}" for i in range(1, 11)]


def ew_rev6() -> pd.DataFrame:
    """REV6 rows of the equal-weighted full set, read in chunks (the file holds every signal)."""
    with zipfile.ZipFile(OSAP / "PredictorAltPorts_DecilesEW.zip") as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        parts = []
        with z.open(name) as f:
            for chunk in pd.read_csv(f, usecols=["signalname", "port", "date", "ret"],
                                     dtype={"port": str}, chunksize=2_000_000):
                parts.append(chunk[chunk.signalname == "REV6"])
    return pd.concat(parts)


def vw_rev6() -> pd.DataFrame:
    df = pd.read_parquet(OSAP / "PredictorAltPorts_DecilesVW.parquet",
                         columns=["signalname", "port", "date", "ret"], filters=[("signalname", "=", "REV6")])
    return df


def wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ym"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m")
    df["port"] = df["port"].astype(str).str.strip().str.zfill(2).where(df["port"].astype(str).str.strip() != "LS", "LS")
    w = df.pivot(index="ym", columns="port", values="ret").sort_index()
    return w


def stats(x: pd.Series) -> tuple[float, float, float, int]:
    x = x.dropna()
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1)
    return m, sd, m / (sd / math.sqrt(n)), n


def published() -> dict:
    s = (ROOT / "lab_data.js").read_text(encoding="utf-8")
    d = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    exp = next(e for e in d["experiments"] if e["id"] == "revisions")
    atp = [r for r in d["after_the_paper"] if r["experiment"] == "revisions"]
    sc = next(r for r in d["scorecard"] if r["key"] == "revisions")
    return {"result": exp["result"], "after": atp, "scorecard": sc["measured"], "takeaway": exp.get("takeaway")}


def main() -> int:
    ew, vw = wide(ew_rev6()), wide(vw_rev6())
    # The record's month set: months in which all ten deciles have a return under BOTH weightings.
    both = ew[PORTS].dropna().index.intersection(vw[PORTS].dropna().index)
    print(f"EW months {ew.index.min()}-{ew.index.max()} ({len(ew)}), VW {vw.index.min()}-{vw.index.max()} "
          f"({len(vw)}), all ten deciles in both: {len(both)}")
    E, V = ew.loc[both], vw.loc[both]

    def window(df, a, b):
        return df[(df.index >= a) & (df.index <= b)]

    # ---- 1. The three compared figures, the paper's window, equal weights, x6
    P = window(E, "197701", "199212")
    print(f"\nPaper window 197701-199212: {len(P)} months")
    means = [P[p].mean() for p in PORTS]
    sp = P["10"] - P["01"]
    m, sd, t, n = stats(sp)
    pub = published()
    here = {r["key"]: r["here"] for r in pub["result"]["rows"]}
    print(f"  most downward x6 = {6 * means[0]:.4f}  (page {here['mean1']:.4f}, note 2.75; paper 4.6, tol 2.9)")
    print(f"  most upward   x6 = {6 * means[9]:.4f}  (page {here['mean10']:.4f}, note 11.40; paper 12.3, tol 2.9)")
    print(f"  spread        x6 = {6 * m:.4f}  (page {here['spread']:.4f}, note 8.65; paper 7.7, tol 2.9)")
    print(f"  spread a month {m:.4f}, t {t:.4f}  (page {pub['result']['spread']['mean']:.4f}, "
          f"t {pub['result']['spread']['t']:.4f}; note 1.44, t 8.42)")
    print("  within:", [abs(a - b) <= 2.9 for a, b in ((6 * means[0], 4.6), (6 * means[9], 12.3), (6 * m, 7.7))])
    print("  six-month equivalents of the ten deciles:", " ".join(f"{6 * x:.2f}" for x in means))
    print("  monotone rising across all ten:", all(b > a for a, b in zip(means, means[1:])))
    for i in range(10):
        assert abs(means[i] - pub["result"]["means"][i]) < 5e-5, (i, means[i], pub["result"]["means"][i])
    print("  all ten decile means equal the published payload's to 5e-5")
    print(f"  volatility, a month: decile 1 {P['01'].std(ddof=1):.2f}, decile 10 {P['10'].std(ddof=1):.2f}")
    PV = window(V, "197701", "199212")
    mv, _, tv, _ = stats(PV["10"] - PV["01"])
    print(f"  by market value: {mv:.4f}% a month, x6 = {6 * mv:.2f}, t {tv:.2f}  (note 4.19 / 0.70, t 2.92)")

    # ---- 2. Since the paper, both weightings
    print("\nPeriods (spread a month, t, months):")
    for label, a, b in (("paper 1977-1992", "197701", "199212"), ("after 1993-2024", "199301", "209912"),
                        ("since 2007", "200701", "209912"), ("whole record", "000000", "209912")):
        e = stats(window(E, a, b)["10"] - window(E, a, b)["01"])
        v = stats(window(V, a, b)["10"] - window(V, a, b)["01"])
        print(f"  {label:16s} EW {e[0]:+.4f} (t {e[2]:.2f}, n {e[3]})   VW {v[0]:+.4f} (t {v[2]:.2f}, n {v[3]})"
              f"   VW/EW {v[0] / e[0]:.2f}")
    A = window(E, "199301", "209912")
    print(f"  since 1993 volatility, a month: decile 1 {A['01'].std(ddof=1):.2f}, decile 10 {A['10'].std(ddof=1):.2f}")
    e_p, e_a = stats(P["10"] - P["01"])[0], stats(A["10"] - A["01"])[0]
    print(f"  EW weaker after the paper by {100 * (1 - e_a / e_p):.1f}% (note: about 37%)")
    print("  published after_the_paper rows:", json.dumps(pub["after"]))
    print("  published scorecard sentence:", pub["scorecard"])

    # ---- 3. Internal consistency of the files
    print("\nLong-short column against decile 10 minus decile 1, every month it is published:")
    for nm, df in (("EW", ew), ("VW", vw)):
        ls = df[["LS", "01", "10"]].dropna()
        dev = (ls["LS"] - (ls["10"] - ls["01"])).abs().max()
        print(f"  {nm}: {len(ls)} months {ls.index.min()}-{ls.index.max()}, largest |LS - (10 - 01)| = {dev:.2e}")

    # ---- 4. The two copies of the release: parquet PS deciles vs experiment 17's Drive csv
    pq_ps = wide(pd.read_parquet(OSAP / "PredictorAltPorts_DecilesVW.parquet", columns=["signalname", "port", "date", "ret"],
                                 filters=[("signalname", "=", "PS")]))
    csv = pd.read_csv(OSAP / "PS_OriginalCuts_VW.csv")
    csv["ym"] = pd.to_datetime(csv["date"]).dt.strftime("%Y%m")
    csv = csv.set_index("ym")
    for p in ("01", "10"):
        j = pq_ps[[p]].join(csv[["port" + p]], how="inner").dropna()
        print(f"  PS decile {p}: {len(j)} common months, largest |parquet - Drive csv| = "
              f"{(j[p] - j['port' + p]).abs().max():.2e}")

    # ---- 5. Rule 8: a result of no effect
    null = (4.6 + 12.3) / 2
    print(f"\nNull: every group earns the same; the record's common level {null:.2f}. Ends sit "
          f"{abs(null - 4.6):.2f} and {abs(12.3 - null):.2f} from it (tolerance 2.9); spread 0 vs 7.7 "
          f"misses by 7.7. Matches: {sum([abs(null - 4.6) <= 2.9, abs(12.3 - null) <= 2.9, abs(0 - 7.7) <= 2.9])} of 3 "
          f"(page: null_within = {pub['result']['null_within']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
