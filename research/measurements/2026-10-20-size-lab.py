"""The size effect among large companies, on Ken French's public data (factor-lab question 4).

Written for research/2026-10-20-size-weight.md. Experiment 4 of the Factor Lab measured the ninth
size decile against the tenth; this extends it in the four directions the size literature says
matter, on files the lab already holds (data/lab/french/, gitignored, read from the main checkout):

  1. The large-company size spread (deciles 8 and 9 minus 10, quintile 4 minus 5) by period,
     including the years since Asness et al.'s sample ended (2013 on).
  2. Size among large companies *holding value or momentum fixed*: in the 5x5 size-by-book-to-market
     and size-by-prior-return files, the fourth size quintile minus the fifth inside each trait
     quintile, averaged - the double sort Asness, Frazzini, Israel, Moskowitz & Pedersen (2018) use
     with quality.
  3. Whether the smaller of the large companies are the less profitable / faster-investing ones:
     the value-weighted operating profitability and asset growth the same 5x5 file publishes for
     each portfolio, quintile 4 against quintile 5. This is the "junk" mechanism, measured among
     large companies.
  4. SMB's alpha against the market, value, profitability and investment (FF5) and momentum -
     the regression in which Asness et al. report SMB's alpha roughly doubling - by period.

And one locator: how many firms the top size deciles hold and their average market value in the
latest month, so the S&P 500's place in the groups is a number rather than a guess.

Every figure is a long-short return between portfolios of all listed US stocks, free of costs. It
measures the trait among large companies, never the screener (rule 5 does not apply because no
screener backtest is read, but nothing here is a measurement of the ranking either).

Run from the repo root:  python research/measurements/2026-10-20-size-lab.py
Writes research/measurements/2026-10-20-size-lab.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import factor_lab as L  # noqa: E402  (read_table, mean_t only)

# The library's files are gitignored and live in the main checkout; a worktree has none.
RAW = ROOT / "data" / "lab" / "french"
if not RAW.exists():
    RAW = Path(r"C:\Users\smitc\OneDrive\Documents\Screener\data\lab\french")
OUT = Path(__file__).with_suffix(".json")

PERIODS = {
    "1926-2026": ("192607", "209912"),
    "1963-2026": ("196307", "209912"),
    "1963-1990 (FF1992 sample)": ("196307", "199012"),
    "1981-2026 (after Banz)": ("198101", "209912"),
    "1963-2012 (AFIMP sample end)": ("196307", "201212"),
    "2000-2026": ("200001", "209912"),
    "2013-2026 (after AFIMP sample)": ("201301", "209912"),
}


def mt(x) -> dict | None:
    r = L.mean_t(list(x))
    return None if r is None else {"mean": round(r["mean"], 3), "t": round(r["t"], 2), "n": len(x)}


def table(fname: str, marker: str) -> tuple[list[str], np.ndarray]:
    try:
        dates, rows, _ = L.read_table(RAW / fname, marker)
    except ValueError:
        # the characteristic tables carry a multi-line title; their column line starts with ","
        import zipfile
        with zipfile.ZipFile(RAW / fname) as z:
            lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        i = next(k for k, line in enumerate(lines) if marker in line)
        while not lines[i].lstrip().startswith(","):
            i += 1
        dates, rows = [], []
        for line in lines[i + 1:]:
            parts = [c.strip() for c in line.split(",")]
            if not parts[0].isdigit():
                break
            dates.append(parts[0])
            rows.append([float(c) for c in parts[1:]])
    a = np.array(rows, float)
    a[a <= -99] = np.nan
    return dates, a


def window(dates, start, end):
    return np.array([start <= d <= end for d in dates])


def size_deciles() -> dict:
    out = {}
    f = "Portfolios_Formed_on_ME_CSV.zip"
    for w, mark in (("vw", "Average Value Weight Returns -- Monthly"),
                    ("ew", "Average Equal Weighted Returns -- Monthly")):
        d, a = table(f, mark)
        dec = a[:, -10:]
        q = a[:, 4:9]
        res = {}
        for name, (s, e) in PERIODS.items():
            m = window(d, s, e) & ~np.isnan(dec).any(axis=1)
            res[name] = {"d9_minus_d10": mt(dec[m, 8] - dec[m, 9]),
                         "d8_minus_d10": mt(dec[m, 7] - dec[m, 9]),
                         "q4_minus_q5": mt(q[m, 3] - q[m, 4]),
                         "d1_minus_d10": mt(dec[m, 0] - dec[m, 9])}
            # January and the other eleven months, for the large-company spread
            jan = np.array([x[4:] == "01" for x in d])
            res[name]["q4_minus_q5_january"] = mt((q[:, 3] - q[:, 4])[m & jan])
            res[name]["q4_minus_q5_feb_dec"] = mt((q[:, 3] - q[:, 4])[m & ~jan])
        out[w] = res
    # the locator: firms and average size (USD millions) in the top deciles, latest month
    d, n = table(f, "Number of Firms in Portfolios")
    _, sz = table(f, "Average Firm Size")
    out["latest_month"] = d[-1]
    out["firms_latest"] = {f"d{k + 1}": int(n[-1, -10 + k]) for k in range(6, 10)}
    out["avg_size_usd_bn_latest"] = {f"d{k + 1}": round(sz[-1, -10 + k] / 1000, 1) for k in range(6, 10)}
    out["firms_latest"]["q4"], out["firms_latest"]["q5"] = int(n[-1, 7]), int(n[-1, 8])
    out["avg_size_usd_bn_latest"]["q4"] = round(sz[-1, 7] / 1000, 1)
    out["avg_size_usd_bn_latest"]["q5"] = round(sz[-1, 8] / 1000, 1)
    return out


def double_sort(fname: str, label: str) -> dict:
    """Quintile 4 minus quintile 5 of size inside each quintile of the other trait (value-weighted),
    and the average across the five: size among large companies with the other trait held fixed."""
    d, a = table(fname, "Average Value Weighted Returns -- Monthly")
    q4 = a[:, 15:20]
    q5 = a[:, 20:25]
    res = {}
    for name, (s, e) in PERIODS.items():
        m = window(d, s, e) & ~np.isnan(a[:, 15:25]).any(axis=1)
        if m.sum() < 12:
            continue
        diff = q4[m] - q5[m]
        res[name] = {"average_across_trait_quintiles": mt(diff.mean(axis=1)),
                     "by_trait_quintile_low_to_high": [mt(diff[:, k]) for k in range(5)]}
    return {"file": fname, "trait": label, "periods": res}


def junk_among_large() -> dict:
    """Value-weighted operating profitability and asset growth of size quintiles 4 and 5 inside
    each book-to-market quintile (the 5x5 file publishes both per portfolio, per year). Averaged
    over formation years; the difference says whether the smaller of the large are the less
    profitable and faster-investing - the junk Asness et al. control for."""
    f = "25_Portfolios_5x5_CSV.zip"
    out = {}
    for key, mark in (("operating_profitability", "Value Weight Average of OP Calculated as"),
                      ("asset_growth_log", "Value Weight Average of investment")):
        d, a = table(f, mark)
        res = {}
        for name, (s, e) in (("1963-2026", ("1963", "2099")), ("2000-2026", ("2000", "2099")),
                             ("2013-2026", ("2013", "2099"))):
            m = np.array([s <= x[:4] <= e for x in d]) & ~np.isnan(a[:, 15:25]).any(axis=1)
            q4, q5 = a[m, 15:20], a[m, 20:25]
            res[name] = {"q4_mean": round(float(np.nanmean(q4)), 4), "q5_mean": round(float(np.nanmean(q5)), 4),
                         "q4_below_q5_share_of_cells": round(float(np.mean(q4 < q5)), 3), "years": int(m.sum())}
        out[key] = res
    return out


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ b
    s2 = e @ e / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    return {"alpha": round(float(b[0]), 3), "alpha_t": round(float(b[0] / se[0]), 2),
            "betas": [round(float(v), 3) for v in b[1:]], "betas_t": [round(float(v / s), 2) for v, s in zip(b[1:], se[1:])],
            "n": int(len(y))}


def smb_regressions() -> dict:
    d5, f5 = table("F-F_Research_Data_5_Factors_2x3_CSV.zip", ",Mkt-RF,SMB,HML,RMW,CMA,RF")
    dm, mom = table("10_Portfolios_Prior_12_2_CSV.zip", "Value Weight Returns -- Monthly")
    wml = dict(zip(dm, mom[:, 9] - mom[:, 0]))
    keep = [i for i, x in enumerate(d5) if x in wml]
    d = [d5[i] for i in keep]
    f = f5[keep]
    w = np.array([wml[x] for x in d])
    mkt, smb, hml, rmw, cma = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    res = {}
    for name, (s, e) in PERIODS.items():
        m = window(d, s, e)
        if m.sum() < 60:
            continue
        res[name] = {
            "smb_raw": mt(smb[m]),
            "capm": ols(smb[m], mkt[m][:, None]),
            "ff3_plus_mom": ols(smb[m], np.column_stack([mkt[m], hml[m], w[m]])),
            "ff5_ex_smb": ols(smb[m], np.column_stack([mkt[m], hml[m], rmw[m], cma[m]])),
            "ff5_ex_smb_plus_mom": ols(smb[m], np.column_stack([mkt[m], hml[m], rmw[m], cma[m], w[m]])),
        }
    return {"regressors": {"capm": ["Mkt-RF"], "ff3_plus_mom": ["Mkt-RF", "HML", "WML"],
                           "ff5_ex_smb": ["Mkt-RF", "HML", "RMW", "CMA"],
                           "ff5_ex_smb_plus_mom": ["Mkt-RF", "HML", "RMW", "CMA", "WML"]},
            "note": "WML = top minus bottom value-weighted prior-return decile (10_Portfolios_Prior_12_2).",
            "periods": res}


def smb_within_profitability() -> dict:
    """Small minus big inside each profitability tercile (all listed stocks, NYSE median split):
    whether size pays more among the profitable, the conditional Asness et al. emphasise."""
    d, a = table("6_Portfolios_ME_OP_2x3_CSV.zip", "Average Value Weighted Returns -- Monthly")
    res = {}
    for name, (s, e) in PERIODS.items():
        m = window(d, s, e) & ~np.isnan(a).any(axis=1)
        if m.sum() < 60:
            continue
        res[name] = {"weak": mt(a[m, 0] - a[m, 3]), "middle": mt(a[m, 1] - a[m, 4]), "robust": mt(a[m, 2] - a[m, 5])}
    return res


def jkp_market_equity() -> dict | None:
    """A second, independent construction: Jensen, Kelly & Pedersen's (2023) `market_equity`
    factor (US, all stocks, capped value weights, tercile long-short signed by the original paper's
    direction). It is an all-company factor, not a large-company one - read beside SMB, not as a
    measurement among the S&P 500. Optional: skipped if the file is absent."""
    import zipfile
    p = RAW.parent / "jkp" / "usa_all_factors_monthly_vw_cap.zip"
    if not p.exists():
        return None
    import pandas as pd
    with zipfile.ZipFile(p) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    me = df[df["name"] == "market_equity"].sort_values("date")
    d = [x.replace("-", "")[:6] for x in me["date"].astype(str)]
    r = me["ret"].to_numpy(float) * 100
    res = {}
    for name, (s, e) in PERIODS.items():
        m = window(d, s, e)
        if m.sum() >= 60:
            res[name] = mt(r[m])
    return {"direction": int(me["direction"].iloc[0]), "first": d[0], "last": d[-1], "periods": res}


def main() -> None:
    out = {"source": "Kenneth R. French Data Library (CRSP 202608 vintage); returns in percent a month",
           "size_groups": size_deciles(),
           "size_within_value": double_sort("25_Portfolios_5x5_CSV.zip", "book-to-market"),
           "size_within_momentum": double_sort("25_Portfolios_ME_Prior_12_2_CSV.zip", "prior 12-2 return"),
           "junk_among_large": junk_among_large(),
           "smb_regressions": smb_regressions(),
           "smb_within_profitability": smb_within_profitability(),
           "jkp_market_equity": jkp_market_equity()}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    sg = out["size_groups"]
    print("latest month", sg["latest_month"], "firms", sg["firms_latest"], "avg $bn", sg["avg_size_usd_bn_latest"])
    for w in ("vw", "ew"):
        for p, r in sg[w].items():
            print(w, p, "q4-q5", r["q4_minus_q5"], "d9-d10", r["d9_minus_d10"], "d8-d10", r["d8_minus_d10"],
                  "jan", r["q4_minus_q5_january"], "feb-dec", r["q4_minus_q5_feb_dec"])
    for k in ("size_within_value", "size_within_momentum"):
        for p, r in out[k]["periods"].items():
            print(k, p, r["average_across_trait_quintiles"])
    print("junk among large", json.dumps(out["junk_among_large"]))
    for p, r in out["smb_regressions"]["periods"].items():
        print("SMB", p, "raw", r["smb_raw"], "| capm a", r["capm"]["alpha"], r["capm"]["alpha_t"],
              "| ff3+mom a", r["ff3_plus_mom"]["alpha"], r["ff3_plus_mom"]["alpha_t"],
              "| ff5 a", r["ff5_ex_smb"]["alpha"], r["ff5_ex_smb"]["alpha_t"], "betas", r["ff5_ex_smb"]["betas"],
              "| ff5+mom a", r["ff5_ex_smb_plus_mom"]["alpha"], r["ff5_ex_smb_plus_mom"]["alpha_t"])
    for p, r in out["smb_within_profitability"].items():
        print("SMB within OP", p, r)
    print("JKP market_equity", out["jkp_market_equity"])
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
