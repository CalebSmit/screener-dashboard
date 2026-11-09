"""How far does a completed stock-funded acquisition move a stock's scored valuation before
its earnings catch up - measured on this project's own committed runs.

Fox completed its acquisition of Roku on 2026-11-06 (8-K of that date, "the Closing Date").
The source added the newly issued Class A shares to FOXA's share count and market value the
following session; FOXA's net income, revenue and book value are still Fox standalone and will
be until the December quarter is filed. Every metric whose denominator is market value or
enterprise value therefore moved at once, and nothing in the pipeline said so.

Run with no arguments. Reads `runs/*/` and writes nothing.

    python -I -X utf8 research/measurements/2026-11-09-post-acquisition-market-value.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# The metrics whose denominator is the market value or the enterprise value built from it.
MC_DRIVEN = ("earnings_yield", "fcf_yield", "ev_ebitda", "ev_sales", "pb_ratio", "size_log_mcap")
PAIRS = {"GOOGL": "GOOG", "FOXA": "FOX", "NWSA": "NWS"}


def runs_in_order():
    out = []
    for d in sorted((ROOT / "runs").glob("*")):
        meta = d / "meta.json"
        if not (meta.exists() and (d / "05_final_scored.parquet").exists()):
            continue
        try:
            start = json.loads(meta.read_text(encoding="utf-8")).get("start_time", "")
        except Exception:  # noqa: BLE001
            continue
        out.append((start, d))
    return [d for _, d in sorted(out)]


def main() -> None:
    order = runs_in_order()
    print("runs with a scored artifact: %d" % len(order))

    print("\n== 1. FOXA's market value, share count and scored valuation, run by run")
    print("%-22s %-10s %13s %13s %8s %9s %7s %9s"
          % ("run start", "run", "marketCap", "shares(mc/px)", "price", "earn_yld", "rank", "composite"))
    hist = []
    for d in order[-8:]:
        start = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("start_time", "")
        raw = pd.read_parquet(d / "00_raw_fetch.parquet").set_index("Ticker")
        sc = pd.read_parquet(d / "05_final_scored.parquet").set_index("Ticker")
        if "FOXA" not in raw.index:
            continue
        mc = float(raw.loc["FOXA", "marketCap"])
        px = float(raw.loc["FOXA", "currentPrice"])
        m = pd.read_parquet(d / "02_outliers_flagged.parquet").set_index("Ticker")
        ey = float(m.loc["FOXA", "earnings_yield"])
        rank = sc.loc["FOXA", "Rank"] if "FOXA" in sc.index else float("nan")
        comp = sc.loc["FOXA", "Composite"] if "FOXA" in sc.index else float("nan")
        hist.append((start, d.name, mc, mc / px, px, ey, rank, comp))
        print("%-22s %-10s %13.0f %13.0f %8.2f %9.5f %7s %9s"
              % (start[:19], d.name, mc, mc / px, px, ey, rank, comp))

    if len(hist) >= 2:
        a, b = hist[-2], hist[-1]
        print("\n   last two runs: price %.2f -> %.2f (%+.2f%%), market value %+.1f%%, "
              "implied shares %+.1f%%, earnings yield %+.1f%%"
              % (a[4], b[4], 100 * (b[4] / a[4] - 1), 100 * (b[2] / a[2] - 1),
                 100 * (b[3] / a[3] - 1), 100 * (b[5] / a[5] - 1)))
        print("   rank %s -> %s, composite %s -> %s" % (a[6], b[6], a[7], b[7]))

    print("\n== 2. Every market-value-driven metric, across the same two runs")
    if len(order) >= 2:
        prev, cur = order[-2], order[-1]
        mp = pd.read_parquet(prev / "02_outliers_flagged.parquet").set_index("Ticker")
        mc_ = pd.read_parquet(cur / "02_outliers_flagged.parquet").set_index("Ticker")
        for col in MC_DRIVEN:
            if col not in mp.columns or col not in mc_.columns:
                continue
            a, b = mp.loc["FOXA", col], mc_.loc["FOXA", col]
            if pd.isna(a) or pd.isna(b):
                print("   %-16s %12s -> %12s" % (col, a, b))
                continue
            print("   %-16s %12.5f -> %12.5f  (%+.1f%%)"
                  % (col, a, b, 100 * (b / a - 1) if a else float("nan")))

    print("\n== 3. How unusual is a market value that moves while the price does not?")
    print("   Across every consecutive pair of runs, the stocks whose market value moved by")
    print("   more than 5% MORE than their price did - i.e. the share count behind it changed.")
    rows = []
    for prev, cur in zip(order, order[1:]):
        try:
            rp = pd.read_parquet(prev / "00_raw_fetch.parquet").set_index("Ticker")
            rc = pd.read_parquet(cur / "00_raw_fetch.parquet").set_index("Ticker")
            sp = json.loads((prev / "meta.json").read_text(encoding="utf-8")).get("start_time", "")
            sc_ = json.loads((cur / "meta.json").read_text(encoding="utf-8")).get("start_time", "")
        except Exception:  # noqa: BLE001
            continue
        common = rp.index.intersection(rc.index)
        for col in ("marketCap", "currentPrice"):
            if col not in rp.columns or col not in rc.columns:
                common = common[:0]
        if not len(common):
            continue
        sh_p = pd.to_numeric(rp.loc[common, "marketCap"], errors="coerce") / \
            pd.to_numeric(rp.loc[common, "currentPrice"], errors="coerce")
        sh_c = pd.to_numeric(rc.loc[common, "marketCap"], errors="coerce") / \
            pd.to_numeric(rc.loc[common, "currentPrice"], errors="coerce")
        move = (sh_c / sh_p - 1.0).dropna()
        big = move[move.abs() > 0.05]
        rows.append((sp[:10], sc_[:10], len(common), len(big),
                     ", ".join("%s %+.1f%%" % (t, 100 * v)
                               for t, v in big.sort_values(key=abs, ascending=False).head(6).items())))
    print("   %-11s %-11s %6s %6s  %s" % ("from", "to", "common", "moved", "largest"))
    for r in rows:
        print("   %-11s %-11s %6d %6d  %s" % r)
    tot = sum(r[3] for r in rows)
    print("\n   %d share-count moves over 5%% across %d run pairs (%.2f a run)"
          % (tot, len(rows), tot / max(1, len(rows))))

    print("\n== 4. The share-count basis of each pair of listed lines, newest run")
    cur = order[-1]
    raw = pd.read_parquet(cur / "00_raw_fetch.parquet").set_index("Ticker")
    for kept, second in PAIRS.items():
        if kept not in raw.index or second not in raw.index:
            continue
        bk = float(raw.loc[kept, "marketCap"]) / float(raw.loc[kept, "currentPrice"])
        bs = float(raw.loc[second, "marketCap"]) / float(raw.loc[second, "currentPrice"])
        print("   %-5s/%-5s  %15.0f vs %15.0f   ratio %.6f  %s"
              % (kept, second, bk, bs, bk / bs,
                 "one share count" if abs(bk / bs - 1) < 1e-4 else "TWO SHARE COUNTS"))


if __name__ == "__main__":
    main()
