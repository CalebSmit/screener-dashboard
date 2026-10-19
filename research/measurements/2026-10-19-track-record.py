"""What the track record's figures can and cannot show - the measurements behind
research/2026-10-19-track-record.md.

Run (offline - reads the committed snapshots and the gitignored price cache that the data loop
keeps at data/track/prices.parquet; copy it from the main checkout if this is a fresh worktree):

    python research/measurements/2026-10-19-track-record.py

It prints:

1. that the record rebuilds from disk exactly as data/track_record.json published it;
2. the day-by-day difference between the top 25 and RSP: its size, its spread, and how many
   standard errors from zero it is (the "how long would it take" arithmetic in note section 3);
3. the per-period record (5 of 7 ahead of RSP) against a coin toss;
4. the top fifth minus bottom fifth, period by period;
5. the price path of the two picks left out (EA, HOLX) as the screener recorded it, which shows
   whether a takeover premium was missed (note section 5);
6. the turnover at each rebalance and the one-way cost at which the lead over RSP disappears
   (queue item 6) - read from track_record.build itself, so it measures what the page shows.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import track_record as tr  # noqa: E402


def cached_prices() -> pd.DataFrame:
    return pd.read_parquet(tr.PRICE_CACHE)


_BUILD = tr.build
_LATE = tr.runs_after_the_close


def record(prices: pd.DataFrame, mode: str = "current") -> dict:
    """track_record.build_from_disk on the cached prices, with the daily curves kept.

    ``mode``: "current" - the code as committed; "published" - no run treated as after the close
    (the rule until 2026-10-19, which is what data/track_record.json holds); "next_day" - every
    rebalance enters one trading day later, the sensitivity in note section 3."""
    tr.load_prices = lambda tickers, start, end=None, download=None: prices.copy()
    tr.build = lambda *a, **k: _BUILD(*a, **{**k, "return_curves": True})
    if mode == "published":
        tr.runs_after_the_close = lambda dates, px, **k: []
    elif mode == "next_day":
        tr.runs_after_the_close = lambda dates, px, **k: list(dates)
    try:
        return tr.build_from_disk(today=date(2026, 10, 19), write=False)
    finally:
        tr.build, tr.runs_after_the_close = _BUILD, _LATE


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    prices = cached_prices()
    old = record(prices, "published")
    out = record(prices)
    nxt = record(prices, "next_day")
    ref = json.loads((ROOT / "data" / "track_record.json").read_text(encoding="utf-8"))
    same = all(old.get(k) == ref.get(k) for k in ("total", "spread", "max_dd", "beat_rsp", "periods_compared"))
    print(f"1. under the old entry rule the rebuild matches the published record: {same}")
    print(f"   published (old rule): window {old['start']} -> {old['end']}; total {old['total']};"
          f" spread {old['spread']}; ahead of RSP {old['beat_rsp']} of {old['periods_compared']}")
    print(f"   runs read as after the close: {out.get('after_close')}; untestable (no recorded prices): {out.get('untimed')}")
    print(f"   corrected rule: window {out['start']} -> {out['end']}, {out['days']} days;"
          f" total {out['total']}; spread {out['spread']}; ahead of RSP {out['beat_rsp']} of {out['periods_compared']}")
    print(f"   sensitivity - every rebalance one trading day later: total {nxt['total']}; spread {nxt['spread']};"
          f" ahead of RSP {nxt['beat_rsp']} of {nxt['periods_compared']}")
    for a, b in zip(old["periods"], out["periods"]):
        print(f"   {a['date']}: entry {a['entry']} -> {b['entry']}; top {a['top']:+.4f} -> {b['top']:+.4f};"
              f" RSP {a.get('RSP', float('nan')):+.4f} -> {b.get('RSP', float('nan')):+.4f}")

    # 2. day by day, top 25 against RSP, on the published weekly series' daily parent
    curves = out.get("_curves") or {}
    top, rsp = curves.get("top"), curves.get("RSP")
    if top is not None and rsp is not None:
        top, rsp = top.align(rsp, join="inner")
        r_top, r_rsp = top.pct_change().dropna(), rsp.pct_change().dropna()
        ex = (r_top - r_rsp).dropna()
        n = len(ex)
        se = ex.std(ddof=1) / math.sqrt(n)
        t = ex.mean() / se
        print(f"2. trading days {n}; mean daily excess {ex.mean() * 1e4:.2f} bp; sd {ex.std(ddof=1) * 1e4:.1f} bp;"
              f" t = {t:.2f}")
        print(f"   cumulative: top {top.iloc[-1] / 100 - 1:+.4f}, RSP {rsp.iloc[-1] / 100 - 1:+.4f};"
              f" difference {top.iloc[-1] / 100 - rsp.iloc[-1] / 100:+.4f}")
        te_ann = ex.std(ddof=1) * math.sqrt(252)
        print(f"   tracking error (daily sd x sqrt(252), a volatility scaling, not a return): {te_ann:.3f}")
        need = (2.0 / t) ** 2 * n if t > 0 else float("inf")
        print(f"   at this same daily lead and noise, two standard errors needs {need:.0f} trading days"
              f" = {need / 252:.1f} years")
        for ir in (0.25, 0.5, 1.0):
            print(f"   an information ratio of {ir}: two standard errors after {(2 / ir) ** 2:.0f} years")

    # 3. periods ahead of RSP against a coin toss
    k, m = out["beat_rsp"], out["periods_compared"]
    p = sum(math.comb(m, j) for j in range(k, m + 1)) / 2 ** m
    print(f"3. ahead of RSP in {k} of {m} periods; P(>= {k} of {m} | coin toss) = {p:.3f}")
    print(f"   as track_record.build publishes them: beat_rsp_chance {out.get('beat_rsp_chance')};"
          f" excess {out.get('excess')}")
    lens = [(pd.Timestamp(x["exit"]) - pd.Timestamp(x["entry"])).days for x in out["periods"]]
    print(f"   period lengths, calendar days: {lens}")

    # 4. spread per period
    sp = [x["q1"] - x["q5"] for x in out["periods"] if x.get("q1") is not None and x.get("q5") is not None]
    print(f"4. top fifth minus bottom fifth per period (pp): {[round(v * 100, 2) for v in sp]}")
    sp_a = np.array(sp)
    print(f"   positive in {int((sp_a > 0).sum())} of {len(sp_a)}; mean {sp_a.mean() * 100:.2f}pp;"
          f" t across periods {sp_a.mean() / (sp_a.std(ddof=1) / math.sqrt(len(sp_a))):.2f}")

    # 5. the two picks left out
    rec = tr.recorded_closes(["EA", "HOLX"])
    for t_ in ("EA", "HOLX"):
        s = rec[t_].dropna() if t_ in rec.columns else pd.Series(dtype=float)
        if len(s):
            print(f"5. {t_}: recorded {s.index.min().date()} {s.iloc[0]:.2f} -> {s.index.max().date()} {s.iloc[-1]:.2f};"
                  f" range {s.min():.2f}-{s.max():.2f} ({(s.max() / s.min() - 1) * 100:.1f}% wide) over {len(s)} runs")

    # 6. turnover and break-even cost
    to = out.get("turnover") or {}
    print(f"6. turnover block: {json.dumps(to)}")
    print(f"   per period: {[x.get('turnover') for x in out['periods']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
