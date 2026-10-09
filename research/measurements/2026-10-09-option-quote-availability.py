"""Is the option chain usable at the hour the data loop runs?

Queue item 2 of ``plan/context-layer.md``: "Option quotes at 2 AM are the previous close; some
chains have no bid/ask. Measure the share of stocks with ``os == 'ok'``".

Two things this prints:

1. **From the record.** ``data/context_log/*.parquet`` holds ``_ctx_opt_status`` for every stock on
   every run date, so the share is a read, not an estimate. No network needed.
2. **A live probe**, so the *reason* is visible rather than inferred: for a sample of tickers it
   fetches the same expiry the pipeline would and prints what the at-the-money rows actually carry
   (bid, ask, lastPrice, impliedVolatility, openInterest) and the status
   ``context_signals.options_context`` derives from them. Pass ``--probe N`` to size the sample;
   ``--probe 0`` skips the network entirely.

Run it at different hours to see whether the fields are a function of the clock.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd  # noqa: E402

STATUSES = ["ok", "partial", "stale-quotes", "no-atm", "no-chain"]


def from_the_record() -> None:
    files = sorted(glob.glob(os.path.join("data", "context_log", "*.parquet")))
    if not files:
        print("No context log yet - data/context_log/ is empty.")
        return
    print("Option status by run date, from data/context_log/ (the record the plan requires):")
    header = f"{'run date':<12}{'n':>5}" + "".join(f"{s:>14}" for s in STATUSES) + f"{'usable %':>10}"
    print(header)
    print("-" * len(header))
    for f in files:
        d = pd.read_parquet(f, columns=None)
        date = os.path.basename(f).replace(".parquet", "")
        if "_ctx_opt_status" not in d.columns:
            print(f"{date:<12}{len(d):>5}   (no _ctx_opt_status column)")
            continue
        vc = d["_ctx_opt_status"].value_counts(dropna=False)
        row = f"{date:<12}{len(d):>5}"
        for s in STATUSES:
            row += f"{int(vc.get(s, 0)):>14}"
        row += f"{100.0 * int(vc.get('ok', 0)) / max(len(d), 1):>9.1f}%"
        print(row)
    print()
    print("'ok' means both the straddle (bid/ask mid on the at-the-money call and put) and the")
    print("at-the-money implied volatility came back usable. Everything the page shows about")
    print("options - expected move, ATM IV, put skew - needs one or both of those.")


def probe(n: int) -> None:
    if n <= 0:
        return
    import yfinance as yf

    from context_signals import choose_expiry, options_context

    tickers = ["AAPL", "MSFT", "JPM", "XOM", "KO", "NVDA", "WMT", "PG", "UNH", "CAT",
               "T", "MRK", "HON", "LIN", "ADBE", "DE", "GIS", "APA", "BBY", "HST"][:n]
    today = datetime.now(timezone.utc).date()
    print()
    print(f"Live probe at {datetime.now().strftime('%Y-%m-%d %H:%M')} local "
          f"/ {datetime.now(timezone.utc).strftime('%H:%M')} UTC, {len(tickers)} tickers.")
    print(f"{'ticker':<8}{'expiry':<12}{'spot':>9}{'strike':>9}{'c bid/ask':>14}{'p bid/ask':>14}"
          f"{'c last':>8}{'c IV':>8}{'c OI':>9}{'status':>14}")
    counts = {s: 0 for s in STATUSES}
    counts["no-expiry"] = 0
    for tk in tickers:
        t = yf.Ticker(tk)
        try:
            spot = float(t.fast_info["lastPrice"])
        except Exception:  # noqa: BLE001
            try:
                spot = float(t.history(period="5d")["Close"].iloc[-1])
            except Exception as e:  # noqa: BLE001
                print(f"{tk:<8}no price: {type(e).__name__}")
                continue
        exp, spans = choose_expiry(t.options, today, None)
        if not exp:
            counts["no-expiry"] += 1
            print(f"{tk:<8}{'(no expiry in window)':<12}")
            continue
        ch = t.option_chain(exp)
        ctx = options_context(ch.calls, ch.puts, spot, exp, today, spans)
        status = ctx.get("_ctx_opt_status", "?")
        counts[status] = counts.get(status, 0) + 1
        strike = ctx.get("_ctx_opt_atm_strike")
        c = p = None
        if strike is not None:
            cr = ch.calls.loc[ch.calls["strike"] == strike]
            pr = ch.puts.loc[ch.puts["strike"] == strike]
            c = cr.iloc[0].to_dict() if len(cr) else None
            p = pr.iloc[0].to_dict() if len(pr) else None

        def ba(row):
            if not row:
                return "-"
            return f"{row.get('bid')}/{row.get('ask')}"

        print(f"{tk:<8}{exp:<12}{spot:>9.2f}{(strike if strike else float('nan')):>9.2f}"
              f"{ba(c):>14}{ba(p):>14}"
              f"{(c or {}).get('lastPrice', float('nan')):>8.2f}"
              f"{(c or {}).get('impliedVolatility', float('nan')):>8.3f}"
              f"{(c or {}).get('openInterest', float('nan')):>9.0f}{status:>14}")
    print()
    print("Probe totals: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=10, help="tickers to fetch live (0 = none)")
    a = ap.parse_args()
    from_the_record()
    probe(a.probe)
