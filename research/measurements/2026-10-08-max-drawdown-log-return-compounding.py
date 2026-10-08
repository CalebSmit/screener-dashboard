"""Does `max_drawdown_1y` measure the drawdown of the price path?

`factor_engine.fetch_fundamentals` stores **log** returns:

    daily_ret = np.log(closes / closes.shift(1))

and `compute_metrics` step 16d builds the path it measures the drawdown on with

    _cum = np.cumprod(1 + _daily_all)          # 1 + log-return, compounded

The price path is `np.exp(np.cumsum(log_ret))` (equivalently `cumprod(1 + simple)`).
`cumprod(1 + log r)` is a different series: since ln(1+r) <= r it drifts below the
true path, and the drift is path-dependent, so the peak-to-trough ratio taken on it
is not the stock's actual largest fall.

This script downloads real 13-month histories and reports, per ticker, the drawdown
under both definitions, plus what the difference does to the cross-sectional ranking
(the only thing the screener actually uses the number for).

    python research/measurements/2026-10-08-max-drawdown-log-return-compounding.py
"""
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TICKERS = ("AAPL MSFT NVDA AMZN META GOOGL BRK-B JPM XOM UNH JNJ PG V HD MA CVX ABBV "
           "KO PEP COST MRK WMT CRM BAC AMD NFLX ADBE TMO LIN ACN MCD CSCO ABT WFC DHR "
           "TXN NEE PM VZ INTC CMCSA COP CAT HON UNP LOW SPGI MS BA GS").split()


def engine_mdd(log_ret: np.ndarray) -> float:
    """Exactly step 16d as it stands today."""
    cum = np.cumprod(1 + log_ret)
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def price_path_mdd(log_ret: np.ndarray) -> float:
    """The drawdown of the actual price path."""
    cum = np.exp(np.cumsum(log_ret))
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def main() -> None:
    import yfinance as yf

    rows = []
    for t in TICKERS:
        try:
            hist = yf.Ticker(t).history(period="13mo", auto_adjust=True)
            closes = hist["Close"].dropna()
            lr = np.log(closes / closes.shift(1)).dropna().values
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"  !! {t}: {type(exc).__name__}: {exc}")
            continue
        if len(lr) < 200:
            print(f"  -- {t}: only {len(lr)} returns, skipped (engine needs 200)")
            continue
        e, p = engine_mdd(lr), price_path_mdd(lr)
        rows.append({"ticker": t, "n": len(lr), "engine": e, "price_path": p,
                     "diff_pp": (e - p) * 100})

    df = pd.DataFrame(rows)
    if df.empty:
        print("no data")
        return
    df = df.sort_values("diff_pp")

    print(f"\n{len(df)} tickers with >= 200 daily returns\n")
    print(f"{'ticker':8s} {'engine':>9s} {'price path':>11s} {'diff (pp)':>10s}")
    for r in df.itertuples():
        print(f"{r.ticker:8s} {r.engine*100:8.2f}% {r.price_path*100:10.2f}% {r.diff_pp:10.3f}")

    d = df["diff_pp"]
    print(f"\ndifference, engine minus price path, in percentage points of drawdown:")
    print(f"  mean {d.mean():+.3f}   median {d.median():+.3f}   "
          f"min {d.min():+.3f}   max {d.max():+.3f}   |max| {d.abs().max():.3f}")
    print(f"  engine reports a LARGER fall than reality for "
          f"{(d < 0).sum()}/{len(d)} tickers")

    # The screener only uses the cross-sectional rank.
    re_, rp = df["engine"].rank(), df["price_path"].rank()
    print(f"\nranking impact over these {len(df)} names "
          f"(the screener scores the percentile, not the level):")
    print(f"  Spearman {re_.corr(rp, method='spearman'):.6f}")
    moved = (re_ - rp).abs()
    print(f"  places moved: max {moved.max():.0f}, mean {moved.mean():.2f}, "
          f"{(moved > 0).sum()} of {len(df)} move at all")


if __name__ == "__main__":
    main()
