"""Replay of the pre-registered momentum panic state (research/2026-10-09-momentum-regime.md).

State at a month-end: the S&P 500 total-return index's trailing 24-month return is negative AND its
trailing 126-trading-day realised volatility is above the 80th percentile of that volatility over the
whole history. Prints how often it fires and when. One request (^SP500TR, full history).
"""
import numpy as np
import pandas as pd
import yfinance as yf

px = yf.Ticker("^SP500TR").history(period="max", auto_adjust=True)["Close"].dropna()
px.index = pd.to_datetime(px.index).tz_localize(None)
lr = np.log(px).diff().dropna()
vol126 = lr.rolling(126).std() * np.sqrt(252)
cut = vol126.quantile(0.80)
me = px.resample("ME").last().index
rows = []
for d in me:
    p = px[:d]
    if len(p) < 520:
        continue
    t24 = p.iloc[-1] / p[: d - pd.DateOffset(months=24)].iloc[-1] - 1 if len(p[: d - pd.DateOffset(months=24)]) else np.nan
    v = vol126[:d].iloc[-1]
    rows.append((d, t24, v, bool(t24 < 0 and v > cut)))
df = pd.DataFrame(rows, columns=["month", "ret24", "vol126", "panic"]).dropna()
print(f"history {px.index[0].date()} .. {px.index[-1].date()}; 126-day vol 80th pct = {cut:.1%}")
print(f"month-ends evaluated: {len(df)}; panic state: {int(df.panic.sum())} ({df.panic.mean():.1%})")
runs = (df.panic != df.panic.shift()).cumsum()
for _, g in df[df.panic].groupby(runs[df.panic]):
    print(f"  {g.month.iloc[0]:%Y-%m} .. {g.month.iloc[-1]:%Y-%m} ({len(g)} months)")
print(f"in the state today ({df.month.iloc[-1]:%Y-%m}): {df.panic.iloc[-1]}")
