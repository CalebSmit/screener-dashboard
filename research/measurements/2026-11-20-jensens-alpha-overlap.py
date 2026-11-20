"""How much of Jensen's alpha is the 12-month return the Momentum category already scores?

research/2026-11-20-jensens-alpha-in-momentum.md, section 2. Descriptive only: rank correlations
between signals, no forward return is computed anywhere in this file (CLAUDE.md rules 4 and 5).

Part A - the live run (default: run_context.latest_run_dir()). Sector percentiles built the
engine's way - midpoint rank (k - 0.5)/n within GICS sector - for the scored metric and for
three alternatives, then Spearman correlations across the stocks that have all of them:

  jensens_alpha  as scored: r12 - [rf + beta (m12 - rf)], r12 unskipped (factor_engine ~2906)
  alpha_12_1     the same CAPM line over the 11 months ending a month ago (the 12-1 window)
  return_12_1, return_6m, return_12m, return_1m

Part B - month-ends 2012-01 .. 2026-09 from backtest v2's own daily prices (cache/v2_daily.parquet,
adjusted closes; market = ^SP500TR from cache/v2_index.parquet, rf = ^IRX). Every stock with a full
year of prices at the month-end; whole-cross-section ranks (no sector in that file). Per month:
Spearman between the scored alpha, its 12-1 version, a CAPM residual momentum (daily residuals
from a 12-month CAPM fit, summed over the 12-1 window and divided by their standard deviation - a
simplification of Blitz, Huij & Martens 2011, who use a 36-month Fama-French 3-factor fit on
monthly returns), return_12_1 and return_1m. Reported as the median and 10th/90th percentile
across months.

Run:  python research/measurements/2026-11-20-jensens-alpha-overlap.py [run_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sector_pct(s: pd.Series, sector: pd.Series) -> pd.Series:
    """Midpoint percentile within sector, the engine's (k - 0.5)/n (metric audit fix 7)."""
    out = pd.Series(np.nan, index=s.index)
    for _, idx in s.groupby(sector).groups.items():
        v = s.loc[idx].dropna()
        if len(v):
            out.loc[v.index] = (v.rank(method="average") - 0.5) / len(v)
    return out


def close_on_or_before(closes: dict, target: pd.Timestamp) -> float:
    keys = sorted(k for k in closes if pd.Timestamp(k) <= target)
    return float(closes[keys[-1]]) if keys else float("nan")


def part_a(run_dir: Path) -> dict:
    rm = pd.read_parquet(run_dir / "01_raw_metrics.parquet")
    rf_ = pd.read_parquet(run_dir / "00_raw_fetch.parquet")
    if "ticker" in rm.columns:
        rm = rm.set_index("ticker")
    elif "Ticker" in rm.columns:
        rm = rm.set_index("Ticker")
    tick_col = "ticker" if "ticker" in rf_.columns else "Ticker"
    rf_ = rf_.set_index(tick_col)
    mc = json.load(open(run_dir / "market_closes.json"))
    closes = mc["closes"]
    last = pd.Timestamp(max(closes))
    m_now = close_on_or_before(closes, last)
    m_1m = close_on_or_before(closes, last - pd.Timedelta(days=30))
    m_12m = close_on_or_before(closes, last - pd.Timedelta(days=365))
    m12_1 = m_1m / m_12m - 1

    df = rm[["Sector", "return_12_1", "return_6m", "return_12m", "beta", "jensens_alpha",
             "_ja_rf", "_ja_mkt"]].copy()
    df["return_1m"] = rf_["price_latest"] / rf_["price_1m_ago"] - 1
    rf = float(df["_ja_rf"].dropna().iloc[0])
    rf11 = (1 + rf) ** (11 / 12) - 1
    df["alpha_12_1"] = df["return_12_1"] - (rf11 + df["beta"] * (m12_1 - rf11))
    # Check the scored figure is what this file thinks it is before reading anything off it.
    recon = df["return_12m"] - (rf + df["beta"] * (df["_ja_mkt"] - rf))
    max_err = float((recon - df["jensens_alpha"]).abs().max())

    cols = ["jensens_alpha", "alpha_12_1", "return_12_1", "return_6m", "return_12m", "return_1m"]
    pct = pd.DataFrame({c: sector_pct(df[c], df["Sector"]) for c in cols}).dropna()
    corr = pct.corr(method="spearman")
    # How far the beta term moves a stock: alpha = r12 - beta (m12 - rf) - rf, so across stocks
    # the beta term's spread against r12's spread says how much of alpha is not r12.
    beta_term = df["beta"] * (df["_ja_mkt"] - rf)
    return {
        "run": run_dir.name,
        "market_last_close": str(last.date()),
        "m12 (scored)": round(float(df["_ja_mkt"].dropna().iloc[0]), 4),
        "m12_1": round(m12_1, 4),
        "m_1m": round(m_now / m_1m - 1, 4),
        "rf": rf,
        "reconstruction_max_abs_err": max_err,
        "n_stocks": int(len(pct)),
        "sd_return_12m": round(float(df["return_12m"].std()), 4),
        "sd_beta_term": round(float(beta_term.std()), 4),
        "spearman_sector_pct": corr.round(3).to_dict(),
    }


def part_b() -> dict:
    d = pd.read_parquet(ROOT / "cache" / "v2_daily.parquet", columns=["date", "adj_close", "ticker"])
    d["date"] = pd.to_datetime(d["date"])
    px = d.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    ix = pd.read_parquet(ROOT / "cache" / "v2_index.parquet")
    ix["date"] = pd.to_datetime(ix["date"])
    mkt = ix[ix.ticker == "^SP500TR"].set_index("date")["close"].reindex(px.index).ffill()
    irx = ix[ix.ticker == "^IRX"].set_index("date")["close"].reindex(px.index).ffill() / 100.0

    lr = np.log(px / px.shift(1))
    mlr = np.log(mkt / mkt.shift(1))
    month_ends = px.groupby(px.index.to_period("M")).apply(lambda g: g.index.max())
    # Month-end simple returns for the 36-month residual-momentum regression.
    mpx = px.groupby(px.index.to_period("M")).last()
    mret = mpx / mpx.shift(1) - 1
    mmkt = mkt.groupby(mkt.index.to_period("M")).last().pct_change()
    mrf = (irx.groupby(irx.index.to_period("M")).last() / 12.0).shift(1).fillna(0.0)
    month_ends = [t for t in month_ends if pd.Timestamp("2012-01-01") <= t <= pd.Timestamp("2026-09-30")]

    pairs = [("jensens_alpha", "return_12_1"), ("jensens_alpha", "return_12m"),
             ("jensens_alpha", "return_1m"), ("alpha_12_1", "return_12_1"),
             ("resid_mom", "return_12_1"), ("resid_mom", "alpha_12_1"),
             ("jensens_alpha", "alpha_12_1"), ("return_12_1", "return_1m"),
             ("alpha_12_1", "return_1m"), ("resid_mom", "return_1m")]
    rows = []
    for t in month_ends:
        t0, t1 = t - pd.DateOffset(years=1), t - pd.DateOffset(days=30)
        win = lr.loc[(lr.index > t0) & (lr.index <= t)]
        mwin = mlr.loc[win.index]
        ok = win.notna().sum() >= 240
        w = win.loc[:, ok]
        if w.shape[1] < 100:
            continue
        p_t = px.loc[:t].iloc[-1][w.columns]
        p_1 = px.loc[:t1].iloc[-1][w.columns]
        p_12 = px.loc[:t0].iloc[-1][w.columns]
        m_t, m_1, m_12 = (mkt.loc[:t].iloc[-1], mkt.loc[:t1].iloc[-1], mkt.loc[:t0].iloc[-1])
        rf = float(irx.loc[:t].iloc[-1])
        r12, r12_1, r1 = p_t / p_12 - 1, p_1 / p_12 - 1, p_t / p_1 - 1
        mk = mwin.values
        X = w.fillna(0.0).values
        mv = (mk - mk.mean())
        beta = (mv @ (X - X.mean(axis=0))) / (mv @ mv)
        beta = pd.Series(beta, index=w.columns)
        m12, m12_1 = m_t / m_12 - 1, m_1 / m_12 - 1
        rf11 = (1 + rf) ** (11 / 12) - 1
        a12 = r12 - (rf + beta * (m12 - rf))
        a12_1 = r12_1 - (rf11 + beta * (m12_1 - rf11))
        # Residual momentum, Blitz, Huij & Martens (2011) construction with CAPM in place of
        # FF3: a 36-month regression of monthly excess returns (with intercept), residuals of
        # months t-11..t-1 (the 12-1 window) summed and divided by their own sd. NOT fitted on
        # the 12-month window: with an intercept those residuals sum to zero, which makes the
        # 12-1 sum exactly minus the latest month's (the first version of this file did that).
        rm_ = pd.Series(np.nan, index=w.columns)
        k = mret.index.get_indexer([t.to_period("M")])[0]
        if k >= 35:
            Y = mret.iloc[k - 35:k + 1].reindex(columns=w.columns)
            Y = Y.loc[:, np.isfinite(Y.values).all(axis=0)]
            mx = mmkt.reindex(mret.index).iloc[k - 35:k + 1].values
            rfx = mrf.reindex(mret.index).iloc[k - 35:k + 1].values
            if Y.shape[1] and np.isfinite(mx).all() and np.isfinite(rfx).all():
                Ye = Y.values - rfx[:, None]
                Xd = np.column_stack([np.ones(36), mx - rfx])
                coef, *_ = np.linalg.lstsq(Xd, Ye, rcond=None)
                e = Ye - Xd @ coef
                e121 = e[-12:-1]
                rm_.loc[Y.columns] = e121.sum(axis=0) / e121.std(axis=0, ddof=1)
        f = pd.DataFrame({"jensens_alpha": a12, "alpha_12_1": a12_1, "return_12_1": r12_1,
                          "return_12m": r12, "return_1m": r1, "resid_mom": rm_})
        f = f.dropna(subset=["jensens_alpha", "alpha_12_1", "return_12_1", "return_1m"])
        if f["resid_mom"].notna().sum() < 100:
            f = f.drop(columns="resid_mom")
        c = f.corr(method="spearman")
        rows.append({"month": str(t.date()), "n": len(f),
                     **{f"{a}~{b}": (c.loc[a, b] if a in c and b in c else np.nan) for a, b in pairs}})
    res = pd.DataFrame(rows)
    summ = {k: {"median": round(float(res[k].median()), 3), "p10": round(float(res[k].quantile(.1)), 3),
                "p90": round(float(res[k].quantile(.9)), 3), "months": int(res[k].notna().sum())}
            for k in res.columns if "~" in k}
    return {"months": len(res), "first": res["month"].iloc[0], "last": res["month"].iloc[-1],
            "median_n": int(res["n"].median()), "spearman_across_months": summ}


if __name__ == "__main__":
    import run_context
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(run_context.latest_run_dir())
    print(json.dumps({"A_live_run": part_a(run_dir), "B_history": part_b()}, indent=1, default=str))
