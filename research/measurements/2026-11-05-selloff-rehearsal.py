"""Rehearse the 2026-11-06 data run against the 2026-11-05 sell-off, before it happens.

On 2026-11-05 the S&P 500 fell 4.4% (the technology sector ETF -8.0%, staples -0.8%, the VIX
+70% to 28). The 02:00 run on 11-06 is the first one this project has run after a day like it:
the input gates in ``scripts/check_run_health.py`` were replayed on 2026-10-22 over every page
published since 2026-07-28, and no day in that window moved the market by more than ~2%. A
rank-continuity limit set on calm days could refuse a real market move as a scrambled run, and a
refused run leaves the published ranking standing for every later run to be compared with.

This script asks the question with the engine's own code, before the run does:

1. It takes the 11-05 02:00 run's own metric table (``02_outliers_flagged.parquet``, prices to
   the 11-04 close) and re-scores it with the engine's scoring functions under that run's own
   effective weights - the **control**, which must reproduce the published ranks exactly.
2. It downloads every stock's 11-04 and 11-05 closes (one bulk chart request per ticker, no
   ``.info`` or statements - a tenth of a full run's traffic) and moves each metric a one-day
   price change can move, by its own definition in ``factor_engine.py``:

   * valuation (earnings / FCF yield, EV multiples, size, price-target upside) through
     ``lookahead.restate_at_price`` - the diagnostic's own restatement, EV restated as
     ``ev + mc*(r-1)``;
   * volatility, beta: the day appended to the engine's own 1-year moments (its published
     ``_vol_sd``, ``_beta_cov``, ``_beta_var``), the oldest day kept - an approximation of one
     day in ~250, stated rather than hidden;
   * max drawdown: the worse of the published one and the fall from the published peak to the
     11-05 close;
   * Jensen's alpha: the engine's own CAPM terms (``_ja_*``) with the stock's and the market's
     12-month returns carried one day forward;
   * ``return_12_1`` and ``return_6m`` are **unchanged by construction**: both end at the price
     a month ago (the skip-month convention, ``factor_engine`` metrics 13 and 14), so a one-day
     move cannot reach them;
   * everything else - filings, analyst estimates - does not move with the price.

3. It re-scores the shocked table the same way and puts it through ``rank_continuity`` against
   the page the site serves now, with the gate's own thresholds.

Nothing here is decision-grade for the methodology (rule 5) and nothing is published: it says
whether tomorrow's run is likely to be refused, and why.

Run from the repo root:  python research/measurements/2026-11-05-selloff-rehearsal.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402
import lookahead  # noqa: E402
from factor_engine import (apply_percentile_transform, compute_category_scores,  # noqa: E402
                           compute_composite, compute_sector_percentiles, rank_stocks)

RUN = ROOT / "runs" / "7bf31db85519"           # the 2026-11-05 02:00 run (11-04 closes)
DAY0, DAY1 = "2026-11-04", "2026-11-05"
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def run_cfg() -> dict:
    cfg = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
    eff = json.loads((RUN / "effective_weights.json").read_text(encoding="utf-8"))
    cfg["factor_weights"] = dict(eff["factor_weights"])
    cfg["metric_weights"] = copy.deepcopy(eff["metric_weights"])
    return cfg


def score(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = compute_sector_percentiles(df.copy())
    s = apply_percentile_transform(s, cfg)
    s = compute_category_scores(s, cfg)
    s = compute_composite(s, copy.deepcopy(cfg))
    return rank_stocks(s)


def closes(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    want = [t.replace(".", "-") for t in tickers] + ["^SP500TR"]
    px = yf.download(want, start="2026-10-29", end="2026-11-06", interval="1d",
                     auto_adjust=True, progress=False, threads=True)["Close"]
    px.columns = [c.replace("-", ".") if c != "^SP500TR" else c for c in px.columns]
    px.index = px.index.strftime("%Y-%m-%d")
    return px.loc[[DAY0, DAY1]]


def shock(base: pd.DataFrame, raw: pd.DataFrame, px: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    df = base.set_index("Ticker").copy()
    r = (px.loc[DAY1] / px.loc[DAY0]).reindex(df.index)
    m = float(px.loc[DAY1, "^SP500TR"] / px.loc[DAY0, "^SP500TR"])
    x, xm = r - 1.0, m - 1.0
    rw = raw.set_index("Ticker").reindex(df.index)

    # Valuation, size and price-target upside: the diagnostic's own restatement.
    snap = df.copy()
    snap["market_cap"] = pd.to_numeric(rw["marketCap"], errors="coerce")
    # The engine's own EV fallback (factor_engine.compute_metrics): market value + debt - cash
    # where the quote carries none. Without it six stocks lose every EV metric here, which the
    # run itself would not do (Pfizer fell 166 places in the first draft for that reason alone).
    def num(c):
        return pd.to_numeric(rw[c], errors="coerce") if c in rw.columns else pd.Series(np.nan, index=rw.index)
    ev = num("enterpriseValue")
    debt = num("totalDebt").fillna(num("totalDebt_bs"))
    cash = num("totalCash").fillna(num("cash_bs"))
    ev = ev.where(ev.notna() & (ev != 0), num("marketCap") + debt - cash)
    snap["enterprise_value"] = ev
    snap["price"] = pd.to_numeric(rw["price_latest"], errors="coerce")
    snap["pt_mean"] = pd.to_numeric(rw["targetMeanPrice"], errors="coerce")
    rest = lookahead.restate_at_price(snap, r, clamps=cfg.get("metric_clamps"))
    for c in lookahead.RESTATED_HERE:
        if c in df.columns:
            ok = r.notna()
            df.loc[ok, c] = rest.loc[ok, c]

    # Volatility: one more day in the engine's own daily SD (n ~ 251 -> 252).
    sd = pd.to_numeric(df["_vol_sd"], errors="coerce")
    n = 251
    sd_new = np.sqrt((sd ** 2 * (n - 1) + x ** 2) / n)
    df["volatility"] = (sd_new * np.sqrt(252)).where(sd.notna() & x.notna(), df["volatility"])

    # Beta: the engine's annualised covariance and market variance, the day appended.
    cov = pd.to_numeric(df["_beta_cov"], errors="coerce") / 252
    var = pd.to_numeric(df["_beta_var"], errors="coerce") / 252
    cov_n = (cov * (n - 1) + x * xm) / n
    var_n = (var * (n - 1) + xm ** 2) / n
    df["beta"] = (cov_n / var_n).where(cov.notna() & var.notna() & x.notna(), df["beta"])

    # Max drawdown: the worse of the published one and the fall from the 52-week high to today.
    # (`_mdd_peak` is the peak *before the worst trough*, not the running high - the first draft
    # used it and undercounted today's fall for every stock that had since made a new high.)
    peak = pd.to_numeric(rw["_ctx_high_52w"], errors="coerce").fillna(
        pd.to_numeric(rw["fiftyTwoWeekHigh"], errors="coerce"))
    p1 = pd.to_numeric(rw["price_latest"], errors="coerce") * r
    dd_today = p1 / np.maximum(peak, p1) - 1
    df["max_drawdown_1y"] = np.minimum(pd.to_numeric(df["max_drawdown_1y"], errors="coerce"),
                                       dd_today.where(dd_today.notna(), 0))

    # Jensen's alpha: the engine's CAPM terms, the stock's and the market's 12m return one day on.
    ri = (1 + pd.to_numeric(df["_ja_ret12"], errors="coerce")) * r - 1
    rm = (1 + pd.to_numeric(df["_ja_mkt"], errors="coerce")) * m - 1
    rf = pd.to_numeric(df["_ja_rf"], errors="coerce")
    ja = ri - (rf + df["beta"] * (rm - rf))
    df["jensens_alpha"] = ja.where(ja.notna() & x.notna(), df["jensens_alpha"])

    info = {"market_day_return": xm, "n_priced": int(x.notna().sum()),
            "median_stock_return": float(x.median()),
            "worst": x.nsmallest(8).round(4).to_dict(), "best": x.nlargest(5).round(4).to_dict()}
    return df.reset_index(), info


def main() -> None:
    cfg = run_cfg()
    published = pd.read_parquet(RUN / "05_final_scored.parquet")
    base = pd.read_parquet(RUN / "02_outliers_flagged.parquet")
    base = base[base["Ticker"].isin(published["Ticker"])].reset_index(drop=True)
    raw = pd.read_parquet(RUN / "00_raw_fetch.parquet")

    ctrl = score(base, cfg)
    j = ctrl.merge(published[["Ticker", "Rank", "Composite"]], on="Ticker", suffixes=("", "_pub"))
    ctrl_exact = int((j["Rank"] == j["Rank_pub"]).sum())
    print(f"control: {ctrl_exact} of {len(j)} ranks reproduce the published run exactly; "
          f"max composite gap {float((j['Composite'] - j['Composite_pub']).abs().max()):.4f}")

    px = closes(base["Ticker"].tolist())
    shocked, info = shock(base, raw, px, cfg)
    print(f"market (S&P 500 TR) {info['market_day_return']:+.2%}, median stock "
          f"{info['median_stock_return']:+.2%}, {info['n_priced']} priced; worst {info['worst']}")
    sim = score(shocked, cfg)

    def as_payload(t: pd.DataFrame, when: str) -> dict:
        return {"table_data": t[["Ticker", "Rank"]].to_dict("records"),
                "kpis": {"run_timestamp": when}}

    base_p = h.published_payload("HEAD")
    pub_when = (base_p.get("kpis") or {}).get("run_timestamp")
    c_pub = h.rank_continuity(as_payload(sim, DAY1), base_p)
    c_ctrl = h.rank_continuity(as_payload(sim, DAY1), as_payload(ctrl, DAY0))
    lim = (h.RANK_MIN_SPEARMAN, h.RANK_MAX_NEW_IN_TOP, h.RANK_MAX_FALLEN)
    passes = (c_pub["spearman"] >= lim[0] and len(c_pub["new_in_top"]) <= lim[1]
              and len(c_pub["fallen"]) <= lim[2])
    print(f"vs the page the site serves ({pub_when}): Spearman {c_pub['spearman']:.3f}, "
          f"{len(c_pub['new_in_top'])} new in the top 25 {c_pub['new_in_top']}, "
          f"{len(c_pub['fallen'])} fallen below #40 {c_pub['fallen']} -> "
          f"{'PASS' if passes else 'FAIL'} (limits >= {lim[0]}, <= {lim[1]}, <= {lim[2]})")

    mv = sim[["Ticker", "Rank", "Composite", "Sector"] + [f"{c}_score" for c in CATS if f"{c}_score" in sim.columns]] \
        .merge(ctrl[["Ticker", "Rank", "Composite"] + [f"{c}_score" for c in CATS if f"{c}_score" in ctrl.columns]],
               on="Ticker", suffixes=("", "_0"))
    mv["dr"] = mv["Rank_0"] - mv["Rank"]
    cat_shift = {c: float((mv[f"{c}_score"] - mv[f"{c}_score_0"]).abs().mean())
                 for c in CATS if f"{c}_score" in mv.columns}
    by_sector = mv.groupby("Sector")["dr"].mean().sort_values().round(1).to_dict()
    top = mv.sort_values("Rank").head(25)[["Ticker", "Rank_0", "Rank", "Composite_0", "Composite"]]
    print("mean |category score change|:", {k: round(v, 2) for k, v in cat_shift.items()})
    print("mean rank change by sector (+ = up):", by_sector)
    print("biggest risers:", mv.nlargest(6, "dr")[["Ticker", "Rank_0", "Rank"]].values.tolist())
    print("biggest fallers:", mv.nsmallest(6, "dr")[["Ticker", "Rank_0", "Rank"]].values.tolist())
    print("simulated top 25:\n" + top.to_string(index=False))
    print(f"vs the control (same morning, no shock): Spearman {c_ctrl['spearman']:.3f}, "
          f"{len(c_ctrl['new_in_top'])} new, {len(c_ctrl['fallen'])} fallen")

    out = {"control_exact": ctrl_exact, "control_n": len(j), **info,
           "vs_published": {**c_pub, "run_timestamp": pub_when, "passes": passes},
           "vs_control": c_ctrl, "category_mean_abs_change": cat_shift, "sector_mean_rank_change": by_sector,
           "top25": top.to_dict("records")}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
