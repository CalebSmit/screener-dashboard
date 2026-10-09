"""Insider trades: the SEC's own Form 4 filings against Yahoo's compiled feed, per stock.

Why: on 2026-10-08 the context layer switched its insider source from Yahoo's
``insider_transactions`` to EDGAR Form 4s (``insider_activity.refresh``). This measures what
the switch changes on the same day: coverage, open-market buy and sale counts, and how much of
the sale value the filings mark as made under a Rule 10b5-1 plan - a fact Yahoo cannot carry -
and how many purchase lines come from 10%+ holders who are neither officers nor directors
(counted apart since the same night). Both sides use ``summarise_rows``, so "buys" here means
officer-and-director buys.

Reads the newest run's raw fetch (Yahoo rows in ``_ctx_insider``) and the SEC cache in
``data/insider/filings.json``. Makes no requests. Prints a summary and writes nothing.

    python research/measurements/2026-10-08-insider-sec-vs-yahoo.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import insider_activity as ia  # noqa: E402


def main(today: date | None = None) -> dict:
    runs = sorted((d for d in (ROOT / "runs").iterdir() if (d / "00_raw_fetch.parquet").exists()),
                  key=lambda d: (d / "00_raw_fetch.parquet").stat().st_mtime)
    raw = pd.read_parquet(runs[-1] / "00_raw_fetch.parquet")
    today = today or date.today()
    cache = json.loads(ia.CACHE_PATH.read_text(encoding="utf-8"))
    n = sec_n = 0
    from datetime import timedelta
    s_since = (today - timedelta(days=90)).isoformat()
    y_buy = s_buy = y_sell = s_sell = both_buy = only_y = only_s = 0
    sell_v = plan_v = 0.0
    h_buy = h_val = 0
    lines = {"officer_or_director": 0, "holder_only": 0}
    for _, r in raw.iterrows():
        tk = r["Ticker"]
        y = ia.summarise_rows(json.loads(r["_ctx_insider"]) if isinstance(r.get("_ctx_insider"), str) else [], today)
        rows = ia.sec_rows_for(cache, tk, today)
        n += 1
        if rows is None:
            continue
        sec_n += 1
        s = ia.summarise_rows(rows, today)
        y_buy += bool(y["buy_n"]); s_buy += bool(s["buy_n"])
        y_sell += bool(y["sell_n"]); s_sell += bool(s["sell_n"])
        both_buy += bool(y["buy_n"]) and bool(s["buy_n"])
        only_y += bool(y["buy_n"]) and not s["buy_n"]
        only_s += bool(s["buy_n"]) and not y["buy_n"]
        sell_v += s["sell_value"]; plan_v += s["sell_planned_value"] or 0.0
        h_buy += bool(s["holder_buy_n"]); h_val += s["holder_buy_value"]
        for x in rows:
            if x["code"] == "P" and x["date"] >= s_since:
                lines["holder_only" if ia.holder_only(x.get("role")) else "officer_or_director"] += 1
    out = {"run": runs[-1].name, "stocks": n, "with_fresh_sec_record": sec_n,
           "stocks_with_buys": {"yahoo": y_buy, "sec": s_buy, "both": both_buy, "only_yahoo": only_y, "only_sec": only_s},
           "stocks_with_sales": {"yahoo": y_sell, "sec": s_sell},
           "sec_purchase_lines_90d": lines, "stocks_with_10pct_holder_buys": h_buy,
           "holder_buy_value_usd": round(h_val),
           "sec_sale_value_usd": round(sell_v), "share_of_sale_value_on_10b5_1_plans": round(plan_v / sell_v, 3) if sell_v else None,
           "cache": cache.get("_meta")}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
