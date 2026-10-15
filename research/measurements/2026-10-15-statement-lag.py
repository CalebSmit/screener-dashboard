"""How many stocks are scored on statements older than the company's latest filed quarter?

The filing check found Camden (CPT) scored on the quarter to 2026-03-31 while its 10-Q for the
quarter to 2026-06-30 had been on EDGAR since July: the source had not caught up. This counts it
across the run: for each stock, the newest statement period the run used (payload ``asof``) against
the newest 10-Q / 10-K period end the company had filed (SEC cache), and how long ago that filing was.

Usage: python research/measurements/2026-10-15-statement-lag.py   (JSON beside it)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

t = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
payload = json.loads(t[t.index("{"):t.rindex("}") + 1])
run = pd.Timestamp(str(payload["kpis"]["run_timestamp"])[:10])
facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
facts = facts[facts["form"].isin(("10-K", "10-Q")) & (facts["filed"] <= run)]
# a company's newest reported period: the latest end of a duration fact (an income or cash-flow
# figure for a quarter or a year-to-date) filed by the run date
dur = facts[facts["start"].notna()]
newest = dur.groupby("ticker").apply(lambda g: g.loc[g["end"].idxmax(), ["end", "filed"]])
rows = []
for tk, d in payload["stock_detail"].items():
    used = (d.get("asof") or {}).get("is")
    if not used or tk not in newest.index:
        continue
    used = pd.Timestamp(used)
    end, filed = newest.loc[tk, "end"], newest.loc[tk, "filed"]
    behind = (end - used).days
    rows.append({"ticker": tk, "used": str(used.date()), "filed_end": str(end.date()),
                 "filed_on": str(filed.date()), "days_behind": int(behind),
                 "days_since_filed": int((run - filed).days), "data_source": d.get("data_source")})
df = pd.DataFrame(rows)
lag = df[df["days_behind"] > 45].sort_values("days_behind", ascending=False)
out = {"run_date": str(run.date()), "compared": int(len(df)),
       "a_quarter_or_more_behind": int(len(lag)),
       "share": round(len(lag) / len(df), 4) if len(df) else None,
       "stocks": lag.to_dict("records")}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "stocks"}))
print(lag.to_string(index=False))
