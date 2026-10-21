"""Does Tiingo's public ticker list carry the departed S&P 500 members free sources could not price?

2026-10-20 (owner-run, late). `research/2026-10-20-free-survivorship-sources.md` found that free,
account-free sources close 37% of backtest v2's 2012-2019 survivorship gap and only 1.2% of its
acquisitions - half the gap. Tiingo publishes the list of every ticker it carries, with the first and
last date of its price history, as a public file that needs no account:
https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip (saved, gitignored, under
data/survivorship/tiingo/). This checks each departed name against it.

A name counts as covered only if a Stock row under its symbol has history starting by its first month
in the index (40 days' grace) and ending within 40 days before to 270 days after its last month - so a
symbol reused by a later company does not count. A list entry is not data: the prices themselves need
a free Tiingo account, and this measures only whether asking would be worth it.

Usage: python research/measurements/2026-10-20-tiingo-list-coverage.py   (writes the JSON beside it)
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research" / "measurements" / "2026-10-20-survivorship-free-sources.json"
LIST = ROOT / "data" / "survivorship" / "tiingo" / "supported_tickers.zip"
OUT = Path(__file__).with_suffix(".json")


def covered(rows: pd.DataFrame, v: dict) -> bool:
    first, last = pd.Timestamp(v["first_in_index"]), pd.Timestamp(v["last_in_index"])
    for _, r in rows.iterrows():
        try:
            s, e = pd.Timestamp(r.startDate), pd.Timestamp(r.endDate)
        except (TypeError, ValueError):
            continue
        if s <= first + pd.Timedelta(days=40) and last - pd.Timedelta(days=40) <= e <= last + pd.Timedelta(days=270):
            return True
    return False


def main() -> None:
    cls = json.loads(SRC.read_text(encoding="utf-8"))["classification"]
    with zipfile.ZipFile(LIST) as z:
        t = pd.read_csv(z.open(z.namelist()[0]))
    t["ticker"] = t["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    t = t[t["assetType"] == "Stock"]
    out = {"list_rows": int(len(t)), "by_cause": {}}
    for cause in sorted({v["cause"] for v in cls.values()}):
        names = [k for k, v in cls.items() if v["cause"] == cause]
        hit = [k for k in names if covered(t[t.ticker == k.upper().replace(".", "-")], cls[k])]
        months = sum(cls[k]["n_departed_months_2012_2019"] for k in names)
        months_hit = sum(cls[k]["n_departed_months_2012_2019"] for k in hit)
        out["by_cause"][cause] = {"names": len(names), "names_listed": len(hit), "member_months": months,
                                  "member_months_listed": months_hit, "listed": sorted(hit)}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for c, r in out["by_cause"].items():
        print(f"{c:58} {r['names_listed']:3}/{r['names']:3} names  {r['member_months_listed']:5}/{r['member_months']:5} member-months")


if __name__ == "__main__":
    main()
