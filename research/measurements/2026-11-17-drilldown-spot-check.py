"""Tuesday's spot check: do three drilldowns still reproduce the engine from the page's own payload?

Independent of `calc_trace` on purpose - plain arithmetic on what the published payload carries:
for each stock, the composite rebuilt from its per-category contributions, and two or three
metric values rebuilt from the published inputs. Reports, does not assert.

    python research/measurements/2026-11-17-drilldown-spot-check.py [TICKER ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as H  # noqa: E402


def main(tickers: list[str]) -> None:
    payload = H.fold_shards(H._parse_payload_file(ROOT / "dashboard_data.js"), H._dir_reader(ROOT))
    rows = {r["Ticker"]: r for r in payload["table_data"]}
    print("run", payload["kpis"]["run_timestamp"])
    for t in tickers or [payload["table_data"][0]["Ticker"], "AMZN", "JPM"]:
        s, r = payload["stock_detail"][t], rows[t]
        inp, raw = s.get("inp") or {}, s.get("raw") or {}
        contrib = s.get("contrib") or {}
        total = sum(v for v in contrib.values() if isinstance(v, (int, float)))
        print(f"\n{t}  rank {r['Rank']}  composite {r['Composite']}  "
              f"sum of {len(contrib)} category contributions {total:.2f}  "
              f"(difference {r['Composite'] - total:+.2f}; coverage {s.get('metric_count')}/{s.get('metric_total')})")
        checks = []
        if inp.get("netIncome") and inp.get("marketCap"):
            checks.append(("earnings_yield", inp["netIncome"] / inp["marketCap"]))
        if inp.get("totalAssets") and inp.get("totalAssets_prior"):
            checks.append(("asset_growth", (inp["totalAssets"] - inp["totalAssets_prior"]) / inp["totalAssets_prior"]))
        if inp.get("netIncome") and inp.get("totalEquity"):
            checks.append(("roe", inp["netIncome"] / inp["totalEquity"]))
        for name, mine in checks:
            pub = raw.get(name)
            if pub is None:
                print(f"  {name:<16} not published for this stock (mine {mine:.4f})")
                continue
            print(f"  {name:<16} published {pub:.4f}  from inputs {mine:.4f}  "
                  f"{'MATCH' if abs(pub - mine) <= 5e-4 * max(1, abs(pub)) else 'DIFFERS'}")
        print(f"  as of: {s.get('asof')}  source: {s.get('data_source')}  "
              f"newer_filing: {s.get('newer_filing')}  filing age: {(s.get('flags') or {}).get('stmt_age_days')}")


if __name__ == "__main__":
    main(sys.argv[1:])
