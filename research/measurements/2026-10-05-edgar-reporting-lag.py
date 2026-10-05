"""Measure the actual fundamental reporting lag for the S&P 500 universe.

Supports research/2026-10-05-fundamental-reporting-lag.md. Measures, from SEC
EDGAR's free submissions API, the distribution of (filingDate - reportDate) for
10-K and 10-Q filings — i.e. how many days after a fiscal period ends its full
financial statements become public. This is the lag a point-in-time backtest
must respect for the 49.0pp of composite weight that needs filings data
(see research/2026-10-01-lookahead-bias-size.md).

Also makes one companyconcept call to confirm that the XBRL fact-level API
carries a `filed` date per fact, which is what plan/backtest-v2.md step 3
depends on.

Deterministic sample: tickers sorted, every STRIDE-th name, so the sample is
reproducible without randomness. Respects EDGAR's access policy: declared
User-Agent, <10 requests/second.

Output: 2026-10-05-edgar-reporting-lag.json beside this script.

Run from the repo root: python research/measurements/2026-10-05-edgar-reporting-lag.py
"""

import json
import statistics
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")

UA = {"User-Agent": "Caleb Smit caleb.smit@icloud.com (personal research)"}
PAUSE = 0.15  # seconds between requests; EDGAR allows 10/s, stay well under
STRIDE = 8  # every 8th ticker of the sorted universe -> ~63 names
EARLIEST_PERIOD = "2019-06-30"  # covers the backtest window (2020-01 onward)

# SEC deadlines for large accelerated filers (public float >= $700M), which is
# effectively every S&P 500 constituent: Exchange Act Rules 13a-13/15d-13.
DEADLINE_DAYS = {"10-K": 60, "10-Q": 40}


def fetch_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main():
    universe = json.load(open(ROOT / "sp500_tickers.json"))
    tickers = sorted(row["Ticker"] for row in universe)
    sample = tickers[::STRIDE]

    # Ticker -> CIK map from EDGAR's own file.
    cik_map_raw = fetch_json("https://www.sec.gov/files/company_tickers.json")
    cik_by_ticker = {
        v["ticker"].upper(): int(v["cik_str"]) for v in cik_map_raw.values()
    }

    lags = {"10-K": [], "10-Q": []}
    per_company = {}
    unresolved = []

    for ticker in sample:
        # EDGAR uses '-' where the universe file uses '.' (BRK.B -> BRK-B).
        cik = cik_by_ticker.get(ticker.upper()) or cik_by_ticker.get(
            ticker.upper().replace(".", "-")
        )
        if cik is None:
            unresolved.append(ticker)
            continue
        time.sleep(PAUSE)
        try:
            sub = fetch_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        except Exception as exc:  # noqa: BLE001 - record and continue
            unresolved.append(f"{ticker} ({exc})")
            continue
        recent = sub["filings"]["recent"]
        company_lags = {"10-K": [], "10-Q": []}
        for form, filed, period in zip(
            recent["form"], recent["filingDate"], recent["reportDate"]
        ):
            if form not in lags or not period or period < EARLIEST_PERIOD:
                continue
            lag = (date.fromisoformat(filed) - date.fromisoformat(period)).days
            lags[form].append(lag)
            company_lags[form].append(lag)
        per_company[ticker] = {
            form: {"n": len(v), "median": statistics.median(v) if v else None}
            for form, v in company_lags.items()
        }

    def describe(values, deadline):
        values = sorted(values)
        n = len(values)
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "median": statistics.median(values),
            "mean": round(statistics.fmean(values), 1),
            "p10": values[int(0.10 * (n - 1))],
            "p90": values[int(0.90 * (n - 1))],
            "min": values[0],
            "max": values[-1],
            "deadline_days": deadline,
            "share_over_deadline": round(
                sum(v > deadline for v in values) / n, 4
            ),
        }

    # One companyconcept call: does each fact carry its own `filed` date?
    time.sleep(PAUSE)
    aapl_cik = cik_by_ticker["AAPL"]
    concept = fetch_json(
        f"https://data.sec.gov/api/xbrl/companyconcept/CIK{aapl_cik:010d}"
        "/us-gaap/Assets.json"
    )
    facts = concept["units"]["USD"]
    fact_fields = sorted(facts[0].keys())
    all_have_filed = all("filed" in f and "end" in f for f in facts)

    result = {
        "measured": date.today().isoformat(),
        "universe_file": "sp500_tickers.json",
        "sample_rule": f"sorted tickers, every {STRIDE}th",
        "sample_size": len(sample),
        "resolved": len(per_company),
        "unresolved": unresolved,
        "period_floor": EARLIEST_PERIOD,
        "lag_days": {
            form: describe(v, DEADLINE_DAYS[form]) for form, v in lags.items()
        },
        "companyconcept_check": {
            "endpoint": "api/xbrl/companyconcept/CIK0000320193/us-gaap/Assets.json",
            "n_facts_usd": len(facts),
            "fact_fields": fact_fields,
            "every_fact_has_filed_and_end": all_have_filed,
        },
        "per_company": per_company,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_company"}, indent=2))


if __name__ == "__main__":
    main()
