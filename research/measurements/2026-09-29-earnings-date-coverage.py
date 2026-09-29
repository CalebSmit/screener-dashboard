"""How well-behaved is the provider's next-earnings date across the S&P 500?

Context: `plan/dashboard-north-star.md` gap 4 proposes surfacing earnings-date
proximity, on the strength of Akepanidtaworn, Di Mascio, Imas & Schmidt (2023,
*Journal of Finance* 78(6)) - announcement-day sells beat non-announcement-day
sells by more than +150 bp/year and are the only sells in that sample to beat a
random-disposal counterfactual.

The design decisions that shipped 2026-09-29 all rest on numbers this script
reproduces, and every one of them was a question the plan could not answer:

1. **Coverage** - is the field present often enough to be worth a surface?
2. **Estimate rate** - how often is the "date" the provider's guess? This is
   what decides whether `isEarningsDateEstimate` is a nice-to-have or the
   load-bearing part of the feature.
3. **Stale dates** - how often is the "next" date already in the past? This is
   what decides whether the drop-never-relabel guard is real or theoretical.
4. **Windows** - is `earningsTimestampEnd` ever different from the start? This
   decides whether the payload carries one date per stock or two.
5. **Timezone safety** - the timestamps are UNIX epochs and the dashboard reads
   the UTC date. If any stamp sat near a UTC midnight, a UTC read and a US
   market-date read would disagree and the site would publish an off-by-one day.
6. **Ambiguity of `earningsTimestamp`** - the obvious-looking third field.

This is a **live** measurement: it re-fetches from the provider, so the numbers
move with the calendar. The figures quoted in `METHODOLOGY_CHANGELOG.md`
2026-09-29 and on the dashboard footnote are the 2026-09-29 run, printed at the
bottom for comparison.

Run:  python research/measurements/2026-09-29-earnings-date-coverage.py
      python research/measurements/2026-09-29-earnings-date-coverage.py --quick
        (a 80-name random sample; ~1 minute instead of ~8)
"""

import collections
import datetime
import os
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

# What the 2026-09-29 full run reported, so a later reader can see drift.
BASELINE_2026_09_29 = {
    "universe": 503,
    "with_start_date": 503,
    "future": 492,
    "stale": 11,
    "windows": 0,
    "estimated_of_future": 209,
    "estimate_rate_pct": 42.5,
    "missing_estimate_flag": 0,
    "utc_vs_eastern_mismatch": 0,
    "days_away_min_median_max": (0, 30, 86),
}


def universe():
    """Tickers from the newest run's raw fetch - no extra network call."""
    import glob

    import pandas as pd
    files = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "00_raw_fetch.parquet")),
                   key=os.path.getmtime, reverse=True)
    if not files:
        raise SystemExit("no run directory with 00_raw_fetch.parquet found")
    return pd.read_parquet(files[0])["Ticker"].dropna().unique().tolist()


def fetch(tickers):
    import yfinance as yf

    def one(t):
        try:
            i = yf.Ticker(t).info or {}
            return (t, i.get("earningsTimestampStart"), i.get("earningsTimestampEnd"),
                    i.get("isEarningsDateEstimate"), i.get("earningsTimestamp"))
        except Exception:
            return (t, None, None, None, None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, tickers))


def as_utc_date(ts):
    return datetime.datetime.fromtimestamp(float(ts), datetime.UTC).date()


def main():
    quick = "--quick" in sys.argv
    tickers = universe()
    if quick:
        random.seed(20260929)
        tickers = random.sample(tickers, min(80, len(tickers)))
    today = datetime.date.today()

    print(f"Fetching {len(tickers)} tickers (as of {today})...")
    rows = fetch(tickers)

    have = [r for r in rows if r[1]]
    print(f"\n1. Coverage")
    print(f"   universe            : {len(rows)}")
    print(f"   with a start date   : {len(have)} "
          f"({100 * len(have) / max(1, len(rows)):.1f}%)")

    print(f"\n5. Timezone safety (checked first - it gates every date below)")
    hours = collections.Counter(
        datetime.datetime.fromtimestamp(float(r[1]), datetime.UTC).strftime("%H:%M")
        for r in have)
    print(f"   UTC times seen      : {dict(hours)}")
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        bad = [r[0] for r in have
               if as_utc_date(r[1])
               != datetime.datetime.fromtimestamp(float(r[1]), et).date()]
        print(f"   UTC vs US/Eastern   : {len(bad)} disagree {bad[:10]}")
        if bad:
            print("   ^^ the UTC read in generate_dashboard._epoch_to_date is "
                  "no longer safe; this is a defect, not drift.")
    except Exception as exc:
        print(f"   timezone check skipped: {exc}")

    fut = [r for r in have if as_utc_date(r[1]) >= today]
    stale = [r for r in have if as_utc_date(r[1]) < today]
    print(f"\n3. Stale dates (the drop-never-relabel guard)")
    print(f"   in the future       : {len(fut)}")
    print(f"   already in the past : {len(stale)}")
    print(f"     {[(r[0], as_utc_date(r[1]).isoformat()) for r in stale[:12]]}")

    win = [r for r in have if r[2] and float(r[2]) != float(r[1])]
    print(f"\n4. Windows (start != end)")
    print(f"   count               : {len(win)} {[r[0] for r in win[:10]]}")

    est = [r for r in fut if r[3] is True]
    noflag = [r for r in fut if r[3] is None]
    print(f"\n2. Estimate rate - the load-bearing number")
    print(f"   flagged estimate    : {len(est)} of {len(fut)} "
          f"({100 * len(est) / max(1, len(fut)):.1f}%)")
    print(f"   flag absent entirely: {len(noflag)}")

    print(f"\n6. Why `earningsTimestamp` is not captured")
    ambiguous = [(r[0], as_utc_date(r[4]).isoformat(), as_utc_date(r[1]).isoformat())
                 for r in have if r[4] and as_utc_date(r[4]) < today]
    agrees = [r[0] for r in have if r[4] and r[1] and float(r[4]) == float(r[1])]
    print(f"   equals the NEXT date: {len(agrees)} {agrees[:6]}")
    print(f"   is a PAST date      : {len(ambiguous)} {ambiguous[:6]}")
    print("   -> no single label is true of every row, so the field is unread.")

    days = sorted((as_utc_date(r[1]) - today).days for r in fut)
    if days:
        print(f"\n   days away min/median/max: "
              f"{days[0]} / {days[len(days) // 2]} / {days[-1]}")
        for h in (7, 14, 30, 45):
            print(f"     within {h:>2}d: {sum(1 for x in days if x <= h)}")

    print(f"\n--- 2026-09-29 full-universe baseline, for drift comparison ---")
    for k, v in BASELINE_2026_09_29.items():
        print(f"   {k:<26} {v}")


if __name__ == "__main__":
    main()
