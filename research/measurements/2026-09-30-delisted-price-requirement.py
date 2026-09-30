"""Size the delisted-price requirement that gates steps 2-4 of backtest v2.

`plan/backtest-v2.md` says the next decision on priority 3 is a procurement
one: **cost a price source for delisted tickers**, because that determines
whether v2 can *remove* survivorship bias or only *report* it. Eight of the
ten sessions before 2026-09-30 deferred it. You cannot cost a data purchase
without knowing exactly what you need to buy, and the 2026-09-24 measurement
deliberately did not establish that:

  * it decomposed renames from exits **only for the oldest month** (2020-01-31),
    because doing the whole window needs CIKs from all 80 revisions;
  * it probed price availability on a **30-name sample** of that month's exits;
  * and its availability test was ``len(hist) > 200`` — *any* 200 rows.

The third is the one that matters most, and it is why this script exists. A
backtest does not need "some history" for a delisted name. It needs prices
**for the months that name was actually in the index**. A company acquired in
2023 whose free series stops in 2021 passes a 200-row test and still leaves a
two-year hole in exactly the window the backtest would read. This script asks
the coverage question the way the harness would ask it.

What it reports, all as counts over the full population:

  1. The 142 names ever in the index 2020-01..2026-08 and absent today,
     decomposed into ticker renames and genuine exits **across the whole
     window** by SEC CIK.
  2. For every genuine exit: does a free price series exist, and does it cover
     the name's own index-membership months?
  3. The residual — name-months a free-data v2 would still be missing. That
     residual, not the name count, is what a vendor has to sell.

**No backtest is run and no return is computed here**, so `CLAUDE.md` rule 5
does not reach it: there is no backtest number to bench. This sizes a defect in
the tool that would one day validate a methodology change; it cannot justify
one.

Run:
    python research/measurements/2026-09-30-delisted-price-requirement.py
    python research/measurements/2026-09-30-delisted-price-requirement.py --refresh-ciks

The CIK map is cached at ``data/universe_history/ticker_ciks.json`` so a re-run
costs no network and the published numbers stay pinned to specific Wikipedia
revisions. ``--refresh-ciks`` rebuilds it (80 HTTP requests).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import universe_history as uh  # noqa: E402

BACKTEST_START = "2020-01-01"  # mirrors backtest.BACKTEST_START
CIK_CACHE = ROOT / "data" / "universe_history" / "ticker_ciks.json"

# A monthly-rebalanced backtest needs, for each name, a price at each month-end
# it was a constituent. "Covered" below means the free series brackets those
# month-ends; see `covers_membership`.
MIN_ROWS = 200  # the 2026-09-24 test, kept so the two numbers are comparable


def current_universe() -> list:
    """Today's constituent list — the one `backtest.py` projects backwards."""
    records = json.loads((ROOT / "sp500_tickers.json").read_text())
    return [uh.normalize_ticker(r["Ticker"]) for r in records]


# ---------------------------------------------------------------------------
# 1. Rename decomposition across the whole window
# ---------------------------------------------------------------------------
def build_cik_map(cache, dates, session=None) -> dict:
    """Union of ticker -> CIK over every revision in the window.

    The 2026-09-24 script could only decompose the oldest month, because a
    company that both joined and left mid-window appears in **neither**
    endpoint revision and so has no CIK to match on. Reading all 80 revisions
    fixes that at the cost of 80 requests.

    Later revisions win on conflict, which matters for exactly the case being
    measured: when a ticker is reassigned or a registrant renamed, the newest
    record is the one that describes today's index, and today's index is what
    membership is tested against.
    """
    session = session or uh._session()
    out = {}
    for i, d in enumerate(dates, 1):
        revid = cache[d].revid
        try:
            m = uh.parse_ticker_ciks(uh.fetch_revision_html(revid, session=session))
        except uh.UniverseHistoryError:
            m = {}
        out.update(m)
        print(f"  [{i}/{len(dates)}] {d} rev {revid}: {len(m)} CIKs "
              f"(union {len(out)})", flush=True)
    return out


def load_cik_map(cache, dates, refresh: bool) -> tuple:
    """Return (map, source) where source is 'cache' or 'network'."""
    if not refresh and CIK_CACHE.exists():
        payload = json.loads(CIK_CACHE.read_text())
        return {k: int(v) for k, v in payload["ciks"].items()}, "cache"
    print(f"Building CIK map from {len(dates)} revisions...")
    m = build_cik_map(cache, dates)
    CIK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CIK_CACHE.write_text(json.dumps({
        "built_from": {"first": dates[0], "last": dates[-1],
                       "n_revisions": len(dates)},
        "revids": {d: cache[d].revid for d in dates},
        "ciks": {k: int(v) for k, v in sorted(m.items())},
    }, indent=1))
    return m, "network"


def decompose(absent, cik_map, today) -> dict:
    """Split absent names into renames, genuine exits and unresolved.

    A name is a **rename** when its CIK is also carried by a ticker that is in
    the index today: same SEC registrant, new symbol. Anything else with a
    resolvable CIK genuinely left. Unresolved names are reported rather than
    assumed either way.
    """
    live_ciks = {cik_map[t] for t in today if t in cik_map}
    renamed, exited, unknown = [], [], []
    for t in absent:
        cik = cik_map.get(t)
        if cik is None:
            unknown.append(t)
        elif cik in live_ciks:
            new = sorted(s for s in today if cik_map.get(s) == cik)
            renamed.append((t, new[0] if new else "?"))
        else:
            exited.append(t)
    return {"renamed": renamed, "exited": exited, "unknown": unknown}


# ---------------------------------------------------------------------------
# 2. Does a free series cover the months the name was in the index?
# ---------------------------------------------------------------------------
def membership_months(cache, dates, ticker) -> list:
    """The month-end dates on which `ticker` was a constituent."""
    return [d for d in dates if ticker in cache[d].tickers]


def covers_membership(index, months) -> dict:
    """Which of a name's membership month-ends does the series price?

    A monthly-rebalanced backtest reads one price per month-end. Requiring an
    exact date would fail on every holiday, so a month-end counts as covered
    when the series has any row in that calendar month. That is the same
    tolerance a real harness would use and it is deliberately generous — it
    makes the shortfall reported below a *floor*.
    """
    have = {(ts.year, ts.month) for ts in index}
    covered = [m for m in months if (int(m[:4]), int(m[5:7])) in have]
    return {"n_months": len(months), "n_covered": len(covered),
            "missing": [m for m in months if m not in set(covered)]}


def probe(cache, dates, tickers) -> list:
    """Census, not a sample: every genuine exit gets fetched once."""
    import yfinance as yf

    rows = []
    for i, t in enumerate(sorted(tickers), 1):
        months = membership_months(cache, dates, t)
        rec = {"ticker": t, "n_months": len(months), "rows": 0,
               "first": None, "last": None, "n_covered": 0,
               "missing": months, "error": None}
        try:
            hist = yf.Ticker(t).history(period="max", auto_adjust=True)
        except Exception as exc:  # network/parse failures are data absence here
            rec["error"] = type(exc).__name__
            hist = None
        if hist is not None and len(hist):
            rec["rows"] = len(hist)
            rec["first"] = str(hist.index[0].date())
            rec["last"] = str(hist.index[-1].date())
            rec.update(covers_membership(hist.index, months))
        rows.append(rec)
        print(f"  [{i}/{len(tickers)}] {t:<6} rows={rec['rows']:<6} "
              f"covered {rec['n_covered']}/{rec['n_months']} months",
              flush=True)
    return rows


# ---------------------------------------------------------------------------
def main() -> int:
    refresh = "--refresh-ciks" in sys.argv
    skip_prices = "--no-prices" in sys.argv

    cache = uh.load_cache()
    if not cache:
        print("No cached universe history. Run: python universe_history.py --refresh")
        return 1

    dates = sorted(d for d in cache if d >= BACKTEST_START)
    today = current_universe()
    ever = set()
    for d in dates:
        ever |= set(cache[d].tickers)
    absent = sorted(ever - set(today))

    print("=" * 78)
    print("DELISTED-PRICE REQUIREMENT — what backtest v2 would have to buy")
    print("=" * 78)
    print(f"Point-in-time months : {len(dates)}  ({dates[0]} .. {dates[-1]})")
    print(f"Distinct names ever  : {len(ever)}")
    print(f"In today's list      : {len(today)}")
    print(f"Absent from today    : {len(absent)}")
    print()

    cik_map, source = load_cik_map(cache, dates, refresh)
    dec = decompose(absent, cik_map, today)
    n_ren, n_exit, n_unk = (len(dec["renamed"]), len(dec["exited"]),
                            len(dec["unknown"]))
    print(f"CIK map: {len(cik_map)} tickers (from {source})")
    print("-" * 78)
    print(f"Of the {len(absent)} names absent from today's list:")
    print(f"  {n_ren:>4} are ticker renames (same SEC registrant, new symbol)")
    print(f"  {n_exit:>4} genuinely left the index")
    print(f"  {n_unk:>4} unresolved (no CIK in any revision in the window)")
    if dec["renamed"]:
        print()
        print("  Renames:")
        pairs = [f"{a}->{b}" for a, b in dec["renamed"]]
        for i in range(0, len(pairs), 6):
            print("    " + "  ".join(f"{p:<13}" for p in pairs[i:i + 6]))
    if dec["unknown"]:
        print()
        print("  Unresolved: " + ", ".join(dec["unknown"]))

    exited = dec["exited"] + dec["unknown"]  # unresolved treated as exits here
    exit_months = sum(len(membership_months(cache, dates, t)) for t in exited)
    live_months = sum(len(membership_months(cache, dates, t)) for t in today)
    print()
    print("-" * 78)
    print("The name-month, not the name, is the unit a vendor sells.")
    print(f"  Membership name-months, exited names : {exit_months}")
    print(f"  Membership name-months, current names: {live_months}")
    print(f"  Exited share of the true panel       : "
          f"{100 * exit_months / (exit_months + live_months):.1f}%")
    print("  (Unresolved names are counted as exits above — the conservative")
    print("   direction for a shortfall figure.)")

    if skip_prices:
        return 0

    print()
    print("-" * 78)
    print(f"Price census over all {len(exited)} exited names "
          f"(no sampling)...")
    rows = probe(cache, dates, exited)

    none_at_all = [r for r in rows if r["rows"] == 0]
    pass_old = [r for r in rows if r["rows"] > MIN_ROWS]
    full = [r for r in rows if r["n_months"] and r["n_covered"] == r["n_months"]]
    partial = [r for r in rows if r["rows"] and r["n_covered"] < r["n_months"]]
    missing_months = sum(len(r["missing"]) for r in rows)

    print()
    print("-" * 78)
    print("RESULT")
    print(f"  Exited names                              : {len(rows)}")
    print(f"  ...with no downloadable series at all      : {len(none_at_all)}")
    print(f"  ...passing the 2026-09-24 test (>{MIN_ROWS} rows)  : {len(pass_old)}"
          f"  ({100 * len(pass_old) / max(len(rows), 1):.0f}%)")
    print(f"  ...covering EVERY membership month         : {len(full)}"
          f"  ({100 * len(full) / max(len(rows), 1):.0f}%)")
    print(f"  ...partial coverage                        : {len(partial)}")
    print()
    print(f"  Name-months required                       : {exit_months}")
    print(f"  Name-months a free source cannot supply    : {missing_months}"
          f"  ({100 * missing_months / max(exit_months, 1):.1f}%)")
    print(f"  ==> Residual survivorship after a free-data v2, as a share of the")
    print(f"      true panel: {100 * missing_months / (exit_months + live_months):.2f}%"
          f" of all name-months.")

    if partial:
        print()
        print("  Partial-coverage names — these are the ones a row-count test")
        print("  calls 'available' and a backtest still cannot use:")
        for r in sorted(partial, key=lambda r: -len(r["missing"]))[:20]:
            print(f"    {r['ticker']:<6} series {r['first']}..{r['last']}  "
                  f"covers {r['n_covered']}/{r['n_months']}, "
                  f"missing {len(r['missing'])} months "
                  f"(first gap {r['missing'][0] if r['missing'] else '-'})")

    if none_at_all:
        print()
        print("  No series at all: "
              + ", ".join(r["ticker"] for r in none_at_all))

    out = ROOT / "research" / "measurements" / "2026-09-30-delisted-price-census.json"
    out.write_text(json.dumps({
        "as_of": dates[-1], "months": len(dates),
        "n_ever": len(ever), "n_today": len(today), "n_absent": len(absent),
        "renamed": dec["renamed"], "exited": dec["exited"],
        "unresolved": dec["unknown"],
        "exit_name_months": exit_months, "live_name_months": live_months,
        "missing_name_months": missing_months,
        "rows": rows,
    }, indent=1))
    print()
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
