"""Measure the survivorship bias in `backtest.py`'s universe.

Step 1 of `plan/backtest-v2.md` is "quantify the damage first" — before
building anything, find out how much survivorship and look-ahead are actually
worth in *this* setup, because that tells you how hard to work on the rest.

This script does the survivorship half. It answers one question with no
modelling in it at all:

    **How many companies does the v1 backtest silently delete from each
    historical month?**

`backtest.py` builds its universe once, from today's constituent list, and
applies it to every rebalance date from 2020-01 onward. Any company that left
the index in between is absent — not weighted zero, absent, as though it had
never been in the S&P 500. This counts them, month by month, using the
point-in-time membership reconstructed by `universe_history.py`.

**This is a count, not a return.** It measures a property of the harness, not
the performance of a strategy, so `CLAUDE.md` rule 5 (backtest output is
benched until 2027-02-11) does not reach it — there is no backtest number here
to bench. Nothing in this script can justify a methodology change; it sizes a
known defect in the tool that would eventually validate one.

Run:  python research/measurements/2026-09-24-survivorship-gap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import universe_history as uh  # noqa: E402

BACKTEST_START = "2020-01-01"  # mirrors backtest.BACKTEST_START


def current_universe() -> list:
    """Today's constituent list — the one `backtest.py` projects backwards.

    Read from the repo's own `sp500_tickers.json` rather than the network, so
    the measurement is reproducible and does not depend on the day it is run.
    """
    records = json.loads((ROOT / "sp500_tickers.json").read_text())
    return [uh.normalize_ticker(r["Ticker"]) for r in records]


def decompose_renames(cache, dropped, first_date: str, last_date: str) -> dict:
    """Split the deleted names into genuine exits and mere ticker renames.

    The gross gap overstates survivorship bias, because a ticker that is
    absent from today's list has not necessarily left the index — it may be
    the same registrant under a new symbol. Matching on SEC CIK separates
    them; see :func:`universe_history.parse_ticker_ciks` for why neither
    symbol nor company name can.

    Costs two HTTP requests: the endpoint revisions, whose ids are already in
    the cache, so the comparison is reproducible against those exact revisions.
    """
    session = uh._session()
    then_ciks = uh.parse_ticker_ciks(
        uh.fetch_revision_html(cache[first_date].revid, session=session))
    now_ciks = uh.parse_ticker_ciks(
        uh.fetch_revision_html(cache[last_date].revid, session=session))
    if not then_ciks or not now_ciks:
        return {"usable": False}

    live_ciks = set(now_ciks.values())
    renamed, exited, unknown = [], [], []
    for t in dropped:
        cik = then_ciks.get(t)
        if cik is None:
            unknown.append(t)
        elif cik in live_ciks:
            new_sym = next(s for s, c in now_ciks.items() if c == cik)
            renamed.append((t, new_sym))
        else:
            exited.append(t)
    return {"usable": True, "renamed": renamed, "exited": exited,
            "unknown": unknown, "n_then_ciks": len(then_ciks),
            "n_now_ciks": len(now_ciks)}


def probe_price_availability(tickers, limit: int) -> dict:
    """Do the deleted names still have downloadable price history?

    This is the feasibility question for step 2 of the plan. Reconstructing
    the universe is worth nothing if the companies it restores have no prices
    to compute returns from — a point-in-time universe with no data for the
    delisted names would restore their *names* and still drop their *returns*,
    which is the same bias wearing a better label.

    Acquired-for-cash companies usually retain history up to the delisting
    date; some bankruptcies and ticker changes do not. This samples rather
    than fetching all of them, and reports the sample size so the number is
    not mistaken for a census.
    """
    import yfinance as yf

    sample = sorted(tickers)[:limit]
    have, missing = [], []
    for t in sample:
        try:
            hist = yf.Ticker(t).history(period="max", auto_adjust=True)
            (have if len(hist) > 200 else missing).append(t)
        except Exception:
            missing.append(t)
    return {"sampled": len(sample), "have": have, "missing": missing}


def main() -> int:
    probe = 0
    if "--probe-prices" in sys.argv:
        probe = int(sys.argv[sys.argv.index("--probe-prices") + 1])

    cache = uh.load_cache()
    if not cache:
        print("No cached universe history. Run:\n"
              "  python universe_history.py --refresh")
        return 1

    today = current_universe()
    dates = sorted(d for d in cache if d >= BACKTEST_START)

    print("=" * 78)
    print("SURVIVORSHIP GAP — what today's constituent list deletes from history")
    print("=" * 78)
    print(f"Today's list: {len(today)} tickers (sp500_tickers.json)")
    print(f"Point-in-time months: {len(dates)}  ({dates[0]} .. {dates[-1]})")
    print()

    gaps = [uh.survivorship_gap(cache[d], today) for d in dates]

    print(f"{'month':<12}{'in index':>10}{'survive':>10}"
          f"{'deleted':>10}{'% deleted':>11}")
    print("-" * 78)
    for g in gaps:
        print(f"{g['as_of']:<12}{g['n_then']:>10}{g['n_survivors']:>10}"
              f"{g['n_dropped']:>10}{g['pct_dropped']:>10.1f}%")

    worst = max(gaps, key=lambda g: g["n_dropped"])
    first = gaps[0]

    # Every distinct company that was in the index at some point in the window
    # and is not in it today. This is the set v1 has never once tested.
    ever = set()
    for d in dates:
        ever |= set(cache[d].tickers)
    never_tested = sorted(ever - set(today))

    print()
    print("-" * 78)
    print(f"Oldest month ({first['as_of']}): {first['n_dropped']} of "
          f"{first['n_then']} constituents are absent from today's list "
          f"({first['pct_dropped']:.1f}%).")
    print(f"Worst month  ({worst['as_of']}): {worst['n_dropped']} deleted "
          f"({worst['pct_dropped']:.1f}%).")
    print(f"Distinct companies in the index during the window but not today: "
          f"{len(never_tested)}")
    print(f"Universe ever seen across the window: {len(ever)} names, "
          f"against {len(today)} in any single v1 run.")
    print()
    print("Names the v1 backtest has never tested (first 40):")
    for i in range(0, min(len(never_tested), 40), 8):
        print("  " + "  ".join(never_tested[i:i + 8]))

    # Turnover: how much of the index changes between consecutive months.
    print()
    print("-" * 78)
    changes = []
    for prev, cur in zip(dates, dates[1:]):
        a, b = set(cache[prev].tickers), set(cache[cur].tickers)
        changes.append((cur, len(b - a), len(a - b)))
    total_add = sum(c[1] for c in changes)
    total_rem = sum(c[2] for c in changes)
    print(f"Index changes over the window: {total_add} additions, "
          f"{total_rem} removals across {len(changes)} month transitions "
          f"({total_rem / max(len(changes), 1) * 12:.1f} removals/year).")

    # The gross figure counts renames as deletions. Separate them by CIK.
    print()
    print("-" * 78)
    print("Decomposing the oldest month's gap: exits vs ticker renames...")
    # Scoped to the oldest month on purpose: every one of its names appears in
    # that month's revision, so every CIK resolves and the split is exact
    # (0 unknown). Decomposing the whole window's 142 would need CIKs from all
    # 80 revisions — the names that both joined and left mid-window are absent
    # from either endpoint — and that is a re-fetch, not a free extension.
    dec = decompose_renames(cache, first["dropped"], dates[0], dates[-1])
    if not dec.get("usable"):
        print("  CIK column unavailable in an endpoint revision; "
              "the gross figure above is not decomposed.")
    else:
        n_exit, n_ren = len(dec["exited"]), len(dec["renamed"])
        n_unk = len(dec["unknown"])
        print(f"  Of {first['n_dropped']} names absent from today's list "
              f"on {dates[0]}:")
        print(f"    {n_exit} genuinely left the index  "
              f"({100 * n_exit / first['n_then']:.1f}% of that month's universe)")
        print(f"    {n_ren} are the same SEC registrant under a new ticker")
        print(f"    {n_unk} could not be matched (no CIK in the old revision)")
        if dec["renamed"]:
            shown = ", ".join(f"{a}->{b}" for a, b in dec["renamed"][:12])
            print(f"    renames, e.g.: {shown}")
        print()
        print(f"  ==> Survivorship bias, net of renames: {n_exit} of "
              f"{first['n_then']} names "
              f"({100 * n_exit / first['n_then']:.1f}%) are absent from every "
              f"v1 backtest of {dates[0]}.")

    print()
    print("-" * 78)
    print("Staleness of the Wikipedia revisions used (days behind the month-end):")
    st = sorted(cache[d].staleness_days for d in dates)
    print(f"  min {st[0]}, median {st[len(st) // 2]}, max {st[-1]}")

    if probe:
        print()
        print("-" * 78)
        # Probe the genuine exits, not the renames: a renamed company's prices
        # are available under its new ticker, so including them would flatter
        # the feasibility number with names that were never at risk.
        targets = dec["exited"] if dec.get("usable") else never_tested
        print(f"Price availability for {len(targets)} exited names "
              f"(sample of {probe})...")
        res = probe_price_availability(targets, probe)
        n = res["sampled"]
        print(f"  {len(res['have'])}/{n} have usable history "
              f"({100 * len(res['have']) / max(n, 1):.0f}%)")
        if res["missing"]:
            print(f"  no history: {', '.join(res['missing'])}")
        print("  Names without history cannot be restored to a point-in-time")
        print("  backtest from this data source; that residual is the floor on")
        print("  how much survivorship bias a free-data v2 can remove.")
        print()
        print("  The split is not random. Companies dropped from the index but")
        print("  still publicly traded keep a full series; companies acquired,")
        print("  taken private or wound up return nothing at all. So the names")
        print("  this source cannot restore are exactly the terminal outcomes —")
        print("  the half of survivorship bias that matters most.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
