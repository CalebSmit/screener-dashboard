"""Measurements behind the 2026-09-16 synthesis (design section of
``research/2026-09-14-sell-discipline-and-hold-bands.md``).

Run from the repo root::

    python research/measurements/2026-09-16-hold-band-and-input-churn.py

Every number quoted in §8 of that note is printed here. Re-run it as the run
series grows - §9 pre-registers a re-measurement at 60+ comparable runs, and
this script is what that re-measurement means.

**What this is and is not.** Descriptive statistics on *published scores only*:
where a name ranked, and how many metrics fed its score. No forward returns, no
information coefficient, no backtest. `CLAUDE.md` rules 4 and 5 restrict what
may justify a methodology change; neither bites here, because nothing below is
a return or a performance estimate. What it measures is the *behaviour of our
own ranking* - how far a top-25 name travels over a given calendar gap, and how
often its score is fed by a changing set of metrics.

Two estimators appear below and the difference matters:

* **Path simulation** walks one book along the real run sequence. It is what a
  turnover figure means in practice, but it yields one observation per run and
  is fragile where the series has gaps.
* **Pairwise migration** takes *every* ordered pair of comparable runs at a
  given calendar spacing. It is path-independent and yields 20-70x more
  observations. Where the two disagree, prefer this one - §6.3 of the note
  reported "a 25/50 band never fires" from a single 18-run path, which the
  pairwise estimator shows to be an artifact of that path.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import history  # noqa: E402  - needs the repo root on sys.path first

BANDS = (25, 30, 35, 40, 50)
TARGET = 25  # config.yaml -> portfolio.num_stocks


def load():
    kept, excluded = history.select_comparable_runs()
    by_date = {s.date: s for s in kept}
    dates = [s.date for s in kept]
    ts = {d: pd.Timestamp(d) for d in dates}
    return kept, excluded, by_date, dates, ts


def pairs_at(dates, ts, lo, hi):
    """Every ordered pair of runs separated by lo..hi calendar days.

    These pairs **overlap heavily** - 34 runs yield 72 pairs at 12-18 day
    spacing, so each run feeds many pairs and the resulting observations are
    nowhere near independent. Use ``disjoint_pairs_at`` for anything that needs
    an honest count. This is the same trap
    ``research/2026-08-10-ic-evidence-independence.md`` found in the IC series
    and ``improvement_engine._effective_observations()`` now guards against;
    it applies just as much to rank-migration statistics.
    """
    return [(a, b) for i, a in enumerate(dates) for b in dates[i + 1:]
            if lo <= (ts[b] - ts[a]).days <= hi]


def disjoint_pairs_at(dates, ts, lo, hi):
    """A maximal set of pairs at this spacing that share no run and do not
    straddle one another - the non-overlapping analogue of the above."""
    out, cursor = [], None
    for i, a in enumerate(dates):
        if cursor is not None and a < cursor:
            continue
        for b in dates[i + 1:]:
            if lo <= (ts[b] - ts[a]).days <= hi:
                out.append((a, b))
                cursor = b
                break
    return out


SPACINGS = [
    ("1 day", 1, 1),
    ("2-4 days", 2, 4),
    ("5-9 days (weekly)", 5, 9),
    ("12-18 days", 12, 18),
    ("25-35 days (monthly)", 25, 35),
    ("50-70 days", 50, 70),
]


def section_1_migration(by_date, dates, ts):
    print("=" * 78)
    print("1. Where does a top-25 name sit after N days?  [pairwise, path-independent]")
    print("=" * 78)
    print("Of the names ranked <=25 at t0, the distribution of their rank at t1.")
    print("The 'B=' columns are breach rates: the fraction outside a hold band of")
    print("that width, i.e. the one-sided turnover a review at that spacing implies.")
    print()
    print("Reported twice. ALL PAIRS uses every pair at the spacing and overlaps")
    print("heavily; DISJOINT uses a maximal non-overlapping set, so its pair count")
    print("is the honest number of independent looks. Quote the disjoint column.")
    print()
    for tag, picker in (("ALL PAIRS (overlapping)", pairs_at),
                        ("DISJOINT (independent)", disjoint_pairs_at)):
        print(f"-- {tag}")
        head = f"{'spacing':<24}{'pairs':>6}{'obs':>7}{'p50':>6}{'p90':>6}{'p95':>6}"
        head += "".join(f"{'B=' + str(b):>9}" for b in BANDS)
        print(head)
        for label, lo, hi in SPACINGS:
            ps = picker(dates, ts, lo, hi)
            obs = [by_date[b].ranks[t]
                   for a, b in ps
                   for t, r0 in by_date[a].ranks.items()
                   if r0 <= TARGET and t in by_date[b].ranks]
            if not obs:
                print(f"{label:<24}{len(ps):>6}{0:>7}   (no pairs at this spacing)")
                continue
            s = pd.Series(obs, dtype=float)
            row = (f"{label:<24}{len(ps):>6}{len(s):>7}"
                   f"{s.quantile(.50):>6.0f}{s.quantile(.90):>6.0f}{s.quantile(.95):>6.0f}")
            row += "".join(f"{(s > b).mean() * 100:8.2f}%" for b in BANDS)
            print(row)
        print()
    print("Note: even in the disjoint column one 'obs' is one ticker in one run-")
    print("pair, and the ~25 tickers within a run move together. The independent")
    print("unit is the PAIR, not the observation.")
    print()


def section_2_round_trip(by_date, dates, ts):
    print("=" * 78)
    print("2. Wasted trades: of the names that breach band B, how many are back")
    print("   inside the top 25 at the *next* review one cadence later?")
    print("=" * 78)
    print("This is what a hold band is for. NMV (2016) motivate hysteresis as")
    print("avoiding round-trips at the boundary; this measures them directly.")
    print()
    print("WARNING: these triples overlap even more than the pairs in section 1.")
    print("The disjoint triple count is printed per cadence - it is the honest")
    print("number of independent looks, and it is very small.")
    print()
    for label, lo, hi in [("1 day", 1, 1), ("2-4 days", 2, 4), ("5-9 days (weekly)", 5, 9)]:
        # How many t0<t1<t2 triples share no run at all?
        disjoint, cursor = 0, None
        for i, d0 in enumerate(dates):
            if cursor is not None and d0 < cursor:
                continue
            j = next((k for k in range(i + 1, len(dates))
                      if lo <= (ts[dates[k]] - ts[d0]).days <= hi), None)
            if j is None:
                continue
            d2 = next((d for d in dates[j + 1:]
                       if lo <= (ts[d] - ts[dates[j]]).days <= hi), None)
            if d2 is not None:
                disjoint += 1
                cursor = d2
        print(f"-- review cadence: {label}  ({disjoint} fully disjoint triples)")
        for band in BANDS:
            breached = returned = 0
            for i, d0 in enumerate(dates):
                # nearest d1 at this spacing, then nearest d2 the same distance on
                for j in range(i + 1, len(dates)):
                    if not (lo <= (ts[dates[j]] - ts[d0]).days <= hi):
                        continue
                    d1 = dates[j]
                    d2 = next((d for d in dates[j + 1:]
                               if lo <= (ts[d] - ts[d1]).days <= hi), None)
                    if d2 is None:
                        break
                    a, b, c = by_date[d0], by_date[d1], by_date[d2]
                    for t, r0 in a.ranks.items():
                        if (r0 <= TARGET and t in b.ranks and b.ranks[t] > band
                                and t in c.ranks):
                            breached += 1
                            returned += c.ranks[t] <= TARGET
                    break
            if breached:
                print(f"   B={band:<3} {breached:>4} breaches, {returned:>3} back inside "
                      f"top {TARGET} next review = {returned / breached * 100:5.1f}% wasted")
            else:
                print(f"   B={band:<3}    no breach/next-review triples at this spacing")
        print()


def section_3_turnover(by_date, dates, ts):
    print("=" * 78)
    print("3. Implied one-sided turnover along the real run path  [path simulation]")
    print("=" * 78)
    print("Sell on breach of band B, refill to 25 from the highest-ranked non-held.")
    print("Novy-Marx & Velikov (2016): anomalies under ~50% monthly one-sided")
    print("turnover mostly survive trading costs; few above it do.")
    print()

    def simulate(review_dates, band):
        held, sells = None, 0
        for d in review_dates:
            snap = by_date[d]
            order = sorted(snap.ranks, key=lambda t: snap.ranks[t])
            if held is None:
                held = set(order[:TARGET])
                continue
            keep = {t for t in held if t in snap.ranks and snap.ranks[t] <= band}
            sells += len(held) - len(keep)
            for t in order:
                if len(keep) >= TARGET:
                    break
                keep.add(t)
            held = keep
        years = (ts[review_dates[-1]] - ts[review_dates[0]]).days / 365.25
        return sells, years

    def every(cadence_days):
        out = [dates[0]]
        for d in dates[1:]:
            if (ts[d] - ts[out[-1]]).days >= cadence_days:
                out.append(d)
        return out

    dense = [d for d in dates if d >= "2026-08-10"]
    for cname, rdates, caveat in [
        ("every run (dense window 2026-08-10..latest)", dense, ""),
        ("monthly (>=28d apart, full series)", every(28),
         "  [THIN - spans the 2026-04..07 data gap]"),
        ("quarterly (>=85d apart, full series)", every(85),
         "  [THIN - one real review; not decision-grade]"),
    ]:
        span = (ts[rdates[-1]] - ts[rdates[0]]).days
        print(f"-- {cname}: {len(rdates)} reviews over {span} days{caveat}")
        for band in BANDS:
            sells, years = simulate(rdates, band)
            if years <= 0:
                continue
            monthly = (sells / years / 12) / TARGET * 100
            print(f"   band {band:>3}: {sells:>3} sells, {sells / years:6.1f}/yr, "
                  f"monthly one-sided turnover {monthly:6.1f}%")
        print()


def pct_cols(df):
    return [c for c in df.columns
            if c.endswith("_pct") and c not in ("_beta_overlap_pct",
                                                "portfolio_turnover_pct")]


def section_4_input_churn(by_date, dates):
    print("=" * 78)
    print("4. Is a rank move information, or an input going missing?")
    print("=" * 78)
    print("A metric percentile that changes between present and absent means the")
    print("category renormalised over a different metric set. The score moved")
    print("because the measurement changed, not because the company did.")
    print("This is the FCX case in CLAUDE.md priority 1.5, counted.")
    print()

    index = history.snapshot_index()
    frames = {d: pd.read_parquet(index[d]).set_index("Ticker") for d in dates}
    usable = [d for d in dates if pct_cols(frames[d])]
    print(f"{len(dates)} comparable runs; {len(usable)} carry metric percentile "
          f"columns ({usable[0]}..{usable[-1]}).")
    print("Earlier snapshots predate that schema - 15 columns, no percentiles.")
    print()

    rows = []
    for d0, d1 in zip(usable, usable[1:]):
        if (pd.Timestamp(d1) - pd.Timestamp(d0)).days > 7:
            continue
        a, b = frames[d0], frames[d1]
        cols = sorted(set(pct_cols(a)) & set(pct_cols(b)))
        shared = a.index.intersection(b.index)
        av, bv = a.loc[shared, cols], b.loc[shared, cols]
        lost = (av.notna() & bv.isna()).sum(axis=1)
        gained = (av.isna() & bv.notna()).sum(axis=1)
        ra, rb = by_date[d0].ranks, by_date[d1].ranks
        for t in shared:
            if t in ra and t in rb:
                rows.append({"rank0": ra[t], "rank1": rb[t], "drank": rb[t] - ra[t],
                             "churn": int(lost[t]) + int(gained[t])})
    df = pd.DataFrame(rows)
    print(f"{len(df)} ticker-transitions over consecutive runs <=7 days apart.")
    print()
    print("How common is an input-availability change?")
    for k in (1, 2, 4):
        print(f"   >={k} metric(s) lost or gained: {(df.churn >= k).mean() * 100:6.2f}%")
    print()
    print("Does it move the rank?  |rank change| by amount of churn:")
    for label, sub in [("stable (churn=0)", df[df.churn == 0]),
                       ("churn 1", df[df.churn == 1]),
                       ("churn 2-3", df[(df.churn >= 2) & (df.churn <= 3)]),
                       ("churn >=4", df[df.churn >= 4])]:
        if len(sub):
            a = sub.drank.abs()
            print(f"   {label:<18} n={len(sub):>6}  p50={a.quantile(.5):5.1f}  "
                  f"p90={a.quantile(.9):6.1f}  p99={a.quantile(.99):7.1f}")
    print()
    print("Direction - does churn push down, or just scatter?")
    for label, sub in [("churn=0", df[df.churn == 0]), ("churn=1", df[df.churn == 1]),
                       ("churn>=2", df[df.churn >= 2])]:
        print(f"   {label:<10} n={len(sub):>6}  median signed drank={sub.drank.median():5.1f}"
              f"  share worsening={(sub.drank > 0).mean() * 100:5.1f}%")
    print()

    top = df[df.rank0 <= TARGET]
    left, stay = top[top.rank1 > TARGET], top[top.rank1 <= TARGET]
    a_, c_ = int((left.churn > 0).sum()), int((stay.churn > 0).sum())
    print(f"The event a holdings panel surfaces - a top-{TARGET} name leaving the top "
          f"{TARGET}:")
    print(f"   exits    : {a_:>3} of {len(left):>3} had input churn "
          f"({a_ / len(left) * 100:.1f}%)")
    print(f"   stayers  : {c_:>3} of {len(stay):>3} had input churn "
          f"({c_ / len(stay) * 100:.1f}%)")
    try:
        from scipy import stats
        odds, p = stats.fisher_exact([[a_, len(left) - a_], [c_, len(stay) - c_]])
        print(f"   Fisher exact: odds ratio {odds:.2f}, p = {p:.4f}")
        u, p2 = stats.mannwhitneyu(df[df.churn >= 2].drank.abs(),
                                   df[df.churn == 0].drank.abs(), alternative="greater")
        print(f"   |rank change| churn>=2 vs churn=0, Mann-Whitney p = {p2:.3g}")
    except ImportError:
        print("   (scipy not available - significance tests skipped)")
    print()


def main():
    kept, excluded, by_date, dates, ts = load()
    print()
    print(f"Comparable runs: {len(kept)}  ({dates[0]} .. {dates[-1]})")
    dropped = ", ".join("{} ({})".format(e["date"], e["reason"]) for e in excluded)
    print("Excluded: " + (dropped or "none"))
    print("Gaps > 7 days in the series: ", end="")
    gaps = [f"{a}->{b} ({(ts[b] - ts[a]).days}d)"
            for a, b in zip(dates, dates[1:]) if (ts[b] - ts[a]).days > 7]
    print(", ".join(gaps) or "none")
    print()
    section_1_migration(by_date, dates, ts)
    section_2_round_trip(by_date, dates, ts)
    section_3_turnover(by_date, dates, ts)
    section_4_input_churn(by_date, dates)


if __name__ == "__main__":
    main()
