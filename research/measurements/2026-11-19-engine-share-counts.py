"""Where do the research engine's wrong market caps come from? Measured 2026-11-19 (13:00).

The morning's measurement (``2026-11-19-market-cap-floor-never-fires.py``, part 4) found 16
tickers whose point-in-time market cap reads under $100M somewhere in ``cache/v2_metrics`` and
Berkshire Hathaway under $2B in every month. ``pit_record.market_fields`` builds market cap as
price x a share count taken from the filings: the more recently *dated* of the latest
``CommonStockSharesOutstanding`` (balance sheet) and ``EntityCommonStockSharesOutstanding``
(cover page) instants. This script follows that choice back to the filed facts, for each ticker.

Part 1 (``--trace``): for every affected ticker, the share-count facts visible in a month where
the cap is absurd - concept, period end, filed date, form, value - and which one the rule picks.

Part 2 (``--sweep``): for every ticker in the facts cache, at every month-end the engine scores,
the count the current rule picks against the company's own weighted-average share count for the
latest period (``WeightedAverageNumberOf(Diluted)SharesOutstanding`` - a different concept in the
same filing, on a ``shares`` unit, which cannot be per-class for a company reporting one EPS).
Reports how many ticker-months disagree by more than 2x, 5x and 10x in either direction.

Read-only. ``python research/measurements/2026-11-19-engine-share-counts.py --trace``
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pit_record as pr  # noqa: E402

OUT = ROOT / "logs"          # gitignored: the per-name-month tables are ~6 MB each
SHARE = ("CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding")
WAVG = ("WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic")


def absurd_months() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted((ROOT / "cache" / "v2_metrics").glob("????-??-??.parquet")):
        d = pd.read_parquet(p, columns=["Ticker", "size_log_mcap"])
        s = pd.to_numeric(d["size_log_mcap"], errors="coerce")
        for t in d.loc[-s < math.log(1e8), "Ticker"]:
            out.setdefault(t, []).append(p.stem)
    out.setdefault("BRK-B", [])
    return out


def trace(facts: pd.DataFrame) -> None:
    bad = absurd_months()
    all_months = [p.stem for p in sorted((ROOT / "cache" / "v2_metrics").glob("????-??-??.parquet"))]
    for t, months in sorted(bad.items(), key=lambda kv: -len(kv[1])):
        probe = sorted({months[0], months[len(months) // 2], months[-1]}) if months else [all_months[-1]]
        print(f"\n=== {t}: {len(months)} months under $100M; first {months[0] if months else '-'}, "
              f"last {months[-1] if months else '-'}")
        for m in probe:
            F = pr.Filed(facts, t, m)
            a, b = F.latest_instant("shares_out"), F.latest_instant("cover_shares")
            got = [x for x in (a, b) if x]
            pick = max(got, key=lambda x: x[1]) if got else None
            print(f"  as of {m}: balance-sheet {a and (f'{a[0]:,.0f}', str(a[1].date()))}  "
                  f"cover {b and (f'{b[0]:,.0f}', str(b[1].date()))}  -> picked "
                  f"{pick and f'{pick[0]:,.0f}'}")
            raw = facts[(facts["ticker"] == t) & facts["concept"].isin(SHARE + WAVG)
                        & (pd.to_datetime(facts["filed"]) <= pd.Timestamp(m))]
            raw = raw.sort_values(["filed", "concept", "end"]).tail(10)
            for _, r in raw.iterrows():
                print(f"      {r['concept'][:44]:44s} end {str(r['end'])[:10]} filed {str(r['filed'])[:10]} "
                      f"{r['form']:6s} {r['val']:>18,.0f}")


def sweep(facts: pd.DataFrame) -> None:
    months = [p.stem for p in sorted((ROOT / "cache" / "v2_metrics").glob("????-??-??.parquet"))]
    sub = facts[facts["concept"].isin(SHARE + WAVG) & facts["form"].isin(pr.FORMS)].copy()
    for c in ("start", "end", "filed"):
        sub[c] = pd.to_datetime(sub[c], errors="coerce")
    sub = sub.dropna(subset=["end", "val"])
    rows = []
    for t, g in sub.groupby("ticker"):
        inst = g[g["concept"].isin(SHARE)]
        wav = g[g["concept"].isin(WAVG)]
        for m in months:
            ts = pd.Timestamp(m)
            i = inst[inst["filed"] <= ts]
            w = wav[wav["filed"] <= ts]
            if i.empty or w.empty:
                continue
            i = i.sort_values(["filed"]).drop_duplicates(["concept", "end"], keep="last")
            picked = np.nan
            best_end = None
            for c in SHARE:                      # the rule as shipped: later-dated instant wins,
                gi = i[i["concept"] == c]        # the balance-sheet concept on a tie
                if gi.empty:
                    continue
                r = gi.sort_values(["end", "filed"]).iloc[-1]
                if best_end is None or r["end"] > best_end:
                    picked, best_end = float(r["val"]), r["end"]
            w = w.sort_values("filed").drop_duplicates(["concept", "start", "end"], keep="last")
            w = w[w["end"] == w["end"].max()]
            ref = float(w.sort_values("filed").iloc[-1]["val"])
            if picked > 0 and ref > 0:
                rows.append((t, m, picked, ref, picked / ref))
    d = pd.DataFrame(rows, columns=["ticker", "month", "picked", "wavg", "ratio"])
    print(f"ticker-months compared: {len(d):,} across {d['ticker'].nunique()} tickers")
    for k in (2, 5, 10):
        lo, hi = d[d["ratio"] < 1 / k], d[d["ratio"] > k]
        print(f"  picked < wavg/{k}: {len(lo):,} ticker-months, {lo['ticker'].nunique()} tickers | "
              f"picked > {k}x wavg: {len(hi):,}, {hi['ticker'].nunique()} tickers")
    for label, part in (("LOW (<1/2)", d[d["ratio"] < 0.5]), ("HIGH (>2x)", d[d["ratio"] > 2])):
        by = part.groupby("ticker").agg(n=("month", "count"), first=("month", "min"),
                                        last=("month", "max"), med=("ratio", "median"))
        print(f"\n{label}: {len(by)} tickers")
        print(by.sort_values("n", ascending=False).to_string())
    d.to_csv(OUT / "2026-11-19-engine-share-counts.sweep.csv", index=False)


# ----------------------------------------------------------------------------------------------
# Part 3 (``--evaluate``): the rule as shipped against a candidate rule, on every name-month the
# engine scores. The candidate was written down before this part was first run:
#
#   1. Three sources, each the latest visible on the day: the balance-sheet count, the cover-page
#      count, and the weighted-average basic count (diluted where basic is absent) of the latest
#      period.
#   2. A source dated more than STALE_DAYS before the month-end is not used (a company reports a
#      count every quarter; an older one is a tag the filer stopped using).
#   3. Each count is put on today's share basis by the splits dated after the day its filing was
#      **filed**, not after the month-end: a filing gives retroactive effect to a split effected
#      before it is issued (SEC SAB Topic 4.C), and knows nothing of one after.
#   4. Preference order: the later-dated of the two instants, the other instant, the weighted
#      average. Take the first that agrees within AGREE x with at least one other source. One
#      source alone is used as it is. Several sources and no two agreeing: unknown (NaN).
#   5. Backstop: a market value under FLOOR is unknown (NaN), never a small company.
STALE_DAYS = 460
AGREE = 1.5
FLOOR = 50e6

# Version 2 - written AFTER version 1's result was read, and labelled as such wherever it is
# quoted. Version 1 removed every market value under $100M but turned 766 name-months the shipped
# rule had right into "unknown": where exactly two sources exist and one carries a units slip
# (ConocoPhillips' weighted average is filed in thousands for 97 months), a vote of two cannot
# say which. And it kept agreeing pairs that were both wrong (Garmin's two instants, both in
# thousands). So a referee that carries no share-unit scaling at all:
#
#   R = net income / diluted EPS, both for the latest period that reports both (|EPS| >= 0.10,
#       so the cent EPS is rounded to moves R by under ~5%), on the filed-date split basis.
#
#   4'. Where R exists: the first source in the preference order within REFEREE x of R; none
#       within it -> R itself where no source was fresh, else unknown. Where R does not exist:
#       version 1's vote.
#   6.  A declared exception, not a rule: Berkshire Hathaway reports its count and its EPS in
#       Class A equivalents and the line scored is Class B; one Class A share converts into
#       1,500 Class B (its 10-K, note on common stock). EQUIVALENT_UNITS multiplies its count.
REFEREE = 3.0
MIN_EPS = 0.10
EQUIVALENT_UNITS = {"BRK-B": 1500.0}
EPS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")
NI = ("NetIncomeLoss",)


# Version 3 - written after version 2's result was read, and the last one this session runs.
# Version 2 cut "had a value, now unknown" from 766 to 559 but put 152 name-months under the
# backstop where version 1 had 66: the referee is not unit-safe either (Halliburton, 43 months,
# where version 1's vote was right). Each mechanism fails where the other holds, so:
#
#   4''. Version 1's vote first. Only where the vote gives nothing, or gives a market value
#        under FLOOR at the month's price, does the referee decide: the first source within
#        REFEREE x of R that clears FLOOR; none -> R itself if it clears FLOOR; else unknown.
#   Steps 1-3, 5 and 6 as before.
#
# ACCEPTANCE, written and committed before version 3 was run. All four, or it is not accepted:
#   a. no name-month under $100M, and none over $4T other than the four companies version 1
#      showed are really that large (Apple, Alphabet's two lines, NVIDIA; 2025-07 onward);
#   b. "the shipped rule had a market value, the candidate says unknown": at most 150 name-months
#      (0.23% of 65,458), every ticker listed;
#   c. at most 3 month-to-month jumps of more than 5x in the count on today's share basis;
#   d. against the live fetch, newest month: at least 474 within 10% and at most 5 off by 2x.
# A fifth number is reported and is not a criterion: how many name-months' market value moves by
# more than 10% - that is the size of the rerun's possible effect, not a measure of correctness.


def _referee(f: pd.DataFrame, ts: pd.Timestamp, sp):
    e = f[f["concept"].isin(EPS) & f["start"].notna() & (f["val"].abs() >= MIN_EPS)]
    n = f[f["concept"].isin(NI) & f["start"].notna()]
    if e.empty or n.empty:
        return None
    m = e.merge(n, on=["start", "end"], suffixes=("_e", "_n"))
    m = m[(ts - m["end"]).dt.days <= STALE_DAYS]
    if m.empty:
        return None
    r = m.sort_values(["end", "filed_e"], kind="mergesort").iloc[-1]
    v = r["val_n"] / r["val_e"]
    return float(v) * _split_after(sp, r["filed_e"]) if v > 0 else None


def _visible(g: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """``pit_record.Filed``'s selection for instants and durations of the share concepts."""
    f = g[g["filed"] <= ts]
    if f.empty:
        return f
    f = f.sort_values(["filed", "_amend"], kind="mergesort")
    return f.drop_duplicates(["concept", "start", "end"], keep="last")


def _latest(f: pd.DataFrame, concepts, instant: bool):
    for c in concepts:
        g = f[f["concept"] == c]
        g = g[g["start"].isna()] if instant else g[g["start"].notna()]
        if g.empty:
            continue
        r = g.sort_values(["end", "filed"], kind="mergesort").iloc[-1]
        return float(r["val"]), r["end"], r["filed"]
    return None


def _split_after(sp, day) -> float:
    if sp is None:
        return 1.0
    v = sp.loc[sp["date"] > day, "ratio"]
    return float(np.prod(v)) if len(v) else 1.0


def evaluate(version: int = 1) -> None:
    import backtest_v2 as b2
    facts = b2._load_facts(ROOT)
    facts = facts[facts["concept"].isin(SHARE + WAVG + EPS + NI) & facts["form"].isin(pr.FORMS)].copy()
    for c in ("start", "end", "filed"):
        facts[c] = pd.to_datetime(facts[c], errors="coerce")
    facts = facts.dropna(subset=["end", "val"])
    facts["_amend"] = facts["form"].str.endswith("/A").astype(int)
    by_t = {t: g for t, g in facts.groupby("ticker")}
    import universe_history as uh
    splits = pd.read_parquet(ROOT / "cache" / "splits.parquet")
    splits["ticker"] = splits["ticker"].map(uh.normalize_ticker)
    splits["date"] = pd.to_datetime(splits["date"])
    sp_by = {t: g for t, g in splits.groupby("ticker")}

    rows = []
    for p in sorted((ROOT / "cache" / "v2_metrics").glob("????-??-??.parquet")):
        ts = pd.Timestamp(p.stem)
        d = pd.read_parquet(p, columns=["Ticker", "size_log_mcap"])
        for t, slm in zip(d["Ticker"], pd.to_numeric(d["size_log_mcap"], errors="coerce")):
            g = by_t.get(t)
            if g is None:
                continue
            f = _visible(g, ts)
            if f.empty:
                continue
            sp = sp_by.get(t)
            bs, cv = _latest(f, SHARE[:1], True), _latest(f, SHARE[1:], True)
            wa = _latest(f, (WAVG[1], WAVG[0]), False)
            # the rule as shipped
            got = [x for x in (bs, cv) if x]
            cur = max(got, key=lambda x: x[1])[0] * _split_after(sp, ts) if got else np.nan
            # the candidate
            inst = sorted([x for x in (bs, cv) if x and (ts - x[1]).days <= STALE_DAYS],
                          key=lambda x: x[1], reverse=True)
            pref = inst + ([wa] if wa and (ts - wa[1]).days <= STALE_DAYS else [])
            vals = [x[0] * _split_after(sp, x[2]) for x in pref if x[0] > 0]
            cand, why = np.nan, "none"
            if len(vals) == 1:
                cand, why = vals[0], "alone"
            elif len(vals) > 1:
                why = "disagree"
                for i, v in enumerate(vals):
                    if any(j != i and 1 / AGREE <= v / w <= AGREE for j, w in enumerate(vals)):
                        cand, why = v, ("first" if i == 0 else f"choice{i + 1}")
                        break
            cap = math.exp(-slm) if np.isfinite(slm) else np.nan
            if version == 3:
                price = cap / cur if np.isfinite(cap) and np.isfinite(cur) and cur > 0 else np.nan
                unit = EQUIVALENT_UNITS.get(t, 1.0)

                def clears(v):          # unknown price: nothing to test against, so it stands
                    return not np.isfinite(price) or v * unit * price >= FLOOR
                cand2, why2, ref = cand, why, np.nan
                if not np.isfinite(cand2) or not clears(cand2):
                    ref = _referee(f, ts, sp)
                    cand2, why2 = np.nan, (why if not np.isfinite(cand) else "under-floor")
                    if ref is not None:
                        for i, v in enumerate(vals):
                            if 1 / REFEREE <= v / ref <= REFEREE and clears(v):
                                cand2, why2 = v, f"referee-chose{i + 1}"
                                break
                        else:
                            if clears(ref):
                                cand2, why2 = ref, "referee"
                    ref = ref if ref is not None else np.nan
                cand2 *= unit
                rows.append((t, p.stem, cap, cur, cand, why, cand2, why2, ref))
                continue
            # version 2
            ref = _referee(f, ts, sp)
            cand2, why2 = cand, why
            if ref is not None:
                cand2, why2 = np.nan, ("ref-refuses" if vals else "none")
                for i, v in enumerate(vals):
                    if 1 / REFEREE <= v / ref <= REFEREE:
                        cand2, why2 = v, ("first" if i == 0 else f"choice{i + 1}")
                        break
                if not vals:
                    cand2, why2 = ref, "referee"
            cand2 *= EQUIVALENT_UNITS.get(t, 1.0)
            cap = math.exp(-slm) if np.isfinite(slm) else np.nan
            rows.append((t, p.stem, cap, cur, cand, why, cand2, why2, ref))
    d = pd.DataFrame(rows, columns=["ticker", "month", "cap_cached", "cur", "cand", "why",
                                    "cand2", "why2", "referee"])
    d["price"] = d["cap_cached"] / d["cur"]            # the engine's own price for the month
    d["cap_cand"] = d["price"] * d["cand"]
    d.loc[d["cap_cand"] < FLOOR, ["cand", "cap_cand"]] = np.nan
    d["cap_cand2"] = d["price"] * d["cand2"]
    d.loc[d["cap_cand2"] < FLOOR, ["cand2", "cap_cand2"]] = np.nan
    d.to_csv(OUT / ("2026-11-19-engine-share-counts.evaluate-v3.csv" if version == 3
                    else "2026-11-19-engine-share-counts.evaluate.csv"), index=False)
    if version >= 2:
        print(f"VERSION {version} (written after the version before it was read):")
        d = d.drop(columns=["cand", "why", "cap_cand"]).rename(
            columns={"cand2": "cand", "why2": "why", "cap_cand2": "cap_cand"})
    n = len(d)
    print(f"name-months: {n:,}; tickers {d['ticker'].nunique()}; with a cached market value {d['cap_cached'].notna().sum():,}")
    print("how the candidate chose:", d["why"].value_counts().to_dict())
    print(f"candidate unknown (NaN): {int(d['cand'].isna().sum()):,} "
          f"(shipped rule unknown: {int(d['cur'].isna().sum()):,})")
    r = d["cand"] / d["cur"]
    print(f"candidate within 2% of the shipped count: {int(((r - 1).abs() <= 0.02).sum()):,}; "
          f"differs by more than 1.5x either way: {int(((r > 1.5) | (r < 1 / 1.5)).sum()):,} name-months, "
          f"{d.loc[(r > 1.5) | (r < 1 / 1.5), 'ticker'].nunique()} tickers")

    print("\nA. implausible market values (name-months / tickers):")
    for label, col in (("shipped", "cap_cached"), ("candidate", "cap_cand")):
        lo, hi = d[d[col] < 1e8], d[d[col] > 4e12]
        print(f"   {label:9s}: under $100M {len(lo):4d} / {lo['ticker'].nunique():2d}   "
              f"over $4T {len(hi):3d} / {hi['ticker'].nunique():2d}  {sorted(hi['ticker'].unique())}")

    print("\nB. month-to-month jumps in the count on today's share basis (a real count moves a few % a month):")
    d = d.sort_values(["ticker", "month"])
    for label, col in (("shipped", "cur"), ("candidate", "cand")):
        j = d.groupby("ticker")[col].transform(lambda s: s / s.shift(1))
        for k in (1.5, 5.0):
            m = (j > k) | (j < 1 / k)
            print(f"   {label:9s}: by more than {k}x: {int(m.sum()):4d} jumps, {d.loc[m, 'ticker'].nunique():3d} tickers")

    print("\nC. the newest month against the live fetch's sharesOutstanding (one cross-section):")
    import run_context
    raw = pd.read_parquet(run_context.latest_run_dir() / "00_raw_fetch.parquet", columns=["Ticker", "sharesOutstanding", "marketCap", "currentPrice"])
    raw["live"] = pd.to_numeric(raw["marketCap"], errors="coerce") / pd.to_numeric(raw["currentPrice"], errors="coerce")
    last = d[d["month"] == d["month"].max()].merge(raw[["Ticker", "live"]], left_on="ticker", right_on="Ticker")
    for label, col in (("shipped", "cur"), ("candidate", "cand")):
        q = last[col] / last["live"]
        ok = q.notna()
        print(f"   {label:9s}: compared {int(ok.sum())}; within 10% {int(((q - 1).abs() <= 0.10).sum())}; "
              f"off by more than 2x {int(((q > 2) | (q < 0.5)).sum())}; unknown {int((~ok).sum())}")
    q = last["cand"] / last["live"]
    off = last[(q > 1.25) | (q < 0.8)].assign(q=q)
    print("   candidate off by more than 25%:", [(t, round(v, 3)) for t, v in zip(off["ticker"], off["q"])])
    q = last["cur"] / last["live"]
    off = last[(q > 1.25) | (q < 0.8)].assign(q=q)
    print("   shipped   off by more than 25%:", [(t, round(v, 4)) for t, v in zip(off["ticker"], off["q"])])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--version", type=int, default=1, choices=(1, 2, 3))
    a = ap.parse_args()
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet") if (a.trace or a.sweep) else None
    if a.trace:
        trace(facts)
    if a.sweep:
        sweep(facts)
    if a.evaluate:
        evaluate(a.version)
