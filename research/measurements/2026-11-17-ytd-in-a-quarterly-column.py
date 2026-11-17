"""How often does the source put a year-to-date figure in a quarterly cash-flow column, and can
that be recognised from the statement alone?

The 2026-11-17 morning session found DexCom (rank 8) scored on $2.209bn of operating cash flow
against $1.414bn filed: the source's 2026-09-30 *quarterly* column holds $1.120bn, which is the
nine months to that date as filed. This measures the question that session left for research:

  1. PREVALENCE. Every operating-cash-flow column of every stock's quarterly statement is
     compared with the company's own filings (the SEC cache): the discrete quarter the filings
     imply, and the year-to-date figure for the same period end. A column is
       right   within TOL of the filed quarter
       ytd     over TOL from the quarter and within TOL of the filed year-to-date figure
       other   neither
     (first fiscal quarters are excluded from "ytd": there the two are the same number).
  2. DETECTORS, each using nothing but what the source itself returns - scoring may not read a
     filing - judged against that ground truth:
       A  "exceeds its fiscal year so far": a column at fiscal position k >= 2 that is within
          TOL of (itself + the earlier columns of its fiscal year) cannot be told apart, so the
          usable form is the *implied* one: the column is at least the sum of the earlier
          quarters of its fiscal year, with the same sign. Fires on seasonality by construction.
       B  "a completed fiscal year does not add up": the four quarterly columns of a fiscal year
          the annual statement also carries, against the annual figure.
       C  "the source disagrees with itself": the source's own trailing-twelve-month figure
          against the sum of its four newest quarterly columns.

    python research/measurements/2026-11-17-ytd-in-a-quarterly-column.py --fetch   # ~500 tickers
    python research/measurements/2026-11-17-ytd-in-a-quarterly-column.py           # the tables

The statements are pickled to the gitignored data/measurements/2026-11-17-cashflow/ so the tables
can be re-run without refetching; the JSON summary is written beside this file.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

STORE = ROOT / "data" / "measurements" / "2026-11-17-cashflow"
PROPS = ("quarterly_cashflow", "cashflow", "ttm_cashflow")
LABEL = "Operating Cash Flow"
CONCEPT = "NetCashProvidedByUsedInOperatingActivities"
TOL = 0.02          # "the same figure": 2% of the larger - the Data Provenance panel's own bar
FAR = 0.10          # the cross-check's "over 10%"


def universe() -> list[str]:
    import check_run_health as H
    import run_context
    payload = H.load_payload(Path(run_context.latest_run_dir()))
    return sorted(r["Ticker"] for r in payload["table_data"])


def fetch(tickers: list[str]) -> None:
    import yfinance as yf
    STORE.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for i, tk in enumerate(tickers):
        p = STORE / f"{tk}.pkl"
        if p.exists():
            done += 1
            continue
        got = {}
        try:
            t = yf.Ticker(tk)
            for k in PROPS:
                try:
                    got[k] = getattr(t, k)
                except Exception as exc:  # noqa: BLE001 - record and carry on
                    got[k] = None
                    got[k + "_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            got["error"] = f"{type(exc).__name__}: {exc}"
        q = got.get("quarterly_cashflow")
        if q is None or getattr(q, "empty", True):
            failed += 1          # not saved: a later --fetch retries it
        else:
            with open(p, "wb") as f:
                pickle.dump(got, f)
            done += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(tickers)}  saved {done}  empty {failed}", flush=True)
        time.sleep(0.4)
    print(f"FETCH DONE saved {done} empty {failed} of {len(tickers)}", flush=True)


# ------------------------------------------------------------------ the filings' side
def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _first(rows) -> float:
    return float(rows.sort_values("filed").iloc[0]["val"])


def filed_at(f, end):
    """(discrete quarter, year-to-date, fiscal position 1-4) the filings imply for the period
    ending ``end``; any of them None when the filings cannot say."""
    near = (f["end"] - end).abs() <= timedelta(days=4)
    at = f[near & f["start"].notna()]
    if at.empty:
        return None, None, None
    span = (at["end"] - at["start"]).dt.days
    longest = at[span == span.max()]
    days = int(span.max())
    ytd = _first(longest)
    pos = min(4, max(1, round(days / 91.3)))
    if pos == 1:
        return ytd, ytd, 1
    start = longest["start"].iloc[0]
    # the year-to-date figure one quarter earlier, same fiscal-year start
    prev_end = end - timedelta(days=91)
    prev = f[((f["end"] - prev_end).abs() <= timedelta(days=10))
             & ((f["start"] - start).abs() <= timedelta(days=4))]
    if prev.empty:
        return None, ytd, pos
    pspan = (prev["end"] - prev["start"]).dt.days
    prev = prev[pspan == pspan.max()]
    return ytd - _first(prev), ytd, pos


# ------------------------------------------------------------------ the source's side
def op_row(stmt):
    if stmt is None or getattr(stmt, "empty", True) or LABEL not in stmt.index:
        return None
    row = stmt.loc[LABEL]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    row = row.dropna()
    row.index = pd.to_datetime(row.index)
    return row.sort_index(ascending=False).astype(float)


def fiscal_position(date, fy_ends) -> tuple | None:
    """(fiscal-year end the column belongs to, position 1-4) from the annual statement's own
    dates - nothing but the source."""
    if not len(fy_ends):
        return None
    newest = max(fy_ends)
    # project fiscal-year ends a year either side of the newest the source carries
    ends = sorted(set(list(fy_ends) + [newest + pd.DateOffset(years=1), newest + pd.DateOffset(years=2)]))
    for e in ends:
        gap = (e - date).days
        if -10 <= gap <= 285:
            return e, 4 - min(3, max(0, round(gap / 91.3)))
    return None


def analyse() -> dict:
    from sec_fundamentals import FACTS_PATH
    import check_run_health as H
    import run_context
    payload = H.load_payload(Path(run_context.latest_run_dir()))
    rank = {r["Ticker"]: r["Rank"] for r in payload["table_data"]}
    facts = pd.read_parquet(FACTS_PATH)
    facts = facts[(facts["concept"] == CONCEPT) & facts["form"].isin(("10-K", "10-Q", "10-K/A", "10-Q/A"))]
    by = dict(tuple(facts.groupby("ticker")))

    cols = []      # one dict per quarterly column
    stocks = {}    # per-stock detector readings
    files = sorted(STORE.glob("*.pkl"))
    for p in files:
        tk = p.stem
        with open(p, "rb") as fh:
            got = pickle.load(fh)
        q, a, ttm = op_row(got.get("quarterly_cashflow")), op_row(got.get("cashflow")), op_row(got.get("ttm_cashflow"))
        if q is None or not len(q):
            continue
        fy_ends = list(a.index) if a is not None else []
        f = by.get(tk)
        st = stocks.setdefault(tk, {"rank": rank.get(tk), "n_cols": len(q)})
        pos_of = {}
        for d in q.index:
            fp = fiscal_position(d, fy_ends)
            pos_of[d] = fp
        for i, (d, v) in enumerate(q.items()):
            fq = fy = fpos = None
            if f is not None:
                fq, fy, fpos = filed_at(f, d)
            fp = pos_of[d]
            # detector A: the column against the earlier columns of its own fiscal year
            a_fire = a_ready = False
            if fp and fp[1] >= 2:
                earlier = [q[e] for e in q.index if e < d and pos_of[e] and pos_of[e][0] == fp[0]]
                if len(earlier) == fp[1] - 1:
                    a_ready = True
                    s = sum(earlier)
                    a_fire = (v > 0 and s > 0 and v >= s) or (v < 0 and s < 0 and v <= s)
            verdict = None
            if fq is not None:
                if rel(v, fq) <= TOL:
                    verdict = "right"
                elif fpos and fpos >= 2 and fy is not None and rel(v, fy) <= TOL:
                    verdict = "ytd"
                else:
                    verdict = "other"
            cols.append({"t": tk, "rank": rank.get(tk), "date": str(d.date()), "newest": i == 0,
                         "in_ltm": i < 4, "v": v, "filed_q": fq, "filed_ytd": fy,
                         "pos_src": fp[1] if fp else None, "pos_filed": fpos, "verdict": verdict,
                         "a_ready": a_ready, "a_fire": a_fire})
        # detector B: completed fiscal years
        b = []
        if a is not None:
            for e, av in a.items():
                qs = [q[d] for d in q.index if pos_of[d] and abs((pos_of[d][0] - e).days) <= 10]
                if len(qs) == 4:
                    b.append({"fy": str(e.date()), "sum_q": sum(qs), "annual": av, "off": rel(sum(qs), av)})
        st["b"] = b
        # detector C: the source's own trailing figure against its four newest quarters
        if ttm is not None and len(ttm) and len(q) >= 4:
            four = q.iloc[:4]
            consecutive = all(75 <= (four.index[j] - four.index[j + 1]).days <= 110 for j in range(3))
            st["c"] = {"ttm": float(ttm.iloc[0]), "ttm_date": str(ttm.index[0].date()),
                       "sum4": float(four.sum()), "newest_q": str(q.index[0].date()),
                       "consecutive": consecutive, "off": rel(float(ttm.iloc[0]), float(four.sum()))}

    df = pd.DataFrame(cols)
    out = {"stocks_fetched": len(files), "stocks_with_a_row": len(stocks), "columns": len(df)}
    judged = df[df["verdict"].notna()]
    out["columns_judged_against_filings"] = len(judged)
    out["verdicts"] = judged["verdict"].value_counts().to_dict()
    multi = judged[judged["pos_filed"] >= 2]
    out["columns_at_fiscal_position_2_plus"] = len(multi)
    out["verdicts_position_2_plus"] = multi["verdict"].value_counts().to_dict()
    print(f"\nstocks fetched {len(files)}, with an operating-cash-flow row {len(stocks)}, columns {len(df)}")
    print(f"columns the filings can judge: {len(judged)}  ->  {out['verdicts']}")
    print(f"  of them at fiscal position 2-4 (where year-to-date differs from the quarter): {len(multi)}"
          f"  ->  {out['verdicts_position_2_plus']}")

    ytd = judged[judged["verdict"] == "ytd"].sort_values(["t", "date"])
    print(f"\nYEAR-TO-DATE COLUMNS: {len(ytd)} columns on {ytd['t'].nunique()} stocks")
    for r in ytd.itertuples():
        print(f"  #{r.rank!s:<5}{r.t:<6}{r.date}  source {r.v / 1e9:>8.3f}bn  filed quarter {r.filed_q / 1e9:>8.3f}bn"
              f"  filed YTD {r.filed_ytd / 1e9:>8.3f}bn  pos {r.pos_filed}  newest={r.newest}  in LTM={r.in_ltm}")
    out["ytd_columns"] = [{k: getattr(r, k) for k in ("t", "rank", "date", "v", "filed_q", "filed_ytd", "pos_filed", "newest", "in_ltm")}
                          for r in ytd.itertuples()]

    other = judged[(judged["verdict"] == "other")]
    far = other[pd.Series([rel(r.v, r.filed_q) > FAR for r in other.itertuples()], index=other.index, dtype=bool)]
    print(f"\nOTHER (neither the quarter nor the year-to-date figure): {len(other)} columns, "
          f"{len(far)} of them over 10% from the filed quarter, on {far['t'].nunique()} stocks; "
          f"in the four newest columns: {int(far['in_ltm'].sum())}")
    out["other_columns"] = len(other)
    out["other_over_10pct"] = len(far)
    out["other_over_10pct_in_ltm"] = int(far["in_ltm"].sum())
    for r in far[far["in_ltm"]].sort_values("rank").head(25).itertuples():
        print(f"  #{r.rank!s:<5}{r.t:<6}{r.date}  source {r.v / 1e9:>8.3f}bn  filed quarter {r.filed_q / 1e9:>8.3f}bn"
              f"  filed YTD {(r.filed_ytd or float('nan')) / 1e9:>8.3f}bn  pos {r.pos_filed}")

    # ---- detector A
    ready = judged[judged["a_ready"]]
    fired = ready[ready["a_fire"]]
    tp = fired[fired["verdict"] == "ytd"]
    missed = ready[(ready["verdict"] == "ytd") & ~ready["a_fire"]]
    not_ready_ytd = judged[(judged["verdict"] == "ytd") & ~judged["a_ready"]]
    print(f"\nDETECTOR A (column >= the earlier quarters of its fiscal year): testable on {len(ready)} judged columns; "
          f"fires on {len(fired)} ({len(fired) / max(len(ready), 1):.1%}); true year-to-date among them {len(tp)}; "
          f"year-to-date columns it is testable on and misses {len(missed)}; year-to-date columns it cannot test {len(not_ready_ytd)}")
    new = ready[ready["newest"]]
    new_f = new[new["a_fire"]]
    print(f"  on the NEWEST column only: testable {len(new)}, fires {len(new_f)} ({len(new_f) / max(len(new), 1):.1%}), "
          f"true year-to-date {int((new_f['verdict'] == 'ytd').sum())}, false alarms {int((new_f['verdict'] == 'right').sum())} right "
          f"+ {int((new_f['verdict'] == 'other').sum())} other")
    out["A"] = {"testable": len(ready), "fires": len(fired), "true": len(tp), "missed": len(missed),
                "untestable_ytd": len(not_ready_ytd), "newest_testable": len(new), "newest_fires": len(new_f),
                "newest_true": int((new_f["verdict"] == "ytd").sum())}
    by_pos = {}
    for k in (2, 3, 4):
        r_k = ready[ready["pos_src"] == k]
        by_pos[k] = (len(r_k), int(r_k["a_fire"].sum()))
        print(f"  fiscal position {k}: fires on {by_pos[k][1]} of {by_pos[k][0]}")
    out["A"]["by_position"] = by_pos

    # ---- detector B
    b_rows = [(tk, s["rank"], b) for tk, s in stocks.items() for b in s.get("b", [])]
    b_off = [x for x in b_rows if x[2]["off"] > TOL]
    ytd_stocks = set(ytd["t"])
    ytd_fy = {(r.t) for r in ytd.itertuples()}
    print(f"\nDETECTOR B (four quarters of a completed fiscal year against the annual statement): "
          f"{len(b_rows)} stock-years testable on {len({x[0] for x in b_rows})} stocks; "
          f"{len(b_off)} differ by over {TOL:.0%}, {sum(1 for x in b_off if x[2]['off'] > FAR)} by over 10%")
    for tk, rk, b in sorted(b_off, key=lambda x: -x[2]["off"])[:25]:
        print(f"  #{rk!s:<5}{tk:<6}FY {b['fy']}  quarters {b['sum_q'] / 1e9:>8.3f}bn  annual {b['annual'] / 1e9:>8.3f}bn  "
              f"({b['off']:.1%})  has a filed-YTD column: {tk in ytd_stocks}")
    out["B"] = {"stock_years": len(b_rows), "stocks": len({x[0] for x in b_rows}), "off_2pct": len(b_off),
                "off_10pct": sum(1 for x in b_off if x[2]["off"] > FAR),
                "off_on_ytd_stocks": sorted({x[0] for x in b_off} & ytd_fy)}

    # ---- detector C
    c_rows = [(tk, s["rank"], s["c"]) for tk, s in stocks.items() if "c" in s]
    c_same = [x for x in c_rows if x[2]["ttm_date"] == x[2]["newest_q"] and x[2]["consecutive"]]
    c_off = [x for x in c_same if x[2]["off"] > TOL]
    ltm_ytd = set(ytd[ytd["in_ltm"]]["t"])
    print(f"\nDETECTOR C (the source's trailing twelve months against its four newest quarters): "
          f"{len(c_rows)} stocks carry a trailing figure, {len(c_same)} dated at the newest quarter with four consecutive quarters; "
          f"{len(c_off)} differ by over {TOL:.0%}")
    print(f"  stocks with a year-to-date column inside the four newest: {sorted(ltm_ytd)}; "
          f"of them C fires on {sorted({x[0] for x in c_off} & ltm_ytd)}")
    for tk, rk, c in sorted(c_off, key=lambda x: -x[2]["off"])[:25]:
        print(f"  #{rk!s:<5}{tk:<6}trailing {c['ttm'] / 1e9:>8.3f}bn  four quarters {c['sum4'] / 1e9:>8.3f}bn  ({c['off']:.1%})"
              f"  year-to-date column in the four: {tk in ltm_ytd}")
    out["C"] = {"with_trailing": len(c_rows), "comparable": len(c_same), "off_2pct": len(c_off),
                "ltm_ytd_stocks": sorted(ltm_ytd), "fires_on_ltm_ytd": sorted({x[0] for x in c_off} & ltm_ytd),
                "fires_elsewhere": sorted({x[0] for x in c_off} - ltm_ytd)}

    dest = Path(__file__).with_suffix(".json")
    dest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {dest.name}")
    return out


def impact(tk: str) -> dict:
    """What the published rank of ``tk`` rests on: the two scored metrics that read operating
    cash flow (fcf_yield, accruals) restated with the filed twelve months, each new sector
    percentile read off the published (value, percentile) pairs of its sector peers, and the
    composite moved by the published weights. An estimate - it does not rerun the engine, and
    every other stock is held still - checked by first rebuilding the two published category
    scores from the published percentiles."""
    import numpy as np
    import check_run_health as H
    import run_context
    import sec_crosscheck as X
    from sec_fundamentals import FACTS_PATH
    payload = H.load_payload(Path(run_context.latest_run_dir()))
    det, w = payload["stock_detail"], payload["weights"]
    d = det[tk]
    inp, raw, pct = d["inp"], d["raw"], d["pct"]
    facts = pd.read_parquet(FACTS_PATH)
    f = facts[(facts["ticker"] == tk) & (facts["concept"] == CONCEPT) & facts["form"].isin(X.FORMS)]
    filed, how = X.flow_ttm(f, pd.Timestamp(d["asof"]["cf"]))
    ocf = inp["operatingCashFlow"]
    ev = (ocf + inp["capex"]) / raw["fcf_yield"]          # the engine's own denominator, backed out
    new = {"fcf_yield": (filed + inp["capex"]) / ev,
           "accruals": (inp["netIncome"] - filed) / inp["totalAssets"]}
    assert abs((inp["netIncome"] - ocf) / inp["totalAssets"] - raw["accruals"]) < 1e-6, "accruals is not (NI - OCF) / assets"
    cats = {"fcf_yield": "valuation", "accruals": "quality"}
    for cat in set(cats.values()):
        mw = w["metric_weights"][cat]
        have = {m: x for m, x in mw.items() if x and pct.get(m) is not None}
        rebuilt = sum(x * pct[m] for m, x in have.items()) / sum(have.values())
        assert abs(rebuilt - d["cat_scores"][cat]) < 0.01, (cat, rebuilt, d["cat_scores"][cat])
    peers = [s for t, s in det.items() if s["sector"] == d["sector"] and t != tk]
    out = {"ticker": tk, "rank": d["rank"], "composite": d["composite"], "run_ocf": ocf, "filed_ocf": filed, "how": how}
    delta = 0.0
    for m, cat in cats.items():
        pairs = sorted((s["raw"][m], s["pct"][m]) for s in peers
                       if s["raw"].get(m) is not None and s["pct"].get(m) is not None)
        xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
        p_new = float(np.interp(new[m], xs, ys))
        mw = w["metric_weights"][cat]
        share = mw[m] / sum(x for k, x in mw.items() if x and pct.get(k) is not None)
        d_comp = w["factor_weights"][cat] / 100 * share * (p_new - pct[m])
        delta += d_comp
        out[m] = {"published": raw[m], "restated": new[m], "pct_published": pct[m], "pct_restated": p_new, "composite_points": d_comp}
        print(f"  {m:<10} {raw[m]:+.4f} -> {new[m]:+.4f}   sector percentile {pct[m]:.1f} -> {p_new:.1f}   composite {d_comp:+.2f}")
    comp = d["composite"] + delta
    others = sorted((r["Composite"] for r in payload["table_data"] if r["Ticker"] != tk), reverse=True)
    rank = 1 + sum(1 for c in others if c > comp)
    out.update({"composite_restated": comp, "rank_restated": rank})
    print(f"{tk}: operating cash flow {ocf / 1e9:.3f}bn published, {filed / 1e9:.3f}bn filed ({how}); "
          f"composite {d['composite']:.2f} -> {comp:.2f}; rank {d['rank']} -> about {rank}, every other stock held still")
    return out


def history(tickers: list[str]) -> list[dict]:
    """The same stocks on every published page git still holds: the run's operating cash flow
    against the filed twelve months to the date the page gave the statement, and the rank."""
    import subprocess
    import sec_crosscheck as X
    from sec_fundamentals import FACTS_PATH
    facts = pd.read_parquet(FACTS_PATH)
    facts = facts[(facts["concept"] == CONCEPT) & facts["form"].isin(X.FORMS)]
    log = subprocess.run(["git", "log", "--format=%H %ad %s", "--date=short", "--", "dashboard_data.js"],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout.splitlines()
    rows, seen = [], set()
    for line in log:
        sha, day, subject = line.split(" ", 2)
        if not subject.startswith("data:") or day in seen:
            continue
        seen.add(day)
        js = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"], capture_output=True, cwd=ROOT).stdout.decode("utf-8", "replace")
        try:
            payload = json.loads(js[js.index("{"): js.rindex("}") + 1])
        except ValueError:
            continue
        for tk in tickers:
            d = (payload.get("stock_detail") or {}).get(tk) or {}
            ocf, when = (d.get("inp") or {}).get("operatingCashFlow"), (d.get("asof") or {}).get("cf")
            if ocf is None or when is None:
                continue
            filed, _ = X.flow_ttm(facts[facts["ticker"] == tk], pd.Timestamp(when))
            rows.append({"day": day, "t": tk, "rank": d.get("rank"), "asof": when, "run": ocf, "filed": filed,
                         "off": None if filed is None else (ocf - filed) / max(abs(ocf), abs(filed))})
    rows.sort(key=lambda r: (r["t"], r["day"]))
    for tk in tickers:
        mine = [r for r in rows if r["t"] == tk]
        print(f"\n{tk}: {len(mine)} published pages carry its operating cash flow")
        last = None
        for r in mine:                       # print a row only when the figure or its date changes
            key = (r["asof"], r["run"])
            if key != last:
                off = "   n/a" if r["off"] is None else f"{r['off']:+.1%}"
                filed = float("nan") if r["filed"] is None else r["filed"] / 1e9
                print(f"  from {r['day']}  statement {r['asof']}  run {r['run'] / 1e9:>8.3f}bn  filed {filed:>8.3f}bn  {off}  rank {r['rank']}")
            last = key
        print("  rank by day: " + "  ".join(f"{r['day'][5:]}:{r['rank']}" for r in mine))
        judged = [r for r in mine if r["off"] is not None]
        print(f"  over 10% from the filing on {sum(1 for r in judged if abs(r['off']) > FAR)} of {len(judged)} pages; "
              f"first {mine[0]['day'] if mine else '-'}, last {mine[-1]['day'] if mine else '-'}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--impact", metavar="TICKER")
    ap.add_argument("tickers", nargs="*")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.tickers or universe())
    elif args.history:
        history(args.tickers or ["DXCM", "GM"])
    elif args.impact:
        res = impact(args.impact)
        Path(__file__).with_name(Path(__file__).stem + f"-impact-{args.impact}.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8")
    else:
        analyse()
