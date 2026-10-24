"""The filings of each member's *predecessor registrant*, for backtest v2 (2026-10-24).

`research/2026-10-20-free-survivorship-sources.md` section 6, fix 2. 773 of the 48,282
member-months in 2012-2019 are companies backtest v2 **has today** that it cannot score, because
`data/sec/pit/facts.parquet` is fetched by today's CIK and a holding-company reorganisation or
redomicile leaves the older filings under a different registrant number. Measured on
`cache/v2_census_presample.json`: Apache 96 months, BlackRock 96, Exxon 96, Disney 88, Cigna 85,
DuPont 68, Broadcom 49, Alphabet 63 across two classes, Medtronic 37, and fourteen smaller.

Those filings were public at the time - they are the company's own, under the name it then had -
so joining them is **not** look-ahead; omitting them is a hole in the point-in-time record. The
rule is general and takes no list of companies:

    for every ticker in today's universe, fetch every CIK the membership history
    (``ciks_by_month.json``) ever recorded for it that is not the CIK it carries today,
    and file those facts under today's ticker.

`pit_record` then selects by filed date as it always does: before the successor's first filing only
the predecessor's figures have been filed, after it the successor's restatement is the later filing
and wins. No special case, and no date to maintain.

Writes `data/sec/pit/facts_predecessor.parquet` (gitignored, same eight columns as
`facts.parquet`, keyed by **today's** ticker) and a summary beside this script. The concept list
and the form filter are imported from the 2026-10-09 census script so the two fetches cannot
drift apart.

    python research/measurements/2026-10-24-predecessor-facts.py            # fetch and write
    python research/measurements/2026-10-24-predecessor-facts.py --pairs    # list, fetch nothing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "sec" / "pit" / "facts_predecessor.parquet"
SUMMARY = Path(__file__).with_suffix(".json")
HERE = Path(__file__).parent


def _census_script():
    """The 2026-10-09 census script, whose filename cannot be imported normally."""
    p = HERE / "2026-10-09-xbrl-point-in-time-census.py"
    spec = importlib.util.spec_from_file_location("pit_census_2026_10_09", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def base_tickers() -> list[str]:
    """Today's scored universe - the tickers ``backtest_v2.Inputs.base`` is keyed by."""
    runs = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no runs/*/00_raw_fetch.parquet - cannot know today's universe")
    return sorted(set(pd.read_parquet(runs[-1])["Ticker"]))


def predecessor_pairs() -> list[dict]:
    """(ticker, predecessor CIK, the months the history gave it) for today's universe."""
    uh_dir = ROOT / "data" / "universe_history"
    by_month = json.loads((uh_dir / "ciks_by_month.json").read_text(encoding="utf-8"))
    today = {k: int(v) for k, v in
             json.loads((uh_dir / "ticker_ciks.json").read_text(encoding="utf-8"))["ciks"].items()}
    base = set(base_tickers())
    seen: dict[tuple[str, int], list[str]] = {}
    for month in sorted(by_month):
        for tk, cik in (by_month[month].get("ciks") or {}).items():
            if tk not in base:
                continue
            cik = int(cik)
            if today.get(tk) == cik:
                continue                      # the CIK already fetched under this ticker
            seen.setdefault((tk, cik), []).append(month)
    return [{"ticker": tk, "cik": cik, "months": len(ms), "first": ms[0], "last": ms[-1],
             "today_cik": today.get(tk)}
            for (tk, cik), ms in sorted(seen.items())]


def download(pairs: list[dict], log=print) -> pd.DataFrame:
    """One ``companyfacts`` request per predecessor CIK, same concepts and forms as today's."""
    import insider_activity as ia
    census = _census_script()
    ua = ia.user_agent()
    if not ua:
        raise SystemExit("SEC identity needed (data/sec/user_agent.txt)")
    e = ia.Edgar(ua)
    rows, got = [], {}
    t0 = time.time()
    for p in pairs:
        try:
            j = e.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{p['cik']:010d}.json").json()
        except Exception as ex:                                             # noqa: BLE001
            log(f"  {p['ticker']} CIK {p['cik']}: {type(ex).__name__}")
            got[f"{p['ticker']}:{p['cik']}"] = "fetch failed"
            continue
        gaap = (j.get("facts") or {}).get("us-gaap") or {}
        n0 = len(rows)
        for c in census.CONCEPTS:
            for unit, facts in ((gaap.get(c) or {}).get("units") or {}).items():
                if unit != "USD":
                    continue
                for f in facts:
                    if f.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                        continue
                    rows.append((p["ticker"], p["cik"], c, f.get("start"), f.get("end"),
                                 f.get("filed"), f.get("form"), f.get("fp"), f.get("val")))
        got[f"{p['ticker']}:{p['cik']}"] = len(rows) - n0
        log(f"  {p['ticker']:6s} CIK {p['cik']:<9d} {len(rows) - n0:6d} facts  "
            f"({j.get('entityName', '?')})")
    df = pd.DataFrame(rows, columns=["ticker", "cik", "concept", "start", "end", "filed",
                                     "form", "fp", "val"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    log(f"  {len(df)} facts in {time.time() - t0:.0f}s")
    return df, got


TENURE_LAG_DAYS = 120
"""How long after the last month the history named a registrant its filings may still be read:
one quarter plus the 10-Q deadline, so the final period it held the symbol is reported."""


def admit(pred: pd.DataFrame, today: pd.DataFrame,
          pairs: list[dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """Keep only the predecessor facts filed **before** today's registrant's first filing.

    Not every CIK the membership history gives a symbol is a predecessor, and the entity names the
    fetch prints make that plain: ``IR``'s earlier CIK is *Trane Technologies plc* and ``Q``'s is
    *IQVIA Holdings* - companies that trade today under their own symbols - while ``CB``'s is Chubb
    Corp and ``JCI``'s is Johnson Controls Inc, two companies absorbed by the registrant that now
    holds the symbol. Their figures must never reach a scored record.

    **The rule that separates them is the cut, not an overlap test.** The first draft of this
    function refused a pair whose predecessor kept filing after the successor started, on the
    theory that a real reorganisation hands over cleanly. Measured, that was wrong in both
    directions: Apache Corp (96 of the 773 months), EIDP and Dow Chemical all still file as
    debt-only subsidiaries of their own holding companies, so three genuine reorganisations were
    refused; and SanDisk Corp and Twenty-First Century Fox - different companies - were admitted,
    because they stopped filing before today's holder of the symbol was created.

    The cut is enough on its own, and provably so. ``pit_record`` selects a concept's value by
    filed date and the **later filing always wins**, so a row kept here can only be chosen for a
    month in which today's registrant had filed *nothing at all* - which is exactly the gap being
    filled, and exactly the state in which v2 scores the name not at all (``no_filings``) today.
    A row filed on the same day as today's registrant's first filing is dropped too: ``<``, not
    ``<=``, so a same-day handover (Alphabet and Google Inc both filed on 2015-10-29) resolves to
    the successor.

    **The second bound: tenure.** The cut alone would let a registrant that lost the symbol
    *decades* before today's holder was created still be read - ``Q``'s earlier CIK is IQVIA, which
    held the symbol for three months in 2017 and filed continuously until 2025. A registrant's
    filing can only inform a month in which it was that symbol's registrant, so rows filed more
    than ``TENURE_LAG_DAYS`` after the last month the membership history named it are dropped too.
    The effective bound is the earlier of the two.

    ``overlap_days`` is still recorded per pair, as a diagnostic that says which pairs are a
    reorganisation and which are a symbol changing hands - it no longer decides anything.
    """
    first_today = (today.dropna(subset=["filed"]).groupby("ticker")["filed"].min().to_dict())
    tenure = {(p["ticker"], int(p["cik"])): pd.Timestamp(p["last"]) + pd.Timedelta(days=TENURE_LAG_DAYS)
              for p in (pairs or [])}
    keep, verdicts = [], {}
    for (tk, cik), g in pred.groupby(["ticker", "cik"]):
        cut, ten = first_today.get(tk), tenure.get((tk, int(cik)))
        last_pred = g["filed"].max()
        bound = min([b for b in (cut, ten) if b is not None], default=None)
        sub = g if bound is None else g[g["filed"] < bound]
        verdicts[f"{tk}:{cik}"] = {
            "rows": int(len(g)), "kept": int(len(sub)),
            "cut": str(cut.date()) if cut is not None else None,
            "tenure_bound": str(ten.date()) if ten is not None else None,
            "bound_used": str(bound.date()) if bound is not None else None,
            "bound_is": (None if bound is None else
                         "tenure" if (cut is None or bound < cut) else "today's first filing"),
            "pred_last_filed": str(last_pred.date()) if pd.notna(last_pred) else None,
            "overlap_days": (int((last_pred - cut).days)
                             if (cut is not None and pd.notna(last_pred)) else None)}
        if len(sub):
            keep.append(sub)
    out = (pd.concat(keep) if keep else pred.iloc[:0]).drop(columns=["cik"])
    return out, verdicts


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true", help="list the pairs and fetch nothing")
    args = ap.parse_args()
    pairs = predecessor_pairs()
    for p in pairs:
        print(f"{p['ticker']:6s} predecessor CIK {p['cik']:<9d} today {p['today_cik']}  "
              f"{p['months']:3d} months {p['first']}..{p['last']}")
    out = {"generated": pd.Timestamp.utcnow().isoformat(), "pairs": pairs}
    if args.pairs:
        print(json.dumps({"n_pairs": len(pairs)}, indent=1))
        return out
    raw, got = download(pairs)
    today = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    df, verdicts = admit(raw, today, pairs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    out["facts_fetched_by_pair"] = got
    out["verdicts"] = verdicts
    out["pairs_with_rows_kept"] = sorted(k for k, v in verdicts.items() if v["kept"])
    out["pairs_fully_cut"] = sorted(k for k, v in verdicts.items() if not v["kept"])
    out["facts_fetched"] = int(len(raw))
    out["facts_kept"] = int(len(df))
    out["tickers"] = sorted(set(df["ticker"])) if len(df) else []
    out["earliest_filed"] = str(df["filed"].min().date()) if len(df) else None
    SUMMARY.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} of {len(raw)} facts kept for "
          f"{len(out['tickers'])} tickers; {len(out['pairs_fully_cut'])} pairs cut to nothing")
    return out


if __name__ == "__main__":
    main()
