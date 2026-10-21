"""How much of backtest v2's 2012-2019 survivorship gap can FREE sources close?

``plan/research-engine.md`` item 4(d), first half (2026-10-20). v2 scores only members that still
have a record today (or were renamed into one), so 2012's months hold 58% of their members and
2019's 78%. The owner uses no paid data (2026-10-15). SEC filings are free for departed companies;
prices are the gap. This measures, per departed member-month, which free route could supply a
price - it builds no monthly records and reruns nothing.

Stages (each cached; ``--force`` redoes it)::

    python research/measurements/2026-10-20-survivorship-free-sources.py departures
    python research/measurements/2026-10-20-survivorship-free-sources.py classify   # SEC + Wikipedia
    python research/measurements/2026-10-20-survivorship-free-sources.py yahoo      # route (a)
    python research/measurements/2026-10-20-survivorship-free-sources.py deals      # route (b), sample
    python research/measurements/2026-10-20-survivorship-free-sources.py report     # writes the JSON

Inputs (gitignored, main checkout): ``cache/v2_census_presample.json`` (who v2 could not score,
by month and reason), ``data/universe_history/*`` (membership and per-month CIKs). Downloads are
cached in ``data/survivorship/`` and never committed. The SEC User-Agent is read at run time from
``data/sec/user_agent.txt``; SEC requests are paced under its 10-per-second limit.

Rules this respects: nothing here is imported by production or by the research engine; no file in
``v2_experiments.RESULT_PATHS`` is touched; under CLAUDE.md rule 5 no number here decides anything.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import universe_history as uh  # noqa: E402


def _main_checkout(root: Path) -> Path:
    """The checkout that holds the gitignored caches (a worktree does not)."""
    if (root / "cache" / "v2_census_presample.json").exists():
        return root
    for p in root.parents:
        if (p / "cache" / "v2_census_presample.json").exists():
            return p
    raise SystemExit("cache/v2_census_presample.json not found in this checkout or any parent")


MAIN = _main_checkout(ROOT)
DATA = MAIN / "data" / "survivorship"
DATA.mkdir(parents=True, exist_ok=True)
OUT_JSON = HERE.with_suffix(".json")
CENSUS = MAIN / "cache" / "v2_census_presample.json"
BY_MONTH = MAIN / "data" / "universe_history" / "ciks_by_month.json"
FLAT = MAIN / "data" / "universe_history" / "ticker_ciks.json"
SURVIVOR_REASONS = ("no_current_record", "symbol_reused", "no_price")   # departed or reused
START, END = "2012-01-31", "2019-12-31"

# ----------------------------------------------------------------------------- SEC access

_LAST = [0.0]


def sec_get(url: str, binary: bool = False, tries: int = 4):
    import requests
    ua = (MAIN / "data" / "sec" / "user_agent.txt").read_text(encoding="utf-8").strip()
    for attempt in range(tries):
        wait = 0.125 - (time.time() - _LAST[0])          # <= 8 requests a second
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()
        r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
                         timeout=60)
        if r.status_code == 200:
            return r.content if binary else r.text
        if r.status_code == 404:
            return None
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"SEC {url}: status {r.status_code}")


def submissions(cik: int) -> dict | None:
    p = DATA / "submissions" / f"CIK{cik:010d}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    p.parent.mkdir(exist_ok=True)
    txt = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if txt is None:
        return None
    p.write_text(txt, encoding="utf-8")
    return json.loads(txt)


def filings_frame(sub: dict, lo=None, hi=None) -> pd.DataFrame:
    """The filings the submissions JSON lists; older pages are fetched (and cached) only when they
    overlap ``lo``..``hi`` (all of them when no window is given)."""
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):
        if lo is not None and (pd.Timestamp(extra.get("filingTo", "2100-01-01")) < pd.Timestamp(lo)
                               or pd.Timestamp(extra.get("filingFrom", "1900-01-01")) > pd.Timestamp(hi)):
            continue
        p = DATA / "submissions" / extra["name"]
        if not p.exists():
            txt = sec_get(f"https://data.sec.gov/submissions/{extra['name']}")
            if txt is None:
                continue
            p.write_text(txt, encoding="utf-8")
        frames.append(pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))))
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    return df


# ----------------------------------------------------------------------------- stage 1

def _snapshots() -> list[dict]:
    """The tracked membership history, 2012-01 .. 2026-08, as plain dicts."""
    return json.loads(uh.CACHE_PATH.read_text(encoding="utf-8"))["snapshots"]


def departures(force: bool = False) -> dict:
    """Every member-month 2012-2019 that v2 could not score, by the census's reason, plus each
    departed symbol's membership months over the full 2012-2026 history and its CIK(s)."""
    p = DATA / "departures.json"
    if p.exists() and not force:
        return json.loads(p.read_text(encoding="utf-8"))
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    by_month = json.loads(BY_MONTH.read_text(encoding="utf-8"))
    flat = {k: int(v) for k, v in json.loads(FLAT.read_text(encoding="utf-8"))["ciks"].items()}
    member_months = sum(c["members"] for c in census)
    rows, reasons = [], Counter()
    for c in census:
        for reason, tickers in c["not_scoreable"].items():
            reasons[reason] += len(tickers)
            for t in tickers:
                rows.append({"month": c["month"], "ticker": t, "reason": reason})
    # membership over the whole cached history, to know when each symbol left
    months_in = defaultdict(list)
    for snap in _snapshots():
        for t in snap["tickers"]:
            months_in[t].append(snap["as_of"])
    tickers = {}
    for r in rows:
        if r["reason"] not in SURVIVOR_REASONS:
            continue
        d = tickers.setdefault(r["ticker"], {"months": [], "reasons": Counter(), "ciks": Counter()})
        d["months"].append(r["month"])
        d["reasons"][r["reason"]] += 1
        cik = (by_month.get(r["month"], {}).get("ciks") or {}).get(r["ticker"])
        if cik:
            d["ciks"][int(cik)] += 1
    for t, d in tickers.items():
        d["months"] = sorted(d["months"])
        all_months = sorted(months_in.get(t, []))
        d["first_in_index"] = all_months[0] if all_months else None
        d["last_in_index"] = all_months[-1] if all_months else None
        d["reasons"] = dict(d["reasons"])
        if d["ciks"]:
            d["cik"] = d["ciks"].most_common(1)[0][0]
            d["cik_source"] = "ciks_by_month (the month's own revision)"
        elif t in flat:
            d["cik"] = flat[t]
            d["cik_source"] = "ticker_ciks.json (a later revision; check)"
        else:
            d["cik"] = None
            d["cik_source"] = None
        d["ciks"] = {str(k): v for k, v in d["ciks"].items()}
    out = {"member_months": member_months, "by_reason": dict(reasons),
           "rows": rows, "tickers": tickers}
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- stage 2

def _revision_names(revid: int) -> dict:
    """symbol -> security name from one Wikipedia revision of the constituents list (cached)."""
    p = DATA / "wiki_revisions" / f"{revid}.html"
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        p.write_text(uh.fetch_revision_html(revid), encoding="utf-8")
        time.sleep(0.5)
    html = p.read_text(encoding="utf-8")
    out = {}
    for f in pd.read_html(io.StringIO(html)):
        cols = [str(c).lower() for c in f.columns]
        sym = next((i for i, c in enumerate(cols) if c in ("ticker symbol", "symbol", "ticker")), None)
        name = next((i for i, c in enumerate(cols) if "security" in c or c == "company"), None)
        if sym is None or name is None or len(f) < 400:
            continue
        for s, n in zip(f.iloc[:, sym], f.iloc[:, name]):
            out[uh.normalize_ticker(str(s))] = str(n)
        break
    return out


def _norm_name(s: str) -> str:
    s = s.upper().replace("&AMP;", "&")
    s = re.sub(r"[^A-Z0-9& ]", " ", s)
    drop = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "PLC", "THE", "HOLDINGS",
            "HOLDING", "GROUP", "INCORPORATED", "LLC", "NV", "SA", "AG", "LP", "DE", "NEW"}
    return " ".join(w for w in s.split() if w not in drop)


def _cik_lookup() -> dict:
    """SEC's bulk name -> CIK list (every name any registrant has filed under)."""
    p = DATA / "cik-lookup-data.txt"
    if not p.exists():
        txt = sec_get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", binary=True)
        p.write_bytes(txt)
    names = defaultdict(set)
    for line in p.read_bytes().decode("latin-1").splitlines():
        m = re.match(r"^(.*):(\d{10}):\s*$", line)
        if m:
            names[_norm_name(m.group(1))].add(int(m.group(2)))
    return names


def resolve_missing_ciks(dep: dict) -> dict:
    """Name-match a CIK for the symbols whose months predate the revisions' CIK column (2014-05).
    The match must have filed a 10-K during the symbol's membership years, or it is rejected."""
    missing = {t: d for t, d in dep["tickers"].items() if not d.get("cik")}
    if not missing:
        return {}
    snaps = {s["as_of"]: s for s in _snapshots()}
    lookup = _cik_lookup()
    resolved = {}
    for t, d in sorted(missing.items()):
        snap = snaps[d["months"][-1]]
        name = _revision_names(snap["revid"]).get(t)
        if not name:
            resolved[t] = {"name": None, "cik": None, "why": "symbol not in its own revision's table"}
            continue
        key = _norm_name(name)
        cands = set(lookup.get(key, set()))
        if not cands:
            # Wikipedia writes "Penney (J.C.)", "Abercrombie & Fitch Company A"; SEC writes
            # "PENNEY J C CO INC", "ABERCROMBIE & FITCH CO /DE/". Rank every SEC name sharing the
            # first word by string similarity and check the closest 25.
            import difflib
            w = key.split()
            pool = [(difflib.SequenceMatcher(None, key, k2).ratio(), c)
                    for k2, ciks in lookup.items() if w and k2.split()[:1] == w[:1] for c in ciks]
            cands = {c for _, c in sorted(pool, reverse=True)[:25]}
        good = []
        y0, y1 = int(d["months"][0][:4]), int(d["months"][-1][:4])
        for cik in sorted(cands)[:25]:
            sub = submissions(cik)
            if not sub:
                continue
            f = filings_frame(sub, f"{y0}-01-01", f"{y1 + 1}-12-31")
            if f.empty:
                continue
            k = f[(f["form"] == "10-K") & (f["filingDate"].dt.year >= y0) &
                  (f["filingDate"].dt.year <= y1 + 1)]
            if len(k):
                import difflib
                sim = difflib.SequenceMatcher(None, key, _norm_name(sub.get("name") or "")).ratio()
                formers = [_norm_name(x.get("name") or "") for x in sub.get("formerNames", [])]
                sim = max([sim] + [difflib.SequenceMatcher(None, key, x).ratio() for x in formers])
                good.append((cik, sub.get("name"), round(sim, 3)))
        good.sort(key=lambda g: -g[2])
        pick = None
        if len(good) == 1 and good[0][2] >= 0.5:
            pick, why = good[0][0], "unique name match with 10-Ks in the membership years"
        elif len(good) > 1 and good[0][2] >= 0.6 and good[0][2] - good[1][2] >= 0.15:
            pick, why = good[0][0], f"closest of {len(good)} name matches with 10-Ks (similarity margin >= 0.15)"
        else:
            why = f"{len(good)} candidates with 10-Ks, none clearly closest - left unresolved"
        resolved[t] = {"name": name, "cik": pick, "candidates": good[:5], "why": why}
    return resolved


SEC_FORMS_ACQ = {"DEFM14A", "PREM14A", "SC 14D9", "SC TO-T", "SC 13E3", "DEFM14C", "PREM14C",
                 "SC 14D9/A", "SC TO-T/A"}
SEC_FORMS_DEREG = {"15-12B", "15-12G", "15-15D", "25-NSE", "25", "15-12B/A"}


def _wiki_changes() -> pd.DataFrame:
    p = DATA / "wiki_changes.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exit_reasons", ROOT / "research" / "measurements" / "2026-09-30-exit-reasons.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = mod.load_changes()
    df = df.astype(str)
    df.to_parquet(p)
    return df


def _wiki_reasons() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exit_reasons", ROOT / "research" / "measurements" / "2026-09-30-exit-reasons.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = _wiki_changes()
    cols = list(df.columns)
    date_col = next(c for c in cols if "date" in c.lower())
    reason_col = next(c for c in cols if c.lower().strip() == "reason" or c.lower().endswith("reason"))
    rem_cols = [c for c in cols if "removed" in c.lower()]
    tick_col = next((c for c in rem_cols if "ticker" in c.lower() or "symbol" in c.lower()), rem_cols[0])
    out = defaultdict(list)
    for _, row in df.iterrows():
        t = uh.normalize_ticker(str(row[tick_col]))
        if t and t != "NAN":
            out[t].append({"date": str(row[date_col]), "reason": str(row[reason_col]),
                           "class": mod.classify(str(row[reason_col]))})
    return out


def _sec_fate(cik: int, last_in_index: str) -> dict:
    sub = submissions(cik)
    if not sub:
        return {"sec": "no submissions"}
    left = pd.Timestamp(last_in_index)
    # every filing from 18 months before the index exit onward: the company's END may come years
    # after it left the index (a demoted member acquired later), and that end is what decides
    # whether a free price source still serves its history
    f = filings_frame(sub, left - pd.Timedelta(days=550), pd.Timestamp("2100-01-01"))
    f = f[f["filingDate"] >= left - pd.Timedelta(days=550)]
    if not f["form"].isin(["10-K", "10-Q"]).any():
        # the registrant stopped reporting before the symbol left the index (the symbol passed to
        # a successor registrant: Mylan Inc -> Mylan N.V.); read its whole history instead
        f = filings_frame(sub)
        left = f.loc[f["form"].isin(["10-K", "10-Q"]), "filingDate"].max() if len(f) else left
        left = left if pd.notna(left) else pd.Timestamp(last_in_index)
    periodic = f[f["form"].isin(["10-K", "10-Q", "10-K405", "20-F", "40-F"])]
    last_periodic = periodic["filingDate"].max() if len(periodic) else None
    end = last_periodic if last_periodic is not None else left
    near = f[(f["filingDate"] >= min(left, end) - pd.Timedelta(days=550)) &
             (f["filingDate"] <= end + pd.Timedelta(days=400))]
    items = " ".join(str(x) for x in near.get("items", pd.Series(dtype=str)).fillna(""))
    forms = set(near["form"])
    return {
        "sec_name": sub.get("name"),
        "former_names": [x.get("name") for x in sub.get("formerNames", [])],
        "sec_tickers_now": sub.get("tickers", []),
        "last_periodic_report": str(last_periodic.date()) if last_periodic is not None else None,
        "acq_forms_near_exit": sorted(forms & SEC_FORMS_ACQ),
        "dereg_forms_near_exit": sorted(forms & SEC_FORMS_DEREG),
        "item_2_01_near_exit": "2.01" in items,
        "item_1_03_near_exit": "1.03" in items,
        "item_3_01_near_exit": "3.01" in items,
    }


def classify_fate(t: str, d: dict, sec: dict, wiki: list) -> tuple[str, str]:
    """(cause, evidence). Cause is the company's fate that keeps it out of v2's records."""
    if d["reasons"].get("no_price") or d["reasons"].get("symbol_reused"):
        return ("symbol reused by another company",
                "census: today's holder of the symbol is a different registrant")
    alive = sec.get("last_periodic_report") and sec["last_periodic_report"] >= "2026-05-01"
    w = wiki[-1]["class"] if wiki else None
    today = {uh.normalize_ticker(x["Ticker"]) for x in
             json.loads((ROOT / "sp500_tickers.json").read_text(encoding="utf-8"))}
    now = {uh.normalize_ticker(x) for x in sec.get("sec_tickers_now") or []}
    if now & today:
        return ("in today's index under another symbol - not carried",
                f"SEC lists {sorted(now)}; today's universe has {sorted(now & today)}")
    if alive:
        return ("still filing - left the index (moved/demoted) or renamed",
                f"last periodic report {sec['last_periodic_report']}; wiki: {w}")
    if sec.get("item_1_03_near_exit") or w == "bankruptcy/receivership":
        return ("bankrupt / distressed delisting", f"8-K 1.03={sec.get('item_1_03_near_exit')}; wiki: {w}")
    if (sec.get("acq_forms_near_exit") or sec.get("item_2_01_near_exit")
            or w == "acquired / merged / taken private"):
        return ("acquired / merged / taken private",
                f"forms {sec.get('acq_forms_near_exit')}; 8-K 2.01={sec.get('item_2_01_near_exit')}; wiki: {w}")
    if w == "spin-off / restructuring":
        return ("spun off / restructured", f"wiki: {w}")
    if sec.get("last_periodic_report"):
        return ("stopped filing - cause not parsed",
                f"last periodic {sec['last_periodic_report']}; dereg {sec.get('dereg_forms_near_exit')}; wiki: {w}")
    return ("unresolved", "no CIK / no filings")


def classify(force: bool = False) -> dict:
    p = DATA / "classified.json"
    if p.exists() and not force:
        return json.loads(p.read_text(encoding="utf-8"))
    dep = departures()
    names = resolve_missing_ciks(dep)
    wiki = _wiki_reasons()
    out = {}
    for t, d in sorted(dep["tickers"].items()):
        cik = d.get("cik") or (names.get(t) or {}).get("cik")
        src = d.get("cik_source") or ((names.get(t) or {}).get("why") if cik else None)
        sec = _sec_fate(cik, d["last_in_index"]) if cik else {}
        cause, ev = classify_fate(t, d, sec, wiki.get(t, []))
        out[t] = {"cik": cik, "cik_source": src, "name_from_wiki": (names.get(t) or {}).get("name"),
                  "name_match": names.get(t), "first_in_index": d["first_in_index"],
                  "last_in_index": d["last_in_index"], "n_departed_months_2012_2019": len(d["months"]),
                  "census_reasons": d["reasons"], "wiki_removals": wiki.get(t, []), **sec,
                  "cause": cause, "evidence": ev}
        print(f"{t:6s} {str(cik):>8s} {cause[:40]:40s} {ev[:70]}", flush=True)
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- stage 3: route (a)

def yahoo(force: bool = False) -> pd.DataFrame:
    """Daily history 2010-11-01 .. 2020-02-29 for every departed symbol, as yfinance serves it."""
    p = DATA / "yahoo_daily.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    import yfinance as yf
    dep = departures()
    tickers = sorted(dep["tickers"])
    frames = []
    for i in range(0, len(tickers), 40):
        chunk = [t.replace(".", "-") for t in tickers[i:i + 40]]
        raw = yf.download(chunk, start="2010-11-01", end="2020-03-01", auto_adjust=False,
                          actions=True, group_by="ticker", progress=False, threads=True)
        for t in chunk:
            try:
                g = raw[t].dropna(how="all")
            except KeyError:
                continue
            g = g.dropna(subset=["Close"])
            if g.empty:
                continue
            g = g.reset_index().rename(columns={"Date": "date"})
            g["ticker"] = t
            frames.append(g)
        print(f"yahoo: {min(i + 40, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(p)
    return df


def yahoo_alt(force: bool = False) -> pd.DataFrame:
    """Second pass of route (a): for a departed symbol Yahoo does not serve, the symbol the SAME
    registrant trades under today (SEC submissions ``tickers``) - e.g. a company that changed its
    symbol after leaving the index. Plus two controls (AAPL, MSFT) that must answer, so an empty
    reply is evidence of absence and not of a blocked host. Stored under the departed symbol."""
    p = DATA / "yahoo_alt.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    import yfinance as yf
    have = set(yahoo()["ticker"])
    cls = classify()
    alt = {}
    for t, c in cls.items():
        if t.replace(".", "-") in have:
            continue
        for s in c.get("sec_tickers_now") or []:
            s = s.replace(".", "-")
            if s != t.replace(".", "-"):
                alt[t] = s
                break
    want = sorted(set(alt.values()) | {"AAPL", "MSFT"})
    raw = yf.download(want, start="2010-11-01", end="2020-03-01", auto_adjust=False, actions=True,
                      group_by="ticker", progress=False, threads=True)
    frames = []
    for t, s in list(alt.items()) + [("AAPL", "AAPL"), ("MSFT", "MSFT")]:
        try:
            g = raw[s].dropna(subset=["Close"])
        except KeyError:
            continue
        if g.empty:
            continue
        g = g.reset_index().rename(columns={"Date": "date"})
        g["ticker"] = t.replace(".", "-")
        g["via_symbol"] = s
        frames.append(g)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(p)
    print("alt symbols tried:", alt)
    return df


def _public_float_points(cik: int) -> list[dict]:
    """(date, public float USD) and the cover-page share count from the company's own XBRL
    (``dei``), via companyconcept - the identity check for a Yahoo series."""
    out = []
    for tax, concept in (("dei", "EntityPublicFloat"), ("dei", "EntityCommonStockSharesOutstanding"),
                         ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")):
        pth = DATA / "concepts" / f"{cik}_{concept}.json"
        if not pth.exists():
            pth.parent.mkdir(exist_ok=True)
            txt = sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{tax}/{concept}.json")
            pth.write_text(txt or "{}", encoding="utf-8")
        j = json.loads(pth.read_text(encoding="utf-8"))
        for u, facts in (j.get("units") or {}).items():
            for f in facts:
                out.append({"concept": concept, "end": f["end"], "val": f["val"], "form": f.get("form"),
                            "filed": f.get("filed")})
    return out


def _split_history(symbol: str) -> pd.Series:
    """Every split Yahoo records for ``symbol``, to today (cached)."""
    p = DATA / "yahoo_splits.json"
    cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if symbol not in cache:
        import yfinance as yf
        try:
            s = yf.Ticker(symbol).splits
            cache[symbol] = {str(pd.Timestamp(k).tz_localize(None).date()): float(v) for k, v in s.items()}
        except Exception:  # noqa: BLE001 - an unanswered symbol has no recorded split
            cache[symbol] = {}
        p.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    s = pd.Series(cache[symbol], dtype=float)
    s.index = pd.to_datetime(s.index)
    return s[s > 0]


def identity_check(cik: int, g: pd.DataFrame) -> dict:
    """Is the Yahoo series this registrant? Compare the series' as-traded close on each public-float
    date with float / shares outstanding (nearest cover-page count within 300 days after). Float
    excludes affiliates' shares, so the ratio should sit a little under 1; a different company's
    prices land anywhere."""
    pts = pd.DataFrame(_public_float_points(cik))
    if pts.empty:
        return {"checked": 0, "verdict": "no dei facts"}
    fl = pts[pts.concept == "EntityPublicFloat"].drop_duplicates("end")
    sh = pts[pts.concept == "EntityCommonStockSharesOutstanding"].copy()
    sh["end"] = pd.to_datetime(sh["end"])
    wa = pts[pts.concept == "WeightedAverageNumberOfSharesOutstandingBasic"].copy()
    wa["end"] = pd.to_datetime(wa["end"])
    via = str(g["via_symbol"].iloc[0]) if "via_symbol" in g else None
    g = g.set_index("date").sort_index()
    # the FULL split history to today: yfinance's closes are adjusted for every later split,
    # including those after this download's window (GME 4:1 in 2022, NBR 1:50 in 2020)
    splits = _split_history(via) if via else pd.Series(dtype=float)
    ratios = []
    for _, r in fl.iterrows():
        d = pd.Timestamp(r["end"])
        if d < g.index[0] or d > g.index[-1] + pd.Timedelta(days=5):
            continue
        s = sh[(sh["end"] >= d) & (sh["end"] <= d + pd.Timedelta(days=300)) & (sh["val"] > 0)]
        # several share classes report separately on one cover date - add them up
        s = s[s["end"] == s["end"].min()].drop_duplicates("val") if len(s) else s
        if s.empty:   # no cover-page count: the weighted average for the period ending nearest
            w = wa[(wa["val"] > 0) & ((wa["end"] - d).abs() <= pd.Timedelta(days=120))]
            if len(w):
                s = w.assign(gap=(w["end"] - d).abs()).sort_values("gap").head(1)
        if s.empty or not r["val"]:
            continue
        close = g["Close"].asof(d)
        after = float(splits[splits.index > d].prod()) if len(splits) else 1.0
        traded = close * after
        implied = float(r["val"]) / float(s["val"].sum())
        ratios.append(implied / traded if traded else None)
    ratios = [x for x in ratios if x]
    if not ratios:
        return {"checked": 0, "verdict": "no overlapping float dates"}
    ok = [0.45 <= x <= 1.25 for x in ratios]
    return {"checked": len(ratios), "median_ratio": float(pd.Series(ratios).median()),
            "share_ok": sum(ok) / len(ok),
            "verdict": "same company" if sum(ok) / len(ok) >= 0.75 else "MISMATCH"}


def month_end_close(g: pd.DataFrame, me: str, tol_days: int = 5) -> float | None:
    s = g.set_index("date")["Adj Close"].sort_index()
    d = pd.Timestamp(me)
    w = s[(s.index <= d) & (s.index > d - pd.Timedelta(days=tol_days + 2))]
    return float(w.iloc[-1]) if len(w) else None


def yahoo_coverage() -> dict:
    """Per departed member-month: does route (a) give (i) the month-end close and the next one
    (a monthly return), and (ii) 13 months of daily history to the month-end (the momentum and
    risk inputs v2 scores with)? Only series that pass the identity check count."""
    dep, cls = departures(), classify()
    yd = yahoo()
    alt = yahoo_alt()
    controls = sorted(set(alt.get("ticker", pd.Series(dtype=str))) & {"AAPL", "MSFT"})
    alt = alt[~alt["ticker"].isin(["AAPL", "MSFT"])] if len(alt) else alt
    yd = pd.concat([yd.assign(via_symbol=yd["ticker"]), alt], ignore_index=True)
    yd["date"] = pd.to_datetime(yd["date"]).dt.tz_localize(None)
    series = {t: g.sort_values("date") for t, g in yd.groupby("ticker")}
    per_ticker = {"_controls_answered": controls}
    for t, d in dep["tickers"].items():
        g = series.get(t.replace(".", "-"))
        c = cls[t]
        if g is None or g.empty:
            per_ticker[t] = {"rows": 0, "identity": None}
            continue
        idc = identity_check(c["cik"], g) if c.get("cik") else {"checked": 0, "verdict": "no CIK"}
        reused = bool(d["reasons"].get("symbol_reused") or d["reasons"].get("no_price"))
        if reused and str(g["via_symbol"].iloc[0]) == t.replace(".", "-"):
            # the census established that today's holder of this symbol is another registrant, so
            # Yahoo's series under it is that company's history. A price-level check cannot always
            # tell (CB: Chubb Corp and ACE traded at similar prices, ratio 0.95), so it is refused
            # outright - including AA, where Yahoo appears to have spliced Alcoa Inc.'s history in.
            idc = {**idc, "float_verdict": idc.get("verdict"), "verdict": "REFUSED - symbol reused"}
        elif idc.get("verdict") != "same company":
            # second tier: the series is under the symbol this registrant files under TODAY, and it
            # runs unbroken from 13 months before the first departed month - a reused symbol cannot
            # do both. Catches a company whose own XBRL has no usable float/share pair (VFC) and a
            # controlled company whose float is far below its market value (COTY: JAB's stake).
            via = str(g["via_symbol"].iloc[0])
            now = {s.replace(".", "-") for s in (c.get("sec_tickers_now") or [])}
            start_needed = pd.Timestamp(d["months"][0]) - pd.DateOffset(months=13)
            gaps = g["date"].diff().dt.days.max() if len(g) > 1 else 999
            if via in now and g["date"].iloc[0] <= start_needed + pd.Timedelta(days=7) and gaps <= 10:
                idc = {**idc, "float_verdict": idc.get("verdict"), "verdict": "same company",
                       "tier": "continuity (symbol is the registrant's current SEC ticker; no gap)"}
        else:
            idc = {**idc, "tier": "float (SEC public float / shares vs Yahoo close)"}
        per_ticker[t] = {"rows": int(len(g)), "first": str(g["date"].iloc[0].date()),
                         "last": str(g["date"].iloc[-1].date()), "identity": idc,
                         "via_symbol": str(g["via_symbol"].iloc[0])}
    month_rows = []
    for r in dep["rows"]:
        if r["reason"] not in SURVIVOR_REASONS:
            continue
        t, me = r["ticker"], r["month"]
        pt = per_ticker.get(t, {})
        g = series.get(t.replace(".", "-"))
        usable = g is not None and (pt.get("identity") or {}).get("verdict") == "same company"
        nxt = (pd.Timestamp(me) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        p0 = month_end_close(g, me) if usable else None
        p1 = month_end_close(g, nxt) if usable else None
        hist = False
        if usable and p0 is not None:
            s = g[(g["date"] <= pd.Timestamp(me)) &
                  (g["date"] > pd.Timestamp(me) - pd.DateOffset(months=13))]
            hist = len(s) >= 240
        month_rows.append({"month": me, "ticker": t, "cause": cls[t]["cause"],
                           "return": p0 is not None and p1 is not None, "history13": hist,
                           "price_only": p0 is not None})
    return {"per_ticker": per_ticker, "month_rows": month_rows}


# ----------------------------------------------------------------------------- stage 4: route (b)

DEAL_RE = [
    re.compile(r"(?:right to receive|converted into the right to receive|entitled to receive)[^.$]{0,160}?"
               r"\$\s?(\d{1,4}(?:\.\d{1,4})?)\s*(?:per share)?\s*in cash", re.I),
    re.compile(r"\$\s?(\d{1,4}(?:\.\d{1,4})?)\s+per share\s+in cash", re.I),
    re.compile(r"\$\s?(\d{1,4}(?:\.\d{1,4})?)\s+in cash(?:,)?\s+(?:without interest\s+)?(?:per share|for each)", re.I),
]
STOCK_RE = re.compile(r"(\d{1,2}\.\d{2,5})\s+(?:of a\s+)?(?:share|shares)\s+of\s+[A-Z][\w\s.,&]{2,60}?common stock", re.I)
PRE_ANN_RE = re.compile(r"(?:closing (?:sale )?price|closing price per share)[^.]{0,220}?"
                        r"(?:last (?:full )?trading day|day (?:prior|before))[^.]{0,200}?\$\s?(\d{1,4}\.\d{2})", re.I)


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = raw.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&").replace("&#36;", "$")
    return re.sub(r"\s+", " ", raw)


def deal_terms(cik: int, last_in_index: str) -> dict:
    """Route (b): the consideration per share the company's own merger filings state. Reads the
    target's DEFM14A / SC 14D9 / 8-K item 2.01 nearest its exit (cached), and also looks for the
    'closing price on the last trading day before announcement' that proxies state."""
    sub = submissions(cik)
    if not sub:
        return {"found": False, "why": "no submissions"}
    left = pd.Timestamp(last_in_index)
    f = filings_frame(sub, left - pd.Timedelta(days=400), left + pd.Timedelta(days=120))
    near = f[(f["filingDate"] >= left - pd.Timedelta(days=400)) &
             (f["filingDate"] <= left + pd.Timedelta(days=120))].copy()
    pref = {"DEFM14A": 0, "SC 14D9": 1, "DEFM14C": 2, "8-K": 3}
    near = near[near["form"].isin(pref)]
    if "items" in near:
        near = near[(near["form"] != "8-K") | near["items"].fillna("").str.contains("2.01")]
    near["p"] = near["form"].map(pref)
    near = near.sort_values(["p", "filingDate"])
    tried = []
    for _, r in near.head(3).iterrows():
        acc = r["accessionNumber"].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{r['primaryDocument']}"
        pth = DATA / "deal_docs" / f"{cik}_{acc}.txt"
        if not pth.exists():
            pth.parent.mkdir(exist_ok=True)
            raw = sec_get(url, binary=True)
            pth.write_text(_strip_html((raw or b"").decode("latin-1"))[:3_000_000], encoding="utf-8")
        text = pth.read_text(encoding="utf-8")
        cash = None
        for rx in DEAL_RE:
            m = rx.search(text)
            if m:
                cash = float(m.group(1))
                break
        stock = STOCK_RE.search(text)
        pre = PRE_ANN_RE.search(text)
        tried.append({"form": r["form"], "filed": str(r["filingDate"].date()), "url": url,
                      "cash_per_share": cash,
                      "stock_ratio": float(stock.group(1)) if stock else None,
                      "pre_announcement_close": float(pre.group(1)) if pre else None})
        if cash is not None:
            break
    hit = next((x for x in tried if x["cash_per_share"] is not None), None)
    return {"found": hit is not None, "doc": hit or (tried[0] if tried else None), "tried": len(tried)}


def deals(force: bool = False, n: int = 45) -> dict:
    p = DATA / "deals.json"
    if p.exists() and not force:
        return json.loads(p.read_text(encoding="utf-8"))
    cls = classify()
    import random
    acq = sorted(t for t, c in cls.items() if c["cause"].startswith("acquired") and c.get("cik"))
    random.Random(20261020).shuffle(acq)          # a fixed-seed sample when n < population
    out = {}
    for t in acq[:n]:
        try:
            out[t] = deal_terms(cls[t]["cik"], cls[t]["last_in_index"])
        except Exception as e:  # noqa: BLE001 - recorded, not hidden
            out[t] = {"found": False, "why": f"error {e!r}"}
        print(t, out[t].get("found"), (out[t].get("doc") or {}).get("cash_per_share"), flush=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- route (c): 10-K Item 5

ITEM5_RE = re.compile(r"market for (?:the )?registrant", re.I)


def tenk_item5(cik: int, year: int) -> dict:
    """Does the company's 10-K filed in ``year`` carry a quarterly price table in Item 5? Looks for
    'High' and 'Low' with quarter labels within 6,000 characters after the Item 5 heading (the
    second occurrence - the first is the table of contents)."""
    sub = submissions(cik)
    if not sub:
        return {"found": None, "why": "no submissions"}
    f = filings_frame(sub, f"{year}-01-01", f"{year}-12-31")
    k = f[(f["form"] == "10-K") & (f["filingDate"].dt.year == year)]
    if k.empty:
        return {"found": None, "why": f"no 10-K filed in {year}"}
    r = k.iloc[0]
    acc = r["accessionNumber"].replace("-", "")
    pth = DATA / "tenk_docs" / f"{cik}_{acc}.txt"
    if not pth.exists():
        pth.parent.mkdir(exist_ok=True)
        raw = sec_get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{r['primaryDocument']}",
                      binary=True)
        pth.write_text(_strip_html((raw or b"").decode("latin-1")), encoding="utf-8")
    text = pth.read_text(encoding="utf-8")
    hits = [m.start() for m in ITEM5_RE.finditer(text)]
    table = False
    for h in hits[1:] or hits:
        win = text[h:h + 6000]
        if (re.search(r"\bhigh\b", win, re.I) and re.search(r"\blow\b", win, re.I)
                and re.search(r"first quarter|fourth quarter|\bQ[14]\b|quarter ended", win, re.I)):
            table = True
            break
    incorporated = bool(re.search(r"incorporated (?:herein )?by reference", text[hits[-1]:hits[-1] + 3000], re.I)) if hits else False
    return {"found": table, "filed": str(r["filingDate"].date()), "item5_mentions": len(hits),
            "incorporated_by_reference_nearby": incorporated}


def tenk_sample() -> dict:
    p = DATA / "tenk.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    cls = classify()
    picks = []
    for cause in sorted({c["cause"] for c in cls.values()}):
        ts = sorted(t for t, c in cls.items() if c["cause"] == cause and c.get("cik"))
        picks += ts[: 4]
    out = {}
    for t in picks:
        c = cls[t]
        y_last = int(c["last_in_index"][:4])
        res = {}
        for y in sorted({2013, min(max(y_last, 2013), 2018), 2019, 2020}):
            if y > y_last + 1:
                continue
            res[str(y)] = tenk_item5(c["cik"], y)
        out[t] = {"cause": c["cause"], "years": res}
        print(t, {y: v.get("found") for y, v in res.items()}, flush=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- route (d): SEC 13F

def thirteenf_prices(zip_name: str, period: str) -> pd.DataFrame:
    """Quarter-end prices implied by institutional holders' 13F reports (SEC Form 13F data sets,
    public, from 2013 Q3): VALUE (thousands of dollars before 2023) / shares, per CUSIP, median
    over filers. Quarter-end only - a 3-month return, never a monthly one."""
    import zipfile
    zp = DATA / "13f" / zip_name
    if not zp.exists():
        zp.parent.mkdir(exist_ok=True)
        zp.write_bytes(sec_get("https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
                               + zip_name, binary=True))
    with zipfile.ZipFile(zp) as z:
        names = {n.split("/")[-1].upper(): n for n in z.namelist()}
        sub = pd.read_csv(z.open(names["SUBMISSION.TSV"]), sep="\t", dtype=str)
        cov = pd.read_csv(z.open(names["COVERPAGE.TSV"]), sep="\t", dtype=str)
        info = pd.read_csv(z.open(names["INFOTABLE.TSV"]), sep="\t", dtype=str,
                           usecols=["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP",
                                    "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"])
    per = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    keep = set(sub.loc[per == pd.Timestamp(period), "ACCESSION_NUMBER"])
    if "ISAMENDMENT" in cov:
        keep -= set(cov.loc[cov["ISAMENDMENT"].fillna("N").str.upper().str.startswith("Y"), "ACCESSION_NUMBER"])
    info = info[info["ACCESSION_NUMBER"].isin(keep) & (info["SSHPRNAMTTYPE"] == "SH")
                & info["PUTCALL"].isna()]
    info = info.assign(value=pd.to_numeric(info["VALUE"], errors="coerce") * 1000.0,
                       shares=pd.to_numeric(info["SSHPRNAMT"], errors="coerce"))
    info = info[(info["shares"] >= 1000) & (info["value"] > 0)]
    info["px"] = info["value"] / info["shares"]
    g = info.groupby("CUSIP").agg(px=("px", "median"), filers=("px", "size"),
                                  name=("NAMEOFISSUER", lambda s: s.mode().iloc[0]))
    return g.reset_index()


def thirteenf(period: str = "2015-12-31", zip_name: str = "2016q1_form13f.zip") -> dict:
    """Match each departed company that was a member at ``period`` to a 13F CUSIP by issuer name
    (SEC conformed name vs the filers' NAMEOFISSUER), then compare the implied price with Yahoo's
    close where Yahoo has the series - the accuracy check - and count coverage by cause."""
    p = DATA / f"13f_{period}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    cls, dep = classify(), departures()
    px = thirteenf_prices(zip_name, period)
    px = px[px["filers"] >= 5]
    px["key"] = px["name"].map(_norm_name)
    by_key = {k: g.sort_values("filers", ascending=False).iloc[0] for k, g in px.groupby("key")}
    yd = yahoo()
    yd["date"] = pd.to_datetime(yd["date"]).dt.tz_localize(None)
    month = period
    out = {}
    for t, d in dep["tickers"].items():
        if month not in d["months"]:
            continue
        c = cls[t]
        names = [c.get("sec_name")] + list(c.get("former_names") or []) + [c.get("name_from_wiki")]
        hit = None
        for n in names:
            if n and _norm_name(n) in by_key:
                hit = by_key[_norm_name(n)]
                break
        g = yd[yd["ticker"] == t.replace(".", "-")]
        yclose = None
        if len(g):
            s = g.set_index("date")["Close"].sort_index()
            w = s[(s.index <= pd.Timestamp(period)) & (s.index > pd.Timestamp(period) - pd.Timedelta(days=7))]
            yclose = float(w.iloc[-1]) if len(w) else None
        out[t] = {"cause": c["cause"], "matched": hit is not None,
                  "cusip": hit["CUSIP"] if hit is not None else None,
                  "filers": int(hit["filers"]) if hit is not None else 0,
                  "px_13f": float(hit["px"]) if hit is not None else None, "px_yahoo_close": yclose}
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- report

def _tenk_summary() -> dict:
    p = DATA / "tenk.json"
    if not p.exists():
        return {}
    t = json.loads(p.read_text(encoding="utf-8"))
    by_year = defaultdict(Counter)
    for v in t.values():
        for y, r in v["years"].items():
            by_year[y][str(r.get("found"))] += 1
    return {"names": len(t), "table_found_by_filing_year": {y: dict(c) for y, c in sorted(by_year.items())},
            "detail": t}


def _13f_summary() -> dict:
    p = DATA / "13f_2015-12-31.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text(encoding="utf-8"))
    both = [(t, v["px_13f"] / v["px_yahoo_close"]) for t, v in r.items()
            if v["px_13f"] and v["px_yahoo_close"]]
    by_cause = defaultdict(Counter)
    for v in r.values():
        by_cause[v["cause"]]["matched" if v["matched"] else "unmatched"] += 1
    return {"departed_members_at_2015_12_31": len(r),
            "matched_by_issuer_name": sum(v["matched"] for v in r.values()),
            "by_cause": {k: dict(c) for k, c in by_cause.items()},
            "compared_with_yahoo": len(both),
            "within_1pct_of_yahoo_close": sum(1 for _, x in both if abs(x - 1) <= 0.01),
            "outside_3pct": sorted([[t, round(x, 3)] for t, x in both if abs(x - 1) > 0.03],
                                   key=lambda z: z[1]),
            "median_filers": float(pd.Series([v["filers"] for v in r.values() if v["matched"]]).median()),
            "detail": r}


def report() -> dict:
    dep, cls = departures(), classify()
    cov = yahoo_coverage()
    dl = deals() if (DATA / "deals.json").exists() else {}
    rows = pd.DataFrame(cov["month_rows"])
    total_members = dep["member_months"]
    by_cause = rows.groupby("cause").agg(member_months=("ticker", "size"),
                                         names=("ticker", "nunique"),
                                         monthly_return=("return", "sum"),
                                         return_and_13m_history=("history13", "sum")).reset_index()
    names_by_cause = defaultdict(list)
    for t, c in cls.items():
        names_by_cause[c["cause"]].append(t)
    # route (b): exit months covered - one per acquired name with a stated cash price
    acq_found = [t for t, x in dl.items() if x.get("found")]
    id_verdicts = Counter((v.get("identity") or {}).get("verdict", "no series")
                          if v.get("rows") else "no series"
                          for k, v in cov["per_ticker"].items() if not k.startswith("_"))
    year = rows.assign(year=rows["month"].str[:4]).groupby("year").agg(
        departed=("ticker", "size"), monthly_return=("return", "sum"),
        return_and_13m=("history13", "sum"))
    members_by_year = Counter()
    for c in json.loads(CENSUS.read_text(encoding="utf-8")):
        members_by_year[c["month"][:4]] += c["members"]
    built_by_year = Counter()
    for c in json.loads(CENSUS.read_text(encoding="utf-8")):
        built_by_year[c["month"][:4]] += c["built"]
    year["members"] = [members_by_year[y] for y in year.index]
    year["built_now"] = [built_by_year[y] for y in year.index]
    year["coverage_now"] = year["built_now"] / year["members"]
    year["coverage_with_route_a"] = (year["built_now"] + year["return_and_13m"]) / year["members"]
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "window": [START, END],
        "member_months": total_members,
        "not_scored_by_census_reason": dep["by_reason"],
        "departed_member_months": int(len(rows)),
        "departed_names": int(rows["ticker"].nunique()),
        "cik_resolution": Counter(("resolved" if c.get("cik") else "unresolved") for c in cls.values()),
        "by_cause": by_cause.to_dict("records"),
        "names_by_cause": {k: sorted(v) for k, v in names_by_cause.items()},
        "route_a_yahoo": {
            "identity_verdicts": dict(id_verdicts),
            "member_months_with_monthly_return": int(rows["return"].sum()),
            "member_months_with_return_and_13m_history": int(rows["history13"].sum()),
            "share_of_departed_with_return": float(rows["return"].mean()),
            "share_of_departed_with_return_and_history": float(rows["history13"].mean()),
            "per_ticker": cov["per_ticker"],
        },
        "route_b_deal_price": {
            "acquired_names_sampled": len(dl), "cash_price_found": len(acq_found),
            "pre_announcement_close_found": sum(1 for x in dl.values()
                                                if (x.get("doc") or {}).get("pre_announcement_close")),
            "detail": dl,
        },
        "remaining_gap_after_route_a": rows.loc[~rows["history13"]].groupby("cause").size().to_dict(),
        "route_c_10k_item5": _tenk_summary(),
        "route_d_13f_2015q4": _13f_summary(),
        "by_year": year.reset_index().to_dict("records"),
        "classification": cls,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(by_cause.to_string())
    print(year.round(3).to_string())
    print(json.dumps({k: out["route_a_yahoo"][k] for k in out["route_a_yahoo"] if k != "per_ticker"},
                     indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["departures", "classify", "yahoo", "yahoo_alt", "deals", "tenk", "13f",
                                      "report"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=45)
    a = ap.parse_args()
    if a.stage == "departures":
        d = departures(a.force)
        print(len(d["rows"]), "not-scored member-months;", len(d["tickers"]), "departed symbols;",
              d["by_reason"], "of", d["member_months"])
        print(Counter(bool(v.get("cik")) for v in d["tickers"].values()))
    elif a.stage == "classify":
        c = classify(a.force)
        print(Counter(v["cause"] for v in c.values()))
    elif a.stage == "yahoo":
        y = yahoo(a.force)
        print(len(y), "rows;", y["ticker"].nunique() if len(y) else 0, "symbols")
    elif a.stage == "yahoo_alt":
        y = yahoo_alt(a.force)
        print(len(y), "rows;", sorted(y["ticker"].unique()) if len(y) else [])
    elif a.stage == "deals":
        d = deals(a.force, a.n)
        print(sum(1 for v in d.values() if v.get("found")), "of", len(d))
    elif a.stage == "tenk":
        tenk_sample()
    elif a.stage == "13f":
        r = thirteenf()
        print(Counter((v["cause"], v["matched"]) for v in r.values()))
    else:
        report()


if __name__ == "__main__":
    main()
