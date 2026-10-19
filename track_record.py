"""The ranking's live track record: how its top names have actually done since each run.

WHY THIS EXISTS - 2026-10-08 (owner, ``plan/context-layer.md``). The most persuasive thing a
screener can show is not another metric; it is what happened next. Every comparable run since
2026-02-20 left a snapshot of every stock's rank, so this is an **out-of-sample** record: no
look-ahead and no survivorship in the selection. Not "no backfill" (corrected 2026-10-19,
``research/2026-10-19-track-record.md`` section 2): the snapshots for 2026-02-20 to 03-01 were
written on 2026-03-09 by ``improvement_engine.backfill_from_existing_runs`` from each run's saved
output - each run's own ranking, recorded three weeks later - and the first predates the public page.
It is a **model** record: no money was invested (GIPS 2020 4.C.48 calls this theoretical
performance), and no return in it is annualised (2.A.12).

What it measures, and what it is not:

* It measures **the ranking**: the top 25 by composite, and the top fifth against the bottom
  fifth, equal-weighted, rebalanced at the first comparable run of each calendar month and held
  through any month with no run (as someone following it would have done).
* Entry is the close of the **first trading day on or after** the run date (the 02:00 run uses the
  previous close, so a reader could not have traded at that price). A run made **after** its
  date's close enters at the next close instead (``runs_after_the_close``): read from the run's
  recorded start time (``run_started_at``, in every snapshot from 2026-10-19) and, for older
  snapshots that record no time, inferred from the run's own recorded prices. Until 2026-10-19 the
  2026-08-10 rebalance, made in the evening, entered at a close set before its ranking existed.
* It is **not** a portfolio to follow. The Model Portfolio panel was removed on 2026-08-26 because
  a fixed buy list on a public page is the closest this tool came to a recommendation; this page
  lists no current holdings and no weights to copy.
* It is **not** evidence for changing the methodology (CLAUDE.md rule 4): a few months of one
  market is far too short, and the methodology itself changed during the window - the page marks
  each change from ``METHODOLOGY_CHANGELOG.md``.
* No trading costs or taxes. Prices are dividend-adjusted closes (total return).
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / "track_record.json"
PRICE_CACHE = ROOT / "data" / "track" / "prices.parquet"
TOP_N = 25
QUINTILE = 0.20
BENCHMARKS = {"RSP": "S&P 500 equal-weight (RSP)", "SPY": "S&P 500 (SPY)"}
CIK_MAP = ROOT / "data" / "universe_history" / "ticker_ciks.json"
SNAPSHOT_DIR = ROOT / "improvement" / "snapshots"
COVERED = 0.8      # a ticker with closes on fewer of the window's days than this is not priced


# --------------------------------------------------------------------------- schedule
def rebalance_dates(run_dates: list[str]) -> list[str]:
    """The first run of each calendar month, in order."""
    seen, out = set(), []
    for d in sorted(run_dates):
        m = d[:7]
        if m not in seen:
            seen.add(m)
            out.append(d)
    return out


def portfolio_from_snapshot(ranks: dict[str, int]) -> dict[str, list[str]]:
    ordered = [t for t, _ in sorted(ranks.items(), key=lambda kv: kv[1])]
    n = len(ordered)
    q = max(1, int(round(n * QUINTILE)))
    return {"top": ordered[:TOP_N], "q1": ordered[:q], "q5": ordered[-q:]}


# --------------------------------------------------------------------------- prices
def load_prices(tickers: list[str], start: date, end: date | None = None, download=None) -> pd.DataFrame:
    """Dividend-adjusted daily closes, cached; only the missing tail is downloaded."""
    end = end or date.today()
    cached = None
    if PRICE_CACHE.exists():
        try:
            cached = pd.read_parquet(PRICE_CACHE)
        except (OSError, ValueError):
            cached = None
    need_all = cached is None or not set(tickers).issubset(cached.columns) or cached.index.min().date() > start
    dl_start = start if need_all else (cached.index.max().date() - timedelta(days=7))
    if download is None:
        import yfinance as yf

        def download(tks, s, e):
            df = yf.download(tks, start=s.isoformat(), end=(e + timedelta(days=1)).isoformat(),
                             auto_adjust=True, progress=False, threads=True)
            return df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tks[0]})
    fresh = download(sorted(set(tickers)), dl_start, end)
    fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
    if cached is not None and not need_all:
        prices = pd.concat([cached.loc[cached.index < fresh.index.min()], fresh]).sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]
        # A column that exists is not a column that is complete (2026-10-09: PSKY had 9
        # closes in the cache over a 160-day window while trading every day, so it dropped
        # out of three bottom-fifth baskets). Availability is tested by coverage of the
        # window the record reads, never by the column's presence: any ticker missing more
        # than a fifth of the window's trading days is downloaded again from the start.
        win = prices.loc[prices.index >= pd.Timestamp(start)]
        if len(win) >= 20:
            frac = win.reindex(columns=sorted(set(tickers))).notna().mean()
            thin = sorted(t for t, f in frac.items() if f < 0.8)
            if thin:
                try:
                    redo = download(thin, start, end)
                    redo.index = pd.to_datetime(redo.index).tz_localize(None)
                    for t in thin:
                        if t in redo.columns and (redo[t].notna().sum() > (win[t].notna().sum() if t in win.columns else 0)):
                            prices[t] = redo[t].reindex(prices.index).combine_first(prices[t]) if t in prices.columns else redo[t].reindex(prices.index)
                except Exception:  # noqa: BLE001 - a failed repair keeps what we had
                    pass
    else:
        prices = fresh.sort_index()
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICE_CACHE)
    return prices


# --------------------------------------------------------------------------- picks the source lost
# 2026-10-15 (plan/context-layer.md item 6). The free price source stops serving a company that is
# renamed or taken over, and the record used to drop it from every basket it was ever in - a
# survivorship gap inside an otherwise out-of-sample record. Two repairs, each said on the page:
# a renamed company is priced under its current symbol (the SEC registrant number, CIK, survives a
# rename - the same rule backtest v2 uses); a company that stopped trading is priced from the
# screener's own record of its price at each run, which ends where its trading ended.
def renamed_symbols(tickers, current, cik_path: Path = CIK_MAP) -> dict:
    """old symbol -> current symbol, for picks no longer in the universe whose SEC registrant now
    trades under another symbol that is."""
    try:
        ciks = {k: int(v) for k, v in json.loads(cik_path.read_text(encoding="utf-8"))["ciks"].items()}
    except (OSError, KeyError, ValueError, TypeError):
        return {}
    by_cik: dict = {}
    for t in current:
        if t in ciks:
            by_cik.setdefault(ciks[t], []).append(t)
    out = {}
    for t in tickers:
        if t in current or t not in ciks:
            continue
        cands = sorted(x for x in by_cik.get(ciks[t], []) if x != t)
        if cands:
            out[t] = cands[0]
    return out


def recorded_closes(tickers, snap_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """The price the screener recorded for each ticker at each run (``price_at_scoring``), dated
    to the trading day before the run - the 02:00 run scores on the previous close. Dividends are
    not in it, so a basket priced this way is a price return for that name."""
    rows: dict = {}
    want = set(tickers)
    for p in sorted(snap_dir.glob("*.parquet")):
        try:
            day = pd.Timestamp(p.name[:10]) - pd.offsets.BDay(1)
            df = pd.read_parquet(p, columns=["Ticker", "price_at_scoring"])
        except (OSError, ValueError, KeyError):
            continue
        for t, v in zip(df["Ticker"], df["price_at_scoring"]):
            if t in want and pd.notna(v) and v > 0:
                rows.setdefault(t, {})[day] = float(v)
    return pd.DataFrame(rows).sort_index()


def fill_unpriced(prices: pd.DataFrame, tickers, start, renames: dict, recorded: pd.DataFrame):
    """(prices, what was filled): each pick with closes on fewer than ``COVERED`` of the window's
    days is priced under its current symbol when renamed, or else from the recorded closes -
    carried between runs and never past the last one."""
    out = prices.copy()
    win = out.index[out.index >= pd.Timestamp(start)]
    filled = {"renamed": {}, "from_records": []}
    if len(win) == 0:
        return out, filled

    def cover(col):
        return float(out[col].reindex(win).notna().mean()) if col in out.columns else 0.0

    for t in sorted(set(tickers)):
        if cover(t) >= COVERED:
            continue
        new = renames.get(t)
        if new and cover(new) >= COVERED:
            out[t] = out[new]
            filled["renamed"][t] = new
            continue
        if t in recorded.columns:
            rec = recorded[t].dropna()
            if rec.empty:
                continue
            # carried from one run to the next, from the first record to the last and no further
            carried = rec.reindex(out.index.union(rec.index)).ffill().reindex(out.index)
            carried[(carried.index < rec.index.min()) | (carried.index > rec.index.max())] = np.nan
            base = out[t] if t in out.columns else pd.Series(np.nan, index=out.index)
            if (base.isna() & carried.notna()).any():       # listed only if the record added prices
                out[t] = base.combine_first(carried)
                filled["from_records"].append(t)
    return out, filled


# --------------------------------------------------------------------------- arithmetic
def run_basket(prices: pd.DataFrame, names: list[str], start: pd.Timestamp, end: pd.Timestamp | None,
               start_value: float) -> tuple[pd.Series, dict]:
    """Equal-weight buy-and-hold of ``names`` from the first close on/after ``start`` through
    ``end`` (exclusive of the next period's entry day). A name whose prices stop (acquired,
    removed) is held at its last price - cash, in effect - and reported."""
    window = prices.loc[(prices.index >= start) & ((prices.index <= end) if end is not None else True)]
    if window.empty:
        return pd.Series(dtype=float), {"held": 0, "stale": []}
    entry = window.iloc[0]
    have = [t for t in names if t in window.columns and pd.notna(entry.get(t)) and entry.get(t) > 0]
    if not have:
        return pd.Series(dtype=float), {"held": 0, "stale": []}
    px = window[have].ffill()
    cutoff = window.index[-1] - pd.Timedelta(days=7)   # a missing last day or two is a late print, not a stop
    stale = [t for t in have if window[t].last_valid_index() is not None and window[t].last_valid_index() < cutoff]
    units = (start_value / len(have)) / entry[have]
    value = (px * units).sum(axis=1)
    last = px.iloc[-1] * units
    end_w = {t: float(v) for t, v in (last / last.sum()).items()}     # weights after the period's drift
    return value, {"held": len(have), "missing": sorted(set(names) - set(have)), "stale": stale,
                   "names": have, "end_w": end_w}


def entry_day(index: pd.DatetimeIndex, run_date: str, after_close: bool = False) -> pd.Timestamp | None:
    """The close a reader of the run could first trade at: the first trading day on or after the
    run's date, or strictly after it when the run was made after that day's close."""
    lo = pd.Timestamp(run_date) + (pd.Timedelta(days=1) if after_close else pd.Timedelta(0))
    after = index[index >= lo]
    return after[0] if len(after) else None


MARKET_TZ = "America/New_York"
CLOSE_HOUR = 16                     # the regular session's close, 16:00 New York time


def after_the_close(started, run_date: str) -> bool | None:
    """Whether a run that started at ``started`` (ISO 8601, with its UTC offset) came after the
    close of ``run_date``: started on a later New York date, or on that date at or after 16:00
    New York time. None when the stamp cannot be read. Early-close days (13:00) are not modelled;
    a run between 13:00 and 16:00 on one would be read as before the close."""
    try:
        ts = pd.Timestamp(str(started))
        if ts.tzinfo is None:                       # naive: this machine's local time
            ts = ts.tz_localize(datetime.now().astimezone().tzinfo)
        et = ts.tz_convert(MARKET_TZ)
    except (TypeError, ValueError):
        return None
    day = pd.Timestamp(run_date).date()
    return et.date() > day or (et.date() == day and et.hour >= CLOSE_HOUR)


def run_started_at(path: Path) -> str | None:
    """The run's start time recorded in a snapshot (``run_started_at``), or None for a snapshot
    written before 2026-10-19 or unreadable."""
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    if "run_started_at" not in df.columns:
        return None
    vals = df["run_started_at"].dropna()
    return str(vals.iloc[0]) if len(vals) else None


def runs_after_the_close(run_dates, prices: pd.DataFrame, snap_dir: Path = SNAPSHOT_DIR,
                         tol: float = 0.002, floor: float = 0.25) -> list[str]:
    """Run dates whose ranking had already seen that day's close.

    2026-10-19, second change: a snapshot written from today on records the run's start time
    (``run_started_at``), and where it does that time decides - ``after_the_close``. The inference
    below stays for every older snapshot, which records only a date.

    2026-10-19 (research/2026-10-19-track-record.md section 6). A snapshot records its date and
    not its time, and the entry rule assumed a run made before that day's close - true of the
    02:00 data loop, false of the 2026-08-10 rebalance run, whose snapshot was committed at 19:22
    CDT and whose ``price_at_scoring`` is that day's close for 47% of stocks within 0.2% against
    10% for the previous day's. That period entered at a price set before the ranking existed.
    The time of a run is read from its own recorded prices, not assumed: a run is after the close
    when more of its prices sit within ``tol`` of the run date's close than of the previous
    trading day's, and at least ``floor`` of them do. A snapshot with no recorded prices (the
    runs before 2026-03-09, and 2026-04-14) cannot be tested and keeps the run date's close;
    the page says so."""
    out = []
    files = {}
    for f in sorted(Path(snap_dir).glob("*.parquet")):
        files[f.name[:10]] = f
    for d in sorted(run_dates):
        f = files.get(d)
        day = pd.Timestamp(d)
        if f is None or day not in prices.index:
            continue
        started = run_started_at(f)
        if started is not None:
            late = after_the_close(started, d)
            if late is not None:
                if late:
                    out.append(d)
                continue
        prev = prices.index[prices.index < day]
        if not len(prev):
            continue
        try:
            rec = pd.read_parquet(f, columns=["Ticker", "price_at_scoring"]).set_index("Ticker")["price_at_scoring"]
        except (OSError, ValueError, KeyError):
            continue

        def share(when):
            row = prices.loc[when]
            common = [t for t in rec.index if t in row.index and pd.notna(row[t]) and pd.notna(rec[t])
                      and rec[t] > 0 and row[t] > 0]
            if len(common) < 50:
                return 0.0
            return float(((rec[common] / row[common] - 1).abs() < tol).mean())

        same, before = share(day), share(prev[-1])
        if same >= floor and same > before:
            out.append(d)
    return out


def untimed_runs(run_dates, snap_dir: Path = SNAPSHOT_DIR) -> list[str]:
    """Run dates whose snapshot records neither the run's start time nor its prices, so whether
    the run came after the close cannot be tested."""
    out = []
    for d in sorted(run_dates):
        f = next(iter(sorted(Path(snap_dir).glob(f"{d}_*.parquet"))), None)
        if f is None:
            continue
        try:
            cols = pd.read_parquet(f).columns
        except (OSError, ValueError):
            continue
        if "price_at_scoring" not in cols and "run_started_at" not in cols:
            out.append(d)
    return out


def one_way_turnover(before: dict | None, names: list) -> float | None:
    """Half the sum of absolute weight changes from the drifted basket to equal weights on
    ``names``: the share of the basket sold, which equals the share bought."""
    if before is None or not names:
        return None
    target = {t: 1.0 / len(names) for t in names}
    keys = set(before) | set(target)
    return 0.5 * sum(abs(target.get(t, 0.0) - before.get(t, 0.0)) for t in keys)


def break_even_cost(top: float | None, bench: float | None, turnovers: list) -> float | None:
    """The cost per dollar traded, paid on every sale and every purchase, at which the basket's
    lead over the benchmark is gone: (1 + top) x prod(1 - 2 c TO_k) = 1 + bench.

    ``None`` when there is no lead to lose or nothing was traded. The first purchase is not
    charged, because a buyer of the benchmark pays one too; the benchmark's own running costs
    are already inside its price."""
    if top is None or bench is None or top <= bench or not turnovers or max(turnovers) <= 0:
        return None

    def net(c):
        v = 1.0 + top
        for to in turnovers:
            v *= max(0.0, 1.0 - 2.0 * c * to)
        return v

    lo, hi = 0.0, 0.5 / max(turnovers)
    if net(hi) > 1.0 + bench:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if net(mid) > 1.0 + bench else (lo, mid)
    return (lo + hi) / 2


def max_drawdown(series: pd.Series) -> float | None:
    if series is None or len(series) < 2:
        return None
    peak = series.cummax()
    return float((series / peak - 1).min())


def methodology_dates(path: Path | None = None) -> list[dict]:
    """Dated headings in METHODOLOGY_CHANGELOG.md - the moments the ranking's rules changed."""
    path = path or (ROOT / "METHODOLOGY_CHANGELOG.md")
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^##\s+(\d{4}-\d{2}-\d{2})\s*[-—:]?\s*(.*)", line)
            if m:
                out.append({"date": m.group(1), "title": m.group(2).strip()[:120]})
    except OSError:
        pass
    seen, uniq = set(), []
    for e in out:
        if e["date"] not in seen:
            seen.add(e["date"])
            uniq.append(e)
    return sorted(uniq, key=lambda e: e["date"])


def coin_toss_chance(k: int, n: int) -> float | None:
    """P(at least ``k`` of ``n`` periods ahead) if each period were a fair coin toss."""
    if n <= 0:
        return None
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def excess_record(top: pd.Series, bench: pd.Series) -> dict | None:
    """How far the basket's lead over the benchmark is from zero, measured day by day.

    The daily difference in returns has a mean and a standard deviation; ``t`` is the mean over
    its standard error. Two standard errors is the usual minimum for telling a lead from luck
    (research/2026-10-19-track-record.md section 3). ``years_to_two`` is how long a record would
    need to be for the same daily lead, with the same noise, to reach two: (2 / t)^2 x days,
    in years of 252 trading days. It extrapolates the lead seen so far, which is the point - it
    is the most optimistic reading the record allows. ``None`` with fewer than 20 days."""
    a, b = top.align(bench, join="inner")
    ex = (a.pct_change() - b.pct_change()).dropna()
    n = len(ex)
    if n < 20 or ex.std(ddof=1) == 0:
        return None
    t = float(ex.mean() / (ex.std(ddof=1) / math.sqrt(n)))
    years = (2.0 / t) ** 2 * n / 252.0 if t > 0 else None
    return {"days": n, "t": round(t, 2), "years_to_two": (round(years, 1) if years is not None else None)}


def build(snapshots: list, prices: pd.DataFrame, today: date | None = None, changelog: list[dict] | None = None,
          return_curves: bool = False, after_close: list[str] | None = None,
          untimed: list[str] | None = None) -> dict:
    """``snapshots`` are history.RunSnapshot objects (date, ranks). Returns the payload block."""
    today = today or date.today()
    by_date = {s.date: s for s in snapshots}
    rebal = rebalance_dates(list(by_date))
    if len(rebal) < 1 or prices.empty:
        return {"available": False}
    baskets = {"top": [], "q1": [], "q5": []}
    bench = {b: [] for b in BENCHMARKS}
    values = {k: 100.0 for k in list(baskets) + list(bench)}
    periods = []
    prev_top = None
    prev_w = None                              # the top basket's weights at the end of the last period
    late = set(after_close or [])              # runs made after their date's close enter a day later
    unpriced = {k: set() for k in baskets}     # picks with no price at entry, per basket
    picks = {k: 0 for k in baskets}
    for i, d in enumerate(rebal):
        start = entry_day(prices.index, d, d in late)
        if start is None:                      # no close after this run yet: nothing to measure
            start = pd.Timestamp(d) + pd.Timedelta(days=1)
        # hold until the day before the next entry day
        end = None
        if i + 1 < len(rebal):
            nxt = entry_day(prices.index, rebal[i + 1], rebal[i + 1] in late)
            end = nxt - pd.Timedelta(days=1) if nxt is not None else None
        port = portfolio_from_snapshot(by_date[d].ranks)
        row = {"date": d, "entry": None}
        for k in baskets:
            v, info = run_basket(prices, port[k], start, end, values[k])
            picks[k] += len(port[k])
            unpriced[k] |= set(info.get("missing") or [])
            if v.empty:
                continue
            baskets[k].append(v)
            row[k] = round(float(v.iloc[-1] / values[k] - 1), 5)
            values[k] = float(v.iloc[-1])
            if k == "top":
                row["entry"] = v.index[0].strftime("%Y-%m-%d")
                row["exit"] = v.index[-1].strftime("%Y-%m-%d")
                row["held"] = info["held"]
                row["stale"] = info["stale"]
                row["changed"] = (len(set(port["top"]) - set(prev_top)) if prev_top is not None else None)
                to = one_way_turnover(prev_w, info["names"])
                row["turnover"] = round(to, 4) if to is not None else None
                prev_w = info["end_w"]
        for b in bench:
            v, _ = run_basket(prices, [b], start, end, values[b])
            if v.empty:
                continue
            bench[b].append(v)
            row[b] = round(float(v.iloc[-1] / values[b] - 1), 5)
            values[b] = float(v.iloc[-1])
        prev_top = port["top"]
        periods.append(row)

    def join(parts):
        if not parts:
            return pd.Series(dtype=float)
        s = pd.concat(parts)
        return s[~s.index.duplicated(keep="last")]

    curves = {k: join(v) for k, v in {**baskets, **bench}.items()}
    base = curves["top"]
    if base.empty:
        return {"available": False}
    # weekly samples for the chart, every curve on the same dates, starting at 100
    idx = base.index
    wk = idx[pd.Series(idx).groupby(pd.Series(idx).dt.to_period("W-FRI")).transform("max").eq(pd.Series(idx)).values]
    if wk[0] != idx[0]:
        wk = idx[:1].append(wk)
    series = {"d": [d.strftime("%Y-%m-%d") for d in wk]}
    for k, s in curves.items():
        s = s.reindex(idx).ffill()
        series[k] = [round(float(x), 3) if pd.notna(x) else None for x in s.loc[wk].values]
    total = {k: (round(float(curves[k].iloc[-1] / 100 - 1), 5) if not curves[k].empty else None) for k in curves}
    first, last = idx[0], idx[-1]
    beat = [p for p in periods if p.get("top") is not None and p.get("RSP") is not None]
    marks = [m for m in (changelog if changelog is not None else methodology_dates())
             if first.strftime("%Y-%m-%d") <= m["date"] <= last.strftime("%Y-%m-%d")]
    tos = [p["turnover"] for p in periods if p.get("turnover") is not None]
    be = break_even_cost(total.get("top"), total.get("RSP"), tos)
    out = {
        "available": True,
        "as_of": today.isoformat(),
        "start": first.strftime("%Y-%m-%d"), "end": last.strftime("%Y-%m-%d"),
        "days": int((last - first).days),
        "rule": {"top_n": TOP_N, "quintile": QUINTILE, "rebalance": "first comparable run of each month",
                 "entry": "close of the first trading day on or after the run's date; the next one for a run made after that close"},
        "benchmarks": BENCHMARKS,
        "total": total,
        "spread": (round(total["q1"] - total["q5"], 5) if total.get("q1") is not None and total.get("q5") is not None else None),
        "max_dd": {k: (round(max_drawdown(curves[k]), 5) if not curves[k].empty else None) for k in curves},
        "periods": periods,
        "beat_rsp": sum(1 for p in beat if p["top"] > p["RSP"]),
        "periods_compared": len(beat),
        # How often a fair coin would do at least as well, and how far the lead is from zero day
        # by day: the two numbers that say what a record this short can show (2026-10-19).
        "beat_rsp_chance": (round(coin_toss_chance(sum(1 for p in beat if p["top"] > p["RSP"]), len(beat)), 3)
                            if beat else None),
        "excess": excess_record(curves["top"], curves["RSP"]) if not curves["RSP"].empty else None,
        "series": series,
        "methodology_changes": marks,
        # Names the ranking picked that the free price source no longer serves (taken
        # private, acquired, renamed) - left out of their basket, which is a survivorship
        # gap the page states rather than hides (plan/context-layer.md item 6).
        "unpriced": {k: sorted(v) for k, v in unpriced.items()},
        "picks": picks,
        "gaps": [p["date"] for i, p in enumerate(periods[1:], 1)
                 if (pd.Timestamp(p["date"]) - pd.Timestamp(periods[i - 1]["date"])).days > 45],
        # Turnover and the cost it implies (plan/context-layer.md item 6, 2026-10-19). The record
        # is gross of costs, so it states how much the top 25 traded and the one cost per dollar
        # traded at which its lead over RSP would be gone - a figure that needs no assumed cost to
        # be true. GIPS 2020 for Firms 2.A.13 requires real returns net of transaction costs; this
        # is a model record (4.C.48), so it says what it leaves out instead of estimating it.
        # Which rebalance runs entered the next day because they had already seen their date's
        # close, and which could not be tested because their snapshot records no prices.
        "after_close": sorted(late & set(rebal)),
        "untimed": sorted(set(untimed or []) & set(rebal)),
        "turnover": ({"rebalances": len(tos),
                      "mean": round(float(np.mean(tos)), 4), "min": round(min(tos), 4), "max": round(max(tos), 4),
                      "traded": round(2 * sum(tos), 4),
                      "break_even_cost": (round(be, 5) if be is not None else None),
                      "basis": "top 25; one-way, from the drifted basket to equal weights; first purchase not counted"}
                     if tos else None),
    }
    if return_curves:            # research/measurements only; never written to the payload
        out["_curves"] = dict(curves)
    return out


def build_from_disk(today: date | None = None, write: bool = True, live=None) -> dict:
    """Load the comparable snapshots and prices, build, and write ``data/track_record.json``.

    ``live`` is ``(date, {ticker: rank})`` for the run being published, which is not in the
    snapshot directory yet (the snapshot is written after the dashboard)."""
    import history
    today = today or date.today()
    kept, _ = history.select_comparable_runs()
    if live and live[1] and not any(s.date == live[0] for s in kept):
        kept = list(kept) + [history.RunSnapshot(date=live[0], ranks={t: int(r) for t, r in live[1].items() if r == r},
                                                 composites={}, cat_scores={})]
    if not kept:
        return {"available": False}
    picks = sorted({t for s in kept for t in s.ranks})
    current = set(max(kept, key=lambda s: s.date).ranks)
    renames = renamed_symbols(picks, current)
    tickers = sorted(set(picks) | set(renames.values())) + list(BENCHMARKS)
    start = date.fromisoformat(min(s.date for s in kept))
    prices = load_prices(tickers, start, today)
    thin = [t for t in picks if t not in prices.columns
            or prices[t].reindex(prices.index[prices.index >= pd.Timestamp(start)]).notna().mean() < COVERED]
    prices, filled = fill_unpriced(prices, picks, start, renames, recorded_closes(thin) if thin else pd.DataFrame())
    dates = [s.date for s in kept]
    out = build(kept, prices, today, after_close=runs_after_the_close(rebalance_dates(dates), prices),
                untimed=untimed_runs(dates))
    if out.get("available"):
        out["filled"] = filled
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out


def load() -> dict | None:
    try:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    res = build_from_disk()
    print(json.dumps({k: v for k, v in res.items() if k not in ("series",)}, indent=1)[:4000])
