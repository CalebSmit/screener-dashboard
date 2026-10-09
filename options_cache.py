"""Option quotes, fetched at an hour when they exist and read back at the hour the loops run.

WHY THIS EXISTS - measured 2026-10-09 (``research/measurements/2026-10-09-option-quote-availability.py``,
``plan/context-layer.md`` queue item 2).

The context layer's options panel needs a *quote*: the expected move is the at-the-money call plus
put mid, and the ATM implied volatility and put skew come off the same chain. Yahoo serves that
chain around the clock, but **overnight it serves it empty**: every strike comes back with
``bid`` and ``ask`` of ``0.00`` and ``impliedVolatility`` of ``0.000``, while ``lastPrice`` and
``openInterest`` survive. Measured on the full universe, same code, same day:

====================================  ==========  =====  ========
Run                                   Hour (ET)       n  ``ok``
====================================  ==========  =====  ========
2026-10-08, owner-run                      21:27    503  **394**
2026-10-09, the 02:00 data loop            03:00    503  **0**
====================================  ==========  =====  ========

So the loop that publishes the site fetched ~1,000 option requests a night and showed a number for
nobody, and the 06:00 code loop sits in the same dead window. The fix is not a looser quality bar -
there is no quote to accept - it is to **fetch when quotes exist and read the cache when they do
not**, which is how ``market_context`` (FRED), ``insider_activity`` (Form 4s) and ``track_record``
(closes) already work.

Two pieces:

* :func:`refresh` runs after the close, keeps only readings that are actually usable, and writes
  them with the date the quotes belong to. ``scripts/refresh-option-quotes.ps1`` is its scheduled
  wrapper.
* :func:`quotes_are_live` lets the 02:00 context pass **find out** rather than consult a hard-coded
  clock: it probes a couple of chains, and if they come back empty it skips the option half of the
  pass for the night and reads the cache instead. A guessed window would go wrong the day the
  source changes its behaviour; a probe costs two requests and cannot.

Everything here is display-only context (CLAUDE.md settled row "ctx"): no field it produces may
reach ``raw``, ``pct``, a category score or the composite.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

CACHE_DIR = Path("data") / "options"
CACHE_NAME = "quotes.json"

#: A reading older than this is dropped rather than shown. Three calendar days carries a Friday
#: close through the weekend to Monday's run; past that the quotes describe a different market.
MAX_AGE_DAYS = 3

#: Probed to find out whether quotes are being served. Liquid, always-listed, and spread across
#: sectors so one halted name cannot decide the night.
PROBE_TICKERS = ("AAPL", "MSFT", "JPM", "XOM", "KO")
PROBE_LIMIT = 3

#: What the page is told when the source serves no quotes at this hour and the cache has nothing.
STATUS_CLOSED = "quotes-closed"

USABLE_STATUSES = ("ok", "partial")

# Pacing for the refresh pass. The same shape as context_fetch: a small pool and a per-request
# sleep, measured at 503 stocks in 206 s for options *and* insider together.
WORKERS = 2
PACE_SECONDS = 0.35
DEFAULT_BUDGET_SECONDS = 900


def cache_path(root: str | os.PathLike | None = None) -> Path:
    return (Path(root) if root is not None else Path(".")) / CACHE_DIR / CACHE_NAME


# --------------------------------------------------------------------------- read / write
def load(root: str | os.PathLike | None = None) -> dict:
    """The cache, or an empty one. A missing or corrupt file is not an error - context is optional."""
    p = cache_path(root)
    if not p.exists():
        return {"readings": {}}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        warnings.warn(f"option quote cache unreadable ({type(e).__name__}: {e}); ignoring it")
        return {"readings": {}}
    if not isinstance(d, dict) or not isinstance(d.get("readings"), dict):
        return {"readings": {}}
    return d


def save(readings: dict, quote_date: str, root: str | os.PathLike | None = None) -> Path:
    """Write the cache atomically, so a refresh killed half way cannot leave a torn file."""
    p = cache_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "quote_date": quote_date,
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "readings": readings,
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- read back
def _parse(d: str | None) -> date | None:
    try:
        return datetime.strptime(str(d), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def reading_for(cache: dict, ticker: str, today: date,
                max_age_days: int = MAX_AGE_DAYS) -> dict | None:
    """A cached reading for ``ticker``, or ``None`` if there is nothing honest to show.

    Refused when the quote date is missing, older than ``max_age_days``, in the future, or when
    the expiry it priced has already passed. ``_ctx_opt_days`` is **recomputed against today**,
    not served from the cache: the page prints "the average move options are paying for by
    <date> (N days)", and N shrinks every day the reading is reused.
    """
    entry = (cache.get("readings") or {}).get(ticker)
    if not isinstance(entry, dict):
        return None
    qd = _parse(entry.get("_ctx_opt_quote_date") or cache.get("quote_date"))
    if qd is None:
        return None
    age = (today - qd).days
    if age < 0 or age > max_age_days:
        return None
    exp = _parse(entry.get("_ctx_opt_expiry"))
    if exp is None or exp < today:
        return None
    out = {k: v for k, v in entry.items() if k.startswith("_ctx_")}
    out["_ctx_opt_quote_date"] = qd.strftime("%Y-%m-%d")
    out["_ctx_opt_days"] = (exp - today).days
    return out


# --------------------------------------------------------------------------- the probe
def quotes_are_live(probe_status, tickers=PROBE_TICKERS, limit: int = PROBE_LIMIT) -> tuple[bool, str]:
    """Are option quotes being served right now?

    ``probe_status(ticker)`` returns the ``_ctx_opt_status`` a real fetch produced (or ``None`` if
    the fetch itself failed). Quotes count as live as soon as **one** probe comes back usable;
    they count as not live only after every probe has come back and none did. A fetch error is
    neither - it says nothing about the hour - so an all-errors probe reports live and lets the
    pass try, rather than silently skipping the night on a network blip.
    """
    seen: list[str] = []
    errors = 0
    for t in list(tickers)[:limit]:
        try:
            s = probe_status(t)
        except Exception as e:  # noqa: BLE001 - context is optional by design
            errors += 1
            seen.append(f"{t}:{type(e).__name__}")
            continue
        seen.append(f"{t}:{s}")
        if s in USABLE_STATUSES:
            return True, f"quotes live ({', '.join(seen)})"
    if errors and errors == len(seen):
        return True, f"probe inconclusive, all errors ({', '.join(seen)}) - trying anyway"
    return False, f"no quotes being served ({', '.join(seen)})"


# --------------------------------------------------------------------------- the refresh pass
def _records_from_latest_run(root: Path) -> list[dict]:
    """Ticker, price and earnings date from the most recent run's raw fetch."""
    import pandas as pd

    files = sorted((root / "runs").glob("*/00_raw_fetch.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit("No runs/*/00_raw_fetch.parquet - run the screener before refreshing quotes.")
    cols = ["Ticker", "price_latest", "earningsTimestampStart"]
    d = pd.read_parquet(files[0], columns=cols)
    return [r for r in d.to_dict("records") if r.get("Ticker")]


def refresh(records: list[dict], root: str | os.PathLike | None = None,
            budget_seconds: float = DEFAULT_BUDGET_SECONDS, log=print,
            fetch=None, probe=None) -> dict:
    """Fetch every stock's chain and store the readings that are usable.

    Only ``ok`` and ``partial`` readings are kept. A chain with no quotes contributes nothing, so
    a refresh run at the wrong hour leaves yesterday's good cache alone instead of replacing it
    with 500 empty entries - which is why the probe runs first and the pass stops if quotes are
    not being served.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    started = time.time()
    today = datetime.now(timezone.utc).date()
    fetch = fetch or _fetch_one
    probe = probe or (lambda t: (fetch(t, None, None, today) or {}).get("_ctx_opt_status"))

    live, why = quotes_are_live(probe)
    log(f"  Option quotes: {why}")
    if not live:
        log("  Option quote refresh: nothing fetched - the source is serving no quotes at this hour. "
            "Cache left as it was.")
        return {"kept": 0, "of": len(records), "seconds": round(time.time() - started),
                "live": False, "why": why}

    kept: dict[str, dict] = {}
    failed = 0
    todo = list(records)
    i = 0
    stopped = None
    while i < len(todo):
        if time.time() - started > budget_seconds:
            stopped = "time budget"
            break
        chunk = todo[i:i + WORKERS * 5]
        i += len(chunk)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {}
            for r in chunk:
                futs[pool.submit(fetch, r["Ticker"], r.get("price_latest"),
                                 r.get("earningsTimestampStart"), today)] = r["Ticker"]
                time.sleep(PACE_SECONDS)
            for f in as_completed(futs):
                tk = futs[f]
                try:
                    ctx = f.result() or {}
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    warnings.warn(f"{tk}: option quotes unavailable: {type(e).__name__}: {e}")
                    continue
                if ctx.get("_ctx_opt_status") in USABLE_STATUSES:
                    ctx = {k: v for k, v in ctx.items() if k.startswith("_ctx_opt")}
                    ctx["_ctx_opt_quote_date"] = today.strftime("%Y-%m-%d")
                    kept[tk] = ctx

    # Never replace a good cache with a worse one: if this pass kept nothing, leave the file.
    if kept:
        prev = load(root)
        merged = dict(prev.get("readings") or {})
        merged.update(kept)
        # drop entries that can no longer be served, so the file does not grow without bound
        merged = {k: v for k, v in merged.items()
                  if reading_for({"readings": {k: v}}, k, today, MAX_AGE_DAYS + 2) is not None}
        p = save(merged, today.strftime("%Y-%m-%d"), root)
        log(f"  Option quote refresh: kept {len(kept)} of {len(todo)} usable, {failed} failed, "
            f"{len(merged)} in the cache, {round(time.time() - started)}s"
            f"{f' (stopped: {stopped})' if stopped else ''} -> {p}")
    else:
        log(f"  Option quote refresh: no usable reading out of {len(todo)} tried "
            f"({failed} failed). Cache left as it was.")
    return {"kept": len(kept), "of": len(todo), "failed": failed, "live": True,
            "seconds": round(time.time() - started), "stopped": stopped, "why": why}


def _fetch_one(ticker: str, price, earnings_ts, today: date) -> dict:
    """One stock's chain, through exactly the pipeline's own expiry choice and quality bars."""
    import numpy as np
    import yfinance as yf

    from context_signals import choose_expiry, options_context

    t = yf.Ticker(ticker)
    if price is None or not np.isfinite(price) or price <= 0:
        try:
            price = float(t.fast_info["lastPrice"])
        except Exception:  # noqa: BLE001
            return {}
    ed = None
    try:
        ed = datetime.fromtimestamp(int(earnings_ts), tz=timezone.utc).date() if earnings_ts else None
    except (TypeError, ValueError, OSError, OverflowError):
        ed = None
    exp, spans = choose_expiry(t.options, today, ed)
    if not exp:
        return {}
    ch = t.option_chain(exp)
    return options_context(ch.calls, ch.puts, float(price), exp, today, spans)


def main(argv=None) -> int:
    """Entry point for the scheduled after-close refresh. Writes only the gitignored cache."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS,
                    help="seconds to spend before keeping what it has")
    ap.add_argument("--limit", type=int, default=0, help="only the first N stocks (for a smoke test)")
    a = ap.parse_args(argv)

    root = Path(__file__).resolve().parent
    records = _records_from_latest_run(root)
    if a.limit:
        records = records[:a.limit]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] Option quote refresh: {len(records)} stocks, budget {a.budget:.0f}s")
    stats = refresh(records, root=root, budget_seconds=a.budget)
    age = ""
    c = load(root)
    if c.get("quote_date"):
        age = f" cache quote_date={c['quote_date']}, {len(c.get('readings') or {})} readings"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] done: {stats}.{age}")
    # A refresh at an hour with no quotes is a correct no-op, not a failure: the scheduled task
    # must not report red for it, or a real fault becomes invisible among the noise.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
