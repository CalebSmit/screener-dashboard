"""The context pass: option chains and insider trades, fetched AFTER the core data is safe.

WHY A SEPARATE PASS - measured 2026-10-08. The first draft fetched options and insider trades
inside each stock's core fetch. Three extra requests per stock tripped Yahoo's rate limiter at
batch 9 of 17; the fetcher backed off to one worker, and the **core** data for the rest of the
universe - the numbers the score is built from - came in slower and at more risk. Context must
never cost a scored number. So the core fetch now does only what it did before (plus the trend
context, which needs no extra request), and this pass runs after it:

* its own small worker pool and per-request pacing;
* on a rate-limit error it pauses and halves its pace, and after repeated limits it stops;
* a time budget, after which it keeps what it has - a stock with no options or insider context
  that night shows "not fetched this run", which is honest and harmless.

Fields written are the same ``_ctx_opt_*`` and ``_ctx_insider`` the page reads. Display only.

THE OPTION HALF ALSO HAS AN HOUR PROBLEM - measured 2026-10-09. Yahoo serves the chain overnight
with ``bid``, ``ask`` and ``impliedVolatility`` all zero, so the 02:00 run spent ~1,000 requests
and produced a usable reading for **0 of 503** stocks, against 394 of 503 at 21:27 ET the evening
before. This pass therefore **probes** a couple of chains before fetching the rest: if quotes are
not being served it skips the option half entirely and reads ``options_cache`` instead, which the
after-close task fills. Insider fetching is unaffected - Form 4 data does not care what hour it is.
The probe rather than a hard-coded window, because a guessed clock goes wrong the day the source
changes and two requests cannot.
"""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

DEFAULT_BUDGET_SECONDS = 900
WORKERS = 2
PACE_SECONDS = 0.35
MAX_RATE_LIMITS = 4


def _is_rate_limit(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return "too many requests" in s or "rate limit" in s or "429" in s


def _usable_price(p) -> float | None:
    """This run's own price, if it is one. Lets the probe cost no extra quote request."""
    try:
        return float(p) if p is not None and np.isfinite(p) and float(p) > 0 else None
    except (TypeError, ValueError):
        return None


def _options_for(ticker: str, price: float | None, earnings_ts, today) -> dict:
    """One stock's option context, or ``{}``. Shared with the after-close refresh."""
    import yfinance as yf
    from context_signals import choose_expiry, options_context
    if price is None or not np.isfinite(price) or price <= 0:
        return {}
    t = yf.Ticker(ticker)
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


def _context_for(ticker: str, price: float | None, earnings_ts, today, with_options: bool = True) -> dict:
    import yfinance as yf
    from insider_activity import rows_from_yahoo
    out: dict = {}
    if with_options:
        out.update(_options_for(ticker, price, earnings_ts, today))
    t = yf.Ticker(ticker)
    out["_ctx_insider"] = json.dumps(rows_from_yahoo(t.insider_transactions, today), separators=(",", ":"))
    return out


def enrich(raw: list[dict], budget_seconds: float = DEFAULT_BUDGET_SECONDS, log=print,
           root=None, options_live=None) -> dict:
    """Add options and insider context to each fetched record, in place. Returns stats.

    ``options_live`` overrides the probe (tests, and a caller that already knows). When quotes are
    not being served, the option half is skipped and ``options_cache`` supplies the last reading
    from an hour when they were, labelled with its date; a stock with no cached reading gets
    ``_ctx_opt_status = "quotes-closed"``, which the page states plainly.
    """
    import options_cache

    started = time.time()
    today = datetime.now(timezone.utc).date()
    todo = [r for r in raw if "_error" not in r and r.get("Ticker")]
    done = failed = limited = 0
    pace = PACE_SECONDS
    stopped = None

    if options_live is None:
        # Probe only tickers this run already has a price for, so finding out costs no quote
        # request. Preferred names first, then whatever the universe offers.
        priced = {r["Ticker"]: _usable_price(r.get("price_latest")) for r in todo}
        candidates = [t for t in options_cache.PROBE_TICKERS if priced.get(t)]
        candidates += [t for t, p in priced.items() if p and t not in candidates]
        live, why = options_cache.quotes_are_live(
            lambda t: (_options_for(t, priced.get(t), None, today) or {}).get("_ctx_opt_status"),
            tickers=candidates)
    else:
        live, why = bool(options_live), "caller supplied"
    log(f"  Option quotes: {why}")

    if not live:
        cache = options_cache.load(root)
        served = 0
        for r in todo:
            got = options_cache.reading_for(cache, r["Ticker"], today)
            if got:
                r.update(got)
                served += 1
            else:
                r["_ctx_opt_status"] = options_cache.STATUS_CLOSED
        qd = cache.get("quote_date")
        log(f"  Options: skipped the fetch and read the cache - {served} of {len(todo)} stocks have a "
            f"reading from an hour when quotes were being served"
            + (f" (quote date {qd})" if qd else "") + ".")
        if not served:
            # Says the remedy rather than only the symptom: an empty cache looks identical to a
            # working one until someone notices every options panel is blank.
            log("  Options: the cache is empty, so no stock shows an expected move. It is filled by "
                "the 'Screener Option Quotes' task (weekdays 20:00, scripts/refresh-option-quotes.ps1); "
                "register it with scripts/register-tasks.ps1 if Get-ScheduledTask does not list it.")

    i = 0
    while i < len(todo):
        if time.time() - started > budget_seconds:
            stopped = "time budget"
            break
        chunk = todo[i:i + WORKERS * 5]
        i += len(chunk)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {}
            for r in chunk:
                futs[pool.submit(_context_for, r["Ticker"], r.get("price_latest"),
                                 r.get("earningsTimestampStart"), today, live)] = r
                time.sleep(pace)
            for f in as_completed(futs):
                r = futs[f]
                try:
                    # the cached option reading must survive the insider update
                    got = f.result()
                    r.update({k: v for k, v in got.items()
                              if live or not k.startswith("_ctx_opt")})
                    done += 1
                except Exception as e:  # noqa: BLE001 - context is optional by design
                    if _is_rate_limit(e):
                        limited += 1
                    else:
                        failed += 1
                        warnings.warn(f"{r['Ticker']}: context unavailable: {type(e).__name__}: {e}")
        if limited:
            if limited >= MAX_RATE_LIMITS:
                stopped = "rate limited"
                break
            time.sleep(30)
            pace = min(pace * 2, 3.0)
    stats = {"done": done, "failed": failed, "rate_limited": limited, "of": len(todo),
             "seconds": round(time.time() - started), "stopped": stopped,
             "options_live": bool(live), "options_why": why}
    log(f"  Context pass: {'options + ' if live else ''}insider context for {done} of {len(todo)} "
        f"stocks{f' (stopped: {stopped})' if stopped else ''}, {stats['seconds']}s")
    return stats
