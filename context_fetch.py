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


def _context_for(ticker: str, price: float | None, earnings_ts, today) -> dict:
    import yfinance as yf
    from context_signals import choose_expiry, options_context
    from insider_activity import rows_from_yahoo
    out: dict = {}
    t = yf.Ticker(ticker)
    if price is not None and np.isfinite(price) and price > 0:
        ed = None
        try:
            ed = datetime.fromtimestamp(int(earnings_ts), tz=timezone.utc).date() if earnings_ts else None
        except (TypeError, ValueError, OSError, OverflowError):
            ed = None
        exp, spans = choose_expiry(t.options, today, ed)
        if exp:
            ch = t.option_chain(exp)
            out.update(options_context(ch.calls, ch.puts, float(price), exp, today, spans))
    out["_ctx_insider"] = json.dumps(rows_from_yahoo(t.insider_transactions, today), separators=(",", ":"))
    return out


def enrich(raw: list[dict], budget_seconds: float = DEFAULT_BUDGET_SECONDS, log=print) -> dict:
    """Add options and insider context to each fetched record, in place. Returns stats."""
    started = time.time()
    today = datetime.now(timezone.utc).date()
    todo = [r for r in raw if "_error" not in r and r.get("Ticker")]
    done = failed = limited = 0
    pace = PACE_SECONDS
    stopped = None
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
                                 r.get("earningsTimestampStart"), today)] = r
                time.sleep(pace)
            for f in as_completed(futs):
                r = futs[f]
                try:
                    r.update(f.result())
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
             "seconds": round(time.time() - started), "stopped": stopped}
    log(f"  Context pass: options + insider context for {done} of {len(todo)} stocks"
        f"{f' (stopped: {stopped})' if stopped else ''}, {stats['seconds']}s")
    return stats
