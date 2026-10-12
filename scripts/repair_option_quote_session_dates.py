"""Re-date the option-quote cache to the session its quotes came from, and re-derive the run's copy.

WHY THIS EXISTS - 2026-10-12. ``options_cache.refresh`` stamped the collecting machine's UTC
date. The refresh runs weekdays at 20:00 local = 21:00 Eastern, which is already the next UTC
day, so every stamp named the session *after* the one the quotes belonged to. The Friday
2026-10-09 refresh wrote ``quote_date 2026-10-10`` - a Saturday - and the 2026-10-12 02:00 run
published "Quotes are the close of Oct 10" for 455 of 501 stocks.
``research/measurements/2026-10-12-option-quote-session-date.py`` has the measurement;
``options_cache.market_session_date`` is the fix.

This repairs what the broken stamp left behind:

1. **The cache** (``data/options/quotes.json``, gitignored machine state). Its ``written`` field
   is a real UTC instant, so the session is recoverable exactly:
   ``market_session_date(written)``. Every reading's ``_ctx_opt_quote_date`` is restamped with
   it. Nothing else in the file is touched.

2. **The newest run's copy of the option fields**
   (``runs/<id>/00_raw_fetch.parquet``). This is *not* a hand-edit of a provenance artifact: the
   parquet's ``_ctx_opt_*`` columns are a copy of what ``options_cache.reading_for`` served the
   run from the cache, so they are **re-derived through that same function** against the
   corrected cache - exactly what the 02:00 pass would have produced. ``_ctx_opt_days`` is
   recounted from ``--as-of`` the same way.

Idempotent: a cache that already names sessions is left alone and reported as such.

    python scripts/repair_option_quote_session_dates.py            # report only
    python scripts/repair_option_quote_session_dates.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import options_cache  # noqa: E402

OPT_PREFIX = "_ctx_opt"


def newest_run_fetch(root: Path) -> Path | None:
    files = sorted((root / "runs").glob("*/00_raw_fetch.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def repair_cache(root: Path, apply: bool) -> tuple[str | None, str | None, int]:
    """Returns (old stamp, new stamp, readings restamped). New stamp is None if nothing to do."""
    cache = options_cache.load(root)
    old = cache.get("quote_date")
    written = cache.get("written")
    if not cache.get("readings"):
        return old, None, 0
    if options_cache.is_session_date(old):
        return old, None, 0

    try:
        at = datetime.strptime(str(written), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        print(f"  cannot recover the session: 'written' is {written!r}")
        return old, None, 0
    new = options_cache.market_session_date(at).strftime("%Y-%m-%d")

    readings = {}
    for ticker, entry in cache["readings"].items():
        e = dict(entry)
        if e.get("_ctx_opt_quote_date") == old or e.get("_ctx_opt_quote_date") is None:
            e["_ctx_opt_quote_date"] = new
        readings[ticker] = e
    if apply:
        options_cache.save(readings, new, root)
    return old, new, len(readings)


def repair_run(root: Path, as_of: date, apply: bool) -> tuple[int, int]:
    """Re-derive the newest run's option columns from the cache. Returns (served, of)."""
    import pandas as pd

    p = newest_run_fetch(root)
    if p is None:
        print("  no runs/*/00_raw_fetch.parquet - nothing to re-derive")
        return 0, 0
    df = pd.read_parquet(p)
    if "Ticker" not in df.columns:
        return 0, 0
    cache = options_cache.load(root)
    opt_cols = [c for c in df.columns if c.startswith(OPT_PREFIX)]

    served = 0
    updates: dict[int, dict] = {}
    for i, ticker in df["Ticker"].items():
        got = options_cache.reading_for(cache, str(ticker), as_of)
        if got:
            served += 1
            updates[i] = {k: v for k, v in got.items() if k.startswith(OPT_PREFIX)}
        else:
            updates[i] = {"_ctx_opt_status": options_cache.STATUS_CLOSED}

    if apply:
        for col in opt_cols:
            if col != "_ctx_opt_status":
                df[col] = None
        for i, vals in updates.items():
            for col in opt_cols:
                if col in vals:
                    df.at[i, col] = vals[col]
            if "_ctx_opt_status" in df.columns:
                df.at[i, "_ctx_opt_status"] = vals.get(
                    "_ctx_opt_status", options_cache.STATUS_CLOSED)
        df.to_parquet(p, index=False)
    print(f"  {p.parent.name}/00_raw_fetch.parquet: {served} of {len(df)} stocks get a reading")
    return served, len(df)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the changes")
    ap.add_argument("--root", default=".")
    ap.add_argument("--as-of", default=None,
                    help="the date the reading is read back on (default: today)")
    a = ap.parse_args()
    root = Path(a.root)
    as_of = date.fromisoformat(a.as_of) if a.as_of else datetime.now().date()

    print(f"Option quote session dates{'' if a.apply else '  (report only)'}")
    print("1. cache")
    old, new, n = repair_cache(root, a.apply)
    if new is None:
        print(f"  quote_date {old!r} already names a session (or there is nothing cached)")
    else:
        print(f"  quote_date {old} -> {new}; {n} readings restamped"
              f"{'' if a.apply else ' (not written)'}")

    print(f"2. newest run, re-derived through reading_for as of {as_of}")
    repair_run(root, as_of, a.apply)

    if a.apply:
        c = options_cache.load(root)
        bad = [t for t, e in (c.get("readings") or {}).items()
               if not options_cache.is_session_date(e.get("_ctx_opt_quote_date"))]
        print(f"3. verify: cache quote_date={c.get('quote_date')}, "
              f"{len(bad)} readings still naming a non-session day")
        return 1 if bad else 0
    print("\nRe-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
