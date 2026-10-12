"""The option-quote stamp named the wrong session, and what it cost.

Measured 2026-10-12.

The `Screener Option Quotes` task runs weekdays at 20:00 local (21:00 Eastern), and
`options_cache.refresh` stamped `datetime.now(timezone.utc).date()` - which has already rolled
over to the next day. So every refresh named the session *after* the one its quotes came from.

The Friday 2026-10-09 refresh wrote `quote_date 2026-10-10` (cache header: `written
2026-10-10T01:03:01Z`), and the 2026-10-12 02:00 run published that date for 455 of 501 stocks.
2026-10-10 was a Saturday. The drilldown sentence is: "Quotes are the close of <date>; options
are not quoted at the hour this run fetches, so the last session's close is used."

Run from the repo root:
    python research/measurements/2026-10-12-option-quote-session-date.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import options_cache as oc  # noqa: E402

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def published_quote_dates() -> Counter:
    """What the live context payload tells readers, per stock."""
    p = Path("dashboard_context.js")
    if not p.exists():
        return Counter()
    d = json.loads(p.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
    return Counter(v.get("oqd") for v in (d.get("ctx") or {}).values())


def main() -> int:
    print("1. The stamp, before and after")
    print("   collected at (UTC)        old stamp (UTC date)   new stamp (session)")
    cases = [
        datetime(2026, 10, 10, 1, 3, 1, tzinfo=timezone.utc),   # the real Friday refresh
        datetime(2026, 10, 14, 1, 3, 0, tzinfo=timezone.utc),   # a Tuesday evening refresh
        datetime(2026, 10, 10, 18, 0, 0, tzinfo=timezone.utc),  # a hand-run Saturday
        datetime(2026, 10, 11, 18, 0, 0, tzinfo=timezone.utc),  # a hand-run Sunday
        datetime(2026, 10, 12, 14, 0, 0, tzinfo=timezone.utc),  # a weekday morning run
    ]
    for t in cases:
        old = t.date()
        new = oc.market_session_date(t)
        flag = "" if new == old else "   <- differs"
        print(f"   {t:%Y-%m-%d %H:%M}Z           {old} ({DAYS[old.weekday()]})      "
              f"{new} ({DAYS[new.weekday()]}){flag}")

    print("\n2. What the live site published (dashboard_context.js as committed)")
    pub = published_quote_dates()
    if not pub:
        print("   dashboard_context.js not present")
    for d, n in pub.most_common():
        if d is None:
            print(f"   {n:>4} stocks  no quote date (quotes-closed)")
            continue
        wd = date.fromisoformat(d).weekday()
        note = "  <- NOT A TRADING SESSION" if wd > 4 else ""
        print(f"   {n:>4} stocks  {d} ({DAYS[wd]}){note}")

    print("\n3. The staleness bound the off-by-one produced")
    print("   Quotes from the Tuesday 2026-10-13 close - a mid-week case, so both stamps are")
    print("   weekdays and the weekend tripwire plays no part. Read on Oct 14-19:")
    exp = "2026-12-18"
    for label, stamp in (("session date (fixed)", "2026-10-13"), ("UTC date (old)", "2026-10-14")):
        served = []
        for day in range(14, 20):
            cache = {"readings": {"AAA": {"_ctx_opt_quote_date": stamp,
                                          "_ctx_opt_expiry": exp,
                                          "_ctx_opt_status": "ok"}}}
            got = oc.reading_for(cache, "AAA", date(2026, 10, day))
            served.append(f"{day:02d}:{'served ' if got else 'refused'}")
        print(f"   stamped {stamp} {label:<22} {' '.join(served)}")
    print(f"   MAX_AGE_DAYS = {oc.MAX_AGE_DAYS}: the old stamp served the close for four calendar")
    print("   days, not three. The Friday case is refused outright now - a Saturday stamp names")
    print("   no session, so the weekend tripwire in reading_for declines to describe it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
