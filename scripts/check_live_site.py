"""After the data loop pushes, is the live site actually serving that run? Never fails a run.

WHY THIS EXISTS - 2026-11-17. A push to ``main`` is not a published page: GitHub Pages builds and
deploys it afterwards, on its own, and nothing in either loop looked at the site once the push
returned. The data loop logged "Published to main - live dashboard refreshed." on the strength of
``git push`` exiting 0. When a Pages deployment failed or hung - the owner received GitHub's
"run failed" emails in early November and had to ask whether they were true - the morning brief
still said the dashboard had been refreshed. The owner's question on 2026-11-10, "why did the
dashboard not update", had no answer in any log.

The check is the one a reader would make: open the published page and look at which run it shows.
``dashboard_data.js`` opens with ``window.SCREENER_DATA = {"kpis": {"run_timestamp": ...`` - the
stamp is in the first 60 bytes - so this asks the live site for its first 400 bytes (an HTTP range
request, a few hundred bytes a poll rather than the 5 MB payload) and compares that stamp with the
one the loop just committed. Each request carries a throwaway query string, because Pages serves
through a CDN with ``Cache-Control: max-age=600`` and a cached copy would otherwise read as stale
for ten minutes after a good deploy.

It waits up to ``BUDGET_S`` (20 minutes; a Pages deploy normally takes one to three), polling every
``INTERVAL_S``. Output is one line, and it always exits 0: the push has already happened, so a
slow or failed deploy is something to *say* - in the brief, with where to look - not a reason to
discard the evidence the run collected. ``ATTENTION - LIVE SITE`` is the marker the brief reads.

The same reasoning, from the article the owner shared that day on running a site with an agent:
"Open each published URL to confirm the intended change appeared."
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://calebsmit.github.io/screener-dashboard/"   # == run_screener.DASHBOARD_URL (tested)
PAYLOAD = "dashboard_data.js"
DEPLOYS = "https://github.com/CalebSmit/screener-dashboard/actions"
HEAD_BYTES = 400
BUDGET_S = 20 * 60
INTERVAL_S = 30

STALE = "ATTENTION - LIVE SITE NOT UPDATED"
STAMP = re.compile(r'"run_timestamp"\s*:\s*"([^"]+)"')


def stamp_of(text: str | None) -> str | None:
    """The run_timestamp a payload's opening bytes name, or None."""
    if not text:
        return None
    m = STAMP.search(text[:HEAD_BYTES * 2])
    return m.group(1) if m else None


def local_stamp(path: Path | None = None) -> str | None:
    path = path or ROOT / PAYLOAD
    try:
        with open(path, encoding="utf-8") as fh:
            return stamp_of(fh.read(HEAD_BYTES))
    except OSError:
        return None


def fetch_head(url: str = SITE + PAYLOAD, timeout: float = 20.0) -> str | None:
    """The live payload's first bytes, past the CDN cache; None on any failure."""
    req = urllib.request.Request(f"{url}?live-check={int(time.time())}",
                                 headers={"Range": f"bytes=0-{HEAD_BYTES - 1}",
                                          "User-Agent": "screener-live-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(HEAD_BYTES * 2).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - any failure is "could not read", never a crash
        return None


def wait_for(expected: str, fetch: Callable[[], str | None] = fetch_head,
             budget_s: float = BUDGET_S, interval_s: float = INTERVAL_S,
             clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep) -> tuple[str, str | None, float]:
    """Poll until the live stamp equals ``expected`` or the budget runs out.

    Returns (state, last stamp seen, seconds waited); state is 'live', 'stale' (the site answered
    but with another run) or 'unreachable' (it never answered at all)."""
    start = clock()
    seen: str | None = None
    while True:
        got = stamp_of(fetch())
        if got is not None:
            seen = got
            if got == expected:
                return "live", seen, clock() - start
        if clock() - start + interval_s > budget_s:
            return ("stale" if seen is not None else "unreachable"), seen, clock() - start
        sleep(interval_s)


def message(state: str, expected: str | None, seen: str | None, waited: float) -> str:
    mins = waited / 60
    if state == "live":
        return f"Live site: serving this run ({expected}), {waited:.0f} s after the push."
    if state == "stale":
        return (f"{STALE} - {mins:.0f} min after the push the site still serves run {seen}, "
                f"not this run ({expected}). The commit is on main; the Pages deployment did not "
                f"publish it. Look at {DEPLOYS} and re-run the failed deployment.")
    if state == "unreachable":
        return (f"Live site: could not be read for {mins:.0f} min after the push "
                f"(network or GitHub) - not checked, nothing implied.")
    return "Live site: not checked - this run's own payload has no run_timestamp."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=BUDGET_S, help="seconds to wait (default 1200)")
    ap.add_argument("--interval", type=float, default=INTERVAL_S, help="seconds between polls")
    args = ap.parse_args(argv)
    try:
        expected = local_stamp()
        if expected is None:
            print(message("no-stamp", None, None, 0))
            return 0
        state, seen, waited = wait_for(expected, budget_s=args.budget, interval_s=args.interval)
        print(message(state, expected, seen, waited))
    except Exception as exc:  # noqa: BLE001 - this check may never fail its caller
        print(f"Live site: could not check ({type(exc).__name__}) - not checked, nothing implied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
