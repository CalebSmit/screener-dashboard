"""Collapse the regeneration diff into distinct path classes.

The discipline this serves: a change to the generator must be shown to move only
what it intended to move, and `scripts/diff_payload.py --group` still prints one
line per ticker for the per-stock history paths. This folds those together so the
*classes* of change are countable by eye.

Usage: python research/measurements/2026-10-14-regen-diff-classes.py [before] [after]
Reads two payload files. No network, writes nothing.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys

BEFORE = sys.argv[1] if len(sys.argv) > 1 else "dashboard_data.js"
AFTER = sys.argv[2] if len(sys.argv) > 2 else "logs/rehearse-publish/site/dashboard_data.js"

LINE = re.compile(r"^\s+(added|changed|removed|unchanged)\s+(\d+)\s+(\S+)")
TICKER_IN_HISTORY = re.compile(r"^history\.\*\.[A-Za-z0-9.\-]+\.")


def main() -> None:
    out = subprocess.run(
        [sys.executable, "scripts/diff_payload.py", BEFORE, AFTER, "--group"],
        capture_output=True, text=True, check=False).stdout
    lines = out.splitlines()
    for ln in lines[:12]:
        print(ln)
    try:
        start = lines.index("By generalised path:")
    except ValueError:
        print("\ndiff_payload produced no grouped section")
        return

    buckets: collections.Counter = collections.Counter()
    for ln in lines[start + 1:]:
        m = LINE.match(ln)
        if not m:
            break
        kind, n, path = m.group(1), int(m.group(2)), m.group(3)
        buckets[(kind, TICKER_IN_HISTORY.sub("history.*.<TICKER>.", path))] += n

    print("\nDistinct classes of change:")
    for (kind, path), n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<10}{n:>6}  {path}")
    print(f"\n{len(buckets)} classes in total.")


if __name__ == "__main__":
    main()
