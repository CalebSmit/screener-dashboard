"""When did the context layer stop reaching the published payload?

Measured 2026-10-12. The live `dashboard_data.js` carries no `ctx` on any of 501 stocks and
no `market` / `track` keys at all, so the "Before you decide" panel, the market backdrop and
the track record - the owner's 2026-10-08 item - show nothing on the public site.

This walks the last N commits that touched `dashboard_data.js` and reports, per commit, how
many stocks carried a `ctx` dict and whether the two top-level context keys were present.

Run from the repo root:  python research/measurements/2026-10-12-context-layer-published.py
"""

from __future__ import annotations

import json
import subprocess
import sys

N = 20


def payload_at(commit: str) -> dict | None:
    try:
        raw = subprocess.run(
            ["git", "show", f"{commit}:dashboard_data.js"],
            capture_output=True, check=True,
        ).stdout.decode("utf-8", "replace")
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(raw.split("=", 1)[1].strip().rstrip(";"))
    except (ValueError, IndexError):
        return None


def main() -> int:
    commits = subprocess.run(
        ["git", "log", f"-n{N}", "--format=%H\t%ad\t%s", "--date=short", "--", "dashboard_data.js"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace").splitlines()

    print(f"{'date':<11} {'ctx stocks':>12}  {'market':>6} {'track':>5}  subject")
    print("-" * 92)
    for line in commits:
        sha, dt, subject = line.split("\t", 2)
        d = payload_at(sha)
        if d is None:
            print(f"{dt:<11} {'unparseable':>12}  {'-':>6} {'-':>5}  {subject[:44]}")
            continue
        sd = d.get("stock_detail") or {}
        n_ctx = sum(1 for v in sd.values() if v.get("ctx"))
        print(f"{dt:<11} {f'{n_ctx} of {len(sd)}':>12}  "
              f"{'Y' if 'market' in d else 'N':>6} {'Y' if 'track' in d else 'N':>5}  {subject[:44]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
