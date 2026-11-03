"""Is the growth-exposure drift the market, or is it the screener's own edits?

`research/2026-11-02-growth-exposure-drift.md` section 3 records the one thing it did not
separate: "Five changelog entries changed how scores are computed between 10-14 and 10-21, mostly
for banks; they may account for part of it. This note does not separate the two." The proposal's
three options turn on that answer - re-fitting the weight to a neutral point the screener itself
moved would bake in its own edits.

This reads every committed `dashboard_data.js` and, per run, computes:

  * `rho`           - Spearman(Composite, Growth score), the exposure the note tracks
  * `rho_ex_fin`    - the same with Financials removed, which is the cohort the 10-14..10-21
                      changelog entries rescored (bank P/B on common equity, ROE and ROA on net
                      income to common)
  * `gv`, `gv_ex_fin` - Spearman(Growth, Valuation), the offset the note says weakened
  * `mv`            - `kpis.methodology_version`, so a run's scoring rules are named rather than
                      inferred from the date

If the drift is the screener's own edits, it is concentrated in Financials and `rho_ex_fin` is
flat. If it is the cross-section, both move together.

    python -X utf8 research/measurements/2026-11-03-growth-drift-decomposition.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def commits() -> list[tuple[str, str]]:
    out = _git("log", "--format=%H\t%ad\t%s", "--date=short", "--", "dashboard_data.js")
    rows = []
    for line in out.splitlines():
        sha, date, subject = line.split("\t", 2)
        if subject.startswith("data: screener run"):
            rows.append((sha, date))
    return list(reversed(rows))


def payload(sha: str) -> dict | None:
    raw = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"], cwd=ROOT,
                         capture_output=True, encoding="utf-8", errors="replace").stdout
    if not raw or "=" not in raw:
        return None
    try:
        return json.loads(raw[raw.index("=") + 1:].rstrip().rstrip(";"))
    except ValueError:
        return None


def rho(rows: list[dict], a: str, b: str) -> float | None:
    xs = [(r.get(a), r.get(b)) for r in rows]
    xs = [(x, y) for x, y in xs if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(xs) < 50:
        return None
    return float(spearmanr([x for x, _ in xs], [y for _, y in xs]).statistic)


def main() -> int:
    print(f"{'run':<12}{'n':>5}{'nFin':>6}{'rho':>9}{'rho_exFin':>11}"
          f"{'gv':>9}{'gv_exFin':>10}  methodology_version")
    prev_mv = None
    for sha, date in commits():
        d = payload(sha)
        if not d:
            print(f"{date:<12}  (payload unreadable at {sha[:8]})")
            continue
        rows = d.get("table_data") or []
        # the payload's column names have moved over the record; take whichever is present
        gkey = "Growth" if rows and "Growth" in rows[0] else "growth_score"
        vkey = "Valuation" if rows and "Valuation" in rows[0] else "valuation_score"
        ckey = "Composite" if rows and "Composite" in rows[0] else "composite"
        if not rows or gkey not in rows[0] or ckey not in rows[0]:
            print(f"{date:<12}  (no growth/composite column at {sha[:8]})")
            continue
        ex = [r for r in rows if r.get("Sector") != "Financials"]
        mv = str((d.get("kpis") or {}).get("methodology_version") or "-")[:12]
        mark = " <- scoring rules changed" if prev_mv and mv != prev_mv else ""
        prev_mv = mv

        def f(v):
            return "     -   " if v is None else f"{v:+9.4f}"

        print(f"{date:<12}{len(rows):>5}{len(rows) - len(ex):>6}"
              f"{f(rho(rows, ckey, gkey))}{f(rho(ex, ckey, gkey)):>11}"
              f"{f(rho(rows, gkey, vkey))}{f(rho(ex, gkey, vkey)):>10}  {mv}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
