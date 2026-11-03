"""The Growth weight at which the ranking would be growth-neutral, every published run.

`research/2026-11-02-growth-exposure-drift.md` reads three weeks: neutral at 13.5 on 2026-10-13,
at 9.6 on 2026-11-02, and calls it a drift away from a neutral point. Three weeks is the window
in which the 2026-10-13 review was made; the record goes back to August. This runs that note's
own sweep (`2026-10-13-growth-weight-sweep.py`, imported, not reimplemented) over every committed
payload, so the question "was 13% ever the neutral point" is answered from the whole record
rather than from the fortnight after the review.

The other 87 points are held as `config.yaml` has them **today**, for every run - the same choice
the 11-02 note made when it pointed the sweep at the 10-30 and 11-02 payloads. It is what makes
the runs comparable; it also means a historical row is "what this run's cross-section would do
under today's weights", not a reconstruction of that morning's configuration.

    python -X utf8 research/measurements/2026-11-03-neutral-weight-history.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]

_spec = importlib.util.spec_from_file_location(
    "sweep", ROOT / "research" / "measurements" / "2026-10-13-growth-weight-sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def commits() -> list[tuple[str, str]]:
    out = subprocess.run(["git", "log", "--format=%H\t%ad\t%s", "--date=short", "--",
                          "dashboard_data.js"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    rows = []
    for line in out.splitlines():
        sha, date, subject = line.split("\t", 2)
        if subject.startswith("data: screener run"):
            rows.append((sha, date + (" (evening)" if "evening" in subject else "")))
    return list(reversed(rows))


def scores_at(sha: str) -> pd.DataFrame | None:
    raw = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"], cwd=ROOT,
                         capture_output=True, encoding="utf-8", errors="replace").stdout
    if not raw or "{" not in raw:
        return None
    try:
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        table = pd.DataFrame(data["table_data"]).set_index("Ticker")
    except (ValueError, KeyError):
        return None
    cols = [c + "_score" for c in CATS]
    if any(c not in table.columns for c in cols) or "Composite" not in table.columns:
        return None
    s = table[cols].apply(pd.to_numeric, errors="coerce")
    s.columns = CATS
    return s.assign(Composite=pd.to_numeric(table["Composite"], errors="coerce"))


def neutral_weight(s: pd.DataFrame, cfg: dict) -> tuple[float | None, float]:
    """(the Growth weight at which exposure crosses zero, the exposure at the configured weight),
    freed points to Quality - the note's headline column."""
    rows = []
    for g in range(0, 27):
        w = {**cfg, "growth": g, "quality": cfg["quality"] + (cfg["growth"] - g)}
        rows.append((g, sweep.blend(s, w).corr(s["growth"], method="spearman")))
    at_cfg = dict(rows)[cfg["growth"]]
    for (g0, e0), (g1, e1) in zip(rows, rows[1:]):
        if (e0 <= 0 <= e1) or (e1 <= 0 <= e0):
            return (float(g0) if e0 == e1 else g0 + (-e0 / (e1 - e0))), at_cfg
    return None, at_cfg


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]
    print(f"other seven weights held at today's config; Growth configured at {cfg['growth']}\n")
    print(f"{'run':<22}{'neutral weight':>15}{'exposure at 13':>16}")
    for sha, label in commits():
        s = scores_at(sha)
        if s is None:
            print(f"{label:<22}   (no category scores in this payload)")
            continue
        nw, at = neutral_weight(s, cfg)
        print(f"{label:<22}{'  -  ' if nw is None else f'{nw:>15.1f}'}{at:>+16.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
