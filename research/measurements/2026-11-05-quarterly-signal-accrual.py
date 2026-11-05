"""How the context layer's own gate would count a signal that only changes once a quarter.

For `research/2026-11-03-following-the-smart-money.md` section 3, which proposes logging each
stock's 13F holder count and its change on the prior quarter as a context-log candidate.

`context_eval` counts effective observations with `improvement_engine._effective_observations`
over the **run dates** it has logs for, spacing them 30 days apart. A 13F reading has **four
vintages a year** (one per quarter end, published 45 days later), so three consecutive monthly
windows read one and the same cross-section against three different forward months. This measures
the gap between the two counts.

Also measured, because the note rests on it: the cross-sectional dispersion of the one
institutional-ownership field the fetch already pulls.

Run: ``python research/measurements/2026-11-05-quarterly-signal-accrual.py``
Writes ``research/measurements/2026-11-05-quarterly-signal-accrual.json``.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import context_eval as ce                              # noqa: E402
import run_context                                     # noqa: E402
from improvement_engine import _effective_observations  # noqa: E402

OUT = Path(__file__).with_suffix(".json")

#: 13F deadline: 45 days after each quarter end (SEC rule 13f-1).
QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def weekdays(start: date, end: date) -> list[str]:
    """Every weekday between the two dates - the data loop's own cadence (Mon-Fri)."""
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def vintages(start: date, end: date) -> list[str]:
    """The filing dates a 13F-derived reading would actually change on, in the same window."""
    out = []
    for year in range(start.year - 1, end.year + 1):
        for m, dd in QUARTER_ENDS:
            published = date(year, m, dd) + timedelta(days=45)
            if start <= published <= end:
                out.append(published.isoformat())
    return sorted(out)


def accrual(years: int = 2) -> dict:
    """What the gate would read against what the signal actually provides."""
    end = date(2026, 11, 5)
    out = {}
    for n in range(1, years + 1):
        start = date(end.year - n, end.month, end.day)
        dates = weekdays(start, end)
        vs = vintages(start, end)
        eff = _effective_observations(dates, "1m")
        out[f"{n}y"] = {
            "run_dates_logged": len(dates),
            "effective_observations_the_gate_would_read": eff,
            "distinct_signal_vintages": len(vs),
            "overstatement_factor": round(eff / len(vs), 2) if vs else None,
            "first_vintage": vs[0] if vs else None,
            "last_vintage": vs[-1] if vs else None,
        }
    out["gate"] = {
        "effective_observations_required": ce.evaluate(write=False).get(
            "gate_effective_observations"),
        "horizon_days": [ce.HORIZON_MIN_DAYS, ce.HORIZON_MAX_DAYS],
        "non_overlap_span_days": 30,
    }
    # at four vintages a year, when would eight genuine observations exist?
    out["eight_genuine_vintages_from_today"] = vintages(date(2026, 11, 5), date(2029, 12, 31))[:8]
    return out


def ownership_dispersion() -> dict:
    """`heldPercentInstitutions`, the field the fetch already pulls, across the live universe.

    The note cites Gompers & Metrick (2001): institutions favour large, liquid stocks, so in the
    S&P 500 ownership is uniformly high and there is little to sort on. This is that claim on this
    screener's own data - and a check of whether the field is even a fraction.
    """
    df = pd.read_parquet(run_context.latest_run_dir() / "00_raw_fetch.parquet")
    s = pd.to_numeric(df["heldPercentInstitutions"], errors="coerce")
    q = s.dropna().quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return {
        "run": run_context.latest_run_dir().name,
        "coverage": [int(s.notna().sum()), int(len(s))],
        "quantiles": {str(k): round(float(v), 4) for k, v in q.items()},
        "mean": round(float(s.mean()), 4),
        "sd": round(float(s.std()), 4),
        "above_0_70": [int((s > 0.70).sum()), round(100 * float((s > 0.70).mean()), 1)],
        # a share of shares outstanding cannot exceed 1; this one does, for a quarter of the index
        "above_1_00": [int((s > 1.0).sum()), round(100 * float((s > 1.0).mean()), 1)],
    }


def main() -> dict:
    result = {"accrual": accrual(), "ownership": ownership_dispersion()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
