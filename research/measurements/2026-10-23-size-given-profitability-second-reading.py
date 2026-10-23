"""Second reading of Factor Lab experiment 16 (`size-given-profitability`), 2026-10-23.

`OWNER_FOCUS.md` 2026-10-20 item 2. The paper-side half - every one of the seventeen reference
figures re-read in Asness, Frazzini, Israel, Moskowitz & Pedersen (2018), Table 3 Panel A, p. 495 -
is recorded in `research/2026-10-20-size-weight.md`, "Second reading (2026-10-23)". This script is
the independent half: it recomputes the experiment's measured figures, and the pre-registered
criterion that decides whether the Size weight reopens, straight from the Kenneth French library
file, with **no import of `factor_lab`**. A transcription or construction error in the experiment
would show up here as a disagreement rather than be reproduced.

Run:  python research/measurements/2026-10-23-size-given-profitability-second-reading.py
Needs the gitignored `data/lab/french/25_Portfolios_ME_OP_5x5_CSV.zip` (`factor_lab.py --fetch`).
"""

from __future__ import annotations

import json
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ZIP = ROOT / "data" / "lab" / "french" / "25_Portfolios_ME_OP_5x5_CSV.zip"
OUT = Path(__file__).with_suffix(".json")

BLOCK = "Average Value Weighted Returns -- Monthly"
PAPER_START, PAPER_END = "196307", "201212"     # the library's first profitability month onward
MISSING = (-99.99, -999.0)


def monthly() -> dict[str, list[float]]:
    """{YYYYMM: [25 returns]} from the value-weighted monthly block, library column order.

    Column ``5 * s + p`` is size quintile ``s`` (0 smallest) by profitability quintile ``p``
    (0 weakest) - the header order SMALL LoOP, ME1 OP2, ... BIG HiOP.
    """
    with zipfile.ZipFile(ZIP) as z:
        text = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    out: dict[str, list[float]] = {}
    inside = False
    for line in text:
        s = line.strip()
        if not inside:
            inside = s.startswith(BLOCK)
            continue
        parts = [p.strip() for p in s.split(",")]
        if not parts or not parts[0]:
            continue
        if not (parts[0].isdigit() and len(parts[0]) == 6):
            if out:                     # the next block's title ends this one
                break
            continue
        vals = [float(v) for v in parts[1:26]]
        if len(vals) == 25 and not any(v in MISSING for v in vals):
            out[parts[0]] = vals
    return out


def stat(series: list[float]) -> dict:
    m = statistics.fmean(series)
    se = statistics.stdev(series) / len(series) ** 0.5
    return {"mean": round(m, 4), "t": round(m / se, 4), "n_months": len(series)}


def window(data: dict[str, list[float]], start: str, end: str) -> list[list[float]]:
    return [data[m] for m in sorted(data) if start <= m <= end]


def measure(rows: list[list[float]]) -> dict:
    """Every figure the experiment computes, from the 25 columns of each month."""
    def col(s: int, p: int) -> list[float]:
        return [r[5 * s + p] for r in rows]

    small_minus_big = {p: [a - b for a, b in zip(col(0, p), col(4, p))] for p in range(5)}
    four_minus_big = {p: [a - b for a, b in zip(col(3, p), col(4, p))] for p in range(5)}
    smbq = [statistics.fmean([small_minus_big[p][i] for p in range(5)]) for i in range(len(rows))]
    fourq = [statistics.fmean([four_minus_big[p][i] for p in range(5)]) for i in range(len(rows))]
    qj = {s: [a - b for a, b in zip(col(s, 4), col(s, 0))] for s in (0, 4)}
    return {
        "small_minus_big_by_profitability_quintile": {f"p{p + 1}": stat(v)
                                                      for p, v in small_minus_big.items()},
        "smbq_average_of_the_five": stat(smbq),
        "fourth_size_group_minus_largest_average": stat(fourq),
        "quality_minus_junk_largest_size_group": stat(qj[4]),
        "quality_minus_junk_smallest_size_group": stat(qj[0]),
    }


def main() -> int:
    data = monthly()
    months = sorted(data)
    paper = window(data, PAPER_START, PAPER_END)
    result = {
        "file": "25_Portfolios_ME_OP_5x5_CSV.zip (202608 CRSP vintage)",
        "months_available": [months[0], months[-1], len(months)],
        # Asness et al. (2018) Table 3 Panel A, p. 495, July 1957 - December 2012, re-read from the
        # publisher's open-access PDF on 2026-10-23. Small - Big column and its t-statistics; the
        # Quality - Junk row; SMBQ from the text on the same page; the fourth-size-group figure
        # derived as the mean of columns "4" minus "Big" across the five rows.
        "paper_table3A": {
            "small_minus_big": {"junk": 0.23, "p2": 0.46, "p3": 0.54, "p4": 0.42, "quality": 0.44},
            "small_minus_big_t": {"junk": 1.21, "p2": 3.02, "p3": 3.44, "p4": 2.82, "quality": 2.87},
            "smbq": 0.42, "smbq_t": 3.18,
            "fourth_minus_big_rowwise": [0.28, 0.19, 0.25, 0.30, 0.25],
            "fourth_minus_big_average": 0.254,
            "quality_minus_junk_big": 0.42, "quality_minus_junk_big_t": 2.78,
            "quality_minus_junk_small": 0.62, "quality_minus_junk_small_t": 4.78,
        },
        "measured_over_the_papers_window": {"window": [PAPER_START, PAPER_END], **measure(paper)},
        # The pre-registered rule from research/2026-10-20-size-weight.md section 9 item 1, which
        # the experiment prints beside the comparison: size quintile 4 minus 5 inside each
        # profitability quintile, value-weighted, averaged across the five, 1963-2026. Refuted if
        # the average is no larger than the unconditional 0.136% a month, or its t is below 1.5.
        "criterion_1963_to_now": {
            "unconditional_spread": 0.136, "min_t": 1.5,
            **measure(window(data, "196307", "202612"))["fourth_size_group_minus_largest_average"],
        },
        "criterion_since_2013": measure(window(data, "201301", "209912"))[
            "fourth_size_group_minus_largest_average"],
    }
    print(json.dumps(result, indent=2))
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
