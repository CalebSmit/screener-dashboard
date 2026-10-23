"""Second reading of Factor Lab experiment 17 (`piotroski`), 2026-10-23.

`OWNER_FOCUS.md` 2026-10-20 item 2: every new lab experiment needs a reader who did not build it -
the paper reopened, each reference figure re-read against `factor_lab.py`, and the tolerances
checked in `git log` to have been committed before the comparison ran.

This script is the *independent* half: it recomputes the experiment's four computed figures and the
verdict note's value-weighted pair straight from the Open Source Asset Pricing files, with no import
of `factor_lab`, so a transcription or construction error in the experiment would show up as a
disagreement rather than be reproduced. Paper-side verification is in
`research/2026-10-20-lab-osap.md`, "Second reading (2026-10-23)".

Run:  python research/measurements/2026-10-23-piotroski-second-reading.py
Needs the gitignored `data/lab/osap/` library files (`factor_lab.py --fetch` writes them).
"""

from __future__ import annotations

import csv
import io
import json
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OSAP = ROOT / "data" / "lab" / "osap"
OUT = Path(__file__).with_suffix(".json")

START, END = "1976-01", "1996-12"          # SignalDoc's SampleStartYear/SampleEndYear for PS
SIGNAL = "PS"


def vw_series() -> dict[str, tuple[float, float]]:
    """{YYYY-MM: (lowest-score decile, highest-score decile)} from the authors' own cut."""
    out: dict[str, tuple[float, float]] = {}
    with (OSAP / "PS_OriginalCuts_VW.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lo, hi = row["port01"], row["port10"]
            if lo not in ("", "NA") and hi not in ("", "NA"):
                out[row["date"][:7]] = (float(lo), float(hi))
    return out


def ew_rows() -> tuple[dict[str, tuple[float, float]], dict[str, list[float]]]:
    """The same deciles equally weighted, from the alternative-portfolio set (long format).

    Returns the return pairs and, separately, each end decile's average *score* - the file carries
    it as ``signallag``, which is what the verdict note's "averaged a score of 2.7 / 7.5" asserts.
    """
    by_month: dict[str, dict[str, float]] = {}
    lags: dict[str, list[float]] = {"01": [], "10": []}
    with zipfile.ZipFile(OSAP / "PredictorAltPorts_DecilesEW.zip") as z:
        with z.open(z.namelist()[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            for row in csv.DictReader(text):
                if row.get("signalname") != SIGNAL:
                    continue
                port, ret, month = str(row["port"]).zfill(2), row.get("ret"), row["date"][:7]
                if ret not in (None, "", "NA"):
                    by_month.setdefault(month, {})[port] = float(ret)
                lag = row.get("signallag")
                if port in lags and lag not in (None, "", "NA") and START <= month <= END:
                    lags[port].append(float(lag))
    out = {m: (p["01"], p["10"]) for m, p in by_month.items() if "01" in p and "10" in p}
    return out, lags


def summarise(series: dict[str, tuple[float, float]], start: str = START, end: str = END) -> dict:
    months = sorted(m for m in series if start <= m <= end)
    lo = [series[m][0] for m in months]
    hi = [series[m][1] for m in months]
    sp = [h - l for h, l in zip(hi, lo)]
    mean_sp = statistics.fmean(sp)
    se = statistics.stdev(sp) / len(sp) ** 0.5
    return {
        "n_months": len(months),
        "first": months[0], "last": months[-1],
        "mean_lowest_pct_year": round(statistics.fmean(lo) * 12, 4),
        "mean_highest_pct_year": round(statistics.fmean(hi) * 12, 4),
        "spread_pct_year": round(mean_sp * 12, 4),
        "t_spread": round(mean_sp / se, 4),
    }


def main() -> int:
    ew, lags = ew_rows()
    result = {
        "window": [START, END],
        "paper_table3B": {"low": 7.8, "high": 31.3, "spread": 23.5, "t": 5.594,
                          "n_high_bm_firm_years": 14043, "n_low": 396, "n_high": 1448,
                          "source": "Piotroski (2000), Table 3 Panel B, printed p. 17"},
        "paper_table3A_for_contrast": {"low": -9.6, "high": 13.4, "spread": 23.0, "t": 5.590,
                                       "note": "one-year MARKET-ADJUSTED; OSAP's SignalDoc names "
                                               "Table 3A as the key table but quotes Panel B's "
                                               "t-statistic 5.594 (Panel A's is 5.590)"},
        "equally_weighted": summarise(ew),
        "value_weighted": summarise(vw_series()),
        "mean_score_in_window": {"lowest_decile": round(statistics.fmean(lags["01"]), 4),
                                 "highest_decile": round(statistics.fmean(lags["10"]), 4)},
        # research/2026-10-20-lab-osap.md section 7 asserts two things about the other periods,
        # which the four tolerance figures do not cover: that the value-weighted gap was the larger
        # "in every period", and that since publication neither weighting is distinguishable from
        # zero ("t 1.38 and -0.55"). Both are re-measured here rather than taken from the note.
        "other_periods": {
            label: {"ew": summarise(ew, a, b), "vw": summarise(vw_series(), a, b)}
            for label, a, b in (("after_the_paper", "1997-01", "9999-99"),
                                ("since_2007", "2007-01", "9999-99"),
                                ("everything", "0000-00", "9999-99"))
        },
    }
    print(json.dumps(result, indent=2))
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
