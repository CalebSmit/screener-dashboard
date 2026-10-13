"""Factor Lab: the papers the screener cites, rerun on public data.

The screener's methodology rests on published research (CLAUDE.md "Your mandate"), and until
now a reader had to take each citation on trust. The lab reproduces a paper's headline result
from data anyone can download, publishes how far the reproduction lands from the paper, and
lets the reader change the conditions in the browser (``lab.html``).

Every experiment carries the same four things, and the page shows all four:

* the **claim** being tested, with its source;
* the **conditions** used - data, sample, estimator, settings;
* the **difference from the reference result**, figure by figure, against tolerances written
  down here rather than chosen on the page; and
* the **details still needed** to call it verified.

What this module is NOT, and must not become:

* **Not evidence for the screener's own parameters.** A slider that finds a better-looking
  setting is how backtests lie. Methodology stays research-led (CLAUDE.md rules 4 and 5); the
  lab shows how a published result behaves, including when it weakens.
* **Not a backtest of the screener.** It reruns the paper's own long-short factor portfolios,
  with no costs and no connection to the ranking the site publishes.
* **Not imported by scoring code.** ``tests/test_factor_lab.py`` enforces it.

Data: Kenneth R. French's Data Library (copyright Eugene F. Fama and Kenneth R. French). The
raw files are downloaded to ``data/lab/french/`` (gitignored) and never committed; what is
published in ``lab_data.js`` is the derived monthly series the page needs, with attribution.

    python factor_lab.py --fetch     # download the source files (network)
    python factor_lab.py --build     # rebuild lab_data.js from them
    python factor_lab.py --report    # print the comparison with the paper
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "lab" / "french"
OUT = ROOT / "lab_data.js"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_HOME = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
FRENCH_FILES = {
    "momentum_monthly": ("10_Portfolios_Prior_12_2_CSV.zip", "Value Weight Returns -- Monthly"),
    "momentum_daily": ("10_Portfolios_Prior_12_2_Daily_CSV.zip", "Average Value Weighted Returns -- Daily"),
}
MISSING = -99.0                      # the library marks missing data -99.99 or -999

# Trading-day windows the page offers for the volatility forecast. 126 is the paper's.
VOL_WINDOWS = (21, 63, 126, 252)


# --------------------------------------------------------------------------- the data
def fetch(raw_dir: Path = RAW_DIR, timeout: float = 120.0) -> list[Path]:
    """Download the source files. Network; called by ``--fetch`` only, never by a test."""
    import urllib.request
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, _ in FRENCH_FILES.values():
        req = urllib.request.Request(FRENCH_BASE + name, headers={"User-Agent": "screener-factor-lab"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if not data.startswith(b"PK"):
            raise RuntimeError(f"{name}: not a zip archive ({len(data)} bytes)")
        (raw_dir / name).write_bytes(data)
        out.append(raw_dir / name)
    return out


def read_table(zip_path: Path, marker: str) -> tuple[list[str], list[list[float]], str]:
    """(dates, rows, vintage line) of the table whose title line contains ``marker``.

    A library file holds several tables one after another, each a title line, a header line
    and rows that begin with a date. The table ends at the first line that does not."""
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    try:
        i = next(k for k, line in enumerate(lines) if marker in line)
    except StopIteration:
        raise ValueError(f"{zip_path.name}: no table titled {marker!r}") from None
    dates, rows = [], []
    for line in lines[i + 2:]:
        parts = [c.strip() for c in line.split(",")]
        if not parts[0].isdigit():
            break
        dates.append(parts[0])
        rows.append([float(c) for c in parts[1:]])
    if not rows:
        raise ValueError(f"{zip_path.name}: table {marker!r} is empty")
    return dates, rows, lines[0].strip()


def winners_minus_losers(rows: list[list[float]]) -> list[float | None]:
    """Top decile minus bottom decile, in the units of the file (percent). ``None`` where
    either leg is marked missing."""
    out = []
    for r in rows:
        lo, hi = r[0], r[-1]
        out.append(None if (lo <= MISSING or hi <= MISSING) else hi - lo)
    return out


def vol_forecasts(daily_dates: list[str], daily_wml_pct: list[float | None],
                  months: list[str], window: int) -> list[float | None]:
    """For each month, the annualised volatility forecast known when the month begins:
    the square root of 252 x the mean squared daily return over the last ``window`` trading
    days of earlier months. Barroso & Santa-Clara's estimator is 21 x sum(r^2) / 126 for the
    monthly variance; this is the same quantity, annualised. ``None`` until ``window`` days
    of history exist."""
    days = [(d, r / 100.0) for d, r in zip(daily_dates, daily_wml_pct) if r is not None]
    out, j = [], 0
    sq = [r * r for _, r in days]
    prefix = [0.0]
    for v in sq:
        prefix.append(prefix[-1] + v)
    for m in months:
        while j < len(days) and days[j][0][:6] < m:      # j = number of days before month m
            j += 1
        if j < window:
            out.append(None)
        else:
            out.append(math.sqrt(252.0 * (prefix[j] - prefix[j - window]) / window))
    return out


def momentum_data(raw_dir: Path = RAW_DIR) -> dict:
    """The derived series the momentum experiments need, from the two library files."""
    mfile, mmark = FRENCH_FILES["momentum_monthly"]
    dfile, dmark = FRENCH_FILES["momentum_daily"]
    m_dates, m_rows, vintage = read_table(raw_dir / mfile, mmark)
    d_dates, d_rows, _ = read_table(raw_dir / dfile, dmark)
    wml = winners_minus_losers(m_rows)
    dwml = winners_minus_losers(d_rows)
    keep = [i for i, v in enumerate(wml) if v is not None]
    months = [m_dates[i] for i in keep]
    return {
        "months": months,
        "wml": [round(wml[i], 2) for i in keep],
        "vol": {str(w): [None if v is None else round(v, 6)
                         for v in vol_forecasts(d_dates, dwml, months, w)] for w in VOL_WINDOWS},
        "vintage": vintage,
        "daily_first": d_dates[0], "daily_last": d_dates[-1],
    }


# --------------------------------------------------------------------------- the method
def stats(x: list[float]) -> dict | None:
    """The paper's Table 3 columns for a series of monthly returns in percent: best and worst
    month, annualised mean (12 x) and standard deviation (12^0.5 x, n-1), excess kurtosis and
    skewness (plain moments), and the Sharpe ratio (mean / standard deviation - these are
    zero-investment portfolios, so there is no risk-free rate to subtract)."""
    n = len(x)
    if n < 3:
        return None
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / (n - 1)
    sd = math.sqrt(var)
    pop = math.sqrt(sum((v - mean) ** 2 for v in x) / n)
    if pop == 0:
        return None
    skew = sum(((v - mean) / pop) ** 3 for v in x) / n
    kurt = sum(((v - mean) / pop) ** 4 for v in x) / n - 3.0
    amean, asd = 12.0 * mean, math.sqrt(12.0) * sd
    return {"n": n, "max": max(x), "min": min(x), "mean": amean, "std": asd,
            "kurt": kurt, "skew": skew, "sharpe": amean / asd}


def run(data: dict, start: str, end: str, window: int = 126, target: float = 0.12,
        cap: float | None = None) -> dict:
    """Plain and risk-managed momentum over ``start``..``end`` (YYYYMM, inclusive).

    Each month's exposure is ``target / forecast``: the target annualised volatility over the
    volatility forecast made from earlier months' daily returns, optionally capped (an
    investor who cannot use leverage has a cap of 1). A month with no forecast yet is left
    out of both series, so the two are always compared over the same months."""
    vol = data["vol"][str(window)]
    months, plain, managed, weights = [], [], [], []
    for m, r, v in zip(data["months"], data["wml"], vol):
        if m < start or m > end or v is None or v <= 0:
            continue
        w = target / v
        if cap is not None:
            w = min(w, cap)
        months.append(m)
        plain.append(r)
        managed.append(w * r)
        weights.append(w)
    return {"months": months, "plain": plain, "managed": managed, "weights": weights,
            "plain_stats": stats(plain), "managed_stats": stats(managed),
            "weight_stats": ({"min": min(weights), "max": max(weights),
                              "mean": sum(weights) / len(weights)} if weights else None)}


def growth(returns_pct: list[float]) -> list[float]:
    """What one dollar becomes, month by month, compounding the returns."""
    out, v = [], 1.0
    for r in returns_pct:
        v *= 1 + r / 100.0
        out.append(v)
    return out


def deepest_fall(returns_pct: list[float]) -> float:
    """The deepest fall of that compounded value below its own earlier high, as a fraction."""
    v = peak = 1.0
    worst = 0.0
    for r in returns_pct:
        v *= 1 + r / 100.0
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return worst


# --------------------------------------------------------------------------- the record
STAT_LABELS = {"max": "Best month (%)", "min": "Worst month (%)", "mean": "Mean, a year (%)",
               "std": "Volatility, a year (%)", "kurt": "Excess kurtosis", "skew": "Skewness",
               "sharpe": "Sharpe ratio"}

# How far a reproduced figure may sit from the paper's and still count as matching. Written
# 2026-10-12, after the first comparison was seen, as round numbers a reader can argue with:
# a quarter of a tenth on a ratio, one point on an annual percentage, three points on a
# single extreme month. The page prints every raw difference beside the verdict, so a
# stricter reader does not have to accept them.
TOLERANCES = {"max": 3.0, "min": 3.0, "mean": 1.0, "std": 1.0, "kurt": 1.0, "skew": 0.15,
              "sharpe": 0.05}

EXPERIMENTS = [
    {
        "id": "risk-managed-momentum",
        "title": "Risk-managed momentum",
        "paper": {
            "authors": "Pedro Barroso and Pedro Santa-Clara",
            "year": 2015,
            "title": "Momentum has its moments",
            "journal": "Journal of Financial Economics 116(1), 111-120",
            "doi": "10.1016/j.jfineco.2014.11.010",
        },
        "claim": ("Momentum's risk is predictable from its own recent daily returns. Scaling the "
                  "strategy by that forecast, to hold its volatility near a constant target, nearly "
                  "doubles its Sharpe ratio (0.53 to 0.97) and removes most of its crash risk: the "
                  "worst month goes from -78.96% to -28.40%, and excess kurtosis from 18.24 to 2.68."),
        "reference": {
            "source": ("Table 3 of the authors' April 2012 working paper, \"Managing the risk of "
                       "momentum\", sample 1927:03 to 2011:12. The published abstract quotes the "
                       "same Sharpe ratios, kurtosis and skewness."),
            "start": "192703", "end": "201112", "window": 126, "target": 0.12, "cap": None,
            "plain": {"max": 26.18, "min": -78.96, "mean": 14.46, "std": 27.53,
                      "kurt": 18.24, "skew": -2.47, "sharpe": 0.53},
            "managed": {"max": 21.95, "min": -28.40, "mean": 16.50, "std": 16.95,
                        "kurt": 2.68, "skew": -0.42, "sharpe": 0.97},
            "weights": {"min": 0.13, "max": 2.00, "mean": 0.90},
        },
        "conditions": [
            ["Portfolio", "Winners minus losers: the top decile of US stocks by return over months "
                          "t-12 to t-2, minus the bottom decile, value-weighted (the paper's WML)."],
            ["Data", "Kenneth R. French's Data Library, \"10 Portfolios Formed on Momentum\", monthly "
                     "and daily files."],
            ["Risk forecast", "The realised variance of the strategy's daily returns over the previous "
                              "126 trading days (about six months), known before the month begins."],
            ["Exposure", "Each month, target volatility of 12% a year divided by the forecast. Above 1 "
                         "means borrowing to hold more than the plain strategy."],
            ["Costs", "None. These are paper portfolios; the returns ignore trading costs, as the "
                      "paper's headline table does."],
        ],
        "open_details": [
            "The reference figures were read from the authors' 2012 working paper, through a text "
            "copy of it. The published article's own tables have not been read here (it is behind a "
            "paywall), so the page numbers a citation needs are still missing.",
            "The paper took its daily returns before July 1963 from Daniel and Moskowitz; this page "
            "uses the library's daily file for the whole period. That is the likeliest cause of the "
            "gap in the risk-managed worst month.",
            "The library is rebuilt from a newer version of the underlying database than the paper "
            "used, so every figure can differ slightly from the 2011 vintage.",
            "The library's daily file begins in November 1926, so the first two months of the paper's "
            "sample (March and April 1927) have no six-month forecast and are left out of both series.",
            "Whether the paper computed kurtosis and skewness with or without a small-sample "
            "correction is not stated; plain moments are used here.",
        ],
        "in_the_screener": (
            "The screener's momentum regime rule takes its 126-day volatility window from this paper. "
            "It does not use the paper's scaling: the screener only ever cuts momentum's weight, and "
            "only in a market panic state, because the above-one exposures shown here belong to a "
            "long-short portfolio and not to one slice of a long-only ranking."),
    },
]


def compare(reference: dict, measured: dict, tolerances: dict = TOLERANCES) -> list[dict]:
    """One row per reported figure: the paper's, ours, the difference and whether it is
    inside the tolerance."""
    rows = []
    for leg in ("plain", "managed"):
        for key in ("max", "min", "mean", "std", "kurt", "skew", "sharpe"):
            ref, got = reference[leg][key], measured[leg + "_stats"][key]
            diff = got - ref
            rows.append({"leg": leg, "stat": key, "label": STAT_LABELS[key], "paper": ref,
                         "here": got, "diff": diff, "tolerance": tolerances[key],
                         "within": abs(diff) <= tolerances[key]})
    return rows


def reproduce(data: dict, exp: dict) -> dict:
    """Run an experiment at the paper's own settings and compare."""
    ref = exp["reference"]
    res = run(data, ref["start"], ref["end"], ref["window"], ref["target"], ref["cap"])
    rows = compare(ref, res)
    return {"rows": rows, "within": sum(r["within"] for r in rows), "total": len(rows),
            "months": len(res["months"]), "first": res["months"][0], "last": res["months"][-1],
            "weights": res["weight_stats"]}


# --------------------------------------------------------------------------- the build
def build_payload(raw_dir: Path = RAW_DIR, now: datetime | None = None) -> dict:
    data = momentum_data(raw_dir)
    exps = []
    for exp in EXPERIMENTS:
        exps.append({**exp, "result": reproduce(data, exp)})
    return {
        "built": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d"),
        "attribution": {
            "text": "Data: Kenneth R. French Data Library. Copyright Eugene F. Fama and Kenneth R. French.",
            "url": FRENCH_HOME, "vintage": data["vintage"],
            "note": ("The monthly series below is derived from the library's files (top decile "
                     "minus bottom decile, and volatility forecasts computed from the daily file). "
                     "Download the originals from the library; this page does not republish them."),
        },
        "tolerances": TOLERANCES, "stat_labels": STAT_LABELS, "vol_windows": list(VOL_WINDOWS),
        "momentum": {k: data[k] for k in ("months", "wml", "vol", "daily_first", "daily_last")},
        "experiments": exps,
    }


def write(payload: dict, out: Path = OUT) -> Path:
    out.write_text("window.LAB_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    return out


def load_published(path: Path = OUT) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def report(payload: dict) -> str:
    lines = []
    for exp in payload["experiments"]:
        r = exp["result"]
        lines.append(f"{exp['title']} - {exp['paper']['authors']} ({exp['paper']['year']}): "
                     f"{r['within']} of {r['total']} figures within tolerance, "
                     f"{r['months']} months {r['first']}..{r['last']}")
        for row in r["rows"]:
            lines.append(f"  {row['leg']:8} {row['label']:24} paper {row['paper']:8.2f}  here "
                         f"{row['here']:8.2f}  diff {row['diff']:+6.2f}  "
                         f"{'within' if row['within'] else 'OUTSIDE'} +/-{row['tolerance']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the source files (network)")
    ap.add_argument("--build", action="store_true", help="rebuild lab_data.js")
    ap.add_argument("--report", action="store_true", help="print the comparison with the paper")
    args = ap.parse_args(argv)
    if args.fetch:
        for p in fetch():
            print("fetched", p.name)
    if args.build:
        payload = build_payload()
        print("wrote", write(payload).name, f"({OUT.stat().st_size / 1024:.0f} KB)")
        print(report(payload))
    elif args.report:
        print(report(load_published()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
