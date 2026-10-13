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
    "factors_monthly": ("F-F_Research_Data_Factors_CSV.zip", ",Mkt-RF,SMB,HML,RF"),
    "factors_daily": ("F-F_Research_Data_Factors_daily_CSV.zip", ",Mkt-RF,SMB,HML,RF"),
    "five_factors": ("F-F_Research_Data_5_Factors_2x3_CSV.zip", ",Mkt-RF,SMB,HML,RMW,CMA,RF"),
    "size_portfolios": ("Portfolios_Formed_on_ME_CSV.zip", "Average Value Weight Returns -- Monthly"),
}
SIZE_EQUAL_MARKER = "Average Equal Weighted Returns -- Monthly"
MISSING = -99.0                      # the library marks missing data -99.99 or -999

# lab.html loads lab_data.js and lab_engine.js as ?v=ASSET_VERSION. Bump it (and the two
# script tags) whenever either file's shape changes, so a browser holding last week's copy of
# one never runs it against this week's copy of the page. 2 = the second experiment;
# 3 = the third; 4 = scene links and the chart readout (the page changed, the files did not,
# but one number moving with every release of the page is simpler to keep true); 5 = the fourth
# experiment.
ASSET_VERSION = 5

# Trading-day windows the page offers for the volatility forecast. 126 is the paper's.
VOL_WINDOWS = (21, 63, 126, 252)
# For the market's own state (experiment 2): the windows its volatility is measured over, and
# how many months back "the market has fallen" looks. 126 days and 24 months are the paper's,
# and the screener's.
MARKET_VOL_WINDOWS = (63, 126, 252)
BEAR_LOOKBACKS = (12, 24, 36)


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
    # A titled table has its column names on the next line; the factor files have no title,
    # and are found by the column-name line itself.
    first = i + 1 if lines[i].lstrip().startswith(",") else i + 2
    for line in lines[first:]:
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


def market_vol(daily_dates: list[str], daily_ret_pct: list[float], months: list[str],
               window: int) -> list[float | None]:
    """For each month, the market's annualised volatility as known when the month begins: the
    standard deviation (n-1) of its daily returns over the last ``window`` trading days of
    earlier months, x 252^0.5. ``None`` until that many days exist."""
    s1, s2 = [0.0], [0.0]
    for r in daily_ret_pct:
        s1.append(s1[-1] + r / 100.0)
        s2.append(s2[-1] + (r / 100.0) ** 2)
    out, j = [], 0
    for m in months:
        while j < len(daily_dates) and daily_dates[j][:6] < m:
            j += 1
        if j < window:
            out.append(None)
            continue
        a, b = s1[j] - s1[j - window], s2[j] - s2[j - window]
        var = max(0.0, (b - a * a / window) / (window - 1))
        out.append(math.sqrt(252.0 * var))
    return out


def trailing_return(m_dates: list[str], m_ret_pct: list[float], months: list[str],
                    lookback: int, min_months: int = 6) -> list[float | None]:
    """For each month, the market's compounded return over the ``lookback`` months before it
    (as many as the file holds, if fewer - but at least ``min_months``). Negative is the
    paper's bear-market indicator."""
    pos = {d: k for k, d in enumerate(m_dates)}
    out = []
    for m in months:
        k = pos.get(m)
        if k is None or k < min_months:
            out.append(None)
            continue
        v = 1.0
        for r in m_ret_pct[max(0, k - lookback):k]:
            v *= 1 + r / 100.0
        out.append(v - 1.0)
    return out


def market_data(months: list[str], raw_dir: Path = RAW_DIR) -> dict:
    """The market series experiment 2 needs, aligned to the momentum months."""
    mfile, mmark = FRENCH_FILES["factors_monthly"]
    dfile, dmark = FRENCH_FILES["factors_daily"]
    m_dates, m_rows, _ = read_table(raw_dir / mfile, mmark)
    d_dates, d_rows, _ = read_table(raw_dir / dfile, dmark)
    excess = {d: r[0] for d, r in zip(m_dates, m_rows)}
    total = [r[0] + r[3] for r in m_rows]                    # Mkt-RF + RF
    daily_total = [r[0] + r[3] for r in d_rows]
    return {
        "mktrf": [round(excess[m], 2) for m in months],
        "vol": {str(w): [None if v is None else round(v, 6)
                         for v in market_vol(d_dates, daily_total, months, w)] for w in MARKET_VOL_WINDOWS},
        "trail": {str(n): [None if v is None else round(v, 6)
                           for v in trailing_return(m_dates, total, months, n)] for n in BEAR_LOOKBACKS},
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


def size_data(raw_dir: Path = RAW_DIR) -> dict:
    """Monthly returns (percent) of the ten size groups, smallest first, equal- and
    value-weighted. The file's last ten columns are the deciles."""
    name, vw_mark = FRENCH_FILES["size_portfolios"]
    dates, vw, vintage = read_table(raw_dir / name, vw_mark)
    dates_ew, ew, _ = read_table(raw_dir / name, SIZE_EQUAL_MARKER)
    if dates != dates_ew:
        raise ValueError("size file: the equal- and value-weighted tables cover different months")
    keep = [i for i in range(len(dates))
            if all(v > MISSING for v in vw[i][-10:]) and all(v > MISSING for v in ew[i][-10:])]
    return {"months": [dates[i] for i in keep], "vintage": vintage,
            "vw": [[round(vw[i][-10 + d], 2) for i in keep] for d in range(10)],
            "ew": [[round(ew[i][-10 + d], 2) for i in keep] for d in range(10)]}


def mean_t(x: list[float]) -> dict | None:
    """Average, standard deviation (n-1) and t-statistic of a monthly series."""
    n = len(x)
    if n < 6:
        return None
    mean = sum(x) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
    return {"mean": mean, "std": sd, "t": mean / (sd / math.sqrt(n)) if sd else None}


def size_run(size: dict, start: str, end: str, weighting: str = "ew") -> dict:
    """Each size group's average month over a period, and three differences with their
    t-statistics: smallest minus largest, and the eighth and ninth groups minus the tenth -
    the size effect among large companies."""
    idx = [i for i, m in enumerate(size["months"]) if start <= m <= end]
    months = [size["months"][i] for i in idx]
    dec = [[size[weighting][d][i] for i in idx] for d in range(10)]
    diff = lambda a, b: [x - y for x, y in zip(dec[a], dec[b])]            # noqa: E731
    return {"months": months, "deciles": [mean_t(x) for x in dec],
            "small_big": mean_t(diff(0, 9)), "nine_ten": mean_t(diff(8, 9)), "eight_ten": mean_t(diff(7, 9))}


FACTOR_KEYS = ["mktrf", "smb", "hml", "rmw", "cma"]
FACTOR_NAMES = {"mktrf": "Market", "smb": "Size", "hml": "Value", "rmw": "Profitability",
                "cma": "Investment"}
FACTOR_LONG = {"mktrf": "all stocks minus Treasury bills", "smb": "small minus big",
               "hml": "high book-to-market minus low", "rmw": "robust profitability minus weak",
               "cma": "conservative investment minus aggressive"}
FACTOR_CATEGORY = {"mktrf": None, "smb": "Size", "hml": "Valuation", "rmw": "Quality", "cma": "Investment"}


def factor_data(raw_dir: Path = RAW_DIR) -> dict:
    """The five monthly factor series, in percent, from the library's 2x3 file."""
    name, mark = FRENCH_FILES["five_factors"]
    dates, rows, vintage = read_table(raw_dir / name, mark)
    out = {"months": dates, "vintage": vintage}
    for k, key in enumerate(FACTOR_KEYS):
        out[key] = [round(r[k], 2) for r in rows]
    return out


def correlation(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def factor_run(factors: dict, start: str, end: str) -> dict:
    """Each factor's average month, volatility, t-statistic and compounded growth over a
    period, their correlations, and each factor regressed on the other four - the intercept is
    what the others leave unexplained, and ``r2`` how much of its variation they account for."""
    idx = [i for i, m in enumerate(factors["months"]) if start <= m <= end]
    months = [factors["months"][i] for i in idx]
    series = {k: [factors[k][i] for i in idx] for k in FACTOR_KEYS}
    n = len(months)
    stats_, spanned = {}, {}
    for k in FACTOR_KEYS:
        x = series[k]
        if n < 6:
            stats_[k] = None
            continue
        mean = sum(x) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
        stats_[k] = {"mean": mean, "std": sd, "t": mean / (sd / math.sqrt(n)) if sd else None,
                     "growth": growth(x)[-1]}
    if n >= 12:
        for k in FACTOR_KEYS:
            others = [o for o in FACTOR_KEYS if o != k]
            coef = ols(series[k], [[1.0] * n] + [series[o] for o in others])
            fitted = [coef[0] + sum(c * series[o][j] for c, o in zip(coef[1:], others)) for j in range(n)]
            my = sum(series[k]) / n
            sst = sum((v - my) ** 2 for v in series[k])
            sse = sum((v - f) ** 2 for v, f in zip(series[k], fitted))
            spanned[k] = {"intercept": coef[0], "slopes": dict(zip(others, coef[1:])),
                          "r2": 1 - sse / sst if sst else None}
    corr = {a: {b: (correlation(series[a], series[b]) if n >= 6 else None) for b in FACTOR_KEYS}
            for a in FACTOR_KEYS}
    return {"months": months, "series": series, "stats": stats_, "corr": corr, "spanned": spanned}


def factor_figures(factors: dict, start: str, end: str) -> dict:
    r = factor_run(factors, start, end)
    out = {"months": len(r["months"])}
    for k in FACTOR_KEYS:
        out["mean_" + k], out["std_" + k], out["t_" + k] = (r["stats"][k][s] for s in ("mean", "std", "t"))
    out["corr_hml_cma"] = r["corr"]["hml"]["cma"]
    out["hml_intercept"] = r["spanned"]["hml"]["intercept"]
    out["hml_on_cma"] = r["spanned"]["hml"]["slopes"]["cma"]
    out["hml_r2"] = r["spanned"]["hml"]["r2"]
    out["rmw_intercept"] = r["spanned"]["rmw"]["intercept"]
    out["cma_intercept"] = r["spanned"]["cma"]["intercept"]
    return out


STATE_NAMES = ["Market up, calm", "Market up, volatile", "Market down, calm",
               "Market down and volatile (the panic state)"]


def crash_run(mom: dict, market: dict, start: str, end: str, lookback: int = 24,
              window: int = 126, threshold: float = 0.206) -> dict:
    """Momentum's months sorted into four market states, each known before the month begins.

    "Down": the market's compounded return over the previous ``lookback`` months is negative
    (Daniel & Moskowitz's bear-market indicator). "Volatile": its volatility over the previous
    ``window`` trading days is above ``threshold`` a year. Down and volatile together is the
    panic state - at the defaults, the screener's own momentum regime rule. A month with no
    reading for either is left out."""
    trail, vol = market["trail"][str(lookback)], market["vol"][str(window)]
    months, wml, state = [], [], []
    for i, m in enumerate(mom["months"]):
        if m < start or m > end or trail[i] is None or vol[i] is None:
            continue
        months.append(m)
        wml.append(mom["wml"][i])
        state.append((2 if trail[i] < 0 else 0) + (1 if vol[i] > threshold else 0))
    n = len(months)
    cells = []
    for s in range(4):
        xs = [r for r, st in zip(wml, state) if st == s]
        cells.append({"state": s, "name": STATE_NAMES[s], "n": len(xs),
                      "share": len(xs) / n if n else 0.0,
                      "mean": sum(xs) / len(xs) if xs else None,
                      "worst": min(xs) if xs else None,
                      "losing": sum(1 for r in xs if r < 0) / len(xs) if xs else None})
    rest = [r for r, st in zip(wml, state) if st != 3]
    return {"months": months, "wml": wml, "state": state, "cells": cells,
            "all_mean": sum(wml) / n if n else None,
            "rest_mean": sum(rest) / len(rest) if rest else None}


def ols(y: list[float], columns: list[list[float]]) -> list[float]:
    """Least-squares coefficients of y on the given columns (include a column of ones for an
    intercept), by the normal equations. Small and exact enough for four regressors."""
    k = len(columns)
    a = [[sum(ci * cj for ci, cj in zip(columns[i], columns[j])) for j in range(k)] +
         [sum(ci * yi for ci, yi in zip(columns[i], y))] for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        for r in range(k):
            if r != i:
                f = a[r][i] / a[i][i]
                a[r] = [x - f * z for x, z in zip(a[r], a[i])]
    return [a[i][k] / a[i][i] for i in range(k)]


def crash_figures(mom: dict, market: dict, start: str, end: str, gamma_start: str,
                  paper_worst: list[list]) -> dict:
    """Every figure of the paper's that experiment 2 reproduces, measured on the published data."""
    idx = [i for i, m in enumerate(mom["months"]) if start <= m <= end]
    months = [mom["months"][i] for i in idx]
    wml = [mom["wml"][i] for i in idx]
    mkt = [market["mktrf"][i] for i in idx]
    bear = [None if market["trail"]["24"][i] is None else (1.0 if market["trail"]["24"][i] < 0 else 0.0)
            for i in idx]
    ws, ms = stats(wml), stats(mkt)
    ones = [1.0] * len(wml)
    a0, b0 = ols(wml, [ones, mkt])
    keep = [k for k, b in enumerate(bear) if b is not None]
    y = [wml[k] for k in keep]
    bb, mm = [bear[k] for k in keep], [mkt[k] for k in keep]
    _, alpha_b, _, beta_b = ols(y, [[1.0] * len(y), bb, mm, [u * v for u, v in zip(bb, mm)]])
    g = [k for k in keep if months[k] >= gamma_start]
    g0, gb = ols([wml[k] for k in g], [[1.0] * len(g), [bear[k] for k in g]])
    ranked = sorted(zip(wml, months))[:len(paper_worst)]
    ours = {m: r for r, m in ranked}
    paper = {m: r for m, r in paper_worst}
    return {
        "wml_mean": ws["mean"], "wml_std": ws["std"], "wml_sharpe": ws["sharpe"],
        "mkt_mean": ms["mean"], "mkt_std": ms["std"],
        "beta0": b0, "alpha0": a0, "beta_bear": beta_b, "alpha_bear": alpha_b,
        "bear_months": sum(1 for b in bear if b == 1.0), "months": len(months),
        "gamma0": g0, "gamma_bear": gb,
        "worst1": ranked[0][0], "worst2": ranked[1][0],
        "worst_overlap": len(set(ours) & set(paper)),
        "worst": [{"month": m, "here": r, "paper": paper.get(m)} for r, m in ranked],
        "missed": [{"month": m, "paper": r, "here": mom["wml"][mom["months"].index(m)]}
                   for m, r in paper_worst if m not in ours],
    }


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
        "kind": "risk-managed",
        "topic": "Momentum",
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
    {
        "id": "momentum-crashes",
        "kind": "crashes",
        "topic": "Momentum",
        "title": "Momentum crashes",
        "paper": {
            "authors": "Kent Daniel and Tobias J. Moskowitz",
            "year": 2016,
            "title": "Momentum crashes",
            "journal": "Journal of Financial Economics 122(2), 221-247",
            "doi": "10.1016/j.jfineco.2015.12.002",
        },
        "claim": ("Momentum's worst losses are partly forecastable. They come in panic states - after "
                  "the market has fallen over two years, while its volatility is high - and arrive as "
                  "the market rebounds. After a two-year market decline, momentum lost 0.67% a month on "
                  "average, against a gain of 1.96% a month otherwise."),
        "reference": {
            "source": ("The authors' draft of June 16, 2015 (the version on Kent Daniel's website): "
                       "Table 1 (page 8), Table 2 (page 11), the text on Table 3 (pages 15-16) and "
                       "Table 5, column 1 (page 20). Sample 1927:01 to 2013:03."),
            "start": "192701", "end": "201303", "gamma_start": "192707",
            "lookback": 24, "window": 126, "threshold": 0.206,
            # Written 2026-10-12 BEFORE the comparison was run (plan/factor-lab.md rule 3), in the
            # same round sizes as experiment 1. What was already known when they were written:
            # experiment 1 had shown this library's winners-minus-losers averaging about 15% a
            # year where this paper reports 17.9%, so the three figures built on that mean
            # (the mean, the alpha, the average outside bear markets) were expected to fall
            # outside; and the five worst months had been printed by an earlier probe.
            "figures": [
                {"key": "wml_mean", "label": "Momentum: mean, a year (%)", "paper": 17.9, "tolerance": 1.0, "where": "Table 1"},
                {"key": "wml_std", "label": "Momentum: volatility, a year (%)", "paper": 30.0, "tolerance": 1.0, "where": "Table 1"},
                {"key": "wml_sharpe", "label": "Momentum: Sharpe ratio", "paper": 0.60, "tolerance": 0.05, "where": "Table 1"},
                {"key": "mkt_mean", "label": "Market: mean excess return, a year (%)", "paper": 7.7, "tolerance": 0.3, "where": "Table 1"},
                {"key": "mkt_std", "label": "Market: volatility, a year (%)", "paper": 18.8, "tolerance": 0.3, "where": "Table 1"},
                {"key": "beta0", "label": "Momentum's market beta", "paper": -0.576, "tolerance": 0.05, "where": "Table 3 (1)"},
                {"key": "alpha0", "label": "Momentum's alpha, a month (%)", "paper": 1.852, "tolerance": 0.10, "where": "Table 3 (1)"},
                {"key": "beta_bear", "label": "Change in beta after a market decline", "paper": -1.131, "tolerance": 0.15, "where": "Table 3 (2)"},
                {"key": "alpha_bear", "label": "Change in alpha after a market decline, a month (%)", "paper": -2.04, "tolerance": 0.35, "where": "Table 3 (2)"},
                {"key": "bear_months", "label": "Months that followed a two-year market decline", "paper": 183, "tolerance": 5, "where": "Footnote 5", "count": True},
                {"key": "gamma0", "label": "Momentum's average month otherwise (%)", "paper": 1.955, "tolerance": 0.15, "where": "Table 5 (1)"},
                {"key": "gamma_bear", "label": "Difference after a market decline, a month (%)", "paper": -2.626, "tolerance": 0.40, "where": "Table 5 (1)"},
                {"key": "worst1", "label": "Worst month (%)", "paper": -74.36, "tolerance": 3.0, "where": "Table 2"},
                {"key": "worst2", "label": "Second-worst month (%)", "paper": -60.98, "tolerance": 3.0, "where": "Table 2"},
                {"key": "worst_overlap", "label": "Of the paper's 15 worst months, how many are among the 15 worst here", "paper": 15, "tolerance": 3, "where": "Table 2", "count": True},
            ],
            "worst": [["193208", -74.36], ["193207", -60.98], ["200101", -49.19], ["200904", -45.52],
                      ["193909", -43.83], ["193304", -43.14], ["200903", -42.28], ["200211", -37.04],
                      ["193806", -33.36], ["200908", -30.54], ["193106", -29.72], ["193305", -28.90],
                      ["200111", -25.31], ["200110", -24.98], ["197401", -24.04]],
        },
        "conditions": [
            ["Portfolio", "Winners minus losers, as in experiment 1: the library's top momentum decile "
                          "minus its bottom decile, value-weighted."],
            ["Market", "The value-weighted return of all US-listed stocks, and the one-month Treasury "
                       "bill rate, from the same library - the series the paper itself used."],
            ["Market fell", "The market's compounded return over the previous 24 months is negative. Known "
                            "before the month begins."],
            ["Volatile", "The market's daily returns over the previous 126 trading days have a standard "
                         "deviation above the threshold, as a rate a year. The paper uses this "
                         "volatility as a continuous measure; the threshold is how the screener turns "
                         "it into a yes or no."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper builds its own momentum deciles from the underlying stock database, ranking "
            "all listed firms; this page uses the library's deciles, which are formed differently. "
            "The paper's portfolio earns about three points a year more, and every figure built on "
            "that average sits outside its tolerance for that reason, not because the pattern "
            "differs. The authors publish their own portfolio returns on Kent Daniel's website; "
            "rerunning on those is the next check.",
            "The reference figures were read from the authors' 2015 draft. The published article's "
            "tables have not been read here.",
            "The paper's text gives momentum's Sharpe ratio as 0.71 while its Table 1 gives 0.60. "
            "The table's own mean and volatility (17.9 / 30.0) give 0.60, which is used.",
            "The library's market file begins in July 1926, so for months before July 1928 the "
            "two-year market return is measured over the months that exist (at least six).",
            "Only the first column of the paper's Table 5 is reproduced. Its other columns use the "
            "market's variance as a continuous measure in units the draft does not state.",
        ],
        "verdict_note": ("All of those describe the portfolio itself: the paper builds its own momentum "
                         "deciles and this page uses the library's. Every figure about what happens "
                         "after a market decline is within."),
        "in_the_screener": (
            "This is the study the screener's momentum regime rule is built on. At the settings "
            "shown first - a 24-month market return below zero and 126-day volatility above 20.6% - "
            "the fourth state below is the screener's rule, applied to a century of the paper's "
            "kind of data. One difference: this page measures the market as all US-listed stocks, "
            "as the paper does, while the screener reads the S&P 500 total-return index, so the two "
            "can disagree near the line. This page finds the state going into April 2020 and "
            "January 2023, when the S&P 500's own two-year return was still just positive. The "
            "screener's thresholds were fixed before this page existed and a test pins them; "
            "nothing found by moving the controls here is used to change them."),
    },
    {
        "id": "five-factors",
        "kind": "factors",
        "topic": "Value, profitability, investment, size",
        "title": "The five factors",
        "paper": {
            "authors": "Eugene F. Fama and Kenneth R. French",
            "year": 2015,
            "title": "A five-factor asset pricing model",
            "journal": "Journal of Financial Economics 116(1), 1-22",
            "doi": "10.1016/j.jfineco.2014.10.010",
        },
        "claim": ("Average stock returns line up with four things besides the market: size, value, "
                  "profitability and investment. From 1963 to 2013 the four factors built on them "
                  "averaged 0.29%, 0.37%, 0.25% and 0.33% a month, each more than two standard errors "
                  "from zero. And once profitability and investment are in the model, the value factor "
                  "adds nothing: the other four explain all but -0.04% a month of it."),
        "reference": {
            "source": ("The published article, Table 4 (page 7), Panels A and C, and Table 6 (page 11), "
                       "the 2x3 factors. Sample July 1963 to December 2013, 606 months."),
            "start": "196307", "end": "201312",
            # Written 2026-10-12 BEFORE the comparison was run. Nothing had been computed on this
            # file when they were set. The library is the authors' own data, rebuilt since on a
            # newer database, so most figures were expected to land within; no figure was
            # expected to miss for a known reason.
            "figures": [
                {"key": "mean_mktrf", "label": "Market: average month (%)", "paper": 0.50, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "mean_smb", "label": "Size: average month (%)", "paper": 0.29, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "mean_hml", "label": "Value: average month (%)", "paper": 0.37, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "mean_rmw", "label": "Profitability: average month (%)", "paper": 0.25, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "mean_cma", "label": "Investment: average month (%)", "paper": 0.33, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "std_mktrf", "label": "Market: volatility, a month (%)", "paper": 4.49, "tolerance": 0.10, "where": "Table 4A"},
                {"key": "std_smb", "label": "Size: volatility, a month (%)", "paper": 3.07, "tolerance": 0.10, "where": "Table 4A"},
                {"key": "std_hml", "label": "Value: volatility, a month (%)", "paper": 2.88, "tolerance": 0.10, "where": "Table 4A"},
                {"key": "std_rmw", "label": "Profitability: volatility, a month (%)", "paper": 2.14, "tolerance": 0.10, "where": "Table 4A"},
                {"key": "std_cma", "label": "Investment: volatility, a month (%)", "paper": 2.01, "tolerance": 0.10, "where": "Table 4A"},
                {"key": "t_mktrf", "label": "Market: t-statistic", "paper": 2.74, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "t_smb", "label": "Size: t-statistic", "paper": 2.31, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "t_hml", "label": "Value: t-statistic", "paper": 3.20, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "t_rmw", "label": "Profitability: t-statistic", "paper": 2.92, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "t_cma", "label": "Investment: t-statistic", "paper": 4.07, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "corr_hml_cma", "label": "Correlation of value with investment", "paper": 0.70, "tolerance": 0.05, "where": "Table 4C"},
                {"key": "hml_intercept", "label": "Value left unexplained by the other four, a month (%)", "paper": -0.04, "tolerance": 0.05, "where": "Table 6"},
                {"key": "hml_on_cma", "label": "Value's loading on investment", "paper": 1.04, "tolerance": 0.08, "where": "Table 6"},
                {"key": "hml_r2", "label": "Share of value's variation the other four explain", "paper": 0.51, "tolerance": 0.05, "where": "Table 6"},
                {"key": "rmw_intercept", "label": "Profitability left unexplained by the other four, a month (%)", "paper": 0.43, "tolerance": 0.05, "where": "Table 6"},
                {"key": "cma_intercept", "label": "Investment left unexplained by the other four, a month (%)", "paper": 0.28, "tolerance": 0.05, "where": "Table 6"},
            ],
        },
        "conditions": [
            ["Factors", "Each is a portfolio that holds one kind of stock and shorts its opposite: small "
                        "minus big, high book-to-market minus low, robust profitability minus weak, "
                        "conservative investment minus aggressive. The market factor is all stocks "
                        "minus Treasury bills."],
            ["Data", "Kenneth R. French's Data Library, \"Fama/French 5 Factors (2x3)\" - the authors' own "
                     "series, rebuilt since the paper on a newer version of the stock database."],
            ["Statistics", "The average and standard deviation of the monthly returns, and the "
                           "t-statistic: the average divided by its standard error."],
            ["Redundancy", "Each factor regressed on the other four over the same months. The "
                           "intercept is what the others leave unexplained."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The library has been rebuilt on a newer database since the paper, so small differences "
            "from the 2013 vintage are expected in every figure.",
            "Only the 2x3 version of the factors is reproduced. The paper also reports 2x2 and "
            "2x2x2x2 versions, which the library publishes separately.",
            "The paper's main tests - how well the model explains 25 and 32 sorted portfolios - are "
            "not reproduced here. This page checks the factors themselves.",
        ],
        "in_the_screener": (
            "Four of the screener's eight categories lean on these factors: Valuation on value, "
            "Quality on profitability, Investment on investment, and Size on size. Two cautions "
            "carry over. The screener ranks about 500 large stocks and holds none short, so it can "
            "capture only part of what a long-short portfolio of all stocks earns - and the size "
            "factor is mostly a small-stock effect, which the methodology's own note on Size says. "
            "And value and investment overlap heavily, which is a reason to read the Valuation and "
            "Investment categories together rather than as two independent votes."),
    },
    {
        "id": "size-effect",
        "kind": "size",
        "topic": "Size",
        "title": "The size effect, and whether it reaches large stocks",
        "paper": {
            "authors": "Eugene F. Fama and Kenneth R. French",
            "year": 1992,
            "title": "The cross-section of expected stock returns",
            "journal": "Journal of Finance 47(2), 427-465",
            "doi": "10.1111/j.1540-6261.1992.tb04398.x",
        },
        "claim": ("Smaller companies earned more. Sorting US stocks into ten groups by market value, the "
                  "smallest group averaged about 1.5% a month from 1963 to 1990 and the largest about "
                  "0.9%. The question this page adds, because the screener ranks only large companies: "
                  "is any of that difference found between the two largest groups?"),
        "reference": {
            "source": ("The published article, Table II, Panel A (page 436): average monthly returns of "
                       "portfolios formed on size, equal-weighted, July 1963 to December 1990. The paper "
                       "splits the smallest and largest tenths in half; those halves are combined here "
                       "by the number of firms in each, to compare with the library's ten groups."),
            "start": "196307", "end": "199012", "weighting": "ew",
            # Written 2026-10-12 BEFORE the comparison was run; nothing had been computed on this
            # file. One tolerance for every decile. The smallest tenth was the one thought most
            # likely to miss: it holds the most firms, the least liquid, and its reference figure
            # is derived from two halves rather than read.
            "figures": [
                {"key": "dec1", "label": "Smallest tenth: average month (%)", "paper": 1.546, "tolerance": 0.05, "where": "Table II A (its two halves, averaged by number of firms)"},
                {"key": "dec2", "label": "Size decile 2: average month (%)", "paper": 1.29, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec3", "label": "Size decile 3: average month (%)", "paper": 1.24, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec4", "label": "Size decile 4: average month (%)", "paper": 1.25, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec5", "label": "Size decile 5: average month (%)", "paper": 1.29, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec6", "label": "Size decile 6: average month (%)", "paper": 1.17, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec7", "label": "Size decile 7: average month (%)", "paper": 1.07, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec8", "label": "Size decile 8: average month (%)", "paper": 1.1, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec9", "label": "Size decile 9: average month (%)", "paper": 0.95, "tolerance": 0.05, "where": "Table II A"},
                {"key": "dec10", "label": "Largest tenth: average month (%)", "paper": 0.89, "tolerance": 0.05, "where": "Table II A (its two halves, averaged by number of firms)"},
                {"key": "small_big", "label": "Smallest tenth minus largest, a month (%)", "paper": 0.656, "tolerance": 0.08, "where": "Table II A, derived"},
            ],
        },
        "conditions": [
            ["Portfolios", "Each June, every US-listed stock is put in one of ten groups by market value, "
                           "using the New York Stock Exchange's own stocks to set the nine dividing "
                           "lines. The tenth group is the largest companies."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Size\"."],
            ["Weighting", "Equal-weighted, as in the paper's table: every company in a group counts the "
                          "same. The value-weighted version, where bigger companies count for more, is "
                          "one click away."],
            ["Large stocks", "The ninth and tenth groups. The screener's methodology takes these as the "
                             "range its roughly 500 companies fall in."],
            ["Costs", "None. The smallest groups are also the costliest to trade, which these "
                      "returns ignore."],
        ],
        "open_details": [
            "The paper's sample is non-financial firms that also appear in the accounting database "
            "(its page 429); the library's size groups are all listed stocks. That is a different "
            "set of companies, most of all among the small ones, and it is the likeliest reason "
            "eight of the eleven figures miss. Rebuilding the paper's exact sample needs the "
            "accounting database, which is not free.",
            "The paper reports the smallest and largest tenths as two halves each. Their combined "
            "figures here are weighted by the paper's own count of firms in each half, which is an "
            "approximation of what the paper would have printed for the whole tenth.",
            "The library has been rebuilt on a newer database, and the smallest group is where "
            "that matters most.",
            "Which size groups the screener's companies actually fall in has not been measured here. "
            "The methodology says the largest two; the true range may reach a little lower.",
            "This checks the paper's table of averages, not its regressions, which are its main "
            "evidence.",
        ],
        "verdict_note": ("The paper's table uses only non-financial companies with accounting data on "
                         "file; the library's groups hold every listed stock. Every group of smaller "
                         "companies sits below the paper's figure for it, and the only matches are among "
                         "the four largest groups."),
        "in_the_screener": (
            "The screener's Size category scores a company higher for being smaller, within the "
            "large companies it ranks. Its own methodology says no published study establishes a "
            "size premium among stocks this large. The last two bars of the chart are that "
            "question, and the sentence above the chart gives the answer for whatever period is "
            "chosen. What follows for the Size weight is a question for a research note, not for "
            "this page."),
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


def reproduce_crashes(mom: dict, market: dict, exp: dict) -> dict:
    """Experiment 2 at the paper's settings: each reference figure beside what the published
    data gives, the worst months side by side, and the four-state table at the screener's
    settings (which the page recomputes on load and checks against this)."""
    ref = exp["reference"]
    got = crash_figures(mom, market, ref["start"], ref["end"], ref["gamma_start"], ref["worst"])
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    scene = crash_run(mom, market, ref["start"], ref["end"], ref["lookback"], ref["window"], ref["threshold"])
    return {"rows": rows, "within": sum(r["within"] for r in rows), "total": len(rows),
            "months": got["months"], "worst": got["worst"], "missed": got["missed"],
            "cells": scene["cells"], "rest_mean": scene["rest_mean"], "all_mean": scene["all_mean"],
            "scene_months": len(scene["months"])}


def reproduce_factors(factors: dict, exp: dict) -> dict:
    """Experiment 3 at the paper's sample: each reference figure beside the published data's,
    and the per-factor table the page recomputes on load and checks against this."""
    ref = exp["reference"]
    got = factor_figures(factors, ref["start"], ref["end"])
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    r = factor_run(factors, ref["start"], ref["end"])
    return {"rows": rows, "within": sum(x["within"] for x in rows), "total": len(rows),
            "months": got["months"], "stats": r["stats"],
            "hml_intercept": r["spanned"]["hml"]["intercept"]}


def reproduce_size(size: dict, exp: dict) -> dict:
    """Experiment 4 at the paper's sample and weighting, against its table."""
    ref = exp["reference"]
    r = size_run(size, ref["start"], ref["end"], ref["weighting"])
    got = {f"dec{d + 1}": r["deciles"][d]["mean"] for d in range(10)}
    got["small_big"] = r["small_big"]["mean"]
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    return {"rows": rows, "within": sum(x["within"] for x in rows), "total": len(rows),
            "months": len(r["months"]), "means": [d["mean"] for d in r["deciles"]],
            "nine_ten": r["nine_ten"]}


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
    market = market_data(data["months"], raw_dir)
    factors = factor_data(raw_dir)
    size = size_data(raw_dir)
    exps = []
    for exp in EXPERIMENTS:
        result = (reproduce_crashes(data, market, exp) if exp["kind"] == "crashes"
                  else reproduce_factors(factors, exp) if exp["kind"] == "factors"
                  else reproduce_size(size, exp) if exp["kind"] == "size"
                  else reproduce(data, exp))
        exps.append({**exp, "result": result})
    return {
        "built": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d"),
        "attribution": {
            "text": "Data: Kenneth R. French Data Library. Copyright Eugene F. Fama and Kenneth R. French.",
            "url": FRENCH_HOME, "vintage": data["vintage"],
            "note": ("The monthly series here are derived from the library's files (top momentum "
                     "decile minus bottom decile, the market's excess return, and volatility and "
                     "trailing-return readings computed from the daily and monthly files). "
                     "Download the originals from the library; this page does not republish them."),
        },
        "tolerances": TOLERANCES, "stat_labels": STAT_LABELS, "vol_windows": list(VOL_WINDOWS),
        "market_vol_windows": list(MARKET_VOL_WINDOWS), "bear_lookbacks": list(BEAR_LOOKBACKS),
        "state_names": STATE_NAMES,
        "factor_keys": FACTOR_KEYS, "factor_names": FACTOR_NAMES, "factor_long": FACTOR_LONG,
        "factor_category": FACTOR_CATEGORY,
        "factors": {k: factors[k] for k in ["months"] + FACTOR_KEYS},
        "size": {k: size[k] for k in ("months", "ew", "vw")},
        "momentum": {k: data[k] for k in ("months", "wml", "vol", "daily_first", "daily_last")},
        "market": market,
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
                     f"{r['months']} months" + (f" {r['first']}..{r['last']}" if "first" in r else ""))
        for row in r["rows"]:
            lines.append(f"  {row.get('leg', ''):8} {row['label'][:52]:52} paper {row['paper']:8.2f}  here "
                         f"{row['here']:8.2f}  diff {row['diff']:+6.2f}  "
                         f"{'within' if row['within'] else 'OUTSIDE'} +/-{row['tolerance']}")
        for c in r.get("cells", []):
            mean = "   n/a" if c["mean"] is None else f"{c['mean']:+6.2f}"
            lines.append(f"    state {c['state']} {c['name'][:42]:42} {c['n']:5} months  mean {mean}% a month")
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
