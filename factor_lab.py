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
published in ``lab_data.js`` is the derived monthly series the page needs, with attribution. The
heaviest series are published beside it, one file each (``lab_data_<key>.js``, ``LAZY_KEYS``), and
the page loads one only when an experiment that reads it is opened.

    python factor_lab.py --fetch     # download the source files (network)
    python factor_lab.py --build     # rebuild lab_data.js and lab_data_<key>.js from them
    python factor_lab.py --report    # print the comparison with the paper
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
    "variance_portfolios": ("Portfolios_Formed_on_VAR_CSV.zip", "Value Weighted Returns -- Monthly"),
    "earnings_yield_portfolios": ("Portfolios_Formed_on_E-P_CSV.zip", "Value Weight Returns -- Monthly"),
    "beta_portfolios": ("Portfolios_Formed_on_BETA_CSV.zip", "Value Weighted Returns -- Monthly"),
    "profitability_portfolios": ("Portfolios_Formed_on_OP_CSV.zip", "Average Value Weight Returns -- Monthly"),
    "cash_flow_yield_portfolios": ("Portfolios_Formed_on_CF-P_CSV.zip", "Value Weight Returns -- Monthly"),
    "accruals_portfolios": ("Portfolios_Formed_on_AC_CSV.zip", "Value Weighted Returns -- Monthly"),
    "investment_portfolios": ("Portfolios_Formed_on_INV_CSV.zip", "Average Value Weight Returns -- Monthly"),
    "split_value": ("6_Portfolios_2x3_CSV.zip", "Average Value Weighted Returns -- Monthly"),
    "split_profitability": ("6_Portfolios_ME_OP_2x3_CSV.zip", "Average Value Weighted Returns -- Monthly"),
    "split_investment": ("6_Portfolios_ME_INV_2x3_CSV.zip", "Average Value Weighted Returns -- Monthly"),
    "split_momentum": ("6_Portfolios_ME_Prior_12_2_CSV.zip", "Average Value Weighted Returns -- Monthly"),
    "quint_value": ("25_Portfolios_5x5_CSV.zip", "Average Value Weighted Returns -- Monthly"),
    "quint_momentum": ("25_Portfolios_ME_Prior_12_2_CSV.zip", "Average Value Weighted Returns -- Monthly"),
    "double_size_op": ("25_Portfolios_ME_OP_5x5_CSV.zip", "Average Value Weighted Returns -- Monthly"),
}
# The 25 size-by-profitability portfolios (experiment "size-given-profitability"): column
# 5 * s + p is size quintile s (0 smallest .. 4 largest) and operating-profitability quintile p
# (0 weakest .. 4 most profitable), the library's own order (SMALL LoOP, ME1 OP2, ..., BIG HiOP).
DOUBLE_SIZE, DOUBLE_TRAIT = 5, 5
# The 25 size-by-trait portfolios: five size groups, five trait groups inside each, in that
# order. The lowest and highest trait group of the smallest and of the largest size group.
QUINT_KEYS = ["value", "momentum"]
QUINT_COLS = {"small": (0, 4), "big": (20, 24)}
# Factors computed separately among small and big companies. Each file's columns are
# small-low, small-middle, small-high, big-low, big-middle, big-high; the sign says which end
# the factor is long (investment is conservative minus aggressive: low minus high).
SPLIT_KEYS = ["value", "profitability", "investment", "momentum"]
SPLIT_PARTS = ["small", "big", "long", "short"]
SPLIT_SIGN = {"value": 1, "profitability": 1, "investment": -1, "momentum": 1}
SPLIT_NAMES = {"value": "Value", "profitability": "Profitability", "investment": "Investment",
               "momentum": "Momentum"}
SPLIT_CATEGORY = {"value": "Valuation", "profitability": "Quality", "investment": "Investment",
                  "momentum": "Momentum"}
# Sorted-portfolio experiments (kind "sort"): which file, which two tables, which columns.
SORTS = {
    "low-volatility": {"file": "variance_portfolios", "ew": "Equal Weighted Returns -- Monthly",
                       "columns": [0, 1, 2, 3, 4]},
    # the file's last ten columns are the deciles; the first is companies with losses
    "earnings-yield": {"file": "earnings_yield_portfolios", "ew": "Equal Weight Returns -- Monthly",
                       "columns": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
    # five quintiles first, then the ten deciles
    "market-beta": {"file": "beta_portfolios", "ew": "Equal Weighted Returns -- Monthly",
                    "columns": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
    # three terciles, then the five quintiles, then the ten deciles: the quintiles, as the paper
    "profitability": {"file": "profitability_portfolios", "ew": "Average Equal Weighted Returns -- Monthly",
                      "columns": [3, 4, 5, 6, 7], "excess": True},
    # the file's last ten columns are the deciles; the first is companies with negative cash flow
    "cash-flow-yield": {"file": "cash_flow_yield_portfolios", "ew": "Equal Weight Returns -- Monthly",
                        "columns": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
    # the file's last ten columns are the deciles, lowest accruals first; listed here highest first,
    # so the last group is the low-accrual end the paper buys and the screener favours, and the
    # spread (last minus first) is the paper's hedge return
    "accruals": {"file": "accruals_portfolios", "ew": "Equal Weighted Returns -- Monthly",
                 "columns": [14, 13, 12, 11, 10, 9, 8, 7, 6, 5]},
    # three terciles, five quintiles, then the ten deciles, slowest asset growth first; listed here
    # fastest first, so the last group is the slow-growth end the paper buys and the screener's
    # Investment category favours, and the spread (last minus first) is the paper's 1 minus 10
    "asset-growth": {"file": "investment_portfolios", "ew": "Average Equal Weighted Returns -- Monthly",
                     "columns": [17, 16, 15, 14, 13, 12, 11, 10, 9, 8]},
    # Not the French library: Chen & Zimmermann's Open Source Asset Pricing portfolios
    # (``source: "osap"``), built by its authors from CRSP and Compustat. Their Piotroski score
    # (signal "PS") is sorted into deciles among the highest fifth of companies by book-to-market;
    # the columns are the port labels of the lowest and highest decile, lowest score first. With a
    # 0-9 score the middle deciles are often empty (the cut points tie), so only the ends are read.
    # "vw" is the authors' own cut (Original_Cuts, value-weighted, as their documentation records
    # for PS); "ew" is the same deciles equally weighted, from their alternative-portfolio set.
    "piotroski": {"source": "osap", "signal": "PS", "columns": ["01", "10"],
                  "vw": "PS_OriginalCuts_VW.csv", "ew": "PredictorAltPorts_DecilesEW.zip"},
}
# Chen & Zimmermann (2022), "Open source cross-sectional asset pricing", Critical Finance Review
# 11(2), 207-264. Portfolio returns are a free download (Google Drive, linked from the data page);
# the code is GPL-2.0 and the authors ask that users of the data cite the paper. Files are kept in
# data/lab/osap/ (gitignored) and never committed. plan/factor-lab.md, "Free data to evaluate".
OSAP_DIR = ROOT / "data" / "lab" / "osap"
OSAP_HOME = "https://www.openassetpricing.com/data/"
OSAP_RELEASE = "October 2025 release (v2.0.0)"
OSAP_DRIVE_IDS = {                    # the release's own files, as its Google Drive folder lists them
    "PS_OriginalCuts_VW.csv": "1fkoDBda7SKboz8NSJbElG5F-pu1oaPdb",
    "PredictorAltPorts_DecilesEW.zip": "1pFBX0LTyoSwH7i8mT4kPCS_3QghlbODD",
}
# Jensen, Kelly & Pedersen's global factor library (experiment 15). Licensed CC BY-NC 4.0 by its
# authors (the licence is printed at the foot of jkpfactors.com/data and in DATA_LICENSE of
# github.com/bkelly-lab/jkp-data): sharing and adapting are allowed for non-commercial use with
# credit, a link to the licence, and a note of what was changed. The files are downloaded to
# data/lab/jkp/ (gitignored) and never committed; lab_data.js publishes the derived monthly
# series with that attribution. File names follow the download page's own script: the S3 keys
# wrap each part in literal square brackets, URL-encoded.
JKP_DIR = ROOT / "data" / "lab" / "jkp"
JKP_HOME = "https://jkpfactors.com/"
JKP_LICENCE = "https://creativecommons.org/licenses/by-nc/4.0/"
JKP_BASE = "https://jkpfactors-data.s3.amazonaws.com/public/"
JKP_REPO = "https://raw.githubusercontent.com/bkelly-lab/jkp-data/main/"
JKP_ALL = "usa_all_factors_monthly_vw_cap.zip"            # the 153 US factors, capped value weights
JKP_MARKET = "usa_mkt_monthly_vw.zip"                      # the US market's excess return, value-weighted
JKP_DETAILS = "factor_details.xlsx"                        # each factor's paper, sample and sign
JKP_CLUSTERS = "cluster_labels.csv"                        # each factor's theme (the paper's 13)
# The library's size groups, by NYSE breakpoints: mega is above the 80th percentile, large the
# 50th to the 80th. Mega is the nearest public group to the companies the screener ranks.
JKP_SIZES = ("mega", "large")
JKP_END = "202012"                                         # the paper's sample ends with 2020


def jkp_key(*parts: str) -> str:
    return "_".join(f"%5B{p}%5D" for p in parts) + ".zip"


def fetch_jkp(raw_dir: Path = JKP_DIR, timeout: float = 120.0) -> list[Path]:
    """Download the JKP files the lab reads. Network; called by ``--fetch`` only, never by a
    test. One request per factor and size group (306 small files), as the download page makes."""
    import time
    import urllib.request
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(JKP_BASE + jkp_key("usa", "all_factors", "monthly", "vw_cap"), JKP_ALL),
            (JKP_BASE + jkp_key("usa", "mkt", "monthly", "vw"), JKP_MARKET),
            (JKP_REPO + "src/jkp/data/resources/factor_details.xlsx", JKP_DETAILS),
            (JKP_REPO + "src/jkp/data/resources/cluster_labels.csv", JKP_CLUSTERS)]
    out = []
    for url, name in jobs:
        req = urllib.request.Request(url, headers={"User-Agent": "screener-factor-lab"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            (raw_dir / name).write_bytes(r.read())
        out.append(raw_dir / name)
    for name in jkp_names(raw_dir):
        for size in JKP_SIZES:
            req = urllib.request.Request(JKP_BASE + "factor/" + jkp_key("usa", name, size),
                                         headers={"User-Agent": "screener-factor-lab"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if not data.startswith(b"PK"):
                raise RuntimeError(f"{name} {size}: not a zip archive ({len(data)} bytes)")
            (raw_dir / f"usa_{name}_{size}.zip").write_bytes(data)
            out.append(raw_dir / f"usa_{name}_{size}.zip")
            time.sleep(0.1)
    return out


def _zip_csv(path: Path) -> list[dict]:
    import csv
    import io
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def jkp_names(raw_dir: Path = JKP_DIR) -> list[str]:
    return sorted({r["name"] for r in _zip_csv(raw_dir / JKP_ALL)})


def _yyyymm(date: str) -> str:
    return date[:4] + date[5:7]


def jkp_data(raw_dir: Path = JKP_DIR) -> dict:
    """The 153 US factors as the library publishes them, plus the same characteristics within the
    mega and large size groups, monthly, in basis points (whole numbers, to keep the published
    file small; every calculation divides by 100 first, in Python and in the browser alike).

    * ``all``: the library's factor - the top third minus the bottom third by the characteristic,
      capped value weights, already signed the way the original paper found it paid.
    * ``mega`` / ``large``: the library's size-group portfolios, which weight every stock in the
      group by its rank on the characteristic (centred, each side summing to one) and are NOT
      signed; they are multiplied here by the factor's own sign from the first file.
    Each series is ``[first month index, value, value, ...]`` over ``months``, None for a gap.
    Every factor's meta: its theme, whether its paper found it significant, and the paper's own
    sample (the "in-sample period" the paper's out-of-sample tests start from)."""
    import csv
    import openpyxl
    rows = _zip_csv(raw_dir / JKP_ALL)
    sign = {}
    by_name: dict[str, dict[str, float]] = {}
    for r in rows:
        sign[r["name"]] = int(float(r["direction"]))
        by_name.setdefault(r["name"], {})[_yyyymm(r["date"])] = float(r["ret"]) * 100
    market = {_yyyymm(r["date"]): float(r["ret"]) * 100 for r in _zip_csv(raw_dir / JKP_MARKET)}
    sized = {s: {} for s in JKP_SIZES}
    for name in by_name:
        for s in JKP_SIZES:
            sized[s][name] = {_yyyymm(r["date"]): sign[name] * float(r["ret"]) * 100
                              for r in _zip_csv(raw_dir / f"usa_{name}_{s}.zip")}
    months = sorted(set(market) | {m for v in by_name.values() for m in v}
                    | {m for g in sized.values() for v in g.values() for m in v})
    months = [m for m in months if m in market]                       # a factor month needs the market's
    pos = {m: i for i, m in enumerate(months)}
    with open(raw_dir / JKP_CLUSTERS, newline="", encoding="utf-8") as fh:
        theme = {r["characteristic"]: r["cluster"] for r in csv.DictReader(fh)}
    wb = openpyxl.load_workbook(raw_dir / JKP_DETAILS, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    head, *body = list(ws.iter_rows(values_only=True))
    col = {h: i for i, h in enumerate(head)}
    details = {}
    for r in body:
        name = r[col["abr_jkp"]]
        if name in by_name and name not in details:
            a, b = (int(y) for y in str(r[col["in-sample period"]]).split("-"))
            details[name] = {"sig": int(r[col["significance"]]), "in_start": f"{a}01", "in_end": f"{b}12",
                             "label": r[col["name_new"]], "cite": r[col["cite"]]}

    def series(values: dict[str, float]) -> list:
        ms = sorted(m for m in values if m in pos)
        first, last = pos[ms[0]], pos[ms[-1]]
        return [first] + [None if months[i] not in values else round(values[months[i]] * 100)
                          for i in range(first, last + 1)]

    names = sorted(by_name)
    return {
        "months": months,
        "mkt": [round(market[m] * 100) for m in months],
        "factors": [{"name": n, "theme": theme[n], "sign": sign[n], **details[n]} for n in names],
        "themes": sorted(set(theme[n] for n in names)),
        "series": {"all": [series(by_name[n]) for n in names],
                   **{s: [series(sized[s][n]) for n in names] for s in JKP_SIZES}},
    }


JKP_MIN_MONTHS = 60          # the paper's own minimum for a factor to count (p. 2487)
JKP_T = 1.96                 # "significant": the t-statistic clears 1.96 in the paper's direction
JKP_IDIO = 10 / math.sqrt(12)    # every factor scaled to 10% a year of idiosyncratic volatility (p. 2487)


def jkp_window(jkp: dict, universe: str, k: int, start: str, end: str) -> tuple[list[float], list[float]]:
    """Factor k's returns and the market's, in percent, over the months between start and end
    (inclusive) where both exist. Plain loops, in the same order as the browser's."""
    ser, months, mkt = jkp["series"][universe][k], jkp["months"], jkp["mkt"]
    first = ser[0]
    y, x = [], []
    for j in range(1, len(ser)):
        i = first + j - 1
        m = months[i]
        if m < start or m > end or ser[j] is None or mkt[i] is None:
            continue
        y.append(ser[j] / 100)
        x.append(mkt[i] / 100)
    return y, x


def jkp_capm(y: list[float], x: list[float]) -> dict | None:
    """Average and its t-statistic; CAPM alpha, beta, the alpha's t-statistic (plain OLS standard
    error) and the residual volatility. None below the paper's minimum of months."""
    n = len(y)
    if n < JKP_MIN_MONTHS:
        return None
    sy = sx = 0.0
    for a, b in zip(y, x):
        sy += a
        sx += b
    my, mx = sy / n, sx / n
    syy = sxx = sxy = 0.0
    for a, b in zip(y, x):
        syy += (a - my) * (a - my)
        sxx += (b - mx) * (b - mx)
        sxy += (b - mx) * (a - my)
    sd = math.sqrt(syy / (n - 1))
    beta = sxy / sxx
    alpha = my - beta * mx
    ss = 0.0
    for a, b in zip(y, x):
        e = a - alpha - beta * b
        ss += e * e
    s = math.sqrt(ss / (n - 2))
    se = math.sqrt(s * s * (1 / n + mx * mx / sxx))
    return {"n": n, "mean": my, "t_mean": my / (sd / math.sqrt(n)) if sd else 0.0,
            "alpha": alpha, "beta": beta, "t_alpha": alpha / se if se else 0.0, "resid": s}


def jkp_run(jkp: dict, start: str, end: str, universe: str = "all", test: str = "capm",
            which: str = "sig") -> dict:
    """The replication rate over a period: of the factors in the set ("sig", the ones significant in
    their own papers, or "all" 153) with at least 60 months in it, the share whose average return
    ("raw") or CAPM alpha ("capm") has a t-statistic above 1.96 in the paper's direction. And the
    same by theme, with each theme's average alpha (percent a month, unscaled)."""
    stats_by, hits, counted = [], 0, 0
    themes = {t: {"theme": t, "n": 0, "hits": 0, "alpha_sum": 0.0} for t in jkp["themes"]}
    for k, f in enumerate(jkp["factors"]):
        if which == "sig" and not f["sig"]:
            stats_by.append(None)
            continue
        y, x = jkp_window(jkp, universe, k, start, end)
        r = jkp_capm(y, x)
        stats_by.append(r)
        if r is None:
            continue
        ok = (r["t_alpha"] if test == "capm" else r["t_mean"]) > JKP_T
        counted += 1
        hits += ok
        th = themes[f["theme"]]
        th["n"] += 1
        th["hits"] += ok
        th["alpha_sum"] += r["alpha"]
    rows = [{"theme": t["theme"], "n": t["n"], "hits": t["hits"],
             "rate": 100 * t["hits"] / t["n"] if t["n"] else None,
             "alpha": t["alpha_sum"] / t["n"] if t["n"] else None} for t in themes.values()]
    total = 0.0
    for t in themes.values():
        total += t["alpha_sum"]
    return {"n": counted, "hits": hits, "rate": 100 * hits / counted if counted else None,
            "alpha": total / counted if counted else None, "themes": rows, "factors": stats_by}


def _prev_month(m: str) -> str:
    y, mo = int(m[:4]), int(m[4:])
    return f"{y - (mo == 1)}{12 if mo == 1 else mo - 1:02d}"


def jkp_outside(jkp: dict, end: str = JKP_END) -> dict:
    """The paper's out-of-sample test (Section III.B.2) on the factors significant in their own
    papers, among all stocks: each factor scaled to 10% a year of idiosyncratic volatility over its
    whole record to ``end``, then its CAPM alpha inside its paper's sample, before it and after it
    (each window needs 60 months, as in the paper's Figure 8)."""
    pre, post, both, a_in, a_post = [], [], [], [], []
    for k, f in enumerate(jkp["factors"]):
        if not f["sig"]:
            continue
        full = jkp_capm(*jkp_window(jkp, "all", k, "000000", end))
        if full is None:
            continue
        scale = JKP_IDIO / full["resid"]
        r_pre = jkp_capm(*jkp_window(jkp, "all", k, "000000", _prev_month(f["in_start"])))
        r_post = jkp_capm(*jkp_window(jkp, "all", k, next_month(f["in_end"]), end))
        yp, xp = jkp_window(jkp, "all", k, "000000", _prev_month(f["in_start"]))
        yq, xq = jkp_window(jkp, "all", k, next_month(f["in_end"]), end)
        r_both = jkp_capm(yp + yq, xp + xq)
        if r_pre is not None:
            pre.append(r_pre["alpha"] > 0)
        if r_both is not None:
            both.append(r_both["alpha"] > 0)
        if r_post is not None:
            post.append(r_post["alpha"] > 0)
            y, x = jkp_window(jkp, "all", k, f["in_start"], f["in_end"])
            n = len(y)
            if n >= 3:   # the paper's own sample, clipped to the data; alpha needs only a line
                r_in = jkp_capm_any(y, x)
                a_in.append(r_in * scale)
                a_post.append(r_post["alpha"] * scale)
    mean = lambda v: sum(v) / len(v) if v else None                      # noqa: E731
    share = lambda v: 100 * sum(v) / len(v) if v else None                # noqa: E731
    return {"alpha_in": mean(a_in), "alpha_post": mean(a_post), "pos_pre": share(pre),
            "pos_post": share(post), "pos_both": share(both),
            "n_pre": len(pre), "n_post": len(post), "n_both": len(both), "n_pair": len(a_in)}


def jkp_capm_any(y: list[float], x: list[float]) -> float:
    """CAPM alpha with no minimum of months (a paper's own sample can be short)."""
    n = len(y)
    sy = sx = 0.0
    for a, b in zip(y, x):
        sy += a
        sx += b
    my, mx = sy / n, sx / n
    sxx = sxy = 0.0
    for a, b in zip(y, x):
        sxx += (b - mx) * (b - mx)
        sxy += (b - mx) * (a - my)
    return my - (sxy / sxx) * mx


def jkp_figures(jkp: dict, ref: dict) -> dict:
    """Every figure experiment 15 compares, at the paper's settings."""
    start = jkp["months"][0] if ref["start"] == "FIRST" else ref["start"]
    end = ref["end"]
    got = {"rate_raw_all": jkp_run(jkp, start, end, "all", "raw", "all")["rate"],
           "rate_raw_sig": jkp_run(jkp, start, end, "all", "raw", "sig")["rate"],
           "rate_capm": jkp_run(jkp, start, end, "all", "capm", "sig")["rate"],
           "rate_capm_mega": jkp_run(jkp, start, end, "mega", "capm", "sig")["rate"],
           "rate_capm_large": jkp_run(jkp, start, end, "large", "capm", "sig")["rate"]}
    got.update(jkp_outside(jkp, end))
    return got


def reproduce_replication(jkp: dict, exp: dict) -> dict:
    """Experiment 15 at the paper's settings, against its figures."""
    ref = exp["reference"]
    got = jkp_figures(jkp, ref)
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    start = jkp["months"][0] if ref["start"] == "FIRST" else ref["start"]
    r = jkp_run(jkp, start, ref["end"], ref["universe"], ref["test"], ref["set"])
    return {"rows": rows, "within": sum(x["within"] for x in rows), "total": len(rows),
            "months": sum(1 for m in jkp["months"] if start <= m <= ref["end"]),
            "rate": r["rate"], "n": r["n"], "hits": r["hits"],
            "counts": {k: got[k] for k in ("n_pre", "n_post", "n_both", "n_pair")}}


SIZE_EQUAL_MARKER = "Average Equal Weighted Returns -- Monthly"
MISSING = -99.0                      # the library marks missing data -99.99 or -999

# lab.html loads lab_data.js and lab_engine.js as ?v=ASSET_VERSION. Bump it (and the two
# script tags) whenever either file's shape changes, so a browser holding last week's copy of
# one never runs it against this week's copy of the page. 2 = the second experiment;
# 3 = the third; 4 = scene links and the chart readout (the page changed, the files did not,
# but one number moving with every release of the page is simpler to keep true); 5 = the fourth
# experiment; 6 = the front page; 7 = sorted-portfolio experiments; 8 = the sixth experiment; 9 = the seventh; 10 = the table of
# results inside and after their papers' samples; 11 = the eighth experiment and the scorecard;
# 12 = the two sides of each big-company factor; 13 = the scorecard's Growth row after the review;
# 14 = its Valuation and Momentum rows after theirs; 15 = the ninth experiment; 16 = the tenth; 17 = the eleventh; 18 = the twelfth; 19 = the thirteenth (a new kind, combo); 20 = the grouped menu (a group per experiment);
# 21 = the 2026-10-16 audit (when each experiment's tolerances were written, what a null result would match);
# 22 = the engine computes every comparison with the paper (E.compare*), and the page fills it from them;
# 23 = the fourteenth experiment (a sort compared under both weightings, `also`) and the Investment group;
# 24 = experiment 15, Jensen, Kelly & Pedersen's 153 factors (a new kind, replication) and its data's attribution;
# 25 = size inside each profitability fifth (queue row 4b; a new kind, double) and the note's pre-registered rule;
# 26 = Piotroski on Open Source Asset Pricing's portfolios, and a second data attribution (merged 2026-10-20 night);
# 27 = the heavy series moved to lab_data_<key>.js, loaded when an experiment opens (LAZY_KEYS) - the page
#      builds their addresses with this same ?v=, read off its lab_data.js tag;
# 28 = the 153 factors' size groups and each of the eight sorts in parts of their own (LAZY_INTO),
#      loaded when shown or chosen, and lab_engine.js's compareReplication marks a figure whose series
#      has not loaded as pending.
ASSET_VERSION = 28

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


def fetch_osap(osap_dir: Path = OSAP_DIR, timeout: float = 600.0) -> list[Path]:
    """Download the Open Source Asset Pricing files from the release's public Google Drive folder.
    Network; ``--fetch`` only. Drive answers a much-downloaded shared file with a "Quota exceeded"
    page for up to a day (seen 2026-10-20); that is reported, not worked around - try again later."""
    import urllib.request
    osap_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, file_id in OSAP_DRIVE_IDS.items():
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "screener-factor-lab"}),
                                    timeout=timeout) as r:
            data = r.read()
        if data[:200].lstrip().lower().startswith(b"<!doctype html"):
            raise RuntimeError(f"{name}: Google Drive returned a page, not the file (quota exceeded?) - retry later")
        (osap_dir / name).write_bytes(data)
        out.append(osap_dir / name)
    return out


def _osap_month(date: str) -> str:
    """'1976-01-31', '1976m1' or '197601' as '197601'."""
    d = date.strip().strip('"')
    if "m" in d:
        y, m = d.split("m")
        return f"{int(y)}{int(m):02d}"
    if "-" in d:
        return d[:4] + d[5:7]
    return d[:6]


def read_osap(path: Path, signal: str) -> dict[str, dict[str, float]]:
    """{port: {yyyymm: monthly return in percent}} for one signal from an OSAP portfolio file - a
    single signal's csv, or a zip of the full set (one csv, a ``signalname`` column per row).
    Read as text, row by row: the full-set file is large and only one signal is wanted."""
    import csv
    import io

    def rows():
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                with z.open(member) as f:
                    yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
        else:
            with path.open(encoding="utf-8", newline="") as f:
                yield from csv.DictReader(f)

    out: dict[str, dict[str, float]] = {}

    def put(port: str, date: str, ret: str | None) -> None:
        if ret in (None, "", "NA"):
            return
        port = port.strip().strip('"')
        port = port if not port.isdigit() else f"{int(port):02d}"
        out.setdefault(port, {})[_osap_month(date)] = float(ret)

    for row in rows():
        if "signalname" in row and row["signalname"] != signal:
            continue
        if "port" in row:                              # long: one row per port and month
            put(row["port"], row["date"], row.get("ret"))
        else:                                          # wide (a single signal's file): port01 ... portLS
            for col, ret in row.items():
                if col and col.startswith("port"):
                    put(col[4:], row["date"], ret)
    if not out:
        raise ValueError(f"{path.name}: no portfolio returns for signal {signal!r}")
    return out


def osap_sort_data(name: str, osap_dir: Path = OSAP_DIR) -> dict:
    """A sort read from Open Source Asset Pricing files, in the shape ``sort_data`` gives: each
    group's monthly return (percent), lowest first, value- and equal-weighted, over the months in
    which every group has a return under both weightings."""
    spec = SORTS[name]
    by_w = {w: read_osap(osap_dir / spec[w], spec["signal"]) for w in ("vw", "ew")}
    cols = spec["columns"]
    months = sorted(set.intersection(*(set(by_w[w][c]) for w in ("vw", "ew") for c in cols)))
    return {"months": months, "source": "osap",
            "vw": [[round(by_w["vw"][c][m], 4) for m in months] for c in cols],
            "ew": [[round(by_w["ew"][c][m], 4) for m in months] for c in cols]}


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


def sort_data(name: str, raw_dir: Path = RAW_DIR) -> dict:
    """Monthly returns (percent) of a sorted-portfolio file's groups, lowest first, value- and
    equal-weighted."""
    spec = SORTS[name]
    if spec.get("source") == "osap":
        return osap_sort_data(name, raw_dir.parent / "osap")       # data/lab/osap beside data/lab/french
    fname, vw_mark = FRENCH_FILES[spec["file"]]
    dates, vw, _ = read_table(raw_dir / fname, vw_mark)
    dates_ew, ew, _ = read_table(raw_dir / fname, spec["ew"])
    if dates != dates_ew:
        raise ValueError(f"{fname}: the equal- and value-weighted tables cover different months")
    cols = spec["columns"]
    keep = [i for i in range(len(dates)) if all(vw[i][c] > MISSING and ew[i][c] > MISSING for c in cols)]
    out = {"months": [dates[i] for i in keep],
           "vw": [[round(vw[i][c], 2) for i in keep] for c in cols],
           "ew": [[round(ew[i][c], 2) for i in keep] for c in cols]}
    if spec.get("excess"):
        # the one-month T-bill rate (the factors file's RF) for the same months, for a paper that
        # reports returns over it
        mfile, mmark = FRENCH_FILES["factors_monthly"]
        f_dates, f_rows, _ = read_table(raw_dir / mfile, mmark)
        rf = {d: r[3] for d, r in zip(f_dates, f_rows)}
        out["rf"] = [round(rf[m], 2) if m in rf else None for m in out["months"]]
    return out


def sort_run(data: dict, start: str, end: str, weighting: str = "vw") -> dict:
    """Each group's average month and volatility over a period, and the last group minus the
    first, month by month, with its t-statistic."""
    idx = [i for i, m in enumerate(data["months"]) if start <= m <= end]
    groups = [[col[i] for i in idx] for col in data[weighting]]
    return {"months": [data["months"][i] for i in idx], "groups": [mean_t(g) for g in groups],
            "spread": mean_t([b - a for a, b in zip(groups[0], groups[-1])])}


def split_data(raw_dir: Path = RAW_DIR) -> dict:
    """Each factor among small companies and among big ones, monthly, in percent, over the
    months all four files share.

    The big-company factor is also split at the market's return into its two sides: "long" is
    the group the factor holds minus the market, "short" the market minus the group it is
    short. The two add up to the factor, month by month, and only the first is something a
    holder who shorts nothing can keep."""
    mfile, mmark = FRENCH_FILES["factors_monthly"]
    m_dates, m_rows, _ = read_table(raw_dir / mfile, mmark)
    market = {d: r[0] + r[3] for d, r in zip(m_dates, m_rows)}              # Mkt-RF + RF
    raw = {}
    for key in SPLIT_KEYS:
        fname, mark = FRENCH_FILES["split_" + key]
        dates, rows, _ = read_table(raw_dir / fname, mark)
        s = SPLIT_SIGN[key]
        held, shorted = (5, 3) if s > 0 else (3, 5)                         # big-high and big-low columns
        raw[key] = {d: (s * (r[2] - r[0]), s * (r[5] - r[3]), r[held] - market[d], market[d] - r[shorted])
                    for d, r in zip(dates, rows) if d in market and all(v > MISSING for v in r[:6])}
    months = sorted(set.intersection(*(set(v) for v in raw.values())))
    out = {"months": months}
    for key in SPLIT_KEYS:
        out[key] = {part: [round(raw[key][m][i], 4) for m in months]
                    for i, part in enumerate(SPLIT_PARTS)}
    return out


def quintile_data(raw_dir: Path = RAW_DIR) -> dict:
    """Value and momentum inside the smallest and the largest fifth of companies by size: the
    highest trait group minus the lowest, monthly, in percent. Same shape as `split_data`, so
    the same arithmetic and the same page serve both; "long" and "short" split the
    largest-fifth spread at the market's return."""
    mfile, mmark = FRENCH_FILES["factors_monthly"]
    m_dates, m_rows, _ = read_table(raw_dir / mfile, mmark)
    market = {d: r[0] + r[3] for d, r in zip(m_dates, m_rows)}              # Mkt-RF + RF
    (s_lo, s_hi), (b_lo, b_hi) = QUINT_COLS["small"], QUINT_COLS["big"]
    raw = {}
    for key in QUINT_KEYS:
        fname, mark = FRENCH_FILES["quint_" + key]
        dates, rows, _ = read_table(raw_dir / fname, mark)
        raw[key] = {d: (r[s_hi] - r[s_lo], r[b_hi] - r[b_lo], r[b_hi] - market[d], market[d] - r[b_lo])
                    for d, r in zip(dates, rows)
                    if d in market and all(r[c] > MISSING for c in (s_lo, s_hi, b_lo, b_hi))}
    months = sorted(set.intersection(*(set(v) for v in raw.values())))
    out = {"months": months}
    for key in QUINT_KEYS:
        out[key] = {part: [round(raw[key][m][i], 4) for m in months] for i, part in enumerate(SPLIT_PARTS)}
    return out


def double_data(raw_dir: Path = RAW_DIR) -> dict:
    """The 25 size-by-profitability portfolios, value-weighted, monthly, in percent: ``vw[c]`` is
    column c = 5 x size quintile + profitability quintile, both counted from 0 (smallest, weakest).
    Only months where all 25 are present."""
    fname, mark = FRENCH_FILES["double_size_op"]
    dates, rows, vintage = read_table(raw_dir / fname, mark)
    n = DOUBLE_SIZE * DOUBLE_TRAIT
    keep = [i for i in range(len(dates)) if all(v > MISSING for v in rows[i][:n])]
    return {"months": [dates[i] for i in keep], "vintage": vintage,
            "vw": [[round(rows[i][c], 2) for i in keep] for c in range(n)]}


def double_run(data: dict, start: str, end: str) -> dict:
    """Size inside each profitability quintile, over a period: every portfolio's average month;
    in each profitability row the smallest size quintile minus the largest (``small_big``) and the
    fourth minus the largest (``four_big``) - the size effect among large companies - each a
    monthly difference with its t-statistic; the same two averaged equally across the five rows
    month by month (``smbq``, Asness et al.'s SMBQ, and ``fourq``); and, inside the largest and the
    smallest size quintile, the most profitable fifth minus the least (``trait_big``, ``trait_small``)."""
    idx = [i for i, m in enumerate(data["months"]) if start <= m <= end]
    col = [[c[i] for i in idx] for c in data["vw"]]
    top = DOUBLE_SIZE - 1

    def cell(s: int, p: int) -> list[float]:
        return col[DOUBLE_TRAIT * s + p]

    def diff(a: list[float], b: list[float]) -> list[float]:
        return [x - y for x, y in zip(a, b)]

    sb = [diff(cell(0, p), cell(top, p)) for p in range(DOUBLE_TRAIT)]
    fb = [diff(cell(top - 1, p), cell(top, p)) for p in range(DOUBLE_TRAIT)]
    avg = lambda rows: [sum(v) / len(v) for v in zip(*rows)] if rows and rows[0] else []   # noqa: E731
    return {"months": [data["months"][i] for i in idx],
            "cells": [mean_t(c) for c in col],
            "rows": [{"small_big": mean_t(sb[p]), "four_big": mean_t(fb[p])} for p in range(DOUBLE_TRAIT)],
            "smbq": mean_t(avg(sb)), "fourq": mean_t(avg(fb)),
            "trait_big": mean_t(diff(cell(top, DOUBLE_TRAIT - 1), cell(top, 0))),
            "trait_small": mean_t(diff(cell(0, DOUBLE_TRAIT - 1), cell(0, 0)))}


def double_figures(data: dict, start: str, end: str) -> dict:
    """The figures a size-by-trait experiment compares with its paper, by key."""
    r = double_run(data, start, end)
    got = {"smbq": r["smbq"]["mean"], "t_smbq": r["smbq"]["t"], "fourq": r["fourq"]["mean"],
           "qj_big": r["trait_big"]["mean"], "t_qj_big": r["trait_big"]["t"],
           "qj_small": r["trait_small"]["mean"], "t_qj_small": r["trait_small"]["t"]}
    for p, row in enumerate(r["rows"], 1):
        got[f"sb_{p}"], got[f"t_sb_{p}"] = row["small_big"]["mean"], row["small_big"]["t"]
    return got


def criterion_result(data: dict, crit: dict) -> dict:
    """A rule written into a research note before its data was fetched, read on that data: the
    fourth size quintile minus the largest, averaged across the trait's five rows, over the rule's
    window. Refuted when the average is no larger than ``unconditional`` or its t-statistic is below
    ``min_t`` - the note's own words, carried on the record as ``rule``."""
    r = double_run(data, crit["start"], crit["end"])
    q = r["fourq"]
    return {"first": r["months"][0], "last": r["months"][-1], "months": len(r["months"]),
            "mean": q["mean"], "t": q["t"],
            "by_row": [row["four_big"]["mean"] for row in r["rows"]],
            "refuted": bool(q["mean"] <= crit["unconditional"] or q["t"] < crit["min_t"])}


def split_run(split: dict, start: str, end: str) -> dict:
    """Each factor's average month and t-statistic among small companies, among big ones, the
    difference between the two month by month, and the two sides of the big-company factor."""
    idx = [i for i, m in enumerate(split["months"]) if start <= m <= end]
    out = {"months": [split["months"][i] for i in idx], "factors": {}}
    for key in (k for k in split if k != "months"):
        part = {p: [split[key][p][i] for i in idx] for p in SPLIT_PARTS}
        out["factors"][key] = {**{p: mean_t(v) for p, v in part.items()},
                               "gap": mean_t([a - b for a, b in zip(part["small"], part["big"])])}
    return out


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


def residuals(y: list[float], x: list[float]) -> list[float]:
    """What is left of ``y`` after the part that moves with ``x`` (one regressor and an intercept;
    the intercept does not change a correlation, so only the slope is removed)."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return [b - beta * a for a, b in zip(x, y)]


def annualised(x: list[float]) -> dict | None:
    """A monthly percent series as a year: mean x 12, volatility x sqrt(12), and the Sharpe ratio
    (their ratio - for a long-short spread the T-bill rate cancels, so none is subtracted)."""
    s = mean_t(x)
    if s is None:
        return None
    return {**s, "mean_a": s["mean"] * 12, "std_a": s["std"] * math.sqrt(12),
            "sharpe": s["mean"] / s["std"] * math.sqrt(12) if s["std"] else None}


def combo_run(split: dict, factors: dict, start: str, end: str, group: str = "big") -> dict:
    """Value and momentum together (Asness, Moskowitz & Pedersen 2013): each spread, their 50/50
    mix, and how they move together - raw, and on what is left after the market's move, which is
    the correlation the paper reports. ``group`` is ``big`` (the big half of the market, nearest the
    paper's large, liquid universe) or ``all`` (the average of the small and big halves)."""
    mk = dict(zip(factors["months"], factors["mktrf"]))
    idx = [i for i, m in enumerate(split["months"]) if start <= m <= end and m in mk]

    def series(key):
        if group == "big":
            return [split[key]["big"][i] for i in idx]
        return [(split[key]["small"][i] + split[key]["big"][i]) / 2 for i in idx]
    v, mo = series("value"), series("momentum")
    mkt = [mk[split["months"][i]] for i in idx]
    out = {"months": [split["months"][i] for i in idx], "value": annualised(v), "momentum": annualised(mo),
           "combo": annualised([(a + b) / 2 for a, b in zip(v, mo)]), "corr": None, "corr_resid": None}
    if len(idx) >= 6:
        out["corr"] = correlation(v, mo)
        out["corr_resid"] = correlation(residuals(v, mkt), residuals(mo, mkt))
    return out


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
# 0.05 on the Sharpe ratio (half of a tenth), one point on an annual percentage (the mean and
# the volatility), one on excess kurtosis, 0.15 on skewness, and three points on a single
# extreme month (the best and the worst). The page prints every raw difference beside the
# verdict, so a stricter reader does not have to accept them.
TOLERANCES = {"max": 3.0, "min": 3.0, "mean": 1.0, "std": 1.0, "kurt": 1.0, "skew": 0.15,
              "sharpe": 0.05}

# When each experiment's tolerances were written, against when its figures were first seen
# (plan/factor-lab.md rule 3). Every reference record carries one of these keys as
# "tolerances_written", and a "tolerances_note" from its own comment wherever it is not
# "before"; the page prints the sentence and the note under the comparison, from one function.
TOLERANCES_WRITTEN = {
    "before": "These tolerances were written before the comparison was first run.",
    "after_first_comparison": "These tolerances were written after the first comparison had been seen.",
    "after_figures_seen": ("These tolerances were written before this comparison was run, but after "
                           "its figures had been measured elsewhere."),
}

EXPERIMENTS = [
    {
        "id": "risk-managed-momentum",
        "group": "Momentum",
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
            "tolerances_written": "after_first_comparison",
            "tolerances_note": ("They were set after the first comparison was seen, as round numbers a reader can "
                                "argue with, so read the differences themselves."),
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
        "group": "Momentum",
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
            "tolerances_written": "before",
            "tolerances_note": ("Written before the comparison was run, but not blind: experiment 1 had already shown "
                                "this library's momentum earning about 15% a year where this paper reports 17.9%, so "
                                "the figures built on that average were expected to miss, and an earlier probe had "
                                "printed the five worst months."),
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
        "group": "Across the factors",
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
            "tolerances_written": "before",
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
        "group": "Across the factors",
        "kind": "size",
        "topic": "Size",
        "title": "The size effect, and whether it reaches large stocks",
        "tab": "The size effect",
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
            "tolerances_written": "before",
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
    {
        "id": "low-volatility",
        "group": "Risk",
        "kind": "sort",
        "topic": "Risk",
        "title": "Do riskier stocks earn more?",
        "tab": "Risk and return",
        "paper": {
            "authors": "Andrew Ang, Robert J. Hodrick, Yuhang Xing and Xiaoyan Zhang",
            "year": 2006,
            "title": "The cross-section of volatility and expected returns",
            "journal": "Journal of Finance 61(1), 259-299",
            "doi": "10.1111/j.1540-6261.2006.00836.x",
        },
        "claim": ("They did not. Sorting US stocks each month into five groups by how volatile they had "
                  "just been, the most volatile fifth earned about 0.1% a month from 1963 to 2000, "
                  "against about 1.1% for the calmest fifth - a difference of nearly one percentage "
                  "point a month, with more than twice the risk."),
        "reference": {
            "source": ("The authors' NBER working paper 10852 (October 2004), Table VI, Panel A (page 48): "
                       "value-weighted portfolios sorted on total volatility, July 1963 to December "
                       "2000. Monthly means and standard deviations of total returns."),
            "start": "196307", "end": "200012", "weighting": "vw",
            # Written 2026-10-12 BEFORE the comparison was run; nothing had been computed on this
            # file. Expected beforehand: the most volatile fifth would miss, and the difference
            # with it - the library measures volatility over 60 days where the paper uses one
            # month, and draws its dividing lines differently, both of which make the library's
            # extreme group less extreme.
            "tolerances_written": "before",
            "figures": [
                {"key": "mean1", "label": "Calmest fifth: average month (%)", "paper": 1.06, "tolerance": 0.10, "where": "Table VI A"},
                {"key": "mean2", "label": "Second fifth: average month (%)", "paper": 1.15, "tolerance": 0.10, "where": "Table VI A"},
                {"key": "mean3", "label": "Middle fifth: average month (%)", "paper": 1.22, "tolerance": 0.10, "where": "Table VI A"},
                {"key": "mean4", "label": "Fourth fifth: average month (%)", "paper": 0.99, "tolerance": 0.10, "where": "Table VI A"},
                {"key": "mean5", "label": "Most volatile fifth: average month (%)", "paper": 0.09, "tolerance": 0.10, "where": "Table VI A"},
                {"key": "std1", "label": "Calmest fifth: volatility, a month (%)", "paper": 3.71, "tolerance": 0.40, "where": "Table VI A"},
                {"key": "std2", "label": "Second fifth: volatility, a month (%)", "paper": 4.48, "tolerance": 0.40, "where": "Table VI A"},
                {"key": "std3", "label": "Middle fifth: volatility, a month (%)", "paper": 5.63, "tolerance": 0.40, "where": "Table VI A"},
                {"key": "std4", "label": "Fourth fifth: volatility, a month (%)", "paper": 7.15, "tolerance": 0.40, "where": "Table VI A"},
                {"key": "std5", "label": "Most volatile fifth: volatility, a month (%)", "paper": 8.30, "tolerance": 0.40, "where": "Table VI A"},
                {"key": "spread", "label": "Most volatile fifth minus calmest, a month (%)", "paper": -0.97, "tolerance": 0.20, "where": "Table VI A"},
            ],
        },
        "verdict_note": ("Every miss is in the more volatile groups. The library measures volatility over "
                         "60 days and draws its dividing lines differently from the paper, so its most "
                         "volatile fifth is a tamer set of stocks, and earned far more than the paper's."),
        "sort": {
            "data": "low-volatility",
            "groups": ["Calmest fifth", "Second fifth", "Middle fifth", "Fourth fifth", "Most volatile fifth"],
            "ends": ["calmest", "most volatile"],
            "low": "calmest fifth", "high": "most volatile fifth",
            "spread": "Most volatile fifth minus calmest",
            "accent": [0],
            "key_accent": "The calmest fifth - the end the screener's Risk category favours",
            "key_rest": "The other four groups",
            "chart": "by how volatile the stocks had been",
            "presets": [["The paper \u00b7 1963\u20132000", "196307", "200012"],
                        ["After the paper \u00b7 2001\u2013now", "200101", "LAST"],
                        ["Since 2010", "201001", "LAST"], ["Everything", "FIRST", "LAST"]],
            "weightings": [["vw", "By market value \u00b7 paper"], ["ew", "Equally"]],
        },
        "conditions": [
            ["Portfolios", "Each month, every US-listed stock is put in one of five groups by the "
                           "variance of its daily returns over the previous 60 trading days."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Variance\"."],
            ["Weighting", "Value-weighted, as in the paper: bigger companies count for more within a "
                          "group. The equal-weighted version is one click away."],
            ["Returns", "Total returns, not returns above Treasury bills, as in the paper's table."],
            ["Costs", "None. The most volatile group is also the costliest to trade."],
        ],
        "open_details": [
            "The paper measures volatility over the previous month; the library uses 60 trading days. "
            "The library also sets its dividing lines differently. Both make the library's most "
            "volatile group less extreme than the paper's.",
            "The reference figures are from the authors' 2004 working paper. The published article's "
            "table has not been read here.",
            "The paper's headline is about volatility left over after the market, size and value are "
            "accounted for. This page reruns its simpler table, on total volatility; the library "
            "publishes the other sort too.",
            "Only the averages and standard deviations are reproduced, not the paper's risk-adjusted "
            "returns.",
        ],
        "in_the_screener": (
            "The screener's Risk category scores a company higher for moving less: lower volatility, "
            "lower sensitivity to the market, a shallower worst fall. The methodology calls the "
            "low-volatility anomaly one of the most robust findings in finance. This is that finding "
            "on public data, and the controls show how it has held since. The screener ranks large "
            "companies only, where the most volatile stocks are far tamer than the group shown here."),
    },
    {
        "id": "earnings-yield",
        "group": "Value",
        "kind": "sort",
        "topic": "Valuation",
        "title": "Does a high earnings yield pay?",
        "tab": "Earnings yield",
        "paper": {
            "authors": "Eugene F. Fama and Kenneth R. French",
            "year": 1992,
            "title": "The cross-section of expected stock returns",
            "journal": "Journal of Finance 47(2), 427-465",
            "doi": "10.1111/j.1540-6261.1992.tb04398.x",
        },
        "claim": ("Sorting companies with positive earnings into ten groups by earnings divided by share "
                  "price, the group with the highest earnings yield averaged about 1.7% a month from "
                  "1963 to 1990 and the group with the lowest about 1.0%, rising almost step by step "
                  "in between."),
        "reference": {
            "source": ("The published article, Table IV, Panel B (page 443): average monthly returns of "
                       "portfolios formed on earnings-price ratio, equal-weighted, July 1963 to December "
                       "1990. The paper splits the lowest and highest tenths in half; those halves are "
                       "combined here by the number of firms in each."),
            "start": "196307", "end": "199012", "weighting": "ew",
            # Written 2026-10-12 BEFORE the comparison was run; nothing had been computed on this
            # file. The same sizes as experiment 4, which reruns another table of the same paper.
            # Expected beforehand: the two end groups are the likeliest to miss - the paper draws
            # its dividing lines across all its stocks and leaves out financial companies, the
            # library draws them on New York Stock Exchange stocks and keeps financials. Unlike
            # the size table, both samples need accounting data, so a closer match than
            # experiment 4's was expected.
            "tolerances_written": "before",
            "figures": [
                {"key": "mean1", "label": "Lowest earnings yield: average month (%)", "paper": 0.984, "tolerance": 0.05, "where": "Table IV B (its two halves, averaged by number of firms)"},
                {"key": "mean2", "label": "Earnings-yield decile 2: average month (%)", "paper": 0.94, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean3", "label": "Earnings-yield decile 3: average month (%)", "paper": 1.03, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean4", "label": "Earnings-yield decile 4: average month (%)", "paper": 1.18, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean5", "label": "Earnings-yield decile 5: average month (%)", "paper": 1.22, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean6", "label": "Earnings-yield decile 6: average month (%)", "paper": 1.33, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean7", "label": "Earnings-yield decile 7: average month (%)", "paper": 1.42, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean8", "label": "Earnings-yield decile 8: average month (%)", "paper": 1.46, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean9", "label": "Earnings-yield decile 9: average month (%)", "paper": 1.57, "tolerance": 0.05, "where": "Table IV B"},
                {"key": "mean10", "label": "Highest earnings yield: average month (%)", "paper": 1.73, "tolerance": 0.05, "where": "Table IV B (its two halves, averaged by number of firms)"},
                {"key": "spread", "label": "Highest earnings yield minus lowest, a month (%)", "paper": 0.746, "tolerance": 0.08, "where": "Table IV B, derived"},
            ],
        },
        "verdict_note": ("The misses are small and mostly at the two ends, where the paper and the library "
                         "draw their dividing lines differently. The rise from group to group and the gap "
                         "between the ends both match."),
        "sort": {
            "data": "earnings-yield",
            "groups": ["Lowest earnings yield", "Decile 2", "Decile 3", "Decile 4", "Decile 5", "Decile 6", "Decile 7", "Decile 8", "Decile 9", "Highest earnings yield"],
            "ends": ["lowest earnings yield", "highest"],
            "low": "group with the lowest earnings yield", "high": "group with the highest",
            "spread": "Highest earnings yield minus lowest",
            "accent": [9],
            "key_accent": "The highest earnings yield - the end the screener's Valuation category favours",
            "key_rest": "The other nine groups",
            "chart": "by earnings yield",
            "presets": [["The paper \u00b7 1963\u20131990", "196307", "199012"],
                        ["After the paper \u00b7 1991\u2013now", "199101", "LAST"],
                        ["Since 2007", "200701", "LAST"], ["Everything", "FIRST", "LAST"]],
            "weightings": [["ew", "Equally \u00b7 paper"], ["vw", "By market value"]],
        },
        "conditions": [
            ["Portfolios", "Each June, companies with positive earnings are put in one of ten groups by "
                           "their earnings for the last fiscal year divided by their market value at "
                           "the end of December."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Earnings/Price\"."],
            ["Weighting", "Equal-weighted, as in the paper's table. The value-weighted version is one "
                          "click away."],
            ["Left out", "Companies with losses. The paper reports them as a separate group; so does "
                         "the library, and they are not shown here."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper draws the lines between groups across all the stocks in its sample and "
            "leaves out financial companies; the library draws them on New York Stock Exchange "
            "stocks and keeps financials. The end groups are where that matters most.",
            "The lowest and highest tenths are combined from the paper's halves by its own count "
            "of firms, which approximates what it would have printed for the whole tenth.",
            "The paper's main result on earnings yield is in its regressions: once size and "
            "book-to-market are accounted for, earnings yield adds little. That is not reproduced "
            "here.",
        ],
        "in_the_screener": (
            "The screener's Valuation category is built from yields of this kind - free cash flow "
            "and earnings set against what the whole business is priced at. This is the nearest "
            "sort the public data carries: earnings against the share price. Two differences "
            "matter. The screener compares each company with others in its own sector, which this "
            "sort does not; and it ranks large companies only, where the spread between groups "
            "has been narrower."),
    },
    {
        "id": "market-beta",
        "group": "Risk",
        "kind": "sort",
        "topic": "Risk",
        "title": "Does market risk pay?",
        "tab": "Market beta",
        "paper": {
            "authors": "Eugene F. Fama and Kenneth R. French",
            "year": 1992,
            "title": "The cross-section of expected stock returns",
            "journal": "Journal of Finance 47(2), 427-465",
            "doi": "10.1111/j.1540-6261.1992.tb04398.x",
        },
        "claim": ("If investors are paid for bearing market risk, stocks that move more with the market "
                  "- higher beta - should earn more. Sorted into ten groups by beta, they did not: from "
                  "1963 to 1990 the average return was flat, about 1.2% to 1.3% a month, from the "
                  "lowest-beta group to the highest."),
        "reference": {
            "source": ("The published article, Table II, Panel B (page 437): average monthly returns of "
                       "portfolios formed on pre-ranking beta, equal-weighted, July 1963 to December "
                       "1990. The lowest and highest tenths are combined from the paper's halves by "
                       "the number of firms in each."),
            "start": "196307", "end": "199012", "weighting": "ew",
            # Written 2026-10-12 BEFORE the comparison was run; nothing had been computed on this
            # file. The same sizes as experiments 4 and 6, which rerun other tables of this
            # paper. Expected beforehand, from experiment 4: the levels would sit below the
            # paper's in most groups (its sample is non-financial firms with accounting data;
            # the library's is every stock), and the flatness - the gap between the end groups -
            # would match.
            "tolerances_written": "before",
            "figures": [
                {"key": "mean1", "label": "Lowest beta: average month (%)", "paper": 1.2, "tolerance": 0.05, "where": "Table II B (its two halves, averaged by number of firms)"},
                {"key": "mean2", "label": "Beta decile 2: average month (%)", "paper": 1.32, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean3", "label": "Beta decile 3: average month (%)", "paper": 1.26, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean4", "label": "Beta decile 4: average month (%)", "paper": 1.31, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean5", "label": "Beta decile 5: average month (%)", "paper": 1.3, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean6", "label": "Beta decile 6: average month (%)", "paper": 1.3, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean7", "label": "Beta decile 7: average month (%)", "paper": 1.23, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean8", "label": "Beta decile 8: average month (%)", "paper": 1.23, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean9", "label": "Beta decile 9: average month (%)", "paper": 1.33, "tolerance": 0.05, "where": "Table II B"},
                {"key": "mean10", "label": "Highest beta: average month (%)", "paper": 1.238, "tolerance": 0.05, "where": "Table II B (its two halves, averaged by number of firms)"},
                {"key": "spread", "label": "Highest beta minus lowest, a month (%)", "paper": 0.038, "tolerance": 0.08, "where": "Table II B, derived"},
            ],
        },
        "verdict_note": ("The gap between the end groups, which is the paper's point, matches. Every miss "
                         "sits below the paper's figure, as experiment 4 found for another panel of the "
                         "same table: the paper leaves out financial companies and firms without "
                         "accounting data."),
        "sort": {
            "data": "market-beta",
            "groups": ["Lowest beta", "Decile 2", "Decile 3", "Decile 4", "Decile 5", "Decile 6", "Decile 7", "Decile 8", "Decile 9", "Highest beta"],
            "ends": ["lowest beta", "highest"],
            "low": "group least sensitive to the market", "high": "group most sensitive",
            "spread": "Highest beta minus lowest",
            "accent": [0],
            "key_accent": "The lowest beta - the end the screener's Risk category favours",
            "key_rest": "The other nine groups",
            "chart": "by sensitivity to the market (beta)",
            "presets": [["The paper \u00b7 1963\u20131990", "196307", "199012"],
                        ["After the paper \u00b7 1991\u2013now", "199101", "LAST"],
                        ["Since 2010", "201001", "LAST"], ["Everything", "FIRST", "LAST"]],
            "weightings": [["ew", "Equally \u00b7 paper"], ["vw", "By market value"]],
        },
        "conditions": [
            ["Portfolios", "Each June, every US-listed stock with enough history is put in one of ten "
                           "groups by its beta: how much it has moved with the market over the "
                           "previous five years of monthly returns."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Market Beta\"."],
            ["Weighting", "Equal-weighted, as in the paper's table. The value-weighted version is one "
                          "click away."],
            ["Returns", "Total returns. Read the volatility column beside them: a higher-beta group "
                        "that earns the same has earned it with more risk."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper's sample is non-financial firms that also appear in the accounting database "
            "(its page 429); the library's groups are every listed stock. Experiment 4 reruns "
            "another panel of the same table and shows how much that lowers the averages.",
            "The paper estimates beta over two to five years and assigns it through size groups; "
            "the library uses five years and a different estimator.",
            "The lowest and highest tenths are combined from the paper's halves by its own count "
            "of firms.",
        ],
        "in_the_screener": (
            "Beta is one of the three measures in the screener's Risk category, which scores a "
            "company higher for moving less with the market. This is the published result that "
            "makes that defensible: if higher beta has not been paid for, a lower-beta company has "
            "given up little return for carrying less market risk. It is a statement about risk "
            "taken for the same return, not a promise of a higher one."),
    },
    {
        "id": "big-and-small",
        "group": "Across the factors",
        "kind": "split",
        "topic": "Value, profitability, investment and momentum among large companies",
        "title": "Do the factors work among large companies?",
        "tab": "Large companies",
        "paper": {
            "authors": "Eugene F. Fama and Kenneth R. French",
            "year": 2015,
            "title": "A five-factor asset pricing model",
            "journal": "Journal of Financial Economics 116(1), 1-22",
            "doi": "10.1016/j.jfineco.2014.10.010",
        },
        "claim": ("Less well than among small ones. Split by company size, from 1963 to 2013 the value "
                  "factor averaged 0.53% a month among small companies and 0.21% among big ones; "
                  "profitability 0.33% and 0.17%; investment 0.45% and 0.22%. Among big companies only "
                  "investment reaches two standard errors."),
        "reference": {
            "source": ("The published article, Table 4, Panel A, second block (page 7): the 2x3 factors "
                       "computed separately among small and big stocks, July 1963 to December 2013."),
            "start": "196307", "end": "201312",
            # Written 2026-10-12 BEFORE the comparison was run; nothing had been computed on
            # these files. (It then landed 12 of 18, not the 17 or so expected: the misses are
            # recorded in verdict_note and the tolerances were left as written.) The same sizes as experiment 3, which reruns the first block of the
            # same table and landed 20 of 21 within. Expected: the same - these are the authors'
            # own portfolios, rebuilt on a newer database.
            "tolerances_written": "before",
            "figures": [
                {"key": "mean_value_small", "label": "Value among small companies: average month (%)", "paper": 0.53, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_value_small", "label": "Value among small companies: t-statistic", "paper": 4.05, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_value_big", "label": "Value among big companies: average month (%)", "paper": 0.21, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_value_big", "label": "Value among big companies: t-statistic", "paper": 1.69, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_value_gap", "label": "Value small minus big: average month (%)", "paper": 0.32, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_value_gap", "label": "Value small minus big: t-statistic", "paper": 2.94, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_profitability_small", "label": "Profitability among small companies: average month (%)", "paper": 0.33, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_profitability_small", "label": "Profitability among small companies: t-statistic", "paper": 3.06, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_profitability_big", "label": "Profitability among big companies: average month (%)", "paper": 0.17, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_profitability_big", "label": "Profitability among big companies: t-statistic", "paper": 1.81, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_profitability_gap", "label": "Profitability small minus big: average month (%)", "paper": 0.16, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_profitability_gap", "label": "Profitability small minus big: t-statistic", "paper": 1.48, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_investment_small", "label": "Investment among small companies: average month (%)", "paper": 0.45, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_investment_small", "label": "Investment among small companies: t-statistic", "paper": 5.49, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_investment_big", "label": "Investment among big companies: average month (%)", "paper": 0.22, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_investment_big", "label": "Investment among big companies: t-statistic", "paper": 2.0, "tolerance": 0.25, "where": "Table 4A"},
                {"key": "mean_investment_gap", "label": "Investment small minus big: average month (%)", "paper": 0.23, "tolerance": 0.03, "where": "Table 4A"},
                {"key": "t_investment_gap", "label": "Investment small minus big: t-statistic", "paper": 2.29, "tolerance": 0.25, "where": "Table 4A"},
            ],
        },
        "verdict_note": ("Every miss is in value or profitability, where today's library gives big "
                         "companies a little more than the paper printed. On today's data all three "
                         "factors sit almost exactly at two standard errors among big companies, so the "
                         "paper's line between the one that reaches it and the two that fall short does "
                         "not survive the rebuild."),
        "conditions": [
            ["Portfolios", "Each June, US stocks are split into small and big at the median market "
                           "value of New York Stock Exchange companies, and within each half into low, "
                           "middle and high on one trait. A factor among big companies is the high "
                           "group minus the low group, both from the big half."],
            ["Data", "Kenneth R. French's Data Library: the \"6 Portfolios\" files formed on size and "
                     "book-to-market, operating profitability, investment, and prior return."],
            ["Momentum", "Shown beside the paper's three. It is not in the paper's table, so it has no "
                         "reference figure and is not counted in the comparison."],
            ["Weighting", "Value-weighted within each of the six portfolios, as in the paper."],
            ["The two sides", "Each big-company factor is also split at the market's return: the group "
                              "it holds minus the market, and the market minus the group it is short. "
                              "The market is all US stocks, value-weighted. This split is not in the "
                              "paper's table and is not counted in the comparison."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "\"Big\" here is the larger half of New York Stock Exchange companies. The screener's "
            "companies are larger still - roughly the top fifth - so this is the nearest public "
            "evidence for its universe, not that universe itself.",
            "The library has been rebuilt on a newer database since the paper.",
            "The traits are the factors' own: book-to-market, operating profitability and asset "
            "growth. The screener measures value, quality and investment with different metrics, "
            "compared within sectors.",
        ],
        "in_the_screener": (
            "Every company the screener ranks sits in the \"big\" half, so these are the versions of "
            "the factors that bear on it most directly, and they are the weaker versions. This is "
            "the experiment to read before trusting a long-run average quoted for all stocks, and "
            "it is the main input to the open questions about the screener's weights listed on the "
            "front page."),
    },
    {
        "id": "largest-fifth",
        "group": "Across the factors",
        "kind": "split",
        "topic": "Value and momentum in the largest fifth of companies",
        "title": "Are value and momentum still there in the largest companies?",
        "tab": "The largest fifth",
        "paper": {
            "authors": "Ronen Israel and Tobias J. Moskowitz",
            "year": 2013,
            "title": "The role of shorting, firm size, and time on market anomalies",
            "journal": "Journal of Financial Economics 108(2), 275-301",
            "doi": "10.1016/j.jfineco.2012.11.005",
        },
        "claim": ("Momentum, yes; value, barely. Sorting US stocks into fifths by size and then by each "
                  "trait, from 1926 to 2011 the highest-value fifth of the largest companies earned 3.7% a "
                  "year more than the lowest (t-statistic 1.90), against 11.2% among the smallest. Past "
                  "winners beat past losers by 7.5% a year among the largest (t-statistic 2.95) and 10.9% "
                  "among the smallest."),
        "reference": {
            "source": ("The published article, Table 3 (page 283): returns of the fifth-minus-first "
                       "portfolios within the smallest and largest size quintiles, in percent a year."),
            "start": "192701", "end": "201112", "scale": 12,
            # Written 2026-10-13 BEFORE the comparison was run; nothing had been computed on these
            # two files. Half a point a year on a spread and a quarter on a t-statistic. Expected:
            # most within - the paper used this library's 2011 files - with value the likelier
            # miss, because the paper's value sample starts six months earlier (July 1926) and
            # the library has been rebuilt since. (It landed 5 of 12: all four figures for the
            # largest fifth within, every miss involving the smallest fifth. The tolerances
            # stay as written; verdict_note says where the misses are.)
            "tolerances_written": "before",
            "figures": [
                {"key": "mean_value_small", "label": "Value in the smallest fifth: high minus low, % a year", "paper": 11.22, "tolerance": 0.5, "where": "Table 3"},
                {"key": "t_value_small", "label": "Value in the smallest fifth: t-statistic", "paper": 3.87, "tolerance": 0.25, "where": "Table 3"},
                {"key": "mean_value_big", "label": "Value in the largest fifth: high minus low, % a year", "paper": 3.7, "tolerance": 0.5, "where": "Table 3"},
                {"key": "t_value_big", "label": "Value in the largest fifth: t-statistic", "paper": 1.9, "tolerance": 0.25, "where": "Table 3"},
                {"key": "mean_value_gap", "label": "Value smallest minus largest: high minus low, % a year", "paper": 7.13, "tolerance": 0.5, "where": "Table 3"},
                {"key": "t_value_gap", "label": "Value smallest minus largest: t-statistic", "paper": 2.1, "tolerance": 0.25, "where": "Table 3"},
                {"key": "mean_momentum_small", "label": "Momentum in the smallest fifth: high minus low, % a year", "paper": 10.87, "tolerance": 0.5, "where": "Table 3"},
                {"key": "t_momentum_small", "label": "Momentum in the smallest fifth: t-statistic", "paper": 4.5, "tolerance": 0.25, "where": "Table 3"},
                {"key": "mean_momentum_big", "label": "Momentum in the largest fifth: high minus low, % a year", "paper": 7.49, "tolerance": 0.5, "where": "Table 3"},
                {"key": "t_momentum_big", "label": "Momentum in the largest fifth: t-statistic", "paper": 2.95, "tolerance": 0.25, "where": "Table 3"},
                {"key": "mean_momentum_gap", "label": "Momentum smallest minus largest: high minus low, % a year", "paper": 3.42, "tolerance": 0.5, "where": "Table 3"},
                {"key": "t_momentum_gap", "label": "Momentum smallest minus largest: t-statistic", "paper": 1.56, "tolerance": 0.25, "where": "Table 3"},
            ],
        },
        "verdict_note": ("All four figures for the largest fifth, which is where the screener's "
                         "companies are, match. Every miss involves the smallest fifth. The paper "
                         "describes forming each trait's groups inside each size group; the library's "
                         "files today form them independently of size, and have been rebuilt on a "
                         "newer database."),
        "split": {
            "data": "quintiles", "keys": ["value", "momentum"],
            "big": "the largest fifth of companies", "small": "the smallest fifth",
            "big_short": "Largest fifth", "small_short": "Smallest fifth",
            "key_big": "In the largest fifth of companies by size, where most of the screener's sit",
            "key_small": "In the smallest fifth",
            "halves": ("The smallest and largest fifths are by market value, on New York Stock Exchange "
                       "breakpoints; under each trait is the screener category it stands behind."),
            "against": ("The paper's figures are in percent a year, so each average month here is "
                        "multiplied by twelve for the comparison. Its risk-adjusted figures are not "
                        "rerun."),
            "presets": [["The paper \u00b7 1927\u20132011", "192701", "201112"],
                        ["After the paper \u00b7 2012\u2013now", "201201", "LAST"],
                        ["Since 2000", "200001", "LAST"], ["1963\u20131999", "196307", "199912"],
                        ["Everything", "FIRST", "LAST"]],
        },
        "conditions": [
            ["Portfolios", "The library's 25 portfolios formed on size and book-to-market, and on size "
                           "and prior return: five size groups on New York Stock Exchange breakpoints, "
                           "five groups on the trait within them. The spread is the highest group minus "
                           "the lowest, inside one size group."],
            ["Momentum", "Return over the prior twelve months, leaving out the latest one."],
            ["Weighting", "Value-weighted within each portfolio, as in the paper."],
            ["Sample", "From January 1927, the first month both files cover. The paper's value sample "
                       "starts in July 1926."],
            ["The two sides", "In the largest fifth each spread is also split at the market's return: "
                              "the high group minus the market, and the market minus the low group. "
                              "The paper does this with a risk-adjusted return instead, so its long-side "
                              "figures are not comparable with these and are not in the comparison."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper's risk-adjusted (market-model) figures, which are its headline, are not rerun "
            "here: 2.2% a year for value and 10.2% for momentum in the largest fifth.",
            "The library has been rebuilt on a newer database since the paper.",
            "The largest fifth by New York Stock Exchange breakpoints averaged $36 billion in 2011 "
            "by the paper's count. It is close to the upper part of the screener's universe, not "
            "the same list.",
        ],
        "in_the_screener": (
            "This is the nearest published test of the two traits where the screener's companies "
            "are. Its \"After the paper\" and \"Since 2000\" settings are the ones to read: the "
            "review of Valuation's and Momentum's weights rests on them, and on whether what "
            "experiment 8 shows for the larger half of companies also shows in the largest fifth."),
    },
    {
        "id": "profitability",
        "group": "Quality",
        "kind": "sort",
        "topic": "Quality",
        "title": "Do profitable companies earn more?",
        "tab": "Profitability",
        "paper": {
            "authors": "Robert Novy-Marx",
            "year": 2013,
            "title": "The other side of value: The gross profitability premium",
            "journal": "Journal of Financial Economics 108(1), 1-28",
            "doi": "10.1016/j.jfineco.2013.01.003",
        },
        "claim": ("Sorted into five groups by gross profits (sales minus the cost of goods sold) against "
                  "assets, the most profitable fifth of companies beat the least profitable by 0.31% a "
                  "month from 1963 to 2010, weighting each company by its market value - even though "
                  "profitable companies tend to carry high prices against their book value."),
        "reference": {
            "source": ("The paper, Table 2, Panel A: value-weighted average excess returns (over the "
                       "one-month T-bill) of quintiles sorted on gross profits-to-assets, NYSE break "
                       "points, financial firms excluded, July 1963 to December 2010. Read from the "
                       "accepted draft of June 2012, whose Table 2 is the published one (0.31% a "
                       "month, t 2.49, both in its text and in the published abstract)."),
            "start": "196307", "end": "201012", "weighting": "vw", "excess": True,
            # Written 2026-10-15 BEFORE the comparison was run; nothing had been computed on this
            # file. Wider than the 0.05 of experiments 4-7, which rerun a table on the same
            # variable the library sorts on: here the library sorts on a DIFFERENT profitability -
            # operating profits (also net of selling, general and administrative costs and
            # interest) against book equity, not gross profits against assets - and keeps the
            # financial firms the paper drops. The paper's own footnote 4 says keeping financials
            # cuts its spread to 0.25%. Expected beforehand: the pattern rises with profitability
            # and the spread lands below 0.31; a miss is a finding about the two measures, not a
            # number to tune.
            "tolerances_written": "before",
            "figures": [
                {"key": "mean1", "label": "Least profitable fifth: average excess month (%)", "paper": 0.31, "tolerance": 0.10, "where": "Table 2 A"},
                {"key": "mean2", "label": "Profitability quintile 2: average excess month (%)", "paper": 0.41, "tolerance": 0.10, "where": "Table 2 A"},
                {"key": "mean3", "label": "Profitability quintile 3: average excess month (%)", "paper": 0.52, "tolerance": 0.10, "where": "Table 2 A"},
                {"key": "mean4", "label": "Profitability quintile 4: average excess month (%)", "paper": 0.41, "tolerance": 0.10, "where": "Table 2 A"},
                {"key": "mean5", "label": "Most profitable fifth: average excess month (%)", "paper": 0.62, "tolerance": 0.10, "where": "Table 2 A"},
                {"key": "spread", "label": "Most profitable minus least, a month (%)", "paper": 0.31, "tolerance": 0.10, "where": "Table 2 A (H-L)"},
            ],
        },
        "verdict_note": ("Every figure lands within its tolerance, and the pattern - returns rising with "
                         "profitability - matches. The spread is at the edge of its range, below the "
                         "paper's, as expected beforehand: the public file sorts on a narrower "
                         "profitability and keeps the financial companies the paper drops, which the "
                         "paper itself reports lowers its spread."),
        "sort": {
            "data": "profitability",
            "groups": ["Least profitable", "Quintile 2", "Quintile 3", "Quintile 4", "Most profitable"],
            "ends": ["least profitable", "most"],
            "low": "least profitable fifth", "high": "most profitable fifth",
            "spread": "Most profitable minus least",
            "accent": [4],
            "key_accent": "The most profitable fifth - the end the screener's Quality category favours",
            "key_rest": "The other four groups",
            "chart": "by operating profitability",
            "presets": [["The paper \u00b7 1963\u20132010", "196307", "201012"],
                        ["After the paper \u00b7 2011\u2013now", "201101", "LAST"],
                        ["Since 2000", "200001", "LAST"], ["Everything", "FIRST", "LAST"]],
            "weightings": [["vw", "By market value \u00b7 paper"], ["ew", "Equally"]],
        },
        "conditions": [
            ["Portfolios", "Each June, US-listed companies are put in one of five groups by operating "
                           "profits for the last fiscal year (sales minus the cost of goods sold, selling "
                           "and administrative costs and interest) divided by book equity, with the "
                           "dividing lines drawn on New York Stock Exchange stocks."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Operating Profitability\"."],
            ["Weighting", "By market value, as in the paper's table. The equal-weighted version is one "
                          "click away."],
            ["Returns", "The comparison with the paper subtracts the T-bill rate's average over the same "
                        "months, as the paper's table does; the chart shows total returns."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper sorts on gross profits against total assets; the public file sorts on operating "
            "profits against book equity. Novy-Marx argues gross profit is the cleaner measure because "
            "it sits above the costs companies spend on growth (advertising, research), and finds "
            "operating-profit measures weaker in his sample. The screener's Quality category uses both "
            "kinds: gross profit against assets, and return on invested capital.",
            "The paper leaves out financial companies; the public file keeps them. The paper's own "
            "footnote 4 reports that keeping them lowers its spread from 0.31% to 0.25% a month.",
            "The paper's stronger result is after adjusting for value (its Fama-French alpha of 0.52% a "
            "month), because profitable companies tend to have low book-to-market ratios. That is not "
            "reproduced here.",
        ],
        "in_the_screener": (
            "Quality is 22% of the composite, and gross profit against assets - this paper's measure - is "
            "22% of Quality. The screener compares it within each sector, which this sort does not, and "
            "ranks large companies only; experiment 8 finds profitability the one factor that holds up "
            "among large companies."),
    },
    {
        "id": "cash-flow-yield",
        "group": "Value",
        "kind": "sort",
        "topic": "Valuation",
        "title": "Does a high cash-flow yield pay?",
        "tab": "Cash-flow yield",
        "paper": {
            "authors": "Josef Lakonishok, Andrei Shleifer and Robert W. Vishny",
            "year": 1994,
            "title": "Contrarian investment, extrapolation, and risk",
            "journal": "Journal of Finance 49(5), 1541-1578",
            "doi": "10.1111/j.1540-6261.1994.tb04772.x",
        },
        "claim": ("Sorting NYSE and AMEX stocks each year into ten groups by cash flow (earnings plus "
                  "depreciation) against market value, the highest cash-flow yields beat the lowest by "
                  "about 8% in the first year after the sort, size-adjusted, from 1968 to 1990 - the "
                  "'value' stocks the market had priced for disappointment."),
        "reference": {
            "source": ("The NBER working paper (w4360, 1993), Table 2: size-adjusted returns "
                       "of equal-weighted C/P deciles, held through the first year after formation (AB1), portfolios "
                       "formed each April 1968-1989. The published article's table was not available to "
                       "read; the working paper's text and table agree with each other (8.8% a year over "
                       "five years, 42.9% cumulative). Its deciles are size-adjusted, so they are compared "
                       "here as each decile against the average of the ten, in percent a year."),
            "start": "196807", "end": "199006", "weighting": "ew", "relative": True, "scale": 12,
            # Written 2026-10-15 BEFORE the comparison was run; nothing had been computed on this
            # file. The paper's AB1 row, each decile minus its ten-decile average (0.72%), in percent
            # a year. Wider than the 0.05%-a-month of experiments 4-7 (0.6% a year): the library
            # forms its groups in June on NYSE breakpoints with NASDAQ stocks, adds deferred taxes to
            # cash flow, and reports raw monthly returns compounded here as 12x the average month,
            # where the paper used April formation, NYSE/AMEX, annual buy-and-hold and a size
            # adjustment. Expected beforehand: the rise from low to high yield matches; the lowest
            # decile (where the paper's -4.9% sits) is the likeliest to miss.
            "tolerances_written": "before",
            "figures": [
                {"key": "mean1", "label": "Lowest cash-flow yield, against the average (% a year)", "paper": -5.62, "tolerance": 2.0, "where": "Table 2, AB1, less the ten-decile average"},
                {"key": "mean2", "label": "Decile 2, against the average (% a year)", "paper": -1.92, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean3", "label": "Decile 3, against the average (% a year)", "paper": -0.92, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean4", "label": "Decile 4, against the average (% a year)", "paper": -0.72, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean5", "label": "Decile 5, against the average (% a year)", "paper": 0.28, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean6", "label": "Decile 6, against the average (% a year)", "paper": 0.18, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean7", "label": "Decile 7, against the average (% a year)", "paper": 0.68, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean8", "label": "Decile 8, against the average (% a year)", "paper": 2.48, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean9", "label": "Decile 9, against the average (% a year)", "paper": 2.78, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "mean10", "label": "Highest cash-flow yield, against the average (% a year)", "paper": 2.78, "tolerance": 2.0, "where": "Table 2, AB1"},
                {"key": "spread", "label": "Highest cash-flow yield minus lowest (% a year)", "paper": 8.4, "tolerance": 4.0, "where": "Table 2, AB1, decile 10 minus decile 1"},
            ],
        },
        "verdict_note": ("Every figure lands within its tolerance and the rise from the lowest yield to the "
                         "highest matches. The public sort's spread is wider than the paper's, mostly at "
                         "the low end - it keeps NASDAQ stocks, which the paper's sample did not."),
        "sort": {
            "data": "cash-flow-yield",
            "groups": ["Lowest cash-flow yield", "Decile 2", "Decile 3", "Decile 4", "Decile 5", "Decile 6", "Decile 7", "Decile 8", "Decile 9", "Highest cash-flow yield"],
            "ends": ["lowest cash-flow yield", "highest"],
            "low": "group with the lowest cash-flow yield", "high": "group with the highest",
            "spread": "Highest cash-flow yield minus lowest",
            "accent": [9],
            "key_accent": "The highest cash-flow yield - the end the screener's Valuation category favours",
            "key_rest": "The other nine groups",
            "chart": "by cash-flow yield",
            "presets": [["The paper \u00b7 1968\u20131990", "196807", "199006"],
                        ["After the paper \u00b7 1990\u2013now", "199007", "LAST"],
                        ["Since 2007", "200701", "LAST"], ["Everything", "FIRST", "LAST"]],
            "weightings": [["ew", "Equally \u00b7 paper"], ["vw", "By market value"]],
        },
        "conditions": [
            ["Portfolios", "Each June, companies with positive cash flow are put in one of ten groups by "
                           "cash flow for the last fiscal year (earnings before extraordinary items, plus "
                           "depreciation and deferred taxes) divided by market value, with the dividing "
                           "lines drawn on New York Stock Exchange stocks."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Cash Flow/Price\"."],
            ["Weighting", "Equal-weighted, as in the paper. The value-weighted version is one click away."],
            ["Returns", "The comparison with the paper sets each group against the average of the ten, "
                        "because the paper's returns are size-adjusted; the chart shows total returns."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper reports size-adjusted returns (each stock against companies of its size) and "
            "holds each portfolio for a year from each April; the public file reports raw monthly returns "
            "of portfolios formed each June, and includes NASDAQ stocks.",
            "The paper's table was read from its 1993 NBER working paper; the published article's "
            "version could not be opened.",
            "Its stronger results are over five years after formation; only the first year is "
            "compared here, the one that matches a monthly-rebalanced screen.",
        ],
        "in_the_screener": (
            "Free-cash-flow yield is the screener's heaviest single metric - 45% of Valuation, about "
            "10% of the composite - and it differs from this sort in two ways: it subtracts capital "
            "spending from operating cash flow, and it divides by the value of the whole business "
            "(enterprise value), not of the shares alone. It is also compared within each sector."),
    },
    {
        "id": "accruals",
        "group": "Quality",
        "kind": "sort",
        "topic": "Quality",
        "title": "Do companies whose earnings are mostly accruals do worse?",
        "tab": "Accruals",
        "paper": {
            "authors": "Richard G. Sloan",
            "year": 1996,
            "title": "Do stock prices fully reflect information in accruals and cash flows about future earnings?",
            "journal": "The Accounting Review 71(3), 289-315",
            # no DOI exists for this article; JSTOR's stable address is its permanent reference
            "url": "https://www.jstor.org/stable/248290",
        },
        "claim": ("Earnings made of accruals - working capital building up on the balance sheet rather than "
                  "cash coming in - persist less than earnings backed by cash, and the market does not seem "
                  "to notice. Sorting NYSE and AMEX companies each year into ten groups by accruals, the "
                  "lowest-accrual group beat the highest by 10.4% in the following year, size-adjusted, "
                  "over 1962-1991."),
        "reference": {
            "source": ("Table 6, first column: time-series means of equal-weighted, size-adjusted returns held "
                       "through the year after formation, 40,679 firm-years 1962-1991 (lowest decile +4.9%, "
                       "t 2.65; highest -5.5%, t -3.98; hedge 10.4%, t 4.71). Read from the published "
                       "article; its text layer carries only the two end rows, so the eight middle rows were "
                       "read from the rendered page. Its returns are size-adjusted, so each decile is compared "
                       "against the average of the ten (0.11% a year in the paper), in percent a year."),
            "start": "196307", "end": "199206", "weighting": "ew", "relative": True, "scale": 12,
            # The order these were written in, because it matters: the two ends and the spread, with
            # their tolerances, were written BEFORE any comparison, when only the end rows of the table
            # had been read (its text layer lacked the rest), and were then compared - 1 of 3 within.
            # The full table was then read from the rendered page. The ends' paper values were
            # corrected from "against zero" to "against the paper's own ten-decile average" (0.11%),
            # which moved neither verdict; their 3.0 tolerances were kept as written, neither widened
            # nor narrowed. The eight middle deciles were added with experiment 11's 2.0, written
            # before THEY were compared.
            "tolerances_written": "after_first_comparison",
            "tolerances_note": ("The two end deciles and the spread were written and compared first (1 of 3 within); "
                                "the ends' paper values were then corrected to the paper's own ten-decile average, "
                                "which moved neither verdict, and the eight middle deciles were added, with their "
                                "tolerances, before they were compared."),
            "figures": [
                {"key": "mean10", "label": "Lowest accruals, against the average (% a year)", "paper": 4.79, "tolerance": 3.0, "where": "Table 6, size-adjusted, year t+1, lowest decile (0.049) less the ten-decile average (0.0011)"},
                {"key": "mean9", "label": "Decile 2, against the average (% a year)", "paper": 2.69, "tolerance": 2.0, "where": "Table 6, decile 2 (0.028)"},
                {"key": "mean8", "label": "Decile 3, against the average (% a year)", "paper": 2.29, "tolerance": 2.0, "where": "Table 6, decile 3 (0.024)"},
                {"key": "mean7", "label": "Decile 4, against the average (% a year)", "paper": 1.09, "tolerance": 2.0, "where": "Table 6, decile 4 (0.012)"},
                {"key": "mean6", "label": "Decile 5, against the average (% a year)", "paper": -0.01, "tolerance": 2.0, "where": "Table 6, decile 5 (0.001)"},
                {"key": "mean5", "label": "Decile 6, against the average (% a year)", "paper": 0.89, "tolerance": 2.0, "where": "Table 6, decile 6 (0.010)"},
                {"key": "mean4", "label": "Decile 7, against the average (% a year)", "paper": -0.31, "tolerance": 2.0, "where": "Table 6, decile 7 (-0.002)"},
                {"key": "mean3", "label": "Decile 8, against the average (% a year)", "paper": -2.21, "tolerance": 2.0, "where": "Table 6, decile 8 (-0.021)"},
                {"key": "mean2", "label": "Decile 9, against the average (% a year)", "paper": -3.61, "tolerance": 2.0, "where": "Table 6, decile 9 (-0.035)"},
                {"key": "mean1", "label": "Highest accruals, against the average (% a year)", "paper": -5.61, "tolerance": 3.0, "where": "Table 6, highest decile (-0.055) less the ten-decile average"},
                {"key": "spread", "label": "Lowest accruals minus highest (% a year)", "paper": 10.4, "tolerance": 4.0, "where": "Table 6, size-adjusted, year t+1, hedge portfolio (0.104)"},
            ],
        },
        "verdict_note": ("The direction reproduces - low accruals beat "
                         "high in the paper's years, t 4.7 - but the library's gradient is flatter: the spread "
                         "is half the paper's and the deciles next to the ends miss, as expected beforehand from "
                         "a weaker measure of accruals (no depreciation term, scaled by book equity rather than "
                         "assets). Published as a miss; the tolerances stay as written."),
        "sort": {
            "data": "accruals",
            "groups": ["Highest accruals", "Decile 9", "Decile 8", "Decile 7", "Decile 6", "Decile 5", "Decile 4", "Decile 3", "Decile 2", "Lowest accruals"],
            "ends": ["highest accruals", "lowest"],
            "low": "group with the highest accruals", "high": "group with the lowest",
            "spread": "Lowest accruals minus highest",
            "accent": [9],
            "key_accent": "The lowest accruals - the end the screener's Quality category favours",
            "key_rest": "The other nine groups",
            "chart": "by accruals, highest first",
            "presets": [["The paper \u00b7 1963\u20131992", "196307", "199206"],
                        ["After the paper \u00b7 1992\u2013now", "199207", "LAST"],
                        ["Since 2007", "200701", "LAST"], ["Everything", "FIRST", "LAST"]],
            "weightings": [["ew", "Equally \u00b7 paper"], ["vw", "By market value"]],
        },
        "conditions": [
            ["Portfolios", "Each June, companies are put in one of ten groups by accruals - the change in "
                           "operating working capital per share over the last fiscal year, divided by book "
                           "equity per share - with the dividing lines drawn on New York Stock Exchange stocks."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Accruals\"."],
            ["Weighting", "Equal-weighted, as in the paper. The value-weighted version is one click away."],
            ["Returns", "The comparison with the paper sets each end against the average of the ten, "
                        "because the paper's returns are size-adjusted; the chart shows total returns."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper measures accruals as the change in non-cash current assets less the change in "
            "current liabilities (excluding short-term debt and taxes payable) less depreciation, over "
            "average total assets; the library uses the change in operating working capital over book "
            "equity, with no depreciation term.",
            "The paper holds each group for a year from four months after the fiscal year end and "
            "adjusts for size; the public file reports raw monthly returns of groups formed each June.",
            "The table's middle rows are not in the published PDF's text layer and were read from the "
            "rendered page; the two ends and the spread were compared before they were read (the "
            "tolerances' comment in the source records the order).",
        ],
        "in_the_screener": (
            "Accruals are 5% of the Quality category, scored lower-is-better, as Sloan's result implies; "
            "the screener's measure is net income less operating cash flow over total assets, and it is "
            "compared within each sector."),
    },
    {
        "id": "value-and-momentum",
        "group": "Value",
        "kind": "combo",
        "topic": "Valuation and Momentum",
        "title": "Do value and momentum offset each other?",
        "tab": "Value + momentum",
        "paper": {
            "authors": "Clifford S. Asness, Tobias J. Moskowitz and Lasse Heje Pedersen",
            "year": 2013,
            "title": "Value and momentum everywhere",
            "journal": "Journal of Finance 68(3), 929-985",
            "doi": "10.1111/jofi.12021",
        },
        "claim": ("Value and momentum each earned a premium among US stocks from 1972 to 2011, and they "
                  "moved against each other (a correlation of -0.53 once the market's move is taken out), "
                  "so a 50/50 mix of the two had a Sharpe ratio about twice either one's: 0.63 against "
                  "0.29 for value and 0.33 for momentum."),
        "reference": {
            "source": ("Table I, Panel A, US stocks, January 1972 to July 2011, read from the published "
                       "article: the P3-P1 spreads (top third minus bottom third by each signal, weighted by "
                       "market value, in a universe of large, liquid stocks), annualised, and the "
                       "correlation of value's and momentum's returns after the market's move is removed "
                       "(the table's caption). Compared with the big-company half of the library's 2x3 "
                       "portfolios, the nearest public construction to the paper's universe."),
            "start": "197201", "end": "201107", "group": "big",
            # Written 2026-10-15 before this record's comparison was run - but not before every figure
            # had been seen: research/2026-10-15-value-momentum-correlation.md had measured the
            # big-company Sharpe ratios and residual correlation an hour earlier (0.27, 0.35, 0.49,
            # -0.22). So the tolerances are not chosen here: they follow experiment 10, the lab's
            # other rerun on a different construction of the same idea, at 0.10% a month for an
            # average (1.2% a year), with the volatility and Sharpe ratio scaled from it (2.0% a year;
            # 0.10) and the correlation at 0.15. Read "outside" here as the record of a construction
            # difference, not a verdict a reader should weigh twice.
            "tolerances_written": "after_figures_seen",
            "tolerances_note": ("A research note had measured the big-company Sharpe ratios and the correlation an "
                                "hour before, so these tolerances were not chosen for this comparison but follow "
                                "experiment 10's scale; read an \"outside\" here as the record of a construction "
                                "difference."),
            "figures": [
                {"key": "mean_value", "label": "Value: average a year (%)", "paper": 3.7, "tolerance": 1.2, "where": "Table I A, value P3-P1"},
                {"key": "mean_momentum", "label": "Momentum: average a year (%)", "paper": 5.4, "tolerance": 1.2, "where": "Table I A, momentum P3-P1"},
                {"key": "mean_combo", "label": "50/50 mix: average a year (%)", "paper": 4.6, "tolerance": 1.2, "where": "Table I A, combination P3-P1"},
                {"key": "std_value", "label": "Value: volatility a year (%)", "paper": 12.8, "tolerance": 2.0, "where": "Table I A"},
                {"key": "std_momentum", "label": "Momentum: volatility a year (%)", "paper": 16.4, "tolerance": 2.0, "where": "Table I A"},
                {"key": "std_combo", "label": "50/50 mix: volatility a year (%)", "paper": 7.2, "tolerance": 2.0, "where": "Table I A"},
                {"key": "sharpe_value", "label": "Value: Sharpe ratio", "paper": 0.29, "tolerance": 0.10, "where": "Table I A"},
                {"key": "sharpe_momentum", "label": "Momentum: Sharpe ratio", "paper": 0.33, "tolerance": 0.10, "where": "Table I A"},
                {"key": "sharpe_combo", "label": "50/50 mix: Sharpe ratio", "paper": 0.63, "tolerance": 0.10, "where": "Table I A"},
                {"key": "corr_resid", "label": "Correlation of value with momentum, after the market", "paper": -0.53, "tolerance": 0.15, "where": "Table I A, last row"},
            ],
        },
        "verdict_note": ("Each premium and the mix's average return "
                         "match the paper; the hedge is milder here (-0.22 after the market, against -0.53), so "
                         "the mix is more volatile and its Sharpe ratio lower (0.49 against 0.63). The likeliest "
                         "reason is the library pricing value at the previous December, which the record's "
                         "open details explain."),
        # printed under the comparison: three of the misses are one difference, said once
        "against_note": ("Given each spread's own average and volatility, the mix's volatility and Sharpe ratio "
                         "follow mechanically from how strongly value and momentum move against each other - "
                         "the correlation in the last row - so those two misses restate that one difference "
                         "rather than adding to it."),
        "conditions": [
            ["Portfolios", "Value is high book-to-market minus low; momentum is high past 12-month return "
                           "(skipping the last month) minus low. Each is measured among big companies - the "
                           "half of the market above the NYSE median by size - or, one click away, as the "
                           "average of the big and small halves."],
            ["The mix", "Half value and half momentum each month: the average of the two spreads."],
            ["Correlation", "Measured on what is left of each spread after the part that moves with the "
                            "market, as the paper does, and also on the raw spreads."],
            ["Data", "Kenneth R. French's Data Library: the 2x3 portfolios on size and book-to-market and "
                     "on size and prior return, and the market factor."],
            ["Costs", "None. These are paper portfolios that hold some stocks and short others."],
        ],
        "open_details": [
            "The paper sorts into thirds within its own large, liquid universe and prices value at a "
            "more recent date; the library splits at the 30th and 70th percentiles of NYSE stocks and "
            "prices book-to-market at the previous December. A value measure on a stale price is partly "
            "a short-term momentum signal in reverse, which is the likeliest reason the correlation "
            "here is milder.",
            "The paper's market benchmark for the residual correlation is the MSCI US index; here it is "
            "the library's value-weighted market of all US stocks.",
        ],
        "in_the_screener": (
            "Valuation (22%) and Momentum (13%) are both in the composite, and the screener's value "
            "ratios use the current price - the construction the paper's evidence says opposes momentum "
            "more strongly. When the momentum regime rule cuts Momentum in a market panic, half the "
            "freed weight goes to Valuation; this experiment is the check on that choice."),
    },
    {
        "id": "asset-growth",
        "group": "Investment",
        "kind": "sort",
        "topic": "Investment and Growth",
        "title": "Do companies that grow their assets fastest earn less?",
        "tab": "Asset growth",
        "paper": {
            "authors": "Michael J. Cooper, Huseyin Gulen and Michael J. Schill",
            "year": 2008,
            "title": "Asset growth and the cross-section of stock returns",
            "journal": "Journal of Finance 63(4), 1609-1651",
            "doi": "10.1111/j.1540-6261.2008.01370.x",
        },
        "claim": ("Sorted each June into ten groups by the growth of their total assets over the last fiscal "
                  "year, the slowest-growing tenth of US non-financial companies beat the fastest-growing "
                  "tenth by 1.73% a month equally weighted (t-statistic 8.45) and by 1.05% a month weighted "
                  "by market value (t 5.04), July 1968 to June 2003 - the returns falling step by step from "
                  "the slowest growers to the fastest."),
        "reference": {
            "source": ("The published article, Table II, Panels B.1 (p. 1618, equal-weighted) and B.2 (p. 1619, "
                       "value-weighted): average monthly raw returns of asset-growth deciles in the year after "
                       "formation (the 'YEAR 1' row), July 1968 to June 2003, with the 10-1 spread and its "
                       "t-statistic; the text on p. 1622 restates the two ends and both spreads. Read from the "
                       "text layer of a copy of the publisher's PDF. The paper's spread is high growth minus "
                       "low (-1.73%); here the groups are listed fastest growth first, so the spread is the "
                       "slowest minus the fastest and carries the opposite sign."),
            "start": "196807", "end": "200306", "weighting": "ew", "also": "vw",
            # Written 2026-10-19 BEFORE the comparison was run: the library's investment file had been
            # downloaded and only its header and first rows (1963-64, before the paper's sample) looked
            # at to find the columns; nothing had been computed on it. The library sorts on the paper's
            # own variable (total assets t-1 over t-2, less one, formed each June), so this is not a
            # different-measure rerun - but three things differ: the library keeps financial companies
            # and utilities (the paper drops SIC 6000-6999), draws its decile lines on NYSE stocks only
            # (the paper's deciles split all its firms, NASDAQ included), and does not require two years
            # on Compustat. NYSE lines put more small, extreme-growth NASDAQ companies in the end groups
            # and fewer in the middle; expected beforehand: the shape matches, the equal-weighted ends
            # and spread land below the paper's, the value-weighted ones closer.
            # Group averages: 0.10% a month, experiment 10's scale for a construction difference (the
            # same-file reruns of experiments 4-7 used 0.05). Spreads: 0.20% a month, about one
            # standard error of the paper's own estimate (1.73 / 8.45 = 0.205; 1.05 / 5.04 = 0.208).
            # t-statistics: 1.5 - the spread's tolerance is one unit of t, plus half a unit for a
            # different volatility from a different set of companies.
            # Plan rule 8, checked from the record before committing: a result of no effect matches 6
            # of these 24 (three middle deciles in each weighting) and misses both spreads, both
            # t-statistics and every end group.
            "tolerances_written": "before",
            "figures": [
                {"key": "mean1", "label": "Fastest asset growth (decile 10): average month, equally weighted (%)", "paper": 0.26, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1, decile 10 (p. 1618)"},
                {"key": "mean2", "label": "Decile 9: average month, equally weighted (%)", "paper": 0.85, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean3", "label": "Decile 8: average month, equally weighted (%)", "paper": 1.14, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean4", "label": "Decile 7: average month, equally weighted (%)", "paper": 1.18, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean5", "label": "Decile 6: average month, equally weighted (%)", "paper": 1.24, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean6", "label": "Decile 5: average month, equally weighted (%)", "paper": 1.34, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean7", "label": "Decile 4: average month, equally weighted (%)", "paper": 1.41, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean8", "label": "Decile 3: average month, equally weighted (%)", "paper": 1.54, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean9", "label": "Decile 2: average month, equally weighted (%)", "paper": 1.76, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1 (p. 1618)"},
                {"key": "mean10", "label": "Slowest asset growth (decile 1): average month, equally weighted (%)", "paper": 1.99, "tolerance": 0.10, "where": "Table II, Panel B.1, year 1, decile 1 (p. 1618; text p. 1622)"},
                {"key": "spread", "label": "Slowest growth minus fastest, equally weighted, a month (%)", "paper": 1.73, "tolerance": 0.20, "where": "Table II, Panel B.1, year 1, spread (10-1) = -0.0173, sign reversed (p. 1618; text p. 1622)"},
                {"key": "t_spread", "label": "Its t-statistic", "paper": 8.45, "tolerance": 1.5, "where": "Table II, Panel B.1, year 1, t(spread) = -8.45, sign reversed (p. 1618)"},
                {"key": "vw_mean1", "label": "Fastest asset growth (decile 10): average month, by market value (%)", "paper": 0.43, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1, decile 10 (p. 1619; text p. 1622)"},
                {"key": "vw_mean2", "label": "Decile 9: average month, by market value (%)", "paper": 0.77, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean3", "label": "Decile 8: average month, by market value (%)", "paper": 0.92, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean4", "label": "Decile 7: average month, by market value (%)", "paper": 1.02, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean5", "label": "Decile 6: average month, by market value (%)", "paper": 1.00, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean6", "label": "Decile 5: average month, by market value (%)", "paper": 1.00, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean7", "label": "Decile 4: average month, by market value (%)", "paper": 1.16, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean8", "label": "Decile 3: average month, by market value (%)", "paper": 1.22, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean9", "label": "Decile 2: average month, by market value (%)", "paper": 1.24, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1 (p. 1619)"},
                {"key": "vw_mean10", "label": "Slowest asset growth (decile 1): average month, by market value (%)", "paper": 1.48, "tolerance": 0.10, "where": "Table II, Panel B.2, year 1, decile 1 (p. 1619; text p. 1622)"},
                {"key": "vw_spread", "label": "Slowest growth minus fastest, by market value, a month (%)", "paper": 1.05, "tolerance": 0.20, "where": "Table II, Panel B.2, year 1, spread (10-1) = -0.0105, sign reversed (p. 1619; text p. 1622)"},
                {"key": "vw_t_spread", "label": "Its t-statistic", "paper": 5.04, "tolerance": 1.5, "where": "Table II, Panel B.2, year 1, t(spread) = -5.04, sign reversed (p. 1619)"},
            ],
        },
        "verdict_note": ("The shape reproduces in both weightings - returns fall from the slowest growers to "
                         "the fastest, and both spreads are well clear of zero (t 8.0 and 4.1) - but the "
                         "public sort's ends are closer together than the paper's, so both spreads land "
                         "below it and outside their tolerances: 1.22% against 1.73% equally weighted, 0.67% "
                         "against 1.05% by market value. The equal-weighted shortfall was expected beforehand "
                         "(financial companies kept, decile lines drawn on NYSE stocks); the value-weighted "
                         "one was not. The middle groups match closely."),
        "sort": {
            "data": "asset-growth",
            "groups": ["Fastest asset growth", "Decile 9", "Decile 8", "Decile 7", "Decile 6", "Decile 5", "Decile 4", "Decile 3", "Decile 2", "Slowest asset growth"],
            "ends": ["fastest asset growth", "slowest"],
            "low": "group with the fastest asset growth", "high": "group with the slowest",
            "spread": "Slowest asset growth minus fastest",
            "accent": [9],
            "key_accent": "The slowest asset growth - the end the screener's Investment category favours",
            "key_rest": "The other nine groups",
            "chart": "by asset growth, fastest first",
            "presets": [["The paper · 1968–2003", "196807", "200306"],
                        ["After the paper · 2003–now", "200307", "LAST"],
                        ["Since 2014", "201401", "LAST"], ["Everything", "FIRST", "LAST"]],
            "weightings": [["ew", "Equally · paper"], ["vw", "By market value · paper"]],
        },
        "conditions": [
            ["Portfolios", "Each June, companies are put in one of ten groups by the growth of their total "
                           "assets over the last fiscal year (assets at the fiscal year ending in the previous "
                           "calendar year, divided by assets a year before that, less one) - the paper's own "
                           "measure - with the dividing lines drawn on New York Stock Exchange stocks."],
            ["Data", "Kenneth R. French's Data Library, \"Portfolios Formed on Investment\"."],
            ["Weighting", "The paper reports both, and both are compared: equally weighted (its first panel) "
                          "and by market value. The chart opens on equal weighting; the other is one click away."],
            ["Returns", "Raw monthly returns, as in the paper's table, held from July to the next June."],
            ["Costs", "None. These are paper portfolios."],
        ],
        "open_details": [
            "The paper leaves out financial companies (SIC codes 6000-6999); the public file keeps them, "
            "and utilities.",
            "The paper's deciles divide all its companies; the public file's dividing lines are drawn on "
            "New York Stock Exchange stocks only, so its end groups hold more small, fast- or slow-growing "
            "NASDAQ companies than the paper's.",
            "The paper's stronger claims are after adjusting for size and value (Fama-French alphas, Table "
            "II Panel C) and among large companies alone (a value-weighted alpha spread of 0.81% a month, "
            "t 2.91, in its largest size third). Neither is reproduced here: the public file has no "
            "size-by-asset-growth deciles.",
        ],
        "in_the_screener": (
            "Asset growth is the whole of the screener's Investment category (5% of the composite), scored "
            "lower-is-better, as this paper implies. Growth (13%) pulls the other way: it scores revenue "
            "growth higher-is-better, and across the screener's companies revenue growth and asset growth "
            "have a rank correlation of +0.51, so for many companies the two categories score one fact in "
            "opposite directions. This sort is across the whole market and does not compare within "
            "sectors, as the screener does."),
    },
]


# Experiment 15 (2026-10-20): Jensen, Kelly & Pedersen (2023) on their own public library. Kept as
# its own block and appended to EXPERIMENTS below it, so another experiment added the same night
# does not collide with it line for line.
JKP_REPLICATION = {
    "id": "replication-crisis",
    "group": "Across the factors",
    "kind": "replication",
    "topic": "Every category at once",
    "title": "How many published factors hold up - and among the largest companies?",
    "tab": "153 factors",
    "paper": {
        "authors": "Theis Ingerslev Jensen, Bryan Kelly and Lasse Heje Pedersen",
        "year": 2023,
        "title": "Is there a replication crisis in finance?",
        "journal": "Journal of Finance 78(5), 2465-2518",
        "doi": "10.1111/jofi.13249",
    },
    "claim": ("Of 153 US factors from the published literature, built one uniform way, most hold up: of the "
              "119 whose own papers found them significant, 82.4% have a CAPM alpha distinguishable from "
              "zero over the full US sample to 2020, and the rate is nearly as high among the largest "
              "companies (77.3% for mega caps, 79.8% for large caps). After their papers' samples ended "
              "the factors earned about half as much (an average alpha of 0.26% a month against 0.49%), "
              "but 83% of them stayed positive."),
    "reference": {
        "source": ("The published article (Journal of Finance 78(5), 2023), read from the copy its authors "
                   "keep in their public code repository: Figure 1 and its text (pp. 2467-2468), the "
                   "internal-validity section (pp. 2489-2491, the size groups on p. 2491) and the "
                   "out-of-sample section (pp. 2494-2495, with Figure 8's caption on p. 2496). The "
                   "'replication rate' is the share of factors whose t-statistic clears 1.96 in the "
                   "direction the original paper found. Which 119 factors were significant in their "
                   "original papers, and each paper's own sample, are read from the authors' "
                   "factor-details file, which marks 119 significant and 34 not, as the paper counts."),
        "start": "FIRST", "end": JKP_END, "universe": "all", "test": "capm", "set": "sig",
        # Written 2026-10-20 BEFORE any comparison was run: the library's files had been downloaded
        # and parsed for their shape (columns, months, gaps, the 119/34 split of the factor-details
        # file) and nothing had been computed on a return. Four differences are known beforehand:
        #  * Vintage. The paper used the library as built in 2021 (SAS); today's files are the 2025
        #    Python rebuild, extended to 2025, whose release notes compare the two closely but not
        #    exactly. Months to 2020 only are compared.
        #  * Standard errors. The paper's are the diagonal of a bootstrapped covariance of alphas
        #    with a block structure across factors (Appendix B); here, each factor's own OLS standard
        #    error. Close for one factor; enough to move a factor sitting at t = 1.96 either side.
        #  * The size groups. The paper's mega- and large-cap rates are empirical-Bayes posteriors
        #    (Figure 5); here they are plain OLS counts. In the paper's own US numbers the two
        #    coincided (82.4% each, p. 2490), but shrinkage can move a group either way. And the
        #    public size-group files weight stocks by their rank on the characteristic, which the
        #    paper's text does not describe for Figure 5.
        #  * The out-of-sample windows. Which factors the two averages (0.49%, 0.26%) are taken over
        #    is not stated; here, Figure 8 Panel B's set (significant originally, five years or more
        #    after the original sample). Alphas are re-estimated inside each window and scaled, as
        #    the paper scales every factor, to 10% a year of idiosyncratic volatility measured over
        #    the whole sample to 2020.
        # Tolerances: one factor is 0.84 points of 119 and 0.65 of 153. Rates on all stocks, 5.0
        # points (about six factors crossing 1.96 either way); the size groups, 8.0, for the
        # different estimator on top; the shares positive, 5.0. The two average alphas, 0.06% a
        # month - about 12% of the in-sample figure and roughly two standard errors of an average
        # of a hundred-odd correlated factors at 10% volatility.
        # Plan rule 8, from the record: a result of no effect - every factor earning nothing - leaves
        # a replication rate at the 2.5% that clears 1.96 by chance, half the factors positive after
        # their samples, and both averages at zero. It matches none of the ten.
        "tolerances_written": "before",
        "figures": [
            {"key": "rate_raw_all", "label": "All 153 factors: share with a significant average return (%)", "paper": 55.6, "tolerance": 5.0, "where": "Figure 1, second bar; text p. 2467"},
            {"key": "rate_raw_sig", "label": "The 119 significant in their papers: share with a significant average return (%)", "paper": 61.3, "tolerance": 5.0, "where": "Figure 1, third bar; text p. 2467"},
            {"key": "rate_capm", "label": "The 119: share with a significant CAPM alpha (%)", "paper": 82.4, "tolerance": 5.0, "where": "Figure 1, fourth bar (p. 2468); 98 of 119, p. 2489"},
            {"key": "rate_capm_mega", "label": "Mega caps only: share of the 119 with a significant CAPM alpha (%)", "paper": 77.3, "tolerance": 8.0, "where": "Figure 5 Panel A; text p. 2491 (empirical Bayes)"},
            {"key": "rate_capm_large", "label": "Large caps only: share of the 119 with a significant CAPM alpha (%)", "paper": 79.8, "tolerance": 8.0, "where": "Figure 5 Panel A; text p. 2491 (empirical Bayes)"},
            {"key": "alpha_in", "label": "Average alpha inside the original papers' samples, a month (%, at 10% volatility)", "paper": 0.49, "tolerance": 0.06, "where": "Section III.B.2, p. 2494"},
            {"key": "alpha_post", "label": "Average alpha after the original samples ended, a month (%, at 10% volatility)", "paper": 0.26, "tolerance": 0.06, "where": "Section III.B.2, p. 2494"},
            {"key": "pos_pre", "label": "Share with a positive alpha before the original sample (%)", "paper": 82.6, "tolerance": 5.0, "where": "p. 2495; Figure 8 Panel A, 102 factors (p. 2496)"},
            {"key": "pos_post", "label": "Share with a positive alpha after the original sample (%)", "paper": 83.3, "tolerance": 5.0, "where": "p. 2495; Figure 8 Panel B, 115 factors (p. 2496)"},
            {"key": "pos_both", "label": "Share with a positive alpha before and after, together (%)", "paper": 87.4, "tolerance": 5.0, "where": "p. 2495; Figure 8 Panel C, 119 factors (p. 2496)"},
        ],
    },
    "verdict_note": "",
    "replication": {
        "presets": [["The paper · to 2020", "FIRST", JKP_END], ["After the paper · 2021–now", "202101", "LAST"],
                    ["Since 2000", "200001", "LAST"], ["Since 2014", "201401", "LAST"], ["Everything", "FIRST", "LAST"]],
        "universes": [["all", "All stocks · paper"], ["mega", "Mega caps"], ["large", "Large caps"]],
        "tests": [["capm", "CAPM alpha · paper"], ["raw", "Average return"]],
        "sets": [["sig", "The 119 · paper"], ["all", "All 153"]],
        # the screener category each of the paper's themes stands behind, where there is one
        "theme_category": {"Value": "Valuation", "Profitability": "Quality", "Quality": "Quality",
                           "Accruals": "Quality", "Momentum": "Momentum", "Investment": "Investment",
                           "Low Risk": "Risk", "Size": "Size"},
        "universe_names": {"all": "all US stocks", "mega": "mega caps (larger than 80% of NYSE companies)",
                           "large": "large caps (between the NYSE median and its 80th percentile)"},
    },
    "conditions": [
        ["Factors", "The 153 characteristics of the paper, each built the library's one way: every month, stocks "
                    "above the NYSE 20th percentile by size are split into thirds by the characteristic (smaller "
                    "stocks are placed by the same lines), each third is weighted by market value capped at the "
                    "NYSE 80th percentile, and the factor is the third the original paper found paid most minus "
                    "the third it found paid least."],
        ["Among the largest companies", "The library's size-group files: the same characteristics inside mega "
                                        "caps (above the NYSE 80th percentile) and inside large caps (50th to "
                                        "80th), each stock weighted by its rank on the characteristic. The "
                                        "files are not signed; each is turned the way its paper found it paid."],
        ["The test", "A factor 'replicates' when its t-statistic is above 1.96 in that direction - on its CAPM "
                     "alpha (its return after the part that moves with the market) as the paper's headline does, "
                     "or on its plain average return. A factor counts once it has 60 months in the period, the "
                     "paper's own minimum."],
        ["Before and after", "Each factor's original paper sample comes from the authors' factor-details file. "
                             "The alphas before and after it are re-estimated in each window, with each factor "
                             "scaled to 10% a year of idiosyncratic volatility over its whole record to 2020, as "
                             "the paper scales them."],
        ["Data", "Jensen, Kelly and Pedersen's Global Factor Data (jkpfactors.com), United States, monthly, "
                 "and its value-weighted US market return. Licensed CC BY-NC 4.0 by its authors: the series "
                 "here are derived from it (rounded to a basis point and re-signed), for non-commercial use."],
        ["Costs", "None. These are paper portfolios that hold some stocks and short others."],
    ],
    "open_details": [
        "The library has been rebuilt since the paper (2021 SAS code; 2025 Python code) and now runs to 2025. "
        "Only months to December 2020 are compared, but the rebuild itself can move a factor across the line.",
        "The paper's standard errors come from a bootstrap of all factors together (Appendix B); here each "
        "factor's own OLS standard error is used.",
        "The paper's size-group rates are empirical-Bayes posteriors, which shrink every alpha toward zero "
        "and then borrow strength across factors; here they are plain OLS counts. The paper's text does not "
        "say how its size-group factors are weighted; the public files weight by rank.",
        "The paper does not say which factors its two average alphas (0.49%, 0.26%) are taken over; here, "
        "the 115 that were significant originally and have five years after their sample, the set of its "
        "Figure 8 Panel B.",
        "Not reproduced: the multiple-testing rate (75.6%, which needs the paper's bootstrap critical value), "
        "the empirical-Bayes posteriors themselves, the regression slopes of Figure 8 (estimated by GLS on "
        "the bootstrapped covariance) and the rates for small, micro and nano caps.",
    ],
    "in_the_screener": (
        "Six of the screener's categories lean on themes this paper tests: Valuation (value), Quality "
        "(profitability and quality), Momentum, Investment, Risk (low risk) and Size. This experiment is the "
        "widest check the lab has on whether that literature holds up at all, and the only one that repeats "
        "every test inside the mega caps, which is where the screener's companies sit. It tests long-short "
        "factors one characteristic at a time across the whole market; the screener combines them, within "
        "sectors, and holds only the top of its ranking."),
}
EXPERIMENTS.append(JKP_REPLICATION)

# Queue row 4b (plan/factor-lab.md), set by research/2026-10-20-size-weight.md section 9: the one
# published case for the Size weight is conditional on quality, and nobody has published it among
# large companies alone. Kept self-contained (its own kind, "double") so it can be merged beside
# experiments being added in parallel.
SIZE_GIVEN_PROFITABILITY = {
    "id": "size-given-profitability",
    "group": "Across the factors",
    "kind": "double",
    "topic": "Size",
    "title": "Does size pay once profitability is held fixed?",
    "tab": "Size, given profitability",
    "paper": {
        "authors": "Clifford Asness, Andrea Frazzini, Ronen Israel, Tobias J. Moskowitz and Lasse H. Pedersen",
        "year": 2018,
        "title": "Size matters, if you control your junk",
        "journal": "Journal of Financial Economics 129(3), 479-509",
        "doi": "10.1016/j.jfineco.2018.05.006",
    },
    "claim": ("Once low-quality (\"junk\") companies are held apart, smaller companies earned more in every "
              "group but the junkiest. Sorting US stocks into fifths by size and, separately, by quality, "
              "from 1957 to 2012 the smallest fifth beat the largest by 0.42% to 0.54% a month in the four "
              "higher-quality groups, and by 0.23% among the junk. The question this page adds, because the "
              "screener ranks only large companies: does the fourth size group still beat the largest once "
              "profitability is held fixed?"),
    "reference": {
        "source": ("The published article (open access), Table 3, Panel A (page 495), and its text on the same "
                   "page: 25 value-weighted portfolios from independent sorts on size and on quality, average "
                   "monthly returns over T-bills and their t-statistics, July 1957 to December 2012. The fourth "
                   "size group minus the biggest is derived from the table's two columns; the paper gives it no "
                   "t-statistic."),
        "start": "196307", "end": "201212",
        # Written 2026-10-20 BEFORE the comparison was run, from the paper alone, and committed on
        # their own: nothing had been computed on the library's 25 size-by-profitability file, which
        # was fetched for the first time that night (only its table titles were read, to find the
        # value-weighted table). The note that queued this experiment had measured related figures on
        # other files - SMB inside the 2x3 profitability terciles, 0.228 (t 1.75) over 1963-2026 -
        # which is why these tolerances are written for a stand-in, not for a rerun.
        #
        # Why they are wide. The paper sorts on its quality score (QMJ: profitability, growth, safety
        # and payout, re-ranked every month); the library sorts once a year on operating profitability
        # alone. Its sample starts in 1957; the library's profitability files start in July 1963, so
        # the window is 594 months against 666 and every t-statistic is expected about 6% lower for
        # that alone. So 0.15% a month on an average or a spread (one paper standard error is about
        # 0.15-0.19 on these rows), and 1.0 on a t-statistic. Expected: the four upper rows and the
        # averages mostly within, the weakest-profitability row the likeliest miss (operating
        # profitability's lowest fifth is not the paper's junk), and the quality spreads among the
        # smallest the other likely one. By plan rule 8 a result of no effect - every spread and
        # t-statistic zero - matches none of the seventeen (the smallest paper figure is 0.23, the
        # smallest t-statistic 1.21).
        "tolerances_written": "before",
        "figures": [
            {"key": "sb_1", "label": "Weakest profitability fifth: smallest size group minus largest, a month (%)", "paper": 0.23, "tolerance": 0.15, "where": "Table 3A, Junk row"},
            {"key": "t_sb_1", "label": "Weakest fifth: t-statistic", "paper": 1.21, "tolerance": 1.0, "where": "Table 3A, Junk row"},
            {"key": "sb_2", "label": "Second fifth: smallest minus largest, a month (%)", "paper": 0.46, "tolerance": 0.15, "where": "Table 3A, row 2"},
            {"key": "t_sb_2", "label": "Second fifth: t-statistic", "paper": 3.02, "tolerance": 1.0, "where": "Table 3A, row 2"},
            {"key": "sb_3", "label": "Middle fifth: smallest minus largest, a month (%)", "paper": 0.54, "tolerance": 0.15, "where": "Table 3A, row 3"},
            {"key": "t_sb_3", "label": "Middle fifth: t-statistic", "paper": 3.44, "tolerance": 1.0, "where": "Table 3A, row 3"},
            {"key": "sb_4", "label": "Fourth fifth: smallest minus largest, a month (%)", "paper": 0.42, "tolerance": 0.15, "where": "Table 3A, row 4"},
            {"key": "t_sb_4", "label": "Fourth fifth: t-statistic", "paper": 2.82, "tolerance": 1.0, "where": "Table 3A, row 4"},
            {"key": "sb_5", "label": "Most profitable fifth: smallest minus largest, a month (%)", "paper": 0.44, "tolerance": 0.15, "where": "Table 3A, Quality row"},
            {"key": "t_sb_5", "label": "Most profitable fifth: t-statistic", "paper": 2.87, "tolerance": 1.0, "where": "Table 3A, Quality row"},
            {"key": "smbq", "label": "Smallest minus largest, averaged across the five (SMBQ), a month (%)", "paper": 0.42, "tolerance": 0.15, "where": "p. 495, text"},
            {"key": "t_smbq", "label": "That average: t-statistic", "paper": 3.18, "tolerance": 1.0, "where": "p. 495, text"},
            {"key": "fourq", "label": "Fourth size group minus largest, averaged across the five, a month (%)", "paper": 0.254, "tolerance": 0.15, "where": "Table 3A, derived: columns 4 and Big"},
            {"key": "qj_big", "label": "Largest size group: most profitable fifth minus least, a month (%)", "paper": 0.42, "tolerance": 0.15, "where": "Table 3A, Big column"},
            {"key": "t_qj_big", "label": "Largest size group: t-statistic", "paper": 2.78, "tolerance": 1.0, "where": "Table 3A, Big column"},
            {"key": "qj_small", "label": "Smallest size group: most profitable fifth minus least, a month (%)", "paper": 0.62, "tolerance": 0.15, "where": "Table 3A, Small column"},
            {"key": "t_qj_small", "label": "Smallest size group: t-statistic", "paper": 4.78, "tolerance": 1.0, "where": "Table 3A, Small column"},
        ],
    },
    # The screener's own test, separate from the comparison with the paper: written into
    # research/2026-10-20-size-weight.md (section 9, item 1) and committed before the library's file
    # was fetched. Its words are carried verbatim; its numbers are the ones it names. The page
    # prints the rule, the figure and the reading, whichever way it falls.
    "criterion": {
        "source": "research/2026-10-20-size-weight.md, section 9, item 1",
        "rule": ("Refutes the case for the weight if, over 1963-2026, the average of the five is no larger "
                 "than the unconditional Q4 - Q5 spread (0.136% a month, value-weighted, section 2.1), or "
                 "its t-statistic is below 1.5 - i.e. if holding profitability fixed does not strengthen "
                 "size among large companies the way it does across the market."),
        "measure": ("Size quintile 4 minus quintile 5 inside each profitability quintile, value-weighted, "
                    "averaged across the five, month by month."),
        "start": "196307", "end": "202612",
        "unconditional": 0.136, "min_t": 1.5,
        "unconditional_source": ("The library's size quintile 4 minus quintile 5, value-weighted, July 1963 "
                                 "to August 2026 (research/measurements/2026-10-20-size-lab.json)."),
        "weight": "Size weight (5% of the score)",
    },
    "double": {
        "rows": ["Weakest profitability", "Second fifth", "Middle fifth", "Fourth fifth", "Most profitable"],
        "cols": ["Smallest", "2", "3", "4", "Largest"],
        "presets": [["The paper's years · 1963–2012", "196307", "201212"],
                    ["After the paper · 2013–now", "201301", "LAST"],
                    ["Since 1981", "198101", "LAST"], ["Since 2000", "200001", "LAST"],
                    ["Everything", "FIRST", "LAST"]],
    },
    "verdict_note": ("The averages across the five profitability fifths match, including the fourth size group's. "
                     "The misses are two of the five rows and the gaps between profitable and weak companies, "
                     "which here are about half the paper's: operating profitability is one part of the paper's "
                     "quality score, not the whole of it."),
    "conditions": [
        ["Portfolios", "Each June, every US-listed stock is put in one of five size groups and, separately, one "
                       "of five groups by operating profitability, both on New York Stock Exchange breakpoints. "
                       "The 25 portfolios are the overlaps."],
        ["Quality", "Operating profitability (sales minus the cost of goods, selling and administrative costs "
                    "and interest, over book equity) stands in for the paper's quality score, which also counts "
                    "growth, safety and payout and is not published in this form."],
        ["Data", "Kenneth R. French's Data Library, \"25 Portfolios Formed on Size and Operating "
                 "Profitability\". It includes financial companies and utilities."],
        ["Sample", "From July 1963, the first month the library sorts on profitability. The paper starts in "
                   "July 1957, so its window is six years longer."],
        ["Weighting", "Value-weighted within each portfolio, as in the paper."],
        ["Large stocks", "The fourth and fifth size groups. In August 2026 the fourth averaged about $15 billion "
                         "and the fifth about $190 billion; the screener's roughly 500 companies sit across "
                         "both."],
        ["Costs", "None. These are paper portfolios."],
    ],
    "open_details": [
        "Operating profitability is one of the paper's four parts of quality. Its quality score is built from "
        "company accounts and is not public in a size-by-quality grid, so the paper's own sort cannot be "
        "rebuilt here.",
        "The paper does not state its size breakpoints; the library's are New York Stock Exchange quintiles. "
        "If the paper used all stocks, its largest fifth is a larger group than the library's.",
        "The paper re-ranks every month; the library re-forms its groups once a year, in June.",
        "The paper reports returns over T-bills. Every figure compared here is a difference between two "
        "portfolios, which the T-bill does not change, so none needs it.",
        "The paper's regressions on its quality factor, its main evidence, are not rerun.",
    ],
    "in_the_screener": (
        "The screener's Size category scores a company higher for being smaller, among the large companies "
        "it ranks, beside a Quality category that carries 22% of the score. The research note behind the "
        "Size weight found no size premium among large companies on its own and kept the weight on this "
        "paper's case - that size pays once quality is controlled - and wrote down, before this file was "
        "fetched, the result that would refute that case among large companies. That rule, and what the "
        "data says, are printed beside the comparison. A refutation reopens the weight for a research note "
        "to decide; this page does not set it."),
}
EXPERIMENTS.append(SIZE_GIVEN_PROFITABILITY)


# Experiment 17 (2026-10-20): appended after experiments 15 and 16 so their published numbers hold.
PIOTROSKI_OSAP = {
    "id": "piotroski",
    "group": "Quality",
    "kind": "sort",
    "topic": "Quality",
    "title": "Among high book-to-market companies, do the financially strong earn more?",
    "tab": "Piotroski score",
    "paper": {
        "authors": "Joseph D. Piotroski",
        "year": 2000,
        "title": ("Value investing: the use of historical financial statement information to separate "
                  "winners from losers"),
        "journal": "Journal of Accounting Research 38 (Supplement), 1-41",
        "doi": "10.2307/2672906",
    },
    "claim": ("Among the fifth of companies with the highest book-to-market ratios, a score counting nine "
              "signs of financial health - profits, cash flow, improving returns on assets, cash flow "
              "above profits, falling leverage, rising liquidity, no new shares, rising margins and "
              "asset turnover - separated later winners from losers: companies scoring 8 or 9 returned "
              "31.3% in the following year and those scoring 0 or 1 returned 7.8%, a difference of 23.5% "
              "(t-statistic 5.59), 1976-1996."),
    "reference": {
        "source": ("Table 3, Panel B (one-year raw returns, each firm held for twelve months), the High F_SCORE (8-9), Low "
                   "F_SCORE (0-1) and High-Low rows, 14,043 high book-to-market firm-years, read from the "
                   "University of Chicago GSB Selected Paper 84 reprint (January 2002) of the published "
                   "article, printed page 17 (PDF page 19). The paper's figures are means of equally "
                   "weighted firm returns over a year; here they are compared with the average month of "
                   "the authors' end deciles times twelve, in percent a year."),
        "start": "197601", "end": "199612", "weighting": "ew", "scale": 12,
        # Written 2026-10-20 BEFORE any Open Source Asset Pricing return had been read: at the time of
        # writing Google Drive was refusing every file in the release with a "Quota exceeded" page, so
        # nothing - not even a header - had been seen. The period is the years the paper names and the
        # window the OSAP documentation uses as the paper's sample (1976-1996).
        # The comparison is NOT like for like, in four ways known beforehand, all recorded on the page:
        # (1) the paper's groups are scores 0-1 (396 firm-years, 2.8%) and 8-9 (1,448, 10.3%); OSAP's are
        #     deciles cut at the 10th and 90th percentiles of a 0-9 score, and with ties each end holds
        #     a wider band of scores - expected to narrow the spread, lifting the low end most;
        # (2) the paper holds each firm a year from the fifth month after its fiscal year end; OSAP
        #     re-forms monthly on the latest annual score - a twelve-times monthly average is not a
        #     buy-and-hold year, and compounding puts the paper's levels a point or two higher;
        # (3) OSAP's score uses EBIT over sales where the paper uses gross margin, and a fall in shares
        #     outstanding where it uses "no equity issued";
        # (4) CRSP and Compustat have been revised in the 25 years since.
        # Expected beforehand: the direction reproduces; the spread lands below 23.5 and may well be
        # outside; the high end below 31.3. Tolerance for every percent-a-year figure: 4.2, one standard
        # error of the paper's own spread (23.5 / 5.594 = 4.20); for the t-statistic 1.5, experiment 14's
        # (one unit of t plus half for a different volatility). The paper gives no standard error for
        # either end group, so the ends take the spread's.
        # Plan rule 8, checked from the record before committing: a result of no effect matches 0 of
        # these 4 - each end sits 11.75 from the null's common level (19.55), and the spread and t miss.
        "tolerances_written": "before",
        "figures": [
            {"key": "mean1", "label": "Lowest score decile: average a year (%)", "paper": 7.8, "tolerance": 4.2, "where": "Table 3, Panel B, Low F_Score (0-1), mean one-year raw return 0.078 (p. 17)"},
            {"key": "mean2", "label": "Highest score decile: average a year (%)", "paper": 31.3, "tolerance": 4.2, "where": "Table 3, Panel B, High F_Score (8-9), mean one-year raw return 0.313 (p. 17)"},
            {"key": "spread", "label": "Highest score minus lowest, a year (%)", "paper": 23.5, "tolerance": 4.2, "where": "Table 3, Panel B, High-Low, mean 0.235 (p. 17); the abstract's \"23% annual return\""},
            {"key": "t_spread", "label": "Its t-statistic", "paper": 5.594, "tolerance": 1.5, "where": "Table 3, Panel B, High-Low, t-statistic 5.594 (p. 17)"},
        ],
    },
    "verdict_note": ("The highest-scoring end matches the paper (30.4% a year against 31.3%); the lowest "
                     "does not (26.1% against 7.8%), so the gap is a fifth of the paper's - 4.3% a year, "
                     "t 1.5 - and outside. Expected in direction, not in size: the public portfolios' "
                     "lowest decile averaged a score of 2.7 over these years and its highest 7.5, where the "
                     "paper's groups are 0-1 and 8-9, and they are re-formed monthly rather than held a "
                     "year. Weighted by market value, the authors' own cut, the gap was 12.6% a year, "
                     "t 3.6. Published as a miss; the tolerances stay as written."),
    "sort": {
        "data": "piotroski",
        "groups": ["Lowest score decile", "Highest score decile"],
        "ends": ["lowest score", "highest"],
        "low": "decile with the lowest scores", "high": "decile with the highest",
        "spread": "Highest score minus lowest",
        "accent": [1],
        "key_accent": "The highest scores - the end the screener's Quality category favours",
        "key_rest": "The lowest scores",
        "chart": "by Piotroski score, lowest first",
        "presets": [["The paper · 1976–1996", "197601", "199612"],
                    ["After the paper · 1997–2024", "199701", "LAST"],
                    ["Since 2007", "200701", "LAST"], ["Everything · 1972–2024", "FIRST", "LAST"]],
        "weightings": [["ew", "Equally · paper"], ["vw", "By market value · OSAP's cut"]],
    },
    "conditions": [
        ["Portfolios", "Each month, companies in the highest fifth by book-to-market are scored 0 to 9 on "
                       "Piotroski's nine signals from their latest annual accounts, and the lowest and highest "
                       "tenths by score are held for the month. Only the two end groups are read: with a "
                       "score that takes ten values the middle deciles are often empty."],
        ["Data", "Chen and Zimmermann's Open Source Asset Pricing, " + OSAP_RELEASE + ": the published "
                 "portfolio returns for their signal \"PS\", which the authors built from CRSP and "
                 "Compustat. Cite: Chen, A. Y. and T. Zimmermann (2022), \"Open source cross-sectional "
                 "asset pricing\", Critical Finance Review 11(2), 207-264."],
        ["What this checks", "A rerun of the authors' published portfolio returns, not of the score: this "
                             "lab cannot see a single company's accounts. It checks that a careful public "
                             "construction of the signal earns what the paper reports, and what it has "
                             "earned since; it does not check the paper's own firm-by-firm numbers."],
        ["Weighting", "Equally weighted, as the paper's averages of firm returns are, from the authors' "
                      "equal-weighted deciles. Their own cut weights by market value; it is one click away."],
        ["Returns", "Raw monthly returns. The comparison sets twelve times the average month against the "
                    "paper's average one-year return."],
        ["Costs", "None. These are paper portfolios."],
    ],
    "open_details": [
        "The paper's end groups are scores 0-1 and 8-9; the public portfolios are deciles of the score, "
        "and because the score takes only ten values, each end decile holds a wider band of scores.",
        "The paper holds each company for a year from the fifth month after its fiscal year end; the "
        "public portfolios are re-formed every month on the latest annual score.",
        "The public construction measures margin as EBIT over sales (the paper: gross margin) and equity "
        "issuance as a rise in shares outstanding (the paper: any common equity issued), and requires "
        "every input to be present.",
        "The paper finds the effect concentrated in small and medium-sized companies (Table 4: among the "
        "largest third, high minus low 15.2%, t 1.88). The public files have no size split for this "
        "signal, so that is not reproduced here; the market-value weighting is the nearest view of the "
        "large companies.",
        "The paper's period: its text says firms are identified each year 1976-1996, and Table 1 counts "
        "firm-years 1975-1995; returns run from the fifth month after each fiscal year end. The window "
        "here, January 1976 to December 1996, is the one the public library's documentation uses.",
    ],
    "in_the_screener": (
        "The Piotroski score is 16% of the Quality category (Quality is 22% of the composite, so about "
        "3.5 points), halved for companies whose valuation score is below the middle, because the paper "
        "defines the effect among high book-to-market companies. The screener ranks S&P 500 companies "
        "within their sectors on its own version of the nine signals; the paper, and this rerun, sort the "
        "highest book-to-market fifth of all listed companies, most of them small."),
}

EXPERIMENTS.append(PIOTROSKI_OSAP)


# One sentence per experiment for the lab's front page. Each is a statement about what the
# published data shows, and tests/test_factor_lab.py checks every one against that data, so a
# rebuild that makes one untrue fails the suite instead of leaving it on the page.
TAKEAWAYS = {
    "risk-managed-momentum": ("Scaling momentum by its own recent volatility roughly doubled its Sharpe "
                              "ratio in the paper's sample, and the public data agrees. After the sample "
                              "ended, both versions earned less."),
    "momentum-crashes": ("Momentum's worst months came after the market had fallen and while it was "
                         "volatile. That pattern reproduces closely; the levels of the paper's own "
                         "portfolio do not."),
    "five-factors": ("Four factors averaged about a third of a percent a month for fifty years. Since "
                     "the paper, none has been distinguishable from zero - over a period too short to "
                     "settle it."),
    "size-effect": ("Smaller companies earned more, but by less than the paper's table shows once every "
                    "listed stock is counted. Between the two largest groups the difference is small."),
    "low-volatility": ("The calmest fifth of stocks earned about as much as the rest, with half the "
                       "volatility of the most volatile fifth. The paper's large penalty for volatile "
                       "stocks does not appear on the public version of the sort."),
    "earnings-yield": ("Returns rose step by step with earnings yield, in the paper's sample and for "
                       "decades after it. Since 2007 the gap has all but closed, and among large "
                       "companies it reversed."),
    "big-and-small": ("Among big companies - all the screener ranks - value, profitability and investment "
                      "each earned roughly half what they did among small ones in the paper's sample. "
                      "Since 2000 only profitability has stayed distinguishable from zero there."),
    "largest-fifth": ("In the largest fifth of companies, past winners beat past losers by nearly 8% a year "
                      "from 1927 to 2011, as the paper found. Since 2000 the gap has averaged under 3% a year "
                      "and cannot be told from zero. Value there was never reliably different from zero."),
    "profitability": ("The most profitable fifth of companies beat the least profitable by about 0.2% a "
                      "month on the public sort, against the paper's 0.31% - every figure within its "
                      "tolerance. Since the paper the gap has been about as large but cannot be told "
                      "from zero, and the profitable end has been the calmer one."),
    "value-and-momentum": ("Among big companies, value and momentum moved against each other in every period "
                           "measured, and more so after 2011 (-0.37). But since then neither earned anything, so "
                           "the mix that once beat both now earns about nothing too: a hedge, not a return."),
    "accruals": ("Low-accrual companies beat high-accrual ones in the paper's years, by about half the "
                 "paper's margin on the library's measure. Since then the gap has narrowed: weighted by "
                 "market value it is 0.09% a month since 2007, not distinguishable from zero."),
    "cash-flow-yield": ("High cash-flow yields beat low ones in the paper's sample, and equally weighted they "
                        "still have since. Weighted by market value - where large companies count most - "
                        "the gap has been about zero since 1990 and below zero since 2007, though not "
                        "distinguishably."),
    "asset-growth": ("Companies that grew their assets slowest beat the fastest growers in the paper's years on "
                     "the public sort too, by less than the paper found. Since the paper, equally weighted the "
                     "gap is a quarter of its old size, and weighted by market value it has been about zero."),
    "replication-crisis": ("Of the 119 factors their own papers found significant, 83% have a CAPM alpha clearly "
                           "above zero on the public library to 2020, and 76% inside the mega caps alone - "
                           "close to the paper's 77%. Since 2014 only 9 of them clear that bar among mega caps, "
                           "over a period too short to settle it."),
    "size-given-profitability": ("Holding profitability fixed, the fourth size group beat the largest by about 0.24% a "
                                 "month in the paper's years, close to the paper's 0.25%, and by 0.17% over 1963-2026 - "
                                 "enough to pass the test the screener's research note set in advance. Since 2013 it "
                                 "has been below zero, over a period too short to settle it."),
    "piotroski": ("On the authors' public portfolios, the highest-scoring tenth of high book-to-market companies "
                  "beat the lowest by about 4% a year equally weighted in the paper's years - a fifth of the "
                  "paper's 23.5%, and not distinguishable from zero - and by about 13% weighted by market value. "
                  "Since 1997 neither gap can be told from zero."),
    "market-beta": ("From the lowest-beta group to the highest, average returns were flat in the paper's "
                    "sample, and the difference has not been distinguishable from zero since - while the "
                    "highest-beta group has been about twice as volatile."),
}

# What the lab can and cannot put behind each of the screener's eight categories. "experiments"
# are ids above; "cannot" names studies the methodology cites that need data the lab does not
# have, and says what that data is. Kept honest by hand and by one test: every experiment id
# exists, and every category of the screener's config appears exactly once.
CATEGORY_MAP = [
    {"category": "Valuation", "key": "valuation",
     "experiments": ["earnings-yield", "cash-flow-yield", "value-and-momentum", "five-factors", "big-and-small", "largest-fifth",
                     "replication-crisis"],
     "has": "Stocks sorted by earnings yield and by cash-flow yield, and the value factor (book value "
            "against price).",
     "gap": "The screener's heaviest valuation measure is free cash flow (after capital spending) against "
            "the value of the whole business, compared within sectors. The public sorts use cash flow "
            "before capital spending against the shares alone, across the whole market."},
    {"category": "Quality", "key": "quality",
     "experiments": ["profitability", "accruals", "piotroski", "five-factors", "big-and-small", "replication-crisis"],
     "has": "Stocks sorted by profitability and by accruals, the profitability factor - robustly "
            "profitable companies minus weakly profitable ones - and Piotroski's financial-strength score "
            "among high book-to-market companies, on the portfolios Chen and Zimmermann publish.",
     "gap": "The Piotroski rerun reads published portfolio returns, not company accounts, so it checks a "
            "careful public construction of the score, not the paper's own numbers, and only among high "
            "book-to-market companies. Beneish (1999), the methodology's earnings-manipulation score, "
            "cannot be rerun here: it needs each company's accounts for decades, and the accruals sort "
            "covers one of its eight inputs."},
    {"category": "Growth", "key": "growth", "experiments": ["asset-growth"],
     "has": "Nothing on the category's own measures. The asset-growth sort is Investment's measure, but it is "
            "the one Growth's revenue growth moves with, and the two categories score it in opposite "
            "directions.",
     "gap": "The methodology's case for Growth is an argument about forward-looking information, not a "
            "published return premium, so there is no headline result to rerun. Mohanram (2005), which it "
            "cites, needs company accounts, and the library has no sort on sales or earnings growth."},
    {"category": "Momentum", "key": "momentum",
     "experiments": ["risk-managed-momentum", "momentum-crashes", "value-and-momentum", "big-and-small", "largest-fifth",
                     "replication-crisis"],
     "has": "Momentum's risk, when it crashes, its size among big companies, and how it offsets value.",
     "gap": "Jegadeesh and Titman (1993), the study the methodology cites for momentum itself, builds its "
            "portfolios differently from the public file. A direct rerun is still to do."},
    {"category": "Risk", "key": "risk", "experiments": ["low-volatility", "market-beta", "replication-crisis"],
     "has": "Stocks sorted by how volatile they have been, and by how much they move with the market "
            "(beta): two of the category's three measures.",
     "gap": "The depth of a stock's worst fall, the third measure. No public sort exists for it."},
    {"category": "Analyst revisions", "key": "revisions", "experiments": [],
     "has": "Nothing, and probably never.",
     "gap": "Cannot be rerun here: Chan, Jegadeesh and Lakonishok (1996) and the later work on earnings "
            "surprises. They need a history of analysts' estimates as they stood at the time, which is "
            "sold, not published."},
    {"category": "Size", "key": "size", "experiments": ["size-effect", "size-given-profitability", "five-factors", "replication-crisis"],
     "has": "The size effect by size group, and the size factor.",
     "gap": "Which size groups the screener's own companies fall in has not been measured."},
    {"category": "Investment", "key": "investment", "experiments": ["asset-growth", "five-factors", "big-and-small", "replication-crisis"],
     "has": "Stocks sorted by asset growth - the category's one measure - and the investment factor: "
            "companies that grow their assets slowly, minus those that grow them fast.",
     "gap": "It overlaps heavily with value; the five-factor experiment shows by how much."},
]


# Where each category stands on what the lab has measured, for the front page. This is the
# lab's whole route into the methodology and it is deliberately not a code path: nothing here
# sets a weight (tests/test_factor_lab.py::test_no_scoring_module_reads_the_lab). A row raises
# a question; the answer is a research note from the literature and from practice, then a
# METHODOLOGY_CHANGELOG.md entry - plan/factor-lab.md, "From the lab to the weights".
# "open" is written by hand; every figure in "measured" is computed below, and the words
# around the figures are checked against the data by the tests.
SCORECARD_OPEN = {
    "valuation": ("Reviewed 2026-10-12: on book value there was never a reliable premium among the "
                  "largest companies, and the published studies find it larger on earnings and cash-flow "
                  "measures, which are the ones the screener weights. Those measures, compared within "
                  "sectors, have still not been rerun here."),
    "quality": ("Whether the screener's financial-strength and manipulation scores add to plain "
                "profitability among large companies. The financial-strength score now has a rerun on "
                "published portfolios, among high book-to-market companies of every size; neither score "
                "can be tested among large companies alone on public data."),
    "growth": ("Researched 2026-10-12 and the weight reviewed 2026-10-13: index providers treat "
               "growth as a style, not a rewarded factor, and the studies of sales growth and "
               "growth forecasts run flat to negative. The 3-point reduction that followed was "
               "refused - this is the one category whose score runs against the others, so its "
               "weight is what keeps the ranking from leaning either way on growth, and the "
               "evidence does not settle which way it should. The open question is now what the "
               "category is made of, not what it is worth."),
    "momentum": ("Reviewed 2026-10-12: every published study finds momentum among large companies, "
                 "and none has data past 2014. Outside the panic state the crash rule watches for, it "
                 "has averaged about 0.3% a month since 2000. The lab and the literature disagree, so "
                 "the weight stays, as the least well-supported of the large ones."),
    "risk": ("The category lowers risk for a similar return. It is not evidence of a higher one, and "
             "the methodology now says so."),
    "revisions": ("It rests on the cited studies alone. The screener's own record, run by run, is the "
                  "only measurement it will get."),
    "size": ("Researched 2026-10-20: no size premium among large companies on its own; the weight rests on "
             "the case that size pays once quality is held fixed. Holding profitability fixed passed the test "
             "written for it in advance, by a margin over the plain spread too small to tell from zero. Since "
             "2013 the spread with profitability held fixed has been below zero."),
    "investment": "How much of this category is Valuation counted a second time.",
}


def screener_weights(path: Path = ROOT / "config.yaml") -> dict:
    """The category weights as the screener's config has them, for display beside the evidence.
    Read at build time; a test fails if the published copy and the config part company."""
    import yaml
    return dict(yaml.safe_load(path.read_text(encoding="utf-8"))["factor_weights"])


def scorecard(split: dict, size: dict, weights: dict, asset_growth: dict) -> list[dict]:
    """One row per category of the screener: its weight, what the lab has measured that bears
    on it among large companies, and what that leaves open."""
    whole, late, recent = (split_run(split, "196307", "209912"), split_run(split, "200001", "209912"),
                           split_run(split, "201401", "209912"))

    def big(r, k):
        return r["factors"][k]["big"]

    def small(r, k):
        return r["factors"][k]["small"]

    idx = range(len(split["months"]))
    overlap = correlation([split["value"]["big"][i] for i in idx], [split["investment"]["big"][i] for i in idx])
    nine_ten = size_run(size, size["months"][0], "209912", "ew")["nine_ten"]
    ag_paper, ag_after = (sort_run(asset_growth, a, b, "vw")["spread"] for a, b in
                          (("196807", "200306"), ("200307", "209912")))
    measured = {
        "valuation": (f"Among big companies the value factor has averaged {big(whole, 'value')['mean']:.2f}% a month "
                      f"since 1963 (t-statistic {big(whole, 'value')['t']:.2f}), about half its "
                      f"{small(whole, 'value')['mean']:.2f}% among small ones, and "
                      f"{big(recent, 'value')['mean']:+.2f}% a month since 2014. Sorted on earnings yield, "
                      f"the gap among large companies reversed after 2007."),
        "quality": (f"Profitability is the factor that carries over best: {big(whole, 'profitability')['mean']:.2f}% "
                    f"a month among big companies since 1963 (t-statistic {big(whole, 'profitability')['t']:.2f}) "
                    f"against {small(whole, 'profitability')['mean']:.2f}% among small ones, and the only one of "
                    f"the four still distinguishable from zero among big companies since 2000. Under half of "
                    f"it is profitable companies beating the market "
                    f"({whole['factors']['profitability']['long']['mean']:.2f}% a month); the rest is "
                    f"unprofitable ones lagging it."),
        "growth": (f"Nothing on the category's own measures - there is no published return premium for growth "
                   f"to rerun. The nearest sort is Investment's asset growth, which revenue growth moves with: "
                   f"by market value, the slowest growers beat the fastest by {ag_paper['mean']:.2f}% a month in "
                   f"its paper's years (t-statistic {ag_paper['t']:.2f}) and by {ag_after['mean']:+.2f}% since "
                   f"(t-statistic {ag_after['t']:.2f})."),
        "momentum": (f"Among big companies momentum has averaged {big(whole, 'momentum')['mean']:.2f}% a month since "
                     f"1963 (t-statistic {big(whole, 'momentum')['t']:.2f}), under half its "
                     f"{small(whole, 'momentum')['mean']:.2f}% among small ones - and "
                     f"{big(late, 'momentum')['mean']:+.2f}% a month since 2000 "
                     f"(t-statistic {big(late, 'momentum')['t']:.2f})."),
        "risk": ("Sorted by volatility or by beta, riskier stocks have not earned reliably more, while "
                 "being about twice as volatile."),
        "revisions": ("Nothing. A history of analysts' estimates as they stood at the time is sold, not "
                      "published."),
        "size": (f"Between the two largest size groups the smaller has earned {nine_ten['mean']:.2f}% a month more "
                 f"over the whole record (t-statistic {nine_ten['t']:.2f}), and the difference cannot be told "
                 f"from zero in any shorter period checked."),
        "investment": (f"Among big companies the investment factor has averaged {big(whole, 'investment')['mean']:.2f}% "
                       f"a month since 1963 (t-statistic {big(whole, 'investment')['t']:.2f}) and "
                       f"{big(recent, 'investment')['mean']:+.2f}% since 2014. It moves with value: among big "
                       f"companies the two have a correlation of {overlap:.2f}."),
    }
    by_key = {c["key"]: c for c in CATEGORY_MAP}
    return [{"key": k, "category": by_key[k]["category"], "weight": w, "measured": measured[k],
             "open": SCORECARD_OPEN[k], "experiments": by_key[k]["experiments"]}
            for k, w in weights.items()]


def next_month(m: str) -> str:
    y, mo = int(m[:4]), int(m[4:])
    return f"{y + (mo == 12)}{(mo % 12) + 1:02d}"


def after_the_paper(mom: dict, market: dict, factors: dict, size: dict, sorts: dict,
                    split: dict | None = None, quintiles: dict | None = None, jkp: dict | None = None,
                    double: dict | None = None) -> list[dict]:
    """Each experiment's headline result in its paper's sample and in the months since, side by
    side - the lab's one table about what happens to a published result after publication
    (plan/factor-lab.md queue item 8; McLean & Pontiff 2016 is the study of the question).

    `sign` is the direction the paper claims (+1, -1, or 0 where its claim is "no difference");
    `weaker` says whether the later figure is smaller in that direction. Nothing here is a test
    of significance: the later periods are short, and each row carries its months."""
    exps = {x["id"]: x for x in EXPERIMENTS}
    end = "209912"
    rows = []

    def add(exp_id, label, unit, sign, inside, after):
        weaker = None if sign == 0 or after["value"] is None else sign * after["value"] < sign * inside["value"]
        rows.append({"experiment": exp_id, "label": label, "unit": unit, "sign": sign,
                     "inside": inside, "after": after, "weaker": weaker})

    def cell(value, months, t=None):
        return {"value": value, "t": t, "months": months}

    ref = exps["risk-managed-momentum"]["reference"]
    a, b = run(mom, ref["start"], ref["end"]), run(mom, next_month(ref["end"]), end)
    for leg, name in (("plain", "Plain momentum"), ("managed", "Risk-managed momentum")):
        add("risk-managed-momentum", f"{name}: Sharpe ratio", "ratio", 1,
            cell(a[leg + "_stats"]["sharpe"], len(a["months"])), cell(b[leg + "_stats"]["sharpe"], len(b["months"])))

    ref = exps["momentum-crashes"]["reference"]
    a = crash_run(mom, market, ref["start"], ref["end"])
    b = crash_run(mom, market, next_month(ref["end"]), end)
    add("momentum-crashes", "Momentum's average month in the panic state", "pct", -1,
        cell(a["cells"][3]["mean"], a["cells"][3]["n"]), cell(b["cells"][3]["mean"], b["cells"][3]["n"]))

    ref = exps["five-factors"]["reference"]
    a, b = factor_run(factors, ref["start"], ref["end"]), factor_run(factors, next_month(ref["end"]), end)
    for k in ("hml", "rmw", "cma", "smb"):
        add("five-factors", f"{FACTOR_NAMES[k]} factor: average month", "pct", 1,
            cell(a["stats"][k]["mean"], len(a["months"]), a["stats"][k]["t"]),
            cell(b["stats"][k]["mean"], len(b["months"]), b["stats"][k]["t"]))

    ref = exps["size-effect"]["reference"]
    a = size_run(size, ref["start"], ref["end"], ref["weighting"])
    b = size_run(size, next_month(ref["end"]), end, ref["weighting"])
    add("size-effect", "Smallest size group minus largest, a month", "pct", 1,
        cell(a["small_big"]["mean"], len(a["months"]), a["small_big"]["t"]),
        cell(b["small_big"]["mean"], len(b["months"]), b["small_big"]["t"]))

    for exp_id, sign in (("low-volatility", -1), ("earnings-yield", 1), ("market-beta", 0),
                         ("profitability", 1), ("cash-flow-yield", 1), ("accruals", 1), ("asset-growth", 1),
                         ("piotroski", 1)):
        x = exps[exp_id]
        ref = x["reference"]
        data = sorts[x["sort"]["data"]]
        a = sort_run(data, ref["start"], ref["end"], ref["weighting"])
        b = sort_run(data, next_month(ref["end"]), end, ref["weighting"])
        add(exp_id, x["sort"]["spread"] + ", a month", "pct", sign,
            cell(a["spread"]["mean"], len(a["months"]), a["spread"]["t"]),
            cell(b["spread"]["mean"], len(b["months"]), b["spread"]["t"]))

    if split is not None and "big-and-small" in exps:
        ref = exps["big-and-small"]["reference"]
        a, b = split_run(split, ref["start"], ref["end"]), split_run(split, next_month(ref["end"]), end)
        for k in SPLIT_KEYS:
            add("big-and-small", f"{SPLIT_NAMES[k]} among big companies: average month", "pct", 1,
                cell(a["factors"][k]["big"]["mean"], len(a["months"]), a["factors"][k]["big"]["t"]),
                cell(b["factors"][k]["big"]["mean"], len(b["months"]), b["factors"][k]["big"]["t"]))

    if split is not None and "value-and-momentum" in exps:
        ref = exps["value-and-momentum"]["reference"]
        a = combo_run(split, factors, ref["start"], ref["end"], ref["group"])
        b = combo_run(split, factors, next_month(ref["end"]), end, ref["group"])
        add("value-and-momentum", "Value and momentum, 50/50, among big companies: Sharpe ratio", "ratio", 1,
            cell(a["combo"]["sharpe"], len(a["months"])), cell(b["combo"]["sharpe"], len(b["months"])))

    if quintiles is not None and "largest-fifth" in exps:
        ref = exps["largest-fifth"]["reference"]
        a = split_run(quintiles, ref["start"], ref["end"])
        b = split_run(quintiles, next_month(ref["end"]), end)
        for k in QUINT_KEYS:
            add("largest-fifth", f"{SPLIT_NAMES[k]} in the largest fifth: average month", "pct", 1,
                cell(a["factors"][k]["big"]["mean"], len(a["months"]), a["factors"][k]["big"]["t"]),
                cell(b["factors"][k]["big"]["mean"], len(b["months"]), b["factors"][k]["big"]["t"]))

    if jkp is not None and "replication-crisis" in exps:
        # the replication paper's own sample ends with 2020; "since" is the five years after it
        span = lambda a, b: sum(1 for m in jkp["months"] if a <= m <= b)            # noqa: E731
        for u, name in (("all", "all stocks"), ("mega", "mega caps")):
            a = jkp_run(jkp, "000000", JKP_END, u, "capm", "sig")
            b = jkp_run(jkp, next_month(JKP_END), end, u, "capm", "sig")
            add("replication-crisis", f"The 119 published factors among {name}: average CAPM alpha, a month",
                "pct", 1, cell(a["alpha"], span("000000", JKP_END)), cell(b["alpha"], span(next_month(JKP_END), end)))

    if double is not None and "size-given-profitability" in exps:
        ref = exps["size-given-profitability"]["reference"]
        a, b = double_run(double, ref["start"], ref["end"]), double_run(double, next_month(ref["end"]), end)
        add("size-given-profitability", "Fourth size group minus largest, profitability held fixed, a month", "pct", 1,
            cell(a["fourq"]["mean"], len(a["months"]), a["fourq"]["t"]),
            cell(b["fourq"]["mean"], len(b["months"]), b["fourq"]["t"]))
    return rows


def compare(reference: dict, measured: dict, tolerances: dict = TOLERANCES) -> list[dict]:
    """One row per reported figure: the paper's, ours, the difference and whether it is
    inside the tolerance."""
    rows = []
    for leg in ("plain", "managed"):
        for key in STAT_ORDER:
            ref, got = reference[leg][key], measured[leg + "_stats"][key]
            diff = got - ref
            rows.append({"leg": leg, "stat": key, "label": STAT_LABELS[key], "paper": ref,
                         "here": got, "diff": diff, "tolerance": tolerances[key],
                         "within": abs(diff) <= tolerances[key]})
    return rows


# What a result of no effect at all would land within tolerance on (plan/factor-lab.md rule 8).
# A tolerance wide enough to pass a null result is not evidence of a replication, so every
# experiment publishes how many of its figures the null would match, beside its own verdict.
# The null is built from the record alone - the paper's figures and nothing measured:
#   * every group earns the same: a group's average is the average of the paper's group figures
#     (zero for a table already set against its own average, `relative`);
#   * every spread, difference, t-statistic, correlation, loading, intercept, Sharpe ratio and
#     premium is zero;
#   * a figure the null says nothing about - a volatility, the market's own figures, a count of
#     months, a list of worst months - stays at the paper's value, which the null then matches.
# Experiment 1's claim is about scaling, not momentum: its null is the risk-managed strategy
# matching the plain one, figure for figure.
NULL_KEEPS_PREFIX = ("mkt_", "worst", "bear_months")      # the market's own figures, counts, worst months
NULL_KEEPS_SUFFIX = ("_mktrf",)
_NULL_VOL = re.compile(r"(^|_)std\d*($|_)")                # a volatility: std1, std_value, wml_std
# a group's level: mean3, dec7, or vw_mean3 for a second weighting of the same table (`also`),
# whose groups all earn that weighting's own average under the null
_NULL_LEVEL = re.compile(r"((?:[a-z]+_)?)(mean|dec)\d+$")
STAT_ORDER = ("max", "min", "mean", "std", "kurt", "skew", "sharpe")
# Shares of many factors (experiment 15): with no effect anywhere, a factor's t-statistic clears
# 1.96 in its paper's direction by chance 2.5% of the time, and half the factors land positive.
NULL_CHANCE = {"rate_": 2.5, "pos_": 50.0}
NULL_MEANING = {
    "replication": ("“No effect” here means every factor earning nothing: 2.5% of them would still clear "
                    "1.96 in their paper's direction by chance, half would land positive after their samples, "
                    "and every average alpha would be zero."),
    "risk-managed": ("“No effect” here means the risk-managed strategy matching the plain one figure "
                     "for figure."),
    "default": ("“No effect” here means every group earning the same and every spread, t-statistic, "
                "correlation and premium at zero; volatilities, the market's own figures and counts of "
                "months stay at the paper's values, because no effect on returns says nothing about them."),
}


def null_rows(exp: dict) -> list[dict]:
    """The paper's figure, the null result's value and the tolerance, in the comparison's order."""
    ref = exp["reference"]
    if exp["kind"] == "risk-managed":
        return [{"paper": ref[leg][k], "null": ref["plain"][k], "tolerance": TOLERANCES[k]}
                for leg in ("plain", "managed") for k in STAT_ORDER]
    figs = ref["figures"]
    levels: dict[str, list[float]] = {}
    for f in figs:
        m = _NULL_LEVEL.match(f["key"])
        if m:
            levels.setdefault(m.group(1), []).append(f["paper"])
    centres = {pre: 0.0 if ref.get("relative") else sum(v) / len(v) for pre, v in levels.items()}
    out = []
    for f in figs:
        k = f["key"]
        if k.startswith(NULL_KEEPS_PREFIX) or k.endswith(NULL_KEEPS_SUFFIX) or _NULL_VOL.search(k):
            null = f["paper"]
        elif k.startswith(tuple(NULL_CHANCE)):
            null = next(v for pre, v in NULL_CHANCE.items() if k.startswith(pre))
        elif _NULL_LEVEL.match(k):
            null = centres[_NULL_LEVEL.match(k).group(1)]
        else:
            null = 0.0
        out.append({"paper": f["paper"], "null": null, "tolerance": f["tolerance"]})
    return out


def null_within(exp: dict) -> int:
    """How many of an experiment's figures a result of no effect would land within tolerance."""
    return sum(abs(r["null"] - r["paper"]) <= r["tolerance"] for r in null_rows(exp))


def null_meaning(exp: dict) -> str:
    return NULL_MEANING.get(exp["kind"], NULL_MEANING["default"])


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


def rf_mean(data: dict, start: str, end: str) -> float:
    """The one-month T-bill rate's average over a period, percent a month, from the sort's own
    published ``rf`` (the library's RF for the same months)."""
    vals = [r for m, r in zip(data["months"], data["rf"]) if start <= m <= end and r is not None]
    return sum(vals) / len(vals)


def reproduce_sort(data: dict, exp: dict) -> dict:
    """A sorted-portfolio experiment at the paper's sample and weighting, against its table.

    With ``reference["excess"]`` the paper reports returns over T-bills, so each group's average
    has the T-bill rate's average over the same months taken off - the mean of a difference is
    the difference of the means, exactly. The spread between two groups is unchanged by it."""
    ref = exp["reference"]
    r = sort_run(data, ref["start"], ref["end"], ref["weighting"])
    rf = rf_mean(data, ref["start"], ref["end"]) if ref.get("excess") else 0.0
    # ``relative``: each group against the average of all the groups - comparable with a paper
    # that reports size-adjusted returns, whose groups average close to zero; ``scale``: the
    # paper's unit (12 for percent a year from percent a month)
    scale = ref.get("scale", 1)

    def figures(run: dict, prefix: str = "") -> dict:
        centre = (sum(s["mean"] for s in run["groups"]) / len(run["groups"])) if ref.get("relative") else rf
        out = {f"{prefix}spread": run["spread"]["mean"] * scale, f"{prefix}t_spread": run["spread"]["t"]}
        for g, s in enumerate(run["groups"], 1):
            out[f"{prefix}mean{g}"], out[f"{prefix}std{g}"] = (s["mean"] - centre) * scale, s["std"]
        return out

    got = figures(r)
    # ``also``: a paper that reports the same table under a second weighting (experiment 14: equal
    # and value weights) - those figures carry the weighting as a prefix, ``vw_mean1``, ``vw_spread``
    if ref.get("also"):
        got.update(figures(sort_run(data, ref["start"], ref["end"], ref["also"]), ref["also"] + "_"))
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    return {"rows": rows, "within": sum(x["within"] for x in rows), "total": len(rows),
            "months": len(r["months"]), "means": [g["mean"] for g in r["groups"]],
            "spread": r["spread"]}


def reproduce_split(split: dict, exp: dict) -> dict:
    """A split experiment at the paper's sample, against its table. `scale` turns an average
    month into the paper's unit (12 where the paper reports percent a year)."""
    ref = exp["reference"]
    r = split_run(split, ref["start"], ref["end"])
    scale = ref.get("scale", 1)
    got = {}
    for key, parts in r["factors"].items():
        for part, s in parts.items():
            got[f"mean_{key}_{part}"], got[f"t_{key}_{part}"] = s["mean"] * scale, s["t"]
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    return {"rows": rows, "within": sum(x["within"] for x in rows), "total": len(rows),
            "months": len(r["months"]),
            "big_means": {k: v["big"]["mean"] for k, v in r["factors"].items()}}


def reproduce_combo(split: dict, factors: dict, exp: dict) -> dict:
    """The value-and-momentum experiment at the paper's sample, against its table."""
    ref = exp["reference"]
    r = combo_run(split, factors, ref["start"], ref["end"], ref["group"])
    got = {"corr_resid": r["corr_resid"]}
    for k in ("value", "momentum", "combo"):
        got[f"mean_{k}"], got[f"std_{k}"], got[f"sharpe_{k}"] = r[k]["mean_a"], r[k]["std_a"], r[k]["sharpe"]
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    return {"rows": rows, "within": sum(x["within"] for x in rows), "total": len(rows),
            "months": len(r["months"]), "sharpe": {k: r[k]["sharpe"] for k in ("value", "momentum", "combo")},
            "corr_resid": r["corr_resid"]}


def reproduce_double(data: dict, exp: dict) -> dict:
    """A size-by-trait experiment at the paper's sample, against its table, and - separately, never
    counted in the comparison - the screener's own pre-registered rule read on the same data."""
    ref = exp["reference"]
    got = double_figures(data, ref["start"], ref["end"])
    rows = []
    for f in ref["figures"]:
        here = got[f["key"]]
        diff = here - f["paper"]
        rows.append({**f, "here": here, "diff": diff, "within": abs(diff) <= f["tolerance"]})
    r = double_run(data, ref["start"], ref["end"])
    return {"rows": rows, "within": sum(x["within"] for x in rows), "total": len(rows),
            "months": len(r["months"]), "means": [c["mean"] for c in r["cells"]],
            "smbq": r["smbq"], "fourq": r["fourq"],
            "criterion": criterion_result(data, exp["criterion"])}


def reproduce(data: dict, exp: dict) -> dict:
    """Run an experiment at the paper's own settings and compare."""
    ref = exp["reference"]
    res = run(data, ref["start"], ref["end"], ref["window"], ref["target"], ref["cap"])
    rows = compare(ref, res)
    return {"rows": rows, "within": sum(r["within"] for r in rows), "total": len(rows),
            "months": len(res["months"]), "first": res["months"][0], "last": res["months"][-1],
            "weights": res["weight_stats"]}


# --------------------------------------------------------------------------- the build
def build_payload(raw_dir: Path = RAW_DIR, now: datetime | None = None, jkp_dir: Path = JKP_DIR) -> dict:
    data = momentum_data(raw_dir)
    jkp = jkp_data(jkp_dir)
    market = market_data(data["months"], raw_dir)
    factors = factor_data(raw_dir)
    size = size_data(raw_dir)
    sorts = {name: sort_data(name, raw_dir) for name in SORTS}
    split = split_data(raw_dir)
    quintiles = quintile_data(raw_dir)
    double = double_data(raw_dir)
    exps = []
    for exp in EXPERIMENTS:
        result = (reproduce_split(quintiles if exp.get("split", {}).get("data") == "quintiles" else split, exp)
                  if exp["kind"] == "split"
                  else reproduce_crashes(data, market, exp) if exp["kind"] == "crashes"
                  else reproduce_factors(factors, exp) if exp["kind"] == "factors"
                  else reproduce_size(size, exp) if exp["kind"] == "size"
                  else reproduce_sort(sorts[exp["sort"]["data"]], exp) if exp["kind"] == "sort"
                  else reproduce_combo(split, factors, exp) if exp["kind"] == "combo"
                  else reproduce_replication(jkp, exp) if exp["kind"] == "replication"
                  else reproduce_double(double, exp) if exp["kind"] == "double"
                  else reproduce(data, exp))
        for row, n in zip(result["rows"], null_rows(exp)):            # what no effect at all would show
            row["null"] = n["null"]
        result["null_within"], result["null_meaning"] = null_within(exp), null_meaning(exp)
        exps.append({**exp, "takeaway": TAKEAWAYS[exp["id"]], "result": result})
    return {
        "built": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d"),
        "attribution": {
            "text": "Data: Kenneth R. French Data Library. Copyright Eugene F. Fama and Kenneth R. French.",
            "url": FRENCH_HOME, "vintage": data["vintage"],
            "note": ("The monthly series here are derived from the library's files (top momentum "
                     "decile minus bottom decile, the market's excess return, and volatility and "
                     "trailing-return readings computed from the daily and monthly files). "
                     "Download the originals from the library; this page does not republish them."),
            # a second source, for the experiments that read it (SORTS entries with source "osap")
            "others": [{
                "text": ("Data: Open Source Asset Pricing, Andrew Y. Chen and Tom Zimmermann, " + OSAP_RELEASE
                         + ". Cite: Chen and Zimmermann (2022), \"Open source cross-sectional asset pricing\", "
                         "Critical Finance Review 11(2), 207-264."),
                "url": OSAP_HOME,
                "note": ("The Piotroski experiment reads the authors' published monthly returns for the lowest and "
                         "highest deciles of their Piotroski-score portfolios, equally and value-weighted, which "
                         "they built from CRSP and Compustat. Download the originals from their data page."),
            }],
        },
        "tolerances": TOLERANCES, "tolerances_written": TOLERANCES_WRITTEN,
        "stat_labels": STAT_LABELS, "vol_windows": list(VOL_WINDOWS),
        "market_vol_windows": list(MARKET_VOL_WINDOWS), "bear_lookbacks": list(BEAR_LOOKBACKS),
        "state_names": STATE_NAMES, "category_map": CATEGORY_MAP,
        "attribution_jkp": {
            "text": ("Data: Global Factor Data, Jensen, Kelly and Pedersen (2023), \"Is there a replication crisis "
                     "in finance?\", Journal of Finance 78(5). Copyright (c) 2023 Theis Ingerslev Jensen, "
                     "Bryan T. Kelly, Lasse Heje Pedersen. Licensed CC BY-NC 4.0."),
            "url": JKP_HOME, "licence": JKP_LICENCE,
            "note": ("Changed here: the factor returns are rounded to a basis point, the size-group series are "
                     "turned the way each factor's paper found it paid, and nothing after the files' own months "
                     "is added. Non-commercial use only, under the licence."),
        },
        "after_the_paper": after_the_paper(data, market, factors, size, sorts, split, quintiles, jkp, double),
        "scorecard": scorecard(split, size, screener_weights(), sorts["asset-growth"]),
        "factor_keys": FACTOR_KEYS, "factor_names": FACTOR_NAMES, "factor_long": FACTOR_LONG,
        "factor_category": FACTOR_CATEGORY,
        "factors": {k: factors[k] for k in ["months"] + FACTOR_KEYS},
        "size": {k: size[k] for k in ("months", "ew", "vw")},
        "sorts": sorts,
        "quintiles": quintiles,
        "double": {k: double[k] for k in ("months", "vw")},
        "split": split, "split_keys": SPLIT_KEYS, "split_names": SPLIT_NAMES, "split_category": SPLIT_CATEGORY,
        "momentum": {k: data[k] for k in ("months", "wml", "vol", "daily_first", "daily_last")},
        "jkp": jkp,
        "market": market,
        "experiments": exps,
    }


# The series the page's first screen - the overview and the menu - never reads. Each is written to
# its own file, lab_data_<key>.js, which adds it to window.LAB_DATA, and lab.html loads it only when
# an experiment that reads it is opened (PARTS there names which kind reads which). Measured
# 2026-10-26 (plan/factor-lab.md, "The first paint"): with every series in lab_data.js the page
# fetched 1.18 MB gzipped before the menu appeared - 6.4 s on a throttled phone, of which the 153
# JKP factors were 0.71 MB. Content is unchanged by the split: load_published() puts it back
# together, and tests/test_factor_lab.py checks the parts against the build.
def sort_part(name: str) -> str:
    """The part holding one sort's series (a valid identifier: lab_data_<key>.js assigns LAB_DATA.<key>)."""
    return "sorts_" + name.replace("-", "_")


# A part that is one branch of an object, and the path it is put back at (round three, 2026-10-27).
# The 153 factors carried all three universes in one 716 KB gzipped part, and opening the experiment
# took 4.9-5.1 s on the phone profile: it now opens on "all" alone, draws the mega-cap bars when their
# series arrives, and loads the large-cap series only when chosen. The eight sorts shared one 212 KB
# part though each experiment reads one of them (1.4-1.55 s to open any): each is now its own, and
# `sorts` in lab_data.js is the empty object they are put back into. lab.html's INTO restates this
# map (the sorts' entries derived from the experiments, as here from SORTS); a branch of a part is
# listed after that part.
LAZY_INTO = {"jkp_mega": ("jkp", "series", "mega"), "jkp_large": ("jkp", "series", "large"),
             **{sort_part(name): ("sorts", name) for name in SORTS}}
LAZY_KEYS = ("jkp", "jkp_mega", "jkp_large", *(sort_part(name) for name in SORTS), "size", "double", "split", "quintiles")


def part_path(key: str, out: Path = OUT) -> Path:
    return out.with_name(f"lab_data_{key}.js")


def _without(obj: dict, path: tuple) -> dict:
    """A copy of obj with the value at path taken out, copying only the dicts along the path."""
    if len(path) == 1:
        return {k: v for k, v in obj.items() if k != path[0]}
    return {**obj, path[0]: _without(obj[path[0]], path[1:])}


def _at(obj: dict, path: tuple):
    for p in path:
        obj = obj[p]
    return obj


def split_parts(payload: dict) -> tuple[dict, dict]:
    """lab_data.js's content and each lazy part's, with every branch LAZY_INTO names taken out of
    whatever holds it - a part, or lab_data.js itself."""
    tops = [k for k in LAZY_KEYS if k not in LAZY_INTO]
    first = {k: v for k, v in payload.items() if k not in tops}
    parts = {k: payload[k] for k in tops}
    for key, path in LAZY_INTO.items():
        parts[key] = _at(payload, path)
        if path[0] in parts:
            parts[path[0]] = _without(parts[path[0]], path[1:])
        else:
            first = _without(first, path)
    return first, {k: parts[k] for k in LAZY_KEYS}


def _parse_js_value(path: Path):
    """The value a part file assigns - an object, or a list for a LAZY_INTO branch."""
    text = path.read_text(encoding="utf-8")
    body = text[text.index(" = ") + 3:].rstrip()
    return json.loads(body[:-1] if body.endswith(";") else body)


def _parse_js_object(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def write(payload: dict, out: Path = OUT) -> Path:
    def dump(o) -> str:
        return json.dumps(o, separators=(",", ":"))
    first, parts = split_parts(payload)
    out.write_text("window.LAB_DATA = " + dump(first) + ";\n", encoding="utf-8")
    for key, content in parts.items():
        part_path(key, out).write_text(f"window.LAB_DATA.{key} = " + dump(content) + ";\n", encoding="utf-8")
    return out


def load_published(path: Path = OUT) -> dict:
    """The whole published payload: lab_data.js with each lab_data_<key>.js put back."""
    data = _parse_js_object(path)
    for key in LAZY_KEYS:
        if key in LAZY_INTO:
            *up, last = LAZY_INTO[key]
            try:
                holder = _at(data, tuple(up))
            except KeyError:
                continue
            if last not in holder and part_path(key, path).exists():
                holder[last] = _parse_js_value(part_path(key, path))
        elif key not in data and part_path(key, path).exists():
            data[key] = _parse_js_object(part_path(key, path))
    return data


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
        for p in fetch_osap():
            print("fetched", p.name)
    if args.build:
        payload = build_payload()
        for p in [write(payload)] + [part_path(k) for k in LAZY_KEYS]:
            print("wrote", p.name, f"({p.stat().st_size / 1024:.0f} KB)")
        print(report(payload))
    elif args.report:
        print(report(load_published()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
