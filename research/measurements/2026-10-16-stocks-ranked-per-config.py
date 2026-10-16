"""How many stocks does each research-engine configuration actually rank? (2026-10-16 audit)

``v2_experiments.py`` compares every registered weighting with the baseline month by month. The
pairing is by month only: each configuration's ranking passes the engine's coverage filter against
*its own* weighted metrics (``backtest_v2.rank_month``), so a screen built on one or two measures
ranks a different, usually smaller, set of stocks than the baseline, and its IC is computed on that
set (``v2_experiments.per_month``). This counts, for every registered configuration, the stocks it
ranks with a next-month return in each era, and how many of them the baseline also ranks that month
(the intersection a same-stocks comparison would use - queued in ``plan/research-engine.md``).

It also measures one construction question the audit raised: the rank correlation between earnings
yield and gross profit to assets, the two halves of the ``profitability-and-value`` screen. Novy-Marx
(2013) pairs profitability with *book-to-market*, which it is negatively correlated with; whether the
earnings-yield version is a hedge at all is an empirical question.

It does NOT run an experiment: no IC, no spread, nothing appended to ``ledger.jsonl``. It reads
``cache/v2_metrics/*.parquet`` and ``cache/v2_daily.parquet`` and writes
``research/measurements/2026-10-16-stocks-ranked-per-config.json``. Not decision-grade (rule 5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v2_experiments as X  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
ERAS = {"presample": X.PRESAMPLE, "sample": X.SAMPLE}


def ranked_with_return(scores: pd.DataFrame, fwd: pd.DataFrame) -> dict[str, set]:
    """The stocks ``per_month`` would compute a month's IC on: ranked, with a next-month return,
    in a month of at least 50 such stocks."""
    out = {}
    for m, g in scores.groupby("month"):
        if m not in fwd.index:
            continue
        ret = fwd.loc[m].reindex(g["Ticker"].values).values
        keep = g.loc[pd.notna(ret) & g["Composite"].notna().values, "Ticker"]
        if len(keep) >= 50:
            out[m] = set(map(str, keep))
    return out


def ey_gpa_correlation(months: dict) -> dict:
    """Mean monthly Spearman correlation of earnings yield with gross profit to assets, raw and
    within sector, over every month both are present for at least 50 stocks."""
    raw, sect = [], []
    for m, (df, _) in sorted(months.items()):
        if "earnings_yield" not in df or "gross_profit_assets" not in df:
            continue
        d = df[["Sector", "earnings_yield", "gross_profit_assets"]].dropna()
        if len(d) < 50:
            continue
        raw.append(d["earnings_yield"].rank().corr(d["gross_profit_assets"].rank()))
        pe = d.groupby("Sector")["earnings_yield"].rank(pct=True)
        pg = d.groupby("Sector")["gross_profit_assets"].rank(pct=True)
        sect.append(pe.corr(pg))
    return {"months": len(raw), "raw_spearman_mean": round(float(np.mean(raw)), 3),
            "within_sector_percentile_corr_mean": round(float(np.mean(sect)), 3)}


_W: dict = {}


def _init() -> None:
    _W["months"] = X.load_months()
    _W["fwd"] = X.forward_returns(sorted(_W["months"]))
    _W["cfg0"] = X.base_config()


def _sets_for(tr: dict) -> dict[str, set]:
    if not _W:
        _init()
    cfg = X.apply_transform(_W["cfg0"], tr)
    return ranked_with_return(X.rank_all(_W["months"], cfg, whole_market=(tr["kind"] == "whole_market")),
                              _W["fwd"])


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    months = X.load_months()
    entries = [e for e in X.load_registry() if not e.get("void")]
    with ProcessPoolExecutor(max_workers=5, initializer=_init) as pool:   # one config per worker
        all_sets = dict(zip([e["id"] for e in entries], pool.map(_sets_for, [e["transform"] for e in entries])))
    base = all_sets["baseline"]
    result = {"what": __doc__.splitlines()[0], "data": X.data_fingerprint(months), "configs": {}}
    for e in entries:
        sets = all_sets[e["id"]]
        row = {}
        for era, (a, b) in ERAS.items():
            ms = [m for m in sets if a <= m <= b]
            if not ms:
                continue
            row[era] = {"months": len(ms),
                        "mean_stocks": round(float(np.mean([len(sets[m]) for m in ms])), 1),
                        "mean_common_with_baseline": round(float(np.mean(
                            [len(sets[m] & base.get(m, set())) for m in ms])), 1)}
        result["configs"][e["id"]] = row
        print(e["id"], row, flush=True)
    result["earnings_yield_vs_gross_profit_assets"] = ey_gpa_correlation(months)
    print(result["earnings_yield_vs_gross_profit_assets"])
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
