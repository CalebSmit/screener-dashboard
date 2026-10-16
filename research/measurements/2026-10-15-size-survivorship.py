"""Is the Size category's 2012-2019 IC survivorship? (research/2026-10-15-research-engine-first-batch.md)

v2 scores only the members that are still in the index today (or renamed into it), so the further
back a month, the more of its members are missing - 58% present in 2012, 96% in 2025. Among the
members that *are* present, the small ones are disproportionately those that did well afterwards
(a small company that fell out of the index is missing; one that thrived is not). If that is what
the Size-only screen's IC is picking up, its IC should be largest in the months with the lowest
coverage. This prints the Size-only and baseline IC against each month's coverage.

Reads cache/v2_metrics, cache/v2_daily.parquet and both censuses. Not decision-grade (rule 5).
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


def main() -> None:
    months = X.load_months()
    fwd = X.forward_returns(sorted(months))
    base = X.base_config()
    size = X.apply_transform(base, {"kind": "only_category", "category": "size"})
    pm = {"size": X.per_month(X.rank_all(months, size), fwd)["ic"],
          "baseline": X.per_month(X.rank_all(months, base), fwd)["ic"]}
    cov = {}
    for name in ("v2_census.json", "v2_census_presample.json"):
        for c in json.loads((ROOT / "cache" / name).read_text(encoding="utf-8")):
            if "members" in c:
                cov[c["month"]] = c["built"] / c["members"]
    df = pd.DataFrame(pm).join(pd.Series(cov, name="coverage"), how="inner").dropna()
    df["tercile"] = pd.qcut(df["coverage"], 3, labels=["lowest", "middle", "highest"])
    print(f"{len(df)} months\n")
    print(df.groupby("tercile", observed=True).agg(
        coverage=("coverage", "mean"), size_ic=("size", "mean"), baseline_ic=("baseline", "mean"),
        months=("size", "size")).round(4).to_string())
    for k in ("size", "baseline"):
        r = df[k].corr(df["coverage"])
        t = r * np.sqrt((len(df) - 2) / (1 - r * r))
        print(f"\ncorrelation of {k} IC with coverage: {r:+.3f} (t {t:+.2f})")
    df["year"] = df.index.str[:4]
    print("\n" + df.groupby("year").agg(coverage=("coverage", "mean"), size_ic=("size", "mean"),
                                         baseline_ic=("baseline", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
