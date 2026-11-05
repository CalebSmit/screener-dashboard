"""Can the research engine hold the experiment `research/2026-11-03-short-interest-days-to-cover.md`
section 5 asks for?

The proposal: register an experiment that **drops `short_interest_ratio` from Revisions**, and a
second that **scores FINRA days to cover in its place** "if the v2 record can carry it".

This script answers, by measurement rather than by argument:

 1. What the live screener puts on the metric (the thing the proposal wants a verdict on).
 2. What backtest v2's baseline puts on it, and on the Revisions category as a whole.
 3. Whether each of the three ways the registry can express "the baseline without this leg"
    produces a ranking that differs from the baseline at all - over every month on file.
 4. What the monthly records actually carry in the column.
 5. Whether the FINRA panel the note names can cover the months the engine's confirmation rule
    needs.

Run: ``python research/measurements/2026-11-05-short-interest-registration.py``
Writes ``research/measurements/2026-11-05-short-interest-registration.json``.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backtest_v2 as v2          # noqa: E402
import pit_record as pr           # noqa: E402
import v2_experiments as X        # noqa: E402

METRIC = "short_interest_ratio"
CATEGORY = "revisions"
OUT = Path(__file__).with_suffix(".json")


def live_weights() -> dict:
    """What the published screener scores the metric at."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cat_w = float(cfg["factor_weights"][CATEGORY])
    total = sum(float(v) for v in cfg["factor_weights"].values())
    within = cfg["metric_weights"][CATEGORY]
    return {
        "factor_weight_revisions": cat_w,
        "factor_weight_total": total,
        "within_category_weight": float(within[METRIC]),
        "within_category_total": sum(float(v) for v in within.values()),
        "composite_points": round(cat_w / total * 100 * float(within[METRIC])
                                 / sum(float(v) for v in within.values()), 3),
        "revisions_metrics": {k: float(v) for k, v in within.items()},
    }


def v2_weights() -> dict:
    """What v2's baseline - the configuration every registered experiment starts from - puts on it."""
    base = X.base_config()                      # = v2.v2_config(config.yaml)
    return {
        "factor_weight_revisions": float(base["factor_weights"][CATEGORY]),
        "revisions_metrics": {k: float(v) for k, v in base["metric_weights"][CATEGORY].items()},
        "metric_in_UNMEASURABLE_METRICS": METRIC in v2.UNMEASURABLE_METRICS,
        "categories_with_weight": {k: float(v) for k, v in base["factor_weights"].items()
                                   if float(v) > 0},
        "categories_at_zero": [k for k, v in base["factor_weights"].items() if not float(v)],
        # the registry registers only-X and drop-X for each category it can measure; whether
        # Revisions is among them is a fact about what a session before today already knew
        "registry_has_only_or_drop_revisions": [
            e["id"] for e in X.load_registry()
            if (e["transform"].get("category") == CATEGORY)],
    }


def arms() -> dict:
    """The three ways `experiments:` can express "the baseline without this leg", each written the
    way the registry's own precedents write it, and what each does to the configuration.

    ``drop-revenue-growth`` is the precedent for a metric-level drop: ``kind: config`` setting the
    metric's weight to 0. ``kind: metrics`` is the only other kind that names metrics.
    """
    base = X.base_config()
    live = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    keep = {cat: {m: float(w) for m, w in ms.items() if m != METRIC and float(w) > 0}
            for cat, ms in live["metric_weights"].items()}
    keep = {cat: ms for cat, ms in keep.items() if ms}
    out = {}
    for name, tr in {
        # the faithful one, written exactly as the registry's own metric-level precedent
        "config-set-weight-0": {"kind": "config",
                                "set": {f"metric_weights.{CATEGORY}.{METRIC}": 0}},
        # the whole category away, which is what "without this leg" reduces to if Revisions holds
        # nothing else v2 can measure
        "drop-category-revisions": {"kind": "drop_category", "category": CATEGORY},
        # every other metric at its live weight; the only kind that can name a metric positively
        "metrics-every-other-metric": {"kind": "metrics", "metrics": keep},
    }.items():
        cfg = X.apply_transform(base, tr)
        out[name] = {
            "transform": tr,
            "config_identical_to_baseline": cfg == base,
            "factor_weight_revisions": float(cfg["factor_weights"][CATEGORY]),
            "metric_weight": float(cfg["metric_weights"][CATEGORY].get(METRIC, 0.0)),
            "weight_put_back_on_unmeasurable_metrics": sorted(
                m for cat, ms in cfg["metric_weights"].items() for m, w in ms.items()
                if float(w) > 0 and m in v2.UNMEASURABLE_METRICS),
        }
    return out


def rankings_differ(months: dict) -> dict:
    """Does any arm rank any stock differently from the baseline, in any month on file?

    This is the measurement the decision rests on: a registered row whose ranking equals the
    baseline's reports a difference of exactly zero *by construction*, which reads as "removing it
    costs nothing" and is in fact "this test was not run".
    """
    base = X.base_config()
    base_scores = X.rank_all(months, base).set_index(["month", "Ticker"]).sort_index()
    out = {"months": int(base_scores.index.get_level_values(0).nunique()),
           "stock_months": int(len(base_scores))}
    for name, info in arms().items():
        cfg = X.apply_transform(base, info["transform"])
        s = X.rank_all(months, cfg).set_index(["month", "Ticker"]).sort_index()
        joined = base_scores[["Composite", "Rank"]].join(
            s[["Composite", "Rank"]], how="outer", lsuffix="_base", rsuffix="_arm")
        comp_diff = (joined["Composite_base"] - joined["Composite_arm"]).abs()
        out[name] = {
            "rows_only_in_one_side": int(joined[["Composite_base", "Composite_arm"]].isna().any(axis=1).sum()),
            "stock_months_with_a_different_composite": int((comp_diff > 1e-12).sum()),
            "max_absolute_composite_difference": float(comp_diff.max()),
            "stock_months_with_a_different_rank": int((joined["Rank_base"] != joined["Rank_arm"]).sum()),
        }
    return out


def record_column(months: dict) -> dict:
    """What the 176 monthly records hold in the column the proposal wants weighted."""
    present, nan_frac, rows = 0, [], 0
    for m, (df, _state) in months.items():
        if METRIC in df.columns:
            present += 1
            nan_frac.append(float(df[METRIC].isna().mean()))
            rows += int(len(df))
    return {
        "months_with_the_column": present,
        "months_total": len(months),
        "all_values_nan_in_every_month": bool(nan_frac and min(nan_frac) == 1.0),
        "min_nan_fraction": min(nan_frac) if nan_frac else None,
        "stock_months": rows,
        "pit_record_class": next((cls for cls, fields in pr.FIELD_CLASSES.items()
                                  for f in fields if "short" in f.lower()), None),
        "short_fields_by_class": {cls: [f for f in fields if "short" in f.lower()]
                                  for cls, fields in pr.FIELD_CLASSES.items()
                                  if any("short" in f.lower() for f in fields)},
    }


def finra_coverage(months: dict) -> dict:
    """Could the second half of the proposal - FINRA days to cover in the metric's place - be judged
    by the engine's own confirmation rule?

    The rule (registry.yaml header): an experiment is CONFIRMED only if its 2012-2019 difference
    clears the bar *and* points the same way as its 2020-2026 difference. FINRA's biweekly
    exchange-listed files begin 2017-12-29
    (research/measurements/2026-10-23-finra-short-interest-probe.py).
    """
    first = pd.Timestamp("2017-12-29")
    reg = yaml.safe_load((ROOT / "research/experiments/registry.yaml").read_text(encoding="utf-8"))
    lo, hi = [pd.Timestamp(d) for d in reg["presample"]["months"]]
    ms = sorted(pd.Timestamp(m) for m in months)
    pre = [m for m in ms if lo <= m <= hi]
    post = [m for m in ms if m > hi]
    return {
        "finra_first_file": str(first.date()),
        "presample_months": len(pre),
        "presample_months_covered": sum(m >= first for m in pre),
        "post_months": len(post),
        "post_months_covered": sum(m >= first for m in post),
        "months_total": len(ms),
        "months_covered": sum(m >= first for m in ms),
    }


def main() -> dict:
    months = X.load_months()
    result = {
        "live": live_weights(),
        "v2_baseline": v2_weights(),
        "arms": arms(),
        "rankings": rankings_differ(months),
        "record_column": record_column(months),
        "finra": finra_coverage(months),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
