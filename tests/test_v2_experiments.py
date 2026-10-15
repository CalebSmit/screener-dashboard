"""The research engine (``v2_experiments.py``): its rules are what make a result worth showing.

Pinned: the discovery and held-out periods are fixed in code and do not overlap; an experiment
must be registered before it can run; every registered transform builds the configuration it
says; a comparison with the baseline is month by month on the same months; the verdict needs the
held-out months to agree with discovery and reports the multiple-testing bar honestly; every run is
appended to the ledger; and no production module imports the engine.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v2_experiments as X  # noqa: E402


def test_the_held_out_months_are_fixed_and_never_overlap_discovery():
    assert X.DISCOVERY == ("2020-01-31", "2023-06-30")
    assert X.HOLDOUT == ("2023-07-31", "2026-08-31")
    assert X.DISCOVERY[1] < X.HOLDOUT[0]
    assert X.HLZ_T == 3.0


def test_every_registered_experiment_has_a_prediction_and_a_source_and_builds():
    reg = X.load_registry()
    ids = [e["id"] for e in reg]
    assert len(ids) == len(set(ids)) and "baseline" in ids
    base = X.base_config()
    for e in reg:
        assert e["hypothesis"] and e["source"] and e["prediction"], e["id"]
        cfg = X.apply_transform(base, e["transform"])
        assert sum(cfg["factor_weights"].values()) > 0, e["id"]


def test_the_transforms_do_what_they_say():
    base = X.base_config()
    only = X.apply_transform(base, {"kind": "only_category", "category": "quality"})
    assert {k for k, v in only["factor_weights"].items() if v} == {"quality"}
    drop = X.apply_transform(base, {"kind": "drop_category", "category": "growth"})
    assert drop["factor_weights"]["growth"] == 0 and drop["factor_weights"]["valuation"] == base["factor_weights"]["valuation"]
    eq = X.apply_transform(base, {"kind": "equal_categories"})
    assert {v for v in eq["factor_weights"].values() if v} == {1}
    mf = X.apply_transform(base, {"kind": "metrics", "metrics": {"valuation": {"ev_ebitda": 50}, "quality": {"roic": 50}}})
    assert {k for k, v in mf["factor_weights"].items() if v} == {"valuation", "quality"}
    assert [m for m, w in mf["metric_weights"]["valuation"].items() if w] == ["ev_ebitda"]
    assert [m for m, w in mf["metric_weights"]["quality"].items() if w] == ["roic"]
    sw = X.apply_transform(base, {"kind": "config", "set": {"factor_neutralization.enabled": True}})
    assert sw["factor_neutralization"]["enabled"] is True
    assert base["factor_weights"]["growth"] > 0                     # the baseline itself is untouched
    with pytest.raises(ValueError):
        X.apply_transform(base, {"kind": "made-up"})


def test_an_unregistered_experiment_cannot_be_run():
    with pytest.raises(SystemExit, match="register before running"):
        X.run(["not-in-the-registry"])


def _pm(ic, months):
    return pd.DataFrame({"ic": ic, "top_minus_universe": [x / 10 for x in ic], "top": 0.01,
                         "universe": 0.0, "n": 400}, index=months)


def test_the_comparison_is_paired_month_by_month():
    months = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-31", "2026-08-31", freq="ME")]
    assert len(months) == 80
    rng = np.random.default_rng(1)
    base_ic = rng.normal(0, 0.12, 80)
    exp = _pm(base_ic + 0.05, months)                   # better by exactly 0.05 every month
    s = X.summarise(exp, _pm(base_ic, months))
    for period in ("discovery", "holdout"):
        d = s[period]["ic_vs_baseline"]
        assert d["mean"] == pytest.approx(0.05) and d["se"] == pytest.approx(0.0, abs=1e-9)
    assert s["discovery"]["ic"]["n"] + s["holdout"]["ic"]["n"] == s["all"]["ic"]["n"]


def test_the_verdict_needs_the_held_out_months_to_agree():
    def res(d_mean, d_p, h_mean, h_p, h_t):
        return {"summary": {"discovery": {"ic_vs_baseline": {"mean": d_mean, "p": d_p}},
                            "holdout": {"ic_vs_baseline": {"mean": h_mean, "p": h_p, "t": h_t}}}}
    e = {"id": "x"}
    assert X.verdict(e, res(0.03, 0.01, 0.002, 0.6, 0.5), 24) == "looked different in discovery; did not hold out"
    assert X.verdict(e, res(0.0, 0.7, 0.03, 0.01, 2.6), 24) == "different only in the held-out months - not confirmed"
    assert X.verdict(e, res(0.03, 0.01, 0.03, 0.01, 2.6), 24).endswith("but not the multiple-testing bar")
    assert X.verdict(e, res(0.03, 0.001, 0.04, 0.0001, 3.9), 24).endswith("clears the multiple-testing bar")
    assert X.verdict(e, res(0.0, 0.8, 0.0, 0.9, 0.1), 24) == "not distinguishable from the baseline"
    assert X.verdict(e, {"summary": {"holdout": {}}}, 24) == "baseline"


def test_no_production_module_imports_it():
    pat = re.compile(r"^\s*(import v2_experiments|from v2_experiments)\b", re.M)
    for p in ROOT.glob("*.py"):
        if p.name in ("v2_experiments.py", "conftest.py"):
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), p.name


def test_a_stock_with_no_composite_is_left_out_of_the_month_not_ranked(monkeypatch):
    """2026-10-15: a narrow experiment (one category, one metric) can leave a stock with no composite;
    the first full run crashed casting its NaN rank to an integer. ``rank_month`` now drops it."""
    import backtest_v2 as v2
    fe = v2.fe
    df = pd.DataFrame({"Ticker": ["A", "B", "C"], "Sector": ["X", "X", "X"]})
    monkeypatch.setattr(fe, "applicable_coverage", lambda d, c: (pd.Series(9, index=d.index), pd.Series(9, index=d.index)))
    for name in ("apply_universe_filters", "apply_percentile_transform", "compute_category_scores"):
        monkeypatch.setattr(fe, name, lambda d, c: d)
    monkeypatch.setattr(fe, "compute_sector_percentiles", lambda d: d)
    monkeypatch.setattr(fe, "compute_composite", lambda d, c: d.assign(Composite=[70.0, np.nan, 40.0]))
    cfg = {"data_quality": {"min_data_coverage_pct": 50}, "factor_weights": {}}
    out = v2.rank_month(df, cfg, {})
    assert list(out["Ticker"]) == ["A", "C"] and list(out["Rank"]) == [1, 2]
