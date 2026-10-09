"""Trap flags say what their names say (2026-10-09, research/2026-10-09-trap-flags.md).

A *value* trap is a cheap stock that is cheap for a reason (Piotroski 2000 works within the
cheapest stocks); a *growth* trap is a high-growth stock with weak fundamentals (Mohanram 2005).
Before 2026-10-09 neither condition was required: the value flag fired on any broadly weak stock
and the growth flag on low-growth stocks with weak quality and revisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factor_engine import apply_growth_trap_flags, apply_value_trap_flags  # noqa: E402


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _universe(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(n)],
        "valuation_score": rng.uniform(0, 100, n),
        "growth_score": rng.uniform(0, 100, n),
        "quality_score": rng.uniform(0, 100, n),
        "momentum_score": rng.uniform(0, 100, n),
        "revisions_score": rng.uniform(0, 100, n),
    })


def test_a_value_trap_is_always_cheap(cfg):
    df = apply_value_trap_flags(_universe(), cfg)
    cut = df["valuation_score"].quantile(cfg["value_trap_filters"]["valuation_percentile"] / 100)
    assert df["Value_Trap_Flag"].any()
    assert (df.loc[df["Value_Trap_Flag"], "valuation_score"] >= cut).all()


def test_a_weak_stock_that_is_not_cheap_is_not_a_value_trap(cfg):
    df = _universe()
    df.loc[0, ["valuation_score", "quality_score", "momentum_score", "revisions_score"]] = [10, 1, 1, 1]
    df.loc[1, ["valuation_score", "quality_score", "momentum_score", "revisions_score"]] = [99, 1, 1, 50]
    df = apply_value_trap_flags(df, cfg)
    assert not df.loc[0, "Value_Trap_Flag"]           # weak everywhere, but not cheap
    assert df.loc[1, "Value_Trap_Flag"]               # cheap, weak on two of three


def test_a_growth_trap_always_has_high_growth(cfg):
    df = apply_growth_trap_flags(_universe(), cfg)
    cut = df["growth_score"].quantile(cfg["growth_trap_filters"]["growth_ceiling_percentile"] / 100)
    assert df["Growth_Trap_Flag"].any()
    assert (df.loc[df["Growth_Trap_Flag"], "growth_score"] >= cut).all()


def test_growth_trap_needs_one_weakness_beside_high_growth(cfg):
    df = _universe()
    df.loc[0, ["growth_score", "quality_score", "revisions_score"]] = [5, 1, 1]      # low growth: never
    df.loc[1, ["growth_score", "quality_score", "revisions_score"]] = [99, 90, 90]   # high growth, strong: no
    df.loc[2, ["growth_score", "quality_score", "revisions_score"]] = [99, 1, 90]    # high growth, weak quality
    df.loc[3, ["growth_score", "quality_score", "revisions_score"]] = [99, 90, 1]    # high growth, weak revisions
    df = apply_growth_trap_flags(df, cfg)
    assert df.loc[[0, 1, 2, 3], "Growth_Trap_Flag"].tolist() == [False, False, True, True]


def test_missing_valuation_never_flags(cfg):
    df = _universe()
    df.loc[0, ["valuation_score", "quality_score", "momentum_score", "revisions_score"]] = [np.nan, 1, 1, 1]
    assert not apply_value_trap_flags(df, cfg).loc[0, "Value_Trap_Flag"]


def test_the_published_flags_follow_the_rule():
    """On the committed payload, every flagged stock meets its defining condition."""
    import json
    p = ROOT / "dashboard_data.js"
    if not p.exists():
        pytest.skip("no payload")
    t = p.read_text(encoding="utf-8", errors="replace")
    d = json.loads(t[t.find("{"):t.rfind("}") + 1])
    rows = pd.DataFrame(d["table_data"])
    vt = rows["Value_Trap_Flag"].fillna(False).astype(bool)
    gt = rows["Growth_Trap_Flag"].fillna(False).astype(bool)
    if vt.sum() > 0.15 * len(rows):
        pytest.skip("payload predates the 2026-10-09 trap rule (value flags on ~24% of stocks)")
    # the engine's cut is a quantile of the same scores; the payload rounds them, hence the slack
    v_cut, g_cut = rows["valuation_score"].quantile(0.70), rows["growth_score"].quantile(0.70)
    assert (rows.loc[vt, "valuation_score"] >= v_cut - 0.05).all()
    assert (rows.loc[gt, "growth_score"] >= g_cut - 0.05).all()
