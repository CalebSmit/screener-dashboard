"""Investor profiles on the rankings table (plan/investor-profiles.md, CLAUDE.md priority 6).

One definition of each named weighting (``presets.py``); the dashboard publishes the ranking the
engine computes under it, with the run's own volatility-regime adjustment - exactly what
``run_screener.py --preset <name>`` would publish from the same scores - and the page only
switches between them.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
from presets import PRESETS  # noqa: E402


def test_the_regime_rule_is_unchanged_by_the_refactor():
    base = PRESETS["balanced"]["factor_weights"]
    low = fe.apply_momentum_regime(base, "LOW VOL")
    assert low["momentum"] == round(13 * 1.15, 2) and low["valuation"] == round(22 - 13 * 0.15, 2)
    high = fe.apply_momentum_regime(base, "HIGH VOL")
    freed = 13 * 0.30
    assert high["momentum"] == round(13 * 0.70, 2)
    assert high["quality"] == round(22 + freed / 2, 2) and high["valuation"] == round(22 + freed / 2, 2)
    assert fe.apply_momentum_regime(base, "NORMAL") == base
    for r in ("LOW VOL", "HIGH VOL", "NORMAL"):
        assert fe.infer_momentum_regime(base, fe.apply_momentum_regime(base, r)) == r


def test_the_published_run_is_the_balanced_preset_under_its_regime():
    """The run's own weights are the Balanced preset with the regime applied - the anchor that
    makes the other profiles comparable with it."""
    w = json.loads((max((d for d in (ROOT / "runs").iterdir() if (d / "effective_weights.json").exists()),
                        key=lambda d: (d / "effective_weights.json").stat().st_mtime) / "effective_weights.json").read_text())
    if not w.get("base_factor_weights"):
        pytest.skip("run predates base weights")
    assert w["base_factor_weights"] == PRESETS["balanced"]["factor_weights"]
    regime = fe.infer_momentum_regime(w["base_factor_weights"], w["factor_weights"])
    assert fe.apply_momentum_regime(w["base_factor_weights"], regime) == pytest.approx(w["factor_weights"])


def _payload():
    t = (ROOT / "dashboard_data.js").read_text(encoding="utf-8", errors="replace")
    return json.loads(t[t.find("{"):t.rfind("}") + 1])


def test_each_profile_is_what_the_preset_would_publish():
    d = _payload()
    P = d.get("profiles")
    if not P:
        pytest.skip("payload predates investor profiles")
    import yaml
    run = max((x for x in (ROOT / "runs").iterdir() if (x / "05_final_scored.parquet").exists()),
              key=lambda x: (x / "05_final_scored.parquet").stat().st_mtime)
    df = pd.read_parquet(run / "05_final_scored.parquet")
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) if (run / "config.yaml").exists() \
        else yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for item in P["list"]:
        key = item["key"]
        fw = fe.apply_momentum_regime(PRESETS[key]["factor_weights"], P["regime"])
        assert item["weights"] == pytest.approx(fw)
        if item.get("published"):
            continue
        c = copy.deepcopy(cfg)
        c["factor_weights"] = fw
        comp = fe.compute_composite(df.copy(), c).set_index("Ticker")["Composite"]
        pub = P["c"][key]
        diffs = [abs(round(float(comp[t]), 2) - v[0]) for t, v in pub.items()]
        assert max(diffs) < 1e-9, key
        ranks = comp.rank(ascending=False, method="min")
        assert all(int(ranks[t]) == v[1] for t, v in pub.items()), key


def test_profiles_never_overwrite_the_published_columns():
    d = _payload()
    if not d.get("profiles"):
        pytest.skip("payload predates investor profiles")
    assert "balanced" not in d["profiles"]["c"]          # the published ranking is the table itself
    assert all(r.get("Composite") is not None for r in d["table_data"][:10])
