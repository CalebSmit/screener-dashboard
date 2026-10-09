"""The run-level overview in What Changed (``run_overview.py``; CLAUDE.md priority 4's residual).

Every number in the sentences is checked against the inputs it is computed from, and the text
never advises.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_overview as ro  # noqa: E402
from stock_summary import advice_terms_in  # noqa: E402


def _history():
    return {
        "available": True,
        "dates": ["2026-09-09", "2026-10-08", "2026-10-09"],
        "current_date": "2026-10-09",
        "noise": {"material_threshold": 36},
        "compare": {"prev": {"date": "2026-10-08", "gap_days": 1}, "m1": {"date": "2026-09-09", "gap_days": 30}},
        "movers": {"prev": {"n_up": 0, "n_down": 1}, "m1": {"n_up": 3, "n_down": 2}},
        "delta": {
            "A": {"m1": {"cat": {"momentum": 10.0, "valuation": -5.0}}, "prev": {"cat": {"momentum": 1.0}}},
            "B": {"m1": {"cat": {"momentum": -10.0, "quality": 2.0}}},
            "C": {"new": True},
        },
        "series": {
            "A": {"r": [1, 2, 3]}, "B": {"r": [30, 20, 10]}, "C": {"r": [None, None, 26]},
            "D": {"r": [5, 26, 24]},
        },
    }


FW = {"valuation": 20, "quality": 20, "growth": 10, "momentum": 15, "risk": 10, "revisions": 15, "size": 5, "investment": 5}


def test_movers_are_the_panels_own_counts():
    o = ro.overview(_history(), FW)
    assert o["m1"]["n_up"] == 3 and o["m1"]["n_down"] == 2 and o["m1"]["threshold"] == 36
    assert "3 stocks moved up the ranking and 2 moved down by 36 places or more" in o["m1"]["text"][0]
    assert "Since Sep 9 (30 days)" in o["m1"]["text"][0]
    assert "Since Oct 8 (1 day)" in o["prev"]["text"][0]


def test_category_shares_weight_absolute_changes():
    s = ro.category_shares(_history(), "m1", FW)
    # |15*10| + |15*-10| = 300 momentum; |20*-5| = 100 valuation; |20*2| = 40 quality
    assert s["momentum"] == pytest.approx(300 / 440)
    assert s["valuation"] == pytest.approx(100 / 440)
    assert s["quality"] == pytest.approx(40 / 440)
    assert sum(s.values()) == pytest.approx(1.0)
    text = ro.overview(_history(), FW)["m1"]["text"][1]
    assert "Momentum accounted for 68%" in text and "Valuation, Quality and Growth together 32%" in text


def test_top_kept_counts_names_in_both_top_lists():
    h = _history()
    assert ro.top_kept(h, "2026-09-09") == 2           # A (1 -> 3) and D (5 -> 24); B was 30
    assert ro.top_kept(h, "2026-10-08") == 2           # A and B (20 -> 10); D was 26
    assert "2 of today's top 25 were also in the top 25 on Sep 9." in ro.overview(h, FW)["m1"]["text"]


def test_a_quiet_run_says_so():
    h = _history()
    h["movers"]["prev"] = {"n_up": 0, "n_down": 0}
    assert "no stock moved 36 places or more" in ro.overview(h, FW)["prev"]["text"][0]


def test_no_history_no_overview():
    assert ro.overview({"available": False}, FW) == {}


def test_the_overview_never_advises():
    o = ro.overview(_history(), FW)
    assert advice_terms_in(json.dumps(o)) == []


def test_the_published_overview_matches_the_published_history():
    """On the committed payload, the overview is what the module computes from its own history."""
    p = ROOT / "dashboard_data.js"
    if not p.exists():
        pytest.skip("no payload")
    t = p.read_text(encoding="utf-8", errors="replace")
    d = json.loads(t[t.find("{"):t.rfind("}") + 1])
    h = d.get("history") or {}
    if "overview" not in h:
        pytest.skip("payload predates the run overview")
    fresh = ro.overview({k: v for k, v in h.items() if k != "overview"}, d["weights"]["factor_weights"])
    assert fresh == h["overview"]
    assert advice_terms_in(json.dumps(h["overview"])) == []
