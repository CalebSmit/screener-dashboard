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


def test_rank_25_is_in_the_top_25_on_both_dates_and_26_is_not():
    """Claim ``history.run_overview``: 'today's top 25 that were also in the top 25'.

    The fixture above has no name at exactly 25 on either date, so excluding rank 25 at
    the baseline (``<= n`` to ``< n``) passed it (mutation audit 2026-10-20,
    ``plan/claims-mutation-audit.md``). Pin the edge on each date."""
    h = _history()
    h["series"] = {"E": {"r": [25, 9, 25]},      # 25 at the baseline and today: kept
                   "F": {"r": [26, 9, 1]},       # 26 at the baseline: not kept
                   "G": {"r": [1, 9, 26]},       # 26 today: not kept
                   "H": {"r": [24, 9, 2]}}       # inside on both: kept
    assert ro.TOP_N == 25
    assert ro.top_kept(h, "2026-09-09") == 2


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


def test_one_kept_name_reads_singular():
    h = _history()
    h["series"] = {"A": {"r": [1, 2, 3]}, "B": {"r": [40, 40, 40]}}
    assert "1 of today's top 25 was also in the top 25" in " ".join(ro.overview(h, FW)["m1"]["text"])


# ---------------------------------------------------------------------------
# history.market_move - the market and the ranking side by side (2026-11-05)
# ---------------------------------------------------------------------------
SESSIONS = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]


@pytest.mark.parametrize("started,want", [
    ("2026-11-06T02:00:04-06:00", "2026-11-05"),   # the 02:00 data loop: the close before it
    ("2026-11-05T02:00:04-06:00", "2026-11-04"),
    ("2026-11-05T18:30:00-06:00", "2026-11-05"),   # an evening run: that day's close
    ("2026-11-05T15:00:00-06:00", "2026-11-05"),   # 16:00 New York exactly is the close
    ("2026-11-05T14:59:00-06:00", None),           # 15:59 New York: mid-session, priced at no close
    ("2026-11-05T13:21:50-06:00", None),           # the 2026-10-20 case: an owner run at lunchtime
    ("2026-11-08T09:00:00-06:00", "2026-11-06"),   # a Sunday: Friday's close
    ("2026-11-02T08:00:00-06:00", None),           # before the first session in the series
    (None, None), ("not a time", None),
])
def test_a_run_is_priced_at_the_last_close_before_it_started(started, want):
    assert ro.session_of(started, SESSIONS) == want


def test_a_holiday_is_skipped_by_the_index_s_own_calendar():
    # no 2026-11-26 (Thanksgiving) in the index's sessions: a run on the 27th at 02:00 names the 25th
    assert ro.session_of("2026-11-27T02:00:00-06:00", ["2026-11-24", "2026-11-25", "2026-11-27"]) == "2026-11-25"


def test_the_market_move_is_the_ratio_of_the_two_closes():
    closes = {"2026-11-03": 15000.0, "2026-11-04": 15237.3, "2026-11-05": 14572.6}
    m = ro.market_move(closes, "2026-11-05T02:00:04-06:00", "2026-11-06T02:00:04-06:00")
    assert m == {"ret": round(14572.6 / 15237.3 - 1, 5), "from": "2026-11-04", "to": "2026-11-05"}
    # two runs priced at the same close (a re-run before the next session): no move to state
    assert ro.market_move(closes, "2026-11-05T02:00:00-06:00", "2026-11-05T08:00:00-06:00") is None
    # a session the series does not carry: nothing, rather than a neighbour's figure
    assert ro.market_move({"2026-11-04": 1.0}, "2026-11-05T02:00:00-06:00", "2026-11-06T02:00:00-06:00") is None
    assert ro.market_move({}, "a", "b") is None


def test_move_words_print_one_decimal_and_name_the_direction():
    assert ro.move_words(-0.04362) == "fell 4.4%"
    assert ro.move_words(0.01582) == "rose 1.6%"
    assert ro.move_words(0.0004) == "was flat"
    assert ro.move_words(-0.0004) == "was flat"


def test_the_rank_correlation_is_spearman_over_the_stocks_ranked_on_both_dates():
    import pandas as pd
    h = _history()
    # A (1 -> 3), B (30 -> 10), D (5 -> 24); C is unranked at the baseline and left out
    want = pd.Series([1, 30, 5]).corr(pd.Series([3, 10, 24]), method="spearman")
    assert ro.rank_correlation(h, "2026-09-09") == pytest.approx(want)
    assert ro.rank_correlation(h, "2026-01-01") is None
    h["series"] = {"A": {"r": [1, 2, 1]}, "B": {"r": [2, 3, 2]}, "C": {"r": [3, 1, 3]}}
    assert ro.rank_correlation(h, "2026-09-09") == pytest.approx(1.0)


def test_the_typical_figure_is_the_median_of_consecutive_runs():
    import pandas as pd
    h = {"dates": ["d0", "d1", "d2", "d3"], "series": {
        "A": {"r": [1, 1, 3, 1]}, "B": {"r": [2, 2, 1, 2]}, "C": {"r": [3, 3, 2, 3]}, "D": {"r": [4, 4, 4, 4]}}}
    pairs = []
    for i in range(1, 4):
        x = [h["series"][t]["r"][i - 1] for t in "ABCD"]
        y = [h["series"][t]["r"][i] for t in "ABCD"]
        pairs.append(pd.Series(x).corr(pd.Series(y), method="spearman"))
    assert ro.typical_run_to_run(h) == pytest.approx(sorted(pairs)[1])
    h["dates"], h["series"] = ["d0", "d1"], {"A": {"r": [1, 2]}, "B": {"r": [2, 1]}, "C": {"r": [3, 3]}}
    assert ro.typical_run_to_run(h) is None            # one pair is not a typical day


def test_the_sentence_states_the_market_and_the_correlation_it_publishes():
    h = _history()
    h["current_started_at"] = "2026-10-09T02:00:00-05:00"
    h["compare"]["prev"]["started_at"] = "2026-10-08T02:00:00-05:00"
    closes = {"2026-10-06": 100.0, "2026-10-07": 104.0, "2026-10-08": 99.5}
    h = ro.with_market(h, {"source": "^SP500TR", "closes": closes})
    assert h["compare"]["prev"]["market"] == {"ret": round(99.5 / 104.0 - 1, 5), "from": "2026-10-07",
                                              "to": "2026-10-08", "src": "^SP500TR"}
    assert "market" not in h["compare"]["m1"]           # no start time for that run: no figure
    o = ro.overview(h, FW)
    rho = ro.rank_correlation(h, "2026-10-08")
    line = next(t for t in o["prev"]["text"] if "rank correlation" in t)
    assert line == (f"Over the same span the S&P 500 fell 4.3% with dividends included (Oct 7 close to "
                    f"Oct 8 close), and the two rankings have a rank correlation of {rho:.3f}, where 1.000 "
                    f"is the same order.")
    assert o["prev"]["rho"] == round(rho, 3)
    m1 = next(t for t in o["m1"]["text"] if "rank correlation" in t)
    assert m1.startswith("The two rankings have a rank correlation of ")
    assert not advice_terms_in(" ".join(o["prev"]["text"] + o["m1"]["text"]))


def _published():
    f = ROOT / "dashboard_data.js"
    if not f.exists():
        pytest.skip("MISSING DATA: no published dashboard_data.js")
    t = f.read_text(encoding="utf-8")
    return json.loads(t[t.index("{"):].rstrip().rstrip(";"))


def test_the_published_market_move_is_the_runs_own_closes():
    """Every market figure the page carries, against the closes saved with the run the payload
    names (``runs/<run_id>/market_closes.json``) and the start times the history published. Holds
    on every run: a payload with no figure has nothing to check."""
    d = _published()
    H = d.get("history") or {}
    moves = {k: v["market"] for k, v in (H.get("compare") or {}).items() if v and v.get("market")}
    if not moves:
        pytest.skip("the published page states no market move (a run before 2026-11-06, or no closes)")
    run_id = (d.get("kpis") or {}).get("run_id")
    f = ROOT / "runs" / str(run_id) / "market_closes.json"
    if not f.exists():
        pytest.skip(f"MISSING DATA: {f} - the run the payload names is not on this machine")
    closes = json.loads(f.read_text(encoding="utf-8"))["closes"]
    for k, m in moves.items():
        sessions = sorted(closes)
        assert m["from"] == ro.session_of(H["compare"][k]["started_at"], sessions), (k, m)
        assert m["to"] == ro.session_of(H["current_started_at"], sessions), (k, m)
        assert m["ret"] == round(closes[m["to"]] / closes[m["from"]] - 1, 5), (k, m)


def test_the_published_correlation_is_the_histories_own():
    d = _published()
    H = d.get("history") or {}
    ov = H.get("overview") or {}
    checked = 0
    for k, o in ov.items():
        if o.get("rho") is None:
            continue
        assert o["rho"] == round(ro.rank_correlation(H, H["compare"][k]["date"]), 3), k
        assert f"rank correlation of {o['rho']:.3f}," in " ".join(o["text"]), k
        checked += 1
    if "prev" in ov and ov["prev"].get("rho_typical") is not None:
        assert ov["prev"]["rho_typical"] == round(ro.typical_run_to_run(H), 3)
    if not checked:
        pytest.skip("the published page states no rank correlation (a run before 2026-11-06)")

