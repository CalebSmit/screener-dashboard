"""Valuation against the stock's own history (``valuation_history.py``) - display-only context.

The construction rules the page states are each checked here on hand-built filings: the
trailing-twelve-month arithmetic, "only what had been filed by then", the split adjustment, and
the market-value self-check that decides whether a stock is shown at all.
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

import valuation_history as vh  # noqa: E402
from stock_summary import advice_terms_in  # noqa: E402


def _facts(rows):
    df = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "form", "fp", "val"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def _d(s):
    return vh._day(pd.Timestamp(s))


NI = "NetIncomeLoss"
SH = "WeightedAverageNumberOfDilutedSharesOutstanding"


def test_ttm_is_last_year_plus_this_ytd_minus_last_ytd():
    f = _facts([
        ("X", NI, "2023-01-01", "2023-12-31", "2024-02-15", "10-K", "FY", 100.0),
        ("X", NI, "2023-01-01", "2023-03-31", "2023-05-01", "10-Q", "Q1", 20.0),
        ("X", NI, "2024-01-01", "2024-03-31", "2024-05-01", "10-Q", "Q1", 30.0),
    ])
    s = vh.ttm_series(f, vh.NI)
    i = list(s["end"]).index(_d("2024-03-31"))
    assert s["val"][i] == pytest.approx(110.0)                 # 100 + 30 - 20
    assert s["avail"][i] == _d("2024-05-01")                   # the day its last component was filed


def test_only_what_had_been_filed_by_then():
    f = _facts([
        ("X", NI, "2023-01-01", "2023-12-31", "2024-02-15", "10-K", "FY", 100.0),
        ("X", NI, "2022-01-01", "2022-12-31", "2023-02-15", "10-K", "FY", 80.0),
    ])
    s = vh.ttm_series(f, vh.NI)
    assert vh._as_of(s, _d("2024-01-31")) == 80.0              # FY2023 not filed until Feb 15
    assert vh._as_of(s, _d("2024-02-29")) == 100.0


def test_first_filed_value_is_kept_over_a_restatement():
    f = _facts([
        ("X", NI, "2023-01-01", "2023-12-31", "2024-02-15", "10-K", "FY", 100.0),
        ("X", NI, "2023-01-01", "2023-12-31", "2025-02-15", "10-K", "FY", 70.0),   # restated a year later
    ])
    assert vh.ttm_series(f, vh.NI)["val"].tolist() == [100.0]


def test_a_stale_figure_is_not_used():
    f = _facts([("X", NI, "2020-01-01", "2020-12-31", "2021-02-15", "10-K", "FY", 100.0)])
    assert vh._as_of(vh.ttm_series(f, vh.NI), _d("2023-06-30")) is None


def test_shares_filed_before_a_split_are_scaled_to_todays_units():
    f = _facts([
        ("X", SH, "2024-01-01", "2024-03-31", "2024-05-01", "10-Q", "Q1", 100.0),
        ("X", SH, "2024-07-01", "2024-09-30", "2024-11-01", "10-Q", "Q3", 1000.0),
    ])
    splits = [["2024-06-10", 10.0]]
    sh = vh.share_series(f)
    assert vh._shares_as_of(sh, _d("2024-05-31"), splits) == pytest.approx(1000.0)   # 100 pre-split x 10
    assert vh._shares_as_of(sh, _d("2024-11-30"), splits) == pytest.approx(1000.0)   # already post-split


def _history(price=10.0, mcap=1000.0):
    rows = []
    for y in range(2019, 2027):
        rows.append(("X", NI, f"{y - 1}-01-01", f"{y - 1}-12-31", f"{y}-02-15", "10-K", "FY", 50.0))
        rows.append(("X", SH, f"{y - 1}-01-01", f"{y - 1}-12-31", f"{y}-02-15", "10-K", "FY", 100.0))
    f = _facts(rows)
    idx = pd.date_range("2021-09-01", periods=60, freq="MS")
    closes = pd.Series(np.linspace(5, 20, 60), index=idx)
    return f, closes, price, mcap


def test_a_stock_whose_shares_do_not_reproduce_its_market_value_is_not_shown():
    f, closes, px, _ = _history()
    today = pd.Timestamp("2026-10-09")
    assert vh.for_ticker(f, closes, [], px, 1000.0, today) is not None      # 10 x 100 = 1000
    assert vh.for_ticker(f, closes, [], px, 1200.0, today) is None          # 17% apart
    assert vh.for_ticker(f, closes, [], px, 1100.0, today) is not None      # 9% apart


def test_the_percentile_is_against_the_stocks_own_month_ends():
    f, closes, px, mcap = _history()
    b = vh.for_ticker(f, closes, [], px, mcap, pd.Timestamp("2026-10-09"), fcf=False)
    ey = b["ey"]
    assert ey["now"] == pytest.approx(50 / 1000)
    hist = [x / 10000 for x in ey["s"] if x is not None]
    # price rose from 5 to 20 against flat earnings, so the yield fell; 10 sits in the middle
    assert ey["pct"] == pytest.approx(100 * np.mean(np.array(hist) < 0.05), abs=2)
    assert ey["lo"] <= ey["med"] <= ey["hi"] and ey["n"] == 60 and len(ey["s"]) == 60
    assert "fy" not in b


def test_too_little_history_shows_nothing():
    f, closes, px, mcap = _history()
    assert vh.for_ticker(f, closes.iloc[-20:], [], px, mcap, pd.Timestamp("2026-10-09")) is None


def test_the_block_rides_the_context_log_as_scalars_only():
    """The context log keeps the two percentiles (each signal's out-of-sample record) and drops
    the JSON block, like the weekly series and insider rows."""
    src = (ROOT / "context_signals.py").read_text(encoding="utf-8")
    assert '"_ctx_valhist"' in src
    ev = (ROOT / "context_eval.py").read_text(encoding="utf-8")
    assert "_ctx_vh_ey_pct" in ev and "_ctx_vh_fy_pct" in ev


def test_it_is_context_and_never_reaches_a_score():
    for name in ("factor_engine.py", "calc_trace.py", "metric_lineage.py", "presets.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "valuation_history" not in src and "_ctx_valhist" not in src and "_ctx_vh_" not in src, name
    # and it is not the backtest's point-in-time layer, which production may not import
    assert not re.search(r"^\s*(import|from)\s+pit_fundamentals\b",
                         (ROOT / "valuation_history.py").read_text(encoding="utf-8"), re.M)


def test_the_card_never_advises():
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    card = src[src.index("function vhChartSvg"):src.index("// ---- the teaser under")]
    text = re.sub(r"<[^>]+>", " ", " ".join(re.findall(r"'([^']*)'", card)))
    assert advice_terms_in(text) == []


def test_the_published_blocks_are_internally_consistent():
    p = ROOT / "dashboard_context.js"
    if not p.exists():
        pytest.skip("no context file")
    t = p.read_text(encoding="utf-8")
    d = json.loads(t[t.find("{"):t.rfind("}") + 1])
    blocks = {k: c["vh"] for k, c in (d.get("ctx") or {}).items() if isinstance(c, dict) and "vh" in c}
    if not blocks:
        pytest.skip("context file predates the valuation history")
    for tk, b in blocks.items():
        assert abs(b["chk"]) <= vh.MCAP_TOLERANCE, tk
        for k in ("ey", "fy"):
            if k in b:
                y = b[k]
                assert 0 <= y["pct"] <= 100 and y["lo"] <= y["med"] <= y["hi"], tk
                assert y["n"] >= vh.MIN_MONTHS and len(y["s"]) <= vh.MONTHS, tk
                # the series is stored in whole basis points, so a month within half a basis point
                # of today may round to either side of it
                hist = np.array([x / 10000 for x in y["s"] if x is not None])
                lo_b = 100 * float((hist < y["now"] - 0.00005).mean())
                hi_b = 100 * float((hist <= y["now"] + 0.00005).mean())
                assert lo_b - 0.1 <= y["pct"] <= hi_b + 0.1, tk


def test_a_filing_during_the_month_counts_at_that_month_end():
    """Monthly bars are labelled with the 1st; the close is the month's last day."""
    rows = [("X", SH, "2021-01-01", "2021-12-31", "2022-02-01", "10-K", "FY", 100.0)]
    for y in range(2019, 2027):
        rows.append(("X", NI, f"{y - 1}-01-01", f"{y - 1}-12-31", f"{y}-02-15", "10-K", "FY", float(y)))
        rows.append(("X", SH, f"{y - 1}-01-01", f"{y - 1}-12-31", f"{y}-02-15", "10-K", "FY", 100.0))
    f = _facts(rows)
    idx = pd.date_range("2021-09-01", periods=60, freq="MS")
    closes = pd.Series(10.0, index=idx)
    b = vh.for_ticker(f, closes, [], 10.0, 1000.0, pd.Timestamp("2026-10-09"), fcf=False)
    s = dict(zip(idx, b["ey"]["s"]))
    # FY2023 (value 2024) was filed 2024-02-15: it is in the February 2024 month-end, not January's
    assert s[pd.Timestamp("2024-01-01")] == round(2023 / 1000 * 10000)
    assert s[pd.Timestamp("2024-02-01")] == round(2024 / 1000 * 10000)
