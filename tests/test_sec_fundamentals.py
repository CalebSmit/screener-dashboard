"""Earnings variability from the SEC's XBRL frames (``sec_fundamentals``), a Quality candidate.

No network: frames are written to a temporary cache and the ticker map is stubbed.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sec_fundamentals as sf  # noqa: E402

TODAY = date(2026, 10, 9)


def test_the_five_years_are_the_last_complete_calendar_years():
    assert sf.years_for(TODAY) == [2021, 2022, 2023, 2024, 2025]
    assert sf.years_for(date(2026, 2, 1)) == [2020, 2021, 2022, 2023, 2024]   # 10-Ks not all in yet


def _facts():
    """10-K facts: AAA steady; BBB switches net-income tag; CCC negative equity one year; DDD a gap."""
    rows = []
    for i, y in enumerate(range(2021, 2026)):
        end = f"{y}-12-31"
        start = f"{y}-01-01"
        rows.append(("AAA", "NetIncomeLoss", start, end, f"{y + 1}-02-15", "10-K", 10.0 + (5 if y == 2023 else 0)))
        rows.append(("AAA", "StockholdersEquity", None, end, f"{y + 1}-02-15", "10-K", 100.0))
        tag = "NetIncomeLoss" if y < 2023 else "ProfitLoss"
        rows.append(("BBB", tag, start, end, f"{y + 1}-02-15", "10-K", 20.0))
        rows.append(("BBB", "StockholdersEquity", None, end, f"{y + 1}-02-15", "10-K", 200.0 + 10 * i))
        rows.append(("CCC", "NetIncomeLoss", start, end, f"{y + 1}-02-15", "10-K", 5.0))
        rows.append(("CCC", "StockholdersEquity", None, end, f"{y + 1}-02-15", "10-K", -50.0 if y == 2024 else 50.0))
        if y != 2022:
            rows.append(("DDD", "NetIncomeLoss", start, end, f"{y + 1}-02-15", "10-K", 8.0))
            rows.append(("DDD", "StockholdersEquity", None, end, f"{y + 1}-02-15", "10-K", 80.0))
    # a quarterly fact and a restatement: neither may disturb the annual series
    rows.append(("AAA", "NetIncomeLoss", "2025-01-01", "2025-03-31", "2025-05-01", "10-Q", 3.0))
    rows.append(("AAA", "NetIncomeLoss", "2021-01-01", "2021-12-31", "2023-02-15", "10-K", 11.0))
    df = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "form", "val"])
    df["fp"] = None
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def test_roe_rows_pair_income_with_equity_at_the_same_fiscal_year_end():
    h = sf.roe_history(["AAA", "BBB", "CCC", "DDD", "ZZZ"], TODAY, facts=_facts())
    assert "ZZZ" not in h
    assert [r[3] for r in h["AAA"]] == [0.11, 0.1, 0.15, 0.1, 0.1]       # the 2021 restatement wins
    assert all(r[0].endswith("-12-31") for r in h["AAA"])
    assert len(h["BBB"]) == 5                                          # a tag switch keeps every year
    assert h["CCC"][3][3] is None                                      # equity not positive -> no ROE


def test_earnings_variability_needs_five_consecutive_years():
    h = sf.roe_history(["AAA", "CCC", "DDD"], TODAY, facts=_facts())
    assert sf.earnings_variability(h["AAA"]) == pytest.approx(statistics.stdev([0.11, 0.1, 0.15, 0.1, 0.1]))
    assert sf.earnings_variability(h["CCC"]) is None                   # one year of negative equity
    assert sf.earnings_variability(h["DDD"]) is None                   # 2022 missing: not five consecutive


def test_no_sec_identity_and_no_cache_means_no_rows(monkeypatch, tmp_path):
    import insider_activity as ia
    monkeypatch.setattr(ia, "user_agent", lambda: None)
    monkeypatch.setattr(sf, "FACTS_PATH", tmp_path / "missing.parquet")
    assert sf.roe_history(["AAA"], TODAY) == {}


def test_it_is_a_weight_zero_candidate_and_lower_is_better():
    import yaml

    import factor_engine as fe
    import improvement_engine as ie
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["metric_weights"]["quality"]["earnings_variability"] == 0
    assert "earnings_variability" in fe.CAT_METRICS["quality"]
    assert fe.METRIC_DIR["earnings_variability"] is False
    assert "earnings_variability" in ie.CANDIDATE_METRICS


def test_the_published_roe_rows_rebuild_the_metric():
    """On the committed payload: every stock's earnings variability is the sample standard
    deviation of the five ROEs published beside it."""
    p = ROOT / "dashboard_data.js"
    t = p.read_text(encoding="utf-8", errors="replace")
    d = json.loads(t[t.find("{"):t.rfind("}") + 1])
    rows = [s for s in d["stock_detail"].values() if s.get("roe5")]
    if not rows:
        pytest.skip("payload predates earnings variability")
    n = 0
    for s in rows:
        pub = s["raw"].get("earnings_variability")
        calc = sf.earnings_variability(s["roe5"])
        assert (pub is None) == (calc is None)
        if pub is not None:
            assert pub == pytest.approx(calc, rel=1e-9)
            n += 1
    assert n > 300
