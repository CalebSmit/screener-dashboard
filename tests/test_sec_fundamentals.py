"""Earnings variability from the SEC's XBRL frames (``sec_fundamentals``), a Quality candidate.

No network: frames are written to a temporary cache and the ticker map is stubbed.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sec_fundamentals as sf  # noqa: E402

TODAY = date(2026, 10, 9)


def test_the_five_years_are_the_last_complete_calendar_years():
    assert sf.years_for(TODAY) == [2021, 2022, 2023, 2024, 2025]
    assert sf.years_for(date(2026, 2, 1)) == [2020, 2021, 2022, 2023, 2024]   # 10-Ks not all in yet


class _FakeEdgar:
    requests = 0

    class s:  # noqa: N801 - mimics requests.Session on Edgar
        @staticmethod
        def get(*a, **k):
            raise AssertionError("no network in tests")


@pytest.fixture
def frames(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "FRAMES_DIR", tmp_path)
    import insider_activity as ia
    monkeypatch.setattr(ia, "ticker_map", lambda edgar=None: {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4})

    def put(concept, period, data):
        (tmp_path / f"{concept}_{period}.json").write_text(json.dumps(data))

    for i, y in enumerate(range(2021, 2026)):
        # AAA: steady 10% ROE except one year; BBB reports under the fallback tag ProfitLoss;
        # CCC has negative equity in one year; DDD is missing a year entirely.
        put("NetIncomeLoss", f"CY{y}", {"1": [10 + (5 if y == 2023 else 0), f"{y}-12-31"],
                                         "3": [5, f"{y}-12-31"],
                                         **({"4": [8, f"{y}-12-31"]} if y != 2022 else {})})
        put("ProfitLoss", f"CY{y}", {"2": [20, f"{y}-06-30"], "1": [999, f"{y}-12-31"]})
        put("NetIncomeLossAvailableToCommonStockholdersBasic", f"CY{y}", {})
        put("StockholdersEquity", f"CY{y}Q4I", {"1": [100, f"{y}-12-31"], "2": [200 + 10 * i, f"{y}-12-31"],
                                                 "3": [-50 if y == 2024 else 50, f"{y}-12-31"], "4": [80, f"{y}-12-31"]})
        put("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", f"CY{y}Q4I", {})
    return tmp_path


def test_roe_rows_use_the_first_concept_with_a_value(frames):
    h = sf.roe_history(["AAA", "BBB", "CCC", "DDD", "ZZZ"], TODAY, edgar=_FakeEdgar())
    assert "ZZZ" not in h                                   # not in the SEC ticker map
    assert [r[3] for r in h["AAA"]] == [0.1, 0.1, 0.15, 0.1, 0.1]   # NetIncomeLoss, not ProfitLoss's 999
    assert h["BBB"][0] == [2021, 20, 200, 0.1]              # fell back to ProfitLoss
    assert h["CCC"][3][3] is None                           # equity not positive -> no ROE
    assert h["DDD"][1][1] is None and h["DDD"][1][3] is None


def test_earnings_variability_needs_all_five_years(frames):
    h = sf.roe_history(["AAA", "CCC", "DDD"], TODAY, edgar=_FakeEdgar())
    assert sf.earnings_variability(h["AAA"]) == pytest.approx(statistics.stdev([0.1, 0.1, 0.15, 0.1, 0.1]))
    assert sf.earnings_variability(h["CCC"]) is None        # one year of negative equity
    assert sf.earnings_variability(h["DDD"]) is None        # one year not reported


def test_no_sec_identity_means_no_requests(monkeypatch):
    import insider_activity as ia
    monkeypatch.setattr(ia, "user_agent", lambda: None)
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
