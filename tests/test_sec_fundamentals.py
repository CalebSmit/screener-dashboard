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


def test_a_partial_refresh_never_replaces_a_complete_cache(tmp_path, monkeypatch):
    import insider_activity as ia
    good = _facts()
    path = tmp_path / "facts.parquet"
    good.to_parquet(path)
    import os
    os.utime(path, (0, 0))                                   # old enough to refresh
    monkeypatch.setattr(sf, "FACTS_PATH", path)
    monkeypatch.setattr(ia, "ticker_map", lambda edgar=None: {f"T{i}": i + 1 for i in range(10)})

    class E:
        requests = 0

        def get(self, url):
            if url.endswith("CIK0000000001.json"):
                class R:
                    @staticmethod
                    def json():
                        return {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
                            {"end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "val": 1}]}}}}}
                return R()
            raise RuntimeError("down")

    out = sf.refresh_companyfacts([f"T{i}" for i in range(10)], edgar=E(), log=lambda *a: None)
    assert len(out) == len(good)                             # 1 of 10 reached: the old cache stands
    assert len(pd.read_parquet(path)) == len(good)


def test_stale_five_years_are_not_published_as_current():
    """BKNG stopped tagging net income after FY2015; its 2011-2015 table must not pass as current."""
    f = _facts()
    old = f[f["ticker"] == "AAA"].copy()
    for c in ("start", "end", "filed"):
        old[c] = old[c] - pd.DateOffset(years=10)
    old["ticker"] = "OLD"
    h = sf.roe_history(["OLD"], TODAY, facts=pd.concat([f, old]))
    assert sf.earnings_variability(h["OLD"]) is None


def test_a_subset_run_never_rewrites_the_cache(tmp_path, monkeypatch):
    path = tmp_path / "facts.parquet"
    _facts().to_parquet(path)
    import os
    os.utime(path, (0, 0))                                  # stale: a full run would refresh it
    monkeypatch.setattr(sf, "FACTS_PATH", path)

    class Boom:
        def get(self, url):
            raise AssertionError("a subset run must not download")

    out = sf.refresh_companyfacts(["AAA"], edgar=Boom(), allow_write=False)
    assert len(out) == len(_facts()) and os.path.getmtime(path) == 0


# ------------------------------------------------- a combined filing under a co-registrant's record
# 2026-11-18: from mid-May the SEC's companyfacts holds a jointly filed 10-Q under one of the other
# registrants on it, so 19 scored companies' records stopped at their first-quarter 10-Q.
def _record(*facts):
    """A companyfacts record: facts are (accession, form, filed, end, value) of us-gaap Assets."""
    return {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"accn": a, "form": form, "filed": filed, "end": end, "val": v} for a, form, filed, end, v in facts]}}}}}


class _Edgar:
    """PARENT (CIK 1) files with SUB (CIK 2) and SUB2 (CIK 3); ALONE (CIK 9) files by itself."""

    def __init__(self, first_registrant=1):
        self.urls = []
        self.first = first_registrant

    def get(self, url):
        self.urls.append(url)
        body = self._body(url)

        class R:
            text = body if isinstance(body, str) else ""

            @staticmethod
            def json():
                return body
        return R()

    def _body(self, url):
        old = [("K", "10-K", "2026-02-10", "2025-12-31", 100), ("Q1", "10-Q", "2026-05-01", "2026-03-31", 110)]
        if url.endswith("companyfacts/CIK0000000001.json") or url.endswith("companyfacts/CIK0000000009.json"):
            return _record(*old)
        if url.endswith("companyfacts/CIK0000000002.json"):
            return _record(("Q2", "10-Q", "2026-08-01", "2026-06-30", 120))
        if url.endswith("companyfacts/CIK0000000003.json"):
            return _record(("Q3", "10-Q", "2026-11-01", "2026-09-30", 130),
                           ("OTHER", "10-Q", "2026-11-02", "2026-09-30", 7))   # the subsidiary's own filing
        if "submissions/CIK0000000001" in url or "submissions/CIK0000000009" in url:
            return {"filings": {"recent": {
                "accessionNumber": ["Q3", "8K", "Q2", "Q1", "K"],
                "form": ["10-Q", "8-K", "10-Q", "10-Q", "10-K"],
                "filingDate": ["2026-11-01", "2026-09-01", "2026-08-01", "2026-05-01", "2026-02-10"]}}}
        if url.endswith("-index-headers.html"):
            if "/data/9/" in url:
                return "<FILER>\nCENTRAL INDEX KEY: 9\n"
            order = [1, 2, 3] if self.first == 1 else [2, 1, 3]
            return "".join(f"&lt;FILER&gt;\nCOMPANY DATA:\nCENTRAL INDEX KEY:\t\t\t{c:010d}\n" for c in order)
        raise AssertionError(url)


def _refresh(tmp_path, monkeypatch, edgar, tickers=("PARENT",)):
    import insider_activity as ia
    monkeypatch.setattr(sf, "FACTS_PATH", tmp_path / "facts.parquet")
    monkeypatch.setattr(ia, "ticker_map", lambda edgar=None: {"PARENT": 1, "ALONE": 9})
    said = []
    out = sf.refresh_companyfacts(list(tickers), edgar=edgar, log=said.append, today=date(2026, 11, 18))
    return out, said


@pytest.mark.parametrize("filed,today,gap", [
    (["2026-02-10", "2026-05-01", "2026-08-01", "2026-11-01"], date(2026, 11, 18), None),
    (["2026-02-10", "2026-05-01"], date(2026, 11, 18), "2026-05-01"),                  # stopped in May
    (["2026-02-10", "2026-05-01", "2026-11-01"], date(2026, 11, 18), "2026-05-01"),    # one quarter out
    (["2025-10-20", "2026-02-27"], date(2026, 3, 1), None),         # third-quarter 10-Q to 10-K: 130 days
    ([], date(2026, 11, 18), None),                                 # nothing recent is not a gap to follow
    (["2021-02-10"], date(2026, 11, 18), None),
])
def test_a_gap_is_more_than_150_days_between_periodic_filings(filed, today, gap):
    assert sf.filing_gap(filed, today) == gap


def test_a_combined_filing_is_read_from_the_co_registrant_that_holds_it(tmp_path, monkeypatch):
    out, said = _refresh(tmp_path, monkeypatch, _Edgar())
    got = {(str(r.end.date()), r.val) for r in out.itertuples()}
    assert got == {("2025-12-31", 100), ("2026-03-31", 110), ("2026-06-30", 120), ("2026-09-30", 130)}
    assert set(out["ticker"]) == {"PARENT"}                 # filed under the listed company's ticker
    assert 7 not in set(out["val"])                         # never the co-registrant's other filings
    assert any("1 companies' filings were under a co-registrant's record (2 filings, 2 facts): PARENT" in s
               for s in said)
    assert not any("still have a gap" in s for s in said)


def test_a_filing_the_listed_company_does_not_lead_is_not_followed(tmp_path, monkeypatch):
    """The facts are the parent's because the parent is the first registrant; where it is not,
    the dimensionless facts may be someone else's."""
    out, said = _refresh(tmp_path, monkeypatch, _Edgar(first_registrant=2))
    assert set(out["val"]) == {100, 110}
    assert any("still have a gap" in s and "PARENT" in s for s in said)


def test_a_company_filing_alone_is_left_as_it_is_and_named(tmp_path, monkeypatch):
    e = _Edgar()
    out, said = _refresh(tmp_path, monkeypatch, e, tickers=("ALONE",))
    assert set(out["val"]) == {100, 110}
    assert any("1 companies still have a gap" in s and "ALONE" in s for s in said)
    assert not any("companyfacts/CIK0000000002" in u for u in e.urls)


def test_a_company_with_no_gap_costs_one_request(tmp_path, monkeypatch):
    class Whole(_Edgar):
        def _body(self, url):
            assert "companyfacts/CIK0000000001" in url, url
            return _record(("K", "10-K", "2026-02-10", "2025-12-31", 100), ("Q1", "10-Q", "2026-05-01", "2026-03-31", 1),
                           ("Q2", "10-Q", "2026-08-01", "2026-06-30", 2), ("Q3", "10-Q", "2026-11-01", "2026-09-30", 3))
    e = Whole()
    out, said = _refresh(tmp_path, monkeypatch, e)
    assert len(e.urls) == 1 and len(out) == 4 and len(said) == 1


def test_a_failed_follow_up_keeps_the_companys_own_facts(tmp_path, monkeypatch):
    class Down(_Edgar):
        def _body(self, url):
            if "submissions" in url:
                raise RuntimeError("down")
            return super()._body(url)
    out, said = _refresh(tmp_path, monkeypatch, Down())
    assert set(out["val"]) == {100, 110}
