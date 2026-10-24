"""The two free data fixes to backtest v2's point-in-time record (2026-10-24).

`research/2026-10-20-free-survivorship-sources.md` section 6, and
`research/2026-10-24-v2-pre-2014-renames.md`.

1. **Pre-2014 renames.** The membership revisions carry no CIK column before 2014-05, so a
   company renamed before then has no registrant number anywhere in the history and
   `Inputs.current_symbol` - which matches by CIK and nothing else - cannot find it.
   `universe_history.PRE_REVISION_CIKS` supplies three, and the tests here pin *which* three and
   why the other five are refused, because the refusals are the rule rather than exceptions to it.

2. **Predecessor registrants.** `facts.parquet` is fetched by today's CIK, so the years before a
   holding-company reorganisation have no filings. `backtest_v2._load_facts` joins a side file;
   the admission bounds live in `research/measurements/2026-10-24-predecessor-facts.py` and are
   tested here on synthetic frames, because the real fetch needs the network.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest_v2 as v2                                                      # noqa: E402
import universe_history as uh                                                 # noqa: E402
import v2_experiments as X                                                    # noqa: E402


def _builder():
    p = ROOT / "research" / "measurements" / "2026-10-24-predecessor-facts.py"
    spec = importlib.util.spec_from_file_location("pred_facts_2026_10_24", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- 1. pre-2014 renames
EXPECTED = {"BFB": 14693, "MHP": 64040, "SAI": 1336920}
# Refused, each for its own reason - see PRE_REVISION_CIKS' comment and the research note.
REFUSED = {"ACE", "TYC", "CMCSK", "PCS", "SAIC"}


def test_the_three_admitted_renames_and_no_others():
    """The constant is a short, closed list. A sixth entry is a data-version change that stales
    every ledger row, so it may not arrive without this test being changed deliberately."""
    assert uh.PRE_REVISION_CIKS == EXPECTED


def test_none_of_the_five_refused_symbols_is_in_the_map():
    """ACE and TYC would take a symbol that was itself a member; CMCSK is a second share class of
    a company already scored; PCS failed its own price-against-filed-float check; SAIC's CIK and
    its price series name two different companies."""
    for sym in REFUSED:
        assert sym not in uh.PRE_REVISION_CIKS, sym


def test_the_revisions_always_win_over_the_constant(tmp_path):
    """``_load_ciks`` may only *supply* a registrant the history lacks. If it could overwrite one,
    a stale constant would silently outrank the point-in-time source."""
    d = tmp_path / "data" / "universe_history"
    d.mkdir(parents=True)
    (d / "ticker_ciks.json").write_text('{"ciks": {"MHP": 999999, "AAPL": 320193}}', encoding="utf-8")
    out = v2._load_ciks(tmp_path)
    assert out["MHP"] == 999999, "the revisions' CIK must win"
    assert out["BFB"] == EXPECTED["BFB"], "a symbol the file lacks must be supplied"
    assert out["AAPL"] == 320193


def test_the_constant_is_still_used_when_the_file_is_missing(tmp_path):
    assert v2._load_ciks(tmp_path) == EXPECTED


def test_each_admitted_symbol_resolves_to_a_different_live_symbol():
    """The point of the fix: BFB -> BF-B, MHP -> SPGI, SAI -> LDOS. Built through the real
    ``current_symbol``, on a base keyed by the live symbols, so the wiring is what is tested."""
    base = {"BF-B": {}, "SPGI": {}, "LDOS": {}, "AAPL": {}}
    ciks = dict(uh.PRE_REVISION_CIKS, **{"BF-B": 14693, "SPGI": 64040, "LDOS": 1336920,
                                         "AAPL": 320193})
    inp = _inputs(base, ciks)
    assert inp.current_symbol("BFB", members={"BFB"}, taken=set()) == "BF-B"
    assert inp.current_symbol("MHP", members={"MHP"}, taken=set()) == "SPGI"
    assert inp.current_symbol("SAI", members={"SAI"}, taken=set()) == "LDOS"


def test_a_rename_is_refused_when_the_live_symbol_is_itself_a_member():
    """The ACE/CB and TYC/JCI shape, which is why those two are not in the map: one current record
    cannot stand for two members, and the symbol's earlier price history is the other company's."""
    inp = _inputs({"CB": {}}, {"ACE": 896159, "CB": 896159})
    assert inp.current_symbol("ACE", members={"ACE", "CB"}, taken=set()) is None
    assert inp.current_symbol("ACE", members={"ACE"}, taken={"CB"}) is None


def _inputs(base: dict, ciks: dict) -> v2.Inputs:
    empty_daily = pd.DataFrame({"ticker": [], "date": pd.to_datetime([]), "close": [],
                                "adj_close": [], "volume": []})
    empty_idx = pd.DataFrame({"ticker": [], "date": pd.to_datetime([]), "close": []})
    splits = pd.DataFrame({"ticker": [], "date": pd.to_datetime([]), "ratio": []})
    facts = pd.DataFrame({"ticker": [], "concept": [], "start": pd.to_datetime([]),
                          "end": pd.to_datetime([]), "filed": pd.to_datetime([]),
                          "form": [], "fp": [], "val": []})
    return v2.Inputs(empty_daily, empty_idx, splits, facts, base, ciks, {})


# --------------------------------------------------------------------------- 2. predecessor facts
def _facts(ticker: str, filed: list[str], cik: int | None = None) -> pd.DataFrame:
    cols = {"ticker": ticker, "concept": "Assets", "start": pd.NaT,
            "end": pd.to_datetime(filed), "filed": pd.to_datetime(filed),
            "form": "10-Q", "fp": "Q1", "val": 1.0}
    df = pd.DataFrame(cols)
    if cik is not None:
        df.insert(1, "cik", cik)
    return df


def test_a_predecessor_row_filed_after_todays_first_filing_is_dropped():
    """The whole safety argument rests on this: ``pit_record`` takes the latest filing, so a kept
    row may only be read in a month where today's registrant had filed nothing at all."""
    b = _builder()
    pred = _facts("XOM", ["2015-05-01", "2026-09-01"], cik=34088)
    today = _facts("XOM", ["2026-08-03"])
    out, verdicts = b.admit(pred, today)
    assert list(out["filed"].dt.strftime("%Y-%m-%d")) == ["2015-05-01"]
    assert verdicts["XOM:34088"]["bound_is"] == "today's first filing"


def test_a_row_filed_on_the_same_day_is_dropped():
    """Alphabet and Google Inc both filed on 2015-10-29. ``<`` not ``<=``, so the successor wins a
    same-day handover - the predecessor must not be able to tie."""
    b = _builder()
    out, _ = b.admit(_facts("GOOGL", ["2015-10-29"], cik=1288776),
                     _facts("GOOGL", ["2015-10-29"]))
    assert out.empty


def test_an_overlapping_predecessor_is_not_refused_for_overlapping():
    """Apache Corp, EIDP and Dow Chemical still file as debt-only subsidiaries of their own
    holding companies. The first draft of the rule refused exactly these three genuine
    reorganisations, so a regression to it must fail here."""
    b = _builder()
    pred = _facts("APA", ["2015-05-01", "2024-11-07"], cik=6769)
    out, verdicts = b.admit(pred, _facts("APA", ["2021-05-07"]))
    assert len(out) == 1, "the pre-2021 row fills a real gap and must be kept"
    assert verdicts["APA:6769"]["overlap_days"] > 0, "recorded, but it decides nothing"


def test_the_tenure_bound_cuts_a_registrant_that_lost_the_symbol_long_ago():
    """IQVIA held ``Q`` for three months in 2017 and filed until 2025; today's holder of ``Q``
    first filed in 2025. Without the tenure bound its 2018-2025 filings would be readable for
    eight years in which it did not hold the symbol."""
    b = _builder()
    pred = _facts("Q", ["2017-09-01", "2019-01-01", "2024-01-01"], cik=1478242)
    pairs = [{"ticker": "Q", "cik": 1478242, "last": "2017-10-31"}]
    out, verdicts = b.admit(pred, _facts("Q", ["2025-11-18"]), pairs)
    assert list(out["filed"].dt.strftime("%Y-%m-%d")) == ["2017-09-01"]
    assert verdicts["Q:1478242"]["bound_is"] == "tenure"


def test_the_tenure_lag_reaches_the_last_period_it_held_the_symbol():
    """A quarter plus the filing deadline: the final period a registrant held the symbol is
    reported after it has lost it, and that report is the figure a point-in-time record wants."""
    b = _builder()
    assert b.TENURE_LAG_DAYS == 120
    pred = _facts("Q", ["2018-01-15"], cik=1478242)      # 76 days after 2017-10-31
    out, _ = b.admit(pred, _facts("Q", ["2030-01-01"]),
                     [{"ticker": "Q", "cik": 1478242, "last": "2017-10-31"}])
    assert len(out) == 1


def test_a_symbol_absorbed_by_todays_registrant_is_cut_to_nothing():
    """Chubb Corp and Johnson Controls Inc: today's registrant has filed since long before, so
    every row falls on the wrong side of the cut and none can reach a scored record."""
    b = _builder()
    out, verdicts = b.admit(_facts("CB", ["2013-02-01", "2015-11-06"], cik=20171),
                            _facts("CB", ["2009-08-07"]))
    assert out.empty
    assert verdicts["CB:20171"]["kept"] == 0


def test_the_output_carries_no_cik_column():
    """It is concatenated straight onto ``facts.parquet``, which has eight columns and no CIK."""
    b = _builder()
    out, _ = b.admit(_facts("XOM", ["2015-05-01"], cik=34088), _facts("XOM", ["2026-08-03"]))
    assert "cik" not in out.columns


# --------------------------------------------------------------------------- 3. the join
def test_load_facts_without_the_side_file_is_not_an_error(tmp_path, monkeypatch):
    d = tmp_path / "data" / "sec" / "pit"
    d.mkdir(parents=True)
    _facts("AAPL", ["2015-01-01"]).to_parquet(d / "facts.parquet")
    out = v2._load_facts(tmp_path)
    assert len(out) == 1


def test_load_facts_concatenates_the_side_file(tmp_path):
    d = tmp_path / "data" / "sec" / "pit"
    d.mkdir(parents=True)
    _facts("AAPL", ["2015-01-01"]).to_parquet(d / "facts.parquet")
    _facts("XOM", ["2013-01-01"]).to_parquet(d / "facts_predecessor.parquet")
    out = v2._load_facts(tmp_path)
    assert sorted(out["ticker"]) == ["AAPL", "XOM"]
    assert list(out.columns) == ["ticker", "concept", "start", "end", "filed", "form", "fp", "val"]


def test_the_side_file_is_a_recorded_data_source():
    """A result that ran with the predecessors joined must not share a fingerprint with one that
    ran without them, or the ledger would mix two different records inside one data version."""
    assert X.DATA_SOURCES["sec_facts_predecessor"] == "data/sec/pit/facts_predecessor.parquet"


def test_the_side_file_is_in_no_production_path():
    """v2 and its research engine only. The same rule as ``lookahead.py``: a point-in-time rebuild
    must never become an input to the live run."""
    allowed = {"backtest_v2.py", "v2_experiments.py", "pit_record.py"}
    for p in ROOT.glob("*.py"):
        if p.name in allowed:
            continue
        assert "facts_predecessor" not in p.read_text(encoding="utf-8"), p.name


@pytest.mark.parametrize("sym,live", [("BFB", "BF-B"), ("MHP", "SPGI"), ("SAI", "LDOS")])
def test_each_admitted_rename_names_a_company_in_todays_universe(sym, live):
    """If a target leaves the index the entry stops helping - it should then be reconsidered, not
    left asserting a resolution that cannot happen."""
    runs = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
    if not runs:
        pytest.skip("no runs/*/00_raw_fetch.parquet in this checkout")
    base = set(pd.read_parquet(runs[-1])["Ticker"])
    assert live in base, f"{sym} resolves to {live}, which is no longer in the universe"
