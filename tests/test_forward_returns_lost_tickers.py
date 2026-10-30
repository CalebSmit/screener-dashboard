"""Forward returns keep the stocks the price source stopped serving (2026-10-29).

`compute_forward_returns` skipped any snapshot ticker missing from the downloaded prices. The free
source stops serving a company that is renamed or taken over, so those stocks fell out of the
evidence panel on every date that held them: measured that day, 1-3 stocks on every `1m` date from
2026-08-10 to 2026-09-28 (WBD, PSKY, BK, CTRA, EA, HOLX, AVB, EQR, SATS) and up to 8 on a `3m`
date. Four had been renamed (BK -> BNY, MMC -> MRSH, EQR -> VMRK, SATS -> ECHO); the rest no
longer trade. The track record had closed the same gap on 2026-10-15 (`track_record.fill_unpriced`);
the engine now applies the same rules, so the two records cannot disagree about one stock.
"""
import json

import numpy as np
import pandas as pd
import pytest

import improvement_engine as ie

RUN = "2026-03-02"            # a Monday
LATER = "2026-04-13"          # 42 days on: 1w and 1m are due


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    imp = tmp_path / "improvement"
    for d in ("snapshots", "price_cache", "proposals"):
        (imp / d).mkdir(parents=True)
    monkeypatch.setattr(ie, "IMPROVEMENT_DIR", imp)
    monkeypatch.setattr(ie, "SNAPSHOTS_DIR", imp / "snapshots")
    monkeypatch.setattr(ie, "PRICE_CACHE_DIR", imp / "price_cache")
    monkeypatch.setattr(ie, "PROPOSALS_DIR", imp / "proposals")
    monkeypatch.setattr(ie, "PERFORMANCE_HISTORY_PATH", imp / "performance_history.csv")
    ciks = tmp_path / "ticker_ciks.json"
    ciks.write_text(json.dumps({"ciks": {"OLD": 111, "NEW": 111, "GONE": 222, "KEEP": 333}}))
    monkeypatch.setattr(ie, "TICKER_CIKS_PATH", ciks)
    return tmp_path


STAYERS = [f"S{i:02d}" for i in range(20)]


def _snapshot(day, tickers, prices=None):
    df = pd.DataFrame({"Ticker": tickers, "Composite": np.linspace(90, 10, len(tickers)),
                       "Rank": np.arange(1, len(tickers) + 1), "in_portfolio": False})
    for c in ie.CATEGORY_SCORES:
        df[c] = 50.0
    df["price_at_scoring"] = [(prices or {}).get(t, 100.0) for t in tickers]
    df.to_parquet(ie.SNAPSHOTS_DIR / f"{day}_{day.replace('-', '')}.parquet", index=False)


def _frame(extra=None):
    idx = pd.bdate_range("2026-02-20", "2026-04-10")
    data = {t: np.linspace(100, 110, len(idx)) for t in STAYERS}
    data.update(extra or {})
    return pd.DataFrame({k: (v if not np.isscalar(v) else np.full(len(idx), v)) for k, v in data.items()},
                        index=idx)


def _stub(monkeypatch, frame):
    seen = []

    def fetch(tickers, start, end):
        seen.append(list(tickers))
        return frame[[c for c in frame.columns if c in tickers]].copy()
    monkeypatch.setattr(ie, "_fetch_prices_for_returns", fetch)
    return seen


def _row(ticker):
    ph = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
    got = ph[(ph["run_date"] == RUN) & (ph["ticker"] == ticker)]
    return None if got.empty else got.iloc[0]


def test_a_renamed_stock_is_priced_under_its_new_symbol(monkeypatch):
    """BK became BNY: the source no longer knows BK, so it was dropped from every date."""
    _snapshot(RUN, STAYERS + ["OLD"])
    _snapshot(LATER, STAYERS + ["NEW"])
    idx = pd.bdate_range("2026-02-20", "2026-04-10")
    seen = _stub(monkeypatch, _frame({"NEW": np.linspace(50, 60, len(idx))}))
    ie.compute_forward_returns(LATER)
    assert "NEW" in seen[0], "the new symbol was never asked for"
    row = _row("OLD")
    assert row is not None, "the renamed stock fell out of the panel"
    new = pd.Series(np.linspace(50, 60, len(idx)), index=idx)
    entry = new[new.index >= RUN].iloc[0]
    end = new[new.index >= pd.Timestamp(RUN) + pd.Timedelta(days=30)].iloc[0]
    assert row["fwd_return_1m"] == pytest.approx(end / entry - 1)


def test_a_stock_that_stopped_trading_is_held_at_its_last_recorded_close(monkeypatch):
    """A takeover: recorded at 40 on the runs after the ranking, then gone. Held as cash."""
    _snapshot(RUN, STAYERS + ["GONE"], {"GONE": 39.0})
    _snapshot("2026-03-03", STAYERS + ["GONE"], {"GONE": 40.0})      # records 03-02's close
    _snapshot("2026-03-06", STAYERS + ["GONE"], {"GONE": 44.0})      # records 03-05's close
    _snapshot(LATER, STAYERS)
    _stub(monkeypatch, _frame())
    ie.compute_forward_returns(LATER)
    row = _row("GONE")
    assert row is not None, "the stock that stopped trading fell out of the panel"
    assert row["fwd_return_1m"] == pytest.approx(44.0 / 40.0 - 1)
    assert row["fwd_return_1w"] == pytest.approx(44.0 / 40.0 - 1)


def test_no_position_is_held_without_an_observed_entry(monkeypatch):
    """Last recorded before the run's own close: there was no price to enter at."""
    _snapshot("2026-02-27", STAYERS + ["GONE"], {"GONE": 40.0})      # records 02-26's close
    _snapshot(RUN, STAYERS + ["GONE"], {"GONE": 41.0})               # records 02-27's close
    _snapshot(LATER, STAYERS)
    _stub(monkeypatch, _frame())
    ie.compute_forward_returns(LATER)
    assert _row("GONE") is None


def test_a_stock_still_in_the_universe_is_never_filled(monkeypatch):
    """A gap in the source for a stock that still trades stays a gap, as before."""
    _snapshot(RUN, STAYERS + ["KEEP"], {"KEEP": 40.0})
    _snapshot("2026-03-03", STAYERS + ["KEEP"], {"KEEP": 41.0})
    _snapshot(LATER, STAYERS + ["KEEP"], {"KEEP": 45.0})
    _stub(monkeypatch, _frame())
    ie.compute_forward_returns(LATER)
    assert _row("KEEP") is None


def test_a_renamed_stock_whose_new_symbol_is_missing_is_not_held_as_cash(monkeypatch):
    """It still trades under the new symbol; freezing its price would invent a return."""
    _snapshot(RUN, STAYERS + ["OLD"], {"OLD": 50.0})
    _snapshot("2026-03-03", STAYERS + ["OLD"], {"OLD": 51.0})
    _snapshot(LATER, STAYERS + ["NEW"])
    _stub(monkeypatch, _frame())                                      # NEW not served either
    ie.compute_forward_returns(LATER)
    assert _row("OLD") is None


def test_the_stayers_are_unchanged(monkeypatch):
    _snapshot(RUN, STAYERS + ["GONE"])
    _snapshot(LATER, STAYERS)
    _stub(monkeypatch, _frame())
    ie.compute_forward_returns(LATER)
    ph = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
    assert set(STAYERS) <= set(ph.loc[ph["run_date"] == RUN, "ticker"])


def test_the_engine_and_the_track_record_share_one_rule():
    """No second copy of the rename or the recorded-close arithmetic in the engine."""
    src = (ie.ROOT / "improvement_engine.py").read_text(encoding="utf-8")
    assert "track_record.renamed_symbols(" in src and "track_record.fill_unpriced(" in src
    assert "track_record.recorded_closes(" in src
