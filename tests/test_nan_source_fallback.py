"""Two-source metric fallbacks must fall back on a present-but-NaN value.

`compute_metrics()` reads nine numeric inputs that have two possible sources,
and every one of them was written as::

    d.get("preferred", d.get("backup", np.nan))

That idiom only reaches ``backup`` when the ``preferred`` *key is absent*.
`_fetch_single_ticker_inner()` writes every one of those keys unconditionally
(via `_safe()` or `_stmt_val()`, both of which return NaN rather than omitting
the key), so the key is always present and the fallback can only fire on an
exception path - never on the missing-data path it was written for.

Measured on four retained raw fetches (`runs/*/00_raw_fetch.parquet`,
503 names each, identical every run):

| first source NaN, backup available | names |
|---|---|
| `totalDebt` -> `totalDebt_bs`      | 1 (FISV) |
| `totalDebt_bs` -> `totalDebt`      | 3 (ANET, ERIE, ISRG) |
| `totalCash` -> `cash_bs`           | 1 (FISV) |
| `currentPrice` -> `price_latest`   | 0 |
| `ebit_annual` -> `ebit`            | 0 |

Cost of the dead fallback, measured as an exact A/B of the whole 503-name
universe against the pre-change module - metrics gained, none lost, and only
`_metric_count` otherwise altered:

| name | regains | weight restored |
|---|---|---|
| ANET | `roic`, `net_debt_to_ebitda` | 45 of 100 quality |
| ISRG | `roic`, `net_debt_to_ebitda` | 45 of 100 quality |
| FISV | `fcf_yield`, `ev_ebitda`, `ev_sales` | 80 of 100 valuation |

ERIE has a rescuable debt figure too but regains nothing - its EBIT is missing
as well, so ROIC stays NaN for an unrelated reason. Reproduce with
`research/measurements/2026-09-25-dead-two-source-fallbacks.py`.

The records below are shaped from those four real names. The price cases have
measured incidence **zero**, so they pin a safety net rather than a live fix;
they are here because `_current_price` feeds `price_at_scoring` in every
improvement-engine snapshot, and a name with no baseline price contributes no
forward return at all.

One site is deliberately NOT given a fallback - see
`TestReturn12mRefusesCrossSourcePrices`.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import factor_engine as fe

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def market_returns():
    """A 260-day market series, long enough for the >=200 guards."""
    rng = np.random.default_rng(12345)
    idx = pd.bdate_range("2025-09-01", periods=260)
    return pd.Series(rng.normal(0.0004, 0.01, 260), index=idx)


def _base_record(**over):
    """A record complete enough to compute the metrics under test."""
    rec = {
        "Ticker": "TEST",
        "shortName": "Test Co",
        "sector": "Technology",
        "industry": "Software",
        "marketCap": 1.0e10,
        "enterpriseValue": 1.1e10,
        "trailingEps": 4.0,
        "forwardEps": 5.0,
        "currentPrice": 100.0,
        "price_latest": 100.0,
        "totalDebt": 1.0e9,
        "totalDebt_bs": 1.0e9,
        "totalCash": 5.0e8,
        "cash_bs": 5.0e8,
        "totalEquity": 6.0e9,
        "totalAssets": 1.0e10,
        "totalRevenue": 4.0e9,
        "ebit": 8.0e8,
        "ebitda": 1.0e9,
        "incomeTaxExpense": 1.5e8,
        "pretaxIncome": 7.5e8,
        "sharesOutstanding": 1.0e8,
        "bookValue": 20.0,
        "targetMeanPrice": 120.0,
        "numberOfAnalystOpinions": 10,
        "fiftyTwoWeekHigh": 125.0,
        "_fy1_eps_current": 5.2,
        "_fy1_eps_90d_ago": 5.0,
    }
    rec.update(over)
    return rec


def _metrics(market_returns, **over):
    df = fe.compute_metrics([_base_record(**over)], market_returns)
    assert len(df) == 1
    return df.iloc[0]


def _val(row, col):
    return float(row[col]) if col in row.index and pd.notna(row[col]) else None


# --------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------

class TestCoalesceHelper:

    def test_returns_first_present_value(self):
        assert fe._coalesce({"a": 1.0, "b": 2.0}, "a", "b") == 1.0

    def test_skips_present_but_nan_first_key(self):
        """This is the whole point: dict.get returns the NaN and stops."""
        assert fe._coalesce({"a": np.nan, "b": 2.0}, "a", "b") == 2.0

    def test_skips_present_but_none_first_key(self):
        assert fe._coalesce({"a": None, "b": 2.0}, "a", "b") == 2.0

    def test_skips_absent_first_key(self):
        assert fe._coalesce({"b": 2.0}, "a", "b") == 2.0

    def test_all_nan_yields_nan(self):
        assert pd.isna(fe._coalesce({"a": np.nan, "b": np.nan}, "a", "b"))

    def test_all_absent_yields_nan(self):
        assert pd.isna(fe._coalesce({}, "a", "b"))

    def test_no_keys_yields_nan(self):
        assert pd.isna(fe._coalesce({"a": 1.0}))

    def test_zero_is_a_value_not_a_miss(self):
        """0.0 debt is the real reading for a debt-free company (ANET, ISRG)."""
        assert fe._coalesce({"a": 0.0, "b": 9.0}, "a", "b") == 0.0

    def test_chains_past_more_than_one_nan(self):
        assert fe._coalesce({"a": np.nan, "b": np.nan, "c": 3.0}, "a", "b", "c") == 3.0

    def test_does_not_mutate_the_record(self):
        d = {"a": np.nan, "b": 2.0}
        fe._coalesce(d, "a", "b")
        assert set(d) == {"a", "b"} and pd.isna(d["a"])


# --------------------------------------------------------------------------
# Recurrence guard on the root cause
# --------------------------------------------------------------------------

class TestNestedGetIdiomIsGone:

    def test_no_nested_dict_get_fallback_in_factor_engine(self):
        """`d.get(A, d.get(B, ...))` is the dead-fallback idiom itself.

        It reads as a two-source fallback and cannot be one. Any new
        occurrence should use `_coalesce` instead, so the pattern is barred
        at source rather than re-audited every few months.
        """
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        hits = re.findall(r'\bd\.get\(\s*"[^"]+"\s*,\s*d\.get\(', src)
        assert hits == [], (
            f"{len(hits)} nested d.get() fallback(s) remain - these cannot "
            f"fall back on a present-but-NaN value; use _coalesce()"
        )

    def test_coalesce_is_actually_used(self):
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        assert src.count("_coalesce(d,") >= 9, (
            "expected _coalesce at the nine two-source sites"
        )


# --------------------------------------------------------------------------
# Quality: ROIC and net debt / EBITDA (ANET, ERIE, ISRG shape)
# --------------------------------------------------------------------------

class TestBalanceSheetDebtFallsBackToInfo:
    """`totalDebt_bs` absent from a filing that otherwise parsed.

    Measured on three real names: ANET and ISRG report `.info` totalDebt of
    exactly 0.0 with no "Total Debt" line on the quarterly balance sheet;
    ERIE reports 63.157M. All three lose roic and net_debt_to_ebitda today.
    """

    def test_roic_computes_when_only_info_debt_is_available(self, market_returns):
        row = _metrics(market_returns, totalDebt_bs=np.nan, totalDebt=0.0)
        assert _val(row, "roic") is not None, "roic lost despite .info debt"

    def test_net_debt_to_ebitda_computes_when_only_info_debt_is_available(
            self, market_returns):
        row = _metrics(market_returns, totalDebt_bs=np.nan, totalDebt=0.0)
        assert _val(row, "net_debt_to_ebitda") is not None

    def test_debt_free_company_gets_the_zero_not_a_nan(self, market_returns):
        """debt=0 must reach invested capital as 0, not be read as missing."""
        zero = _metrics(market_returns, totalDebt_bs=0.0, totalDebt=0.0)
        fell_back = _metrics(market_returns, totalDebt_bs=np.nan, totalDebt=0.0)
        assert _val(fell_back, "roic") == pytest.approx(_val(zero, "roic"))

    def test_balance_sheet_debt_still_takes_precedence(self, market_returns):
        """The filing figure wins when it is there - temporal consistency."""
        both = _metrics(market_returns, totalDebt_bs=2.0e9, totalDebt=1.0e9)
        only_bs = _metrics(market_returns, totalDebt_bs=2.0e9, totalDebt=np.nan)
        assert _val(both, "roic") == pytest.approx(_val(only_bs, "roic"))

    def test_both_debt_sources_nan_still_yields_nan(self, market_returns):
        """No fabrication: absent data stays absent."""
        row = _metrics(market_returns, totalDebt_bs=np.nan, totalDebt=np.nan)
        assert _val(row, "roic") is None

    def test_balance_sheet_cash_falls_back_to_info_cash(self, market_returns):
        row = _metrics(market_returns, cash_bs=np.nan, totalCash=5.0e8)
        assert _val(row, "roic") is not None


# --------------------------------------------------------------------------
# Valuation: enterprise value (FISV shape)
# --------------------------------------------------------------------------

class TestEnterpriseValueFallsBackToBalanceSheet:
    """`.info` returned nothing usable; the balance sheet is fresh.

    FISV on every retained run: enterpriseValue, totalDebt and totalCash all
    NaN, while totalDebt_bs = 28.034B and cash_bs = 245M (both filed
    2026-06-30) and marketCap = 24.45B. Both EV metrics are lost.
    """

    FISV = dict(enterpriseValue=np.nan, totalDebt=np.nan, totalCash=np.nan,
                totalDebt_bs=2.8034e10, cash_bs=2.45e8, marketCap=2.445e10)

    def test_ev_ebitda_computes_from_balance_sheet_debt_and_cash(
            self, market_returns):
        row = _metrics(market_returns, **self.FISV)
        assert _val(row, "ev_ebitda") is not None, "ev_ebitda lost despite BS data"

    def test_ev_sales_computes_from_balance_sheet_debt_and_cash(
            self, market_returns):
        row = _metrics(market_returns, **self.FISV)
        assert _val(row, "ev_sales") is not None

    def test_reconstructed_ev_matches_mcap_plus_debt_minus_cash(
            self, market_returns):
        row = _metrics(market_returns, **self.FISV)
        expected_ev = 2.445e10 + 2.8034e10 - 2.45e8
        assert _val(row, "ev_sales") == pytest.approx(expected_ev / 4.0e9, rel=1e-6)

    def test_info_cash_and_debt_still_take_precedence_for_ev(self, market_returns):
        """`.info` matches yfinance's own EV definition; keep preferring it."""
        both = _metrics(market_returns, enterpriseValue=np.nan,
                        totalDebt=1.0e9, totalCash=5.0e8,
                        totalDebt_bs=9.9e9, cash_bs=0.0)
        expected_ev = 1.0e10 + 1.0e9 - 5.0e8
        assert _val(both, "ev_sales") == pytest.approx(expected_ev / 4.0e9, rel=1e-6)

    def test_no_debt_or_cash_anywhere_leaves_ev_metrics_nan(self, market_returns):
        row = _metrics(market_returns, enterpriseValue=np.nan,
                       totalDebt=np.nan, totalCash=np.nan,
                       totalDebt_bs=np.nan, cash_bs=np.nan)
        assert _val(row, "ev_ebitda") is None
        assert _val(row, "ev_sales") is None


# --------------------------------------------------------------------------
# Price-denominator sites: measured incidence zero, real safety net
# --------------------------------------------------------------------------

class TestPriceFallsBackToLatestClose:
    """`currentPrice` present but NaN, with the 13-month series' last close
    available. Incidence on the live universe is zero; these pin the net."""

    NO_INFO_PRICE = dict(currentPrice=np.nan, price_latest=100.0)

    def test_peg_ratio_computes(self, market_returns):
        assert _val(_metrics(market_returns, **self.NO_INFO_PRICE),
                    "peg_ratio") is not None

    def test_price_target_upside_computes(self, market_returns):
        assert _val(_metrics(market_returns, **self.NO_INFO_PRICE),
                    "price_target_upside") is not None

    def test_proximity_52w_high_computes(self, market_returns):
        assert _val(_metrics(market_returns, **self.NO_INFO_PRICE),
                    "proximity_52w_high") is not None

    def test_fy1_revision_3m_still_computes(self, market_returns):
        """Already guarded by hand on 2026-09-10; must survive the refactor."""
        assert _val(_metrics(market_returns, **self.NO_INFO_PRICE),
                    "fy1_revision_3m") is not None

    def test_current_price_passthrough_uses_the_latest_close(self, market_returns):
        """Feeds `price_at_scoring` in every improvement-engine snapshot."""
        row = _metrics(market_returns, **self.NO_INFO_PRICE)
        assert _val(row, "_current_price") == pytest.approx(100.0)

    def test_bank_pb_ratio_computes(self, market_returns):
        row = _metrics(market_returns, currentPrice=np.nan, price_latest=100.0,
                       sector="Financial Services",
                       industry="Banks - Diversified", priceToBook=np.nan,
                       bookValue=20.0)
        assert _val(row, "pb_ratio") == pytest.approx(5.0)

    def test_info_price_takes_precedence_over_latest_close(self, market_returns):
        row = _metrics(market_returns, currentPrice=110.0, price_latest=100.0)
        assert _val(row, "_current_price") == pytest.approx(110.0)

    def test_no_price_anywhere_leaves_metrics_nan(self, market_returns):
        row = _metrics(market_returns, currentPrice=np.nan, price_latest=np.nan)
        assert _val(row, "peg_ratio") is None
        assert _val(row, "price_target_upside") is None
        assert _val(row, "_current_price") is None


# --------------------------------------------------------------------------
# The one site that must NOT get a fallback
# --------------------------------------------------------------------------

class TestReturn12mRefusesCrossSourcePrices:
    """`return_12m` compares two dates and must use one price scale.

    It is the only site whose fallback runs the other way -
    `d.get("price_latest", d.get("currentPrice"))`. That order is deliberate
    and the `currentPrice` leg is deleted rather than repaired, for two
    reasons:

    1. It cannot change any outcome. `price_latest` and `price_12m_ago` are
       written by the same guarded block in `_fetch_single_ticker_inner`
       (`if hist is not None and len(hist) >= 10`), so whenever
       `price_latest` is unavailable `price_12m_ago` is too, and the metric
       is NaN regardless of what supplies the near endpoint.
    2. If it ever did fire it would be wrong. `price_12m_ago` comes from
       `t.history(auto_adjust=True)`; `.info["currentPrice"]` does not.
       Dividing an unadjusted price by an adjusted one across a split is
       exactly the defect that put MNST on the live site at momentum 71.5
       when its true split-adjusted 12-1 return was the 3rd percentile
       (fixed 2026-08-26, `check_price_series_integrity`).
    """

    def test_does_not_substitute_info_price_for_the_near_endpoint(
            self, market_returns):
        """Impossible from the fetcher; asserted so a future edit cannot
        reintroduce a cross-scale division."""
        d = _base_record(currentPrice=100.0)
        d.pop("price_latest")
        d["price_12m_ago"] = 80.0
        d["price_1m_ago"] = 95.0
        row = fe.compute_metrics([d], market_returns).iloc[0]
        assert _val(row, "return_12m") is None, (
            "return_12m fell back to .info price - that is a cross-scale "
            "comparison, see check_price_series_integrity"
        )

    def test_uses_the_adjusted_series_even_when_info_price_differs(
            self, market_returns):
        row = _metrics(market_returns, currentPrice=200.0, price_latest=100.0,
                       price_12m_ago=80.0, price_1m_ago=95.0)
        assert _val(row, "return_12m") == pytest.approx((100.0 - 80.0) / 80.0)

    def test_skip_month_momentum_is_unaffected(self, market_returns):
        """return_12_1 uses two history endpoints and no .info price at all."""
        row = _metrics(market_returns, currentPrice=200.0, price_latest=100.0,
                       price_12m_ago=80.0, price_1m_ago=95.0)
        assert _val(row, "return_12_1") == pytest.approx((95.0 - 80.0) / 80.0)

    def test_the_reversed_order_is_documented_at_the_site(self):
        """A bare `d.get("price_latest", np.nan)` beside ten _coalesce calls
        reads like an oversight; the reason must stay next to it."""
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        i = src.index('rec["return_12m"]')
        window = src[max(0, i - 1400):i]
        assert "auto_adjust" in window or "split" in window.lower(), (
            "return_12m's deliberate single-source price needs its reason "
            "written at the site"
        )


# --------------------------------------------------------------------------
# The comment that promised the fallback must describe what happens
# --------------------------------------------------------------------------

class TestDocumentedIntentMatchesTheCode:

    def test_price_series_derived_note_names_the_precedence_counterexample(self):
        """The note above PRICE_SERIES_DERIVED_FIELDS used to justify keeping
        `price_latest` on the claim that `info["currentPrice"]` takes
        precedence over it *everywhere* it is used. `return_12m` prefers
        `price_latest`, so the justification was false at one of the seven
        sites. The conclusion still holds; the note must carry the real
        reason, which means naming the counterexample."""
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        i = src.index("PRICE_SERIES_DERIVED_FIELDS = (")
        note = src[max(0, i - 2000):i]
        assert "return_12m" in note, (
            "the note justifies keeping price_latest without mentioning the "
            "one site that prefers it"
        )

    def test_roic_comment_admits_the_debt_source_may_differ(self):
        """The invested-capital comment claims equity, debt and cash all come
        from the same filing. With the fallback live, debt may come from
        `.info` when the filing carries no Total Debt line."""
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        i = src.index('rec["roic"] = (nopat / ic)')
        window = src[max(0, i - 2600):i]
        assert ".info" in window, (
            "ROIC's temporal-consistency claim needs the .info debt fallback "
            "stated beside it"
        )
