"""The dividend yield: one derivation, dollar inputs only, and a named basis.

On 2026-10-14 the source stopped serving `.info["dividendRate"]` - 0 of 12
probed tickers, every sibling field still served
(`research/measurements/2026-10-14-dividend-field-probe.py`). `dividend_yield`
fell from 402 of 501 stocks to 6, `tests/test_input_churn.py` went red, and the
live Company Snapshot read "Dividend Yield: None" for Coca-Cola, J&J and
Procter & Gamble - a false statement, not a gap.

Two things went wrong and both are pinned here:

1. The metric read **one** field with no fallback, so one source change emptied it.
2. The dashboard derived the same quantity a **second** time from its own copy of
   that field (`CLAUDE.md` row 0.8), so fixing the engine alone would have left
   the page showing "None".
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import factor_engine as F

ROOT = Path(__file__).resolve().parents[1]
GEN = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")

KO_PRICE = 90.11
KO_RATE = 2.08           # dollars a share a year
KO_YIELD = KO_RATE / KO_PRICE   # 2.308%


class TestTheRateFallback:
    def test_the_declared_forward_rate_is_preferred(self):
        y, basis = F.dividend_yield_from_fields(
            {"dividendRate": KO_RATE, "trailingAnnualDividendRate": 1.0,
             "currentPrice": KO_PRICE})
        assert y == pytest.approx(KO_YIELD)
        assert basis == "forward"

    def test_the_trailing_rate_carries_it_when_the_source_drops_the_forward_one(self):
        """The 2026-10-14 case: `dividendRate` absent for the whole universe."""
        y, basis = F.dividend_yield_from_fields(
            {"dividendRate": np.nan, "trailingAnnualDividendRate": KO_RATE,
             "currentPrice": KO_PRICE})
        assert y == pytest.approx(KO_YIELD)
        assert basis == "trailing"

    def test_a_missing_key_is_not_a_zero(self):
        """The field absent entirely, which is how the source served it that day."""
        y, basis = F.dividend_yield_from_fields(
            {"trailingAnnualDividendRate": KO_RATE, "currentPrice": KO_PRICE})
        assert y == pytest.approx(KO_YIELD)
        assert basis == "trailing"

    def test_a_genuine_non_payer_reads_zero_not_unknown(self):
        y, basis = F.dividend_yield_from_fields(
            {"trailingAnnualDividendRate": 0.0, "currentPrice": 250.0})
        assert y == 0.0
        assert basis == "trailing"

    def test_no_rate_at_all_is_unknown(self):
        y, basis = F.dividend_yield_from_fields({"currentPrice": KO_PRICE})
        assert np.isnan(y)
        assert basis is None

    @pytest.mark.parametrize("price", [None, 0, -1.0, np.nan])
    def test_no_usable_price_is_unknown(self, price):
        y, basis = F.dividend_yield_from_fields(
            {"trailingAnnualDividendRate": KO_RATE, "currentPrice": price})
        assert np.isnan(y)
        assert basis is None

    def test_the_price_fallback_matches_the_rest_of_the_block(self):
        """`_coalesce(d, "currentPrice", "price_latest")`, as its neighbours use."""
        y, _ = F.dividend_yield_from_fields(
            {"trailingAnnualDividendRate": 2.0, "price_latest": 50.0})
        assert y == pytest.approx(0.04)


class TestUnitsCannotBeMisread:
    """Dollar amounts only. The units trap is the reason, and it is measured."""

    def test_the_percent_shaped_yield_field_is_never_read(self):
        """`.info["dividendYield"]` was 2.35 for a stock yielding 2.31% on
        2026-10-14, and a fraction in older yfinance releases. Reading it is a
        silent 100x error the release after next."""
        y, basis = F.dividend_yield_from_fields(
            {"dividendYield": 2.35, "trailingAnnualDividendYield": 0.0232,
             "currentPrice": KO_PRICE})
        assert np.isnan(y), "a yield field was read; only dollar rates may be"
        assert basis is None

    def test_only_dollar_rate_fields_are_consulted(self):
        assert F.DIVIDEND_RATE_FIELDS == ("dividendRate", "trailingAnnualDividendRate")
        for field in F.DIVIDEND_RATE_FIELDS:
            assert "Yield" not in field, f"{field} is a yield, not a rate"
            assert field in F.DIVIDEND_BASIS

    def test_an_impossible_yield_is_refused_and_the_next_field_tried(self):
        """A percent served where dollars were expected is ~100x too big."""
        y, basis = F.dividend_yield_from_fields(
            {"dividendRate": 2.35 * KO_PRICE,      # as if a percent reached the rate
             "trailingAnnualDividendRate": KO_RATE,
             "currentPrice": KO_PRICE})
        assert y == pytest.approx(KO_YIELD)
        assert basis == "trailing"

    def test_the_bound_is_clear_of_every_real_yield(self):
        """It must reject only the impossible: the highest S&P 500 yields are
        under 10%, so a 25% ceiling never touches a real reading."""
        assert F.DIVIDEND_YIELD_MAX == 0.25
        y, _ = F.dividend_yield_from_fields(
            {"trailingAnnualDividendRate": 0.099 * 100.0, "currentPrice": 100.0})
        assert y == pytest.approx(0.099), "a 9.9% yield must survive the bound"


class TestTheFieldIsFetched:
    def test_the_fallback_field_is_harvested_from_info(self):
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        assert 'rec["trailingAnnualDividendRate"] = _safe(info, "trailingAnnualDividendRate")' in src

    def test_the_schema_carries_it(self):
        import schemas
        assert "trailingAnnualDividendRate" in schemas.RawTickerData.model_fields


class TestOneDerivationOnly:
    """Row 0.8: the page shows the engine's numbers, never its own re-derivation."""

    def test_the_generator_does_not_divide_a_dividend_rate_by_a_price(self):
        offenders = [ln.strip() for ln in GEN.splitlines()
                     if re.search(r"_div(idend)?_rate\s*/", ln)]
        assert not offenders, (
            "the generator is deriving the yield itself again: " + str(offenders))

    def test_the_generator_does_not_carry_the_raw_rate_at_all(self):
        """Not available means it cannot be re-derived by the next editor."""
        assert '("dividendRate", "_dividend_rate")' not in GEN

    def test_both_payload_fields_read_the_engines_column(self):
        assert '"dividend_yield": round(_div_yield, 4)' in GEN
        assert '"div_yield": round(_div_yield, 4)' in GEN
        assert '_div_yield = _safe(row.get("dividend_yield"))' in GEN


class TestThePageSaysWhichBasis:
    def test_the_basis_is_published(self):
        assert '"dividend_basis": _div_basis' in GEN
        assert '("_dividend_basis", "_dividend_basis")' in GEN

    def test_the_row_names_the_basis_it_is_showing(self):
        assert "last 12 months paid" in GEN
        assert "declared annual rate" in GEN

    def test_unknown_is_not_rendered_as_none(self):
        """"None" asserts the company pays no dividend. For 495 stocks on
        2026-10-14 that was false - the figure was simply absent."""
        row = GEN[GEN.index("label: 'Dividend Yield'"):][:600]
        assert "f.dividend_yield === null" in row and "'—'" in row, (
            "an absent yield must render as unknown, not as 'None'")
        assert "f.dividend_yield === 0 ? 'None'" in row, (
            "'None' must be reserved for a measured zero")
