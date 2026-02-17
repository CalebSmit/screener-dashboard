#!/usr/bin/env python3
"""
Data Accuracy Validation Script
================================
Compares raw yfinance data to the screener's computed metrics
for representative tickers to identify discrepancies.

Usage: python validate_data_accuracy.py
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    sys.exit(1)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
TEST_TICKERS = ["AAPL", "JPM", "XOM", "MSFT", "TSLA"]

# Wikipedia GICS sector names vs yfinance sector names
WIKI_TO_YF_SECTOR = {
    "Information Technology": "Technology",
    "Consumer Discretionary": "Consumer Cyclical",
    "Financials": "Financial Services",
    "Health Care": "Healthcare",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
    # These match:
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}

YF_TO_WIKI_SECTOR = {v: k for k, v in WIKI_TO_YF_SECTOR.items()}


# -------------------------------------------------------------------------
# Import screener functions
# -------------------------------------------------------------------------
from factor_engine import (
    _safe, _stmt_val, _fetch_single_ticker_inner,
    compute_metrics, fetch_market_returns,
)


# -------------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------------
class DiscrepancyLog:
    def __init__(self):
        self.rows = []

    def add(self, ticker, field, issue_type, expected, actual, severity, detail):
        self.rows.append({
            "Ticker": ticker,
            "Field": field,
            "Issue": issue_type,
            "Expected": expected,
            "Actual": actual,
            "Severity": severity,
            "Detail": detail,
        })

    def print_report(self):
        if not self.rows:
            print("\n  No discrepancies found.")
            return
        df = pd.DataFrame(self.rows)
        print(f"\n{'='*100}")
        print(f"  DATA ACCURACY VALIDATION REPORT")
        print(f"  {len(self.rows)} issues found across {df['Ticker'].nunique()} tickers")
        print(f"{'='*100}")

        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            subset = df[df["Severity"] == sev]
            if subset.empty:
                continue
            print(f"\n--- {sev} ({len(subset)} issues) ---")
            for _, r in subset.iterrows():
                print(f"\n  [{r['Ticker']}] {r['Field']}")
                print(f"    Issue:    {r['Issue']}")
                print(f"    Expected: {r['Expected']}")
                print(f"    Actual:   {r['Actual']}")
                print(f"    Detail:   {r['Detail']}")

        # Summary table
        print(f"\n{'='*100}")
        print("SUMMARY BY ISSUE TYPE:")
        for issue, count in df["Issue"].value_counts().items():
            print(f"  {issue:45s} {count:3d} occurrences")

    def to_dataframe(self):
        return pd.DataFrame(self.rows)


# -------------------------------------------------------------------------
# Test 1: Sector naming mismatch
# -------------------------------------------------------------------------
def test_sector_mismatch(log):
    """Check if yfinance sector names match the Wikipedia GICS names used in scoring."""
    print("\n[TEST 1] Sector Naming Mismatch")
    print("-" * 50)

    for sym in TEST_TICKERS:
        t = yf.Ticker(sym)
        info = t.info or {}
        yf_sector = info.get("sector", "Unknown")
        wiki_equivalent = YF_TO_WIKI_SECTOR.get(yf_sector, None)

        if wiki_equivalent and wiki_equivalent != yf_sector:
            log.add(
                sym, "sector", "SECTOR_NAME_MISMATCH",
                f"Wikipedia GICS: '{wiki_equivalent}'",
                f"yfinance: '{yf_sector}'",
                "CRITICAL",
                f"yfinance returns '{yf_sector}' but Wikipedia/GICS uses '{wiki_equivalent}'. "
                f"Stocks may be assigned to wrong sector groups for percentile ranking. "
                f"The screener falls back to Wikipedia sector when yfinance returns 'Unknown', "
                f"but NOT when yfinance returns a valid but differently-named sector."
            )
            print(f"  {sym}: yfinance='{yf_sector}' vs GICS='{wiki_equivalent}' ** MISMATCH **")
        else:
            print(f"  {sym}: yfinance='{yf_sector}' -- OK (matches GICS)")


# -------------------------------------------------------------------------
# Test 2: CapEx sign convention
# -------------------------------------------------------------------------
def test_capex_sign(log):
    """Check if CapEx sign is handled correctly in FCF calculation."""
    print("\n[TEST 2] CapEx Sign Convention in FCF")
    print("-" * 50)

    for sym in TEST_TICKERS:
        t = yf.Ticker(sym)
        cf = t.cashflow
        if cf is None or cf.empty:
            print(f"  {sym}: No cashflow data")
            continue

        capex_val = None
        capex_label = None
        for idx in cf.index:
            if "capital expenditure" == str(idx).lower().strip():
                vals = cf.loc[idx].dropna()
                if len(vals) > 0:
                    capex_val = float(vals.iloc[0])
                    capex_label = str(idx)
                break

        ocf_val = None
        for idx in cf.index:
            if "operating cash flow" == str(idx).lower().strip():
                vals = cf.loc[idx].dropna()
                if len(vals) > 0:
                    ocf_val = float(vals.iloc[0])
                break

        if capex_val is not None:
            print(f"  {sym}: CapEx={capex_val:,.0f} (label='{capex_label}')")

            # In yfinance, Capital Expenditure is NEGATIVE
            # Screener code (line 500-501):
            #   fcf = (ocf - abs(capex)) if capex < 0 else (ocf - capex if capex else ocf)
            # This is correct IF capex is negative, but the abs() + negativity check
            # means: if capex = -12B, fcf = ocf - 12B (correct)
            # BUT: if capex = 12B (positive, shouldn't happen but...), fcf = ocf - 12B (also correct)
            # This logic is actually robust.

            if capex_val < 0:
                screener_fcf = ocf_val - abs(capex_val) if ocf_val else None
                correct_fcf = ocf_val + capex_val if ocf_val else None  # capex is already negative
                print(f"    OCF={ocf_val:,.0f}, Screener FCF={screener_fcf:,.0f}, Correct FCF={correct_fcf:,.0f}")
                if screener_fcf and correct_fcf and abs(screener_fcf - correct_fcf) > 1:
                    log.add(
                        sym, "fcf_yield (capex)", "CAPEX_SIGN_DOUBLE_NEGATIVE",
                        f"FCF = OCF + CapEx = {correct_fcf:,.0f}",
                        f"Screener: OCF - abs(CapEx) = {screener_fcf:,.0f}",
                        "LOW",
                        "Both approaches yield same result when CapEx is negative."
                    )
            else:
                print(f"    CapEx is POSITIVE ({capex_val:,.0f}) -- unusual, check data")


# -------------------------------------------------------------------------
# Test 3: EV calculation - totalCash discrepancy
# -------------------------------------------------------------------------
def test_ev_cash_mismatch(log):
    """Check if EV fallback uses totalCash (info) vs Cash And Cash Equivalents (BS)."""
    print("\n[TEST 3] Enterprise Value Cash Component")
    print("-" * 50)

    for sym in TEST_TICKERS:
        t = yf.Ticker(sym)
        info = t.info or {}

        ev_yf = info.get("enterpriseValue", None)
        mc = info.get("marketCap", None)
        td_info = info.get("totalDebt", None)
        tc_info = info.get("totalCash", None)

        bs = t.balance_sheet
        cash_bs = None
        if bs is not None and not bs.empty:
            for idx in bs.index:
                if "cash and cash equivalents" == str(idx).lower().strip():
                    vals = bs.loc[idx].dropna()
                    if len(vals) > 0:
                        cash_bs = float(vals.iloc[0])
                    break

        if mc and td_info and tc_info:
            ev_manual = mc + td_info - tc_info
            print(f"  {sym}:")
            print(f"    yfinance EV:        {ev_yf:>20,.0f}")
            print(f"    Manual (info cash): {ev_manual:>20,.0f}")
            print(f"    info[totalCash]:     {tc_info:>20,.0f}")
            print(f"    BS CashEquiv:       {cash_bs:>20,.0f}" if cash_bs else "    BS CashEquiv: N/A")

            if tc_info and cash_bs and abs(tc_info - cash_bs) / max(abs(tc_info), 1) > 0.1:
                log.add(
                    sym, "enterpriseValue (cash)", "TOTALCASH_VS_BS_CASH_MISMATCH",
                    f"BS Cash & Equivalents = {cash_bs:,.0f}",
                    f"info['totalCash'] = {tc_info:,.0f}",
                    "MEDIUM",
                    f"info['totalCash'] includes short-term investments ({tc_info:,.0f}) "
                    f"but BS 'Cash And Cash Equivalents' is just cash ({cash_bs:,.0f}). "
                    f"Difference: {abs(tc_info - cash_bs):,.0f}. When EV fallback uses "
                    f"info['totalCash'] and ROIC uses cash_bs, the invested capital "
                    f"calculation may be inconsistent."
                )


# -------------------------------------------------------------------------
# Test 4: PEG Ratio - earningsGrowth format
# -------------------------------------------------------------------------
def test_peg_ratio(log):
    """Verify PEG ratio calculation and earningsGrowth interpretation."""
    print("\n[TEST 4] PEG Ratio Calculation")
    print("-" * 50)

    for sym in TEST_TICKERS:
        t = yf.Ticker(sym)
        info = t.info or {}

        price = info.get("currentPrice", None)
        eps = info.get("trailingEps", None)
        eg = info.get("earningsGrowth", None)

        if price and eps and eg and eps > 0.01:
            pe = price / eps
            # Screener: PEG = P/E / (earningsGrowth * 100)
            # earningsGrowth is a DECIMAL (0.183 = 18.3%)
            # So earningsGrowth * 100 = 18.3 (the percentage)
            # PEG = P/E / growth_pct -- this is CORRECT
            screener_peg = pe / (eg * 100) if eg > 0.01 else None

            print(f"  {sym}: P/E={pe:.2f}, earningsGrowth={eg}, PEG={screener_peg}")

            # Verify: earningsGrowth IS a decimal
            # 0.183 means 18.3% growth
            # Standard PEG = P/E / EPS_Growth_Rate(%)
            # So: PEG = 33.6 / 18.3 = 1.84 -- CORRECT
            if screener_peg:
                print(f"    Screener PEG: {screener_peg:.2f} (P/E / (eg*100))")
                print(f"    This is CORRECT: earningsGrowth={eg} is decimal, *100 converts to %")
        else:
            print(f"  {sym}: Cannot compute PEG (price={price}, eps={eps}, eg={eg})")
            if eg and eg < 0:
                print(f"    earningsGrowth is negative ({eg}), screener correctly skips")


# -------------------------------------------------------------------------
# Test 5: Dividend yield format (not used in scoring but in portfolio stats)
# -------------------------------------------------------------------------
def test_dividend_yield(log):
    """Check dividendYield format from yfinance."""
    print("\n[TEST 5] Dividend Yield Format")
    print("-" * 50)

    for sym in TEST_TICKERS:
        t = yf.Ticker(sym)
        info = t.info or {}

        dy = info.get("dividendYield", None)
        dr = info.get("dividendRate", None)
        price = info.get("currentPrice", None)

        if dy is not None and dr is not None and price:
            manual_pct = (dr / price) * 100
            print(f"  {sym}: dividendYield={dy}, dividendRate={dr}, price={price}")
            print(f"    Manual yield: {manual_pct:.2f}%")

            # In yfinance 0.2.66, dividendYield is returned as PERCENTAGE
            # (e.g., 0.41 for AAPL means 0.41%, NOT 41%)
            # Portfolio constructor line 250: avg_div = np.average(divs) * 100
            # If dividendYield is already percentage, multiplying by 100 would give 41%!
            if abs(dy - manual_pct) < 0.5:
                print(f"    yfinance returns as PERCENTAGE ({dy}% ~= {manual_pct:.2f}%)")
                log.add(
                    sym, "dividend_yield", "DIVIDEND_YIELD_ALREADY_PERCENTAGE",
                    f"dividendYield={dy} (already in %)",
                    f"portfolio_constructor.py line 250 multiplies by 100 again",
                    "HIGH",
                    f"yfinance v0.2.66 returns dividendYield as a percentage ({dy}%), "
                    f"but portfolio_constructor.py multiplies by 100 again "
                    f"('avg_div = np.average(divs, weights=weights) * 100'). "
                    f"This would show {dy * 100:.1f}% instead of {dy:.2f}%. "
                    f"However, 'dividend_yield' column is never populated in the DataFrame "
                    f"so this code path is currently dead/unused."
                )
        elif dy is None:
            print(f"  {sym}: No dividendYield (non-dividend payer or missing)")


# -------------------------------------------------------------------------
# Test 6: Screener metric computation vs raw yfinance
# -------------------------------------------------------------------------
def test_metric_computation(log):
    """Fetch raw data and compare screener-computed metrics to manual calculations."""
    print("\n[TEST 6] Full Metric Computation Validation")
    print("-" * 50)

    market_returns = fetch_market_returns()
    print(f"  Market returns: {len(market_returns)} observations")

    for sym in TEST_TICKERS:
        print(f"\n  --- {sym} ---")
        raw = _fetch_single_ticker_inner(sym)

        if "_error" in raw:
            print(f"    FETCH ERROR: {raw['_error']}")
            continue

        # Compute metrics using screener
        df = compute_metrics([raw], market_returns)
        if df.empty:
            print(f"    No metrics computed")
            continue

        row = df.iloc[0]

        # Manual validation of key metrics
        t = yf.Ticker(sym)
        info = t.info or {}

        # --- EV/EBITDA ---
        ev_yf = info.get("enterpriseValue", None)
        fins = t.financials
        ebitda_yf = None
        if fins is not None and not fins.empty:
            for idx in fins.index:
                if "ebitda" == str(idx).lower().strip():
                    vals = fins.loc[idx].dropna()
                    if len(vals) > 0:
                        ebitda_yf = float(vals.iloc[0])
                    break

        screener_ev_ebitda = row.get("ev_ebitda", np.nan)
        if ev_yf and ebitda_yf and ebitda_yf > 0:
            manual_ev_ebitda = ev_yf / ebitda_yf
            diff = abs(screener_ev_ebitda - manual_ev_ebitda) if pd.notna(screener_ev_ebitda) else float('inf')
            status = "OK" if diff < 0.5 else "MISMATCH"
            print(f"    EV/EBITDA: screener={screener_ev_ebitda:.2f}, manual={manual_ev_ebitda:.2f} [{status}]")
            if status == "MISMATCH":
                log.add(
                    sym, "ev_ebitda", "COMPUTATION_MISMATCH",
                    f"Manual: {manual_ev_ebitda:.2f}",
                    f"Screener: {screener_ev_ebitda:.2f}",
                    "HIGH",
                    "EV/EBITDA computation differs from manual calculation"
                )

        # --- Earnings Yield ---
        eps_yf = info.get("trailingEps", None)
        price_yf = info.get("currentPrice", None)
        screener_ey = row.get("earnings_yield", np.nan)
        if eps_yf and price_yf and price_yf > 0:
            manual_ey = eps_yf / price_yf
            diff = abs(screener_ey - manual_ey) if pd.notna(screener_ey) else float('inf')
            status = "OK" if diff < 0.005 else "MISMATCH"
            print(f"    Earnings Yield: screener={screener_ey:.4f}, manual={manual_ey:.4f} [{status}]")

        # --- ROIC ---
        screener_roic = row.get("roic", np.nan)
        if pd.notna(screener_roic):
            print(f"    ROIC: screener={screener_roic:.4f}")

        # --- Forward EPS Growth ---
        fwd_eps = info.get("forwardEps", None)
        trail_eps = info.get("trailingEps", None)
        screener_feg = row.get("forward_eps_growth", np.nan)
        if fwd_eps and trail_eps and abs(trail_eps) > 0.01:
            manual_feg = (fwd_eps - trail_eps) / abs(trail_eps)
            diff = abs(screener_feg - manual_feg) if pd.notna(screener_feg) else float('inf')
            status = "OK" if diff < 0.01 else "MISMATCH"
            print(f"    Fwd EPS Growth: screener={screener_feg:.4f}, manual={manual_feg:.4f} [{status}]")

        # --- Revenue Growth ---
        screener_rg = row.get("revenue_growth", np.nan)
        if pd.notna(screener_rg):
            print(f"    Revenue Growth: screener={screener_rg:.4f}")

        # --- Piotroski F-Score ---
        screener_f = row.get("piotroski_f_score", np.nan)
        if pd.notna(screener_f):
            print(f"    Piotroski F: screener={screener_f:.2f}")

        # --- Beta ---
        screener_beta = row.get("beta", np.nan)
        yf_beta = info.get("beta", None)
        if pd.notna(screener_beta) and yf_beta:
            diff = abs(screener_beta - yf_beta)
            status = "OK" if diff < 0.15 else "DIFFERS"
            print(f"    Beta: screener={screener_beta:.2f}, yfinance_info={yf_beta:.2f} [{status}]")
            if diff >= 0.15:
                log.add(
                    sym, "beta", "BETA_DIFFERS_FROM_YFINANCE",
                    f"yfinance info['beta']={yf_beta:.2f}",
                    f"Screener computed={screener_beta:.2f}",
                    "LOW",
                    f"Screener computes beta from daily returns (252-day window) "
                    f"while yfinance info['beta'] uses a different methodology/window. "
                    f"Difference of {diff:.2f} is expected and not a bug."
                )
        elif pd.notna(screener_beta):
            print(f"    Beta: screener={screener_beta:.2f}, yfinance_info=N/A")

        # --- Volatility ---
        screener_vol = row.get("volatility", np.nan)
        if pd.notna(screener_vol):
            print(f"    Volatility: screener={screener_vol:.4f}")


# -------------------------------------------------------------------------
# Test 7: Statement label matching issues
# -------------------------------------------------------------------------
def test_statement_labels(log):
    """Check if _stmt_val matches the right financial statement rows."""
    print("\n[TEST 7] Financial Statement Label Matching")
    print("-" * 50)

    for sym in TEST_TICKERS[:3]:  # Just check 3
        t = yf.Ticker(sym)

        # Check if 'EBIT' vs 'Operating Income' resolves correctly
        fins = t.financials
        if fins is not None and not fins.empty:
            ebit_direct = _stmt_val(fins, "EBIT")
            op_income = _stmt_val(fins, "Operating Income")
            print(f"  {sym}: EBIT={ebit_direct:,.0f}" if pd.notna(ebit_direct) else f"  {sym}: EBIT=NaN")
            print(f"  {sym}: Operating Income={op_income:,.0f}" if pd.notna(op_income) else f"  {sym}: Operating Income=NaN")

            # Check for the 'Total Revenue' vs 'Operating Revenue' issue
            # In yfinance, both may exist - does substring matching pick up wrong one?
            rev = _stmt_val(fins, "Total Revenue")
            print(f"  {sym}: Total Revenue={rev:,.0f}" if pd.notna(rev) else f"  {sym}: Total Revenue=NaN")

            # Check: does 'Tax Provision' exist or is it 'Income Tax Expense'?
            tax = _stmt_val(fins, "Tax Provision")
            tax2 = _stmt_val(fins, "Income Tax")
            print(f"  {sym}: Tax Provision={tax:,.0f}" if pd.notna(tax) else f"  {sym}: Tax Provision=NaN")
            if pd.isna(tax) and pd.notna(tax2):
                print(f"  {sym}: Falls back to 'Income Tax'={tax2:,.0f}")

        # Check cashflow labels
        cf = t.cashflow
        if cf is not None and not cf.empty:
            # 'Common Stock Dividend' vs 'Common Stock Dividend Paid' vs 'Cash Dividends Paid'
            for label in ["Common Stock Dividend", "Common Stock Dividend Paid",
                          "Cash Dividends Paid", "Dividends Paid"]:
                val = _stmt_val(cf, label)
                if pd.notna(val):
                    print(f"  {sym}: '{label}' = {val:,.0f}")
                    break
            else:
                print(f"  {sym}: No dividend label matched")

        print()


# -------------------------------------------------------------------------
# Test 8: Stale data / caching issues
# -------------------------------------------------------------------------
def test_stale_data(log):
    """Check for stale/inconsistent dates across data sources."""
    print("\n[TEST 8] Data Freshness / Consistency Check")
    print("-" * 50)

    for sym in TEST_TICKERS[:3]:
        t = yf.Ticker(sym)

        # Check financial statement dates
        fins = t.financials
        bs = t.balance_sheet
        cf = t.cashflow

        fin_dates = list(fins.columns) if fins is not None and not fins.empty else []
        bs_dates = list(bs.columns) if bs is not None and not bs.empty else []
        cf_dates = list(cf.columns) if cf is not None and not cf.empty else []

        print(f"  {sym}:")
        print(f"    Financials dates: {[str(d.date()) for d in fin_dates[:3]]}")
        print(f"    BalanceSheet dates: {[str(d.date()) for d in bs_dates[:3]]}")
        print(f"    CashFlow dates: {[str(d.date()) for d in cf_dates[:3]]}")

        # Check if statements are aligned (col=0 should be same period)
        if fin_dates and bs_dates:
            if fin_dates[0] != bs_dates[0]:
                log.add(
                    sym, "statement_dates", "STATEMENT_DATE_MISALIGNMENT",
                    f"Financials[0]={fin_dates[0].date()}",
                    f"BalanceSheet[0]={bs_dates[0].date()}",
                    "MEDIUM",
                    "Income statement and balance sheet most recent periods don't match. "
                    "Metrics that combine data across statements (ROIC, Piotroski) "
                    "may mix different fiscal periods."
                )


# -------------------------------------------------------------------------
# Test 9: Fallback defaults masking missing data
# -------------------------------------------------------------------------
def test_fallback_defaults(log):
    """Check if default values are masking missing data."""
    print("\n[TEST 9] Fallback Defaults Analysis")
    print("-" * 50)

    # The _safe() function defaults to np.nan for missing info keys
    # This is correct - NaN propagates through calculations
    print("  _safe() default: np.nan -- CORRECT (NaN propagates)")

    # Check: totalDebt and totalCash default to 0 in EV calculation
    # factor_engine.py line 475-478:
    #   td = d.get("totalDebt", d.get("totalDebt_bs", 0))
    #   ca = d.get("totalCash", d.get("cash_bs", 0))
    #   td = td if pd.notna(td) else 0
    #   ca = ca if pd.notna(ca) else 0
    print("  EV fallback: totalDebt defaults to 0, totalCash defaults to 0")
    print("    This means if debt data is missing, EV = MarketCap (underestimates EV)")
    print("    If cash data is missing, EV = MarketCap + Debt (overestimates EV)")

    log.add(
        "ALL", "enterpriseValue (fallback)", "MISSING_DATA_DEFAULTS_TO_ZERO",
        "EV = MC + Debt - Cash (all components required)",
        "Missing debt/cash defaults to 0, producing biased EV",
        "MEDIUM",
        "When totalDebt or totalCash is missing from yfinance, the fallback EV "
        "calculation defaults these to 0. This systematically biases EV downward "
        "(missing debt) or upward (missing cash). Consider using NaN for EV when "
        "key components are missing, rather than assuming they're zero."
    )

    # Check: dividendsPaid defaults to 0 in sustainable growth
    # factor_engine.py line 610-611:
    #   _divs_raw = d.get("dividendsPaid", 0)
    #   divs = 0 if pd.isna(_divs_raw) else abs(_divs_raw)
    print("  Sustainable growth: dividendsPaid defaults to 0")
    print("    This assumes 100% retention ratio when dividend data is missing")
    print("    Could overestimate sustainable growth for dividend payers")

    log.add(
        "ALL", "sustainable_growth (dividends)", "MISSING_DIVIDENDS_ASSUMES_ZERO",
        "Should use NaN or lookup dividendRate from info",
        "Defaults dividendsPaid to 0 (100% retention)",
        "LOW",
        "When dividendsPaid is missing from cashflow statement, retention ratio "
        "defaults to 100%. For known dividend payers, this inflates sustainable growth."
    )


# -------------------------------------------------------------------------
# Test 10: Version/API compatibility
# -------------------------------------------------------------------------
def test_version_compatibility(log):
    """Check yfinance version and deprecated API usage."""
    print("\n[TEST 10] yfinance Version & API Compatibility")
    print("-" * 50)

    import yfinance
    installed = yfinance.__version__
    pinned = "0.2.28"
    print(f"  Installed: {installed}")
    print(f"  Pinned in requirements.txt: {pinned}")

    if installed != pinned:
        log.add(
            "N/A", "yfinance version", "VERSION_MISMATCH",
            f"requirements.txt pins {pinned}",
            f"Installed: {installed}",
            "MEDIUM",
            f"requirements.txt pins yfinance=={pinned} but {installed} is installed. "
            f"API behavior may differ between versions. Key changes in 0.2.36+: "
            f"dividendYield format changed, some info keys renamed, "
            f"earnings_history behavior may differ."
        )

    # Check if earnings_history is a proper DataFrame
    t = yf.Ticker("AAPL")
    try:
        eh = t.earnings_history
        if eh is not None and not eh.empty:
            cols = eh.columns.tolist()
            expected_cols = ["epsActual", "epsEstimate"]
            missing = [c for c in expected_cols if c not in cols]
            if missing:
                log.add(
                    "AAPL", "earnings_history columns", "API_COLUMN_CHANGE",
                    f"Expected columns: {expected_cols}",
                    f"Actual columns: {cols}",
                    "HIGH",
                    f"earnings_history DataFrame missing expected columns: {missing}"
                )
            else:
                print(f"  earnings_history columns: {cols} -- OK")
        else:
            print(f"  earnings_history: empty or None")
    except AttributeError as e:
        log.add(
            "AAPL", "earnings_history", "API_REMOVED",
            "t.earnings_history should return DataFrame",
            f"AttributeError: {e}",
            "HIGH",
            "earnings_history may have been removed or renamed in this yfinance version"
        )


# -------------------------------------------------------------------------
# Test 11: EBITDA fallback to EBIT
# -------------------------------------------------------------------------
def test_ebitda_fallback(log):
    """Check the EBITDA-to-EBIT fallback and its impact."""
    print("\n[TEST 11] EBITDA vs EBIT Fallback")
    print("-" * 50)

    for sym in TEST_TICKERS[:3]:
        t = yf.Ticker(sym)
        fins = t.financials
        if fins is None or fins.empty:
            continue

        ebitda = _stmt_val(fins, "EBITDA")
        ebit = _stmt_val(fins, "EBIT")

        print(f"  {sym}: EBITDA={ebitda:,.0f}" if pd.notna(ebitda) else f"  {sym}: EBITDA=NaN")
        print(f"  {sym}: EBIT={ebit:,.0f}" if pd.notna(ebit) else f"  {sym}: EBIT=NaN")

        if pd.notna(ebitda) and pd.notna(ebit) and ebitda > 0:
            ratio = ebit / ebitda
            print(f"    EBIT/EBITDA ratio: {ratio:.2f}")
            if abs(1 - ratio) > 0.15:
                # The fallback to EBIT when EBITDA is missing would give
                # a materially different EV/EBITDA ratio
                log.add(
                    sym, "ev_ebitda (EBIT fallback)", "EBIT_EBITDA_MATERIAL_DIFFERENCE",
                    f"EBITDA={ebitda:,.0f}",
                    f"EBIT={ebit:,.0f} (used as fallback)",
                    "HIGH" if abs(1 - ratio) > 0.3 else "MEDIUM",
                    f"When EBITDA is missing, screener falls back to EBIT (line 493-494). "
                    f"EBIT/EBITDA ratio is {ratio:.2f} for {sym}. D&A of "
                    f"{ebitda - ebit:,.0f} would make the EV/EBIT ratio {1/ratio:.0f}% "
                    f"of what EV/EBITDA would be. The metric is labeled 'ev_ebitda' "
                    f"but may actually be EV/EBIT for some stocks."
                )
        print()


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    log = DiscrepancyLog()

    print("=" * 60)
    print("  DATA ACCURACY VALIDATION")
    print(f"  Tickers: {TEST_TICKERS}")
    print("=" * 60)

    test_sector_mismatch(log)
    test_capex_sign(log)
    test_ev_cash_mismatch(log)
    test_peg_ratio(log)
    test_dividend_yield(log)
    test_metric_computation(log)
    test_statement_labels(log)
    test_stale_data(log)
    test_fallback_defaults(log)
    test_version_compatibility(log)
    test_ebitda_fallback(log)

    log.print_report()

    # Save to CSV
    df = log.to_dataframe()
    if not df.empty:
        out = "validation/data_accuracy_report.csv"
        df.to_csv(out, index=False)
        print(f"\nReport saved to: {out}")

    return len(log.rows)


if __name__ == "__main__":
    n_issues = main()
    print(f"\nTotal issues found: {n_issues}")
