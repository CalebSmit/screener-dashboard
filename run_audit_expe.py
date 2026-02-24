#!/usr/bin/env python3
"""
EXPE Metric Audit — Before/After Comparison

Fetches EXPE data from yfinance and computes each of the 7 metrics
using both current (pre-fix) logic and proposed fix logic.
Prints raw yfinance fields and a comparison table.
"""

import numpy as np
import warnings

def fmt(v, pct=False, mult=False):
    """Format a value for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    if pct:
        return f"{v*100:.1f}%"
    if mult:
        return f"{v:.2f}x"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"{v:.4f}"


def main():
    import yfinance as yf
    import pandas as pd

    print("=" * 70)
    print("  EXPE METRIC AUDIT — Raw Fields & Before/After Comparison")
    print("=" * 70)

    t = yf.Ticker("EXPE")
    info = t.info or {}
    fins = t.financials
    bs = t.balance_sheet
    cf = t.cashflow

    # Helper to pull from statements
    def sv(stmt, label, col=0):
        if stmt is None or stmt.empty:
            return np.nan
        target = label.lower().strip()
        for idx in stmt.index:
            if target == str(idx).lower().strip():
                vals = stmt.loc[idx].dropna()
                if len(vals) > col:
                    return float(vals.iloc[col])
        for idx in stmt.index:
            idx_low = str(idx).lower().strip()
            if idx_low.startswith(target) or idx_low.endswith(target):
                vals = stmt.loc[idx].dropna()
                if len(vals) > col:
                    return float(vals.iloc[col])
        return np.nan

    def si(key, default=np.nan):
        v = info.get(key, default)
        return default if v is None else v

    # -- Raw Data --
    print("\n-- Raw yfinance Fields --\n")

    ev = si("enterpriseValue")
    mc = si("marketCap")
    ebit = sv(fins, "EBIT")
    if np.isnan(ebit):
        ebit = sv(fins, "Operating Income")
    ebitda_reported = sv(fins, "EBITDA")
    da_cf = sv(cf, "Depreciation And Amortization")
    if np.isnan(da_cf):
        da_cf = sv(cf, "Reconciled Depreciation")
    if np.isnan(da_cf):
        da_cf = sv(cf, "Depreciation Amortization Depletion")
    fwd_eps = si("forwardEps")
    trail_eps = si("trailingEps")
    cur_price = si("currentPrice")
    total_debt_info = si("totalDebt")
    total_debt_bs = sv(bs, "Total Debt")
    total_cash_bs = sv(bs, "Cash And Cash Equivalents")
    total_equity = sv(bs, "Stockholders Equity")
    total_equity_prior = sv(bs, "Stockholders Equity", 1)
    total_assets = sv(bs, "Total Assets")
    total_assets_prior = sv(bs, "Total Assets", 1)
    total_revenue = sv(fins, "Total Revenue")
    net_income = sv(fins, "Net Income")
    tax_exp = sv(fins, "Tax Provision")
    if np.isnan(tax_exp):
        tax_exp = sv(fins, "Income Tax")
    pretax = sv(fins, "Pretax Income")
    divs_paid = sv(cf, "Common Stock Dividend")
    if np.isnan(divs_paid):
        divs_paid = sv(cf, "Dividends Paid")
    payout_ratio_info = si("payoutRatio")

    fields = [
        ("enterpriseValue", ev),
        ("marketCap", mc),
        ("EBIT (financials)", ebit),
        ("EBITDA (financials, reported)", ebitda_reported),
        ("D&A (cashflow)", da_cf),
        ("forwardEps", fwd_eps),
        ("trailingEps", trail_eps),
        ("currentPrice", cur_price),
        ("totalDebt (info)", total_debt_info),
        ("totalDebt (balance sheet)", total_debt_bs),
        ("Cash (balance sheet)", total_cash_bs),
        ("Stockholders Equity (yr 0)", total_equity),
        ("Stockholders Equity (yr 1)", total_equity_prior),
        ("Total Assets (yr 0)", total_assets),
        ("Total Assets (yr 1)", total_assets_prior),
        ("Total Revenue", total_revenue),
        ("Net Income", net_income),
        ("Tax Provision", tax_exp),
        ("Pretax Income", pretax),
        ("Dividends Paid (cashflow)", divs_paid),
        ("payoutRatio (info)", payout_ratio_info),
    ]
    for name, val in fields:
        print(f"  {name:40s} = {fmt(val)}")

    # -- Metric Computations --
    print("\n" + "=" * 70)
    print("  METRIC COMPUTATIONS: CURRENT vs FIXED")
    print("=" * 70)

    results = []

    # --- Fix 1: EV/EBITDA ---
    print("\n-- Fix 1: EV/EBITDA --")
    old_ebitda = ebitda_reported
    if pd.notna(ebit) and pd.notna(da_cf) and da_cf >= 0:
        new_ebitda = ebit + da_cf
    else:
        new_ebitda = ebitda_reported
    old_ev_ebitda = (ev / old_ebitda) if (pd.notna(ev) and pd.notna(old_ebitda) and old_ebitda > 0 and ev > 0) else np.nan
    new_ev_ebitda = (ev / new_ebitda) if (pd.notna(ev) and pd.notna(new_ebitda) and new_ebitda > 0 and ev > 0) else np.nan
    print(f"  EBITDA (reported):     {fmt(old_ebitda)}")
    print(f"  EBITDA (EBIT + D&A):   {fmt(new_ebitda)}")
    print(f"  EBIT = {fmt(ebit)}, D&A = {fmt(da_cf)}")
    print(f"  Current EV/EBITDA:     {fmt(old_ev_ebitda, mult=True)}")
    print(f"  Fixed EV/EBITDA:       {fmt(new_ev_ebitda, mult=True)}")
    results.append(("EV/EBITDA", old_ev_ebitda, new_ev_ebitda, "12-15x"))

    # --- Fix 2: Forward EPS Growth ---
    print("\n-- Fix 2: Forward EPS Growth --")
    if pd.notna(fwd_eps) and pd.notna(trail_eps) and abs(trail_eps) > 0.01:
        raw_growth = (fwd_eps - trail_eps) / max(abs(trail_eps), 1.0)
        old_feg = float(np.clip(raw_growth, -0.75, 3.0))
        new_feg = float(np.clip(raw_growth, -0.75, 1.50))
    else:
        old_feg = np.nan
        new_feg = np.nan
    print(f"  forwardEps = {fwd_eps}, trailingEps = {trail_eps}")
    print(f"  Raw growth = {fmt(raw_growth if pd.notna(fwd_eps) else np.nan, pct=True)}")
    print(f"  Current (clamp [-75%, +300%]): {fmt(old_feg, pct=True)}")
    print(f"  Fixed   (clamp [-75%, +150%]): {fmt(new_feg, pct=True)}")
    results.append(("Forward EPS Growth", old_feg, new_feg, "15-25%"))

    # --- Fix 3: PEG Ratio ---
    print("\n-- Fix 3: PEG Ratio --")
    pe = (cur_price / trail_eps) if (pd.notna(cur_price) and pd.notna(trail_eps) and trail_eps > 0.01) else np.nan
    old_peg = (pe / (old_feg * 100)) if (pd.notna(pe) and pd.notna(old_feg) and old_feg > 0.01) else np.nan
    new_peg = (pe / (new_feg * 100)) if (pd.notna(pe) and pd.notna(new_feg) and new_feg > 0.01) else np.nan
    print(f"  P/E = {fmt(pe, mult=True)}")
    print(f"  Current PEG: {fmt(old_peg, mult=True)}")
    print(f"  Fixed PEG:   {fmt(new_peg, mult=True)}")
    results.append(("PEG Ratio", old_peg, new_peg, "1.0-2.5x"))

    # --- Fix 4: ROIC ---
    print("\n-- Fix 4: ROIC --")
    if pd.notna(ebit):
        tax_rate = 0.21
        if pd.notna(tax_exp) and pd.notna(pretax) and pretax > 0:
            tax_rate = max(0, min(tax_exp / pretax, 0.5))
        nopat = ebit * (1 - tax_rate)
    else:
        nopat = np.nan
        tax_rate = np.nan
    print(f"  EBIT = {fmt(ebit)}, tax_rate = {tax_rate:.2%}" if pd.notna(tax_rate) else f"  EBIT = NaN")
    print(f"  NOPAT = {fmt(nopat)}")

    if pd.notna(total_equity) and pd.notna(total_debt_bs) and pd.notna(total_cash_bs):
        op_cash = 0.02 * total_revenue if pd.notna(total_revenue) and total_revenue > 0 else 0
        excess_cash_old = max(0, total_cash_bs - op_cash)
        ic_old = total_equity + total_debt_bs - excess_cash_old

        excess_cash_new = max(0, total_cash_bs - op_cash)
        excess_cash_new = min(excess_cash_new, 0.5 * total_cash_bs)  # cap at 50%
        ic_new = total_equity + total_debt_bs - excess_cash_new
        if pd.notna(total_assets) and total_assets > 0:
            ic_new = max(ic_new, 0.10 * total_assets)  # floor at 10% TA

        old_roic = (nopat / ic_old) if (pd.notna(nopat) and ic_old > 0) else np.nan
        new_roic = (nopat / ic_new) if (pd.notna(nopat) and ic_new > 0) else np.nan
    else:
        ic_old = np.nan
        ic_new = np.nan
        old_roic = np.nan
        new_roic = np.nan
        op_cash = np.nan
        excess_cash_old = np.nan
        excess_cash_new = np.nan

    print(f"  Total Equity = {fmt(total_equity)}")
    print(f"  Total Debt (BS) = {fmt(total_debt_bs)}")
    print(f"  Total Cash (BS) = {fmt(total_cash_bs)}")
    print(f"  Operating Cash (2% Rev) = {fmt(op_cash)}")
    print(f"  Excess Cash (current): {fmt(excess_cash_old)}")
    print(f"  Excess Cash (fixed, 50% cap): {fmt(excess_cash_new)}")
    print(f"  IC (current): {fmt(ic_old)}")
    print(f"  IC (fixed, 10% TA floor): {fmt(ic_new)}")
    print(f"  10% of Total Assets = {fmt(0.10 * total_assets if pd.notna(total_assets) else np.nan)}")
    print(f"  Current ROIC: {fmt(old_roic, pct=True)}")
    print(f"  Fixed ROIC:   {fmt(new_roic, pct=True)}")
    results.append(("ROIC", old_roic, new_roic, "10-20%"))

    # --- Fix 5: Debt/Equity ---
    print("\n-- Fix 5: Debt/Equity --")
    old_de = (total_debt_info / total_equity) if (pd.notna(total_debt_info) and pd.notna(total_equity) and total_equity > 0) else np.nan
    new_de = (total_debt_bs / total_equity) if (pd.notna(total_debt_bs) and pd.notna(total_equity) and total_equity > 0) else np.nan
    print(f"  totalDebt (info) = {fmt(total_debt_info)}")
    print(f"  totalDebt (BS)   = {fmt(total_debt_bs)}")
    print(f"  Equity           = {fmt(total_equity)}")
    print(f"  Current D/E (info): {fmt(old_de, mult=True)}")
    print(f"  Fixed D/E (BS):     {fmt(new_de, mult=True)}")
    results.append(("Debt/Equity", old_de, new_de, "2.4-3.5x"))

    # --- Fix 6: Sustainable Growth Rate ---
    print("\n-- Fix 6: Sustainable Growth Rate --")
    if pd.notna(net_income) and pd.notna(total_equity) and total_equity > 0 and net_income > 0:
        old_roe = net_income / total_equity
        if pd.notna(total_equity_prior) and total_equity_prior > 0:
            avg_eq = (total_equity + total_equity_prior) / 2
        else:
            avg_eq = total_equity
        new_roe = net_income / avg_eq

        # Old: use dividendsPaid
        if pd.notna(divs_paid):
            divs = abs(divs_paid)
            old_ret = max(0, 1 - divs / net_income)
        else:
            old_ret = np.nan
        old_sgr = (old_roe * old_ret) if pd.notna(old_ret) else np.nan

        # New: prefer payoutRatio from info
        if pd.notna(payout_ratio_info) and 0 <= payout_ratio_info <= 2.0:
            new_ret = max(0, 1 - min(payout_ratio_info, 1.0))
        elif pd.notna(divs_paid):
            divs = abs(divs_paid)
            new_ret = max(0, 1 - divs / net_income)
        else:
            new_ret = np.nan
        new_sgr = float(np.clip(new_roe * new_ret, 0, 1)) if pd.notna(new_ret) else np.nan
    else:
        old_roe = np.nan
        new_roe = np.nan
        old_sgr = np.nan
        new_sgr = np.nan
        old_ret = np.nan
        new_ret = np.nan

    print(f"  Net Income = {fmt(net_income)}")
    print(f"  Equity (current yr) = {fmt(total_equity)}")
    print(f"  Equity (prior yr)   = {fmt(total_equity_prior)}")
    print(f"  ROE (current, single yr): {fmt(old_roe, pct=True)}")
    print(f"  ROE (fixed, avg equity):  {fmt(new_roe, pct=True)}")
    print(f"  payoutRatio (info) = {payout_ratio_info}")
    print(f"  dividendsPaid (CF) = {fmt(divs_paid)}")
    print(f"  Retention (current): {fmt(old_ret, pct=True)}")
    print(f"  Retention (fixed):   {fmt(new_ret, pct=True)}")
    print(f"  Current SGR: {fmt(old_sgr, pct=True)}")
    print(f"  Fixed SGR:   {fmt(new_sgr, pct=True)}")
    results.append(("Sustainable Growth", old_sgr, new_sgr, "40-50%"))

    # --- Fix 7: Asset Growth ---
    print("\n-- Fix 7: Asset Growth --")
    ag = ((total_assets - total_assets_prior) / total_assets_prior) if (pd.notna(total_assets) and pd.notna(total_assets_prior) and total_assets_prior > 0) else np.nan
    print(f"  Total Assets (yr 0) = {fmt(total_assets)}")
    print(f"  Total Assets (yr 1) = {fmt(total_assets_prior)}")
    print(f"  Asset Growth: {fmt(ag, pct=True)}")
    print(f"  (Uses annual balance sheet — confirmed, no change needed)")
    results.append(("Asset Growth", ag, ag, "3-8%"))

    # -- Summary Table --
    print("\n" + "=" * 70)
    print("  SUMMARY: BEFORE vs AFTER")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'Current':>12s} {'Fixed':>12s} {'Benchmark':>14s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*14}")
    for name, old, new, bench in results:
        if "Growth" in name or "ROIC" in name or "Sustainable" in name:
            o = fmt(old, pct=True)
            n = fmt(new, pct=True)
        elif "EV/EBITDA" in name or "PEG" in name or "Debt" in name:
            o = fmt(old, mult=True)
            n = fmt(new, mult=True)
        else:
            o = fmt(old, pct=True)
            n = fmt(new, pct=True)
        print(f"  {name:<25s} {o:>12s} {n:>12s} {bench:>14s}")
    print()


if __name__ == "__main__":
    main()
