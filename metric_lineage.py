"""Where every scored metric comes from: formula, inputs, caveats - and a check.

WHY THIS EXISTS - 2026-10-07 (``plan/calculation-transparency.md``, stages T1/T2).

The owner asked to *see the numbers going into the calculations* behind each score.
This is the single table behind that: for every metric the dashboard publishes it says
what the formula is, which reported figures feed it, what period they cover, and where
the scorer does something a reader would not guess from the name. The drilldown renders
it; ``tests/test_metric_lineage.py`` holds it to account.

Three rules, all learned from the 2026-10-06 audit:

1. **The inputs are the figures the scorer used, as fetched.** ``ev_used`` is the EV the
   scorer settled on (Yahoo's value unless it failed a cross-check), not the raw API
   field the page used to show. Nothing is clipped or rounded for display.
2. **An equation is only shown when it reproduces.** ``RECOMPUTE`` holds a function for
   each metric whose value can be rebuilt from the inputs published beside it. The build
   and the tests rebuild it for every stock; a stock whose inputs do not reproduce its
   value is flagged on the page rather than shown with a formula that does not hold.
   Metrics built from a price series, analyst history or a provider ratio say so and show
   no equation.
3. **Caveats are written down, not discovered.** Where the code differs from the label
   (a "1-year" growth rate that spans 12-21 months, a "6-month" return that skips the
   latest month, a ratio where negative values rank best) the registry says so in words.
   These are *descriptions of what the code does*, not changes to it - methodology
   changes need research (CLAUDE.md rule 4) and are tracked as open items.

Display-only: nothing here feeds a metric, a weight or a rank.
"""

from __future__ import annotations

import math

# fmt codes understood by the page: usd ($B/$M), price ($0.00), pct (fraction -> %),
# ratio (0.00), num (as is), shares (millions)
USD, PRICE, PCT, RATIO, NUM = "usd", "price", "pct", "ratio", "num"

# Clamp bounds the scorer applies (config.yaml metric_clamps); pinned by a test.
FEG_CLAMP = (-0.75, 1.50)
PTU_CLAMP = (-0.50, 1.00)
PEG_CAP = 50.0
MIN_ANALYSTS = 3


def _ok(*xs):
    return all(x is not None and not (isinstance(x, float) and math.isnan(x)) for x in xs)


def _first(i, *keys):
    for k in keys:
        v = i.get(k)
        if _ok(v):
            return v
    return None


# --------------------------------------------------------------------------- recompute
def _ev_ebitda(i):
    ev, e = i.get("ev_used"), i.get("ebitda_used")
    return ev / e if _ok(ev, e) and e > 0 and ev > 0 else None


def _fcf_yield(i):
    f, ev = i.get("fcf_used"), i.get("ev_used")
    return f / ev if _ok(f, ev) and ev > 0 else None


def _earnings_yield(i):
    ni, mc = i.get("netIncome"), i.get("marketCap")
    return ni / mc if _ok(ni, mc) and mc > 0 else None


def _ev_sales(i):
    ev, r = i.get("ev_used"), i.get("totalRevenue")
    return ev / r if _ok(ev, r) and r > 0 and ev > 0 else None


def _pb(i):
    pb = i.get("priceToBook")
    if _ok(pb) and pb > 0:
        return pb
    bv, p = i.get("bookValue"), i.get("currentPrice")
    return p / bv if _ok(bv, p) and bv > 0 else None


def _roic(i):
    ebit = i.get("ebit")
    if not _ok(ebit):
        return None
    tax, pre = i.get("incomeTaxExpense"), i.get("pretaxIncome")
    rate = 0.21
    if _ok(pre) and pre <= 0:
        rate = 0.0
    elif _ok(tax, pre) and pre > 0:
        rate = max(0.0, min(tax / pre, 0.5))
    nopat = ebit * (1 - rate)
    eq = i.get("totalEquity")
    debt = _first(i, "totalDebt_bs", "totalDebt")
    cash = _first(i, "cash_bs", "totalCash")
    if not _ok(eq, debt, cash):
        return None
    rev = i.get("totalRevenue")
    operating = 0.02 * rev if _ok(rev) and rev > 0 else 0.0
    excess = min(max(0.0, cash - operating), 0.5 * cash)
    ic = eq + debt - excess
    ta = i.get("totalAssets")
    if _ok(ta) and ta > 0:
        ic = max(ic, 0.10 * ta)
    return nopat / ic if ic > 0 else None


def _gpa(i):
    gp, ta = i.get("grossProfit"), i.get("totalAssets")
    return gp / ta if _ok(gp, ta) and ta > 0 else None


def _accruals(i):
    ni, ocf, ta = i.get("netIncome"), i.get("operatingCashFlow"), i.get("totalAssets")
    return (ni - ocf) / ta if _ok(ni, ocf, ta) and ta > 0 else None


def _net_debt_ebitda(i):
    debt = _first(i, "totalDebt_bs", "totalDebt")
    e = i.get("ebitda_nd_used")
    if not (_ok(debt) and _ok(e) and e > 0):
        return None
    cash = _first(i, "cash_bs", "totalCash") or 0.0
    nd = debt - cash
    return 0.0 if nd <= 0 else nd / e


def _op_leverage(i):
    e0 = _first(i, "ebit_annual", "ebit")
    e1 = i.get("ebit_prior")
    r0 = _first(i, "totalRevenue_annual", "totalRevenue")
    r1 = _first(i, "totalRevenue_annual_prior", "totalRevenue_prior")
    if not _ok(e0, e1, r0, r1) or e1 == 0 or r1 <= 0 or (e0 > 0) != (e1 > 0):
        return None
    rc = (r0 - r1) / abs(r1)
    if abs(rc) < 0.01:
        return None
    return ((e0 - e1) / abs(e1)) / rc


def _feg(i):
    f, t = i.get("forwardEps"), i.get("trailingEps")
    if not (_ok(f, t) and abs(t) > 0.01):
        return None
    ratio = f / t
    if ratio > 2.0 or ratio < 0.3:
        return None
    g = (f - t) / max(abs(t), 1.0)
    return min(max(g, FEG_CLAMP[0]), FEG_CLAMP[1])


def _peg(i):
    p, t, g = _first(i, "currentPrice", "price_latest"), i.get("trailingEps"), _feg(i)
    if not (_ok(p, t) and t > 0.01 and g is not None and g > 0):
        return None
    return min((p / t) / (g * 100), PEG_CAP)


def _rev_growth(i):
    r, p = i.get("totalRevenue"), i.get("totalRevenue_prior")
    return (r - p) / p if _ok(r, p) and p > 0 else None


def _rev_cagr(i):
    r3 = i.get("totalRevenue_3yr_ago")
    r0 = _first(i, "totalRevenue_annual", "totalRevenue")
    return (r0 / r3) ** (1 / 3) - 1 if _ok(r0, r3) and r3 > 0 and r0 > 0 else None


def _ptu(i):
    t, p, n = i.get("targetMeanPrice"), _first(i, "currentPrice", "price_latest"), i.get("numberOfAnalystOpinions")
    if not (_ok(t, p, n) and p > 0 and n >= MIN_ANALYSTS):
        return None
    return min(max((t - p) / p, PTU_CLAMP[0]), PTU_CLAMP[1])


def _short(i):
    s = i.get("shortRatio")
    return s if _ok(s) and s >= 0 else None


def _size(i):
    mc = i.get("marketCap")
    return -math.log(mc) if _ok(mc) and mc > 0 else None


def _asset_growth(i):
    a, p = i.get("totalAssets"), i.get("totalAssets_prior")
    return (a - p) / p if _ok(a, p) and p > 0 else None


def _equity_ratio(i):
    e, a = i.get("totalEquity"), i.get("totalAssets")
    return e / a if _ok(e, a) and a > 0 else None


def _roe(i):
    y = i.get("returnOnEquity")
    if _ok(y):
        return y
    ni, e = i.get("netIncome"), i.get("totalEquity")
    return ni / e if _ok(ni, e) and e > 0 else None


def _roa(i):
    y = i.get("returnOnAssets")
    if _ok(y):
        return y
    ni, a = i.get("netIncome"), i.get("totalAssets")
    return ni / a if _ok(ni, a) and a > 0 else None


def _ret_12_1(i):
    a, b = i.get("price_12m_ago"), i.get("price_1m_ago")
    return (b - a) / a if _ok(a, b) and a > 0 else None


def _ret_6m(i):
    a, b = i.get("price_6m_ago"), i.get("price_1m_ago")
    return (b - a) / a if _ok(a, b) and a > 0 else None


def _fy1_rev(i):
    c, a, p = i.get("_fy1_eps_current"), i.get("_fy1_eps_90d_ago"), _first(i, "currentPrice", "price_latest")
    return (c - a) / p if _ok(c, a, p) and p > 0 else None


RECOMPUTE = {
    "ev_ebitda": _ev_ebitda, "fcf_yield": _fcf_yield, "earnings_yield": _earnings_yield,
    "ev_sales": _ev_sales, "pb_ratio": _pb, "roic": _roic, "gross_profit_assets": _gpa,
    "accruals": _accruals, "net_debt_to_ebitda": _net_debt_ebitda,
    "operating_leverage": _op_leverage, "forward_eps_growth": _feg, "peg_ratio": _peg,
    "revenue_growth": _rev_growth, "revenue_cagr_3yr": _rev_cagr,
    "price_target_upside": _ptu, "short_interest_ratio": _short, "size_log_mcap": _size,
    "asset_growth": _asset_growth, "equity_ratio": _equity_ratio, "roe": _roe, "roa": _roa,
    "return_12_1": _ret_12_1, "return_6m": _ret_6m, "fy1_revision_3m": _fy1_rev,
}


# --------------------------------------------------------------------------- the table
def _L(formula, inputs=(), caveat=None, how=None, kind="ratio"):
    return {"formula": formula, "how": how, "inputs": [list(x) for x in inputs],
            "caveat": caveat, "kind": kind}


EV_NOTE = ("Uses the scorer's own enterprise value: Yahoo's figure, unless it is missing or "
           "more than 10% (25% for financials) away from market cap + debt - cash, in which "
           "case that sum is used.")
GROWTH_WINDOW = ("The 'prior' figure is the fiscal year before the latest completed fiscal year, "
                 "so this covers roughly 12-21 months, not exactly one year.")

LINEAGE = {
    # ---- valuation
    "ev_ebitda": _L("Enterprise value / EBITDA",
                    [("Enterprise value (used)", "ev_used", USD), ("EBITDA (used)", "ebitda_used", USD),
                     ("EBIT", "ebit", USD), ("D&A (cash-flow)", "da_cf", USD)],
                    how="EBITDA = trailing-12-month EBIT + depreciation & amortisation; Yahoo's reported EBITDA is used only when either is missing. Needs EBITDA > 0.",
                    caveat=EV_NOTE),
    "fcf_yield": _L("Free cash flow / enterprise value",
                    [("Free cash flow (used)", "fcf_used", USD), ("Enterprise value (used)", "ev_used", USD),
                     ("Operating cash flow", "operatingCashFlow", USD), ("Capital expenditure", "capex", USD)],
                    how="Free cash flow = trailing-12-month operating cash flow - capital expenditure. Divided by enterprise value, not market cap.",
                    caveat=EV_NOTE),
    "earnings_yield": _L("Net income / market cap",
                         [("Net income (TTM)", "netIncome", USD), ("Market cap", "marketCap", USD)],
                         how="One definition for every stock: trailing-12-month net income over market cap. Negative earnings give a negative yield."),
    "ev_sales": _L("Enterprise value / revenue",
                   [("Enterprise value (used)", "ev_used", USD), ("Revenue (TTM)", "totalRevenue", USD)],
                   caveat=EV_NOTE),
    "pb_ratio": _L("Price / book value",
                   [("Price / book (Yahoo)", "priceToBook", RATIO), ("Book value per share", "bookValue", PRICE),
                    ("Price", "currentPrice", PRICE)],
                   how="Yahoo's price-to-book when positive, otherwise price / book value per share. Banks and insurers only."),
    # ---- quality
    "roic": _L("After-tax operating profit / invested capital",
               [("EBIT (TTM)", "ebit", USD), ("Income tax", "incomeTaxExpense", USD), ("Pre-tax income", "pretaxIncome", USD),
                ("Equity", "totalEquity", USD), ("Debt (balance sheet)", "totalDebt_bs", USD),
                ("Cash (balance sheet)", "cash_bs", USD), ("Revenue (TTM)", "totalRevenue", USD),
                ("Total assets", "totalAssets", USD)],
               how="NOPAT = EBIT x (1 - tax rate); the tax rate is tax / pre-tax income capped at 50%, zero for a loss, 21% if unreported. Invested capital = equity + debt - excess cash (cash above 2% of revenue, at most half of cash), floored at 10% of total assets."),
    "gross_profit_assets": _L("Gross profit / total assets",
                              [("Gross profit (TTM)", "grossProfit", USD), ("Total assets", "totalAssets", USD)],
                              how="Novy-Marx's gross profitability. Uses ending, not average, assets."),
    "net_debt_to_ebitda": _L("Net debt / EBITDA",
                             [("Debt (balance sheet)", "totalDebt_bs", USD), ("Cash (balance sheet)", "cash_bs", USD),
                              ("EBITDA (used)", "ebitda_nd_used", USD)],
                             how="Net debt = debt - cash, and a net-cash company is set to exactly 0.0. Needs EBITDA > 0.",
                             caveat="This EBITDA is EBIT + D&A when D&A is non-negative, which can differ from the EBITDA used for EV/EBITDA and from the figure in Company Snapshot."),
    "piotroski_f_score": _L("Count of nine pass/fail financial-health signals",
                            how="Each signal is 1 (pass) or 0 (fail); a signal whose inputs are missing is not testable and is neither. A score needs at least 6 testable signals, so scores based on 6 and on 9 signals are not strictly comparable.",
                            caveat="Signal 1 is 'net income > 0' (the published test uses ROA > 0, which has the same sign). Comparisons use this year's and the prior fiscal year's statements, with the prior-year window described under Revenue growth.",
                            kind="components"),
    "accruals": _L("(Net income - operating cash flow) / total assets",
                   [("Net income (TTM)", "netIncome", USD), ("Operating cash flow (TTM)", "operatingCashFlow", USD),
                    ("Total assets", "totalAssets", USD)],
                   how="Sloan's accruals, cash-flow form. Lower is better, so negative accruals rank best."),
    "operating_leverage": _L("% change in EBIT / % change in revenue (annual)",
                             [("EBIT (annual)", "ebit_annual", USD), ("EBIT (prior year)", "ebit_prior", USD),
                              ("Revenue (annual)", "totalRevenue_annual", USD),
                              ("Revenue (prior year)", "totalRevenue_annual_prior", USD)],
                             how="Missing when EBIT changes sign, prior EBIT is zero, or revenue moves less than 1%.",
                             caveat="Lower is scored as better and negative values - EBIT falling while revenue rises, or the reverse - are not treated specially, so they rank at the top of the sector. This is recorded as an open research item."),
    "beneish_m_score": _L("Beneish (1999) eight-index manipulation score",
                          how="M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI - 0.172 SGAI + 4.679 TATA - 0.327 LVGI, from annual statements. Lower is better. Needs at least 5 of the 8 indices computed from real data; the rest default to neutral values.",
                          caveat="TATA uses the cash-flow form of accruals, not Beneish's original balance-sheet form.",
                          kind="components"),
    "roe": _L("Return on equity (banks and insurers)",
              [("Return on equity (Yahoo)", "returnOnEquity", PCT), ("Net income (TTM)", "netIncome", USD),
               ("Equity", "totalEquity", USD)],
              how="Yahoo's ratio when it reports one, otherwise net income / ending equity."),
    "roa": _L("Return on assets (banks and insurers)",
              [("Return on assets (Yahoo)", "returnOnAssets", PCT), ("Net income (TTM)", "netIncome", USD),
               ("Total assets", "totalAssets", USD)],
              how="Yahoo's ratio when it reports one, otherwise net income / ending assets."),
    "equity_ratio": _L("Equity / total assets (banks and insurers)",
                       [("Equity", "totalEquity", USD), ("Total assets", "totalAssets", USD)]),
    # ---- growth
    "forward_eps_growth": _L("(Forward EPS - trailing EPS) / max(|trailing EPS|, $1)",
                             [("Forward EPS", "forwardEps", PRICE), ("Trailing EPS", "trailingEps", PRICE)],
                             how="Clipped to -75% .. +150%. Left out when the two EPS figures are on very different bases (forward / trailing above 2x or below 0.3x) or trailing EPS is near zero.",
                             caveat="Trailing EPS is as reported (GAAP) while forward EPS is the analyst consensus, which is usually adjusted."),
    "peg_ratio": _L("(Price / trailing EPS) / (forward EPS growth x 100)",
                    [("Price", "currentPrice", PRICE), ("Trailing EPS", "trailingEps", PRICE),
                     ("Forward EPS", "forwardEps", PRICE)],
                    how="Capped at 50; missing unless growth and trailing EPS are positive."),
    "revenue_growth": _L("(Revenue - prior revenue) / prior revenue",
                         [("Revenue (TTM)", "totalRevenue", USD), ("Prior revenue", "totalRevenue_prior", USD)],
                         caveat=GROWTH_WINDOW),
    "revenue_cagr_3yr": _L("(Revenue / revenue three years earlier) ^ (1/3) - 1",
                           [("Revenue (latest annual)", "totalRevenue_annual", USD),
                            ("Revenue (three years earlier)", "totalRevenue_3yr_ago", USD)],
                           how="Both endpoints are annual statements."),
    "sustainable_growth": _L("Return on equity x retention ratio",
                             [("Net income (TTM)", "netIncome", USD), ("Equity", "totalEquity", USD),
                              ("Equity (year earlier)", "totalEquity_prior", USD),
                              ("Dividends paid", "dividendsPaid", USD), ("Payout ratio", "payoutRatio", PCT)],
                             how="ROE uses average equity; retention = 1 - dividends / net income. Clipped to 0..100%. Needs positive net income and equity.",
                             caveat="This ROE is not the Yahoo ROE shown for banks."),
    # ---- momentum
    "return_12_1": _L("(Price 1 month ago - price 12 months ago) / price 12 months ago",
                      [("Price ~12 months ago", "price_12m_ago", PRICE), ("Price ~1 month ago", "price_1m_ago", PRICE)],
                      how="Dividend- and split-adjusted closing prices; 'a month' is 30 calendar days.",
                      caveat="Skips the most recent month on purpose (Jegadeesh & Titman)."),
    "return_6m": _L("(Price 1 month ago - price ~6 months ago) / price ~6 months ago",
                    [("Price ~6 months ago", "price_6m_ago", PRICE), ("Price ~1 month ago", "price_1m_ago", PRICE)],
                    caveat="Despite the '6M' label this is the return from about six months ago to about one month ago - it skips the latest month, like 12-1."),
    "jensens_alpha": _L("12-month return - [risk-free + beta x (market return - risk-free)]",
                        [("Price now", "price_latest", PRICE), ("Price ~12 months ago", "price_12m_ago", PRICE)],
                        how="Beta, the risk-free rate (13-week T-bill) and the S&P 500's 12-month return are shared inputs; the equation is not reproduced per stock.",
                        caveat="The stock's return includes dividends; the market return is the S&P 500 price index, which does not - this tilts alpha upward by roughly the index's dividend yield.",
                        kind="series"),
    # ---- risk
    "volatility": _L("Annualised standard deviation of daily log returns",
                     [("Volatility (as scored)", "volatility_1y", PCT)],
                     how="About 13 months of daily adjusted closes, sample standard deviation x sqrt(252). Needs at least 200 daily returns.",
                     caveat="Labelled 1-year, but the window is about 13 months.", kind="series"),
    "beta": _L("Slope of the stock's daily log returns on the S&P 500's",
               how="cov(stock, market) / var(market) over about 13 months of common trading days; needs at least 200 and 80% overlap. Raw, not shrunk toward 1. Lower is scored as better.",
               kind="series"),
    "sharpe_ratio": _L("(12-month return - risk-free) / volatility", kind="series",
                       how="Weight 0 in the composite since 2026-09-02; shown for reference."),
    "sortino_ratio": _L("(12-month return - risk-free) / downside deviation", kind="series",
                        how="Weight 0 in the composite; shown for reference.",
                        caveat="The denominator is the standard deviation of the shortfalls about their own mean, not the root-mean-square shortfall."),
    "max_drawdown_1y": _L("Largest peak-to-trough fall in the price path", kind="series",
                          how="About 13 months of daily returns; a negative fraction, so a smaller fall scores higher.",
                          caveat="Labelled 1-year, but the window is about 13 months."),
    # ---- revisions
    "fy1_revision_3m": _L("(Current-year EPS estimate now - 90 days ago) / price",
                          [("Estimate now", "_fy1_eps_current", PRICE), ("Estimate 90 days ago", "_fy1_eps_90d_ago", PRICE),
                           ("Price", "currentPrice", PRICE)],
                          how="Scaled by price, not by the estimate, so a near-zero estimate cannot blow it up."),
    "analyst_surprise": _L("Median of (actual EPS - estimate) / max(|estimate|, $0.10) over the last 4 quarters", kind="series",
                           how="Quarters with a missing figure or an estimate near zero are skipped; at least 2 valid quarters are needed."),
    "price_target_upside": _L("(Mean analyst target - price) / price",
                              [("Mean target", "targetMeanPrice", PRICE), ("Price", "currentPrice", PRICE),
                               ("Analysts", "numberOfAnalystOpinions", NUM)],
                              how="Clipped to -50% .. +100%; needs at least 3 analysts."),
    "earnings_acceleration": _L("Latest quarter's surprise - previous quarter's surprise", kind="series",
                                caveat="A change in the surprise ratio between the two most recent valid quarters, not an acceleration of earnings growth."),
    "consecutive_beat_streak": _L("Recency-weighted count of quarters that beat estimates", kind="series",
                                  how="Each quarter that beat its estimate adds its position (1 oldest .. 4 newest).",
                                  caveat="Not a streak length: a beat in the first and last quarters scores 1 + 4 = 5. The maximum depends on how many quarters were valid."),
    "short_interest_ratio": _L("Days to cover (Yahoo 'short ratio')",
                               [("Short ratio", "shortRatio", RATIO)],
                               how="Shares sold short / average daily volume, passed through unchanged. Lower is scored as better."),
    # ---- size / investment
    "size_log_mcap": _L("- ln(market cap)",
                        [("Market cap", "marketCap", USD)],
                        how="Sign-flipped so that a smaller company scores higher. The raw value is a negative number."),
    "asset_growth": _L("(Total assets - assets a year earlier) / assets a year earlier",
                       [("Total assets", "totalAssets", USD), ("Total assets (year earlier)", "totalAssets_prior", USD)],
                       how="Lower is scored as better (Fama-French conservative investment)."),
}

# --------------------------------------------------------------------------- equations
# The line printed under every metric in the workings, filled with that stock's own figures:
# "$4.53B free cash flow ÷ $31.2B enterprise value = 14.5%". Owner, 2026-10-07: "we don't
# see any numbers anywhere ... the numbers going into the scoring."
#
# Each entry is (exact, [templates]). A template is text with {input_key|label} slots; the
# page uses the FIRST template whose slots all have a value for the stock (that is how the
# scorer's own fallbacks run - e.g. Yahoo's price-to-book before price / book per share).
#
# exact=True means the template IS the arithmetic: with ÷ × − ^ ln read as operators,
# ``evaluate_template`` turns it into a number, and tests/test_metric_lineage.py checks that
# number against the published value for every stock. So if the scoring formula changes in
# factor_engine and this line is not updated, the suite fails - the page cannot drift from
# the engine. exact=False is for a formula with steps a one-line equation cannot carry
# (tax rates, floors, clamps); its line lists the real inputs and points at the full detail.
EQUATIONS = {
    "ev_ebitda": (True, ["{ev_used|enterprise value} ÷ {ebitda_used|EBITDA}"]),
    "fcf_yield": (True, ["{fcf_used|free cash flow} ÷ {ev_used|enterprise value}"]),
    "earnings_yield": (True, ["{netIncome|net income} ÷ {marketCap|market cap}"]),
    "ev_sales": (True, ["{ev_used|enterprise value} ÷ {totalRevenue|revenue}"]),
    "pb_ratio": (True, ["{priceToBook|price-to-book, as reported}",
                        "{currentPrice|price} ÷ {bookValue|book value per share}"]),
    "roic": (False, ["{ebit|EBIT} after tax ÷ ({totalEquity|equity} + {totalDebt_bs|debt} − excess cash)",
                     "{ebit|EBIT} after tax ÷ invested capital"]),
    "gross_profit_assets": (True, ["{grossProfit|gross profit} ÷ {totalAssets|total assets}"]),
    "accruals": (True, ["({netIncome|net income} − {operatingCashFlow|operating cash flow}) ÷ {totalAssets|total assets}"]),
    "net_debt_to_ebitda": (False, ["({totalDebt_bs|debt} − {cash_bs|cash}) ÷ {ebitda_nd_used|EBITDA}, and 0 for net cash",
                                   "net debt ÷ {ebitda_nd_used|EBITDA}, and 0 for net cash"]),
    "operating_leverage": (False, ["EBIT change ({ebit_prior|prior} → {ebit_annual|latest}) ÷ revenue change ({totalRevenue_annual_prior|prior} → {totalRevenue_annual|latest})"]),
    "forward_eps_growth": (False, ["({forwardEps|forward EPS} − {trailingEps|trailing EPS}) ÷ trailing EPS (at least $1), clipped"]),
    "peg_ratio": (False, ["({currentPrice|price} ÷ {trailingEps|trailing EPS}) ÷ (EPS growth × 100), capped at 50"]),
    "revenue_growth": (True, ["({totalRevenue|revenue} − {totalRevenue_prior|prior revenue}) ÷ {totalRevenue_prior|prior revenue}"]),
    "revenue_cagr_3yr": (True, ["({totalRevenue_annual|revenue} ÷ {totalRevenue_3yr_ago|revenue 3 years earlier}) ^ (1/3) − 1",
                                "({totalRevenue|revenue} ÷ {totalRevenue_3yr_ago|revenue 3 years earlier}) ^ (1/3) − 1"]),
    "sustainable_growth": (False, ["ROE ({netIncome|net income} ÷ average equity) × share of earnings kept"]),
    "price_target_upside": (True, ["({targetMeanPrice|mean target} − {currentPrice|price}) ÷ {currentPrice|price}",
                                   "({targetMeanPrice|mean target} − {price_latest|price}) ÷ {price_latest|price}"]),
    "short_interest_ratio": (True, ["{shortRatio|days of trading to cover the short position}"]),
    "size_log_mcap": (True, ["−ln({marketCap|market cap})"]),
    "asset_growth": (True, ["({totalAssets|total assets} − {totalAssets_prior|a year earlier}) ÷ {totalAssets_prior|a year earlier}"]),
    "equity_ratio": (True, ["{totalEquity|equity} ÷ {totalAssets|total assets}"]),
    "roe": (True, ["{returnOnEquity|return on equity, as reported}",
                   "{netIncome|net income} ÷ {totalEquity|equity}"]),
    "roa": (True, ["{returnOnAssets|return on assets, as reported}",
                   "{netIncome|net income} ÷ {totalAssets|total assets}"]),
    "return_12_1": (True, ["({price_1m_ago|price 1 month ago} − {price_12m_ago|12 months ago}) ÷ {price_12m_ago|12 months ago}"]),
    "return_6m": (True, ["({price_1m_ago|price 1 month ago} − {price_6m_ago|6 months ago}) ÷ {price_6m_ago|6 months ago}"]),
    "fy1_revision_3m": (True, ["({_fy1_eps_current|EPS estimate now} − {_fy1_eps_90d_ago|90 days ago}) ÷ {currentPrice|price}",
                               "({_fy1_eps_current|EPS estimate now} − {_fy1_eps_90d_ago|90 days ago}) ÷ {price_latest|price}"]),
}

# Metrics with no per-stock equation: one plain line saying what the number is made of.
SOURCES = {
    "piotroski_f_score": "Pass/fail financial-health signals that passed",
    "beneish_m_score": "−4.84 plus eight weighted indices from the annual statements",
    "jensens_alpha": "12-month return minus the return its beta predicted",
    "volatility": "Daily price swings over about 13 months, annualised",
    "beta": "How far it moves with the S&P 500, from about 13 months of daily returns",
    "sharpe_ratio": "12-month return above the risk-free rate, per unit of volatility",
    "sortino_ratio": "12-month return above the risk-free rate, per unit of downside swing",
    "max_drawdown_1y": "Largest fall from a peak over about 13 months",
    "analyst_surprise": "Median beat or miss against the EPS estimate, last 4 quarters",
    "earnings_acceleration": "Latest quarter's surprise minus the one before it",
    "consecutive_beat_streak": "Quarters that beat the estimate, recent ones counting more",
}

def template_slots(template: str) -> list[tuple[str, str]]:
    """The (input_key, label) pairs in a template, in order."""
    import re
    return [(k, lab) for k, lab in re.findall(r"\{([A-Za-z0-9_]+)\|([^}]*)\}", template)]


def choose_template(metric: str, inp: dict):
    """The template the page shows for this stock: the first whose slots all have values."""
    entry = EQUATIONS.get(metric)
    if not entry:
        return None
    for t in entry[1]:
        if all(_ok(inp.get(k)) for k, _ in template_slots(t)):
            return t
    return None


def evaluate_template(template: str, inp: dict):
    """Read an exact template as arithmetic and return its value (or None)."""
    import re
    expr = re.sub(r"\{([A-Za-z0-9_]+)\|[^}]*\}", lambda m: repr(float(inp[m.group(1)])), template)
    expr = (expr.replace("÷", "/").replace("×", "*").replace("−", "-")
            .replace("^", "**").replace("ln(", "_ln("))
    if re.search(r"[A-Za-z]", expr.replace("_ln", "")):
        raise ValueError(f"template has words left after substitution: {template!r}")
    try:
        v = eval(expr, {"__builtins__": {}}, {"_ln": math.log})  # noqa: S307 - our own templates
    except (ZeroDivisionError, ValueError, OverflowError):
        return None
    return v.real if isinstance(v, complex) else v


# --------------------------------------------------------------------------- not used
# Why a metric listed under a category carries no weight in the score a reader is looking
# at (owner, 2026-10-07: "it should say a little bit about why something was not used in the
# score"). The page picks the reason by rule from the published weight tables:
#   - weighted in the bank table but not in this one      -> BANK_ONLY
#   - this IS the bank table and the metric is weighted in the generic one -> NOT_FOR_BANKS
#   - a specific, documented reason below                  -> NOT_USED_BECAUSE[metric]
#   - otherwise                                           -> CANDIDATE
# Every specific reason cites a decision that is written down (config.yaml comments and
# METHODOLOGY_CHANGELOG.md); none is new methodology.
BANK_ONLY = ("Used only for banks and insurers, where it replaces measures built on enterprise "
             "value, cash flow or operating profit, which do not mean the same thing for a lender.")
NOT_FOR_BANKS = ("Not used for banks and insurers: a lender's debt is its raw material, so measures "
                 "built on enterprise value, free cash flow, operating profit or leverage do not "
                 "describe it the way they describe other companies.")
CANDIDATE = ("Tracked, but given no weight: a candidate the screener records so its record can be "
             "measured before it is ever allowed to move a score.")
NOT_USED_BECAUSE = {
    "peg_ratio": ("Removed from the score: P/E divided by growth counts valuation a second time "
                  "(earnings yield is already 1 / P/E) and breaks when growth is negative."),
    "sharpe_ratio": ("Removed from the score 2026-09-02: it is 12-month return divided by risk, so it "
                     "mostly repeated the momentum signal (correlation +0.94 with 12-1 return)."),
    "sortino_ratio": ("Removed from the score 2026-09-02, with Sharpe: return divided by downside "
                      "risk, so it mostly repeated the momentum signal."),
}


def published_not_used() -> dict:
    return {"bank_only": BANK_ONLY, "not_for_banks": NOT_FOR_BANKS, "candidate": CANDIDATE,
            "because": dict(NOT_USED_BECAUSE)}


# Fetch fields the page needs per stock, in a stable order. Derived from the table so a
# new input cannot be named without being published.
ENGINE_KEYS = ("ev_used", "ebitda_used", "fcf_used", "ebitda_nd_used")
INPUT_KEYS = tuple(dict.fromkeys(
    k for entry in LINEAGE.values() for _, k, _ in entry["inputs"]
    if k not in ENGINE_KEYS))

# Metrics whose value is a provider ratio, an analyst figure or a long price series: no
# per-stock equation is shown for these, and the page says so.
NOT_RECOMPUTED = tuple(m for m, e in LINEAGE.items() if m not in RECOMPUTE)


def published_lineage() -> dict:
    """What goes to the browser: no functions, and which metrics carry an equation."""
    out = {}
    for m, e in LINEAGE.items():
        out[m] = {"f": e["formula"], "in": e["inputs"], "k": e["kind"]}
        if e["how"]:
            out[m]["how"] = e["how"]
        if e["caveat"]:
            out[m]["cav"] = e["caveat"]
        if m in RECOMPUTE:
            out[m]["eq"] = 1
        if m in EQUATIONS:
            out[m]["x"] = EQUATIONS[m][1]
            out[m]["xe"] = 1 if EQUATIONS[m][0] else 0
        if m in SOURCES:
            out[m]["src"] = SOURCES[m]
    return out
