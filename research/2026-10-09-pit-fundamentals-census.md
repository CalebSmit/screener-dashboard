# Point-in-time fundamentals from free SEC data: how much of the backtest's blind half can be fixed (2026-10-09)

`plan/backtest-v2.md` step 1 left one piece unmeasured: the **49.0 points of composite weight**
whose inputs come from filings, which `backtest.py` v1 holds at *today's* values for every month
back to 2020 (a negative reporting lag of up to 80 months). The plan named SEC XBRL as the free
route because every fact carries the date it was **filed**. This note measures how far that route
goes. Reproduce: `research/measurements/2026-10-09-xbrl-point-in-time-census.py` (JSON beside it).

**Panel:** the 2026-10-01 look-ahead measurement's - 81 rebalance month-ends, 2020-01 .. 2026-09,
the current 503 names, 40,743 name-months. **Source:** one `companyfacts` request per company
(503 requests, 1.17 million 10-K/10-Q facts). **Coverage** = the share of name-months for which a
value of that input had been *filed by the month-end* for a period ending in the previous 15
months - i.e. a backtest could have used it then. Any of the listed tags counts, because filers
switch tags (ASC 606 moved most revenue to `RevenueFromContractWithCustomer...`).

## Results

| Input | Companies | Name-month coverage | First filed after period end, median (10-Q / 10-K) |
|---|---|---|---|
| Total assets | 502 | **96.8%** | 33 / 53 days |
| Shareholders' equity | 502 | **96.8%** | 33 / 53 |
| Net income | 502 | **96.2%** | 34 / 55 |
| Revenue | 500 | **95.5%** | 34 / 56 |
| Operating cash flow | 501 | **95.5%** | 33 / 54 |
| Cash | 487 | 84.9% | 34 / 53 |
| Current assets / liabilities | 424 | 81.0% | 33 / 52 |
| D&A | 433 | 75.9% | 34 / 54 |
| Operating income | 420 | 75.5% | 33 / 54 |
| Long-term debt | 459 | 71.2% | 33 / 53 |
| Capex | 398 | 68.3% | 34 / 54 |
| Cost of revenue | 358 | 58.8% | 33 / 55 |
| Gross profit (tag) | 258 | **39.0%** | 34 / 57 |

**Reporting lags are short and regular:** a quarter's figures are first filed a median ~33 days
after it ends (90th percentile ~40 for balance-sheet items), a fiscal year's ~54 days. The long
tails in the JSON (p90 of 380-600 days on some flows) are year-to-date and annual figures that
first appear inside a later filing, not slow filers. **A 45-day lag for quarters and 75 days for
annual figures would cover the large majority of filings**, and the facts' own `filed` dates make
even that assumption unnecessary.

## What it means for the 49.0 points

- **Rebuildable now, ~95% coverage:** everything built from net income, assets, equity, revenue and
  operating cash flow - ROA/ROE/equity ratio (banks), accruals, asset growth (Investment),
  revenue growth and the revenue CAGR, sustainable growth, Piotroski's profitability, cash-flow and
  accrual signals, and the fundamentals behind earnings yield and EV/sales.
- **Rebuildable with tag mapping, 68-85%:** ROIC and net debt / EBITDA (operating income, D&A,
  debt, cash), the FCF yield (capex), Piotroski's leverage and liquidity signals.
- **The weak spot is gross profit** (39% tagged; 59% have a cost-of-revenue tag to derive it from).
  That is gross profit / assets (22% of Quality) and Piotroski's margin signal - the inputs a free
  v2 would have to treat most carefully, or measure as partially point-in-time.
- **Not reachable from filings at all:** the 9.0pp of analyst estimates (forward EPS growth, FY1
  revisions, surprise history) - as the plan expected - plus short interest, which has no free
  history either.

**Limits.** Current constituents only (survivorship is the other half of v2 and is measured
separately); coverage counts the *availability* of a figure, not that the engine's exact definition
can be rebuilt from it (ROIC's tax and excess-cash rules, the EBITDA fallback); and tag mapping is
the real work - every percentage above is a lower bound that better concept lists will raise.

## Next

Step 3 of the plan, in its own order: the 34.1 price points first (no data needed), then a
`pit_fundamentals.py` that returns, for a ticker and a date, the latest figure **filed** by then for
each input above - built on this census's facts cache - and finally v2 recomputing each fundamentals
metric per rebalance month from it. Do not wire any piece into `backtest.py` alone: a half-fixed
backtest is what the plan forbids.
