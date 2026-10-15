# Scoring history from the filings alone: the record rebuilt, checked against today, and the look-ahead's second half measured (2026-10-15)

**Owner-run.** `plan/backtest-v2.md` step 3. Backtest v1 scores every month since 2020 on today's
fundamentals. The 2026-10-01 note measured the *price* half of that look-ahead (63.2% of
name-months land in a different decile once prices are the month's own) and called it a lower bound,
because the 49.0 points of composite weight built from filings were frozen in both of its arms. The
2026-10-09 census showed the free SEC facts could cover those inputs; this note builds the piece that
uses them and measures what it finds.

Reproduce: `research/measurements/2026-10-15-pit-record-vs-live.py` and
`research/measurements/2026-10-15-lookahead-fundamentals-half.py` (JSON beside each). Module:
`pit_record.py`, `tests/test_pit_record.py`.

## 1. The design choice: rebuild the record, not the metrics

The engine does not score inputs; it scores a **raw record** - the ~100 fields
`factor_engine._fetch_single_ticker_inner` builds from Yahoo's statements, which `compute_metrics`
turns into 46 metrics. A v2 that re-derived each metric from filings would be a second
implementation of 29 formulas: exactly the "one computation, never a second" failure that row 0.8 of
`CLAUDE.md` exists to prevent, and the reason the 2026-10-14 dividend outage took two fixes.

So `pit_record.py` rebuilds the **record**, field for field, from facts filed on or before a date,
and v2 hands it to the engine's own `compute_metrics`. One set of formulas scores the live run and
every historical month; a formula change reaches both at once.

- **What a field means is the engine's, not this module's.** Trailing twelve months are a fiscal
  year where one ends at the latest quarter, else year-to-date + prior year - prior year-to-date.
  Measured on the live run: the fetch fills every `*_prior` flow with the **fiscal year before the
  latest one** (503 of 503 records), not the trailing twelve months a year earlier, so the rebuild
  does the same (no weighted metric reads them since 2026-10-09). Yahoo's "Total Debt" includes lease
  liabilities - MGM's is 4.9x its borrowings alone - so the rebuild adds them. Where a company does not
  tag operating income, EBIT = pretax income + interest, as the fetch falls back from Yahoo's
  "Operating Income" row to its "EBIT" row.
- **Of several versions of one figure, the latest filed by the date wins** - what an investor
  would have read; a restatement filed afterwards is invisible.
- **A field the filings cannot supply is NaN, never today's value.**
- **Every raw field the engine reads is classified**: `statement` (rebuilt), `market` (filings
  over a price), `price_history` (the backtest's own panel), `no_free_history` (analyst estimates,
  short interest - the 9.0pp the plan expected to be permanently unmeasurable) or `static`. A test
  parses `compute_metrics` and fails on any field nobody has classified - which is how a new input
  would otherwise leak today's value into every historical month. It caught two on its first run.
- **Splits.** A split-adjusted price panel is in today's shares; a 2022 filing is in 2022's.
  `split_after` puts the filings' share counts and EPS on the price's basis (NVDA's 2022 filings are
  in pre-10:1 shares). Split history: 1,510 splits for 360 tickers, `cache/splits.parquet`.

The SEC cache gained the concepts the record needs (tax, pretax income, interest, dividends,
short-term investments, current debt, leases, receivables, PP&E, SG&A, diluted EPS, share counts
from the balance sheet and the cover page): 2,262,395 facts for 502 companies, 0 failed. Nothing in a
published run reads the new ones.

## 2. Checked where the answer is known: today

Rebuilt as of 2026-10-15 for the run of that morning (503 stocks) and scored by the same engine at
the same configuration as the live record. Price-history and analyst fields are the live ones in
every arm, so this isolates the fundamentals rebuild.

| | Composite Spearman | Top 25 kept | Top decile kept | Median rank move |
|---|---|---|---|---|
| **S** - statement fields from filings | **0.933** | 19 / 25 | 43 / 50 | 17 places |
| **B** - statement + market fields from filings | **0.885** | 16 / 25 | 35 / 50 | 23 places |

Per category (arm B): valuation 0.883, quality 0.868, growth 0.967, investment 0.978, size 0.955;
momentum, risk and revisions 1.000 by construction.

Per weighted fundamentals metric (arm B; coverage = share of stocks with a live value that the
rebuild also gives one):

| Metric | Composite weight | Coverage | Spearman | Within 10% |
|---|---|---|---|---|
| `fcf_yield` | 9.90 | 83% | 0.927 | 79% |
| `roic` | 6.38 | 90% | 0.931 | 66% |
| `ev_ebitda` | 5.50 | 71% | 0.813 | 71% |
| `size_log_mcap` | 5.00 | 96% | 0.960 | 99% |
| `asset_growth` | 5.00 | 99% | 0.986 | 86% |
| `gross_profit_assets` | 4.84 | 65% | 0.966 | 89% |
| `earnings_yield` | 4.40 | 95% | 0.959 | 90% |
| `net_debt_to_ebitda` | 4.40 | 73% | 0.780 | 63% |
| `piotroski_f_score` | 3.52 | 97% | 0.845 | 60% |
| `revenue_growth` | 3.25 | 98% | 0.937 | 80% |
| `ev_sales` | 2.20 | 94% | 0.968 | 86% |
| `revenue_cagr_3yr` | 1.95 | 97% | 0.966 | 94% |
| `sustainable_growth` | 1.95 | 74% | 0.996 | 91% |
| `beneish_m_score` | 1.76 | 85% | 0.870 | 92% |
| `accruals` | 1.10 | 96% | 0.982 | 87% |

**How to read it.** The rebuild is not Yahoo, and does not need to be: a v2 needs every month built
*the same way*, and the rebuild is one consistent source throughout, where Yahoo's history is not
available at all. What the table measures is how far the rebuilt screener is from the live one -
**0.89 at the composite** - which bounds how much a v2 result can be claimed for the live ranking.
The gaps are definitional, not errors found in either source (consistent with
`research/2026-10-14-yahoo-vs-sec.md`, where the inputs with a clean counterpart agreed 94-100%):

- **Debt and EBITDA** (net debt / EBITDA 0.78, EV/EBITDA 0.81). Yahoo's operating income and its
  debt total are its own constructions; the filings' `OperatingIncomeLoss` is the company's, and
  lease liabilities are tagged inconsistently. These two metrics are the weakest point of a free v2.
- **Gross profit** is tagged by 65% of companies (the census found 39% tagged directly; the rest
  derive it from a cost-of-revenue tag). The other 35% are scored on the rest of Quality.
- **Piotroski** is an integer: one signal flipping is a 1-point move, so "within 10%" understates
  agreement (Spearman 0.85).

What the rebuild could improve with more tag work - better EBITDA and debt coverage - is listed in
section 5. None of it changes the conclusion below.

## 3. The look-ahead's second half, month by month

The 2026-10-01 panel: 81 rebalance months, 2020-01 .. 2026-09, the current constituents with a price
that month (468-499 a month, **39,320 name-months**). Each month, three versions of the universe
are scored by the engine's own `compute_metrics` and composite at today's configuration:

- **V0** - today's filings, market value at today's price: what v1 holds constant;
- **V1** - today's filings, market value at the month's price: v1 with the price half fixed;
- **PIT** - the filings as they stood at the month's end, at its price: both halves fixed.

The 12-1 and 6-1 returns are the month's own in all three and the analyst fields are today's (no free
history), so they cancel; all three read statements from the SEC rebuild, so a difference between
arms is *when* the filings are read, never *where from*. Deciles as `backtest._assign_deciles`, as the
2026-10-01 measurement applied them.

| | Name-months in a different decile | 2+ deciles | v1's top decile kept | Median monthly Spearman |
|---|---|---|---|---|
| Price half (V0 -> V1) | **60.1%** | 27.0% | 62.1% | 0.827 |
| **Fundamentals half (V1 -> PIT)** | **72.6%** | **43.2%** | **48.2%** | **0.606** |
| Both (V0 -> PIT) | **75.8%** | 48.5% | 43.3% | 0.537 |

*For scale: an unrelated ranking moves 90% of names and keeps 10% of a top decile.*

The fundamentals half by year - share of name-months in a different decile: **2020 80.0%, 2021 80.3%,
2022 78.9%, 2023 76.6%, 2024 74.6%, 2025 70.0%, 2026 40.4%.** It falls to 7.0% in 2026-08 and 2.6% in
2026-09, where the filings read "then" and "now" are nearly the same documents - the measurement's own
zero point, behaving as it must.

**Three checks before believing it.**

1. **The price half reproduces the 2026-10-01 result independently.** That note restated the payload's
   metrics at historical prices and found 63.2% of name-months changing decile; this one rebuilds
   market value from filed share counts at the panel price and finds 60.1%. Two constructions, one
   answer.
2. **The rebuilt history is right where it can be checked - with one known distortion.** Market
   values at 2020-06 from filed share counts and the panel price: Apple $1,530bn, Microsoft $1,464bn,
   NVIDIA $232bn, Tesla $200bn, JPMorgan $244bn. Without `split_after`, NVIDIA's would have been a tenth
   of that. They run **low for dividend payers**: the panel is `auto_adjust=True`, so a 2020 price is
   reduced by every dividend since - JPMorgan's ~$287bn reads $244bn, Tesla's (no dividend) is right.
   The 2026-10-01 note measured the same drag (`dividend_drag`). It sits **inside the price half**
   (V0 -> V1); the fundamentals half compares V1 and PIT at the *same* price, so the drag is common to
   both and cancels from the headline number. A v2 must value on split-adjusted, not
   dividend-adjusted, closes (section 5).
3. **Not a coverage artifact.** At 2020-06 every weighted metric has the same coverage in V1 and PIT to
   within 5 points (e.g. `fcf_yield` 77% / 76%, `gross_profit_assets` 59% / 59%), so the composite is
   renormalised the same way in both. What differs is the values, and they differ the way six years of
   business change says they should: size is 0.955 correlated between 2020 and 2026, earnings yield
   0.36, Piotroski 0.20, revenue growth 0.14, asset growth 0.09.

## 4. What this means

- **Backtest v1's history is not this screener's history.** In the months that make up most of its
  window, under half of v1's top decile (43%) is what the screener - rebuilt from prices and filings -
  would have put there at the time, and the rank correlation is 0.54. Rule 5 of `CLAUDE.md` already
  says v1 decides nothing; this is the size of the reason. The 2026-10-01 note called its 63.2% "a
  lower bound"; the measured total is **75.8%**, and it is still a bound - the 9.0pp of analyst inputs
  stay frozen in every arm here, and survivorship (11.4% of the panel) is separate.
- **The fundamentals half is the larger half** (72.6% against 60.1%), which confirms the plan's
  2026-10-01 re-sequencing: point-in-time fundamentals before the point-in-time universe.
- **It is now fixable for free.** Everything V0 -> PIT moved is rebuilt from SEC filings and a price
  panel, both free, by the engine's own code. The record a v2 needs exists and is tested.
- **Nothing in the published screener changes.** No weight, metric or score moved; the live run reads
  none of this. Under rule 5 no backtest number is quoted here as evidence for or against any weight,
  and none should be until v2 exists and 2027-02-11 has passed.

## 5. What v2 still needs

In order (also in `plan/backtest-v2.md` step 3):

1. **The v2 scoring loop.** Per month: the point-in-time universe (`universe_history.members_on`),
   records from `pit_record.record_as_of`, the panel price, analyst fields set to NaN (the honest
   treatment - the engine renormalises within the category; report it as the filings-and-prices
   screener and say so), the engine's composite, then deciles and forward returns. A name in the
   month's universe with no current raw record (delisted) has no static fields or price here - that is
   step 2's work, and v2 must count and report those names rather than drop them silently.
2. **Two price series, not one.** Valuation needs the close as traded (split-adjusted only, so a
   2020 market value is 2020's); returns need the total-return series. The panel has only the latter.
3. **Daily prices for the risk metrics.** Volatility, beta, Jensen's alpha and the 13-month drawdown
   (13.3pp of composite weight between them) are computed from daily returns; the monthly panel cannot reproduce
   them. One adjusted daily history per name, cached, then the engine's own risk code per month.
4. **Better tags where the rebuild is weakest:** EBITDA (operating income + D&A, 71-73% coverage) and
   total debt. Each improvement is measurable with `2026-10-15-pit-record-vs-live.py`.
5. Then step 2 (delisted names), step 4 (the A/B harness with confidence intervals) and step 5.

**Limits of this note.** Current constituents only. Analyst inputs frozen. Market value from filed
share counts can miss multi-class structures (size correlates 0.96 with Yahoo's, not 1.0). The
`*_prior` flows follow the engine's prior-fiscal-year meaning, not a trailing year a year earlier.
And "the filings as they stood" uses each fact's `filed` date - a fact filed after the month-end is
invisible, which is the point, but a 10-K's figures are visible from the day it was filed even if a
reader would take days to digest it.
