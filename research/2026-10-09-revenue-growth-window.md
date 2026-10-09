# "Year-over-year" revenue growth spanned 12 to 23 months (2026-10-09)

**Question (CLAUDE.md 0.9(b), opened 2026-10-07).** The transparency build found that
`revenue_growth` (25% of Growth) compared trailing-twelve-month revenue with "the fiscal year
before the latest completed one". How long a window is that, does it differ across companies,
and what is the right comparison? Research-led per rule 4; no backtest or IC used.

**Decision:** latest quarter vs the same quarter a year earlier; where Yahoo has no such pair, the
latest fiscal year vs the one before. Both are exactly 12 months. Changelog 2026-10-09.

## 1. What the code did

`totalRevenue_prior` was meant to be the TTM a year earlier (quarters 5-8). Yahoo supplies five
quarters, so it fell back to the annual statement's column 1. **Measured on run `a2d76219dc0a`:
502 of 502 stocks used the fallback** (`research/measurements/2026-10-09-revenue-growth-window.py`;
a separate live sample: 19 of 20 tickers have exactly five quarters).

## 2. How long the window is - and that it depends on the fiscal calendar

Span between the end of the TTM (the latest reported quarter) and the end of the fiscal year it
was compared with, using each company's fiscal year-end from its own SEC filing (466 stocks):

| Fiscal year ends | Span |
|---|---|
| December (363 stocks) | **18 months** |
| January-February | 17-18 |
| March-April | 15 |
| **May-July** | **12** |
| August-November | 20-21 |

Overall: 12 months for 32 stocks, 18 for 363, 21 for 39, up to 23. The median "YoY" growth read
**11.0%**; the same companies' true fiscal-year-over-fiscal-year growth has a median of **6.95%**.
Two problems follow:

1. **Inflation.** An 18-month change is reported as a one-year one.
2. **Inconsistency across companies** - the one that matters for a score built from sector
   percentiles. Microsoft (June year-end) was compared over 12 months while Alphabet (December)
   was compared over 18, in the same sector, on the same day. Which company looks like the faster
   grower depended partly on its fiscal calendar and on the date of the run.

## 3. The choice

The fetch carries two comparisons that are exactly one year apart:

- **Latest quarter vs the same quarter a year earlier.** Current, and seasonally matched. In the
  earnings-announcement literature quarterly revenue is modelled as a *seasonal* random walk - its
  benchmark is the same quarter of the prior year - e.g. Jegadeesh & Livnat (2006, *Journal of
  Accounting and Economics* 41, 147-171), whose revenue surprise (SURGE) is measured against
  Q(t-4) plus drift. It is also the convention practitioners see daily: "quarterly revenue
  growth (yoy)" is how data vendors (Yahoo's own `revenueGrowth` field among them) and earnings
  coverage report the top line. Cost: one quarter is noisier than four.
- **Latest fiscal year vs the one before.** Steadier, and the academic default for annual sales
  growth (Compustat `SALE` over its lag). Cost: up to a year old at any moment, and it largely
  repeats the latest year of `revenue_cagr_3yr`, which Growth already holds on the annual basis
  (Phase 13, F20).

**Quarter first, annual as the fallback** - because the Growth category already has a slow
annual leg (3-year CAGR) and a forward-looking one (forward EPS growth); what it lacked was a
current top-line reading that is honest about its window. Both branches are 12 months, so the
fallback never reintroduces the inconsistency. A live sample of 25 stocks had a same-quarter pair
exactly 365 days apart for all 25; `_revg_basis` records which branch each stock used, and the
page shows the quarters and their dates.
Cross-check: the new figure equals Yahoo's own `revenueGrowth` field to three decimals for both
stocks tested (Alphabet 0.242, Microsoft 0.177), so the definition is the vendor's and the arithmetic
is ours.

**Not chosen:** TTM vs TTM (the textbook practitioner figure) needs eight quarters, which this
source does not supply; the SEC's XBRL data could provide it, but Q4 is reported only inside the
annual figure and would have to be derived - a larger build, worth it only if the one-quarter
noise proves to matter.

## 4. Fit with the rest of the screener

- Growth: forward EPS growth 45 (forward, analysts), revenue growth 25 (now: current top line,
  12 months), 3-year revenue CAGR 15 (annual, smoothed), sustainable growth 15 (ROE x retention).
  The quarter definition makes revenue growth **less** redundant with the CAGR than the annual
  alternative would.
- **Same defect, fixed the same day:** Piotroski signals 3 (ROA up), 8 (gross margin up) and 9
  (asset turnover up) compared TTM figures with that same fiscal-year-before-last fallback.
  Piotroski (2000) defines each on **annual** data, year t against t-1, with ROA and turnover on
  beginning-of-year total assets - so those three now compare the latest fiscal year with the one
  before, on beginning-of-year assets, and a missing annual input leaves the signal untestable
  rather than reviving the mix. Signals 5-7 already compared the latest quarter-end balance sheet
  with the same quarter a year earlier, which is consistent, and are unchanged. On the 10-stock
  test fixture this changed 2 of 10 F-scores.
- **Still to do:** the Company Snapshot's "YoY" display lines compare the same TTM and fallback
  figures; they are display-only and should show the same 12-month comparison.

## Sources

- Jegadeesh, N. & Livnat, J. (2006). "Revenue surprises and stock returns." *Journal of
  Accounting and Economics* 41(1-2), 147-171. Working paper read: https://pages.stern.nyu.edu/~jlivnat/JAE%20submission.pdf ;
  record: https://ideas.repec.org/a/eee/jaecon/v41y2006i1-2p147-171.html
- Piotroski, J. (2000). "Value Investing: The Use of Historical Financial Statement Information
  to Separate Winners from Losers." *Journal of Accounting Research* 38 (supplement), 1-41.
- Measurement: `research/measurements/2026-10-09-revenue-growth-window.py`.
