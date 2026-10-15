# Should the top five be there? An audit of the ranking's leaders (2026-10-15)

**Owner-run.** Owner: *"look into the top 5 stocks in the screener and make sure they should actually
be there."* The run of 2026-10-15 02:00, ranked by composite. For each name: are the inputs the
company's own figures, does each high score come from what the metric is meant to measure, and what
would a careful reader want said before acting on the rank.

## 1. The five

| Rank | Stock | Composite | Inputs against its filings | Verdict |
|---|---|---|---|---|
| 1 | Expedia (EXPE) | 74.90 | all five within 0.5% | **Earned, one measure overstated** |
| 2 | APA Corp (APA) | 71.40 | all four checked within 0.5% | **Earned on the data; cyclical caveat** |
| 3 | Host Hotels (HST) | 71.35 | all five within 0.5% | **Flattered by a one-off gain** |
| 4 | Incyte (INCY) | 70.57 | not checkable this week (SEC data lags its 10-Q) | **Earned** |
| 5 | Bristol-Myers (BMY) | 68.44 | all five within 0.5% | **Clean** |

**Expedia.** Top decile on free-cash-flow yield (13.6%, 99th percentile in its sector), ROIC (64%),
accruals (cash earnings well above book earnings), Piotroski (8 of 9), forward EPS growth and 12-1
momentum. Two things a reader should know: its operating cash flow includes customers' prepayments
for travel (the "merchant float" - real cash, but cash it owes back in services), and its equity is
**4.2% of total assets** after years of buybacks, so return on equity reads ~170% and **sustainable
growth sits at its 100% cap (97th percentile)**. That one measure is an artefact of the small equity
base. Its weight is 1.95 composite points; at a median score instead, Expedia would lose under 1 point
and stay first. ROIC is already protected: the engine floors invested capital at 10% of assets, a
rule written with Expedia in mind.

**APA.** A beta of **-0.94** and Jensen's alpha of **+122%** look like errors; they are not. Recomputed
from the cached daily history against the S&P 500 total-return index over the same year: APA -0.94,
Exxon -0.66, Devon -0.68, EOG -0.69, Occidental -0.76 - the whole energy sector moved against the
market this year, and APA's price rose 114%. The real caveat is **cyclical**: an oil producer after a
price rise looks cheap on trailing earnings (EV/EBITDA 3.6, 93rd percentile) at what may be the
cycle's top - the classic value trap the trap flag is meant to catch, and did not, because analysts
expect flat earnings (forward EPS growth -0.2%) rather than a fall.

**Host Hotels.** Net income of **$1.03bn** against **$0.89bn** of operating income. Its filings show
why: in the first half of 2026 net income was $0.731bn against $0.612bn of operating income, the
first quarter alone $0.494bn against $0.319bn - about $0.2bn from selling hotels. That gain lifts its
earnings yield (95th percentile in Real Estate), Piotroski's profit signals and sustainable growth.
For a REIT, net income is also the wrong yardstick in practice: the industry reports FFO, which
excludes property-sale gains precisely because they do not recur.

**Incyte.** Revenue growth of 37.7% is real: its quarterly revenue went from $1.216bn to $1.674bn
year on year, and the source's own "unusual items" are small. It is penalised on asset growth (35%,
6th percentile), as the Investment category intends. The filing check could not run because SEC's
company-facts feed did not yet carry its second-quarter 10-Q - a gap in the check, not in the data.

**Bristol-Myers.** Nothing found. Its 2024 Karuna write-off has left the trailing twelve months, so
its earnings yield (7.7%) is on ordinary earnings.

## 2. How common each caveat is

Measured on the same run (`ranking_audit.py`):

- **Net income above 1.25x after-tax operating income: 46 stocks.** In the top 50: HST (#3, 1.46x)
  and Western Digital (#28, 2.56x).
- **Equity under 10% of assets with sustainable growth at its cap: 4 stocks** (EXPE #1, GDDY #31,
  MA #217, MTD #381). 12 stocks in all sit at the cap.

## 3. What shipped

`ranking_audit.py` makes this audit repeatable rather than a one-off:

- **In each stock's view**, a note headed "In this company's reported figures" on the metric affected - earnings yield when net
  income runs well above operations, sustainable growth when buybacks have hollowed out equity - in
  plain words, with the figures (50 stocks on this run). Display-only; claim `drilldown.ranking_audit`.
- **Every morning**, the health check names the top-25 stocks whose rank rests on something worth a
  second look (non-operating earnings, thin equity, an input more than 10% off its filing, a period
  behind its filings, annual figures used), and the brief carries the line. On this run: EXPE (equity
  4.2% of assets), HST (net income 1.46x after-tax operating income), CF (net income 16% off its
  filing - minority interests, a definitional difference).

Nothing here changes a score. `tests/test_ranking_audit.py`, 8 tests.

## 4. The methodology question it raises - for research, not yet a proposal

*Researched the same afternoon: `research/2026-10-15-earnings-definition.md` - an OPEN proposal to put
earnings yield on earnings without unusual items, and operating income left as it is.*

Both caveats are the screener scoring exactly what it defines. Whether the definitions are right is a
methodology question, and the honest status is: **not researched yet**.

- **Earnings yield on earnings before non-recurring items.** The accounting literature on the
  persistence of earnings components (Fairfield, Sweeney & Yohn 1996, *The Accounting Review* 71(3);
  the later work on special items) is the place to start; index providers' value definitions and the
  REIT industry's FFO convention are the practitioner side. The source serves "Normalized Income" and
  "Total Unusual Items" rows on the same statement the fetch already reads, so a change would cost no
  extra requests - but its definition of "unusual" is the source's, which is a reason for care, not a
  reason to proceed. **This session could not verify the papers' effect sizes** (the abstract pages
  were unreachable), so it proposes nothing: a Monday should read them and, if they hold, write the
  proposal with the 46-stock measurement above as its scope.
- **Sustainable growth with a hollowed-out equity base** is a smaller question (1.95 points, 4 stocks):
  whether to treat it as not meaningful below an equity-to-assets floor, as the ROIC floor already
  does for invested capital.
- **Cyclical earnings** (APA) is the hardest: normalised or mid-cycle earnings are standard in
  practice for commodity producers and need history the screener does not keep.

## 5. Reproduce

The figures above are from `dashboard_data.js` of the 2026-10-15 run, SEC company facts for HST, the
cached daily history (`cache/v2_daily.parquet`) for the betas, and the source's quarterly statement
for INCY, each read in this session; `ranking_audit.top_flags(payload)` and
`ranking_audit.stock_notes(detail)` rebuild sections 2 and 3 from any run's payload.
