# Multi-Factor Stock Screener

A free, public screener that scores the companies in the S&P 500 on eight
categories - valuation, quality, growth, momentum, risk, analyst revisions,
size and investment - ranks them, and shows the arithmetic behind every number.
It is built to be taught from: an investment club can open any stock and follow
its score back to the company's own figures.

**It is decision support for research, not investment advice.** It does not
say what to own or how much, and on 2012-2026 history its ranking has not been
shown to predict returns (see "Has it been tested?" below).

## Open it

| Page | What it is for |
|------|----------------|
| [The dashboard](https://calebsmit.github.io/screener-dashboard/) | The ranking, every stock's sheet, comparisons and your own watch list. Updated on weekday mornings. |
| [Has it been tested?](https://calebsmit.github.io/screener-dashboard/research.html) | The ranking rerun on 2012-2026 history, with every pre-registered experiment and its result, including the ones that failed. |
| [Factor Lab](https://calebsmit.github.io/screener-dashboard/lab.html) | The published studies behind the categories, rerun on public data, with how far each rerun lands from its paper. You can change the conditions in the browser. |
| Methodology | Every metric, weight, rule and limitation. On the dashboard it is the **Methodology** button; in this repository it is [SCREENER_OVERVIEW.md](SCREENER_OVERVIEW.md). |

## How to read it

The dashboard's own short guide walks the same path.

1. **Start with the ranking.** Each company has eight category scores from 0
   to 100 and one composite. Every score compares a company with its own
   sector, where about 50 is the middle, and on all eight a higher score adds
   more to the composite - so a high Risk score means a calmer price and a high
   Size score a smaller company. The key above the table, "What the columns
   mean", says this column by column.
2. **Read the composite as points.** It is the weighted sum of the eight
   scores: a composite of 70 is 70 points out of 100, and it does not mean the
   stock beats 70% of the others.
3. **Open a stock.** "Why it ranks here" gives the reasons in a few sentences.
   "The workings" rebuilds the score: each metric's value, its sector
   percentile, the weight it carried and the points it added, so any number can
   be checked with a calculator.
4. **Treat a trap flag as a prompt to read on.** A value-trap flag marks a
   stock that scores high on Valuation but is weak on at least two of Quality,
   Momentum and Revisions; a growth-trap flag marks high Growth with weak
   Quality or Revisions. A flag changes neither the score nor the rank.
5. **Read the context.** "Before you decide" shows the price trend, options,
   insider trades and the market backdrop beside the score. None of it enters
   the score.
6. **Compare and follow.** Compare puts two stocks side by side. My holdings
   keeps the names you follow, in your own browser (no purchase price is asked
   for), and shows where each ranks on every run.

### In a club

A ranking like this is a structured way to choose what to research, and the
research is still the work: the company's filings, its competitors, what would
have to be true for the price to be wrong. The stock sheet's "The case against"
and the methodology's **Limitations** are written to start that argument, and
"Has it been tested?" is the honest answer to "does this work?".

## The eight categories

| Category | Weight | The question it asks |
|----------|--------|----------------------|
| Valuation | 22% | Is the stock priced low against what the business generates? |
| Quality | 22% | Is this a well-run, durable business? |
| Growth | 13% | Is the business growing, and can it keep it up? |
| Momentum | 13% | Has the market been rewarding the stock recently? |
| Risk | 10% | How bumpy has the ride been? |
| Revisions | 10% | What do analysts expect, and are they turning more or less optimistic? |
| Size | 5% | Is the company small for its sector? |
| Investment | 5% | Is it expanding its asset base conservatively or aggressively? |

Weights sum to 100. Banks, insurers and other bank-like financial companies are
scored on their own Valuation and Quality metrics, because enterprise value and
return on invested capital do not fit a business whose liabilities are its
raw material. On any one run the momentum regime rule or missing data can move
a weight; each stock's sheet shows the weight it was actually scored with.

The registry holds **46 metrics**: **27** general ones that carry weight, **4**
bank-specific ones (price-to-book, ROE, ROA and the equity ratio) that replace
some of the general ones for a bank-like company - which is scored on **24** in
all - and **15** candidates that are computed and shown but carry no weight. A
candidate joins the score only through a research note and a changelog entry.
`tests/test_readme_front_door.py` fails if these figures part from `config.yaml`.

## What it cannot do

- **Predict returns, as far as anyone has shown.** Rerun on 2012-2026 history,
  rebuilt from the filings as they stood each month, the ranking did no better
  than chance, and that rebuild lacks the analyst inputs and the companies that
  have since left the index. The live track record is months long, which shows
  nothing either way.
- **Compare sectors with each other.** A high Valuation score means low-priced
  for its sector, and says nothing about the sector itself.
- **See past free data.** Everything comes from Yahoo Finance, checked each
  run against the companies' own SEC filings where a filing has the figure.
  Analysts' estimate history reaches back only about 90 days.
- **Say what to own or how much.** There is no model portfolio on the
  dashboard and no position sizing.

The methodology's **Limitations** section has the full list, with the numbers.

## How it is built and checked

- **Two scheduled loops.** A data run at 02:00 on weekdays fetches, scores and
  publishes; a code session improves the system. A data run is discarded
  rather than published if it fails its health checks - thin price or analyst
  coverage, a collapse in score dispersion, stale prices, too few stocks
  scored, or a ranking that breaks from the last one without a recorded
  methodology change (`scripts/check_run_health.py`).
- **The page proves its own arithmetic.** Before publishing, the build rebuilds
  every category score and composite from the published inputs and refuses to
  publish if any stock fails to reproduce. `python scripts/audit_stock.py JPM`
  repeats the calculation for any stock with code that shares nothing with the
  scoring engine.
- **Every sentence that says how a number is computed is registered** in
  `claims.py` with the code that makes it true and the test that checks it, and
  the data run checks those claims before it publishes.
- **Methodology changes are written down.** Each one is in
  `METHODOLOGY_CHANGELOG.md` with its evidence: published research and
  documented professional practice. Backtests and the live record are not yet
  allowed to justify a change.
- **More than 3,000 tests** run before any code change reaches the public site.

## Run it yourself

```bash
# Python 3.9+
pip install -r requirements.txt

python run_screener.py --dry-run          # check config and wiring (~10 s)
python run_screener.py                    # full run (uses today's cache if present)
python run_screener.py --refresh          # force fresh data
python run_screener.py --tickers AAPL,MSFT,GOOGL   # quick test on 3 stocks
python run_screener.py --preset value     # balanced / value / growth / momentum
python run_screener.py --show-weights     # print the effective weights and exit
python run_screener.py --no-portfolio     # scores only, no model portfolio
```

On a machine with TLS interception (e.g. Avast), set `CURL_CA_BUNDLE` to
`.certs/combined_ca.pem` before running: yfinance's curl_cffi backend honors
`CURL_CA_BUNDLE`, and other SSL environment variables do not affect the data
path under Python 3.13.

Tests: `python scripts/run_tests.py tests/ test_screener.py -q`.

### Configuration

All tuneable parameters live in **`config.yaml`**:

- **`universe`** - index selection, minimum market cap and volume, exclusions.
- **`factor_weights`** - category weights (sum to 100).
- **`metric_weights`** / **`bank_metric_weights`** - weights within each category (each sums to 100).
- **`value_trap_filters`** / **`growth_trap_filters`** - the trap thresholds.
- **`portfolio`** - the workbook's model portfolio: size, weighting, position and sector caps.
- **`caching`** - cache format and age; caches are tagged with a hash of the scoring config.
- **`data_quality`** - outlier reporting (flagged, never clipped), coverage thresholds.
- **`metric_clamps`** - bounds on the two metrics that are clamped.
- **`improvement`** - the evidence-recording engine; it records and proposes, and applying a weight change is switched off.

### Output files

| File | Description |
|------|-------------|
| `factor_output.xlsx` | Up to **7 sheets**: ReadMe (disclaimers), FactorScores (every stock, all eight category scores and the composite), ScreenerDashboard (the top 25), ModelPortfolio (research use only), DataValidation (the top 10's raw values and data-quality flags), WeightSensitivity (each category weight moved 5 percentage points either way, with the top-20 overlap, when available) and FactorCorrelation (Spearman correlations of every metric's sector percentile, when available) |
| `cache/factor_scores_<hash>_YYYYMMDD.parquet` | The scored universe, tagged with the config hash |
| `runs/<run_id>/` | Raw fetch, scored data and config snapshot for each run |
| `validation/data_quality_log.csv` | Per-ticker data-quality issues |
| `improvement/` | Snapshots and live information-coefficient history |

`generate_dashboard.py` writes the dashboard (`index.html`) and its two data
files (`dashboard_data.js`, `dashboard_context.js`); `SCREENER_OVERVIEW.md` is
written by `run_screener.py`. All four are generated - edit the code that
writes them, never the files.

## Disclaimer

This tool is provided for **educational and research purposes only**. It does
not constitute investment advice, and nothing on the dashboard is a
recommendation to buy or sell any security. Past results of any backtest or
track record do not guarantee future results. Do your own research, and consult
a qualified financial adviser before making investment decisions.

## Dependencies

See `requirements.txt` for the full list. Core libraries: pandas, numpy and
scipy (data and statistics), yfinance (market data), openpyxl (the workbook),
PyYAML (configuration), pyarrow (Parquet) and markdown (the methodology page).
