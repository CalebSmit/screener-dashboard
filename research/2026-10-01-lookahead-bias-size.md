# How big is `backtest.py`'s look-ahead bias?

**Date:** 2026-10-01
**Plan item:** `plan/backtest-v2.md` step 1, the second half — `CLAUDE.md`
priority 3
**Reproduce:** `python research/measurements/2026-10-01-lookahead-price-component.py`
(inputs are the committed `dashboard_data.js`; output JSON is committed beside
the script)
**Code shipped:** `lookahead.py`, `tests/test_lookahead.py` (43 tests)

---

## The answer

**63.2% of the name-months in `backtest.py`'s panel are assigned to the wrong
decile**, purely because the harness holds price-dependent metrics at today's
values instead of restating them at the rebalance month's own price. **30.3%**
are wrong by two deciles or more. Only **59.5%** of the names v1 puts in its top
decile belong there.

Set that against the survivorship figure measured on 2026-09-24 and restated
2026-09-30: **11.4% of the panel is absent.** The two biases are now measured on
the same unit and look-ahead is **5.5x larger** — and the look-ahead number is a
**lower bound**, because 49.0 points of composite weight are still frozen in
*both* arms of the experiment.

The decision rule this plan set for itself in step 1 was: *"If it's 0.5% a year,
v1 is usable with a caveat. If it's 4%, every existing validation claim needs
retracting."* Survivorship came in at the retract end. Look-ahead is an order of
magnitude past it. **v1's decile sort is not a noisy version of the right
answer; at the start of the window it is close to a different answer.**

| | Survivorship (2026-09-30) | **Look-ahead (today)** |
|---|---|---|
| Unit | share of name-month panel | share of name-month panel |
| Measured | **11.4% absent** | **63.2% mis-decile'd** |
| Residual after the cheap fix | 4.89% | not yet known — 49.0pp of weight unmeasured |
| Cost to fix | $199/yr (Sharadar) | free for 34.1pp; a data source for the rest |
| Decays toward the present? | yes, 23.0% → 0.6% | yes, 73.9% → 2.0% |

---

## 1. What "look-ahead" actually is in this harness

`backtest.simulate_monthly_scores()` takes **one** Phase-1 snapshot of all 44
metrics, recomputes a few of them from the trailing price panel at each
month-end, and leaves the rest at their snapshot values. The module docstring
describes this as:

> *Look-ahead bias: Fundamental scores (Valuation, Quality, Growth, Revisions)
> are held constant from the Phase 1 snapshot. [...] Only Momentum and Risk
> metrics are recomputed from trailing prices.*

**That understates it in two ways, and the first is the whole finding.**

### 1a. The four-bucket decomposition

`lookahead.weight_buckets()` derives this from `config.yaml` rather than writing
it down, so a reweight cannot leave it stale:

| Bucket | Share of composite weight | What it means |
|---|---|---|
| **Recomputed** | **16.9%** | `return_12_1`, `return_6m`, `volatility`, `beta` — honestly point-in-time |
| **Price-restatable, held constant** | **28.0%** | A ratio of a price-independent fundamental to a market value. One month-end price restates it **exactly** |
| **Price-derived, held constant** | **6.1%** | `jensens_alpha`, `max_drawdown_1y` — derived from nothing but a price *history*, which the harness already holds |
| **Needs point-in-time data** | **49.0%** | Filings and analyst estimates. This is the bucket that needs a data source |
| **Held constant, total** | **83.1%** | |

So **34.1 points of composite weight are held at a future value despite
depending on nothing but price** — data the harness has loaded in memory at the
moment it decides not to use it. That is not a data-availability problem. It is
the harness ranking 2020 on 2026's share prices.

### 1b. "Only Momentum and Risk are recomputed" is not what the code does

`simulate_monthly_scores`'s `dynamic_cols` list holds four names. Momentum and
Risk carry **six** weighted metrics between them. `jensens_alpha` (25% of
momentum) and `max_drawdown_1y` (28.57% of risk) are **not** in that list, so
**6.1 of the 23 points** the docstring implies are recomputed are in fact frozen
at today's values — both of them pure functions of a price history.

`tests/test_lookahead.py::test_recomputed_matches_backtests_dynamic_cols` parses
`backtest.py` for that literal and fails if it moves, so this classification
cannot quietly go stale the way the docstring did.

---

## 2. The measurement

Two arms, identical in every respect but one:

* **Arm A (v1):** every static metric at its snapshot value; momentum/risk
  recomputed at the rebalance month exactly as `simulate_monthly_scores` does.
* **Arm B:** the same, with the six price-restatable metrics restated at that
  month's price.

81 rebalance months (2020-01 .. 2026-09), 502 names, **40,662 name-months**. The
snapshot is the committed `dashboard_data.js` from the 2026-10-01 data run; the
price panel is monthly adjusted closes from yfinance.

The restatement algebra holds every price-independent quantity — net income, FCF,
EBITDA, revenue, share count, the analyst target, the non-equity part of
enterprise value — at its snapshot value and moves only the price:

```
market_cap(m) = market_cap · r                  where r = P(m) / P(snapshot)
ev(m)         = ev + market_cap · (r − 1)       debt and cash do not move with the share price
earnings_yield(m) = earnings_yield / r
fcf_yield(m)      = fcf_yield · ev / ev(m)
ev_ebitda(m)      = ev_ebitda · ev(m) / ev
ev_sales(m)       = ev_sales  · ev(m) / ev
size_log_mcap(m)  = size_log_mcap − log(r)
price_target_upside(m) = pt_mean / (price · r) − 1, clamped
```

### What the gap is *not*

**It is not the restatement being more honest than arm A.** Arm B is still
look-ahead: it uses today's net income against 2020's price. The gap measures
only how much of v1's ranking is driven by the *price* half of a valuation
ratio being taken from the future. The 49.0 points that need point-in-time
filings are frozen in both arms, so the true figure is larger by an unknown
amount.

### Results

| Measure | Panel | 2020 | 2023 | 2026 |
|---|---|---|---|---|
| Decile assignment differs | **63.2%** | 71.2% | 67.5% | 35.4% |
| Differs by ≥ 2 deciles | **30.3%** | 39.9% | 34.6% | 7.0% |
| Quintile differs | **44.7%** | — | — | — |
| v1's top decile retained | **59.5%** | 43.3% | 54.7% | 85.6% |
| Spearman of the two composites | **0.77** median | 0.72 | 0.77 | 0.97 |
| Median \|rank change\| (of 502) | **45.5** | 53.3 | 45.5 | 14.1 |

Worst month, **2020-01**: 73.9% of names change decile, 42.2% by two or more,
Spearman **0.70**, median rank change **56.5** places, p90 **195.8** places, and
**43.1%** of v1's top decile survives.

### The monotone signature, which is how you know it is real

The error decays toward the present — 73.9% in 2020-01 to 2.0% in 2026-09 — with
a rank correlation of **0.86** between a month's age and its decile-change rate.
That is the same signature the survivorship measurement found (23.0% → 0.6%) and
for the same structural reason: the further back the test reaches, the more the
snapshot it is applying has drifted from what was knowable. At the snapshot month
itself the two arms are **identical to machine precision**, which is the
measurement's own null check.

---

## 3. Verification: is arm A really v1?

A measurement of a harness is worthless if it is measuring a different harness.
Two checks, both reproducible from the committed script:

**The null.** `restate_at_price` at `r = 1` reproduces the snapshot with a
maximum absolute difference of **0.0** across all five restated metrics, so the
entire measured migration is attributable to the restatement and nothing else —
not to payload rounding, not to the reconstruction. A regression test pins it.

**The chain.** Scoring the snapshot through `compute_sector_percentiles →
compute_category_scores → compute_composite` — `simulate_monthly_scores`'s exact
chain — gives a composite that differs from the published site by a median of
**0.61** points and a median of **8** rank places. That gap is **not** rounding:
repeating it from the full-precision run cache moves it only to 0.56 and 7.

It is one missing call. `run_screener.py` inserts
**`adjust_momentum_weight()`** between the category scores and the composite, and
`backtest.py` does not. Adding that one call collapses the gap: median absolute
composite difference **0.61 → 0.00**, median rank difference **8 → 2** places
(p95 7, three names over 1 point) — and run against the **full-precision** run
cache instead of the 4dp payload it goes to **0.00 and 0 places** (p95 1, one
name over 1 point). The committed script reports the payload figures, because the
payload is what a reader can check; the cache is gitignored.

So arm A is a faithful instance of `backtest.py`'s chain, the 4dp rounding
accounts for the last two rank places, and the rest of the original gap is a real
difference between the backtest and the live pipeline.

### Which is a second finding, recorded but not fixed

On the 2026-10-01 run the regime step moves **momentum 13 → 14.95** and
**valuation 22 → 20.05**. So **v1 backtests a weighting the site does not
publish**, and it has done so since the regime adjustment was added. Worse for
v2: the regime is read from the *current* run's volatility, so a historical
rebalance would need its own month's regime — a third look-ahead vector, inside
the weights rather than the metrics. Not fixed here, for the same reason nothing
else is: `plan/backtest-v2.md` forbids a half-fixed backtest, and this is a
finding about what v2 must get right, not a patch for v1.

---

## 4. Limitations, stated rather than buried

* **Adjusted prices reinvest dividends**, so `P_adj(m)/P_adj(T)` is below the raw
  price ratio by the cumulative payout. Corrected approximately with each name's
  own trailing yield, the headline moves from **73.9% → 75.3%** at the oldest
  month and **66.3% → 67.3%** at the midpoint. The conclusion does not depend on
  it, and the correction makes the bias slightly *larger*.
* **The ratio denominator is the panel's own latest close**, not the payload's
  price date; those differ by a median of **1.71%** (p90 4.93%) because the
  payload is an intraday-1-October figure and the panel's last bar is September.
  A near-uniform level shift in `r` barely moves a *cross-sectional* rank, and the
  newest month's Spearman of **0.9999** is the evidence for that. Using one price
  source for both ends of the ratio was preferred to mixing an unadjusted price
  into an adjusted series.
* **Share count is held at today's value.** That is deliberate — the experiment
  isolates price — but a real point-in-time market cap would also undo six years
  of buybacks and issuance, which would add to the gap, not subtract.
* **Both arms freeze 49.0pp of weight**, so this is a floor. How much the
  fundamentals half adds is the open question, and it needs point-in-time
  filings.
* **`dashboard_data.js` rounds `raw` to 4dp** and omits seven weight-0 metrics.
  Neither can bias the arm-A-vs-arm-B comparison, because both arms read the same
  rounded snapshot and the `r = 1` null is exactly zero.
* **The universe is today's constituents**, as v1's is. Survivorship is held
  fixed by construction so that the two biases are measured separately rather
  than compounded.

---

## 5. What the literature and practice say about the size of this

The measurement above is this repository's own, so the external evidence is about
whether a bias this large is plausible and what practitioners do about it.

* **Banz, R. W. & Breen, W. J. (1986), "Sample-Dependent Results Using
  Accounting and Market Data: Some Evidence", *Journal of Finance* 41(4),
  779-793** (September 1986; no published abstract, so the finding below is
  quoted from the secondary literature rather than the paper's own words). This is
  the paper that **named** both defects — *look-ahead bias* and
  *ex-post-selection bias*. Method: run the same tests on the standard Compustat
  database and on a bias-free one. Result: **rates of return on portfolios chosen
  from accounting data differed significantly between the two databases, implying
  different conclusions** about the relationship between accounting data and
  prices; they criticised the existing size and P/E literature on exactly these
  two grounds. The effect size that matters here is not a percentage — it is that
  the *conclusion* flipped, which is the same shape as this measurement's
  "63.2% of the panel is in a different decile".
* **The practitioner answer is a product you buy.** S&P Global sells
  point-in-time Compustat snapshots — *"a consistent view of historical financial
  data, both reported data and subsequent restatements, the way it appeared at the
  end of any month"*, with snapshots beginning in **1987**, marketed explicitly to
  *"avoid look ahead bias"*. A separate paid dataset existing for this single
  purpose is the professional statement that the ordinary file cannot answer the
  question, and it is why the fundamentals half of this bias is the expensive half.
* **Documented academic practice is a reporting lag, not a snapshot.** Fama &
  French (1992, *JF* 47(2), 427-465) match accounting data for fiscal years
  ending in calendar year *t−1* to returns from **July of year *t*** onward — a
  deliberate **minimum six-month gap**, chosen so that every filing used was
  already public. v1 applies a **negative** lag of up to **80 months**.
* **Where academia and practice agree, unusually.** No camp argues that a
  held-constant fundamental snapshot is acceptable over a multi-year window; the
  disagreement is only about how much lag is enough. That is a question v2 has to
  answer and this measurement does not.

**None of this is cited as justification for a methodology change**, and nothing
in this session changes a score. It is cited for why a look-ahead measurement was
worth a session at all.

### Sources, all accessed 2026-10-01

* Banz & Breen (1986), publication record:
  <https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1986.tb04548.x> and
  <https://ideas.repec.org/a/bla/jfinan/v41y1986i4p779-93.html> (confirms
  41(4), 779-793, September 1986; records "No abstract is available for this
  item", which is why the finding is quoted from secondary citations)
* *Journal of Finance* 41(4) issue listing:
  <https://afajof.org/issue/volume-41-issue-4/>
* Compustat point-in-time snapshots:
  <https://www.spglobal.com/content/dam/spglobal/mi/en/documents/general/Compustat-Brochure_Digital.pdf>
  and <https://www.marketplace.spglobal.com/en/datasets/compustat-financials-(8)>
* Fama & French (1992), "The Cross-Section of Expected Stock Returns",
  *JF* 47(2), 427-465 — the July-of-year-*t* convention is §I of that paper

---

## 6. What this changes about the plan

1. **The sequencing question is settled: look-ahead is the bigger bias, by 5.5x
   on a common unit.** `plan/backtest-v2.md` guessed the other way — step 2 is
   titled "Point-in-time universe (bigger bias, usually)". For this screener it
   is not. Step 3 outranks step 2.
2. **Do not buy the delisted-price feed yet.** The 2026-09-30 decision was to
   defer the $199/yr purchase until look-ahead was sized. It is now sized, and it
   argues the same way with more force: spending money to cut an 11.4% bias while
   a ≥63.2% bias is untouched buys nothing a reader can use.
3. **34.1pp of the 83.1pp held constant is free to fix** — the price-restatable
   28.0pp plus the price-derived 6.1pp. No vendor, no licence, no permission.
   That is the cheapest honest improvement available to v2 and it should be the
   first thing v2 does, before any purchase.
4. **The next measurement is the fundamentals half, and it is the one remaining
   unknown in step 1.** 49.0pp of weight needs point-in-time filings. SEC EDGAR's
   XBRL `companyconcept` endpoint is free, carries a `filed` date per fact, and
   covers most of the Quality, Growth and Investment inputs — so the *reporting
   lag* and the *drift* can both be measured without a vendor. Analyst estimates
   (`forward_eps_growth`, `fy1_revision_3m`, `analyst_surprise` — 9.0pp of the
   49) have no free retrospective source and may have to be reported as
   permanently unmeasurable rather than fixed.
5. **v2 must apply `adjust_momentum_weight` per rebalance month, from that
   month's regime.** Recorded in §3.

**No `METHODOLOGY_CHANGELOG.md` entry.** Nothing in `config.yaml`,
`factor_engine.py` or the published payload moved; no stock's composite or rank
changes. The changelog records changes to how stocks are scored, and an entry
that moved no number would dilute a file whose value is that every entry did.
The finding is recorded where it binds: here, `plan/backtest-v2.md`, and
`CLAUDE.md` priority 3.

**Rule 5 compliance.** No return, no Sharpe, no information coefficient and no
decile spread is computed anywhere in the measurement. Every figure above is a
property of the scoring harness — a weight share, a rank, a decile assignment, a
correlation between two rankings. There is no backtest output in this session to
bench.
