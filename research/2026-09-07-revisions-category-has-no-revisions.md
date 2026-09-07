# The Revisions category contains no revisions

**Date:** 2026-09-07 (Monday, research)
**Author:** nightly code session
**Status:** complete. Recommends a methodology change; the change itself is
Wednesday's synthesis and Thursday's build.

---

## The question

The screener's sixth category is called **Revisions** and carries **10% of the
composite**. None of its five scored metrics is an estimate revision.

```
revisions (10% of composite)
  analyst_surprise          38   median EPS surprise, last 4 reported quarters
  earnings_acceleration     20   latest quarter's surprise % minus the prior quarter's
  consecutive_beat_streak   20   recency-weighted count of beats in last 4 quarters
  price_target_upside       12   consensus target / price - 1        (a level, not a change)
  short_interest_ratio      10   days to cover                       (not analyst data at all)
```

`config.yaml -> metric_weights.revisions`. The first three are all functions of
the *same* four rows of `Ticker.earnings_history`
(`factor_engine.py:1089-1130`) and together are **78% of the category = 7.8% of
the composite**.

So the decision this note informs is: **is a surprise-history category the
right thing to be spending 10% of the composite on in an S&P 500 universe, and
is a real revisions signal available?**

Two sub-questions turn out to matter more than the headline:

1. Does the *documented* effect these metrics are proxying (post-earnings
   announcement drift) still exist in large caps?
2. `config.yaml` and `SCREENER_OVERVIEW.md` both assert that a real revisions
   metric is impossible with the current data source. Is that true?

---

## 1. What the literature says

### 1.1 The foundational result: earnings momentum is real, and revisions are the strongest leg of it

**Chan, Jegadeesh & Lakonishok (1996), "Momentum Strategies", *Journal of
Finance* 51(5), 1681–1713.** The canonical study of earnings momentum. It
compares three surprise measures head to head:

| Signal | Definition | 6-month spread, top vs bottom decile | Sample |
|---|---|---|---|
| SUE | YoY EPS change / SD of earnings innovations | **+7.5%** | 1973–1993 |
| **REV6** | **6-month moving average of (consensus forecast revision / price)** | **+7.7%** (**+8.7%** at 12 months) | **IBES, 1977–1993** |
| ABR | 3-day abnormal return around the announcement | reported separately; smallest of the three | 1973–1993 |

Effect sizes and sample periods as reported in Jegadeesh's later survey
*Momentum* (2001), §6 "Earnings Momentum", which reproduces CJL's tables.

Two conditions worth carrying:

- **The revision leg is the strongest and the most robust.** Jegadeesh: "the
  analyst forecast revision strategy is remarkably robust. The profitability of
  this strategy is not sensitive to the specific definition of forecast
  revisions nor is it sensitive to the source of analyst forecasts. Also, both
  the SUE strategy and the forecast revision strategy have persisted for a
  fairly long period of time after the initial publication of the evidence."
- **It is short-lived.** "Similar to the SUE based strategy, the profitability
  of analyst forecast revision strategy is also relatively short lived." The
  6→12 month increment is +1.0% against +7.7% in the first six months.

**Replication before CJL.** Stickel (1991), Zacks database, NYSE/AMEX,
1981–1984: top-vs-bottom **5%** portfolios earn **+7.07%** on consensus
forecast revisions and **+6.36%** on individual analyst revisions. An earlier
study cited in the same survey forms Up/Down portfolios at a ±5% revision
threshold and finds **+3.1%**. Different definitions, different databases,
same sign and rough magnitude — which is what "robust" means here.

**Are surprise and revisions the same signal?** No. CJL report a rank
correlation of **0.440** between SUE and forecast revisions — the highest pair
in their table — and conclude the momentum variables "do not reflect the same
information. Rather, they capture different aspects of improvement or
deterioration in a company's performance."

### 1.2 The complication: in large caps, price momentum *is* earnings momentum

**Novy-Marx (2015), "Fundamentally, Momentum is Fundamental Momentum", NBER
Working Paper 20984.** US, **January 1975 – December 2012**. Signals: SUE (YoY
EPS change scaled by the SD of earnings innovations over the last eight
announcements) and CAR3 (3-day abnormal return around the most recent
announcement). Notably, **it uses no analyst-revision signal at all.**

Fama-MacBeth regressions (Table 1), coefficient on prior-year return
`r2,12` with t-stats in brackets:

| Specification | Full sample 1975–2012 | 1994–2012 |
|---|---|---|
| `r2,12` alone | 0.59 [2.84] | 0.38 [1.05] |
| `r2,12` + SUE + CAR3 + controls | 0.15 [0.70] | −0.00 [−0.00] |

Earnings surprise subsumes price momentum. But the number that matters most
for *this* screener is the size breakdown (Table 3), **average monthly excess
returns by size quintile, value-weighted**:

| Strategy | Q1 (small) | Q2 | Q3 | Q4 | **Q5 (large)** |
|---|---|---|---|---|---|
| Price momentum `WML` | 1.43 [5.48] | 0.88 [3.95] | 0.69 [3.06] | 0.47 [1.97] | **0.35 [1.48]** |
| **SUE** | 1.50 [15.6] | 0.76 [7.17] | 0.53 [5.21] | 0.26 [2.75] | **0.26 [2.46]**, α 0.29 [2.83] |
| **CAR3** | 1.29 [14.8] | 0.76 [9.41] | 0.46 [5.22] | 0.32 [3.77] | **0.20 [2.12]**, α 0.15 [1.66] |

Monotonic decay in size for all three. In the largest quintile — 333 names,
**72% of market cap**, the closest thing in the paper to an S&P 500 universe —
price momentum is **insignificant** (t=1.48) and CAR3's alpha is
**insignificant** (t=1.66). **SUE survives**: 0.26%/month, alpha 0.29%/month
with t=2.83.

That is the fair reading, and it cuts both ways. A *properly standardised*
earnings surprise still earned an alpha in large caps over 1975–2012. But the
margin is thin, and it is the smallest of any subsample in the paper.

### 1.3 The decisive result: PEAD is gone in this universe

**Martineau (2022), "Rest in Peace Post-Earnings Announcement Drift",
*Critical Finance Review* 11(3–4), 613–646.** Compustat sample 593,654
announcements (1973–2019); I/B/E/S sample **312,462 announcements
(1984–2019)**; NYSE/AMEX/NASDAQ, price > $1, market cap > $5M.

The surprise definition is the important part: **analyst earnings surprise =
(actual EPS − median analyst forecast) / price 5 days prior**, using forecasts
issued within 90 days of the announcement. That is an *analyst-estimate*
surprise — the same family as this screener's `analyst_surprise`, not a
time-series SUE.

Findings:

- **"Since 2006, analyst earnings surprises fail to positively predict
  post-announcement returns over 60 days for all-but-microcap stocks"** (and
  since 2016 for microcaps). 60-day drift coefficient for
  all-but-microcap: **2006–2010 = −0.001 (n.s.); 2016–2019 = −0.002,
  statistically significant and negative.**
- The return did not vanish, it **moved to the announcement date**. Two-day
  announcement return `BHAR[0,1]` for large stocks: **~20bps in 1984–1990 →
  ~120bps in 2016–2019**. Top-vs-bottom surprise spread on the announcement
  day reaches ~120bps by 2016–2019.
- Conclusion: prices now fully reflect the surprise on the announcement date.
  For large stocks, PEAD has been **non-existent since 2006**.

This is directly on point. **The screener is S&P 500 only** — every name in it
is "all-but-microcap", and most are in the top decile of US market cap. The
drift the surprise metrics are proxying is documented as absent in exactly this
universe for the last twenty years, and the return has relocated to a two-day
window a daily screener holding a monthly-horizon view cannot trade.

### 1.4 What the surprise metrics *are* legitimately evidence for

Being fair to the current construction: there is real support for a
meet-or-beat indicator, but it is not a drift story.

**Bartov, Givoly & Hayn (2002), "The rewards to meeting or beating earnings
expectations", *Journal of Accounting and Economics* 33(2), 173–204.** Firms
that meet or beat consensus "enjoy a **higher return over the quarter** than
firms with similar quarterly earnings forecast errors that fail to meet these
expectations" — a premium of almost 3%. The premium survives when the
meet-or-beat was likely achieved through earnings or expectations management.

Two limits, both from the paper itself:

1. **The ~3% premium is contemporaneous** — a return earned *over the quarter
   in which the MBE occurs*, not a forward return available to someone
   screening after the fact.
2. **"Leading indicator of future performance" means future *fundamentals*, not
   future returns.** The paper's Table 9 discussion is about firm performance
   "for both of the years following the MBE year".

So `consecutive_beat_streak` has a defensible reading as a **quality/
persistence** signal — habitual beaters tend to keep performing — which is a
different claim from "this predicts next month's return", and arguably belongs
in a different category. It does not rescue the PEAD rationale.

---

## 2. What practice does

### 2.1 Barra: revisions are a factor; surprise is not

**MSCI, "Datasheet — Barra US Total Market Equity Trading Model" (USFAST),
March 2015.** 24 style factors, 60 industry factors, 19,700+ assets. The
`Sentiment` factor is defined verbatim as:

> **Sentiment** — "Explains the return differences between stocks based on
> sell-side analyst revisions and news sentiment"
> Descriptors: • **Revision ratio** • **Change in analyst-predicted
> earnings-to-price** • **Change in analyst-predicted earnings per share**
> • Positive sentiment based on Event Sentiment Score • Positive sentiment
> based on Composite Sentiment Score • Sentiment dispersion based on
> Composite Sentiment Score • At-the-money skew

The first three descriptors are exactly the construct this screener says it
cannot build. And searched across the whole datasheet, **the word "surprise"
appears zero times** — in 24 style factors and their full descriptor lists,
there is no earnings-surprise descriptor anywhere. Barra models risk rather
than alpha, so this is evidence about what *co-moves* and is worth a factor of
its own; but the choice of revisions over surprise as the thing to measure is
explicit.

### 2.2 Zacks: the screener implements one of four components, and it is the weakest one

The **Zacks Rank** is the most widely distributed revisions-based ranking sold
to retail and advisors (redistributed by Fidelity among others), recalculated
nightly. Its four components:

| Zacks component | What it is | This screener |
|---|---|---|
| **Agreement** | share of analysts revising in the same direction | **absent** |
| **Magnitude** | size of the recent change in consensus estimate for current and next fiscal year | **absent** |
| **Upside** | Zacks' proprietary "most accurate estimate" vs consensus | absent (not replicable — proprietary) |
| **Surprise** | "a company's last few quarters' earnings per share surprises" | **this is the whole category** |

The overlap is exact and unflattering: Zacks' **Surprise** component is
literally "a company's last few quarters' EPS surprises", which is
`analyst_surprise` (median of the last four) plus `consecutive_beat_streak`
(recency-weighted count of the last four). The screener implements the one
component of four that is not a revision — in a system named for revisions.

Note what Zacks calls the product: *"Harnessing the Power of Earnings Estimate
Revisions"*. Agreement and Magnitude are the *estimate revision* legs, and they
map one-to-one onto data the screener already has access to (§3.2).

### 2.3 Where academia and practice disagree — and where they do not

**They agree on the signal.** Both camps put analyst estimate revisions at the
centre: CJL/Stickel measure it and find the largest, most robust spread; Barra
and Zacks build production systems on it. Nobody in either camp builds a
"revisions" construct out of surprise history alone.

**They disagree on whether surprise still works.** The academic verdict since
Martineau (2022) is that analyst-surprise drift is dead outside microcaps.
Practitioners kept the surprise leg anyway — Zacks still ships it as one of
four. Two honest reasons for that gap:

- **Horizon.** Zacks recalculates nightly and is marketed on 1–3 month views;
  academic PEAD tests use a fixed 60-trading-day window from the announcement.
  A practitioner blending surprise with three revision signals is not making a
  standalone PEAD bet.
- **Commercial inertia.** Surprise is intuitive, easy to display, and has been
  in the product since the 1980s. That is not evidence.

I take the academic side on the specific question of whether *surprise alone*
predicts forward returns in large caps, because Martineau's test uses the
analyst-based surprise definition this screener actually computes, on this
screener's actual universe, over the modern period. But I take the practitioner
side on the shape of the category: a revisions category should contain
revisions, and surprise can stay in it as one leg among several rather than as
78% of it.

---

## 3. Where this contradicts what we currently do

### 3.1 The category's stated rationale rests on an effect documented as absent here

`SCREENER_OVERVIEW.md:149`, public methodology page:

> "Estimate revisions and analyst targets are among the most powerful
> short-term return predictors. **When a company consistently beats earnings
> estimates, the stock price usually follows — but with a lag, which creates an
> opportunity.**"

The bolded sentence is post-earnings announcement drift, stated as the
mechanism justifying 10% of the composite. Martineau (2022) finds that drift
"non-existent since 2006" for all-but-microcap stocks, with the 2016–2019
coefficient significantly *negative*. The first half of the sentence
("estimate revisions … are among the most powerful short-term return
predictors") is well supported — and describes a signal the category does not
contain.

### 3.2 The claim that a revisions metric is impossible is false

Stated twice in public-facing docs and once in config:

- `config.yaml`, `metric_weights.revisions`: *"EPS forecast revision (change in
  consensus FWD EPS over 3-6 months) would be ideal here, but yfinance does not
  provide historical consensus data. Future enhancement: integrate I/B/E/S data
  from FactSet or Refinitiv."*
- `SCREENER_OVERVIEW.md:151` and `:476`: *"yfinance does not provide historical
  consensus EPS estimates, so the Revisions category cannot include the single
  most powerful revisions signal (change in forward EPS consensus over time).
  This would require a paid data source like FactSet or Refinitiv I/B/E/S."*

**Measured today against the installed yfinance 0.2.66:**

`Ticker.eps_trend` returns consensus EPS **now vs 7, 30, 60 and 90 days ago**,
for the current quarter, next quarter, current year (FY1) and next year (FY2):

```
        current  7daysAgo  30daysAgo  60daysAgo  90daysAgo     (AAPL, 2026-09-07)
0q      1.97754   1.97656    1.97549    2.00836    2.00767
+1q     2.91090   2.90859    2.90495    2.94829    2.94699
0y      8.81228   8.81249    8.80411    8.75958    8.75324
+1y     9.56691   9.53127    9.55399    9.68258    9.65517
```

`Ticker.eps_revisions` returns the up/down analyst counts over the last 7 and
30 days — the **Agreement / diffusion** construct:

```
        upLast7days  upLast30days  downLast30days  downLast7Days
0q                1             7              14              0
+1y               0             6              24              2
```

The `90daysAgo` column *is* "change in consensus forward EPS over 3 months". It
is Barra's "Change in analyst-predicted earnings per share" and Zacks'
"Magnitude"; `eps_revisions` is Zacks' "Agreement". Neither needs FactSet or
Refinitiv.

**Coverage, measured on a deterministic sample of 84 S&P 500 names (every 6th
ticker in `sp500_tickers.json`), 2026-09-07:**

| Signal | Coverage | 95% Wilson CI |
|---|---|---|
| 90-day FY1 consensus revision | **82/84 = 97.6%** | [91.7%, 99.3%] |
| 30-day FY1 consensus revision | 83/84 = 98.8% | [93.6%, 99.8%] |
| 30-day up/down diffusion | 80/84 = 95.2% | [88.4%, 98.1%] |
| `analyst_surprise` (for comparison) | **82/84 = 97.6%** | [91.7%, 99.3%] |

Coverage is **identical to the metric already carrying 38% of the category**.
The sparsity concern quoted in `SCREENER_OVERVIEW.md:149` as the reason the
category is capped at 10% does not apply to the revision signals specifically.

**Measured request cost** (instrumenting every `YfData` network method): a
fresh `Ticker` asking only for `.eps_trend` costs **1 quoteSummary request**.
Asking for `.info` first does *not* make it free — `.info` uses different
modules. But once `eps_trend` is fetched, **`eps_revisions` and
`earnings_estimate` cost 0 additional requests** — same cached response. So the
honest figure is **+1 HTTP request per ticker (~502/run), which yields all
three endpoints.** Not free; cheap and bounded.

### 3.3 The three surprise metrics are less independent than the 38/20/20 split implies

Spearman on the live published payload, 2026-09-07 run, 498 names with data:

|  | analyst_surprise | earnings_acceleration | consecutive_beat_streak |
|---|---|---|---|
| analyst_surprise | 1.000 | −0.070 | **0.497** |
| earnings_acceleration | −0.070 | 1.000 | 0.017 |
| consecutive_beat_streak | 0.497 | 0.017 | 1.000 |

`analyst_surprise` and `consecutive_beat_streak` correlate at **+0.497** — both
are "does this firm habitually beat", computed off the same four rows. Their
combined 58 points of category weight buy less diversification than two
independent metrics would. `earnings_acceleration` is genuinely independent
(−0.07) — it is a delta, not a level — and is the one surprise metric with no
overlap problem.

For context, this is the same shape of finding as
`research/2026-09-02-category-independence-synthesis.md`: metrics that look
distinct in the config sharing a single underlying input.

### 3.4 The surprise signal is structurally stale, and the metric does not know it

`analyst_surprise` is the **median of the last four reported quarters**
(`factor_engine.py:1106`). Even at its freshest it is dominated by data 3, 6
and 9 months old. There is no time-since-announcement term anywhere in the
computation.

Measured on the same 84-name sample, days between the last reported quarter and
2026-09-07:

```
count 83   mean 66.0   sd 11.4   min 38   median 69   max 99
  0-30  days since report:   0  ( 0.0%)
 30-60  days since report:  10  (12.0%)
 60-90  days since report:  71  (85.5%)
 90+    days since report:   2  ( 2.4%)
```

**Not one name in the sample is inside 30 days of its last report**, and 85.5%
sit in the 60–90 day band. Martineau's drift window is 60 trading days from
the announcement; on this run date the median name is already **69 calendar
days** past it.

Be careful with this number: S&P 500 reporting is clustered, so the
distribution swings with the calendar — 2026-09-07 falls mid-quarter, between
the July and October seasons, which is the stale end of the cycle. On a late-
July run date it would look much fresher. **The point is not that today's
number is representative; it is that the metric applies identical weight
regardless, and never knows which it is looking at.** A stock two days past a
big beat and a stock 89 days past the same beat receive the same
`analyst_surprise` treatment, and the median-of-four aggregation deliberately
blends across that distinction.

A revision signal has no such problem: consensus estimates are updated
continuously, so `eps_trend`'s 90-day window is 90 days old on every name, on
every run date, by construction.

### 3.5 What is working, and should not be disturbed

The revisions category is the **most independent category in the screener**.
Spearman against the other seven on the published 2026-09-07 run:

```
revisions vs momentum   +0.192      revisions vs risk       -0.157
revisions vs size       -0.124      revisions vs quality    +0.046
revisions vs growth     +0.040      revisions vs investment -0.045
revisions vs valuation  +0.023
```

Maximum absolute correlation **0.192**. Against the 2026-08-26 finding that
momentum and risk were 23% of composite weight off a single `Ticker.history()`
call, this category is the counter-example: it is genuinely orthogonal, and
`revisions_score` correlates with the composite at only +0.237, so it is doing
independent work rather than restating the ranking.

**This is an argument for fixing the category, not shrinking it.** A 10% slot
that is uncorrelated with everything else is valuable real estate; the problem
is what is parked in it.

---

## 4. What would change my mind

Falsifiable, in priority order:

1. **Martineau's result does not extend to a daily-rebalanced screener.** His
   test is a fixed 60-trading-day window from the announcement date. If
   surprise measured *as this screener measures it* — a 4-quarter median,
   sampled on arbitrary dates — turned out to carry forward-return information
   in S&P 500 names, the category is fine as-is. Testable here eventually via
   per-metric IC, but not on 3 effective observations at `1m`.
2. **The revision signal has no cross-sectional dispersion in this universe.**
   If `eps_trend`'s 90-day change is near-identical across S&P 500 names, it
   cannot rank them. **Partly checked and it survives**: on the 84-name sample
   the 90-day FY1 revision has p10 = −1.75%, median +1.72%, p90 = +15.76%,
   and **0% tied values** — real dispersion, cleanly rankable. But see the
   caution in §5 about the mean (+19.9%) versus median (+1.7%): the raw ratio
   has a fat right tail and is unusable unscaled.
3. **The diffusion variant is too coarse to rank.** **Confirmed as a problem.**
   `(up − down)/(up + down)` over 30 days is **62% tied** on the sample — many
   names pin at exactly +1.0. In a rank-scored system that is a large tie
   block. Magnitude (`eps_trend`, 0% tied) is the better primary construct
   here; diffusion is at best a secondary.
4. **A revision metric duplicates something already scored.** Measured on the
   sample: 90-day revision vs `analyst_surprise` Spearman = **+0.346**. CJL
   reported **0.440** between SUE and forecast revisions and called them
   distinct. +0.346 is lower than that, so the duplication case fails on the
   project's own coherence standard.
5. **The +1 request/ticker is unaffordable.** It is not — the data loop runs
   11.8–13.6 minutes against a 3-hour limit — but if fetch reliability
   regressed, the trade would need revisiting.

---

## 5. Recommendation

**Change warranted.** Specifically:

**(a) Correct the two false factual claims immediately.** `SCREENER_OVERVIEW.md`
lines 151 and 476, and the `config.yaml` comment, state that a forward-EPS
revision metric requires FactSet or Refinitiv. It requires `Ticker.eps_trend`,
which is already installed, and covers 97.6% of the universe. This is a
truthfulness fix to a public methodology page with no methodology content —
done today (§7).

**(b) Add a real revisions metric — `fy1_revision_3m`.** The Barra "Change in
analyst-predicted earnings per share" / Zacks "Magnitude" construct:

```
fy1_revision_3m = (eps_trend['0y','current'] - eps_trend['0y','90daysAgo'])
                  / |eps_trend['0y','90daysAgo']|
```

Design notes for Thursday, from the measurements above:

- **Scale by price, not by the estimate.** CJL's REV6 is
  *revision / stock price*, not revision / estimate. My sampled ratio has mean
  **+19.9%** against a median of **+1.7%** and sd **1.55** — a fat right tail
  driven by small denominators, exactly the pathology the `analyst_surprise`
  code already guards against with its `max(|e|, 0.10)` floor. Scaling by price
  is the literature's construction *and* removes the denominator problem.
  Rank-scoring makes the screener insensitive to monotone transforms
  (established 2026-09-03), so this matters only for the outlier tail — but
  that tail is where it matters.
- **FY1 (`0y`), not `0q`.** Quarterly consensus is mechanically dragged around
  by the reporting calendar; the annual number is what Barra and Zacks use.
- **90-day window.** Matches CJL's 6-month MA in spirit at the resolution
  available, and is the longest window `eps_trend` offers.
- **Not diffusion as the primary metric** — 62% ties (§4.3).

**(c) Reweight within the category, and say why in the changelog.** A defensible
target, argued from §1 and §2 rather than from any return series:

| Metric | Now | Proposed | Argument |
|---|---|---|---|
| `fy1_revision_3m` | — | **35** | CJL's strongest leg (+7.7%/6mo, decile); Barra + Zacks both build on it; 97.6% coverage; +0.346 vs surprise so it is additive |
| `analyst_surprise` | 38 | **15** | Martineau: no drift in this universe since 2006. Retained, not dropped — SUE keeps a t=2.83 alpha in Novy-Marx's largest quintile, and this is one leg of four in Zacks |
| `consecutive_beat_streak` | 20 | **10** | +0.497 with `analyst_surprise`; BGH supports it as a persistence signal, but that is a *contemporaneous* and *fundamental* result, not forward return |
| `earnings_acceleration` | 20 | **20** | unchanged — genuinely independent (−0.07) and the only surprise metric with no overlap |
| `price_target_upside` | 12 | **10** | unchanged in role; sell-side optimism bias already noted in config |
| `short_interest_ratio` | 10 | **10** | unchanged; not analyst data, but not the subject of this note |

Surprise-family weight falls from **78 → 45** and the category gains the signal
it is named after. **The category's 10% composite weight does not change** —
this note is about what is *inside* the slot, and §3.5 argues the slot itself is
earning its place.

**(d) Do not add a recency/time-since-announcement term to the surprise
metrics.** It was the obvious fix suggested by §3.4 and I considered it at
length. It is the wrong one: conditioning on recency would sharpen a signal
that Martineau shows does not exist in this universe, adding a term and a
paragraph of explanation to buy a better estimate of zero. The cost is also
real — every stock's score would start depending on its reporting calendar, and
the movers panel would show names shifting for no reason a student could see.
Fixing the freshness of the *wrong* signal is worse than replacing it with a
signal that is fresh by construction.

**What this note does not license:** changing the category's 10% composite
weight, or removing any metric outright. Both would need their own argument.

---

## 6. Wednesday's design section — the hypothesis and the coherence question

**Hypothesis.** Replacing 33 points of surprise-family weight with a 3-month
FY1 consensus revision makes the revisions category measure the construct it is
named for, without disturbing its independence from the other seven categories.

**The coherence question Wednesday must answer.** §3.5 shows the category is the
screener's most orthogonal. A revision signal correlates +0.346 with
`analyst_surprise` (§4.4) — lower than CJL's 0.440 — but nobody has measured
what it correlates with *across* categories. Two specific risks:

1. **Growth.** `forward_eps_growth` carries **45%** of the growth category and
   is built on the same FY1 consensus. A *level* and a *change in the level*
   are different objects, but they come from one number. If
   `fy1_revision_3m ~ forward_eps_growth` comes back high, the screener would
   be spending both a growth slot and a revisions slot on one estimate — the
   2026-08-26 momentum/risk failure in a new place. **Measure this first.**
2. **Momentum.** Novy-Marx's whole thesis is that price momentum *is* earnings
   momentum. Revisions and `return_12_1` are plausibly the same bet with a
   3-month lead. Current `revisions ~ momentum` is +0.192; if adding revisions
   pushes it materially above ~0.35 that is a coherence cost to weigh against
   the signal's literature support.

**The measurement that would refute it.** Recompute the published composite with
the proposed weights through the real `compute_composite` (the exact method the
2026-09-03 session used, err 0.0), and report: the two correlations above, the
change in `revisions ~ composite`, and the top-50 turnover. If the change moves
fewer names than deleting the category outright — the calibration bar 09-03
established, where deleting size moved 6 of the top 50 — then it is a
presentational change and should be argued on explainability alone, honestly.

**Explainability check, since the tool must be teachable.** "Analysts have
raised their earnings estimate for this company by 4% over the last three
months" is a sentence a student understands immediately, and is closer to the
"why does it rank there" summaries of priority 4 than "the median of its last
four quarterly EPS surprises is in the 71st percentile". That is a real
benefit, and it is *not* evidence — it belongs in the changelog under expected
effect, not under Evidence.

---

## 7. Done today

- This note.
- Corrected the two false claims in `SCREENER_OVERVIEW.md` (lines 151, 476) and
  the matching `config.yaml` comment. Facts only: no weight, threshold, metric
  or code path changed. See `NIGHTLY_LOG.md` 2026-09-07.

Not done today, deliberately: no metric added, no weight changed. Monday is
research; (b) and (c) above are Wednesday's synthesis and Thursday's build, and
the growth-overlap measurement in §6 must come first.

---

## Sources

- Chan, L.K.C., Jegadeesh, N. & Lakonishok, J. (1996). "Momentum Strategies."
  *Journal of Finance* 51(5), 1681–1713.
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1996.tb05222.x
- Jegadeesh, N. (2001). "Momentum" (survey), §6 Earnings Momentum — reproduces
  CJL's effect sizes and the Stickel (1991) replication.
  https://breesefine7110.tulane.edu/wp-content/uploads/sites/16/2015/10/Momentum-2001.pdf
- Novy-Marx, R. (2015). "Fundamentally, Momentum is Fundamental Momentum."
  NBER Working Paper 20984. https://www.nber.org/papers/w20984 (text used:
  https://mysimon.rochester.edu/novy-marx/research/FMFM.pdf)
- Martineau, C. (2022). "Rest in Peace Post-Earnings Announcement Drift."
  *Critical Finance Review* 11(3–4), 613–646.
  https://www.nowpublishers.com/article/Details/CFR-0122 ·
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111607
- Bartov, E., Givoly, D. & Hayn, C. (2002). "The rewards to meeting or beating
  earnings expectations." *Journal of Accounting and Economics* 33(2), 173–204.
  https://leeds-faculty.colorado.edu/rocks/bartov_givoly_hayn_2002.pdf
- Stickel, S. (1991), via Jegadeesh (2001) §6.
- MSCI (2015). "Datasheet — Barra US Total Market Equity Trading Model"
  (USFAST), March 2015. Style factor and descriptor tables, pp. 2–5.
  https://cdn2.hubspot.net/hubfs/2174119/Return%20Downloads/USFAST%20Datasheet%20-%20Barra%20US%20Total%20Market%20Equity%20Trading%20Model%20(1).pdf
- Zacks Investment Research. "Zacks Rank Guide" / *Harnessing the Power of
  Earnings Estimate Revisions*, components Agreement / Magnitude / Upside /
  Surprise. https://www.zackstrade.com/wp-content/uploads/2024/08/zacks-rank-guide-2022-09.pdf ·
  redistribution documented at https://www.fidelity.com/trading/research-firms/zacks-investment-research

**Measurements in this note** were taken on 2026-09-07 against yfinance 0.2.66,
the published `dashboard_data.js` from the 02:00 data run, and a deterministic
84-name sample (`sp500_tickers.json[::6]`). No figure from `backtest.py` or from
`improvement/live_ic_history.csv` is used anywhere above — the backtest is
benched until 2027-02-11 (rule 5) and the IC history holds 3 effective
observations at `1m` (rule 4).
