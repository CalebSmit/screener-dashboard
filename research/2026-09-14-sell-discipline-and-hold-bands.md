# Sell discipline: what the evidence supports, and what it forbids

**Date:** 2026-09-14 (Monday, research)
**Author:** nightly code session
**Status:** research complete. No code, weight or threshold has changed.
Wednesday's synthesis section is §8, left for 2026-09-16.

---

## The question

Priority 5 — the sell-side workflow — is the top open north-star item and has
been untouched for **40 days**. `plan/dashboard-north-star.md` describes it as
"a client-side watchlist/holdings list, deterioration flags against it, and a
'review queue' of owned names whose scores dropped materially."

Before building that, one thing has to be settled:

> **When should this screener tell a holder that something has deteriorated
> enough to be worth reviewing — and what must a sell surface avoid doing?**

The second half of that question turns out to matter more than the first. The
selling literature is unusual: it is less a body of evidence about *which rule
earns more* and more a body of evidence about *how sell decisions go wrong*.
Three of the four most-cited results are failure studies. A surface built
without reading them will reproduce the documented failures, and it will look
perfectly reasonable while doing so.

This is also the question the north-star plan itself parked as Monday research
question 3 ("What sell disciplines have evidence behind them? … What actually
survives out-of-sample?").

---

## 1. Selling is where professional skill disappears

**Akepanidtaworn, Di Mascio, Imas & Schmidt (2023), "Selling Fast and Buying
Slow: Heuristics and Trading Performance of Institutional Investors", *Journal
of Finance* 78(6), 3055–3098** (circulated as NBER w29076, 2021).

This is the most important paper for this feature, and it is recent enough that
most retail-facing tools predate it.

**Data.** 783 institutional portfolios, average size **$573 million**, complete
daily holdings and trades, **2000–2016**: 89 million fund-security-trading
dates and 4.4 million trades (2.0 million sells, 2.4 million buys). These are
experienced PMs running concentrated, high-tracking-error mandates — not retail
accounts.

**Method.** Each decision is compared to a counterfactual built *from the
manager's own holdings*: for a sell, "what if you had instead sold a randomly
chosen position you did not trade that day?" The authors argue, correctly, that
benchmarking a sale against an index is the wrong comparison, because the
relevant alternative to selling X is selling Y.

**Findings, with magnitudes:**

| Decision | vs. no-skill counterfactual |
|---|---|
| **Buys** | **outperform by over +100 bp/year** per dollar of purchase volume |
| **Sells** | **underperform by −80 bp/year** (preferred specification, factor-neutral random-sell) |

For scale, the authors note that active mutual fund management fees run 20–50
bp/year, and the cost gap between mutual funds and institutional separately
managed accounts is 10–35 bp/year. **The selling deficit is larger than the fee
these managers charge.** Robust to counterfactuals matched on size, value,
idiosyncratic volatility, prior returns, momentum, and characteristic
selectivity, and to sample splits for outflow pressure and price impact.

**The mechanism — and this is the part that constrains the design.** The
deficit is not a skill deficit. It is an *attention* deficit, and it shows up as
a specific, nameable heuristic:

> "PMs in our sample have substantially greater propensities to sell positions
> that are extreme on the salient dimension of prior returns: both the worst and
> best performing assets in the portfolio are sold at rates **more than 50
> percent higher** than assets that just under- or over-performed … In contrast,
> we observe no similar tendency to focus on extremes on the buying side."

The pattern survives controls for position size and holding length, and
survives stock-date fixed effects: *the same stock on the same day* is more
likely to be sold out of a portfolio where its return looks extreme relative to
its neighbours than out of one where it does not stand out.

**The natural experiment that proves it is attention, not ability.** On days
when a holding reports earnings — an exogenous, pre-scheduled shock to
attention — selling decisions **outperform non-announcement-day sells by more
than +150 bp/year**, and announcement-day sells actually *beat* the random-sell
counterfactual. Buying performance is unchanged on those days, as predicted,
because attention was already there. The authors conclude that PMs "do not lack
the fundamental skill to sell well — it is just not transferred."

**A second-order cost worth noting:** PMs rarely re-purchase what they sell, so
a sale permanently removes a name from the consideration set. Selling badly
does not just cost the trade; it costs the future idea.

**Heterogeneity, directly relevant here:** the selling deficit is *worst* among
**fundamentals-oriented managers running concentrated portfolios with high
tracking error**, and *smallest* among managers running momentum strategies. A
25-name, fundamentals-driven, high-active-share screen is squarely in the
worst-affected group.

### Why this is a design constraint, not a fun fact

The obvious way to build Priority 5 is a **review queue ranked by size of
move**. That is, literally and exactly, a machine for restricting the
consideration set to prior-return extremes — the documented error, automated and
presented as a feature. The paper's own diagnosis is that managers fail because
their attention goes to the extremes; a surface that sorts by move size does
their attention allocation for them, in the direction that loses money.

---

## 2. The reference point a sell surface must not offer

**Odean (1998), "Are Investors Reluctant to Realize Their Losses?", *Journal of
Finance* 53(5), 1775–1798.** 10,000 accounts at a large discount brokerage,
**1987–1993**.

**Effect size.** Over the full year, the proportion of gains realized (PGR) was
**0.233** against a proportion of losses realized (PLR) of **0.155** —
a **1.50× ratio**, difference −0.078, **t = −32**. Investors sell winners half
again as readily as losers.

**And they are wrong to.** Excess returns vs the CRSP value-weighted index,
after the sale of a winner vs after a *paper* loss that was held:

| Horizon | Winners sold | Losers held | Difference | p |
|---|---|---|---|---|
| 84 trading days | +0.47% | −0.56% | **+1.03%** | 0.002 |
| 252 trading days (1yr) | +2.35% | −1.06% | **+3.41%** | 0.001 |
| 504 trading days (2yr) | +6.45% | +2.87% | **+3.58%** | 0.014 |

The behaviour reverses in December (PGR 0.162 vs PLR 0.197, t = 4.6) — tax-loss
selling — which is itself evidence the effect is a framing artefact rather than
a considered view, since the same investors can clearly sell losers when a
salient reason to do so is placed in front of them.

**The design implication is narrow and concrete.** The disposition effect is
defined *relative to the purchase price*. It requires a cost basis to exist as
a reference point. A holdings surface that prominently displays "you are up
18% / down 12% on this position" installs the reference point that produces the
bias. Every retail portfolio tracker does this. The evidence says a decision
surface should not lead with it.

This one is cheap to honour and expensive to retrofit, which is why it belongs
in the note *before* the build rather than in a review afterwards.

---

## 3. What churn costs

**Barber & Odean (2000), "Trading Is Hazardous to Your Wealth", *Journal of
Finance* 55(2), 773–806.** 66,465 households at a large discount broker,
**1991–1996**.

| | Annual return |
|---|---|
| Market | **17.9%** |
| Average household (75% annual turnover) | 16.4% |
| **Highest-turnover households** | **11.4%** |

A **6.5 percentage point** gap between the most active traders and the market,
on the same asset class, in the same period. The paper also confirms Odean
(1999): "the stocks investors buy subsequently underperform the stocks they
sell" — the trades are not merely costly, they are backwards.

This is the number to put in front of an investment club. A sell surface that
generates more review events than a member can thoughtfully act on is not a
neutral addition; it has a measurable expected cost.

---

## 4. The one construction rule that is both well-evidenced and standard practice

### 4.1 The literature: trading hysteresis beats every other cost mitigation

**Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading
Costs", *Review of Financial Studies* 29(1), 104–147** (NBER w20721). CRSP +
Compustat, **July 1963 – December 2012**.

The paper evaluates three cost-mitigation techniques and reaches an unambiguous
ranking:

> "introducing a buy/hold spread, which allows investors to continue to hold
> stocks that they would not actively trade into, is **the single most effective
> simple cost mitigation strategy**."

**The rule, stated precisely.** An *sS* rule (after the Arrow–Harris–Marschak
1951 inventory model, which has an inaction region between the two thresholds):
you buy only when the signal enters the top **S**%, and you hold — restricting
sales — until it leaves the top **s**%. Their worked example is a **10%/20%**
rule: buy on entry to the top 10%, sell only on exit from the top 20%.

The justification is a statement about the signal, not about costs:

> "there is not much of a difference in expected returns between stocks in the
> 75–80% range of the distribution of a given return predictor and those in the
> 80–85% range."

**Effect size** (Table 5, UMD-like momentum factors, July 1973 – Dec 2012,
monthly %, net-on-net Fama-French 4-factor regression):

| Cost mitigation | Gross | T-costs | **Net** | **Net α** | t(α) |
|---|---|---|---|---|---|
| Restrict to low-cost universe | 0.66 | 0.35 | 0.31 | 0.17 | 3.06 |
| Staggered quarterly rebalancing | 0.62 | 0.26 | 0.37 | 0.19 | 6.62 |
| **Trading hysteresis (buy/hold spread)** | **0.77** | **0.26** | **0.51** | **0.33** | **8.81** |

Hysteresis nets **0.51%/month against 0.31%** for the low-cost-universe
approach — and its net alpha is roughly double either alternative, with the
largest t-statistic. The authors adopt the hysteresis-constructed momentum
factor for the rest of the paper on this basis.

**Two calibration facts worth carrying:**

- Round-trip costs for typical value-weighted strategies average **in excess of
  50 bp**; equal-weighted are two to three times higher.
- "Transaction costs generally reduce realized spreads by **more than 1% of the
  monthly one-sided turnover**" — i.e. 20% monthly turnover costs at least
  20 bp/month. A usable rule of thumb for telling a student what a trade costs.
- Anomalies with **under 50% one-sided monthly turnover** mostly keep
  significant net spreads; **few above it do**.

### 4.2 The practice: every major index provider does exactly this

This is the rare case where documented practice and the literature converge on
the same rule, arrived at independently.

**MSCI Momentum Indexes Methodology, July 2025**, §3.1.1:

> "To reduce Index turnover and enhance Index stability, buffer rules are
> applied at **50% of the fixed number of securities** … the MSCI ACWI Momentum
> Index targets 500 securities and the buffers are applied between **rank 251
> and 750**. The securities in the Parent Index with a Momentum rank at or above
> 250 will be added … on a priority basis. The existing constituents that have a
> Momentum rank **between 251 and 750** are then successively added until the
> number of securities … reaches 500."

Buy band: top 250. Hold band: top 750. A **tripled** band, on a quarterly
review cycle (February/May/August/November).

MSCI also applies a **turnover buffer** that implements only half of each
weight change (`x + (y−x)/2`) — and, tellingly, **"the turnover buffer is not
applied on deletions."** Additions are damped; exits are not. Asymmetry is
deliberate at the weight level too.

**S&P Dow Jones Indices** states the same idea as a *principle* rather than a
parameter (S&P Select Industry Indices Methodology, "Turnover"):

> "S&P Dow Jones Indices believes turnover in index membership should be avoided
> when possible. At times a company may appear to temporarily violate one or
> more of the addition criteria. However, **the addition criteria are for
> addition to an index, not for continued membership.** As a result, an index
> constituent that appears to violate criteria for addition to that index will
> not be deleted unless ongoing conditions warrant an index change."

That sentence is the whole finding in one line: **the buy test and the hold test
are different tests.** This screener currently has only one test.

**S&P Quality Indices** parameterise it as a **20% buffer**: stocks ranked in
the top 80% of the target stock count are auto-included; current constituents
within the top 120% of target count are then added in score order. For a
25-name target: auto-buy the top 20, hold to rank 30.
*(Verification note: spglobal.com returned HTTP 403 to automated fetches, so
this parameterisation is corroborated across two independent search retrievals
rather than read from the primary PDF, unlike every other citation in this
note. The S&P DJI principle quote above **is** from a primary document. Treat
the 80/120 figures as indicative until someone opens the PDF by hand.)*

### 4.3 The convergence

| Source | Buy band | Hold band | Ratio |
|---|---|---|---|
| Novy-Marx & Velikov worked example | top 10% | top 20% | **2.0×** |
| MSCI Momentum (ACWI) | top 250 of 500 target | top 750 | **3.0×** |
| S&P Quality | top 80% of target | top 120% of target | **1.5×** |

Three independent sources, two of them commercial products with real money
tracking them, all land on a hold band between **1.5× and 3× the buy band**.
None of them uses a symmetric rule. That is about as strong a practice consensus
as this kind of question produces.

---

## 5. What the evidence does *not* support: stop-losses

This is where academia and practice genuinely disagree, and the disagreement
matters because a price-based stop is the first thing most people reach for.

**Kaminski & Lo (2014), "When do stop-loss rules stop losses?", *Journal of
Financial Markets* 18, 234–254.**

The paper defines the **"stopping premium"** — the change in expected return
from overlaying a stop-loss rule — and derives it analytically per
return-generating process.

**Proposition 1 is a negative result, and it is unconditional.** If returns
follow a random walk (IID), the stopping premium is **Δμ = −p₀π**, where p₀ is
the probability of being stopped out and π the risk premium. It is *always*
negative:

> "If the portfolio follows a random walk … the stopping premium is always
> negative … stop-loss rules simply force the portfolio out of higher-yielding
> assets on occasion, thereby lowering the overall expected return without
> adding any benefits. In such cases, stop-loss rules never stop losses."

**The positive result is narrow.** Stops add value only when returns carry
positive serial correlation, and the premium is "directly proportional to the
magnitude of return persistence." Empirically, applied to a **stocks-versus-
bonds allocation using daily index futures, January 1993 – November 2011**, one
calibration using monthly-interval stops "can increase the return by **1.5%** and
decrease the volatility by **5%**, causing an increase in the Sharpe Ratio by as
much as **20%**." They find **no value at short sampling frequencies.**

**Why this does not license a per-stock stop in this tool.** The supporting
evidence is an **asset-class-level overlay** — index futures, switching between
equities and the risk-free asset, at monthly frequency. Practice applies stops
to *individual stocks* at *daily or intraday* frequency. That is the regime
Kaminski & Lo explicitly find worthless, on an object they did not test. The gap
between "portfolio-level momentum overlay at monthly frequency helps" and "put a
−15% stop on each holding" is not a detail; it is the entire conditioning set.

A second reason to refuse: a stop-loss is by construction a rule that fires on
**prior-return extremes**, which is precisely the heuristic §1 identifies as the
cause of the −80 bp/year institutional selling deficit.

**A second, smaller disagreement.** MSCI's Appendix III "Conditional
Rebalancing" triggers an unscheduled rebalance when the *parent index's*
annualised volatility rises month-on-month past the **95th percentile** of its
historical monthly volatility changes, and then scores momentum on 6-month
price momentum alone instead of the usual 6/12-month blend. This is explicitly
drawdown management, not cost management, and Novy-Marx & Velikov's framework
neither supports nor evaluates it — they only study cost mitigation. Practice is
going beyond the literature here.

Note what the trigger is *not*, though: it is a **market-level** condition, not
a single-name one. Even the most aggressive mainstream practitioner rule for
selling early does not key off one stock's move.

---

## 6. What I measured on this screener

All figures below are descriptive statistics of the published ranking's
stability, computed from `improvement/snapshots/` through
`history.select_comparable_runs()` (which applies the existing Spearman ≥ 0.50
comparability gate). **They are not backtest results and contain no forward
returns or IC**, so rules 4 and 5 of `CLAUDE.md` do not bite.

Series: 32 comparable runs, 2026-02-20 → 2026-09-14; consecutive pairs no more
than 7 days apart, per `MAX_NOISE_PAIR_GAP_DAYS`.

### 6.1 The top of the ranking is far stickier than the universe

Absolute rank change between consecutive runs:

| Population | n | p50 | p90 | p95 | max |
|---|---|---|---|---|---|
| Whole universe | 13,542 | 7 | 30 | **43** | 347 |
| **Names currently ranked in the top 25** | **675** | **1** | **6** | **10** | 72 |

A top-25 name moves by a *median of one rank* per run. The universe median is
seven. This is not a small difference and I had not expected it to be this
large.

### 6.2 Therefore the existing movers panel is nearly blind to holdings

`history.py` sets the "material mover" threshold at the 95th percentile of
universe-wide rank change — currently **43 ranks** (`rank_change_noise()`,
measured, n=13,542). By construction 5% of universe moves clear it.

**Top-25 names clear it 1 time in 675 holding-days — 0.15%.** A name can fall
from rank 3 to rank 27 — out of any sane buy band, a real deterioration worth a
holder's attention — and never once appear in "What Changed."

This is not a defect in the movers panel: it is a universe-discovery surface and
43 is the right threshold *for that job*. It is a demonstration that **a
holdings surface cannot be built on the movers panel's threshold**, which is the
cheapest and most tempting way to build Priority 5.

### 6.3 A strict sell rule would mostly generate trades that undo themselves

Dense weekday window, 18 runs, 2026-08-20 → 2026-09-14. A "sell signal" is a
name inside the buy band at run *t* whose rank at run *t+1* is outside the hold
band. "Returned" means it is back inside the buy band within five runs (≈ one
week).

| Rule | Holding-days | Sell signals | Returned within 5 runs |
|---|---|---|---|
| **Strict — buy 25, sell on leaving 25** | 425 | **31 (7.3%)** | **22 of 31 — 71%** |
| S&P-Quality-style 20% buffer — 20/30 | 341 | 1 (0.3%) | 0 |
| **MSCI/NMV-style doubling — 25/50** | 425 | **0 (0.0%)** | — |

**Nearly three in four sells from the strict rule are undone within a week.**
The worst next-run rank observed for *any* top-25 name in this window was 46, so
a hold band at 50 was never breached at all.

**Honest caveat on the zeroes.** Eighteen runs across 25 calendar days is a
short and fairly quiet window; "0 signals" means "no deterioration large enough
to breach a 2× band happened here", not "this band never fires". The robust
numbers are the **71% round-trip rate** on the strict rule and the rank-stability
distributions in §6.1, which are computed over the full 32-run comparable
series. The band widths should be re-measured once the series is longer.

---

## 7. Where the evidence contradicts what we do, or would naively do

1. **We have one test, not two.** `config.yaml → portfolio.num_stocks: 25` is a
   pure top-N cut. There is no hold band anywhere in the system. Every source in
   §4 — two of them commercial index products — uses a different, wider test for
   continued membership than for entry. S&P DJI states it as a principle.

2. **Ranking a review queue by size of move automates the documented error.**
   §1: institutional PMs sell prior-return extremes at rates **>50% higher**
   than middling positions, and this is the identified cause of a **−80 bp/year**
   deficit. A move-ranked queue is that heuristic, implemented.

3. **A holdings surface that leads with gain/loss vs cost basis installs the
   disposition effect.** §2: PGR/PLR = 1.50, t = −32, and the winners sold beat
   the losers held by **+3.41%** over the following year.

4. **The movers panel's 43-rank threshold cannot serve a holdings surface** —
   §6.2, it fires for a top-25 name 0.15% of the time.

5. **A per-stock stop-loss has no support** in the one paper that studied stops
   rigorously, whose positive result is confined to asset-class overlays at
   monthly frequency and whose random-walk result is negative by proposition.

6. **The screener's rebalance cadence is already right.** `config.yaml` line
   221 records "Rebalance cadence: quarterly (manual)". MSCI Momentum reviews
   quarterly; NMV's staggered-quarterly variant beats the low-cost-universe
   variant on net alpha. Nothing here argues for reviewing more often, and §3
   argues against it. **No change warranted** — recording this so a future
   session does not relitigate it.

---

## 8. Wednesday's design section

*Written 2026-09-16 (synthesis). Every number below is reproduced by
`research/measurements/2026-09-16-hold-band-and-input-churn.py`, which reads
`improvement/snapshots/` through the same comparability gate `history.py` uses.
Descriptive statistics on published scores only — no forward returns, no IC, no
backtest, so `CLAUDE.md` rules 4 and 5 do not bite.*

> **Update 2026-09-15 (Tuesday, product day).** The **list** half of Priority 5
> shipped ahead of this section - the My Holdings panel, built to constraints
> 2 and 3 below rather than to the band. `METHODOLOGY_CHANGELOG.md` 2026-09-15;
> `tests/test_holdings_panel.py`, 61 tests. Nothing in §8 was pre-empted: the
> panel sets **no threshold of any kind**, which is exactly what question 1
> exists to decide. Two of the five questions moved, though, and the synthesis
> should start from where they now are:
>
> - **Question 2 gained a measurement.** Across the 500 live stocks with a
>   one-month category delta, the largest mover is **Risk 34.0%, Revisions
>   29.0%, Momentum 26.4%, Valuation 3.2%, Investment 3.2%, Growth 2.8%, Size
>   1.2%, Quality 0.2% (one stock in 500)**. ~90% of one-month category
>   movement comes from the three categories fed by daily prices and estimates.
>   **A deterioration trigger keyed to Quality or Growth would essentially
>   never fire at monthly cadence** - which rules out the most intuitive form of
>   "fundamental deterioration" and pushes question 2's answer toward Revisions
>   and toward question 4's earnings dates.
> - **Question 3 is answered in the shipped code**: the queue shows
>   **everything**, annotated, never a filtered subset. §1 is unambiguous that
>   the failure is a restricted consideration set, and the alternative reading
>   would have automated the documented error. If the synthesis wants to revisit
>   it, it needs an argument against Akepanidtaworn et al., not a UI preference.
>
> Questions **1 (band width), 4 (earnings dates) and 5 (implied turnover)** are
> untouched and still open.

The questions it should settle, in priority order:

1. **Band width.** §4.3 gives a practice range of 1.5×–3×. §6.3 says 2× (25/50)
   never fired in the sample window and 1.5× (20/30) fired once. Is a 2× band
   too wide to ever be useful on this screener, and what evidence would tell us?
   A band that never fires is not a conservative feature, it is a dead one.
2. **What the trigger should key on instead of rank.** §1's earnings-day result
   (+150 bp/year) says information-driven sells are the good ones. This screener
   already carries `fy1_revision_3m`, `Composite_Confidence`, the trap flags and
   the category `contrib` attribution. A deterioration flag anchored to *why the
   score fell* rather than *how far it fell* is both better-evidenced and more
   in keeping with "decision support, not a recommendation engine".
3. **Whether the review queue shows everything or only the flagged.** §1 says
   the failure is a restricted consideration set. The coherent answer may be
   that the queue lists **all** holdings every time, with the flagged ones
   annotated rather than filtered — which is a different UI than the north-star
   plan's "review queue of names that dropped materially".
4. **Where earnings dates fit.** North-star gap 4 ("nothing about timing or
   catalysts") and §1's natural experiment point at the same feature from
   opposite directions. That is a real coherence finding and should not be lost.
5. **What must be measured afterwards.** Turnover implied by whatever rule is
   chosen, in trades per holding per year, against NMV's >1%-of-monthly-turnover
   cost rule of thumb.

---

### 8.0 The correction that reframes the rest: §6.3 measured the wrong thing

§6.3 reported that a 25/50 band "produced zero signals" across an 18-run window
and treated that as evidence the band might be too wide to be useful. **That
zero was an artifact of the estimator, not a property of the band.**

§6.3 walked *one path* through 18 consecutive runs. Taking instead **every
ordered pair of comparable runs** at a given calendar spacing — 34 runs now,
2026-02-20 to 2026-09-16 — a 2× band fires at **1.65%** of weekly holding-looks
and **1.93%** of monthly ones. It fires. It is rare, not dead.

But the pairwise estimator has a defect of its own, and it is one this project
has already paid for once. **34 runs yield 72 pairs at 12–18 day spacing**, so
each run feeds many pairs and the observations are nowhere near independent —
the identical trap `research/2026-08-10-ic-evidence-independence.md` found in
the IC series, which `improvement_engine._effective_observations()` now guards
against. It applies to rank-migration statistics exactly as it applies to ICs,
and nobody had noticed.

Restricting to a **maximal non-overlapping set of pairs** changes the numbers
substantially:

| spacing | pairs (all → disjoint) | B=25 | B=35 | B=50 |
|---|---|---|---|---|
| 1 day | 20 → 20 | 8.60% → 8.60% | 1.80% → 1.80% | 0.60% → 0.60% |
| 5–9 days | 68 → **8** | 15.00% → 16.50% | 4.24% → **8.00%** | 1.65% → **4.50%** |
| 12–18 days | 72 → **4** | 19.13% → 26.73% | 7.49% → **15.84%** | 3.99% → **10.89%** |
| 25–35 days | 37 → **2** | 24.14% → 24.00% | 9.44% → **16.00%** | 1.93% → **4.00%** |

The overlapping estimator **understates breach rates at the wide bands by
2–3×** — the same order as the ~2.35× independence overstatement the IC note
measured. And the honest pair counts are **8 weekly, 4 fortnightly, 2 monthly**.

So the true position is not "a 2× band never fires". It is: **we have two
independent monthly looks at this question.** That is the same evidential state
the improvement engine is in — 2–3 effective observations against a gate of 8 —
arrived at independently, measured by the same estimator, and it earns the same
answer: *do not act yet, and say so plainly.*

### 8.1 Q1 — band width: do not commit, and fix the stopping rule's unit

**A band is warranted. Its width is not determinable from what we have.**

The evidence *for* a band is the wasted-trade rate, and it is the one result
robust across every cadence measured. Of the names that breach a strict top-25
boundary, the share back inside the top 25 at the very next review:

| cadence (disjoint triples) | B=25 | B=30 | B=35 | B=40 | B=50 |
|---|---|---|---|---|---|
| 1 day (9) | **37.5%** | 12.5% | 0.0% | 0.0% | 0.0% |
| 2–4 days (7) | **47.4%** | 44.4% | 31.2% | 33.3% | 25.0% |
| 5–9 days (3) | **31.0%** | 20.8% | 5.9% | 3.4% | 0.0% |

**The strict rule — the only rule this screener has — wastes between a third
and a half of the trades it implies**, at every cadence, and widening the band
monotonically reduces that at every cadence. Both statements survive the
independence correction because they are directional and they replicate three
times out of three.

**Where the waste stops mattering does not replicate.** At B=35 the three rows
say 0.0%, 31.2% and 5.9%. Those rest on 9, 7 and 3 disjoint triples and on 4,
16 and 34 breach events. There is no width in the 1.5×–3× practice range that
this data distinguishes from any other.

**§9's stopping rule is right in spirit and wrong in unit.** It asks for "60+
comparable runs". Run count is not the binding quantity: 60 runs at daily
cadence is still only two or three independent *monthly* looks, because the
independent unit is a non-overlapping window at the review cadence, not a file
in `improvement/snapshots/`. **Restated:**

> **Do not commit to a band width until there are at least 8 disjoint
> observation windows at the review cadence the band will govern.** Today:
> **2 monthly**. At one per month of continuous running from the dense window's
> start (2026-08-10), that is roughly **2027-04** — within a month of the date
> the improvement engine reaches its own 8-observation gate, for the same
> reason. Re-run the measurement script; read the DISJOINT column.

**Pre-registered decision rule, so this is mechanical later rather than
re-litigated:** at ≥8 disjoint monthly windows, choose the narrowest band whose
wasted-trade rate is below half the strict rule's, and whose implied monthly
one-sided turnover is under NMV's 50%. If two widths qualify, take the
narrower — signal given up is a real cost and the practice range's upper end
(3×) comes from 500-name quarterly indices, not 25-name screens.

### 8.2 Q5 — turnover: the cadence binds, not the band

| review cadence | strict top-25, monthly one-sided turnover |
|---|---|
| every run (23 reviews, 37 contiguous days, one path) | **121.8%** |
| monthly spacing (disjoint pairwise) | **24.0%** |

Novy-Marx & Velikov (2016): anomalies under ~50% monthly one-sided turnover
mostly survive trading costs; few above it do. **Daily action on a strict
top-25 rule sits at more than twice the level above which NMV find anomalies
stop surviving costs** — and that conclusion tolerates a 2.4× error in the
estimate before it changes. Monthly review of the same rule sits comfortably
inside.

**This is the coherence gap, and it is a product defect rather than a
methodology one.** `config.yaml` line 221 records "Rebalance cadence: quarterly
(manual)". The dashboard regenerates **every weekday** and says **nothing**
about cadence anywhere — grepping `generate_dashboard.py` for "quarterly"
returns one unrelated data-source label. A holdings panel that redraws a rank
every morning implicitly invites a reader to act on it every morning, and the
methodology's own answer to that is a number 2.4× outside the region the
literature says survives costs.

**So the first thing to build is not a band. It is telling the reader what
cadence the tool is built for.** That costs nothing, needs no width, and is
what the evidence most clearly supports.

### 8.3 Q2 — the trigger keys on input stability, not size of move

The most intuitive reading of "fundamental deterioration" was already ruled out
on 2026-09-15: at monthly cadence the largest category move is Risk 34.0%,
Revisions 29.0%, Momentum 26.4% — and **Quality 0.2%, one stock in 500**. A
trigger keyed to the fundamentals categories would essentially never fire.

What the measurement adds is the opposite failure — moves that are **not
information at all**. When a metric's availability changes between runs, its
category renormalises over a different metric set and the score moves because
the *measurement* changed, not the company. This is the FCX case recorded in
`CLAUDE.md` priority 1.5, now counted over 24 run-pairs and 12,044
ticker-transitions:

| input churn | n | median &#124;rank change&#124; | p90 | share worsening |
|---|---|---|---|---|
| none | 11,450 | **6** | 24 | 44.6% |
| 1 metric | 447 | 7 | 28 | 47.0% |
| 2–3 metrics | 143 | **21** | 70 | 52.4% |
| ≥4 metrics | 4 | 48.5 | 129 | — |

**Three things follow, and the first is not a statistical claim at all.**

1. **The mechanism is arithmetic.** Renormalising a category over a different
   metric set moves the score; that is what the code does, by design. So the
   existence of this effect needs no significance test. Only its size is
   uncertain.
2. **A single lost metric is indistinguishable from noise** (median 7 vs 6).
   **Two or more triples the median rank move** (21 vs 6). Any flag should arm
   at ≥2, not ≥1 — otherwise it fires on 4.93% of transitions instead of 1.22%
   and mostly says nothing.
3. **It is noise, not deterioration.** Churn ≥2 leaves 52.4% of names worse off
   against a 44.6% baseline — nearly symmetric. It scatters ranks; it does not
   systematically push them down. **That is exactly the signal a sell surface
   must not present as a reason to sell.**

Among the specific event a holdings panel exists to surface — a top-25 name
leaving the top 25 — **6 of 43 exits (14.0%) coincided with input churn against
3.8% of the names that stayed.** Treat that ratio as indicative: it is 6 events
across 24 correlated run-pairs, and the significance tests the script prints
(Fisher p=0.009, Mann-Whitney p=7×10⁻²⁵) assume an independence that §8.0 has
just shown does not hold. The mechanism is certain; the magnitude is
provisional. The fix is justified anyway because it costs one sentence.

**And the tool already knows.** `stock_summary._sentence_confidence()` states
the coverage **level** — "The score rests on 41 of 45 metrics" — but never the
**change**. `history.py` carries rank, composite and category scores between
runs, and no metric count. One missing quantity is the whole gap.

**This unifies three separately-recorded items**: §8 question 2, `CLAUDE.md`
priority 1.5's open product gap ("the movers panel cannot distinguish 'moved on
new information' from 'moved because two inputs went missing'"), and yesterday's
`change_driver`, which names the category that moved but not whether the move
was real.

### 8.4 Q4 — earnings dates: available, unread, and display-only

Monday's claim that "the fetch already touches the provider response that
carries it" is **correct, and now verified rather than assumed.** The `.info`
dict already pulled at `factor_engine.py:744` carries, checked live against
AAPL/HST/EXPE today:

- `earningsTimestampStart` / `earningsTimestampEnd` — the **next** scheduled report
- `earningsTimestamp` — the last one reported
- **`isEarningsDateEstimate`** — whether that date is confirmed or a guess

Nothing in the repository reads any of them; a grep for `earningsDate`,
`earningsTimestamp` and `calendar` finds no consumer. So this is a genuine
zero-API-cost addition, on the same footing as the business descriptions added
2026-08-26.

Two constraints, both of which follow from things already settled:

- **Display-only, never scored.** Same rule as the descriptions; a proximity-to-
  earnings number entering `raw`/`pct` would be a new factor smuggled in as a
  UI feature.
- **`isEarningsDateEstimate` must be shown, not hidden.** EXPE's next date is
  flagged as an estimate today. Presenting an estimated date with the same
  confidence as a confirmed one is precisely the false precision this tool
  exists not to emit.

The evidence positively endorses this one: earnings-day sells beat
non-announcement-day sells by **+150 bp/year** (§1) and are the *only* selling
behaviour in Akepanidtaworn et al. that beats its counterfactual. It is the one
place the literature says attention is well spent.

### 8.5 Q3 — settled in shipped code

Answered 2026-09-15: the queue lists **all** holdings every time, annotated
rather than filtered. Revisiting it requires an argument against Akepanidtaworn
et al., not a UI preference.

### 8.6 What this implies for the other seven categories

The synthesis question proper, and the answer is uncomfortable.

**A hold band on composite rank is, at monthly cadence, mostly a band on
price.** ~90% of one-month category movement comes from Risk, Revisions and
Momentum; `CLAUDE.md` priority 1.5 records that momentum and risk are **23% of
composite weight and 100% derived from a single `Ticker.history()` call**. The
five fundamentals categories barely move a rank between quarterly filings —
Quality moved materially for one stock in 500 over a month.

So a rank-triggered sell rule is substantially a price-triggered sell rule
wearing eight categories as a costume. That is not a stop-loss, but it shares
the input Kaminski & Lo (2014) found has no support at single-name frequency,
and it deserves saying out loud rather than being discovered later.

**Two consequences:**

1. **`change_driver` is load-bearing, not decorative.** It is the only thing on
   the surface that distinguishes "this fell because the business changed"
   (essentially never, monthly) from "this fell because the price moved"
   (usually). It shipped 2026-09-15 looking like a nicety; it is the mechanism
   that keeps a rank move interpretable.
2. **Any future deterioration trigger must be category-aware.** A Quality or
   Growth deterioration is rare enough that when it *does* happen it is far more
   informative than the same-sized Momentum move — and a composite-rank band
   weights them identically. Do not build a composite-only trigger and call the
   fundamentals covered.

### 8.7 What Thursday should build, in order

1. **Say what cadence the tool is for.** §8.2. No threshold, no new data, and
   it is the gap the evidence most clearly supports.
   → **Built 2026-09-17.** `portfolio.review_cadence` is a config key rather
   than a comment, surfaced as `D.cadence` and stated on the holdings panel,
   the What Changed footnote and the holdings footnote (long form, with the
   121.8% / 24.0% / ~50% turnover numbers and the NMV citation).
   `tests/test_review_cadence.py`, 40 tests, 37 failing against the pre-change
   generator.
2. **Flag input-availability change on the holdings surface.** §8.3. Arm at
   ≥2 metrics, state it as a caveat on the move rather than a reason to act,
   and extend `history.py` to carry a per-ticker metric count so the sentence
   can be built. Note the pre-2026-03-09 snapshots carry no percentile columns —
   handle their absence rather than assuming the schema.
   → **Built 2026-09-17.** `history.py` carries per-ticker metric *availability*
   (the missing set, not a count) and emits `ch: [lost, gained]`;
   `stock_summary._sentence_input_churn` arms at ≥2. Fires for **27 of 502**
   stocks on the 2026-09-17 run against the ~1-month baseline — higher than
   §8.3's 1.22% because that figure was measured on consecutive runs ≤7 days
   apart and the drilldown's preferred baseline is ~28 days.
   `tests/test_input_churn.py`, 58 tests, 52 failing against the pre-change code.

   **One design point the plan did not anticipate.** A *count* is not enough: a
   net count difference of zero can hide one metric dropping out as another
   returns. The implementation compares the availability **sets**, over the
   intersection of columns both runs carry — the snapshot schema has grown
   (`fy1_revision_3m_pct` appears part-way through), and counting a column that
   did not exist yet as a metric that went missing would flag the entire
   universe on the day a metric was added.
3. **Earnings dates, display-only, with the estimate flag.** §8.4.
   → **Not built 2026-09-17.** Items 1 and 2 were taken together as one coherent
   change — both are about not inviting action the tool cannot justify — and
   this one is separable: it touches the fetch layer rather than the history
   spine, and needs a full refetch to populate. Still endorsed by the evidence
   (§8.4); it is the next session's item, not a deferral on the merits.

**Explicitly not built:** the hold band. §8.1 — 2 independent monthly looks,
and a pre-registered rule for when to revisit.

## 9. What would change my mind

- **On the band:** ~~if, over a longer series, a 2× band still produces zero
  signals across a period containing a genuine large drawdown in a top-25 name,
  the band is too wide and the rule is decorative. Re-measure §6.3 at 60+
  comparable runs before committing to a width.~~

  **Superseded 2026-09-16 (§8.0, §8.1).** The premise was wrong twice over. A 2×
  band does *not* produce zero signals — that was an artifact of measuring one
  path through 18 runs; over all comparable pairs it fires at 1.65–4.50% of
  holding-looks depending on the estimator. And "60+ comparable runs" is the
  wrong unit: run count is not the binding quantity, because pairs drawn from a
  daily series overlap almost completely. The replacement, restated in the unit
  that actually binds: **do not commit to a width until there are ≥8 disjoint
  observation windows at the review cadence the band will govern** — today
  there are **2 monthly**, projecting to roughly 2027-04. Re-run
  `research/measurements/2026-09-16-hold-band-and-input-churn.py` and read the
  DISJOINT column, not the ALL PAIRS one.
- **On hysteresis generally:** NMV's result is a *cost*-mitigation result for a
  long/short factor portfolio. If a hold band were shown to materially degrade
  the signal for a long-only 25-name book — the tool's actual use — the cost
  saving could be outweighed. NMV's own argument that the 75–80% and 80–85%
  buckets have near-identical expected returns is the reason to doubt this, but
  it is an argument, not a measurement on this universe.
- **On the disposition-effect constraint:** this one I do not expect to move.
  It is a design cost of essentially zero and the evidence is 28 years old and
  replicated.

---

## Sources

**Literature**

1. Akepanidtaworn, K., Di Mascio, R., Imas, A., & Schmidt, L. (2023). "Selling
   Fast and Buying Slow: Heuristics and Trading Performance of Institutional
   Investors." *Journal of Finance* 78(6), 3055–3098. NBER Working Paper 29076
   (read in full for this note).
2. Odean, T. (1998). "Are Investors Reluctant to Realize Their Losses?"
   *Journal of Finance* 53(5), 1775–1798. Tables V and VI read directly.
3. Barber, B. M., & Odean, T. (2000). "Trading Is Hazardous to Your Wealth: The
   Common Stock Investment Performance of Individual Investors." *Journal of
   Finance* 55(2), 773–806.
4. Novy-Marx, R., & Velikov, M. (2016). "A Taxonomy of Anomalies and Their
   Trading Costs." *Review of Financial Studies* 29(1), 104–147. NBER Working
   Paper 20721; Table 5 and §5.3 read directly.
5. Kaminski, K. M., & Lo, A. W. (2014). "When do stop-loss rules stop losses?"
   *Journal of Financial Markets* 18, 234–254. Proposition 1 and the empirical
   section read directly.

**Documented practice**

6. MSCI (July 2025). *MSCI Momentum Indexes Methodology*, §3.1.1 Buffer Rules,
   §3.1.2 Turnover Buffer, Appendix III Conditional Rebalancing. Primary
   document, read directly.
7. S&P Dow Jones Indices. *S&P Select Industry Indices Methodology*, "Turnover".
   Primary document, read directly.
8. S&P Dow Jones Indices. *S&P Quality Indices Methodology* — 20% buffer
   (top 80% / top 120% of target count). **Secondary retrieval only**; see the
   verification note in §4.2.

**Measured in this repository (descriptive; no forward returns, no IC, no
backtest)**

9. `improvement/snapshots/`, 32 comparable runs 2026-02-20 → 2026-09-14, via
   `history.select_comparable_runs()` and `history.rank_change_noise()`.
