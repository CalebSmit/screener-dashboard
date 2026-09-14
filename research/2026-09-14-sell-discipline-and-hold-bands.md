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

*To be written 2026-09-16 (synthesis). Left deliberately empty.*

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

## 9. What would change my mind

- **On the band:** if, over a longer series, a 2× band still produces zero
  signals across a period containing a genuine large drawdown in a top-25 name,
  the band is too wide and the rule is decorative. Re-measure §6.3 at 60+
  comparable runs before committing to a width.
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
