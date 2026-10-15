# The options signal: what the expected move, ATM IV, put skew and put/call open interest are worth

**2026-10-12. Research day. `plan/context-layer.md` queue item 4 — one research note per
context signal. This is the note for the options panel.**

No production code follows from this note today. It ends with four recommendations, three of
which are changes to *what the page says*, not to any score. The options signal is context-only
and this note does not argue for changing that — it argues the opposite, with sources.

> ### Correction, 2026-10-14 — read this before §4 and §5
>
> **One conclusion below is wrong and is withdrawn: that Xing–Zhang–Zhao's skew decays faster
> than this screener acts.** This note had XZZ second-hand from a survey. Read from the paper,
> its **Table 4 Panel B** measures the same quintile spread at a **four-week** holding period —
> **6.52% a year, t 2.70** — and at 6–7% a year with *rising* t-statistics (3.73 → 4.31) out to
> **28 weeks**; the abstract and §I say the predictability "persists for at least 6 months".
> §4 below conflated the paper's weekly *rebalancing* with its decay *horizon*.
>
> §5's headline sentence — "Every option-implied signal with real support in the literature
> decays faster than this screener acts" — is therefore **not true of XZZ**. It stands for Pan &
> Poteshman, Cremers & Weinbaum and Fodor et al., whose horizons are as described.
>
> **What still holds:** the skew is not a candidate for the score today, for the other two
> reasons given here (post-publication shrinkage, and the context layer's own observation gate),
> and all three findings in "Findings against what the page does today" were correct and are now
> fixed. **What changes:** the prior for reading the first `1m` record from `context_eval.py`.
> Do not expect nothing at one month; the paper claims something there.
>
> Full working, the construction the page now computes, and the measured effect of the change:
> **`research/2026-10-14-volatility-skew-construction.md`**.

---

## 1. The question

The "Before you decide" panel shows four option-derived numbers for ~455 of 501 stocks. None of
them enters a score (CLAUDE.md settled row "ctx"). The questions this note answers:

1. For each of the four, what does the literature actually find, how big is it, and under what
   conditions did it hold?
2. How do quant shops and index providers treat option-implied data in *stock selection*?
3. Does what the page displays match what the evidence supports — in construction, in horizon,
   and in the sentence attached to it?

## 2. What the screener computes today

From `context_signals.options_context` (read 2026-10-12), on **one** expiry — the first expiry
after a report due within 60 days, else the expiry nearest 30 days (`choose_expiry`):

| Shown as | Construction in the code |
|---|---|
| Expected move | (ATM call mid + ATM put mid) ÷ spot — a straddle over the price |
| Implied volatility (at the money) | mean of the ATM call IV and ATM put IV |
| Put skew | IV of the put nearest **0.90 × spot** (accepted within ±0.04 moneyness) **minus the ATM IV above** |
| Put / call open interest | Σ put open interest ÷ Σ call open interest, **all strikes on that expiry**, shown as a level |

Quotes come from an after-close cache (`options_cache`), because overnight the source returns
every bid, ask and IV as zero. The session they belong to is named on the page — a defect in
that labelling was found and fixed the same day this note was written (`NIGHTLY_LOG.md`
2026-10-12).

## 3. The literature

### 3.1 Put skew — the best-supported of the four, with caveats the page does not carry

**Xing, Zhang & Zhao (2010), "What Does the Individual Option Volatility Smirk Tell Us About
Future Equity Returns?", *JFQA* 45(3), 641–662.** Stocks in the steepest-smirk quintile
underperformed the flattest by roughly **10.9% a year, risk-adjusted, over 1996–2005**. The
effect is strongest around earnings and the smirk predicts earnings surprises, which is the
mechanism they argue for: informed traders buy protection before bad news.

**Their skew is OTM put IV minus at-the-money *call* IV** (confirmed in Fu, Arisoy, Shackleton &
Umutlu's survey of option-implied measures, which restates the definition). The screener's is OTM
put IV minus the **average of the ATM call and ATM put** IV, at 0.90 moneyness rather than the
0.95 that is conventional. Because ATM put IV normally sits slightly above ATM call IV, averaging
raises the subtracted leg and **compresses** the measure relative to Xing–Zhang–Zhao's. It is a
variant of their signal, not their signal.

**Conditions the effect held under, which matter here:** a 1996–2005 US sample; weekly rebalancing;
quintile sorts, not a single stock's reading. Related work finds smaller gaps in later data, and
the general rule — McLean & Pontiff (2016), *JF* 71(1) — is that published cross-sectional
predictors return **26% less out of sample and 58% less after publication**, across 97 variables.
Xing–Zhang–Zhao has been published for sixteen years.

### 3.2 Implied volatility level — the evidence points the other way from intuition

**Bakshi & Kapadia (2003)** show delta-hedged option positions lose money on average, i.e. a
negative volatility risk premium: implied exceeds subsequently realized volatility. **Carr & Wu
(2009), "Variance Risk Premiums", *RFS* 22(3)** quantify it with synthetic variance swaps on five
indices and 35 individual stocks and find the crucial asymmetry: the premium is **strongly
negative for the S&P 500 and Dow**, but **mostly insignificant for individual stocks**, leading
them to conjecture the market prices variance risk in the market portfolio, not in each single
name. Bakshi & Kapadia attribute the smaller single-stock premium to the idiosyncratic component.

So a *high ATM IV on one stock* is not a reliable signal of anything priced. What single-stock
volatility *is* associated with is the **idiosyncratic-volatility puzzle** — Ang, Hodrick, Xing &
Zhang (2006) find the highest idiosyncratic-volatility quintile underperformed the lowest by
about **−1.06% a month**. That is a realised-volatility result and the screener already carries
realised volatility and beta inside its Risk category, so the options panel's IV adds little that
the score does not already have.

### 3.3 Changes in implied volatility — stronger than the level

**An, Ang, Bali & Cakici (2014), "The Joint Cross Section of Stocks and Options", *JF* 69(5).**
Stocks with large increases in **call** implied volatility over the previous month earn high
future returns, and large increases in **put** implied volatility earn low ones; decile sorts on
lagged changes in call IV produce spreads of roughly **1% a month**, persisting up to six months —
a longer horizon than most option-implied signals. **This is a change, not a level**, and the
screener stores only the level.

**Cremers & Weinbaum (2010), *JFQA* 45(2), 335–367** find stocks with relatively expensive calls
beat those with relatively expensive puts by about **50 bp a week** (1996–2005), using the implied
volatility spread between matched call and put pairs — and explicitly report that **predictability
declined over their sample**, consistent with the mispricing being arbitraged away.

### 3.4 Put / call open interest — the weakest link, and the construction is wrong for the evidence

This is where the literature and the page diverge most.

**Pan & Poteshman (2006), "The Information in Option Volume for Future Stock Prices", *RFS* 19(3),
871–908** is the canonical result, and it is routinely cited loosely. What they actually find:
put-call ratios built from **non-public, signed, open-buy volume** (a proprietary CBOE dataset
identifying volume initiated by buyers opening new positions) predict **>40 bp the next day and
>1% over the next week**.

Then they test the public version, and this is the part that matters for this screener. Using the
Lee–Ready algorithm on publicly observable trades and quotes, the public signal predicts "for only
the next one or two trade days", after which **prices reverse** — which they read as price
pressure, not information. In a bivariate regression containing both, the public signal has **no
predictability at all**. Their own words: *"the publicly observed option volume has very little,
if any, predictability for future stock prices."*

Open interest is a further step removed again — a stock rather than a flow, netting longs against
shorts, with no direction. There *is* supporting evidence for open interest, but it is about the
**change**, not the level: **Fodor, Krieger & Doran (2011),** *Financial Markets and Portfolio
Management*, find large increases in put open interest precede poor returns and that the recent
**change in the call-to-put open-interest ratio** predicts returns **over the following few
weeks** — with the call side "considerably less pronounced".

**So the one number the page shows for positioning — the level of the put/call open-interest ratio,
on a single expiry — is the construction with the least support in the literature**, and the
supported variant (the change, over weeks) is not what is stored.

### 3.5 Expected move

The straddle-over-price number is not a return predictor and is not presented as one. It is a
*restatement of the option price* — what the market charges for the move. Its evidential status is
different from the other three: it is correct by construction (modulo the quote quality bars the
code already applies) and is genuinely useful for timing a purchase around a report. Nothing in
the literature contradicts it because it does not claim to predict anything.

## 4. Documented practice

**Index providers do not use option-implied data to select stocks.** MSCI's factor indexes build
Value, Quality and Momentum from Barra GEMLT model factors — Book-to-Price, Earnings and Dividend
Yields, LT Reversal, Leverage, Earnings Variability, Beta — plus an Analyst Sentiment score for
Momentum. Fundamentals, price history and estimates; no option-implied input appears in the
published factor list. Option-implied data appears in their and S&P's product lines as *derivative
products* (VIX, the CBOE SKEW index, buy-write and put-write indices), which are index-level
volatility instruments, not stock-selection inputs.

**Sell-side derivatives desks do use exactly these numbers** — implied-move tables ahead of
earnings, skew and risk-reversal screens, cheap/expensive-vol rankings — and publish them to
clients continuously. But note what they are used *for*: structuring and timing an expression of a
view, and hedging. They are an execution and risk-management layer on top of a thesis, not the
thesis.

**The two are not in conflict, and the apparent disagreement dissolves once horizon is accounted
for.** Every option-implied cross-sectional signal in section 3 has a half-life measured in days
to a month: Pan & Poteshman one day to one week; Cremers & Weinbaum weekly; Fodor et al. a few
weeks; Xing–Zhang–Zhao weekly rebalanced; An et al. is the outlier at up to six months. A quarterly
or semi-annually rebalanced factor index cannot harvest a one-week signal after costs, so index
providers rationally exclude them. A derivatives desk trading daily can. Neither is wrong about
the data.

There is a second practical reason, which does not bind here but is worth recording: option-implied
factors require an options market. For the S&P 500 that is near-universal — this screener gets a
usable chain for ~455 of 501 — but for any broader universe coverage collapses, and a factor you
cannot compute for a third of your universe is not a factor you can build an index on.

## 5. What this means for this screener

The screener's stated horizon is long-term and its review cadence is monthly at the fastest
(`review_cadence`). Set that against the measured horizons above and the conclusion is not close:

> **Every option-implied signal with real support in the literature decays faster than this
> screener acts.** Keeping them out of the score is not a provisional position pending more
> evidence — it is what the evidence says.

This is a stronger statement than the context layer's standing rule, which says a signal may join
the score once it has a research note *and* a measured `1m` record meeting the observation gate.
For the options signals specifically, **the research note is the argument against joining**, and a
future `1m` record should be read with that prior: at a one-month horizon these signals are
largely spent.

That does not make the panel worthless. It makes it a *timing and sizing* surface — which is what
the owner asked for when he raised technicals ("timing of investments") — and the page should say
so in those terms rather than implying a prediction.

### Findings against what the page does today

1. **The page attributes Xing–Zhang–Zhao's ~11%/yr to a skew it does not compute.** Theirs is OTM
   put IV − ATM **call** IV; ours subtracts the **average of ATM call and put** IV, at 0.90
   moneyness. Direction is the same; the number is not theirs. Either match the construction or
   say the measure is a variant.
2. **The page omits the decay.** It says "with smaller gaps in later data", which is true but
   vague where a figure exists: McLean & Pontiff's 58% post-publication decline across 97
   predictors is the honest frame for a 2010 result quoted in 2026.
3. **Put / call open interest is shown as a bare number with no sentence attached.** The panel's
   explanatory paragraph covers the expected move and the skew and says nothing about it. A number
   on a teaching tool with no interpretation is the worst of both worlds: it implies significance
   it has not earned, and a student cannot learn anything from it. The evidence supports the
   *change* in the ratio over weeks, not the level — and Pan & Poteshman specifically find the
   public, unsigned version carries "very little, if any" predictability.
4. **ATM IV largely duplicates what the Risk category already scores.** Realised volatility and
   beta are in the score; the volatility risk premium that would make *implied* different from
   *realised* is, per Carr & Wu, mostly insignificant for individual stocks. This is an overlap
   worth naming in the "fits the whole screener" sense, not a defect.

### Recommendations (for a later session — no code today)

- **R1.** Correct the skew sentence: describe the measure actually computed, and cite
  Xing–Zhang–Zhao for the *direction and mechanism* rather than attaching their effect size to a
  different construction. Register the corrected sentence in `claims.py`.
- **R2.** Either drop the put/call open-interest row, or replace the level with the **change**
  over a few weeks and give it a sourced sentence (Fodor, Krieger & Doran 2011). Dropping it is
  the cheaper honest option and is the one this note leans to: the context log already records the
  level, so nothing is lost for later study.
- **R3.** Add the post-publication decay caveat wherever a pre-2015 cross-sectional effect size is
  quoted on the page — this applies beyond the options panel.
- **R4.** Record in `plan/context-layer.md` that the options signals are **not** candidates for
  joining the score, with this note as the reason, so a future session does not re-litigate it
  from the `1m` record alone.

R1–R3 are page-sentence changes, each needing a registered claim; R4 is a plan edit. None touches
a score.

## 6. Sources

- An, B.-J., Ang, A., Bali, T. G. & Cakici, N. (2014). "The Joint Cross Section of Stocks and
  Options." *Journal of Finance* 69(5), 2279–2337.
- Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006). "The Cross-Section of Volatility and Expected
  Returns." *Journal of Finance* 61(1).
- Bakshi, G. & Kapadia, N. (2003). "Volatility Risk Premiums Embedded in Individual Equity Options:
  Some New Insights." *Journal of Derivatives*; and "Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium." *Review of Financial Studies* 16(2).
- Carr, P. & Wu, L. (2009). "Variance Risk Premiums." *Review of Financial Studies* 22(3),
  1311–1341.
- Cremers, M. & Weinbaum, D. (2010). "Deviations from Put-Call Parity and Stock Return
  Predictability." *JFQA* 45(2), 335–367.
- Fodor, A., Krieger, K. & Doran, J. (2011). "Do Option Open-Interest Changes Foreshadow Future
  Equity Returns?" *Financial Markets and Portfolio Management*.
- Fu, X., Arisoy, Y. E., Shackleton, M. & Umutlu, M. (2016). "Option-Implied Volatility Measures
  and Stock Return Predictability." (Used here only to confirm Xing–Zhang–Zhao's definition.)
- McLean, R. D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?"
  *Journal of Finance* 71(1), 5–32.
- Pan, J. & Poteshman, A. M. (2006). "The Information in Option Volume for Future Stock Prices."
  *Review of Financial Studies* 19(3), 871–908. (Read in full:
  <https://www.mit.edu/~junpan/volume.pdf>.)
- Xing, Y., Zhang, X. & Zhao, R. (2010). "What Does the Individual Option Volatility Smirk Tell Us
  About Future Equity Returns?" *JFQA* 45(3), 641–662.
- MSCI factor index methodology (Barra GEMLT factor list; Analyst Sentiment in Momentum):
  MSCI Core Multiple-Factor Indexes Methodology and "Foundations of Factor Investing".

**Not used as evidence, per CLAUDE.md rules 4 and 5:** no backtest number and no reading from this
system's own IC history appears above. The `1m` horizon currently holds 4 effective observations
against a gate of 8.
