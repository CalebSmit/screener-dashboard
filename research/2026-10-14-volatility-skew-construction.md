# The put skew: the paper's construction, the paper's horizons, and one correction

**2026-10-14** (code session, 13:00 slot). Closes the three open page-sentence findings in
`plan/context-layer.md` item 4, and **corrects one conclusion** in
`research/2026-10-12-option-implied-signals.md`.

Everything below is read from the papers themselves. The 10-12 note had Xing, Zhang & Zhao's
definition second-hand from a survey; the 06:00 session today obtained the paper and read the
definition (pp. 644–645); this session read the rest of it, which is where the correction comes
from. Files are in the gitignored `data/papers/`; the download scripts are
`research/measurements/2026-10-14-fetch-xzz.py` and
`research/measurements/2026-10-14-fetch-decay-and-volume-papers.py`.

---

## 1. What the page said, and what it was showing

The options panel printed, for ~455 of 501 stocks:

> Steeper put skew has been followed by weaker returns — stocks with the steepest skew trailed
> those with the flattest by about 11% a year, risk-adjusted, in 1996–2005 (Xing, Zhang & Zhao
> 2010), with smaller gaps in later data.

beside a row labelled **"Put skew (90% puts vs at the money)"**, computed as

```
old = IV(put nearest 0.90 × spot, within ±0.04)  −  mean(IV(ATM call), IV(ATM put))
```

Neither leg is the paper's.

## 2. The paper's construction, from the paper

**Xing, Y., Zhang, X. & Zhao, R. (2010).** "What Does the Individual Option Volatility Smirk
Tell Us About Future Equity Returns?" *Journal of Financial and Quantitative Analysis* 45(3),
641–662. Definition at pp. 644–645, equation (1):

> SKEW<sub>i,t</sub> = VOL<sup>OTMP</sup><sub>i,t</sub> − VOL<sup>ATMC</sup><sub>i,t</sub>

with, verbatim from the text:

| Element | The paper's rule |
|---|---|
| OTM put | strike/price **below 0.95 but above 0.80**; among those, the one **nearest 0.95** |
| ATM call | strike/price **between 0.95 and 1.05**; among those, the one **nearest 1** |
| Liquidity | **positive open interest**; options of **10 to 60 days** to expiration |
| Frequency | daily SKEW averaged over a week, **Tuesday close to Tuesday close** |

They note (footnote 3) that alternative *moneyness measures* — Bollen & Whaley's delta, Ni's
volatility-adjusted ratio — give "quantitatively similar results", and (footnote 4, §III.B)
that a volume-weighted SKEW gives 10.06% a year against the moneyness-based 10.90%. **Neither
robustness check covers substituting the mean of the ATM call and put for the ATM call**, which
is what this screener was doing.

### Why the old measure was neither published quantity

Write *s* for the paper's skew and *v* for the ATM put IV minus the ATM call IV — the
**put-call implied-volatility spread**, which is Cremers & Weinbaum's (2010) signal and a
different claim about a different thing. Then

```
mean(IV_atmc, IV_atmp) = IV_atmc + v/2
old = IV(put @0.90) − IV_atmc − v/2
```

so the old number was (a skew measured at 0.90 rather than 0.95) **minus half the put-call IV
spread**. It mixed two signals the literature keeps apart and reported the sum under one
paper's name.

## 3. How much that mattered, measured

`research/measurements/2026-10-14-skew-construction-paired.py` — 25 names, every 20th in the
committed universe, both constructions computed from the **same fetched chain** so nothing is
attributable to quotes moving:

| | |
|---|---|
| available, old → new | **22 → 23** of 25 (the new construction is available for slightly more) |
| new − old, IV points | mean **−4.23**, median **−2.36**, range −40.39 to +3.47 |
| rank correlation | Spearman **0.675**, Pearson 0.616 |
| **sign disagreements** | **7 of 22 (32%)** — one construction called the skew steep where the other called it flat |

A third of the sample disagreed on the *direction*. That is the size of the defect: the page
was telling a reader whether a stock's puts were bid up relative to at-the-money, with a
paper's effect size attached, and for seven names in twenty-two the paper's own measure said
the opposite sign.

### A plausibility check the old measure fails

`research/measurements/2026-10-14-skew-distribution-vs-paper.py`, against Table 1 (p. 647),
in IV points:

| | mean | 5% | 25% | 50% | 75% | 95% | max |
|---|---|---|---|---|---|---|---|
| **paper**, 1996–2005, ~840 firms/week | +6.40 | −0.99 | +2.40 | +4.76 | +8.43 | +19.92 | — |
| old | +3.65 | −3.51 | −2.77 | +0.18 | +4.82 | +14.42 | **+46.56** |
| new | −0.94 | −8.23 | −2.83 | −0.20 | +1.49 | +4.02 | +6.18 |

The old construction produced **+46.6 IV points** for one stock in twenty-two — beyond the
paper's 95th percentile of +19.92, for a measure whose whole range over a decade of weekly
cross-sections was roughly −1 to +20. The new one produces nothing outside that range.

**What does *not* reproduce, and is reported rather than tuned away:** our level sits about 7
points below theirs (median −0.20 against +4.76). Two candidate explanations were checked and
one was ruled out.

- *Ruled out — strike granularity.* A real strike grid does not put the legs exactly 0.05 of
  moneyness apart as the paper's nominal 0.95→1.00 does. Measured: the realised gap averages
  **0.064** (median 0.060, range 0.030–0.098), **wider** than 0.050 in 18 of 23 names. A wider
  gap would make our skew *larger*, not smaller, so this is not the cause.
- *Most likely — the sample is all large companies.* Skew is a small-company phenomenon in
  their own data: in Table 3 Panel A the steepest-skew quintile averages **$4.29bn** of market
  value against $13.8–14.7bn for the intermediate quintiles, and the flattest $7.82bn. This
  screener ranks the S&P 500 only. It is the same pattern the Factor Lab keeps finding — an
  effect measured across all stocks is smaller among the largest — and the standing reading
  rule applies: large-company evidence outranks all-stock evidence.

That is a limitation of what the number can support, not a reason to change the construction.

## 4. The correction: XZZ is **not** a sub-monthly signal

`research/2026-10-12-option-implied-signals.md` §4 lists "Xing–Zhang–Zhao weekly rebalanced"
among signals with "a half-life measured in days to a month", and concludes in §5:

> **Every option-implied signal with real support in the literature decays faster than this
> screener acts.** [...] a future `1m` record should be read with that prior: at a one-month
> horizon these signals are largely spent.

**For XZZ that is wrong, on the paper's own Table 4.** The note conflated the paper's
*rebalancing frequency* (weekly — that is how the portfolios are formed) with its *decay
horizon*, which the paper measures separately and at length in §III.C. From the abstract:
"This predictability persists for at least 6 months." From §I, contrasting itself with Pan &
Poteshman explicitly: "we find no quick reversals of the stock price movements following
publicly observable volatility smirks. In fact, the predictability from volatility smirks
persists for at least 6 months."

Table 4 Panel B — quintile holding-period returns, annualised, Fama-French 3-factor adjusted,
Newey-West t-statistics:

| holding period | 1 week | 4 weeks | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|---|
| low − high skew, % a year | 10.90 | **6.52** | 6.88 | 7.14 | 5.99 | 6.50 | 7.04 | 6.38 |
| t | 2.93 | **2.70** | 3.73 | 4.23 | 4.32 | 4.34 | 4.33 | 4.31 |

The spread falls by ~40% from one week to four and then **stops falling**, holding 6–7% a year
out to 28 weeks with t-statistics that *rise*. The paper's own summary: "The predictability of
volatility skew lasts over the 28-week horizon and then slowly dies out."

Panel A (Fama-MacBeth on the *n*-th week ahead) is weaker and non-monotonic — SKEW coefficient
t −4.09 at week 1, −3.27 at 4, −1.89 at 8, −1.30 at 12, −2.47 at 16, −2.03 at 20, −1.82 at 24 —
which is what a cumulative holding-period result looks like decomposed into single weeks. Panel
C puts SKEW's own first-order weekly autocorrelation at 0.660, falling to ~0.20 by lag 8, so the
*signal* is not persistent even though its *return effect* is.

### What the correction changes, and what it does not

**It does not make the skew a candidate for the score today.** The 10-12 note's verdict was
load-bearing on three legs and only one of them breaks:

1. ~~horizon~~ — **withdrawn for XZZ.** At four weeks the paper measures 6.52% a year, t 2.70;
   the screener's review cadence is monthly at its fastest. The horizon argument stands for Pan
   & Poteshman (1–2 days), Cremers & Weinbaum (weekly) and Fodor et al. (a few weeks); it does
   not stand for XZZ.
2. **Post-publication shrinkage stands.** See §5 — 58% across 97 predictors, and the paper has
   been published for sixteen years.
3. **The context layer's own rule stands** (CLAUDE.md row "ctx"): a signal joins the score only
   with a research note **and** a measured `1m` record meeting the improvement engine's
   observation gate, then a changelog entry. `context_eval.py`'s first one-month window closes
   **2026-11-07** and the gate is ~8 months of daily logs away.

**What it changes is the prior for reading that record.** The 10-12 note told the next session
to expect nothing at one month. The paper says otherwise for this one signal, so:

> When `context_eval.py` reports a `1m` record for `_ctx_opt_skew`, read it as a test of a
> finding the paper **does** claim at that horizon — not as a measurement of something already
> known to be spent. A flat `1m` IC is then evidence *against* XZZ surviving in large
> companies in the 2020s, which is worth writing up either way. Note the series before
> 2026-10-14 is the old construction and the series after is the paper's; the first five log
> dates (2026-10-08 to 10-14) are therefore **not comparable** with later ones and should be
> excluded, not spliced.

**Proposal status: NONE.** No weight, score or threshold changes here, and none is proposed. If
a future session wants to propose one it starts from §3's limitation — the level sits well below
the paper's in a large-company universe — not from this correction.

## 5. The decay figure, from the primary source

**McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
Predictability?" *Journal of Finance* 71(1), 5–32.

> "We study the out-of-sample and post-publication return-predictability of **97 variables**
> that academic studies show to predict cross-sectional stock returns. Portfolio returns are
> **26% lower out-of-sample** and **58% lower post-publication**. [...] We estimate a 32%
> (58% − 26%) lower return from publication-informed trading."

They "can reject the hypothesis that return-predictability disappears entirely, and [...] that
post-publication return-predictability does not change", and the decline is robust to a general
time trend, other authors' time indicators, and time fixed effects (§I).

**A version trap worth recording.** Five copies were downloaded. The 2012-01, 2012-10 and
2013-05 working papers all say **82 characteristics** and about **35%** post-publication, not 97
and 58%. Only the "Journal of Finance, Forthcoming" manuscript carries the published figures.
Citing the number off a working paper found by search would have put a wrong figure on a public
page with a correct-looking citation. The page quotes 97 and 58%.

One refinement, stated with its own hedge: predictors constructible "using only financial data,
such as volume, prices, returns, and shares outstanding" — their **Market** category, which is
where an option-implied measure sits — had the **largest** post-publication decline, "although
it is not significantly different from the declines of the other predictors" (§IV, Table IV). So
it is a reason not to expect option-implied signals to do *better* than average, and not a
quantified extra penalty. The page does not claim one.

## 6. Put/call open interest: off the page

The panel printed a bare `put OI / call OI` ratio with **no sentence at all** in its explanatory
paragraph. Two primary sources bear on it:

**Pan, J. & Poteshman, A. M. (2006).** "The Information in Option Volume for Future Stock
Prices." *Review of Financial Studies* 19(3), 871–908. Their headline result uses volume
*initiated by buyers to open new positions*, which is **not public data**: low-ratio stocks beat
high-ratio stocks by >40bp the next day and >1% over the next week. The relevant finding here is
the decomposition. Using the Lee–Ready algorithm on publicly observable CBOE trade and quote
records they build the public analogue, and find (§5, and the paper's summary):

> "the publicly observed option volume has very little, if any, predictability for future stock
> prices"

and in a bivariate regression with the non-public signal, the public one "no longer has any
significantly" predictive content.

**XZZ's own control** is the closest public analogue measured over multiple horizons. Their PCR
is the previous week's put volume over call volume — publicly computable, like ours. Table 4
Panel A, PCR t-statistics predicting the *n*-th week ahead: **3.62\*\*\*** at week 1, then
**0.02, 1.81, 0.13, −0.70, 1.37, −0.56** at weeks 4, 8, 12, 16, 20, 24. Significant for one
week; nothing from four weeks out.

Both are about **flow**. Ours was a **level**, and of open interest rather than volume — a step
further from anything measured. On a tool whose cadence is monthly, a number with no sentence,
no published support at the reader's horizon, and no support for its functional form is the
worst of the available options: it implies significance it has not earned and a student can
learn nothing from it.

**Decision: removed from the page and from the payload** (`generate_dashboard.CTX_FIELDS`), and
**still recorded** in `data/context_log/` and evaluated by `context_eval.py` as
`options_put_call_oi`. If its own record ever says something, it can come back with a sentence.
`tests/test_option_implied_claims.py::test_the_context_log_still_records_it_so_the_evidence_keeps_accruing`.

## 7. What shipped

| | |
|---|---|
| `context_signals._xzz_skew`, `_xzz_leg` | the paper's construction, with the `XZZ_*` constants named as the paper's |
| published legs | `_ctx_opt_skew_put_iv`, `_ctx_opt_skew_strike`, `_ctx_opt_skew_call_iv`, `_ctx_opt_skew_call_strike`, so the row shows `$90.00 put 38% − $100.00 call 30%` against its own `+8.0 pts` (CLAUDE.md row 0.8) |
| `_ctx_opt_skew_in_window` | whether this expiry is inside 10–60 days. Measured on the 2026-10-14 cache: **413 of the 481 stocks with a skew (85.9%)** are inside; outside it the panel says the paper's figures are context, not a reading of this number |
| the sentence | names the construction, quotes **10.9% at one week and 6.5% at four**, names the 10–60 day window, and gives the 58% shrinkage as an upper bound |
| put/call OI | off the page and out of the payload; still in the context log |
| claims register | `context.put_skew`, 11 `checked_by` entries |
| tests | `tests/test_option_implied_claims.py` (11), 11 new in `tests/test_context_layer.py` |

**No score moved.** Every field here is `_ctx_*`, which CLAUDE.md row "ctx" keeps out of `raw`,
`pct`, every category score and the composite.

### A test that could not see the defect

The only option-chain fixture in the suite (`_chain`, used by three tests) gave the ATM call and
the ATM put **the same implied volatility**, so `mean(call, put) == call` and the two
constructions were numerically identical on it. The one assertion on the skew —
`out["_ctx_opt_skew"] == pytest.approx(0.38 - 0.30)` — is true of *both* measures, so the suite
passed unchanged after the construction was replaced, which is how the mismatch survived from
2026-10-08 to today. This is the 2026-10-12 lesson again in a new place: *a claim whose check
cannot distinguish the claim from its alternative is an unchecked claim.* The added fixture
(`_asym_chain`) breaks the symmetry on purpose, and one test asserts the old arithmetic is
**not** what comes out.

## 8. Sources

- Xing, Y., Zhang, X. & Zhao, R. (2010). "What Does the Individual Option Volatility Smirk Tell
  Us About Future Equity Returns?" *JFQA* 45(3), 641–662. Read from the paper: definition
  pp. 644–645; Table 1 p. 647; Table 3 pp. 650–651; Table 4 pp. 653–654.
- McLean, R. D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance* 71(1), 5–32. Read from the "Journal of Finance,
  Forthcoming" manuscript — abstract, §I, §IV and Table IV. **Not** the 2012/2013 working
  papers, whose figures differ (82 predictors, ~35%).
- Pan, J. & Poteshman, A. M. (2006). "The Information in Option Volume for Future Stock Prices."
  *RFS* 19(3), 871–908. Read from the author's copy — abstract, §1, §4, §5.
- Cremers, M. & Weinbaum, D. (2010). "Deviations from Put-Call Parity and Stock Return
  Predictability." *JFQA* 45(2). **Cited second-hand**, for the identity of the put-call IV
  spread in §2 only; not obtained this session, and no figure of theirs is quoted on the page.
- `research/2026-10-12-option-implied-signals.md` — the note this corrects in one respect and
  leaves standing in the rest.
