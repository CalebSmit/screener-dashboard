# Synthesis: are the eight categories eight bets?

**Date:** 2026-09-02
**Author:** nightly code session (synthesis day)
**Follows:** `research/2026-08-31-size-factor-in-a-large-cap-universe.md`
**Status:** complete. One methodology change shipped — see
`METHODOLOGY_CHANGELOG.md` 2026-09-02.

---

## The question Wednesday is supposed to ask

Monday's note ended with three candidate changes and an instruction: decide
whether they fit the screener *as a whole*, rather than treating the size
category in isolation. The right way to do that is to stop looking at size and
look at the whole 8×8 category structure — what overlaps with what, what is
redundant, and what the composite is implicitly betting on.

Doing that surfaced something larger than any of the three candidates.

---

## Part 1 — The category structure, measured

Spearman correlations between the eight published category scores, from the
2026-09-02 02:00 data run (N = 499 stocks with all eight scores):

|  | val | qual | grow | mom | risk | rev | size | inv |
|---|---|---|---|---|---|---|---|---|
| **valuation** | 1.000 | 0.100 | −0.349 | −0.156 | 0.029 | 0.005 | 0.337 | 0.228 |
| **quality** | 0.100 | 1.000 | 0.171 | −0.019 | 0.036 | 0.022 | −0.090 | −0.023 |
| **growth** | −0.349 | 0.171 | 1.000 | 0.068 | −0.140 | 0.043 | −0.266 | −0.331 |
| **momentum** | −0.156 | −0.019 | 0.068 | 1.000 | **0.516** | 0.189 | −0.220 | −0.072 |
| **risk** | 0.029 | 0.036 | −0.140 | **0.516** | 1.000 | −0.051 | −0.249 | 0.035 |
| **revisions** | 0.005 | 0.022 | 0.043 | 0.189 | −0.051 | 1.000 | −0.130 | −0.041 |
| **size** | 0.337 | −0.090 | −0.266 | −0.220 | −0.249 | −0.130 | 1.000 | 0.280 |
| **investment** | 0.228 | −0.023 | −0.331 | −0.072 | 0.035 | 0.280 | −0.041 | 1.000 |

One number dominates. **momentum ~ risk = +0.516** is the largest of the 28
pairs, by a wide margin — half again the next largest (valuation ~ growth at
−0.349).

Spanning regressions say the same thing. Regressing each category's rank on
the other seven:

| Category | R² | Unspanned | Weight |
|---|---|---|---|
| **risk** | **0.378** | 62.2% | 10.00% |
| **momentum** | **0.352** | 64.8% | 14.95% |
| size | 0.271 | 72.9% | 5.00% |
| growth | 0.265 | 73.5% | 13.00% |
| valuation | 0.242 | 75.8% | 20.05% |
| investment | 0.158 | 84.2% | 5.00% |
| revisions | 0.092 | 90.8% | 10.00% |
| quality | 0.073 | 92.7% | 22.00% |

Risk and momentum were the two **least** independent categories in the
screener. Together they carry ~25% of composite weight.

---

## Part 2 — Why, and it is not a coincidence

`CLAUDE.md` already records that momentum and risk are "100% derived from one
`Ticker.history()` call" — a shared *data source*, noted after the MNST split
incident as a blast-radius concern. That is true but it is not the explanation
here, because a shared data source does not by itself make two scores
correlate. Volatility and 12-month return both come from the same price series
and correlate at −0.013.

The actual cause is that two of the five scored risk metrics were not risk
metrics.

`factor_engine.py` builds them from the *same numerator*:

```
sharpe_ratio   (:1923)  = (return_12m − rf) / volatility
sortino_ratio  (:1946)  = (return_12m − rf) / downside_deviation
```

Across the S&P 500 the cross-sectional spread in trailing returns is far wider
than the spread in dispersion, so the shared numerator dominates both ratios
and the denominators barely matter. Measured on the published metric
percentiles:

| Pair | 2026-08-31 | 2026-09-01 | 2026-09-02 |
|---|---|---|---|
| `sharpe_ratio` ~ `sortino_ratio` | +0.993 | +0.994 | +0.993 |
| `sharpe_ratio` ~ `return_12_1` | +0.940 | +0.940 | +0.944 |
| `sortino_ratio` ~ `return_12_1` | +0.936 | +0.933 | +0.940 |
| **`sharpe_ratio` ~ `volatility`** | **+0.029** | **+0.032** | **+0.025** |

Three consecutive runs, essentially identical. The risk category was scoring
**five metrics that were three distinct things**: two of the five were each
other, and both of those were the momentum signal rather than a risk measure.
30% of the category — 3% of composite — was momentum wearing a risk label.

`SCREENER_OVERVIEW.md` justified the design with *"Five metrics give a more
complete risk picture than two."* A +0.993 correlation between two of the five
refutes that sentence directly. This is the sharpest kind of finding available
to a synthesis day: a documented design rationale that the tool's own output
contradicts.

### The user-facing consequence

This is what makes it a defect rather than a modelling preference. On the
2026-09-02 run:

| Ticker | Published risk | Risk on dispersion alone | Momentum |
|---|---|---|---|
| SNDK | 31.1 | **1.6** | 94.2 |
| MRNA | 34.7 | **6.8** | 82.8 |
| VRT | 29.1 | **3.2** | 75.3 |
| FIX | 37.5 | **12.0** | 98.3 |
| MU | 41.6 | **17.2** | 94.8 |
| WDC | 39.9 | **15.7** | 94.0 |

Every one a high-momentum name. The public site was telling a student that a
violently volatile stock was mid-pack on risk **because it had gone up**. For
a tool whose stated purpose is to be teachable, a category that does not mean
what its name says is the whole problem.

---

## Part 3 — What the literature and practice actually use to measure risk

Both measure risk with **dispersion**. Neither ranks a cross-section on a
return-divided-by-risk ratio.

**Literature.** Ang, Hodrick, Xing and Zhang (2006), *The Cross-Section of
Volatility and Expected Returns*, JF 61(1), 259–299: sorts on **idiosyncratic
volatility**; quintile 1-minus-5 spread over **1%/month**, robust at
**−0.63%/month, t = −3.30** after excluding the smallest growth firms.
Frazzini and Pedersen (2014), *Betting Against Beta*, JFE 111(1), 1–25: sorts
on **beta**; the BAB factor realises a Sharpe of **0.78** over 1926–March 2012.

Note carefully what Frazzini and Pedersen do with the Sharpe ratio. They use
it to **evaluate the resulting portfolio**, not to rank the cross-section.
That is the correct use of the statistic — it is a portfolio-evaluation
measure — and it is precisely the use this screener was *not* making of it.

**Practice.** The Barra US Equity Model (USE4) builds its Residual Volatility
style factor from dispersion descriptors — daily standard deviation,
cumulative range, residual sigma — with Beta as its own descriptor. MSCI's
Minimum Volatility indexes optimise against those Barra BETA and RESVOL
exposures, leaving them unconstrained while constraining every other style
factor to ±0.25 sd. No index provider selects for low risk with a Sharpe
ratio.

*(Source caveat, same convention as the 08-31 note: the specific USE4
descriptor weights — 0.74 DASTD + 0.16 CMRA + 0.10 HSIGMA — come from a
**secondary** source. The primary MSCI PDF was not text-extractable this
session. The substantive claim, that all the descriptors are dispersion
measures, is not in doubt; treat the decimals as indicative.)*

**Academia and practice do not disagree here.** That is worth saying, because
the 08-31 note found a real divergence on size. On how to measure risk
cross-sectionally, the literature and the index providers are doing the same
thing, and this screener was doing something else.

---

## Part 4 — The change, and what it does to the whole

Shipped today (`METHODOLOGY_CHANGELOG.md` 2026-09-02): `sharpe_ratio` and
`sortino_ratio` go to **weight 0** within risk; the surviving 30/20/20 is
renormalised over 70 to 42.86 / 28.57 / 28.57, preserving their relative
emphasis exactly. Both ratios stay computed and stay visible on each stock's
detail page — weight-0 candidates, the same treatment `proximity_52w_high` and
`peg_ratio` already get. They are informative; they are not risk.

Spanning regressions after the change:

| Category | R² before | R² after |
|---|---|---|
| **risk** | 0.378 | **0.188** |
| **momentum** | 0.352 | **0.115** |
| size | 0.271 | 0.269 |
| growth | 0.265 | 0.263 |
| valuation | 0.242 | 0.246 |
| investment | 0.158 | 0.161 |
| revisions | 0.092 | 0.104 |
| quality | 0.073 | 0.076 |

The two least-independent categories become two of the **most** independent —
momentum moves from third-most-spanned to second-least — and nothing else
moves more than 0.012. The change is surgical: it removes an overlap without
disturbing the rest of the structure.

Effect on the ranking: composite Spearman **0.990**, median absolute rank
change **10 places**, p90 35, max 83, and **3 of the top 50 change** (out:
DELL, FOX, STLD; in: ADBE, CB, EXE). Real, but not violent — which is what a
3%-of-composite correction should look like.

**An honest negative.** I also computed an "effective number of independent
bets" (entropy of the eigenspectrum of the weight-scaled correlation matrix):
**5.18 → 5.29 of 8**. That barely moves, because the measure is dominated by
how concentrated the *weights* are rather than by how correlated the
categories are — quality and valuation alone are 42%. It is the wrong
instrument for this question and I am reporting it rather than quietly
dropping it, since a reader might otherwise expect a bigger number. The
spanning R² table is the informative one.

**What deliberately did not change: the eight category weights.** The
measurement says the risk category was mismeasuring risk, not that risk
deserves more or less of the composite. There is a real second-order effect —
roughly 3% of composite that was labelled risk was behaving as momentum, so
momentum's realised weight falls back toward its stated 13–15% and risk's
rises toward its stated 10% — but that is the categories *becoming what the
documentation always said they were*, not a new bet. Re-deciding the category
weights needs its own research, and doing both at once would make neither
attributable.

---

## Part 5 — Disposition of Monday's three candidates

**Candidate 3 (winsorise-before-rank) — already done, and verified live
today.** Shipped by the 2026-09-01 morning session. Confirmed this morning
against the published payload: the six megacaps now carry six distinct market
caps (NVDA $5,250B, AAPL $4,745B, GOOGL $4,097B, MSFT $3,720B, AMZN $2,750B,
META $1,474B) where they previously shared one clipped value. Closed.

**Candidate 2 (are `size` and `investment` one bet or two?) — answered: two.**
They correlate **+0.280**, the fifth-largest pair, and they are the *first*
and *sixth* least-spanned categories: size is 73% unspanned by the other
seven, investment 84%. Two single-metric 5% categories that each retain
three-quarters or more of their own variance are not a disguised 10% bet, and
merging them would lose information. Asness et al.'s result that CMA absorbs
*part* of SMB's alpha is consistent with +0.280 — partial overlap, not
redundancy. **No change; the question is closed** rather than left open.

**Candidate 1 (is the size tilt more aggressive than intended?) — still open,
deliberately deferred.** It is a genuine question and Monday's Finding B is
right that the `log` in `size_log_mcap` is annihilated by the subsequent rank
transform. But it is a question about *one* 5% category's transfer function,
and it competes today against a 25%-of-weight overlap that was making a
category mean something other than its name. Depth over breadth: one thing,
done properly. Monday's own refutation criterion still stands — if the
compressed version changes the top 50 by fewer than ~2 names it is complexity
for nothing, and that is a cheap thing to measure first.

---

## What this leaves open, in priority order

1. **Candidate 1 above** — the size tilt's aggressiveness, measured against
   its own refutation criterion before any code is written.
2. **`growth` ~ `investment` = −0.331** is a genuine internal tension nobody
   has written down. `investment` scores *low* asset growth well (the CMA
   proxy); `growth` scores fast revenue and EPS growth well. Companies growing
   fast generally grow assets to do it, so the screener is systematically
   rewarding growth with one hand and penalising how it is funded with the
   other. That may be correct — it is close to what the five-factor model
   does — but it is currently an accident of construction rather than a
   documented choice, and it is 18% of composite weight. Worth a research day.
3. **Confirm this change on the live site.** The 2026-09-03 02:00 data run is
   the first to publish it. The check is one line: the momentum/risk category
   correlation in the published payload should read near **+0.15**, not
   **+0.52**.
4. **Re-examine `max_drawdown_1y`.** It survived today as a dispersion
   measure, correctly, but it correlates **+0.539** with `return_12_1` and
   **+0.636** with `volatility`. It is the weakest of the three survivors on
   the "is this really independent risk information" test. Not a defect;
   a thing to look at once, with sources.

---

## Sources

- Ang, A., Hodrick, R.J., Xing, Y., and Zhang, X. (2006). "The Cross-Section
  of Volatility and Expected Returns." *Journal of Finance* 61(1), 259–299.
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2006.00836.x
- Frazzini, A. and Pedersen, L.H. (2014). "Betting Against Beta." *Journal of
  Financial Economics* 111(1), 1–25.
  https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta
- Asness, C., Frazzini, A., Israel, R., Moskowitz, T., and Pedersen, L.H.
  (2018). "Size Matters, If You Control Your Junk." *JFE* 129(3), 479–509.
  (Cited here only for the CMA/SMB overlap point carried from the 08-31 note.)
- MSCI. *The Barra US Equity Model (USE4) Methodology Notes*, August 2011.
  https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf
  — read directly; the descriptor *weights* quoted in Part 3 are from a
  secondary summary, flagged there.
- MSCI. *MSCI Minimum Volatility Indexes Methodology*, September 2017.
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_Minimum_Volatility_Methodology_Sep2017.pdf

Internal measurements: `dashboard_data.js` as published on `main` from the
2026-09-02 02:00 data run, and `runs/*/03_percentiles.parquet` for the
2026-08-31, 09-01 and 09-02 runs. `factor_engine.py:1923` (`sharpe_ratio`),
`:1946` (`sortino_ratio`), `:2433` (`CAT_METRICS["risk"]`); `config.yaml`
`metric_weights.risk`.

**No backtest number appears in this note** (`CLAUDE.md` rule 5) and **no
figure from `live_ic_history.csv` or `performance_history.csv` is used as
evidence** (rule 4). The `1m` optimization horizon holds 3 effective
observations against a gate of 8.
