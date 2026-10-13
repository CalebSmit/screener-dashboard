# Value and momentum among large companies: decay, drought, or never there? (2026-10-12, owner-run)

**Question.** The second of the five questions in `plan/factor-lab.md`. Experiment 8 found that
among big US companies the value factor has averaged 0.19% a month since 1963 (t 1.56) and about
zero since 2014, and momentum 0.08% a month since 2000 (t 0.24). Valuation and Momentum are 35
points of the composite. Does the published literature read those numbers as decay, as a long
drought, or as something that was never there - and what follows for the two weights? Research-
led per CLAUDE.md rule 4: the only returns below are published ones and the Factor Lab's reruns
of Kenneth French's portfolios; none is a return of the screener's ranking.

**Finding.** They are two different stories.

- **Value among large companies was never reliably there on book-to-market** - the measure the
  lab and the textbooks use. That is the published view, not a new one. The same papers find the
  premium larger among large companies on earnings and cash-flow measures, which are the ones the
  screener weights, and almost entirely on the long side.
- **Momentum among large companies is strong in every published sample, and has been absent
  since 2000 on the public data** - a stretch longer and lower than any in the record, and one
  that the literature, which ends in 2011-2014, has not yet had to explain.

**No weight is proposed to change.** For momentum the lab points one way and the literature and
practice the other, and the reading rules written before this was researched say the weight then
stays. What changes is the methodology page, which quoted neither the large-company evidence nor
the last quarter-century (generator; `tests/test_large_company_evidence.py`).

---

## 1. What the published record says

Read from the papers (PDFs fetched 2026-10-12, a second reader's figures spot-checked against the
files). Returns are gross of costs throughout.

**Israel & Moskowitz, "The role of shorting, firm size, and time on market anomalies"
(*Journal of Financial Economics* 2013).** US stocks, July 1926 to December 2011, quintiles on
NYSE size breakpoints with value or momentum quintiles formed inside each, value-weighted. The
largest quintile averaged $36 billion in market value at the end of 2011 - larger than the
Russell 1000's average, so it is a fair stand-in for the top of the S&P 500. Table 3, percent a
year, t-statistics in brackets:

| Largest size quintile | Value (book-to-market) | Momentum (12-2) |
|---|---|---|
| High minus low, return | 3.70 (1.90) | 7.49 (2.95) |
| High minus low, CAPM alpha | 2.19 (1.14) | 10.24 (4.23) |
| Long side alone, CAPM alpha | 1.97 (1.21) | 3.92 (3.83) |
| Share of the alpha on the long side | 89.9% | 38.3% |
| *Smallest quintile, high minus low alpha* | *12.99 (4.52)* | *13.12 (5.59)* |

Their conclusions: the value premium falls steadily with size and is insignificant in the two
largest quintiles; momentum is present and significant in every size group with no reliable
relation to size; and against the market they cannot reject an even split between long and short
sides in any group. Two details matter here. On **other value measures** the largest quintile's
premium is small but larger than on book-to-market: earnings-to-price, cash-flow-to-price,
dividend yield, long-term reversal and a composite have t-statistics of 2.87, 2.82, 2.47, 2.10
and 2.77 (p. 15; earnings and cash-flow measures from 1951). And by **period**, the all-stock
momentum alpha is 10.29, 10.26, 9.47 and 8.87% a year in 1926-49, 1950-69, 1970-89 and 1990-2011
(Table 1) - no fade through 2011. *Not usable:* the paper's Table A3 panel for 2000-2011 repeats
the full-sample figures, apparently a typesetting error.

**Asness, Frazzini, Israel & Moskowitz, "Fact, fiction, and value investing" (*Journal of
Portfolio Management* 2015).** Exhibit 5, market-adjusted percent a year: large-company HML 1.7
(t 1.16) over 1926-2014; by period 0.1 (0.06) in 1926-62, **6.0 (2.52) in 1963-81**, 1.7 (0.93)
in 1982-2014. The one significant period is the sample the original studies used. Their comment
on a large-company premium on this measure: "Perhaps there never was." What lifts large-company
value in that paper is not a better measure but company: Exhibit 6 puts the Sharpe ratio of
large-company value at 0.25, momentum at 0.49 and a 60/40 mix at 0.65 (1927-2014).

**The same authors, "Fact, fiction and momentum investing" (*JPM* 2014).** Exhibit 5, annualised:
momentum among big stocks 6.8% over 1927-2013, 5.5% over 1963-2013, 4.5% over 1991-2013 (Sharpe
0.24, which over 23 years is a t-statistic near 1.15 - arithmetic done here, not in the paper).
They state there is no evidence of weakening in the two decades since discovery, citing Israel &
Moskowitz. The lab's 0.46% a month for 1963-2013 is their 5.5% a year.

**Bhattacharya, Li & Sonaer, "Has momentum lost its momentum?" (read as the SSRN working paper;
published in *Review of Quantitative Finance and Accounting* 2017).** Equal-weighted deciles,
six-month formation and holding, 1965-2012. Large companies (above the NYSE median): 0.88% a month
in 1965-98 (p 0.0001) and **0.90% in 1999-2012 (p 0.28)** - the same point estimate, no longer
distinguishable from zero because its volatility rose. All stocks: 0.69% in 1999-2012 (p 0.29),
and -0.19% in 2006-12. The authors read it as arbitraged away.

**Hou, Xue & Zhang, "Replicating anomalies" (read as NBER w23394; *Review of Financial Studies*
2020).** 1967-2014, NYSE breakpoints, value-weighted: 286 of 447 anomalies are insignificant,
but price momentum survives at reduced size - 0.60 to 0.82% a month against Jegadeesh & Titman's
0.9 to 1.1% - and so does book-to-market (0.59%, t 2.84), across all sizes.

**McLean & Pontiff, "Does academic research destroy stock return predictability?" (*Journal of
Finance* 2016).** 97 predictors, equal-weighted long-short quintiles: 0.58% a month in sample,
0.40% out of sample before publication, 0.26% after - declines of 26% and 58%. The paper rejects
both no decay and full disappearance, and finds that predictors concentrated in large, liquid
stocks earn less before and after. Its per-predictor results are in an appendix that was not
obtained.

## 2. What the lab adds: twelve to fifteen more years

`research/measurements/2026-10-12-large-company-value-momentum.py`, from the lab's published
data (French's value-weighted portfolios of the larger half of NYSE companies, July 1963 to
August 2026). Average month in percent, t-statistic in brackets; "held side" is the group the
factor holds minus the market.

| Among big companies | 1963-1999 | 2000-2026 | 2000s | 2010s | 2020-26 |
|---|---|---|---|---|---|
| Momentum | 0.58 (2.97) | 0.08 (0.24) | 0.10 | 0.16 | -0.08 |
| Momentum, held side | 0.33 (3.90) | 0.04 (0.31) | 0.06 | 0.04 | -0.01 |
| Value | 0.23 (1.67) | 0.14 (0.62) | 0.48 | -0.26 | 0.22 |
| Value, held side | 0.20 (2.13) | 0.19 (1.07) | 0.40 | -0.15 | 0.37 |

- **Momentum.** Of the 439 windows of 320 months in the record, the one ending in August 2026 is
  the lowest. Three decades in a row have been near zero; before 2000 the worst decade was the
  1980s at 0.19. Among small companies momentum was 0.07 in the 2000s and 0.38 and 0.53 since,
  neither distinguishable from zero.
- **Momentum inside and outside the panic state** the screener's regime rule watches for (as
  known before each month): since 2000, **-2.02% a month over the 35 months inside it and +0.33%
  (t 1.14) over the 285 outside**; over the whole record, -2.01 inside and +0.50 (t 3.04)
  outside. The crashes account for most of the shortfall since 2000, not all of it.
- **Value.** The latest 320-month window is unremarkable: 199 of the 439 are at or below it.
  Value among big companies has always alternated - the 1990s and the 2010s were both about
  -0.25 - and its held side is nearly the whole of it in every long period.

## 3. Reading it

**Value (22 points).** Not decay. On book-to-market there was never a reliable large-company
premium to decay, and both AQR and the lab say so. Three things support the category as the
screener builds it rather than as the lab measures it: the published large-company premia are
larger on earnings and cash-flow measures (the screener's 45/25/20/10 is free-cash-flow yield,
EV/EBITDA, earnings yield and EV/sales; book value carries no weight outside banks); about 90%
of what premium there is sits on the long side, which a ranking that shorts nothing keeps; and
value's contribution next to momentum is larger than alone. One thing cuts the other way: the
lab's earnings-yield experiment found that sort's gap closing after 2007 and reversing among
large companies, so "earnings measures do better" is a statement about 1951-2011.

**Momentum (13 points).** This is the uncomfortable one. Every published study finds momentum
among large companies, including the one written to test whether it had gone (the point
estimate held through 2012). The lab's data, which runs twelve years past the latest of them,
shows the value-weighted large-company version flat since 2000 - and flatter still on the held
side, which was only 38% of the premium to begin with. The reading rules protect a result from
being called gone on less than twenty years inside two standard errors; this is 26.7 years, so
they do not protect it. What still stands behind the weight:

1. the 1927-1999 record and the absence of fade through 2011 in Israel & Moskowitz;
2. the +0.33% a month since 2000 outside the panic state - two-thirds of the long-run figure,
   not distinguishable from zero, and the reason the regime rule exists;
3. how it combines with value (0.65 against 0.25 and 0.49);
4. practice: momentum is in MSCI's six rewarded factors and in S&P's quality-value-momentum set
   (`research/2026-10-12-growth-category.md`, section 3).

**The lab and the literature disagree, so the weight stays** - that is the rule, written before
this was looked at, and it is the right outcome here for a second reason: the lab's sort (top
against bottom 30% on 12-2 return, value-weighted, across the market) is not the screener's
measure (three momentum metrics, ranked within sectors). But the disagreement should not be
resolved by forgetting it. It is recorded here as **the composite's least well-supported large
weight on current evidence**, ahead of Growth only because Growth has no premium to lose.

## 4. What changed tonight

The methodology's Valuation and Momentum sections each gain a paragraph (generator) saying what
the evidence is among large companies, what a holder who shorts nothing keeps, and - for
momentum - what the public data shows since 2000 and what the weight therefore rests on. The
momentum figures on that page are checked against the lab's data by
`tests/test_large_company_evidence.py`, so a refresh that changes the picture fails a test.
**No weight, metric or threshold changed**; no `METHODOLOGY_CHANGELOG.md` entry.

## 5. Left open

- **Rerun Israel & Moskowitz in the lab** (largest size quintile, long and short sides, by
  period, extended to 2026). It is the published form of experiment 8 and would say whether the
  flat stretch is as stark in the largest fifth as in the larger half.
- **Does sector-relative momentum behave differently from the market-wide sort?** The published
  work on industry-neutral momentum has not been read; it bears directly on whether the lab's
  result transfers to the screener's metrics.
- **The regime rule's size.** Inside the panic state big-company momentum has averaged -2% a
  month; the rule cuts the weight to 70%, not to zero. Whether 0.70 is the right multiplier is a
  fair question, and not one to answer by fitting it to these numbers
  (`research/2026-10-09-momentum-regime.md` set it in advance).
- **If Growth's 3 points move** (the proposal in the Growth note), this note is a reason they
  should not go to Momentum.

**What would change this reading:** a published study with data past 2014 on momentum among large
US companies, either way; or the lab's own rerun of the largest quintile showing the stretch
since 2000 is specific to the larger-half, value-weighted construction.

## 6. Sources

Read in the relevant tables, 2026-10-12:

* Israel, R. & Moskowitz, T. (2013). The role of shorting, firm size, and time on market anomalies. *Journal of Financial Economics* 108(2), 275-301. Published version: Tables 1, 3, 4, A3, A4; Figs. 3, A1.
* Asness, C., Frazzini, A., Israel, R. & Moskowitz, T. (2014). Fact, fiction and momentum investing. *Journal of Portfolio Management* 40(5). Exhibits 4, 5.
* Asness, C., Frazzini, A., Israel, R. & Moskowitz, T. (2015). Fact, fiction, and value investing. *Journal of Portfolio Management* 42(1). Exhibits 3, 5, 6.
* Bhattacharya, D., Li, W.-H. & Sonaer, G. (2017). Has momentum lost its momentum? *Review of Quantitative Finance and Accounting* 48. Read as the SSRN working paper: Tables 1, 3, 7.
* Hou, K., Xue, C. & Zhang, L. (2020). Replicating anomalies. *Review of Financial Studies* 33(5). Read as NBER Working Paper 23394 (2017).
* McLean, R. D. & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance* 71(1), 5-32. Tables I, II, IV, V.
* Data: Kenneth R. French Data Library, through the Factor Lab's published series (experiment 8).

Volume and issue numbers of the *JPM*, *RQFA* and *RFS* versions are from memory; the copies read
were the publisher's PDF (the two AQR papers) and working papers (the other two). Not obtained:
McLean & Pontiff's internet appendix.
