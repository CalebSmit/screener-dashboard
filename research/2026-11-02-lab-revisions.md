# Factor Lab: analyst revisions (Chan, Jegadeesh & Lakonishok 1996), on Open Source Asset Pricing

**Date:** 2026-11-02. **Kind:** a lab rerun (experiment 18, `lab.html#revisions`).
**Not a proposal.** Nothing here changes a weight or a metric; the lab never justifies a screener
parameter by itself (`plan/factor-lab.md`, rule 1, and "From the lab to the weights"). Section 8
lists questions for a Monday note, not answers.
**Second read: pending** (`OWNER_FOCUS.md` 2026-10-20 item 2: a reader who did not build it must
reopen the paper, re-read every reference figure, recompute the measured ones without importing
`factor_lab`, and confirm the pre-registration in `git log`).

## 1. Why

Analyst Revisions (10% of the composite) was the one category of the eight with no experiment
behind it: the lab's category map said "Nothing, and probably never", because the studies need a
history of analysts' estimates as they stood at the time, which is sold. The 2026-10-20 OSAP
evaluation (`research/2026-10-20-lab-osap.md` section 2) found that Chen & Zimmermann publish the
*portfolio returns* of the Chan-Jegadeesh-Lakonishok revision signal (`REV6`), built by them from
I/B/E/S. A portfolio-level rerun needs only those returns.

It matters to the screener more than most reruns, because the category's heaviest measure is the
same construction: `fy1_revision_3m` (48 of the category's 100 points, about 4.8 composite points)
is the 90-day change in the consensus forecast of current-year EPS divided by price, and the code
cites CJL for the price scaling (`factor_engine.py`, the `fy1_revision_3m` block).

## 2. The paper, and where each figure came from

Louis K. C. Chan, Narasimhan Jegadeesh and Josef Lakonishok, "Momentum strategies", *Journal of
Finance* 51(5), 1681-1713 (December 1996), doi:10.1111/j.1540-6261.1996.tb05222.x (checked against
Crossref).

**The version read.** No free copy of the published article was found. The figures come from the
**NBER working paper No. 5375 (December 1995)**, `nber.org/system/files/working_papers/w5375/w5375.pdf`,
40 PDF pages. Its text layer is unusable OCR (`pypdf` returns letter salad), so the pages were
rendered to images with `pypdfium2` and read from the images at 3x; PDF page numbers below. A
journal revision can change a table: **that the published Table V carries the same numbers is not
verified**, and the record and the page say so.

- **Sample and method** (pp. 6-8): all domestic primary NYSE, AMEX and Nasdaq stocks with CRSP and
  Compustat coverage, supplemented by I/B/E/S. At the beginning of every month from January 1977 to
  January 1993, stocks are ranked and put in ten deciles; **breakpoints are NYSE stocks'**; stocks
  equally weighted. Returns are buy-and-hold over the periods after formation, **skipping the first
  five days**; a delisted stock's return is replaced by the value-weighted market's for the rest of
  the period.
- **The signal** (p. 8): `REV6_it = sum_{j=0..6} (f_{i,t-j} - f_{i,t-j-1}) / P_{i,t-j-1}`, where *f*
  is "the consensus (mean) IBES estimate" of the firm's earnings - the text calls it "a six-month
  moving average of past changes". **The paper is inconsistent with itself**: the formula sums seven
  monthly changes of the mean; every table caption says "moving average of the past six months'
  revisions in median IBES estimates of current fiscal-year earnings".
- **The table reproduced - Table 5, Panel A (p. 35)**, "Mean returns and characteristics for
  portfolios classified by revision in analyst forecasts", row "Return 6 months after portfolio
  formation": portfolio 1 (Low) **0.046**, portfolio 10 (High) **0.123**; the spread 0.077 is
  derived. The full row: 0.046, 0.070, 0.072, 0.079, 0.083, 0.082, 0.087, 0.106, 0.116, 0.123. First
  year: 0.132 and 0.229 - the text's (p. 13) "sorting stocks on REV6 yields the largest spread in
  one-year returns (9.7%)".
- **No standard error in Table 5.** The nearest the paper prints is **Table 9 (p. 40)**: the
  zero-cost portfolio long the top and short the bottom third on both REV6 and past return, held six
  months, monthly three-factor intercept **1.431% (t 5.91)**, January 1977-December 1993 - a standard
  error of 0.242% a month.
- **Large companies - Table 8 Panel A (p. 39)**: among firms above the NYSE median, first-year
  returns by REV6 decile run from **0.134** to **0.210** (gap 7.6%, against 9.7% for all firms) - and
  REV6 is the one earnings-news measure whose gap survives there (SUE 0.147 to 0.176; announcement
  return 0.140 to 0.183).
- **Regressions - Table 7 (p. 38)**: in monthly cross-sectional regressions of six-month returns on
  percentile ranks, REV6 alone has t **5.45**; with past return, SUE and announcement return in the
  same regression, **3.10**.

## 3. The public construction, read before any return

OSAP's code at the release's tag (`v2.0.0`), read before the tolerances were written:

- `Signals/pyCode/Predictors/REV6.py`: I/B/E/S unadjusted summary file, `fpi == "1"` (the current
  fiscal year), **`meanest`**; monthly change divided by the **prior month's** |price|; REV6 = the
  **sum of the current and six lagged** monthly changes, missing if any of the seven is missing. That
  is the paper's printed formula exactly (seven terms, the mean).
- `Portfolios/Code/01_PortfolioFunction.R` and `30_PredictorAltPorts.R`: the decile sets
  (`q_cut = 0.1`) use breakpoints from **all stocks** unless the signal's documentation names a
  quantile filter, and REV6's names none; the signal is lagged a month before it meets returns;
  `PredictorAltPorts_DecilesEW` forces equal weights, `..._DecilesVW` value weights; portfolios are
  re-formed **every month** and held one.
- `SignalDoc.csv` (REV6): sample 1977-1992, "Key Table in OP = 7 R6", "T-Stat = 4.07", stock weight
  EW, no long-short quantile, and the note *"Table 5 has portfolio returns, but no t-stats ... We
  find only monthly rebalancing works in port sorts."*

**The data.** Equal-weighted deciles: `PredictorAltPorts_DecilesEW.zip`, the October 2025 release's
Drive copy already on disk for experiment 17. Value-weighted deciles:
`Portfolios/Full Sets Alt/PredictorAltPorts_DecilesVW.parquet` from the **same release** on Hugging
Face (branch `2025.10` of `OpenSourceAP/CrossSection`). Since their October 2026 release the authors
publish there instead of Google Drive ("Google Drive could not handle the download volume", their
data page). The two copies are the same release: the parquet's PS deciles 01 and 10 equal experiment
17's Drive file `PS_OriginalCuts_VW.csv` in all 636 months, to the last digit. `factor_lab.fetch_osap`
now downloads both (`OSAP_HF_FILES`); `read_osap` reads parquet as well as csv and zip.

## 4. Tolerances, written and committed first

Commit **`4e1f6a7`** holds the paper's three figures, the tolerances and the record's account of
what was expected. At that point the equal-weighted file had last been read for signal PS and the
value-weighted file had been downloaded but not opened: **no REV6 return had been read.**

| Figure | Paper | Tolerance |
|---|---|---|
| Most downward revisions, six months (%) | 4.6 | 2.9 |
| Most upward revisions, six months (%) | 12.3 | 2.9 |
| Most upward minus most downward, six months (%) | 7.7 | 2.9 |

**Why 2.9.** One standard error from the paper's own Table 9 (0.242% a month x 6 = 1.45 over six
months) for sampling, and the same again for five construction differences known beforehand:
(1) **holding period** - the paper's six-month figure averages a signal one to six months old, OSAP's
re-formed portfolios always hold the freshest one; (2) the paper's **five-day skip**; (3) **NYSE
against all-stock breakpoints**; (4) **mean against median** (the captions) and seven terms; (5)
**delisting** treatment and thirty years of data revisions. The ends take the spread's tolerance, as
in experiment 17. The comparison sets six times the average month of the re-formed deciles, January
1977 to December 1992 (OSAP's documented sample), against the paper's six-month means.

**Expected beforehand** (written in the record): the direction reproduces; (1) and (2) both push the
public spread **above** 7.7, so it may land outside on the high side; (3) cuts the other way or not
at all; the levels differ by the compounding a six-month hold adds.

**Plan rule 8**, checked from the record before committing: a result of no effect matches **0 of 3**
(each end sits 3.85 from the null's common level of 8.45; the spread misses).

**Why only the ends.** The paper prints all ten deciles and the page shows all ten, but a null
result lands near the paper's middle deciles under any tolerance (they run 7.0-11.6 around a centre
of 8.7), which is what rule 8 forbids. After the run the record gained `"compares": "ends"` so the
layout test accepts a ten-group sort compared at its ends; no figure or tolerance moved.

## 5. What reproduced

**3 of 3 within** (`lab.html#revisions`), 192 months.

| Figure | Paper | Here | Diff. | |
|---|---|---|---|---|
| Most downward revisions, six months (%) | 4.60 | 2.75 | -1.85 | within |
| Most upward revisions, six months (%) | 12.30 | 11.40 | -0.90 | within |
| Most upward minus most downward (%) | 7.70 | 8.65 | +0.95 | within |

The spread is **1.44% a month, t 8.42**, and lands above the paper's as predicted. Returns rise from
each decile to the next across all ten (six-month equivalents 2.75, 5.91, 6.83, 7.10, 7.96, 8.39,
9.02, 10.21, 10.86, 11.40), the shape of the paper's row. Both ends sit a little below the paper's
levels, the low end more - consistent with the six-month compounding the comparison leaves out,
and with the paper's delisting rule (a delisted stock earns the market for the rest of the hold,
which flatters the low end), though neither is measured here.

**Internal consistency.** In both files the published long-short column equals decile 10 minus
decile 1 in all 580 months (September 1976 to December 2024).

**Volatility does not separate in the paper's years** (6.53% against 6.69% a month for the two ends);
since 1993 the downward-revision end is the more volatile (9.55% against 8.23%).

## 6. Since the paper

Most upward minus most downward, a month (t-statistic), from the published series:

| Period | Equally weighted | By market value |
|---|---|---|
| The paper's years, 1977-1992 (192 months) | **+1.44% (8.42)** | +0.70% (2.92) |
| After the paper, 1993-2024 (384) | **+0.90% (4.54)** | +0.50% (1.79) |
| Since 2007 (216) | +0.83% (2.76) | +0.78% (1.96) |
| Whole record, 1976-2024 (580) | +1.08% (7.51) | +0.57% (2.87) |

Unlike most results the lab has rerun, the equal-weighted gap is still clearly distinguishable from
zero after its paper (t 4.54 over 32 years), though weaker by about 37%. By market value it was half
the size to begin
with and since 1993 sits just inside the usual bar (t 1.79); since 2007 it is 0.78% a month with t
1.96. By the plan's reading rules that is "not confirmed", not "gone".

## 7. What this does and does not verify

It verifies that a careful, documented, public construction of the CJL revision signal - built from
I/B/E/S by authors with no stake in the result - earns over the paper's own years what the paper's
working-paper table reports, within one-to-two of the paper's own standard errors, with the same
monotone shape; and that the effect is still there after publication across all covered stocks.

It does **not** verify: the published article's Table V (not obtained); the paper's own firm-level
numbers (the lab sees no forecasts); the paper's six-month holding construction (OSAP re-forms
monthly, which by its own note is the only construction that "works" - a fact about the signal's
decay worth keeping in mind); the effect **among large companies alone** (no size cut is published
for this signal; market-value weighting is the nearest view); the paper's regression t-statistics.

**Two observations on the sources, neither a defect in the experiment:**

- **OSAP's `T-Stat = 4.07` for REV6 is R6's.** In the working paper's Table 7 Panel A, 4.07 is the
  t-statistic on **past return (R6)** in the first regression; REV6's own are 5.45 alone and 3.10
  with the other three. The published Table VII may differ; recorded so a reader comparing the page
  with `SignalDoc.csv` does not take the experiment's figures for a mis-citation (the same kind of
  slip the 2026-10-23 second reading found in OSAP's Piotroski row).
- **`SignalDoc.csv`'s definition says "for the next quarter"**; the code reads `fpi == "1"`, the
  current fiscal year, which is what the paper uses. The code is right; the description is loose.

## 8. What it means for the Revisions category

**No weight change, and none proposed.** The lab never sets a parameter (rule 1); this is one
construction across all covered stocks, and by the reading rules evidence among large companies
outranks evidence across all stocks.

What it adds:

1. **The category's heaviest leg now has a rerun behind it.** `fy1_revision_3m` is CJL's
   construction (FY1 consensus change over price), and the construction earned what the paper says,
   in the paper's years and after them.
2. **Among large companies the evidence is weaker, as the paper itself found.** The paper's largest
   half kept about four-fifths of the first-year gap (7.6% against 9.7%); the public value-weighted
   gap is about half the equal-weighted one in every period, t 1.79 since 1993.
3. **The other 52 points are still unrerun.** Surprises (25), the price target (13.5) and short
   interest (13.5) have no experiment; `EarningsSurprise` in OSAP is the route for the first.

**Questions for a Monday note** (not answered here; each would follow "From the lab to the weights"):

- **The window.** The screener measures revisions over 90 days; CJL over six to seven months; OSAP's
  note says the signal works only with monthly re-formation, i.e. its information is front-loaded.
  What do the literature and practitioners (Barra's sentiment descriptors, I/B/E/S-based quant
  screens) use, and is there evidence that a shorter window is better or worse among large companies?
- **Among large companies.** Is there published evidence on revision portfolios restricted to large
  caps since the 1990s (the public files have no size cut)? The question 8 row in `plan/factor-lab.md`
  already asks what the category's 10% is on; this is the leg that row says is researched least by
  the lab and most by the literature.
- **Revisions and momentum.** CJL's Table 7 keeps REV6 significant with past return in the same
  regression (t 3.10), which is the premise of the category's "independence budget" against
  Momentum (`config.yaml`, `tests/test_revisions_momentum_budget.py`). Does later work - Novy-Marx
  (2015) argues momentum is largely earnings momentum - change how much of the 48 points is a second
  Momentum bet?

## 9. Reproducing it

`python factor_lab.py --fetch --build` (the fetch needs Google Drive for the Drive files - expect its
quota page and retry - and Hugging Face for the parquet), then `lab.html#revisions`. The record is
`REVISIONS_OSAP` in `factor_lab.py`; its pin is
`tests/test_factor_lab.py::test_the_revisions_tolerances_are_the_ones_committed_before_the_comparison`
and the published figures are checked by `test_the_revisions_sort_is_published_as_measured`.
