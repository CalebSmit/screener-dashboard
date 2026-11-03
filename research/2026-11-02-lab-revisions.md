# Factor Lab: analyst revisions (Chan, Jegadeesh & Lakonishok 1996), on Open Source Asset Pricing

**Date:** 2026-11-02. **Kind:** a lab rerun (experiment 18, `lab.html#revisions`).
**Not a proposal.** Nothing here changes a weight or a metric; the lab never justifies a screener
parameter by itself (`plan/factor-lab.md`, rule 1, and "From the lab to the weights"). Section 8
lists questions for a Monday note, not answers.
**Second read: DONE 2026-11-02** by a reader who did not build it (`OWNER_FOCUS.md` 2026-10-20
item 2): the paper reopened from a fresh download, every reference figure re-read, every measured
figure recomputed without importing `factor_lab`, the pre-registration confirmed in `git log`. The
three compared figures and the 3-of-3 count stand; two sentences of this note were corrected
(sections 6 and 8) and one quotation made verbatim (section 3). See "Second reading" at the end.

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
  EW, no long-short quantile, and the note *"Table 5 has portfolio returns, but no-tstats. ... We
  find only monthly rebalancing works in port sorts."* (quoted verbatim since the second reading).

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
with and since 1993 sits just short of the usual bar (t 1.79 against 1.96); since 2007 it is 0.78% a
month with t 1.96. By the plan's reading rules that is "not confirmed", not "gone".

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
   gap is about half the equal-weighted one in the paper's years, since 1993 and over the whole
   record (0.48, 0.55 and 0.53 of it), t 1.79 since 1993 - though not since 2007, when the two are
   close (0.78% against 0.83% a month).
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

## Second reading (2026-11-02, a different reader)

`OWNER_FOCUS.md` 2026-10-20 item 2: a lab experiment is not checked until a reader who did not build
it has reopened the paper, re-read every reference figure, recomputed the measured ones without the
lab's code and confirmed in `git log` that the tolerances came first. All four were done, in a
separate worktree, by a session that did not write the experiment and did not open the builder's
scratch files. **The three compared figures, the 3-of-3 count and the null's 0 of 3 all stand. No
figure, tolerance or published sentence on the lab page changed.** Two sentences of this note did
(below), and nothing was rebuilt.

**1. The paper, re-read from a fresh download.** `nber.org/system/files/working_papers/w5375/w5375.pdf`,
572,050 bytes, SHA-256 `df7059e0...9f723dd1a944`, 40 pages. `pypdf` returns letter salad, as the
note says, so the pages were rendered with `pypdfium2` at 3x and read from the images.

| Figure | Record | Source, as read | Verdict |
|---|---|---|---|
| Most downward revisions, six months | 4.6 | Table 5 Panel A (PDF p. 35), "Return 6 months after portfolio formation", 1(Low) **0.046** | exact |
| Most upward revisions, six months | 12.3 | the same row, 10(High) **0.123** | exact |
| Spread, six months | 7.7 | derived, 0.123 - 0.046 = 0.077 | exact |
| The full six-month row (section 2) | 0.046 ... 0.123 | 0.046, 0.070, 0.072, 0.079, 0.083, 0.082, 0.087, 0.106, 0.116, 0.123 | exact |
| First year, the two ends; "9.7%" | 0.132, 0.229 | Table 5 Panel A; text PDF p. 13 (printed 11): REV6 "yields the largest spread in one-year returns (9.7%)" | exact |
| Table 9 intercept, the tolerance's basis | 1.431, t 5.91 | Table 9 (p. 40), "Difference" row, January 1977-December 1993; 1.431 / 5.91 = 0.242, x6 = 1.45, x2 = 2.9 | exact |
| Table 8, largest half, first year | 0.134 to 0.210 | Table 8 Panel A (p. 39), "Revision in analyst forecasts"; SUE 0.147 to 0.176, announcement return 0.140 to 0.183 | exact |
| Table 7 t-statistics | 5.45; 3.10; R6 4.07 | Table 7 Panel A (p. 38): REV6 alone (5.45), all four together (3.10); R6 alone (4.07) | exact |
| The formula and the skip | seven terms, mean, prior price; five days | p. 8 (printed 6): the sum over j = 0..6 of (f(t-j) - f(t-j-1)) / p(t-j-1), "the concensus (mean) IBES estimate"; "we skip the first five days after portfolio formation" | exact |

So the record's three reference figures and every other paper figure this note quotes are
transcribed correctly from the stated place, on the stated quantity. One wording point, left
unchanged because it is an interpretation rather than an error: the paper says "the first five
days", and the record and page say "five trading days"; the paper does not say which.

**2. The published article.** No free copy of *Journal of Finance* 51(5) was found either (the open
web, IDEAS/RePEc, AQR's bibliography, and the CFA Digest's abstract, which prints no figures). The
nearest independent evidence is a later review by one of the paper's own authors - Jegadeesh and
Titman, "Momentum", draft of 23 October 2001 (45 pages, SHA-256 `962cbde3...9928484d9`, read with
`pypdf`) - whose Table 11 cites "Chan, Jegadeesh and Lakonishok (1996)", the published article, for
the revision deciles: six months **12.3 up, 4.6 down, difference 7.7**; twelve months 22.9 and 13.2.
The six-month figures are the working paper's exactly, and so are the twelve-month levels (0.229,
0.132). The review prints the twelve-month difference as 8.7, in its table and its text; 22.9 - 13.2
is 9.7, so that is the review's own arithmetic slip, not a different figure. **So the published
Table V is still not read directly, but the three figures compared here are the ones the paper's
co-author attributes to the published version.** The record's and the page's "not obtained" remain
literally true and were left as they are.

**3. Every measured figure, recomputed without `factor_lab`.**
`research/measurements/2026-11-02-lab-revisions-second-read.py` reads REV6 straight from
`PredictorAltPorts_DecilesEW.zip` (chunked, pandas) and `PredictorAltPorts_DecilesVW.parquet`, keeps
the months in which all ten deciles have a return under both weightings (580, September 1976 to
December 2024 - the same set either way), and computes each decile's average month, the monthly
decile 10 minus decile 1 with t = mean / (sd(n-1) / sqrt(n)), and six times the average for the
comparison - the construction the record's `scale: 6` and its "Returns" condition describe.

| Figure | Published | Recomputed | |
|---|---|---|---|
| Most downward, six months, 1977-1992 EW | 2.75 | 2.7490 | agrees |
| Most upward | 11.40 | 11.4034 | agrees |
| Spread | 8.65 | 8.6544 | agrees |
| Spread a month, t | 1.44, 8.42 | 1.4424, 8.4188 | agrees |
| Within 2.9 | 3 of 3 | 3 of 3 | agrees |
| The ten six-month equivalents | 2.75 ... 11.40 | 2.75 5.91 6.83 7.10 7.96 8.39 9.02 10.21 10.86 11.40, rising every step | agrees (all ten means equal the payload's to 5e-5) |
| By market value, paper's years | 4.19, 0.70% a month, t 2.92 (page: 4.2, t 2.9) | 4.19, 0.6986, t 2.92 | agrees |
| 1993-2024 (384 months) | EW 0.90, t 4.54; VW 0.50, t 1.79 | 0.9011, t 4.54; 0.4959, t 1.79 | agrees |
| Since 2007 (216) | EW 0.83, t 2.76; VW 0.78, t 1.96 | 0.8292, t 2.76; 0.7796, t 1.96 | agrees |
| Whole record (580) | EW 1.08, t 7.51; VW 0.57, t 2.87 | 1.0765, t 7.51; 0.5743, t 2.87 | agrees |
| Volatility of the two ends, a month | 6.53 / 6.69; since 1993 9.55 / 8.23 | the same | agrees |
| "Weaker by about 37%" | 37% | 37.5% | agrees |
| Long-short column = decile 10 - decile 1 | all 580 months, both files | largest difference 9.4e-14 | agrees |
| Parquet PS deciles = experiment 17's Drive csv | all 636 months | largest difference 6.1e-17 | agrees |

The scorecard sentence (1.44, t 8.42; 0.90, t 4.54; 0.50, t 1.79), the "after the paper" row
(1.4424 in 192 months, 0.9011 in 384) and the verdict note (8.7, t 8.4; 2.7 against 4.6; 4.2, t 2.9)
are these numbers, correctly rounded. The takeaway's "about half that" (by market value since 1993)
is 0.55 of the equal-weighted figure - fair.

**4. The pre-registration.** `4e1f6a7` (2026-11-02 19:45:21 -06:00) carries the three paper figures,
the 2.9 tolerances, the window (197701-199212), `weighting: "ew"`, `scale: 6`,
`tolerances_written: "before"`, an empty `verdict_note` and the test that pins them; `b1cf2f8`
(19:53:47) is the run, and `4e1f6a7` is its ancestor. `git diff 4e1f6a7 HEAD` on the record changes
**no figure, tolerance, `where`, window, weighting or scale**, and the pin test is untouched. Besides
the result fields it changes two things: it adds `"compares": "ends"` (disclosed in a comment beside
it, in `b1cf2f8`'s message and in section 4), and it rewords the `source` line from "equally weighted
buy-and-hold returns over the six months" to "equally weighted returns of each group held for the six
months" (disclosed in `b1cf2f8`'s message as made for the advice-term check; it says the same thing,
and section 4 does not mention it). As in the earlier second readings, git shows the order of the
commits, not that no return was looked at before the first: `SORTS["revisions"]` already existed at
`4e1f6a7`, so "no REV6 return had been read" rests on the builder's statement. Eight minutes from
tolerances to result is consistent with it.

**5. The null.** A result of no effect puts every group at the paper's centre, (4.6 + 12.3) / 2 =
8.45, and the spread at 0: each end sits 3.85 from it against a tolerance of 2.9, and the spread
misses by 7.7. **0 of 3**, and the payload's `null_within` is 0.

**6. The construction claims in section 3.** OSAP's `REV6.py` at tag `v2.0.0` keeps `fpi == "1"`,
takes `meanest`, divides each monthly change by the lagged absolute price, sums the current change
and six lags and leaves REV6 missing if any is missing; `SignalDoc.csv`'s REV6 row has the sample,
`Key Table in OP = 7 R6`, `T-Stat = 4.07` and "for the next quarter" as section 7 says - so the
observation that OSAP's 4.07 is the past-return t-statistic stands, and OSAP's own "Key Table" label
(`7 R6`) points the same way. The portfolio code's all-stock breakpoints (`q_filt`) were not
re-read.

**Corrected in this note (not on the page):**

- Section 8, point 2 said the value-weighted gap is "about half the equal-weighted one in every
  period". It is in three of the four (0.48, 0.55, 0.53), but since 2007 the two are close - 0.78%
  against 0.83% a month, 0.94 of it - as section 6's own table shows. Reworded to say so.
- Section 6 said the value-weighted gap since 1993 "sits just inside the usual bar (t 1.79)", which
  reads as clearing it; t 1.79 is short of 1.96, as the sentence's own conclusion ("not confirmed")
  says. Reworded to "just short of the usual bar".
- Section 3's quotation of `SignalDoc.csv` made verbatim ("no-tstats").
