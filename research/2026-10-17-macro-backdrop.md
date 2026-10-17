# The market backdrop: what the five readings are worth, and whether any of them belongs in a score

**2026-10-17. `plan/context-layer.md` queue item 4 — one research note per context signal. This
is the note for the macro backdrop**, the last signal but one with none of its own (the track
record is the other). It covers all five readings the panel prints — yield curve, credit spreads,
volatility, inflation, labour market — and the three factor notes beside them.

**What it concludes, up front.** No macro reading is a candidate for the score, and that is a
finding rather than a holding position: §7 gives the reason from the literature and from
practice, and it is the same reason index providers keep macroeconomic series out of factor
construction while asset allocators use them daily. **Three of the page's own sentences are
wrong about their sources' own records and are corrected in the same commit as this note** (§8).
Two are fair as written and are left alone.

Every DOI below was resolved through Crossref before it was written down
(`research/measurements/2026-10-17-fetch-macro-sources.py --dois`), and every paper quoted was
read from the PDF rather than from memory. Where a figure is second-hand it says so.

---

## 1. What the panel shows today

`market_context.py` fetches ten FRED series, and `readings()` turns five of them into one
descriptive sentence each. The text below is what the live payload carried on 2026-10-16
(`dashboard_context.js`, `market.readings`); two of these five already have a correction in
flight from the 2026-10-16 sessions that reaches the page at the next full run.

| k | the sentence's empirical claim | source it cites |
|---|---|---|
| `curve` | an inverted 10y−3m curve "has preceded every US recession since the late 1960s, with lags of roughly 6 to 24 months and at least one false alarm" | Estrella & Mishkin 1998; the NY Fed's model |
| `credit` | "Spreads widen when lenders want more for default risk; wide spreads have coincided with stress for indebted companies" | **none** |
| `vol` | momentum's "sharpest losses in rebounds after high-volatility declines" | Daniel & Moskowitz 2016 |
| `inflation` | the Fed's 2% goal is "measured there on a different index, PCE" | implied (FOMC) |
| `jobs` | "at +0.50pp or more it has marked the start of every US recession since 1970" | Sahm 2019 |

And three `FACTOR_NOTES`: quality in downturns (Asness, Frazzini & Pedersen 2019), momentum and
rebounds (Daniel & Moskowitz 2016), rates and long-duration stocks (Dechow, Sloan & Soliman
2004; Weber 2018).

---

## 2. The labour-market reading: Sahm's rule says less than the page claims

### What the source actually says

Claudia Sahm, "Direct Stimulus Payments to Individuals", in *Recession Ready: Fiscal Policies to
Stabilize the American Economy*, The Hamilton Project / Washington Center for Equitable Growth,
May 2019, pp. 67–92. Read from the chapter PDF
(<https://www.hamiltonproject.org/assets/files/Sahm_web_20190506.pdf>; the book is at
<https://equitablegrowth.org/wp-content/uploads/2019/05/AutomaticStabilizers_FullBook_web_20190513.pdf>).
No DOI; a stable URL in its place, per the lab's rule for a paper without one.

The chapter is **a fiscal-policy proposal, not a recession forecast**. The 0.50pp rise is the
*trigger* for automatic stimulus payments, chosen because the unemployment rate is timely and
simple, and Sahm's own claims about it are narrower than the page's:

- p. 77: "Even a modest rise in the unemployment rate such as 0.50 percentage points … has
  occurred only **during or closely following** recessions. In other words, by this rule the
  stimulus payments would have been triggered only in recessions."
- p. 77: "Early in each recession since 1970, the unemployment rate rose at least a 0.50
  percentage points. **On average, payments would have been triggered within three months of the
  start of the past six recessions.** The automatic trigger would have been met **four months
  after the 2008–9 recession began and two months after the 2001 recession began.**"
- p. 78, under "some concerns": "the unemployment rate **tends to lag the business cycle**, such
  that unemployment usually peaks after the recession has ended."
- Endnote 7: "Earlier in the postwar period (not shown in figure 2) **the only false positive by
  this rule was in 1959**, and it was followed six months later by a recession."
- Endnote 8: "In general, the real-time data trigger **a few months later** than would the fully
  revised data." (The page reads the real-time series, which is right, and is therefore on the
  slower of the two.)

So the author of the rule says it triggers *after* a recession has begun, by about three months,
and explicitly warns that unemployment lags the cycle. The page says it has "marked **the
start**" of every recession. That is not what the source claims.

### What the rule's own record shows

`research/measurements/2026-10-17-sahm-rule-record.py`, from two free FRED series — the Sahm
real-time series FRED publishes (`SAHMREALTIME`, 801 months from December 1959) and the NBER
recession indicator (`USREC`). USREC flags the month *after* the NBER peak as the first
recession month, so both conventions are reported.

**(a) Sensitivity — the page's claim is right.** All **8 of 8** NBER recessions since 1970 have a
crossing of +0.50pp in the window from twelve months before the first recession month to the
recession's end. Nothing was missed.

**(b) "The start" — the page's claim is wrong.** The crossing came **after** the recession had
begun in **6 of 8**, on either convention:

| recession's first month | first crossing | vs first recession month | vs NBER peak |
|---|---|---|---|
| 1970-01 | 1969-10 | −3 | −2 |
| 1973-12 | 1974-03 | +3 | +4 |
| 1980-02 | 1980-04 | +2 | +3 |
| 1981-08 | 1980-08 | −12 | −11 |
| 1990-08 | 1990-11 | +3 | +4 |
| 2001-04 | 2001-06 | +2 | +3 |
| 2008-01 | 2008-04 | +3 | +4 |
| 2020-03 | 2020-04 | +1 | +2 |

Mean **+0.9 months** against the NBER peak. Of the two early readings, one (1981-08) is the 1980
recession's own crossing continuing through the double dip rather than a lead. The +3 for 2008
against Sahm's own "+4" is the convention, not a disagreement: she counts from the December 2007
peak. The record reproduces her "within three months of the start" almost exactly — and three
months *after* the start is not "the start".

**(c) False alarms — the page is silent, and there are two.**

| episode | peak | what followed |
|---|---|---|
| 1976-11 (one month, exactly +0.50) | +0.50pp | no recession for **39 months** (next: 1980-02) |
| 2024-07 … 2024-09 (+0.53, +0.57, +0.50) | +0.57pp | **no recession on record since**, through the latest NBER data (September 2026) |

The 2024 episode postdates the chapter, so it is not something Sahm got wrong — it is something
the record now contains and the page does not mention. It also sits **inside the ten-year window
the panel itself charts**, which is the strongest argument for saying it: a reader looking at the
panel's own spark line is looking at a false alarm while being told the rule has never produced
one.

The 1976 case is a boundary touch (exactly +0.50, "0.50 percentage points or more" triggers) and
Sahm's figure 2 covers 1970–2018 while her text says the rise "has occurred only during or
closely following recessions"; November 1976 is twenty months after the 1973–75 recession ended,
which is not "closely following". Endnote 7 names 1959 as the only pre-1970 false positive and
does not mention 1976. This note records the disagreement rather than resolving it: on FRED's own
real-time series, published by the rule's author, the 1976 touch is there.

**FRED's own note** (<https://fred.stlouisfed.org/series/SAHMREALTIME>) uses the words the page
copied — "signals the start of a recession when the three-month moving average … rises by 0.50
percentage points or more" — and carries a caveat only about data revisions, not about timing or
false positives. So the page inherited a wording defect from its source's source; that does not
make it true.

**Practice.** The rule is used as a real-time *coincident* labour-market gauge, which is what the
record supports: it is one of the inputs in the published recession-probability commentary of
market economists, and Sahm herself said publicly during the 2024 trigger that it did not mean a
recession was under way. Sahm's own Q&A (p. 88) adds a comparison worth carrying: asked about
alternatives, she writes that "financial market indicators tend to produce **more** false
positives (in part due to monetary policy responses)" than the unemployment trigger — which is
the right frame for the yield-curve reading two rows above it on the same panel.

---

## 3. The yield-curve reading: right about the record, wrong about the lag and the false alarm

### What the sources say

- **Estrella & Mishkin (1998), "Predicting U.S. Recessions: Financial Variables as Leading
  Indicators", *Review of Economics and Statistics* 80(1), 45–61, DOI
  [10.1162/003465398557320](https://doi.org/10.1162/003465398557320).** This is the paper the page
  cites. It is an out-of-sample probit horse race across financial variables at one to eight
  quarters; the term spread wins beyond one quarter. **It does not state a lag range in months and
  it does not count false alarms.** The NY Fed's scan of the underlying research paper
  (<https://www.newyorkfed.org/medialibrary/media/research/staff_reports/research_papers/9609.pdf>)
  has no text layer, so the article's own tables were not read here; what matters for the page is
  only that neither the published abstract nor any part of this literature attributes those two
  specific facts to it, and the facts themselves are measured below.
- **Estrella & Trubin (2006), "The Yield Curve as a Leading Indicator: Some Practical Issues",
  Federal Reserve Bank of New York *Current Issues in Economics and Finance* 12(5)**
  (<https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci12-5.pdf>; no DOI).
  **This is the paper that actually says what the page is trying to say**, and it is the NY Fed's
  own account of the indicator it publishes. Read from the PDF:
  - p. 4: the monthly-average spread between the ten-year constant-maturity rate and the
    three-month secondary-market rate **on a bond-equivalent basis** "has turned negative before
    each recession in the period from January 1968 to July 2006".
  - p. 5: "all six NBER recessions since 1968 have been preceded by **at least three negative
    monthly average observations in the twelve months before the start** of the recession.
    Moreover, when inversion on a monthly average basis is used as an indicator, **there have
    been no false signals over this period.**"
  - p. 5: daily and intraday inversions "often prove to be false signals"; monthly or quarterly
    averages are the reliable frequency. The level of the spread, not its change, carries the
    signal (p. 5).

### What the record shows

`research/measurements/2026-10-17-yield-curve-record.py`, from `GS10`, `TB3MS` and `USREC`,
monthly, 1953–2026. **The bill's basis turns out to matter and the source's basis is the one to
use:** a discount rate understates a bond-equivalent yield, so `GS10 − TB3MS` is the *larger*
spread, and on that basis the 1990 recession has **no** inversion in the 36 months before it —
7 of 8. Converted to the bond-equivalent basis Estrella & Trubin use, the record is:

- **(a) 8 of 8** recessions since 1968 preceded by an inversion, and **8 of 8** with at least
  three negative monthly averages in the prior twelve — Estrella & Trubin's sentence, reproduced,
  and extended from their six recessions to all eight. **The page's claim is true.**
- **(b) Measured leads, first month of the inversion to the first recession month: 6, 6, 9, 10,
  13, 15, 17 and 9 months** (median 13, minimum 6, maximum 17). The page says "roughly 6 to 24
  months". The **lower bound is exactly right and the upper bound is 7 months beyond anything on
  record.** (For 2020 the inversion a reader would have acted on is June–September 2019, nine
  months ahead; the one-month touch in February 2020 is not a signal anyone traded.)
- **(c) The one unambiguous false alarm is September 1966 – February 1967** — six months
  inverted, deepest −0.49pp, no recession within 24 months. It is **outside the period the page's
  own sentence names**: "since the late 1960s" starts after it, which is precisely why Estrella &
  Trubin begin at January 1968 and can report no false signals. From 1968 to today, **no
  completed episode has yet failed**, and the open case is **November 2022 – November 2024, the
  longest inversion in the series (25 months) and the deepest since 1981 (−1.71pp), with no
  recession declared in the 22 months since it ended** — its conventional 24-month window closes
  in November 2026, so the script reports it as "not yet decidable" rather than as a failure.

So the sentence's three parts are: true, overstated, and true only of a period it excludes. And
none of the three comes from the paper it credits.

**Practice.** The NY Fed publishes a recession probability from this spread monthly, which is the
documented practice the page gestures at; Estrella & Trubin is its methodological statement.
Note what that practice is: a *published descriptive probability*, not a position. The spread is
not an input to any factor index this note could find.

---

## 4. The credit reading: the mechanism clause is the part the evidence contradicts

The page says: "Lower-grade corporate bonds yield X pp over Treasuries, higher than N% of the
last ten years. **Spreads widen when lenders want more for default risk**; wide spreads have
coincided with stress for indebted companies." It cites nothing.

**Gilchrist & Zakrajšek (2012), "Credit Spreads and Business Cycle Fluctuations", *American
Economic Review* 102(4), 1692–1720, DOI
[10.1257/aer.102.4.1692](https://doi.org/10.1257/aer.102.4.1692)** (read from the NBER working
paper, w17021, May 2011, <https://www.nber.org/system/files/working_papers/w17021/w17021.pdf>)
decomposes a bond-level credit-spread index into expected defaults and a residual, the **excess
bond premium**. From the abstract:

> "we construct a credit spread index that is — compared with the standard default-risk
> indicators — a considerably more powerful predictor of economic activity … **the predictive
> content of credit spreads is due primarily to movements in the excess bond premium.**
> Innovations in the excess bond premium that are orthogonal to the current state of the economy
> are shown to lead to significant declines in economic activity and equity prices."

Two consequences for the page:

1. **The clause attributing a widening spread to default risk names the component that does
   *not* carry the information.** The informative part is the premium lenders charge *above*
   compensation for expected default — in the authors' reading, the risk-bearing capacity of the
   financial sector, i.e. the supply of credit.
2. **Baa-over-10-year — the series the page uses (`BAA10Y`) — is one of the "standard default-risk
   indicators" the paper's index outperforms.** That is not a reason to drop it (it is free, long
   and daily, and the page states a percentile rather than a forecast), but the page should not
   explain it with the mechanism the paper rules out.

The second half of the page's sentence — "wide spreads have coincided with stress for indebted
companies" — is fair, modest, and the direction every strand of this literature agrees on.

---

## 5. The volatility and inflation readings: both fair, left alone

**Volatility.** Daniel & Moskowitz (2016), "Momentum crashes", *Journal of Financial Economics*
122(2), 221–247, DOI
[10.1016/j.jfineco.2015.12.002](https://doi.org/10.1016/j.jfineco.2015.12.002). This is the
best-covered claim on the panel: it has a research note
(`research/2026-10-09-momentum-regime.md`), a published lab replication (experiment 2, 9 of 15
figures within tolerances written beforehand, with every figure about what follows a market
decline matching), and a scoring rule built on it with its three numbers fixed in advance
(`factor_engine.market_panic_state`). Nothing to add. The 2026-10-16 sessions already corrected
this sentence's description of the rule, which had said the screener "raises it when calm" when
the rule never raises momentum's weight; that correction reaches the page at the next full run.

**Inflation.** "against the Federal Reserve's 2% goal (measured there on a different index,
PCE)" is a statement of fact about the FOMC's *Statement on Longer-Run Goals and Monetary Policy
Strategy*, which specifies 2% inflation "as measured by the annual change in the price index for
personal consumption expenditures". The panel reads CPI and says so. Fair, and the distinction is
worth teaching. The "Latest reading: 2026-09" formatting defect was fixed on 2026-10-16
(`_month_name`) and reaches the page at the next run.

---

## 6. The three factor notes

**Quality in downturns** — Asness, Frazzini & Pedersen, "Quality minus junk", *Review of
Accounting Studies* 24(1), 34–112, DOI
[10.1007/s11142-018-9470-2](https://doi.org/10.1007/s11142-018-9470-2) (read from the working
paper). The page's sentence is supported:

- QMJ "delivers positive returns in **23 out of 24 countries** that we study" over June 1951 –
  December 2012, with an information ratio above 1 (p. 5, p. 25). "Across decades and countries"
  is fair.
- "QMJ returns are **high during market downturns** … if anything QMJ exhibits a mild positive
  convexity, that is, **it benefits from flight to quality during crises**" (p. 5). And p. 20:
  "in down markets, investors may exhibit flight to quality in the sense that prices of
  unprofitable stocks drop more than the prices of profitable stocks, **even adjusting for their
  betas**."

Two precisions worth recording without changing the sentence. First, the convexity AFP measure is
**mild and statistically weak** — the quadratic term is insignificant for QMJ overall and only
marginally significant (t 2.0) for the profitability leg; the down-market strength is mostly the
*linear* negative market beta. Second, QMJ is **long–short**, and p. 20's own mechanism is that
junk falls more — so a long-only holder of quality keeps only part of it. This screener has
already measured that for itself: lab experiment 8 found a long-only holder keeps **under half**
of the profitability factor among large companies. The page's words ("have tended to hold up
better when markets fall") are a relative statement and remain true; the quantitative caveat
lives in the lab, where a reader can rerun it.

**Momentum and rebounds** — same source as §5, same verdict.

**Rates and long-duration stocks** — Dechow, Sloan & Soliman (2004), "Implied Equity Duration: A
New Measure of Equity Risk", *Review of Accounting Studies* 9(2–3), 197–228, DOI
[10.1023/B:RAST.0000028186.44328.3f](https://doi.org/10.1023/B:RAST.0000028186.44328.3f); Weber
(2018), "Cash flow duration and the term structure of equity returns", *Journal of Financial
Economics* 128(3), 486–503, DOI
[10.1016/j.jfineco.2018.03.003](https://doi.org/10.1016/j.jfineco.2018.03.003). Checked on
2026-10-09 (`research/2026-10-09-context-panel-claims.md`, addendum) against both papers: 1.10% a
month, 1963–2013. Nothing reopened here; **the DOIs are this note's addition**, since the
2026-10-16 audit found the recent notes carried none.

---

## 7. Is any macro reading a candidate for the score? No — and here is the argument

The context layer's rule (CLAUDE.md row `ctx`) is that a signal joins the score only with a
research note *and* a measured `1m` record meeting the engine's observation gate. This note
answers the prior question — whether it is worth waiting for that record — and the answer for
macro is no, for reasons that are about the *kind* of signal rather than its strength.

**1. A macro reading is one observation per month for the whole universe, not 500 observations.**
Every stock in the run gets the same yield curve. A signal with no cross-sectional variation
cannot enter a cross-sectional ranking at all; it could only *change the weights*, which is
factor timing. That is a different claim needing different evidence, and `context_eval.py` — which
measures each signal's Spearman IC across stocks — **structurally cannot evaluate it**. Worth
stating plainly, because it would otherwise look like a signal whose record simply has not
accrued yet.

**2. The literature on timing factors from such signals is discouraging, from the people most
motivated to find it.** Asness, Chandra, Ilmanen & Israel (2017), "Contrarian Factor Timing Is
Deceptively Difficult", *Journal of Portfolio Management* 43(5), 72–87, DOI
[10.3905/jpm.2017.43.5.072](https://doi.org/10.3905/jpm.2017.43.5.072), conclude that contrarian
value timing of factors is "generally, a weak addition for long-term investors holding
well-diversified factors", that it "can reduce diversification and detract from the performance
of a multi-style strategy that already includes value", and that results stay "fairly weak" at
longer holding periods and at market extremes (read from AQR's own summary of the article, marked
second-hand; the published tables were not obtained). This is AQR arguing against a product it
could sell.

**3. Documented practice separates the two uses cleanly.** MSCI's factor indexes are built from
**fundamental** factors — financial-statement and price descriptors — and MSCI's own framing
distinguishes "fundamental, statistical and macroeconomic" factor models with its index work in
the first category (MSCI Barra Factor Indexes Methodology,
<https://www.msci.com/eqb/methodology/meth_docs/MSCI_Barra_Factor_Indexes_Methodology_Mar18.pdf>;
MSCI FaCS methodology,
<https://www.msci.com/downloads/web/msci-com/research-and-insights/blog-post/factors-and-esg-the-truth-behind-three-myths/factors-and-esg-the-truth-behind-three-myths-MSCI-FaCS-Methodology.pdf>).
Macro series are the asset-allocation layer's inputs, not the stock-selection layer's. The NY Fed
*publishes* a recession probability from the curve and takes no position on it. Both halves of
that are the practice this panel imitates.

**4. The one place this screener already reacts to the market is deliberately not built from
these series.** `adjust_momentum_weight` / `market_panic_state` reads the **universe's own**
two-year return and 126-day volatility, with three numbers fixed in advance and never re-fitted,
and it only ever cuts momentum's weight. That is the narrowest possible version of macro-aware
weighting: one direction, one category, pre-registered thresholds, computed from the stocks being
ranked rather than from an external series that could be revised. **If a macro reading were ever
to touch the score, this is the shape it would have to take** — and the panel's job stays what
the owner asked for, context for timing and sizing.

**Conclusion: macro stays beside the score, permanently rather than provisionally.** Not for want
of evidence that the curve and the Sahm rule describe the cycle — §2 and §3 show they do — but
because a universe-wide monthly reading is the wrong shape for a cross-sectional ranking, and
because the one published literature on using it to move weights says it is hard to do well.

---

## 8. Findings against what the page does today

Three sentences on the live page are wrong about their own sources' records. All three are
display-only context text in `market_context.readings`; **no number, score or rank changes.**
Fixed in the same commit as this note, with `claims.py` entries and tests, and reaching the live
page at the next full run (the data loop rebuilds `data/market_context.json` every run).

**(a) The labour-market sentence overstates the rule and omits its false alarms.** "At +0.50pp
or more it has marked **the start** of every US recession since 1970" — the crossing came *after*
the recession began in 6 of 8, mean +0.9 months against the NBER peak, and Sahm's own chapter
says "within three months of the start" and that unemployment "tends to lag the business cycle".
Two episodes crossed with no recession following, one of them in 2024, inside the window the
panel charts. **Corrected to**: it has reached +0.50pp around the start of every US recession
since 1970 — usually in the first months of one rather than before it — and twice, in 1976 and
2024, without a recession following.

**(b) The yield-curve sentence's lag range is unsupported and its false alarm is misattributed.**
"Lags of roughly 6 to 24 months and at least one false alarm (Estrella & Mishkin 1998)" — the
measured leads are 6 to 17 months; the one unambiguous false alarm (1966–67) is outside the
period the sentence names; and Estrella & Mishkin 1998 states neither fact. **Corrected to**: the
measured 6-to-17-month range, the NY Fed's own "no false signals on monthly averages since 1968"
(Estrella & Trubin 2006), and the open 2022–24 case named as the open case.

**(c) The credit sentence explains the spread with the component that carries no information,
and cites nothing.** **Corrected to**: a widening spread is partly expected defaults and partly
the premium charged above them, and Gilchrist & Zakrajšek (2012) find it is the second that has
predicted activity — with the citation added.

**Left alone deliberately:** the volatility and inflation sentences (§5, both fair, both with a
formatting/description correction already in flight from 2026-10-16), and all three factor notes
(§6, fair as written; the QMJ long-only caveat is measured in the lab rather than asserted here).

---

## 9. What this note does not do

- **It proposes no weight change and registers no research-engine experiment.** A macro reading
  cannot be an experiment in `v2_experiments.py`, which compares cross-sectional weightings; §7.1
  is the reason.
- **It cites no backtest number and no IC from this system's own history** (CLAUDE.md rules 4
  and 5).
- **It does not claim the curve or the Sahm rule is useless.** Both describe the cycle with a
  record this note measured. The claim is about *where they belong* — beside the ranking, with
  their limits stated.

## Sources

| source | DOI / stable URL | how read |
|---|---|---|
| Sahm (2019), *Recession Ready* ch. 3, Hamilton Project | <https://www.hamiltonproject.org/assets/files/Sahm_web_20190506.pdf> | PDF, pp. 67–92, endnotes 7–8 |
| Estrella & Mishkin (1998), *REStat* 80(1), 45–61 | [10.1162/003465398557320](https://doi.org/10.1162/003465398557320) | DOI verified; NY Fed scan has no text layer, so not read — see §3 |
| Estrella & Trubin (2006), FRBNY *Current Issues* 12(5) | <https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci12-5.pdf> | PDF, pp. 3–5 |
| Gilchrist & Zakrajšek (2012), *AER* 102(4), 1692–1720 | [10.1257/aer.102.4.1692](https://doi.org/10.1257/aer.102.4.1692) | NBER w17021 PDF, abstract and §1 |
| Asness, Frazzini & Pedersen (2019), *RAS* 24(1), 34–112 | [10.1007/s11142-018-9470-2](https://doi.org/10.1007/s11142-018-9470-2) | working-paper PDF, pp. 5, 20, 25 |
| Daniel & Moskowitz (2016), *JFE* 122(2), 221–247 | [10.1016/j.jfineco.2015.12.002](https://doi.org/10.1016/j.jfineco.2015.12.002) | covered by `research/2026-10-09-momentum-regime.md` + lab experiment 2 |
| Dechow, Sloan & Soliman (2004), *RAS* 9(2–3), 197–228 | [10.1023/B:RAST.0000028186.44328.3f](https://doi.org/10.1023/B:RAST.0000028186.44328.3f) | covered by `research/2026-10-09-context-panel-claims.md` |
| Weber (2018), *JFE* 128(3), 486–503 | [10.1016/j.jfineco.2018.03.003](https://doi.org/10.1016/j.jfineco.2018.03.003) | covered by `research/2026-10-09-context-panel-claims.md` |
| Asness, Chandra, Ilmanen & Israel (2017), *JPM* 43(5), 72–87 | [10.3905/jpm.2017.43.5.072](https://doi.org/10.3905/jpm.2017.43.5.072) | **second-hand** (AQR's own article summary) |
| MSCI Barra Factor Indexes Methodology (Mar 2018) | <https://www.msci.com/eqb/methodology/meth_docs/MSCI_Barra_Factor_Indexes_Methodology_Mar18.pdf> | documented practice |
| FRED series notes, `SAHMREALTIME` | <https://fred.stlouisfed.org/series/SAHMREALTIME> | series page |

Measurements, both reproducible from free data with no key:
`research/measurements/2026-10-17-sahm-rule-record.py`,
`research/measurements/2026-10-17-yield-curve-record.py`,
`research/measurements/2026-10-17-fetch-macro-sources.py`.
