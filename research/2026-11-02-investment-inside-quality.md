# Should Investment be folded into Quality? A weight-preserving fold changes almost nothing the evidence speaks to

2026-11-02, owner-run night.

**Proposal status: REFUSED 2026-11-04** (the 06:00 session, which did not write this note) - the
fold is refused and the recommendation accepted: Investment stays standalone and the practitioner
convention now appears on the methodology page, with the large-company caveat
`research/2026-10-19-revenue-growth-vs-asset-growth.md` §11.2 queued. Section 4 was rebuilt
independently and run over eight payloads rather than one; every figure held, and one of the
listed costs does not exist. The review is **section 6**.

No return, IC, backtest or track-record figure from this system is used as evidence anywhere in
this note (CLAUDE.md rules 4 and 5). The factor figures are published research and the lab's
public-library replications. Section 4 measures structure and data coverage on the published
payload, not returns. No weight, metric or file other than this note changed.

## The question

`plan/factor-lab.md`, "The questions, in the order to research them", row 3: **should the
screener's standalone Investment category (5%, one metric, asset growth scored lower-is-better)
be folded into Quality, as most practitioners do, or stay standalone, as in the academic
construction?**

`research/2026-10-19-revenue-growth-vs-asset-growth.md` settled, and this note does not redo:
nine of eleven provider families house asset growth inside Quality, Research Affiliates as
"Profitability minus Investment" (§9.4-9.5, §11.3 item 2); asset growth is half of a position on
*growth in asset turnover* (§3); its large-company effect is half-size and lives in the extreme
deciles (§6.2-6.3). What is left: what investment adds **beside** profitability and value among
**large** companies, exactly how practitioners structure the fold, and what a fold would do here.

## Finding

1. **The academic models keep investment separate because it is the one that survives, not the
   one that is redundant.** In Fama & French (2015) it is *value* that the other four factors
   absorb (HML intercept -0.04%, t -0.47); CMA keeps 0.28% a month (t 5.03) after the other four
   (Table 6, p. 11). In Hou, Xue & Zhang (2015) the investment factor has a Fama-French alpha of
   0.33% (t 4.85) and correlates 0.06 with the profitability factor (pp. 660, 662).
2. **But among large companies, and next to a timely value measure and momentum - this
   screener's exact conditions - investment adds little.** FF2015's big-stock CMA is 1.03-2.00
   standard errors from zero and "almost all the average value of CMA is from small stocks"
   (p. 8). Asness (2014) finds that with timely value and momentum in the model "it's actually
   CMA that's mostly redundant". The lab's big-company investment factor is -0.06% a month
   (t -0.35) since 2003.
3. **Practice folds it, but in two different ways, and the cleaner precedent keeps the factor
   visible.** MSCI's FaCS keeps Investment Quality as its own risk-model factor and *groups* it
   as 25% of the Quality group (FaCS Methodology, January 2018, p. 8); Research Affiliates
   blends it into one Quality score. Scientific Beta runs it as its own index and argues
   profitability and investment "cannot be combined in a composite manner to represent a single
   risk factor" (2017, p. 35). AQR's published QMJ penalises asset growth only implicitly,
   through residual-income growth (RAS 2019, p. 43, footnote 12).
4. **Here a weight-preserving fold is exactly neutral for 455 of 499 stocks and the whole top 25,
   and non-neutral in five places**: renormalisation inside Quality (44 stocks, up to 5.19
   composite points), the momentum panic rule (one top-25 swap, hypothetically), the investor
   profiles (zero to two top-25 swaps each), the trap flags that read the Quality score (14 flip,
   none in the top 25), and the health gate (Quality's dispersion falls 15.3% against a 20% bar).
5. **Recommendation: keep Investment standalone, and say on the page that practitioners usually
   house it inside Quality and why this screener does not.** The live questions about these 5
   points are weight and shape, and a fold changes neither; it would only bury the leg that makes
   the asset-turnover position legible to a student.

---

## 1. What investment adds beside profitability and value, among large companies

### 1.1 In the full cross-section, investment is the survivor

**Fama & French (2015).** The famous result runs the opposite way to the overlap concern in the
factor-lab row. Regressing each factor on the other four (Table 6, p. 11, July 1963 - December
2013, 2x3 factors):

| Left-hand factor | Intercept, % a month | t | Main slopes |
|---|---|---|---|
| HML (value) | **-0.04** | **-0.47** | CMA 1.04 (t 23.03), RMW 0.23 (t 5.36) |
| CMA (investment) | **0.28** | **5.03** | HML 0.45 (t 23.03), RMW -0.21 (t -7.84) |

*"In the five-factor model, HML is redundant for describing average returns"* (p. 12); HML and
CMA correlate "about 0.70" (p. 8). The overlap is real, but it is value that is spanned. And
conditionally, investment and profitability move *against* each other (-0.21) - the statistical
reason a blend of the two diversifies.

**Hou, Xue & Zhang (2015)**, *RFS* 28(3). The investment factor earns 0.45% a month (t 4.95);
its Fama-French three-factor alpha is 0.33% (t 4.85) and its Carhart alpha 0.28% (t 3.85); it
correlates 0.69 with HML (p. 660). The investment and ROE factors correlate 0.06 (p. 662): the
q-model is built so that profitability and investment are two separate dimensions. Hsu, Kalesnik
& Kose (2019, p. 57) summarise the literature the same way: *"most models considered in academia
include profitability and investment as separate factors."*

### 1.2 Among large companies, next to timely value and momentum, it adds little

Three routes, each closer to this screener's conditions than the full-market results above:

- **Size.** FF2015 Table 4: big-stock CMA 0.22 (t 2.00), 0.11 (t 1.50), 0.07 (t 1.03) across three
  constructions; *"almost all the average value of CMA is from small stocks"* (p. 8).
- **Timely value and momentum.** Asness (2014, AQR) reruns FF2015 Table 6 with value on the
  current price (HML-DEV) and momentum added: CMA's t-statistic *"comes close but doesn't break
  2.0"* and *"it's actually CMA that's mostly redundant."* The screener prices valuation on today's
  market value and carries 13% Momentum - Asness's setting, not Fama and French's. (His tables are
  images; only the text was verified.)
- **After publication.** The lab's experiment 14 (`research/2026-10-19-lab-asset-growth.md` §5):
  the value-weighted asset-growth spread was 0.67% a month (t 4.05) in Cooper, Gulen & Schill's
  sample and **-0.07% (t -0.29)** since; the big-company investment factor 0.37% (t 2.63) then,
  **-0.06% (t -0.35)** since. Hsu, Kalesnik & Kose (2019) Table 6 (p. 56): low asset growth
  0.30% (t 3.97) before publication, 0.07% (t 0.48) after.

Against that: Cooper, Gulen & Schill's (2008) large-firm value-weighted alpha spread, -0.81% a
month (t -2.91), and Scientific Beta's long-leg Carhart alpha of +1.66% on the largest 500 US
stocks, 1975-2015 (both as recorded in the 10-19 note).

### 1.3 Does QMJ include growth or investment?

Not as a variable. The 2013 SSRN draft (p. 10) averages profitability, growth, safety and payout
(net equity issuance, net debt issuance, net payout over profits). The published paper (*RAS*
24(1), 2019) builds the headline score from **three** - profitability, growth, safety (p. 43,
eq. 6) - with payout as an explicit variant (p. 75). Its growth is growth in residual income, and
footnote 12 (p. 43) states the consequence: *"Growth in residual income increases in the growth in
net income and decreases in asset growth, all else equal."* So AQR penalises asset growth inside
Quality implicitly, and keeps the issuance terms - the financing side of the same expansion - in a
separate payout component, which Asness (2014) calls *"strongly related"* to the Fama-French
investment factor.

### 1.4 Novy-Marx

Novy-Marx's "Quality Investing" (working paper, undated copy; references run to 2013) races seven
quality notions (p. 3): Graham's criteria, Grantham's, Greenblatt's return on capital, Sloan's
accruals, Piotroski's F-score, gross profitability, and low volatility/beta. **None is an
investment measure**; gross profitability *"largely subsumes the power of other notions of
quality"*, *"especially among large caps stocks and for long-only investors"* (p. 28). No
Novy-Marx critique of the investment factor itself was found, and none is cited.

### 1.5 What the literature implies for the fold

Two questions are run together in the factor-lab row. **Is investment a distinct dimension from
profitability?** Yes, everywhere: FF2015 (-0.21 conditionally), HXZ (0.06), Scientific Beta's
largest 500 (-0.18), this screener's scores (-0.061, §4) - which is what practice cites to bundle
two near-orthogonal signals as "Quality". **Is it worth 5 points beside value and momentum among
large companies?** Weakly at best (§1.2). That is a question about **weight**, the same whichever
heading the 5 points sit under.

## 2. Exactly how practitioners structure it

| Provider, document | Quality definition | Investment's place | Verified |
|---|---|---|---|
| **MSCI FaCS**, "MSCI FaCS Methodology", January 2018, p. 8 | Quality group = 0.25 Profitability + 0.25 **Investment Quality** + 0.25 Earnings Quality - 0.125 Leverage - 0.125 Earnings Variability | A full factor in the risk model, **grouped** under Quality for reporting | Yes, text and worked Microsoft example |
| **MSCI Barra** USSLOW datasheet, December 2014, p. 3 | - | "Management Quality": asset growth, investment and net issuance growth, a standalone style factor | Yes |
| **MSCI Quality Indexes**, May 2022, p. 4 | ROE, debt to equity, earnings variability | **Absent** | Yes |
| **S&P Quality Indices**, December 2014, p. 17 | ROE, accruals ratio (change in net operating assets over average total assets, inverted), financial leverage | In substance, via the accruals ratio | Yes (the 10-19 note read the March 2025 edition, Appendix A p. 22) |
| **FTSE Russell** Global Factor ground rules v8.8 | Profitability (ROA, accruals, change in asset turnover) and leverage | No asset growth; change in asset turnover instead | 10-19 note §9.2 |
| **AQR**, QMJ (RAS 2019) | Profitability, growth, safety; payout optional | Implicit, through residual-income growth (p. 43, fn. 12) | Yes |
| **Research Affiliates**, RAFI Multi-Factor rulebook v5.3 §2.8.1 p. 10; Hsu, Kalesnik & Kose (2019) | "Quality = average of Profitability minus the average of Investment" | **Blended** into one Quality score | 10-19 note; HKK pp. 54, 57 re-read tonight |
| **Scientific Beta**, "The Dimensions of Quality Investing", May 2017 | High Profitability and Low Investment as two "dimensions of quality" | **Two separate indices**, combined equally in a multi-beta "Quality" index (p. 36) | Yes |
| **Dimensional** | Profitability in all products | Investment used only as a small-cap exclusion | 10-19 note §9.4 |
| **Robeco**, "Guide to factor investing in equity markets" (undated), p. 16 | "high profitability, high earnings quality and conservative management (no exuberant M&A activity, for example)" | Inside Quality, by description | Yes, wording only; no construction given |
| **Northern Trust**, Invesco | - | Inside Quality (Invesco via licensed FTSE/S&P scores) | 10-19 note |
| **J.P. Morgan** | - | - | **Not verified**: no primary document obtained tonight |

Three things to take from the table.

**"Most practitioners fold it" hides a split.** MSCI's FaCS - the closest structure, because it
publishes a factor *and* a group - keeps investment as its own measured factor and adds it into
Quality one level up. Research Affiliates collapses it into one score. A screener that publishes
category scores and a drilldown is closer to FaCS.

**The one shop that tested the bundle as a factor objected to it.** Scientific Beta, on the 500
largest US stocks (Exhibit 14, p. 35, weekly, 1975-2015), measures the investment factor at -0.18
with profitability, +0.30 with value and **+0.68 with low volatility**, and writes that the two
*"cannot be combined in a composite manner to represent a single risk factor"* - then sells an
equal-weight combination of the two labelled "Quality". Both are true: two signals that investors
like to hold together.

**Practitioners' reason for folding is not the factor-lab row's question.** HKK (p. 57) argue
that profitability plus investment is the most parsimonious *definition of quality*; their stated
weakness of investment, *"a weaker multifactor alpha because it is correlated with the value
factor"* (p. 54), is about how much it is worth, not where it is filed.

## 3. What folding would mean concretely here

### 3.1 What moves, and the arithmetic

`factor_engine.CAT_METRICS["investment"]` holds one metric, `asset_growth`, at 5% x 100% =
**5.00 composite points**; nothing else needs rehoming. A fold that keeps every metric's composite
weight sets Quality to **27** and rescales by 22/27:

| Metric (generic) | Today, % of Quality | Composite points | Folded, % of Quality (x22/27) | Composite points |
|---|---|---|---|---|
| roic | 29 | 6.38 | 23.6296 | 6.38 |
| gross_profit_assets | 22 | 4.84 | 17.9259 | 4.84 |
| net_debt_to_ebitda | 20 | 4.40 | 16.2963 | 4.40 |
| piotroski_f_score | 16 | 3.52 | 13.0370 | 3.52 |
| beneish_m_score | 8 | 1.76 | 6.5185 | 1.76 |
| accruals | 5 | 1.10 | 4.0741 | 1.10 |
| **asset_growth** | - | **5.00** | **18.5185** (= 5/27) | **5.00** |

The bank table rescales the same way (Piotroski 12.2222, accruals 8.1481, ROE 28.5185, ROA
20.3704, equity ratio 12.2222, asset growth 18.5185). Banks get asset growth today through the
generic Investment table, so the bank Quality table must gain it explicitly or 46 banks lose it.
ROE and ROA keep exactly 7.70 and 5.50 points, so the bank P/B-ROE identity
(`tests/test_bank_pb_roe_identity.py`) survives in points; the Piotroski low-valuation profile's
freed 1.76 points still go to ROIC and gross profitability. None of these weights is a whole
number (Risk's 42.86 is precedent); rounding to 18.5 would move every composite and must not be
done silently.

### 3.2 What else would have to change in the same commit

- **Coverage counts: unchanged** - `asset_growth` stays weighted (27 of 27 generic, 24 of 24
  bank on the 10-30 payload).
- **Everything keyed on eight categories**: `factor_weights`, the four presets, the engine's
  category bounds (`"investment": (0.0, 15.0)`), `history.py`, `portfolio_constructor.py`, the
  correlation and sensitivity panels, the drilldown's category list and `contrib`,
  `stock_summary.py`, the methodology generator, and **48 test files**.
- **History.** `scripts/check_run_health.py` reads `improvement/dispersion_history.csv`
  **positionally** (`header=None`), so dropping a column shifts every later one; the engine's
  Investment IC series ends and Quality's restarts as a different object.
- **The Factor Lab.** `factor_lab.CATEGORY_MAP`'s Investment entry (four experiments) merges into
  Quality's, the scorecard test that ties it to `config.yaml` moves with it, and Growth's entry,
  which calls asset growth "Investment's measure", is rewritten.
- **The changelog** entry (rule 3) says plainly that the fold rests on practice, not a return.

### 3.3 The explainability trade

Seven categories is not materially easier than eight, and "quality means profitable, honestly
reported and not over-expanding" is a good sentence for a student. Three things count the other
way, specific to this screener:

- **The asset-turnover position becomes invisible.** The 10-19 note found `revenue_growth` and
  `asset_growth` net to a position on revenue outgrowing the balance sheet (+0.892 with the
  spread). Today the drilldown shows Growth rewarding one leg and a separate Investment line
  penalising the other; folded, the second leg is one of seven Quality rows, readable only by
  someone who already knows to look.
- **Quality would carry a fourth balance-sheet-growth signal.** Accruals are already counted three
  times inside Quality (CLAUDE.md 0.12, measured, deliberately unchanged), and `asset_growth`'s
  percentile correlates +0.246 with accruals' and +0.286 with Beneish's (§4). No points are
  double-counted, but the category becomes more "how fast did the balance sheet grow" than its
  name says.
- **A defensibility feature loses resolution** (rule 7). The weight-sensitivity panel shows
  Investment's top 25 as the most sensitive to its own weight (average Jaccard 0.709, lowest of
  eight; 10-30 payload). Folded, that is averaged into Quality's row.

## 4. Measurement: is a weight-preserving fold neutral? (structure, not returns)

On the committed `dashboard_data.js` - the **2026-10-30 02:00 run**, 499 stocks, regime NORMAL
(the 11-02 run was refused and is unpublished). A scratch script (not committed) reads each
stock's sector-percentiles (`stock_detail[t].pct`), the published weight profiles and coverage;
it reproduces every published composite to within 0.005 and every Quality score exactly (218
generic, 235 Piotroski-low-valuation, 46 bank profiles), then rescores Quality with asset growth
inside it as `compute_category_scores` would and recomputes the composite over seven categories.

**How Investment sits against the other categories** (Spearman, 499 stocks): Valuation +0.249,
Quality **-0.061**, Growth **-0.404**, Momentum -0.061, Risk +0.155, Revisions -0.073, Size
+0.272. Non-banks only (453): with Quality -0.087, with Valuation +0.232. The other seven explain
20.1% of its rank variance (15.8% on 2026-09-02, `research/2026-09-02-category-independence-synthesis.md`).
Against Quality's own metrics' percentiles: ROIC -0.145, gross profitability -0.051, net debt /
EBITDA -0.155, Piotroski +0.004, accruals +0.246, Beneish +0.286; for the 46 banks, ROA +0.319
and equity ratio +0.410. So investment is nearly orthogonal to Quality as scored, as every
source in section 1 found.

**The composite.** If every stock has every metric, a weight-preserving fold is an identity:
27 x (22Q + 5I)/27 = 22Q + 5I. Measured, that holds for **455 of 499 stocks** (|change| <= 0.005).
**The top 25 does not change, and no top-25 stock changes rank**; 8 of the top 50 move at least
one place, 57 stocks more than one place, 6 more than five. (The rank-25/26 gap is 0.031 points.)

The fold is **not** neutral in five constructions:

1. **Renormalisation inside Quality (44 stocks).** A category score averages only the metrics a
   stock has. Today asset growth is 5 of 100 points whatever Quality lacks; folded, it is 18.52%
   of the Quality weight *present*, and a stock missing asset growth gives its 5 points to Quality
   alone. 42 stocks miss a weighted Quality metric (Beneish 32, gross profitability 9, net debt /
   EBITDA 5, accruals 3, ROIC 2, Piotroski 2), 3 miss asset growth (DPZ, FDXF, HONA), and ERIE
   has no Quality score. Largest moves: **ERIE 43.64 -> 38.46** (rank 387 -> 450; asset growth
   becomes its whole 27-point Quality), **BX 41.29 -> 36.67** (418 -> 470); the rest under a point.
2. **The momentum panic rule.** `apply_momentum_regime` gives half of momentum's freed 3.9 points
   to `fw["quality"]`, so folded, asset growth gets **5.36 points instead of 5.00**. Applied
   hypothetically to this run (today is NORMAL): 484 composites change, one top-25 swap (NEM out,
   WDC in).
3. **The investor profiles.** The presets carry different Quality:Investment ratios (22:5, Value
   25:4, Growth 15:4, Momentum 12:5) but there is **one** within-Quality table, so folded, asset
   growth's points become Value 5.37 (CAH out, IT in), Growth 3.52 (no top-25 change), Momentum
   3.15 (FTNT, SWK out; DELL, INCY in), and almost every composite in each profile moves.
   Avoiding it needs per-profile metric tables, which do not exist and would need explaining.
4. **Everything that reads the Quality score.** Quality changes for every stock (mean absolute
   5.54 points, largest 24.15, Spearman 0.945 old against new), moving toward Valuation (+0.080 ->
   +0.162) and away from Growth (+0.185 -> +0.054). The trap flags read it: **value traps 45 ->
   46** (DAL, GEHC, WFC flagged; KHC, ZBH cleared), **growth traps 67 -> 72** (CAT, CPAY, DASH,
   GNRC, JBL, LITE, URI flagged; FANG, JCI cleared) - fast asset growers flagged, slow ones
   cleared; none in the top 25. A defensibility feature (rule 7) would change as a side effect:
   `research/2026-10-09-trap-flags.md` set its floors on a Quality with no asset growth in it.
5. **The health gate.** Quality's standard deviation falls 19.12 -> **16.22**, **15.3%** below its
   trailing ten-run median of 19.16 against `MAX_DISPERSION_DROP = 0.20`: the first run would pass
   with three-quarters of the margin spent by the methodology, not the data.

**A second cross-section, as a check on the first.** The same script on the 2026-11-02 payload
(the refused run, as regenerated in the working tree tonight): 43 stocks non-neutral (ERIE -5.40),
top 25 unchanged, 15 trap flags flipped (4 value, 11 growth; none in the top 25), no top-25 swap
in the hypothetical panic, profile swaps Value 2, Growth 1, Momentum 1, Quality's dispersion
19.04 -> 16.17. The pattern holds; the counts move from run to run.

So in its normal state a weight-preserving fold is **mostly a relabelling**, and must be justified
as one, by explainability and practice. Its non-neutral edges are exactly where the machinery
treats a category as a unit - which is what a category is.

## 5. Recommendation

**Keep Investment as a standalone category. Do the sound part of the practitioner finding in
words, not in structure.**

1. **A fold changes nothing the evidence is about.** The live questions about these 5 points are
   *how much* (§1.2: weak among large companies, next to timely value and momentum, since 2003)
   and *what shape* (10-19 note §6.3: an extreme-decile effect scored linearly). Both are the same
   question inside Quality or outside it.
2. **Practice is split on the form that matters.** The structure closest to a published drilldown
   (FaCS) keeps investment as its own factor and groups it one level up; the shop that tested the
   bundle as a factor (Scientific Beta) says it is two. One blended Quality score copies Research
   Affiliates, not the field.
3. **The costs are measured** (§4): 44 stocks re-weighted by missing data, two of three profiles'
   top 25 changed, 14 trap flags flipped without research, a gate's margin consumed, positional history
   broken, 48 test files - and the asset-turnover position hidden from the student.

**Instead, for a build day, one commit with its claim registered:** the methodology page's
Investment section says that most practitioners house asset growth inside Quality (MSCI at a
quarter of its Quality group; Research Affiliates as profitability minus investment), that the
academic models and Scientific Beta keep it separate, and that the screener does so to keep its 5
points visible beside the revenue growth it offsets - in the same paragraph as the large-company
caveat the 10-19 note §11.2 already queued.

**What would reopen the fold:** a Monday that changes asset growth's weight or shape. If its weight
falls far enough that a one-metric category stops earning a line, housing the residual inside
Quality becomes natural - decided with the weight, one change at a time (`plan/factor-lab.md`,
"From the lab to the weights"), not before it. A middle path - keep the scoring, *display*
Investment under a "Quality and investment" heading, FaCS-style - changes no number but is design
work nightly sessions do not take on (CLAUDE.md 0.8/0.9), and buys less than the paragraph above.

## What this does not settle

- **Whether 5 points is the right weight for asset growth among large companies.** The evidence is
  weak (§1.2) and no change is proposed; that is its own Monday, with the tail-shape question
  (10-19 note §11.3 item 1) beside it.
- **Section 4 is one run's cross-section.** The 44-stock count moves with coverage, and the panic
  and profile swaps would differ on another day: it shows that and where the fold is non-neutral.
- **Unverified:** J.P. Morgan's quality definition (not obtained); Asness (2014)'s coefficients
  (tables are images - only the text was read); QMJ's published *results* (definitions only); the
  March 2015 Scientific Beta edition against the May 2017 one read. Fairfield, Whisenant & Yohn
  (2003), which would put accruals and asset growth in one specification, is still unobtained.
- **Nothing here says investment earns anything in this screener.** Section 4 is structure.

## Sources

**Literature**

- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial
  Economics* 116(1), 1-22. https://doi.org/10.1016/j.jfineco.2014.10.010 - Table 4 (p. 7), p. 8,
  Table 6 (p. 11), §7 (p. 12). Read from `data/papers/ff2015.pdf`.
- Hou, K., Xue, C., & Zhang, L. (2015). Digesting anomalies: An investment approach. *Review of
  Financial Studies* 28(3), 650-705. https://doi.org/10.1093/rfs/hhu068 - pp. 660, 662. Read from
  `data/papers/hxz_pub_a.pdf` (published text).
- Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. *Review of Accounting
  Studies* 24(1), 34-112. Publisher's version via
  https://research-api.cbs.dk/ws/portalfiles/portal/60211462/lasse_heje_pedersen_et_al_quality_minus_junk_publishersversion.pdf
  - p. 43 (eq. 6, footnote 12), p. 75 (payout). The October 2013 draft,
  https://ssrn.com/abstract=2312432, p. 10.
- Asness, C. S. (2014). Our model goes to six and saves value from redundancy along the way. AQR,
  Cliff's Perspective, 17 December 2014.
  https://www.aqr.com/insights/perspectives/our-model-goes-to-six-and-saves-value-from-redundancy-along-the-way
  - pp. 1, 5. Read from `data/papers/aqr-model-goes-to-six.pdf`.
- Hsu, J., Kalesnik, V., & Kose, E. (2019). What is quality? *Financial Analysts Journal* 75(2),
  44-61. https://doi.org/10.1080/0015198X.2019.1567194 - p. 54, Table 6 (p. 56), p. 57. Read from
  `data/papers/ra-what-is-quality-faj-2019.pdf`.
- Novy-Marx, R. Quality investing. Working paper, University of Rochester (undated copy).
  https://mysimon.rochester.edu/novy-marx/research/QDoVI.pdf - pp. 1-2, 28.
- Cooper, M. J., Gulen, H., & Schill, M. J. (2008). Asset growth and the cross-section of stock
  returns. *Journal of Finance* 63(4), 1609-1651. https://doi.org/10.1111/j.1540-6261.2008.01370.x
  - via the 10-19 note §6.1.

**Practice**

- MSCI (2018). MSCI FaCS Methodology (Bonne, Roisenberg, Aylur Subramanian, Melas), January 2018,
  p. 8.
  https://www.msci.com/downloads/web/msci-com/research-and-insights/blog-post/factors-and-esg-the-truth-behind-three-myths/factors-and-esg-the-truth-behind-three-myths-MSCI-FaCS-Methodology.pdf
  (hosting URL from search; text read from `data/papers/msci-facs-methodology.pdf`).
- MSCI (2014). Model datasheet: Barra US Total Market Equity Model for Long-Term Investors
  (USSLOW), December 2014, p. 3. `data/papers/barra-usslow-datasheet.pdf` (URL not re-checked).
- MSCI (2022). MSCI Quality Indexes Methodology, May 2022, p. 4.
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_Quality_Indexes_Methodology_May2022.pdf
- S&P Dow Jones Indices (2014). S&P Quality Indices Methodology, December 2014, p. 17.
  `data/papers/spdji_quality_b.pdf`; current edition:
  https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/methodology-sp-quality-indices.pdf
- ERI Scientific Beta (2017). The Dimensions of Quality Investing: High Profitability and Low
  Investment Smart Factor Indices, May 2017, pp. 10, 35-36.
  `data/papers/scibeta-the-dimensions-of-quality-investing.pdf`; a March 2015 edition:
  https://conferences.pionline.com/uploads/conference_admin/ERI_Scientific_Beta_Publication_Dimensions_of_Quality_Investing1.pdf
- Robeco. Guide to factor investing in equity markets (undated), p. 16.
  `data/papers/robeco-guide-factor-investing.pdf` (URL not re-checked).
- Research Affiliates RAFI Multi-Factor rulebook v5.3; FTSE Russell Global Factor Index Series
  ground rules v8.8; Dimensional, Northern Trust, Invesco - as cited and verified in
  `research/2026-10-19-revenue-growth-vs-asset-growth.md` §9.

**This project**

- `research/2026-10-19-revenue-growth-vs-asset-growth.md` (§3, §4, §6, §9, §11.3);
  `research/2026-10-19-lab-asset-growth.md` (§5); `research/2026-10-12-growth-category.md`;
  `research/2026-11-02-growth-exposure-drift.md`; `research/2026-09-02-category-independence-synthesis.md`;
  `research/2026-10-12-factor-lab-findings.md` §3; `DECISIONS.md` "The four weight questions the
  lab raised"; `config.yaml`; `factor_engine.py` (`CAT_METRICS`, `metric_weight_profiles`,
  `compute_category_scores`, `apply_momentum_regime`, `compute_composite`, the trap flags);
  `scripts/check_run_health.py` (`check_dispersion`); `dashboard_data.js` (2026-10-30 run).

---

## 6. Review, 2026-11-04 (06:00 session, which did not write this note)

**Verdict: the fold is REFUSED and section 5's recommendation is ACCEPTED.** Investment stays
standalone; the practitioner convention, the large-company caveat and the joint position with
Growth are now on the methodology page behind claim `overview.investment_standalone`
(`METHODOLOGY_CHANGELOG.md` 2026-11-04). This is the first of seven reviewed proposals whose
recommendation was accepted as written - the six before it were refused as written - and the
reason is visible in the note's own structure: it recommends *no change to any number*, and it
measured the costs of the change it declines before declining it.

### 6.1 What was re-measured, and on how many runs

Section 4's script was not committed, so the review could not read it. It was rebuilt from
`factor_engine` instead of from the note - `research/measurements/2026-11-04-investment-fold-review.py`,
committed - and run on **the committed payload plus the last eight published runs**
(2026-10-22 to 2026-11-04), which closes the note's own caveat that "section 4 is one run's
cross-section". The reproduction is faithful: published minus recomputed is **3.5e-07** on
Quality, **0.0** on Investment, and **4.99e-03** on the composite, the last being the payload's
own two-decimal rounding.

| Section 4's claim | the note (2026-10-30) | this review (2026-11-04) | over eight runs |
|---|---|---|---|
| composites unchanged by the fold | 455 of 499 | **455 of 499** | **455 on all eight** |
| top 25 changed | none | **none** | **0 in, 0 out, all eight** |
| stocks missing a weighted Quality metric | 42 | **42** | - |
| stocks missing asset growth | DPZ, FDXF, HONA | **DPZ, FDXF, HONA** | - |
| stocks with no Quality score | ERIE | **ERIE** | - |
| largest composite move | 5.19 points | **5.103 points** (BX) | - |
| Quality dispersion drop | 15.3% | **15.4%** | **15.0% to 15.4%** |
| trap flags flipped | 14 | **15** (4 value, 11 growth) | **13 to 17** |
| Investment vs Quality (Spearman) | -0.061 | **-0.068** | - |

Every claim holds. The fold's arithmetic was re-derived from `config.yaml` as well: 5/27 =
**18.5185%**, the rescale factor 22/27 = **0.814815**, and every metric's composite points are
preserved by construction. The dispersion result is the one worth keeping as a range rather than
as a number: the fold would spend **three-quarters of the health gate's margin** on the first run
after it shipped, on all eight runs measured, and the gate does read Quality (see 6.2).

### 6.2 One of the listed costs does not exist

Section 3.2 lists, among what a fold would have to change: *"`scripts/check_run_health.py` reads
`improvement/dispersion_history.csv` **positionally** (`header=None`), so dropping a column shifts
every later one."* The positional read is real. The consequence is not, twice over:

1. `investment_disp` is the **last** of the file's nine columns, so dropping it shifts nothing.
2. `check_run_health.CATEGORIES` is `["valuation", "quality", "growth", "momentum", "risk"]` -
   **five** categories. The dispersion gate never reads `revisions_disp`, `size_disp` or
   `investment_disp`, at any position.

So this cost is struck, which makes the fold marginally *cheaper* than the note argued - the same
direction as the 2026-10-30 review, and the reason a reviewer measures a cost list rather than
reading it. It does not change the verdict: the gate reads Quality at position 2, so section 4
item 5 - the 15.4% dispersion drop - stands, and is the live one. (A separate observation, not a
defect: the file **has** a header row, which `header=None` loads as data and
`pd.to_numeric(..., errors="coerce").dropna()` then discards. It works; it works by accident.)

### 6.3 The coherence argument the note did not make

This is a synthesis session, so the question it owes is not only "is the fold neutral" but "does
the result fit the other seven categories". Reading Scientific Beta's Exhibit 14 (p. 35) rather
than only the sentence the note quotes beside it, the investment factor's correlations **on the
500 largest US stocks, 1975-2015** are:

| Investment factor vs | correlation |
|---|---|
| Profitability | **-0.18** |
| Size | -0.02 |
| Momentum | -0.04 |
| Value | +0.30 |
| **Low Volatility** | **+0.68** |

**Among large companies the investment factor's strongest documented relationship by a wide margin
is with low volatility, not with profitability** - and this screener's own scores agree in
direction, if far more weakly: Investment against Quality is **-0.068** today, against Risk
**+0.166**, against Valuation **+0.228** and Size **+0.259**. Folding asset growth into Quality
would house it under the category it has the *least* relationship to, which is a coherence
argument against the fold independent of explainability. It also reframes the remaining open
questions: if these 5 points overlap anything in this screener, it is Risk's 10 and Valuation's
22, not Quality's 22. **That is a question for a Monday, and nothing here authorises moving a
weight on it** - the correlation is between *factors* in a long/short construction, not between
this screener's categories, and +0.166 against Risk is a weak reading on one run.

Exhibit 15, on the facing page and not cited by the note, measures the same thing by overlap: the
low-investment and high-profitability factor indices **share about 50% of their stocks (110 of
250)** out of a 500-stock universe, which Scientific Beta notes is *"not very different from that
of other factor indices such as value, size and momentum, each of which shares roughly 50%"*.
That is a direct measurement of the factor-lab row's question, on a universe the size of this
screener's, and it points the same way as the -0.18.

**And one consequence of the fold that the note states but understates.** ERIE misses all six
weighted Quality metrics and has no Quality score at all. Folded, its "Quality" score would be
**asset growth alone** - a stock with no quality data would be published with a quality number,
because renormalisation would hand asset growth the whole 27 points. A category score that can be
composed entirely of the one metric that is not about quality is a worse outcome than a 5-point
relabelling is worth.

### 6.4 Sources: every load-bearing figure re-read from the PDF bytes

Read with `pypdf` from the files in the gitignored `data/papers/`, not from the note:

| Source | Checked | Result |
|---|---|---|
| **Fama & French (2015)** Table 6, p. 11 | HML intercept -0.04 (t -0.47), slopes CMA 1.04 / RMW 0.23 at t 23.03 / 5.36; CMA intercept 0.28 (t 5.03), slopes HML 0.45 / RMW -0.21 (t -7.84) | **exact, all eight figures** |
| **FF2015** Table 4, p. 7 | big-company CMA 0.22 (t 2.00), 0.11 (t 1.50), 0.07 (t 1.03) across the 2x3, 2x2 and 2x2x2 sorts | **exact, all six** |
| **FF2015** p. 8 | *"In the 2x2x2 sorts the average value of CMA_B is 0.07% per month (t = 1.03), and almost all the average value of CMA is from small stocks"*; *"the average value of CMA_B is only 1.03-2.00 standard errors from zero"* | **exact, and the page is right** |
| **FF2015** variable definition | *"Inv is the growth of total assets for the fiscal year ending in t-1 divided by total assets at the end of t-2"* | **exact** - see 6.5 |
| **MSCI FaCS Methodology**, Jan 2018 | *"we assigned a 25% weight to each of the three SES quality factors and a -12.5% weight to each of the other two"*; Exhibit 2's weight column; the Microsoft worked example, 0.25x0.169 + 0.25x(-0.055) + 0.25x0.403 - 0.125x(-0.442) - 0.125x0.289 = 0.148 | **exact**; the weights sentence is on p. 8 as cited, the worked example on **p. 9** |
| **ERI Scientific Beta**, May 2017 | Exhibit 14 (p. 35): investment vs profitability -0.18, value +0.30, low volatility +0.68; weekly, 1975-2015, EDHEC Risk LTTR US = the 500 largest US stocks | **exact**; the *"cannot be combined in a composite manner"* conclusion is on **p. 34**, not p. 35 |
| **Hsu, Kalesnik & Kose (2019)** p. 57 | *"most models considered in academia include profitability and investment as separate factors"* | **exact** |
| **Asness (2014)** p. 5 | *"using up-to-date timely value, and making the obvious addition of momentum to the Five Factor Model, it's actually CMA that's mostly redundant"*, and *"it's still statistically almost, barely, but not quite interesting"* | **exact**, and the note's "comes close but doesn't break 2.0" is a fair reading of the second |
| **Factor Lab**, big-company investment factor | recomputed over FF2015's own sample, 1963-07 to 2013-12 | **+0.2165% a month, t 1.991** - an independent reproduction of their Table 4 CMA_B of 0.22 at t 2.00 |

Two one-page citation slips (the Scientific Beta quote, the MSCI worked example) and **no wrong
figure**. Sections 1 and 2 are sound as written.

**One of its own figures has drifted, and the cause is the lab, not the note.** Section 1.2 and
Finding 2 give the lab's big-company investment factor since 2003 as **-0.06% a month (t -0.35)**.
Recomputed today it is **-0.049% (t -0.312)**: the lab's French-library data has gained months
since 11-02. The sign, the significance and the argument are unchanged - which is why the page
states this one as a band held to the lab's live data (`tests/test_investment_standalone.py`)
rather than as a literal that goes quietly stale.

### 6.5 A correction the review made to its own first draft

The page paragraph first said the screener uses total-asset growth "not the per-share version Fama
& French use to sort CMA". **That is wrong**, and the 10-19 note says so at its line 319: FF2015's
`Inv` *is* total-asset growth, confirmed above from the paper itself. The per-share construction
belongs to Fama & French's earlier anomaly work, which is how §11.2 item 2 put it ("not FF2008's
per-share version") and how the page now puts it. Recorded because the error is instructive: the
distinction is real and worth stating, and getting its attribution backwards would have put a
false sentence about a cited paper on the public methodology page. Writing a metric's provenance
down is what exposed it - the same lesson CLAUDE.md 0.10 draws about writing a metric's arithmetic
down.

### 6.6 What shipped, and what did not

**Shipped:** four paragraphs in `build_screener_overview()`'s Investment section - the measure and
why it matters, the large-company reading, the joint position with Growth, and why the category is
standalone when most providers fold it - with claim `overview.investment_standalone` and
`tests/test_investment_standalone.py` (6 tests, in the 02:00 publish gate by derivation). That
closes items 1, 2 and 3 of `research/2026-10-19-revenue-growth-vs-asset-growth.md` §11.2, open
since 2026-10-19; item 4 was already on the page in the Growth section.

**Not shipped, deliberately:** no weight, metric, threshold or scoring line changed, and
`CAT_METRICS` is untouched. The FaCS-style display option in section 5 ("Quality and investment"
as a heading over an unchanged score) is still design work nightly sessions do not take on, and
still buys less than the paragraph.

**Left open, unchanged by this review:** whether 5 points is the right weight for asset growth
among large companies, and the tail-shape question (§11.3 item 1 of the 10-19 note) - both their
own Mondays. The trigger to reopen the fold is the one section 5 names: a weight change that makes
a one-metric category stop earning a line. 6.3 adds a second question for a Monday, about Risk and
Valuation rather than Quality.

