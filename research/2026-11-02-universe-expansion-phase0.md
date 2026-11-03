# Below the large caps: what changes when the screener leaves the S&P 500 (Phase 0)

2026-11-02, owner-run evening; Phase 0 of plan/universe-expansion.md

**Proposal status: OPEN** - a later session (not the one that wrote this) decides whether to adopt
the three pre-registrations in section 7: the S&P 1500 as the first wider universe, the NYSE
20th-percentile market cap as the phase-4 floor, and percentiles ranked within sector *and* size
band.

**Question.** The owner wants NYSE and NASDAQ stocks beyond the S&P 500, phased in
(`plan/universe-expansion.md`). Phase 0 asks five questions before any code moves: which universe
comes first; where the microcap floor sits; how percentiles should be ranked once small companies
share the pool; what thinning analyst coverage does to the Revisions category and the coverage
filter; and which of this repo's research notes stop applying below the large caps. Research-led
per CLAUDE.md rule 4. No IC series, backtest number or return of the screener's own ranking is used
anywhere below. The measured numbers in section 8 are data-coverage counts, not returns.

**Finding.**

1. **S&P 1500 first. The expectation holds, with two caveats.** The S&P 400 and 600 apply the same
   committee rules as the S&P 500: positive GAAP earnings in the latest quarter and over the
   trailing four, a 0.10 float minimum and a liquidity ratio. The S&P 600's addition floor ($1.2B,
   July 2025) sits close to the NYSE 20th percentile ($1.22B, June 2026). That is the line the
   anomaly literature draws between microcaps and everything else. So the S&P 1500 is, nearly, "all
   but microcaps, with a profitability screen". Caveat one: the screens apply **at addition only**,
   so current members need not still pass them. Caveat two: `config.yaml`'s
   `universe.min_market_cap: 2e9` would silently drop the bottom of the S&P 600 (section 7).
2. **The floor: the NYSE 20th-percentile market cap, taken from Ken French's free breakpoint file
   each June and held for twelve months. Today that is $1.224B.** Fama & French (2008), Hou, Xue &
   Zhang (2020) and Jensen, Kelly & Pedersen (2023) all use this line. In Fama & French's sample,
   stocks below it are about 60% of all US stocks and about 3% of market value. With microcaps
   handled this way, 65% of 452 published anomalies fail a t of 1.96 (HXZ). Practice sets floors
   far lower (Russell $30M, MSCI's $537M). That is because their goal is market coverage, not
   signal quality.
3. **Rank within sector *and* size band.** All three index providers keep size bands apart when
   they score style. Russell scores the R1000, R2000 and microcaps separately because "valuations
   of small stocks differ from those of large stocks" (its rule 8.2.3). MSCI z-scores within the
   Standard and Small Cap indexes separately. S&P ranks within each parent index. The repo change
   is small: `compute_sector_percentiles` groups by `Sector` alone and would group by
   `(Sector, size_band)`. Because the S&P 500 band *is* today's universe, its percentiles come out
   byte-identical by construction.
4. **Analyst coverage roughly halves at each step down.** Median analyst count on Yahoo: 20 for the
   S&P 500 (this morning's run), 11 for an S&P 400 sample, 6 for an S&P 600 sample. 10% of the S&P
   600 sample has fewer than three analysts, against 1% of the S&P 500. **For a stock with no
   coverage, today's code scores the whole Revisions category (10 points) on days-to-cover alone.**
   Its coverage (22 of 27 weighted metrics, 81.5%) also sits just above the 80% line where the
   coverage discount starts. So the stock passes every filter and is never discounted.
5. **Eighteen notes stop applying as written** (section 5). Five matter most before anything
   publishes: the Size-weight decision, the value/momentum-among-large-companies note, the
   asset-growth notes, the Revisions note and the reporting-lag note. In the first and fourth,
   evidence from small stocks points the other way.

---

## 1. Which universe first: S&P 1500, or all NYSE + NASDAQ above a floor

### What S&P actually requires (documented practice)

*S&P U.S. Indices Methodology*, "Eligibility Criteria". Read from the December 2022 text, the newest
copy that could be retrieved (spglobal.com returns 403 to non-browser clients; mirror in Sources).
Later changes are taken from S&P DJI's own press releases:

| Rule | S&P Composite 1500 (500 / 400 / 600) | S&P Total Market Index |
|---|---|---|
| **Financial viability** | "The sum of the most recent four consecutive quarters' GAAP earnings (net income excluding discontinued operations) should be positive as should the most recent quarter." REITs may use FFO. Confirmed unchanged by the June 2026 megacap consultation, which also declined cap-based exceptions | None |
| **Market cap at addition** (unadjusted company level, July 2025) | 500: >= $22.7B; 400: $8.0-22.7B; 600: $1.2-8.0B. Reviewed each quarter. Designed to capture the TMI's cumulative market cap at about the 85th (500), 85th-93rd (400) and 93rd-99th (600) percentiles | None |
| **Float-adjusted market cap** | >= 50% of the index's minimum | - |
| **Float (IWF)** | >= 0.10 | >= 0.10 |
| **Liquidity** | Float-adjusted liquidity ratio (annual dollar value traded / float-adjusted cap) >= **0.75** at addition. It was 1.00 until the January 2023 consultation. The December 2022 text adds 250,000 shares traded in each of the six months before evaluation; whether that survived 2023 is *not verified*. "Current constituents have no minimum requirement" | FALR >= 0.10 |
| **Excluded types** | BDCs, LPs, MLPs, royalty trusts, ADRs, preferreds, closed-end funds, unit trusts and others | Same list |
| **Migrations** | A member moving between the 500, 400 and 600 "do[es] not need to meet the financial viability, liquidity, or 50%" float tests | - |

Two things follow. First, the earnings screen is a **junk filter at the boundary**, applied at entry
and never again. A member that turns unprofitable stays in; one that shrinks out of the S&P 500 can
migrate to the 400 without re-passing it. Second, the 400 and 600 bands map closely onto academic
size groups. Using June 2025 NYSE breakpoints, the nearest date to the July 2025 thresholds (Ken
French's `ME_Breakpoints`, CRSP vintage 2026-09):

| | Addition range (Jul 2025) | NYSE percentile it corresponds to (Jun 2025) |
|---|---|---|
| S&P 500 | >= $22.7B | ~80th ($23.4B) - JKP's "mega" line |
| S&P 400 | $8.0-22.7B | ~62nd-80th (60th = $6.8B) |
| S&P 600 | $1.2-8.0B | ~25th-62nd (20th = $0.87B, 30th = $1.67B) |

### The alternative: all NYSE + NASDAQ above a floor

FTSE Russell's rules show what a rules-based all-listed universe needs. *Russell US Equity Indexes
Construction and Methodology* v7.2, August 2026, sections 5-6:

- **Exchange:** US exchange only. OTC, bulletin-board and pink-sheet stocks are excluded.
- **Price:** close of at least $1.00 on rank day (or a 30-day average of at least $1.00).
- **Size and float:** total market cap of at least $30M, and at least 5% of shares in free float.
- **Excluded types:** royalty trusts, LLCs, closed-end funds and BDCs, blank-cheque companies and
  SPACs, LPs, ETFs and tokenized shares. REITs and partnerships that generate UBTI are also
  screened out.
- **Construction:** the largest 4,000 companies by total market cap form the R3000E. The R2000 is
  ranks 1,001-3,000 and the Microcap index ranks 2,001-4,000.
- **Other rules:** ±2.5% cumulative bands at the breakpoints, and a second share class needs daily
  dollar volume above the global median ($175,000 at the 2026 rank date).
- **No earnings screen.**

The NASDAQ Trader directory the plan names also carries ETF and test-issue flags. Even so, a
from-scratch universe would have to rebuild Russell's whole exclusion list itself, from free data.

### Literature

- **The profitability screen is a quality control at the boundary.** Asness, Frazzini, Israel,
  Moskowitz & Pedersen (2018, *JFE* 129(3)) show the size premium is masked by junk and appears
  once quality is controlled. That note is already read in `research/2026-10-20-size-weight.md`.
  The S&P 600 admits only companies that were profitable at entry. A floor-based universe admits
  every loss-maker above the floor.
- **S&P's own research** (Dash & Soe 2009, "A Tale of Two Small-Cap Benchmarks", and its ten-year
  follow-up) credits most of the S&P 600's long-run lead over the Russell 2000 to that
  profitability screen. *Read only through summaries (the Indexology pages return 403); this is
  vendor research about the vendor's own index, so it is weak evidence.* It is cited only to show
  that the provider itself names the screen as what separates its small-cap index.

### Answer

**S&P 1500 first.** Five reasons:

1. It keeps the screener's implicit premise, a profitable company at entry, which is what the large-cap
   research notes assume.
2. It stays at about 1,500 names, about 3x today's fetch (plan table).
3. Free constituent lists exist (Wikipedia; the S&P 500 already uses the same pattern).
4. Its bottom edge falls at roughly the academic microcap line, so the papers in section 2 apply
   to everything in it.
5. Analyst coverage is still there for most names (section 4).

The **all-listed universe above the NYSE 20th percentile** turns out to be *about the same
size*. Microcaps are ~60% of listed stocks (FF 2008; HXZ's 1967-2014 average: 2,406 of 3,938, or
61%), so ~1,500-1,800 names remain above the line (*an estimate from those shares; today's count is
not measured*). What separates the two universes is composition, not size: loss-makers, recent IPOs
and the exclusion work above.

**What the expectation missed:** the S&P 1500 is a **committee-chosen** universe whose screens
apply at entry. A screener ranking it inherits a quality tilt it did not choose. The methodology
page must say so, the same way it states the S&P 500's.

---

## 2. Microcaps, and where the floor goes

### Literature

**Fama & French (2008), "Dissecting Anomalies", *Journal of Finance* 63(4), 1653-1678.** Read from
the published PDF. The sample is NYSE, Amex and (after 1972) NASDAQ stocks, July 1963 to December
2005. Microcaps are stocks below the NYSE 20th percentile of end-June market cap, small stocks fall
between the 20th and 50th, and big stocks sit above the median. At end-June 2005 those lines were
$610M and $2.3B.

- Microcaps are about **60% of stocks but about 3% of market cap**. Big stocks are over 90% of
  market cap. Between 1973 and 2005, microcaps ranged from 48% to 69% of firms and 1.0% to 5.0% of
  market cap.
- Microcaps dominate equal-weighted sorts and all-stock regressions. They "typically account for
  more than 60% of the stocks in extreme sort portfolios". From practice, anomaly returns found
  only in microcaps "are probably not realizable" because those stocks are expensive to trade.
- **Size itself is a microcap effect** (Table IV, monthly slopes on log market cap): microcaps
  -0.46 (t -6.95), small stocks -0.03 (t -0.37), big stocks -0.08 (t -1.96).
- **The asset-growth anomaly has a size gradient.** Value-weighted hedge returns: microcaps -0.57%
  a month (t -4.42), small -0.31% (t -2.43), big -0.02% (t -0.10). Regression slopes: -0.83
  (t -6.82), -0.57 (t -3.10), -0.17 (t -0.86).
- **Momentum is strong above the microcap line:** slopes of 0.82 (t 4.65) for small stocks and
  0.78 (t 3.92) for big, against 0.41 (t 2.51) for microcaps. Book-to-market: 0.23, 0.30 and 0.17
  (t 3.19, 3.41, 1.79).

**Hou, Xue & Zhang (2020), "Replicating Anomalies", *Review of Financial Studies* 33(5), 2019-2133.**
The published abstract (via RePEc) reports what happens with microcaps handled by NYSE breakpoints
and value-weighted returns:

- **65% of 452 anomalies fail |t| >= 1.96**, including 96% of the trading-frictions category.
- At the multiple-testing hurdle of 2.78, **82% fail**.

The 2017 NBER version (read in full) gives 64% of 447 and 85% at t >= 3. It also explains why: many
original studies used all-exchange breakpoints with equal weights, so microcaps filled the extreme
deciles. That version also updates FF's Table I for 1967-2014:

- **Share of names and of market value:** microcaps are 61% of firms but 3.28% of market cap;
  small stocks 6.77%; big stocks 90%.
- **Cross-sectional return spread:** the standard deviation of monthly returns is 19.1% for
  microcaps, 11.9% for small stocks and 8.9% for big stocks.

**Jensen, Kelly & Pedersen (2023), "Is There a Replication Crisis in Finance?", *Journal of Finance*
78(5).** Read from the PDF the Factor Lab already holds (`research/2026-10-20-lab-jkp.md`). JKP form
breakpoints on non-micro stocks too, but they **disagree with HXZ on how much survives**.

- **Replication rates by size group:** using empirical-Bayes rates, mega 77.3%, large 79.8%, small
  85.7%, micro 85.7% and nano 68.1%.
- **Where the ordering breaks:** only nano stocks (below the NYSE 1st percentile: 458 of 4,356 US
  stocks at end-2020) "exhibit notable deviation". The rank correlation of cluster alphas is 73%
  between mega and micro, but 36% between mega and nano.

The disagreement is about method, not data. HXZ count a decile spread's raw t-statistic. JKP use
CAPM alphas, shrink each toward its theme cluster, and value-weight terciles with capped weights.
**Both put the line at the NYSE 20th percentile when forming breakpoints.** The open question
between them is whether the factors are weak everywhere (HXZ) or about as strong in every size
group except the very smallest (JKP).

### Practice

| Who | Floor | What it is for |
|---|---|---|
| Russell 3000E | $30M total cap, price >= $1, float >= 5% | Coverage of the investable market (largest 4,000) |
| MSCI USA IMI | UMSR = the full cap where cumulative float cap reaches 99% of the DM universe: **$537M** (April 2026); float cap >= half that; 12-month ATVR >= 20%, 3-month ATVR >= 20% in each of the last four quarters, trading frequency >= 90% | 99% coverage, liquid enough to replicate |
| S&P SmallCap 600 | $1.2B at addition + earnings + FALR 0.75 | A small-cap benchmark that is investable and profitable |
| S&P TMI | None; FALR >= 0.10 | Total coverage |
| Academic factor construction (FF, HXZ, JKP) | **NYSE 20th percentile** for breakpoints | Keep microcaps from driving the result |

Index providers set floors low because a benchmark has to cover the market. A screener has a
different job: making *relative* calls among stocks, so its floor should sit where the evidence for
those calls is cleanest and the stocks are cheap enough to trade. FF 2008's "probably not
realizable" and HXZ's microcap argument are both about the screener's problem, not the index's.

### The pre-registered floor

**Total company market cap >= the NYSE 20th-percentile breakpoint in Ken French's
`ME_Breakpoints` file for the most recent June, applied from July and held for twelve months.
Value for July 2026-June 2027: $1,224.13M** (the June 2026 row, n = 1,079 NYSE stocks). Recent
Junes for scale: $0.674B (2022), $0.760B (2023), $0.837B (2024), $0.870B (2025).

Why this number and not another:

- **It is the line all three replication papers use**, so their findings apply above it.
- **It lines up with S&P's own small-cap entry floor** ($1.2B), so phase 4 admits no company smaller
  than S&P already admits.
- **It moves with the market**, so it never needs re-tuning. Fixing it each June follows Fama-French
  timing.
- **It is free and published monthly.**

MSCI's $537M (about the NYSE 10th percentile) is the practice-based runner-up. It is not chosen
because the band between it and the 20th percentile is exactly where FF and HXZ show microcaps
driving results, and where analyst coverage thins fastest (section 4). Russell's price-$1 and $30M
rules are almost certainly redundant above $1.2B (*not measured*). Apply them anyway as exclusions
in phase 4, at no cost. **The floor applies in phase 4 only.** Phases 2-3 rank the S&P 1500 as S&P
defines it. Phase 2 should count how many S&P 600 members sit below the floor, because existing
members are not removed when they shrink.

---

## 3. Percentiles: within sector, or within sector and size band?

### How the repo ranks today

`factor_engine.compute_sector_percentiles` groups by `Sector` (11 GICS sectors) and ranks each
metric with `_directed_pct` (midpoint rank, (r - 0.5)/n x 100). Where a sector has fewer than
`SECTOR_MIN_PEERS = 10` valid values, it falls back to a **universe-wide** rank for that metric.
Missing values stay missing, and `compute_category_scores` renormalises over the metrics present.
The Size category is -ln(cap), ranked the same way, within sector. `Composite_Pct` is ranked across
the whole universe.

### What happens if small companies simply join the pool

A small company's value, quality and risk would be ranked against large companies in its sector.
Three of the eight categories would then measure size as much as the trait they are named for:

- **Risk (10 points).** Volatility, beta and drawdown are strongly size-dependent: monthly return
  spread of 19.1%, 11.9% and 8.9% for micro, small and big (HXZ 2017, Table 2). The S&P 600 would
  sit at the bottom of Risk in every sector.
- **Size (5 points)** would rank every S&P 600 stock near the top of its sector. The 5-point tilt
  was sized for the 500 largest (`research/2026-10-20-size-weight.md`); here it would become a
  small-cap tilt.
- **Valuation (22 points).** Russell's rule 8.2.3 states the provider's reason for separating
  bands: small and large stocks are valued differently, so pooling distorts relative valuations.

Pooling would also **change every S&P 500 percentile**, which breaks the plan's first fixed
commitment.

### What practice does

- **FTSE Russell** (rule 8.2.3-8.2.4) runs its value/growth scoring "separately to the stocks in
  the Russell 1000 and Russell 2000 and to the smallest 1,000 stocks in the Russell Microcap".
- **MSCI** (*Global Investable Market Value and Growth Index Methodology*, May 2023, section 2.2):
  variables are "standardized within each individual MSCI Market Standard and Small Cap Index". They
  are winsorized at the 5th and 95th percentiles and z-scored on the float-cap-weighted mean and
  standard deviation. Its multi-factor indexes z-score value and quality within sector *inside a
  size-defined parent* (Diversified Multiple-Factor methodology, as read in the size-weight note).
- **S&P** (*U.S. Style Indices Methodology*, June 2022) standardises each factor across the whole
  Total Market Index, after winsorizing at the 90th percentile. It then **ranks "within each parent
  index"**: "the S&P MidCap 400 constituent with the highest Value Score would have a Value Rank of
  1 ... the lowest ... 400".
- **Barra USE4** (Menchero, Orr & Wang 2011) takes the other road. It standardises each descriptor
  across the whole estimation universe ("cap-weighted mean ... equal-weighted standard deviation")
  and handles size explicitly with **Size and Non-Linear Size factors** and industry factors. A risk
  model can do that because it estimates everything jointly; a percentile screener cannot.

**Literature:** FF 2008 and JKP form their breakpoints within size groups, or on non-micro stocks
only, for the same reason. **Practice and literature agree here.** Ranking across size bands
without an explicit size control mixes the factor with size.

### What would change in this repo

1. **A `size_band` column** set at fetch time. In phases 2-3 it is the S&P parent index (500 / 400 /
   600). In phase 4 it is NYSE breakpoints: above the 80th percentile, 50th-80th, and 20th-50th (the
   JKP mega / large / small groups).
2. **`compute_sector_percentiles` groups by `(Sector, size_band)`.** The S&P 500 band is today's
   universe, so its percentiles are unchanged by construction. That is the phase-1 exit test, met
   for free.
3. **The fallback scope changes.** Below `SECTOR_MIN_PEERS`, fall back to the **band-wide** rank,
   not the universe-wide one. Falling back across bands would bring back the size mixing the band
   removes. The case exists today: the S&P 400 has **6 Communication Services** members (Wikipedia
   list, 2026-11-02), against 22 in the 500 and 20 in the 600. Russell's precedent is a hierarchy
   (subsector, then sector, supersector, industry) with a five-member minimum. The ten-peer minimum
   stays.
4. **`Composite_Pct` is computed within band.**
5. **One question this note does not settle:** whether phase 3 shows a *merged* S&P 1500 list.
   Band-relative composites share a 0-100 scale, so a merged sort means "best within its band", not
   "best of the 1,500". Showing three band rankings side by side is more honest; a merged view needs
   that label.

---

## 4. Data coverage: analysts thin out, and what that does to the score

### Measured (section 8 has the method)

| | Median analysts (Yahoo `numberOfAnalystOpinions`) | p25 | No count, or < 3 analysts | With a price target |
|---|---|---|---|---|
| S&P 500 (run `1957c294714c`, all 503) | **20** | 15 | 5 of 503 (1.0%) | 499 of 503 |
| S&P 400 (random 40) | **11** | 9 | 0 of 40 | 40 of 40 |
| S&P 600 (random 40) | **6** | 4.25 | 4 of 40 (10%; Wilson 95% 4.0-23.1%) | 38 of 40 |

12 of the 40 S&P 600 names (30%; 18.1-45.4%) have fewer than five analysts. As a cross-check,
Alger reported about 22 analysts for a typical S&P 500 company against 7 for the S&P 600 (citing
FactSet, c. 2019). *That figure is second-hand: the PDFs now return 404.* Yahoo's count is
recommendations, not the I/B/E/S growth-estimate contributors Russell counts, so the two are close
but not the same.

### Literature: neglect is information, not a score

**Hong, Lim & Stein (2000), "Bad News Travels Slowly: Size, Analyst Coverage, and the Profitability
of Momentum Strategies", *Journal of Finance* 55(1), 265-295.** All US stocks, 1980-96.

- **Size:** momentum profits fall as firm size rises and are "effectively zero for the largest
  firms". In the smallest decile, momentum is negative because of thin trading.
- **Coverage, holding size fixed:** momentum earns 1.13% a month with low residual analyst coverage
  against 0.72% with high, mostly on the loser side. The mechanism is slow diffusion of firm news
  where few analysts look.

*Figures from the NBER Digest summary. The NBER scan has no text layer, so the paper's tables were
not read.*

Israel & Moskowitz (2013, *JFE* 108(2), 275-301; abstract via RePEc/AQR): the value premium shrinks
with size, while momentum shows "no reliable relation with size". **The implication for the
screener:** a thinly covered stock's missing estimates mean the market may be slower to price its
news. That is a reason to treat its Momentum and Revisions readings with care, not a number to put
in the score.

### Practice: what to do when the estimates are not there

- **FTSE Russell** (rule 8.10.1) substitutes the subsector mean when the I/B/E/S growth figure is
  missing. With **one analyst** it scores 1/3 the stock's own value and 2/3 the subsector's; with
  **two**, 2/3 and 1/3; with **three or more**, 100% its own. This is a documented shrinkage
  schedule for thin coverage.
- **MSCI** (Value and Growth, section 2.3.1) drops a missing variable from the average: "only
  available variable z-scores are used". That is what this repo does.
- **S&P Style** sets a missing factor to zero before standardising.
- **Barra USE4** leaves partial data alone. When *every* descriptor of a factor is missing, it
  regresses the stock's exposure on industry and size: stocks "with similar attributes ... are
  likely to share similar exposures".

### The concrete consequence here

As of this config, 27 metrics carry weight for most stocks (24 for banks). Five need analysts:
`fy1_revision_3m`, `analyst_surprise`, `price_target_upside` and `consecutive_beat_streak`
(Revisions), plus `forward_eps_growth` (45% of Growth). Together they are **14.5 points of the
composite**: 8.65 of Revisions' 10, and 5.85 of Growth's 13. For a stock with no coverage:

- **Revisions becomes days-to-cover alone.** `compute_category_scores` renormalises over the metrics
  present, so `short_interest_ratio`'s 13.5 points become 100% of the category: 10 composite points
  on short interest. A category named for analyst revisions would score a stock no analyst covers.
- **Coverage is 22 of 27 (81.5%).** That passes the coverage filter (`coverage_minimum(27, 60)` =
  16) and sits just above the 80% line where `coverage_discount` starts. The stock is neither
  dropped nor discounted. One more missing metric costs (0.80 - 0.778) x 0.15 = 0.3% of the
  composite.
- **`forward_eps_growth` may not go missing at all.** When the consensus inputs are absent, the code
  falls back to Yahoo's `forwardEps` over GAAP trailing EPS (the `fy2_over_trailing` basis). In the
  probe, Yahoo returned a `forwardEps` for **all 40** S&P 600 names, including the two with no
  analyst count (TRST, JOE). *Where Yahoo gets a forward EPS for a stock with no analysts is not
  documented and not verified.* This is the FOX pattern from `research/2026-10-19-share-classes.md`,
  and it will occur more often.
- **The universe-wide guards do not fire.** `auto_disable_sparse_category` (30%) and
  `check_run_health`'s `MIN_TARGET_COVERAGE` (50%) look at the whole universe. On the S&P 1500 they
  would pass, while the bottom band could be thin. In phase 4 the target-coverage gate could fail an
  otherwise sound run.

Three options for the deciding session, matching the practice above:

- **(a)** Keep renormalising (MSCI-like).
- **(b)** Russell's shrinkage: a stock with fewer than three analysts has its analyst metrics pulled
  toward the sector-band median (50) by 2/3 or 1/3.
- **(c)** A minimum share of category weight before a category is scored. For example, if under
  50% of Revisions' weight is present, the category is NaN and `compute_composite`'s existing
  category renormalisation spreads its 10 points over the other seven.

(c) is the easiest to explain ("no analysts, no analyst score"), and (b) follows the most specific
published practice. **Either one changes the S&P 500 for the ~5 names with no coverage today.** So
it is a methodology change with its own changelog entry, decided on its own, not a side effect of
phase 3.

*Measured the same evening on the published 2026-10-30 payload (499 stocks):* **one** stock's
Revisions category rests on `short_interest_ratio` alone - Loews (L, rank 146, Revisions 31.3, no
analyst count) - and **six** more are missing at least one of the four analyst metrics but not all.
So in the S&P 500 today option (c) would move one stock; options (a)-(c) only diverge materially
below the large caps, which is why the choice belongs to phase 2's measurements rather than to a
change made now.

---

## 5. Which research notes stop applying below the large caps

Each was researched for the S&P 500, or measured on it. "Stops applying" means its *conclusion*
cannot be carried down without new reading. The note itself stays true for the S&P 500.

| Note | Why it does not carry |
|---|---|
| `2026-08-31-size-factor-in-a-large-cap-universe.md` | Built on "a large-cap-only universe". Its point that the S&P 500's earnings screen weakens the small-is-junk mechanism carries to the S&P 1500 but **not** to a floor-based universe |
| `2026-10-20-size-weight.md` | The 5% rests on there being no premium among large companies plus a conditional quality case. FF 2008 finds no size slope among *small* stocks either (-0.03, t -0.37), so its conclusion may survive, but within each band it is untested |
| `2026-10-12-value-and-momentum-among-large-companies.md` | Explicitly large-company. Israel & Moskowitz find the value premium larger among smaller stocks; FF 2008 finds momentum as strong among small as among big |
| `2026-10-12-factor-lab-findings.md` | The lab's reruns are big-half and largest-fifth portfolios |
| `2026-10-20-lab-jkp.md` | Reads JKP's mega and large groups only; the small group's rate (85.7%) differs |
| `2026-10-19-lab-asset-growth.md` | "Among large companies since 2003 it has earned about nothing". FF 2008: the anomaly is present among small stocks (-0.31%/mo, t -2.43) and absent among big. The Investment category's case is **stronger** below the large caps |
| `2026-10-19-revenue-growth-vs-asset-growth.md` | Its own table shows small companies earning 2.27x the big-company figure; its large-company conditioning does not transfer |
| `2026-10-20-lab-osap.md` | Piotroski's spread is largest among small firms (0.270 against 0.152 among the largest third). The F-score's role in Quality may deserve more weight below large caps (a question, not a proposal) |
| `2026-10-12-growth-category.md`, `2026-11-02-growth-exposure-drift.md` | The 13.5-point neutral point and the drift are measured on the S&P 500 cross-section; a band's neutral point must be measured on that band |
| `2026-09-07-revisions-category-has-no-revisions.md` | Martineau's "no drift in this universe since 2006" is a large-cap result (the note itself records the drift as dead "outside microcaps"); its 97.6% coverage and its section 8.4 spanning budget are S&P 500 measurements |
| `2026-10-26-price-target-upside.md` | "71% price" and the spanning test are S&P 500 measurements, where median coverage is 20 analysts, not 6 |
| `2026-10-05-fundamental-reporting-lag.md` | Rests on large accelerated filers (float >= $700M: 10-K in 60 days, 10-Q in 40). Much of the S&P 600 sits below $700M float, where the SEC's deadlines are longer (75 days for an accelerated filer's 10-K, 90 for a non-accelerated filer's - the standard Form 10-K schedule, *not re-read for this note*) |
| `2026-10-09-insider-purchases.md` | Its context sentence is calibrated to large firms, where Lakonishok & Lee find the signal near zero and the small-firm effect much larger. The panel's wording would understate it for small caps |
| `2026-10-09-short-term-reversal.md` | The "partly reverse" sentence is tuned to "the largest US companies, today", and the note's own sources (Nagel 2012) treat reversal as a return to supplying liquidity - a quantity that differs by size; HXZ list the one-month reversal among anomalies that fail once microcaps are controlled. Re-read for small caps before the sentence appears on their pages |
| `2026-10-12-option-implied-signals.md`, `2026-10-14-volatility-skew-construction.md` | Option coverage at `MAX_REL_SPREAD` was measured on S&P 500 chains; small-cap chains are thinner (not measured) |
| `2026-09-14-sell-discipline-and-hold-bands.md`, `2026-09-21-position-sizing-and-how-much.md` | Turnover-cost arithmetic and the hold-band rule assume large-cap trading costs |
| `2026-10-09-bank-like-financials.md` | Validated on the S&P 500's 59 bank-like names. The S&P 600 has **118** Financials (Wikipedia list), many of them regional banks, and the default-to-bank rule will fire far more often |
| `2026-10-14-yahoo-vs-sec.md` | Agreement rates measured on S&P 500 filers; small-filer tagging is untested |

`2026-10-09-momentum-regime.md` carries over unchanged: the panic rule reads the market, not the
universe. The research engine and backtest-v2 notes stay S&P 500 by the plan's own scope.

---

## 6. What this does not settle

- **Category weights by band.** FF 2008 and Israel & Moskowitz point to investment and value
  mattering *more* among smaller stocks, and momentum about the same. But a different weight table
  per band makes the tool harder to explain, and no note here argues for one. Phase 3 should
  publish the S&P 400 and 600 bands **on the S&P 500's weights, saying so plainly**. Any band-specific
  weight is a separate Monday.
- **Merged or side-by-side** S&P 1500 ranking (section 3, point 5).
- **Which coverage rule (section 4: a, b or c).** All three change ~5 S&P 500 names.
- **A liquidity floor for phase 4.** Not proposed. Measure dollar volume above the $1.2B floor in
  phase 4 first; add an MSCI-style traded-value ratio only if illiquid names appear.
- **Point-in-time history for the 400/600.** None is free, so nothing here can be backtested even
  after 2027-02-11.
- **Two unresolved figures.** (1) FF 2008 quotes $2.3B as the end-June 2005 NYSE median. The
  current breakpoint file gives $1.79B for June 2005 (its 20th percentile, $604M, matches FF's
  $610M). This is probably a CRSP vintage difference; not resolved. (2) Whether S&P's
  250,000-shares-a-month rule survived the 2023 liquidity consultation.
- **The probe is a sample.** 40 names per index, from Yahoo's recommendation count. Phase 2's full
  shadow run replaces it.

---

## 7. Recommended phase 1-2 adjustments

**Three pre-registrations, for the deciding session to adopt or refuse before phase 2 runs:**

1. Universe order: S&P 1500 (phases 2-3), then the all-listed universe above the floor (phase 4).
2. Floor (phase 4 only): NYSE 20th-percentile breakpoint, June reading, held twelve months
   ($1,224.13M for July 2026-June 2027); plus Russell's exclusions (price >= $1, US exchange, no
   SPACs, LPs, trusts, closed-end funds or ETFs).
3. Percentiles within `(Sector, size_band)`, with band-wide fallback below ten peers.

**Phase 1 (engineering, no published change) - additions to the plan's list:**

- **`config.yaml` `universe.min_market_cap: 2e9` must become per-universe.**
  `apply_universe_filters` drops everything under it after the fetch. On an S&P 600 run it would
  silently remove every member below $2B; the index admits from $1.2B, and existing members can be
  smaller still. Keep 2e9 for the S&P 500 (byte-identical) and set 0 for S&P index bands, where
  index rules govern. Phase 4 uses the floor above.
- Add `size_band` and the grouped percentiles above. **Exit check:** S&P 500 percentiles unchanged,
  which holds by construction.
- **Constants from the plan's "about 500" list that should become size-band-aware rules, not just
  universe-size parameters:**
  - `check_run_health.MIN_TARGET_COVERAGE` (0.50) and
    `run_screener.CATEGORY_AUTO_DISABLE_PCT` (30) should be evaluated **per band**. A universe-wide
    average hides a thin bottom band (section 4).
  - The input gates `RANK_MIN_SPEARMAN` (0.90), `RANK_MAX_NEW_IN_TOP` (8) and the top-25/top-40
    rules were set by replaying S&P 500 pages
    (`research/measurements/2026-10-22-input-gates-replay.py`). They must be **re-derived per
    universe from that universe's own replay**, never copied. Small-cap ranks will move more.
  - `universe_history.MIN/MAX_PLAUSIBLE_MEMBERS` (495/515) become per index: about 400 and about
    600 with the same small tolerance.
  - `SECTOR_MIN_PEERS` (10) stays; its fallback scope changes.
  - `MIN_SCORED`, `$universe = 500`, `MIN_UNIVERSE` and `N = ... || 500` are universe *counts*
    and can come straight from the `universe:` block.
  - `MIN_CATEGORY_COVERAGE` (0.90) can stay universe-wide: it guards a feed failure, not a band.

**Phase 2 (shadow run) - measure, in addition to the plan's list:**

- Per band: the analyst-count distribution; the share of stocks whose Revisions score would rest on
  `short_interest_ratio` alone; how often `forward_eps_growth` takes the `fy2_over_trailing`
  fallback.
- How many S&P 600 members sit below the NYSE 20th percentile.
- Sector-band groups under ten peers (S&P 400 Communication Services is known).
- Share of names below $700M float (the reporting-lag premise).
- Per-band `sec_crosscheck` counts.
- Day-to-day rank continuity per band, as the input to re-deriving the bot-style gates for this
  universe.

**Before phase 3 publishes anything:** the context-panel sentences tuned to large companies
(insider purchases, short-term reversal) need band-aware wording or must be withheld on small-cap
pages. The methodology page must say which evidence was gathered on which universe (the plan
already requires this).

---

## 8. Measurements made for this note

- **S&P 500 analyst counts:** `runs/1957c294714c/00_raw_fetch.parquet` (the 2026-11-02 02:00 run),
  column `numberOfAnalystOpinions`. 503 rows, 499 with a count; median 20, p25 15, p10 12; 4
  missing and 1 below three.
- **S&P 400/600 probe:**
  - Constituent lists from Wikipedia ("List of S&P 400 companies" / "List of S&P 600 companies",
    fetched 2026-11-02; 400 and 603 rows).
  - 40 tickers drawn from each list after shuffling with `random.Random(20261102)`. One
    `yfinance` `.info` call per ticker, sequential, with a 1-second pace (80 requests at about
    19:15 local, far from the 02:00 loop).
  - Fields read: `numberOfAnalystOpinions`, `targetMeanPrice`, `forwardEps`, `marketCap`.
  - The script and JSON live in the session scratchpad and were not committed, because this note
    is the only file this session writes. The description above is enough to rerun it.
- **NYSE breakpoints:** Ken French's `ME_Breakpoints_CSV.zip` (built from the 202609 CRSP file),
  June rows 2020-2026.
- **Sector counts** for the 400/600: the same Wikipedia lists' GICS Sector column; for the 500,
  `sp500_tickers.json`.
- **Weighted metric sets:** `factor_engine.weighted_metric_sets(load_config())` = 27 generic and 24
  bank; `coverage_minimum(27, 60)` = 16.

---

## Sources

- Fama, E.F. & French, K.R. (2008). Dissecting Anomalies. *Journal of Finance* 63(4), 1653-1678.
  doi:10.1111/j.1540-6261.2008.01371.x. Read from
  https://www.johnhcochrane.com/s/fama_french_dissecting_anomalies_JF.pdf (pp. 1654-1655, 1657,
  1663, 1668-1669, Table IV).
- Hou, K., Xue, C. & Zhang, L. (2020). Replicating Anomalies. *Review of Financial Studies* 33(5),
  2019-2133. doi:10.1093/rfs/hhy131. Abstract via
  https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2019-2133..html; full text of NBER Working Paper
  23394 (2017), https://www.nber.org/papers/w23394 (abstract, section 3.1, Table 2).
- Jensen, T.I., Kelly, B. & Pedersen, L.H. (2023). Is There a Replication Crisis in Finance?
  *Journal of Finance* 78(5). doi:10.1111/jofi.13249. Read from the copy in `data/lab/jkp/`
  (pp. 2491-2492, Figure 5; Figure 12 text).
- Hong, H., Lim, T. & Stein, J.C. (2000). Bad News Travels Slowly: Size, Analyst Coverage, and the
  Profitability of Momentum Strategies. *Journal of Finance* 55(1), 265-295. NBER w6553; figures
  via https://www.nber.org/digest/nov98/bad-news-travels-slowly (the paper's tables not read).
- Israel, R. & Moskowitz, T.J. (2013). The role of shorting, firm size, and time on market
  anomalies. *Journal of Financial Economics* 108(2), 275-301. Abstract via
  https://ideas.repec.org/a/eee/jfinec/v108y2013i2p275-301.html.
- Asness, C., Frazzini, A., Israel, R., Moskowitz, T.J. & Pedersen, L.H. (2018). Size matters, if
  you control your junk. *Journal of Financial Economics* 129(3), 479-509 (as read in
  `research/2026-10-20-size-weight.md`).
- Menchero, J., Orr, D.J. & Wang, J. (2011). *The Barra US Equity Model (USE4) Methodology Notes*.
  MSCI. https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf
  (section 2.2-2.3, pp. 8-9).
- S&P Dow Jones Indices. *S&P U.S. Indices Methodology*, December 2022 text, "Eligibility
  Criteria" and Appendix A. Read from
  https://www.betashares.com.au/wp-content/uploads/2016/10/methodology-sp-us-indices.pdf; the
  current text is at https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf
  (403 to this session).
- S&P Dow Jones Indices press releases: "Update to S&P Composite 1500 Market Cap Guidelines and
  Results of ... Consultation on Market Capitalization and Liquidity Eligibility Criteria"
  (2023-01-04); "Update to S&P Composite 1500 Market Cap Guidelines" (2025-07-01); "Consultation on
  Treatment of MegaCap Companies - Results" (2026-06-04). https://press.spglobal.com/
- S&P Dow Jones Indices. *S&P U.S. Style Indices Methodology*, June 2022 ("Style Scores",
  "Establishing Style Baskets"). Copy retrieved from https://cbonds.ru/download/indexdocs/indexdescrs/1306.
- S&P Dow Jones Indices research on the S&P 600 vs the Russell 2000 (Dash & Soe 2009; "A Tale of
  Two Small-Cap Benchmarks: 10 Years Later"),
  https://www.spglobal.com/spdji/en/research/article/a-tale-of-two-small-cap-benchmarks-10-years-later
  (summaries only; vendor research).
- FTSE Russell. *Russell US Equity Indexes Construction and Methodology* v7.2, August 2026
  (sections 4.2, 5.6-5.14, 6.5, 6.9-6.10, 8.2, 8.10).
  https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/russell-us-indexes-construction-and-methodology.pdf
- MSCI. *Global Investable Market Indexes Methodology Summary*, July 2026 (Table 3, size-segment
  targets). https://www.msci.com/eqb/methodology/meth_docs/MSCI_Jul26_GIMIMethod_Summary.pdf
- MSCI. *Global Investable Market Value and Growth Index Methodology*, May 2023 (sections 2.2-2.3).
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_GIMIVGMethod_May2023.pdf
- Kenneth R. French Data Library, `ME_Breakpoints_CSV.zip` (CRSP 202609).
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Alger, analyst coverage of large vs small caps (22 vs 7, citing FactSet, c. 2019), as summarised
  in search results; the source PDFs (la-francaise.com, AOM148/AOM44) now return 404 - *not
  verified*.
- Wikipedia, "List of S&P 400 companies" and "List of S&P 600 companies", accessed 2026-11-02.
- U.S. SEC, Form 10-K and Form 10-Q general instructions (filing deadlines by filer status). The
  large-accelerated figures are from `research/2026-10-05-fundamental-reporting-lag.md`; the
  accelerated and non-accelerated deadlines were not re-read for this note.
