# Dashboard inventory (as of 2026-10-07)

**Read this before changing the dashboard.** There is far more in it than a
first look suggests, and the most common failure mode will be rebuilding
something that already exists. Refresh this doc when you materially change the
layout.

Generated from `index.html` (237,952 chars) and `dashboard_data.js`.

**Refresh this file whenever you change the layout.** It went stale on
2026-08-25 because the session that shipped the time dimension could not
edit it - `.claude/` was blocked as sensitive. The plans now live in
`plan/` precisely so that cannot happen again; there is no excuse for
leaving it wrong.

## Top-level views (as rebuilt 2026-10-07)

| Section | Contents |
|---|---|
| **Top bar** | Sticky, full-width: product name, data date, jump links to every section (`goToSection()` opens a collapsed one first), Methodology. **The Refresh Data button is gone** - it opened an `EventSource` to `localhost:7720` and could only ever work on the owner's machine (`refresh_server.py` still exists for local use) |
| **Stat strip** | One bar, four cells: universe, value-trap flags, growth-trap flags, **metric coverage** (average, on the applicable basis) |
| **Top 5 Stocks** | Five cards in a grid (three across below 1100px, stacked on a phone): rank and sector, ticker and composite, the eight categories as a 4x2 aligned strip. Reads `table_data` directly, excluding trap-flagged names |
| **Full Universe Rankings** | The workhorse. Scrolls **with the page**; header row sticky below the top bar; **windowed rows** (only what is on screen plus a margin is in the DOM; `renderWindow()`, fixed `--row-h`); score cells tinted by value; designed filters (search with `/` shortcut, sector, trap flags, composite minimum, clear); flags as words, blank when none; phone: each row is a 96px card and a "Sort by" select replaces the header. **Section order is pinned by tests: Top 5, What Changed, then this** |
| **My Holdings** | Unchanged in behaviour (every property in the section below is still pinned). The seven cited rationales moved behind "Why this panel works this way" |
| **What Changed** | Five movers each way, "Show all N" for the rest; footnote behind "How to read this" |
| **Factor Analytics** | **Where each sector scores** - a sector x category matrix (median or average, one hue, shading relative within each column; the old bar chart was flat at ~50 for every sector) and Trap Rate by Sector (one hue, value-labelled) |
| **Defensibility & Diagnostics** | Same two analyses; the correlation heatmap is one hue by magnitude with pairs above 0.7 outlined; status colour is only a dot |
| **Methodology** | Reading surface: contents rail built from its own headings, ~72-character measure |

The **stock drilldown** (`openStockDetail`) is a **side sheet** (a bottom sheet on a phone) with a
fixed identity header (ticker, company, sector, rank, composite), jump links, and a body that
scrolls. In order: **Why it ranks here** (headline, then grouped: what drives the score / what
changed / context / read with care), the score cards (composite banner + 4x2), About, Rank
History, Analyst Price Targets, Company Snapshot, Sector Peers, Data Provenance, **Score
Contribution Breakdown** (category points, the coverage-discount line when one applies, the
composite; each row opens its workings) and **The workings**.

## The workings - every number behind a score (2026-10-07)

`plan/calculation-transparency.md`. **Read `DECISIONS.md` 0.8b before changing any of this.**
For each category: a table of every metric with weight in the table this stock was scored with -
raw value, sector percentile, **weight used**, points - totalling the category score. A header note
says which table (bank, Piotroski halved, ...) and why (`weights.profile_labels`). Each metric row
opens: the **formula**, the **reported figures behind it** (`stock_detail[t].inp`, as fetched), the
nine Piotroski signals or eight Beneish indices where it is one, whether the inputs **rebuild the
value** (`inp_bad` lists the stocks where they do not), **who it was ranked against** ("Ranked 3rd
of 47 in Consumer Discretionary", sector median and quartiles, or the universe where a sector has
fewer than `sector_min_peers` values) and the registry's plain **caveat** where the code differs
from the label.

New payload keys: `weights.profiles`, `weights.profile_labels`, `weights.coverage_discount`,
`lineage`, `lineage_check`, `sector_stats`, `sector_min_peers`; per stock `wp`, `cov`, `inp`,
`pio`, `bn`, `asof`, `inp_bad`. `peers` is tickers only. All are display-only (asserted absent
from `raw`/`pct`). Payload: 1,268,733 B gzipped (was 1,278,885).

Not yet: a "download this stock's workings" CSV; equations for the 12 series-based and
provider-ratio metrics (they say so instead); the analyst-history inputs (the per-quarter EPS
actuals and estimates are not retained at fetch).

## Visual design system - stage 1 DONE 2026-10-06, applied across the page 2026-10-07

Owner directive 2026-10-05 (`OWNER_FOCUS.md`): the page read as generated and
did not feel good to use. Stage 1 replaced the *tokens*, not the structure:
`:root` in `generate_dashboard.py` now carries a neutral ramp, one accent, an
Inter-based type stack with `tabular-nums` figures and a 14px base, one radius,
named-property transitions at 120-160ms and `prefers-reduced-motion`. Deleted,
not restyled: the rainbow header rule, per-category and per-sector hue maps,
glows and the entrance animations. Every value and its source is in
`plan/dashboard-design-system.md` - change a token there and in `:root` together.

**2026-10-07** applied it surface by surface (`plan/dashboard-redesign-master.md`). Measured
with `scripts/shot_dashboard.py` (now also reports the feel numbers): **DOM nodes 9,928 ->
2,779; transitioned elements 662 -> 99; row click to paint 70 ms; sort to paint 21 ms; layout
shift 0.001; payload gzipped flat.** `tests/test_dashboard_browser.py` (25 tests, Chromium via
Playwright, skipped where unavailable) holds those budgets.

Presentation-only stages leave `dashboard_data.js` byte-identical (check with
`scripts/diff_payload.py`); 2026-10-07 added payload keys deliberately, listed above.

Still the old structure or not yet judged: the lower drilldown blocks (Rank History, Price Targets,
Company Snapshot, Peers) were re-chromed but not redesigned; the peers table still colours cells
green/red against the stock; the Holdings panel's populated state was not re-looked-at; contrast
was checked for tokens, not every pair in use.

## My Holdings - the sell-side surface (2026-09-15)

Priority 5 / north-star gap 2. A `localStorage` list under
`screener_holdings_v1` holding **tickers and nothing else**, rendered as one
card per name: rank, composite, an eight-category score-and-delta strip, and
the `change` / `change_driver` / `flags` / `confidence` sentences lifted from
`stock_detail[t]["summary"]`. Above it, a concentration line (names, sectors,
largest sector share, count inside the top 25 and top 100, trap-flag count) -
the honest half of "how much / does it fit", since the list holds no weights.

**Added 2026-09-17:** the `input_churn` sentence joined `HOLDINGS_FACTS`, and a
cadence note now sits above the concentration line (see the two sections below).

**Added 2026-09-29:** the `earnings` sentence joined `HOLDINGS_FACTS` - the next
scheduled report date, on the row of every name you own. See its own section
below; the membership list is exact and pinned by a test, so adding to it needs
a citation, not a preference.

**Added 2026-09-22:** a **Concentration block** below that line - see its own
section further down. The fit line and the block are different things and both
are needed: the line reports what the list *contains*, the block interprets the
count against published thresholds and reports the risk spread.

**Zero payload cost.** It is a *view* over fields the payload already carried;
nothing was added to `stock_detail` for it.

**Three properties are research constraints, not styling.** All three have
tests, and the sources are in `research/2026-09-14-sell-discipline-and-hold-bands.md`
and `METHODOLOGY_CHANGELOG.md` 2026-09-15:

1. **Every saved name renders, every time** - never a filtered subset.
   Akepanidtaworn et al. (2023, *JF* 78(6)) trace an 80 bp/year institutional
   selling deficit to a restricted consideration set.
2. **Ordered by rank, never by size of move.** The rank change is shown for
   context; it is not the sort key and not a filter. Same source.
3. **No cost basis, share count or P&L anywhere**, in the code or in storage -
   Odean (1998). A hand-edited key containing a position dict is read for its
   ticker and written back clean. This is also why the panel works equally as a
   watchlist.

**There is no exit rule and no hold band, deliberately - and the reason written
here until 2026-09-17 was wrong on both numbers.** Corrected per
`METHODOLOGY_CHANGELOG.md` 2026-09-16 and §8 of the research note, which is where
the working is:

- "A 25/50 band fired zero times" came from walking **one path** through 18 runs.
  Over all comparable pairs a 2x band fires at **1.65-4.50%** of holding-looks.
- "60+ comparable runs" is the **wrong unit**. Pairs from a daily series overlap
  almost completely, so 34 runs are **2 independent monthly looks**, not 34
  observations - the same independence trap as the IC series.

**What is established:** a band *is* warranted (the strict top-25 rule wastes
**31-47%** of the trades it implies, at every cadence measured), but its **width
is not determinable** - at 1.4x the three cadences report 0.0%, 31.2% and 5.9%.
**The pre-registered rule binds:** no width until **>= 8 disjoint observation
windows at the review cadence the band will govern**; today **2 monthly**,
roughly **2027-04**. Do not pick 50 because MSCI doubles, and **do not reuse the
movers panel's threshold** - measured, it fires for a top-25 name 0.15% of the
time.

The empty panel ships **collapsed**; a saved list auto-expands it.
`tests/test_holdings_panel.py`, 61 tests (60 fail against the pre-change
generator); nine drive the emitted script under Node against a stubbed DOM.

## Concentration - "how much / does it fit?" (2026-09-22)

North-star question 4, which had **no surface at all** between the Model
Portfolio's removal on 2026-08-26 and this. `holdingsConcentration(rows)`,
rendered below the fit line inside `#holdings-fit`. Sources and design:
`research/2026-09-21-position-sizing-and-how-much.md` §8.4. Zero payload cost -
`raw.volatility` was already in `stock_detail`, and the thresholds are literature
constants in a `NAME_MARKS` array in the emitted script. Verified: regenerating
left `dashboard_data.js` byte-identical.

Three lines:

1. **Name count against the published counts** - 30-40 (Statman 1987), ~50
   (Campbell et al. 2001), 63 for a 10% shortfall risk over 20 years (Domian et
   al. 2007), with a computed "below all three / above N of the three".
2. **The equal-split slice**, with the published single-position caps for scale:
   UCITS 5% (10%/40%), RIC 25/5/50, S&P Select Sector's 24% re-cap.
3. **The widest risk gap on the list**, in raw annualised volatility.

**Four things not to tidy**, all with tests in
`tests/test_holdings_concentration.py` (33 tests, all 33 fail against the
pre-change generator; 17 drive the emitted script under Node):

- **No target weight for any stock, ever.** The equal-split figure is arithmetic
  on the *length* of the list - identical for every name on it. Printing a
  per-stock weight is what got the Model Portfolio deleted, and sizing by a
  conviction score is the single most error-sensitive thing the estimation
  literature identifies (Chopra & Ziemba 1993 via Ziemba & MacLean 2011: mean
  errors ~20x covariance errors, ~100x near zero risk aversion; DeMiguel,
  Garlappi & Uppal 2009: none of 14 models consistently beat 1/N).
- **The risk line uses RAW annualised volatility, never the volatility
  percentile.** That percentile is sector-relative *and* direction-inverted, so a
  high value means "calm for its sector". Measured by
  `research/measurements/2026-09-22-holdings-risk-comparability.py`: it orders
  the pair **backwards for 23.9%** of the 111,417 cross-sector pairs, worst case
  a name reading as the safer holding while carrying **2.00x** the volatility.
  A test encodes exactly that inversion.
- **The three counts are always quoted with their condition** - all measure
  *randomly selected* portfolios, so they bound the question for a pre-screened
  large-cap list rather than settle it.
- **The caps are described as caps, not targets.** They constrain the top end
  and say nothing about distribution below it.

The risk line is omitted below two holdings with volatility (coverage is 501 of
502, so the absent case is real); the rest of the block still renders.

## Next earnings date - "is today the day to look?" (2026-09-29)

North-star gap 4, open since 2026-08-10. Three `.info` fields captured at fetch
(`earningsTimestampStart`, `earningsTimestampEnd`, `isEarningsDateEstimate`),
carried as `stock_detail[t]["earn"]` = `{"d", "end"?, "est"}`, and rendered as
an `earnings` sentence in the baked summary - so it lands on **both** the
drilldown and every My Holdings row from one implementation. Zero API cost: the
fields ride the `.info` dict the fetch already pulls. Changelog 2026-09-29;
`tests/test_earnings_date.py`, 54 tests, 49 failing against the pre-change tree.

**Why this fact.** Announcement-day sells are the **only** sells in
Akepanidtaworn et al. (2023, *JF* 78(6)) that beat their counterfactual, by
more than **+150 bp/year**, against a -80 bp/year deficit for sells overall.
The same paper is already why this panel refuses to rank by size of move; this
is its positive half. It also matters mechanically: the fundamentals categories
barely move between filings (largest one-month Quality move: **one stock in
500**), so a report is when Valuation, Quality and Growth are actually replaced.
That is why the sentence sits immediately after `confidence`.

**Five things not to tidy**, all measured across all 503 tickers on 2026-09-29
and reproducible with `research/measurements/2026-09-29-earnings-date-coverage.py`:

- **Never scored.** Tests assert the fields are absent from `METRIC_COLS`,
  `METRIC_DIR`, every weight block in `config.yaml`, and from `raw`/`pct`.
- **`est` is always emitted and never defaults to true-looking.** **209 of the
  492 future dates (42.5%)** are provider estimates, not company-announced
  schedules. This is the load-bearing part of the feature, not a footnote.
- **No countdown, no colour ramp, no badge** - the evidence says announcement
  days are when attention is well spent, not that a near report is good or bad
  news. A test asserts the prose is identical bar the day count, and another
  asserts the CSS rule carries no red, amber, bold or uppercase.
- **A past date is dropped, never relabelled.** **11 of 503** carried one.
- **`earningsTimestamp` stays uncaptured.** It equals the *next* date for 27
  tickers and is a *past* date for 56 - no label is true of every row. A test
  greps the fetcher to keep the obvious-looking field unread.

The horizon reads "36 days after this run", not "in 36 days": the summary is
baked at build time and the site rebuilds on weekdays only, so a reader-relative
phrasing would decay into a falsehood over a weekend while reading as current.

Cost: **+7.4 KB gzipped (+0.62%)** on the payload, +0.8 KB on `index.html`.
`end` is omitted when it equals the start - which it did for all 503 - so the
common case carries one date, not two.

## Review cadence, stated on the surfaces that move (2026-09-17)

`config.yaml -> portfolio.review_cadence` (a real key since 2026-09-17; it was a
bare comment before, which is why the page could not state it). Surfaced as
`D.cadence` and rendered by `cadenceText(long)` / `cadenceLine(long)` in three
places: the holdings panel above the concentration line (short form, both empty
and populated states), the What Changed footnote (short form), and the holdings
footnote (long form, with the turnover numbers and the Novy-Marx & Velikov
citation).

**The gap it closes:** the site is rebuilt every weekday and, until this landed,
said nothing about how often acting on it was intended. Acting on the strict
top-25 rule at every run implies **121.8%** monthly one-sided turnover against
**24.0%** at monthly review, where NMV (2016) find few anomalies survive costs
above **~50%**.

**It is a sentence, not a lock** - the tool does not know what a reader is doing.
Read from the *run's own* config snapshot, not the working tree, so a republished
old run states what it was configured for; `configured: false` marks the
quarterly fallback so it is distinguishable from a real setting.
`tests/test_review_cadence.py`, 40 tests (37 fail against the pre-change
generator); eleven drive the emitted script under Node.

## Input-availability churn - "is this move information?" (2026-09-17)

`history.py` records per-ticker metric availability per run and emits
`ch: [lost, gained]` on a delta entry when it changed; `stock_summary.py` turns
`>= 2` into an `input_churn` sentence, shown on the drilldown and on every
holdings row (amber left rule, `.holding-note-input_churn`).

**Why it exists:** when a metric percentile flips between present and absent, its
category renormalises over a different metric set and the score moves as
arithmetic, with no company event. That is `CLAUDE.md` priority 1.5's FCX case,
and it closes the product gap recorded there - the movers panel could not
distinguish "moved on new information" from "moved because two inputs went
missing".

**Three things not to tidy**, all with tests in `tests/test_input_churn.py`
(58 tests, 52 fail against the pre-change code):

1. **Arms at 2, not 1.** One changed metric moves the median rank by 7 against a
   baseline of 6 - noise. Two or more triples it to 21.
2. **Worded as a caveat, never as deterioration.** Churn leaves 52.4% of names
   worse off against a 44.6% base rate; it scatters ranks rather than pushing
   them down, and the sentence reads identically whether the stock rose or fell.
3. **Only columns both runs carry are compared**, and pre-2026-03-09 snapshots
   (15 columns, no percentiles) yield `None` rather than `(0, 0)` - "cannot tell"
   must not render as "nothing changed".

On the 2026-09-17 run it fires for **27 of 502** stocks against the ~1-month
baseline. Payload cost is one optional two-integer key on ~5% of delta entries.

## Why It Ranks Here - the deterministic summary (2026-09-08)

Built by `stock_summary.py` **at run time**, stored per stock as
`stock_detail[t]["summary"]` = `[{"k": kind, "t": sentence}, ...]`, rendered by
`renderSummary()` as the first block of the drilldown. **Thirteen kinds**:
`rank`, `drivers`, `weakest`, `best_inputs`, `worst_input`, `change`,
`change_driver`, `input_churn`, `target`, `peers`, `flags`, `confidence`,
`earnings`. A kind is omitted when it cannot be
stated exactly, so a thin stock gets a shorter summary rather than a hedged one.

**`change_driver` was added 2026-09-15** and says *why* a stock moved, not how
far: the category that moved furthest since the baseline, the direction of its
**score** (stated explicitly, because a high Risk score means low risk), and
what that category now contributes. It and `change` read their baseline from
one place (`_pick_comparison`) so they cannot end up describing different
windows. Measured across the live payload it names Risk 34%, Revisions 29%,
Momentum 26%, Quality 0.2% - the fundamentals categories barely move between
quarterly filings. Cost: **+10.5 KB gzipped (+0.89%)**.

**Every sentence here is now registered in `claims.py` (2026-10-07, stage T0a).**
Each of the thirteen kinds has an entry naming what it asserts about how a number
is computed, the code that makes it true, and the test that checks it. **Adding a
`_sentence_*` function fails the suite until it is registered**
(`tests/test_claims_register.py`, 24 tests, also in the data loop's publish gate).

**The `rank` sentence was false until 2026-10-07** and is the reason the register
exists. It read *"Its composite of 73.7 is a percentile: it scores above 74% of the
universe"* - for the stock ranked **1st of 502** - wrong for 493 of 502 stocks by a
median of 19.6 points. `Composite` has been cardinal since Phase 13 (F1);
`Composite_Pct` is the percentile. It now reads *"Ranks 1st of 502 - ahead of 100%
of the other 501 stocks. Its composite of 73.8 is a 0-100 score computed from its 8
category scores and their weights, not a percentile"*: the share is
`(N - rank) / (N - 1)`, and the category count is the stock's own (two stocks do not
have eight). It survived for months because **a test asserted it** - when correcting
a published claim here, grep the tests as well as the prose.

It stops short of claiming `composite == sum of contributions`, deliberately: the
coverage discount reduces it for 3 of 502 stocks and the payload does not yet carry
the coverage figure that discount reads. **T0b** adds that.

**Known defect, T0b:** `confidence`'s "The score rests on N of 18 metrics" and the
provenance badge's 60/80% colours use `factor_engine`'s hard-coded 18-metric list,
**not** the applicable-metric coverage the discount uses (35 for a bank-like stock,
41 otherwise). 62 stocks read under 80%; 3 were discounted.

**Do not move this into the browser.** Building it here is what makes it
diffable and identical for every reader, which is the entire reason it replaced
the chat.

**Do not let advice language in.** `stock_summary.BANNED_TERMS` +
`advice_terms_in()` are checked against all 502 live summaries by
`tests/test_ai_chat_removed.py`. "Explains why it ranks there, never whether to
buy" is a north-star constraint with teeth, not a style note.

**It says "sector percentile" deliberately** - the percentiles are
sector-relative (`factor_engine.compute_sector_percentiles`), and dropping the
qualifier would publish a false claim about how the number was computed.

## Percentiles are direction-adjusted, and the page says so - 2026-09-11

**Read this before touching any percentile surface.** `compute_sector_percentiles()`
does `ranks = 100 - ranks` wherever `METRIC_DIR` is `False`, so a published
percentile **always means "best in its sector", never "largest"**. That is true
of 13 of the 37 metrics in `metric_meta` - on the live payload HON's EV/EBITDA
of 6.95 is the 99th percentile and AXON's 98.61 is the 0th.

Until 2026-09-11 nothing said so, and the natural reading of the drilldown was
exactly backwards. Three things now state it, and all three must stay:

1. **`metric_meta[m]["dir"]`** - `"higher"` or `"lower"`, **derived in
   `prepare_dashboard_data()` from `factor_engine.METRIC_DIR`.** Do not
   hand-write these. The derivation is what makes it impossible for the page to
   claim a direction the scorer disagrees with, and
   `tests/test_percentile_direction.py` compares the two on every build.
2. **The drilldown metric table** - header reads `Sector Percentile - 100 =
   best`, a `.pctile-convention-note` sits under it, and `dirChip()` renders a
   `↓ better` / `↑ better` chip beside every metric name. The note's "13 of 37"
   count is computed in JS from the payload, not written as a literal.
3. **The summary prose** - `_label_and_value()` appends `, lower is better` for
   inverted metrics only. Higher-is-better metrics are deliberately left plain:
   the ambiguity only exists where percentile and raw value point opposite ways,
   and the prose is ~101 KB gzipped across 502 stocks.

Total cost measured at **+1.2 KB gzipped (+0.1%)**. No score, rank, `raw` or
`pct` value changed - verified cell-by-cell across all 502 stocks.

**The eight category columns also carry definitions now** (`title` on each
`<th>`), naming their scored metrics with weights, the bank carve-out for
Valuation/Quality, and the fact that a **high `Risk` score means low risk**.

**Name only metrics that actually carry weight.** Draft tooltips listed P/B
under Valuation and PEG under Growth; both are weight 0 for non-banks. Read
`config.yaml`'s active weights rather than `CAT_METRICS`, which includes
zero-weight candidates. A test pins this.

## The "Screener AI" chat is gone - DONE 2026-09-08

Owner directive 2026-08-10, priority 4, open 29 days. Removed: the chat FAB and
panel, the Chat Settings dialog (API-key field + model picker), 27 JS functions,
three keyframe blocks and the `AI CHAT PANEL` stylesheet - 891 lines, and 921
lines off `generate_dashboard.py`. The `config_traps` payload key went with it:
its only consumer was the chat's system prompt, and the same thresholds are
already in the Methodology section.

It required each visitor to paste an Anthropic API key into `localStorage` and
called `api.anthropic.com` from the browser - unusable for a student club,
costly per question, a credential-phishing-shaped form on a public page, and
un-reproducible, which is the one that decided it. Changelog 2026-09-08.

`tests/test_ai_chat_removed.py` (75 tests, 58 failing against the pre-change
generator) pins **both halves**: no chat symbol, element id, model id,
`localStorage` key or provider URL survives, *and* the summary block renders. A
partial swap is the dangerous state - a dangling identifier blanks the page with
all four ship gates green.

## The Methodology section is roughly half the file

It embeds a full document: What Is This, Where Does the Data Come From, all 8
factor categories with weights, Bank-Specific Scoring, a 6-step score
calculation walkthrough, Piotroski Conditional Weighting, Data Quality
Safeguards, Value/Growth Trap Detection, Portfolio Construction, What Gets
Output (all 6 Excel sheets), Factor-Exposure Diagnostics, Reproducibility,
Defensibility & Transparency Features, Key Design Decisions, Limitations,
Quick Start, Summary.

This is **an asset for the investment-club audience and the main reason the
tool is defensible** - do not delete it.

**Correction, 2026-08-28: it is already generated from config.** The previous
version of this section warned that the weights were "hardcoded into the
prose" and would go stale. They are not: `run_screener.generate_screener_overview(cfg)`
templates the whole document out of `config.yaml` on every run, and all 8
category weights and ~40 metric weights were checked against `config.yaml`
that morning and matched exactly. **Edit the generator, not the file**
(rule 10).

The real failure was one level down, and worse: the document was faithful to
`config.yaml` while the *screener* was not. A run's momentum weight is scaled
by the volatility regime, so the composite was built at 14.95% while every
surface printed 13%. See below and `METHODOLOGY_CHANGELOG.md` 2026-08-28.

## Payload weight

| Key | Size (MB) | Notes |
|---|---|---|
| `stock_detail` | 2.81 -> ~4.3 | **~88% of the payload.** All 502 stocks. Grew 2026-08-26 with `about`, 2026-09-08 with `summary` (+0.67 raw), 2026-09-15 with `change_driver` (+0.10 raw / +10.5 KB gz) |
| `history` | 0.27 | Added 2026-08-25. 18 accepted run dates, 2 excluded |
| `table_data` | 0.26 | 502 rows, 8 category scores + composite/rank/flags |
| everything else | <0.02 | `portfolio` (0.010) and `spx_weights` removed 2026-08-26 |

**Raw size is the wrong number to optimise.** Pages serves gzip, and the
payload compresses ~4x overall. The business descriptions add ~0.71 MB raw but
only ~60 KB on the wire. Measure gzip before calling anything expensive.

**But do measure.** The 2026-09-08 summaries compress only **6.6x** (665 KB raw
-> 101 KB gzipped), well short of the ~11x the earlier prose achieved, because
every stock's sentences carry different numbers and gzip's 32 KB window cannot
match far back. Wire payload went **1,078 -> 1,179 KB**; removing the chat gave
back 11 KB of page, so the net was **+90 KB (+8%)**. That was judged worth it -
see `plan/dashboard-north-star.md` - but it is the largest single addition since
`about`, and the next thing added should be weighed against a phone on 4G.

`stock_detail` dominates. Per stock: `raw`, `pct`, `cat_scores`, `contrib`,
`composite`, `rank`, `sector`, `company`, `industry`, `about`, `summary`,
`vt`/`gt`, `price`, `pt_mean/high/low`, `num_analysts`, `eps_mismatch`,
`eps_ratio`, `data_source`, `metric_count`/`metric_total`, `financials`,
`flags`, `peers`, `self_metrics`.

**`raw` is the value as fetched, from 2026-09-01.** Until then the pipeline
winsorized every metric at the 1st/99th percentiles immediately before ranking
it, and `raw` carried the *clipped* number — so the site published AAPL, NVDA,
MSFT, GOOG, GOOGL and AMZN with one identical market cap of $2,802.0B against a
true $5,331.2B for NVDA. 301 cells across 33 metrics on 159 stocks were wrong
the same way. Clipping could never have helped, because `pct` is a rank and a
rank is invariant under monotone transforms. Removed; changelog 2026-09-01,
`tests/test_no_winsorization.py`. If a metric ever again shows a cluster of
identical values at its extremes, that is the defect returning.

`industry` and `about` were added 2026-08-26 (owner request). Both are
**display-only** - `about` is the provider's `longBusinessSummary`, rendered
verbatim in a clamped block under the score cards with a "Show more" toggle and
an attribution line. Neither is scored, ranked, or fed to a metric, and
`tests/test_dashboard_surfaces.py` asserts they never appear in `raw`/`pct`.
Both ride the `.info` dict the fetch already pulls, so they cost no API calls.

Adding history will grow this fast. Lazy-load or downsample - do not ship a
10 MB payload to a phone.

## Weights: what the drilldown shows - DONE 2026-08-28

The stock drilldown shows **per-stock effective weights**, not the configured
defaults. Three surfaces read them (`effWeights()` in the emitted JS): the
score cards, the contribution bars, and the category-detail badges.

Two things move a weight away from the Methodology page's number, and both are
now stated on the page by `weightNote()`:

1. **The volatility-regime adjustment**, run-level. Baked into
   `weights.factor_weights`, with `weights.base_factor_weights` kept alongside
   so the page can show `13% -> 15.0%` rather than just asserting 15.0%.
2. **Per-stock renormalisation**, when a category could not be scored. The
   category keeps its row, marked "no data", instead of disappearing - hiding
   it would leave the reader unable to see why the rest add to more than the
   defaults.

**Do not revert these to `D.weights.factor_weights[c]`.** That is the bug:
between February and 2026-08-28 the page printed `Score x 13% = 9.76 pts`,
which is false, for 498 of 502 stocks. `prepare_dashboard_data()` now
reconciles the recorded weights against the published contributions on every
build and will not publish weights that fail to reproduce them.
`tests/test_weight_transparency.py`, 34 tests.

## Other payload keys

`kpis`, `weights` (factor + metric, plus `base_factor_weights` /
`factor_weights_adjusted` / `factor_weights_derived` since 2026-08-28),
`metric_meta` (36 metrics),
`sectors` (11), `sector_composition`, `histogram`, `vt_by_sector`,
`gt_by_sector`, `sector_distributions`, `factor_correlation`,
`weight_sensitivity` (8), `data_quality`, `history`.

`portfolio` and `spx_weights` were removed 2026-08-26 - see below.
`config_traps` was removed 2026-09-08 with the chat that was its only consumer.

## What is genuinely missing

Confirmed against the above, not guessed:

1. ~~Any time dimension.~~ **SHIPPED 2026-08-25.** `history.py` builds a
   quality-gated spine from `improvement/snapshots/`, surfaced as: a
   **What Changed** panel under the KPI row (biggest movers each way, inline
   sparklines, category that moved most), a sortable **delta column** in the
   universe table, and a **Rank History** block in each stock's drill-down.
   Payload key `history` (~0.25 MB): `dates`, `series`, `delta`, `movers`,
   `noise`, `compare`, `excluded`, `available`.

   Two things not to undo. **Runs enter the history only if their ranking
   correlates with the previous accepted run at Spearman >= 0.50** - the
   `2026-07-28` degraded run correlates at 0.016/-0.020 and, ungated, reports
   82% of the universe as material movers. And **the default comparison is the
   ~1-month window, not the previous run**: measured on this repo's snapshots,
   every material one-day mover on 2026-08-25 was a round-trip, while 169 of
   193 one-month moves were genuine trends.
2. ~~Any sell-side workflow.~~ **MOSTLY SHIPPED 2026-09-15** - see the My
   Holdings section above. What remains is the **hold band**: the screener
   still has one test (top 25) where the evidence says entry and continued
   holding should use different, asymmetric tests. Blocked on measurement, not
   on design - re-measure the band at 60+ comparable runs (32 today).
3. **Time-series valuation context.** `pct` is cross-sectional only.
4. ~~Catalyst/earnings-date proximity.~~ **SHIPPED 2026-09-29** - see the
   "Next earnings date" section above. What is *still* missing on this axis is
   a **run-level** view of it: nothing answers "which of my candidates report
   this week" without opening names one at a time. 8 of 503 were inside seven
   days on the ship date, so the set is small enough to be useful and the data
   is already in the payload.
5. ~~Per-stock confidence surfaced.~~ **MOSTLY SHIPPED 2026-09-08.** The
   summary's `confidence` and `target` sentences state metric coverage ("rests
   on 12 of 18 metrics"), name any withheld category and the reweighting it
   caused, flag stale filings with their age, flag an EPS-basis mismatch, and
   call out a thin analyst base (<5). What is still not legible anywhere is
   `data_source` (quarterly vs annual).
6. **Charting breadth.** Three charts for a 3 MB payload is thin - though add
   charts only where they beat a table, not for decoration.

## Model Portfolio removal - DONE 2026-08-26

Owner directive 2026-08-05, restated as a priority 2026-08-26 and shipped the
same evening. The recommendation this file made was followed exactly: **the
dashboard surface went, the construction engine stayed.**

**Removed:** the `Model Portfolio` section, the `Portfolio Sector Allocation vs
S&P 500` chart, `renderPortfolio()`, `renderSectorAlloc()`, the `portfolio`
payload key, and the `spx_weights` key - the latter because that chart was its
only consumer, and a sector split of the S&P 500 against itself says nothing.

**Kept, deliberately:** `portfolio_constructor.py`, the `08_model_portfolio`
artifact, the `ModelPortfolio` Excel sheet, and the `in_portfolio` snapshot
column. The warning in the previous version of this section was correct -
`improvement_engine.record_run_snapshot()` writes `in_portfolio` into every
snapshot and computes **turnover** from it, so deleting construction outright
would have silently damaged the evidence base. Three test modules cover it too.

**Top 5 was checked, not assumed.** It had been reading the portfolio holdings.
It now filters `table_data` for trap-free names and sorts by rank. Verified
identical on live data (`HST, EXPE, APA, EIX, CF` both ways): the sector cap is
8-of-25 and cannot bind on five rows.

Pinned by `tests/test_dashboard_surfaces.py` (30 tests, 29 of which fail
against the pre-removal code). A *partial* removal is the dangerous state -
`D.portfolio` undefined at render time takes the whole script down and the page
goes blank with every ship gate still green.

If the owner later wants the construction engine gone too, the open questions
are whether turnover is actually used by anything and whether the Excel sheet
is still wanted.
