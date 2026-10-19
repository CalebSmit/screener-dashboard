# Investment-club readiness - a first-year student on a phone

**CLAUDE.md priority 7: "can a student open this on a phone and understand what they're looking at?"**
First pass 2026-10-19. Function and comprehension only: the design pass and the function pass of
2026-10-15/16 (`plan/dashboard-investor-pass.md`) are not undone or restyled. No number, rank or
sentence in the payload changed: every regeneration was presentation-only and checked
`dashboard_data.js` and `dashboard_context.js` byte-identical by sha256.

## Method

Headless Chromium (Playwright) at **375x812 with mobile emulation** (`is_mobile`, touch, so
`(pointer: coarse)` matches), the worktree served on `localhost`, the 2026-10-19 02:00 run
(`377ec40838b1`, 501 stocks). Each task was walked as a first-year finance student who knows
what a P/E is and has never seen a factor screen, with screenshots of every screen and the text
of every surface read in full. The same surfaces were then checked at 1440x900. Horizontal scroll
was measured (`scrollWidth` against `innerWidth`) on every screen; console errors were captured.

## The walkthrough

### 1. What is this, and should I trust it?

- **Works.** The first screen states the product ("The S&P 500, ranked on eight factors"), that
  every score traces to its inputs, and "For research and education. Not investment advice." The
  two proof pages are buttons on the first screen. The Track record tile says "8-month record".
- **Confusing: "Trap flags 44 / 62".** The fourth tile on the first screen. Nothing on the phone
  said what a trap flag is: the table's column header has a tooltip ("see Methodology"), but a
  phone hides the header and never shows a tooltip. **Fixed** (below: the key, reached from a
  "What a flag means" link on the tile).
- **Confusing: "Moved materially 107 / 105, rank moves over ±37".** Readable once you know ±37 is
  a measured noise threshold; the tile does not say so. Partly fixed: the key's ▲▼ line says moves
  under 37 places are within ordinary run-to-run variation.
- **Jargon on the tiles further down**: "10y-3m curve +0.83pp", "option-implied moves", "101 EPS
  flags", "Stability moderate (79%)". The Market backdrop's own text used "pp" throughout without
  saying what it is. **Fixed** for "pp" and "FRED" (one line in the section's opening). The rest is
  explained inside each section when opened (Diagnostics names Jaccard similarity and Spearman
  correlation and says what 1.0 and 0.7 mean); left.

### 2. The top-ranked stocks, and why one ranks there

- **Works.** "Leading the ranking" is a swipe row of five cards; tapping one opens the stock sheet,
  which opens on the eight scores, then "Why it ranks here" in plain sentences ("Ranks 1st of 501,
  ahead of 100% of the other 500 stocks").
- **Confusing: which way does a score run?** EXPE, ranked 1st, shows "Risk 21". JPM shows "Size 2".
  A student reads both as bad. A high Risk score means *low* risk, and a high Size score means a
  *smaller* company, so JPM's 2 says it is the largest bank, not that something is wrong. The
  desktop table's header tooltips said this; nothing on a phone did, and the sheet's score cards
  said only "20.5 / 2.0 pts (10% wt)". **Fixed**: one line under the sheet's eight scores, the
  guide's first statement, and the key.
- **Confusing: the abbreviations** Val, Qual, Grow, Mom, Rev, Inv on the leader cards and the table
  cards. **Fixed** by the key, which pairs each with its name.

### 3. Look up a stock I own

- **AAPL** - the search icon (top right) opens the palette; "apple" or "aapl" finds it. Works.
- **A bank** - "bank" finds BAC, MTB and BNY by name (JPMorgan does not contain "bank"; "jpm" or
  "financials" finds it). JPM's sheet reads well; its bank weighting is stated in The workings.
- **A stock that left the index (WBA)** - **dead end in three places.** The palette said "No stock
  or section matches. Try a ticker (AAPL)" - as if the student had mistyped. The table's search
  said "No stocks match these filters. Try a wider sector or a lower composite minimum." My
  holdings' add box said "No matches". **Fixed**: all three now say "WBA is not in this ranking.
  The ranking covers the 501 S&P 500 companies this run scored. A company that has left the index,
  or whose data could not be fetched, is not in it." A name fragment, or a scored ticker hidden by
  another filter, keeps the old message.
- **Holdings** - each card's rank change read "▼ 32 ranks since 2026-09-18" beside "Sep 18"
  everywhere else. **Fixed** to the page's date format.

### 4. Compare two stocks

- **Works**: Compare in the sheet's toolbar, a tray at the bottom, "Side by side", and "Where the
  composite gap comes from" adds the category differences up to the gap.
- **Phone layout hid part of it.** With just two stocks the table ran **52px** past a 375px dialog:
  "JPMorgan Chase & Co." was held on one line and pushed the second column's bars off screen, with
  no cue that it scrolled. **Fixed**: below 420px the company name wraps and the score cells are
  tighter; two stocks fit (313px of 313). Three or four still scroll sideways, with the row labels
  pinned, as designed.
- Left: the tray's chips scroll inside a 97px strip, so the second ticker is half-visible beside
  "2 stocks". The count is stated; low value.

### 5. What does a score of 62 mean, and what are the eight categories?

- The guide (first visit only) says each category is 0-100 against sector peers with the median
  near 50, and the composite is a weighted sum. The sheet's first sentence gives the rank. Nothing
  said what a composite of 62 is relative to the other 500. **Fixed**: the key's Composite line
  quotes the run's median composite (the engine's own `kpis.median_composite`, 50.2 today) - so a
  62 is in the upper half - and sends the reader to the stock for its exact rank.
- Where to find all eight explained on a phone: before this pass, only the Methodology document.
  **Fixed**: the key ("What the columns mean") above the ranking, one line per column.

### 6. Methodology and "Has it been tested?"

- "How it works" (hero) opens Methodology; "Has it been tested?" is a hero button and a footer link.
- **Methodology opens on jargon.** Its first paragraph is this run's momentum-rule reading
  ("...The panic state needs a negative return and volatility above 20.6%...") before the Overview,
  and the Overview names `METRIC_COLS`. Left: the document is generated by
  `run_screener.build_screener_overview()` with its own claims tests; a reordering is a session's
  work of its own (see Open, below).
- **"Has it been tested?" gave no scale.** It opens on "average monthly rank correlation ... -0.003",
  then t-statistics and "5 basis points of cost a side". **Fixed**: a "Reading the numbers" line under
  the three answers says what a rank correlation of +1 and 0 mean, what a t-statistic is and what a
  basis point is (generated by `research_page.py`; `tests/test_research_page.py` holds the committed
  page to the ledger).
- **Reaching Methodology from deep in the page on a phone.** Below 400px the header's Methodology
  button is hidden, and the header does not stay on screen on a phone. From row 300 of the ranking
  the ways out are scrolling back up or to the footer past the remaining cards. Left (see Open).

### Tap targets and gestures

Measured at 375 with `(pointer: coarse)` matching, opened key and an open sheet: no control under
40px except text links inside sentences - the brand (24px, also reachable as the page top), the
guide's "What each column means", the key's "Methodology", and the insider-trade filing dates in
Before you decide. Inline links in a sentence are the exception WCAG 2.5.8 allows. No horizontal
scroll at 375 or 1440 on the page, an open key, an open sheet, the comparison, `research.html` or
`lab.html`. No console errors.

## What changed (all display-only; payload byte-identical)

| Change | Where | Test |
|---|---|---|
| **"What the columns mean"** key above the full ranking (a `why-panel` disclosure, closed by default, every width): the composite with the run's median, each category's direction (from `generate_dashboard.SCORE_KEY`), ▲▼ with the comparison date and the noise threshold (filled from `H`), and the value- and growth-trap rules with thresholds read from `config.yaml` (`_trap_rules`) | `score_key_html()`, `fillScoreKey()`, `openScoreKey()` | `tests/test_club_readiness.py` |
| The key is reached from the Trap flags tile ("What a flag means"), the guide ("What each column means") and the search palette (label "What the columns mean"; "trap", "composite", "risk" find it through a `keys` field) | `renderKPIs`, guide markup, `PAL_ACTIONS` | same |
| One line under the stock sheet's eight scores: 0-100 against the sector, about 50 the median, higher adds more points on all eight, and what high means for Risk, Size and Investment | `openStockDetail` (`.msr-note`) | same, in a browser |
| The guide's first statement adds the direction rule | guide markup | same |
| On a phone the flag chips read "Value trap" / "Growth trap" (CSS `::after`, so the row's text and `test_dashboard_browser`'s check are unchanged) | `_css_investor` | same, in a browser |
| A ticker this run did not score is named in the palette, the table's empty state and My holdings' add box (`notInRunText`, shared) | `_js_table`, `_js_ux`, holdings search | same, in a browser |
| Holdings' rank-change chip uses the page's date format | `renderHoldings` | - |
| Market backdrop says what FRED and a pp are | `renderMarket` | same |
| Two stocks side by side fit 375px | `_css_investor` | same, in a browser |
| "Reading the numbers" on `research.html` | `research_page.py` | same, and `tests/test_research_page.py` |

**Claims registered** (`claims.py`): `dashboard.score_key` (which way each score runs; checked
against the payload by Spearman correlation of each category score with a weighted metric whose
direction the line names, and the largest company in each sector scoring under 5 on Size) and
`dashboard.trap_flag_key` (the thresholds come from config; the engine flags what the key
describes, on constructed cases; a flag changes neither score nor rank, and Leading the ranking
excludes flagged names). `scripts/rehearse_publish.py` now runs `tests/test_club_readiness.py`
with the page tests.

## Open (for a later session)

1. **A way back from deep in the ranking on a phone.** The header is static below 760px and its
   Methodology button hidden below 400px; 501 cards are about 48,000px. Options: a sticky compact
   header on phones, or a "Top" control. Either is a layout change - judge it against the investor
   pass's first-screen budget (`tests/test_dashboard_browser.py`).
2. **"Its weakest scored category is Size at 2 out of 100"** (the summary sentence, payload) reads
   as a business weakness for every megacap. It is a true statement about the score. A change is a
   `stock_summary` sentence change, so it goes through `claims.py` (`summary.weakest`) and is not
   display-only.
3. **Methodology's opening**: the run's momentum-rule note comes before "Overview", and the
   Overview names `METRIC_COLS`. Both are generated (`run_screener.build_screener_overview`,
   `generate_dashboard._load_methodology_html`); edit the generator and rerun the overview claims
   tests.
4. **The score cards' "18.6 pts (22% wt)"** abbreviations; "How it adds up" spells the same
   arithmetic out, so this is low value.
5. **Context jargon left as is**: implied volatility, straddle, put skew, 10b5-1, VIX, Sahm. Each
   card already explains its terms in its research paragraph; the plain-English pass there is
   `plan/context-layer.md` item 4's territory.
6. **Adding a departed stock to My holdings.** The add box now says why it cannot; the panel keeps a
   saved name the run stopped scoring, but cannot take a new one. Allowing it would touch settled
   row 5's storage rules.
