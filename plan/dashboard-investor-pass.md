# The investor pass - making the dashboard read like a product an investor would pay for

**Opened 2026-10-15 (owner-run, night).** Owner: *"can you do a deep ui/ux pass to make it look really
nice and expensive, and premium for investors please? ... lots of things that still looks weak or ai
generated. Need to guide the users eyes, and show things first that makes sense."*

**Presentation only.** No number, rank or sentence in the payload changes: every phase is installed
with a check that `dashboard_data.js` and `dashboard_context.js` are byte-identical before and after
(sha256). The standing rules still bind (CLAUDE.md "The dashboard is the product"): look at 1440 and
375 before and after, colour for meaning, tabular numerals, the page shows the engine's numbers.

## What read as weak, from the live page (1440 and 375, 2026-10-15 22:50)

1. **The top spoke to an operator.** "Factor rankings", four equal buttons, then tiles for "Ranked this
   run", "Moved materially", "Trap flags" and "Metric coverage". An investor's first questions - what is
   on top, how has it done, can I trust it - were not answered above the fold.
2. **A tutorial box** - numbered 1-2-3 circles and a "Got it" button - in the prime position.
3. **Top 5 cards**: eight identical blue progress bars with labels and decimals each, no hierarchy;
   on a phone, five full screens of scrolling.
4. **Seven collapsed sections as a stack of grey rows**, their key numbers in small grey text - it
   read like a settings page.
5. **The drilldown** opens on a wall of text; the score profile is a scroll away.

## The order the eye should take

statement (what this is) -> at a glance (rank, movement, how it has done, caveats) -> the leaders
(Top 5) -> the tiles (your list, what changed, what reports soon, context, evidence) -> the full
ranking. The proof pages - "Has it been tested?" and the Factor Lab - are one quiet step from the
statement, never louder than the primary action.

## Phase 1 - DONE 2026-10-15 (owner-run)

- **Hero**: "The S&P 500, ranked on eight factors", a sentence that names the eight, the disclaimer as a
  quiet note; one primary button, "How it works" secondary, the two proof pages as text links with an
  arrow (buttons on a phone, 2x2).
- **At a glance**: one panel of four divided tiles, not four floating cards. Ranked / Moved materially /
  **Track record** (top 25 against the equal-weight S&P 500, filled from the context file the moment it
  lands, opening the Track Record section) / Trap flags (or the momentum weight when the regime rule
  moved it). Metric coverage stays in Diagnostics' pill.
- **Guide**: the same three statements as a quiet ruled band - no numbers, no box, "Hide".
- **Top 5**: each card's eight scores as one column profile on a 0-100 scale, whole numbers under the
  bars (exact scores in the tooltip and the drilldown), the 70+ columns in full accent. On a phone the
  five cards are a row you swipe.
- **Section tiles**: Holdings full width, the other six in a 2-column grid; each tile states its
  headline number in readable type and one line of what it is for; an opened tile spans the row. On a
  phone, one column.
- **A defect the grid exposed and fixed**: Reporting, Market and Track drew their charts *before* the
  section opened, so they measured the width of the closed tile; they now open, then draw.

## Phase 2 - DONE 2026-10-15 (owner-run)

- **Rankings table, quiet heat**: no blue wash behind 4,000 score cells; each score's ink brightens with
  its value (18% to 100% of the primary ink), so strengths read first and weaknesses recede; the
  composite is the brightest column; rank change in quieter green and red.
- **The stock sheet**: the eight-score panel moved above "Why it ranks here" (and its jump link first -
  `tests/test_dashboard_navigation.py` pins the new order); the header adds the last price beside rank and
  composite (not on a phone, where the header is full); the lead sentence larger, group labels in
  sentence case.

## Phase 3 - DONE 2026-10-15 (owner-run)

- The sector heatmap and the factor-correlation grid use the same quiet ink as the table (the correlation
  grid keeps a lighter wash and its outlines on pairs above 0.7).
- "Top 5 Stocks" is "Leading the ranking" and "Full Universe Rankings" is "The full ranking", each with one
  line saying what it holds and what selecting does.

## Phase 4 - DONE 2026-10-15 (owner-run)

- The Factor Lab's menu: fourteen "Experiment N" chips became five labelled groups (Momentum, Value,
  Quality, Risk, Across the factors) from a `group` field on each experiment record.
- Header nav "Top 5" is "Leaders", matching its section.

## Phase 5 - DONE 2026-10-15 (owner-run)

- **Before you decide**: each card's research paragraph (`.ctx-why`) is clamped to three quiet lines and
  opens on a select - the reading leads, the citations stay one click away (nothing removed).
- **Rank history**: the "±N ranks" band label moved from inside the plot (where it sat on the line) to the
  axis row between the two dates.

## Phase 6 - DONE 2026-10-15 (owner-run)

- Hero actions under the copy (the eye reads down the left edge); the guide tighter, "Hide" in its corner on
  a wide screen. Checked but deliberately unchanged: My Holdings stays above What Changed
  (`tests/test_holdings_panel.py::test_holdings_section_sits_above_what_changed` - a settled decision).

## Phase 7 - DONE 2026-10-16 (owner-run, just after midnight)

- **Link previews**: Open Graph and Twitter tags on the dashboard, `research.html` and `lab.html`, with
  `og-card.png` (1200x630) rendered by `scripts/make_og_card.py`. The card states the product only - no
  rank, score or return - so it cannot go stale. Decorative rising bars were tried and dropped: on an
  investment product they read as implied returns.
- **Track record tile** says the record's length beside the number: "8 months, too short to judge".

## Phase 8 - DONE 2026-10-16 (owner-run)

- **Factor Analytics tile** states a finding instead of a description: the sector with the highest
  value-trap rate and the rate (from `vt_by_sector`, already in the payload).
- **The Methodology dialog**: tables keep a margin from the dialog's edge (they ran into it), metric names
  sit on one line on a wide screen, no blue-tinted stripes, the code blocks lose their accent rule, and an
  inline file path wraps instead of pushing the dialog sideways on a phone (it overflowed by 12px at 375).

## Phase 9 - DONE 2026-10-16 (owner-run)

- **The first screen on a wide display** (>= 1200px): the statement on the left, the four at-a-glance
  figures beside it as a 2x2. The right half of the first screen was empty; the leaders now show above
  the fold at 1440x900. Below 1200 the tiles stay a row under the hero, and on a phone a 2x2 as before.
- **The rankings table**: tickers in ink, not link-blue (the whole row opens the stock; the ticker turns
  blue on hover and on the open row). The composite's bar was an underline with no scale; it is now a
  gauge on a visible 0-100 track, its fill drawn against the track's own width (it had been drawn
  against the cell's, so 71 read as about 79% of the visible line).
- **Diagnostics tile**: its three pill badges are a line of text like the other tiles' headlines, status
  dots kept.
- **Header**: below 1340px the nav clipped "Diagnostics", and at 1000px the brand ran under the first link.
  The brand no longer shrinks, the search keeps its icon and shortcut, the links tighten, and where the
  nav must still scroll (below 1180, phones included) its right edge fades.

## Phase 10 - DONE 2026-10-16 (owner-run)

- **The stock sheet opens on the eight scores.** The composite card above them repeated the header's rank
  and composite word for word; it is hidden (`#modal-score-row > .modal-score-card.composite`), which on
  a phone brings "Why it ranks here" up by about 90px.
- **One heading voice in the sheet**: "Why it ranks here" and "About" were 11-13px uppercase labels beside
  the sentence-case "How it adds up"; "Rank History", "Analyst Price Targets", "Company Snapshot", "Sector
  Peers" and "Data Provenance" were Title Case. All are sentence-case 14px headings now.
- **The score words beside each bar** ("Strong", "Very weak") were 9px capitals in a chip; they are 11.5px
  text.

## Phase 11 - DONE 2026-10-16 (owner-run)

- **One product, three pages**: `research.html` and `lab.html` carried a plain-text title and blue
  "Back to the screener" links. Both now open with the front page's mark and name (a link home) and the
  same quiet three-way nav - Rankings, Has it been tested?, Factor Lab - with the current page marked.
  `research_page.py` generates the research page's bar; `lab.html` is source.

## Phase 12 - DONE 2026-10-16 (owner-run)

- **Four links in browser-default violet** (Chrome's dark-scheme link colour): the Track Record's note
  linking to the tested-history page, and the insider filing dates in Before you decide. Accent now.
- **One segmented control**: Factor Analytics' Median/Average and Value/Growth toggles were solid accent
  buttons in 11px heavy type beside the quiet `seg-control` used by What Changed, Reporting and Track
  Record; they match it now.
- **Chart titles in sentence case**: "Trap rate by sector", "How stable is the ranking?", "Are the factors
  independent?".

## Phase 13 - DONE 2026-10-16 (owner-run, after the 02:00 run)

- **One type scale.** Measured on the opened page at 1440: about 30 distinct text sizes, most of the odd
  ones (11.2, 11.84, 12.16, 12.48, 13.12, 13.44, 14.08px) from 35 scattered `rem` values in older blocks.
  Each is now the nearest step of one half-pixel scale (10 / 10.5 / 11 / 11.5 / 12 / 12.5 / 13 / 13.5 / 14 /
  15 / 17). Left on purpose: code set in `em` (it follows its sentence) and the 9.5px labels under the
  leaders' bars, where eight columns share 230px.

## Phase 14 - DONE 2026-10-16 (owner-run)

- **One corner radius for panels (14px)** across the three pages: measured, the front page's diagnostics
  cards were 10px, the research page's 10-12px, the lab's 10-12px.
- **No monospace figures on the research page**: about 190 figures were set in JetBrains Mono (the look the
  2026-10-05 brief named); they are Inter with tabular numerals, like the front page. Code stays mono.

## Phase 15 - DONE 2026-10-16 (owner-run)

- **320px**: the hero's 2x2 scrolled the page sideways by 5px (its `1fr` columns could not shrink below
  "Has it been tested? ->" plus padding). Columns are `minmax(0, 1fr)` and the buttons tighten below 360px.
  All three pages measured at 320: no horizontal scroll.
- **Checked and left alone**: the first paint before the 6 MB data file lands already shows skeletons in
  the new 2x2 shape; the lab's experiment view reads cleanly at 1440 and 375.

## The professional pass - 2026-10-16 (owner-run, day)

Owner: *"keep making ui/ux improvements please. Really go deep, like a professional front end designer
would do ... no signs of ai language anywhere, or that it looks like it was vibe coded."*

### P1 - copy, dates and details (the audit came first: every visible text node on the page and a sheet)

- **Voice.** About 50 strings rewritten in one register: short declarative sentences, specific numbers, no
  "X, not Y" reflex, no em-dash asides, "we" removed, the disclaimer stated in the hero and footer rather than
  four times. Kept on purpose, because tests and settled decisions hold them: the summary's "never says whether
  to buy, sell or hold", the track record's "not a portfolio to follow", the cadence line's "not by itself a
  reason to act", the storage note's "not an account and is not backed up", and the rank sentence's "not a
  percentile" (the 2026-10-07 correction). They now read as statements, not as hedges.
- **Dashes.** Spaced hyphens used as dashes ("estimate - a beat or miss") and em-dash asides were rewritten
  out of the page's own copy, the summary sentences, the Market notes and the history notes.
- **Dates.** One reader-facing format, "Sep 16" (the year only when it is not the run's): the tiles, What
  Changed, the stock sheet's history, the rank chart line, the Market cards, the track-record table and the
  footer (which printed "2026-10-16 02:00 UTC" under a hero saying "Oct 16, 2026, 2:00 AM CDT", and with no
  stamp would have printed the reader's own clock as the data time). The summary sentences use "Sep 16, 2026"
  (`stock_summary._fmt_day`, which had read "4 Nov 2026").
- **Defects found by reading every line**: the Market cards printed "62th", "2th" and "3th percentile"; and
  the VIX note said the screener "cuts momentum's weight when the universe is turbulent and **raises it when
  calm**", false since the rule was rebuilt on 2026-10-12 (it cuts only in the panic state and never raises).
- **One disclosure icon**: the tiles' SVG chevron replaces a text "▼" (stock sheet), "▸" (notes) and "›"
  (holdings), drawn as a CSS mask so it takes the text colour.
- **Live with the next data run**: the summary sentences, Market notes and history notes are built into the
  payload, so they change on the 2026-10-19 02:00 run; `scripts/rehearse_publish.py` checked them against the
  page tests and the claims gate.

### P2 - the system details: touch, focus, selection, phones

- **Touch targets** (`@media (pointer: coarse)`): measured at 375px, 40-odd controls were under 40px tall,
  the guide's "Hide" 28x23. Every control is now at least 40px on a touch screen; a mouse keeps the compact sizes.
- **One focus ring**: tabbing the page found the brand ring on half the controls and the browser's thin grey
  default on the rest; the filter fields' 12% glow was close to invisible. One 2px accent ring for everything,
  a visible ring on fields, an accent edge on reporting rows.
- **Selected text** in the accent rather than the browser's bright default.
- **The stock sheet on a phone**: its tool row ran 13px past the padding at 375 and was clipped at 320; the list
  position (which repeats the header's rank) is hidden below 420px and the tabs fade at their edge. The hero's
  eyebrow drops "S&P 500" below 420px so it stays on one line.

### P3 - component craft, inspected at 2x

- **Leader cards**: the 50 reference line was drawn over the bars and cut every filled column; the fills now sit
  above it, so the line marks only the empty track.
- **Hyphens in tabular headlines**: Inter's tabular-figure setting widens the ordinary hyphen ("value - trap",
  "10y - 3m"); the non-breaking hyphen keeps its width (measured 21.1px against 25.5px for "a-b" at 14px), so
  the compound words inside the tiles' and summary panel's headlines use U+2011. Figures stay tabular.
- **One heading case**: the seven tile titles were Title Case beside sentence-case section headers, nav and
  headings; all are sentence case. "Defensibility & Diagnostics" is "Diagnostics", as the nav already said
  (the section and its features are unchanged). The search palette gains the three sections it was missing
  (Reporting soon, Market backdrop, Track record).

### P4 - the Methodology document

- **Headings as statements, in sentence case**: "What Is This?", "Where Does the Data Come From?", "Key Design
  Decisions & Why", "Limitations to Be Aware Of" and the rest are "Overview", "Data sources", "Design
  decisions", "Limitations"; "Quick Start" is "Running it yourself"; each category's "Why these?" is "Rationale."
- **The opening and the summary rewritten.** The subtitle promised to explain "how it arrives at its *picks*"
  and the summary asked which stocks "look best"; both framed the ranking as a recommendation. The opening also
  said the candidate metrics are ones "the self-improving engine may activate", false since 2026-08-20
  (`allow_auto_apply` is false; a weight changes only through research and a changelog entry).
- **One typesetting pass** (`run_screener._typeset_md`, `tests/test_overview_typesetting.py`): about 130 dashes
  in three forms become a spaced en dash for a dash and a true minus inside a formula ("(Net Income − Operating
  Cash Flow)"); code is untouched. Three patterns that matched the old hyphen (a claim's detector, two tests) now
  accept the en dash.
- **In the dialog**: the document's own title and subtitle are hidden under the dialog's header, which already
  names it; the doubled section rules (a markdown rule plus the heading's border) are one; the run's momentum
  reading is a quiet note with a human date.

## Next (in order)

1. **Typography and rhythm**: done (phases 13-14).
4. **research.html and lab.html**: bars done (phase 11); their section headers and figures next.
