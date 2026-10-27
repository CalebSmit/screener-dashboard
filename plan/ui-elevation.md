# UI elevation - from "okay" to "great" (opened 2026-10-27, owner-run)

Owner, 2026-10-27: *"do a full in depth ui/ux update to all the tools. right now it looks okay,
but it can look great. Be creative and look into absolutly everything to make sure it is as good
as it can be."*

The 2026-10-05..07 redesign (`plan/dashboard-design-system.md`, `plan/dashboard-redesign-master.md`)
removed what read as generated: GitHub's palette, a hue per category, monospace everywhere, 12px
type. It got the page to **restrained**. What it did not give the tools is **identity** - the four
surfaces are clean, competent and interchangeable with any dark SaaS template. This pass is about
the second half of "premium": a point of view, executed precisely.

## The direction: a research publication with an instrument inside it

Three ideas, and every change should serve at least one:

1. **One typeface, a confident headline voice.** Page titles, section titles and hero
   statements are Inter at weight 600, tracked tight - the style of the leader cards' ticker.
   *Revised 2026-10-27, same day:* the first pass set headlines in a reading serif (Newsreader);
   the owner looked at it on his phone and said *"I like the font of the EXPE word ... I don't
   love how the title font looks"* and asked for consistent fonts. **Do not reintroduce a
   serif or a second display face.** Long lead sentences stay at weight 400; only headlines
   take 600.
2. **The data is the decoration.** Where a surface is a box of text describing numbers, show the
   numbers instead - a small true chart, a magnitude-scaled cell, a distribution. Nothing
   decorative that is not data. (Tufte's data-ink, and the rule this page already keeps: colour
   for meaning.)
3. **Depth through light, not glow.** Layered warm-black surfaces with a 1px inner top
   highlight, hairline rules, generous and regular whitespace on a 4/8 grid, motion that only
   answers a user's action (120-200 ms). No glows, no gradients-as-decoration, no entrance
   choreography.

## Shared tokens (every surface uses these values - change them here first)

| Token | Value | Job |
|---|---|---|
| `--font-display` | `'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif` - the same face as the body; the token names the *role* | page titles, section titles, hero statements |
| Headline weight / tracking | 600; `letter-spacing: -0.022em`, hero titles `-0.032em`; lead sentences 400 at `-0.011em` | the ticker's style (`.top5-ticker`: 600, tracked tight) at headline sizes |
| Hero title size | `clamp(36px, 4.6vw, 56px)`, line-height 1.04 | |
| Section title size | `clamp(22px, 2.2vw, 28px)`, line-height 1.15 | |
| `--hairline-hi` | `inset 0 1px 0 rgba(255,255,255,.04)` | the top edge of a raised card; replaces nothing, adds depth without shadow |
| Heat (score magnitude) | `rgba(57,135,229, a)`, `a = 0.03 + 0.25 * score/100` | sequential, single hue (the accent), **magnitude only**: text stays ink, never coloured |
| Fonts URL | `https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap` | loaded the existing non-blocking way |

Accent, ink, neutrals and the meaning colours are **unchanged** (`plan/dashboard-design-system.md`).
Amber stays the caveat colour, so no surface may use an amber/gold accent for interactivity.
Two families, each with one job: Inter for everything a reader reads (headlines at 600, UI and
figures tabular), JetBrains Mono for literal code. The owner's 2026-10-05 complaint was three
families with no roles, and his 2026-10-27 one was a serif that did not match the rest.

## The rules that do not move

- **Every number, rank and sentence byte-identical.** `dashboard_data.js` / `dashboard_context.js`
  are never rewritten; the page is rebuilt with `python scripts/rebuild_page.py`, which cannot
  touch them, and `tests/test_rebuild_page.py` fails if the committed page is not what the
  generator builds.
- Look at every changed surface at **1440 and 375 px before and after**, and read the screenshot
  (a clipped token does not widen the document).
- First paint stays under the **3,000-node** budget (`tests/test_dashboard_browser.py`); the
  lab's first-paint file stays under its gzip budget.
- A new sentence that says how a number is computed needs a `claims.py` registration. Prefer
  showing the number to writing a new sentence about it.
- Decision support, not a recommendation: no surface may imply "buy".
- WCAG AA contrast for text, visible focus, keyboard reachability, `prefers-reduced-motion`.
- Never weaken a test. A test that pins a presentational choice this pass deliberately changes
  is updated to pin the new choice exactly as strictly, and the commit says so.

## Status

### First pass - 2026-10-27, owner-run (`good/2026-10-27-owner-1`)

Four agents, one surface each, every diff reviewed before merging; the combined tree passed the four
gates and `scripts/rehearse_publish.py`, and `dashboard_data.js` / `dashboard_context.js` /
`lab_data*.js` are byte-identical to the live site.

| Surface | What changed | Where |
|---|---|---|
| Dashboard page | headlines and wordmark in the headline voice; **"498 companies, one dot each"** (the published composites, top 25 and trap flags marked, hover/tap names a stock, click opens it; one SVG built after first paint); leader cards as labelled score rows with a tick at 50; a small true chart on every home tile; heat-shaded score cells; phone list 96 -> 80 px a row (page ~6,900 px shorter); **open sections were capped at 5,000 px, so Reporting soon hid ~100 companies** - uncapped, day headings sticky; footer `#gen-time` labelled true UTC | `_css_page()`, `_js_page()`; `tests/test_dashboard_page_ui.py` (5) |
| Drilldown | masthead: ticker and sector line, the company name as a headline, rank / composite / price, and a strip of all 498 composites with this stock marked; trap chips in the header; heat score tiles with a median tick, unscored as a dashed outline; "Why it ranks here" with its figures emphasised (characters wrapped, never changed - tested on every stock); How it adds up as a receipt with a stacked total; one colour language for charts; folding header; Enter/Space on every card; drag the header down to close on a phone; the Data tab now lights; the rank-history label no longer overprints at 375 | `_css_drilldown()`, `_js_drilldown()`; `tests/test_drilldown_presentation.py` (10) |
| Factor Lab | the four rows of chips became a numbered index grouped by category (sticky side column >= 1080 px, a picker on phone) with each experiment's matched-figures bar; verdicts drawn as one cell per figure with the "no effect would match k" tick; tolerance marks per figure; loading state for lazy series; deep links open the experiment | `lab.html` |
| Has it been tested? | the headline answer drawn as IC estimates with 95% intervals and the check's lead; the trading rule's gap drawn the same way; tables stack on phone | `research_page.py` (`headline_figure`, `excess_figure`) |
| Bot dashboard | text inks at AA (tokens unchanged); direction shown by a triangle as well as colour; the tape scrolls with the page, pauses, stands still under reduced motion; one-row phone masthead and a theme switch; Replay named "Machinery" with its own band and per-session strip; chart end labels no longer collide; a withheld Sharpe shows a dash, never a count in the figure slot | `screener-bot` `bot/dashboard_template.html`, `DESIGN.md` |

New tool: **`scripts/rebuild_page.py`** rebuilds `index.html` / `dashboard.html` from the published
payload without being able to touch it; `tests/test_rebuild_page.py` fails if the committed page is
not what the generator builds.

### Same day, second pass - one typeface (owner)

Owner, looking at the live page on his phone: *"Please focus on the typography and consistent
fonts. I like the font of the EXPE word ... I don't love how the title font looks."* Newsreader is
gone from all three pages and from the font download; every headline that used it - 18 rules
across `generate_dashboard.py`, `lab.html` and `research_page.py` - is now Inter 600 tracked tight,
the ticker's own style, and the two long lead sentences (the drilldown's first "Why it ranks here"
line, the lab's claim) are Inter 400. Presentation only; payload byte-identical.

### Same day, third pass - the header

**Header, 2026-10-27 (owner):** every section link is shown in full or listed, in order, under a
"More" menu (`_js_page`, priority-plus; re-measured on resize, font load and link visibility).
The "Last updated" line left the bar (the hero and footer state the same time) and the search
shows its icon and shortcut at every desktop width. Before: links clipped at 1024, 1152, 1536, 1680
and 1920 px. `tests/test_header_nav_fit.py` (9) opens the page at seven widths.

### Same day, fourth pass - the Factor Lab and the research page, read as a student would

Audited at 375 and 1280 px, several experiments of every kind, settings changed, by pointer, touch
and keyboard. Presentation and access only: no figure, sentence about a computation, data file or
`lab_engine.js` changed.

| Page | What changed | Where |
|---|---|---|
| Factor Lab | bar charts read by tap and keyboard like the line charts (one `readable()` helper); a line chart's label ends on its last month; a failed lazy part keeps the experiment and offers Try again; the tab title names the experiment and opening one moves focus to its heading; the experiment list answers the arrow keys; 44 px controls on a phone; equations wrap; the no-effect line on its own line; a key to the meters; each verdict links to "What it means for the screener"; a way on at the foot; `<noscript>`; print as ink on paper | `lab.html`; `tests/test_lab_reading_and_access.py` (13) |
| Factor Lab, follow-up | phone overview cards fold to title + meter with "What it found" (cards 6,990 -> 3,894 px at 375); "%" on every bar chart's y axis; "Copy a link" confirms in a status region; the hero's figures as rows on a phone | `lab.html`; `tests/test_lab_reading_and_access.py` (18 in all) |
| Has it been tested? | every section has an id; an "On this page" list read off the sections the generator emitted; the foot links back to the answer, the lab and the rankings; 44 px nav, contents and summary on a phone; print as ink on paper | `research_page.py` (`with_contents`); `tests/test_research_page.py` (+2) |

### Same day, round two - the lab and the research page again: phone tables, an axe audit, phone speed

Owner: *"Keep finding improvements please!"* Presentation, access and load order only: no figure, no
`lab_data*.js`, no `lab_engine.js` (so `ASSET_VERSION` stays 27), no ledger, registry or `summary.json`.
Scratch measurements in the gitignored `logs/round2-lab/` (a gzip-serving harness, Playwright, CDP).

**Automated accessibility audit** - axe-core 4.10.2 injected by Playwright for the audit only (the pages
and the suite never load it): the lab's overview and three experiments of different kinds
(risk-managed momentum: line charts; size effect: a lazy part and bars; 153 factors: the 716 KB part)
and the research page, each at 375 and 1280 px - ten audits.

| | Before | After |
|---|---|---|
| Serious / critical | **1 rule on all ten** - `link-in-text-block`, 4-5 nodes a lab audit, 3 on research (links marked by colour alone: 1.12:1 against the muted grey, 1.66:1 against body text) | **0** |
| Moderate | lab: `landmark-complementary-is-top-level` (the figures an `<aside>` inside the banner) and `region` (wordmark, closing note outside every landmark); research: `landmark-one-main` and `region` x14-23 (every word of the page outside a landmark) | **0** |
| With WCAG 2.2 `target-size` enabled as well | - | **0** |

**What changed**

| # | Page | What was wrong | What it is now | Tested in |
|---|---|---|---|---|
| 1 | Lab | The overview's three tables after the cards were **8,313 px at 375 (Inter loaded)**; After the paper squeezed each result's name into a 110 px column four lines deep and hid its drawn drift on a phone | After the paper is a row per result: the name across the width, its two figures under the two column heads, the drift drawn beside them (its key stacked over the column). The scorecard and the category map keep every category, its weight, what the lab measured or covers and its experiments; only the third column folds, behind a 44 px button **named for that column** ("Left open", "Not covered"). Every row is shown; print shows every cell. **5,615 px; page 15,904 -> 13,206.** Desktop byte-for-byte the same layout | `test_lab_reading_and_access.py` (+2) |
| 2 | Lab | On a phone the map's third column lost its header and had no label - "not covered" read as more of what the lab covers | "Not covered:" before it, as the scorecard's "Left open:" already had | same |
| 3 | Lab | The display changes that make tables into rows drop table semantics in some browsers | the three tables carry explicit `table/rowgroup/row/columnheader/cell` roles | same |
| 4 | Lab | axe above: no single main landmark, an aside nested in the banner, links marked by colour alone | `header.bar` (banner) / one `<main>` round the introduction and the lab (the old `main#lab` is `div#lab`) / `footer`; the figures a labelled `section`; links inside sentences, citations, notes underlined at 55% of their colour | +1 |
| 5 | Lab | The menu's two files were discovered by the script tags 460 lines down | `<link rel="preload">` for both from the head, same URLs; a test keeps them equal to the tags. Menu on the phone profile (4x CPU, slow 4G, 375 px) median **1.67 -> 1.57 s** (n = 6 each, interleaved; machine shared, so +-0.2 s noise) | +1 |
| 6 | Lab | An experiment's series was asked for only at the click | asked for at a **mouse button down** or 150 ms of hover or keyboard focus on its card or menu entry - never at a finger coming down (on a phone that starts a scroll as often as a tap, and a scroll past the 153 factors must not fetch 716 KB). A 100 ms mouse press: size effect **0.87 -> 0.67 s**, 153 factors **4.7 -> 4.5 s**; a phone tap unchanged (0.75 s), by design | +1 |
| 7 | Research | The first screen answered "Not measurably." and went straight to "average monthly rank correlation ... 95% interval"; what a rank correlation is sat three cards down | a plain sentence under the headline - the companies it ranked higher did not do measurably better the next month than those it ranked lower, in either period or over the whole record; no edge has been shown - printed **only while every interval includes zero** (`research_page.plain_answer`, claim `research.plain_answer`); the chart's caption gives the scale's ends and its zero | `test_research_page.py` (+2) |
| 8 | Research | The check on the headline chart was "Fed today's filings / lead over the real ranking" - meaningless until the bullet two cards on | "The check: can this test see an edge at all?" / "A ranking that cheats - given today's filings for every past month; its lead over the real one"; the legend says "the cheating ranking's lead" | same |
| 9 | Research | axe above | banner / one main / footer; in-sentence links underlined | +1 |

**Phone speed, measured** (Playwright, 375 px, CDP CPU 4x and Lighthouse's slow 4G, gzip as Pages
serves it, cache off): time to a usable menu ~1.6 s, of which `lab_data.js` (100 KB gz) arriving is
~1.0-1.3 s and the overview's first layout ~260-290 ms of main thread; opening an experiment whose
part must load: size effect 0.75 s, the eight sorts 1.55 s, **153 factors 4.5-5.0 s (the worst
offender)**; a re-run after changing a condition 25-100 ms on every experiment (17 measured, every
control, median of three) - no re-run is slow. The 153 factors are slow because of bytes - 153 factors
x 1,200 months x three universes, already integer basis points, ~1.3 bytes a number gzipped - and the
page needs all three universes on opening, because it re-derives the paper's mega- and large-cap
rates to check itself against the build. Splitting the part by universe would only defer a third.
Not done; see "Seen, not fixed".

Tried and **reverted**: `content-visibility: auto` on the three tables cut the overview's first layout
290 -> 160 ms at 4x, but an element it skips has an empty `innerText`, which a test of the tables'
text caught - a reader's copy or a tool's read of an off-screen table would come back empty too.
Deferring `mark()`'s geometry read to the next frame saved nothing (it was the page's one layout, not
a second one).

**Seen, not fixed**

- **153 factors on a phone: 4.5-5 s.** The real fix is a data-format change in `factor_lab.write`
  (e.g. ship the "all" and "mega" universes first and check the large-cap figures when the third
  arrives) - it moves `LAZY_KEYS`, the part files, the node reassembly test and `ASSET_VERSION`, and
  the self-check would turn asynchronous. A lab session's job, not a presentation pass.
- On a phone the scorecard and the category map list the same experiment links per category twice,
  a section apart. Merging them is a content decision.
- `tests/test_case_against.py::test_each_line_opens_the_section_that_shows_it` (dashboard, untouched
  here) failed once under the parallel suite on a loaded machine and passed alone.

### Same day, night - the drilldown pass (owner-run)

The stock sheet audited as a student would use it - EXPE (#1), JPM (bank), VICI (REIT), ED (utility),
FDXF (three categories unscored), WMB (#490) - at 375 and 1280 px, by keyboard and by heading. Eight
changes, all presentation; payload byte-identical; `tests/test_drilldown_reading.py` (49). What was
wrong, and what it is now (detail in `plan/dashboard-inventory.md`, "The drilldown pass"):

1. **Only one heading in the sheet** (the ticker). Eleven section titles and the eight workings
   categories are now real `h3`/`h4`, a button inside a heading; the dialog is named ticker + company.
2. **A tab only scrolled.** It now moves focus to the section it names; the lit tab is `aria-current`.
3. **Ten tabs did not fit an 880 px sheet** - Peers and Data sat past the edge unannounced. Arrows at
   the edge (mouse) and a fade (touch).
4. **Next / Previous threw focus to Close**, so a second Enter closed the sheet; and a screen reader
   heard only "2 of 498". Focus stays on the control; a status line names the stock and its rank.
5. **No plain-English meaning for 38 metrics** ("EV/EBITDA", "Beneish M-Score", "12-1M Return"). Each
   metric card opens with "What it measures" (definitional only - tested: no digit, no computing verb,
   no advice term), and each category with the score key's own "A higher ... score means" line.
6. **The research under each context reading** opened only by clicking a paragraph - no keyboard, no
   cue. A "Read the research" button now, kept in step with the paragraph's own click.
7. **On a phone the workings' figures had no labels** ("2.51 ● 27 60%" - which is the percentile?).
   "27 pctile · 60% weight · 16.30 pts", drawn by CSS; Calculation is a 44 px target.
8. **Printing with a sheet open printed the page behind it.** It now prints the sheet - every section
   open, light ink, no controls - and puts the sheet back afterwards (Ctrl+P or the toolbar's icon).

Seen and not changed: an unscored category tile says "Not scored: no data" without naming the reason
(a new listing or a price series the run rejected) - the reason is not in the payload per category, so
saying it needs a payload field; the "Strong / Weak" words in How it adds up read "Very Weak" for a low
Risk score, which is a volatile price, not a weak company - worth a wording decision by the owner. **Decided the same night (coordinator):** the four
words now describe the score, not the company - High / Above mid / Below mid / Low.

### Same day, fourth pass - the page outside the sheet (owner-run, 2026-10-27)

Presentation only (payload byte-identical; `scripts/rebuild_page.py`). Opened at 375, 768, 1024, 1280 and
1920 px and read; every item below was a screenshot before it was a fix. `tests/test_page_fourth_pass.py`
(20, browser) asserts each as behaviour.

| What was wrong | What changed |
|---|---|
| Residuals 1-4 below | done - see the list |
| Phone rankings rendered all 498 cards: a 46,600 px page, footer at the end | 50 at a time, "Show the next 50" / "Show all 498" (`PHONE_STEP`, `shownRows`, `#table-more` in `_js_table`); filters, search, sort, the sheet's stepping and deep links read the full list; a new filter or sort starts from the top. Page 11,100 px |
| The table was ~500 tab stops between the filters and the footer | one roving tab stop (`rovingT`, `syncRoving`); arrows / Page / Home / End inside, as before; `aria-describedby="universe-keys"` says so |
| Below 1180 px the spacer rows spanned 15 columns when 14 were shown, so the fixed-layout table grew an empty 16th column that took the company name's width; at 768 px (iPad upright) the company column was 0 px and Inv / Flags were cut off with no way to scroll | spacers span the shown columns (`visibleCols`); columns tighten 761-1300 px; at <= 900 px the name leaves the row (the ticker carries it as a tooltip); 761-768 px no longer stacks the filters in a right-aligned column |
| Desktop: "Valero Energy Corpo..." beside a 108 px flags column blank on 87% of rows | flags 80 px (header "Flags"), composite 100, rank 46; the rare stock with both flags stacks them |
| Phone: a trap flag halved the company name ("Delta Air ...") | the flag sits under the name; two flags fit |
| Phone Menu pill sat over a card's composite on every screen, and over the footer | steps aside while scrolling down, back on the way up and at the end (`.tuck`, `initMobileNav`); footer leaves 96 px |
| Track record tile showed "+13.8%" from 8 months with no word on what that can show | "Too short to tell skill from luck: 0.7 standard errors, where two is the usual bar" whenever the panel's t is below two - the panel's own figure, claim `context.track_luck` (new check) |
| My holdings' remove button: 15x17 px, read aloud as "times" | `aria-label="Remove T from your list"`, 32 px (44 px on touch); Concentration's chevron stays beside its label on a phone |
| The dot picture's axis had no name | "Composite score, 0-100 - higher ranks higher" under it |

### Left for a later pass (owner-run; nightly sessions do not take design work)

1. ~~Phone Reporting soon hides the "Last quarter" and "Insiders" columns~~ **DONE 2026-10-27 (fourth
   pass).** Below 640 px the two figures ride under the company name (`.rp-m`), labelled "Last quarter
   vs estimate" and "Insiders"; the columns stay off so the row still fits.
2. ~~Phone Weighting select clips "Balanced (published)"~~ **DONE** - Weighting and Sort by take the
   full row on a phone; the short controls pair up above them (`grid-auto-flow: dense`).
3. ~~Phone Side-by-side puts the third stock in a sideways scroll~~ **DONE** - below 480 px each
   category's name is a caption row above its scores, so two, three or four stocks share the width.
4. ~~The dot picture is not keyboard-focusable~~ **DONE** - a slider over the companies in score order
   (`distKey`): focus lands on the published #1, arrows step one dot, Page keys ten, Home / End the
   ends, Enter opens the sheet; the tooltip follows and `aria-valuetext` names the stock.
5. ~~`renderFlagsWarnings()` writes to a `#modal-flags` element that does not exist~~ **DONE
   2026-10-27 (13:00 session).** All eleven badges audited against the live page: seven were
   already covered elsewhere, three became Data-provenance sentences with registered claims
   (`drilldown.ltm_annualized`, `drilldown.ev_substituted`, `drilldown.beta_overlap`), and the
   eleventh was deleted because no run had ever set it - which also made a sentence on the public
   methodology page false, corrected in the generator (`overview.no_financial_sector_flag`). The
   dead function is gone. The item was right that thresholds had to be registered first; they are.
   `tests/test_dead_badges.py` (18), `plan/dashboard-inventory.md`, NIGHTLY_LOG.md 2026-10-27.
6. ~~Bot residuals~~ **DONE 2026-10-27 (owner-run evening, screener-bot `7f4328f`)** - all four, plus the
   health ledger (ROADMAP 3.7); see the bot's SESSION_LOG.md. Was: the "Against what you could hold instead" header wraps to four lines on a phone; the
   one-mark empty chart is a large dashed box; ladder labels are not keyboard-reachable; white on
   the map's full gain tile is 4.42:1 (the colour is locked in `DESIGN.md`).
7. Seen in the fourth pass, not fixed: Factor analytics' sector table on a phone shows two score columns
   at a time behind a fade (it scrolls; a transposed phone layout would read better); Diagnostics' tile
   header wraps its chevron under the title on a phone; the table's Delta header is a bare symbol (its
   tooltip and "What the columns mean" explain it).
