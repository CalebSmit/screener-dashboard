# Dashboard redesign - master plan, every surface

**Created 2026-10-06**, after stage 1 shipped. Owner direction, same day: *"get it
set up with a full blown plan to make it look better, every single place on the
dashboard."* The queue entry is in `OWNER_FOCUS.md`; the token-level argument is in
`plan/dashboard-design-system.md`; **this file is the surface-by-surface plan and
the order of work.** The calculation-transparency work (`plan/calculation-transparency.md`)
meets this plan at the stock drilldown and is sequenced with it below.

The governing idea is unchanged: **premium reads as restraint.** Neutrals carry the
UI, one accent carries interactivity, colour appears only when it *means* something,
and everything that is not helping a reader decide is removed rather than restyled.

## How to use this file

- One **stage** is one session's work and ends committed. If a stage is too big for
  the session you have, split it at a step boundary, commit, and log where you
  stopped - the next session starts from the log, not from scratch.
- Every session starts with `scripts/shot_dashboard.py <label>` (desktop 1440,
  mobile 375, drilldown, plus DOM / transition / font census) and **looks at the
  images**. Say in `NIGHTLY_LOG.md` what you saw. Every session ends the same way.
- **Presentation changes leave every number, rank and sentence byte-identical.**
  Verify with `scripts/diff_payload.py` (built in calculation-transparency stage T0b;
  until it exists, compare SHA-256 of `dashboard_data.js` as stage 1 did).
- Edit `generate_dashboard.py`, never the generated files (rule 10). Update
  `plan/dashboard-inventory.md` in the same session (rule 9).
- If a stage proves unnecessary, record why and move on - do not manufacture work.

## What was seen on 2026-10-06 (after stage 1), surface by surface

Looked at the live page at 1440px and 375px, and one bank (JPM) and one ordinary
name (EXPE) in the drilldown. These are observations to re-check, not a spec.
Surfaces marked **not yet inspected** were not opened in this audit; the first
session of their stage looks at them before changing anything.

| # | Surface | What is wrong now | Target |
|---|---|---|---|
| 1 | **Page shell**: header, section headers, chevrons, footer, scroll, anchors | A header *card* costs the first 110px for a title and two pills; section headers are bold titles with a blue tick and a chevron - every one the same weight; no way to jump between sections; native scrollbars and focus rings unreviewed. **The `Refresh Data` button is a dead control for every visitor but the owner:** `triggerRefresh()` opens an `EventSource` to `http://localhost:7720` (`generate_dashboard.py` ~3485), which only exists while `refresh_server.py` runs on the owner's machine, and its own tooltip says so | A quiet top bar (name, data date, Methodology link); sticky section nav / anchors; one section-header style; designed focus rings; page width and gutters from the spacing scale. **Remove the Refresh button from the public page** (the 02:00 data loop is the refresh; keep `refresh_server.py` working for local use if a test or the owner needs it, but a public page must not ship a control that cannot work or that probes a visitor's localhost) - check which tests pin it first |
| 2 | **KPI row** | Three large cards carrying `502`, `120`, `121`, with "24% flagged" printed twice; at 375px the third card orphans on its own row | One compact stat strip: universe, run date, data health, trap counts - information density up, area down; no orphan at any width |
| 3 | **Top 5 cards** | Eight identical blue bars per card, so the eye has nothing to compare; the ticker is large accent-blue; cards carry dead space at the bottom; at 375px it is a sideways carousel | One-hue score encoding with a deliberate scale; ticker, name, composite as the hierarchy; the eight categories as a compact aligned strip; vertical stack on mobile |
| 4 | **My Holdings** | The **empty state is a wall**: seven paragraphs of cited rationale in bold and yellow emphasis open on the first screen of the panel. The research is good and belongs behind a disclosure | Empty state = one sentence plus the add field; the seven rationales move behind "Why this panel works this way", plain weight, no yellow; populated rows keep every property pinned by `test_holdings_panel.py` |
| 5 | **What Changed** | 15 + 15 rows at once; a dense footer paragraph; sparklines are fine | Top 5 each way with "Show all"; footnote behind a disclosure; the comparison-window toggle as a proper segmented control |
| 6 | **Factor Analytics - Factor Scores by Sector** | **Eleven near-identical bars** (median composite is ~48-52 for every sector on an axis of 0-70), so the chart conveys almost nothing; y labels truncate ("ommunication Se...") | Decide whether this chart answers a question at all. Candidate: a sector x category matrix (median score, one-hue intensity) which is where sectors actually differ. Never truncate a label |
| 7 | **Factor Analytics - Trap Rate by Sector** | Bars in red / amber / green - status colour used as a ramp, which is exactly the decoration rule 3 forbids | One hue, sorted, value-labelled; status colour only for a threshold that means something |
| 8 | **Defensibility & Diagnostics** | Header carries three coloured badges; an intro paragraph; KPI-style cards; the correlation heatmap is built from red/amber/green translucent fills (`corrColor`) | Calm summary line; the heatmap in one diverging scheme that is validated for colour-blindness (`scripts/check_contrast.py`, dataviz palette), with the diagonal and legend explained once |
| 9 | **Rankings table** - *the workhorse* | 502 rows in a **nested scroll box**; bare numbers for eight category scores; native `<select>` filters; the Δ column shows a muted `0`; the Flags column shows a check mark that means *no trap flag* but reads as an endorsement, and flagged names show the bare abbreviations `VT` / `GT`; **8,050 of the page's 9,923 DOM nodes (81%) are inside this table** | Page-level scroll with a sticky header; restrained score encoding; designed filter controls with a visible "clear"; Δ shows nothing when nothing moved; a legend for flags; fewer than ~3,000 nodes at first paint; row hover / selected / keyboard states; result count and empty-result state |
| 10 | **Drilldown** | A 920px modal with **5,900px** of content; "Why it ranks here" is eleven sentences of equal weight; the score cards wrap to leave `Size` and `Investment` orphaned on a second row; the percentile-convention note is **printed eight times** (once per category); inactive metrics print as raw snake_case (`operating_margin, current_ratio, ...`); no navigation inside a very long panel. *Its first sentence is also false for every stock - see calculation-transparency "Defect 3".* | A side panel / sheet with a sticky header (ticker, name, rank, composite) and in-panel section nav; "Why it ranks here" gets a headline and a visual hierarchy; the eight categories as one aligned table; the convention note stated once; labels, not identifiers; built **together with** the calculation view (calculation-transparency T4) |
| 11 | **Drilldown blocks**: About, Rank History, Analyst Targets, Company Snapshot, Sector Peers, Data Provenance, Score Contribution | **Not yet inspected** in this audit | Each reviewed against the hierarchy above; Rank History and Targets are the two that most want a real chart treatment |
| 12 | **Charts**: `sector-dist-chart`, `vt-chart`, histogram, sparklines, rank history | One chart style per file; axes, tooltips, and typographic treatment not shared; colour assigned per chart | One chart module: shared axis / grid / tooltip / label rules, one palette, value labels instead of legends where possible, accessible text alternatives |
| 13 | **Methodology** (modal and embedded document) | **Not yet inspected** as a reading surface; it is ~30 headings and half the file | Long-form typography: measure (60-75ch), a table of contents with anchors, designed tables and code blocks, a calm modal. Content is untouched (the claims register owns its truth) |
| 14 | **States**: loading, empty holdings, no filter results, error, offline, N/A cells, stale-data banners | Mostly implicit or absent; N/A is rendered inconsistently (`N/A`, `—`, blank) | One vocabulary: a single glyph/word for "no data" and a different one for "not applicable to this stock"; every list has a designed empty state; a stale-run banner that uses the amber token |
| 15 | **Mobile (375px) and tablet (768px)** | Orphaned KPI card, header consuming the first screen, Top 5 sideways carousel, a wall of text in the drilldown, wide tables | A first-class layout, not a shrink: bottom-anchored drilldown sheet, tables that become stacked rows, 44px targets, no horizontal page scroll |
| 16 | **Motion and feel** | 662 elements carry transitions (stage 1 trimmed the worst); interaction latency never measured | Budgets below; named-property transitions only; `prefers-reduced-motion` honoured; nothing shifts under the cursor |
| 17 | **Accessibility** | Contrast checked for tokens only; keyboard path, focus order, roles and labels unreviewed; the modal's focus handling unknown | Full keyboard path (table -> drilldown -> back); focus trap and return in the sheet; roles/labels on controls and chart alternatives; contrast re-run on every pair in use |
| 18 | **Microcopy and number formatting** | Units, signs and abbreviations vary between surfaces (`$338.77`, `65.7%`, `-10.9%`, `1.7`) | One formatting module: thousands, `$B/$M`, signed deltas with true minus, consistent decimals per quantity; sentence case throughout |

## Measured budgets (re-measure; these are the targets, not the baselines)

| Quantity | Baseline 2026-10-06 | Target | Why this number |
|---|---|---|---|
| DOM nodes at first paint | 9,928 | **< 3,000** | The rankings table is 81% of the nodes (8,050 of 9,923, measured 2026-10-06); a table that renders only what is on screen is the cheapest fix |
| Elements with a CSS transition | 662 | **< 100** | Motion on controls and state changes only |
| Click-to-paint on a row / filter / sort | not measured | **< 100 ms** at the 95th percentile on a mid-range laptop profile (CPU 4x throttle) | The classic response threshold (Card, Moran & Newell; Nielsen); Google's INP rates <= 200 ms "good", so 100 leaves headroom |
| Largest contentful paint, cold | not measured | **< 2.5 s** on a throttled connection | The documented "good" LCP threshold |
| Cumulative layout shift | not measured | **< 0.05** | Nothing shifts under the cursor |
| Gzipped page + payload | ~1.2 MB | **flat, +/- 5%** (calculation-transparency may spend up to its own +150 KB budget separately) | A phone on 4G is the budget |
| Font families | Inter + one mono | Inter for everything; mono for code only | One family, one job |
| Text/surface contrast | AA for tokens | **AA for every pair in use**, via `scripts/check_contrast.py` | Whole-page, not token-level |

`scripts/shot_dashboard.py` already reports DOM nodes, transitioned elements and the
font census. **Extend it** (stage D2) to report click-to-paint and CLS with
Playwright's performance APIs, so "feel" is a number in the log every session.

## The stages, in order

Each stage lists what to look at first, the work, how it is accepted, and what in
the tests it touches. **Fifteen test modules pin dashboard markup, ids and JS names**
(`test_dashboard_surfaces`, `test_holdings_panel`, `test_stock_summary`,
`test_dashboard_js` with `node --check`, ...). Prefer CSS and layout changes; if a
pinned hook must move, change its test in the same commit and say why - never delete
an assertion to get green.

### D2 - The rankings table (surface 9)  *(next)*

Look at: the table at 1440 and 375, scrolling 502 rows, sorting, filtering, opening a
stock from a row.
Work: page-level scroll with a sticky header (remove the nested box); windowed or
chunked rendering so first paint carries only what is on screen (options: chunked
`IntersectionObserver` append, or a fixed-row-height windowed list - choose by the
measurements, and note that find-in-page, keyboard navigation and sort/filter must
keep working, which argues against `content-visibility` alone since it leaves the
nodes in the DOM); score encoding (one hue, intensity by value, number kept legible);
designed filter controls replacing native selects (or a native `<select>` restyled
to the system if it measures better on mobile - argue it); Δ blank when zero; a flags
legend; row hover / selected / `:focus-visible` states; result count and empty
state; flags shown as words ("Value trap", "Growth trap") with the absence of a flag
left blank rather than ticked; extend `shot_dashboard.py` with the feel metrics.
Accepted when: DOM nodes < 3,000, click-to-paint < 100ms, payload byte-identical,
all pinned table tests green.

### D3 - Drilldown shell **together with** calculation-transparency T4 (surfaces 10, 11)

This is one stage, not two: restyling the old category tables and then replacing them
with the calculation view would pay for the same surface twice. **Prerequisites that
must have landed first:** calculation-transparency T0a and T0b (claims fixed, true weights) and
ideally T2/T3 (inputs, percentile context).
Look at: a bank, an ordinary name, a thin-coverage name, an N/A-heavy name, at 1440 and
375.
Work: side panel / sheet with sticky identity header and in-panel section nav; the
summary with a headline and hierarchy; the eight categories as one aligned table that
expands to metrics and then to inputs; the convention note once; labels not
identifiers; each block (surface 11) reviewed; chart treatment for Rank History and
Analyst Targets.
Accepted when: every level of the equation sums on screen and
`tests/test_calculation_reproducibility.py` is green; payload diff shows only added
keys; screenshots at both widths reviewed and described.

### D4 - Top 5, KPIs, My Holdings, What Changed (surfaces 2-5)

Work: the stat strip; Top 5 cards with one-hue encoding and a stack on mobile; the
Holdings empty state reduced to a sentence with the rationale behind a disclosure
(plain weight, no yellow - the content and its citations are unchanged and the
properties in `plan/dashboard-inventory.md` stay pinned); What Changed trimmed to five
a side with "Show all" and a segmented window control.
Accepted when: no orphan card at 320-1440px; the first screen on mobile shows the
rankings entry point; every holdings test green.

### D5 - Analytics, defensibility, shared chart module (surfaces 6-8, 12)

Work: answer first whether *Factor Scores by Sector* earns its place (it is flat as
built); a sector x category matrix is the likely replacement and must be argued from
what the data shows; Trap Rate in one hue; the correlation heatmap on a validated
diverging scheme; one chart module with shared axes, grid, tooltip and label rules; a
text alternative for each chart.
Accepted when: no truncated label, one palette, `check_contrast.py` clean on every
chart pair, the defensibility features of rule 7 all still present.

### D6 - Methodology reading surface and the state vocabulary (surfaces 13, 14, 18)

Work: reading typography, TOC and anchors for the methodology; one "no data" /
"not applicable" vocabulary; designed empty, error and stale-run states; the number
formatting module applied everywhere.
Accepted when: every empty list has a designed state, N/A renders one of two ways
everywhere, and the methodology is comfortable to read for ten minutes.
**Note:** the methodology *text* is generated (`generate_screener_overview`); change
presentation only here. Its truth is the claims register's job
(calculation-transparency T0a).

### D7 - Mobile as a first-class layout (surfaces 1, 15)

Work: the shell, sheet, stacked-row tables and 44px targets at 320, 375, 414 and 768;
no horizontal page scroll; the drilldown as a bottom sheet with a grab handle and
swipe / back-button dismissal.
Accepted when: a student can open the site on a phone, find a stock, read why it
ranks there, and open its workings without pinching or sideways scrolling - and the
log says so from screenshots at each width.

### D8 - Polish, accessibility, performance audit (surfaces 16, 17)

Work: a full keyboard pass; focus order and return; roles/labels; reduced-motion
audit; re-run every budget in the table; fix what misses it; a last look at every
surface at both widths, side by side with the references (Linear, Stripe Dashboard,
Vercel, Mercury) *as documented practice, not for copying*.
Accepted when: every budget in the table is met or the miss is logged with the
reason, and the owner's test passes - *would you screenshot it next to Linear or
Stripe without embarrassment?*

## Order against the rest of the work

| Night | Work | Why here |
|---|---|---|
| Next code session | **calculation-transparency T0a then T0b** (false sentences and claims register; then false weights - correctness defects on the public site) | Outranks presentation; small; unblocks D3 |
| then | **D2** rankings table | The surface people live in; biggest measured win (DOM nodes) |
| then | **T1** lineage registry | No visible change; unblocks T2 |
| then | **T2 / T3** inputs and percentile context | Payload work that D3 consumes |
| then | **D3 + T4** drilldown with the calculation view | The two meet here |
| then | **D4, D5, D6** | Independent of each other; any order |
| then | **D7, D8, T5, T6** | Mobile, audit, independent checker, provenance |

At one stage per code session that is roughly eight to ten sessions. The weekly
rotation still applies around it: **Mondays** write a research note (design practice
for D-stages; how index providers expose factor construction for T-stages -
`plan/calculation-transparency.md` names the sources to start from), **Wednesdays**
check the new look still *explains* the categories and sector-relative percentiles
correctly, **Thursdays** build, **Fridays** harden and teach, and every other Friday
is the retrospective, which judges whether this plan is working and may rewrite it.

## Constraints that do not move

- **No number, score, rank or sentence changes meaning** in a presentation stage.
- **Decision support, not advice** - nothing reads as a recommendation;
  `BANNED_TERMS` stays enforced; the defensibility features (rule 7) stay.
- **Stay a single static page built by `generate_dashboard.py`** - no framework, no
  build step, no new runtime dependency.
- **Keep working offline from the file once loaded**; if webfonts are used they are
  subset and the fallback stack looks right.
- **Look before and after, at 1440 and 375, and describe it.** Judging from CSS is
  how the page got to look generated in the first place.
- **Commit after every stage.** The 2026-10-06 session was cut off by a usage limit
  with nothing committed.

## When this plan is done

The owner opens it and it *feels* like a product someone paid for - calm, confident,
fast, and obvious where to look - on a laptop and on a phone, and any number on it
can be followed to its inputs. The retrospective decides when that is true and may
close the `OWNER_FOCUS.md` items.
