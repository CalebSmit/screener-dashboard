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

1. **Editorial typography.** Serious financial writing - the FT, the Economist, Bloomberg
   Businessweek, Stripe Press - pairs a reading serif for headlines with a precise sans for
   data. The screener is a teaching tool that *explains*; its headlines should read like a
   publication's, not a control panel's. Display serif for page titles, section titles and
   hero statements only; Inter stays the face of every control, label and figure.
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
| `--font-display` | `'Newsreader', 'Iowan Old Style', 'Palatino Linotype', Georgia, serif` | page titles, section titles, hero statements. Never a label, control, table or figure |
| Display weight / tracking | 500, `letter-spacing: -0.015em`, `font-optical-sizing: auto` | |
| Hero title size | `clamp(36px, 4.6vw, 56px)`, line-height 1.04 | |
| Section title size | `clamp(22px, 2.2vw, 28px)`, line-height 1.15 | |
| `--hairline-hi` | `inset 0 1px 0 rgba(255,255,255,.04)` | the top edge of a raised card; replaces nothing, adds depth without shadow |
| Heat (score magnitude) | `rgba(57,135,229, a)`, `a = 0.03 + 0.25 * score/100` | sequential, single hue (the accent), **magnitude only**: text stays ink, never coloured |
| Fonts URL | `https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=JetBrains+Mono:wght@400;500&display=swap` | loaded the existing non-blocking way |

Accent, ink, neutrals and the meaning colours are **unchanged** (`plan/dashboard-design-system.md`).
Amber stays the caveat colour, so no surface may use an amber/gold accent for interactivity.
The family's roles are now three, each with one job: display serif (headlines), Inter (UI and
figures, tabular), JetBrains Mono (literal code). The owner's 2026-10-05 complaint was three
families *with no roles*; that rule still binds.

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
| Dashboard page | display serif headlines and wordmark; **"498 companies, one dot each"** (the published composites, top 25 and trap flags marked, hover/tap names a stock, click opens it; one SVG built after first paint); leader cards as labelled score rows with a tick at 50; a small true chart on every home tile; heat-shaded score cells; phone list 96 -> 80 px a row (page ~6,900 px shorter); **open sections were capped at 5,000 px, so Reporting soon hid ~100 companies** - uncapped, day headings sticky; footer `#gen-time` labelled true UTC | `_css_page()`, `_js_page()`; `tests/test_dashboard_page_ui.py` (5) |
| Drilldown | masthead: ticker and sector line, the company name in the serif, rank / composite / price, and a strip of all 498 composites with this stock marked; trap chips in the header; heat score tiles with a median tick, unscored as a dashed outline; "Why it ranks here" with its figures emphasised (characters wrapped, never changed - tested on every stock); How it adds up as a receipt with a stacked total; one colour language for charts; folding header; Enter/Space on every card; drag the header down to close on a phone; the Data tab now lights; the rank-history label no longer overprints at 375 | `_css_drilldown()`, `_js_drilldown()`; `tests/test_drilldown_presentation.py` (10) |
| Factor Lab | the four rows of chips became a numbered index grouped by category (sticky side column >= 1080 px, a picker on phone) with each experiment's matched-figures bar; verdicts drawn as one cell per figure with the "no effect would match k" tick; tolerance marks per figure; loading state for lazy series; deep links open the experiment | `lab.html` |
| Has it been tested? | the headline answer drawn as IC estimates with 95% intervals and the check's lead; the trading rule's gap drawn the same way; tables stack on phone | `research_page.py` (`headline_figure`, `excess_figure`) |
| Bot dashboard | text inks at AA (tokens unchanged); direction shown by a triangle as well as colour; the tape scrolls with the page, pauses, stands still under reduced motion; one-row phone masthead and a theme switch; Replay named "Machinery" with its own band and per-session strip; chart end labels no longer collide; a withheld Sharpe shows a dash, never a count in the figure slot | `screener-bot` `bot/dashboard_template.html`, `DESIGN.md` |

New tool: **`scripts/rebuild_page.py`** rebuilds `index.html` / `dashboard.html` from the published
payload without being able to touch it; `tests/test_rebuild_page.py` fails if the committed page is
not what the generator builds.

### Left for a later pass (owner-run; nightly sessions do not take design work)

1. Phone Reporting soon hides the "Last quarter" and "Insiders" columns - decide whether phone
   readers should see them.
2. Phone Weighting select clips "Balanced (published)".
3. Phone Side-by-side puts the third stock in a sideways scroll.
4. The dot picture is not keyboard-focusable (the table stays the keyboard route; it has an
   `aria-label` summary).
5. `renderFlagsWarnings()` writes to a `#modal-flags` element that does not exist, so its Beneish
   and channel-stuffing badges have never rendered. Wiring it in would publish thresholds not
   registered in `claims.py` - register them first, or delete the dead code.
6. Bot: the "Against what you could hold instead" header wraps to four lines on a phone; the
   one-mark empty chart is a large dashed box; ladder labels are not keyboard-reachable; white on
   the map's full gain tile is 4.42:1 (the colour is locked in `DESIGN.md`).
