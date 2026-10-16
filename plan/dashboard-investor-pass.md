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

## Next (in order)

1. **Typography and rhythm**: one type scale across the page (title 40 / section 17 / body 15 / caption
   12.5), uppercase labels used only for tile labels, consistent 14px radii on panels.
4. **research.html and lab.html** in the same voice as the front page (their bars and headers).
