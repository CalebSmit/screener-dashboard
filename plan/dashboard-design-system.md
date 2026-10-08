# Dashboard design system

**Created 2026-10-06**, the first build session of the owner's 2026-10-05
"premium" directive (`OWNER_FOCUS.md`). This file is the written argument for
every token: what it is, why that value, and the source. Later sessions change
a token here *and* in `generate_dashboard.py :root` together, or not at all.

The governing idea, from the owner's brief and the documented practice it
names (Linear, Stripe Dashboard, Mercury, Apple's financial surfaces):
**premium reads as restraint.** Neutrals carry the UI; one accent carries
interactivity; color otherwise appears only when it *means* something
(direction of change, a caveat, a trap flag). Decoration is what "AI slop"
pattern-matches on - rainbow rules, a hue per category, glows, staggered
entrance animation - and all of it is removed, not restyled.

## Where the values come from

The neutral ramp, ink roles, accent, and status colors are taken from the
validated dark-mode reference palette in Anthropic's dataviz skill
(`references/palette.md`), which ships with a runnable six-check validator
(lightness band, chroma floor, CVD separation, normal-vision floor, WCAG
contrast). Supplementary contrast ratios for the exact text/surface pairs this
page uses were computed with `scripts/check_contrast.py` (WCAG 2.x formula) on
2026-10-06 - every token below that carries text passes **AA 4.5:1** on every
surface it sits on, with the two deliberate exceptions noted.

That palette is deliberately *not* GitHub's dark theme (`#0d1117/#161b22/...`),
which is what the page used until today and the single biggest "generated
dashboard" tell the owner's brief measured.

## Color tokens

| Token | Value | Role |
|---|---|---|
| `--bg-deep` | `#0d0d0d` | page plane |
| `--bg-primary` | `#141413` | modal ground |
| `--bg-card` | `#1a1a19` | card/surface |
| `--bg-card-hover` | `#202020` | hover wash |
| `--bg-elevated` | `#222221` | raised wells (inputs, tracks, badges) |
| `--border` | `#262625` | hairline dividers |
| `--border-bright` | `#343433` | control borders, emphasized rules |
| `--text-primary` | `#ffffff` | headings, key figures (17.4:1 on card) |
| `--text-secondary` | `#c3c2b7` | body, labels (9.7:1) |
| `--text-muted` | `#8e8c86` | fine print (5.1:1 card; 4.7:1 raised). Was `#898781` until 2026-10-07; raised one step because 4.4:1 on raised surfaces missed AA |
| `--accent` | `#3987e5` | fills, active states, focus, "you" markers |
| `--accent-text` | `#5598e7` | accent as text - links, tickers (≥5.3:1 everywhere) |
| `--accent-glow` | `rgba(57,135,229,.12)` | focus ring / selected wash only (name kept for compat; no glows) |
| `--green` | `#0ca30c` | positive change, up-delta (≥4.75:1) |
| `--red` | `#e66767` | negative change as text (≥4.9:1) |
| `--red-strong` | `#d03b3b` | negative emphasis fills/borders (3.6:1, UI-component grade) |
| `--green-dim` / `--red-dim` / `--amber-dim` | 12% washes | chip backgrounds |
| `--amber` | `#fab219` | caveat marks (input churn, stale data) (9.5:1) |

**Rules with teeth, from the dataviz skill's non-negotiables:**

- *Text wears text tokens, never the series color.* Scores, labels and values
  are ink; a colored mark beside them can carry meaning.
- *Status colors are reserved.* Green/red mean direction of change or
  good/bad state - never decoration, never "the Quality category is green".
- *Sequential = one hue.* Both charts are single-series, so they get the
  accent, not a palette.

**What this deletes:** the per-category hue map (8 saturated hues whose only
job was telling apart rows that are already labelled), the 11-hue sector
pill map, the rainbow header rule, the rainbow contrib-total gradient, and
every `box-shadow` glow. Category and sector identity ride labels and
position, which is how Linear/Stripe-class tables do it.

## Typography

| Token | Value | Job |
|---|---|---|
| `--font-body` | `'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif` | everything: UI, body, headings, **and figures** |
| `--font-heading` | same stack (kept as alias) | headings differentiate by weight/size, not family |
| `--font-mono` | `'JetBrains Mono', ui-monospace, monospace` | literal code only: methodology code blocks, the localStorage key |

- **Two families, each with a job** (the owner's constraint). Three families
  with no roles was the measured state; the heading face (Space Grotesk) and
  body face (DM Sans) collapse into Inter, a UI typeface designed for screens
  at text sizes and the de-facto standard of the product class the owner
  pointed at.
- **Figures are Inter + `font-variant-numeric: tabular-nums`** (CSS Fonts
  Level 4), not monospace. Tabular figures give the vertical alignment that
  was the only legitimate job mono was doing, without setting 6,400 elements
  in a code face. This is standard financial-UI practice: IBM Carbon's data
  table guidance and Apple's HIG both specify tabular/monospaced *digits* for
  numeric columns - neither sets labels or prose in a code face.
- **Base size 14px** (was 12px-dominant: 5,680 of ~9,900 elements at 12px,
  measured 2026-10-06). 14px body for data-dense "productive" surfaces is the
  documented norm: IBM Carbon `body-01` = 14px, Material `body-medium` = 14sp,
  GitHub Primer body = 14px. Micro-caps labels floor at 11px with letter
  spacing; 9-10px sizes are removed.
- Webfonts load from Google Fonts css2 (`display=swap`, latin subset via
  unicode-range, weights 400/500/600/700 Inter + 400/500 JBM only - down from
  12 weight-files to 6). The fallback stack is metric-compatible system UI.

## Space, shape, elevation

- `--gap: 16px` stays the spacing quantum (multiples: 4/8/12/16/24).
- **One radius:** `--radius: 8px`; `--radius-pill: 999px` for pills/chips.
  (Was a mix of 3,4,5,6,8,10,12,14,20px.)
- **Elevation is for things that float:** one overlay shadow
  (`--shadow-overlay`) for the modal and dropdowns. Cards sit flat with
  hairline borders - no resting shadows, no glow halos.

## Motion

- Tokens: `--t-fast: 120ms`, `--t-base: 160ms`, both `ease-out`. Nothing
  animates longer than 200ms except nothing.
- **Entrance animation is gone** - fadeUp/fadeIn/slideRight staggers,
  barGrow, glowPulse are deleted. Motion communicates a state change the user
  caused; a page assembling itself on load is decoration and costs perceived
  speed. (Material motion: "transitions... quick, 100-200ms"; Apple HIG:
  "use motion purposefully".)
- The modal keeps one 160ms ease-out fade/8px rise - it is a caused state
  change.
- `@media (prefers-reduced-motion: reduce)` disables all transitions and
  animations (WCAG 2.3.3 Animation from Interactions; Media Queries L5).
- `transition: all` is banned; transitions name their properties.

## Measured, 2026-10-06, committed page (the "before")

From `scripts/shot_dashboard.py` (playwright, 1440px): **9,928 DOM nodes**,
**662 elements carrying CSS transitions**, JetBrains Mono rendering on
**6,415 elements**, dominant font size **12px (5,680 elements)**, 3 families.
Re-measure after every design session; the numbers belong in the log.

## What stage 1 (this session) deliberately does not touch

The rankings table's nested scroll box and 502-row DOM weight (stage 2), the
drilldown hierarchy (stage 3), mobile layout (stage 4) - per the owner brief's
own ordering. The token system is what makes those changes cheap.
