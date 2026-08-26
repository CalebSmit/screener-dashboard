# Investor Profile Selector — client-side reweighting

**Owner:** Wednesday nightly sessions
**Status:** Not started
**Decide before shipping:** `presets.py` already defines server-side Value /
Growth / Momentum weight vectors, and they differ from the profile weights
originally sketched (e.g. Value is 30/25/8/10/10/8/5/4 in code vs
35/25/5/5/15/10/3/2 as sketched). Shipping both means `--preset value` and the
dashboard's "Value" produce different rankings under the same name - the worst
kind of inconsistency for a tool that sells auditability.

You now have authority to resolve this yourself. **Recommended:** generate the
client-side profile JSON from `presets.py` at dashboard-build time so there is
exactly one definition of each named profile, and add a Long-term/Quality
preset to complete the set. Record the decision in `METHODOLOGY_CHANGELOG.md`.
Add a regression test asserting the CLI and dashboard rankings match for the
same profile.

Note also: the sketched Momentum profile zero-weights Size and Investment,
which is a stronger claim than re-tilting - it asserts the SMB and CMA proxies
carry no information at short horizons. Defensible, but argue it in the
changelog rather than letting it happen because the numbers summed to 100.

## Goal

Let a dashboard user switch between named investor profiles (Value, Growth,
Long-term/Quality, Short-term/Momentum, Balanced) and see the Full Universe
table and the Model Portfolio view re-rank instantly — entirely in the browser,
with no Python re-run and no change to the scoring pipeline.

## Why this is safe to build nightly

The whole feature is front-end. It reads category scores that are already in
the payload and recombines them. It cannot alter `factor_engine.py`,
`config.yaml`, or any scored output. That containment is the point — nightly
work on this can't break the pipeline.

## What the data already gives us

`dashboard_data.js` → `window.SCREENER_DATA`:

- `table_data` — **all 501 stocks**, each with `valuation_score`,
  `quality_score`, `growth_score`, `momentum_score`, `risk_score`,
  `revisions_score`, `size_score`, `investment_score`, plus `Composite`,
  `Rank`, `Value_Trap_Flag`, `Growth_Trap_Flag`, and both severity fields.
- `weights.factor_weights` — the Balanced weights the payload was scored under.
- `portfolio.holdings` — current 14 holdings.
- `stock_detail` — **all 501 stocks** (verified 2026-08-05), each with `raw`
  metric values, `pct` percentiles, `cat_scores`, per-category `contrib`
  attribution, `peers`, analyst price targets, `financials`, `flags`, and
  `data_source` provenance.

Everything needed for reweighting is present. No payload change is required for
milestones 1-3.

## Three things that are easy to get wrong

### 1. `Composite` is a percentile rank, not the weighted sum

Per `README.md` and `SCREENER_OVERVIEW.md`, the pipeline computes a raw weighted
sum of category scores and then converts it to a cross-sectional percentile:
`rank(pct=True) * 100`. A score of 95 means "better than 95% of the universe."

So the client-side recompute is **two steps**, not one:

```js
// 1. raw weighted sum
const raw = CATEGORIES.reduce((s, c) => s + w[c] * row[`${c}_score`], 0) / 100;
// 2. percentile-rank across the full scored universe, ties averaged
```

Skipping step 2 produces numbers on a different scale from every other
composite in the app, and the Balanced profile would not reproduce the
server's own published numbers.

**Acceptance test for milestone 2:** selecting Balanced must reproduce the
existing `Composite` and `Rank` for all 501 rows to within floating-point
tolerance. If it doesn't, the transform is wrong. Use `pandas.rank(pct=True)`
tie semantics (average ranks on ties).

### 2. Trap flags are weight-invariant — but trap *exclusion* still applies

Trap detection thresholds on category-score percentiles (quality / momentum /
revisions floors; growth ceiling), never on the composite or on
`factor_weights`. Reweighting does **not** change any flag, and the UI should
not claim otherwise. (Verified 2026-08-05 against `config.yaml`; do not "fix"
this non-problem by adding a disclaimer.)

However `config.yaml` sets `value_trap_filters.flag_only: false`, meaning
flagged stocks are **excluded from the model portfolio** server-side. Any
client-side rebuild of the Top-5 / Model Portfolio view must apply the same
exclusion, or it will display a portfolio the pipeline would never construct.
The Full Universe table keeps showing trapped stocks, flagged, as it does now.

### 3. The Model Portfolio is more than a top-N slice

`portfolio_constructor.py` applies sector constraints, position caps, and a
weighting scheme — it is not "the top 5 by composite". Do **not** present a
client-side re-ranked top-N as a Model Portfolio. Milestone 3 shows a re-ranked
**Top Names** list and labels it as such; a faithful client-side portfolio
construction is a separate, later question and may not be worth doing at all.

Good news for this feature: `stock_detail` covers the whole universe, so any
stock that rises into the top ranks under a non-Balanced profile already has a
full detail payload. No graceful-degradation path is needed.

One real consequence though: `contrib` (per-category contribution to the
composite) is computed under **Balanced** weights. Under a different profile
those contribution numbers are wrong and must be recomputed client-side from
`cat_scores` × the active weights, or hidden. Showing stale contributions next
to a re-ranked composite would be actively misleading.

## Milestones — one Wednesday each, roughly

**M1 — Profile definitions plumbed through, no UI.**
Add a profile registry to `generate_dashboard.py` that emits a
`SCREENER_DATA.profiles` object. Source the weights from `presets.py` so there
is one definition (pending the owner's decision on the proposals entry). Ship
with Balanced only if the decision is still open. Tests: registry emits valid
vectors, each summing to 100.

**M2 — Recompute engine, verified against the server.**
Pure function: `(rows, weights) -> [{ticker, raw, composite, rank}]`. No DOM.
Unit-test that Balanced reproduces the published `Composite`/`Rank` exactly.
Still no visible UI.

**M3 — Selector UI + table re-sort.**
Pill toggle near the existing filters. On change, recompute, re-sort the Full
Universe table and the Top Names list, and update rank-delta indicators.
Persist the choice in `localStorage`. Keep Balanced as the default so a
first-time visitor sees the published methodology. Show a clear label stating
which profile is active and that the pipeline's own portfolio is Balanced-based.

**M4 — Custom weights via sliders.**
Eight sliders, live renormalization to 100, reset-to-profile, and a shareable
URL hash. This is genuinely new functionality and deserves its own week — do
not rush it into M3.

## Guardrails

- Edit `generate_dashboard.py`. Never hand-edit `dashboard.html`, `index.html`,
  or `dashboard_data.js` — they are generated, and `index.html` is what
  GitHub Pages serves.
- Do not regenerate the data payload to test; work against the committed one.
- The published Balanced numbers are the reference implementation. If a
  client-side result disagrees with the server, the client is wrong until
  proven otherwise.
