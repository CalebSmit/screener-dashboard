# What is the ranking for, given no measured edge? - the question and its pre-registration

2026-11-19 (owner-run, night). **CLAUDE.md priority 0.15**, opened the same evening on the owner's
delegation. This note does **not** answer the question. It writes down, before anything is
measured, what the candidate answers are and what result would count for or against each - so the
Monday session that measures cannot choose its reading after seeing the numbers.

**Proposal status: OPEN** - a Monday research session completes sections 4-5 against the criteria
in section 3, which may not be changed after the first measurement runs. Due: the Monday of
**2026-11-23** (the brief's "Coming up" carries it).

## 1. The fact that makes this the question

The research engine rebuilt 176 months of 2012-2026 from the filings as they stood each month and
asked whether the screener's ranking orders the **next month's** returns. Pooled monthly rank IC
-0.001 (95% interval -0.018 to +0.015 before tonight's rerun; -0.00133 after); none of 27
pre-registered alternative weightings did better; the harness fed leaked data finds t 4.7, so it
can see a signal when there is one (`research/2026-10-15-research-engine-first-batch.md`,
`research.html`). The bot's own rule lagged an equal-weighted S&P 500 in every period tested.

Two things that result does **not** say, and which this note exists to separate:

- It tests **one horizon** (one month) and **one use** (ordering all ~500 names). The screener's
  heaviest categories - Valuation and Quality, 63% of what moves the composite (finding 7 of the
  review) - are the ones the literature finds slow: value's premium is earned over years, not
  weeks (Asness, Moskowitz & Pedersen 2013, *Value and Momentum Everywhere*), and gross
  profitability predicts returns at horizons well beyond a month (Novy-Marx 2013, *The Other Side
  of Value*). A one-month rank IC is the test most likely to miss a slow signal.
- It is measured without the analyst inputs (Revisions, 10%) and without the companies that left
  the index (survivorship) - both stated on `research.html`.

And one thing it does say: in practice an IC of 0.05 is considered good and 0.02-0.03 ordinary for
a monthly cross-sectional forecast (Grinold & Kahn, *Active Portfolio Management*, 2nd ed. 2000,
on the "fundamental law"); an interval spanning -0.018 to +0.015 rules out the good case and
cannot rule out the ordinary one. "No measured edge" is not "measured no edge".

## 2. The candidate answers

| | What the ranking is for | What the page would then say | What it is judged on |
|---|---|---|---|
| A | **A screen**: narrow 500 to ~40 worth a student's reading time | "start your research here", never "these will do better" | Whether the top 40 avoid the worst outcomes (not whether they beat the rest on average) |
| B | **A slow forecast**: the composite orders returns over 6-12 months, not 1 | a horizon on every claim of performance; the review cadence becomes the horizon | Rank IC at 3, 6 and 12 months, non-overlapping |
| C | **Fewer, better factors**: some categories carry the signal and the rest add noise | a smaller, more teachable composite | Category-by-category IC at the horizon B finds, against the composite |
| D | **A teaching tool only**: no return claim at all | the track-record tile becomes a lesson in why short records mean nothing | Nothing to measure: the default if A-C all fail |

These are not exclusive; A and B can both hold.

## 3. Pre-registered criteria (fixed 2026-11-19, before any measurement)

All on the research engine's existing monthly records (`backtest_v2`, 2012-2026, both eras
reported separately and pooled), with its existing look-ahead positive control run alongside -
**a result only counts if the positive control clears its own bar in the same run.**

- **A (screen) holds** if, over non-overlapping months, the top 40 by composite fall in the
  **bottom decile** of next-month return less often than 10% of the time with a one-sided 95% interval
  whose upper end is below 10% - in **both** eras. (A screen that steers readers away from the worst
  names is useful even with no average edge.) Fails otherwise.
- **B (slow forecast) holds** if the pooled rank IC at **6 or 12 months**, measured on
  non-overlapping windows only, has a 95% interval entirely above zero **and** the same sign in both
  eras. With ~29 non-overlapping 6-month windows and ~14 12-month ones, the engine's own power
  calculation must be stated first; if it cannot detect 0.05 at those counts, B is recorded as
  **undecidable**, not as failed.
- **C (fewer factors) holds** only if B holds, and then if a category's own IC at B's horizon is
  above the composite's with a paired difference whose interval excludes zero in both eras. One
  category at a time, Bonferroni across eight.
- **D** is the answer if A, B and C all fail or are undecidable.

**What may not happen:** no new weighting is fitted to these results (rule 5 holds to 2027-02-11;
the research engine reports, it does not set weights); no horizon is added after the first run;
no criterion is relaxed to let an answer through. A finding that the engine cannot decide is a
finding.

## 4. Measurement (to be completed by the Monday session)

*Empty on purpose.* Register each test in `research/experiments/registry.yaml` and commit it before
it runs, as every engine experiment is.

## 5. What it would change on the page (to be completed after section 4)

*Empty on purpose.* Whatever the answer, the home page already states the one-month result
(`page.hero_tested`, 2026-11-19). If A holds, the hero's framing moves from "ranked" toward
"screened". If B holds, every performance figure carries its horizon. If D, the track-record tile is
reframed as a lesson.

## Sources

- Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and Momentum Everywhere. *Journal of
  Finance* 68(3).
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium. *Journal of
  Financial Economics* 108(1).
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
- The project's own: `research/2026-10-15-research-engine-first-batch.md`, `research.html`,
  `research/2026-11-19-fresh-eyes-review.md`.

*Not re-read tonight:* the three published works are cited for their well-known headline findings
only; the Monday session should check each claim against the paper before relying on it, per
`prompts/nightly.md`'s reading rule.
