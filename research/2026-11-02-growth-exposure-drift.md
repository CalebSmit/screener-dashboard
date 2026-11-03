# The ranking's growth exposure drifts: 13% is not a neutral point, it was one

2026-11-02, owner-run session. Answers the third bullet of section 11 of
`research/2026-10-12-growth-category.md` ("what the ranking's growth exposure does over time")
and opens the second ("is a score weight the right instrument for a neutrality the screener
wants?") as a proposal.

**Proposal status: DECIDED 2026-11-03** (06:00 scheduled session, not the one that wrote this) -
**option 3 accepted, options 1 and 2 refused as responses to this evidence, and the premise all
three rest on does not survive the longer record.** The drift is not the cross-section moving away
from a neutral point: it is the decay of a step this screener caused itself on 2026-10-09, back to
the level that held for the two months before. Section 6 is the review; nothing was re-fitted and
no weight moved.

No return is read anywhere in this note (CLAUDE.md rule 4). No weight changed.

## 1. What happened

The 2026-11-02 02:00 data run was refused at its last gate: `tests/test_growth_category.py::
test_the_ranking_really_is_close_to_growth_neutral` measured the composite's Spearman correlation
with the Growth score at **+0.058**, outside the 0.05 band the 2026-10-13 review set for the
methodology page's sentence "at 13% the ranking taken whole sits close to neutral, leaning neither
way". The gate did what it was built to do: the sentence was false for that run, so the run was
not published. The test's own docstring names the two ways out - "either the run has changed
character or the paragraph needs rewriting".

## 2. The measurement

The composite's Spearman correlation with the Growth score, every published run since the review
(each computed from that day's committed `dashboard_data.js`):

| Run | 10-09 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-29 | 10-30 | 11-02 (refused) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rho | -0.011 | -0.012 | -0.009 | -0.008 | -0.004 | -0.008 | +0.002 | +0.013 | +0.017 | +0.022 | +0.028 | +0.030 | +0.031 | +0.038 | +0.049 | **+0.058** |

`research/measurements/2026-10-13-growth-weight-sweep.py`, pointed at the 10-30 and 11-02 runs
(the script reads whichever payload is committed; for 11-02 a copy pointed at the rehearsal's
regenerated payload):

| | 2026-10-13 (the review) | 2026-10-30 | 2026-11-02 |
|---|---|---|---|
| exposure at 13% | -0.009 | +0.049 | +0.058 |
| weight at which exposure is zero (freed points to Quality / pro rata) | 13.5 / 13.3 | 10.2 / 11.0 | 9.6 / 10.6 |
| weights with \|exposure\| <= 0.05 | 11-16 | 8-13 | 7-12 |
| slope near 13 (correlation per point) | +0.0181 | +0.0173 | +0.0178 |
| Growth against the mean of the other seven | -0.388 | -0.348 | -0.342 |

Growth's correlation with each other category's score:

| | Valuation | Quality | Momentum | Risk | Revisions | Size | Investment |
|---|---|---|---|---|---|---|---|
| 10-13 | -0.438 | +0.157 | +0.188 | -0.263 | +0.167 | -0.341 | -0.406 |
| 11-02 | -0.367 | +0.191 | +0.163 | -0.255 | +0.186 | -0.297 | -0.413 |

## 3. What it settles

- **The structure the review relied on held.** Growth is still the one category running against
  the others (-0.342), and the slope is unchanged (+0.018 a point): its weight is still the
  growth-exposure dial.
- **The dial's zero moved 3.9 points in three weeks.** The offset Growth works against weakened -
  mostly Valuation's (-0.438 to -0.367, the largest category weight) - so the same 13 points now
  over-correct. The drift is gradual and one-directional across fifteen runs, including the six
  between 10-22 and 10-30 when no methodology change landed, so it is the cross-section moving,
  not one bad fetch. (Five changelog entries changed how scores are computed between 10-14 and
  10-21, mostly for banks; they may account for part of it. This note does not separate the two.)
- **"At 13% the ranking sits close to neutral" was a fact about one run, stated as a property of
  the weight.** Section 11 of the growth note foresaw exactly this ("-0.01 is one run ... whether
  the neutrality is stable or an accident of this month's cross-section") and chose to hold the
  band in a test until it was known. It is now known: it is not stable at the scale of weeks.

## 4. What it does not settle, and why the weight did not move tonight

Re-fitting the weight to today's neutral point (about 10) would be the 2026-10-13 proposal's
number arrived at by a different route - and it would be a fit to a measurement that moved four
points in three weeks, so the next three weeks could undo it. A weight that is re-fitted whenever
a correlation drifts is a rule, not a weight, and it needs to be designed as one: how often, on
what, with what band, and how it is explained to a student reading the page. That is the open
question, and it is a Monday's, with the literature and practice behind it:

1. **Re-fit on the review cadence.** Monthly, to the neutral point of the run that month, stated
   as a rule. Cheapest to build; the ranking's weights then change every month.
2. **Constrain the exposure instead of the weight** - MSCI's approach (section 11 of the growth
   note): neutralise the composite against the Growth score after it is built. Exact every run,
   but a step a student has to be taught, and it touches row 0.8 (the page must reproduce it).
3. **State the lean and leave the weight.** The weight was set where the review found neutral;
   the page says the lean moves and reports it. Honest and simple, but gives up neutrality as a
   design goal it was the reason for keeping 13.

What shipped tonight is only the honest version of the sentence, so the data loop can publish:

- The methodology page no longer says the ranking "sits close to neutral, leaning neither way" as
  a standing fact. It says what the review measured, that the balance moves with the market, the
  2026-11-02 reading, and that how to hold it is open (this note).
- The publish-gate test that held the ranking to the 0.05 band is replaced by tests that hold the
  new sentence true: the page's dated figures are in this note, the old sentence is in `claims.py`
  `FORBIDDEN`, and the review's structural claim (Growth runs against the rest) is still checked
  against every published run.
- The morning brief reports the published run's exposure every day, against the review's band, so
  the drift stays in front of whoever decides this proposal.

## 6. The review, 2026-11-03 (06:00 scheduled session)

Reviewed by a session that did not write the note, as the routine requires. The note is careful
and its arithmetic reproduces. What it does not survive is its own unanswered question, which
section 3 states and sets aside:

> "Five changelog entries changed how scores are computed between 10-14 and 10-21, mostly for
> banks; they may account for part of it. This note does not separate the two."

Separating them is what this review did, and the answer is not the one the sentence expects.

### 6.1 It is not the banks, and it is not a drift

`research/measurements/2026-11-03-growth-drift-decomposition.py` recomputes the exposure on every
committed payload with Financials removed - the cohort those five entries rescored. If the
screener's own bank edits were behind the move, the exposure outside Financials would be flat. It
is not: **excluding Financials the rise over the same window is larger, +0.0949 against +0.0814**
(10-13 to 11-03). The banks damped it slightly. And the exposure keeps rising through 10-23 to
10-30, five runs that share one `kpis.methodology_version` (`8ff931b2a52c`), so no scoring rule
changed underneath them.

So the note is right that the banks are not the story. The story is one run earlier than any
window it looked at.

### 6.2 The decisive measurement: what the neutral weight has actually been

`research/measurements/2026-11-03-neutral-weight-history.py` runs **this note's own sweep**, the
2026-10-13 script, unmodified, over every published payload rather than over three of them. The
Growth weight at which a run would be exactly growth-neutral (freed points to Quality, the other
seven held at today's configuration):

| period | runs | neutral weight |
|---|---|---|
| 2026-08-12 .. 2026-10-09 **morning** | 40 | **8.8 - 12.0**, mean 9.9 |
| 2026-10-09 **evening** | 1 | **13.6** |
| 2026-10-12 .. 2026-10-16 | 5 | 13.2 - 13.7 |
| 2026-10-19 .. 2026-10-30 | 8 | 12.9 down to 10.2 |
| 2026-11-03 | 1 | **8.8** |

The exposure at the configured 13% tells the same story from the other side: **+0.019 to +0.076 on
all 40 pre-10-09-evening runs**, then -0.012, and **+0.073 today** - back inside the old range, at
the top of it.

The step is one run wide. The 2026-10-09 morning run is neutral at 10.2; **the evening run the
same day is neutral at 13.6**. Between them is `METHODOLOGY_CHANGELOG.md` 2026-10-09, five
entries changing how `operating_leverage`, `revenue_growth`, Piotroski 3/8/9, EBITDA and
`forward_eps_growth` are computed - four of the five inside Growth or Quality, which is where
Growth's offset against Valuation lives. Growth against Valuation moves -0.322 -> -0.443 across
that same pair of runs and is -0.359 today.

### 6.3 What that does to the proposal

**The premise fails.** All three options are framed as answers to "a fixed weight has drifted away
from neutral". The record says 13% was **never** the neutral point: it was neutral for about six
runs, beginning four days before the review that measured it and ending about a week after. The
review read a transient as a property. That is not a criticism of the review - it had five runs of
record at the time and section 11 of `research/2026-10-12-growth-category.md` said in writing that
it did not know whether the neutrality was stable. It now does.

**Option 1, re-fit on the review cadence - REFUSED.** This is the one the measurement kills
outright. A rule that re-fits Growth to the month's neutral point would have set it to ~13.5 in
mid-October (where it already was, by coincidence), ~11.4 in late October and ~8.8 today: **three
moves in a month, every one of them tracking the decay of this screener's own 2026-10-09 edit.**
A weight that follows a correlation is not a weight, and here it would have been following an
artifact of its own methodology changes rather than anything about the market. It also fails the
teachability test the mandate sets: "we re-fit the growth weight monthly to whatever makes the
ranking neutral" is a sentence a student cannot check without rerunning the sweep.

**Option 2, constrain the exposure after the composite is built (MSCI's approach) - REFUSED as a
response to this evidence**, and explicitly *not* refused on its merits. It is the right
instrument for a persistent, exogenous drift. There is no persistent drift here: at the configured
weight the exposure is stationary in a +0.02 to +0.08 band across 40 pre-change runs and has
returned into it. Building a neutralisation step - which touches row 0.8, because the page must
then reproduce it - to hold a quantity that is already stationary is cost without a problem. It
stays where the 10-12 note left it, as an open design question about the *instrument*, to be taken
up if the exposure ever leaves that band and stays out.

**Option 3, state the lean and leave the weight - ACCEPTED.** It is also what shipped on 11-02, so
accepting it is ratification rather than a build. One thing in it was wrong and is corrected on the
page by this review: it said the balance "moves as the scores Growth offsets move **with the
market**". The record does not support that attribution. The one large move in fifteen months of
published runs was this screener's own, and the page now says so, with the measured range.

### 6.4 What this review adds that nobody should have to find again

- **The neutral weight has a long-run level and it is about 10, not 13.** This is recorded, and
  deliberately not acted on. It is suggestive - the 2026-10-12 proposal that was refused on
  2026-10-13 asked for exactly 10 - but a weight is not set by fitting a correlation (that is
  what option 1 was just refused for), and the 10-13 refusal rested on the evidence for what
  growth *earns*, which this measurement says nothing about. If a Monday reopens Growth's weight,
  this series is a fact it must account for; it is not on its own a reason to move anything.
- **A methodology change moves the ranking's factor exposures, and nothing here measures that.**
  The 2026-10-09 evening entry measured rank changes and score changes - carefully - and the
  exposure shift it caused was invisible until this note's series existed, three weeks later.
  Worth one line in the changelog's measurement habit: when a metric inside Growth or Quality
  changes, record the composite's correlation with each category score before and after.
- **A band test on a published run can be a test of an artifact.** The gate that refused the
  11-02 run was holding every run to within 0.05 of a neutrality created by this project's own
  edit six days earlier. Replacing it with sentence tests (which 11-02 did) was right for a
  different reason than the one given.

## 5. Sources

- `research/2026-10-12-growth-category.md` sections 10-11 (the review, the band, the open questions).
- `research/measurements/2026-10-13-growth-weight-sweep.py` (unchanged; run on the 10-30 payload as
  committed and on the refused 11-02 payload as regenerated by `scripts/rehearse_publish.py`).
- The series in section 2: `git show <commit>:dashboard_data.js` for each publish, Spearman of
  `Composite` against `growth_score` over `table_data`.
