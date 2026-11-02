# The ranking's growth exposure drifts: 13% is not a neutral point, it was one

2026-11-02, owner-run session. Answers the third bullet of section 11 of
`research/2026-10-12-growth-category.md` ("what the ranking's growth exposure does over time")
and opens the second ("is a score weight the right instrument for a neutrality the screener
wants?") as a proposal.

**Proposal status: OPEN** - decide how the screener holds the ranking's growth exposure near
neutral now that a fixed 13% weight has drifted from -0.01 to +0.06 in three weeks (re-fit the
weight on a schedule, constrain the exposure as MSCI does, or state the lean and leave it);
answer section 4 first.

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

## 5. Sources

- `research/2026-10-12-growth-category.md` sections 10-11 (the review, the band, the open questions).
- `research/measurements/2026-10-13-growth-weight-sweep.py` (unchanged; run on the 10-30 payload as
  committed and on the refused 11-02 payload as regenerated by `scripts/rehearse_publish.py`).
- The series in section 2: `git show <commit>:dashboard_data.js` for each publish, Spearman of
  `Composite` against `growth_score` over `table_data`.
