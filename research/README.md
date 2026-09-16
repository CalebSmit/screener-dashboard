# Research notes

Monday sessions produce notes here. This directory is the evidence base the rest
of the week builds on, and the reason a methodology change can be defended months
later.

**A note must be complete in one session.** Research used to be split across
Monday (literature) and Tuesday (practice); the 2026-08-21 retrospective ended
that, because a lost Monday left Tuesday with nothing to append to. Tuesday is
now the product day. One self-contained note, literature *and* documented
practice, is the unit.

## Naming

`YYYY-MM-DD-short-topic.md` - e.g. `2026-08-10-momentum-lookback-windows.md`

## `measurements/` - the numbers a note quotes, as runnable code

Added 2026-09-16. When a note quotes statistics computed from this repo's own
data - snapshot series, rank migrations, coverage counts - **put the script that
produced them in `research/measurements/`**, named after the note it supports.

The reason is the mandate's own wording: evidence must be written down *where
someone else can check it*. A number in prose cannot be re-run when the series
grows, and several of these notes explicitly ask a later session to re-measure.
The 2026-09-14 note asked for exactly that and had no script to re-run; the
2026-09-16 synthesis then found that two of its headline numbers were estimator
artifacts, which a re-runnable script would have surfaced immediately.

These are analysis scripts, not pipeline code - nothing in the nightly loops
imports them, and they are exempt from the ship gates' expectations about
coverage. They must still run from a clean checkout.

## What a good note contains

- **The question.** One sentence. What decision does this inform?
- **What the literature actually says.** Author, title, year, venue. The finding
  *with effect sizes* and the conditions under which it held - sample period,
  universe, market. "Momentum works" is not a finding; "12-1 month momentum
  earned ~1%/month in US large caps 1965-1989, with severe crashes in 2009"
  is.
- **Where the evidence contradicts what we currently do.** The most valuable
  part of the note. Be specific about the config value or code path.
- **What would change our mind.** The falsifiable version.
- **Recommendation.** Concrete enough to implement, or an explicit "no change
  warranted, here's why."
- **Wednesday's design section.** Wednesday is synthesis day and reads this note
  to decide how the finding fits the rest of the screener. Give it the
  hypothesis, the implementation sketch, and the measurement that would refute
  it.

## Standards

- Cite primary sources. A blog summarising a paper is a pointer to the paper,
  not a citation.
- Prefer findings that replicated out-of-sample or post-publication. Much of the
  factor literature does not survive publication - assume decay until shown
  otherwise.
- Record disconfirming evidence. A note that concludes "our current approach is
  well-supported, don't touch it" is a successful research session and saves a
  future one from relitigating it.
- Note when a finding is US-large-cap-specific. This screener is S&P 500 only,
  which is a narrow, well-arbitraged universe - effects documented in small caps
  or internationally often will not survive here.
- **Count independent observations, not rows.** This is the mistake this project
  keeps making and it has now been made in three different places. Overlapping
  forward-return windows inflated the IC evidence base by ~2.35x
  (`2026-08-10-ic-evidence-independence.md`), which
  `improvement_engine._effective_observations()` now guards against. The same
  trap caught rank-migration statistics on 2026-09-16: 34 runs yield 72 pairs at
  a fortnight's spacing, and the overlapping estimator understated band-breach
  rates by 2-3x. Before quoting an `n`, ask how many **non-overlapping** looks
  at the question it really represents - and remember that ~500 tickers within
  one run move together, so the independent unit is usually the *run*, not the
  row.
