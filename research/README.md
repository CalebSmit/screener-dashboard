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
