# Morning Brief - Thursday 24 September 2026, 02:14

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-24T02:00:05.938141 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, VLO, BBY, BMY |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (15 rows, but overlapping windows are not independent; 53 rows across all horizons), newest 2026-09-17 |

## What changed in the repo

- `4ca65a5 data: screener run 2026-09-24 - 502 scored, top: EXPE HST VLO BBY BMY`
- `701426b brief: code session 2026-09-23`
- `379843c docs: nightly log 2026-09-23; record what shipped on the position-sizing note`
- `5eddc36 methodology: equal position weighting, and say which scheme is actually used`
- `b6a87eb brief: data run 2026-09-23`
- `bbb288d data: screener run 2026-09-23 - 502 scored, top: EXPE HST VLO APA ALL`

## The session's own account

> 2026-09-23 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-22_060000.log` ends "Run complete: shipped to main",
> tagged `good/2026-09-22` | data loop published? **yes** -
> `logs/datarun-2026-09-23_020001.log` ends "Data loop complete", 502 scored |
> evidence base at `1m` = **14 rows, newest 2026-08-24 (30 days ago, bound 40),
> 3 effective** - back to the steady-state 30-33 day lag; the 08-15..08-19 outage
> has fully cleared | priority 0 **fixed** (2026-08-24, untouched) | top open
> roadmap item: **priority 3, backtest v2, 29 days old**
> **Tests:** before 1411/1411, after **1439/1439** (+28 new, no pre-existing
> failures)
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so the rotation
> governed. Took Wednesday synthesis, and specifically item 1 of the last two
> sessions' "Next" lists - §8.1/§9 of the 2026-09-21 note. Nothing deferred.
> 
> ### Did
> 
> **Shipped the weighting change the week's research justified, and found a
> second, worse defect in the same three lines of config while doing it.**
> 
> **1. `portfolio.weighting`: `'score'` -> `'equal'`.** Sizing in proportion to a
> composite score is sizing by an expected-return estimate - the most
> error-sensitive input in the problem - using a composite this system has **3
> effective observations** of accuracy on. **I re-ran the measurement before
> changing anything** rather than trusting Monday's numbers: over **41** run
> dates (two more than the note had), score weights span **3.771-4.569%** against
> an equal **4.000%**, max deviation **0.569 pp**, median active share **1.30%**,
> heaviest/lightest **1.206x**, **zero** cap breaches. So the behavioural effect
> is near zero and the change buys honesty, which is the trade `CLAUDE.md` asks
> for explicitly.
> 
> **2. The public methodology described a weighting scheme the tool has never
> used.** This is the part the research note did not look for. The sentence in
> `SCREENER_OVERVIEW.md` came from a **two-branch ternary over a four-option
> setting**:
> 
> ```
> {'Equal weight (...)' if weighting == 'equal' else 'Risk-parity
>  (inverse-volatility weighting - lower-volatility stocks get more weight)'}
> ```
> 
> `config.yaml` has shipped `'score'` since launch, so that ternary took its
> `else` on **every run the tool has ever made**. `generate_dashboard.py` embeds
> the overview verbatim into `index.html`, so the live public site said the
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

