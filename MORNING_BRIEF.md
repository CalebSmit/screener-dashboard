# Morning Brief - Friday 18 September 2026, 02:14

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-18T02:00:08.064952 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, CAH, VLO, BMY |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (11 rows, but overlapping windows are not independent; 45 rows across all horizons), newest 2026-09-11 |

## What changed in the repo

- `db3bdd5 data: screener run 2026-09-18 - 502 scored, top: EXPE HST CAH VLO BMY`
- `72799b2 brief: code session 2026-09-17`
- `110583c docs: changelog, session log, and correct the inventory's superseded hold-band numbers`
- `6f61ba8 feat: state the review cadence the tool is built for`
- `1f097d6 feat: flag when a rank move is an input going missing, not the company`
- `fad60bc brief: data run 2026-09-17`
- `090abba data: screener run 2026-09-17 - 502 scored, top: HST EXPE CAH BBY JBHT`

## The session's own account

> 2026-09-17 - BUILD. Implement what the week's research justified. Write tests alongside the code.
> 
> **Health (rule 8, all five):**
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **Yes.** `logs/nightly-2026-09-16_060000.log` ends "Run complete: shipped to main", tagged `good/2026-09-16` |
> | Data loop published? | **Yes.** `logs/datarun-2026-09-17_020000.log` ends "Data loop complete", 502 scored, HEALTH: PASS, 0 fetch failures |
> | Evidence base | **44 rows, newest 2026-09-10, 3 effective observations at `1m`** (11 raw) against a gate of 8. Up from 43 rows / 2026-09-09 yesterday - moving. Effective count unchanged at 3, as expected: 1-month observations accrue about one a month |
> | Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call touched today |
> | Top open roadmap item | **Priority 5, the sell-side workflow - 43 days old.** Two of its three remaining build items shipped today. Next unblocked item is **Priority 3, backtest v2**, plan file dated 2026-08-25 - **23 days** |
> 
> **Tests:** before **1264/1264**, after **1362/1362** (+98: 58 new in
> `test_input_churn.py`, 40 in `test_review_cadence.py`).
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
> deferred. Thursday's build taken as written, implementing §8.7 items 1 and 2 of
> the 2026-09-14 research note - the rotation's intended path, not a swap.
> 
> ### Did
> 
> **Shipped the two §8.7 items that share a single argument: the tool now says how
> often it is meant to be acted on, and says when a move is the measurement
> changing rather than the company.** Both are about not inviting action the
> evidence cannot justify, which is why they went together rather than one per
> session.
> 
> **1. The cadence the tool is built for is now stated on the surfaces that move.**
> `config.yaml` has recorded a quarterly rebalance cadence since launch - as a bare
> *comment*, which the generator could not read. The site regenerates every weekday
> and said nothing: a grep of `generate_dashboard.py` for "quarterly" returned one
> unrelated data-source label. A surface that redraws a rank every morning
> implicitly invites acting on it every morning.
> 
> `portfolio.review_cadence` is now a real key, surfaced as `D.cadence` and
> rendered by `cadenceText(long)` in three places - the holdings panel (short form,
> both empty and populated states), the What Changed footnote (short form), and the
> holdings footnote (long form with the numbers). Read from the **run's own** config
> snapshot, not the working tree, so a republished old run states what it was
> configured for; `configured: false` marks the fallback so it cannot be mistaken
> for a real setting. Both paths are exercised today: the published 2026-09-17 run
> predates the key and falls back cleanly; tonight's data run will carry it.
> 
> **It is a sentence, not a lock**, and that was a deliberate call. The tool does
> not know what a reader is doing. Naming the cadence is decision support;
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

