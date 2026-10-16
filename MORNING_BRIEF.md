# Morning Brief - Friday 16 October 2026, 13:52

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-16T02:00:03.815047 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, HST, INCY, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (31 rows, but overlapping windows are not independent; 85 rows across all horizons), newest 2026-10-09 |
| Context signals | Context signals: 7 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); CF (#8: net income 16% off the filing); MU (#18: a period behind its filings); ES (#19: a period behind its filings) |
| Context coverage | options usable 492/501 (9 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `870c4f5 the refusal recorded: changelog, owner queue, priorities and the day's log`
- `6297aa4 a long filing tag wraps instead of being clipped at 375px`
- `cb4c86b the earnings-yield row says what one-off charge is inside the earnings it scored`
- `420464a the earnings-definition proposal: REVIEWED and REFUSED, with the measurements`
- `260f994 wip: the earnings-definition proposal's input, measured at the window the engine scores`
- `1379e24 wip: measure the open earnings-definition proposal's input against the filings`
- `0498168 professional pass 7: last details; the day's log entry`
- `405b254 professional pass 6: favicons everywhere, the last text chevron, a packed context grid, tidy labels`
- `68e02d7 the claims gate runs the methodology typesetting test`
- `78bf7f0 professional pass 5: the Factor Lab typesets its dashes and minus signs as it draws`
- `0c80999 professional pass 4: the Methodology document - statement headings, one typesetting pass, two framing errors`
- `fa9bab0 professional pass 3: bars drawn above their reference line, unwidened hyphens, one heading case`
- `4a24c20 professional pass 2: 40px touch targets, one focus ring, selection colour, the stock sheet fits a phone`
- `01f0fd6 professional pass 1: one voice in the copy, one date format, one disclosure icon; two defects`
- `ec83782 brief: code session 2026-10-16`

## The session's own account

> 2026-10-16 (13:00 session) - QUEUE. The earnings-definition proposal: reviewed, refused, and the charge inside the earnings put on the row
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-16_060001.log`
> (the retrospective) ends "Run complete: shipped to main"; the owner-run professional pass followed it
> and `main` is at `98fa2a0`. Data loop published? **yes** - `logs/datarun-2026-10-16_020001.log` ends
> "Data loop complete", HEALTH: PASS. Evidence base at `1m`: **31 rows, newest 2026-09-16 (30 days ago,
> bound 40), 4 effective** - inside the bound and in steady state. Priority 0: unchanged,
> `allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.11, the
> context layer** (opened 2026-10-08, **age 8 days**).
> **Tests:** before **2492 passed / 0 failed**, after **2504 passed, 1 skipped / 0 failed** (+13, all
> in the new `tests/test_special_item_note.py`; the skip is its published-payload check, which cannot
> fire until the 2026-10-19 run puts the notes in the committed payload). Dry run exits 0.
> `scripts/rehearse_publish.py` passed twice.
> **Owner queue / rotation:** took the **open proposal**, which Orient makes the first job and which
> the retrospective six hours earlier named as this session's Next. **0.11 (the context layer) was
> deferred** - item 4's two remaining per-signal notes are untouched and nothing there is half-done.
> 
> ### Did
> 
> **Decided the one open proposal** (`research/2026-10-15-earnings-definition.md`, open 1 day): score
> earnings yield on the source's "Normalized Income" - net income with its unusual items removed -
> instead of net income as reported. **REFUSED.** Section 8 of that note is the review; the changelog
> entry is 2026-10-16.
> 
> The research is good and is **not** what it failed on. I re-read the paper carrying the argument:
> Rouen, So & Wang (2021), *JFE* 142(3) 1068-1091, Table 3 Panel A - next-year persistence **0.810**
> for core earnings against **0.637** for net income, **0.178** for the excluded items themselves; and
> MSCI's Fundamental Data Methodology (March 2021, pp. 13-14) builds EPS the same way. Section 4's
> diagnosis is right too: EV/EBITDA measures the recurring business while earnings yield measures it
> plus this year's gains and charges, inside one category.
> 
> **What the refusal rests on is the input, measured at the window the engine actually sums** - the
> note measured the annual row. Both prior proposal decisions were refusals resting on a measurement
> the note had not made, so I measured before deciding:
> 
> - **23 companies have no four-quarter "Normalized Income" at all** - ABT, AMZN, APH, BA, BAC, BRK-B,
>   CHD, CHRW, CINF, CPRT, CPT, DLR, DTE, ES, EXR, GIS, ILMN, LEN, MDLZ, REG, TFC, WELL, ZBH. The
>   proposal's own fallback ("1 company of 497") would rank them on earnings *including* one-offs
>   beside 265 ranked excluding them. That is precisely the defect `factor_engine.py:2094` records
>   Phase 13 fixing **in this exact metric**: *"SINGLE definition across the universe ... the fallback
>   cohort was ranked on an incomparable basis inside one percentile rank."*
> - **The fallback cohort is where the one-offs are biggest.** Of the 18 whose annual figures are both
>   known, **8 carry annual unusual items worth 10%+ of net income**: General Mills 2,193%, **Boeing
>   434%** ($9.70bn gain on a disposal), CPT 65%, DLR 56%, ZBH 37%, AMZN 20%, WELL 16%, ES 11%. The
>   change would remove the small distortions and **keep the two largest in the index** - worse than
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

