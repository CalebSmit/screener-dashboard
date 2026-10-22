# Morning Brief - Thursday 22 October 2026, 07:25

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-22T02:00:04.595430 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, APA, HST, INCY, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (34 rows, but overlapping windows are not independent; 91 rows across all horizons), newest 2026-10-14 |
| Context signals | Context signals: 11 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 422/423 within 0.5%; operating cash flow 405/409 within 0.5%; net income 396/418 within 0.5%; market value 444/484 within 5%; filings cached 6 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); CF (#9: net income 16% off the filing); MU (#14: a period behind its filings) |
| Context coverage | options usable 490/498 (8 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `9bf29e2 log: the gates' numbers, the rerun check, and the three guards that fired`
- `15efe21 research engine: all 31 rows rerun level, and the page reports the stop arm`
- `3a56a63 log + OWNER_FOCUS: the 2026-10-22 stop-loss item, done and moved to Done`
- `5e9d966 research: the stop-loss arm's result, and the page reports it beside the unstopped rule`
- `513c364 research engine: register and build one stop-loss arm on the bot's rule (Han, Zhou & Zhu 2016)`
- `ad2346d brief: data run 2026-10-22`
- `9c0bc1b data: screener run 2026-10-22 - 498 scored, top: EXPE APA HST INCY EG`
- `a45fb93 OWNER_FOCUS: one pre-registered stop-loss test on the bot's rule (Han, Zhou & Zhu 2016), never a sweep`
- `eaef387 OWNER_FOCUS: the screener's half of the plan to real money - no wrong or stale ranking reaches the bot`
- `13e3146 contract: pin share_classes for screener-bot (scored_as + the line's own price)`
- `d5d3a5c brief: code session 2026-10-21-midday`
- `0055951 log: the 2026-10-21 13:00 session - the bank P/B-ROE decision and the guard it replaced`
- `732ad0a research engine: rerun all 30 after the config.yaml comment (0 of 25,460 values moved)`
- `55b9291 record: the bank P/B-ROE decision - changelog, owner queue, section order`
- `859a7dd bank P/B-ROE proposal ACCEPTED: state the pairing on the page, guard the measured loading`

## The session's own account

> 2026-10-22 (06:00 session) - QUEUE/BUILD. One pre-registered stop-loss on the bot's rule, and the prediction failed in two of three periods
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-21 13:00 ends "Run complete:
> shipped to main" at 14:30 (`logs/nightly-2026-10-21_130001-midday.log`). Data loop published?
> **yes** - `logs/datarun-2026-10-22_020001.log` ends "Data loop complete" at 02:24, PUBLISHED CLAIMS
> PASS (1272 tests), 498 scored. Evidence base at `1m` = **34 rows, newest 2026-09-21 (31 days ago,
> bound 40), 4 effective** - inside the bound, no action. Priority 0: `allow_auto_apply` **false**,
> 4 of 8 effective observations. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12,
> **age 10 days**) - not worked; its day is Friday 2026-10-23, and the owner queue outranks it today.
> **Tests:** before **2981 passed / 1 skipped** (364s), after **2988 passed / 1 skipped** (353s) -
> +7, which is exactly the seven new tests in two existing modules, no failures and nothing removed.
> Gates 2 and 3 pass (dry-run exit 0; `index.html` 651,277 bytes; both payloads `node --check` clean
> with their opening assignments; `research.html` 55,107 bytes).
> **Owner queue / rotation:** took `OWNER_FOCUS.md`'s **top open item, 2026-10-22** - one
> pre-registered stop-loss test on the bot's rule. It is a BUILD day and this was a build item, so
> nothing was displaced. No proposal was waiting for a decision (`write_brief.open_proposals()`
> returned `[]`). **Deferred:** the 2026-10-21 item (the rank-continuity gate and the rest of the
> screener's half of the path to real money), the 2026-10-20 item's parts 2-6, and the Factor Lab.
> 
> ### Did
> 
> **The owner asked for exactly one thing and the value is in its being exactly one.** Register a
> stop-loss variant of the trading rule before building it, build that one, report it beside the
> unstopped rule, never sweep the level. All four held: `bot-rule-v1.1-stop10` and the code that runs
> it are in **one commit** (513c364) and no ledger row for it existed before, so the git history is
> the pre-registration.
> 
> **The rule was read from the paper, not from memory.** Han, Zhou & Zhu (2016), "Taming Momentum
> Crashes: A Simple Stop-Loss Strategy", SSRN 2407199, DOI 10.2139/ssrn.2407199 - 52-page PDF
> downloaded and read with `pypdf`; construction on printed pp. 5-6, Table I Panels A-C on pp. 6-7,
> turnover and stop frequency pp. 8-9. **Two things the reading corrected:** the DOI resolves through
> Crossref to SSRN posted content and **no journal** - it is a working paper, and the *Journal of
> Financial Markets* in the owner's item is Kaminski & Lo (2014), a different paper. And the paper's
> rule is not "sell at the close 10% down": it triggers on the day's **lowest** price, waits a day,
> and sells at a limit of `(1-L/2)(1-L)P0` - 85.5% of the month's starting price at L = 10% - or at
> that day's close if the whole day trades below it.
> 
> **So four differences are registered under `construction_note`, each because it changes when the
> stop fires:** the trigger here is the close (the daily panel carries no intraday range, so this
> fires *less* often than the paper's); the sale is at the triggering close, not the next day's limit
> (so it exits *sooner and higher*, and exits names the paper's band would have kept); cash earns
> nothing where the paper earns T-bills; and this is the long side of a 25-name screen, not long-short
> momentum deciles.
> 
> **The prediction, registered before the run:** less negative worst month *and* maximum drawdown in
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

