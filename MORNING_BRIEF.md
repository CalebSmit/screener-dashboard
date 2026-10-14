# Morning Brief - Wednesday 14 October 2026, 02:21

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-14T02:00:10.255065 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, INCY, HST, APA, SNDK |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (29 rows, but overlapping windows are not independent; 81 rows across all horizons), newest 2026-10-07 |
| Context signals | Context signals: 5 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Context coverage | options usable 486/501 (15 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `ccbc795 data: screener run 2026-10-14 - 501 scored, top: EXPE INCY HST APA SNDK`
- `d7c1b71 brief: code session 2026-10-13`
- `1a027ac growth: record the refusal in the docs, and generalise the open-proposal test`
- `99ec010 growth: record the decision on the page, the lab scorecard and the plan`
- `a3d2726 growth: review the waiting proposal and refuse it, with the measurement that decided it`
- `eeff9e2 brief: data run 2026-10-13`
- `a5e84b8 data: screener run 2026-10-13 - 501 scored, top: EXPE HST INCY APA BMY`
- `0bbd9c5 brief: list proposals waiting for a decision, with their age`
- `97e7a29 Factor Lab experiment 9 (the largest fifth), and the nightly sessions set up to carry the work on`
- `97ab4a8 research: value and momentum among large companies - no weight change, and a page that says why`
- `5f39f6f research: what stands behind Growth's 13%, and what the category actually does`
- `55ccf6e Factor Lab: what a holder who shorts nothing keeps of each factor among big companies`
- `d73e26f Factor Lab: the factors among large companies, and how a lab finding reaches a weight`
- `447736d Factor Lab: every headline result inside its paper's sample and since, on the front page`
- `573fd30 methodology page: the Risk and Size sections say what the Factor Lab measured`

## The session's own account

> 2026-10-13 (06:00) - the Growth proposal: reviewed, refused, and the weight stated as deliberate
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-12_060001.log`
> ends "Run complete: shipped to main". Data loop published? **yes** - `logs/datarun-2026-10-13_020001.log`
> ends "Data loop complete", HEALTH: PASS, claims gate 371 passed. Evidence base at `1m`:
> **28 rows, newest 2026-09-11 (32 days ago, bound 40), 4 effective** - inside the bound, and the
> engine's own line in the data log agrees ("4 effective (28 rows) at the 1m horizon"). Priority 0:
> unchanged, `allow_auto_apply` false, 4 of 8 effective observations. Top open roadmap item:
> **0.13, the Factor Lab** (opened 2026-10-12, **age 1 day**).
> **Tests:** before **2227 passed / 0 failed**, after **2235 passed / 0 failed**.
> **Owner queue / rotation:** took the owner item - `OWNER_FOCUS.md`'s Factor Lab item and
> `prompts/nightly.md` section 3 both say the waiting Growth proposal outranks the day's focus.
> **Tuesday's product focus was therefore deferred**, deliberately and with the owner item named;
> nothing in the dashboard was found broken on the way through.
> 
> ### Did
> 
> **Reviewed the proposal in section 7 of `research/2026-10-12-growth-category.md` (Growth 13% ->
> 10%, Quality 22% -> 25%) and refused it.** The decision and its reasoning are section 10 of that
> note; the status line is rewritten, so the morning brief's "Waiting for a decision" list is now
> empty (`write_brief.open_proposals()` returns `[]`).
> 
> **What the review added.** Section 7 measured one point - the composite's Spearman with the Growth
> score is -0.01 at weight 13 and -0.06 at weight 10 - and argued from the evidence that growth as
> the category measures it earns nothing of its own. One point cannot separate the two readings of
> that evidence, because they predict different *curves*:
> 
> - *"Weight belongs where evidence is"* - move the 3 points; the curve is irrelevant.
> - *"A trait that earns nothing should be neither rewarded nor penalised"* - the right weight is
>   whichever leaves the ranking neutral, and moving off it is a disimprovement.
> 
> `research/measurements/2026-10-13-growth-weight-sweep.py`, on the published run of 2026-10-13
> 02:00 (501 stocks; the approximation reproduces the published ranking at Spearman 1.0000, 25 of
> the top 25). Growth swept 0-26:
> 
> | Growth weight | 0 | 5 | 10 | **13** | 16 | 20 | 26 |
> |---|---|---|---|---|---|---|---|
> | composite's Spearman with the Growth score | -0.199 | -0.136 | **-0.060** | **-0.009** | +0.048 | +0.130 | +0.258 |
> 
> - Exposure is **not flat**: monotonic across the sweep, slope **+0.0181 of correlation per point**.
> - **Zero exposure is at weight 13.5** (13.3 pro rata). The configured weight is 13.
> - |exposure| <= 0.05 only over **weights 11-16**. The proposed 10 is just outside it.
> - **Growth is the only category that runs against the composite's grain**: -0.388 against the mean
>   of the other seven, where every other category is between -0.055 and +0.158. Nothing else does
>   this job, so Growth's weight *is* the growth-exposure dial.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

