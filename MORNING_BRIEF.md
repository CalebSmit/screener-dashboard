# Morning Brief - Monday 23 November 2026, 18:00

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **failed** - last ran today |
| Code session (latest) | **failed** - last ran today |
| Dashboard data from | 2026-11-23 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, APA, SNDK |
| Ranking's lean on growth | +0.053 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 19 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (56 rows, but overlapping windows are not independent; 137 rows across all horizons), newest 1m run date 2026-10-21 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Code-session login | 2026-12-03 09:01 (9 days) |
| Context coverage | options usable 494/499 (5 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (**today**) - Build residual momentum in place of `jensens_alpha` (ACCEPTED 2026-11-21, weight 25 unchanged) in a 06:00 session (0.15's section 4 is committed - done 2026-11-21): section 9's build list - 37-month history on the same request with the Risk inputs trimmed to 13, `^SP500TR` to 4y, lineage/equation/claims, backtest v2, the engine rerun, a throwaway full run before the changelog entry. `research/2026-11-20-jensens-alpha-in-momentum.md`
- 2026-11-24 (in 1 day) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 11 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 14 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## Things that needed attention

- hint: Diverging branches can't be fast-forwarded, you need to either:
- hint:
- hint: 	git merge --no-ff
- hint: or:
- hint: 	git rebase
- hint: Disable this message with "git config set advice.diverging false"
- fatal: Not possible to fast-forward, aborting.
- git pull --ff-only failed - local main cannot be brought level with origin/main. Not running on it.
- git pull --ff-only failed - main diverged locally. Resolve by hand.
- Could not fast-forward local main to the published brief: hint: Diverging branches can't be fast-forwarded, you need to either:

## What changed in the repo

- `53dc4a7 data: screener run 2026-11-23 - 499 scored, top: EXPE VLO MU APA SNDK`
- `70492a3 NIGHTLY_LOG 2026-11-22 (owner-run, evening): the phone pass - measurements, gates, what was left`
- `dbe332d Phone pass: on a phone every long part is a heading that opens`
- `1862811 brief: code session 2026-11-22-midday`
- `af2a043 docs: eq4/roe5 shard slice - plan progress, inventory, CLAUDE.md, owner item, NIGHTLY_LOG 13:00`
- `739bc89 shard (eq4): the calendar's phone line waits too, rather than vanishing until the file lands`
- `3ab8056 shard (eq4, roe5): the quarters and ROE years behind four metrics' workings leave first paint - 908 -> 801 KB gzipped, -11.8%`
- `f585248 brief: code session 2026-11-22-weekend`
- `11e048c NIGHTLY_LOG 2026-11-22: gates and what the full run caught`
- `2b398f5 shard (inp): three more published-payload readers fold the shard; docs and log`
- `2d634c2 shard: the reported inputs (inp) leave first paint - 1,125 -> 908 KB gzipped, -19.3%`

## The session's own account

> 2026-11-22 (owner-run, evening) - The phone pass: every long part is a heading that opens. Home, stock sheet, Market, Methodology, research page and Factor Lab at 375px
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-22_130001-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-20_180003.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (55 rows, but overlapping windows are not independent; 135 rows across all horizons), newest 1m run date 2026-10-20 (33 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **41 days**; a standing programme, last advanced **16 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 46d; 0.14 39d. Shipped to readers: 19 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days.
> 
> **Owner request:** *"i want you to do a full review of the UX and frontend of all of these things, the
> dashboards, etc, on the phone version. Right now I tried goign trhough the screener dashboard on my
> phone, and there is too much info and it got way too long and hard to know where to go ... this pass
> is just for the phone."* `logs/.owner-session` held from 21:02; removed at the end.
> 
> ### What I measured first (375 x 812, the live page as committed)
> 
> | Surface | Height | What a phone reader met |
> |---|---|---|
> | Home page | 9,635px; the ranking's section began at 4,514px | hero, four KPI tiles, the dot chart, a 526px guide, an 884px leaders carousel and seven teaser tiles (1,606px) before the first row |
> | Stock sheet (EXPE) | 9,819px in a 484px window | ten sections, seven of them open; "Before you decide" alone was 3,365px |
> | Market backdrop, opened | 6,760px | five readings with a long paragraph each, then ten charts |
> | Methodology dialog | 73,020px | one column; its contents rail is hidden below 900px, so no way to navigate it |
> | `research.html` | 15,518px | the answer, then eight sections end to end |
> | `lab.html` | 14,249px | a 717px hero, then nineteen experiment cards |
> 
> ### Did
> 
> One rule, applied everywhere, at phone widths only: **a long part is a heading that says what is
> inside it, and it opens with one tap.** Nothing left any page; no number, rank or sentence changed
> (`scripts/rebuild_page.py`: payload byte-identical; the research page's `report` changed only
> `research.html`); the desktop pages are untouched (the sub-agent's four 1440px before/after
> screenshots are pixel-identical; `tests/test_phone_pass.py` asserts the dashboard's phone controls
> and folds are absent at 1440px); print shows everything.
> 
> | Surface | Before | After |
> |---|---|---|
> | Home: top of the ranking's first row | ~5,000px | 2,687px |
> | Stock sheet (EXPE) | 9,819px | ~3,500px |
> | Market backdrop, opened | 6,760px | 3,460px |
> | Methodology | 73,020px | 4,956px |
> | `research.html` | 15,518px | 3,516px |
> | `lab.html` overview | 14,249px | 5,904px |
> 
> - **Dashboard** (`generate_dashboard._css_phone`, `_js_phone`, emitted last): the dot chart and the
>   guide are one line each; leader cards keep name, composite and reasons; the seven tiles are a title
>   and their headline; filters other than search and sort sit behind a **Filters** button that counts
>   the ones in use. The sheet's five long sections arrive folded with a one-line hint each; score tiles
>   are shorter; "Why it ranks here" folds "What changed" and "Context" (never "Read with care"). Market
>   backdrop clamps each reading to three lines and puts the ten charts under three headings. The
>   Methodology folds under its eighteen section headings, Overview open, and a cross-reference opens
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

