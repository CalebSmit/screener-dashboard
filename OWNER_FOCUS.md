# Owner focus queue

**This is how the owner tells the nightly session what to work on.**

Write what you want in plain English and save the file. That is the whole
process — no formatting rules, no ticket numbers, no need to say where in the
codebase it lives. The next code session (6:00 AM, Mon–Fri) reads this file
**before** it reads the weekly rotation, and open items here outrank the day's
nominal focus.

You do not need to be precise about *how*. "The sector chart is useless" is a
perfectly good item; working out what to do about it is the session's job. If
an item turns out to be a bad idea, the session will say so in
`NIGHTLY_LOG.md` and explain why rather than silently skipping it.

Two things still outrank this file, and neither is negotiable:

1. **A broken data loop or a failing ship gate.** A stalled pipeline means the
   tool stops improving at all, so it gets fixed first. The session will say in
   the log that it deferred your item and why.
2. **The four ship gates.** Nothing here can authorise a push that fails them.

---

## Open

- **2026-10-21 - The screener's half of the plan to real money: no wrong or stale ranking reaches the
  trading bot.** Owner: *"I want you to start thinking in the way that you need to make me money in
  the stock market with this tool. And that's going to be through the trading bot."* The plan is
  `screener-bot/ROADMAP.md` (rewritten 2026-10-21 from five audits); its **phase 1** is the input
  side, and these items are this repo's. Measured: on a normal day the top 25 gains at most 4 new
  names and the rank correlation with the previous run is >= 0.981 (the 10-09 overhaul: 6 and 0.935;
  the broken 07-28 run: 24 and 0.016). In order, each with its test:
  1. **Rank-continuity gate** in `check_run_health.py` (bot ROADMAP 1.3): FAIL below 0.90 correlation
     with the last published run, more than 8 new top-25 names, or more than 3 top-25 names below #40;
     override only when the methodology version changed and a changelog entry is dated today. Test: a
     shuffled ranking with normal spread fails, a normal day passes, an overhaul with its entry passes.
  2. **The empty-record hole** (1.4): FAIL below 490 scored or coverage exclusions above 2% of the
     universe, and count them toward the 40% fetch gate. Today `meta.json` shows 2 unscored while the
     log says 0 failures.
  3. **Price freshness** (1.5): publish each price's bar date; FAIL unless >= 95% sit on the expected
     session, or if over half are unchanged across a session.
  4. **Data-loop git hygiene** (1.9): fail closed when `git pull --ff-only` fails; push only the data
     commit, never all of local `main`; put `tests/test_bot_payload_contract.py` in the publish gate.

  **Items 1-4 BUILT 2026-10-22 (on a worktree branch; merged by the session that relays it).** All in
  `scripts/check_run_health.py` / `scripts/data-run.ps1`, tested in `tests/test_input_gates.py`, and
  replayed before shipping against every page the site published since 2026-08-21
  (`research/measurements/2026-10-22-input-gates-replay.py` + JSON; 62 versions, 46 data-loop runs):
  **no ordinary day fails any rule**; the known-bad pages do (07-28: Spearman 0.856, 10 new, 7 fallen;
  08-06 synthetic and 08-07 warm-start: -0.017, 22 new, 23 fallen, no prices; 08-10: no prices).
  - *1:* the baseline is `dashboard_data.js` as committed at HEAD (the working-tree file is the run
    itself by then). "Methodology version" = `RunContext.config_hash` - the scored-cache key, which
    covers every weight, the data-quality/universe/sector/trap/regime/share-class rules and the
    `scoring_schema` counter - recorded in `meta.json` and published as `kpis.methodology_version`.
    **One deliberate change to the wording:** the changelog entry may be dated from the published
    run's day to today, not today only - measured, 6 of the 46 data-loop runs (09-02, 09-03, 09-11,
    10-13, 10-20, 10-22) were the first scored under a change documented the day before, and under
    "today" only a refusal there could never clear (every later run is compared with the same stale
    page). The 10-09 overhaul passes outright (0.935, 6 new, 1 fallen) and had the waiver available.
    Known limit, written in the docstring: a bad page published by a code session (ungated here)
    would make good runs fail until item 5 moves the baseline to the last data-loop run.
  - *2:* `universe.json`'s `failed` (fetch errors + empty records + coverage filter) over 2% of the
    universe, or under 490 ranked; history max 0.4% (HONA, VYLR), min 498 ranked. `data-run.ps1`'s
    40% gate now counts every "Excluded from scoring" ticker, once each.
  - *3:* `stock_detail[t].price_date` (the quote's `regularMarketTime` in Eastern, or the last bar
    when the quote is missing). Expected session from a rule-based NYSE calendar (holidays verified
    2022-2027; an unscheduled closure costs one discarded run). Verified live on a 29-stock run at
    12:45 CT: 29/29 dated 2026-10-22. History cannot replay the dates (they did not exist); its
    unchanged-across-a-session share peaked at 1.2% against the 50% line.
  - *4:* pull failure and a local `main` ahead of `origin/main` both stop the run before the
    screener; the push is `<data commit>:refs/heads/main`, only when it is the single commit ahead;
    `test_bot_payload_contract.py` and `test_input_gates.py` are in `check_published_claims.py`.
  The methodology page states the new refusals from the gate's own constants. **5 and 6 are open.**
  5. **Only the data loop's run is tradable** (1.2, this repo's half): after every gate `data-run.ps1`
     writes `signals.json` (top 40, run_id, git_sha, raw-fetch sha256, gate verdicts); the code
     runner never stages it (test it beside `tests/test_gate_ordering.py`). Code sessions' full runs
     keep publishing the page; they stop being what the bot trades.
  6. **Audit trail** (1.10): `run_id`, `git_sha`, a config hash and `raw_sha256` in `kpis` (add them
     to the contract test); archive each published run's raw fetch and log outside OneDrive, never
     pruned.
  The bot's halves (fresh-ranking rule, methodology pause, second price source, pre-trade verifier)
  are owner-run work in the bot repo. **None of this changes a score.**

- **2026-10-20 - Fine-tune and test what the 2026-10-20 night built.** Owner: *"do as much work as
  you can tonight before the 2 am session please! Then have the sessions fine tune things and test
  things."* The night shipped (owner-run, tags `good/2026-10-20-owner-1..`): the overfitting check on
  the research page, the engine calibration test, Factor Lab experiment 15 (Jensen, Kelly & Pedersen
  2023), the Size and bank-overlap research notes, the context layer's close-out (0.11), and the
  measurements listed in `NIGHTLY_LOG.md` 2026-10-20 (owner-run, night). **The sessions' job now is
  to check that work as a second reader and harden it, not to add surfaces.** In order:
  1. **Decide the OPEN proposals** (the morning brief lists them with their ages): share classes
     (`research/2026-10-19-share-classes.md`), and the bank P/B-ROE pairing - state it on the page
     and guard it with a test (`research/2026-10-20-bank-quality-valuation-overlap.md` section 8).
     The deciding session must not be the one that wrote the note.
     **Share classes: DECIDED 2026-10-21 (06:00) - ACCEPTED, with one change the review measured.**
     Alphabet, Fox and News Corp each had two ranked lines; each company is now scored **once**, on
     its more-traded class, as S&P DJI and FTSE Russell do. The note wanted the line re-picked from
     every run's own trading volume. Measured over 89 month-ends, Alphabet's two classes are within
     S&P's own 20% switching margin **a third of the time** and the busier line has actually changed
     **four times** - and the company ranks **179 on GOOGL against 101 on GOOG**, so that rule would
     have moved a company 78 places on a coin flip. The choice is now declared in `config.yaml` and
     a run only *reports* a crossing. Top 25 unchanged; the 16 other Communication Services stocks
     gain 1.565 composite points each (they were being held down by rows that were duplicates); 0.0
     change outside that sector. `METHODOLOGY_CHANGELOG.md` 2026-10-21.
     **The bank P/B-ROE pairing: DECIDED 2026-10-21 (13:00) - ACCEPTED, with the guard it asked
     for replaced by one that can see what it claims.** The question was whether 35 points of a
     bank's Quality score double-count its Valuation inputs. They do not: a bank's ROE is its
     earnings yield times its price-to-book, so ROE *cancels* P/B rather than adding to it, and the
     four weights net to a position on earnings yield and on equity per dollar of assets with
     **nothing on price-to-book by itself** - which is the "high ROE at a low P/B" reading
     Damodaran, Wilcox, Nissim, Mercer Capital and MSCI's multi-factor indexes all build on
     purpose. Six ways of removing the "overlap" were rescored and every one is further from it.
     **So no weight moved** (0 of 501 composites); what was wrong is that the neutrality was the
     arithmetic of four separately chosen weights and nothing said so. The page now states it, with
     every figure generated from `config.yaml`.
     **What the review added:** both of the note's refutation criteria were re-measured on a
     different run (R-squared 0.978, P/B loading -0.069 against earnings yield's +0.283; all six
     candidates reproduce to the third decimal) - and the guard the note specified was measured
     too, and **could not see the one change the note itself calls the mildest route to a
     price-to-book tilt**: moving ROE's weight to ROA reads exactly 0.00 on its first-order
     formula, identical to today, while the loading measured on the engine's own scores is 2.3x
     today's and past the note's own bar. The failing condition is now the measured one.
     `METHODOLOGY_CHANGELOG.md` 2026-10-21 (13:00 session); the note's new section 11.
     **A third proposal arrived 2026-10-22 and was DECIDED 2026-10-23 (06:00) - all three of its
     registrations REFUSED as written; test B accepted in substance.**
     `research/2026-10-22-free-data-that-adds-evidence.md` asked a later session to register three
     pre-registered tests in `research/experiments/registry.yaml`. The decisive finding is that
     **the registry cannot hold any of them**: `experiments:` is 29 weighting transforms (29 of 29
     carry one) and `evaluations:` is 2 bot rules, and shaped as the note writes them the engine
     raises `KeyError: 'transform'` in `report`, `summary_for_page` and three places in
     `tests/test_engine_calibration.py`. For the short-interest test it is worse than an error:
     `v2_config` zeroes the metric and `_record` NaNs its inputs, so the entry would have **run
     clean on an all-NaN column** and published a null caused by the missing panel. Its data is
     real, though - FINRA's files were re-probed independently: 403 before 2017, **200 from
     2017-12-29**, exchange-listed rows present four years before FINRA's own page says, ~100% of
     rows carrying days to cover. So the note's research stands and the build order is now written
     into its section 7, licence check first. Test A refused on its own void condition (both
     coverage rates measured, 69.8% and 74.4%, sit under its 80% bar); test C refused on the trade
     (22 months buy 9.8% of precision, its confirm band fires with probability 0.808 under the null,
     and the months are size-selected by SEC Release 33-9002, so they cannot be pooled into the Size
     near-miss they would most be wanted for). A false "short interest has no free history" was
     corrected in four places, with a mutation-tested guard against the reclassification that
     correction invites. **No proposal is now waiting for a decision**
     (`write_brief.open_proposals()` returns `[]`).
  2. **Second-read every new lab experiment.** For experiment 15 (and any later one from tonight):
     open the paper, re-read each reference figure against `factor_lab.py`, and check in
     `git log` that the tolerances were committed before the comparison ran. A transcription error
     found here is a defect to fix and record; a miss that is real stays a miss. **Experiment 15:
     DONE the same night** - all ten figures re-read in the paper and recomputed independently;
     the licence's copyright notice added (`research/2026-10-20-lab-jkp.md`, "Second reading").
     Any experiment merged after it still needs its second reader.
  3. **Test the tests.** Pick three registered claims a session (`claims.py`), break the code each
     one describes in a scratch copy, and confirm its `checked_by` test fails. A check that still
     passes is checking plumbing, not the claim (CLAUDE.md, 0.11's lesson) - fix it, and keep a
     running table in `plan/claims-mutation-audit.md`. **Started the same night:** 38 mutations on
     22 claims; **15 got through** (8 were untested boundaries, 3 checked the committed payload
     instead of the code), all 15 now caught by registered tests. The plan file lists the next 12.
  4. **The positive control as a ledger row** (`plan/research-engine.md` item 4(c), second half): the
     look-ahead arm as `kind: control`, rerun with everything else. The synthetic half shipped
     tonight (`tests/test_engine_calibration.py`).
  5. **Lab experiment 4b** - size inside each profitability quintile (French
     `25_Portfolios_ME_OP_5x5`), with the refutation criterion already written in
     `research/2026-10-20-size-weight.md` section 9. It is the one result that would reopen Size.
  6. **Backtest v2's two free data fixes, then the extended build**
     (`research/2026-10-20-free-survivorship-sources.md`): 167 member-months of today's members
     dropped because a rename or symbol format is not carried before May 2014 (BF.B, MHP->SPGI,
     SAI->LDOS, PCS->TMUS, ACE->CB, TYC->JCI), and 773 `no_filings` member-months whose older filings
     sit under a predecessor's CIK (Exxon, BlackRock, Disney, Cigna, Alphabet...). Both touch
     `RESULT_PATHS`, so do them together: one rebuild, one rerun. Then the note's section 8 plan for
     the extended universe - **register `only-size-extended` first, with its one-sided reading**
     (free data adds the members that shrank out but almost none that were acquired, so the Size
     test can falsify the survivorship explanation and cannot confirm it).
     **Late the same night:** Tiingo's public ticker list carries 84 of the 152 acquired names (65% of
     acquired member-months) - the hole free sources could not fill. If the owner opens a free Tiingo
     account, he saves the key himself in the gitignored `data/survivorship/tiingo_key.txt` (never in
     chat, never in a tracked file); then spot-check a few series, read Tiingo's terms on derived results,
     and rebuild with it - the Size test would then be able to confirm as well as falsify. Addendum in the note.
     **Licence gate (checked 2026-10-20 late):** the free Starter plan ($0; 500 symbols a month, 50 requests an
     hour, 1,000 a day) is **personal use only** - its pricing page says the data may not be displayed or
     shared with another person or organization. Whether a statistic computed from it (a monthly IC) may
     appear on the public research page or in this public repo is **not settled**. Until Tiingo confirms
     in writing that it may, **nothing derived from Tiingo prices is committed or published**: raw files,
     rebuilt months and any results stay in gitignored folders and out of the ledger. Do not rebuild v2's
     published records with them.
  7. **Watch two numbers and act before they bite:** the full suite's runtime (6:23-7:20 tonight,
     against the Bash tool's 600 s per call - speed it up or split the call, never drop tests), and
     `lab_data.js` at ~2.9 MB (measure the lab's first paint; move the JKP series to their own file
     if it is slow).
     Also: three tests (`test_claims_register::test_confidence_metric_count_is_the_discount_coverage`,
     two in `test_investor_profiles`) fail with FileNotFoundError in a fresh worktree, which has no
     gitignored `runs/` - every agent tonight had to copy it in. Make them skip loudly when no run exists.
  Close this item when 1-6 are done or argued down in the log.

- **2026-10-15 - A research engine: test the methodology the way automated research loops do, so it
  can be shown to people and trusted.** Owner: *"we're trying to come up with the best methodology
  to screen stocks by, and it has to be backed up, it has to be proven, so ... I can show this to
  people, and they can trust me that this screener works ... first, it's going to be a good trading
  bot, but then eventually ... the methodology behind a hedge fund type of thing that I want to
  start."* **First version built and run the same night (owner-run):** `v2_experiments.py`, 27
  experiments from the literature, each pre-registered with a prediction before it ran, judged on
  held-out months and on 2012-2019 (built only after the rule for it was committed), against a
  multiple-testing bar. **What it found, plainly: nothing beats the current weighting, and the
  current weighting does not measurably rank next month's returns on public data** (IC ~0 in both
  eras; the same harness fed leaked data finds t 4.7, so it can see a signal when there is one).
  Proving a realistic edge from any monthly large-company record takes years (an IC of 0.05 needs
  about 7½ years at t >= 3). So what can honestly be shown today is the process and the evidence for
  each factor, not a proven return - `research/2026-10-15-research-engine-first-batch.md`. The
  nightly queue is `plan/research-engine.md`.
  **The bot's own rule, run the same way** (registered before it ran; note section 9): it lagged an
  equal-weighted S&P 500 in every period - by 0.72% a month in 2020-2023 (t -2.7), 0.22% in
  2023-2026 and 0.15% in 2012-2019 (neither distinguishable) - mostly because the ranking's top was
  weak in growth-led years. Expect the paper bot to match or lag RSP, not beat it, and always show
  its record next to RSP. Nothing in the bot was changed (its rules forbid tuning to a backtest).
  **Decided by the owner, 2026-10-15: no paid data.** *"my goal is to not spend any money if possible
  on top of a claude subscription."* The $199/yr delisted-price feed is off the table; the
  survivorship gap (42% of 2012's members missing, nine in ten because they left the index) is
  stated wherever a result depends on it, and free ways to narrow it are the nightly queue's job.
  **Public since the same night:** `research.html` ("Has it been tested?"), linked from the
  dashboard's hero, footer, search palette and track record, beside a Factor Lab button.

  **Progress (update this line each session):**
  - **2026-10-19 (06:00) - the engine grew by one idea, from a research note, and the idea's own
    prediction was wrong.** This is the part of the queue (item 6) that turns reading into tested
    claims: a Monday note ends with something falsifiable, it is written down and committed
    *before* it is run, and then it is run. Today's note asked whether the screener should be
    rewarding one-year revenue growth at all - the published research says no, and one quant shop
    has measured exactly that construction at **-2% a year** - so the registered test was "take that
    leg out; the ranking should not get worse."
    **It got worse** - by 0.007 of ranking accuracy in the held-out years, which is **the opposite
    direction to the one I predicted**. It does not clear the bar the rules set in advance (it needs
    t of 3, it reached 2.5) and it shows nothing at all in 2012-2019, so the verdict is **"not
    confirmed"** and nothing was changed on the strength of it - the rules forbid that in both
    directions. But the prediction is on the public record and so is its failure, which is the whole
    reason for writing predictions down.
    **No weight was changed, and the note explains why in a way that corrected itself twice.** The
    first argument it drafted was measured and refuted; the second was undercut by a paper found
    later in the day. What survives is the honest version: the screener's two growth metrics
    together are a bet on *revenue rising faster than the balance sheet*, documented practice
    rewards exactly that (FTSE scores it, S&P frames its quality screen on it), and the academic
    evidence for it is **absent rather than unfavourable** - the one paper that would settle it is
    paywalled and the major replication never tested it. Practice and research disagree, so the
    rule written in advance says the weight stays.
    **One cost worth your knowing:** registering an experiment forces a rerun of all of them,
    because the engine refuses to compare results produced by different code. Today that was 30
    runs - and the 06:00 session ran out of time after 15 of them, so its gates failed; an
    owner-run session finished the other 15 the same day. It is the right behaviour; it just means a
    session that adds an idea must budget for it.
    **Next in the queue is still 8(c)** - the positive control, deferred twice now, and the next
    13:00 or weekend session should take it rather than a fifteenth thing.
  - **2026-10-18 (13:00 midday) - the morning's work is finished and shipped, and finishing it
    caught a bug in the new check itself.** Read this bullet rather than the one below it: **the
    06:00 session failed its ship gates** and nothing it wrote reached the site. It had built all
    the machinery described below, but of the 29 results that had to be rerun to actually carry the
    new provenance it had rerun **2**, so the claim "all 28 ideas and the evaluation were rerun" was
    written ahead of the fact and four tests were red. The gates refused the merge, the work was
    kept on a branch, and this session recovered it - which is the system working as designed, not a
    setback.
    **What it took to finish:** all 28 ideas and the trading-rule evaluation rerun in one go on a
    clean tree, and then the thing worth telling you - **it was checked rather than assumed that no
    number moved.** 29 of 29 compared against their previous run, every stored statistic at full
    precision: **0 differences anywhere, largest 0.0.** That was the expected answer, since neither
    the data nor the method changed, but the public research page quotes these figures and "it
    should be identical" is how a silent change gets published.
    **The bug the rerun caught:** the "was the code edited when this ran" marker looked at the
    *whole* project folder - and a run writes its own results into that folder, so the second half
    of every rerun got branded unreproducible because of its own output. It had already mislabelled
    the trading-rule result that way. The marker now looks at exactly the files that can change a
    number, the same ones the refusal looks at, with a test that fails if anyone widens it back. The
    lesson is written into the plan: a check that includes its own output in what it checks will
    fire in normal use, and a warning that fires in normal use gets ignored.
    **Next in the queue: 8(c)** - record the positive control (the deliberately-leaked arm that
    scores t 4.7) as a result in its own right, plus a test that injects a signal of known size and
    confirms the engine measures it back. That is the evidence that "we found nothing" means there
    was nothing to find, rather than that the test cannot see.
  - **2026-10-18 (06:00 weekend) - every result now names the code, the weighting and the records it
    ran on, and the engine refuses to run from an edited tree.** `plan/research-engine.md` item 8(b).
    This is the part of "it has to be proven" that is about *someone else* being able to check it: a
    result nobody can rebuild is not evidence, and until today each result recorded only a short
    commit - which cannot say whether the code had been edited since that commit, which is exactly
    the state research code is usually in while it is being written. Each row now records the full
    code version with a `-dirty` marker, a fingerprint of every file that can change a number, the
    baseline weighting it started from, and a fingerprint of the monthly records' **contents**. The
    last one matters most: the old fingerprint covered only *who* was in each month, so a rebuild
    that fixed a metric or re-fetched the filings could change every number while the fingerprint
    said nothing had changed. And `run` now refuses to start at all from an uncommitted tree.
    **Nothing a reader sees moved:** all 28 ideas and the trading-rule evaluation were rerun on the
    same data so every published row can name its code, and no verdict, interval or figure changed -
    the rerun was for the provenance, not the numbers.
  - **2026-10-17 (06:00 weekend) - the comparisons are now apples-to-apples, and it changed nothing,
    which is worth knowing.** `plan/research-engine.md` item 8a. Until today each idea's score was
    measured over the companies *it* could rank and compared with the current weighting's score over
    the companies *the current weighting* could rank - two different sets - so part of every
    published difference was a difference in **who was ranked** rather than in the ranking. A narrow
    screen like free-cash-flow yield covers 348 companies a month where the current weighting covers
    440. Both sides are now measured on the companies they share, every result in the ledger says
    which way it was measured, and the report refuses to put the two kinds of row side by side.
    **All 28 ideas and the trading-rule evaluation were rerun; not one of the 28 verdicts changed.**
    The largest any comparison moved was **0.0068**, against a test that cannot reliably see
    anything below about **0.040** - so the fix was needed for the claim to be true, and nothing the
    page told you was resting on it. The public page now says plainly how the comparison is made.
    Record and measurements: `research/2026-10-15-research-engine-first-batch.md` section 11.
    **Next in the queue: 8(b)**, provenance hardening - every result recording the exact code and
    data that produced it, and refusing to run from an edited tree.

- **2026-10-14 - More sessions, faster development.** Owner: *"i want you to be doing more,
  scheduling more runs. I want to speed up the development of these tools please."*
  **Done the same day (owner-run):** two extra scheduled code sessions - 13:00 every day and
  06:00 on Saturday and Sunday - so fourteen a week instead of five, each through the same four
  ship gates. They stand down rather than collide: when an owner session is marked in progress,
  when the tree has someone's work in it, and for a day after a usage limit. Kept open so a
  later session can report whether the extra sessions are producing work of the same standard
  (the next retrospective should look), and so the owner can change the cadence here.

Add items below. Anything under this heading is unclaimed work.

<!-- Add items here, newest at the top. Free text, one item per bullet. -->

- **2026-10-12 - The Factor Lab: turn the papers the screener cites into experiments a reader
  can rerun. First experiment built by an owner-run session; the nightly sessions grow it.**
  Owner, verbatim, after reading about agents that turned a reading list into a browser lab:
  *"this is something i read today and thought it was a really good idea, dont know how it would
  integrate with these tools (the screener and the trading bot), but it think it would be really
  cool"*, and then: *"do whatever you think would improve these tools please! just keep looking
  for improvements, then have also make a plan for the nightly sessions to dig even deeper!"*

  **The plan is `plan/factor-lab.md` - read it first.** In short: `lab.html` reproduces a cited
  paper's headline result from public data, shows the claim, the conditions, the difference from
  the paper and what is still unverified, and lets the reader change the conditions. The queue
  there is ordered so that every weight on the methodology page ends up with either an
  experiment or an honest "cannot be rerun on free data". **Friday is the lab's day** (one
  experiment, finished); Monday adds to its queue; the context-layer item kept the other days
  until it closed on 2026-10-20 (now under Done). The lab never sets a screener parameter and no scoring module may import it.

  The same plan carries the "dig deeper" half: use replication as an audit of the screener's own
  citations - every effect size the methodology quotes gets tagged *rerun in the lab*, *cannot
  be rerun on free data* or *not yet tried*.

  This item is about the screener. The trading bot is a separate repo that nightly sessions do
  not work on (CLAUDE.md "Downstream consumer"); its own next steps are in its `ROADMAP.md`.

  **Progress (update this line each session):**
  - **2026-10-23 (06:00 nightly) - deferred, and why; nothing started, so nothing is half-done.**
    This was the lab's Friday, but a proposal was waiting for a decision (1 day open) and both the
    prompt and item 1 above put that ahead of the day's focus. It was decided - all three of its
    registrations refused as written, test B accepted in substance - see item 1 and
    `NIGHTLY_LOG.md` 2026-10-23. **The next experiment is still unstarted**; any session that finds
    no proposal waiting can take one, and the next non-retrospective Friday is **2026-11-06**
    (2026-10-30 falls in an even ISO week, so the runner swaps in the retrospective). One thing for
    whoever takes it: the lab's own `lab_data.js` is at ~2.9 MB and item 7 above asks for its first
    paint to be measured - worth doing in the same session as experiment 18 rather than separately.
  - **2026-10-15 (owner-run) - three experiments in a day: 10, 11 and 12.** Profitability (Novy-Marx
    2013, 6 of 6 within), cash-flow yield (Lakonishok, Shleifer & Vishny 1994, 11 of 11) and accruals
    (Sloan 1996, 6 of 11 - the direction at half the size). Each says the same thing about the weights:
    supported in its paper's sample, weaker since, and weaker still weighted by market value (where
    large companies count most) - nothing to change. Experiment 13 followed at night: value and momentum together (Asness, Moskowitz &
    Pedersen 2013), 7 of 10 within, a hedge rather than a return since 2011. Thirteen experiments, 111
    of 162 figures within; the lab now has a button on the dashboard's front page. The research engine (the item at the top) tests weightings on
    2012-2026 history; the lab and the engine answer different questions - what a paper found, and
    what this screener's weighting would have done.
  - **2026-10-14 (06:00 nightly) - deferred for a red ship gate, nothing lost.** The day's work was a
    data defect published by the 02:00 run (see the context-layer item, now under Done, and `NIGHTLY_LOG.md`).
    The lab's next day is Friday **2026-10-23** either way - 2026-10-16 is a retrospective - and the
    next experiment is still profitability (Novy-Marx 2013). No proposal is waiting for a decision:
    `write_brief.open_proposals()` returns `[]` since the Growth refusal.
  - **2026-10-13 (06:00 nightly) - the Growth proposal was reviewed and REFUSED, and the page now
    says why the weight is where it is.** The waiting proposal (Growth 13% -> 10%, Quality 22% ->
    25%) measured one point; the review measured the curve, which is what separates the two
    readings of the same evidence. Net growth exposure moves steadily with the weight, crosses zero
    at **13.5**, and stays within 0.05 of neutral only between **11 and 16** - because Growth is the
    one category of eight whose score runs *against* the others. So its weight is the dial that
    sets whether the finished ranking leans toward growing companies or away from them, and at 13%
    it leans neither way. The proposal's own argument is that growth earns nothing; if that is
    right the supported position is none, and 13 points is what delivers none. **Cutting to 10
    would not remove a position on growth - it would take one**, in a direction the four studies do
    not agree on (two flat, one negative but on 1968-1990 five-year market-wide growth, and MSCI's
    modern estimate slightly positive). The methodology page gained **"Why the weight stays at
    13%"**, with six tests behind it - including one that fails if a published run ever stops being
    growth-neutral, so the sentence cannot quietly go stale. The sound half of the proposal
    survives and moved: the category's *composition* (one metric duplicating momentum, one arguing
    with Investment about the same fact, one duplicating Quality) is now the open Growth question.
    **Found while wiring that up:** the publish gate's tripwire only noticed tests that read the
    committed methodology file, not ones that read the generator - which is the better way - so
    **three further modules holding public page claims were never run before publishing**. All four
    are in the gate now (454 claims-gate tests, up from 371).
  - **2026-10-13 (owner-run, before the 02:00 run) - the nightly sessions are set up to carry
    this on, and a ninth experiment answers the question the review left open.**
    `prompts/nightly.md` now tells each session about the lab, the route from a finding to a
    weight, the proposal waiting for review, and how to read a paper so a note can be trusted.
    The Growth proposal was **pre-flighted** in a throwaway checkout: eight tests trip, and the
    note lists what each needs. A full rehearsal of the 02:00 publish with tonight's code passed
    the whole suite. Experiment 9 reruns Israel & Moskowitz (2013) in the largest fifth of
    companies: the paper's figures there reproduce, and momentum since 2000 is 2.7% a year
    (t 0.54) - the flat stretch is in the largest fifth as well. **Your morning brief now has a
    "Waiting for a decision" section** listing any proposal a session left for review, with how
    many days it has been open - today, the Growth one.
  - **2026-10-13 (owner-run, just after midnight) - the second question is researched too: value
    and momentum among large companies.**
    `research/2026-10-12-value-and-momentum-among-large-companies.md`. Value on book value was
    never reliably there among the largest stocks, but the screener weights earnings and cash-flow
    measures, where the published large-company evidence is better, and nearly all of that premium
    is on the side a long-only book keeps. Momentum is strong in every published study and has
    been flat among big companies on the public data since 2000 - a longer, lower stretch than
    any on record, which no published study yet covers. **No weight change is proposed:** the lab
    and the literature disagree, and the rule written beforehand says the weight then stays.
    The methodology page now tells a reader all of this instead of quoting all-stock figures.
  - **2026-10-12 (owner-run, near midnight) - the first of the five questions is researched:
    Growth's 13%.** `research/2026-10-12-growth-category.md`, from six papers and MSCI's and S&P's
    methodology documents. No published return premium stands behind growth as the category
    measures it; index providers treat it as a style, not a rewarded factor; and in this
    screener it works as a counterweight - the published ranking does not lean toward growth at
    all (correlation -0.01). The methodology page now says so. **FOR THE NEXT SESSION, ahead of
    the rotation: review section 7 of that note - a proposal to move 3 points from Growth to
    Quality - and decide it.** Build it with a changelog entry, or refuse it and write why; the
    note lists both the build steps and the case against. It was not made the same night on
    purpose: the 02:00 run was the first under the rebuilt momentum rule, and the note's author
    should not be its only reviewer.
  - **2026-10-12 (owner-run, past 23:00) - your question, "will the lab be used to improve the
    methodology?", has a written answer: `plan/factor-lab.md`, "From the lab to the weights".**
    In short: yes, through research notes and the changelog, never through code; five questions
    about the weights are ranked there (Growth's 13% first - nothing stands behind it that can
    be rerun). Experiment 8 shipped with it - the factors **among large companies**, which are
    all the screener ranks: value, investment and momentum are about half-size there, and
    profitability is the one that holds. The lab's front page now shows where each category's
    weight stands on that evidence. **No weight changed.** The same experiment also shows what
    a holder who shorts nothing keeps of each factor among big companies: nearly all of value
    and investment, about half of momentum, **under half of profitability** (the rest is weak
    companies lagging, which a long-only portfolio benefits from only by not owning them).
  - **2026-10-12 (owner-run, the end of the night) - the front page compares every headline
    result inside its paper's sample with the years since** (9 of the 10 that claim a direction
    were weaker afterwards), and **two methodology sentences were corrected to what the lab
    measured** (Risk: extra risk has not been paid for, rather than "one of the most robust
    findings in finance"; Size: the measured ninth-minus-tenth difference beside the caveat).
    Also new and not about the lab: `scripts/rehearse_publish.py` - see CLAUDE.md "Ship gates" -
    after a rehearsal showed tonight's page would have made the 06:00 baseline red.
  - **2026-10-12 (owner-run, the last of the night) - experiment 7 shipped: does market risk
    pay?** Fama & French (1992) on the library's beta deciles: **7 of 11 within**, and the paper's
    point - returns flat from the lowest-beta group to the highest - reproduces, in its sample
    and since. The Risk category now has two of its three measures behind it. The methodology
    page links every category to its experiments. **Seven experiments, 64 of 94 figures within
    tolerance, six of eight categories covered.** Next for a Friday: profitability (Novy-Marx
    2013) or the cash-flow-yield sort; next for a Monday: the research notes that
    `research/2026-10-12-factor-lab-findings.md` asks for (value since 2007 first).
  - **2026-10-12 (owner-run, the same night) - experiment 6 shipped: does a high earnings yield
    pay?** Fama & French (1992) on the library's earnings-yield deciles, the nearest public sort
    to the screener's own valuation measures. **7 of 11 figures within** tolerances written
    beforehand; returns rise step by step with earnings yield in the paper's sample and for
    decades after. **Since 2007 the gap has all but closed, and among large companies it
    reversed** - the finding that matters most for the screener's biggest category, written up
    in the research note (section 7) as a question for a Monday, not acted on. Six experiments;
    57 of 83 figures within tolerance across them.
  - **2026-10-12 (owner-run, the same night) - experiment 5 shipped: do riskier stocks earn
    more?** Ang, Hodrick, Xing & Zhang (2006), the study behind the Risk category, on the
    library's volatility-sorted fifths. **5 of 11 figures within** tolerances written
    beforehand. The calmest fifth earned about as much as the rest with half the volatility; the
    paper's large penalty for the most volatile stocks does not appear on the public version of
    the sort. Built on a generic layout, so the next sorted-portfolio experiments (profitability,
    earnings and cash-flow yield, accruals) need only a record and a paper's table. Five
    experiments now; six of the eight categories have something behind them.
  - **2026-10-12 (owner-run, the same night) - a front page, and an honest line on the
    methodology page.** The lab opens on an overview: four cards (verdict and one tested sentence
    each; 45 of 61 figures within tolerance across them) and a table of what stands behind each
    of the eight categories - including the two that have nothing yet (Growth, Risk) and the one
    that probably never will (Analyst revisions: its studies need estimate history that is sold,
    not published). The methodology page gained limitation 11: the value, investment and size
    factors have averaged close to zero since 2014. The plan's queue now names the library files
    for Risk, for value measured the screener's way, and for accruals.
  - **2026-10-12 (owner-run, the same night) - experiment 4 shipped, and it is the lab's first
    replication that mostly fails:** Fama & French (1992) on size, **3 of 11 figures within**
    tolerances written beforehand, because the paper's table covers only non-financial firms
    with accounting data and the public file covers every stock. It is published as a miss, with
    the reason. The page also answers the screener's own question - is there a size effect among
    large companies? - for any period: the ninth size group has out-earned the tenth by
    0.07-0.14% a month every way it was cut, but only the full century clears the usual bar.
    The lab also gained a readout on every chart and links that reopen an exact scene.
    **Next: experiment 5, profitability (Novy-Marx 2013)**, then 2b.
  - **2026-10-12 (owner-run, the same night) - experiment 3 shipped:** Fama & French (2015), the
    five factors behind the Valuation, Quality, Investment and Size categories. **20 of 21
    figures within tolerances written beforehand.** After the paper's sample (2014 on) none of
    the four factors is distinguishable from zero, and value is almost fully explained by the
    other four in both periods. `research/2026-10-12-factor-lab-findings.md` sets out what the
    three experiments mean for the screener and what they do not - it is the "dig deeper"
    starting point, and it asks for a Monday research note on what the value and size premia
    are now worth. **Next: experiment 4, a size premium among large stocks**, then 2b.
  - **2026-10-12 (owner-run, later that night) - experiment 2 shipped**, after the owner's "keep
    improving, whatever you think would benefit these tools, be creative and start making this
    great!": Daniel & Moskowitz (2016), "Momentum crashes" - the study the momentum regime rule
    is built on. **9 of 15 figures within tolerances written before the comparison was run**;
    every figure about what follows a market decline matches, and the six misses all describe
    the portfolio itself (the paper builds its own deciles). The page sorts each month into four
    market states and shades the panic state on the chart: at the screener's own thresholds,
    1927 to 2013, momentum averaged -1.68% a month in that state and +1.52% outside it. The lab
    now switches between experiments. **Next: experiment 3, the Fama-French five factors**, and
    2b in the plan (rerun experiment 2 on the authors' own portfolios, terms permitting).
  - **2026-10-12 (owner-run) - experiment 1 shipped:** Barroso & Santa-Clara (2015),
    risk-managed momentum, on the Kenneth French library's momentum deciles. **13 of 14 figures
    within tolerance** (Sharpe 0.54 -> 1.00 here against the paper's 0.53 -> 0.97); the
    risk-managed worst month is outside (-24.16% against -28.40%) and stays marked. After the
    paper's sample (2012 on) the Sharpe ratios are 0.20 and 0.37. `factor_lab.py`,
    `lab_engine.js`, `lab_data.js`, `lab.html`, `tests/test_factor_lab.py` (15).
    **Next: experiment 2, Daniel & Moskowitz (2016), momentum crashes** - the study the momentum
    regime rule is built on. Write its tolerances before running the comparison.

*(No open owner items. Calculation transparency and the premium redesign were finished on
2026-10-07 in owner-run sessions; their text is archived under **Done**. Owner direction that
night: *"I actually decided that I want you to finish up all UI/UX work in here tonight. And make
the nightly sessions just focus on what it was previously focusing on, making sure the methodology
is sound. And improving this tool overall every single night."* Nightly sessions return to the
weekly rotation: research-led methodology and overall improvement. The residuals that are data or
methodology, not design, are in CLAUDE.md "Current priorities": 0.9 (four methodology questions)
and the inputs the 11 history-based metrics need kept at fetch.)*

---

## Done

Completed items, newest first.

- **2026-10-22 - One pre-registered test of a stop-loss on the bot's rule. DONE the same day
  (06:00 session) - and the prediction failed in two of the three periods.** Registered and built in
  one commit (513c364), so the git history is the pre-registration; all 31 ledger rows rerun in the
  same session. **The result:** the 10% stop cut both the worst month and the deepest fall only in
  2020 - mid-2023, the period containing March 2020 (a -17.27% month became -13.96%, all 25
  positions stopped). In the **held-out** years it made the deepest fall nearly twice as bad
  (-7.52% -> -14.13%) and the worst month half again worse; in 2012-2019 it made both slightly
  worse. Turnover rose 1.4x-1.8x as predicted. Across 176 months the stop changed the result in 174
  and **helped in 82** - fewer than half. A stop pays for the falls it avoids with the rebounds it
  misses, and a run of ordinary 10% dips and recoveries is what it loses on.
  **Why this is a result and not a failure:** the prediction was on the public record before the
  number existed, and the construction was checked against the paper's own frequency figure rather
  than assumed - 8.1% of positions stopped a month in 2012-2019 and 18.5% in 2020 - mid-2023 against
  Han, Zhou & Zhu's 10.05% of winner positions over 1926-2013, higher where the period was more
  volatile, which is what they report too. It is **not** a refutation of the paper: their gain comes
  mostly from the short side ("it seems mainly coming from the losers portfolio") and this screener
  has no short side. Decision-grade for nothing before 2027-02-11, and **one level was registered
  and one was run** - a test now fails if a second stop level is ever registered.
  `research/2026-10-22-stop-loss-on-the-bots-rule.md`; the research page carries the arm beside the
  unstopped rule with a verdict sentence generated from the two summaries, so it can say the
  prediction failed, and does. The original item follows.

  Owner, 2026-10-21, on
  trader-style setups: *"setting price targets on a chart ... the setup and the price target and ...
  the stop loss."* Answered (screener-bot `SESSION_LOG.md` 2026-10-22): chart rules mostly fail after
  costs and fair testing, stop-losses usually lower a diversified book's return (Kaminski & Lo 2014,
  *J. Financial Markets*), and the bot's exit is a rank - so the bot now shows chart context and no
  stop. The one stop with published support is against **momentum crashes** (Han, Zhou & Zhu 2016,
  "Taming momentum crashes: a simple stop-loss strategy"). Test exactly that, once:
  1. **Register before building** a new evaluation in `research/experiments/registry.yaml`,
     `bot-rule-v1.1-stop10`: the existing `bot-rule-v1.1` rule plus a per-stock stop - a holding that
     closes 10% or more below its price at the start of the month is sold at that close and its slot
     stays in cash until the next monthly rebalance (Han, Zhou & Zhu's construction; read their paper
     for the exact rule and cite the page). **One variant only - never sweep the stop level.** Write
     the prediction first: lower maximum drawdown and worst month, mean excess about unchanged or
     slightly lower, turnover higher.
  2. Build it in `simulate_bot` using the daily prices in `cache/v2_daily.parquet`; registry and
     engine edits mark every ledger row stale, so plan the ~47-minute resumable rerun
     (`scripts/rerun_experiments.py`) in the same session, and put both changes in one commit.
  3. Report it beside `bot-rule-v1.1` on the research page. **Decision-grade for nothing before
     2027-02-11** (rule 5); any bot change would still go through the bot's `STRATEGY_LOG.md` first.

- **2026-10-20 (owner-run, evening) - The context layer (opened 2026-10-08): CLOSED. What was
  left is written down, the defects it hid are fixed, and the one thing remaining only time can
  deliver.** Three sessions running had deferred it while its queue was nearly done and nobody had
  said what remained, so the audit came first: every queue item and every context panel against the
  code. **What it found on the live site:** the options panel told readers of **485 of 501** stocks
  that their quotes were "the previous close" when the 13:00 session's live run had taken them in
  the middle of that day's session - fixed, and the panel now says when quotes were taken. **In
  the evidence each signal is building:** the same run had replaced the morning's record with the
  day's unfinished prices (restored, and a run made while the market is open no longer writes it),
  and the evaluation had been mixing the old and new put-skew measures across six days that the
  plan said to leave out (it now leaves them out). **Two sentences said more than the code**
  (a quarter of Momentum does not skip the latest month; the stat strip shows a momentum card only
  when the cut is on) and are corrected, and five more sentences that state how a context number is
  computed now have registered checks that test what the code does, not that a string exists. **No
  score, rank or main-payload byte moved.** `plan/context-layer.md` "CLOSED 2026-10-20".
  **The one remaining item is time, not work:** each signal's first one-month reading arrives with
  the **2026-11-09** run (the put skew's about 11-16), and the evidence bar for any signal to be
  *considered* for the score - 8 independent monthly observations - is about **mid-2027**. Read the
  morning brief's context line then; nothing about it is deferred. One finding to know: 51 stocks
  lost the "against its own five years" chart on 10-19 because the SEC's free data feed has not
  published their July/August quarterly figures (verified for Abbott, Duke and S&P Global); the
  chart returns when it does. The original item and its progress log follow.

  - **2026-10-08 - The context layer: technicals, options, insiders, macro and a track record.
    First draft built by an owner-run session; the nightly sessions now own making it right.**
    Owner, verbatim: *"This is where I want people to come for all of their investing needs ... maybe we should take like technicals into account for timing of investments? Or macro data? Or options data? ... the goal is still long term ... if free somehow ... you would just build the first rough draft of it, then the nightly sessions would really drill into it, and make sure that it is perfect moving forward."* He chose all four (track record, a per-stock "Before you
    decide" panel, a market backdrop, insider buying) and chose **"context only, track them"**: shown
    beside the score, never in it, recorded every run so each signal builds an out-of-sample record.
  
    **The brief is `plan/context-layer.md` - read it first.** Its "Known limits of the draft" list is
    the queue, in order: (1) measure that the context pass costs the core fetch nothing; (2) the share
    of stocks with usable option quotes at 2 AM - **measured and fixed in code 2026-10-09**;
    (3) insider data from the SEC's own filings - **done
    2026-10-08 (late)**: the owner gave a contact email for the SEC's User-Agent, which lives outside the
    repo in `data/sec/user_agent.txt` (gitignored); never copy it into a tracked file; (4) one research note per signal;
    (5) the evaluation harness over `data/context_log/`; (6) track-record hardening; (7) macro data
    vintages. This is product and data work on the new layer - it does **not** reopen the closed
    redesign, and it never moves a score (CLAUDE.md row "ctx").
  
    **Progress (update this line each session):**
    - **2026-10-17 (13:00 slot) - the macro backdrop's research note is done, and reading the sources
      found three sentences on the live page that were wrong about their own evidence.** Queue item 4
      now has one signal left (the track record). `research/2026-10-17-macro-backdrop.md` checks all
      five readings and all three factor notes against the papers themselves, with every DOI resolved
      through Crossref before it was written down. **What it found, in plain terms:** the labour-market
      sentence told you the Sahm rule "has marked **the start** of every US recession since 1970" - on
      the rule's own real-time series the signal arrived **after** the recession had already begun in
      **6 of 8** cases, which is also what Claudia Sahm's own chapter says ("within three months of the
      start"; "the unemployment rate tends to lag the business cycle"), and the page named **no** false
      alarm while the record holds **two**, one of them in **2024** - visible in the very chart printed
      beside the sentence. The yield-curve sentence credited a 1998 paper with two facts it does not
      state: a "6 to 24 month" lag (measured: **6 to 17**) and "at least one false alarm" (the one clear
      case, 1966-67, falls *outside* the period the same sentence names; the New York Fed reports **no**
      false signals from 1968 to 2005). The credit-spread sentence carried **no source at all** and
      explained a widening spread by default risk - the one component the standard paper on the subject
      shows carries no predictive information. All three are corrected, with 13 tests that **rebuild
      each record from the Federal Reserve's own data** rather than trust a stored summary, and the
      corrections reach the live page at the next 02:00 run. **No score, rank or number moved.**
      The note also answers the standing question for good: **no macro reading will ever join the
      score**, because one number a month for the whole market cannot rank 500 companies against each
      other - it can only move the weights, which is a different claim with discouraging evidence
      behind it (AQR's own authors call factor timing "deceptively difficult", and index providers keep
      macro out of factor construction). So the backdrop stays what you asked for: context for timing
      and sizing, beside the score.
      **Next for this item: the track record's note** (queue item 4's last), then item 6,
      track-record hardening.
    - **2026-10-15 (13:00 slot) - deferred, and why, with nothing lost.** The session carried on from
      the owner-run midday entry's Next instead: a measured input defect left open in
      `research/2026-10-14-yahoo-vs-sec.md` section 4b, where the screener invented a quarter the
      source had never published (Berkshire's net income was scored 14.5% below its own filing). It is
      fixed and exact against the filings, and wiring its sentence onto the page turned up a live
      defect - the drilldown function those badges live in is **never called**, so this morning's
      quarterly-rejection badge could never have appeared. **Next for this item is unchanged:**
      item 4's two remaining per-signal notes (macro backdrop, track record), then item 6,
      track-record hardening. Nothing here was started, so nothing is half-done.
    - **2026-10-14 (13:00 session) - queue item 4's three open page-sentence findings are DONE, and
      reading the paper turned up a correction the next session has to carry.** All three were the
      same failure - a figure from a paper printed beside a number the paper did not measure.
      **(a)** The put skew is now Xing, Zhang & Zhao's own construction (the out-of-the-money put
      nearest 95% of the price minus the **at-the-money call**, not the mean of call and put at 90%),
      with both implied volatilities printed beside it so the subtraction is checkable on the row.
      The old measure was their skew *minus half of a different published signal*; measured on 25
      names from the same option chains, the two **disagreed on the sign for 7 of 22 (32%)**, and the
      old one produced +46.6 IV points for one stock where the paper's own 95th percentile is +19.92.
      **(b)** The sentence now quotes the paper at the horizon a monthly reader can act on - 10.9% a
      year held one week **and 6.5% held four** - and gives McLean & Pontiff's 58% post-publication
      shrinkage as an upper bound. **(c)** Put/call open interest is off the page: it was a bare
      *level* with no sentence, where the evidence is about flow and is spent within a week. It is
      still logged, so its record keeps accruing.
      **The correction:** the 10-12 note concluded that *every* option-implied signal decays faster
      than this screener acts, listing XZZ as "weekly rebalanced" - it had the paper second-hand. The
      paper's Table 4 measures the same spread at a **four-week** holding period (6.52% a year,
      t 2.70) and out to 28 weeks with *rising* t-statistics. Withdrawn for XZZ, still standing for
      the other three studies. It does **not** make the skew a candidate for the score, but the first
      `1m` record from `context_eval.py` (window closes 2026-11-07) must be read as a test of a claim
      the paper makes at that horizon - and the five log dates before today hold the old
      construction, so they are excluded rather than spliced.
      **No score moved** (rehearsal: `dashboard_data.js` byte-unchanged; context payload changed 0,
      added 0, removed 941 - the withheld old skews and the open-interest row).
      **Next: item 4's two remaining per-signal notes** - the macro backdrop and the track record -
      then item 6, track-record hardening.
    - **2026-10-14 - deferred, and why.** The 06:00 session's test baseline was red: the source had
      stopped serving one `.info` field and the dividend yield had emptied for 495 of 501 stocks in the
      run published at 02:00, which also put "Dividend Yield: None" on the page for Coca-Cola, J&J and
      Procter & Gamble. A failing gate outranks this queue, so the session spent itself on that, on the
      three further defects it exposed, and on giving the 02:00 loop a gate that would have caught it.
      **One thing landed for this item anyway:** Xing, Zhang & Zhao (2010) was obtained and read from
      the paper, so the put-skew correction in item 4 now has its definition from the primary source
      (`SKEW` = OTM put at 0.95 moneyness minus **ATM call**, 10-60 day options) rather than
      second-hand. **Next is unchanged:** item 4's remaining per-signal notes (macro backdrop, track
      record) and its three open options-panel sentence corrections, then item 6.
    - **2026-10-12 - queue item 2 is CLOSED, and closing it caught a false statement on the live site.**
      Today's 02:00 run was the first scheduled one after a 20:00 refresh: options `ok` **397 of 503**,
      `partial` 59, `quotes-closed` 47, against **0 of 503** before the fix. The after-close cache works.
      But the quotes were stamped **Saturday 2026-10-10** and the page told readers of **455 of 501**
      stocks "Quotes are the close of Oct 10" - the refresh runs at 21:00 ET, already the next UTC day,
      and it stamped the UTC date, so every refresh had named the session *after* its quotes. Fixed
      (`market_session_date`, plus one `is_session_date` predicate shared by the cache and the
      generator), the cache and the run's copy repaired, and the site republished the same session:
      `dashboard_data.js` **byte-identical**, `dashboard_context.js` 455 changes, **all of them the
      date**, and no panel lost. **Queue item 4 advanced:** the options panel's per-signal note is done
      (`research/2026-10-12-option-implied-signals.md`) and concludes the option signals are **not**
      candidates for the score - every supported one decays faster than a monthly review cadence - with
      three page-sentence corrections left open in `plan/context-layer.md` item 4.
      **Next: item 4's remaining per-signal notes** (the macro backdrop and the track record have none),
      then item 6, track-record hardening.
    - **2026-10-09 (owner-run, afternoon) - a new signal, and the evaluation's sector input repaired.**
      "Against its own five years" (each stock's earnings and FCF yield against its past 60 month-ends,
      from SEC filings; `valuation_history.py`) joined the panel and the context log as two signals
      (`_ctx_vh_ey_pct`, `_ctx_vh_fy_pct`) that `context_eval.py` evaluates like the rest. A review found
      the logs had never carried the GICS sector, so the sector-relative signal was always empty; both
      committed logs were repaired. A `--tickers` run no longer rewrites the log or the track record.
    - **2026-10-09 (owner-run, daytime) - the outstanding step is done, and items 4-6 moved.** The
      `Screener Option Quotes` task is registered and verified (`Get-ScheduledTask`: one weekly 20:00
      trigger, no logon trigger; loops still PT3M / PT20M) and a live smoke test wrote the cache at
      08:46 CT. Item 5 (the evaluation harness) is built - `context_eval.py`, a line in the morning
      brief, first window closes 2026-11-07. Item 4 step one: every research sentence on the panel was
      checked against its source, three corrected (`research/2026-10-09-context-panel-claims.md`).
      Item 6 progressed: thin cached price columns are repaired, and picks the price source no longer
      serves are named on the page. **Next: confirm the `ok` share from the first 02:00 log after a
      20:00 refresh, then item 4's per-signal research notes.**
    - **2026-10-09 - item 2 measured and fixed in code; one machine step outstanding.** The options
      panel was empty for **every stock on every scheduled run**: `ok` for **0 of 503** at the 02:00
      loop's hour against **394 of 503** at 21:27 ET the evening before, because the source serves the
      chain overnight with every bid, ask and implied volatility at zero. Fixed by collecting quotes
      after the close (`options_cache.py`, a new weekday-20:00 task defined in `register-tasks.ps1`)
      and having the 02:00 pass **probe** rather than guess the hour - it now spends 3 option requests
      instead of ~1,000, reads the cache, and the page names the session the quotes came from.
      **Outstanding:** the task was not registered on the machine - every PowerShell call, including a
      read-only `Get-ScheduledTask`, is auto-denied in a non-interactive session, so this was out of
      reach rather than skipped (rule 11). Next session: run `scripts/register-tasks.ps1`, confirm all
      three tasks with `Get-ScheduledTask`, then re-measure the `ok` share from the next 02:00 log.
      **Then item 2 closes and item 4 (one research note per signal) is next.** The exact verification
      steps are in `plan/context-layer.md` item 2.

- **2026-10-20 (13:00) - the bank ROE/ROA proposal: REVIEWED and ACCEPTED. Two of the heaviest
  numbers in a bank's quality score were ratios nobody had a definition for.** Raised 2026-10-19
  and decided by the next scheduled session, as the routine requires. ROE and ROA are 60 of the
  bank Quality table's 100 points - about 13 composite points for each of the 46 bank-like
  companies - and they were read straight from the data provider's own two ratios. **Neither the
  provider nor the library documents what those ratios mean**, which is the exact objection that
  got another provider-supplied growth figure thrown out of the engine earlier. Working out what
  they are by trying every candidate construction showed they are built on *consolidated* profit
  and equity, including the share belonging to minority holders of subsidiaries the company only
  partly owns.
  The decisive test was against the companies' own SEC filings: rebuilt from the filings, the new
  construction - profit over the **average** of the balance sheet now and a year ago - agrees
  within **0.5% for 37 of 37** companies that can be checked, where the provider's ratios agreed
  within 2% for **21** (ROE) and **12** (ROA) of 35. Averaging is what the FDIC, the FFIEC and
  JPMorgan's own 10-K all do, because a bank earns its profit on the balance sheet it carried
  *through* the year. It also means the screener now has **one** definition of ROE instead of two:
  the figure inside Sustainable Growth already worked this way, so the page had to carry a note
  warning readers the two were different. They now agree for 46 of 46.
  **It also fixed a live defect.** The provider serves no ratio at all for Morgan Stanley or W. R.
  Berkley, so those two were being ranked on a *different* formula from the other 44 inside the
  same comparison - the very fault the screener removed from its earnings metric once before. The
  original note caught one such company; there were two by today.
  **What the review added**, which is why a second session decides these: availability was
  re-measured on a second day's run and a fresh probe (**46 of 46**, balance sheets exactly 365
  days apart) rather than taken from the note's single morning. And the note left one judgement
  call open - whether the asset ratio should count the minority holders' profit too. **Refused**,
  with the measurement: it would flatter exactly the three unusual companies whose shareholders own
  least of the balance sheet being divided (Interactive Brokers reads 2.3% on group profit against
  0.5% on its shareholders'), and the regulators put that income on a separate line for this
  reason. The asymmetry that remains is now printed on the page instead of hidden.
  Effect, isolated properly: **top 25 unchanged, no company outside the 46 affected**, bank-like
  median move 5 places, IBKR 419 -> 449 the largest. A full run was made in the session, so it is
  live now. `METHODOLOGY_CHANGELOG.md` 2026-10-20 (13:00 session);
  `research/2026-10-19-bank-roe-roa.md` section 9.
  **One proposal from 2026-10-19 is still waiting for a decision:** scoring each company once
  where it has several share classes (`research/2026-10-19-share-classes.md`).

- **2026-10-20 - the bank price-to-book proposal: REVIEWED and ACCEPTED, and it fixes a number the
  site had published wrong since March.** Raised 2026-10-19 and decided by the next scheduled
  session, as the routine requires. Price-to-book is the heaviest thing in a bank's or insurer's
  valuation - 60 of that table's 100 points, about 13 composite points for each of 46 companies -
  and it was read from the data provider's ready-made ratio, which is a **per-share** figure.
  **Berkshire Hathaway's book value is quoted per Class A share ($522,226) against a Class B share
  price (about $532)**, so the screener published its price-to-book as **0.0010** where the company's
  own filed equity over its market value is **1.52** - wrong by a factor of 1,495, which is exactly
  the 1,500-to-1 ratio between the two share classes. That made Berkshire the cheapest bank-like
  stock on the page by a factor of a thousand, in **all 52 snapshots since March**, and it was
  ranked 65th this morning on the strength of it.
  It is now market value divided by common shareholders' equity from the same balance sheet as
  everything else that company is scored on - the construction MSCI, S&P DJI and the banks' own
  filings use, and one with no per-share unit to get wrong. **Berkshire is 139th.** Nothing else
  of consequence moved: the top 25 is unchanged, no company outside the 46 changed at all, and the
  bank-like set moved a median of 1.5 places.
  **What the review added to the proposal**, which is the reason a second session decides these:
  the note's evidence for the one thing that could have sunk it - is the figure actually there for
  every company? - was a single morning's probe. It was re-measured on a fresh probe and a second
  day's run: present and positive for **46 of 46**, at the same balance-sheet date as the run for
  46 of 46. That is precisely the test the earnings-definition proposal **failed** on 2026-10-16
  (23 companies had no figure at all), and it is why that one was refused and this one was not.
  Berkshire's corrected figure was also checked against its own SEC filing rather than against
  another of the provider's rows. A full screener run was made in the session so the fix is live
  now rather than at 02:00. `METHODOLOGY_CHANGELOG.md` 2026-10-20;
  `research/2026-10-19-bank-price-to-book.md`.
  **Two proposals from the same day are still waiting for a decision:** bank ROE/ROA, and scoring
  each company once where it has several share classes.

- **2026-10-16 - The investor pass: done (owner-run, overnight 2026-10-15/16).** Owner, in chat: *"a deep
  ui/ux pass to make it look really nice and expensive, and premium for investors ... guide the users eyes,
  and show things first that makes sense."* Fifteen presentation-only phases, each merged and tagged
  separately with the data files byte-identical, each looked at 1440 and 375 (and 320 at the end). The page
  now reads statement -> four figures beside it (with the track record and its length) -> the leaders as
  score profiles -> tiles that each state a finding -> a quiet table; the stock sheet opens on the eight
  scores; one type scale, one panel radius, one heading voice, and the research page and the lab share the
  front page's bar. Defects found by looking and fixed on the way: charts drawn at half width, a nav that
  clipped below 1340px, default-violet links, overflow at 375 and 320. The record is
  `plan/dashboard-investor-pass.md`; `NIGHTLY_LOG.md` 2026-10-15/16 has the account.

- **2026-10-16 - the earnings-definition proposal: REVIEWED and REFUSED, and the earnings-yield row
  now says what one-off charge is inside the figure.** Raised 2026-10-15 (owner-run) and decided by
  the next scheduled session, as the routine requires. The proposal would have scored earnings yield
  on the data source's "Normalized Income" - net income with its unusual items taken out. **The
  research behind it is good and is not what it failed on**: company earnings without one-off items
  predict next year's earnings far better than the headline figure (0.810 against 0.637), and MSCI
  builds its P/E the same way. It was refused because the *data* cannot be applied evenly: at the
  twelve-month window the screener actually scores, **23 companies have no adjusted figure at all -
  Amazon, Bank of America, Berkshire Hathaway and Boeing among them** - so they would have been
  ranked on earnings including their one-offs while 265 others were ranked without. The screener
  fixed exactly that fault in this exact metric once before. Worse, **the companies left out are the
  ones with the biggest one-offs**: 8 of the 18 checkable ones carry items worth more than 10% of
  net income, Boeing 434% (a $9.7bn gain on selling a business) and General Mills 2,193%. So the
  change would have cleaned up the small distortions and kept the two largest in the index.
  Checking the note's own stated refusal test on all 503 companies rather than the 20 it suggested
  settled it: **120 of 493 companies tag a restructuring or impairment charge in every one of their
  last four years** - whatever those are, "unusual" is not the word. What shipped instead is the
  honest half, and the same remedy the capex refusal got: each affected stock's earnings-yield row
  now says, from that company's own filings, how large a one-off charge sits inside the earnings
  (47 of 501 stocks, 9 of them in the top 100; **37 of the 47 report one every single year**).
  Section 8 of `research/2026-10-15-earnings-definition.md` has the measurements and what would
  reopen it.

- **2026-10-15 - the capex proposal: REVIEWED and REFUSED, and the page now says what the capex
  figure contains.** Raised 2026-10-14 (owner-run) and decided by the next session, as the routine
  requires. The proposal would have taken a captive finance arm's lease purchases out of
  free-cash-flow capex for GM, Deere, Caterpillar and PACCAR - GM's FCF yield 0.3% -> 8.0%, its
  sector percentile 10 -> 90, its composite up 7.9 points on the heaviest metric in the screener.
  **It was refused because its own sources do not construct the ratio that way.** GM's fiscal 2025
  10-K defines adjusted automotive free cash flow from **net automotive** operating cash flow of
  $18.7B; the consolidated figure the screener uses is $26.87B. The proposal removes the finance
  arm's capex and keeps its $8.17B of operating cash flow, which would publish $17.57B of free
  cash flow against GM's own $9.5B - **85% too high**, where today's figure is 81% too low. The
  whole lease cycle, from GM's own tags: $15.79B of vehicles bought, $7.80B of rentals collected
  (inside operating cash flow, so the proposal keeps them), $10.10B received when the vehicles are
  sold at lease end (counted by neither construction). **The clean ratio cannot be computed from
  free data** - a segment-level operating cash flow is MD&A prose, not an XBRL fact - so what
  shipped is the honest alternative: every affected stock's FCF-yield row now says, from its own
  filings, how much of its capex is assets bought to lease out and that its own free-cash-flow
  figure is the industrial business only. Section 5 of `research/2026-10-14-capex-definition.md`
  has the measurements and the two things that would reopen it.

- **2026-10-07 - Premium redesign and calculation transparency: finished.** Owner-run, five
  passes in one evening, verified in a real browser at 1440, 375 and 320px and live on the
  public site. Shipped, beyond the first version: search palette (Ctrl/Cmd+K), stock links,
  J/K stepping, Compare with the composite gap decomposed, workings CSV, Add to Holdings from the
  sheet, analytics that filter the table, every metric's calculation one click away with its
  equation checked against the engine for every stock, reasons for unused metrics, a real rank
  history chart with the ordinary-variation band, brand mark / favicon / page title / meaningful
  stat strip / footer, focus trapped in every dialog, AA contrast on every surface, keyboard
  navigation of the rankings, the shell painting before the data lands (LCP 4.8 s -> ~0.3-1.1 s
  at 10 Mbit/s), and fixes found by looking: the sheet lifting its own header 164px, analytics
  cards running 44px off phone screens, the drilldown's nav order, Peers' green/red verdicts,
  a history rank 1-3 off for 284 stocks. Nightly sessions now return to methodology and overall
  improvement (owner direction, same night).

  <details><summary>Archived text of the two finished items, as they stood when closed</summary>

- **2026-10-06 — Show the numbers that go into every score, and make the
  arithmetic provably right. This builds trust; spend real sessions on it, in
  step with the premium item below.** Owner, verbatim: *"I want it to have more
  data integrity, or calculation integrity - we can see how things score, but we
  don't actually see the numbers going into any calculations in the breakout
  details for each company. This will add trust to the screener."*

  **The full plan is `plan/calculation-transparency.md` - read it first.** It holds
  the goal, the design principles, seven stages (T0a, T0b, T1-T6), the tests and
  the sources. The short version: a reader should be able to follow any number from
  the composite, down through category score, metric percentile and raw value, to the
  company's own reported inputs - see every weight and peer count on the way - and
  recompute it in a spreadsheet and get the same answer.

  **Why this is urgent and not just a feature - three defects, measured on the live
  site 2026-10-06 while writing the plan** (reproduce with the two scripts in
  `research/measurements/2026-10-06-*.py`; do not trust these, re-run them):

  1. **The per-metric weights on the drilldown are not the weights used, for 276 of
     502 stocks (55%).** The page prints the generic configured weight; the engine uses
     bank weights, Piotroski-conditional weights and per-stock renormalisation.
     334 of 4,012 stock-category pairs cannot be reproduced from what is on screen.
     JPM's Valuation panel shows three heavily-weighted metrics as N/A, calls P/B
     (the bank's main metric) "Inactive", and prints a score no arithmetic on the
     page produces. Same class of bug as 2026-08-28.
  2. **"Composite = sum of the points" is false for 2 stocks** (FDXF, L): a coverage
     discount is applied after the weighted average and never shown.
  3. **The first sentence of every drilldown is false.** *"Its composite of 73.7 is a
     percentile: it scores above 74% of the universe"* - for the stock ranked **1st**.
     Off by a median of 19.6 points, by more than 10 points for 75% of stocks. The
     composite has been cardinal since Phase 13; the README, the generated methodology
     page's Step 5 and one plan doc still say percentile, while the same methodology
     page's Limitation 8 says the opposite.

  **Order, and why it outranks the design stages:** these are false statements on a
  public site, so **T0a (fix the false sentences, build the claims register) then T0b
  (true weights, reproducibility test, build-time refusal to publish a page whose
  arithmetic does not add up) come before any presentation stage.** Then T1 (lineage
  registry), then T2/T3 (inputs and percentile context in the payload), then T4 built
  *together with* the drilldown redesign (design stage D3 - do not style the old
  category tables twice), then T5 (an independent checker and a gate) and T6
  (per-input provenance).

  **Constraints that do not move:**
  - **No scoring change of any kind.** This is explanation, not methodology; nothing
    here may be justified by the backtest (rule 5) or the thin IC series (rule 4).
  - **The page shows the engine's numbers, never its own re-derivation.** Refactor
    `compute_category_scores` so weight resolution is one function used by scoring and
    export - a second copy in the generator or in JS is how defect 1 happened.
  - **Say what cannot be shown** ("provider-supplied", "computed from 252 daily
    closes") rather than print a formula that does not reproduce. A false equation is
    worse than none.
  - **Payload budget:** <= +150 KB gzipped inline, else per-ticker shards; measure
    after gzip and log it. Decision support, not advice: `BANNED_TERMS` applies to
    every new sentence. Display-only: inputs enter no metric, weight or rank.
  - **Any discrepancy the new checks find is reported, not smoothed over** - in the
    log and `METHODOLOGY_CHANGELOG.md`. Finding them is the point.

  **Progress (update this line each session):**
  - **2026-10-06 - plan written, nothing built yet.** Next: **T0a**.
  - **2026-10-07 - T0a shipped.** Defect 3 is fixed everywhere and can no longer
    come back. The drilldown's first sentence now reads *"Ranks 1st of 502 -
    ahead of 100% of the other 501 stocks. Its composite of 73.8 is a 0-100
    score computed from its 8 category scores and their weights, not a
    percentile"* - the share is `(N-rank)/(N-1)`, exact for all 502. The plan
    named four places saying the composite was a percentile; there were **six**
    (also `FORENSIC_AUDIT_REPORT.md`, and **a test that asserted the false
    claim**, which is why it survived). `claims.py` + 24 tests now make an
    unregistered or unchecked claim fail the build, and the data loop's publish
    gate runs them. Every number, rank and other sentence byte-identical:
    181,446 payload leaves, **502 changed (all this sentence), 0 added, 0
    removed**; **+309 bytes gzipped** against a 150 KB budget. Dashboard and
    methodology page regenerated, so it is live now rather than at 02:00.
    **A fourth defect was found and recorded, not fixed:** "rests on N of 18
    metrics" and the provenance badge's 60/80% colours use a hard-coded
    18-metric list, not the applicable-metric coverage the discount reads (35
    for a bank, 41 otherwise) - 62 stocks read under 80%, 3 were discounted.
    **Next: T0b** (true per-stock metric weights from the engine, the
    reproducibility test, the build-time refusal to publish) - which now also
    owns that fourth defect. `scripts/diff_payload.py`, which the plan assigned
    to T0b, is already built.
  - **2026-10-07 (owner-run session) - T0b, T1, T2, T3, T4, T5 and most of T6 are built and
    shipped, at the owner's request ("make all improvements right now ... then the nightly
    sessions do an even deeper pass").** The page now prints the weights the engine used
    (bank / Piotroski-conditional / rescaled), every metric opens to its formula, the
    figures behind it, whether they rebuild the value, and who it was ranked against; the
    nine Piotroski signals and eight Beneish indices are listed; the composite chain shows
    the coverage discount. **4,010/4,010 category scores and 502/502 composites rebuild from
    the payload; the build refuses to publish otherwise; `scripts/audit_stock.py --all`
    independently reproduces 502/502.** 24 metric equations rebuild at 99.5-100%. Payload
    gzipped is flat (1,268,733 B vs 1,278,885). **The deep pass - read
    `plan/calculation-transparency.md` "Status, 2026-10-07 evening" for the table - is, in
    order:** (1) the "download this stock's workings" CSV; (2) add the missing inputs at
    fetch (analyst per-quarter EPS, the risk-free rate and market return) so the last 12
    metrics can show an equation; (3) **research the four findings the build surfaced**
    (open item 0.9 in `CLAUDE.md`) - negative `operating_leverage` ranking best is the one
    with a measured effect; (4) put `audit_stock.py --sample 25` in the morning brief.
    **(1) shipped in pass 2 the same night** (Download as CSV).
  - **2026-10-07 (late, owner-run pass 3) - the numbers are on every row, not behind a
    click.** Owner, with a screenshot of the Valuation workings: *"there is nothing like
    showing the actual numbers going into any scores ... make sure the nightly sessions know
    that if something is changed with scoring, that it also updates on there, the frontend
    side of things to match the backend."* Each metric now carries a second line, e.g.
    "$4.5B free cash flow ÷ $31.0B enterprise value = 14.4% · 1st of 46 in sector, median
    4.0% · 100.0 x 45%"; a two-line card on a phone. The equations are templates
    (`metric_lineage.EQUATIONS`) the suite **evaluates against every stock's scored value** -
    19 exact ones reproduce 100% - so a formula changed in the engine without the page fails
    the build, and the data loop's publish gate runs the same check. Standing rule:
    CLAUDE.md 0.8c, `prompts/nightly.md` section 3, `DECISIONS.md` 0.8c.
  - **Commit after every stage.**

  **The test:** a student picks any stock, opens it, and can answer *"where did this
  63 come from?"* three levels down without leaving the panel; downloads the workings;
  a spreadsheet agrees with the page to the displayed precision; and the 02:00 data
  loop would refuse to publish a day on which that stopped being true.

- **2026-10-05 — Make the dashboard look and feel premium. Spend a lot of the
  next two weeks on it.** **The surface-by-surface plan is
  `plan/dashboard-redesign-master.md` (18 surfaces, stages D2-D8, measured budgets,
  order of work) - read it first; the brief below is the owner's words and the
  original measurements.** The calculation-transparency item above takes the first
  code session (T0a, T0b) because those are correctness defects; after that the two
  items interleave as the master plan's order table sets out, meeting at the
  drilldown. Owner, verbatim: *"It looks like AI slop. Make it look
  premium and expensive. Right now it does not look good, and it doesn't feel
  good to move around in it either."* Both halves count: how it **looks** and how
  it **feels to use**. This outranks the day's rotation focus on every day it can
  be worked, Mon-Fri, until the test at the bottom is met; the retrospective
  decides when it is done.

  **What was measured on the live site, 2026-10-05** (the documented failure the
  evidence rule asks for; re-measure before and after, do not trust these):

  - **The palette is GitHub's dark theme, unchanged** - `#0d1117` / `#161b22` /
    `#21262d` / `#e6edf3` / `#7d8590` / `#58a6ff`. That is the default look of
    generated dashboards, which is what "AI slop" is picking up on.
  - **Colour is spent everywhere, so it means nothing.** A rainbow gradient bar
    across the header, blue-gradient headers on the Top 5 cards, eight
    saturated bars per card (one hue per category), coloured sector pills,
    green/red/orange/purple text. Premium reads as restraint.
  - **The type is small and mono-heavy.** 12px is the size of ~2,100 text
    elements and 13px ~770; JetBrains Mono sets **~2,560 elements** - nearly
    every label and number - alongside two other families. Three typefaces with
    no clear role each.
  - **The rankings table is the main thing people use and it is the weakest
    surface.** All 502 rows are in the DOM (~9,900 nodes on the page), the table
    scrolls *inside a box inside the page* (nested scrolling is the single
    worst "doesn't feel good" culprit), scores are bare numbers with no visual
    encoding, and the filters are unstyled native `<select>` controls.
  - **The drilldown has no hierarchy.** "Why it ranks here" is the point of the
    tool and renders as ~11 sentences of identical weight in one block.
  - **Mobile is an afterthought.** Three KPI cards in a two-column grid leave an
    orphan, the header card consumes the first screen, and Top 5 is a sideways
    carousel.
  - **Motion is everywhere and unconsidered** - ~660 elements carry transitions.

  **Direction, not a spec** - you are the designer; argue for what you change:

  1. **A real design system first, in the generator, once.** One neutral scale,
     **one** accent, and colour reserved for *meaning* (better/worse, trap flag,
     stale data) rather than decoration. A deliberate type scale (body 14-15px,
     tabular numerals for every figure, at most two families, each with a job),
     a spacing scale, hairline borders, one radius and one elevation system.
     Tokens as CSS variables so every later change is cheap.
  2. **Then the surface people live in: the rankings table.** Sticky header,
     comfortable row height, proper hover / selected / keyboard states, scores
     shown with restrained visual encoding rather than bare digits, designed
     filter controls, **no nested scroll box**, and rendering that does not put
     502 rows' worth of work in front of the first paint.
  3. **Then Top 5 / KPIs / My Holdings, then the drilldown** (give the summary a
     headline and a hierarchy; a side panel or a well-built sheet, not a dimmed
     wall of text), **then mobile as a first-class layout**, then a polish pass.
  4. **Feel is measurable.** Interactions respond in well under 100ms, motion is
     short and ease-out and honours `prefers-reduced-motion`, scrolling is
     smooth, nothing shifts under the cursor. Record DOM node count and a
     scroll/click responsiveness number before and after.
  5. **Look at it every session.** Open the live page in the browser pane at
     desktop *and* 375px, before and after each change, and say in the log what
     you saw. A design change judged only from the CSS is how this got here.
     References worth studying as documented practice, not copying: Linear,
     Stripe Dashboard, Vercel, Mercury, Apple's financial surfaces.

  **Rotation, so the week still works:** Monday's research day is a design note -
  what premium financial UI actually does and why (type, colour, density, data
  tables) - with sources. Wednesday's synthesis checks that the new look still
  *explains* the eight categories and the sector-relative percentiles correctly.
  Thursday builds. Tuesday and Friday are the product and teach days anyway.

  **Constraints that do not move:**
  - **No number, score, rank or sentence changes meaning.** This is presentation.
    Diff the rebuilt payload against the live one and say it is identical.
  - **Decision support, not advice** - nothing that reads as a recommendation,
    and `BANNED_TERMS` stays enforced. Defensibility features stay (rule 7).
  - **15 test modules pin dashboard markup, ids and JS function names**
    (`test_dashboard_surfaces`, `test_holdings_panel`, `test_stock_summary`,
    `test_dashboard_js` with its `node --check`, and more). Prefer CSS and layout
    changes. If a pinned hook has to move, update its test in the same commit and
    say why - never delete an assertion to get green.
  - **Stay a single static page built by `generate_dashboard.py`** - no
    framework migration, no build step, no new runtime dependency. Edit the
    generator, never the generated files (rule 10).
  - **Keep payload weight flat** (it is ~1.2 MB gzipped; measure after gzip) and
    keep working offline from the file once loaded. If you load webfonts, choose
    them deliberately, subset them, and make the fallback stack look right.
  - Update `plan/dashboard-inventory.md` in the same session (rule 9).

  **Progress (update this line each session):**
  - **2026-10-06 - stage 1 shipped:** design-system tokens in the generator, with the
    written argument in `plan/dashboard-design-system.md` - one neutral ramp and one
    accent, Inter with tabular figures, 14px base, rainbow/glow/entrance-animation
    removed, `color-scheme: dark`. Payload byte-identical. **Next: stage D2, the
    rankings table** (nested scroll box, 502-row DOM, native selects, score
    encoding) - after calculation-transparency T0a/T0b. Baseline to beat: 9,928 DOM
    nodes (8,050 of them in the table), 662 transitioned elements.
  - **2026-10-06 (evening) - full plan written:** every surface audited at 1440 and
    375px and listed with what is wrong and the target; stages D2-D8; budgets for DOM
    nodes, click-to-paint, LCP, CLS and contrast. One finding to act on early: the
    public **Refresh Data button is dead for every visitor** (it connects to
    `localhost:7720`).
  - **2026-10-07 (owner-run session) - a first version of D1 through D7 is built and
    shipped, verified in a real browser at 1440 and 375px.** Measured: DOM nodes **9,928 ->
    2,779**, transitioned elements **662 -> 99**, row-click to paint **70 ms**, sort to paint
    **21 ms**, layout shift **0.001**, payload gzipped flat. `tests/test_dashboard_browser.py`
    (25 tests) holds those budgets. **The deep pass is the work listed per stage in
    `plan/dashboard-redesign-master.md` "Status, 2026-10-07 evening"** - in this order: (1) the
    lower drilldown blocks (Rank History, Price Targets, Company Snapshot, Sector Peers) were
    re-chromed, **not redesigned**, and Peers still colours cells red/green against the stock;
    (2) **look at My Holdings populated** at both widths - never re-inspected; (3) the
    sector matrix has no phone treatment; (4) one shared chart module and alt text; (5) one
    "no data" / "not applicable" vocabulary and a formatting module; (6) D8: a full keyboard
    pass, a real focus trap in the sheet, contrast on **every pair in use**, LCP, 320 and 414px,
    and a side-by-side against Linear / Stripe / Vercel / Mercury. **Be critical:** a first
    version built in one sitting is not "perfect" and the owner asked for perfect - find what
    is still not premium and fix it, and say what you saw in the log.
  - **2026-10-07 (late, owner-run pass 2) - owner, verbatim: *"keep improving the
    UI/UX/Frontend please, it looks alot better now! But do as much as you can right now ...
    and for the fine tunups give that work to the nightly sessions to do and test. Be
    creative!"*** Shipped, presentation only (payload byte-identical): a **search palette**
    (Ctrl/Cmd+K, any stock or section from anywhere); **stock links** (`#stock=TICKER`
    opens it; the back gesture closes the sheet); **J / K stepping** through the table's
    current filter and sort; **Compare** (up to four, side by side, with the composite gap
    taken apart into category points - registered as claim `compare.composite_gap`);
    **Add to Holdings** from any drilldown; **Download as CSV** of a stock's whole workings;
    the drilldown **reordered to match its own jump links** (they skipped four sections);
    **Peers lost its green/red verdicts** (it judged "better" on rules the screener does
    not use, e.g. dividend yield) and gained a peer median; trap rates as HTML bars (the
    canvas clipped sector names); sector matrix and trap bars **filter the rankings** on
    click; section previews on collapsed headers; a dismissible first-visit guide;
    shortcuts sheet (`?`); price targets, snapshot and populated Holdings de-boxed; phone
    filter bar, score grid and contribution rows fixed (two lost the cascade). 18 new tests
    in `tests/test_dashboard_navigation.py`; 3 payload tests for the gap claim in
    `test_calculation_reproducibility.py` (in the data loop's publish gate).
    **The nightly fine-tuning list is now `plan/dashboard-redesign-master.md` "Status,
    pass 2"** - it supersedes the six items above where they overlap. Test what shipped
    tonight first (palette, links, J/K, compare, CSV) on the live site at 1440 and 375px.
  - **Commit after every stage** - the 2026-10-06 session was cut off by a usage
    limit with nothing committed (see `NIGHTLY_LOG.md` 2026-10-06).

  **The test:** the owner opens it and it *feels* like a product someone paid for -
  calm, confident, fast, and obvious where to look. If you would be
  embarrassed to screenshot it next to Linear or Stripe, it is not done.

  </details> The session moves them here with the date and a
pointer to what it did, so this file doubles as a record of what you asked for
and what actually happened.

- **2026-08-26 — Remove the model portfolio.** Removed from the dashboard: the
  Model Portfolio section, its payload key, the sector-allocation chart, and
  the S&P sector-weight table that only that chart used. Top 5 now reads the
  ranking directly and shows the same five names. The `portfolio_constructor`
  engine and its Excel sheet were **kept** — see `METHODOLOGY_CHANGELOG.md`
  2026-08-26 (evening) for why, and say the word if you want those gone too.

- **2026-08-26 — Add "about" sections to the stock drilldown.** Each stock's
  detail view now opens with a plain-English description of what the company
  does, plus its specific industry. Sourced from the same Yahoo Finance
  response the screener already downloads, so it costs no extra requests.
  Descriptive only: never scored, never ranked.

---

## Notes on what this file is for

Good items are about **what the tool should do for you** — a surface that
doesn't help, a question you can't answer, something you find yourself checking
elsewhere. Those are things only you can know.

You do not need to file methodology work here. Deciding what a factor is worth,
which metrics overlap, and how the eight categories fit together is what the
research rotation is for, and it is driven by published research and
professional practice rather than by requests. If you *want* a specific factor
researched, though, say so — that is a legitimate item.

- **2026-08-26 — Layout: Top 5 first, two sections collapsed.** "What Changed"
  moved below "Top 5 Stocks"; both it and "Factor Analytics" now start
  collapsed, so the landing view is the Top 5 plus the full table and
  everything else is one click away.

- **2026-08-26 — Full fresh run to load the descriptions.** Ran with a forced
  refetch; 501 of 502 stocks now carry a real business description on the live
  site.

- **2026-09-01 — General smoothness/reliability pass.** Audited scheduled
  tasks, repo hygiene, and stale documentation claims. Fixed a permanent false
  "High severity" alarm in the data-quality log (bank-only metrics scored
  against the wrong population), closed a synthetic-data-fabrication gap in
  `factor_engine.py`'s own entry point, and made stray git branches from
  interrupted runs self-clean instead of silently accumulating.
