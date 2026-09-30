# What a delisted-price source costs, and whether buying one unblocks backtest v2

**Date:** 2026-09-30
**Question:** `plan/backtest-v2.md` says the next decision on priority 3 is a
procurement one — *cost a price source for delisted tickers, because that
determines whether v2 can remove survivorship bias or only report it.* This
note answers it: what the market charges, what the licences allow, what the
money does **not** buy, and whether buying now is the right next move.
**Status of the answer:** decided. See §7.

**Why this note exists at all.** The 2026-09-24 session measured the
survivorship gap at **4.3% of the universe per year** and found that only ~40%
of exited names have free price history — then correctly declined to wire
`backtest.py` to the new point-in-time universe, because restoring a name's
*symbol* without its *returns* is a different bug, not a fix. It named the
procurement decision as the next step. **Eight of the ten sessions since have
deferred it**, every time for defensible smaller work, and every log entry said
so. The item is 36 days old. It is not blocked on code and never was, which is
exactly why it kept losing to work that was.

---

## 1. What, precisely, would have to be bought

Two scripts produce every number in this section and are checkable:

* `research/measurements/2026-09-30-delisted-price-requirement.py`
* `research/measurements/2026-09-30-exit-reasons.py`

### 1a. How many names, and how many of them are real exits

The 2026-09-24 measurement could only separate renames from exits **for the
oldest month**, because a company that both joined and left mid-window appears
in neither endpoint revision and so has no CIK to match on. Reading the CIK
column from **all 80 revisions** (cached at
`data/universe_history/ticker_ciks.json`, 640 tickers) closes that gap and makes
the decomposition exact for the whole window:

| | Count |
|---|---|
| Point-in-time months, 2020-01-31 .. 2026-08-31 | 80 |
| Distinct names ever in the index over the window | 643 |
| In today's list (`sp500_tickers.json`) | 503 |
| Absent from today's list | 142 |
| — of which ticker renames (same SEC registrant, new symbol) | **19** |
| — of which genuine exits | **123** |
| — of which unresolved (no CIK in any revision) | **0** |

The 19 renames are `ABC→COR, ANTM→ELV, ARNC→HWM, BK→BNY, BLL→BALL, DISCA→WBD,
DISCK→WBD, EQR→VMRK, FB→META, FI→FISV, FLT→CPAY, MMC→MRSH, NLOK→GEN, PEAK→DOC,
PKI→RVTY, RE→EG, SATS→ECHO, UTX→RTX, WLTW→WTW`. Every target resolves to a
ticker in today's list, which is the check that the decomposition is not
silently dropping live companies.

### 1b. The unit a vendor sells is the name-month, not the name

`backtest.py` reads one adjusted monthly close per constituent per month-end
(`_load_or_fetch_prices`, `interval="1mo"`). So the requirement is a panel:

| | Name-months |
|---|---|
| Current names, over their membership months | 35,077 |
| **Exited names, over their membership months** | **4,526** |
| Exited share of the true panel | **11.4%** |

**11.4% is the cleanest statement of this backtest's survivorship bias yet
measured.** The 2026-09-24 figure — 4.3% of the universe per year — is a
turnover rate; this is the share of the panel that is simply not there.

### 1c. "Has a price series" is not the test. This is.

The 2026-09-24 probe tested `len(hist) > 200` on a **30-name sample**. A backtest
does not need 200 rows; it needs a price in each month the name was a
constituent. Running the census over **all 123 exits** and scoring coverage
month by month:

| | Names | % |
|---|---|---|
| Exited names censused (no sampling) | 123 | |
| No downloadable series at all | 51 | 41% |
| Passing the 2026-09-24 test (>200 rows) | 69 | 56% |
| **Covering every membership month** | **66** | **54%** |
| Non-empty series covering *some* months | 0 | 0% |
| Non-empty series covering **no** membership month | 6 | 5% |

| | Name-months |
|---|---|
| Required, exited names | 4,526 |
| **A free source cannot supply** | **1,938 (42.8%)** |
| Residual survivorship after a free-data v2 | **4.89% of the whole panel** |

Two things to take from this.

**First, the 40% figure in `plan/backtest-v2.md` was pessimistic.** The census
puts free coverage at **54% of names and 57.2% of name-months**, not 40%. A
free-data v2 would cut survivorship from 11.4% of the panel to **4.89%** — a
real improvement, and still not zero.

**Second, the row-count test lets three names through that a backtest cannot
use, and the failure mode is worth naming.** Six exits return a *non-empty*
series that contains **zero** of their membership months — the ticker has been
reissued or Yahoo retains only a short recent window:

| Ticker | Series returned | Membership months covered |
|---|---|---|
| INFO | 2024-10-10 .. 2026-09-29 | 0 / 26 |
| LB | 2024-06-28 .. 2026-09-29 | 0 / 19 |
| SBNY | 2024-08-15 .. 2026-09-29 | 0 / 15 |
| AVB | 2026-08-14 .. 2026-08-17 | 0 / 79 |
| EA | 2026-08-04 .. 2026-08-04 | 0 / 79 |
| LEG | 2026-08-26 .. 2026-08-26 | 0 / 23 |

**INFO, LB and SBNY each clear 200 rows.** IHS Markit was absorbed by S&P Global
in March 2022; the series `INFO` now returns starts in October 2024. Silently
joining that to a 2020-2022 backtest would not leave a hole, it would insert
**another company's prices** under a former constituent's symbol — a
survivorship fix that introduces a data-integrity bug. This is the same class of
defect as the 2026-08-26 split-scale finding: a series that is present,
plausible and wrong. Any wiring of `universe_history.py` into `backtest.py` must
assert overlap with the membership window, not row count.

### 1d. Why each name left, and where the free gap actually sits

Classified against the Effective Date / Added / Removed / **Reason** table in
"Historical components of the S&P 500" (409 rows), which cites S&P DJI press
releases. 116 of the 123 exits matched; the 7 unmatched are reported rather
than assumed.

| Removal reason | Names | % of matched | Name-months required | Free source misses |
|---|---|---|---|---|
| Market-cap / representation | 72 | **62%** | 2,755 | 276 (**10%**) |
| Acquired / merged / taken private | 35 | **30%** | 1,339 | 1,339 (**100%**) |
| Other / unparsed | 4 | 3% | 136 | 93 (68%) |
| **Bankruptcy / receivership** | **3** | **3%** | 93 | 93 (100%) |
| Spin-off / restructuring | 2 | 2% | 24 | 0 (0%) |
| (unmatched) | 7 | — | 179 | 137 (77%) |

**This table is the answer to the whole note.** Three readings:

1. **The free gap is not spread across the exits; it *is* the acquisitions.**
   1,339 of the 1,938 missing name-months (**69%**) belong to the 35 acquired
   names, and free data supplies **none** of them — 32 of the 35 return nothing
   at all. Meanwhile market-cap demotions, the 62% majority, are **90% covered
   for free**, because a demoted company keeps trading. The 2026-09-24 session
   said "the split is not random, and it is the bad half"; the split is
   **100% versus 10%**.
2. **Performance delistings — the exact condition Shumway's −30%/−55% applies
   to — are 3 of 116 (2.6%)**, and all three are the March-May 2023 bank
   failures (SIVB, SBNY, FRC). §3 and §4 turn on this number.
3. **So a paid source is both necessary and sufficient here.** Necessary,
   because free data gives 0% of the acquisition months. Sufficient, because
   §4 shows the index exits an acquisition at the last traded close — which is
   precisely what every vendor in §2 sells.

*Caveat, stated because the table invites over-reading:* the reason recorded is
the reason for **index removal**, not the ticker's eventual fate. Nine of the 72
market-cap demotions have no series at all, which means those companies later
delisted for some other reason. The load-bearing claim is only about the state
**at the moment of removal**, which is the moment the backtest's panel ends for
that name — and that is what the table measures.

---

## 2. What the market charges

All figures accessed **2026-09-30**. Prices are USD and are the vendor's own
published list price for the plan that covers this requirement — not the
flagship tier, which is what makes the total so small.

**The data volume is a one-time historical extract, not a feed** — 123 tickers ×
their membership months, 1,938 month-ends of adjusted monthly closes covering
2020-01 to 2026-08. Every vendor below bills monthly with a one-month minimum,
so the *download* costs one month of the cheapest adequate plan.

**But the licence, not the download, sets the price.** Sharadar's terms require
deleting the data within 30 days of the subscription ending (§5). A backtest
harness re-reads its inputs every time it runs, so a $19 pull-and-cancel would
leave nothing behind that v2 could legitimately use. **The honest figure is
therefore $199/yr recurring, for as long as v2 needs the panel** — not $19 once.
Both numbers are small; the recurring one is a standing commitment and a
subscription decision, which is why §7 leaves it to the owner.

| Source | Price for what is needed | Delisted US equities | Carries a delisting return? | Licence shape |
|---|---|---|---|---|
| **yfinance** (status quo) | $0 | partial — measured in §1 | no | no terms; unofficial |
| **Stooq** | $0 | **untested** — see below | no | — |
| **Sharadar Prices**, 10-year history | **$19/mo**, $199/yr | 21,000 active + delisted tickers, from Dec 1997; marketed "99% survivorship-bias-free" | no | personal-use only; **explicitly permits publishing derived results** (§5) |
| **EODHD "Historian"** | **$19.99/mo**, $199/yr | yes; depth by delisting year — pre-2018 delistings are EOD-only | no | non-professional licence forbids "displaying" the data; commercial pricing quote-only |
| **Massive** (ex-Polygon) Stocks Developer | $79/mo (10 yrs history) | yes, inactive tickers flagged | no | — |
| **Norgate Data** Platinum | $346.50/6 mo, **$630/yr** | yes, delisted from 1990, **plus** historical index constituents | no | — (terms page 404s) |
| **Alpha Vantage** premium | from $49.99/mo | `LISTING_STATUS` enumerates delisted symbols; price coverage for them undocumented | no | — |
| **CRSP** via WRDS | no list price; institutional annual contract, sales contact only | yes — the reference dataset | **yes** — `dlret` / `dlstcd` | academic; publication contemplated |

**Stooq is recorded as untested, not as unavailable.** Its CSV endpoint
(`stooq.com/q/d/l/?s=<t>.us&i=d`) returned a JavaScript browser-verification
page for all 20 delisted tickers probed — **and for the AAPL and MSFT
controls**, byte-identical 795-byte challenges. That is a block on this host,
not an absence of data, and the difference matters: the 2026-09-24 session
established yfinance's gaps were genuine absence by exactly this kind of
control, and the same discipline forbids claiming anything about Stooq's
coverage here.

**Norgate's higher price buys something already free.** Platinum is the
cheapest tier carrying delisted securities, and it bundles historical index
constituents — which `universe_history.py` already reconstructs at $0 from
Wikipedia revisions, with an 80-month committed cache. Paying $630 for the half
we have plus the half we need is worse value than $19 for the half we need.

**So the cost is $19 to download and $199/yr to keep.** That is the answer to
the question the plan asked, and it is small enough that **cost was never the
blocker.** Two other things are, and §5 and §7 are what.

---

## 3. What the money does not buy: the delisting return

Every source in §2 except CRSP sells the same thing — prices up to a ticker's
last trade. None of them sells the **delisting return**: what a shareholder
actually received when the security stopped existing.

- **Shumway, Tyler (1997), "The Delisting Bias in CRSP Data", *Journal of
  Finance* 52(1), 327–340.** Verbatim: *"delists for bankruptcy and other
  negative reasons are generally surprises and … correct delisting returns are
  not available for most of the stocks that have been delisted for negative
  reasons since 1962. Using over-the-counter price data, the author shows that
  the omitted delisting returns are large."* The convention the literature took
  from it is to substitute **−30%** for missing performance-related delisting
  returns on NYSE/AMEX.

- **Shumway, Tyler & Warther, Vincent A. (1999), "The Delisting Bias in CRSP's
  Nasdaq Data and Its Implications for the Size Effect", *Journal of Finance*
  54(6), 2361–2379.** Verbatim: *"We estimate that using a corrected return of
  −55 percent for missing performance-related delisting returns corrects the
  bias. We revisit previous work which finds a size effect among Nasdaq stocks.
  After correcting for the delisting bias, there is no evidence that there ever
  was a size effect on Nasdaq."*

That second result is the reason to take this seriously rather than treat it as
a rounding error: **a delisting-return correction did not shade a published
factor finding, it erased one.** A harness that gets this wrong does not produce
a slightly optimistic number; it can produce a factor that is not there. And
`plan/backtest-v2.md` §"Why this matters more than any feature" is explicit that
a biased backtest steers the self-improvement loop toward whatever the bias
favours.

**Both effect sizes are conditional on *performance-related* delistings**, and
that condition is doing all the work. §4 is why it mostly does not hold here.

---

## 4. Why the S&P 500 is the easy case, and how much of it is

Shumway's universe is all of CRSP, where a failing company drifts to the pink
sheets and its final OTC prices are simply not in the database. The S&P 500 is
not that universe, and the difference is written into the index's own rules.

**S&P Dow Jones Indices, *S&P U.S. Indices Methodology*, "Deletions"** (verbatim
from the document):

> A company is deleted from the index if it is involved in a merger,
> acquisition, or significant restructuring such that it no longer meets the
> eligibility criteria: A company delisted as a result of a merger, acquisition
> or other corporate action is **removed at a time announced by S&P Dow Jones
> Indices, normally at the close of the last day of trading or expiration of a
> tender offer.** … **If a stock is moved to the pink sheets or the bulletin
> board, the stock is removed.**

and, under "Other Adjustments":

> In cases where there is **no achievable market price** for a stock being
> deleted, it can be removed at a **zero or minimal price** at the Index
> Committee's discretion.

Three consequences, and they are the crux of this note:

1. **For an acquisition, the last traded close is not an approximation of the
   terminal value — it is the value the index itself exited at.** A vendor
   selling prices through the last day of trading therefore supplies exactly
   what a faithful point-in-time S&P 500 backtest needs. There is no missing
   delisting return to estimate.
2. **The pink-sheet drift that creates Shumway's bias is ruled out by
   construction.** The index removes a constituent *when* it moves to the pink
   sheets, so the untracked OTC decline he priced at −30%/−55% happens after the
   name has already left the panel this backtest models.
3. **Where there is genuinely no price, the index's own convention is zero.**
   That is a documented practitioner rule this project can adopt and cite,
   rather than an assumption it has to defend.

**How many of this window's exits are the easy case is measured, not assumed:
113 of 116, or 97.4%.** Performance delistings are three names — SIVB, SBNY and
FRC, all FDIC receiverships in March-May 2023 — and for those the index's own
zero-or-minimal-price convention applies, so even they need no vendor. The
condition Shumway's effect sizes depend on is met by **2.6%** of this universe's
exits. See §1d.

That is the whole reason a $199/yr feed is adequate where the literature would
suggest only CRSP's `dlret` would do. It is also why the free gap and the
solution line up so neatly: the 1,339 name-months free data cannot supply are
**exactly** the acquisition months, and the acquisition months are **exactly**
the ones the index exits at a price a vendor sells.

**Where academia and practice disagree, and why.** Shumway says "estimate the
missing return, it is large and negative"; S&P DJI says "remove at the last
achievable price, or at zero if there isn't one". They are not in conflict —
they are answering for different universes. Shumway is correcting a *database*
that kept a security after the index would have dropped it. For a screener whose
universe is *defined* as S&P 500 membership, the index's rule is the right one,
because the backtest's job is to model what a constituent-following investor
experienced. This is a case where following documented practice over the
published paper is correct, and it is worth writing down because the reverse is
usually true.

---

## 5. The licence, which is the actual constraint

Two of the cheap vendors publish their terms. They are not the same, and the
difference decides which one this project could use.

**Sharadar** (`sharadar.com/terms`, accessed 2026-09-30) is the only source
surveyed whose licence addresses this exact use:

- *"This License is granted solely to natural persons for personal use"* —
  professional, commercial, institutional and organizational purposes are
  excluded.
- **Publishing derived results is permitted, and needs no attribution:**
  *"Attribution is not required for research outputs, backtest results, models,
  summary statistics, trade logs, and similar derived works that do not display
  the Services Data itself."*
- **Making the data available to others is not:** *"you may not transfer or make
  available access to either the Services or the Service Data to others"*; no
  *"publish, disseminate, re-distribute or share"*.
- **After cancellation the data must go within 30 days**, while *"research
  outputs, backtest results, models, summary statistics … that do not contain
  and cannot reproduce the Services Data"* may be kept.

**EODHD** (`eodhd.com/financial-apis/terms-conditions`, accessed 2026-09-30) is
narrower where it matters: a Non-Professional User must not engage in *"selling,
reselling, retransmitting, redistributing, **displaying**, or granting access to
the Information or Services"*, and the terms say nothing about derived
analytics — their own text directs the ambiguous case to sales.

So the licence answer is:

> **A personal Sharadar subscription would permit backtest v2 to run and its
> results to be published. It would not permit the price cache to be
> committed.**

That second half is a real cost to this project specifically, and it is not a
legal quibble. The mandate's standard is *"evidence must be written down where
someone else can check it."* Every other input to this system honours that:
`data/universe_history/sp500_membership.json` is committed, the improvement
snapshots are committed, `research/measurements/` exists precisely so quoted
numbers can be re-run. A **delisted price cache is the first input this project
would be contractually unable to publish**, which means the first backtest
number nobody outside can reproduce — in a repository whose entire claim is
that its numbers are checkable.

That is survivable with an explicit, documented exception: gitignore the cache,
and say plainly in the changelog which figures rest on an unpublishable input. It
is **not** survivable silently, and it is a real cost to weigh against 11.4% of
the panel — not an obstacle to wave away.

**One tempting workaround does not obviously work.** The licence lets you keep
derived works *"that do not contain and cannot reproduce the Services Data"*, so
committing a derived monthly **return** series instead of prices looks like a way
to keep the evidence checkable. But a monthly return series plus any single
observed price reconstructs the whole price path, so whether it "cannot
reproduce the Services Data" is genuinely doubtful. Do not assume it is
permitted; that is a question for the vendor, and it is the second half of the
one question §7 hands to the owner.

---

## 6. Where the evidence contradicts what we currently do

- **`backtest.py` still projects today's 503 names across the whole window**
  (`_load_or_fetch_prices`, monthly `auto_adjust=True` closes from
  `BACKTEST_START = "2020-01-01"`). That is **11.4% of the true panel absent**
  (§1b), unchanged. This note does not fix it and does not wire anything in —
  see rule 5 and §7.
- **The plan's "only 40% of exited names have downloadable prices" is
  pessimistic and should be corrected to 54% of names / 57.2% of name-months**
  (§1c). It came from a 30-name sample of one month's exits; this is a census of
  all 123.
- **`plan/backtest-v2.md` asks for a price source and does not warn about the
  ticker-reuse failure mode** (§1c). Three exits clear a 200-row availability
  test while covering none of their membership months. Wiring on row count would
  insert a different company's prices under a former constituent's symbol.
- **Nothing in the repo records the delisting-return question at all.**
  `plan/backtest-v2.md` treats "point-in-time universe" as solved once
  membership and prices exist. §3 and §4 are the missing third component, and
  §4 is the reason it turns out to be cheap here.
- **The plan's fallback is better-founded than the plan thought.** It offers
  *"measure and report the survivorship premium rather than pretend it's zero"*
  as a consolation. Given §4, a point-in-time S&P 500 backtest using
  last-traded-price exits and the index's own zero-price convention is not a
  consolation — it is close to the right answer.

## 7. Recommendation

**The procurement question is answered: Sharadar Prices (10-year history), $19 to
download and $199/yr to keep, and its licence expressly permits publishing
backtest results derived from it.** Cost is not the blocker and should never be
cited as one again.

**The case for buying is stronger than the plan assumed**, and it should be
stated at full strength before the recommendation to wait, because the two are
easily confused:

- Free data supplies **0%** of the 1,339 acquisition name-months and **90%** of
  the demotion months (§1d). A vendor is the only way to close the gap, and it
  closes essentially all of it.
- What the vendor sells is **exactly the right number**: the index exits an
  acquisition at the last traded close (§4), and the three bankruptcies take the
  index's own zero-price convention. There is no delisting return left to
  estimate, which is what makes a $199/yr feed adequate where §3 implies only
  CRSP would be.
- A free-data-only v2 leaves **4.89%** of the panel missing. A paid v2 leaves
  approximately none.

**Nonetheless: do not buy it yet.** Buy it when the look-ahead half has been
sized. The reasons are the plan's own, and they are about sequencing, not value:

1. **`plan/backtest-v2.md` is explicit that a half-fixed backtest is worse than
   an honestly-broken one:** *"A v2 that fixes survivorship but not look-ahead
   is still not decision-grade, and shipping it early invites exactly the false
   confidence the bench period exists to prevent."* Survivorship is sized at
   4.3%/yr. **Look-ahead has never been sized at all** — the plan's own step 1
   says "quantify the damage first" and only half of step 1 is done.
2. **Sizing look-ahead costs nothing and needs no vendor.** It is a measurement
   over data already in the repo: how much of a score at month *t* was knowable
   at month *t*, given filing lags. It is the same shape as the 2026-09-24
   survivorship measurement, which is what makes it the obvious next session.
3. **The data cannot be kept after the subscription lapses** (§5), so the clock
   starts when you subscribe, not when the harness is ready. Subscribing before
   a point-in-time fundamentals path exists pays for a panel nothing consumes.
4. **Rule 5 benches the backtest until 2027-02-11 regardless.** Nothing is lost
   by spending the next unit of effort on a free measurement.

**Two questions need the owner, and neither is the price.**

1. **Whether a personal-use licence covers this.** Sharadar's grant is to
   *"natural persons for personal use"* and excludes organizational purposes,
   while publishing derived backtest results is expressly allowed. My reading is
   that it is covered — the subscriber is a natural person doing his own
   research, and what reaches the public site is a derived work the licence names
   — but the site's stated second audience is college investment clubs, so the
   call is his.
2. **Whether a derived return series may be committed** (§5). If not, the first
   figures in this project's history rest on an input nobody outside can check,
   and that is a decision about the project's own evidence standard rather than
   about data.

Both are one email to a vendor's sales desk. **$199/yr is not worth deliberating
over; these two are**, and they are the reason this note stops at a
recommendation rather than a purchase.

## 8. What would change my mind

- **A free source that carries the missing names with publishable terms.** Stooq
  is the live candidate and is untested (§2) — if its CSV endpoint is reachable
  from another host, probe it against the exact name list in §1 before anyone
  pays for anything.
- **Look-ahead turning out to be unfixable.** If point-in-time fundamentals
  cannot be built from EDGAR at acceptable effort, then v2 cannot be
  decision-grade whatever the universe looks like, and the honest destination is
  the plan's fallback — report the premium, buy nothing.
- **A materially different reasons mix than §1d measures.** If performance
  delistings were a large share of exits rather than 2.6% of them, §4 collapses
  and only CRSP's `dlret` would do — which has no list price at all, is sold on
  institutional annual contracts, and would probably end the project's backtest
  ambitions rather than fund them.
- **The reasons table turning out to be unreliable.** §1d rests on a Wikipedia
  table (409 rows, 116 of 123 exits matched) whose rows cite S&P DJI press
  releases but which is a contemporaneous record, not the index. One row is
  already visibly wrong — `MXIM`'s reason reads *"delisted from NASDAQ"* dated
  **2007**, fourteen years before Analog Devices acquired it. If the mix is
  audited against S&P DJI's own announcements and the bankruptcy share is
  materially higher than 3 names, §4 needs redoing.

## 9. Design section — how this fits the rest of the screener

*(This note is a Wednesday, so the synthesis is here rather than in a later
note. The rotation's Monday was lost to a ship-gate repair on 2026-09-28, so
there was no new research note to synthesise; the swap is recorded in
`NIGHTLY_LOG.md`.)*

**What this changes about the screener as a whole: nothing, today — and that is
the finding.** No weight, metric, threshold, percentile or published score moves.
The backtest is benched until 2027-02-11 and is not wired to the point-in-time
universe. What changes is the *shape* of priority 3: it stops being a
procurement question and becomes a measurement question, which a session can act
on without anyone's permission.

**The overlap worth naming.** This project has now hit the same wall three times
from different directions: the evidence base is thin because independent
observations accrue slowly (rule 4, ~1 effective `1m` observation a month, 4
against a gate of 8), and the backtest that could substitute for waiting is
biased. Buying delisted prices looks like a way to short-circuit that. It is
not — it fixes one of two biases in a harness that decides nothing for another
four months. **The binding constraint on this screener's methodology is still
time, and no purchase shortens it.** Anything sold as an accelerant here should
be read against that.

**The sequencing this implies for priority 3**, replacing "cost a price source"
as the next step:

1. **Size look-ahead.** Free, unblocked, and the missing half of the plan's own
   step 1. Method: for each rebalance month, compare a score built from data
   published by that date against the score `backtest.py` actually uses (a
   single Phase-1 snapshot held constant). The gap is the look-ahead premium,
   expressible as a share of the panel like §1b's 11.4%, so the two biases become
   comparable and the plan's decision rule applies to both.
2. **Then decide the fundamentals source** (EDGAR XBRL is named in the plan; the
   owner has a separate project doing primary-source pulls).
3. **Then buy prices**, if steps 1–2 say v2 can be decision-grade. §7's answer
   is on the shelf and will not go stale — it is $199/yr and two licence
   questions.
4. **Adopt the index's own conventions when wiring it**, not CRSP's: exit an
   acquisition at the last traded close, and a no-price deletion at zero (§4),
   citing the S&P DJI methodology. This is the cheap correctness win in this
   note and it costs nothing to apply.

**The generalisable lesson, and it is the fourth instance.** §1c's three names —
INFO, LB, SBNY — return a series that is present, plausible and belongs to
something else. This project has now met that shape four times: the 2026-08-13
stale-cache scoring, the 2026-08-26 split-scale series that `auto_adjust=False`
returned byte-identically, the 2026-09-17 finding that a metric *count* hides one
input dropping out as another returns, and now a row count that certifies a
ticker whose history does not touch the window it is needed for. The standing
rule this earns: **an availability check must assert coverage of the span the
caller will read, not the volume of what came back.** `research/README.md`
records "count independent observations, not rows" for the same reason; this is
its data-integrity twin.

**What it implies for the other seven categories: nothing directly, and one
thing indirectly.** A survivorship-clean panel would change *which* categories
look good, and §4 of `plan/backtest-v2.md` already warns that v1's bias most
flatters "value and distress-adjacent tilts, which is precisely where this
screener's Valuation weighting lives." That remains an untested warning. It is
worth restating that **no weight in `config.yaml` has ever been set from a
backtest number** — checked again today, as the 2026-09-24 session checked it:
no `METHODOLOGY_CHANGELOG.md` entry cites one under **Evidence**. The 2026-08-11
bench rule landed before any could. So there is no accumulated damage to undo
here, only an inability to confirm, which is the state rule 4 describes and
accepts.

## Sources

- Shumway, T. (1997). "The Delisting Bias in CRSP Data." *Journal of Finance*
  52(1), 327–340. Abstract via RePEc:
  https://ideas.repec.org/a/bla/jfinan/v52y1997i1p327-40.html
- Shumway, T. & Warther, V. A. (1999). "The Delisting Bias in CRSP's Nasdaq Data
  and Its Implications for the Size Effect." *Journal of Finance* 54(6),
  2361–2379. Abstract via RePEc:
  https://ideas.repec.org/a/bla/jfinan/v54y1999i6p2361-2379.html
- S&P Dow Jones Indices, *S&P U.S. Indices Methodology*, "Index Maintenance →
  Deletions" and "Other Adjustments". Quoted from the publicly mirrored PDF at
  https://www.betashares.com.au/wp-content/uploads/2016/10/methodology-sp-us-indices.pdf
  (spglobal.com's copy returns HTTP 403 to non-browser clients; the current
  revision is linked from
  https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf).
  Accessed 2026-09-30. The deletion rules quoted are unchanged in substance
  across the revisions surveyed, but this is the mirrored revision, not the
  current one, and should be re-checked before being quoted in public docs.
- Sharadar pricing and licence: https://sharadar.com/subscribe ,
  https://sharadar.com/terms , https://sharadar.com/prices — accessed 2026-09-30
- EODHD pricing and terms: https://eodhd.com/pricing ,
  https://eodhd.com/financial-apis/terms-conditions ,
  https://eodhd.com/financial-apis/delisted-stock-companies-data-2 — accessed
  2026-09-30
- Norgate Data: https://norgatedata.com/stockmarketpackages.php — accessed
  2026-09-30
- Massive (formerly Polygon.io): https://massive.com/pricing — accessed
  2026-09-30
- Alpha Vantage: https://www.alphavantage.co/documentation/ and third-party
  pricing summaries — accessed 2026-09-30. Its delisted price coverage is
  **undocumented**, which is why it is not a candidate.
- CRSP / WRDS: no public list price; institutional annual contracts only
  (https://wrds-www.wharton.upenn.edu/) — accessed 2026-09-30
- S&P 500 removal reasons: "Historical components of the S&P 500", Wikipedia —
  Effective Date / Added / Removed / Reason table, cited to S&P DJI press
  releases. Parsed by `2026-09-30-exit-reasons.py`. Contemporaneous record, not
  the index; treated with the same caveat `universe_history.py` carries.
