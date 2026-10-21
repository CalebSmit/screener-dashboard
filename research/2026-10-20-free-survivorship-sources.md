# How much of backtest v2's 2012-2019 survivorship gap can free sources close?

**2026-10-20, night (owner-run session, delegated).** `plan/research-engine.md` item 4(d), its first
and measurable half: before anyone rebuilds months with departed members, measure how many of the
missing member-months a free, terms-compliant price source can actually supply, by cause, and say
whether that is enough to run the Size test written down in advance. **Nothing was rebuilt and the
research engine was not rerun.** No file in `v2_experiments.RESULT_PATHS` was touched, so no ledger
row is staled. Under CLAUDE.md rule 5 nothing here decides a weight.

**Answer in one paragraph.** Free sources close **37% of the departed member-months** (5,776 of
15,590) with an honest monthly return, and **36.8%** (5,737) with the 13 months of daily history v2
also needs to score them. Coverage of 2012's members would rise from **56.7% to 69.1%**, 2019's from
**76.9% to 87.6%**. But what they close is almost entirely one cause: members that **left the index
and kept trading** (86% of those months covered). **Acquisitions are 51.5% of the departed months and
free data covers 1.2% of them** - one name, FMC Technologies, whose symbol survived its merger.
Bankruptcies: 0%. Deal filings give an exit price for about half the acquired names but never a
monthly return, and the other two free routes (10-K price tables, 13F holdings) give quarterly
figures only. So a free rebuild adds the members that left because they shrank and still omits
those that left because they were bought. That makes item 4(d)'s Size test **runnable in one
direction only**: a Size lead that *stays* at +0.020 or more would still falsify the survivorship
explanation, but a fall below +0.014 would not confirm it, because the members still missing are the
ones whose last months would have helped Size. Section 7 has the exact build plan; its first step is
to pre-register that one-sided reading before any extended month exists.

Code: `research/measurements/2026-10-20-survivorship-free-sources.py` (stages `departures`,
`classify`, `yahoo`, `yahoo_alt`, `deals`, `tenk`, `13f`, `report`); every number below is in
`research/measurements/2026-10-20-survivorship-free-sources.json`. Downloads are cached in the main
checkout's gitignored `data/survivorship/` (SEC submissions, concept files, deal and 10-K documents,
one 13F data set, the Yahoo series). The SEC User-Agent is read at run time and requests are paced at
8 a second, under SEC's 10.

---

## 1. What is missing, exactly

Source: `cache/v2_census_presample.json`, the census `backtest_v2.py` wrote when it built the 96
months of 2012-2019. For every month it names each member it could not score, with a reason.

| | Member-months | Share of the panel |
|---|---|---|
| Members, 2012-01 .. 2019-12 | 48,282 | |
| Built by v2 today | 31,919 | 66.1% |
| `no_current_record` (no raw record today - the company left) | 15,108 | 31.3% |
| `no_price` (today's holder of the symbol is another company) | 434 | 0.9% |
| `symbol_reused` (same, caught by the per-month CIK guard) | 48 | 0.1% |
| **Departed or symbol reused - the survivorship gap** | **15,590** | **32.3%** |
| `no_filings` (in today's index, but no SEC facts for that month) | 773 | 1.6% |

The last row is not survivorship and is reported separately (section 6): those are companies v2
*has* today - Exxon, BlackRock, Disney, Cigna, DuPont, Broadcom, Alphabet, Medtronic and others -
whose older filings sit under a predecessor registrant's CIK. It is a free fix inside v2.

The 15,590 departed member-months belong to **294 symbols**. 247 carry a CIK from the month's own
Wikipedia revision (`ciks_by_month.json`, from 2014-05), 4 more from the flat map; the other 47
predate the revisions' CIK column and were name-matched against SEC's own name list
(`cik-lookup-data.txt`) - accepted only when the match filed a 10-K during the symbol's membership
years and was clearly the closest name. **4 symbols stay unresolved** (JCP, JDSU, KFT, KRFTV;
56 member-months) and are reported as such rather than guessed.

## 2. Why each one left

Classified from each registrant's SEC submissions record (periodic reports, merger proxies
DEFM14A/PREM14A, tender documents SC 14D9 / SC TO-T, going-private SC 13E3, deregistration forms
15/25, and 8-K items 2.01 "completion of acquisition" and 1.03 "bankruptcy"), read around the
company's *last* periodic report rather than its index exit - a member demoted in 2015 and bought in
2023 is, for a price source, an acquisition - plus the removal reason in Wikipedia's
"Historical components of the S&P 500" table that `2026-09-30-exit-reasons.py` already reads.

| Cause | Names | Departed member-months | Share |
|---|---|---|---|
| Acquired / merged / taken private | 152 | **8,034** | **51.5%** |
| Still files today - left the index (demoted, moved) and kept trading | 104 | 6,401 | 41.1% |
| Symbol reused by another company (census) | 12 | 482 | 3.1% |
| Bankrupt / distressed delisting (8-K item 1.03 or the removal reason) | 13 | 438 | 2.8% |
| In today's index under another symbol - not carried by v2 | 8 | 167 | 1.1% |
| Unresolved (no CIK) | 4 | 56 | 0.4% |
| Spun off / restructured | 1 | 12 | 0.1% |

Known misclassifications, stated rather than hand-corrected (the script's rules are what the JSON
records): three utilities bought by other utilities (AGL Resources, Progress Energy, Pepco) count as
"still files" because the acquired subsidiary still files debt reports; GGP counts as bankrupt
because its CIK carries the predecessor's 2009 Chapter 11; and some "still files" names left the
index in 2026 through acquisitions not yet closed (AvalonBay, Electronic Arts, Leggett). None of
these has a free price series either way, so the coverage figures below do not move with them.

**Compared with 2020-2026.** The 2026-09-30 census found acquisitions were 30% of that window's exits
by name and market-cap demotions 62%. Over 2012-2019 acquisitions are the majority of the missing
*months*, because a member that left in 2015 has been missing for every month it was a member, and
because companies demoted in 2012-2019 and bought later (Nordstrom, Foot Locker, Walgreens, U.S.
Steel, Hess, Juniper, Kellanova) are, for a price source, acquisitions too.

## 3. Route (a): Yahoo through yfinance - the only route that gives monthly returns

Daily history 2010-11-01 .. 2020-02-29 for all 294 symbols, then a second pass for any symbol Yahoo
did not serve, under the symbol the **same registrant** trades under today (SEC submissions
`tickers`: GPS -> GAP, LTD/LB -> BBWI, WYN -> TNL, KORS -> CPRI, ...). AAPL and MSFT were requested
in the same call as controls and answered, so an empty reply is evidence of absence, not of a
blocked host.

**The ticker-reuse check.** The 2026-09-30 note found that a series can be present, plausible and
another company's (INFO from 2024). Every series was therefore tested against the registrant's own
filings: the SEC public float (dei `EntityPublicFloat`, at each fiscal Q2 end) divided by its share
count (dei cover-page count within 300 days, else the us-gaap weighted average within 120 days)
must sit between 0.45 and 1.25 of Yahoo's close on that date, put back on the as-traded basis with
**every** split Yahoo records for the symbol to today (without that, GME's 2022 4:1 and Nabors' 2020
1:50 failed a true match). Float excludes affiliates' shares, so controlled companies sit low (Gap
0.60, Campbell 0.63, AutoNation 0.60) - legitimate; a different company's prices land anywhere. Where
the float check cannot run or a controlled company fails it (V.F. has no usable share facts; Coty's
float is 37% of its value because of JAB's stake), a second tier accepts the series only if it is
under the symbol the registrant files under **today** and runs without a gap from 13 months before
the first missing month - a reused symbol cannot do both.

**And one refusal.** For the 12 symbols the census already found reused, the series under the old
symbol is refused outright. The price-level check cannot always see the difference: Yahoo's `CB`
series is ACE's history, and ACE and Chubb Corp traded close enough that it passed (ratio 0.95), as
did `JCI` (Tyco's history) at 1.09. Two similar-priced companies is exactly the case a level check
misses; refusing is the honest response. (`AA` appears to carry Alcoa Inc.'s pre-2016 history
spliced in, ratio 0.97; it is refused too, conservatively.)

| Result per symbol | Symbols |
|---|---|
| No series at all ("possibly delisted") | 176 |
| Series accepted as the same company | 100 |
| Refused - symbol reused | 9 |
| Series, but none of it overlaps a float date (begins after the membership: DOW, FOX, NWSA ...) | 7 |
| Float mismatch | 1 |
| No CIK to check against | 1 |

| Departed member-months with an accepted series | Count | Share of the 15,590 |
|---|---|---|
| Month-end close and next month-end close (a monthly return, adjusted close) | **5,776** | **37.0%** |
| ... and 13 months of daily history to the month-end (v2's momentum and risk inputs) | **5,737** | **36.8%** |

By cause:

| Cause | Departed months | Covered (return + history) | Covered share |
|---|---|---|---|
| Still files - left the index and kept trading | 6,401 | 5,475 | **85.5%** |
| In today's index under another symbol | 167 | 166 | 99.4% |
| Acquired / merged / taken private | 8,034 | 96 | **1.2%** (FMC Technologies only) |
| Symbol reused | 482 | 0 | 0% |
| Bankrupt | 438 | 0 | 0% |
| Unresolved, spun off | 68 | 0 | 0% |

By year (members, built by v2 today, built plus route (a)):

| Year | Member-months | v2 today | With route (a) |
|---|---|---|---|
| 2012 | 6,000 | 56.7% | **69.1%** |
| 2013 | 6,000 | 59.2% | 71.6% |
| 2014 | 6,014 | 60.8% | 73.1% |
| 2015 | 6,035 | 62.7% | 75.3% |
| 2016 | 6,053 | 67.2% | 79.4% |
| 2017 | 6,060 | 71.2% | 82.8% |
| 2018 | 6,060 | 73.9% | 84.8% |
| 2019 | 6,060 | 76.9% | **87.6%** |

**What is left after route (a): 9,853 member-months, and 80.6% of them are acquisitions** (7,938),
then still-filing names Yahoo does not serve (926: AvalonBay, Electronic Arts, Discovery's two
classes now under WBD, whose Yahoo history starts in 2022, Ensco, Mallinckrodt, the three utilities),
reused symbols (482), bankruptcies (438), unresolved (56) and one spin-off (12).

**Why Yahoo has nothing for acquisitions, and why the gap grows.** Yahoo drops a symbol's history
once it is delisted, whenever that happens. Every 2012-2019 member acquired since - in 2016 or in
2025 - has lost its history, so this route covers *less* of 2012 each year that passes. That is
worth knowing about any plan that leans on it: data not cached now may not be there later.
`data/survivorship/yahoo_daily.parquet` and `yahoo_alt.parquet` hold what was served on 2026-10-20.

**Terms.** yfinance is an unofficial client of Yahoo's public endpoints; Yahoo's terms reserve the
data for personal use. This project already runs its live fetch and v2's price cache through it on
that basis, and caches it gitignored, never committed. The rebuild would not change that footing.

## 4. Route (b): the deal price in the target's own merger filings

For every acquired name with a CIK (151), the script read the target's DEFM14A (merger proxy), SC
14D9 (tender recommendation), DEFM14C or 8-K item 2.01 nearest its end, and extracted the cash per
share, any exchange ratio, and the "closing price on the last trading day before announcement" that
proxies state.

| | Names | Share |
|---|---|---|
| Acquired names tried | 151 | |
| No merger document found in the window | 30 | 20% |
| Cash consideration per share extracted | **82** | **54%** |
| Exchange ratio extracted (stock or mixed deals) | 54 | 36% |
| Pre-announcement closing price extracted | 47 | 31% |

**The extraction is not clean enough to use unreviewed.** Read against the deals' published terms,
at least 8 of the 82 cash figures are wrong or superseded: Xilinx and Duke Realty (all-stock deals;
a stray dollar figure), Spectra Energy (all-stock), Cameron ($34 against $14.44 cash), Abiomed (took
the $35 CVR, not the $380 cash), Kraft Foods Group ($72.50 is Heinz's price; Kraft holders received a
$10 special dividend and shares), Kansas City Southern (the terminated CN terms) and Tiffany ($135,
renegotiated to $131.50 after the proxy). Exchange ratios are worse (Pall's "67.85").

**And it never gives a monthly return.** A deal price is the value at exit. Turning it into the exit
month's return needs the previous month-end close, which is exactly what is missing; the
pre-announcement close (31% of names) gives a return from announcement to completion, which spans
several months. Route (b) adds **0** member-months to a monthly IC. It could add one *exit value* per
acquired name - 1 of the ~53 months an acquired name is missing, on average.

## 5. Routes (c) and (d): free prices in other SEC filings - quarterly at best

**(c) 10-K Item 5 price tables** (sample of 20 names, up to four per cause, 41 10-Ks). Quarterly high/low
tables were present in **16 of 19** 10-Ks filed in 2013 and **6 of 7** filed in 2018, and in **0 of
6** filed in 2019 and **0 of 5** in 2020. That break is the SEC's 2018 "Disclosure Update and
Simplification" release, which removed Regulation S-K Item 201(a)'s requirement to disclose high and
low prices (effective 5 November 2018). They are quarterly *ranges*, not closes, so even for 2012-2018
they cannot make a monthly return. **0** member-months.

**(d) SEC Form 13F data sets** (structured, public SEC data, from 2013 Q3). Each institutional manager reports quarter-end
holdings: value and share count per CUSIP, so value / shares is the quarter-end price. Tested on the
2016 Q1 file (positions at 2015-12-31): the median over filers (shares >= 1,000, at least 5 filers)
was matched by issuer name to **118 of 172** departed members of that month (acquired: **60 of 86**;
median 598 filers per security). Against Yahoo's close where both exist, **29 of 39 agree within 1%**;
the other 10 are Yahoo's series being adjusted for corporate actions after 2015 (GameStop's 2022
split at exactly 4.0, Xerox's 2017 reverse split and Conduent spin, ConAgra's 2016 Lamb Weston spin,
BorgWarner's 2023 spin) - and one is the reuse case again (`CB`, 1.135: Yahoo's series is ACE's).
So 13F gives an **as-traded** quarter-end price for acquired members too, and name matching, not the
data, is what limited it to 70%; a CUSIP map would do better. But it is **quarter-end only**: a
3-month return at best, never a monthly one. **0** member-months for the monthly test.

**Not tested, because they need an account the session may not create:** Nasdaq Data Link's frozen
"WIKI" end-of-day prices (community data, described by its users as public domain, frozen in April
2018; whether it carries delisted names was not established), Alpha Vantage and Tiingo free tiers.
Each needs the owner to sign up for a free key; none was assumed to cover anything. Stooq still
answers with a browser challenge and is off-limits. **CRSP and the $199/yr feed are excluded by the
owner's no-paid-data rule.**

| Route | Monthly returns, departed member-months | What else it gives |
|---|---|---|
| (a) Yahoo, identity-checked | **5,776 (37.0%)**; 5,737 with 13 months of history | - |
| (b) Deal price in merger filings | 0 | an exit value for ~54% of acquired names, unreviewed |
| (c) 10-K Item 5 | 0 | quarterly high/low ranges to 2018 |
| (d) 13F data sets | 0 | as-traded quarter-end prices from 2013 Q3, ~70% name-matched |
| **Only the exit value, no monthly return** | | **~152 exit months** (one per acquired name), not usable in a monthly IC |

## 6. Two fixes this found that are not survivorship

Both are inside v2's own data and free; both would be a data-version change (rerun everything,
rule 3), so neither was made tonight.

1. **167 member-months of companies in today's index that v2 drops** because the rename is not
   carried before the revisions' CIK column (2014-05) or the symbol is written differently: BF.B as
   "BFB" in 2012's revision, McGraw-Hill (MHP) -> S&P Global, SAIC -> Leidos, MetroPCS -> T-Mobile,
   ACE -> Chubb, Tyco -> Johnson Controls, Comcast's special class. The CIKs resolved here carry them.
2. **773 `no_filings` member-months** for companies v2 does have today (Exxon 96 months, BlackRock
   96, Disney 88, Cigna 85, DuPont 68, Broadcom 49, Alphabet 63 across two classes, Medtronic 37 ...):
   holding-company reorganizations and redomiciles put the older filings under a predecessor CIK.
   `pit_record` would need the predecessor's facts joined for those months.

## 7. Is it enough to run item 4(d)'s Size test?

The registered prediction: if survivorship drives Size's 2012-2019 lead over the baseline, the lead
falls from **+0.028** to **below +0.014** on an extended universe; the survivorship explanation is
**falsified if it stays at +0.020 or more** (the current ledger has it at +0.027, t 1.83).

**What the free rebuild adds and omits, and which way that leans.** It adds the members that left
because they shrank - small, poor recent returns, the classic survivorship omission that flatters a
small-company screen. It still omits the acquired members, which are disproportionately the
smaller members too (Palepu 1986, *J. Accounting and Economics* 8(1): size lowers the probability of
being a target) and whose announcement month carries the takeover premium (Andrade, Mitchell &
Stafford 2001, *J. Economic Perspectives* 15(2): average target announcement return about +16% over
the three days, more over a longer window). Leaving them out removes small members' best months.
**Both changes push Size's measured lead down.** So:

- **A lead that stays at +0.020 or more on the free extended universe falsifies the survivorship
  explanation** - more strongly than on a complete universe, because the omission leans against it.
- **A lead that falls below +0.014 does not confirm it**, because part or all of the fall could be
  the acquisitions still missing.

*How large is that lean, roughly?* An arithmetic illustration, not a measurement and not computed on
any return data: about 110 acquired members exit within 2012-2019, ~1.15 a month. One stock moving
from the middle to the top of a ~480-stock return ranking while sitting about a quarter of the way
from the middle of the score ranking moves a rank correlation by about 0.003 (rank covariance
120 x 240 / 480 over a rank variance of 480^2/12). So the omitted announcement months could hold
Size's measured IC down by **something like 0.003-0.005** a month - the same size as the gap between
the prediction's two thresholds (0.006). That is why the confirming side is not readable and the
falsifying side is.

**Power.** The two builds share the same 96 months and most of the same stocks, so the quantity to
read is the **month-by-month change** in Size's lead (extended minus current), whose standard error
is far below the 0.040 the design can see for a single comparison. That statistic should be named in
the registration, before the build exists.

**Verdict: runnable, as a one-sided test.** It can falsify the survivorship explanation; it cannot
establish it. Item 4(d) should say so before it runs.

## 8. The build plan for a nightly session

Order matters: the registration must exist, committed and pushed, **before** any extended month is
built (rule 1, and `refuse_known_outcomes` will refuse otherwise).

1. **Register first (one commit, then the rerun it forces).** Add `only-size-extended` to
   `research/experiments/registry.yaml`: the hypothesis and sources (Linnainmaa & Roberts 2018,
   Shumway 1997, this note), the item 4(d) prediction verbatim, **the one-sided reading from section
   7** (falsified at >= +0.020; a fall below +0.014 "not confirming, acquisitions still omitted"),
   and the statistic: the paired month-by-month change in Size-minus-baseline between the
   `presample` and extended builds, over 2012-01..2019-12. The registry is in `RESULT_PATHS`, so the
   same commit stales all 30 ledger rows: budget the ~47-minute rerun with
   `python scripts/rerun_experiments.py --limit 4` in foreground calls, then `report`. If the period
   needs code in `v2_experiments.py` (a data variant for the extended months), put it in this same
   commit so there is one rerun, not two.
2. **Inputs, offline-cached, outside `RESULT_PATHS`.** A tracked builder
   `research/measurements/2026-10-2x-extended-inputs.py` that writes, gitignored:
   - `cache/v2_daily_extended.parquet` - Yahoo daily close and adjusted close from 2010-11-01 for
     exactly the series this note accepted (`route_a_yahoo.per_ticker[*].identity.verdict ==
     "same company"`; 100 symbols), keyed by the **departed** symbol, with `via_symbol` kept. Copy
     from `data/survivorship/` - re-downloading risks the history having gone.
   - `data/sec/pit/facts_extended.parquet` - `pit_record`'s own facts fetch for the departed CIKs
     (all have CIKs). Check per month that `record_as_of` builds a record; count failures.
   - `data/survivorship/static_extended.json` - each departed member's **sector from that month's
     own Wikipedia revision** (the constituents table carries a GICS Sector column back to 2012;
     `wiki_revisions/` already holds some). The engine ranks within sector, so this must be
     point-in-time, not guessed from today's SIC code. Industry and share-class fields that the
     static record carries and the revision does not: state which are missing and how the engine
     treats them.
3. **Code, in `RESULT_PATHS` (so in step 1's commit or followed by another full rerun).**
   `backtest_v2.Inputs` accepts the extra prices, facts and static fields; a build with
   `--tag _presample_extended` admits **only** the member-months listed in this note's JSON (never
   by row count; the INFO lesson), never a census-reused symbol, and records each admitted month in
   the census under `extended`. Tests: admitted set equals the JSON's; no reused symbol admitted; a
   series not overlapping its membership is refused; the `presample` build is byte-identical.
4. **Build** the 96 extended months (`--shard k/4`, `--merge 4`), **then** run the registered
   experiment and report it in this note and `RESULTS.md`, with the one-sided reading. Nothing it
   says moves a weight before 2027-02-11 (rule 5) - and on this design it can only ever remove an
   explanation, not install one.
5. **Separately, and before or after:** the two non-survivorship fixes in section 6 are data fixes
   to v2 itself (rename carry before 2014-05, predecessor-CIK filings). Each is a data-version bump
   and a full rerun; do them together, and not in the middle of step 1-4's sequence.

**What would make the confirming side readable:** monthly prices for acquired members. No free
source found here supplies them. 13F quarter-end prices plus deal values could support a separately
registered *quarterly* version that includes acquired members from 2013 Q3, but with ~25 quarterly
observations it could not tell +0.028 from +0.014 either, so it is not worth building for this
question.

## 9. Sources

- Linnainmaa, J. T. & Roberts, M. R. (2018), "The History of the Cross-Section of Stock Returns",
  *Review of Financial Studies* 31(7), 2606-2649 - testing anomalies in the history before their
  samples (the `presample` period's rule).
- Shumway, T. (1997), "The Delisting Bias in CRSP Data", *Journal of Finance* 52(1), 327-340; and
  Shumway & Warther (1999), *Journal of Finance* 54(6) - omitted delisting returns, and a size effect
  that disappeared once they were corrected.
- Palepu, K. G. (1986), "Predicting Takeover Targets", *Journal of Accounting and Economics* 8(1),
  3-35 - size and target probability.
- Andrade, G., Mitchell, M. & Stafford, E. (2001), "New Evidence and Perspectives on Mergers",
  *Journal of Economic Perspectives* 15(2), 103-120 - target announcement returns.
- S&P Dow Jones Indices, *S&P U.S. Indices Methodology*, "Deletions" (quoted in
  `research/2026-09-30-delisted-price-source-cost.md` section 4): an acquired member is removed at
  the last close or tender expiry.
- SEC, Release 33-10532, "Disclosure Update and Simplification" (2018) - removed Item 201(a)'s
  high/low price table.
- SEC, Form 13F data sets (`sec.gov/data-research/sec-markets-data/form-13f-data-sets`), 2013 Q3
  onward; SEC EDGAR submissions and XBRL `companyconcept` APIs.
