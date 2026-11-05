# "Follow the smart money": hedge-fund 13F holdings and congressional trades

**2026-11-03, owner-run night.** The owner, in chat: *"Another idea to add is to follow the smart
money, like follow the big hedge funds or Wall Street somehow ... we are always taught to not bet
against Wall Street. Or the government, like senators and that type of thing."*

Two questions, read from the papers themselves (two readers; figures marked *(checked)* were
re-found in the PDF text by the second; the rest are the first reader's from the paper or, where
said, from an abstract). The screener's rules frame the answer: free data only; a new signal is
shown as **context** first - beside the score, never in it - and joins the score only after a
research note **and** a measured one-month record (settled row "ctx"); and the page is decision
support, never "do what X did".

**Verdict.** Congress: **do not build it** - the post-2012 evidence is null or negative, the data are
late, ranged and partly scanned, and federal law restricts commercial use of the reports.
Hedge funds: **not a score input; one modest context line is defensible** and is left as an OPEN
proposal with its direction pre-registered (section 3). And the screener already carries the
best-evidenced "follow the informed money" signal: **corporate insiders' open-market purchases**
from their own Form 4 filings (context layer, `insider_activity.py`), filed within two business
days with exact prices - the opposite of both cases below.

## 1. Hedge funds and institutions (13F filings)

What the public can see: managers with $100M+ in 13(f) securities file their **long** US equity
positions **45 days after each quarter end** - no shorts, no written options, and confidential
treatment available (SEC 13F FAQ 2, 25, 41, 43, 52-57). The SEC's free quarterly data sets carry a
CUSIP and no ticker; EDGAR does not label which filers are hedge funds.

- **Copying a fund gets you the fund, minus its fees - not more than the market.** Frank, Poterba,
  Shackelford & Shoven (2004, *J. Law & Economics* 47(2)), 1992-99, semiannual disclosure: before
  expenses the funds beat their copycats by 42 bp over six months; **after** expenses the copycats
  were ahead by 24-25 bp. That is a cheaper route to the same fund return, which is a statement
  about fees.
- **The part you can copy is the less informed part.** Agarwal, Jiang, Tang & Yang (2013, *JF*),
  1999-2007: hedge funds keep about a third of 13F value confidential, and those hidden positions
  beat the disclosed ones by 5.2-7.5% a year (12 months, four-factor and DGTW-adjusted), mostly in
  small, thinly covered stocks.
- **Conviction works in small stocks, not large ones.** Cohen, Polk & Silli (2010), managers' best
  ideas, measured at the true holding date (before anyone could see them): the less-liquid half
  earned **100 bp a month (t = 3.88)**, the more-liquid half **26 bp (t = 1.20, not significant)**
  *(checked, Table 5)*. This screener is the S&P 500.
- **Crowding pays until it does not.** Brown, Howard & Lundblad (*RFS* 2022; working paper read),
  2004-2016: the most hedge-fund-crowded fifth returned 11.3% a year against 8.5% for the least,
  with significantly negative abnormal returns in 2008 - a return with tail risk attached.
- **Changes in holdings: the sign is contested.** Chen, Jegadeesh & Wermers (2000, *JFQA*): among
  large firms, stocks funds bought beat stocks they sold by 1.97% over a year (t = 3.78), counted
  from quarter end - i.e. before the filing is public - with nothing in year two (first reader;
  not re-found in the PDF text). Chen, Hong & Stein (2002): a falling number of mutual-fund holders
  predicted underperformance (abstract). Against: Lehavy & Sloan (2008) find rising investor
  recognition predicts *lower* returns; Dasgupta, Prat & Verardo (2011) find persistently bought
  stocks underperform over 12-30 months (abstracts). Gompers & Metrick (2001): institutions favour
  large, liquid stocks - in the S&P 500 ownership is uniformly high, leaving little to sort on.

**Reading it together.** The positive results are measured before the data are public, sit mostly
in small and illiquid stocks, or rest on holdings the public never sees. After the 45-day lag, in
large companies, there is no robust effect, and the sign of the most natural signal is disputed.
"Not betting against Wall Street" is real advice in one sense: the cheapest way to own what Wall
Street owns, minus its fees, is an index fund - which is the benchmark this screener already
measures itself against.

## 2. Senators and representatives (STOCK Act disclosures)

- **Before 2012, positive but fragile - and uncopyable.** Ziobrowski, Cheng, Boyd & Ziobrowski
  (2004, *JFQA*), Senate 1993-98: Fama-French alpha on purchases **0.323% a month equal-weighted
  and 0.849% trade-weighted** *(checked, Table 2)*, measured from the trade date - when trades were
  disclosed only annually, 5-17 months later. Eggers & Hainmueller (2013) found the result
  significant in at most 3 of 8 specifications, with four senators making nearly half the trades.
- **2004-08, both chambers: Congress lost.** Eggers & Hainmueller (2013, *J. Politics* 75(2)): the
  average congressional portfolio **underperformed a passive index by 2-3% a year** before
  expenses - $100 invested like Congress became **$69 against $80** in an index fund *(checked)*.
- **After the 2012 STOCK Act: no outperformance.** Belmont, Sacerdote, Sehgal & Van Hoek (2022,
  *J. Public Economics* 207), 2012-2020: no aggregate outperformance, none for senators accused of
  informed trading; House purchases underperformed by about 26 bp at six months (first reader,
  from the abstract and the NBER working paper).
- **The one positive recent result is about named leaders.** Wei & Zhou, "Captain Gains on Capitol
  Hill" (NBER WP 34524, November 2025, not peer reviewed): members outperform peers after joining
  congressional leadership. Its sample is reportedly about 20 leaders (press coverage, unverified).
- **The copy-trading ETFs show the tilt, not the information.** Since their 2023-02-06 launch to
  2025-09-30 (Tidal Trust I annual report, filed 2025-12-08): NANC (Democrats' trades) 25.17% a year
  annualised, GOP (Republicans', formerly KRUZ) 15.91%, the S&P 500 21.96%. The report itself
  describes NANC as large-cap growth and tech-heavy and GOP as value-tilted - in a growth-led market,
  the gap is the tilt.

**The data:** a trade must be reported within 30 days of notice and no later than 45 days after it;
amounts are in ranges ($1,001-$15,000 and up); spouses' and children's trades are mixed in; of 515
House trade reports for 2025, about 13% are likely scanned paper (by report-ID prefix; 2 checked) and
would need OCR; no reliable free machine-readable source exists. And **5 U.S.C. 13107(c)** makes it
unlawful to use the reports "for a commercial purpose" other than news publication, with a civil
penalty - a question for a lawyer before anything that trades money (the paper-trading bot included)
ever reads them.

**Why not, on this page specifically:** the panel would show something the post-2012 literature
finds to be noise for the average member; the only positive evidence is about specific named
politicians, so it would read as "follow Senator X" - advice, and partisan advice, on a public tool
built for a college investment club.

## 3. What, if anything, to do

- **Congress: nothing.** Recorded here so a later session does not rebuild the idea from a post.
- **13F:** an open proposal, below.

**Proposal status: DECIDED 2026-11-05 (06:00 session, which did not write this note) - REFUSED as
written, and buildable later on two pre-registered conditions.** The reason is not the literature,
which is this note's own work and stands: it is that logging a **quarterly** signal through the
context layer's **daily** record would overstate its independence by about **3.2x** on the one
count rule 4 and priority 0 exist to protect. Section 4 is the review and carries the conditions.

Costs and limits to weigh: the SEC data sets need a CUSIP-to-ticker map (the SEC's quarterly
  13(f) list carries no ticker; OpenFIGI's free API returns tickers, terms not yet read); the
  reading changes once a quarter and is 45-135 days stale, so it yields about four independent
  observations a year and would take roughly two years to reach the engine's gate of eight; and it
  must never be labelled "smart money" on the page. Yahoo's `heldPercentInstitutions`, already
  fetched (`factor_engine.py`), is a level, not a change, and is not proposed.
- **What the owner's instinct is already served by:** the insider panel - officers and directors
  buying their own company's stock on the open market, which is the "informed money" case with the
  strongest and most timely public evidence in this whole area.

## 4. Review of the proposal (2026-11-05, 06:00 session)

Measured: `research/measurements/2026-11-05-quarterly-signal-accrual.py`, output committed beside
it. The verdicts in sections 1 and 2 were not re-opened; what is reviewed is the one thing left
open - whether to start logging the holder count.

### 4.1 A quarterly signal in a daily record overstates its own independence by 3.2x

`context_eval` already refuses to count overlapping windows: it reads the improvement engine's own
`_effective_observations` over the **run dates** it has logs for, spacing them 30 days apart
(`_non_overlapping(span=30)`), and the gate is **8 effective observations**. That is the right
machinery for a signal that moves every day, which every context signal logged so far does.

A 13F reading does not. It changes on **four filing dates a year** - 45 days after each quarter
end - and is the same cross-section in between, so three consecutive monthly windows rank the
stocks identically against three different forward months. Measured against the data loop's own
weekday cadence:

| window | run dates logged | effective observations the gate would read | distinct signal vintages | overstatement |
|---|---:|---:|---:|---:|
| 1 year | 262 | **13** | **4** | **3.25x** |
| 2 years | 523 | **25** | **8** | **3.12x** |

This is the defect `research/2026-08-10-ic-evidence-independence.md` measured at **2.35x** in the
screener's own IC series, which is why rule 4 says to quote `_n_observations` and never
`_n_raw_observations` - and it would be **worse here, and of a different kind**: there the overlap
is in the forward window, which `_effective_observations` was written to see; here it is in the
**signal**, which it cannot see at all, because nothing in the record says when the reading last
changed. The gate would read 8 effective observations after about **eight months** of logging,
while the signal would have provided **two or three** genuine cross-sections.

**So the counting has to change before the signal is logged, not after.** A context signal carrying
a vintage - the quarter end and the filing date the proposal itself asks to store - must be counted
by vintage, and `context_eval` has no notion of one today. Logging first and fixing the count later
inverts the order rule 4 insists on, and the first thing anyone would do with 8 apparent
observations is read an IC off them.

### 4.2 Counted honestly, the decision point is 2028-08-14

Eight genuine vintages from the next filing deadline: 2026-11-14, 2027-02-14, 2027-05-15,
2027-08-14, 2027-11-14, 2028-02-14, 2028-05-15, **2028-08-14**. That is the earliest date this
signal could be *considered* for the score, about **21 months** after its first reading - and the
note's own section 1 is the reason to be careful about spending that wait: after the 45-day lag, in
large companies, it finds no robust effect and a contested sign. A long wait for a signal whose
prior is "probably nothing" is defensible as evidence-gathering and indefensible as a priority; it
is not refused on the wait, but the wait belongs in the decision.

### 4.3 What this review can and cannot say about "little to sort on"

The note cites Gompers & Metrick (2001) for institutional ownership being uniformly high in the
S&P 500. On this screener's own data (`00_raw_fetch.parquet`, the 2026-11-05 run,
`heldPercentInstitutions`, 502 of 503 covered): median **0.9167**, interquartile range
**0.8208-0.9956**, and **458 of 502 (91.1%)** above 0.70. Two things follow, and only the first
is about the proposal:

- **That field is not a substitute and must not be logged as one.** The note already says it is a
  level rather than a change; it is also **not a fraction** - **123 of 502 (24.5%) exceed 1.00**,
  p90 1.0515 and p99 1.1609 - so it cannot be read as a share of the company and its near-uniformity
  may be partly an artefact of whatever denominator Yahoo uses.
- **It does not establish the point for the proposal's own construct.** Chen, Hong & Stein's signal
  is **breadth** - the *number* of holders - and a count varies where a near-saturated percentage
  cannot. This project has one measurement of it: the 2016 Q1 13F data set, parsed for
  `research/2026-10-20-free-survivorship-sources.md` section 5(d), had a **median of 598 filers
  per security**. So "little to sort on" is supported for concentration and **left open for
  breadth**, which is the honest reading and is weaker than the note's.

### 4.4 The data cost is known, and the identifier is the unsolved part

Also from that 2026-10-20 probe: one quarterly SEC data set is enough, the holder count per CUSIP
is a `groupby`, and **issuer-name matching reached 118 of 172 (~70%)** on the hard cases. So the
build is "one zip, one groupby, one identifier map", and the map is the only unsolved piece - the
note is right that the SEC's 13(f) list carries no ticker, and OpenFIGI's terms are still unread.
**One untested free candidate, recorded so the next session does not start from OpenFIGI:** a
Schedule 13D/13G cover page carries the subject class's CUSIP *and* the filing is indexed under the
subject company's CIK, which would give an exact CUSIP-to-CIK map for widely held US stocks from
EDGAR alone. **Not verified here** - no filing was fetched or parsed for this review - so it is a
lead, not a finding.

### 4.5 The conditions, pre-registered

If a later session builds this, these bind, and the note's direction stands as written
(a rising holder count expected to precede *higher* one-month returns, Chen, Hong & Stein 2002,
against Lehavy & Sloan 2008 and Dasgupta, Prat & Verardo 2011 - either sign informative):

1. **`context_eval` counts by vintage first.** A signal whose log rows carry a vintage is counted
   on distinct vintages, not on spaced run dates, and the gate reads that number. Shipped and
   tested **before** the first holder count is logged.
2. **The identifier route is settled without an unread licence** - either OpenFIGI's terms read and
   found to permit it, or an EDGAR-only map, with its match rate measured against the live universe
   and stated. A signal covering an unmeasured share of the index is not loggable.

**No weight, metric, threshold, published sentence or production line changed by this review.**
