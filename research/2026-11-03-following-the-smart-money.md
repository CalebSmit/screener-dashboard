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

**Proposal status: OPEN** - decide whether to *log* (not show, not score) one line per stock - the number of 13F filers holding it and the change on the prior quarter, dated by quarter end and filing date - as a context-log candidate, with the direction pre-registered here before any data is pulled: **a rising holder count is expected to precede *higher* one-month returns** (Chen, Hong & Stein 2002), against which Lehavy & Sloan (2008) and Dasgupta, Prat & Verardo (2011) predict the opposite, so either sign is informative and neither is assumed.

Costs and limits to weigh: the SEC data sets need a CUSIP-to-ticker map (the SEC's quarterly
  13(f) list carries no ticker; OpenFIGI's free API returns tickers, terms not yet read); the
  reading changes once a quarter and is 45-135 days stale, so it yields about four independent
  observations a year and would take roughly two years to reach the engine's gate of eight; and it
  must never be labelled "smart money" on the page. Yahoo's `heldPercentInstitutions`, already
  fetched (`factor_engine.py`), is a level, not a change, and is not proposed.
- **What the owner's instinct is already served by:** the insider panel - officers and directors
  buying their own company's stock on the open market, which is the "informed money" case with the
  strongest and most timely public evidence in this whole area.
