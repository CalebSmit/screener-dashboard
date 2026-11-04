# A ten-repo "quant stack" against this screener: what each would add

**2026-11-03, owner-run evening.** The owner forwarded a post presenting ten open-source projects
as a hedge-fund stack ("329k+ stars of code that hedge funds would rather you never put together"):
OpenBB, TradingAgents, qlib, RD-Agent, gs-quant, nautilus_trader, Chronos, QuantLib, skfolio and
vectorbt, wired into one pipeline - pull five years of S&P 500 data, forecast every stock 20 days
ahead, mine factors automatically, let an agent team approve trades, backtest 10,000 variants and
"keep the ones that survive fees", then paper trade for two weeks.

Each project was read from primary sources (READMEs, licences, docs, the papers behind them) by two
independent readers; nothing was installed or run. The question asked of each was narrow: **does a
specific part of it make this screener - a monthly, long-only, large-company, explainable ranking
built on free data - better, and what is the strongest reason not?** Star counts are GitHub's on the
day. Companion notes: `research/2026-11-03-kronos-candlestick-foundation-model.md` (Kronos, and
Chronos's figures in that paper) and `research/2026-10-27-tradingagents-ideas.md`.

## Verdicts

| Project | What it is | Verdict for this screener |
|---|---|---|
| **QuantLib** (7.7k, BSD) | Pricing library; American-option implied volatility by finite differences with a discrete dividend schedule | **The one real fit - measure first** (section 1) |
| **skfolio** (2.6k, BSD-3; Nicolini, Manzi & Delatte 2025, arXiv 2507.04176) | Portfolio optimisation: mean-CVaR/CDaR, risk parity, HRP/HERC/NCO, shrinkage, purged CV | **No on the page; at most a Factor Lab teaching replication** (section 2) |
| **OpenBB** (75.1k, Apache-2.0 per its LICENSE today) | A wrapper over many data providers | **No as a dependency.** One free source we lack - Cboe's delayed option chains - could be called directly as a cross-check on Yahoo's quotes; Cboe's terms for republishing are **not yet checked** |
| **qlib** (50.0k, MIT) | Microsoft's quant platform; Alpha158/Alpha360 factor sets; a point-in-time (PIT) store | **No.** Its PIT collector is Chinese A-shares from Baostock and *imputes* 45/90-day publication lags; `pit_record.py` already uses SEC's real per-fact `filed` dates, which is strictly better. Its S&P 500 history parses today's Wikipedia page; `universe_history.py` reads the revision history. Alpha158/360 are 100% short-horizon price and volume |
| **RD-Agent** (15.2k, MIT; Li et al., *R&D-Agent-Quant*, arXiv 2505.15155, NeurIPS 2025) | An LLM loop that proposes, codes and backtests factors | **No - it is the practice our research engine exists to prevent** (section 3) |
| **vectorbt** (9.7k; Apache-2.0 **plus Commons Clause** - not OSI open source; new features go to a paid PRO) | Vectorised backtesting | **No.** "Backtest 10,000 versions and keep the survivors" is backtest overfitting (section 3), and our controls already exist |
| **Chronos** (6.2k, Apache-2.0) | Amazon's general time-series forecaster | **No.** No finance evaluation in its README; on US stocks it ranks below a linear model in the Kronos paper (XNAS RankIC 0.0461 vs DLinear 0.0518, Tables 16-17); a stock-return study finds gains over a random walk "small and sparse" (Noguer i Alonso & Franklin, arXiv 2606.27100, abstract) |
| **TradingAgents** (120.4k, Apache-2.0; arXiv 2412.20138) | LLM agents debate and trade | **No.** Its evidence is 3 stocks for ~60 trading days in 2024 (Table 1), Sharpe 5.6-8.2 that the authors call beyond their "expected empirical range", no costs or variance stated |
| **gs-quant** (13.9k, Apache-2.0) | Goldman Sachs' toolkit | **No.** Pricing, risk and data need Marquee credentials "available to institutional clients"; the free `timeseries` module is rolling-window arithmetic pandas already does |
| **nautilus_trader** (31.2k, LGPL-3.0) | Event-driven execution engine | **No** - an execution engine for a screener that never trades; at most relevant to the separate paper-trading bot |

The stack's design is the wrong shape for this tool even where a part is good: it optimises for
*finding something that backtests well*, and this screener's credibility rests on *not doing that* -
pre-registered experiments, a held-out period, Harvey-Liu-Zhu's t >= 3 and a Bonferroni count
(`v2_experiments.py`), and the probability of backtest overfitting by CSCV (`overfitting.py`).

## 1. QuantLib: our implied volatilities could stop depending on an undocumented field

`context_signals.options_context` takes **Yahoo's `impliedVolatility` field as given** for the ATM
implied volatility and for both legs of the put skew. Its method - model, rate, dividends, which
price it inverts - is **documented nowhere** the reader could find. (The expected move does not
depend on it: it is built from bid/ask mids, and since tonight its divisor is the quote response's
own price - changelog 2026-11-03 evening.)

How the providers whose numbers the literature uses do it:

- **Cboe** (its *American and European-Style Theoretical Options Calculation Methodology*): NBBO
  quotes, a Cox-Ross-Rubinstein binomial tree, discrete dividends projected two fiscal years ahead,
  and the **Treasury constant-maturity curve, cubic-spline interpolated** to each expiry - which FRED
  publishes free.
- **OptionMetrics IvyDB** (its product literature): a proprietary CRR-based model for American options
  with discrete dividends, solved against the **mid of the best closing bid and offer**. Xing, Zhang &
  Zhao (2010) - the skew this panel cites - built their measure on IvyDB's volatilities.

QuantLib-Python (PyPI 1.44, Windows wheels, BSD licence) solves American implied volatility with a
discrete dividend schedule (`VanillaOption.impliedVolatility` with a finite-difference engine). So the
screener *could* invert its own mids the way the paper's data vendor does, with free inputs, at an
estimated seconds-to-a-minute for ~2,000 options a run (unmeasured).

**Why not just switch:** (a) the dividend projection is the hard part - Cboe uses analysts; the free
inputs are a trailing rate and a next ex-date; (b) a correct solver on a bad mid is still wrong, and
Yahoo's after-hours bid/ask is the quote the 20:00 refresh records; (c) one practitioner thread
reports a QuantLib American IV of 141% against Bloomberg's 116% on the same option, unresolved - a
solver set up wrongly fails silently. So the first step is a measurement, not a change.

**Proposal status: DECIDED 2026-11-04 (13:00 session, which did not write this note) - ADOPTED, with one change to the plan: the construction is labelled per reading, so the switch splices nothing instead of discarding a month of evidence.** The reasoning, the disjoint replication and the build order are section 1d; the label shipped the same session.

*The original proposal, as written before the measurement:* a later session pre-registers and runs a paired measurement: for one
evening's quotes, invert the ATM call and put and the 0.95-moneyness put with QuantLib (CRR or finite
differences, discrete dividends from the run's own dividend fields, FRED constant-maturity rate
interpolated to expiry), against Yahoo's `impliedVolatility` for the same contracts; report the
distribution of differences, how many stocks' skew changes sign, and whether the panel's sentence
would change for any. Pre-register before running: adopt only if the median absolute ATM difference
exceeds 1 volatility point **and** a hand-checked sample of 10 contracts reproduces an independent
calculator. The precedent is `research/measurements/2026-10-14-skew-construction-paired.py`. Display
only either way (settled row "ctx").

### 1a. Pre-registration (written and committed 2026-11-03, before any implied volatility was computed)

The owner asked for the work to continue the same night, so the measurement is run by this session
and **the decision stays OPEN for a session that did not write the proposal** - running a
measurement is not deciding on it. Fixed before any number is seen:

- **Sample:** every 10th stock of the newest full run's universe in ticker order (about 50), quotes
  pulled once, after the close, through the pipeline's own expiry choice (`choose_expiry`), at-the-money
  strike and XZZ legs (`_xzz_leg`). No stock is added or dropped after seeing a result; one with no
  usable leg is reported as such.
- **Inputs:** spot = the quote response's price (`chain_spot`); price = bid/ask mid; risk-free rate =
  FRED constant-maturity Treasury yields (1, 3, 6, 12 months), linearly interpolated in time to the
  expiry (Cboe interpolates the same curve with a cubic spline; over 1-12 months the difference is
  far below a volatility point); dividends = the run's `dividendRate`, paid in equal instalments on
  dates projected forward from the stock's last ex-dividend date at its median past interval, within
  the option's life.
- **Model:** QuantLib American exercise, finite-difference engine with the discrete dividend schedule.
  **Independent check:** a Cox-Ross-Rubinstein tree written in this repository from the textbook
  recursion (Hull, *Options, Futures and Other Derivatives*, ch. 13 and 21), with discrete dividends
  by the escrowed-dividend approximation, on 10 contracts chosen by rule (the first 10 ATM calls in
  sample order). "Reproduces" means within 0.5 volatility points on all 10.
- **Reported:** the distribution of QuantLib minus Yahoo at the money (call and put averaged, as the
  panel does) in volatility points; the same for each skew leg; how many stocks' skew changes sign;
  how many would cross the panel's IV bounds.
- **Bar, unchanged from above:** a change is warranted only if the median absolute ATM difference
  exceeds 1 volatility point **and** the independent check reproduces QuantLib. Otherwise Yahoo's
  field stays, and this note says it was measured and found adequate.

### 1b. Result (2026-11-03, about 21:30 ET; the decision is OPEN for a session that did not write this)

`research/measurements/2026-11-03-implied-vol-quantlib-vs-yahoo.py`, as pre-registered, on 51 stocks
(every 10th ticker), quotes pulled once after the close. Self-test first: QuantLib and the CRR tree
each recover a 25.0% volatility from QuantLib's own price, with and without a discrete dividend.

- **Priced at the money: 38 of 51.** Not priced: 11 with a leg that had no usable quote or no Yahoo
  IV, 2 with no common strike, NVR with no expiry - listed in the output, none dropped by choice.
- **ATM IV, QuantLib minus Yahoo: median -3.16 volatility points; median |difference| 3.16; 90th
  percentile 6.09; maximum 9.43** (ZBH: 48.4% against Yahoo's 57.8%, 17 days out).
- **Independent check: 10 of 10 ATM calls reproduce within 0.5 points** (largest gap BIIB, 0.43).
- **Put skew (XZZ legs, the same contracts the panel uses): 30 stocks priced, median difference +2.51
  points, and the sign differs for 17 of 30.**
- No QuantLib ATM value fell outside the panel's IV bounds.
- **The pre-registered bar is MET** (median |ATM difference| 3.16 > 1, and 10 of 10 reproduce).

**What Yahoo's number is, as far as can be told** - a diagnostic run after the result, deciding
nothing (`research/measurements/2026-11-03-yahoo-iv-what-does-it-invert.py`). Against Yahoo's figure
on the same contracts, median |difference| in points: American on the mid (ours) 2.29, American on the
last trade 4.03, European on the mid 2.08, European with no rate on the last trade 4.38 - **none of the
four reproduces it** (each within half a point for 4-7% of contracts). The last trades were a median
**9.8 hours** old when pulled (over 6 hours for 112 of 144), so Yahoo's field is most likely computed
at some earlier moment of the day against a different underlying price - which no reader of the panel
is told. **The skew is where it matters: Yahoo's is negative (out-of-the-money put below the
at-the-money call) for 28 of 48 stocks; ours, from the closing quotes, for 5 of 48.** A put a 5% fall
away being cheaper in volatility terms than an at-the-money call is the unusual case for single
stocks, so a field that says so for most of the index is the likelier error. That inference is
this session's, not a cited measurement - the deciding session should weigh it.

**Cost if adopted:** 9.6 ms per American inversion with a dividend, so three contracts for 500 stocks
is about 14 seconds a refresh; QuantLib 1.44 has Windows wheels (BSD licence) and would join
`requirements.txt`. **What adopting would change:** the 20:00 refresh and the live pass invert the
ATM call, ATM put and the XZZ put from the mids they already keep (the expected move is unaffected);
the context log records which construction each reading used, and `context_eval` excludes the
Yahoo-field skew from the record rather than splicing it (the 2026-10-14 precedent). **What not
adopting would leave:** a put-skew sign on the page that disagrees with the quotes it sits beside for
more than half the sample.

### 1c. The review's pre-registration (written and committed 2026-11-04 13:20, before anything was computed)

The 2026-11-04 13:00 session is the independent reader (prompt section 1; it did not write the
proposal). The author's bar is met, so what is being decided is **adoption**, which is a different
question: *is our construction better than Yahoo's field, or only different from it?* The author
flags its own key inference - that a negative put skew for most of the index is the likelier error -
as uncited, so the review's job is to supply evidence for or against it. Fixed before any number is
seen:

- **A disjoint replication.** The same measurement on `recs[5::10]` - every 10th stock of the same
  universe in ticker order, **offset by five**, so no stock overlaps the author's sample. Same
  expiry choice, legs, rate curve and dividend rule. It runs **mid-session** (about 14:30 ET), not
  after the close, which is deliberate: see the next item.
- **A discriminating test of the staleness explanation.** The author's diagnostic found that none of
  four candidate constructions reproduces Yahoo's field, and that the last trades behind it were a
  median 9.8 hours old. The obvious remaining candidate is the one it could not test after the
  close: **Yahoo's field is an inversion of the last trade against the underlying price at the
  moment of that trade.** Mid-session, each contract's `lastTradeDate` and the day's 1-minute bars
  make that testable. Four candidates, median |difference| from Yahoo's field in volatility points:
  (1) American on the mid at the current spot - ours; (2) American on the last trade at the current
  spot; (3) American on the last trade at **the spot at that trade's minute**; (4) European on the
  last trade at the spot at that trade's minute.
- **Whether Yahoo's skew is internally consistent**, which is the point that decides it either way:
  for each stock's two XZZ legs, the gap between the two legs' last trades and the underlying's move
  over that gap. A skew built from two inversions taken at different spots is not a measure of the
  smirk at one moment, whatever its sign.
- **The dividend gap in the author's independent check.** The CRR cross-check is reported as 10 of
  10 within 0.5 points, but the printout does not say how many of those ten carried a projected
  dividend - and the dividend schedule is the part the author calls hardest. The replication reports
  that count, and **the first ten ATM calls that carry at least one dividend** are checked as well.
- **The literature, read from the paper rather than from memory.** Xing, Zhang & Zhao (2010), the
  source of this panel's measure, reports the distribution of its own SKEW across stocks. Whether a
  negative SKEW is the ordinary case or the unusual one for large US stocks is a citable fact, and
  it is the author's uncited inference restated as something checkable.

**The review's decision rule, fixed now.** **Adopt** only if all three hold: (a) the disjoint sample
also puts the median absolute ATM difference above 1 volatility point; (b) the independent CRR tree
still reproduces QuantLib within 0.5 points, **including on contracts that carry a dividend**; and
(c) at least one of the two independent lines - the paper's own distribution of SKEW, or Yahoo's
field being reproduced by a stale or internally inconsistent snapshot - shows Yahoo's construction
to be the less defensible one. **Refuse** if the replication fails (a), or if (b) fails where
dividends are present, or if the paper says a negative SKEW for most large stocks is ordinary - in
which case the author's inference is wrong and the sign disagreement is not evidence for us.
Display-only either way (settled row "ctx"); adoption would also have to say what happens to the
skew already recorded in `data/context_log/`, which is a cost the proposal names and does not size.

### 1d. The decision (2026-11-04, 13:00 session): ADOPTED, and the switch is made to cost no evidence

All three conditions in 1c are met, so the proposal is **accepted**: the panel's implied
volatilities should be this project's own inversion of its own bid/ask mids, not the source's
undocumented field. The measurement is
`research/measurements/2026-11-04-implied-vol-review-replication.py` (committed), run mid-session
at 14:10 ET on a sample disjoint from the author's.

**The author's pre-registration is genuine.** Section 1a was committed in `57958d5`; the result and
the script arrived in `741215b`. Nothing was computed before the rule for reading it was fixed.

**(a) The replication holds, and the magnitude is smaller than the note's.** 40 of 50 stocks priced
at the money. QuantLib minus Yahoo: **median -1.78 volatility points, median |difference| 2.00,
90th percentile 4.18, maximum 7.51** - against the author's -3.16/3.16/6.09/9.43 after the close.
Above the 1-point bar on both samples, so the finding replicates; the gap is roughly a third
narrower mid-session, which is itself consistent with the field being staler at 21:00 than at 14:10.

**(b) The dividend gap in the author's cross-check is closed, and it holds.** The 11-03 printout
reported 10 of 10 ATM calls reproducing within 0.5 points but did not say how many carried a
dividend - and the dividend schedule is the part the note itself calls hardest. Measured here on
two sets chosen by the same rule: **10 of 10 with no dividend in the option's life, and 9 of 9
with one** (largest gap CF, 60.46% against the CRR tree's 60.82%). A finite-difference engine with
a discrete dividend schedule and an independently written binomial tree using the escrowed-dividend
approximation agree on every contract tried, 19 of 19. The solver is not set up wrongly, which is
the failure mode the note's own practitioner anecdote warns about.

**(c) The paper's own distribution decides it, and the evidence was already in this repository.**
The note's key inference - that a put a 5% fall away being cheaper in volatility terms than an
at-the-money call is the unusual case, so a field reporting it for most of the index is the likelier
error - is marked uncited. It need not be. **Xing, Zhang & Zhao (2010) Table 1, p. 647**, read from
the paper by the 2026-10-14 session and recorded in
`research/measurements/2026-10-14-skew-distribution-vs-paper.py`: across 1996-2005, averaging ~840
firms a week,

    SKEW:  mean +6.40   5% -0.99   25% +2.40   50% +4.76   75% +8.43   95% +19.92   (IV points)

The 5th percentile is **-0.99**, so in the data this measure comes from a **negative SKEW is about a
one-in-sixteen event**.

Against that, the published record - not a 48-stock evening but **every logged day under the current
construction**, read from the committed context logs:

| | days | stocks/day | SKEW negative | median SKEW |
|---|---|---|---|---|
| Xing, Zhang & Zhao, 1996-2005 | ~520 weeks | ~840 | ~6% (5th pct -0.99) | **+4.76 pts** |
| ours as published (source field) | **12** | 476-493 | **42-60%, median 47%** | -0.69 to +1.10 pts |
| QuantLib from our own mids | 1 + 1 | 48 / 26 | 10% / **35%** | ~+2 to +4 pts |

That is about **5,800 stock-days** saying the published field puts the skew negative for roughly
half the index every single day, where the paper's own cross-sections put it there for about one
stock in sixteen. **The author's inference is confirmed**, on a base three orders of magnitude
larger than the one it was drawn from, and adopting moves both the sign frequency and the level
toward the paper's distribution - the ~5-point level gap the 10-14 session measured and could not
explain (median -0.20 against +4.76) narrows by roughly 40%.

**The construction boundary was dated from the data, and the existing predicate is already right.**
The both-leg columns the 10-14 change added appear first in the **2026-10-16** log, so the six logs
from 10-08 to 10-15 are the old 0.90-moneyness measure and the twelve from 10-16 are the paper's.
`context_eval.skew_is_the_papers_measure` keys on the presence of those legs rather than on a date,
which puts the boundary in exactly the right place without anyone having to maintain it - worth
recording because the prose around it is looser than the code: the 10-14 changelog says "before
2026-10-14 ... those five log dates" and `CLAUDE.md` says "before 2026-10-16's log". The data says
six dates, ending 10-15, and `CLAUDE.md` is the accurate one.

#### Three corrections to the note

1. **"5 of 48" does not replicate in magnitude.** Our own construction put the skew negative for
   **9 of 26 (35%)** mid-session, not 10%. The *Yahoo* side replicates exactly - 15 of 26 (58%)
   against 28 of 48 (58%), and 47% on the full universe - so the honest contrast is **35% against
   58%**, or 10-35% against 47-58% across samples, not 10% against 58%. The direction is robust and
   the gap is narrower than the note implies. Both of our figures are small samples; the full
   universe cannot be measured for our construction without re-pulling 500 chains, which is not
   worth a rate-limit risk to the 02:00 run.
2. **The mechanism the note proposes is not confirmed.** It suggests the field is "computed at some
   earlier moment of the day against a different underlying price". That is exactly testable
   mid-session, and the candidate the 11-03 diagnostic could not try after the close was tried
   here: the **last trade inverted against the underlying's price at the minute of that trade**,
   from the day's 1-minute bars. Median |difference| from the field: **American 2.15 points (12% of
   contracts within half a point), European 2.13 (12%)** - against **ours at 2.30 (11%)**. Nothing
   reproduces it. With the 11-03 diagnostic's four candidates that is **eight constructions across
   two sessions and the field matches none of them.**
3. **Which makes the case for adoption stronger than the note's, not weaker.** The argument does
   not need the staleness story or the sign-plausibility inference. It is this project's own
   standing rule: *every number on the page must be checkable from the page*, and the page shows
   the engine's numbers rather than a re-derivation of them (`CLAUDE.md` row 0.8). Ours is specified
   end to end - American exercise, discrete dividends, a named Treasury curve, our own mids - and
   reproduces against independently written code on every contract tried. Theirs is a number whose
   method is documented nowhere and which eight attempts cannot rebuild. For a tool whose product
   is explainability that is the decision, whichever number is closer to the truth.

#### What the review adds that neither session had

**Yahoo's two skew legs are not a single moment.** For the 26 stocks with both legs, the gap between
the two contracts' last trades is a **median 73 minutes** - 25 of 26 are more than five minutes
apart, 14 more than an hour, the largest fourteen days - and the underlying moved a **median 0.289%
(maximum 1.67%)** between those two moments. Stated honestly, this is a fact about when the two
contracts last *traded*, and since no candidate reproduces the field it cannot be asserted that its
volatilities are struck at those moments. It is a reason to doubt a skew built from that source, not
a proof of its method. Ours takes both legs from one quote snapshot by construction.

#### The cost the proposal did not size, and the change to the plan

Both panel figures that depend on an inversion - the ATM implied volatility and the skew - change
construction together, and by this project's own precedent (excluded, never spliced) a switch
retires the readings built the old way. The first draft of this section sized that as **18 logged
days of `options_atm_iv` and 12 of the skew**, read off the logs. **That is wrong by a factor of
about fifteen, and the correct number is one day.** Run through `context_eval.signals_frame` itself
rather than counted by hand, the readings that actually enter the record today are:

| log date | rows | in the ATM-IV record | in the skew record |
|---|---|---|---|
| 2026-10-08 .. 2026-11-03 (17 days) | 503 each | **0** | **0** |
| 2026-11-04 | 503 | **467** | **466** |

Every earlier day is already excluded, by the **2026-11-03 (night)** entry in this changelog: a
reading is counted only if it carries the price its quotes were taken against (`_ctx_opt_spot_src`),
and that column exists only from the 11-04 log. So the record for both signals is **one day old**,
its first one-month reading was already due about **2026-12-04**, and switching construction now
costs **one day of readings and essentially no calendar time** - about a day of delay, not the month
the first draft implied. The lesson is the obvious one: the cost of a change to an evidence series
is a number to get from the code that builds the series, not from the files underneath it.

**The decision still modifies the proposal, for the general reason rather than the sizing: label the
construction per reading, publish ours, and let the source series keep accruing on its own.** One
constant decides which series the record counts (`context_signals.PUBLISHED_IV_CONSTRUCTION`), an
unstamped reading is a `source` reading because nothing here has ever inverted an option price, and
the two are never pooled. The switch then costs **nothing at all** rather than one day, and it
produces something neither construction can give alone: a paired record, which is the only way this
project will ever be able to say which construction predicts better. That part shipped with this
decision (`context_signals`, `context_eval`, 5 tests).

**And it makes the timing argument the opposite of what it looks like.** A month of accrued evidence
would have been a reason to delay the switch until the first reading landed. There is no such
month - the window is as open as it will ever be, and every day of waiting is a day of evidence
accrued on a construction this review has just found indefensible.

**The one cost still unsized is coverage.** Ours needs a usable bid/ask mid where the field needs
nothing: **160 of 196 contracts (82%)** that carried a source IV had a usable mid, and the ATM pair
priced for 40 of 50 stocks. Measure it on the whole universe from one 20:00 refresh before the
switch - and `MAX_REL_SPREAD` and the IV bounds are **not** to be loosened to win coverage back
(`CLAUDE.md` 0.11).

#### Build order for the session that takes it

1. **One inversion, feeding all three contracts** (ATM call, ATM put, XZZ put leg), so the panel's
   subtraction is arithmetic on published numbers and not a second computation (row 0.8).
2. **No new per-ticker network call - and this is the one real obstacle, now measured.** The
   Treasury curve is four FRED series, cached daily with a last-known fallback: easy. The dividend
   *schedule* is not. Checked on the newest run's `00_raw_fetch.parquet` rather than assumed:
   `dividendRate` is there (403 of 503) and `trailingAnnualDividendRate` (503 of 503), but
   **`exDividendDate`, `lastDividendDate` and `lastDividendValue` are all absent**, so nothing in
   the run says *when* the next dividend falls. The 11-03 measurement got the dates from
   `yf.Ticker.dividends` - a request per stock, inside the pass that tripped the rate limiter when
   it ran inside the core fetch (settled row "ctx"). So the build's first step is to add
   **`exDividendDate` to the core `.info` fetch**, which costs no extra request because `.info` is
   already pulled, and then to verify it with a throwaway full run - a fetch-schema change is not
   checkable from unit tests. That is why this is a session of its own and not the tail of this one.
3. **Fall back to the source field and stamp it `source`** where QuantLib, the curve or a mid is
   unavailable. A gate that can only fail closed jams an unattended loop; the same applies to a
   signal.
4. **Flip `PUBLISHED_IV_CONSTRUCTION` and add the generator's matching guard in the same commit** -
   the page must withhold a figure built the way the record refuses to count, as it already does for
   the expected move's divisor (`SPOT_SOURCES`).
5. **Pin QuantLib in `requirements.txt`** (1.44, BSD, Windows wheels; 9.6 ms per inversion, so about
   14 seconds for three contracts across 500 stocks).
6. **Say the construction on the panel and register the claim**, so the number is checkable from the
   page.
7. **Measure coverage before and after**, and report any loss rather than tuning it away.

## 2. skfolio and "how much"

The north star's unanswered question is position size. skfolio can compute risk-parity,
inverse-volatility or minimum-CVaR weights for any list. Printing such weights beside named stocks on
the public page **recreates the Model Portfolio panel removed on 2026-08-26**, with sizes added - the
closest the tool came to a recommendation (CLAUDE.md, "The dashboard is the product"). The honest
teaching content is a result, not a library: across 14 optimisation models and 7 datasets, none
consistently beat equal weighting out of sample (DeMiguel, Garlappi & Uppal 2009, *RFS* 22(5)). If
the Factor Lab ever wants it, that replication needs no optimiser library - and `lab_engine.js` must
reproduce the Python in the browser, which a heavy optimiser makes hard. Not proposed.

## 3. Mining and selecting: why RD-Agent and "keep the survivors" are excluded by design

- **RD-Agent's evidence** (Table 1): CSI 300, tested 2017-01 to 2020-08, daily rebalancing into the
  top 50. Annual return 14.21% (o3-mini) against 5.70% for the Alpha158 library - so "2x" is roughly
  that comparison. The test period sits **inside the language models' training data**; the
  out-of-sample check (Table 2, CSI 500 and NASDAQ 100, 2024 to mid-2025) reports ICs (NASDAQ 100:
  0.0162 vs 0.0040) but **no returns**; there is **no multiple-testing correction and no error bars**
  (App. C.1 declines them), and the paper does not say which split guides its 23-44 iteration loops.
- **"Keep the ones that survive fees"**: under the null, the best Sharpe ratio of N trials grows
  roughly like the square root of 2 ln N, so selecting the winner selects for luck, and a fee filter
  does not remove the selection. The corrections are known: the probability of backtest overfitting
  by CSCV (Bailey, Borwein, Lopez de Prado & Zhu, *J. Computational Finance* 2017), a t >= 3 bar for a
  new factor (Harvey, Liu & Zhu 2016, *RFS* 29(1)) - over 10,000 tests a Bonferroni bar is |z| near
  4.6 - and the deflated Sharpe ratio (Bailey & Lopez de Prado 2014). The research engine implements
  the first two; it scores IC differences rather than Sharpe ratios, so the third is not needed.
- **Two weeks of paper trading** ("if it doesn't beat buy-and-hold, you don't have an edge yet") is
  about ten observations of a daily strategy and less than one of a monthly one. This project's own
  estimate is about 85 months to distinguish an IC of 0.05 from zero (DECISIONS.md, the research
  engine's record).

## 4. What the post claims that could not be verified

The "329k+ stars" total is roughly right (today's counts above sum to about 332k) and irrelevant -
stars measure attention, not evidence; RD-Agent's absolute factor counts behind "70% fewer"; any out-of-sample annual return
for RD-Agent; that the combination has ever been tested end to end by anyone.
