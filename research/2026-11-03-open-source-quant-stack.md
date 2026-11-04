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

**Proposal status: OPEN** - a later session pre-registers and runs a paired measurement: for one
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
