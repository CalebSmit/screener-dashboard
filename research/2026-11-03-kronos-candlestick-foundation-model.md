# Kronos (a candlestick foundation model) and "Day Trader" - do either belong in the screener?

**2026-11-03, owner-run evening.** The owner forwarded a social-media post calling two GitHub repos
"the most BROKEN combo in trading": **Kronos**, a foundation model for price candles, and **Day
Trader**, eleven Claude Code skills for intraday trading. The post's worked example: on BTC candles,
725 of 1,000 sampled futures end down; a paper short with a 1% stop and 2% target - 203 paths hit the
target first, 302 the stop, 495 neither.

**Verdict: no change, and neither is a context-signal candidate today.** The evidence does not
address the question this screener asks, and the post's example is not evidence of anything. The
reasoning, with every number attributed, is below; section 5 says what would reopen it.

Read from primary sources only: the paper (arXiv 2508.02739v1, all 36 pages, read with pypdf), the
Kronos README, model code and Hugging Face card, and the Day Trader README. Nothing was installed or
run. Figures marked *(checked)* were re-read from the PDF text by a second reader the same evening.

## 1. What Kronos is

- Shi, Fu, Chen, Zhao, Xu, Zhang & Li (Tsinghua), *Kronos: A Foundation Model for the Language of
  Financial Markets*, arXiv 2508.02739 (v1, 2 Aug 2025). The README reports AAAI 2026 acceptance
  (AAAI-26 proceedings vol. 40(30), DOI 10.1609/aaai.v40i30.39730 - the AAAI page refused the fetch,
  so the DOI is only partly verified).
- A tokenizer turns each candle (open, high, low, close, volume, amount) into a 20-bit token in coarse
  and fine halves; a decoder-only transformer predicts the next token. Sizes 4.1M (mini), 24.7M,
  102.3M, 499.2M parameters (Table 1); **the 499M "large" model, which holds most of the best
  results, is not released.** Context is capped at 512 candles.
- Pretraining: 12.11 billion candles, 96,569 assets, frequencies from 1-minute to weekly (Table 13),
  **up to June 2024** *(checked, App. D)*; the paper's tests start July 2024. The cut-off of the
  *downloadable* checkpoints is not stated anywhere found.
- MIT licence for code and weights; the card is silent on the training data's own terms.

## 2. What the paper measured, and what it did not

- **The "93% better" headline is price-path shape, not returns:** Kronos-large's RankIC across a
  forecast path averages 0.0267 against 0.0138 for the best general time-series model (Tables 14-15).
- **Return forecasting** is scored by an IC **pooled over all samples in an asset class and frequency,
  not cross-sectionally per date** (p.15) - so it is not the quantity this screener's IC measures
  (rank of stocks against each other on one date, `improvement_engine`), and the two cannot be
  compared.
- **US stocks (XNAS) return RankIC:** small 0.0513, base 0.0544, large 0.0601, against 0.0518 for a
  linear baseline (DLinear) and 0.0461 for Chronos-large *(checked, Tables 16-17)*. **The released
  models do not clearly beat a linear model on US stocks.**
- **Volatility on XNAS:** a GARCH model's R-squared 0.2271 against Kronos's 0.0961-0.1169 (Table 18).
- **Horizons:** daily forecasts use 40 days to predict 12 (Table 8); the investment simulation uses 90
  to predict 10. **No one-month horizon on daily data is tested** - and one month is this screener's
  optimisation horizon.
- **The only trading test is one year of Chinese A-shares** (CSI 300/800, July 2024 - June 2025,
  0.15% per trade, 5-day minimum hold; Table 10). CSI 300 annualised excess return: Kronos-small
  18.05% (information ratio 1.24), Kronos-base 19.11% (1.38), the best baseline MOMENT-large 16.55%
  (1.20) *(checked)*; Kronos-large 21.93% (1.42). **There is no US backtest**, no factor-adjusted
  return (so the signal may be short-term reversal or momentum restated - the README itself calls the
  outputs "raw signals" needing risk-factor neutralisation), and one year is one observation of a
  regime.
- No independent replication was found. A separate study of time-series foundation models on daily
  excess returns (Rahimikia, Ni & Wang, arXiv 2511.18578) finds off-the-shelf ones perform poorly
  zero-shot and fine-tuned; its abstract does not name Kronos.

## 3. Why the post's example is not evidence

The 1,000 "futures" are 1,000 samples from **one** model conditioned on **one** history: they are the
model's own uncertainty, not 1,000 outcomes. "725 of 1,000 end down" says the model leans down; it
says nothing about how often such leans are right. That needs the model's calibration measured on
many independent dates - the post's own step 6 ("re-run on each of the last 5 days") is five
overlapping observations. The stop/target race (203/302/495) is the same samples counted again.
(Taken at face value it is also thin: 203 x 2% - 302 x 1% = +104 percentage points across the
1,000 paths, about +0.10% per path before any cost, with the 495 paths that touch neither level
left out of the sum.) The default Kronos API averages
the sampled paths into one forecast, so counting them needs custom code; the figures are
unverified. The post's other claims (salaries at Citadel, "most BROKEN") are marketing.

This is the same mistake the screener has guarded against since August: counting overlapping
or model-generated observations as independent (`research/2026-08-10-ic-evidence-independence.md`;
settled row 0).

## 4. Day Trader

github.com/Jakeschincariol/day-trader-skill (MIT): a chart scanner, news monitor, entry timer, exit
planner, "Time Machine", risk manager, position sizer, order executor (Alpaca, paper by default,
typed confirmation), price alerts, an earnings analyst and a Congress-trades tracker, on free Yahoo,
Nasdaq symbol lists, SEC EDGAR and the House Clerk. The Time Machine **resamples the stock's own
5-minute bars from the previous 59 sessions** - a bootstrap of history, not a forecast, and it does
not use Kronos. It is intraday workflow tooling that ends in an order. **It does not fit**: this
screener is decision support for a monthly review and never emits a trade (CLAUDE.md, "The dashboard
is the product"; the Model Portfolio decision), and the bot is a separate project.

## 5. Could Kronos be a context-layer signal? Not now - and what would change that

The context layer is the right *kind* of home for an unproven signal (shown beside the score, logged,
never scored until it earns a measured `1m` record - settled row "ctx"). It is a poor candidate:

1. **Never tested at the horizon that matters**, and on US stocks the released sizes are level with a
   linear baseline at the horizon that was tested.
2. **It would duplicate what the screener already has.** It reads only prices and volumes, which is
   the Momentum and Risk categories' input, plus the Trend and Recent move panels.
3. **It cannot be explained to a student.** Every number on the page must be checkable from the page
   (row 0.8); a 100-million-parameter sampler's output is not, and the screener's case is that it is
   teachable.
4. **Cost.** No CPU figure is published and the code keeps no key-value cache. A rough estimate
   (Kronos-small, about 21 daily steps, 10 paths) is 1-7 hours of CPU per daily run for 500 stocks -
   unmeasured, so a one-stock timing test should replace it before anyone relies on it; the post's
   1,000 paths would be about 100 times that.
5. **Time.** Any new context signal needs 8 effective one-month observations before it may be
   considered for the score - about mid-2027 at the earliest, like the signals already logging.

**What would reopen it:** a published, cross-sectional (per-date, across stocks) evaluation on US
large caps at a horizon of about a month, with a released checkpoint whose training cut-off is stated
and which beats a linear baseline after costs. If one appears, the honest first step is a Factor Lab
replication or a research-engine experiment registered before it runs (CLAUDE.md, 0.13 and 3) - not a
daily signal. Until then, nothing here is a proposal.

**Proposal status: none** - a finding; no change is proposed.
