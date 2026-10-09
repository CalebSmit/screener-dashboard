# The momentum regime rule: why it is off, and what would replace it - 2026-10-09 (owner-run)

**Status:** the rule is **switched off** (`config.yaml` `momentum_regime.enabled: false`, changelog
2026-10-09 "metric audit"). This note records the measurement that switched it off and a
pre-registered design for a replacement, so the next session does not re-tune the old one.

## What the rule did

`factor_engine.adjust_momentum_weight` took `current_vol` = the standard deviation of
`momentum_score` **across the ~500 stocks** of one run, ranked it against every earlier run's figure in
`factor_vol_history.csv`, and scaled the Momentum category weight: x0.70 above the 75th percentile
("HIGH VOL", freed weight to Quality and Valuation), x1.15 below the 25th ("LOW VOL", funded from
Valuation). The methodology page said it "tracks market-wide volatility".

## Why that input cannot measure volatility

`momentum_score` is a weighted mean of within-sector **percentile ranks** of three metrics. A
percentile column has a spread fixed by construction (uniform on 0-100, standard deviation ~28.9),
whatever the returns behind it. The spread of a weighted mean of such columns depends only on how
closely the columns' ranks agree with each other - not on the size of returns or the state of the
market. Measured on run `a2d76219dc0a`:

| | |
|---|---|
| spread predicted from the three metrics' rank correlations alone | 25.03 |
| spread measured | 25.07 |
| correlation of the run-to-run spread with the S&P 500's 21-day realised volatility | +0.37 |
| `factor_vol_history.csv` drift | 27.0 -> 25.07 as the metrics' correlations changed |
| rule replayed over its own history | **LOW in 30 of 33 runs where it acted, NORMAL 3, HIGH never** (re-measured) |

So in practice it was a one-way rule that raised momentum from 13 to 14.95 and cut Valuation from 22 to
20.05 on most days, for a reason it did not measure.

## What the literature supports

* **Barroso & Santa-Clara (2015), "Momentum has its moments", *JFE* 116(1).** Momentum's risk varies
  over time and is predictable from its own recent realised volatility (they use 126 trading days);
  scaling exposure by it "virtually eliminates crashes and nearly doubles the Sharpe ratio". The scale
  is usually below one, occasionally above.
* **Daniel & Moskowitz (2016), "Momentum crashes", *JFE* 122(2).** Crashes cluster in "panic" states -
  after market declines and when market volatility is high - and coincide with market rebounds; a
  dynamic strategy built on forecasts of momentum's mean and variance roughly doubles the static
  strategy's alpha and Sharpe ratio.
* **Cooper, Gutierrez & Hameed (2004), "Market states and momentum", *JF* 59(3)**: momentum profits
  follow up-markets (positive trailing three-year market return); after down-markets they vanish.

All three are about the time-series state of the **market or of the momentum portfolio**. None uses a
cross-sectional dispersion of scores.

## A replacement, pre-registered (not built)

1. **Input:** the S&P 500 total-return index (`^SP500TR`, already the run's market series): trailing
   126-day realised volatility, and the trailing 24-month return (Daniel & Moskowitz's down-market
   state). Thresholds from the index's **own long history** (fixed percentiles of 126-day volatility
   since 1990), not from this screener's few dozen runs - so the rule is decided before it is used.
2. **Rule:** momentum's weight is cut (x0.70, the existing magnitude) only in the panic state - trailing
   24-month market return negative **and** 126-day volatility above its long-run 80th percentile.
   Otherwise unchanged. **No upward scaling**: an above-one scale has support for a long-short
   momentum portfolio, not for a 13% sleeve of a long-only composite.
3. **Evidence gate before switching on:** the note above plus a replay over the index history showing
   how often the state fires (expect a few percent of months - 2008-09, 2020, 2022). Record it in the
   changelog with the replay.
4. **What would make it wrong:** if the state fires on more than ~15% of months in the replay, the
   thresholds are not describing panics and should not be used.

**Replay, run the same day** (`research/measurements/2026-10-09-momentum-panic-replay.py`, output beside
it): over ^SP500TR 1988-2026, the 126-day volatility 80th percentile is 20.6%, and the state holds at
**33 of 442 month-ends (7.5%)** - 2001-03..09, 2002-07..2003-05, 2008-09..2009-09 (which contains
Daniel & Moskowitz's March-May 2009 momentum crash) and 2010-07..08. It passes the 15% bound in
point 4, and it is not in force today. Two cautions for the build: the 80th percentile is taken over
the whole history, so the replay's early months use a threshold they could not have known (the live
rule would use the fixed figure); and the state missed 2020 and 2022, when the 24-month market return
stayed positive - consistent with Daniel & Moskowitz's definition, not a defect.

Until then the rule stays off and `factor_vol_history.csv` keeps recording the dispersion figure
(harmless, and the series that would show if the old input ever starts moving with markets).

## Sources
* Barroso, P. & Santa-Clara, P. (2015). Momentum has its moments. *Journal of Financial Economics* 116(1), 111-120.
* Daniel, K. & Moskowitz, T. (2016). Momentum crashes. *Journal of Financial Economics* 122(2), 221-247. https://www.nber.org/papers/w20439
* Cooper, M., Gutierrez, R. & Hameed, A. (2004). Market states and momentum. *Journal of Finance* 59(3), 1345-1365.
