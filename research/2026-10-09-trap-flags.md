# Trap flags: a value trap must be cheap, a growth trap must be growing - 2026-10-09 (owner-run)

**Found** walking the live site at phone width: the headline card reads "Trap flags 122 / 125 -
24% and 25% of stocks". A warning that fires on one stock in four tells a student little, so the
rule behind it was read and measured.

## What the code did

* **Value trap** (`apply_value_trap_flags`): flagged when at least 2 of quality, momentum and
  revisions sat in the bottom 30% of the universe. **Valuation was never consulted.** The
  published methodology page said the opposite - "a stock can score well on valuation (cheap!) but
  be cheap for a reason" - and "about 30% of stocks are typically flagged".
* **Growth trap** (`apply_growth_trap_flags`): 2 of 3 among {growth above the 70th percentile,
  quality below the 35th, revisions below the 35th}. So low quality plus low revisions flagged a
  stock as a *growth* trap whatever its growth.

## Measured on run `a2d76219dc0a` (2026-10-09 02:00, 501 stocks)

| | Value flag | Growth flag |
|---|---|---|
| Flagged | 122 (24%) | 125 (25%) |
| Median valuation percentile of value-flagged | **0.51** - no cheaper than the universe | |
| Value-flagged in the cheaper half | 62 of 122 | |
| Growth-flagged in the bottom half on growth | | **34 of 125** |
| Carrying both flags | 74 | 74 |

Seventy-four stocks were simultaneously a "value trap" and a "growth trap" - the two labels were
measuring one thing, broad weakness, which the Quality, Momentum and Revisions scores already show.

## What the literature and practice define

* **Piotroski (2000)**, "Value Investing: The Use of Historical Financial Statement Information to
  Separate Winners from Losers", *Journal of Accounting Research* 38: the test is run **within the
  highest book-to-market quintile**. Among those cheap firms, buying strong and shorting weak
  fundamentals earned about 23% a year in 1976-1996. The concept a value-trap flag borrows - cheap,
  and cheap for a reason - is defined inside the cheap set.
* **Mohanram (2005)**, "Separating Winners from Losers among Low Book-to-Market Stocks using
  Financial Statement Analysis", *Review of Accounting Studies* 10: the mirror image, **within
  low book-to-market (growth) stocks**, using a fundamentals score (G-score).
* Practitioner usage follows the same definition: "value trap" is applied to stocks that look
  cheap on multiples, and the remedy quant managers describe is to combine value with quality and
  momentum (Asness, Frazzini, Israel & Moskowitz 2015, "Fact, Fiction, and Value Investing",
  *Journal of Portfolio Management*) - exactly the three weakness dimensions this flag already uses.

## The change

* **Value trap** = Valuation score at or above the 70th percentile (the cheapest 30%, the same
  cut the growth ceiling uses) **and** weak on at least 2 of quality / momentum / revisions (the
  existing 30th-percentile floors and 2-of-3 rule, unchanged). Config key
  `value_trap_filters.valuation_percentile: 70`.
* **Growth trap** = Growth score above the 70th percentile **and** quality below the 35th **or**
  revisions below the 35th. Same thresholds; growth is now required rather than one of three.
* Severity scores, thresholds, the 2-of-3 tolerance, NaN handling and the `flag_only` switch are
  unchanged. The methodology page's two false sentences (the "about 30%" rate and severity
  averaged "across the dimensions that triggered the flag" - the code averages all three) are
  corrected in the generator.

**Effect on the same run:** value flags **122 -> 43**, growth flags **125 -> 74**, both **74 -> 5**.
The Top 5 (which skips flagged names) is unchanged - EXPE, HST, APA, BBY, DLTR - and 2 of the top 25
carry a flag before and after. No score, rank or composite changes; the flags are labels, plus the
exclusion rule in the Excel model portfolio and the Top 5.

**How it fits the whole.** The flags exist so a high Valuation (or Growth) score is not read on its
own. A flag on a stock that is neither cheap nor growing answers no question the category scores do
not already answer; scoped to its own category it marks exactly the stocks where one high score is
contradicted by the others - which is the defensibility feature CLAUDE.md rule 7 preserves.

**Not changed, and why:** the 30%/35% floors are conventions, not estimates; nothing here measures
whether a different cut is better, and the backtest cannot decide it before 2027-02-11 (rule 5).
