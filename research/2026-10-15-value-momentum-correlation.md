# Value and momentum together: is the hedge real among large companies, and does it still hold?

**2026-10-15, owner-run (night).** `plan/factor-lab.md` queue item 7. Measurement:
`research/measurements/2026-10-15-value-momentum-correlation.py`. Checks a premise; sets nothing.

## The question

The momentum regime rule (changelog 2026-10-12, evening) cuts Momentum's weight in Daniel &
Moskowitz's panic state and sends half the freed weight to Valuation. Part of the case for Valuation
as the recipient is that value and momentum hedge each other: when one does badly the other tends to
do well. That premise had been cited, not checked.

## The paper

Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere", *Journal of Finance* 68(3),
929-985 - read from the published version (Lasse Pedersen's NYU copy), Table I Panel A. US stocks,
January 1972 to July 2011, a universe of large, liquid companies; portfolios are thirds by each
signal, value-weighted, returns over the T-bill rate:

| US stocks, 1972-01..2011-07 | Value P3-P1 | Momentum P3-P1 | 50/50 combination |
|---|---|---|---|
| Sharpe ratio | 0.29 | 0.33 | **0.63** |
| Correlation, value with momentum | | | **-0.53** (P3-P1); -0.65 (rank-weighted factors) |

Across their eight markets and asset classes the correlation "averag[es] about -0.60", and "in every
market, the value/momentum combination outperforms either value or momentum by itself."

## Rebuilt on public data

Kenneth French's 2x3 portfolios (value: high minus low book-to-market; momentum: high minus low
12-2 month return), the average of the small and big halves ("all") or the big half alone:

| Window | Set | Sharpe, value | Sharpe, momentum | Sharpe, 50/50 | Correlation | On market residuals |
|---|---|---|---|---|---|---|
| 1972-01..2011-07 (the paper's) | all | 0.48 | 0.56 | 0.80 | -0.17 | -0.22 |
| | **big** | **0.27** | **0.35** | **0.49** | **-0.20** | -0.22 |
| 2011-08..2026-08 (after) | all | 0.00 | 0.19 | 0.16 | -0.28 | -0.29 |
| | **big** | **0.00** | **0.01** | **0.01** | **-0.39** | -0.37 |
| 2007-01..2026-08 | big | -0.13 | 0.06 | -0.04 | -0.43 | -0.39 |
| 1927-2026 | big | 0.23 | 0.33 | 0.51 | -0.39 | -0.33 |

**What reproduces.** Among big companies over the paper's window, each premium matches the paper
closely (0.27 against 0.29; 0.35 against 0.33), and the combination beats either alone (0.49).

**What does not.** The correlation is milder: -0.20 against the paper's -0.53. The likely reason is
construction, not data: the library's book-to-market uses market value from the previous December,
while the paper's value measure uses a more current price. A value signal on a stale price is
partly a short-term momentum signal in reverse, so updating the price makes value and momentum more
opposed. That is the argument of Asness & Frazzini (2013), "The Devil in HML's Details" - cited here
from its known thesis, not re-read tonight; a reviewer should check it before a changelog cites a
figure from it.

**What happened next.** The hedge got *stronger* after the paper's sample (big companies: -0.39 since
August 2011, -0.43 since 2007), while both premia among big companies went to about zero (Sharpe
0.00 and 0.01). The combination's Sharpe ratio since 2011 is 0.01.

## What it means for the screener

- **The premise holds, as a statement about risk.** Value and momentum are reliably opposed among
  large companies, in every window measured and more so recently. Sending weight freed from
  Momentum to Valuation in a panic keeps a hedge in the composite rather than removing one. The
  screener's Valuation category prices its ratios at the current price, the construction that the
  paper's evidence says opposes momentum more strongly than the library's.
- **It is not a return argument.** Neither premium earned anything among big companies after 2011
  in this data, and the research engine found the same for the screener's own categories over
  2012-2026 (`research/2026-10-15-research-engine-first-batch.md`). The changelog entry should be read
  as moving weight toward a diversifier, not toward a better return.
- **No change.** The rule's three numbers are pre-registered and are not re-fitted to anything here
  (CLAUDE.md, item 0.12).

A lab experiment for this paper needs a layout the lab does not have (a correlation and a
combination, not a sort); it stays in `plan/factor-lab.md` item 7 for a Friday.
