# One-month reversal in an S&P 500 context panel (2026-10-09)

`plan/context-layer.md` item 4, second per-signal note. The "Before you decide" panel shows each
stock's one-month move beside its sector's median and, past +/-15%, says short-term winners and losers
have tended to partly reverse. Is that fair **for the largest US companies, today**?

## Evidence

- **The original effect.** Jegadeesh (1990, *JF*): negative first-order serial correlation in monthly
  returns; Lehmann (1990, *QJE*): weekly reversal. Nagel (2012, *RFS*): reversal profits are the return
  to supplying liquidity and are far larger when the VIX is high.
- **Faded in its plain form, especially among large stocks.** Chordia, Subrahmanyam & Tong (2014)
  attribute much weaker reversal and momentum profits to rising liquidity and trading activity; a
  later review finds the original one-month strategy "performs poorly after 2000" with value-weighted
  returns "mostly insignificant"; Blitz, van der Grient & Honarvar (2023) find the classic effect has
  "steadily weakened ... to the point of now having vanished entirely in most regions".
- **Survives relative to the industry, even in large caps.** Da, Liu & Schaumburg (2014, *Management
  Science* 60(3)): reversal "still exists even among large stocks ... it is the industry momentum effect
  that makes it difficult to find" - a residual (industry-adjusted) version earns more than the
  standard one. Blitz, Huij, Lansdorp & Verbeek's residual reversal is profitable after costs even in
  the 500 or 100 largest US stocks after 1990. De Groot, Huij & Zhou report positive net reversal
  profits among large caps once trading costs are handled.

## Verdict on the panel

The sentence was fair about the history but silent on the two conditions that matter here. It now
says that in large companies the plain effect has largely faded and what survives is the move
**relative to the stock's industry** - which is exactly the comparison the panel already prints (the
one-month move beside the sector median). The screener's momentum signal skipping the latest month
(12-1) is unaffected and remains standard practice.

**For `context_eval.py`:** `return_1m` is the plain signal; expect a weak or null IC in this universe.
A sector-relative variant (one-month return minus the sector median) is the one with large-cap
support and is cheap to add to the harness from the logs.

## Sources
- Jegadeesh (1990) *JF* 45(3); Lehmann (1990) *QJE* 105(1); Nagel (2012) *RFS* 25(7).
- Chordia, Subrahmanyam & Tong (2014), "Have capital market anomalies attenuated in the recent era of
  high liquidity and trading activity?", *Journal of Accounting and Economics* 58.
- Da, Z., Liu, Q. & Schaumburg, E. (2014). "A Closer Look at the Short-Term Return Reversal."
  *Management Science* 60(3), 658-674. https://academicweb.nd.edu/~zda/Reversal.pdf
- Blitz, van der Grient & Honarvar (2023), SSRN; Blitz, Huij, Lansdorp & Verbeek, "Short-term residual
  reversal", *Journal of Financial Markets* (2013).
- De Groot, Huij & Zhou, "Another Look at Trading Costs and Short-Term Reversal Profits".
  https://www.efmaefm.org/0efmameetings/efma%20annual%20meetings/2011-Braga/papers/0259.pdf
