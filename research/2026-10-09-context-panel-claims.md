# Are the "Before you decide" panel's research sentences fair to their sources? (2026-10-09)

`plan/context-layer.md` queue item 4 asks for one research note per context signal. This is the
first, narrower step every one of those notes needs: **each sentence the panel already prints
about the research, checked against the source**, and corrected where it overstated. It does not
yet ask whether any signal should enter the score - that needs the measured record
`context_eval.py` started keeping today (first one-month window closes 2026-11-07).

| Panel sentence | Source says | Verdict / change |
|---|---|---|
| "a long-only rule of holding only above [the 200-day average] has historically cut drawdowns more than it raised returns (Faber 2007)" | Faber's rule is the **10-month** simple average, tested on **asset-class indices**. Updated through 2011 for the S&P 500: return ~9.6% -> 10.2%, volatility ~15.8% -> 11.8%, maximum drawdown ~-51% -> -23%. | Direction fair; scope was not. The page now says the evidence is for **broad market indices, not single stocks**, and names the 10-month average. |
| "short-term winners and losers have historically tended to partly reverse the following month (Jegadeesh 1990; Lehmann 1990)" | Jegadeesh (1990, *JF*): negative first-order serial correlation in **monthly** returns. Lehmann (1990, *QJE*): **weekly** reversal. Nagel (2012, *RFS*): reversal profits are compensation for supplying liquidity and are far larger when the VIX is high. | Fair, with Lehmann mis-scoped. Now "over a week, Lehmann 1990" and "most strongly when markets are stressed (Nagel 2012)". |
| "Steeper put skew has been followed by weaker returns (Xing, Zhang & Zhao 2010)" | *JFQA* 45(3): stocks with the steepest smirks underperform the flattest by **10.9% a year, risk-adjusted**, 1996-2005, persisting about six months; steep-smirk firms have worse earnings shocks next quarter. A 2015-2017 replication finds the same sign at about half the size. | Fair. Now dated and sized - "about 11% a year, risk-adjusted, in 1996-2005 ... smaller gaps in later data" - so a student does not read a 1990s magnitude as today's. **Definition gap, recorded:** XZZ's smirk is OTM put IV minus ATM *call* IV; the panel's skew is the 90%-strike put's IV minus ATM IV. Same idea, not the identical variable. |
| "Insider purchases have historically been informative and sales mostly have not ... (Lakonishok & Lee 2001; Cohen, Malloy & Pomorski 2012)" | L&L (2001, *RFS*): purchases predict returns, sales do not. CMP (2012, *JF*): the information is in *opportunistic* trades; *routine* ones carry none. | Fair; unchanged. Since 2026-10-08 the panel separates 10b5-1 plan sales and 10%+ holders, which is closer to CMP's distinction than the sentence alone. |

**What each future signal note should add** (Monday standard): effect sizes in large caps
specifically (the S&P 500 is where most of these effects are weakest), the post-publication
record, and how practitioners use the signal - then the measured record from `context_eval.py`.

## Sources
- Faber, M. (2007). "A Quantitative Approach to Tactical Asset Allocation." *Journal of Wealth
  Management*; figures through 2011 as reproduced in https://admissions.juniata.edu/offices/juniata-voices/media/andrew-and-castro.pdf
- Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security Returns." *Journal of Finance* 45(3).
- Lehmann, B. (1990). "Fads, Martingales, and Market Efficiency." *Quarterly Journal of Economics* 105(1).
- Nagel, S. (2012). "Evaporating Liquidity." *Review of Financial Studies* 25(7). https://www.nber.org/papers/w17653
- Xing, Y., Zhang, X. & Zhao, R. (2010). "What Does the Individual Option Volatility Smirk Tell Us
  About Future Equity Returns?" *JFQA* 45(3), 641-662. https://ideas.repec.org/a/cup/jfinqa/v45y2010i03p641-662_00.html
- Lakonishok, J. & Lee, I. (2001). "Are Insider Trades Informative?" *Review of Financial Studies* 14(1).
- Cohen, L., Malloy, C. & Pomorski, L. (2012). "Decoding Inside Information." *Journal of Finance* 67(3).
