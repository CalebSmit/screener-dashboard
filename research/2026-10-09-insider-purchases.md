# Insider purchases as context in an S&P 500 screener (2026-10-09)

`plan/context-layer.md` item 4, the first per-signal note: what the literature says insider
trades are worth, **in companies the size of the S&P 500**, and whether the page's treatment
(officers and directors counted, 10%+ holders apart, plan sales tagged) is right. Context only -
nothing here proposes the signal for the score.

## What the evidence says

- **Purchases, not sales.** Lakonishok & Lee (2001, *RFS* 14(1); NYSE/AMEX/Nasdaq 1975-1995):
  "informativeness ... is coming from purchases, while insider selling appears to have no predictive
  ability." Jeng, Metrick & Zeckhauser (2003, *REStat* 85(2)): purchase portfolios earn abnormal
  returns of **more than 6% a year**; sale portfolios nothing significant.
- **Size is the condition that matters here.** L&L: firms with extensive insider purchases beat
  those with extensive sales by 7.8% over 12 months, **4.8% after size and book-to-market**, and "the
  usefulness of insider trading activity depends on company size ... large companies are priced more
  efficiently ... the biggest potential benefit ... is in the smaller companies." The market's
  reaction around trading and reporting is "around 1% for small firms, and is practically zero for
  large firms" - L&L's large firms are the top three NYSE size deciles, which is roughly the S&P 500.
  For aggregate management trading, the small-company spread was 19% and significant; for large
  companies 5% and not significant.
- **Who trades matters less than is often assumed.** Seyhun's "information hierarchy" (top
  executives, then officers, then directors) is the usual reading of his 1986/1998 work, but JMZ
  find top executives' purchases **do not** earn more than other insiders'. L&L treat holders of
  10%+ who are not management as a **separate category** ("large shareholders") alongside officers
  and directors.
- **Opportunistic vs routine.** Cohen, Malloy & Pomorski (2012, *JF* 67(3)): the information is in
  *opportunistic* trades; routine traders' trades carry none. Rule 10b5-1 plan sales are routine by
  construction.

## Is the page right?

| Page treatment | Verdict |
|---|---|
| Purchases emphasised, sales described as mostly uninformative | **Supported** (L&L; JMZ). |
| Plan sales tagged (Form 4 10b5-1 checkbox) | **Supported** in spirit by CMP's routine/opportunistic split. |
| 10%+ holders counted apart from officers and directors | **Consistent with L&L's own taxonomy**; no source found that says their trades carry the same information as management's. Kept. |
| No officer-vs-director weighting | **Supported**: the hierarchy is contested (JMZ). |
| The sentence citing L&L without the size condition | **Overstated for this universe.** Corrected today: the panel now says the evidence is strongest in smaller companies and weak among the largest, which includes the whole S&P 500. |

## Implication for the record `context_eval.py` keeps

The prior for `insider_buyers` / `insider_buy_value` predicting one-month returns **in this
universe** is weak: the published effects are concentrated in small firms and accrue over 6-12
months, not one. A null result at `1m` should therefore be read as expected, not as a reason to
remove the panel - the panel's job is to show what insiders did, which a reader may weigh for
themselves.

## Sources
- Lakonishok, J. & Lee, I. (2001). "Are Insider Trades Informative?" *Review of Financial Studies*
  14(1), 79-111. Full text read: https://www.lsvasset.com/pdf/research-papers/Insider-Trades-Informative.pdf
- Jeng, L., Metrick, A. & Zeckhauser, R. (2003). "Estimating the Returns to Insider Trading: A
  Performance-Evaluation Perspective." *Review of Economics and Statistics* 85(2), 453-471.
  https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/9919.pdf
- Cohen, L., Malloy, C. & Pomorski, L. (2012). "Decoding Inside Information." *Journal of Finance* 67(3).
- Seyhun, H. N. (1986). "Insiders' Profits, Costs of Trading, and Market Efficiency." *Journal of
  Financial Economics* 16(2), 189-212 (read via secondary summaries; the hierarchy is cited only as
  "the usual reading").
