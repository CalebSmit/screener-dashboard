# One pre-registered stop-loss on the bot's rule, and it did not do what the paper does

**2026-10-22, 06:00 session.** `OWNER_FOCUS.md`'s top open item, in full: register one stop-loss
variant of the trading rule before building it, build exactly that one, report it beside the
unstopped rule, and never sweep the stop level.

**Decision-grade for nothing before 2027-02-11** (CLAUDE.md rule 5). Nothing here changes a weight,
a metric, a threshold or the bot. What it changes is what the public research page says a stop
would have done, which until today it did not say at all.

---

## 1. Why this stop and not another

The owner's question came from trader-style setups — a chart, a price target, a stop. The screener
repo's answer (recorded in the bot's `SESSION_LOG.md`, 2026-10-22) was that chart rules mostly fail
after costs under fair testing, and that stop-losses usually *lower* a diversified long-only book's
return. The standard reference for the second half is **Kaminski, K. & Lo, A. (2014), "When do
stop-loss rules stop losses?", _Journal of Financial Markets_ 18, 234–254** — which is where the
*Journal of Financial Markets* in the owner's item comes from, not from the paper tested here.

The one exception with published support is a stop against **momentum crashes**:

> **Han, Y., Zhou, G. & Zhu, Y. (2016), "Taming Momentum Crashes: A Simple Stop-Loss Strategy."**
> SSRN working paper 2407199 (first draft January 2014; the version read here is September 2016).
> DOI **10.2139/ssrn.2407199**, resolved through Crossref on 2026-10-22, which returns SSRN posted
> content and **no journal**: it is a working paper and is cited as one here, not as peer-reviewed
> work.

Read from the PDF, not second-hand (the copy read is the one hosted at
`smallake.kr/wp-content/uploads/2017/03/SSRN-id2407199.pdf`, 52 pages, which matches the SSRN
abstract page's title, authors and September 2016 date).

**What the paper finds**, CRSP 1926–2013, equal-weighted six-month momentum deciles (Table I):

| | Original momentum | Stop at L = 15% | Stop at L = 10% |
|---|---|---|---|
| Mean excess return a month | 0.99% | 1.93% | **2.32%** |
| Standard deviation | 6.01% | 4.85% | **4.61%** |
| Sharpe ratio | 0.165 | 0.399 | **0.504** |
| Skewness | −1.18 | +1.28 | **+1.54** |
| Worst month | −49.79% | −17.43% | **−15.37%** |

(Panels A–C, printed pp. 6–7.) Turnover and stop frequency, printed pp. 8–9: the sell stop closes
about **10.05%** of the winners' positions in an average month and raises the winners' monthly
turnover from **39.11%** to **49.08%**.

## 2. The rule, exactly as the paper states it

Section III, printed pp. 5–6:

- `P0` is the stock's price at the beginning of the month — "the close price at the end of the
  previous month when we form the momentum portfolio."
- On each trading day *t*, compute `R^L_t = (P^L_t − P0)/P0`, where `P^L_t` is **the lowest price of
  day t**. The first time `R^L_t ≤ −L`, the stop is **triggered**: trigger price `SP_D = (1−L)·P0`.
- The sale is **not** executed that day. On the following day the stock is sold at the limit price
  `LP_D = (1 − L/2)·SP_D` — 85.5% of `P0` at L = 10% — whenever it is reached. If the day's high is
  below `LP_D`, sell at the close. If the day's low stays above `LP_D`, keep the stock and repeat on
  the following trading days of the month. The `L/2` band exists "to make sure that the initial drop
  … is genuine and not due to random fluctuations."
- The proceeds earn the T-bill return for the rest of the month.
- The short side has the mirror-image buy stop. Only the sell side is relevant here.

## 3. Four differences in how it is built here, each one registered before it ran

These are in `research/experiments/registry.yaml` under `bot-rule-v1.1-stop10`'s
`construction_note`, committed in the same commit as the code. Each one changes *when* the stop
fires, so none of them is cosmetic:

1. **The trigger is the close, not the day's low.** `cache/v2_daily.parquet` carries date, close,
   adjusted close and volume — no intraday range. So this stop fires **less often** than the
   paper's, and only on falls that hold to the close.
2. **The sale is at the triggering close, not at the next day's limit.** The paper's `L/2` band
   requires a *further* 5% fall (at L = 10%) before it sells. This exits **sooner and higher** than
   the paper, and it also exits names the paper's band would have kept after a bounce.
3. **Cash earns nothing**, here as everywhere in this harness; the paper earns T-bills on the
   proceeds. The stop is charged that lost yield.
4. **It is the long side of a 25-name screen, not long-short momentum deciles.** The paper sorts
   every CRSP common stock on six-month returns with a one-month gap, above $5, excluding the
   smallest NYSE size decile. There is no short side here and no decile.

## 4. The prediction, written before the run

Registered verbatim: *lower (less negative) maximum drawdown and a less negative worst month in
**every** period; mean excess over the equal-weighted universe about unchanged or slightly lower,
and in no period a difference this record could distinguish from zero; turnover higher.* The
paper's doubled Sharpe ratio was explicitly **not** predicted — it is a long-short result.

## 5. What happened

`research/measurements/2026-10-22-stop-loss-arm.json`, read from the ledger rows themselves (both
on code `good/2026-10-22-owner-1-3-g513c364`, one data vintage, `comparable_provenance: true`):

| | 2012–2019 (96 mo) | 2020 – mid-2023 (42 mo) | mid-2023 – 2026 (38 mo) |
|---|---|---|---|
| Worst month, no stop | −8.13% | −17.27% | −5.27% |
| Worst month, 10% stop | **−8.41%** | **−13.96%** | **−8.36%** |
| Max drawdown, no stop | −14.55% | −20.62% | −7.52% |
| Max drawdown, 10% stop | **−15.43%** | **−18.65%** | **−14.13%** |
| Excess vs the average scored stock, no stop | −0.148% (t −1.09) | −0.701% (t −2.64) | −0.220% (t −0.55) |
| Excess, 10% stop | −0.240% (t −1.71) | −0.690% (t −2.32) | −0.411% (t −0.94) |
| Turnover a year, no stop → stop | 1.42 → **2.00** | 1.74 → **3.05** | 1.35 → **2.42** |
| Positions stopped | 180 | 164 | 125 |

**The prediction failed in two of the three periods, on the one thing a stop is for.** It cut both
the worst month and the deepest fall only in 2020 – mid-2023 — the period containing March 2020. In
the held-out years it made the deepest fall **nearly twice as bad** (−7.52% → −14.13%) and the
worst month half again worse, and in 2012–2019 it made both slightly worse. Turnover rose as
predicted, by **1.4× to 1.8×**. Excess return was about unchanged in 2020 – mid-2023 and about
0.1–0.2 points a month lower in the other two, which is within what this length of record can see.

**The construction checks out against the paper's own frequency figure.** The stop closed 469
positions over 176 months, a median of 1 a month, none at all in 58 of them, and 25 of 25 in March
2020. As a share of positions held that is **8.1%** a month in 2012–2019, **18.5%** in
2020 – mid-2023 and **15.2%** in the held-out years, against the paper's 10.05% of winner positions
over 1926–2013. The same order of magnitude, higher where the period was more volatile — which is
what the paper reports about volatility too (its stopped share correlates 68.49% with market
volatility for the winners, printed p. 9).

## 6. Why it behaves this way, and what that does not license

A stop sells into a fall and holds cash to the month's end, so it pays for the falls it avoids with
the rebounds it misses. Across all 176 months it changed the result in 174 and **helped in 82**
— fewer than half. In the ten worst months for the unstopped rule it helped in **7**; in the 58
months the rule lost money it helped in **36**. The one month it clearly earned its keep is March
2020, where it turned −17.27% into −13.96% with every position stopped.

The mechanism that makes it worse elsewhere is the same one in reverse: a 25-name book puts ~4% in
each holding, and a stopped name is re-bought at the next rebalance at whatever price it has
recovered to. A sequence of ordinary 10% dips and recoveries — which is most of 2023–2026 — is
exactly the sequence this rule loses money on, and the loss shows up in the portfolio's monthly
path, which is what the drawdown measures.

**This is not evidence that a stop-loss is bad**, and it must not be read as any of the following:

- It is not a refutation of Han, Zhou & Zhu. Their result is a long-short momentum portfolio whose
  crash risk comes mostly from the **short** side (their own section IV: "it seems mainly coming
  from the losers portfolio"). There is no short side here.
- It is not decision-grade for the bot or for anything else (rule 5, until 2027-02-11), and it is
  measured on a survivorship-limited universe with no analyst inputs.
- **It is not a reason to try 15% or 20%.** One level was registered and one was run. A stop level
  picked by what this history rewarded would be a number fitted to 176 months of one market, and
  the engine's own rules forbid registering a variation of a near-miss for exactly that reason.

What it *is*: the honest answer to the owner's question, on this screener's own ranking, with the
prediction on the public record ahead of the result and the result disagreeing with it.

## 7. Where it is published

The research page (`research.html`, "What a simple trading rule on the ranking would have done")
now carries the stop arm's table beside the unstopped rule's, the construction, and a verdict
sentence **generated from the two summaries** — so it can say the prediction failed, and does.
`tests/test_research_page.py::test_the_stop_loss_arm_is_shown_beside_the_rule_with_a_verdict_that_can_say_it_failed`
feeds it three fabricated pairs (better everywhere, worse everywhere, mixed) and checks it says a
different thing for each; a sentence that could only ever report success would be checking
plumbing.

**Proposal status: none.** Nothing is proposed. The item asked for one test, once; it is done and
reported, and the next reading of it is not a sweep but the live record after 2027-02-11.
