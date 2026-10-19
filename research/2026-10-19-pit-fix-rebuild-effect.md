# Backtest v2 rebuilt after the fiscal-year fix: what moved, a second defect it exposed, and no verdict changed (2026-10-19)

**The question.** The top-of-ranking audit (`research/2026-10-19-top-of-ranking-data-audit.md`,
section 4) changed `pit_record.Filed` so that a 10-Q cannot overwrite a fiscal year its 10-K
reported. `pit_record` builds every month of backtest v2, so the 176 cached months had to be rebuilt
and every research-engine row rerun before the research page quotes them again. What did the
rebuild change in the records, and did any result change with it?

**The answer.**

* **The fiscal-year rule touched 485 of 67,308 name-months (0.7%), at 53 companies**, spread over
  163 of the 176 months. The inputs it moved are the ones the audit predicted - annual EBIT and
  EBITDA (231 and 198 name-months), tax, revenue, net income, operating cash flow, D&A - and through
  them 14 weighted metrics. The largest single case is Comfort Systems (FIX), the audit's own
  example: April-August 2026, up to **257 places** in the baseline ranking.
* **The first rebuild exposed a second defect, now fixed.** It moved 1,000 name-months, **569 of them
  outside the fiscal-year rule's footprint** with no membership, sector or bank-flag change to
  explain them. Cause: `Filed._one` broke ties on filing date by the row order an unstable sort
  left behind, and the fiscal-year fix changed that order. A filing that carries two instants within
  the 10-day matching tolerance - an accounting change's opening balance beside the period end;
  **911 such same-day pairs in the cache, 523 of them dated 1 January** (ASC 842's adoption date
  beside 31 December) - had one picked arbitrarily, per concept: Air Products' FY2018 current assets
  were read at 2018-10-01 and its current liabilities at 2018-09-30. Ties now go to the end nearest
  the period asked for, then end and value; every sort is stable; an amendment wins over its
  original filed the same day. Two tests (`tests/test_pit_record.py`), the second of which shuffles
  the facts and requires an identical record.
* **No verdict changed.** All 29 registered experiments and the bot evaluation were rerun on the
  final months. Every held-out and 2012-2019 verdict is the same; the largest change in any IC
  difference against the baseline is **0.00065**; still 0 of 27 alternatives better or worse at the
  bar. The baseline's mean monthly IC moved from -0.00282 to **-0.00275** (2012-2019) and stayed
  -0.00003 (2020-2026); the methodology page prints both to three places (-0.003 and 0.000), so its
  generated sentence is unchanged.

## 1. Method

* **Rebuild.** `python backtest_v2.py` for 2020-01..2026-09 and `--tag _presample` for 2012-2019,
  sharded and merged, at commit `b3dfaff` (both fixes). The static fields (sector, industry, share
  class) come from run `377ec40838b1` (2026-10-19 02:00): the run the 2026-10-15 build used has been
  pruned, so it could not be reused. `cache/v2_metrics/_build.json` now records the base run and
  commit for every month (the 2026-10-15 build predates that record and reads `unrecorded`).
  Measured: **no member list and no sector or bank flag differs** between the two builds in any
  month, so the change of base run moved nothing that ranks.
* **Inputs unchanged.** Prices, index, splits and the SEC facts hash identically to what the ledger's
  previous rows ran on (`content_parts` in the ledger); the old months reproduce the ledger's
  `content_hash` 318ed0cfc297 exactly. New `content_hash`: 97c40a1303ae (month files d284a63ea21c0159, base run 377ec40838b1).
* **Attribution** - `research/measurements/2026-10-19-pit-fix-rebuild-effect.py` (JSON beside it):
  the fiscal-year rule's footprint is exact - every name-month where it can bite is rebuilt by the
  build's own `record_at` with and without the 10-K preference (same tie rule on both sides) and
  every input compared. Metric moves are then split into that footprint and the rest, and the first
  rebuild (fiscal-year rule, old ties) against the final one isolates the tie rule.
* **Verdicts** - `research/measurements/2026-10-19-pit-fix-rerun-verdicts.py` pairs each id's last
  row on the old months with its row on the new ones and recomputes both verdicts with the engine's
  own `verdict` / `verdict_presample`.

## 2. What moved in the records

| | Name-months | Companies | Metrics most moved |
|---|---|---|---|
| Fiscal-year rule (footprint) | 485 records; 405 with a metric moved | 53 | operating leverage 225, operating margin 224, ROIC 198, EV/EBITDA 188, Beneish 161, interest coverage 158 |
| Tie rule (first rebuild -> final) | 662 | - | Beneish M-score 653, Piotroski 36, asset growth 21 |
| Before -> after, all causes | 1,035 | 89 | |

Beneish dominates the tie rule because its indices compare two balance sheets a year apart, and the
prior year's is the one that sits beside an adoption-date opening balance. The most-touched companies
under the fiscal-year rule are Starbucks (46 name-months), Royal Caribbean, Axon, Masco and PPG;
FIX's five months (2026-04..08) are the only ones where a ranking moved by more than 100 places.

**Ranks, baseline weighting** (`v2_experiments.rank_all`, every month):

| | Before -> after |
|---|---|
| Per-month Spearman of composites | min **0.9965** (2026-05), median 0.99999; 65 of 176 months identical |
| Footprint name-months | mean shift 8.9 places, max 257.5 (FIX), 61 changed tenth |
| Every other name-month | mean shift 0.24, 99th percentile 3, max 114; 370 changed tenth (of 66,132) |
| Top tenth | 8 of 6,592 month-slots changed hands |

## 3. What it did to the results

| Figure | Before | After |
|---|---|---|
| Baseline IC, 2012-2019 | -0.00282 (95% -0.0229 to +0.0173) | -0.00275 (-0.0228 to +0.0173) |
| Baseline IC, 2020-2026 | -0.00003 | -0.00003 |
| Baseline IC, pooled 176 months | -0.00155 (t -0.19) | -0.00151 (t -0.18) |
| Alternatives better / worse at the bar | 0 / 0 of 27 | 0 / 0 of 27 |
| Verdicts changed (held out or 2012-2019) | | **none** |
| `drop-revenue-growth`, held out | -0.0074 (t -2.50) | -0.0072 (t -2.45) |
| `orthogonalised-categories`, held out | +0.0041 (t 2.39) | +0.0041 (t 2.41) |
| `drop-size`, 2012-2019 | -0.0038 (t -2.71) | -0.0039 (t -2.76) |
| `only-size`, 2012-2019 | +0.0267 (t 1.84) | +0.0266 (t 1.83) |
| Bot rule, excess vs universe, 2020-mid-2023 | -0.73%/month (t -2.72) | -0.70% (t -2.64) |
| Bot rule, 2020-2026 together | -0.49% (t -2.04, p 0.044) | -0.47% (t -2.00, p 0.049) |
| Months of record to show IC 0.05 (research page) | 47 | 48 |

The research page (`research.html`) changed only in those figures. The generated paragraph on the
methodology page (`run_screener._research_engine_sentence`) prints -0.003 and 0.000 either way.
Under CLAUDE.md rule 5 none of this decides anything before 2027-02-11; the point of the table is
that a data fix to the records did not move a conclusion, measured rather than assumed.

## 4. Two things worth remembering

* **A fix that changes a sort can change every tie downstream of it.** The fiscal-year rule was
  correct and tested; it still moved 569 name-months it had nothing to do with, because the record
  depended on row order. The tie rule's second test shuffles the facts and requires an identical
  record - that is the check that would have caught it, and it now runs with the suite.
* **Three figures quoted in other notes moved in the second decimal of t** (`drop-revenue-growth`
  -2.5 -> -2.45, `drop-size` -2.71 -> -2.76, the bot's 2020-mid-2023 gap -2.72 -> -2.64). Each note
  keeps the figure it reported, with a dated line pointing here.
