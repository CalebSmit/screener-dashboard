> **Answered the same day (owner-run): `research/2026-10-15-earnings-definition.md`.** The run's
> operating income is the source's *pre-special-items* row; its "Total Operating Income As Reported"
> row equals the filings 374 of 374. The persistence evidence and MSCI's practice support the
> pre-special-items figure for ROIC and EV/EBITDA, so no change; the open proposal is to make
> earnings yield consistent with it.

# Operating income is the screener's least source-agreeing input, and three weighted metrics stand on it (2026-10-15, 13:00 slot)

**Status: measured, not acted on.** This is a finding, not a proposal. Nothing here changed a
score, a weight or a gate tonight, and nothing should until the question in section 6 is
researched properly on a Monday.

## 1. How it was found

While re-measuring the three-quarter annualisation defect
(`research/measurements/2026-10-15-three-quarter-annualisation.py`), General Mills' scored
operating income came out at **$2.807bn** against the **$0.886bn** it filed for the same twelve
months - while its revenue, net income and operating cash flow agreed with the filing *to the
dollar*. One stock is an anecdote, so it was sized over the universe.

## 2. The size of it

`sec_crosscheck.FIELDS` checks revenue, net income, gross profit, operating cash flow, capex,
total assets and equity. **It does not check `ebit`** - the run's operating income - which feeds
`roic`, `ev_ebitda` and `net_debt_ebitda`. Adding it to the same comparison, on the run of
2026-10-15 (`research/measurements/2026-10-15-operating-income-vs-filings.py`):

| Input | compared | within 0.5% | over 10% | median abs gap |
|---|---|---|---|---|
| operating cash flow | 411 | 406 (98.8%) | 1 | 0.000% |
| net income | 421 | 397 (94.3%) | 7 | 0.000% |
| revenue | 410 | 371 (90.5%) | 8 | 0.000% |
| **operating income** | **322** | **140 (43.5%)** | **70** | **1.594%** |

It is not close. Operating income is the only input of the four where agreement with the
companies' own filings is the exception rather than the rule.

## 3. It is not the trailing rebuild

The obvious suspect was the check itself: `sec_crosscheck.flow_ttm` builds a trailing twelve
months as *year-to-date + prior fiscal year - prior year-to-date*, and a concept that behaves
differently across filings could break it. So the comparison was redone with **no rebuild at
all** - the source's own *annual* operating income (`ebit_annual`, which the payload already
carries) against the company's filed **fiscal year**, one period, one concept
(`research/measurements/2026-10-15-operating-income-annual-check.py`):

- **375 comparable stocks.** 38 more were excluded because they last tagged
  `OperatingIncomeLoss` for a fiscal year the run's statement does not cover - some as far back
  as 2010 - and comparing a 2026 figure with a 2012 one measures nothing. *(The first pass of
  this measurement did not exclude them and reported 121 over 10%; that number is wrong and is
  superseded by the one below. It is recorded here because the correction is the point: a
  filer that stops tagging a concept looks exactly like a filer that disagrees about it.)*
- **158 (42.1%) agree within 0.5%; 90 (24%) disagree by over 10%;** median absolute gap 1.69%.
- **0 of the 90 are a period behind their filings**, so staleness does not explain it.
- The source's figure is **the larger one for 64 of the 90 (71%)** - the direction you get if
  a charge the company put inside operating income has been left out.
- **9 stocks have opposite signs** on the annual figure.

So the gap survives with the reconstruction removed. It is between the two sources' figures for
the same fiscal year, not an artefact of how the check builds a trailing year.

## 4. What the gap looks like, by example

| Stock | Fiscal year | Source $bn | Filed $bn | Gap |
|---|---|---|---|---|
| KHC | 2025-12-27 | +4.637 | **-4.669** | +199% |
| BA | 2025-12-31 | **-5.416** | +4.281 | -179% |
| CNC | 2025-12-31 | -0.312 | **-7.623** | +96% |
| GIS | 2026-05-31 | +2.807 | +0.886 | +68% |
| CVS | 2025-12-31 | +10.385 | +4.660 | +55% |
| REG | 2025-12-31 | +0.583 | +1.123 | **-48%** |
| ESS | 2025-12-31 | +0.602 | +0.899 | **-33%** |

Two shapes, and they point in opposite directions:

- **The majority (71%):** the source's figure is larger, concentrated in companies that took a
  large impairment or write-down - Kraft Heinz, Centene, General Mills, CVS. Checked at the
  fact level for General Mills: its filed quarters to 2026-05-31 sum to **$0.8858bn**, exactly
  its filed fiscal year, and its net income of **-$87.6m** agrees with the filing to the dollar
  - so the company put a roughly $2bn charge *inside* operating income and the source did not.
- **A minority going the other way**, concentrated in real estate (REG, UDR, ESS, WY). There
  the filers' own `OperatingIncomeLoss` is the larger figure, which looks like a different
  subtotal rather than an excluded charge.

A third pattern sits behind both: **38 companies stopped tagging the concept altogether**, and
ECHO's filed figure of **-$17.7bn** against a source figure of -$0.091bn is not a definitional
nuance in either direction - it needs reading before it is called anything.

## 5. Why this matters to the screener, concretely

`ebit` carries weight through three metrics - `roic`, `ev_ebitda` and `net_debt_ebitda`. If the
majority pattern is what it looks like, the screener is scoring a **before-impairment**
operating profit for the companies that took impairments, which flatters exactly the kind of
company a valuation screen surfaces: cheap on an earnings multiple *because* of a write-down the
multiple does not see. That is a direction of error, not a random one, and it is the reason this
is worth a session rather than a footnote.

It also bears on `ebitda`, which is built from `ebit + da_cf` where the source's EBITDA row is
unusable (2026-10-09 audit).

## 6. What it does **not** license, and the question for a Monday

**Do not switch `ebit` to the filings.** Three reasons, all already settled here:

1. Scoring deliberately never reads the SEC cache (CLAUDE.md 0.14). Putting three weighted
   metrics on a weekly cache is the same move the capex proposal was refused for on 2026-10-15,
   and the refusal's reasoning applies unchanged.
2. The 2026-10-15 staleness measurement (`research/2026-10-14-yahoo-vs-sec.md` section 3c) found
   that switching source for a *subset* of stocks costs 20.5 median rank places in definitional
   noise - about as much as a quarter of staleness gains. Mixing two definitions of operating
   income inside one cross-sectional percentile is worse, not better.
3. **The filing is not obviously the right figure either.** The real-estate cases and ECHO say
   `OperatingIncomeLoss` is not one concept across filers. "Agrees with the filing" is the
   yardstick for revenue and cash flow because those are unambiguous; it is not established that
   it is the yardstick here.

**The question, stated so a Monday can answer it:** *should an operating profit used for ROIC
and EV/EBITDA include impairments and restructuring charges, and what do index providers and
practitioners actually do?* There is a real literature and a real practitioner split behind this
- Greenblatt's and Koller's treatments of operating earnings, MSCI's and S&P's quality
methodologies, and the standard analyst distinction between reported and adjusted operating
income - and the screener's existing comment at the `ebit` call site already cites Greenblatt
(2006) and Koller for preferring operating income over Yahoo's "EBIT" row. This is the next
question in the same series, and it has an answer in the sources rather than in the data.

**Until then, do not add operating income to the filing check as a gate.** It would fire for a
quarter of the universe every morning, which is how a gate gets muted. A *note* alongside the
existing per-field counts would be defensible once the normal range is known - the same
precedent `sec_crosscheck` is already following for its other fields, which the 2026-10-14 note
says to leave as a note "until a few weeks of counts show its normal range".

## 7. Reproduce

```
python research/measurements/2026-10-15-operating-income-vs-filings.py
python research/measurements/2026-10-15-operating-income-annual-check.py
```

Both are offline and read-only: the weekly `companyfacts` cache plus the published payload.
Re-run them rather than trusting the numbers above; the whole point of this note is that one of
its own first figures was wrong for a reason worth knowing.
