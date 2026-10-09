# Forward EPS growth: a fixed 12-month horizon on one accounting basis (2026-10-09)

`forward_eps_growth` is 45% of the Growth category (5.85% of the composite) - the largest single
input to Growth. The revenue-growth finding the same morning prompted the same check here.

## What it was

Yahoo's `forwardEps` over `trailingEps`: `(forwardEps - trailingEps) / max(|trailingEps|, $1)`,
clipped to -75%..+150%, NaN where the two differ by more than 2x or less than 0.3x (Phase 13, F5).

**Measured 2026-10-09** (live, AAPL/MSFT/NKE/JPM/NVDA/KO/ORCL/COST): `forwardEps` equals the
consensus for the fiscal year **after** the current one (the `+1y` row of `eps_trend`), for every
stock checked. So the comparison runs from the trailing twelve months to a fiscal year that ends
**13 to 24 months later**, depending on where the company is in its fiscal year: Microsoft (June
year-end) ~24 months, JPMorgan (December) ~18. And `trailingEps` is GAAP while the consensus is
on the analysts' adjusted basis - the reason F5 had to throw away rows whose ratio looked extreme.

## Documented practice

MSCI Fundamental Data Methodology (March 2021), section 2.2.5, **Short-term Forward EPS Growth Rate
(EGRSF)**, "a measure of the expected growth of a security over the next 12 months":

    EGRSF  = (EPS12F - EPS12B) / |EPS12B|
    EPS12F = (M x EPS1 + (12 - M) x EPS2) / 12      EPS1, EPS2: current- and next-FY consensus
    EPS12B = (M x EPS0 + (12 - M) x EPS1) / 12      EPS0: last reported fiscal year
    M      = months remaining before the current fiscal year end

It is one of the five variables in MSCI's growth style definition. Its time-weighting is exactly
what removes the fiscal-calendar dependence.

## What it is now

`EPS12F` as MSCI defines it, from Yahoo's `eps_trend` `0y` and `+1y` consensus and `nextFiscalYearEnd`.
For `EPS12B` the screener uses **the last four reported quarters' actual EPS** from
`earnings_history` - the actuals the analysts' estimates are compared against, so both sides are on
the same (consensus) basis - rather than MSCI's blend of the last fiscal year's EPS with the current
estimate, which would need a GAAP or vendor actual. The $1 denominator floor and the -75%..+150%
clip are kept from the old metric. Where the consensus inputs are missing, the old form (with its F5
guard) is the fallback; `_feg_basis` records which was used.

**Measured on a live sample of 35:** both computable for 35 of 35; rank correlation with the old
values only **0.65**; median 14.8% against the old 34.5%. Examples of what the old form produced:
Albemarle 1,102% before clipping (GAAP trailing EPS depressed by charges) against 87.7%; Intuit 64.5%
against 0.9% (adjusted vs GAAP); KKR 137% against 31%.

## Fit with the rest of Growth

Revenue growth (25%) is now the latest quarter over the same quarter a year earlier and the 3-year
revenue CAGR (15%) is annual - both backward-looking, both on fixed windows. Forward EPS growth is the
category's forward-looking leg, now also on a fixed window. Sustainable growth (15%) is unchanged.

## Sources
- MSCI, *Fundamental Data Methodology*, March 2021, sections 2.2.5 (EGRSF) and 2.1 (EPS12F / EPS12B).
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_Fundamental_Data_Methodology_Mar2021.pdf
- MSCI, *Global Investable Market Value and Growth Index Methodology* (growth variables).
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_GIMIVGMethod_May2023.pdf
