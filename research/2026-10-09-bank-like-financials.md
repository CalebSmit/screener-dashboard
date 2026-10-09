# Which financials should be scored with the bank metric set?

*2026-10-09. Data: run `a2d76219dc0a` (00/01/05 parquet), rescored offline with HEAD scoring code. No live requests.*

## 1. The question and the defect

`_is_bank_like()` sends a Financials stock to the bank set (P/B 60 / earnings yield 40; ROE 35, ROA 25, equity ratio 15, Piotroski 15, accruals 10) if its **Yahoo** industry is listed, and **by default** otherwise. 26 of 59 bank-like stocks get there only by default. That puts fee businesses on P/B and equity ratio. Two examples: TROW's 72% equity/assets ranks at the 99th percentile, and AON's goodwill-inflated 6.1x P/B ranks at the 9th.

Two further defects in the rule:
- **EG** is a miss. Its industry string is `Insurance - Reinsurance`, but the list holds `Reinsurance`.
- **The rule reads Yahoo's *sector*, not GICS.** It runs before `run_screener` overwrites `Sector` with GICS. So XYZ, CPAY, JKHY and FIS (Yahoo "Technology"), GPN ("Industrials") and FISV ("Unknown") never reach it. Today they land in the right place by accident.

## 2. Literature and practice

**Documented:**
- **Fama & French (1992, p.429)** exclude financial firms "because the high leverage that is normal for these firms probably does not have the same meaning as for nonfinancial firms, where high leverage more likely indicates distress." **Novy-Marx (2013)** excludes all firms with a one-digit SIC code of 6 from the gross-profitability tests. The academic convention is a blanket exclusion of SIC 6, which also covers brokers (62xx), asset managers (6282) and insurance agents and brokers (6411). It gives no guidance on sub-industries.
- **Damodaran** (*Investment Valuation*, 3e, ch. 21) sorts financial firms into four groups by how they earn: banks (spread), insurers (premiums plus float income), investment banks, and investment firms (advisory and management fees). For a balance-sheet financial, debt is "raw material", not capital. Equity multiples (P/E, P/B) are therefore "a much better fit … than value multiples such as value to EBITDA". He also argues the P/B-ROE link is *stronger* for financials because book equity tracks marked assets. His worked examples price insurance brokers (MMC, AJG, AON, BRO, ERIE) and JPM's asset-management arm on **P/E**, not P/B.
- **MSCI Barra USE4** handles sector differences with GICS-based **industry factors**. Its example is Life Insurance having the lowest price-to-assets "industry beta". Style descriptors are defined the same way for every stock; I found no financials-specific descriptor. **MSCI Quality** (May 2025) applies ROE, D/E and earnings variability uniformly, with no sector carve-out.
- **S&P DJI (2022 Quality consultation)** stopped applying the accruals ratio to GICS 40 (Financials) and Real Estate. *I saw this only in secondary summaries; the primary PDF returned 403.*
- **Sell-side and valuation practice:**
  - Insurance brokers: EV/EBITDA is "the most prevalent" multiple (IRMI).
  - Asset managers: EV/EBITDA and P/E, with % of AUM as a cross-check (Mercer Capital). A UK court called EV/EBITDA "the best starting point" (Oxera, *Signia*).
  - Investment banks and brokers: P/TBV against ROTE (e.g. the GS vs MS comparisons).

**Inference (mine):** the documented dividing line is *whether liabilities are an operating input*. That is the case with deposits, insurance float, customer credit balances and consolidated insurer liabilities. Where they are, EV, EBITDA, ROIC and GP/A lose their meaning and P/B-ROE is the practitioner standard. Fee businesses with conventional P&Ls are valued like industrials, on EV/EBITDA and P/E. Goodwill-heavy brokers make P/B meaningless.

## 3. Classification (76 GICS Financials)

| Group | Tickers | Set | Why |
|---|---|---|---|
| Banks (diversified, regional) | BAC C JPM WFC PNC TFC USB CFG FITB HBAN KEY MTB RF | **bank** | Deposits are liabilities. JPM's Yahoo EV ($719bn) is below its market cap ($881bn). |
| Custody banks | BNY STT NTRS | **bank** | Bank holding companies with deposits; equity/assets 6.8-8.0%. |
| Consumer finance | AXP COF SYF | **bank** | Deposit-funded lenders. |
| Insurers, incl. reinsurance | AFL GL MET PRU PFG AIG AIZ L ACGL ALL CB CINF HIG PGR TRV WRB EG | **bank** | Float. P/B-ROE is the standard. |
| Multi-sector holding | BRK-B | **bank** | Insurance-dominated balance sheet. |
| Alt managers that consolidate insurers | APO (Athene), KKR (Global Atlantic) | **bank** | Equity/assets 4.3% and 7.5%. KKR's Yahoo EV is $141bn against an $83bn market cap. APO's "gross profit" is 95% of revenue. *Judgement call:* the sell-side uses distributable-earnings P/E. |
| AMP | AMP | **bank** | Owns Ameriprise Bank and RiverSource Life; equity/assets 3.2%. |
| Broker-dealers / IB | GS MS SCHW IBKR RJF HOOD | **bank** | Customer funds sit on balance sheet. IBKR's Yahoo EV is **-$29.6bn**, because $131bn of customer cash is in `totalCash`. GS's EV is $22bn against a $257bn market cap. HOOD is the weakest fit. |
| Insurance brokers | AON AJG BRO WTW MRSH ERIE | **generic** | Fee businesses that hold fiduciary funds only in trust. Practice is EV/EBITDA and P/E. ERIE earns a management fee from the Erie Exchange. |
| Traditional / pure-fee alt managers | BLK TROW BEN IVZ BX ARES | **generic** | Fee P&Ls. Practice is EV/EBITDA and P/E. |
| Exchanges / data, crypto | CME ICE CBOE NDAQ SPGI MCO MSCI FDS, COIN | **generic** | COIN goes with the exchanges: CME also carries clearing margin on its balance sheet and is generic today. |
| Payments | V MA PYPL CPAY GPN FIS FISV JKHY XYZ | **generic** | Unchanged. |

Yahoo supplies **no EBITDA and no gross profit for 42 of the 46** proposed bank-set stocks (all except APO, KKR, HOOD and IBKR). For banks and insurers the generic set cannot even be computed.

## 4. Measured effect (both arms rescored with HEAD code)

**13 stocks switch bank → generic. 0 switch the other way.** The other 13 of the audit's 26 default-path stocks stay in the bank set, but now by an explicit rule: STT, NTRS, AMP, PFG, RJF, APO, KKR, GS, MS, SCHW, IBKR, HOOD and EG.

| Ticker | Rank | Valuation | Quality | What changed |
|---|---|---|---|---|
| TROW | 130 → **89** | 63 → 82 | 65 → 62 | Equity ratio (99th pct) replaced by EV/EBITDA 6.5x (94th) and ROIC 20% (77th) |
| AON | 337 → 343 | 25 → **51** | 79 → 54 | P/B 6.1x (9th) → EV/EBITDA 10.9x (74th); ROE 45% (96th) → ROIC 18% (73rd) |
| WTW | 305 → 252 | 28 → 53 | 65 → 52 | |
| MRSH | 266 → 292 | 19 → 42 | 79 → 53 | |
| BEN | 177 → 282 | 58 → 41 | 40 → 37 | |
| BRO | 306 → 393 | 52 → 53 | 60 → 40 | |
| AJG | 428 → 462 | 23 → 22 | 44 → 27 | |
| ERIE | 328 → 454 | 22 → 48 | 92 → **31** | Only 21 of 29 weighted metrics present (no EBITDA or gross profit from Yahoo) |
| BX | 398 → 411 | 6 → 31 | 82 → 56 | Quality rests on accruals and Beneish alone (no EBITDA or gross profit) |
| BLK / ARES / IVZ / COIN | 432→445 / 449→456 / 109→112 / 501→501 | | | Small moves |

**Knock-on effects.** The generic-metric peer pool inside Financials grows from 17 to 25-28 names, and the bank pool shrinks from 59 to 46. Other Financials therefore move: mean |Δrank| is 7.1, the largest is PYPL at -33, and GS moves +16. Across the universe, 346 ranks move, by a mean of 2.8 places. No stock falls below the 60% coverage filter.

*Caveat:* HEAD's midpoint-percentile fix (2026-10-09) postdates this run, so the "old" ranks are the rescored baseline, not the published ones.

**Side finding, to investigate separately.** AON's Beneish M-score of -0.07 (1.7th percentile) comes almost entirely from DSRI = 3.52. That is probably fiduciary premium receivables, not channel stuffing. Brokers carry fiduciary receivables that Beneish's model was not built for.

## 5. Recommendation

1. **Classify on GICS sector and GICS sub-industry, not Yahoo.** `load_sp500` already downloads `constituents.csv`, which carries a `GICS Sub-Industry` column; keep it, and add it to `sp500_tickers.json`.
2. **Bank-set sub-industries:** Diversified Banks, Regional Banks, Consumer Finance, Commercial & Residential Mortgage Finance, Life & Health Insurance, Multi-line Insurance, Property & Casualty Insurance, Reinsurance, Multi-Sector Holdings, Investment Banking & Brokerage, Diversified Capital Markets.
   **Generic-set sub-industries:** Insurance Brokers, Asset Management & Custody Banks, Financial Exchanges & Data, Transaction & Payment Processing Services.
3. **Ticker overrides to bank, inside Asset Management & Custody Banks:** BNY, STT, NTRS (custody banks); APO, KKR (consolidated insurers); AMP. Each needs a one-line reason in code. The existing `_NON_BANK_FINANCIALS` list becomes redundant, since sub-industry covers it.
4. **Unknown Financials sub-industry:** keep the bank default, because P/B-ROE is the safer guess for an unseen lender. But make it loud: log a `dq_log` entry, and add a test asserting that **no current constituent reaches the default**. The current defect was 26 silent defaults.
5. **Yahoo fallback (when the CSV is unavailable):**
   - Normalise em-dashes to `" - "`.
   - Add `Insurance - Reinsurance`, which fixes EG.
   - Map `Capital Markets` → bank, and `Insurance Brokers` / `Asset Management` / `Financial Data & Stock Exchanges` → generic.
   - Keep ticker overrides PFG and RJF → bank, because Yahoo files them under Asset Management.
6. Ship with a `METHODOLOGY_CHANGELOG.md` entry citing §2 and the moves in §4.

## Sources

- Fama, E. & French, K. (1992). The Cross-Section of Expected Stock Returns. *Journal of Finance* 47(2), §I.A. https://people.hec.edu/rosu/wp-content/uploads/sites/43/2023/09/Fama-French-Cross-section-of-expected-stock-returns-1992.pdf
- Novy-Marx, R. (2013). The Other Side of Value. NBER w15940. https://www.nber.org/system/files/working_papers/w15940/w15940.pdf
- Damodaran, A. *Investment Valuation*, 3e, ch. 21 "Valuing Financial Service Firms". https://pages.stern.nyu.edu/~adamodar/pdfiles/val3ed/c21.pdf
- MSCI (2011). USE4 Methodology Notes, §2. https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf
- MSCI (2025). Quality Indexes Methodology, Appendices I-II. https://www.msci.com/indexes/documents/methodology/2_MSCI_Quality_Indexes_Methodology_20250520.pdf
- S&P DJI (2022). S&P Quality Indices consultation (secondary summary; primary returned 403). https://spglobal.com/spdji/en/documents/indexnews/announcements/20220630-1453817/1453817_spqualityindicesconsultation6-22-2022updated.pdf
- IRMI. Valuation Insights: Insurance Agencies and Brokers. https://www.irmi.com/articles/expert-commentary/valuation-insights-insurance-agencies-and-brokers
- Mercer Capital. Asset Manager Valuation and Rules of Thumb. https://mercercapital.com/asset-manager-valuation-and-rules-of-thumb
- Oxera (2018). The curious case of the valueless valuation (*Signia Wealth*). https://www.oxera.com/wp-content/uploads/2018/07/The-curious-case-of-the-valueless-valuation-1.pdf-1.pdf
- Nasdaq/Zacks. Goldman vs. Morgan Stanley (P/TBV comparison). https://www.nasdaq.com/articles/goldman-vs-morgan-stanley-which-financial-giant-has-more-upside
- GICS sub-industries: datasets/s-and-p-500-companies `constituents.csv`. https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv
- Measurement: `research/measurements/2026-10-09-bank-like-financials.py` (re-runs compute_metrics offline on run a2d76219dc0a with the flag forced).
