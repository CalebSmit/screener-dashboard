const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak
} = require("docx");

// ── Color palette ──
const ACCENT = "1B3A5C";       // Dark navy
const ACCENT_LIGHT = "E8F0FE"; // Light blue bg
const GREEN = "1B7A3D";
const GREEN_BG = "E6F4EA";
const AMBER = "9A6700";
const AMBER_BG = "FFF8E1";
const RED = "C62828";
const RED_BG = "FDECEA";
const GRAY = "5F6368";
const GRAY_BG = "F1F3F4";
const BORDER_COLOR = "DADCE0";
const WHITE = "FFFFFF";

// ── Table helpers ──
const border = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COLOR };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: WHITE };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: ACCENT, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.LEFT, children: [
      new TextRun({ text, bold: true, color: WHITE, font: "Arial", size: 20 })
    ] })]
  });
}

function cell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: [
      new TextRun({ text, font: "Arial", size: 20, bold: opts.bold, color: opts.color || "000000" })
    ] })]
  });
}

function verdictBadge(verdict) {
  const map = {
    "STRONG": { color: GREEN, fill: GREEN_BG },
    "SOUND": { color: GREEN, fill: GREEN_BG },
    "STANDARD": { color: ACCENT, fill: ACCENT_LIGHT },
    "ACCEPTABLE": { color: AMBER, fill: AMBER_BG },
    "CONCERN": { color: RED, fill: RED_BG },
    "MINOR CONCERN": { color: AMBER, fill: AMBER_BG },
    "ENHANCEMENT": { color: AMBER, fill: AMBER_BG },
  };
  const m = map[verdict] || { color: GRAY, fill: GRAY_BG };
  return new TableCell({
    borders,
    width: { size: 1400, type: WidthType.DXA },
    shading: { fill: m.fill, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: verdict, bold: true, color: m.color, font: "Arial", size: 18 })
    ] })]
  });
}

// ── Bullet helper ──
function bullet(text, bold_prefix = "") {
  const runs = [];
  if (bold_prefix) {
    runs.push(new TextRun({ text: bold_prefix, bold: true, font: "Arial", size: 22 }));
  }
  runs.push(new TextRun({ text, font: "Arial", size: 22 }));
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 60, after: 60 },
    children: runs
  });
}

function subBullet(text, bold_prefix = "") {
  const runs = [];
  if (bold_prefix) {
    runs.push(new TextRun({ text: bold_prefix, bold: true, font: "Arial", size: 20 }));
  }
  runs.push(new TextRun({ text, font: "Arial", size: 20 }));
  return new Paragraph({
    numbering: { reference: "subbullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: runs
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 32, color: ACCENT })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 26, color: ACCENT })]
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.spaceBefore || 80, after: opts.spaceAfter || 80 },
    alignment: opts.align,
    children: [new TextRun({
      text, font: "Arial", size: opts.size || 22,
      bold: opts.bold, italics: opts.italics, color: opts.color || "000000"
    })]
  });
}

function spacer() {
  return new Paragraph({ spacing: { before: 120, after: 120 }, children: [] });
}

// ── Accent bar (thin colored line) ──
function accentBar() {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({ children: [
      new TableCell({
        borders: { top: { style: BorderStyle.SINGLE, size: 6, color: ACCENT },
                   bottom: noBorder, left: noBorder, right: noBorder },
        width: { size: 9360, type: WidthType.DXA },
        margins: { top: 0, bottom: 0, left: 0, right: 0 },
        children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: [] })]
      })
    ] })]
  });
}

// ── Build document ──
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "subbullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2013",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({ children: [
        new Paragraph({ alignment: AlignmentType.RIGHT, children: [
          new TextRun({ text: "Multi-Factor Screener \u2014 Methodology Audit", font: "Arial", size: 16, color: GRAY, italics: true })
        ] })
      ] })
    },
    footers: {
      default: new Footer({ children: [
        new Paragraph({ alignment: AlignmentType.CENTER, children: [
          new TextRun({ text: "Page ", font: "Arial", size: 16, color: GRAY }),
          new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: GRAY }),
          new TextRun({ text: " \u2014 Confidential", font: "Arial", size: 16, color: GRAY })
        ] })
      ] })
    },
    children: [

      // ═══════════════════════════════════════════════════════
      // TITLE PAGE
      // ═══════════════════════════════════════════════════════
      spacer(), spacer(), spacer(), spacer(),
      accentBar(),
      spacer(),
      para("METHODOLOGY AUDIT REPORT", { size: 40, bold: true, color: ACCENT, align: AlignmentType.LEFT }),
      para("Multi-Factor Equity Screener", { size: 32, color: ACCENT, align: AlignmentType.LEFT }),
      spacer(),
      para("Credibility & Defensibility Assessment", { size: 24, color: GRAY, italics: true }),
      para("Against CFA Institute Standards & Institutional Quant Practices", { size: 22, color: GRAY }),
      spacer(), spacer(),
      accentBar(),
      spacer(),
      para("Date: February 20, 2026", { color: GRAY }),
      para("Model Version: 8-Factor (v2.0)", { color: GRAY }),
      para("Universe: S&P 500", { color: GRAY }),
      para("Prepared for: Professor Review", { color: GRAY }),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // EXECUTIVE SUMMARY
      // ═══════════════════════════════════════════════════════
      heading1("1. Executive Summary"),
      para("This report audits the scoring methodology of the Multi-Factor Equity Screener for academic and institutional credibility. The model scores S&P 500 stocks across eight factor categories using sector-relative percentile ranking, missing-data-adaptive weighting, and trap detection filters. Each component was evaluated against published academic research, CFA Institute curriculum, and documented practices at institutional quant firms (AQR, MSCI, Dimensional Fund Advisors)."),
      spacer(),

      // Summary verdict table
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3200, 4760, 1400],
        rows: [
          new TableRow({ children: [
            headerCell("Component", 3200),
            headerCell("Assessment", 4760),
            headerCell("Verdict", 1400),
          ] }),
          new TableRow({ children: [
            cell("Factor Selection", 3200, { bold: true }),
            cell("All 8 factors map directly to Fama-French 5 + momentum + quality + revisions. Academically canonical.", 4760),
            verdictBadge("STRONG"),
          ] }),
          new TableRow({ children: [
            cell("Metric Computation", 3200, { bold: true }),
            cell("FCF Yield, ROIC, Piotroski, accruals follow standard formulas. Appropriate guardrails on edge cases.", 4760),
            verdictBadge("STRONG"),
          ] }),
          new TableRow({ children: [
            cell("Factor Weights", 3200, { bold: true }),
            cell("Val/Qual dominant (22% each), consistent with value-quality tilt. Size/Investment at 5% reasonable for S&P 500.", 4760),
            verdictBadge("SOUND"),
          ] }),
          new TableRow({ children: [
            cell("Sector-Relative Scoring", 3200, { bold: true }),
            cell("Sector-relative percentiles with small-sector fallback. Both sector-neutral and cross-sectional are accepted.", 4760),
            verdictBadge("STANDARD"),
          ] }),
          new TableRow({ children: [
            cell("Bank-Specific Logic", 3200, { bold: true }),
            cell("P/B + Earnings Yield for banks, ROE/ROA for quality. Industry standard per CFA curriculum.", 4760),
            verdictBadge("STRONG"),
          ] }),
          new TableRow({ children: [
            cell("Missing Data Handling", 3200, { bold: true }),
            cell("Weight redistribution to available metrics. No imputation. Coverage floor at 60%.", 4760),
            verdictBadge("SOUND"),
          ] }),
          new TableRow({ children: [
            cell("Value/Growth Trap Filters", 3200, { bold: true }),
            cell("Majority 2-of-3 logic (quality + momentum + revisions). Well-calibrated thresholds.", 4760),
            verdictBadge("SOUND"),
          ] }),
          new TableRow({ children: [
            cell("Momentum Construction", 3200, { bold: true }),
            cell("12-1 month canonical signal. 6-month added for regime diversification.", 4760),
            verdictBadge("STRONG"),
          ] }),
          new TableRow({ children: [
            cell("Winsorization", 3200, { bold: true }),
            cell("1st/99th percentile before ranking. Standard practice, though double-compression possible.", 4760),
            verdictBadge("STANDARD"),
          ] }),
          new TableRow({ children: [
            cell("Revisions Sparsity", 3200, { bold: true }),
            cell("~40% coverage via yfinance. Auto-disable below 30% is pragmatic but limits factor contribution.", 4760),
            verdictBadge("ACCEPTABLE"),
          ] }),
        ]
      }),

      spacer(),
      para("Overall Assessment: The model is well-grounded in peer-reviewed factor research and would be defensible in a CFA or institutional context. The methodology choices are transparent, documented, and internally consistent. Three areas for potential enhancement are noted in Section 8.", { italics: true, color: GRAY }),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 2. FACTOR SELECTION
      // ═══════════════════════════════════════════════════════
      heading1("2. Factor Selection & Academic Grounding"),

      para("The screener employs an 8-factor model. Below, each factor is mapped to its academic origin and institutional usage."),
      spacer(),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1400, 1000, 3360, 3600],
        rows: [
          new TableRow({ children: [
            headerCell("Factor", 1400),
            headerCell("Weight", 1000),
            headerCell("Academic Basis", 3360),
            headerCell("Institutional Precedent", 3600),
          ] }),
          new TableRow({ children: [
            cell("Valuation", 1400, { bold: true }),
            cell("22%", 1000, { align: AlignmentType.CENTER }),
            cell("Fama-French HML (1993); Lakonishok, Shleifer & Vishny (1994)", 3360),
            cell("Core factor in AQR Value, MSCI Enhanced Value, DFA. Uses EV/EBITDA, FCF Yield, Earnings Yield, EV/Sales.", 3600),
          ] }),
          new TableRow({ children: [
            cell("Quality", 1400, { bold: true }),
            cell("22%", 1000, { align: AlignmentType.CENTER }),
            cell("Fama-French RMW (2015); Novy-Marx (2013); Piotroski (2000); Sloan (1996)", 3360),
            cell("AQR Quality Minus Junk. Uses ROIC, Gross Profit/Assets, Piotroski F-Score, Accruals, Debt/Equity.", 3600),
          ] }),
          new TableRow({ children: [
            cell("Growth", 1400, { bold: true }),
            cell("13%", 1000, { align: AlignmentType.CENTER }),
            cell("Consensus estimate research (Chan, Karceski, Lakonishok 2003)", 3360),
            cell("Forward EPS growth, PEG ratio, revenue growth, sustainable growth. Blend of forward-looking and realized.", 3600),
          ] }),
          new TableRow({ children: [
            cell("Momentum", 1400, { bold: true }),
            cell("13%", 1000, { align: AlignmentType.CENTER }),
            cell("Jegadeesh & Titman (1993); Carhart (1997) 4-factor model", 3360),
            cell("12-1 month return is the canonical signal. 6-month return adds medium-term regime capture.", 3600),
          ] }),
          new TableRow({ children: [
            cell("Risk", 1400, { bold: true }),
            cell("10%", 1000, { align: AlignmentType.CENTER }),
            cell("Low-volatility anomaly (Baker, Bradley, Wurgler 2011); Frazzini & Pedersen BAB (2014)", 3360),
            cell("MSCI Minimum Volatility, Invesco S&P 500 Low Volatility. Uses annualized vol + market beta.", 3600),
          ] }),
          new TableRow({ children: [
            cell("Revisions", 1400, { bold: true }),
            cell("10%", 1000, { align: AlignmentType.CENTER }),
            cell("Post-earnings announcement drift (Bernard & Thomas 1989); analyst herding literature", 3360),
            cell("Earnings surprise + price target upside. Used by Seeking Alpha Quant, S&P Capital IQ factor models.", 3600),
          ] }),
          new TableRow({ children: [
            cell("Size", 1400, { bold: true }),
            cell("5%", 1000, { align: AlignmentType.CENTER }),
            cell("Fama-French SMB (1993); Banz (1981) small-firm effect", 3360),
            cell("Implemented as -log(market cap). Low weight appropriate: SMB premium weakest in large-cap S&P 500 universe.", 3600),
          ] }),
          new TableRow({ children: [
            cell("Investment", 1400, { bold: true }),
            cell("5%", 1000, { align: AlignmentType.CENTER }),
            cell("Fama-French CMA (2015); Titman, Wei & Xie (2004) asset growth anomaly", 3360),
            cell("YoY total asset growth (lower = better). Conservative firms outperform. Low weight appropriate.", 3600),
          ] }),
        ]
      }),

      spacer(),
      para("Assessment: ", { bold: true }),
      para("The factor selection directly maps to the Fama-French 5-factor model (Market, SMB, HML, RMW, CMA) augmented with Momentum (Carhart) and Revisions (post-earnings drift). This is a superset of the most widely accepted asset pricing frameworks. The 8-factor structure matches or exceeds the factor coverage of commercial products like MSCI FaCS, AQR style premia, and S&P factor indices."),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 3. METRIC COMPUTATION
      // ═══════════════════════════════════════════════════════
      heading1("3. Metric Computation Audit"),

      heading2("3.1 Valuation Metrics"),
      bullet("FCF Yield = (Operating Cash Flow \u2013 CapEx) / Enterprise Value. ", "FCF Yield (40% weight): "),
      subBullet("Uses FCF/EV (unlevered), which is the institutional standard for cross-sector comparability per Wall Street Prep and CFA Level II curriculum."),
      subBullet("Requires both OCF and CapEx to be non-null. This is conservative but correct\u2014assuming zero CapEx would overstate FCF for capital-intensive firms."),
      bullet("EV / EBITDA (25% weight). Capital-structure-neutral. Widely used across sell-side and buy-side. EV is recomputed from Market Cap + Debt \u2013 Cash when yfinance EV is missing, which is appropriate."),
      bullet("Earnings Yield (20% weight). Inverse P/E. Simple, transparent, well-understood."),
      bullet("EV/Sales (15% weight). Lowest weight reflects its weakness (ignores margins). Useful as a fallback for loss-making firms."),

      heading2("3.2 Quality Metrics"),
      bullet("ROIC = NOPAT / Invested Capital. ", "ROIC (30% weight): "),
      subBullet("NOPAT = EBIT \u00d7 (1 \u2013 effective tax rate), with tax rate clamped to [0%, 50%]. The 50% cap prevents negative or extreme tax rates from distorting NOPAT."),
      subBullet("Invested Capital = Equity + Debt \u2013 Excess Cash, where Excess Cash = max(0, Cash \u2013 2% of Revenue). The 2% operating cash buffer is a recognized adjustment to avoid inflating ROIC for cash-hoarding firms (e.g., AAPL, GOOG)."),
      bullet("Novy-Marx (2013) gross profitability factor. Measures operational profitability above COGS. Strong academic support as a quality predictor.", "Gross Profit / Assets (25% weight): "),
      bullet("Standard leverage metric. Correctly excludes negative-equity firms (e.g., buyback-driven negative book value).", "Debt/Equity (20% weight): "),
      bullet("9 binary signals covering profitability, cash flow quality, leverage change, liquidity, dilution, efficiency, and margins. Requires \u22656 testable signals. Not normalized by number of testable signals\u2014this is appropriate per Piotroski (2000) original specification.", "Piotroski F-Score (15% weight): "),
      bullet("Sloan (1996) earnings quality. (Net Income \u2013 OCF) / Total Assets. Lower is better. Well-documented predictor of earnings persistence.", "Accruals (10% weight): "),

      heading2("3.3 Momentum Metrics"),
      bullet("Canonical Jegadeesh-Titman (1993) momentum signal. Excludes most recent month to avoid short-term reversal. This is the exact specification used by Kenneth French\u2019s data library and AQR\u2019s momentum factors.", "12-1 Month Return (50%): "),
      bullet("Medium-term momentum complement. Equal weighting with 12-1M provides signal diversification across lookback horizons. Research shows 6-month momentum produces Sharpe ratios around 1.0.", "6-Month Return (50%): "),

      heading2("3.4 Risk Metrics"),
      bullet("Annualized standard deviation of daily log returns (\u00d7\u221a252). Requires \u2265200 trading days. Standard computation per CFA Level II.", "Volatility (60%): "),
      bullet("Cov(Stock, Market) / Var(Market) using date-aligned daily returns vs. S&P 500. Sample variance (ddof=1). Standard CAPM beta.", "Beta (40%): "),

      heading2("3.5 Revisions Metrics"),
      bullet("Median of last 4 quarterly surprises: (Actual \u2013 Estimate) / max(|Estimate|, $0.10). Requires \u22652 valid quarters. The $0.10 denominator floor prevents near-zero estimates from producing extreme ratios.", "Analyst Surprise (65%): "),
      bullet("(Target Mean \u2013 Current Price) / Current Price, clamped to [-50%, +100%]. Requires \u22653 covering analysts. The analyst count threshold and clamping are appropriate given well-documented sell-side optimism bias.", "Price Target Upside (35%): "),

      heading2("3.6 Size & Investment"),
      bullet("-log(Market Cap). Higher value = smaller company. At 5% weight, this is appropriately conservative for an S&P 500 universe where the small-cap premium is empirically weakest.", "Size: "),
      bullet("YoY total asset growth (lower = better). Fama-French CMA proxy. Conservative investment predicts higher future returns per Titman, Wei & Xie (2004). Appropriately low weight.", "Investment: "),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 4. SCORING METHODOLOGY
      // ═══════════════════════════════════════════════════════
      heading1("4. Scoring Methodology"),

      heading2("4.1 Sector-Relative Percentile Ranking"),
      para("All raw metrics are ranked within their GICS sector using pandas rank(pct=True), producing percentile scores on a 0\u2013100 scale. Metrics where lower is better (e.g., EV/EBITDA, Debt/Equity, Volatility) are inverted (100 \u2013 rank)."),
      spacer(),
      bullet("Ensures a cheap Utility stock is not penalized for having a higher EV/EBITDA than a cheap Tech stock. Both sector-neutral and cross-sectional approaches are used institutionally; AQR uses sector neutrality in long-short funds, while Dimensional uses bands.", "Rationale: "),
      bullet("If a sector has fewer than 10 stocks with valid data for a metric, the model falls back to universe-wide percentile ranking. This prevents inflated/deflated scores in small sectors.", "Small-Sector Fallback: "),

      heading2("4.2 Category Score Aggregation"),
      para("Each category score is a weighted average of its constituent metric percentiles. The critical design choice: missing-data-adaptive weighting."),
      spacer(),
      bullet("If a metric is NaN for a given stock, it contributes zero to both the numerator and denominator of the weighted average. The remaining metrics are effectively upweighted proportionally.", "Missing Data: "),
      bullet("This approach is preferable to mean imputation (which biases toward the mean) and preferable to dropping the stock entirely (which reduces coverage). It is equivalent to pairwise deletion, which is the default method in prevailing factor analysis software.", "Institutional Comparison: "),

      heading2("4.3 Composite Score"),
      para("The composite score is a weighted average of the 8 category scores, then percentile-ranked cross-sectionally. A stock with Composite = 97.5 is in the 97.5th percentile of the full S&P 500 universe."),
      spacer(),
      bullet("The final percentile ranking normalizes across runs and time periods. This is standard practice for factor index construction (MSCI, S&P)."),

      heading2("4.4 Piotroski Conditional Weighting"),
      para("A notable methodological refinement: for stocks with low valuation scores (below 50th percentile), the Piotroski F-Score weight is halved and the freed weight is redistributed to ROIC and Gross Profit/Assets."),
      spacer(),
      bullet("Piotroski (2000) originally demonstrated F-Score\u2019s predictive power primarily in value stocks. For expensive (growth) stocks, F-Score has weak discriminating power. This conditional weighting is an appropriate adaptation supported by the original research.", "Academic Support: "),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 5. BANK-SPECIFIC TREATMENT
      // ═══════════════════════════════════════════════════════
      heading1("5. Bank-Specific Treatment"),

      para("The model correctly identifies bank-like financial institutions (banks, insurers, credit companies) and applies a fundamentally different metric set. This is critical and often overlooked in simpler screeners."),
      spacer(),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2400, 3480, 3480],
        rows: [
          new TableRow({ children: [
            headerCell("Category", 2400),
            headerCell("Non-Bank Metrics", 3480),
            headerCell("Bank Metrics", 3480),
          ] }),
          new TableRow({ children: [
            cell("Valuation", 2400, { bold: true }),
            cell("EV/EBITDA (25%), FCF Yield (40%), Earnings Yield (20%), EV/Sales (15%)", 3480),
            cell("P/B Ratio (60%), Earnings Yield (40%)", 3480),
          ] }),
          new TableRow({ children: [
            cell("Quality", 2400, { bold: true }),
            cell("ROIC (30%), GP/Assets (25%), D/E (20%), F-Score (15%), Accruals (10%)", 3480),
            cell("ROE (35%), ROA (25%), Equity Ratio (15%), F-Score (15%), Accruals (10%)", 3480),
          ] }),
          new TableRow({ children: [
            cell("Growth", 2400, { bold: true }),
            cell("Same for both", 3480, { color: GRAY }),
            cell("Same for both", 3480, { color: GRAY }),
          ] }),
          new TableRow({ children: [
            cell("Momentum/Risk/Rev", 2400, { bold: true }),
            cell("Same for both", 3480, { color: GRAY }),
            cell("Same for both", 3480, { color: GRAY }),
          ] }),
        ]
      }),

      spacer(),
      bullet("EV/EBITDA is meaningless for banks (deposits inflate enterprise value; EBITDA excludes net interest income). FCF Yield is unreliable (banks lack traditional CapEx). These are correctly zeroed out.", "Why EV-based metrics fail for banks: "),
      bullet("P/B is the primary bank valuation metric per CFA Institute Level II (Market-Based Valuation reading) and Wall Street Oasis investment banking interview standards. ROE measures profit per dollar of equity capital; ROA measures bank efficiency. Equity Ratio captures capital adequacy.", "Why P/B, ROE, ROA are correct: "),
      bullet("Payment processors (V, MA, PYPL), exchanges (CME, ICE, NDAQ), and data providers (MSCI, SPGI, FDS) are explicitly excluded from bank treatment via an override list. This is a detail most screeners miss.", "Non-bank financial override: "),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 6. TRAP DETECTION
      // ═══════════════════════════════════════════════════════
      heading1("6. Value & Growth Trap Detection"),

      heading2("6.1 Value Trap Filter"),
      para("A value trap is a stock that appears cheap on valuation metrics but has deteriorating fundamentals. The model flags stocks that breach at least 2 of 3 dimensions:"),
      spacer(),
      bullet("Quality score below the 30th percentile"),
      bullet("Momentum score below the 30th percentile"),
      bullet("Revisions score below the 30th percentile"),
      spacer(),
      para("The 2-of-3 majority logic is a deliberate calibration choice. Simple OR logic (any 1 breach) flagged approximately 60% of the universe, which is too aggressive. Majority logic catches stocks with genuinely broad weakness while tolerating a single weak dimension. This is consistent with the Quality-Value-Momentum (QVM) approach documented in institutional research."),

      heading2("6.2 Growth Trap Filter"),
      para("A growth trap is a high-growth stock with weak underlying fundamentals. The model flags stocks above the 70th percentile in growth that also breach at least 2 of 3:"),
      spacer(),
      bullet("Growth score above the 70th percentile (high growth = suspect)"),
      bullet("Quality score below the 35th percentile"),
      bullet("Revisions score below the 35th percentile"),
      spacer(),
      para("This mirrors the value trap logic for the opposite scenario and catches \u201Cgrowth at any price\u201D stocks that institutional investors typically avoid."),

      heading2("6.3 Missing Data in Trap Flags"),
      bullet("NaN in any category score counts as FALSE (not flagged) for that layer. Missing data does not trigger a trap flag\u2014only confirmed weakness does.", "Conservative default: "),
      bullet("Stocks with NaN in any of the three dimensions are separately flagged with Insufficient_Data_Flag. This preserves information about data completeness without conflating it with fundamental weakness.", "Transparency: "),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 7. DATA QUALITY
      // ═══════════════════════════════════════════════════════
      heading1("7. Data Quality & Robustness"),

      heading2("7.1 Winsorization"),
      bullet("1st and 99th percentile (scipy.stats.mstats.winsorize). Applied to all raw metrics before percentile ranking. Requires at least 10 observations per metric."),
      bullet("The 1st/99th percentile range is the institutional standard for financial data per academic literature and SAS best practices. This removes approximately 2% of data points in each tail."),
      bullet("One potential concern: winsorizing before percentile ranking can cause double-compression of extremes. In practice, this is a minor effect since percentile ranking already handles outliers by construction.", "Double-compression note: "),

      heading2("7.2 Data Coverage"),
      bullet("Stocks with fewer than 60% of metrics available are excluded from the universe. This prevents poorly-covered stocks from receiving misleading scores."),
      bullet("Individual metrics missing more than 70% of values are automatically zero-weighted. This prevents a single sparse metric from distorting category scores."),

      heading2("7.3 Revisions Auto-Disable"),
      para("If analyst revision metrics (surprise + price target upside) have less than 30% coverage across the universe, the entire revisions factor weight is set to zero and proportionally redistributed to the remaining factors. This is a pragmatic adaptation to yfinance data limitations and ensures the model degrades gracefully."),

      heading2("7.4 Forward EPS Growth Guardrails"),
      bullet("Growth rate clamped to [-75%, +150%] to guard against spurious extremes from mixing GAAP trailing EPS with normalized forward consensus estimates."),
      bullet("Denominator floored at $1.00 to prevent near-zero trailing EPS from producing extreme growth ratios."),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 8. AREAS FOR ENHANCEMENT
      // ═══════════════════════════════════════════════════════
      heading1("8. Areas for Enhancement"),

      para("While the model is defensible as-is, three areas could strengthen it further for institutional scrutiny:"),
      spacer(),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [600, 2600, 4260, 1900],
        rows: [
          new TableRow({ children: [
            headerCell("#", 600),
            headerCell("Area", 2600),
            headerCell("Detail", 4260),
            headerCell("Priority", 1900),
          ] }),
          new TableRow({ children: [
            cell("1", 600, { align: AlignmentType.CENTER }),
            cell("Revisions Data Source", 2600, { bold: true }),
            cell("yfinance provides ~40% coverage for analyst surprise and price target data. Integrating I/B/E/S data (via FactSet or Refinitiv) would bring coverage to 90%+ and enable EPS forecast revision as a third revisions metric.", 4260),
            cell("High", 1900, { align: AlignmentType.CENTER, color: RED, bold: true }),
          ] }),
          new TableRow({ children: [
            cell("2", 600, { align: AlignmentType.CENTER }),
            cell("Turnover & Transaction Costs", 2600, { bold: true }),
            cell("The model does not penalize high-turnover signals. Momentum and revisions factors can generate significant quarterly turnover. Adding a turnover decay or transaction cost adjustment would improve the real-world implementability.", 4260),
            cell("Medium", 1900, { align: AlignmentType.CENTER, color: AMBER, bold: true }),
          ] }),
          new TableRow({ children: [
            cell("3", 600, { align: AlignmentType.CENTER }),
            cell("Out-of-Sample Backtest", 2600, { bold: true }),
            cell("While the backtesting framework exists in the codebase, running a published backtest with Sharpe ratio, max drawdown, and factor attribution would substantially strengthen the model\u2019s credibility for institutional audiences.", 4260),
            cell("Medium", 1900, { align: AlignmentType.CENTER, color: AMBER, bold: true }),
          ] }),
        ]
      }),

      spacer(),
      para("None of these are methodology flaws\u2014they are enhancements that would elevate the model from a strong academic screener to an institutional-grade factor model.", { italics: true, color: GRAY }),

      new Paragraph({ children: [new PageBreak()] }),

      // ═══════════════════════════════════════════════════════
      // 9. CONCLUSION
      // ═══════════════════════════════════════════════════════
      heading1("9. Conclusion"),

      para("The Multi-Factor Equity Screener demonstrates a methodologically rigorous approach to stock scoring that is well-aligned with both academic factor research and institutional quant practices. The key strengths of the methodology:"),
      spacer(),
      bullet("All eight factors have direct academic citations and institutional precedent (Fama-French, Jegadeesh-Titman, Novy-Marx, Piotroski, Sloan, Carhart)."),
      bullet("The sector-relative percentile ranking prevents cross-sector valuation distortions while the small-sector fallback ensures robustness."),
      bullet("Bank-specific treatment correctly addresses the fundamental unsuitability of EV-based and FCF-based metrics for financial institutions."),
      bullet("Missing-data-adaptive weighting is superior to mean imputation and avoids coverage loss from listwise deletion."),
      bullet("Value and growth trap filters use well-calibrated majority logic that avoids over-flagging while catching multi-dimensional weakness."),
      bullet("The Piotroski conditional weighting is a sophisticated refinement supported by the original research."),
      spacer(),
      para("A CFA charterholder or hedge fund analyst reviewing this methodology would find it defensible, transparent, and consistent with established factor investing principles. The model appropriately balances academic rigor with practical data constraints."),

      spacer(), spacer(),
      accentBar(),
      spacer(),
      para("References", { bold: true, size: 24, color: ACCENT }),
      para("Fama, E.F. and French, K.R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. Journal of Financial Economics, 33(1), 3\u201356.", { size: 20, color: GRAY }),
      para("Fama, E.F. and French, K.R. (2015). A Five-Factor Asset Pricing Model. Journal of Financial Economics, 116(1), 1\u201322.", { size: 20, color: GRAY }),
      para("Jegadeesh, N. and Titman, S. (1993). Returns to Buying Winners and Selling Losers. Journal of Finance, 48(1), 65\u201391.", { size: 20, color: GRAY }),
      para("Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium. Journal of Financial Economics, 108(1), 1\u201328.", { size: 20, color: GRAY }),
      para("Piotroski, J.D. (2000). Value Investing: The Use of Historical Financial Statement Information. Journal of Accounting Research, 38, 1\u201341.", { size: 20, color: GRAY }),
      para("Sloan, R.G. (1996). Do Stock Prices Fully Reflect Information in Accruals and Cash Flows? The Accounting Review, 71(3), 289\u2013315.", { size: 20, color: GRAY }),
      para("Carhart, M.M. (1997). On Persistence in Mutual Fund Performance. Journal of Finance, 52(1), 57\u201382.", { size: 20, color: GRAY }),
      para("Titman, S., Wei, K.C.J. and Xie, F. (2004). Capital Investments and Stock Returns. Journal of Financial and Quantitative Analysis, 39(4), 677\u2013700.", { size: 20, color: GRAY }),
      para("Baker, M., Bradley, B. and Wurgler, J. (2011). Benchmarks as Limits to Arbitrage. Journal of Financial Economics, 101(1), 1\u201317.", { size: 20, color: GRAY }),

    ]
  }]
});

// ── Write file ──
Packer.toBuffer(doc).then(buffer => {
  const outPath = "/sessions/admiring-lucid-babbage/mnt/Screener-1/Methodology_Audit_Report.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Written:", outPath, `(${(buffer.length / 1024).toFixed(0)} KB)`);
});
