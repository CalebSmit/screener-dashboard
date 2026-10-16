"""The public research page (``research.html``): what the research engine found, for anyone.

WHY THIS EXISTS - 2026-10-15, owner: "people would like to see that". The research engine
(``v2_experiments.py``) tests the screener's weighting against 26 pre-registered alternatives on
2012-2026 history rebuilt from the filings companies had published at the time. Its answer - the
ranking has not been shown to predict returns - belongs on the public site next to the live track
record, which on its own looks better than the evidence. This module turns the engine's committed
ledger into one static page in the dashboard's own look.

**Generated** (CLAUDE.md rule 10): ``python v2_experiments.py report`` writes it whenever the
results change. Never hand-edit ``research.html``. Every number on it comes from
``research/experiments/ledger.jsonl`` through this module; ``tests/test_research_page.py`` rebuilds
the page and fails if the committed one differs, if a number does not match the ledger, or if the
page uses advice language. Never imported by production code.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "research.html"

LABELS = {
    "equal-categories": "Equal weight on every category", "only-valuation": "Valuation alone",
    "only-quality": "Quality alone", "only-growth": "Growth alone", "only-momentum": "Momentum alone",
    "only-risk": "Low risk alone", "only-size": "Size alone", "only-investment": "Investment alone",
    "drop-valuation": "Without Valuation", "drop-quality": "Without Quality", "drop-growth": "Without Growth",
    "drop-momentum": "Without Momentum", "drop-risk": "Without Risk", "drop-size": "Without Size",
    "drop-investment": "Without Investment", "magic-formula": "Greenblatt's Magic Formula",
    "piotroski-only": "Piotroski F-score alone", "profitability-and-value": "Profitability and value",
    "fcf-yield-only": "Free-cash-flow yield alone", "momentum-12-1-only": "12-1 month return alone",
    "low-volatility-only": "Low volatility alone",
    "orthogonalised-categories": "Overlap removed between categories",
    "whole-market-percentiles": "Ranked across the market, not within sectors",
    "value-and-momentum": "Value and momentum", "52-week-high": "Nearness to the 52-week high",
    "profitability-and-low-volatility": "Profitability and low volatility",
}
REPO = "https://github.com/CalebSmit/screener-dashboard"
LIM = 0.06            # the chart's axis, in IC: wider than every difference and every band so far


def label(exp_id: str) -> str:
    return LABELS.get(exp_id, exp_id.replace("-", " ").capitalize())


def num(x: float, d: int = 3) -> str:
    """A signed figure with a real minus sign; one that rounds to zero carries no sign."""
    if round(abs(x), d) == 0:
        return f"{0:.{d}f}"
    return ("+" if x > 0 else "&minus;") + f"{abs(x):.{d}f}"


def pct(x: float, d: int = 2) -> str:
    return num(x * 100, d) + "%"


def _pos(x: float) -> float:
    return max(0.0, min(100.0, (x + LIM) / (2 * LIM) * 100))


def months_needed(ic: float, ic_sd: float, z: float) -> int:
    """Months of independent record for a true mean IC ``ic`` to show at two-sided ``z`` with 80%
    power, when the monthly IC has standard deviation ``ic_sd``."""
    return int(round(((z + 0.84) * ic_sd / ic) ** 2))


def build(reg: list, latest: dict, summary: dict, detectable, evaluations: list) -> str:
    """The page, from the registry, the latest ledger row per id, the engine's page summary,
    the engine's ``detectable(se, n)`` and the registered evaluations."""
    n = summary["registered"]
    base = latest["baseline"]["summary"]
    rows = []
    for e in reg:
        r = latest.get(e["id"])
        if e["id"] == "baseline" or e.get("void") or not r:
            continue
        s = r["summary"]
        if "presample" not in s or "ic_vs_baseline" not in s["presample"]:
            continue
        rows.append({"id": e["id"], "p": s["presample"]["ic_vs_baseline"], "h": s["holdout"]["ic_vs_baseline"],
                     "source": e["source"].split(";")[0].split(" - ")[0].rstrip(".")})
    rows.sort(key=lambda r: -r["p"]["mean"])
    pre_ic, post_ic = summary["baseline_ic"]["presample"], summary["baseline_ic"]["all"]
    ic_sd = base["all"]["ic"]["se"] * math.sqrt(base["all"]["ic"]["n"])
    mde_p = detectable(sorted(r["p"]["se"] for r in rows)[len(rows) // 2], n)
    mde_h = detectable(sorted(r["h"]["se"] for r in rows)[len(rows) // 2], n)

    dots = []
    for r in rows:
        p = r["p"]
        d = detectable(p["se"], n)
        lo, hi = p["mean"] - 1.96 * p["se"], p["mean"] + 1.96 * p["se"]
        dots.append(
            f'<div class="dot-row"><div class="dot-name">{html.escape(label(r["id"]))}</div>'
            f'<div class="track" role="img" aria-label="{html.escape(label(r["id"]))}: {num(p["mean"])}, t {p["t"]:+.1f}">'
            f'<span class="see" style="left:{_pos(-d):.2f}%;width:{_pos(d) - _pos(-d):.2f}%"></span>'
            f'<span class="zero"></span>'
            f'<span class="ci" style="left:{_pos(lo):.2f}%;width:{_pos(hi) - _pos(lo):.2f}%"></span>'
            f'<span class="dot" style="left:{_pos(p["mean"]):.2f}%"></span></div>'
            f'<div class="dot-val num">{num(p["mean"])} <span class="t">t {p["t"]:+.1f}</span></div></div>')

    table = "".join(
        f'<tr><td>{html.escape(label(r["id"]))}<div class="src">{html.escape(r["source"])}</div></td>'
        f'<td class="num">{num(r["h"]["mean"])} <span class="t">({r["h"]["t"]:+.1f})</span></td>'
        f'<td class="num">{num(r["p"]["mean"])} <span class="t">({r["p"]["t"]:+.1f})</span></td></tr>'
        for r in sorted(rows, key=lambda r: label(r["id"])))

    bot_html = ""
    ev = next((e for e in evaluations if e["id"] == "bot-rule-v1.1"), None)
    if ev and latest.get(ev["id"]):
        b = latest[ev["id"]]["summary"]
        brows = "".join(
            f'<tr><td>{lab}</td><td class="num">{pct(b[k]["ret"]["mean"])}</td>'
            f'<td class="num">{pct(b[k]["excess"]["mean"])} <span class="t">(t {b[k]["excess"]["t"]:+.2f})</span></td>'
            f'<td class="num">{b[k]["turnover_a_year"] * 100:.0f}%</td></tr>'
            for k, lab in (("presample", "2012&ndash;2019"), ("discovery", "2020 &ndash; mid-2023"),
                           ("holdout", "mid-2023 &ndash; 2026")) if k in b)
        rule = ev["rule"]
        bot_html = f"""
  <section>
    <h2>What a simple trading rule on the ranking would have done</h2>
    <p>A rule that holds the top {rule['target_count']}, keeps each until it falls past rank {rule['sell_rank']}, skip names flagged as value or growth traps, at most {rule['max_per_sector']} in a sector, {rule['cost_bps']} basis points of cost a side, rebalanced monthly:</p>
    <div class="tw"><table>
      <thead><tr><th>Period</th><th class="num">Return a month</th><th class="num">Against the equal-weighted S&amp;P&nbsp;500</th><th class="num">Turnover a year</th></tr></thead>
      <tbody>{brows}</tbody>
    </table></div>
    <p>It trailed an equal-weighted index of the same stocks in every period, by most in the growth-led years of 2020 and 2021. Read any live record of the ranking next to an equal-weight fund.</p>
  </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Has it been tested? · Multi-Factor Screener</title>
<meta name="description" content="The screener's weighting and 26 pre-registered alternatives, tested on 2012-2026 S&amp;P 500 history rebuilt from the filings published at the time.">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Multi-Factor Screener">
<meta property="og:title" content="Has it been tested? The screener on 2012-2026 history">
<meta property="og:description" content="The ranking and 26 pre-registered alternatives, tested on S&amp;P 500 history rebuilt from the filings published at the time.">
<meta property="og:url" content="https://calebsmit.github.io/screener-dashboard/research.html">
<meta property="og:image" content="https://calebsmit.github.io/screener-dashboard/og-card.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Generated by research_page.py - do not edit. Tokens are the dashboard's and the lab's. */
:root {{
    color-scheme: dark;
    --bg-deep: #0d0d0d; --bg-card: #1a1a19; --bg-elevated: #222221; --border: #262625; --border-bright: #343433;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8e8c86;
    --accent: #3987e5; --accent-text: #5598e7; --accent-glow: rgba(57,135,229,.12); --band: #252524;
    --font-body: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', monospace;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--bg-deep); color: var(--text-secondary); font-family: var(--font-body);
       font-size: 15px; line-height: 1.6; -webkit-font-smoothing: antialiased; }}
a {{ color: var(--accent-text); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }}
.wrap {{ max-width: 960px; margin: 0 auto; padding: 0 16px 72px; }}
@media (min-width: 720px) {{ .wrap {{ padding: 0 32px 96px; }} }}
.bar {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; min-height: 56px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
.mark {{ display: inline-flex; align-items: center; gap: 10px; color: var(--text-primary); font-weight: 600; letter-spacing: -.01em; text-decoration: none; }}
.bmk {{ width: 24px; height: 24px; flex: none; }}
.bm-bg {{ fill: var(--bg-elevated); stroke: var(--border-bright); }} .bm-1 {{ fill: var(--accent); opacity: .5; }} .bm-2 {{ fill: var(--accent); opacity: .75; }} .bm-3 {{ fill: var(--accent); }}
.bar nav {{ display: flex; gap: 4px; font-size: 13.5px; flex-wrap: wrap; }}
.bar nav a {{ color: var(--text-secondary); padding: 6px 10px; border-radius: 8px; }}
.bar nav a:hover {{ color: var(--text-primary); background: var(--bg-card); }}
.bar nav a[aria-current=page] {{ color: var(--text-primary); background: var(--bg-elevated); }}
@media (max-width: 600px) {{ .bar {{ padding: 10px 0 8px; gap: 6px; }} .bar nav {{ margin-left: -10px; }} }}
.lede {{ padding: 40px 0 4px; max-width: 68ch; }}
.eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); }}
h1 {{ margin: 10px 0 14px; color: var(--text-primary); font-size: clamp(28px, 5vw, 40px); line-height: 1.12; font-weight: 700; letter-spacing: -.025em; text-wrap: balance; }}
h2 {{ margin: 0 0 10px; color: var(--text-primary); font-size: 21px; line-height: 1.25; letter-spacing: -.015em; text-wrap: balance; }}
p {{ margin: 0 0 12px; max-width: 68ch; }}
section {{ margin-top: 40px; }}
.fine {{ font-size: 13px; color: var(--text-muted); }}
.num {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.t {{ color: var(--text-muted); font-size: .85em; }}
strong {{ color: var(--text-primary); }}
.answer {{ display: grid; gap: 10px; margin: 24px 0 0; padding: 0; list-style: none; max-width: 76ch; }}
.answer li {{ padding: 14px 16px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; }}
.steps {{ list-style: none; margin: 14px 0; padding: 0; display: grid; gap: 8px; max-width: 76ch; }}
.steps li {{ display: grid; grid-template-columns: 2.2em minmax(0, 1fr); gap: 10px; }}
.steps .num {{ color: var(--accent-text); }}
.chart {{ margin: 16px 0 8px; padding: 16px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; }}
.dot-row, .axis {{ display: grid; grid-template-columns: minmax(0, 15em) minmax(0, 1fr) 7.5em; gap: 12px; align-items: center; }}
.dot-row {{ padding-block: 3px; }}
.dot-name {{ font-size: 13.5px; min-width: 0; overflow-wrap: anywhere; color: var(--text-secondary); }}
.track {{ position: relative; height: 18px; }}
.track span {{ position: absolute; top: 50%; transform: translateY(-50%); }}
.see {{ height: 18px; background: var(--band); border-radius: 3px; }}
.zero {{ left: 50%; width: 1px; height: 18px; background: var(--text-muted); opacity: .7; }}
.ci {{ height: 2px; background: var(--accent); opacity: .6; }}
.dot {{ width: 9px; height: 9px; margin-left: -4.5px; border-radius: 50%; background: var(--accent-text); }}
.dot-val {{ font-size: 12.5px; text-align: right; }}
.axis {{ margin-top: 6px; }}
.ticks {{ display: flex; justify-content: space-between; font: 11.5px var(--font-mono); color: var(--text-muted); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12.5px; color: var(--text-muted); margin-top: 10px; }}
.legend i {{ display: inline-block; vertical-align: middle; margin-right: 6px; }}
.tw {{ overflow-x: auto; margin: 12px 0; max-width: 100%; }}
details {{ max-width: 100%; min-width: 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{ text-align: left; vertical-align: bottom; font-size: 11.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); border-bottom: 1px solid var(--border-bright); padding: 8px 12px 8px 0; }}
td {{ border-bottom: 1px solid var(--border); padding: 9px 12px 9px 0; vertical-align: top; }}
th.num, td.num {{ text-align: right; }}
.src {{ font-size: 12px; color: var(--text-muted); }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 14px 0; }}
.stat {{ padding: 14px 16px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; }}
.stat .v {{ font: 500 26px/1.2 var(--font-mono); color: var(--text-primary); font-variant-numeric: tabular-nums; }}
.stat .k {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
.claims {{ display: grid; gap: 12px; grid-template-columns: 1fr; }}
@media (min-width: 720px) {{ .claims {{ grid-template-columns: 1fr 1fr; }} }}
.claim {{ padding: 16px 18px; background: var(--bg-elevated); border: 1px solid var(--border-bright); border-radius: 10px; }}
.claim h3 {{ margin: 0 0 6px; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); }}
details summary {{ cursor: pointer; color: var(--accent-text); font-size: 14px; margin-top: 8px; }}
.foot .num {{ white-space: normal; overflow-wrap: anywhere; }}
.foot {{ margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 13px; color: var(--text-muted); }}
@media (max-width: 599px) {{
  td .t {{ display: block; }}
  .dot-row, .axis {{ grid-template-columns: minmax(0, 1fr) 6.2em; }}
  .dot-name {{ grid-column: 1 / -1; margin-bottom: -2px; }}
  .axis > div:first-child {{ display: none; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <a class="mark" href="index.html" aria-label="Multi-Factor Screener, the rankings"><svg class="bmk" viewBox="0 0 32 32" aria-hidden="true"><rect x=".5" y=".5" width="31" height="31" rx="8" class="bm-bg"/><rect x="7" y="17" width="4" height="8" rx="1.5" class="bm-1"/><rect x="14" y="12" width="4" height="13" rx="1.5" class="bm-2"/><rect x="21" y="7" width="4" height="18" rx="1.5" class="bm-3"/></svg>Multi-Factor Screener</a>
    <nav aria-label="Site"><a href="index.html">Rankings</a><a href="research.html" aria-current="page">Has it been tested?</a><a href="lab.html">Factor Lab</a></nav>
  </div>

  <header class="lede">
    <div class="eyebrow">The screener, tested on its own history</div>
    <h1>Does the ranking predict returns?</h1>
    <p>Before trusting a ranking, ask whether it would have worked. This page tests the screener's weighting, and {summary['alternatives']} alternatives from the investing literature, on fifteen years of S&amp;P&nbsp;500 history rebuilt from the financial statements companies had actually published at each date.</p>
  </header>

  <ul class="answer">
    <li><strong>Not measurably.</strong> The ranking's average monthly rank correlation with the next month's return was <span class="num">{num(pre_ic)}</span> over 2012&ndash;2019 and <span class="num">{num(post_ic)}</span> over 2020&ndash;2026. Both are zero within the noise.</li>
    <li><strong>No alternative did better.</strong> {"None" if summary['better'] == 0 else summary['better']} of the {summary['alternatives']} alternative weightings cleared the bar set before the test, in either period.</li>
    <li><strong>The test can see a signal when there is one.</strong> Fed information it could not have had at the time (today's filings, for past months), the same test finds a strong one.</li>
  </ul>

  <section>
    <h2>How it was tested</h2>
    <p>Test enough ideas on one history and some look excellent by luck. The test was set up to prevent that:</p>
    <ol class="steps">
      <li><span class="num">1</span><span>Every idea, its source and its predicted result were written down and published in the project's history <em>before</em> it was run.</span></li>
      <li><span class="num">2</span><span>Ideas were compared with the current weighting month by month, on the same stocks, and judged on months they were never tuned on.</span></li>
      <li><span class="num">3</span><span>The years 2012&ndash;2019 were set aside as an untouched test, with the rule for passing it, before any of that data was built.</span></li>
      <li><span class="num">4</span><span>A result counts only if it clears a bar that allows for how many ideas were tried: p&nbsp;&lt;&nbsp;0.05&nbsp;/&nbsp;{n} and a t-statistic of at least 3, the threshold Harvey, Liu &amp; Zhu (2016) propose for a new factor. Every run is kept, including runs a data correction replaced.</span></li>
    </ol>
  </section>

  <section>
    <h2>Each alternative against the current weighting, 2012&ndash;2019</h2>
    <p class="fine">Difference in average monthly rank correlation. The shaded band is what the test could reliably detect; a dot inside it cannot be told apart from zero.</p>
    <div class="chart" role="figure" aria-label="Each alternative's difference from the current weighting, with its interval and the detectable band">
      {''.join(dots)}
      <div class="axis"><div></div><div class="ticks"><span>{num(-LIM, 2)}</span><span>{num(-LIM / 2, 2)}</span><span>0</span><span>{num(LIM / 2, 2)}</span><span>{num(LIM, 2)}</span></div><div></div></div>
      <div class="legend">
        <span><i style="width:9px;height:9px;border-radius:50%;background:var(--accent-text)"></i>difference</span>
        <span><i style="width:18px;height:2px;background:var(--accent);opacity:.6"></i>95% interval</span>
        <span><i style="width:18px;height:10px;background:var(--band);border-radius:2px"></i>detectable</span>
      </div>
    </div>
    <p>Size alone looks best, but its lead follows a gap in the data rather than the market. The test can only include companies that are still in the index today, and among 2012's members the small ones that survived are disproportionately the ones that did well. Size's advantage is largest in the months with the most members missing and disappears where coverage is fullest.</p>
    <details><summary>Every alternative, both periods</summary>
      <div class="tw"><table>
        <thead><tr><th>Alternative</th><th class="num">Mid-2023&ndash;2026 (t)</th><th class="num">2012&ndash;2019 (t)</th></tr></thead>
        <tbody>{table}</tbody>
      </table></div>
    </details>
  </section>
{bot_html}
  <section>
    <h2>How long would it take to prove an edge?</h2>
    <div class="stats">
      <div class="stat"><div class="v">{months_needed(0.05, ic_sd, 1.96)}</div><div class="k">months of record to show a monthly correlation of 0.05 at the usual 5% level</div></div>
      <div class="stat"><div class="v">{months_needed(0.05, ic_sd, 3.0)}</div><div class="k">months for the same with a t-statistic of 3</div></div>
      <div class="stat"><div class="v">{months_needed(0.03, ic_sd, 3.0)}</div><div class="k">months to show 0.03 with a t-statistic of 3</div></div>
    </div>
    <p>The ranking's monthly correlation swings by about <span class="num">{ic_sd:.2f}</span> from one month to the next, so any monthly ranking of large companies needs years of record before a realistic edge can be told apart from luck. That applies to this screener's live track record too. Here, the smallest difference the test could reliably see was about <span class="num">{mde_p:.3f}</span> over 2012&ndash;2019 and <span class="num">{mde_h:.3f}</span> over the held-out months.</p>
  </section>

  <section>
    <h2>What this means</h2>
    <div class="claims">
      <div class="claim"><h3>Supported</h3><p>Every factor the screener uses has a published source, and the <a href="lab.html">Factor Lab</a> reruns the studies behind its categories on public data. Every score on the dashboard can be rebuilt from its inputs. The testing is pre-registered and every run is kept.</p></div>
      <div class="claim"><h3>Not supported</h3><p>That the ranking beats the market, or that its top names will do better than the rest. On public data it has not been shown to predict returns, and the factor premiums it draws on were mostly weaker after their papers were published and weaker again among large companies.</p></div>
    </div>
    <p style="margin-top:14px">So the screener is a structured, checkable way to look at a company across eight dimensions. It is not a forecast.</p>
  </section>

  <section>
    <h2>Limits of the test</h2>
    <ul>
      <li>It cannot include the analyst-based inputs (estimate revisions and forward earnings growth, about a sixth of the live weighting): no free history of them exists.</li>
      <li>Companies that have since left the index are missing, which matters most in the earliest years.</li>
      <li>Large companies only, rebalanced monthly, with costs approximated and no taxes.</li>
    </ul>
  </section>

  <div class="foot">
    Generated from the screener's <a href="{REPO}/tree/main/research/experiments">research log</a> (data version <span class="num">{html.escape(str(summary.get('data_hash')))}</span>); the write-up, the registry of every prediction and every run are in the project's <a href="{REPO}/blob/main/research/2026-10-15-research-engine-first-batch.md">public repository</a>.
    This is research on a screening method for study and teaching. It is not investment advice.
  </div>
</div>
</body>
</html>
"""
