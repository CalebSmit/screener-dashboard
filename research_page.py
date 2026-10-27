"""The public research page (``research.html``): what the research engine found, for anyone.

WHY THIS EXISTS - 2026-10-15, owner: "people would like to see that". The research engine
(``v2_experiments.py``) tests the screener's weighting against 26 pre-registered alternatives on
2012-2026 history rebuilt from the filings companies had published at the time. Its answer - the
ranking has not been shown to predict returns - belongs on the public site next to the live track
record, which on its own looks better than the evidence. This module turns the engine's committed
ledger into one static page in the dashboard's own look.

**Generated** (CLAUDE.md rule 10): ``python v2_experiments.py report`` writes it whenever the
results change. Never hand-edit ``research.html``. Every number on it comes from
``research/experiments/ledger.jsonl`` through this module, plus two committed measurements read
beside it (each month's coverage, and how many stocks each configuration ranks; ``COVERAGE`` and
``PER_CONFIG``); ``tests/test_research_page.py`` rebuilds the page and fails if the committed one
differs, if a number does not match the ledger, or if the page uses advice language. Never imported
by production code.

Corrected 2026-10-16 after an audit found the page claiming more than the ledger supports: the Size
sentence is now a hypothesis generated from what the coverage data can show; the seven screens whose
2020-2026 outcome was on file before registration are marked; the pairing is described as by month;
every headline IC carries its interval; the chart's band and the pass bar are drawn and named
separately; the trading-rule sentence is generated from each period's sign and t; and the screens
that are within-sector versions of market-wide papers say so.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "research.html"
# Two committed measurements the page reads beside the ledger (2026-10-16 audit): each month's
# coverage (the share of that month's members backtest v2 could build), and how many stocks each
# configuration ranks. Both are written by tracked scripts in research/measurements/.
COVERAGE = ROOT / "research" / "measurements" / "2026-10-16-coverage-by-month.json"
PER_CONFIG = ROOT / "research" / "measurements" / "2026-10-16-stocks-ranked-per-config.json"
KNOWN_MARK = "&dagger;"
WITHIN_SECTOR = " (within sector)"
SHORT = {"presample": "2012&ndash;19", "discovery": "2020&ndash;mid-23", "holdout": "mid-23&ndash;26"}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}

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


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def display_label(e: dict) -> str:
    """An experiment's name for the page, as HTML: "(within sector)" where the screen is a
    within-sector version of a market-wide paper (``construction_note``), and a dagger where its
    2020-2026 outcome was on file before it was registered (``outcome_known_at_registration``)."""
    out = html.escape(label(e["id"]) + (WITHIN_SECTOR if e.get("construction_note") else ""))
    return out + (KNOWN_MARK if e.get("outcome_known_at_registration") else "")


def _corr(x, y) -> tuple[float, float]:
    """Pearson correlation and its t-statistic (n - 2 degrees of freedom)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    return r, r * math.sqrt((len(x) - 2) / max(1 - r * r, 1e-12))


def size_coverage_relation(latest: dict, coverage: dict) -> dict | None:
    """Whether Size's 2012-2019 lead can be read as survivorship, from the ledger's own monthly ICs
    and each month's coverage (2026-10-16). Coverage rises with the calendar almost exactly, so the
    across-era relation cannot tell survivorship from time; the within-era one is the only part of
    the data that could, and the page's sentence says only what it finds."""
    months_cov = (coverage or {}).get("months") or {}
    size = (latest.get("only-size") or {}).get("ic_by_month") or {}
    base = (latest.get("baseline") or {}).get("ic_by_month") or {}
    ms = sorted(m for m in months_cov if m in size and m in base)
    if len(ms) < 24:
        return None
    cov = [months_cov[m]["coverage"] for m in ms]
    s = [size[m] for m in ms]
    b = [base[m] for m in ms]
    d = [x - y for x, y in zip(s, b)]
    pre = [i for i, m in enumerate(ms) if m <= "2019-12-31"]
    pick = lambda xs: [xs[i] for i in pre]            # noqa: E731
    years = sorted({m[:4] for m in ms})
    year_cov = lambda y: float(np.mean([months_cov[m]["coverage"] for m in ms if m.startswith(y)]))  # noqa: E731
    r_time, _ = _corr(cov, range(len(ms)))
    r_s, _ = _corr(pick(s), pick(cov))
    r_b, _ = _corr(pick(b), pick(cov))
    r_d, t_d = _corr(pick(d), pick(cov))
    r_da, t_da = _corr(d, cov)
    return {"months": len(ms), "pre_months": len(pre), "r_time": r_time, "pre_r_size": r_s, "pre_r_base": r_b,
            "pre_r_diff": r_d, "pre_t_diff": t_d, "all_r_diff": r_da, "all_t_diff": t_da,
            "first_year": years[0], "first_cov": year_cov(years[0]),
            "last_year": years[-1], "last_cov": year_cov(years[-1])}


def size_paragraph(row: dict, rel: dict | None, bar: float) -> str:
    """The sentence under the chart about the top dot when it is Size: a hypothesis, generated from
    what the coverage relation can and cannot show."""
    if not rel:
        return ""
    p = row["p"]
    reach = "short of" if abs(p["t"]) < bar else "beyond"
    if abs(rel["pre_t_diff"]) < 2:
        moved = "about as much as"
    else:
        moved = "more than" if rel["pre_r_diff"] < 0 else "less than"
    return (f'<p>{row["label"]} shows the largest 2012&ndash;2019 difference, <span class="num">{num(p["mean"])}</span> '
            f'(t&nbsp;{p["t"]:+.2f}), {reach} the pass bar. Survivorship could explain it. The test can only include '
            f"companies that are still in the index today, and among 2012's members the small ones that survived are "
            f"disproportionately the ones that did well. This data cannot separate that explanation from the passage "
            f"of time. The share of each month's members the test could include rose steadily, from "
            f'<span class="num">{rel["first_cov"] * 100:.0f}%</span> in {rel["first_year"]} to '
            f'<span class="num">{rel["last_cov"] * 100:.0f}%</span> in {rel["last_year"]} (its correlation with the '
            f'calendar is <span class="num">{rel["r_time"]:.3f}</span>), and within 2012&ndash;2019 the Size screen\'s '
            f"monthly result moved with that share {moved} the current weighting's did (correlations of "
            f'<span class="num">{num(rel["pre_r_size"])}</span> and <span class="num">{num(rel["pre_r_base"])}</span>). '
            f"Rebuilding the departed members from free filings and prices would test it; that rebuild is queued "
            f"with its prediction written down in advance.</p>")


def bot_sentence(b: dict, periods: list, p_value) -> str:
    """What the trading rule did against the average scored stock, period by period, generated from
    each period's sign and t (2026-10-16: the sentence was hand-written and said "in every period"
    where only one period's gap was distinguishable from zero)."""
    out = []
    for k, lab in periods:
        if k not in b:
            continue
        st = b[k]["excess"]
        word = "below" if st["mean"] < 0 else "above"
        dist = p_value(st) < 0.05
        out.append(f'In {lab} it returned <span class="num">{abs(st["mean"]) * 100:.2f}%</span> a month {word} that '
                   f'average (t&nbsp;{st["t"]:+.2f}), '
                   + ("a gap larger than chance readily produces." if dist else "a gap within the noise."))
    return " ".join(out)


def stop_verdict(base: dict, arm: dict, periods: list) -> str:
    """Whether the pre-registered stop-loss prediction held, period by period, generated from the
    numbers rather than written down beside them (2026-10-22).

    The prediction registered before the arm ran was: a less negative worst month and maximum
    drawdown in **every** period, excess about unchanged or slightly lower, turnover higher. So the
    sentence has to be able to say the prediction failed - which is what it says."""
    kept, broke = [], []
    for k, lab in periods:
        if k not in base or k not in arm:
            continue
        better = (arm[k]["worst_month"] > base[k]["worst_month"]
                  and arm[k]["max_drawdown"] > base[k]["max_drawdown"])
        (kept if better else broke).append(lab)
    if not kept and not broke:
        return ""
    if not broke:
        head = ("The prediction held: the stop cut both the worst month and the deepest fall in "
                "every period measured.")
    elif not kept:
        head = ("<strong>The prediction failed.</strong> The stop did not cut the worst month and "
                "the deepest fall in any period measured - in every one it made at least one of "
                "them worse.")
    else:
        head = (f"<strong>The prediction half failed.</strong> The stop cut both the worst month and "
                f"the deepest fall in {' and '.join(kept)}, and made at least one of them worse in "
                f"{' and '.join(broke)}.")
    tn = [f'{arm[k]["turnover_a_year"] / base[k]["turnover_a_year"]:.1f}x' for k, _ in periods
          if k in arm and k in base and base[k]["turnover_a_year"]]
    turn = (f" Turnover rose as predicted, to {' / '.join(tn)} the unstopped rule's." if tn else "")
    stops = sum(arm[k]["positions_stopped"] for k, _ in periods if k in arm)
    return (f"<p>{head}{turn} The stop closed <span class=\"num\">{stops}</span> positions across the "
            f"periods shown. A stop exits into a fall and holds cash until the next monthly "
            f"rebalance, so it trades the falls it escapes for the rebounds it misses, and the "
            f"periods above disagree about whether that is a good trade. One stop level was tested, "
            f"10%, fixed before the test ran and never searched over &mdash; a level chosen by what "
            f"this history happened to reward would tell you nothing.</p>")


def control_sentence(c: dict | None) -> str:
    """The page's claim that the test can see a signal, from the positive control's ledger row.

    Until 2026-10-25 this bullet asserted the claim in words with no number behind it, on the
    strength of a measurement from 2026-10-15 that was never a ledger row and was built on monthly
    records since rebuilt twice (plan/research-engine.md item 4(c)). Three states, and the page says
    which one it is in rather than asserting the first:

    * **seen** - the control cleared the same bar every alternative is judged against. Quote the
      difference, its t-statistic and the smallest difference this comparison could reliably detect,
      because "it saw a big one" is only informative beside what it can see.
    * **not seen** - say so plainly and say what it costs: every flat result on the page becomes
      uninterpretable. A page that could only report the reassuring outcome would be worthless.
    * **not run on these records** - no claim. ``summary_for_page`` returns ``None`` for a control
      row built on other data, by another method or from another baseline weighting, exactly as it
      drops an alternative on those grounds, so this is also the state in a fresh checkout.
    """
    if not c:
        return ("<strong>Whether the test can see a signal is being re-measured.</strong> The check "
                "feeds the ranking the filings as they stand today for every past month &mdash; "
                "information nobody had at the time &mdash; and asks whether this test spots the "
                "advantage. It is registered with its prediction, and it is not quoted here until it "
                "has been run on the same monthly records as everything above.")
    if not c.get("sees"):
        return ("<strong>Warning: the test did not see a signal that is known to be there.</strong> "
                "Fed the filings as they stand today for every past month &mdash; information nobody "
                "had at the time &mdash; this test put the cheating ranking "
                f'<span class="num">{num(c["ic_vs_baseline"])}</span> ahead of the real one '
                f'(t <span class="num">{c["t"]:+.1f}</span>), which does not clear its own bar. '
                "Until that is explained, read every flat result on this page as *this test cannot "
                "tell*, not as *there was nothing to find*.")
    seen = (f'<strong>The test can see a signal when there is one.</strong> Fed the filings as they '
            f'stand today for every past month &mdash; information nobody had at the time &mdash; the '
            f'same test puts that ranking <span class="num">{num(c["ic_vs_baseline"])}</span> of rank '
            f'correlation a month ahead of the real one over <span class="num">{c["months"]}</span> '
            f'months (t <span class="num">{c["t"]:+.1f}</span>)')
    if c.get("detectable") is not None:
        seen += (f', against the <span class="num">{num(c["detectable"])}</span> it could reliably '
                 f'detect')
    return seen + (". So the flat results above are a finding about the ranking, not a test that "
                   "cannot see anything.")


def _pbo_words(pbo: float) -> str:
    """How a PBO compares with the 0.5 that luck alone gives (Bailey et al. 2017)."""
    if pbo >= 0.55:
        return "worse than a coin toss"
    if pbo > 0.45:
        return "about what luck alone gives"
    return "better than luck alone would give"


def overfitting_section(reg: list, latest: dict) -> str:
    """"If you had kept the best backtest" (2026-10-20): the probability of backtest overfitting
    (Bailey, Borwein, Lopez de Prado & Zhu 2017) over the ledger's configurations, from
    ``overfitting.check``. Every sentence is generated from the numbers - including whether the
    whole-history reading survives in the recent era and without its most frequent winner, because
    a low PBO measures consistency, not cause."""
    import overfitting
    r = overfitting.check(reg, latest)
    if not r:
        return ""
    a, rec, wo = r["all"], r.get("recent"), r.get("without")
    top = a["top_winners"][0] if a["top_winners"] else None
    cards = [(f'{a["pbo"]:.2f}', f"chance the best-looking weighting landed in the bottom half, "
                                 f"{a['first_month'][:4]}&ndash;{a['last_month'][:4]}")]
    if rec:
        cards.append((f'{rec["pbo"]:.2f}', f"the same over {rec['first_month'][:4]}&ndash;{rec['last_month'][:4]} alone"))
    if wo:
        cards.append((f'{wo["pbo"]:.2f}', f"every year, with {html.escape(label(wo['dropped']))} dropped from the set"))
    stats = "".join(f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in cards)

    read = (f"Over {a['first_month'][:4]}&ndash;{a['last_month'][:4]} the PBO was "
            f'<span class="num">{a["pbo"]:.2f}</span>, {_pbo_words(a["pbo"])}, and the winner beat the current '
            f'weighting on the other half in <span class="num">{a["winner_beat_current_out_of_sample"] * 100:.0f}%</span> '
            f"of trials.")
    if top:
        s = (latest.get(top["id"]) or {}).get("summary") or {}
        pre = (s.get("presample") or {}).get("ic_vs_baseline") or {}
        hold = (s.get("holdout") or {}).get("ic_vs_baseline") or {}
        read += (f" But one weighting did most of the winning: {html.escape(label(top['id']))}, in "
                 f'<span class="num">{top["share"] * 100:.0f}%</span> of trials.')
        if pre and hold:
            read += (f" Its lead over the current weighting was <span class=\"num\">{num(pre['mean'])}</span> over "
                     f"2012&ndash;2019 and <span class=\"num\">{num(hold['mean'])}</span> over the held-out months"
                     + (", where it fell behind." if hold["mean"] < 0 < pre["mean"] else "."))
        if top["id"] == "only-size":
            read += " That is the Size result above, which survivorship could explain."
    rest = []
    if rec:
        rest.append(f"Over {rec['first_month'][:4]}&ndash;{rec['last_month'][:4]} alone the PBO was "
                    f'<span class="num">{rec["pbo"]:.2f}</span>, {_pbo_words(rec["pbo"])}, and the winner beat the '
                    f'current weighting out of sample in <span class="num">{rec["winner_beat_current_out_of_sample"] * 100:.0f}%</span> '
                    f"of trials.")
    if wo:
        rest.append(f"Drop {html.escape(label(wo['dropped']))} from the set and the PBO over every year is "
                    f'<span class="num">{wo["pbo"]:.2f}</span>, {_pbo_words(wo["pbo"])}.')
    robust = (rec is None or rec["pbo"] < 0.45) and (wo is None or wo["pbo"] < 0.45)
    so = ("So the best-looking backtest held up consistently, in every reading this page can make. That is a "
          "reason to research it, not to adopt it: a bias in the history can be consistent too."
          if robust else
          "So a low reading over the whole history did not survive either check. Keeping the best of many "
          "backtests would mostly have picked whatever the history happened to reward, which is why the "
          "screener's weights come from published research and every test here is registered before it runs.")
    return f"""
  <section>
    <h2>If you had kept the best backtest</h2>
    <p>A common way to choose a strategy is to backtest many and keep the winner. That rewards luck as well as skill. Bailey, Borwein, L&oacute;pez de Prado &amp; Zhu (2017) measure how much: split the history into equal blocks of months, and for every way of picking half of them (<span class="num">{a['splits']:,}</span> ways here, from {a['blocks']} blocks), take the weighting that did best on that half and see where it ranks on the other half. The <strong>probability of backtest overfitting</strong> (PBO) is the share of those trials in which the winner lands in the bottom half. Luck alone gives about 0.5; a weighting that is genuinely better pushes it toward 0. This uses the current weighting and the {a['configurations'] - 1} alternatives above, each scored month by month against the current one on the stocks both ranked, and reruns nothing.</p>
    <div class="stats">{stats}</div>
    <p>{read}</p>
    <p>{' '.join(rest)}</p>
    <p>{so}</p>
  </section>"""


def headline_figure(summary: dict) -> str:
    """The headline drawn (2026-10-27, plan/ui-elevation.md): the ranking's average monthly rank
    correlation in each era and pooled, each with its 95% interval, and beside them the positive
    control's lead with the smallest lead it could reliably detect - so the reader sees in one
    glance that every interval straddles zero while the check that the test can see a signal sits
    well clear of it. Every figure on it is one the page already prints, from the engine's summary;
    nothing is computed here but the positions."""
    ci = summary.get("baseline_ic_ci95") or {}
    ic = summary.get("baseline_ic") or {}
    if not ci:
        return ""
    rows, aria = [], []

    def row(name: str, sub: str, mean: float, lo=None, hi=None, band=None, kind: str = "fp-a") -> str:
        parts = []
        if band is not None:
            parts.append(f'<span class="fp-band" style="left:{_pos(-band):.2f}%;width:{_pos(band) - _pos(-band):.2f}%"></span>')
        parts.append('<span class="fp-zero"></span>')
        if lo is not None:
            parts.append(f'<span class="fp-ci" style="left:{_pos(lo):.2f}%;width:{_pos(hi) - _pos(lo):.2f}%"></span>')
        parts.append(f'<span class="fp-dot" style="left:{_pos(mean):.2f}%"></span>')
        val = (f'<span class="num">{num(mean)}</span>'
               + (f'<span class="fp-iv num">{num(lo)} to {num(hi)}</span>' if lo is not None else ""))
        return (f'<div class="fp-row {kind}"><div class="fp-name">{name}' + (f'<span>{sub}</span>' if sub else "") + '</div>'
                f'<div class="fp-track">{"".join(parts)}</div><div class="fp-val">{val}</div></div>')

    for key, name in (("presample", "2012&ndash;2019"), ("all", "2020&ndash;2026"), ("pooled", "All months, pooled")):
        if key not in ci or key not in ic:
            continue
        lo, hi = ci[key]
        sub = "" if key != "pooled" else f'<span class="num">{summary["months"]["pooled"]}</span> months'
        band = summary.get("pooled_detectable") if key == "pooled" else None
        rows.append(row(name, sub, ic[key], lo, hi, band))
        aria.append(f"{name.replace('&ndash;', ' to ')}: {num(ic[key])}, 95% interval {num(lo)} to {num(hi)}")
    c = summary.get("control")
    if c and c.get("ic_vs_baseline") is not None:
        rows.append('<div class="fp-sep"><span>The check</span></div>')
        rows.append(row("Fed today&rsquo;s filings", "lead over the real ranking", c["ic_vs_baseline"],
                        band=c.get("detectable"), kind="fp-c"))
        aria.append(f"the ranking fed today's filings: {num(c['ic_vs_baseline'])} ahead of the real one")
    ticks = "".join(f"<span>{t}</span>" for t in (num(-LIM, 2), num(-LIM / 2, 2), "0", num(LIM / 2, 2), num(LIM, 2)))
    label = html.escape("Average monthly rank correlation with the next month's return. " + "; ".join(aria)
                        .replace("&minus;", "-")) + "."
    return f"""
  <figure class="fp" role="img" aria-label="{label}">
    <figcaption><strong>The ranking&rsquo;s average monthly rank correlation</strong> with the next month&rsquo;s return, and its 95% interval</figcaption>
    <div class="fp-sep"><span>The ranking</span></div>
    {''.join(rows)}
    <div class="fp-axis"><div></div><div class="fp-ticks">{ticks}</div><div></div></div>
    <div class="legend">
      <span><i class="lg-dot"></i>average</span>
      <span><i class="lg-ci"></i>95% interval</span>
      <span><i class="lg-dot2"></i>the check&rsquo;s lead</span>
      <span><i class="lg-band"></i>smallest the test could reliably detect</span>
      <span><i class="lg-zero"></i>no relation</span>
    </div>
  </figure>"""


def build(reg: list, latest: dict, summary: dict, detectable, evaluations: list) -> str:
    """The page, from the registry, the latest ledger row per id, the engine's page summary,
    the engine's ``detectable(se, n)`` and the registered evaluations."""
    import v2_experiments as X        # the engine's statistics (Student's t, the pass bar); never production
    n = summary["registered"]
    base = latest["baseline"]["summary"]
    coverage = _load(COVERAGE)
    per_config = _load(PER_CONFIG).get("configs") or {}
    rows = []
    for e in reg:
        r = latest.get(e["id"])
        if e["id"] == "baseline" or e.get("void") or not r:
            continue
        s = r["summary"]
        if "presample" not in s or "ic_vs_baseline" not in s["presample"]:
            continue
        rows.append({"id": e["id"], "e": e, "label": display_label(e),
                     "p": s["presample"]["ic_vs_baseline"], "h": s["holdout"]["ic_vs_baseline"],
                     "source": e["source"].split(";")[0].split(" - ")[0].rstrip(".")})
    rows.sort(key=lambda r: -r["p"]["mean"])
    pre_ic, post_ic = summary["baseline_ic"]["presample"], summary["baseline_ic"]["all"]
    ci = summary.get("baseline_ic_ci95") or {}
    ic_sd = base["all"]["ic"]["se"] * math.sqrt(base["all"]["ic"]["n"])
    mid_p = sorted(rows, key=lambda r: r["p"]["se"])[len(rows) // 2]["p"]
    mid_h = sorted(rows, key=lambda r: r["h"]["se"])[len(rows) // 2]["h"]
    mde_p = detectable(mid_p["se"], n, mid_p.get("n"))
    mde_h = detectable(mid_h["se"], n, mid_h.get("n"))
    bar_p, bar_h = X.pass_bar(n, mid_p.get("n")), X.pass_bar(n, mid_h.get("n"))

    def _interval(key: str) -> str:
        lo, hi = ci[key]
        return f' (95% interval <span class="num">{num(lo)} to {num(hi)}</span>)'

    all_zero = all(lo <= 0 <= hi for lo, hi in ci.values()) if ci else True
    pooled = ""
    if "pooled" in ci:
        pooled = (f' Pooled over all <span class="num">{summary["months"]["pooled"]}</span> months it was '
                  f'<span class="num">{num(summary["baseline_ic"]["pooled"])}</span>{_interval("pooled")}. '
                  + ("Every interval includes zero. " if all_zero else "")
                  + f'The pooled record could reliably detect an average of about '
                  f'<span class="num">{summary["pooled_detectable"]:.3f}</span> at a t-statistic of 3, and no smaller.')
    headline = ("<strong>Not measurably.</strong>" if all_zero else "<strong>Not reliably.</strong>") + \
        (f" The ranking's average monthly rank correlation with the next month's return was "
         f'<span class="num">{num(pre_ic)}</span> over 2012&ndash;2019'
         + (_interval("presample") if "presample" in ci else "")
         + f' and <span class="num">{num(post_ic)}</span> over 2020&ndash;2026'
         + (_interval("all") if "all" in ci else "") + "." + pooled)

    known = [r for r in rows if r["e"].get("outcome_known_at_registration")]
    if known:
        k = len(known)
        which = "each category on its own" if all(r["id"].startswith("only-") for r in known) else "those marked"
        step1_extra = (f" For {WORDS.get(k, k)} of them ({which}, marked {KNOWN_MARK}), the 2020&ndash;2026 result "
                       f"had already been computed in an earlier check of the data before they were registered, so "
                       f"for those only the 2012&ndash;2019 test is blind.")
    else:
        step1_extra = ""

    universe = ""
    b0 = (per_config.get("baseline") or {}).get("sample")
    narrow = [(v["sample"]["mean_stocks"], i) for i, v in per_config.items()
              if i != "baseline" and (v or {}).get("sample") and i in {r["id"] for r in rows}]
    if b0 and narrow:
        m_n, m_id = min(narrow)
        # what each comparison was actually measured over (method version 2, 2026-10-17): read from
        # the ledger row, which records it per month, rather than re-derived here
        common = [v for m, v in (latest.get(m_id, {}).get("n_common_by_month") or {}).items()
                  if X.SAMPLE[0] <= m <= X.SAMPLE[1]]
        overlap = ""
        if common:
            c = sum(common) / len(common)
            # a narrow screen's stocks are usually a near-subset of the baseline's, so the overlap
            # is the screen's own universe and the work pairing does is to hold the *baseline* to
            # it. Say which of those two it is rather than printing the same number twice.
            only_narrow = m_n - c            # ranked by the screen but not by the current weighting
            overlap = (" The narrower screen ranks almost only companies the current weighting also ranks "
                       + ('(fewer than one a month is unique to it), ' if only_narrow < 1 else
                          f'(<span class="num">{only_narrow:.0f}</span> a month are unique to it), ')
                       + f"so pairing on the overlap holds the current weighting to that smaller set instead of "
                       f'crediting it with the <span class="num">{b0["mean_stocks"] - c:.0f}</span> only it can '
                       f"score." if c >= 0.99 * m_n else
                       f' The two had <span class="num">{c:.0f}</span> companies a month in the overlap, and that '
                       f"is what each comparison is measured on.")
        universe = (f" Each comparison is paired by month and measured on the stocks the two rankings share, so a "
                    f"difference is a difference in ranking rather than in which companies were ranked. A weighting "
                    f"can only rank what it has enough data to score: over 2020&ndash;2026 the current weighting "
                    f'ranked <span class="num">{b0["mean_stocks"]:.0f}</span> stocks a month on average and the '
                    f"narrowest alternative, "
                    f'{display_label(next(r["e"] for r in rows if r["id"] == m_id))}, '
                    f'<span class="num">{m_n:.0f}</span>.{overlap}')

    control_li = control_sentence(summary.get("control"))

    dots = []
    for r in rows:
        p = r["p"]
        d = detectable(p["se"], n, p.get("n"))
        bar = X.pass_bar(n, p.get("n"))
        q = X.t_critical(0.05, p["n"] - 1) if p.get("n") else 1.96
        lo, hi = p["mean"] - q * p["se"], p["mean"] + q * p["se"]
        where = "inside the pass bar" if abs(p["t"]) < bar else "beyond the pass bar"
        note = r["e"].get("construction_note")
        dots.append(
            f'<div class="dot-row"><div class="dot-name"' + (f' title="{html.escape(note)}"' if note else "")
            + f'>{r["label"]}</div>'
            f'<div class="track" role="img" aria-label="{html.escape(label(r["id"]))}: {num(p["mean"])}, t {p["t"]:+.1f}, {where}">'
            f'<span class="see" style="left:{_pos(-d):.2f}%;width:{_pos(d) - _pos(-d):.2f}%"></span>'
            f'<span class="zero"></span>'
            f'<span class="tick" style="left:{_pos(-bar * p["se"]):.2f}%"></span>'
            f'<span class="tick" style="left:{_pos(bar * p["se"]):.2f}%"></span>'
            f'<span class="ci" style="left:{_pos(lo):.2f}%;width:{_pos(hi) - _pos(lo):.2f}%"></span>'
            f'<span class="dot" style="left:{_pos(p["mean"]):.2f}%"></span></div>'
            f'<div class="dot-val num">{num(p["mean"])} <span class="t">t {p["t"]:+.1f}</span></div></div>')

    table = "".join(
        f'<tr><td>{r["label"]}<div class="src">{html.escape(r["source"])}</div>'
        + (f'<div class="src">How it is built here: {html.escape(r["e"]["construction_note"])}</div>'
           if r["e"].get("construction_note") else "")
        + f'</td><td class="num">{num(r["h"]["mean"])} <span class="t">({r["h"]["t"]:+.1f})</span></td>'
        f'<td class="num">{num(r["p"]["mean"])} <span class="t">({r["p"]["t"]:+.1f})</span></td></tr>'
        for r in sorted(rows, key=lambda r: label(r["id"])))

    size_html = size_paragraph(rows[0], size_coverage_relation(latest, coverage), bar_p) \
        if rows and rows[0]["id"] == "only-size" else ""

    overfit_html = overfitting_section(reg, latest)

    bot_html = ""
    ev = next((e for e in evaluations if e["id"] == "bot-rule-v1.1"), None)
    if ev and latest.get(ev["id"]):
        b = latest[ev["id"]]["summary"]
        periods = [("presample", "2012&ndash;2019"), ("discovery", "2020 &ndash; mid-2023"),
                   ("holdout", "mid-2023 &ndash; 2026")]

        def _ex(st: dict) -> str:
            q = X.t_critical(0.05, st["n"] - 1)
            return (f'{pct(st["mean"])} <span class="t">(t {st["t"]:+.2f})</span>'
                    f'<div class="t">95%: {pct(st["mean"] - q * st["se"])} to {pct(st["mean"] + q * st["se"])}</div>')
        brows = "".join(
            f'<tr><td class="per">{SHORT[k]}</td><td class="num">{pct(b[k]["ret"]["mean"])}</td>'
            f'<td class="num">{_ex(b[k]["excess"])}</td>'
            f'<td class="num">{pct(b[k]["worst_month"])}</td>'
            f'<td class="num">{pct(b[k]["max_drawdown"])}</td>'
            f'<td class="num">{b[k]["turnover_a_year"] * 100:.0f}%</td></tr>'
            for k, lab in periods if k in b)
        cov_line = ""
        mc = (coverage or {}).get("months") or {}
        if mc:
            ys = sorted({m[:4] for m in mc})
            yc = lambda y: np.mean([v["coverage"] for m, v in mc.items() if m.startswith(y)]) * 100  # noqa: E731
            cov_line = (f" Those are the members still listed today: <span class=\"num\">{yc(ys[0]):.0f}%</span> of "
                        f"each month's members in {ys[0]}, rising to <span class=\"num\">{yc(ys[-1]):.0f}%</span> "
                        f"in {ys[-1]}, so the average leans toward the companies that survived.")
        rule = ev["rule"]
        # The stop-loss arm beside it (2026-10-22, OWNER_FOCUS): one pre-registered variant of the
        # same rule. Rendered only when its row is in the ledger on the same provenance, so the two
        # tables can never be two different vintages of the records read as one comparison.
        stop_html = ""
        sev = next((e for e in evaluations if e["rule"].get("stop_loss_pct")), None)
        srow = latest.get(sev["id"]) if sev else None
        if srow and X.comparable_provenance(srow, latest[ev["id"]]) \
                and srow.get("data_hash") == latest[ev["id"]].get("data_hash"):
            s = srow["summary"]
            srows = "".join(
                f'<tr><td class="per">{SHORT[k]}</td><td class="num">{pct(s[k]["ret"]["mean"])}</td>'
                f'<td class="num">{_ex(s[k]["excess"])}</td>'
                f'<td class="num">{pct(s[k]["worst_month"])}</td>'
                f'<td class="num">{pct(s[k]["max_drawdown"])}</td>'
                f'<td class="num">{s[k]["turnover_a_year"] * 100:.0f}%</td></tr>'
                for k, lab in periods if k in s)
            # Page prose, not the registry's text verbatim, for two reasons that are both rules:
            # the registered entry uses the word the no-advice guard bans (``BANNED_TERMS``), and
            # its prediction contains "in every period", which this page's own tripwire forbids
            # because a *result* sentence once claimed it falsely. A registered prediction may
            # never be edited after it has run, so the page says the same thing in its own words
            # and points at the registered text; RESULTS.md and the research note quote it.
            stop_html = f"""
    <h3>The same rule with a {sev['rule']['stop_loss_pct']:.0f}% stop-loss</h3>
    <p>One pre-registered variant of the rule above: a holding whose adjusted close ends a day {sev['rule']['stop_loss_pct']:.0f}% or more below its price at the start of the month is closed out at that close, and its slot stays in cash until the next monthly rebalance. The construction is Han, Zhou &amp; Zhu (2016), <em>Taming Momentum Crashes: A Simple Stop-Loss Strategy</em> (SSRN&nbsp;2407199) &mdash; the one stop-loss rule with published support behind it. It was registered with its prediction before it ran: a less negative worst month and a shallower deepest fall in all three periods, the gap against the average scored stock about unchanged or slightly lower, and turnover higher.</p>
    <div class="src">Four differences from the paper are recorded with the rule, each because it changes when the stop fires. The paper triggers on the day&rsquo;s lowest price where this reads the close, because the daily record carries no intraday range. The paper waits a day and exits at a limit a further {sev['rule']['stop_loss_pct'] / 2:.0f}% down, where this exits at the triggering close. The paper earns Treasury-bill interest on the proceeds, where cash here earns nothing. And the paper&rsquo;s portfolio is a long-short momentum sort of every listed stock, where this is the long side of a 25-name screen. The registered text is in <code>research/experiments/registry.yaml</code>, with the reading of the paper in <code>research/2026-10-22-stop-loss-on-the-bots-rule.md</code>.</div>
    <div class="tw"><table class="bot">
      <thead><tr><th>Period</th><th class="num">Return a month</th><th class="num">Against the average scored stock</th><th class="num">Worst month</th><th class="num">Deepest fall</th><th class="num">Turnover a year</th></tr></thead>
      <tbody>{srows}</tbody>
    </table></div>
    {stop_verdict(b, s, periods)}"""
        bot_html = f"""
  <section>
    <h2>What a simple trading rule on the ranking would have done</h2>
    <p>A rule that holds the top {rule['target_count']}, keeps each until it falls past rank {rule['sell_rank']}, skip names flagged as value or growth traps, at most {rule['max_per_sector']} in a sector, {rule['cost_bps']} basis points of cost a side, rebalanced monthly. It is compared with the equal-weighted average return of the stocks the test could score each month.{cov_line}</p>
    <div class="tw"><table class="bot">
      <thead><tr><th>Period</th><th class="num">Return a month</th><th class="num">Against the average scored stock</th><th class="num">Worst month</th><th class="num">Deepest fall</th><th class="num">Turnover a year</th></tr></thead>
      <tbody>{brows}</tbody>
    </table></div>
    <p>{bot_sentence(b, periods, X.p_value)} Read any live record of the ranking next to an equal-weight fund.</p>{stop_html}
  </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Has it been tested? · Multi-Factor Screener</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23141413'/%3E%3Crect x='7' y='17' width='4' height='8' rx='1.5' fill='%233987e5' opacity='.55'/%3E%3Crect x='14' y='12' width='4' height='13' rx='1.5' fill='%233987e5' opacity='.8'/%3E%3Crect x='21' y='7' width='4' height='18' rx='1.5' fill='%233987e5'/%3E%3C/svg%3E">
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
<link rel="preload" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=JetBrains+Mono:wght@400;500&display=swap" as="style" onload="this.onload=null;this.rel='stylesheet'">
<noscript><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"></noscript>
<style>
/* Generated by research_page.py - do not edit. Tokens are the dashboard's and the lab's. */
:root {{
    color-scheme: dark;
    --bg-deep: #0d0d0d; --bg-card: #1a1a19; --bg-elevated: #222221; --border: #262625; --border-bright: #343433;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8e8c86;
    --accent: #3987e5; --accent-text: #5598e7; --accent-glow: rgba(57,135,229,.12); --band: #252524;
    --font-body: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', monospace;
    /* plan/ui-elevation.md: the display serif for page and section titles and hero statements only */
    --font-display: 'Newsreader', 'Iowan Old Style', 'Palatino Linotype', Georgia, serif;
    --hairline-hi: inset 0 1px 0 rgba(255,255,255,.04);
    --bar-h: 60px; --t-fast: 140ms;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--bg-deep); color: var(--text-secondary); font-family: var(--font-body);
       font-size: 15px; line-height: 1.6; -webkit-font-smoothing: antialiased; }}
a {{ color: var(--accent-text); text-decoration: none; }}
a:hover {{ text-decoration: underline; text-underline-offset: 3px; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }}
.wrap {{ max-width: 960px; margin: 0 auto; padding: 0 16px 72px; }}
@media (min-width: 720px) {{ .wrap {{ padding: 0 32px 96px; }} }}
/* the site bar, as the dashboard's and the lab's: sticky, a full-bleed translucent band under a hairline */
.bar {{ position: sticky; top: 0; z-index: 30; display: flex; align-items: center; justify-content: space-between; gap: 16px; height: var(--bar-h); }}
.bar::before {{ content: ''; position: absolute; z-index: -1; top: 0; bottom: 0; left: 50%; width: 100vw; transform: translateX(-50%);
               background: color-mix(in srgb, var(--bg-deep) 88%, transparent); -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
               border-bottom: 1px solid var(--border); }}
.mark {{ display: inline-flex; align-items: center; gap: 10px; color: var(--text-primary); font-size: 15px; font-weight: 600; letter-spacing: -.01em; text-decoration: none; }}
.mark:hover {{ text-decoration: none; }}
.bmk {{ width: 24px; height: 24px; flex: none; }}
.bm-bg {{ fill: var(--bg-elevated); stroke: var(--border-bright); }} .bm-1 {{ fill: var(--accent); opacity: .5; }} .bm-2 {{ fill: var(--accent); opacity: .75; }} .bm-3 {{ fill: var(--accent); }}
.bar nav {{ display: flex; gap: 2px; font-size: 13.5px; flex-wrap: wrap; }}
.bar nav a {{ color: var(--text-secondary); padding: 7px 12px; border-radius: 8px; transition: color var(--t-fast), background var(--t-fast); }}
.bar nav a:hover {{ color: var(--text-primary); background: var(--bg-card); text-decoration: none; }}
.bar nav a[aria-current=page] {{ color: var(--text-primary); background: var(--bg-elevated); box-shadow: inset 0 0 0 1px var(--border-bright); }}
@media (max-width: 719px) {{ .bar {{ position: static; height: auto; min-height: 56px; flex-wrap: wrap; padding: 10px 0 8px; gap: 4px; }}
    .bar nav {{ margin-left: -12px; }} .bar nav a {{ padding: 10px 12px; }} }}
.lede {{ padding: 56px 0 8px; max-width: 70ch; }}
.eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--accent-text); }}
h1 {{ margin: 12px 0 20px; color: var(--text-primary); font-family: var(--font-display); font-optical-sizing: auto; font-size: clamp(36px, 4.6vw, 56px);
      line-height: 1.04; font-weight: 500; letter-spacing: -.015em; text-wrap: balance; }}
.lede p {{ font-size: 17px; line-height: 1.6; max-width: 62ch; }}
h2 {{ margin: 0 0 14px; color: var(--text-primary); font-family: var(--font-display); font-optical-sizing: auto; font-size: clamp(24px, 2.4vw, 30px);
      line-height: 1.15; font-weight: 500; letter-spacing: -.01em; text-wrap: balance; }}
h3 {{ color: var(--text-primary); font-size: 15px; font-weight: 600; margin: 28px 0 8px; }}
p {{ margin: 0 0 12px; max-width: 68ch; }}
section {{ margin-top: 64px; padding-top: 40px; border-top: 1px solid var(--border); }}
@media (max-width: 719px) {{ .lede {{ padding-top: 28px; }} .lede p {{ font-size: 16px; }} section {{ margin-top: 44px; padding-top: 28px; }} }}
.fine {{ font-size: 13px; color: var(--text-muted); }}
.num {{ font-family: var(--font-body); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.t {{ color: var(--text-muted); font-size: .85em; }}
strong {{ color: var(--text-primary); }}
.answer {{ display: grid; gap: 12px; margin: 28px 0 0; padding: 0; list-style: none; max-width: none; }}
.answer li {{ padding: 18px 20px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--hairline-hi); }}
.answer li:first-child {{ font-size: 16px; line-height: 1.65; border-color: var(--border-bright); }}
.answer li:first-child strong:first-child {{ display: block; margin-bottom: 4px; font-family: var(--font-display); font-optical-sizing: auto; font-size: clamp(24px, 2.6vw, 30px);
    font-weight: 500; line-height: 1.15; letter-spacing: -.01em; }}
@media (min-width: 720px) {{ .answer {{ grid-template-columns: 1fr 1fr; }} .answer li:first-child {{ grid-column: 1 / -1; }} }}
.terms {{ max-width: 76ch; margin: 12px 0 0; line-height: 1.55; }}
.terms strong {{ color: var(--text-secondary); font-weight: 600; }}
.steps {{ list-style: none; margin: 18px 0; padding: 0; display: grid; gap: 14px; max-width: 76ch; }}
.steps li {{ display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 14px; align-items: start; }}
.steps li > .num {{ display: grid; place-items: center; width: 28px; height: 28px; border: 1px solid var(--border-bright); border-radius: 50%;
    color: var(--text-primary); font-size: 13px; font-weight: 600; }}
.chart {{ margin: 16px 0 8px; padding: 20px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--hairline-hi); }}
@media (hover: hover) {{ .dot-row:hover {{ background: rgba(255,255,255,.025); }} }}
.dot-row {{ border-radius: 6px; }}
/* the headline, drawn: the ranking's average with its 95% interval, era by era, and the positive control beside it */
.fp {{ margin: 20px 0 0; padding: 22px 24px 18px; background: var(--bg-card); border: 1px solid var(--border-bright); border-radius: 14px; box-shadow: var(--hairline-hi); }}
.fp figcaption {{ margin: 0 0 14px; font-size: 13.5px; color: var(--text-secondary); max-width: 70ch; }}
.fp-row, .fp-axis {{ display: grid; grid-template-columns: minmax(0, 12em) minmax(0, 1fr) 11em; gap: 16px; align-items: center; }}
.fp-row {{ padding-block: 8px; }}
.fp-name {{ font-size: 14px; color: var(--text-primary); font-weight: 500; }}
.fp-name > span {{ display: block; font-size: 12px; font-weight: 400; color: var(--text-muted); }}
.fp-track {{ position: relative; height: 26px; }}
.fp-track > span {{ position: absolute; top: 50%; transform: translateY(-50%); }}
.fp-band {{ height: 26px; background: var(--band); border-radius: 4px; }}
.fp-zero {{ left: 50%; width: 1px; height: 34px; background: var(--text-muted); }}
.fp-ci {{ height: 3px; border-radius: 2px; background: var(--accent); }}
.fp-dot {{ width: 13px; height: 13px; margin-left: -6.5px; border-radius: 50%; background: var(--accent-text); box-shadow: 0 0 0 3px var(--bg-card); }}
.fp-c .fp-dot {{ background: var(--text-primary); }}
.fp-val {{ text-align: right; font-size: 15px; font-weight: 600; color: var(--text-primary); }}
.fp-iv {{ display: block; font-size: 12px; font-weight: 400; color: var(--text-muted); }}
.fp-sep {{ margin: 10px 0 2px; font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); }}
.fp-sep:not(:first-of-type) {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }}
.fp-axis {{ margin-top: 6px; }}
.fp-ticks {{ display: flex; justify-content: space-between; font: 11.5px var(--font-body); font-variant-numeric: tabular-nums; color: var(--text-muted); }}
.legend .lg-dot {{ width: 10px; height: 10px; border-radius: 50%; background: var(--accent-text); }}
.legend .lg-dot2 {{ width: 10px; height: 10px; border-radius: 50%; background: var(--text-primary); }}
.legend .lg-ci {{ width: 18px; height: 3px; border-radius: 2px; background: var(--accent); }}
.legend .lg-band {{ width: 18px; height: 10px; border-radius: 2px; background: var(--band); box-shadow: inset 0 0 0 1px var(--border-bright); }}
.legend .lg-zero {{ width: 1px; height: 12px; background: var(--text-muted); }}
@media (max-width: 599px) {{
  .fp {{ padding: 18px 16px 14px; }}
  .fp-row, .fp-axis {{ grid-template-columns: minmax(0, 1fr) auto; gap: 4px 12px; }}
  .fp-name {{ grid-column: 1; grid-row: 1; }} .fp-val {{ grid-column: 2; grid-row: 1; }}
  .fp-track {{ grid-column: 1 / -1; }}
  .fp-axis > div:first-child, .fp-axis > div:last-child {{ display: none; }} .fp-ticks {{ grid-column: 1 / -1; }}
}}
.dot-row, .axis {{ display: grid; grid-template-columns: minmax(0, 15em) minmax(0, 1fr) 7.5em; gap: 12px; align-items: center; }}
.dot-row {{ padding-block: 3px; }}
.dot-name {{ font-size: 13.5px; min-width: 0; overflow-wrap: anywhere; color: var(--text-secondary); }}
.track {{ position: relative; height: 18px; }}
.track span {{ position: absolute; top: 50%; transform: translateY(-50%); }}
.see {{ height: 18px; background: var(--band); border-radius: 3px; }}
.zero {{ left: 50%; width: 1px; height: 18px; background: var(--text-muted); opacity: .7; }}
.ci {{ height: 2px; background: var(--accent); opacity: .6; }}
.tick {{ width: 2px; height: 10px; margin-left: -1px; background: var(--text-primary); opacity: .55; }}
.dot {{ width: 9px; height: 9px; margin-left: -4.5px; border-radius: 50%; background: var(--accent-text); }}
.dot-val {{ font-size: 12.5px; text-align: right; }}
.axis {{ margin-top: 6px; }}
.ticks {{ display: flex; justify-content: space-between; font: 11.5px var(--font-body); font-variant-numeric: tabular-nums; color: var(--text-muted); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 12.5px; color: var(--text-muted); margin-top: 14px; }}
.legend i {{ display: inline-block; vertical-align: middle; margin-right: 6px; }}
.tw {{ overflow-x: auto; margin: 12px 0; max-width: 100%; }}
details {{ max-width: 100%; min-width: 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{ text-align: left; vertical-align: bottom; font-size: 11.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); border-bottom: 1px solid var(--border-bright); padding: 8px 12px 8px 0; }}
td {{ border-bottom: 1px solid var(--border); padding: 11px 12px 11px 0; vertical-align: top; font-variant-numeric: tabular-nums; }}
td.num {{ color: var(--text-primary); }}
@media (hover: hover) {{ tbody tr:hover td {{ background: rgba(255,255,255,.02); }} }}
th.num, td.num {{ text-align: right; }}
.src {{ font-size: 12px; color: var(--text-muted); }}
td.per {{ white-space: nowrap; }}
.bot th.num {{ white-space: normal; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 14px 0; }}
.stat {{ padding: 18px 20px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--hairline-hi); }}
.stat .v {{ font: 600 30px/1.15 var(--font-body); letter-spacing: -.02em; color: var(--text-primary); font-variant-numeric: tabular-nums; }}
.stat .k {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
.claims {{ display: grid; gap: 12px; grid-template-columns: 1fr; }}
@media (min-width: 720px) {{ .claims {{ grid-template-columns: 1fr 1fr; }} }}
.claim {{ padding: 18px 20px; background: var(--bg-elevated); border: 1px solid var(--border-bright); border-radius: 14px; box-shadow: var(--hairline-hi); }}
.claim h3 {{ margin: 0 0 8px; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); }}
.claim p {{ margin: 0; }}
details summary {{ cursor: pointer; color: var(--accent-text); font-size: 14px; margin-top: 12px; padding: 6px 0; }}
details summary:hover {{ color: var(--text-primary); }}
ul {{ padding-left: 20px; max-width: 72ch; }} li::marker {{ color: var(--text-muted); }}
code {{ font-family: var(--font-mono); font-size: .9em; }}
.foot .num {{ white-space: normal; overflow-wrap: anywhere; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition-duration: 0s !important; }} }}
.foot {{ margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 13px; color: var(--text-muted); }}
@media (max-width: 599px) {{
  td .t {{ display: block; }}
  .bot td.num {{ white-space: normal; }}
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
    <li>{headline}</li>
    <li><strong>No alternative did better.</strong> {"None" if summary['better'] == 0 else summary['better']} of the {summary['alternatives']} alternative weightings cleared the bar set before the test, in either period.</li>
    <li>{control_li}</li>
  </ul>
{headline_figure(summary)}
  <p class="fine terms"><strong>Reading the numbers.</strong> A rank correlation runs from &minus;1 to +1: at +1 the ranking would have put every stock's next-month return in exact order, and at 0 it said nothing about them. A t-statistic is a result divided by its own standard error, so the further it is from 0, the less likely chance alone produced the result. A basis point is a hundredth of a percentage point.</p>

  <section>
    <h2>How it was tested</h2>
    <p>Test enough ideas on one history and some look excellent by luck. The test was set up to prevent that:</p>
    <ol class="steps">
      <li><span class="num">1</span><span>Every idea, its source and its predicted result were written down and published in the project's history <em>before</em> it was run.{step1_extra}</span></li>
      <li><span class="num">2</span><span>Ideas were compared with the current weighting month by month and judged on months they were never tuned on.{universe}</span></li>
      <li><span class="num">3</span><span>The years 2012&ndash;2019 were set aside as an untouched test, with the rule for passing it, before any of that data was built.</span></li>
      <li><span class="num">4</span><span>A result counts only if it clears a bar that allows for how many ideas were tried: p&nbsp;&lt;&nbsp;0.05&nbsp;/&nbsp;{n} and a t-statistic of at least 3, the threshold Harvey, Liu &amp; Zhu (2016) propose for a new factor. With p from Student's t, that means a t-statistic of at least <span class="num">{bar_p:.2f}</span> over 2012&ndash;2019's {mid_p.get('n')} months and <span class="num">{bar_h:.2f}</span> over the {mid_h.get('n')} held-out months. Every run is kept, including runs a data correction replaced.</span></li>
    </ol>
  </section>

  <section>
    <h2>Each alternative against the current weighting, 2012&ndash;2019</h2>
    <p class="fine">Difference in average monthly rank correlation. The shaded band is the smallest difference the test could reliably detect (80% of the time). The two short marks either side of zero are the pass bar: a difference counts only if it reaches beyond them, a t-statistic of <span class="num">{bar_p:.2f}</span> over these {mid_p.get('n')} months, in the same direction as 2020&ndash;2026. A dot inside the marks did not pass; it may still be a real difference smaller than the band.{" Rows marked " + KNOWN_MARK + " had their 2020&ndash;2026 result on file before they were registered; this chart is their blind test." if known else ""}</p>
    <div class="chart" role="figure" aria-label="Each alternative's difference from the current weighting, with its interval and the detectable band">
      {''.join(dots)}
      <div class="axis"><div></div><div class="ticks"><span>{num(-LIM, 2)}</span><span>{num(-LIM / 2, 2)}</span><span>0</span><span>{num(LIM / 2, 2)}</span><span>{num(LIM, 2)}</span></div><div></div></div>
      <div class="legend">
        <span><i style="width:9px;height:9px;border-radius:50%;background:var(--accent-text)"></i>difference</span>
        <span><i style="width:18px;height:2px;background:var(--accent);opacity:.6"></i>95% interval</span>
        <span><i style="width:2px;height:10px;background:var(--text-primary)"></i>pass bar</span>
        <span><i style="width:18px;height:10px;background:var(--band);border-radius:2px"></i>smallest difference the test could reliably detect</span>
      </div>
    </div>
    {size_html}
    <details><summary>Every alternative, both periods</summary>
      <div class="tw"><table>
        <thead><tr><th>Alternative</th><th class="num">Mid-2023&ndash;2026 (t)</th><th class="num">2012&ndash;2019 (t)</th></tr></thead>
        <tbody>{table}</tbody>
      </table></div>
    </details>
  </section>
{overfit_html}{bot_html}
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
