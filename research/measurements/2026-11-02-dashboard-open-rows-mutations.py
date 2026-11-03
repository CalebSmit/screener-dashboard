"""Audit rows 189+ (plan/claims-mutation-audit.md, "Results, 2026-11-02 (owner-run, late)"): the
page-printing checks for five claims whose engine side was already pinned -
`drilldown.composite_chain`, `drilldown.peer_rank`, `drilldown.unscored_category`,
`context.track_turnover`, `context.track_luck` - and two claims registered the same night for
the queue's group 2: `dashboard.holdings_fit` (tests/test_holdings_panel.py, named by no claim
until then) and `drilldown.composite_distribution` (the masthead strip, whose only check lived in
tests/test_drilldown_presentation.py, read the committed page and is run by --before as it stood).

Each mutation is one exact single-occurrence replacement in `generate_dashboard.py`, the
generator whose emitted JavaScript prints the sentence. It runs **the claim's own `checked_by`**
(what `scripts/check_published_claims.py` derives), and with `--before` the same list minus every
node in `tests/test_dashboard_printed_figures.py` - which is how a row is recorded as a gap rather
than asserted to be one. The source is restored in `finally` and compared byte for byte.

    python research/measurements/2026-11-02-dashboard-open-rows-mutations.py [--before] [--rows 189-200]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import claims  # noqa: E402

NEW = "tests/test_dashboard_printed_figures.py::"
GD = "generate_dashboard.py"


# Checks written tonight outside the new module: dropped by --before as well.
NEW_ELSEWHERE = {
    "tests/test_holdings_panel.py::test_the_fit_line_prints_its_own_counts",
    "tests/test_holdings_panel.py::test_the_remove_all_dialog_counts_the_list_and_a_refusal_keeps_it",
    "tests/test_holdings_concentration.py::test_the_count_and_its_standing_at_each_published_threshold",
}
# The two claims registered tonight had no checked_by before it; --before runs the checks their
# modules already held that bear on the sentence (the masthead strip's lived in a module no claim
# named, against the committed page, and is run here as it stood).
BEFORE_UNREGISTERED = {
    "drilldown.composite_distribution": (
        "tests/test_drilldown_presentation.py::test_masthead_marks_the_published_composite_among_all_of_them",),
}


def checked_by(cid: str) -> tuple:
    if "--before" in sys.argv and cid in BEFORE_UNREGISTERED:
        return BEFORE_UNREGISTERED[cid]
    nodes = [c for c in claims.CLAIMS if c.id == cid][0].checked_by
    if "--before" in sys.argv:
        nodes = tuple(t for t in nodes if not t.startswith(NEW) and t not in NEW_ELSEWHERE)
    return nodes


CHAIN, PEER, UNSC = "drilldown.composite_chain", "drilldown.peer_rank", "drilldown.unscored_category"
TURN, LUCK = "context.track_turnover", "context.track_luck"
FIT, DIST = "dashboard.holdings_fit", "drilldown.composite_distribution"
REGIME, OVER = "overview.momentum_regime", "history.run_overview"
RATE = "context.rate_sensitivity"

MUTS = [
    # --- drilldown.composite_chain: the total, the discount line, the composite -------------
    (189, "the total prints the composite, not the category points", CHAIN,
     "add up to</span><strong>${{runningTotal.toFixed(1)}}</strong>", "add up to</span><strong>${{composite.toFixed(1)}}</strong>"),
    (190, "the discount line's n prints the denominator", CHAIN,
     "values for ${{cov.n}} of ${{cov.of}}", "values for ${{cov.of}} of ${{cov.of}}"),
    (191, "the coverage share truncated, not rounded", CHAIN,
     "(cov.n / cov.of * 100).toFixed(0)", "Math.floor(cov.n / cov.of * 100)"),
    (192, "the threshold printed as a fixed 80", CHAIN,
     "${{cd ? Math.round(cd.threshold * 100) : 80}}% threshold", "${{80}}% threshold"),
    (193, "the discount rate printed x10 too small", CHAIN,
     "reduced by ${{(disc * 100).toFixed(1)}}%", "reduced by ${{(disc * 10).toFixed(1)}}%"),
    (194, "the cut taken off the composite, not the points", CHAIN,
     "&minus;${{(runningTotal - composite).toFixed(1)}}", "&minus;${{(composite * disc).toFixed(1)}}"),
    (195, "the composite line prints the undiscounted points", CHAIN,
     "Composite Score: <strong>${{fmt(composite,'score')}}</strong>", "Composite Score: <strong>${{fmt(runningTotal,'score')}}</strong>"),
    (196, "the discount line needs a discount above 2%", CHAIN,
     "if (disc > 0) {{", "if (disc > 0.02) {{"),
    # --- drilldown.peer_rank: rank, peer count, direction, sector figures, percentile --------
    (197, "the rank is the count of better peers (off by one)", PEER,
     "return { rank: better + 1, n: n,", "return { rank: better, n: n,"),
    (198, "ties counted as better", PEER,
     "if (lower ? v < mine : v > mine) better++;", "if (lower ? v <= mine : v >= mine) better++;"),
    (199, "a universe-ranked stock counted against its sector only", PEER,
     "if (!fallback && o.sector !== s.sector) return;", "if (o.sector !== s.sector) return;"),
    (200, "a sector of exactly the minimum falls back to the universe", PEER,
     "const fallback = !st || st[0] < (D.sector_min_peers || 10);", "const fallback = !st || st[0] <= (D.sector_min_peers || 10);"),
    (201, "the sector median prints the lower quartile", PEER,
     'sector median <span class="wk-n">${fmtMetric(pc0.st[2], meta.fmt)}', 'sector median <span class="wk-n">${fmtMetric(pc0.st[1], meta.fmt)}'),
    (202, "the middle half ends at the median", PEER,
     "to ${fmtMetric(pc0.st[3], meta.fmt)}", "to ${fmtMetric(pc0.st[2], meta.fmt)}"),
    (203, "the percentile ordinal truncated", PEER,
     "${ordinal(Math.round(pctv0))}", "${ordinal(Math.floor(pctv0))}"),
    (204, "the direction words swapped", PEER,
     "${pc0.lower ? 'lower' : 'higher'} being better", "${pc0.lower ? 'higher' : 'lower'} being better"),
    (205, "13 takes 'rd'", PEER,
     "const suf = (v >= 11 && v <= 13) ? 'th'", "const suf = (v >= 11 && v <= 12) ? 'th'"),
    # --- drilldown.unscored_category: the counts, the names, the weight shared out ------------
    (206, "both counts are the whole table, not the weighted metrics", UNSC,
     "const n = names.length;", "const n = Object.keys(wp.table).length;"),
    (207, "the workings' count alone is the whole table", UNSC,
     "'None of the ' + n + ' metrics this category weights", "'None of the ' + Object.keys(wp.table).length + ' metrics this category weights"),
    (208, "the weight shared out is the category's effective weight (0)", UNSC,
     "its ${{fmtWeight(base[c] || 0)}} was shared out", "its ${{fmtWeight(ew[c] || 0)}} was shared out"),
    (209, "the weight shared out is the configured default (the code before tonight)", UNSC,
     "const base = D.weights.factor_weights || D.weights.base_factor_weights || {{}};",
     "const base = D.weights.base_factor_weights || {{}};"),
    (210, "a weight-0 metric counted as weighted", UNSC,
     "Object.keys(wp.table).filter(m => wp.table[m] > 0).sort(", "Object.keys(wp.table).filter(m => wp.table[m] >= 0).sort("),
    # --- context.track_turnover: the tile, the range, the paragraph, the break-even ---------
    (211, "the tile's mean prints the most traded", TURN,
     "kpi('Traded at each rebalance', cPlain(TO.mean, 0)", "kpi('Traded at each rebalance', cPlain(TO.max, 0)"),
    (212, "the range starts at the mean", TURN,
     "'one way, on average; ' + cPlain(TO.min, 0)", "'one way, on average; ' + cPlain(TO.mean, 0)"),
    (213, "the paragraph's mean counts both sides of each trade", TURN,
     "At each rebalance the top 25 traded ' + cPlain(TO.mean, 0)", "At each rebalance the top 25 traded ' + cPlain(TO.traded / TO.rebalances, 0)"),
    (214, "the break-even cost doubled", TURN,
     "'; a cost of ' + cPlain(TO.break_even_cost, 2)", "'; a cost of ' + cPlain(TO.break_even_cost * 2, 2)"),
    (215, "a null break-even cost still printed", TURN,
     "(TO.break_even_cost !== null && TO.break_even_cost !== undefined ? '; a cost of '", "(TO.break_even_cost !== undefined ? '; a cost of '"),
    # --- context.track_luck: the coin toss, the days, the t, the years ------------------------
    (216, "the coin-toss share is the complement", LUCK,
     "Math.round(T.beat_rsp_chance * 100)", "Math.round((1 - T.beat_rsp_chance) * 100)"),
    (217, "the trading days are the record's calendar days", LUCK,
     "'Measured day by day against RSP over ' + X.days + ' trading days", "'Measured day by day against RSP over ' + T.days + ' trading days"),
    (218, "the years to two to a whole number", LUCK,
     "Number(X.years_to_two).toFixed(1) + ' years of record", "Number(X.years_to_two).toFixed(0) + ' years of record"),
    (219, "a negative lead printed without its sign", LUCK,
     "the top 25&rsquo;s lead is ' + Number(X.t).toFixed(1)", "the top 25&rsquo;s lead is ' + Math.abs(X.t).toFixed(1)"),
    (220, "a null years-to-two still printed", LUCK,
     "(X.years_to_two ? ', and even if the lead", "(X.years_to_two !== undefined ? ', and even if the lead"),
    (221, "periods compared printed as all periods", LUCK,
     "T.beat_rsp + ' of ' + T.periods_compared", "T.beat_rsp + ' of ' + T.periods.length"),
    # --- dashboard.holdings_fit (test_holdings_panel.py, registered tonight) -----------------
    (222, "the top 25 counted strictly below 25", FIT,
     "return r.s.rank <= 25; }}", "return r.s.rank < 25; }}"),
    (223, "the top 100 counted strictly below 100", FIT,
     "return r.s.rank <= 100; }}", "return r.s.rank < 100; }}"),
    (224, "the universe is the table's length, not the run's", FIT,
     "const universe = D.kpis.universe_size || D.table_data.length;", "const universe = D.table_data.length;"),
    (225, "the sector count prints the largest sector's names", FIT,
     "' across <strong>' + sectors.length", "' across <strong>' + topCount"),
    (226, "only value traps counted as flagged", FIT,
     "return r.s.vt || r.s.gt; }}", "return r.s.vt; }}"),
    (227, "the largest sector's share over the sectors, not the names", FIT,
     "Math.round(topCount / n * 100)", "Math.round(topCount / sectors.length * 100)"),
    (228, "a published count met only above it", FIT,
     "return n >= m.n; }}).length;", "return n > m.n; }}).length;"),
    (229, "'above k' counts the marks not cleared", FIT,
     "'above ' + cleared + ' of the three'", "'above ' + (NAME_MARKS.length - cleared) + ' of the three'"),
    (230, "the count beside the standing prints the marks cleared", FIT,
     "'<p><strong>' + n + '</strong> name'", "'<p><strong>' + cleared + '</strong> name'"),
    (231, "the remove-all dialog states the list's cap", FIT,
     "'Remove all ' + holdings.length + ' names", "'Remove all ' + HOLDINGS_MAX + ' names"),
    # --- drilldown.composite_distribution (the masthead strip) -------------------------------
    (232, "the count is the number of bins", DIST,
     "' among the ' + d.n + ' published composites'", "' among the ' + d.bins.length + ' published composites'"),
    (233, "the label's median is the middle of the range", DIST,
     "'; the median is ' + fmt(med, 'score')", "'; the median is ' + fmt(d.lo + (d.hi - d.lo) / 2, 'score')"),
    (234, "the caption under the strip prints the stock's composite", DIST,
     "aria-hidden=\"true\">median ' + fmt(med, 'score')", "aria-hidden=\"true\">median ' + fmt(score, 'score')"),
    (235, "the label's composite rounded to a whole number", DIST,
     "const label = 'Composite ' + fmt(score, 'score')", "const label = 'Composite ' + fmt(Math.round(score), 'score')"),
    (236, "the next bin marked", DIST,
     "(i === on ? ' class=\"cd-on\"' : '')", "(i === on + 1 ? ' class=\"cd-on\"' : '')"),
    # --- overview.momentum_regime: the Methodology dialog's "Momentum rule, this run." -------
    (237, "the panic sentence's threshold prints the volatility", REGIME,
     "'The return is negative and volatility is above ' + pc(m.vol_threshold)", "'The return is negative and volatility is above ' + pc(m.vol)"),
    (238, "the unchanged sentence's threshold is a fixed 20.6%", REGIME,
     "'The panic state needs a negative return and volatility above ' + pc(m.vol_threshold)", "'The panic state needs a negative return and volatility above ' + '20.6%'"),
    (239, "the return printed without taking its size", REGIME,
     "pc(Math.abs(m.ret)) +", "pc(m.ret) +"),
    (240, "the months window prints the trading-day window", REGIME,
     "' over ' + m.return_window_months + ' months", "' over ' + m.vol_window_days + ' months"),
    (241, "the volatility printed daily, not annualised", REGIME,
     "realised volatility of ' + pc(m.vol) + ' a year", "realised volatility of ' + pc(m.vol / Math.sqrt(252)) + ' a year"),
    (242, "the panic sentence chosen on the return alone", REGIME,
     "m.vol_window_days + ' trading days. ' + (m.state === 'PANIC'", "m.vol_window_days + ' trading days. ' + (m.ret < 0"),
    # --- history.run_overview: the What Changed footnote and its section tile ----------------
    (243, "the footnote's up count is the listed movers", OVER,
     "${{mv.n_up}} stocks moved up materially", "${{mv.up.length}} stocks moved up materially"),
    (244, "the shown-up count is the whole count", OVER,
     "const shownUp = Math.min(mv.up.length, mv.n_up)", "const shownUp = mv.n_up"),
    (245, "the shown-down count capped by the up count", OVER,
     "shownDown = Math.min(mv.down.length, mv.n_down)", "shownDown = Math.min(mv.down.length, mv.n_up)"),
    (246, "the round-trip share over the listed movers", OVER,
     "${{mv.n_up + mv.n_down}}</strong> are flagged", "${{mv.up.length + mv.down.length}}</strong> are flagged"),
    (247, "the round-trip sentence printed with none", OVER,
     "if (mv.n_round_trip) {{", "if (mv.n_round_trip >= 0) {{"),
    (248, "the tile reads the last run, not the ~1-month window", OVER,
     "const k = H.movers.m1 ? 'm1' : 'prev', mv = H.movers[k];", "const k = 'prev', mv = H.movers[k];"),
    (249, "the tile's 'of N' is the movers, not the universe", OVER,
     "mv.n_up + ' of ' + N + ' stocks moved up materially", "mv.n_up + ' of ' + (mv.n_up + mv.n_down) + ' stocks moved up materially"),
    (250, "the tile caption's up count prints the down count", OVER,
     "'<div class=\"tv-cap\"><span>' + mv.n_up + ' up</span>", "'<div class=\"tv-cap\"><span>' + mv.n_down + ' up</span>"),
    # --- context.rate_sensitivity: the drilldown's rate card --------------------------------
    (251, "the slope printed per +1 point, not per +0.10", RATE,
     "cPct(c.rb * RATE_STEP, 2) + '</div>'", "cPct(c.rb, 2) + '</div>'"),
    (252, "the window counted in 22-day months", RATE,
     "Math.round(c.rn / 21)", "Math.round(c.rn / 22)"),
    (253, "the sector median printed per +1 point", RATE,
     "cPct(sr.median * RATE_STEP, 2)", "cPct(sr.median, 2)"),
    (254, "the share explained is the correlation, not its square", RATE,
     "cPlain(c.rr2, 0) + ' of its daily swings'", "cPlain(Math.sqrt(c.rr2), 0) + ' of its daily swings'"),
    (255, "'read loosely' only under 1%", RATE,
     "(c.rr2 < 0.05 ?", "(c.rr2 < 0.01 ?"),
    (256, "the sector median looked up by industry", RATE,
     "((D.market || {}).sector_rate || {})[s.sector]", "((D.market || {}).sector_rate || {})[s.industry]"),
]

rows = None
if "--rows" in sys.argv:
    a, _, b = sys.argv[sys.argv.index("--rows") + 1].partition("-")
    rows = range(int(a), int(b or a) + 1)

print(f"mode: {'--before (the new checks dropped)' if '--before' in sys.argv else 'now'}")
caught = missed = 0
for row, name, cid, a, b in MUTS:
    if rows is not None and row not in rows:
        continue
    src = ROOT / GD
    orig = src.read_bytes()
    text = orig.decode("utf-8")
    tests = checked_by(cid)
    try:
        assert text.count(a) == 1, (row, text.count(a))
        src.write_bytes(text.replace(a, b).encode("utf-8"))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                           cwd=ROOT, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        ok = r.returncode != 0
        caught += ok
        missed += not ok
        print(f"{row} {name:<66} {cid:<28} {'CAUGHT' if ok else 'MISSED'}  ({last})", flush=True)
    finally:
        src.write_bytes(orig)
        assert src.read_bytes() == orig
print(f"restored; {caught} caught, {missed} missed")
