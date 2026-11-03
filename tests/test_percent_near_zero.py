"""One rule for every percentage the dashboard prints near zero, and a tripwire for new ones.

WHY THIS EXISTS. Three defects of one family have reached the live public site, in two opposite
directions, and each was fixed in the single formatter it was found in:

* **unsigned**, 2026-10-28 - BXP's last close was 0.03% *below* its 200-day average and the
  case-against line printed "0.0%", which states an equality that is not true. The data loop's
  claims gate refused that night's run. Fixed inline in that one sentence.
* **signed**, 2026-10-29 - AMD's five-year earnings-yield range read "-0.0% - 3.2%" and INTC's
  median "-0.0%": a sign on a zero, a value the page could not format. Fixed inside ``cPlain``.
* **both**, 2026-11-03 - NVR's average analyst target, 0.05% below its price, printed
  "$6262.00 -0.0%" (gate 1 red on the published payload), and ``cPct`` - the *signed* context
  formatter, written beside ``cPlain`` and not given its fix - was still printing an unsigned
  "0.0%" for ESS's close 0.03% below its 200-day average. The all-panel sweep
  (``tests/test_every_panel_reads_clean.py``) cannot see that one: its pattern looks for a
  **signed** zero, so the BXP direction is invisible to it by construction.

So the rule lives in ``pctNearZero`` and every percent formatter goes through it. This module
checks the rule's behaviour at its boundaries, pins the three live cases above, and - the part
that stops a fourth - fails if a formatter in the generated page builds a percentage without it
and is not named in ``EXEMPT`` with a reason.

The rule: a value that is **not** zero takes one more decimal rather than printing as a zero; if
it is still zero at that precision it is zero to the reader and prints **unsigned**, because a
sign on a zero is the glitch and an unsigned zero is true to the precision shown.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None


# ---------------------------------------------------------------------------
# the rule, evaluated in a browser against the generator's own page
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if sync_api is None:
        pytest.skip("MISSING DEP: playwright not installed")
    d = tmp_path_factory.mktemp("pctzero")
    (d / "index.html").write_text(
        gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v"),
        encoding="utf-8")
    # No payload: every function under test is pure arithmetic on its arguments, and the page's
    # own loader tolerates a missing data file. Nothing here reads D.
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page()
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof pctNearZero === 'function'", timeout=60000)
        yield pg
        b.close()


def call(page, expr: str):
    return page.evaluate(f"() => {expr}")


# (percent value already in percent units, decimals, printed magnitude, prints as zero)
RULE = [
    (5.34, 1, "5.3", False),        # ordinary: one decimal, unchanged
    (0.0, 1, "0.0", True),          # a true zero is a zero
    (-0.0495, 1, "0.05", False),    # NVR's analyst target: rounds to zero at 1, so takes 2
    (0.03, 1, "0.03", False),       # BXP / ESS: 0.03% is not 0.0%
    (-0.03, 1, "0.03", False),
    (0.04, 1, "0.04", False),
    (0.049, 1, "0.05", False),      # the boundary of rounding to zero at one decimal
    (0.051, 1, "0.1", False),       # just past it: one decimal is enough, so one decimal
    (-0.051, 1, "0.1", False),
    (0.004, 1, "0.0", True),        # still zero at two decimals: zero to the reader, unsigned
    (-0.004, 1, "0.0", True),
    (0.3, 2, "0.30", False),        # a caller asking for two decimals keeps two
    (0.003, 2, "0.003", False),     # BKR's rate sensitivity: zero at two, so takes three
    (0.0004, 2, "0.00", True),
    (-12.0, 0, "12", False),        # zero decimals is a legal request (turnover)
    (0.4, 0, "0.4", False),         #   and 0.4% is not 0%
]


@pytest.mark.parametrize("pct, d, mag, zero", RULE)
def test_the_rule_at_its_boundaries(page, pct, d, mag, zero):
    out = call(page, f"pctNearZero({pct!r}, {d})")
    assert out["mag"] == mag, out
    assert out["zero"] is zero, out


def test_a_non_number_prints_as_zero_rather_than_NaN(page):
    for expr in ("pctNearZero(NaN, 1)", "pctNearZero(Infinity, 1)",
                 "pctNearZero(null, 1)", "pctNearZero(undefined, 1)"):
        out = call(page, expr)
        assert out["zero"] is True and out["mag"] == "0.0", (expr, out)


def test_the_default_is_one_decimal(page):
    assert call(page, "pctNearZero(5.34)")["mag"] == "5.3"
    assert call(page, "pctNearZero(5.34, null)")["mag"] == "5.3"


# ---------------------------------------------------------------------------
# the formatters, including the three figures that were wrong on the live site
# ---------------------------------------------------------------------------
# (call, printed text). Fractions where the formatter takes a fraction, percent units where it
# takes percent units - which is itself worth pinning, since passing one for the other is how a
# figure lands 100x out.
FORMATTERS = [
    # pctPlain / fmtMetric / fmtInput take a fraction and print no sign for a zero
    ("pctPlain(0.0534, 1)", "5.3%"),
    ("pctPlain(-0.0003, 1)", "-0.03%"),            # ESS's 200-day gap, unsigned before today
    ("pctPlain(-0.00004, 1)", "0.0%"),
    ("pctPlain(0, 1)", "0.0%"),
    ("fmtMetric(-0.0003, 'pct')", "-0.03%"),
    ("fmtInput(-0.0003, 'pct')", "-0.03%"),
    # cPct: signed, and the sign is dropped only when the value prints as a zero
    ("cPct(0.0534, 1)", "+5.3%"),
    ("cPct(-0.0534, 1)", "−5.3%"),
    ("cPct(-0.0003, 1)", "−0.03%"),           # ESS, printed "0.0%" until 2026-11-03
    ("cPct(0.00003, 2)", "+0.003%"),               # BKR's rate sensitivity, printed "0.00%"
    ("cPct(-0.00004, 1)", "0.0%"),
    ("cPct(0, 1)", "0.0%"),
    ("cPct(null, 1)", "&mdash;"),
    # cPlain: the same rule, no sign on a positive
    ("cPlain(0.0534, 1)", "5.3%"),
    ("cPlain(-0.0003, 1)", "-0.03%"),
    ("cPlain(null, 1)", "&mdash;"),
    # cPp: percentage points, same rule
    ("cPp(0.0534, 1)", "+5.3pp"),
    ("cPp(-0.0003, 1)", "−0.03pp"),
    ("cPp(-0.00004, 1)", "0.0pp"),
]


@pytest.mark.parametrize("expr, text", FORMATTERS)
def test_each_formatter_prints_the_rule(page, expr, text):
    assert call(page, expr) == text


def test_the_analyst_target_card_prints_a_small_gap_rather_than_a_signed_zero(page):
    """NVR, 2026-11-03: a mean target of \\$6262.00 against a \\$6265.10 close printed "-0.0%",
    which is what turned gate 1 red. The card's formatter is local to renderPriceTargets, so it
    is reached the way a reader reaches it - by rendering the panel."""
    page.evaluate("""() => {
        window.D = window.D || {};
        document.body.insertAdjacentHTML('beforeend',
            '<div id="modal-price-targets"></div>');
        renderPriceTargets({price: 6265.1, pt_mean: 6262.0, pt_high: 7000, pt_low: 6262.0,
                            num_analysts: 3});
    }""")
    txt = page.evaluate("() => document.getElementById('modal-price-targets').innerText")
    assert "-0.05%" in txt, txt
    assert not re.search(r"[−-]0\.0+%", txt), txt


def test_an_exactly_equal_target_prints_an_unsigned_zero(page):
    page.evaluate("""() => {
        document.getElementById('modal-price-targets').innerHTML = '';
        renderPriceTargets({price: 100, pt_mean: 100, pt_high: 120, pt_low: 90, num_analysts: 3});
    }""")
    txt = page.evaluate("() => document.getElementById('modal-price-targets').innerText")
    assert "0.0%" in txt and "+0.0%" not in txt, txt


# ---------------------------------------------------------------------------
# the tripwire: a new percentage formatter must use the rule
# ---------------------------------------------------------------------------
# A `toFixed(...)` whose result is followed, within a few characters, by a percent sign or "pp" -
# the shape of a percentage this repo formatted itself.
PCT_SHAPE = re.compile(r"toFixed\([^()]*\)[\s)}\]]*(?:\+\s*)?['\"`]?\s*(?:%|pp\b)")

# Lines that build a percentage without the shared rule, each with the reason it is allowed to.
# A line is matched by a distinctive fragment of itself, so an edit to it fails this test and the
# reason gets re-read. **Adding a row is a decision, not a formality**: the question to answer is
# "can the value be non-zero and print as zero, and would that mislead a reader?"
EXEMPT: dict[str, str] = {
    # -- geometry: a CSS width/height/offset or an SVG coordinate, not a figure anyone reads
    "style=\"left:${pctLow}%;width:${(pctHigh - pctLow).toFixed(1)}%\"": "range-bar geometry",
    "style=\"width:${fillPct.toFixed(1)}%": "contribution-bar geometry",
    "style=\"width:${s.contrib[c].toFixed(2)}%": "composite stack geometry",
    "'<span class=\"trap-track\"><i style=\"width:'": "trap-rate bar geometry",
    "'<span class=\"cmp-bar\"><i style=\"width:'": "compare-bar geometry",
    "<i class=\"' + (v >= 0 ? 'up' : 'dn') + '\" style=\"--w:'": "category-gap bar geometry",
    "<span class=\"mk-rate-track\"><i class=\"'": "rate-sensitivity bar geometry",
    "'\" style=\"left:' + left.toFixed(1) + '%;width:'": "track tile bar geometry",
    "tileViz('sec-changed'": "up/down split bar geometry",
    "(i < 7 ? 'wk1' : 'wk2')": "sparkline column geometry",
    "<span class=\"tvp-t\"><i style=\"left:'": "percentile tick geometry",
    "Math.max(4, vt[s].rate / mx * 100).toFixed(1)": "trap-rate column geometry",
    "Math.max(4, Math.min(1, x.avg_jaccard) * 100).toFixed(1)": "stability column geometry",
    "((v - 1) / Math.max(1, N - 1) * 100).toFixed(1)": "rank tick geometry",
    # -- a share of a visible count: unsigned, and the denominator is printed beside it, so a
    #    share rounding to 0% reads as "0 of N" rather than as an equality that is not true
    "trap flags: ' + top + ', ' + Number(vt[top].rate).toFixed(1)":
        "trap rate, printed with its sector",
    "'<span class=\"trap-val\"><strong>'": "trap rate, printed with 'N of M'",
    "c.turnover_every_run.toFixed(0)": "turnover share from the construction config",
    "c.turnover_monthly.toFixed(0)": "turnover share from the construction config",
    "c.turnover_ceiling.toFixed(0)": "turnover ceiling from the literature",
    "(hi.v * 100).toFixed(0)": "the panic rule's volatility bar, a fixed 20.6%",
    "(dq.avg_metric_coverage * 100).toFixed(0)": "metric coverage share",
    "s.plus_jaccard": "category stability share",
    "s.minus_jaccard": "category stability share",
    "fl.beta_overlap_pct.toFixed(0)": "trading-day overlap share, against a printed 80% bar",
    "(avgJ * 100)": "category stability share",
    "secs.map(s => s + ' ' + Number(vt[s].rate).toFixed(1) + '%')": "trap rate per sector",
    # -- a weight: "whole when whole, else one decimal", and no weight in config.yaml is small
    #    enough to round to zero (the smallest is Size at 5%)
    "(Math.abs(pct - Math.round(pct)) < 0.05 ? Math.round(pct) : pct.toFixed(1))": "category weight",
    "(Math.abs(w - Math.round(w)) < 0.05 ? Math.round(w) : w.toFixed(1))": "metric weight",
    # -- already carries the rule inline, and its sentence template is pinned in claims.py
    "'Its last close is −' + gap.toFixed(gap < 0.1 ? 2 : 1)":
        "the case-against trend line, the 2026-10-28 fix; claim drilldown.case_against",
    # -- cannot round to zero: both branches run only when the value is above 2%
    "if (a <= 0.02) agree.push(p[1]); else differ.push": "filing check, only reached above 2%",
    "return p[1] + ' (' + (a * 100).toFixed(a < 0.1 ? 1 : 0) + '%)'":
        "filing check, only reached above 2%",
    # -- a level, not a difference: an inflation rate of 0.0% is a true reading
    "(u === '%' || u === 'pp') ? Number(sr.last).toFixed(2)": "a macro series' own level",
    "'inflation ' + Number(S.CPIAUCSL.last).toFixed(1)": "a macro series' own level",
    # -- the coverage discount, printed beside the count it is taken from
    "below the ${cd ? Math.round(cd.threshold * 100) : 80}% threshold": "coverage share and discount",
}


def _js() -> list[str]:
    """The page's script as published, so a formatter is checked where it ends up and not in
    whichever f-string or raw string it is written in."""
    html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    return html.splitlines()


def test_every_percentage_formatter_uses_the_rule_or_says_why_not():
    unexplained = []
    for line in _js():
        if not PCT_SHAPE.search(line):
            continue
        if "pctNearZero" in line or "pctPlain" in line:
            continue
        if any(frag in line for frag in EXEMPT):
            continue
        unexplained.append(line.strip()[:200])
    assert not unexplained, (
        f"{len(unexplained)} lines build a percentage without pctNearZero and are not in EXEMPT. "
        "Either route them through the rule or add a row saying why a zero cannot mislead there:\n"
        + "\n".join(unexplained))


def test_every_exemption_still_matches_a_line():
    """An exemption whose line has been edited away is a reason nobody will re-read."""
    js = _js()
    dead = [frag for frag in EXEMPT if not any(frag in line for line in js)]
    assert not dead, f"{len(dead)} exemptions match no line: {dead}"


def test_every_exemption_only_ever_names_a_line_that_formats_a_number():
    """A fragment broad enough to reach a line with no formatting in it would silence a real
    defect somewhere else on the page. Every line a fragment matches carries a ``toFixed``.
    (Not "exactly one line": the market backdrop's meta sentence is built in two places, and a
    fragment that names both is doing its job.)

    Not ``PCT_SHAPE``: one exempt line builds its percent sign inside a nested ternary
    (``(u === '%' || u === 'pp') ? ... + (u === '%' ? '%' : 'pp')``) which the pattern cannot
    see, and it is listed anyway, so the list records what the tripwire cannot reach as well as
    what it chooses to allow.
    """
    js = _js()
    bad = {}
    for frag in EXEMPT:
        hits = [line for line in js if frag in line]
        if not hits or any("toFixed" not in h for h in hits):
            bad[frag] = len(hits)
    assert not bad, f"{len(bad)} exemptions reach a line that formats nothing: {bad}"


def test_the_tripwire_fires(monkeypatch):
    """The floor: a formatter added without the rule is caught. Without this, an exemption list
    that happened to match everything would read exactly like coverage."""
    bad = "        const oops = v => (v * 100).toFixed(1) + '%';"
    monkeypatch.setattr(sys.modules[__name__], "_js", lambda: [bad])
    with pytest.raises(AssertionError, match="without pctNearZero"):
        test_every_percentage_formatter_uses_the_rule_or_says_why_not()


def test_the_pattern_tells_a_percentage_from_the_rest():
    for yes in ["(v * 100).toFixed(1) + '%'", "x.toFixed(2) + 'pp'",
                "`${(a - b).toFixed(1)}%`", "Math.abs(v).toFixed(1) + '%'"]:
        assert PCT_SHAPE.search(yes), yes
    for no in ["v.toFixed(2)", "'$' + v.toFixed(2)", "x.toFixed(1) + ' bp'",
               "Math.round(v).toString() + '%'"]:
        assert not PCT_SHAPE.search(no), no
