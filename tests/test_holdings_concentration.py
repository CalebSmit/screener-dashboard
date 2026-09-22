"""The Concentration block on My Holdings - north-star question 4, "how much?".

Shipped 2026-09-22. Until then question 4 had *no* surface at all, and had had
none since the Model Portfolio was removed on 2026-08-26. The fit line added on
2026-09-15 reports what the list *contains* (names, sectors, largest sector,
top-25/100 counts, trap flags); it says nothing about how the count compares to
the published diversification thresholds, and nothing about risk.

The design, and its sources, are section 8.4 of
`research/2026-09-21-position-sizing-and-how-much.md`. Three properties here are
research constraints rather than styling, and this module exists so a later
session does not tidy them away:

1. **No target weight for any stock, ever.** The obvious implementation of
   "how much" - print a recommended weight - is precisely what got the Model
   Portfolio deleted. Sizing by a conviction score is also the single most
   error-sensitive thing the estimation literature identifies: errors in
   expected returns do ~20x the damage of covariance errors, ~100x near zero
   risk aversion (Chopra & Ziemba 1993, via Ziemba & MacLean 2011), and none
   of 14 optimisation models across 7 datasets consistently beat 1/N out of
   sample (DeMiguel, Garlappi & Uppal, *RFS* 22(5), 2009). The equal-split line
   stays on the right side of that by being arithmetic on the *length* of the
   list - identical for every name on it - never a per-stock number.

2. **The risk comparison uses RAW annualised volatility, not the volatility
   percentile the payload also carries.** That percentile is sector-relative
   (`factor_engine.compute_sector_percentiles` groups by `Sector`) and
   direction-inverted (`METRIC_DIR['volatility']` is False, so a high
   percentile is a *low*-volatility stock), which makes it unable to rank risk
   across a mixed-sector list. Measured by
   `research/measurements/2026-09-22-holdings-risk-comparability.py` on the
   live run: the percentile orders the pair backwards for **23.9%** of the
   111,417 cross-sector pairs, worst case a name reading as the safer holding
   while carrying **2.00x** the volatility. `test_risk_line_follows_raw_
   volatility_not_the_sector_percentile` is that finding as a regression.

3. **The published thresholds are quoted with their condition attached.**
   Statman (1987), Campbell et al. (2001) and Domian et al. (2007) all measure
   *randomly selected* portfolios, so they bound the question for a
   pre-screened large-cap list rather than settling it. Quoting the numbers
   without that clause would overstate them.

As in `test_holdings_panel.py`, the functional half drives the real emitted
script under Node against a stubbed DOM - "the string appears in the file" is a
weak check for a block whose entire contract is what it renders.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dashboard as g  # noqa: E402
from test_holdings_panel import _DOM_STUB, _payload  # noqa: E402

NODE = shutil.which("node")
pytestmark_node = pytest.mark.skipif(NODE is None, reason="node not available")


@pytest.fixture(scope="module")
def html() -> str:
    return g.generate_html()


@pytest.fixture(scope="module")
def script(html) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == 1
    return blocks[0]


_EPILOGUE = r"""
globalThis.__api = {
  initHoldings, renderHoldings, addHolding, removeHolding, clearHoldings,
  holdingsFitLine, holdingsConcentration, renderHoldingsFootnote,
  NAME_MARKS, D,
  current: () => holdings.slice(),
};
"""

# Raw annualised volatility per test ticker, and a deliberately CONTRADICTORY
# sector percentile. AAA is the most volatile name in raw terms (0.60) while
# reading as the calmest on the sector percentile (95); BBB is the least
# volatile (0.20) while reading as the riskiest (5). Any implementation that
# reaches for `pct` instead of `raw` gets the pair exactly backwards, which is
# the real-data failure mode measured at 23.9% of cross-sector pairs.
_VOL_RAW = {"AAA": 0.60, "BBB": 0.20, "CCC": 0.30, "DDD": 0.40}
_VOL_PCT = {"AAA": 95.0, "BBB": 5.0, "CCC": 60.0, "DDD": 30.0}


def _payload_with_vol(include=("AAA", "BBB", "CCC", "DDD")) -> dict:
    payload = _payload()
    for t, d in payload["stock_detail"].items():
        if t in include:
            d["raw"] = {"volatility": _VOL_RAW[t]}
            d["pct"] = {"volatility": _VOL_PCT[t]}
        else:
            d.setdefault("raw", {})
            d.setdefault("pct", {})
    return payload


def _run_js(script: str, body: str, tmp_path: Path, payload=None) -> dict:
    payload = payload if payload is not None else _payload_with_vol()
    src = "\n".join([
        _DOM_STUB, script, _EPILOGUE,
        "const A = globalThis.__api;",
        "A.D.table_data.push(...A.D._rows);",
        body,
    ])
    path = tmp_path / "conc.cjs"
    path.write_text(src, encoding="utf-8")
    proc = subprocess.run([NODE, str(path), json.dumps(payload)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[:4000]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _fit_of(script, tickers, tmp_path, payload=None) -> str:
    body = (
        "A.initHoldings();\n"
        + json.dumps(list(tickers)) + ".forEach(A.addHolding);\n"
        "console.log(JSON.stringify({fit: globalThis.__els['holdings-fit'].innerHTML}));"
    )
    return _run_js(script, body, tmp_path, payload)["fit"]


# ---------------------------------------------------------------------------
# It exists and is wired in
# ---------------------------------------------------------------------------

def test_concentration_function_is_emitted(script):
    assert "function holdingsConcentration(" in script


def test_render_calls_it(script):
    """A block nothing calls is a block nobody sees."""
    render = script[script.index("function renderHoldings("):
                    script.index("function holdingsFitLine(")]
    assert "holdingsConcentration(rows)" in render


@pytest.mark.parametrize("n", [30, 50, 63])
def test_published_thresholds_are_present(script, n):
    marks = script[script.index("const NAME_MARKS"):
                   script.index("function holdingsConcentration(")]
    assert f"n: {n}" in marks or f"n:{n}" in marks


@pytest.mark.parametrize("source", ["Statman", "Campbell", "Domian"])
def test_each_threshold_names_its_source(script, source):
    marks = script[script.index("const NAME_MARKS"):
                   script.index("function holdingsConcentration(")]
    assert source in marks


# ---------------------------------------------------------------------------
# Constraint 1: no target weight, ever
# ---------------------------------------------------------------------------

@pytestmark_node
def test_equal_split_depends_only_on_the_count_not_on_which_stocks(script, tmp_path):
    """The line is arithmetic on the length of the list. Two different
    three-name lists must produce the identical figure - that is what makes it
    a statement about the list rather than a weight for a stock."""
    a = _fit_of(script, ["AAA", "BBB", "CCC"], tmp_path)
    b = _fit_of(script, ["BBB", "CCC", "DDD"], tmp_path)
    assert "33.3%</strong> a position" in a
    assert "33.3%</strong> a position" in b


@pytestmark_node
@pytest.mark.parametrize("tickers,expected", [
    (["AAA", "BBB"], "50.0%"),
    (["AAA", "BBB", "CCC"], "33.3%"),
    (["AAA", "BBB", "CCC", "DDD"], "25.0%"),
])
def test_equal_split_arithmetic(script, tmp_path, tickers, expected):
    assert expected + "</strong> a position" in _fit_of(script, tickers, tmp_path)


@pytestmark_node
def test_single_holding_reads_as_english(script, tmp_path):
    """One saved name is a common starting state - somebody has just added
    their first ticker - so it must not read as a broken plural."""
    fit = _fit_of(script, ["AAA"], tmp_path)
    assert "<strong>100%</strong> in one position" in fit
    assert "across 1 name" not in fit
    assert "1 names" not in fit


@pytestmark_node
def test_no_per_stock_weight_is_ever_printed(script, tmp_path):
    """No ticker may be followed by a percentage that reads as its weight.

    The equal-split figure is the only percentage-of-portfolio number on the
    block, and it is not attached to a name.
    """
    fit = _fit_of(script, ["AAA", "BBB", "CCC"], tmp_path)
    block = fit[fit.index("holdings-concentration"):]
    for t in ("AAA", "BBB", "CCC"):
        for m in re.finditer(re.escape(t), block):
            tail = re.sub(r"<[^>]+>", "", block[m.end():m.end() + 60])
            assert not re.match(r"[^.]{0,20}?\b\d+(\.\d+)?%\s*(a position|of|weight)",
                                tail), f"{t} appears followed by what reads as a weight"


def test_block_carries_no_currency_or_cost_basis(script):
    code = script[script.index("function holdingsConcentration("):
                  script.index("function holdingCard(")]
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("//"))
    for term in ("$", "shares", "cost", "paid", "profit", "positionSize"):
        assert term.lower() not in code.lower(), f"{term!r} in the concentration block"


# ---------------------------------------------------------------------------
# Constraint 2: raw volatility, not the sector percentile
# ---------------------------------------------------------------------------

@pytestmark_node
def test_risk_line_follows_raw_volatility_not_the_sector_percentile(script, tmp_path):
    """The 23.9% finding, as a regression test.

    AAA carries 3x BBB's raw volatility while reading as the *calmer* name on
    the sector percentile. The block must name AAA as the most volatile
    holding. An implementation reading `pct` names BBB and is wrong in the same
    direction as 23.9% of real cross-sector pairs.
    """
    fit = _fit_of(script, ["AAA", "BBB"], tmp_path)
    block = fit[fit.index("holdings-concentration"):]
    assert "Widest risk gap" in block
    hi = block.index("AAA")
    lo = block.index("BBB")
    assert hi < lo, "the most volatile name must be named first"
    assert "60%</strong> annualised" in block
    assert "20%</strong>" in block
    assert "3.0&times;</strong> spread" in block


@pytestmark_node
def test_risk_line_reports_the_extremes_of_the_whole_list(script, tmp_path):
    """Four names: the gap is AAA (0.60) against BBB (0.20), not any inner
    pair, and the ratio is computed from the raw numbers."""
    block = _fit_of(script, ["AAA", "BBB", "CCC", "DDD"], tmp_path)
    assert "3.0&times;</strong> spread" in block
    assert "60%</strong> annualised" in block


@pytestmark_node
def test_risk_line_is_omitted_when_there_is_nothing_to_compare(script, tmp_path):
    """One holding has no spread; a block claiming one would be inventing it."""
    fit = _fit_of(script, ["AAA"], tmp_path)
    assert "holdings-concentration" in fit
    assert "Widest risk gap" not in fit


@pytestmark_node
def test_risk_line_is_omitted_when_volatility_is_missing(script, tmp_path):
    """Coverage is 501 of 502 on the live run, so the absent case is real."""
    payload = _payload_with_vol(include=())
    fit = _fit_of(script, ["AAA", "BBB"], tmp_path, payload)
    assert "holdings-concentration" in fit, "the rest of the block still renders"
    assert "Widest risk gap" not in fit


@pytestmark_node
def test_a_name_missing_volatility_does_not_break_the_comparison(script, tmp_path):
    """AAA and BBB carry volatility, CCC does not. The pair is still AAA/BBB."""
    payload = _payload_with_vol(include=("AAA", "BBB"))
    block = _fit_of(script, ["AAA", "BBB", "CCC"], tmp_path, payload)
    assert "3.0&times;</strong> spread" in block
    assert "3</strong> names" in block, "all three still count toward the name total"


def test_block_does_not_read_the_volatility_percentile(script):
    """Belt and braces on the above: the code must not touch `pct` at all."""
    code = script[script.index("function holdingsConcentration("):
                  script.index("function holdingCard(")]
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("//"))
    assert ".pct" not in code and "pct[" not in code
    assert "raw || {}" in code.replace("{{", "{")


# ---------------------------------------------------------------------------
# Constraint 3: thresholds quoted with their condition
# ---------------------------------------------------------------------------

@pytestmark_node
@pytest.mark.parametrize("n,expected", [
    (1, "below all three"),
    (3, "below all three"),
])
def test_standing_against_the_thresholds(script, tmp_path, n, expected):
    fit = _fit_of(script, ["AAA", "BBB", "CCC", "DDD"][:n], tmp_path)
    assert expected in fit


@pytestmark_node
def test_randomly_chosen_condition_is_always_stated(script, tmp_path):
    """The three counts measure randomly selected portfolios. Quoting them
    without that clause overstates them for a pre-screened large-cap list."""
    fit = _fit_of(script, ["AAA", "BBB"], tmp_path)
    assert "randomly chosen" in fit
    assert "bound the question rather than settle it" in fit


@pytestmark_node
def test_published_caps_are_quoted_as_caps_not_targets(script, tmp_path):
    fit = _fit_of(script, ["AAA", "BBB"], tmp_path)
    assert "UCITS" in fit and "5%" in fit
    assert "25%" in fit and "RIC" in fit
    assert "24%" in fit and "Select Sector" in fit
    assert "not targets" in fit


@pytestmark_node
def test_empty_list_renders_no_concentration_block(script, tmp_path):
    out = _run_js(script, """
        A.initHoldings();
        console.log(JSON.stringify({fit: globalThis.__els['holdings-fit'].innerHTML}));
    """, tmp_path)
    assert "holdings-concentration" not in out["fit"]


@pytestmark_node
def test_it_sits_below_the_fit_line(script, tmp_path):
    """The fit line says what the list is; concentration interprets it."""
    fit = _fit_of(script, ["AAA", "BBB"], tmp_path)
    assert fit.index("holdings-fit-line") < fit.index("holdings-concentration")


# ---------------------------------------------------------------------------
# The teaching half
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,detail", [
    ("Chopra", "20&times;"),
    ("Ziemba", "100&times;"),
    ("DeMiguel", "3,000 months"),
])
def test_footnote_sources_the_refusal_to_size(script, source, detail):
    """Every claim a reader might argue with is sourced on the page itself."""
    foot = script[script.index("function renderHoldingsFootnote("):
                  script.index("function setupHoldingsSearch(")]
    assert source in foot and detail in foot


def test_footnote_explains_the_percentile_choice(script):
    foot = script[script.index("function renderHoldingsFootnote("):
                  script.index("function setupHoldingsSearch(")]
    assert "23.9%" in foot
    assert "sector" in foot.lower()
    assert "2.00&times;" in foot


def test_footnote_states_selection_and_weighting_are_separate(script):
    """The coherence point from section 4.4: in every documented institutional
    scheme the alpha signal drives selection and weighting is separate."""
    foot = script[script.index("function renderHoldingsFootnote("):
                  script.index("function setupHoldingsSearch(")]
    assert "selection" in foot and "separate" in foot


def test_no_advice_language_in_the_block(script):
    import stock_summary
    code = script[script.index("function holdingsConcentration("):
                  script.index("function holdingCard(")]
    text = re.sub(r"<[^>]+>", " ", code)
    hits = [t for t in stock_summary.BANNED_TERMS
            if re.search(r"\b" + re.escape(t) + r"\b", text, re.I)]
    assert not hits, f"advice language in the concentration block: {hits}"
