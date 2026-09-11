"""The published percentile is direction-adjusted, and the page must say so.

``factor_engine.compute_sector_percentiles()`` does ``ranks = 100 - ranks`` for
every metric whose ``METRIC_DIR`` entry is ``False``. So a published percentile
always means "better than this share of its sector" and never "larger than".
For the 13 of 37 published metrics where lower is better, the percentile runs
*opposite* to the raw value printed immediately beside it.

Measured on the live 2026-09-11 payload, the two ends of that inversion:

===========  ==========  ===========
Stock        EV/EBITDA   Percentile
===========  ==========  ===========
HON              6.95        99
AXON            98.61         0
===========  ==========  ===========

Until 2026-09-11 nothing on the page stated the convention. A reader seeing
``EV/EBITDA 6.95`` at the 99th percentile had no way to tell it from a raw rank,
and the natural reading - "this company's EV/EBITDA is high for its sector" - is
exactly backwards. That is a comprehension defect in the surface whose whole
purpose is explaining *why* a stock ranks where it does.

These tests pin three things:

1. ``metric_meta[m]["dir"]`` exists for every published metric and **agrees with
   ``factor_engine.METRIC_DIR``**. It is derived, not hand-written, so the
   marker cannot drift from the ranking it describes.
2. The convention is stated in the rendered page, and the direction marker is
   rendered next to each metric name.
3. The universe table's eight category abbreviations carry definitions, and
   those definitions name only metrics that actually carry weight.

See ``METHODOLOGY_CHANGELOG.md`` 2026-09-11.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402
import stock_summary as s  # noqa: E402
from factor_engine import METRIC_DIR  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

CATEGORIES = ["valuation", "quality", "growth", "momentum",
              "risk", "revisions", "size", "investment"]

# The eight category columns in the universe table, with the sort key the
# header carries. Abbreviations a finance student cannot expand on sight are
# the reason this exists.
CATEGORY_COLUMNS = [
    ("valuation_score", "Val"),
    ("quality_score", "Qual"),
    ("growth_score", "Grow"),
    ("momentum_score", "Mom"),
    ("risk_score", "Risk"),
    ("revisions_score", "Rev"),
    ("size_score", "Size"),
    ("investment_score", "Inv"),
]


def _frame() -> pd.DataFrame:
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC", "DDD"],
        "Company": ["Alpha Inc", "Beta Corp", "Gamma Ltd", "Delta SA"],
        "Sector": ["Information Technology", "Financials", "Energy", "Utilities"],
        "Composite": [80.0, 60.0, 55.0, 40.0],
        "Rank": [1, 2, 3, 4],
        "Value_Trap_Flag": [False, False, False, False],
        "Growth_Trap_Flag": [False, False, False, False],
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = 50.0
    return df


@pytest.fixture(scope="module")
def payload() -> dict:
    data = g.prepare_dashboard_data({
        "df": _frame(), "meta": {"run_date": "2026-09-11"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    })
    return json.loads(data)


@pytest.fixture(scope="module")
def html() -> str:
    return g.generate_html()


# ---------------------------------------------------------------------------
# 1. Direction is present, and derived from the scorer rather than restated
# ---------------------------------------------------------------------------

def test_every_published_metric_declares_a_direction(payload):
    meta = payload["metric_meta"]
    assert meta, "metric_meta is empty - the fixture is not exercising the payload"
    missing = [m for m, v in meta.items() if "dir" not in v]
    assert missing == [], f"metrics published with no direction: {missing}"


def test_direction_matches_the_scorer_exactly(payload):
    """The whole point: the page cannot claim a direction the ranking disagrees
    with. If this fails, one of the two moved and the other did not."""
    meta = payload["metric_meta"]
    wrong = {
        m: (v["dir"], METRIC_DIR.get(m, True))
        for m, v in meta.items()
        if v["dir"] != ("higher" if METRIC_DIR.get(m, True) else "lower")
    }
    assert wrong == {}, f"page direction disagrees with METRIC_DIR: {wrong}"


def test_direction_is_only_ever_one_of_two_values(payload):
    values = {v["dir"] for v in payload["metric_meta"].values()}
    assert values <= {"higher", "lower"}, f"unexpected direction values: {values}"


def test_the_inverted_metrics_are_actually_marked_lower(payload):
    """A spot-check with named metrics, so a blanket bug that marked everything
    'higher' and still satisfied the agreement test above cannot pass.

    These seven are lower-is-better on the evidence in ``config.yaml``'s own
    comments: cheaper multiples, less leverage, less manipulation risk, calmer
    price, fewer days to cover, more conservative asset growth.
    """
    meta = payload["metric_meta"]
    for m in ["ev_ebitda", "ev_sales", "pb_ratio", "net_debt_to_ebitda",
              "beta", "volatility", "asset_growth"]:
        assert meta[m]["dir"] == "lower", f"{m} should be lower-is-better"


def test_the_normal_metrics_are_marked_higher(payload):
    meta = payload["metric_meta"]
    for m in ["fcf_yield", "earnings_yield", "roic", "gross_profit_assets",
              "forward_eps_growth", "revenue_growth", "return_12_1",
              "fy1_revision_3m"]:
        assert meta[m]["dir"] == "higher", f"{m} should be higher-is-better"


def test_both_directions_are_actually_present(payload):
    """If every metric pointed the same way the marker would be noise, and the
    convention note would be describing a distinction that does not exist."""
    dirs = [v["dir"] for v in payload["metric_meta"].values()]
    assert dirs.count("lower") > 0 and dirs.count("higher") > 0


def test_the_two_transformed_metrics_are_marked_higher(payload):
    """``size_log_mcap`` publishes -log(mcap) and ``max_drawdown_1y`` publishes a
    negative fraction. METRIC_DIR describes the direction of the *displayed*
    number in both cases, which is what makes deriving the marker from it safe.
    A future change that flips either without changing the displayed value would
    be publishing a false statement, so it is pinned."""
    meta = payload["metric_meta"]
    assert meta["size_log_mcap"]["dir"] == "higher"
    assert meta["max_drawdown_1y"]["dir"] == "higher"


def test_direction_does_not_leak_into_scored_data(payload):
    """``dir`` is display metadata. It must never reach ``raw`` or ``pct`` - the
    same constraint ``about``/``industry`` carry in test_dashboard_surfaces."""
    for detail in payload.get("stock_detail", {}).values():
        assert "dir" not in detail.get("raw", {})
        assert "dir" not in detail.get("pct", {})


# ---------------------------------------------------------------------------
# 2. The convention is stated where the numbers are
# ---------------------------------------------------------------------------

def test_percentile_header_states_the_convention(html):
    """'Percentile Rank' alone is the string this change replaced: it is the
    label that let the inverted reading through."""
    assert "Sector Percentile &mdash; 100 = best" in html


def test_convention_note_is_defined_and_rendered(html):
    assert "PCTILE_CONVENTION" in html
    assert "100 = best in its sector, not largest" in html
    assert "pctile-convention-note" in html


def test_convention_note_says_percentiles_are_sector_relative(html):
    """The percentiles really are sector-relative
    (``compute_sector_percentiles``). Dropping the qualifier would publish a
    false claim about how the number was computed - the same constraint
    ``stock_summary`` carries."""
    assert "not the whole index" in html


def test_convention_count_is_derived_not_hardcoded(html):
    """The '13 of 37' figure is computed from the payload at render time. A
    literal would go stale the next time a metric is added or reweighted, which
    is how three documentation claims went wrong before 2026-09-10."""
    assert "Object.keys(mm).filter(m => mm[m].dir === 'lower').length" in html


def test_direction_chip_is_rendered_next_to_each_metric(html):
    assert "function dirChip(meta)" in html
    assert "${dirChip(meta)}" in html
    assert "metric-dir-lower" in html
    assert "metric-dir-higher" in html


def test_direction_chip_has_an_explanatory_tooltip(html):
    """The arrow alone is not teaching. Both branches must spell it out."""
    assert "Lower is better: a smaller value earns a higher percentile." in html
    assert "Higher is better: a larger value earns a higher percentile." in html


def test_direction_chip_is_styled(html):
    """An unstyled chip inherits the metric-name colour and reads as part of the
    label, which is worse than omitting it."""
    assert ".metric-dir" in html


# ---------------------------------------------------------------------------
# 3. The universe table's abbreviations are defined at point of use
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sort_key,label", CATEGORY_COLUMNS)
def test_every_category_column_carries_a_definition(html, sort_key, label):
    marker = f'data-sort="{sort_key}" title="'
    assert marker in html, f"{label} column has no definition tooltip"
    tip = html.split(marker, 1)[1].split('"', 1)[0]
    assert len(tip) > 60, f"{label} tooltip is too short to teach anything: {tip!r}"
    assert "0-100" in tip, f"{label} tooltip does not state the score range"


def test_composite_column_is_defined_as_the_ranking_key(html):
    marker = 'data-sort="Composite" title="'
    assert marker in html
    tip = html.split(marker, 1)[1].split('"', 1)[0]
    assert "ranking key" in tip


def test_risk_column_warns_that_high_means_low_risk(html):
    """The single most confusable column on the page: a *high* Risk score means
    *low* risk, because the category's three metrics are all inverted before
    scoring. Saying only 'higher is better' would not fix the confusion."""
    marker = 'data-sort="risk_score" title="'
    tip = html.split(marker, 1)[1].split('"', 1)[0]
    assert "LOW risk" in tip


def test_category_tooltips_name_only_metrics_that_carry_weight(html):
    """Two drafts of these tooltips named P/B under Valuation and PEG under
    Growth. Both carry **zero** weight for non-banks - ``config.yaml`` marks
    P/B "Bank-only" and PEG "Removed: double-counts valuation" - so both
    would have taught a student something false about how the score is built.

    P/B is still legitimately named in the Valuation tooltip, but only in the
    sentence about how *banks* are scored, where it carries 60%.
    """
    val = html.split('data-sort="valuation_score" title="', 1)[1].split('"', 1)[0]
    growth = html.split('data-sort="growth_score" title="', 1)[1].split('"', 1)[0]

    # PEG must not be presented as a scored growth input.
    assert "PEG carries no weight" in growth

    # P/B appears only alongside the bank carve-out.
    assert "P/B" in val and "Banks" in val
    before_banks = val.split("Banks", 1)[0]
    assert "P/B" not in before_banks, (
        "P/B is named as a general valuation input, but it carries zero weight "
        "outside bank_metric_weights"
    )


# ---------------------------------------------------------------------------
# 4. The "Why it ranks here" prose carries the same qualifier
# ---------------------------------------------------------------------------

_META = {
    "ev_ebitda": {"label": "EV/EBITDA", "fmt": "ratio", "dir": "lower"},
    "fcf_yield": {"label": "FCF Yield", "fmt": "pct", "dir": "higher"},
    "no_dir": {"label": "Legacy Metric", "fmt": "ratio"},
}


def test_inverted_metric_prose_states_the_direction():
    """Without this, "the 99th sector percentile on EV/EBITDA (6.95)" reads as a
    contradiction, and the repair a reader makes is the wrong one."""
    out = s._label_and_value("ev_ebitda", 6.95, _META)
    assert "lower is better" in out
    assert "6.95" in out


def test_normal_metric_prose_is_left_alone():
    """The qualifier goes only where the ambiguity is. 502 stocks of prose is
    ~101 KB gzipped; a phrase on every metric would not be free."""
    out = s._label_and_value("fcf_yield", 0.105, _META)
    assert "lower is better" not in out
    assert "better" not in out


def test_metric_with_no_direction_does_not_crash():
    """``dir`` is added by generate_dashboard. A caller passing older metadata -
    or a metric added to METRIC_COLS but not metric_meta - must degrade to the
    old string, not raise inside the summary builder."""
    out = s._label_and_value("no_dir", 1.5, _META)
    assert out == "Legacy Metric (1.50)"


def test_the_qualifier_is_not_advice_language():
    """``BANNED_TERMS`` is the machine-checkable form of "explains why it ranks
    there, never whether to buy". "lower is better" describes the ranking rule,
    not an action, and must not trip the guard - checked through the real
    detector rather than by eyeballing the word list."""
    assert s.advice_terms_in(s._label_and_value("ev_ebitda", 6.95, _META)) == []


def test_missing_raw_value_still_has_no_qualifier():
    """No number means no contradiction to resolve, so the phrase would be
    dangling."""
    assert s._label_and_value("ev_ebitda", None, _META) == "EV/EBITDA"


def test_bank_carve_out_is_disclosed_where_it_changes_the_metrics(html):
    """Valuation and Quality are the two categories banks score differently.
    A student comparing JPM with AAPL on 'Qual' is not comparing like with
    like, and the page should say so at the column rather than only in the
    methodology document 30 headings down."""
    for key in ["valuation_score", "quality_score"]:
        tip = html.split(f'data-sort="{key}" title="', 1)[1].split('"', 1)[0]
        assert "Bank" in tip or "banks" in tip
