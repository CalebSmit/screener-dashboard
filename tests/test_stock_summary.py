"""The deterministic per-stock summary - what it says, and what it may never say.

Owner directive 2026-08-10, priority 4 in ``CLAUDE.md``, specified in
``plan/dashboard-north-star.md``. Shipped 2026-09-08 as ``stock_summary.py``
plus a "Why it ranks here" block at the top of the stock drilldown, replacing
the browser-side "Screener AI" chat.

The whole point of the replacement is that the explanation is **reproducible**:
built at run time from the run's own numbers, baked into the payload, identical
for every reader. So the tests here are about two properties:

1. **Every figure is arithmetic on the payload**, not a paraphrase. If the
   composite is 74.7 and Valuation contributes 21.1, the sentence says exactly
   those numbers - a test recomputes them from the fixture.
2. **It explains, it never advises.** ``BANNED_TERMS`` is the machine-checkable
   form of the line in ``plan/dashboard-north-star.md``: "a summary explains why
   a stock ranks where it does; it never says whether to buy it." A public
   screener that emits "attractive entry point" is a liability for a student
   investment club, which is the audience this tool is being built for.

The chat-removal side is pinned in ``tests/test_ai_chat_removed.py``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stock_summary as ss  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

CATEGORIES = ss.CATEGORIES

METRIC_META = {
    "ev_ebitda": {"label": "EV/EBITDA", "fmt": "ratio", "category": "valuation"},
    "fcf_yield": {"label": "FCF Yield", "fmt": "pct", "category": "valuation"},
    "earnings_yield": {"label": "Earnings Yield", "fmt": "pct", "category": "valuation"},
    "roic": {"label": "ROIC", "fmt": "pct", "category": "quality"},
    "piotroski_f_score": {"label": "Piotroski F-Score", "fmt": "int", "category": "quality"},
    "beta": {"label": "Beta", "fmt": "ratio", "category": "risk"},
    "volatility": {"label": "Volatility", "fmt": "pct", "category": "risk"},
}

METRIC_WEIGHTS = {
    "valuation": {"ev_ebitda": 25, "fcf_yield": 45, "earnings_yield": 20, "pb_ratio": 0},
    "quality": {"roic": 27, "piotroski_f_score": 15, "roe": 0},
    "growth": {},
    "momentum": {},
    "risk": {"volatility": 42.86, "beta": 28.57},
    "revisions": {},
    "size": {},
    "investment": {},
}


def _detail(**overrides) -> dict:
    """A complete, self-consistent stock detail record.

    Contributions are deliberately *not* round: they are what the pipeline
    actually writes (score x effective weight / 100), and the summary must
    quote them rather than recompute from the headline weights - the exact
    mistake the 2026-08-28 weight-transparency fix was about.
    """
    detail = {
        "company": "Alpha Inc",
        "sector": "Information Technology",
        "composite": 74.65,
        "rank": 1,
        "cat_scores": {
            "valuation": 95.76, "quality": 83.47, "growth": 49.53,
            "momentum": 97.50, "risk": 37.62, "revisions": 65.06,
            "size": 70.00, "investment": 46.67,
        },
        "contrib": {
            "valuation": 21.07, "quality": 18.36, "growth": 6.44,
            "momentum": 12.68, "risk": 3.76, "revisions": 6.51,
            "size": 3.50, "investment": 2.33,
        },
        "raw": {"ev_ebitda": 8.99, "fcf_yield": 0.081, "earnings_yield": 0.067,
                "roic": 0.244, "piotroski_f_score": 7.0,
                "beta": 0.70, "volatility": 0.31},
        "pct": {"ev_ebitda": 97.4, "fcf_yield": 91.2, "earnings_yield": 96.8,
                "roic": 88.0, "piotroski_f_score": 74.0,
                "beta": 17.3, "volatility": 44.0},
        "vt": False,
        "gt": False,
        "price": 22.05,
        "pt_mean": 25.14,
        "num_analysts": 20,
        "metric_count": 18,
        "metric_total": 18,
        "eps_mismatch": False,
        "flags": {"vt_severity": 0.0, "gt_severity": 0.0, "beneish_flag": False,
                  "channel_stuffing": False, "stale_data": False,
                  "stmt_age_days": 70},
        "peers": [
            {"ticker": "BBB", "composite": 57.4},
            {"ticker": "CCC", "composite": 51.1},
            {"ticker": "DDD", "composite": 44.0},
        ],
    }
    detail.update(overrides)
    return detail


def _build(detail=None, **kw):
    kw.setdefault("universe_size", 502)
    kw.setdefault("metric_meta", METRIC_META)
    kw.setdefault("metric_weights", METRIC_WEIGHTS)
    return ss.build_summary(detail if detail is not None else _detail(), **kw)


def _text(detail=None, **kw) -> str:
    return ss.summary_text(_build(detail, **kw))


def _kinds(summary) -> list:
    return [f["k"] for f in summary]


def _fact(summary, kind) -> str:
    for f in summary:
        if f["k"] == kind:
            return f["t"]
    raise AssertionError(f"no {kind!r} fact in {_kinds(summary)}")


# ---------------------------------------------------------------------------
# 1. It explains. It never advises.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "This looks like a great buy at these levels.",
    "The stock is undervalued versus peers.",
    "An attractive entry point for long-term holders.",
    "We think this is a compelling opportunity.",
    "Investors should avoid this name.",
])
def test_advice_detector_catches_recommendation_language(phrase):
    """The guard has to actually fire, or the clean-summary test proves nothing."""
    assert ss.advice_terms_in(phrase), phrase


@pytest.mark.parametrize("phrase", [
    "Ranks 1st of 502.",
    "Its weakest scored category is Risk at 38 out of 100.",
    "Inside Valuation it sits in the 97th sector percentile on EV/EBITDA (8.99).",
    "The score rests on 18 of 18 metrics.",
])
def test_advice_detector_does_not_fire_on_explanation(phrase):
    assert ss.advice_terms_in(phrase) == []


def test_generated_summary_contains_no_advice_language():
    assert ss.advice_terms_in(_text()) == []


def test_every_edge_case_summary_is_advice_free():
    """Every branch of every sentence builder, checked in one sweep."""
    variants = [
        _detail(),
        _detail(vt=True, flags={**_detail()["flags"], "vt_severity": 42.1}),
        _detail(gt=True, flags={**_detail()["flags"], "gt_severity": 12.0}),
        _detail(flags={**_detail()["flags"], "beneish_flag": True,
                       "channel_stuffing": True, "stale_data": True}),
        _detail(pt_mean=None),
        _detail(pt_mean=18.0),          # target below the price
        _detail(num_analysts=2),        # thin coverage
        _detail(peers=[]),
        _detail(metric_count=12),
        _detail(rank=502, composite=0.2),
    ]
    for detail in variants:
        for delta in (None, {"m1": {"dr": 34, "dc": 2.6}}, {"m1": {"new": True}},
                      {"prev": {"dr": -12, "dc": -1.1}}):
            text = _text(detail, history_delta=delta,
                         history_compare={"m1": {"date": "2026-08-10", "gap_days": 29},
                                          "prev": {"date": "2026-09-07", "gap_days": 1}})
            assert ss.advice_terms_in(text) == [], text


def test_banned_terms_cover_the_examples_in_the_north_star_plan():
    """``plan/dashboard-north-star.md`` names the bad phrasings explicitly."""
    for phrase in ["attractive entry point", "undervalued", "a strong buy"]:
        assert ss.advice_terms_in(phrase), phrase


# ---------------------------------------------------------------------------
# 2. Every figure is arithmetic on the payload
# ---------------------------------------------------------------------------

def test_rank_sentence_quotes_rank_universe_and_composite():
    fact = _fact(_build(), "rank")
    assert "1st of 502" in fact
    assert "74.7" in fact


def test_composite_is_described_as_a_universe_percentile():
    """It is one - ``Composite`` is a cross-sectional percentile rank
    (``SCREENER_OVERVIEW.md``), and a student needs told that once."""
    assert "percentile" in _fact(_build(), "rank")
    assert "above 75% of the universe" in _fact(_build(), "rank")


def test_drivers_are_the_two_largest_contributors_with_exact_points():
    fact = _fact(_build(), "drivers")
    # 21.07 (valuation) and 18.36 (quality) are the two largest contributions.
    assert "Valuation" in fact and "Quality" in fact
    assert "21.1 points" in fact and "18.4 points" in fact
    assert "39.4 of its 74.7 points" in fact   # 21.07 + 18.36
    assert "Momentum" not in fact              # 12.68, third


def test_weakest_is_the_lowest_category_score_not_the_lowest_contribution():
    """Risk scores 37.62 (lowest score) but Investment contributes 2.33 (lowest
    points). A reader asking "what is weak here" means the score."""
    fact = _fact(_build(), "weakest")
    assert "Risk" in fact
    assert "38 out of 100" in fact
    assert "3.8 points" in fact


def test_weakest_explains_what_a_category_score_of_50_means():
    assert "50 is the sector median" in _fact(_build(), "weakest")


def test_best_inputs_come_from_the_leading_category_only():
    """Momentum has no weighted metrics in the fixture; Valuation leads on
    contribution, so the inputs named must be Valuation metrics."""
    fact = _fact(_build(), "best_inputs")
    assert "Inside Valuation" in fact
    assert "EV/EBITDA (8.99)" in fact          # 97.4 pct, highest in valuation
    assert "Earnings Yield (6.7%)" in fact     # 96.8 pct, second
    assert "FCF Yield" not in fact             # 91.2 pct, third


def test_worst_input_is_the_lowest_percentile_weighted_metric_anywhere():
    fact = _fact(_build(), "worst_input")
    assert "Beta (0.70)" in fact               # 17.3 pct, lowest overall
    assert "17th sector percentile" in fact


def test_zero_weight_metrics_are_never_quoted():
    """``pb_ratio`` and ``roe`` carry 0% weight, so they do not move the score
    and naming them would mislead. They are absent from the fixture's `pct`
    map on purpose; adding them must not change the sentences."""
    with_zero = _detail()
    with_zero["pct"] = {**with_zero["pct"], "pb_ratio": 1.0, "roe": 0.5}
    with_zero["raw"] = {**with_zero["raw"], "pb_ratio": 12.0, "roe": 0.02}
    meta = {**METRIC_META,
            "pb_ratio": {"label": "P/B Ratio", "fmt": "ratio", "category": "valuation"},
            "roe": {"label": "ROE", "fmt": "pct", "category": "quality"}}
    text = _text(with_zero, metric_meta=meta)
    assert "P/B Ratio" not in text
    assert "ROE" not in text
    assert "Beta (0.70)" in text                # still the worst weighted input


def test_metric_percentiles_are_always_labelled_sector_relative():
    """They are sector-relative (``factor_engine.compute_sector_percentiles``).
    Calling them plain "percentiles" would publish a false claim about how the
    number was computed."""
    for kind in ("best_inputs", "worst_input"):
        assert "sector percentile" in _fact(_build(), kind)


def test_peer_sentence_places_it_in_the_market_cap_neighbourhood():
    fact = _fact(_build(), "peers")
    assert "3 closest Information Technology names" in fact
    assert "1st of 4" in fact                   # 74.65 beats 57.4, 51.1, 44.0
    assert "BBB at 57.4" in fact


def test_peer_sentence_counts_stocks_that_outrank_it():
    detail = _detail(composite=50.0)
    fact = _fact(_build(detail), "peers")
    assert "3rd of 4" in fact                   # 57.4 and 51.1 are above 50.0


def test_target_sentence_states_the_gap_and_the_analyst_count():
    fact = _fact(_build(), "target")
    assert "$22.05" in fact and "$25.14" in fact
    assert "14.0% above" in fact                # 25.14 / 22.05 - 1
    assert "20 analyst estimates" in fact


def test_target_below_the_price_is_stated_as_below():
    fact = _fact(_build(_detail(pt_mean=18.0)), "target")
    assert "18.4% below" in fact                # 18.00 / 22.05 - 1
    assert "above" not in fact


def test_thin_analyst_coverage_is_disclosed():
    assert "thin basis" in _fact(_build(_detail(num_analysts=2)), "target")
    assert "thin basis" not in _fact(_build(), "target")


def test_missing_price_target_says_so_rather_than_omitting_it():
    """Silence would read as "no gap"; the absence is itself a coverage fact."""
    fact = _fact(_build(_detail(pt_mean=None)), "target")
    assert "No mean analyst price target" in fact


# ---------------------------------------------------------------------------
# 3. Confidence and flags - the uncertainty the north star asks to be visible
# ---------------------------------------------------------------------------

def test_metric_coverage_is_stated():
    assert "rests on 18 of 18 metrics" in _fact(_build(), "confidence")


def test_withheld_categories_are_named_and_the_reweighting_explained():
    """A rejected price series takes out Momentum and Risk together - 23% of
    composite weight off one ``Ticker.history()`` call (changelog 2026-08-26).
    The reader has to be able to see that happened."""
    detail = _detail(metric_count=12)
    detail["cat_scores"] = {**detail["cat_scores"], "momentum": None, "risk": None}
    detail["contrib"] = {**detail["contrib"], "momentum": None, "risk": None}
    fact = _fact(_build(detail), "confidence")
    assert "Momentum and Risk could not be scored" in fact
    assert "reweighted" in fact


def test_withheld_categories_are_excluded_from_drivers_and_weakest():
    detail = _detail()
    detail["cat_scores"] = {**detail["cat_scores"], "risk": None}
    detail["contrib"] = {**detail["contrib"], "risk": None}
    summary = _build(detail)
    assert "Risk" not in _fact(summary, "weakest")
    assert "Investment" in _fact(summary, "weakest")   # 46.67, next lowest


def test_caveats_are_separate_sentences_not_one_run_on():
    """Comma-joining these produced an unreadable run-on for the stocks that
    need them most - FDXF on 2026-09-08: 12 of 18 metrics, three categories
    withheld and 282-day-old filings, all in one sentence."""
    detail = _detail(metric_count=12, eps_mismatch=True,
                     flags={**_detail()["flags"], "stale_data": True,
                            "stmt_age_days": 282})
    detail["cat_scores"] = {**detail["cat_scores"], "momentum": None,
                            "risk": None, "investment": None}
    detail["contrib"] = {**detail["contrib"], "momentum": None,
                         "risk": None, "investment": None}
    fact = _fact(_build(detail), "confidence")
    assert fact.count(". ") >= 3
    assert "282 days old" in fact
    assert "normalised EPS disagree" in fact


def test_trap_flags_are_reported_with_their_severity():
    fact = _fact(_build(_detail(vt=True, flags={**_detail()["flags"],
                                                "vt_severity": 42.1})), "flags")
    assert "value trap" in fact
    assert "severity 42/100" in fact            # matches the badge's X/100


def test_absence_of_flags_is_stated_explicitly():
    assert "no value-trap, growth-trap or accounting flag" in _fact(_build(), "flags")


def test_accounting_flags_are_surfaced():
    fact = _fact(_build(_detail(flags={**_detail()["flags"],
                                       "beneish_flag": True,
                                       "channel_stuffing": True})), "flags")
    assert "Beneish" in fact
    assert "channel-stuffing" in fact


# ---------------------------------------------------------------------------
# 4. The time dimension
# ---------------------------------------------------------------------------

COMPARE = {"m1": {"date": "2026-08-10", "gap_days": 29},
           "prev": {"date": "2026-09-07", "gap_days": 1}}


def test_change_prefers_the_one_month_baseline_over_the_previous_run():
    """Same reason the movers panel does: on 2026-08-25 every material one-day
    mover was a round-trip, while 169 of 193 one-month moves were real trends
    (``plan/dashboard-inventory.md``)."""
    fact = _fact(_build(history_delta={"m1": {"dr": 34, "dc": 2.6},
                                       "prev": {"dr": 1, "dc": 0.1}},
                        history_compare=COMPARE), "change")
    assert "2026-08-10" in fact
    assert "moved up 34 places" in fact
    assert "2026-09-07" not in fact


def test_change_falls_back_to_the_previous_run_when_no_month_baseline():
    fact = _fact(_build(history_delta={"prev": {"dr": -12, "dc": -1.1}},
                        history_compare=COMPARE), "change")
    assert "2026-09-07" in fact
    assert "moved down 12 places" in fact
    assert "composite down 1.1" in fact


def test_a_held_rank_is_reported_as_held_not_omitted():
    fact = _fact(_build(history_delta={"m1": {"dr": 0, "dc": 0.0}},
                        history_compare=COMPARE), "change")
    assert "held its rank" in fact


def test_a_stock_new_to_the_universe_says_so():
    fact = _fact(_build(history_delta={"m1": {"new": True}},
                        history_compare=COMPARE), "change")
    assert "was not in the run of 2026-08-10" in fact


def test_no_history_means_no_change_sentence_not_a_guess():
    assert "change" not in _kinds(_build(history_delta=None))


# ---------------------------------------------------------------------------
# 5. Degradation - a thin stock still gets a valid, shorter summary
# ---------------------------------------------------------------------------

def test_an_almost_empty_detail_produces_a_summary_and_does_not_raise():
    summary = _build({"composite": 30.0, "rank": 400, "cat_scores": {},
                      "contrib": {}, "raw": {}, "pct": {}, "flags": {}})
    assert isinstance(summary, list)
    assert ss.advice_terms_in(ss.summary_text(summary)) == []
    assert "rank" in _kinds(summary)


def test_a_completely_empty_detail_does_not_raise():
    assert _build({}) is not None


@pytest.mark.parametrize("missing", ["composite", "rank", "peers", "price",
                                     "pt_mean", "metric_count", "flags", "pct"])
def test_dropping_any_single_field_never_raises(missing):
    detail = _detail()
    detail.pop(missing, None)
    text = _text(detail)
    assert ss.advice_terms_in(text) == []


def test_facts_that_cannot_be_stated_exactly_are_omitted_not_approximated():
    detail = _detail(peers=[])
    assert "peers" not in _kinds(_build(detail))


def test_every_fact_has_a_kind_and_non_empty_text():
    for fact in _build():
        assert fact["k"] and isinstance(fact["k"], str)
        assert fact["t"] and fact["t"].strip().endswith(".")


# ---------------------------------------------------------------------------
# 6. Formatting helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
    (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (23, "23rd"),
    (100, "100th"), (101, "101st"), (111, "111th"), (502, "502nd"),
])
def test_ordinals(n, expected):
    assert ss._ordinal(n) == expected


@pytest.mark.parametrize("p,expected", [
    (0.0, "the lowest"), (0.4, "the lowest"), (0.6, "the 1st"),
    (17.3, "the 17th"), (99.4, "the 99th"), (99.6, "the highest"),
    (100.0, "the highest"),
])
def test_percentile_phrasing_handles_both_extremes(p, expected):
    """A sector percentile really does reach 0 and 100, and "the 0th sector
    percentile" reads like a bug rather than "worst in its sector"."""
    assert ss._pctile_phrase(p) == expected


@pytest.mark.parametrize("value,fmt,expected", [
    (0.081, "pct", "8.1%"), (8.99, "ratio", "8.99"), (7.0, "int", "7"),
    (None, "pct", "n/a"), (-0.276, "pct", "-27.6%"),
])
def test_metric_formatting_matches_the_front_end(value, fmt, expected):
    """Mirrors ``fmtMetric`` in the emitted JS: the prose and the metric table
    below it must not disagree about the same number."""
    assert ss._fmt_metric(value, fmt) == expected
