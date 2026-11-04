"""The Investment section's new paragraphs, held to the code and the sources that make them true.

Claim `overview.investment_standalone`, added 2026-11-04 when the proposal in
`research/2026-11-02-investment-inside-quality.md` was decided (REFUSE the fold; state the
practitioner convention on the page). The section had no large-company caveat where Size and
Momentum both carry thorough ones - `research/2026-10-19-revenue-growth-vs-asset-growth.md`
section 11.2 queued four page additions and this ships three of them.

These read **the generator's output**, not the committed `SCREENER_OVERVIEW.md` (CLAUDE.md
rule 10): the 02:00 run rewrites that file, so a test reading it passes on a hand-edit and says
nothing about what the next run publishes.

Every figure in the section is one of three kinds, and each kind is checked differently:

* **the engine's own** - the measure `compute_metrics` computes, and the weight in `config.yaml`;
* **the Factor Lab's** - recomputed here from the lab's data with a band, the
  `tests/test_large_company_evidence.py` pattern, so a refresh that changes the picture fails
  this instead of leaving the page wrong;
* **a named measurement's over a named set of runs** - checked against the committed JSON or
  script that produced it. These are *not* asserted of today's run on purpose: a check on a
  claim runs in the 02:00 publish gate, where failing discards the night, so a count that moves
  with coverage must not be stated as a property of every run (CLAUDE.md row `data->red`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import factor_lab as fl

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "research" / "2026-11-02-investment-inside-quality.md"
PAIR_NOTE = ROOT / "research" / "2026-10-19-revenue-growth-vs-asset-growth.md"
SPREAD_JSON = ROOT / "research" / "measurements" / "2026-10-19-spread-exposure-over-time.json"
FOLD_SCRIPT = ROOT / "research" / "measurements" / "2026-11-04-investment-fold-review.py"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def section(cfg) -> str:
    import run_screener
    text = run_screener.build_screener_overview(cfg)
    start = text.index("### 8. Investment")
    return text[start: text.index("## Bank-specific scoring", start)]


def test_the_section_names_the_measure_the_engine_computes(section, cfg):
    """Total-asset growth, not an assets-per-share measure - and the weight is config's."""
    assert "**Which asset-growth measure, and why it matters.**" in section
    assert "year-on-year percentage change in *total assets*" in section
    assert "Cooper, Gulen & Schill (2008)" in section
    # The distinction the paragraph turns on, stated in the direction the paper states it.
    assert "the same variable Fama & French sort CMA on" in section
    assert "not** the assets-per-share version" in section
    assert "a buyback shrinks assets per share without shrinking the asset base" in section
    # Fama & French's own definition, quoted. Verified against data/papers/ff2015.pdf 2026-11-04.
    assert ("Inv is the growth of total assets for the fiscal year ending in t-1 divided by "
            "total assets at the end of t-2") in section
    # The engine's metric really is the total-assets version: asset_growth carries Investment's
    # whole weight, and no per-share asset metric is weighted anywhere.
    inv = cfg["metric_weights"]["investment"]
    assert inv["asset_growth"] == 100, inv
    assert f"### 8. Investment ({cfg['factor_weights']['investment']}% of final score)" in section


def test_the_large_company_reading_is_the_labs_own(section):
    """The lab's large-company investment factor, recomputed, must still say what the page says.

    Bands rather than literals: the lab's data gains a month at each refresh, so the figures
    drift slightly. A band wide enough for that drift still fails if the sign or the
    significance changes, which is the state the page would be wrong in.
    """
    assert "**Among large companies.**" in section
    assert "1.03 and 2.00 standard errors from zero" in section
    assert "0.07% a month (t 1.03)" in section
    assert "almost all the average value of CMA is from small stocks" in section
    assert "lab.html#big-and-small" in section
    assert "Asness (2014)" in section and "redundant one" in section
    assert PAIR_NOTE.name in section

    split = fl.load_published()["split"]
    ff_sample = fl.split_run(split, "196307", "201312")["factors"]["investment"]["big"]
    since_2003 = fl.split_run(split, "200301", "209912")["factors"]["investment"]["big"]

    # "averaged about 0.2% a month over Fama & French's own sample to 2013 - reproducing their
    # figure". Their Table 4 CMA_B for the 2x3 sorts is 0.22 a month at t 2.00.
    assert "about 0.2% a month over Fama & French's own sample to 2013" in section
    assert 0.15 <= ff_sample["mean"] < 0.25, ff_sample
    assert abs(ff_sample["mean"] - 0.22) < 0.05, ff_sample

    # "has averaged slightly *below* zero since 2003 ... t-statistic under 1 in absolute terms"
    assert "below* zero since 2003" in section
    assert "t-statistic under 1 in absolute terms" in section
    assert since_2003["mean"] < 0, since_2003
    assert abs(since_2003["t"]) < 1, since_2003

    # The claim is comparative - the weak half. It fails if the recent era stops being weaker.
    assert ff_sample["mean"] > since_2003["mean"]


def test_the_joint_position_is_the_replays_range(section):
    """"between +0.89 and +0.91 in every one of them" is the 40-run replay's own range."""
    assert "**What this category does jointly with Growth.**" in section
    assert "revenue outgrowing the balance sheet" in section
    assert "between +0.89 and +0.92 in every one of them" in section
    assert "under +0.14" in section
    assert "FTSE Russell scores change in asset turnover directly" in section

    data = json.loads(SPREAD_JSON.read_text(encoding="utf-8"))
    spread = [row["pair_vs_spread"] for row in data["series"]]
    revenue = [row["pair_vs_revenue"] for row in data["series"]]
    assert data["runs"] == len(spread) == 40, data["runs"]
    assert "40 published runs" in section
    # Every run inside the stated range, not just the average - which is what "in every one of
    # them" asserts, and the only reading that makes the sentence falsifiable.
    assert min(spread) >= 0.89 and max(spread) <= 0.92, (min(spread), max(spread))
    # and tight enough to be a real bound: the stated range must not be loose padding
    assert max(spread) > 0.905, max(spread)
    assert max(revenue) < 0.14, max(revenue)
    # and the pair tracks the spread far better than it tracks either leg's raw value
    assert min(spread) > max(revenue) + 0.5


def test_the_fold_measurement_is_the_committed_one(section):
    """The fold result is attributed to a named run range and a committed script."""
    assert "**Why it is its own category, when most practitioners file it under Quality.**" in section
    assert "weight that keeps each metric's share of the composite identical" in section
    assert "455 of about 500 composites unchanged and the top 25 untouched every time" in section
    assert "2026-10-22 to 2026-11-04" in section
    assert "eight published runs" in section
    # The page points at work a reader can rerun, and it exists.
    assert FOLD_SCRIPT.exists(), FOLD_SCRIPT
    assert NOTE.name in section and FOLD_SCRIPT.name in section
    # The review's own numbers are recorded in the note, so the page is not the only record.
    note = NOTE.read_text(encoding="utf-8")
    assert "455 of 499" in note
    assert "**Proposal status: REFUSED" in note or "Proposal status: DECIDED" in note


def test_the_practice_figures_are_the_sources_own(section):
    """The three practice figures, as read from the primary documents on 2026-11-04."""
    # MSCI FaCS Methodology, January 2018, p. 8 (weights) and p. 9 (the Microsoft worked example):
    # "we assigned a 25% weight to each of the three SES quality factors".
    assert "Investment Quality a 25% share of its Quality factor group" in section
    # Research Affiliates / Hsu, Kalesnik & Kose (2019): quality = profitability minus investment.
    assert "profitability *minus* investment" in section
    # Hsu, Kalesnik & Kose (2019) FAJ 75(2), p. 57, quoted verbatim.
    assert ("most models considered in academia include profitability and investment as "
            "separate factors") in section
    # ERI Scientific Beta, May 2017: Exhibit 14 (p. 35) gives -0.18 on the 500 largest US
    # stocks, 1975-2015; the conclusion is on p. 34, quoted verbatim.
    assert "**-0.18**" in section
    assert "500 largest US stocks over 1975-2015" in section
    assert ("cannot be combined in a composite manner to represent a single risk factor"
            in section)
    # The split in practice is the point: the section must not claim practice is unanimous.
    assert "Nine of eleven provider families" in section
    assert "The academic models do the opposite" in section


def test_the_section_carries_a_caveat_like_size_and_momentum(cfg):
    """The defect section 11.2 named: the same standard applied unevenly.

    Size and Momentum each carry a large-company reading. Investment carried none until today.
    """
    import run_screener
    text = run_screener.build_screener_overview(cfg)
    # Each of the four categories whose premium is weaker among large companies than in the
    # full cross-section must say so, in its own words. Matching each section's own marker
    # rather than one phrase: a shared phrase would let a section pass by quoting a neighbour.
    expected = (
        ("### 1. Valuation", "### 2.", "mostly a small-company result"),
        ("### 4. Momentum", "### 5.", "not on the last quarter-century"),
        ("### 7. Size", "### 8.", "largest two market-cap deciles"),
        ("### 8. Investment", "## Bank-specific scoring",
         "almost all the average value of CMA is from small stocks"),
    )
    for start, end, marker in expected:
        i = text.index(start)
        sec = text[i: text.index(end, i)]
        assert marker in sec, (start, marker)
        # and each points the reader at the research behind it
        assert "research/" in sec, start
