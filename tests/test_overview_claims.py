"""`SCREENER_OVERVIEW.md` must describe the tool that exists.

It is the canonical public methodology reference and `generate_dashboard.py`
embeds it verbatim into `index.html`, which GitHub Pages serves. Prose there is
what a student reads to decide whether to trust a number.

Three defects found on 2026-09-25, all live on the public site, all created by
one session on 2026-09-10 that changed the weights and left the prose behind:

1. `fy1_revision_3m` - the **heaviest metric in the Revisions category at
   35%** - had an **empty** "What It Measures" cell, and was the only row in
   the document still labelled with a raw snake_case identifier.
2. The category's "Why these?" paragraph said "Analyst Surprise gets the
   highest weight". It is 15% against the revision metric's 35%, contradicted
   by the table two lines above it.
3. Limitation 5 and a trailing note both said a forward-EPS-consensus-change
   metric was "not feasible with yfinance" and proposed it as a future
   enhancement via FactSet or Refinitiv - for a metric that had been live and
   most-weighted for 15 days.

A fourth, unrelated and older: limitation 1 told the reader "approximately
10-25% of tickers may fail to fetch on a given run". Measured over the 18
scheduled runs from 2026-09-02 to 2026-09-25: **0 failures across 9,036
ticker-fetches**. The figure dated from the launch-period rate-limiting era.

These tests read `config.yaml` rather than hard-coding weights, so they follow
the configuration instead of pinning today's numbers.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
OVERVIEW = ROOT / "SCREENER_OVERVIEW.md"
LIVE_PAGE = ROOT / "index.html"


@pytest.fixture(scope="module")
def overview() -> str:
    return OVERVIEW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def page() -> str:
    if not LIVE_PAGE.exists():
        pytest.skip("index.html not present")
    return LIVE_PAGE.read_text(encoding="utf-8", errors="replace")


def _table_rows(text: str):
    """(line number, cells) for every markdown table row that is not a
    header separator."""
    for n, line in enumerate(text.splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        yield n, cells


def _section(text: str, heading_contains: str) -> str:
    """The text from a heading containing `heading_contains` to the next
    heading of the same or higher level."""
    heads = [(m.start(), len(m.group(1)), m.group(0))
             for m in re.finditer(r"^(#{2,4}) .*$", text, re.M)]
    for i, (start, level, head) in enumerate(heads):
        if heading_contains.lower() not in head.lower():
            continue
        end = len(text)
        for start2, level2, _ in heads[i + 1:]:
            if level2 <= level:
                end = start2
                break
        return text[start:end]
    raise AssertionError(f"no heading containing {heading_contains!r}")


# --------------------------------------------------------------------------
# 1. No metric may go undescribed
# --------------------------------------------------------------------------

class TestEveryTableCellIsFilled:

    def test_no_empty_cells_in_any_table(self, overview):
        """An empty cell is how `fy1_revision_3m` shipped undescribed at 35%
        weight. A blank renders as a blank on the public page."""
        empty = [(n, cells) for n, cells in _table_rows(overview)
                 if any(c == "" for c in cells)]
        assert empty == [], (
            "empty table cell(s) in SCREENER_OVERVIEW.md at line(s) "
            + ", ".join(str(n) for n, _ in empty)
        )

    def test_no_metric_row_is_labelled_with_a_raw_identifier(self, overview, cfg):
        """Metric rows carry plain-English names. A snake_case key in the
        first column means a metric was added to the table and never named."""
        keys = {k for cat in cfg["metric_weights"].values() for k in cat}
        offenders = []
        for n, cells in _table_rows(overview):
            label = cells[0].strip("* ").strip()
            if label in keys and "_" in label:
                offenders.append((n, label))
        assert offenders == [], (
            "raw metric identifier(s) used as a display name: "
            + ", ".join(f"L{n}:{k}" for n, k in offenders)
        )


# --------------------------------------------------------------------------
# 2. The prose must agree with the weights it describes
# --------------------------------------------------------------------------

class TestRevisionsSectionMatchesConfig:

    def test_listed_weights_match_config_exactly(self, overview, cfg):
        """The multiset of percentages in the Revisions table must equal the
        non-zero weights in `config.yaml`. Compared as a multiset because the
        display names differ from the config keys by design."""
        section = _section(overview, "Revisions")
        listed = sorted(int(m) for m in
                        re.findall(r"^\|[^|]+\|\s*(\d+)%\s*\|", section, re.M))
        configured = sorted(v for v in cfg["metric_weights"]["revisions"].values() if v)
        assert listed == configured, (
            f"Revisions table lists {listed}, config has {configured}"
        )

    def test_the_metric_named_highest_weighted_actually_is(self, overview, cfg):
        """The false claim was "Analyst Surprise gets the highest weight"
        while the revision metric carried more than twice as much."""
        weights = cfg["metric_weights"]["revisions"]
        top = max(weights, key=lambda k: weights[k])
        assert top == "fy1_revision_3m", (
            "this test's expectation is tied to the FY1 revision metric being "
            f"heaviest; config now says {top} - update the assertion below"
        )
        section = _section(overview, "Revisions")
        claim = re.search(r"([^.]*highest weight[^.]*)\.", section)
        assert claim is not None, "no 'highest weight' sentence in the section"
        sentence = claim.group(1)
        assert re.search(r"FY1|Revision", sentence), (
            f"the highest-weight sentence names something other than the "
            f"heaviest metric: {sentence.strip()!r}"
        )
        assert "Analyst Surprise gets the highest weight" not in section

    def test_does_not_call_the_live_revision_metric_infeasible(self, overview, cfg):
        """Limitation 5 and a trailing note both claimed the metric was
        impossible with the free feed while it was scored at 35%."""
        assert cfg["metric_weights"]["revisions"]["fy1_revision_3m"] > 0, (
            "fy1_revision_3m is no longer scored; this test needs revisiting"
        )
        for phrase in ("is not feasible with yfinance",
                       "cannot include the single most powerful revisions signal",
                       "Future enhancement: integrate I/B/E/S"):
            assert phrase not in overview, (
                f"overview still claims the live revision metric is absent: "
                f"{phrase!r}"
            )

    def test_the_real_residual_limit_is_still_stated(self, overview):
        """The honest limitation is depth, not absence: yfinance's estimate
        history reaches ~90 days, so revision *persistence* is out of reach.
        Removing the false claim must not remove the true one."""
        assert "90 days" in overview or "90-day" in overview


# --------------------------------------------------------------------------
# 3. The fetch-reliability figure must be the measured one
# --------------------------------------------------------------------------

class TestFetchReliabilityClaim:

    def test_does_not_claim_a_double_digit_fetch_failure_rate(self, overview):
        """Measured 0 failures in 9,036 ticker-fetches over 18 runs. Telling a
        reader that a quarter of the universe may be missing trains them to
        excuse a real gap as normal."""
        assert not re.search(
            r"\d{1,2}\s*-\s*\d{1,2}%\s+of\s+tickers\s+may\s+fail", overview), (
            "the stale 10-25% fetch-failure claim is back"
        )

    def test_states_the_measured_rate_with_its_period(self, overview):
        section = _section(overview, "Limitations")
        assert "9,036" in section and "0 fetch failures" in section, (
            "the measured fetch-reliability figure and its sample are missing"
        )

    def test_still_says_rate_limiting_is_possible(self, overview):
        """Correcting an overstatement must not become the opposite
        overstatement: it is a free unofficial API."""
        section = _section(overview, "Limitations")
        assert "rate limiting" in section.lower()

    def test_points_at_the_gate_that_makes_the_claim_safe(self, overview):
        """The reason a reader can rely on published numbers is not that
        fetches never fail - it is that a bad run is discarded."""
        section = _section(overview, "Limitations")
        assert "check_run_health" in section


# --------------------------------------------------------------------------
# 4. The live page must carry the same text
# --------------------------------------------------------------------------

class TestLivePageCarriesTheCorrections:
    """`index.html` embeds the overview. Correcting the markdown and not
    regenerating leaves the public site saying the old thing - which is how
    all four defects above stayed visible."""

    def test_page_does_not_carry_the_stale_fetch_claim(self, page):
        assert not re.search(
            r"\d{1,2}\s*-\s*\d{1,2}%\s+of\s+tickers\s+may\s+fail", page)

    def test_page_does_not_call_the_revision_metric_infeasible(self, page):
        assert "is not feasible with yfinance" not in page

    def test_page_describes_the_heaviest_revisions_metric(self, page):
        assert "FY1 EPS Revision" in page, (
            "index.html predates the overview correction - regenerate it"
        )

    def test_page_states_the_measured_fetch_reliability(self, page):
        assert "9,036" in page
