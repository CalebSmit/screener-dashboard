"""The published methodology must name the position-weighting scheme it uses.

Why this file exists
--------------------
``SCREENER_OVERVIEW.md`` is the canonical public methodology reference, and
``generate_dashboard.py`` embeds it verbatim into ``index.html`` -- the page
GitHub Pages serves.  Until 2026-09-23 the line describing how the model
portfolio is weighted was built by a **two-branch ternary over a four-option
setting**::

    {'Equal weight (each stock gets ~25%)' if weighting == 'equal'
     else 'Risk-parity (inverse-volatility weighting - lower-volatility
           stocks get more weight)'}

``config.yaml`` has shipped ``portfolio.weighting: 'score'`` since launch, so
that ternary took its ``else`` branch on every run, and the live site told
every reader the portfolio was inverse-volatility weighted.  It was
composite-score weighted, which tilts the *opposite* way: toward the
highest-scoring names rather than the calmest ones.  The same false sentence
was live in ``index.html`` on 2026-09-23 and in every published
``SCREENER_OVERVIEW.md`` before it.

Two separate things are pinned here, because they failed together and could
fail apart:

1. **The description is a mapping, not a binary.**  Every scheme
   ``schemas.PortfolioConfig.validate_weighting`` accepts gets its own branch,
   and only the inverse-volatility branch may mention inverse volatility.  An
   unrecognised scheme names itself rather than borrowing another scheme's
   description -- silently inheriting a wrong description is what happened.
2. **What is published matches what is configured.**  ``SCREENER_OVERVIEW.md``
   and ``index.html`` are checked against the live ``config.yaml`` rather than
   against a hard-coded string, so the test follows the config if it changes
   and fails if the artifacts fall behind it.

Also pinned: ``max_position_pct`` is disclosed as non-binding under equal
weighting.  25 equal-weighted names are 4.00% each against a 5% cap, so it
cannot fire below 20 holdings, and it never fired in 41 run dates under
``'score'`` either (``research/measurements/2026-09-21-position-sizing-dispersion.py``).
A parameter that reads as a live safety control and is not one is the same
failure shape as the always-firing bank-metric alarm fixed 2026-09-01.  It is
kept -- it becomes live if ``num_stocks`` falls -- and it is now labelled.

See ``METHODOLOGY_CHANGELOG.md`` 2026-09-23 and
``research/2026-09-21-position-sizing-and-how-much.md``.
"""

import re
from pathlib import Path

import pytest
import yaml

from run_screener import _max_pos_note, weighting_description

ROOT = Path(__file__).resolve().parent.parent
OVERVIEW = ROOT / "SCREENER_OVERVIEW.md"
INDEX = ROOT / "index.html"
CONFIG = ROOT / "config.yaml"

# Every scheme schemas.PortfolioConfig.validate_weighting accepts.
VALID_SCHEMES = ("equal", "inverse_vol", "score", "markowitz")

# The exact wording that was published for two schemes it does not describe.
INVERSE_VOL_PHRASE = "inverse-volatility"


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _portfolio_cfg() -> dict:
    return _cfg().get("portfolio", {})


def _normalise(text: str) -> str:
    """Collapse whitespace and unify dashes so markdown/HTML compare equal."""
    text = text.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 1. The description is a mapping over every valid scheme
# ---------------------------------------------------------------------------

def test_every_valid_scheme_has_its_own_branch():
    """No valid scheme falls through to the unrecognised-scheme text."""
    for scheme in VALID_SCHEMES:
        desc = weighting_description(scheme, 25)
        assert "unrecognised scheme" not in desc, (
            f"{scheme!r} has no branch in weighting_description()"
        )


def test_the_four_descriptions_are_all_distinct():
    """The bug was two schemes sharing one description. They may not."""
    descs = [weighting_description(s, 25) for s in VALID_SCHEMES]
    assert len(set(descs)) == len(descs), f"duplicate descriptions: {descs}"


@pytest.mark.parametrize("scheme", ["equal", "score", "markowitz"])
def test_only_the_inverse_vol_branch_claims_inverse_volatility(scheme):
    """This is the exact defect: 'score' described as inverse-volatility."""
    assert INVERSE_VOL_PHRASE not in weighting_description(scheme, 25).lower()


def test_inverse_vol_branch_still_says_inverse_volatility():
    for scheme in ("inverse_vol", "risk_parity"):
        assert INVERSE_VOL_PHRASE in weighting_description(scheme, 25).lower()


def test_score_description_says_score_proportional():
    desc = weighting_description("score", 25).lower()
    assert "composite" in desc and "score" in desc


def test_equal_description_reports_the_actual_slice():
    assert "~4%" in weighting_description("equal", 25)
    assert "~5%" in weighting_description("equal", 20)
    assert "~10%" in weighting_description("equal", 10)


def test_markowitz_discloses_its_fallback():
    """portfolio_constructor falls back to 'score' when returns are missing."""
    desc = weighting_description("markowitz", 25).lower()
    assert "experimental" in desc
    assert "fall" in desc  # "falls back"


def test_unrecognised_scheme_names_itself_rather_than_borrowing_one():
    desc = weighting_description("some_new_scheme", 25)
    assert "some_new_scheme" in desc
    assert INVERSE_VOL_PHRASE not in desc.lower()
    assert "Equal weight" not in desc


@pytest.mark.parametrize("scheme", ["EQUAL", " Score ", "Inverse_Vol"])
def test_scheme_matching_is_case_and_space_insensitive(scheme):
    assert "unrecognised scheme" not in weighting_description(scheme, 25)


def test_missing_scheme_falls_back_to_equal():
    """schemas.PortfolioConfig defaults `weighting` to 'equal'."""
    assert weighting_description("", 25) == weighting_description("equal", 25)
    assert weighting_description(None, 25) == weighting_description("equal", 25)


# ---------------------------------------------------------------------------
# 2. The published artifacts match the live config
# ---------------------------------------------------------------------------

def test_overview_states_the_configured_scheme():
    pcfg = _portfolio_cfg()
    expected = weighting_description(
        pcfg.get("weighting", "equal"), pcfg.get("num_stocks", 25)
    )
    text = _normalise(OVERVIEW.read_text(encoding="utf-8"))
    assert _normalise(expected) in text, (
        "SCREENER_OVERVIEW.md does not describe the configured weighting "
        f"scheme {pcfg.get('weighting')!r}. Re-run the screener to regenerate it."
    )


def test_overview_does_not_claim_a_scheme_it_is_not_using():
    pcfg = _portfolio_cfg()
    if pcfg.get("weighting") in ("inverse_vol", "risk_parity"):
        pytest.skip("inverse-vol is configured, so the phrase is truthful")
    text = OVERVIEW.read_text(encoding="utf-8").lower()
    assert "**weighting:** risk-parity" not in text, (
        "SCREENER_OVERVIEW.md describes the portfolio as risk-parity weighted "
        "while config.yaml configures something else -- the 2026-09-23 defect."
    )


@pytest.mark.skipif(not INDEX.exists(), reason="index.html not built")
def test_live_page_states_the_configured_scheme():
    """index.html embeds the overview verbatim and is what Pages serves."""
    pcfg = _portfolio_cfg()
    expected = weighting_description(
        pcfg.get("weighting", "equal"), pcfg.get("num_stocks", 25)
    )
    html = _normalise(INDEX.read_text(encoding="utf-8", errors="replace"))
    if "<strong>Weighting:</strong>" not in html:
        pytest.skip("the embedded methodology section is not present")
    assert _normalise(expected) in html, (
        "the published page describes a weighting scheme config.yaml does not "
        "configure. Regenerate the dashboard."
    )


def test_limitations_section_does_not_call_score_a_volatility_scheme():
    """Limitation 7 listed `score` as using 'single-name volatility only'.

    Score weighting takes no volatility input at all; it uses the composite.
    """
    text = OVERVIEW.read_text(encoding="utf-8")
    assert "single-name volatility only (`inverse_vol` / `score`)" not in text


# ---------------------------------------------------------------------------
# 3. The configured default, and why
# ---------------------------------------------------------------------------

def test_default_weighting_is_equal():
    """Changed from 'score' on 2026-09-23. See the changelog before reverting.

    DeMiguel, Garlappi & Uppal (2009) RFS 22(5): across 14 optimisation models
    and 7 datasets none consistently beat 1/N out of sample.  Chopra & Ziemba
    (1993) via Ziemba & MacLean (2011): errors in expected returns do ~20x the
    damage of covariance errors.  Score weighting sizes by an expected-return
    estimate this system has 3 effective observations on.
    """
    assert _portfolio_cfg().get("weighting") == "equal"


def test_weighting_is_a_scheme_the_schema_accepts():
    assert _portfolio_cfg().get("weighting") in VALID_SCHEMES


def test_valid_schemes_here_match_the_schema_validator():
    """If the schema gains a scheme, this file must gain a branch for it."""
    source = (ROOT / "schemas.py").read_text(encoding="utf-8")
    match = re.search(r'valid = \{([^}]*)\}\s*\n\s*if v not in valid', source)
    assert match, "could not locate validate_weighting's accepted set"
    declared = {s.strip().strip("\"'") for s in match.group(1).split(",") if s.strip()}
    assert declared == set(VALID_SCHEMES), (
        f"schemas.py accepts {declared}, this file knows {set(VALID_SCHEMES)}"
    )


# ---------------------------------------------------------------------------
# 4. max_position_pct is disclosed as non-binding where it cannot bind
# ---------------------------------------------------------------------------

def test_cap_note_fires_for_equal_weighting_below_the_threshold():
    note = _max_pos_note("equal", 25, 5.0)
    assert "not binding" in note
    assert "4.00%" in note
    assert "below 20 holdings" in note


def test_cap_note_threshold_tracks_the_configured_cap():
    assert "below 20 holdings" in _max_pos_note("equal", 25, 5.0)
    assert "below 17 holdings" in _max_pos_note("equal", 25, 6.0)
    assert "below 10 holdings" in _max_pos_note("equal", 25, 10.0)


def test_cap_note_is_silent_when_the_cap_can_actually_bind():
    """Non-equal schemes produce dispersed weights; do not promise anything."""
    for scheme in ("score", "inverse_vol", "markowitz"):
        assert _max_pos_note(scheme, 25, 5.0) == ""


def test_cap_note_is_silent_when_the_cap_is_infeasible():
    """10 equal-weighted names are 10% each, above a 5% cap.

    portfolio_constructor already warns about this case; the overview must not
    additionally claim the cap is 'not binding'.
    """
    assert _max_pos_note("equal", 10, 5.0) == ""


def test_cap_note_survives_degenerate_inputs():
    assert _max_pos_note("equal", 0, 5.0) == ""
    assert _max_pos_note("equal", 25, 0.0) == ""


def test_overview_discloses_the_cap_is_not_binding():
    pcfg = _portfolio_cfg()
    note = _max_pos_note(
        pcfg.get("weighting", "equal"),
        pcfg.get("num_stocks", 25),
        float(pcfg.get("max_position_pct", 5.0)),
    )
    if not note:
        pytest.skip("the configured scheme can bind the cap")
    text = _normalise(OVERVIEW.read_text(encoding="utf-8"))
    assert _normalise(note) in text


def test_max_position_pct_is_still_present_and_unchanged():
    """The cap is inert, not wrong. Deleting it is the mistake to avoid.

    It becomes live the moment `num_stocks` falls below 20 or a dispersed
    weighting scheme is selected, which is why 2026-09-23 documented it
    instead of removing it.
    """
    assert float(_portfolio_cfg().get("max_position_pct")) == 5.0
