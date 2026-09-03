"""The `log` in `size_log_mcap` does not affect any score, and the docs must say so.

Why this file exists
--------------------
Until 2026-09-03, ``SCREENER_OVERVIEW.md`` -- the canonical public methodology
reference, served to an audience that includes college investment clubs --
justified the size metric with this sentence:

    "Using the log transform compresses the enormous range of market caps
    ($2B to $3T+) into a more linear scale that ranks sensibly."

That is false, and the *same document* contradicts it 44 lines later, where it
correctly explains that "Every metric below is scored by its **rank** within its
sector, and a rank does not care how far away an outlier is."

Both cannot be true.  ``compute_sector_percentiles`` ranks the metric one step
after it is computed, and a rank is invariant to every order-preserving
transformation.  Measured on the 2026-09-03 published run (502 stocks)::

    max |rank(-log mcap) - rank(-mcap)|      = 0.0000000000
    max |rank(-log mcap) - rank(-sqrt mcap)| = 0.0000000000

So the log changes nothing about any stock's score.  It is a readability choice
for the stored number, and the document was presenting it as the mechanism that
keeps the size tilt gentle.

Why that mattered enough to fix
-------------------------------
The log *is* the mechanism by which the practitioner standard keeps a size tilt
gentle -- MSCI's Low Size indexes weight holdings in proportion to 1/ln(mcap).
Imported honestly onto this universe, that compression turns a **798x** spread
in market cap into a **1.295x** spread in weight.  This screener instead scores
size linear-in-rank, so a 10x cap ratio becomes a 57-point score gap (CAT $364B
scores 1; UAL $35B scores 59).

The screener's design is defensible -- it is close to the bet S&P 500 Equal
Weight makes -- but it is a *stronger* bet than "log market cap" implies, and a
tool whose product is explainability may not describe its own transfer function
incorrectly.  See ``research/2026-08-31-size-factor-in-a-large-cap-universe.md``
Finding B, and ``NIGHTLY_LOG.md`` 2026-09-03 for why the tilt was documented
rather than compressed.

These tests pin the fact, not the prose: the invariance is asserted against the
real ``compute_sector_percentiles``, so if the pipeline ever stops rank-scoring
size the first two tests fail and the documentation genuinely needs rewriting
again.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import factor_engine as fe
from run_screener import _SIZE_DESCRIPTIONS

ROOT = Path(__file__).resolve().parent.parent
OVERVIEW = ROOT / "SCREENER_OVERVIEW.md"

# The exact claim that was removed. If it ever comes back, so does the bug.
FALSE_CLAIM = "into a more linear scale that ranks sensibly"


def _universe(mcaps, sector="Technology"):
    """A frame shaped like the pipeline's, carrying one sector of >= 10 names."""
    n = len(mcaps)
    df = pd.DataFrame(
        {
            "Ticker": [f"T{i:03d}" for i in range(n)],
            "Sector": [sector] * n,
        }
    )
    for col in fe.METRIC_COLS:
        df[col] = np.nan
    return df


# Spread over three orders of magnitude, the way the real universe is
# ($6.8B to $5,419B on the 2026-09-03 run).
MCAPS = np.array(
    [6.8e9, 12.0e9, 21.0e9, 35.0e9, 58.0e9, 104.0e9, 180.0e9,
     268.0e9, 364.0e9, 700.0e9, 1034.0e9, 2750.0e9, 4743.0e9, 5419.0e9]
)


@pytest.mark.parametrize(
    "alt_name,alt_fn",
    [
        ("no transform at all", lambda mc: -mc),
        ("square root", lambda mc: -np.sqrt(mc)),
        ("cube root", lambda mc: -np.cbrt(mc)),
        ("log base 10", lambda mc: -np.log10(mc)),
    ],
)
def test_log_is_inert_against_any_monotone_transform(alt_name, alt_fn):
    """The pipeline's size percentile is identical under any order-preserving
    re-expression of market cap. This is what makes the old doc claim false."""
    logged = _universe(MCAPS)
    logged["size_log_mcap"] = -np.log(MCAPS)
    logged = fe.compute_sector_percentiles(logged)

    alt = _universe(MCAPS)
    alt["size_log_mcap"] = alt_fn(MCAPS)
    alt = fe.compute_sector_percentiles(alt)

    diff = np.nanmax(
        np.abs(logged["size_log_mcap_pct"] - alt["size_log_mcap_pct"])
    )
    assert diff == pytest.approx(0.0, abs=1e-9), (
        f"size percentile changed by {diff} when the transform was swapped for "
        f"{alt_name}. If this fails the log is no longer inert and "
        f"SCREENER_OVERVIEW.md's size section must be rewritten."
    )


def test_size_percentile_is_linear_in_rank_not_compressed():
    """Equally-spaced *ranks* produce equally-spaced scores, however unequally
    spaced the underlying market caps are. This is the tilt's real shape."""
    df = _universe(MCAPS)
    df["size_log_mcap"] = -np.log(MCAPS)
    df = fe.compute_sector_percentiles(df)

    pct = df["size_log_mcap_pct"].sort_values().to_numpy()
    gaps = np.diff(pct)
    assert np.allclose(gaps, gaps[0]), (
        "size percentiles are not evenly spaced across ranks; the tilt is no "
        f"longer linear-in-rank. gaps={gaps}"
    )
    # And the spread is the full 0-100 range, not MSCI's near-flat one.
    assert pct.max() - pct.min() > 90.0


def test_msci_style_compression_is_near_flat_on_this_universe():
    """Pins the comparison the documentation now makes.

    MSCI Low Size weights holdings proportional to 1/ln(mcap). Over a cap range
    like the real universe's, that is almost no tilt at all -- which is why
    'compress toward MSCI' is not a middle path between keeping and dropping
    the size category, but a disguised way of dropping it.
    """
    inv_ln = 1.0 / np.log(MCAPS)
    weights = inv_ln / inv_ln.sum()
    cap_ratio = MCAPS.max() / MCAPS.min()
    weight_ratio = weights.max() / weights.min()

    assert cap_ratio > 500, "test universe no longer spans a realistic cap range"
    assert weight_ratio < 1.5, (
        f"MSCI-shape weight ratio {weight_ratio:.3f}x is larger than expected; "
        "the documented 1.295x comparison needs rechecking."
    )


def test_overview_does_not_repeat_the_false_claim():
    md = OVERVIEW.read_text(encoding="utf-8")
    assert FALSE_CLAIM not in md, (
        "SCREENER_OVERVIEW.md again claims the log transform makes market caps "
        "'rank sensibly'. Ranking is invariant to the log -- see this module's "
        "docstring. Fix the generator in run_screener.py, not the .md."
    )


def test_overview_states_the_log_does_not_affect_the_score():
    md = OVERVIEW.read_text(encoding="utf-8")
    size_section = md.split("### 7. Size")[1].split("### 8.")[0]
    assert "does not compress the tilt" in size_section
    assert "identical to ten decimal places" in size_section, (
        "the size section no longer shows the reader the evidence that the log "
        "is inert"
    )


def test_overview_discloses_the_tilt_strength_and_its_weakness():
    """The two things Monday's research found and the doc used to omit."""
    md = OVERVIEW.read_text(encoding="utf-8")
    size_section = md.split("### 7. Size")[1].split("### 8.")[0]

    # It is an equal-weight-style tilt, not a log-compressed one.
    assert "equal-weight-style size tilt" in size_section
    # And the extrapolation is labelled rather than papered over.
    assert "largest two market-cap deciles" in size_section
    assert "extrapolates" in size_section


def test_size_metric_description_points_at_rank_scoring():
    desc = _SIZE_DESCRIPTIONS["size_log_mcap"]
    assert "percentile rank" in desc
    assert "which the log does not affect" in desc


def test_overview_size_section_has_no_stale_cap_range():
    """The removed sentence quoted '$2B to $3T+'. The real 2026-09-03 universe
    spans $6.8B to $5,419B, so that range was wrong as well as irrelevant."""
    md = OVERVIEW.read_text(encoding="utf-8")
    size_section = md.split("### 7. Size")[1].split("### 8.")[0]
    assert not re.search(r"\$2B to \$3T", size_section)
