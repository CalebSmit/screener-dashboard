"""`scripts/exposure_shift.py`: the before/after factor-exposure record a scoring change owes.

The habit is `research/2026-11-02-growth-exposure-drift.md` section 6.4; the command exists because
the 2026-11-24 Momentum change skipped it and moved the ranking's growth lean +0.026 -> +0.006
unrecorded (that note's section 7). These tests pin that the script measures what the brief measures
- one definition of "lean on growth", not two - and that its arithmetic is a rank correlation over
the published columns. Synthetic tables only: no premise depends on today's data (row
"premise->red").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exposure_shift as X  # noqa: E402
import write_brief  # noqa: E402


def _table(seed: int, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({f"{c}_score": rng.uniform(0, 100, n) for c in X.CATS})
    t["Composite"] = t["valuation_score"] * 0.6 + t["growth_score"] * 0.1 + rng.normal(0, 10, n)
    return t


def test_growth_exposure_is_the_briefs_own_number():
    """The script's growth row and the brief's "lean on growth" line are the same computation."""
    t = _table(1)
    brief = write_brief.growth_exposure(t.to_dict("records"))
    assert brief.startswith(f"{X.exposures(t)['growth']:+.3f}")


def test_exposure_is_a_rank_correlation():
    """A monotone transform of the composite leaves every exposure unchanged (Spearman, not Pearson)."""
    t = _table(2)
    u = t.copy()
    u["Composite"] = np.exp(u["Composite"] / 20)
    assert X.exposures(t) == pytest.approx(X.exposures(u))


def test_pairs_cover_all_28_and_rank_the_largest_move_first():
    a = _table(3)
    b = a.copy()
    b["momentum_score"] = b["growth_score"] + np.random.default_rng(4).normal(0, 1, len(b))
    assert len(X.pairs(a)) == 28
    top = X.shift(a, b)["pairs_moved_most"][0]
    assert top[0] == "growth~momentum"
    assert top[2] > 0.9


def test_reads_a_payload_file(tmp_path):
    t = _table(5)
    p = tmp_path / "dashboard_data.js"
    p.write_text("window.SCREENER_DATA = " + json.dumps({"table_data": t.to_dict("records")}) + ";",
                 encoding="utf-8")
    assert X.exposures(X.load_table(str(p))) == pytest.approx(X.exposures(t))
