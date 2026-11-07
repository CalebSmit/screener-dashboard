"""A second full run on the same day must be able to publish (2026-11-06).

That day the 02:00 data run was discarded by the claims gate, the 06:00 session fixed both
causes, and the owner approved a 16:00 catch-up run so the site would not sit on Wednesday's
close all weekend. The catch-up loaded the 02:00 run's scored cache (``[CACHE HIT]``) and was
discarded at dashboard generation: ``CalculationMismatch``, 286 of 499 stocks, every insurer
checked against the generic weights because the cache keeps no ``_is_bank_like`` column.
It fetched nothing either, so even a clean publish would have carried the previous session's
prices. Two consequences, each pinned here:

* the data loop - the run that publishes - always passes ``--fresh`` and never reuses the cache;
* ``scripts/rehearse_publish.py`` (gate 5) never rehearses from a run that fetched nothing - the
  discarded catch-up run, as the newest folder in ``runs/``, failed every rehearsal until it was
  moved aside.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_screener as rs  # noqa: E402
import rehearse_publish as rp  # noqa: E402

# A fixed day, never the clock: the cache's window is counted from its file date.
TODAY = datetime(2026, 11, 6, 16, 0)
WRITTEN_TODAY = datetime(2026, 11, 6)
WRITTEN_YESTERDAY = datetime(2026, 11, 5)


def _args(**kw):
    base = {"refresh": False, "fresh": False, "tickers": ""}
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_data_loop_never_reuses_the_scored_cache():
    ps1 = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8")
    calls = re.findall(r"\$args = @\(([^)]*)\)", ps1)
    assert calls, "data-run.ps1 no longer builds the screener's arguments where this test looks"
    assert all("'run_screener.py'" in c and "'--fresh'" in c for c in calls if "run_screener" in c), (
        "the data loop must call run_screener.py with --fresh: a run that loads the day's scored "
        "cache cannot publish (2026-11-06 16:00, CalculationMismatch on 286 of 499 stocks)")


def test_the_flag_exists_and_defaults_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_screener.py"])
    assert rs.parse_args().fresh is False
    monkeypatch.setattr(sys, "argv", ["run_screener.py", "--fresh"])
    assert rs.parse_args().fresh is True


@pytest.mark.parametrize("args, written, reuse", [
    (_args(), WRITTEN_TODAY, True),                     # a hand rerun the same day may reuse it
    (_args(fresh=True), WRITTEN_TODAY, False),          # the data loop never does
    (_args(refresh=True), WRITTEN_TODAY, False),
    (_args(tickers="AAPL,MSFT"), WRITTEN_TODAY, False),  # a trial never reads it
    (_args(), WRITTEN_YESTERDAY, False),                # outside the window
    (_args(fresh=True), WRITTEN_YESTERDAY, False),
])
def test_when_a_run_may_reuse_the_scored_cache(args, written, reuse):
    assert rs.reuse_score_cache(written, 1, args, now=TODAY) is reuse


def test_the_premise_the_cache_cannot_carry_the_bank_flag(tmp_path, monkeypatch):
    """Why a cache-hit run cannot publish: the writer drops every internal column. If this ever
    fails because the cache now keeps them, the cache path may be publishable after all - but it
    still fetched nothing, so check ``check_run_health``'s fetch evidence before relaxing --fresh."""
    import factor_engine as fe
    monkeypatch.setattr(fe, "CACHE_DIR", tmp_path)
    df = pd.DataFrame({"Ticker": ["ALL"], "valuation_score": [63.3], "_is_bank_like": [True]})
    out = pd.read_parquet(fe.write_scores_parquet(df, config_hash="h"))
    assert "_is_bank_like" not in out.columns and "valuation_score" in out.columns


def _run(root: Path, name: str, fetched: bool, mtime: float) -> Path:
    d = root / name
    d.mkdir()
    scored = d / "05_final_scored.parquet"
    pd.DataFrame({"Ticker": [f"T{i}" for i in range(rp.MIN_UNIVERSE)]}).to_parquet(scored)
    (d / "effective_weights.json").write_text("{}")
    if fetched:
        (d / "00_raw_fetch.parquet").write_bytes(b"")
    os.utime(scored, (mtime, mtime))
    return d


def test_the_rehearsal_skips_a_run_that_fetched_nothing(tmp_path):
    _run(tmp_path, "e1174e2f4363", fetched=True, mtime=1_000_000)   # 02:00, fetched
    _run(tmp_path, "bce9b3ce6629", fetched=False, mtime=2_000_000)  # 16:00, [CACHE HIT]
    assert rp.newest_full_run(tmp_path).name == "e1174e2f4363"


def test_a_newer_run_that_fetched_is_still_chosen(tmp_path):
    _run(tmp_path, "older", fetched=True, mtime=1_000_000)
    _run(tmp_path, "newer", fetched=True, mtime=2_000_000)
    assert rp.newest_full_run(tmp_path).name == "newer"
