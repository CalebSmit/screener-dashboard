"""The research engine's overfitting check (``overfitting.py``, 2026-10-20).

Pinned: the procedure behaves as Bailey, Borwein, Lopez de Prado & Zhu (2017) say it must - about 0.5
on pure noise, near 0 when one configuration is genuinely better, and the full C(16, 8) splits; the
committed ledger's reading is the one the research page prints, all three ways; the page's verdict
follows the numbers; the module reads only the ledger, so it is outside the engine's ``RESULT_PATHS``
(editing it must not stale 30 rows); and nothing in production imports it.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import overfitting as O  # noqa: E402
import research_page as rp  # noqa: E402
import v2_experiments as X  # noqa: E402


def _frame(x: np.ndarray) -> pd.DataFrame:
    months = pd.date_range("2012-01-31", periods=x.shape[0], freq="ME").strftime("%Y-%m-%d")
    cols = [O.CURRENT] + [f"c{i}" for i in range(1, x.shape[1])]
    return pd.DataFrame(x, index=months, columns=cols)


def _ledger() -> tuple[list, dict]:
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    return X.load_registry(), latest


def test_pure_noise_gives_about_one_half():
    """Under the null the in-sample winner's out-of-sample rank is uniform, so PBO ~ 0.5."""
    rng = np.random.default_rng(20261020)
    pbos = []
    for _ in range(5):
        x = rng.normal(0, 0.12, size=(176, 28))
        x[:, 0] = 0.0
        pbos.append(O.cscv(_frame(x))["pbo"])
    assert 0.35 < float(np.mean(pbos)) < 0.65, pbos


def test_a_genuinely_better_configuration_drives_it_toward_zero_and_keeps_winning():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.12, size=(176, 28))
    x[:, 0] = 0.0
    x[:, 5] += 0.06                                          # one real edge, half a monthly sd
    r = O.cscv(_frame(x))
    assert r["pbo"] < 0.1
    assert r["top_winners"][0]["id"] == "c5" and r["top_winners"][0]["share"] > 0.9
    assert r["winner_beat_current_out_of_sample"] > 0.9


def test_it_runs_every_split_and_refuses_too_little_history():
    rng = np.random.default_rng(1)
    r = O.cscv(_frame(rng.normal(size=(64, 4))))
    assert r["splits"] == math.comb(16, 8) == 12_870
    assert O.cscv(_frame(rng.normal(size=(40, 4))), blocks=8)["splits"] == math.comb(8, 4)
    with pytest.raises(ValueError):
        O.cscv(_frame(rng.normal(size=(20, 4))))


def test_the_current_weighting_is_the_zero_column_and_only_comparable_rows_enter():
    reg, latest = _ledger()
    perf, skipped = O.candidates(reg, latest)
    assert (perf[O.CURRENT] == 0).all()
    void = {e["id"] for e in reg if e.get("void")}
    assert not void & set(perf.columns)
    assert all(any(s.startswith(v + ":") for s in skipped) for v in void)
    # each column is the experiment's IC lead over the current weighting on the stocks both ranked
    i = next(c for c in perf.columns if c != O.CURRENT)
    m = perf.index[0]
    assert perf.loc[m, i] == pytest.approx(latest[i]["ic_by_month"][m] - latest[i]["base_ic_by_month"][m])


def test_the_page_prints_the_ledger_s_reading_all_three_ways():
    reg, latest = _ledger()
    r = O.check(reg, latest)
    assert r and {"all", "recent", "without"} <= set(r)
    page = rp.PAGE.read_text(encoding="utf-8")
    # the section itself, by its heading: since 2026-10-27 the page's contents list names it first
    sec = page[page.index("<h2>If you had kept the best backtest"):]
    sec = sec[:sec.index("</section>")]
    for key in ("all", "recent", "without"):
        assert f'<div class="v">{r[key]["pbo"]:.2f}</div>' in sec, key
    assert f'{r["all"]["splits"]:,}' in sec
    top = r["all"]["top_winners"][0]
    assert f"{rp.label(top['id'])}, in" in sec and f'{top["share"] * 100:.0f}%' in sec
    assert r["without"]["dropped"] == top["id"]


def test_the_verdict_follows_the_numbers():
    """The closing sentence may call the winner consistent only if it stays below 0.45 in the
    recent era and without its most frequent winner - a low whole-history PBO alone is consistency,
    not cause."""
    reg, latest = _ledger()
    r = O.check(reg, latest)
    page = rp.PAGE.read_text(encoding="utf-8")
    robust = r["recent"]["pbo"] < 0.45 and r["without"]["pbo"] < 0.45
    assert ("held up consistently" in page) == robust
    assert ("did not survive either check" in page) == (not robust)
    for p, words in ((0.21, "better than luck"), (0.5, "about what luck"), (0.73, "worse than a coin toss")):
        assert rp._pbo_words(p).startswith(words)


def test_it_reads_the_ledger_only_and_cannot_stale_it():
    assert "overfitting.py" not in X.RESULT_PATHS
    src = (ROOT / "overfitting.py").read_text(encoding="utf-8")
    assert "LEDGER.open(" not in src and ".write_text(" not in src


def test_no_production_module_imports_it():
    pat = re.compile(r"^\s*(import overfitting|from overfitting)\b", re.M)
    for p in ROOT.glob("*.py"):
        if p.name in ("overfitting.py", "research_page.py", "conftest.py"):
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), p.name
