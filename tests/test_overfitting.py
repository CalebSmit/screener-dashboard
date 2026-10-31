"""The research engine's overfitting check (``overfitting.py``, 2026-10-20).

Pinned: the procedure behaves as Bailey, Borwein, Lopez de Prado & Zhu (2017) say it must - about 0.5
on pure noise, near 0 when one configuration is genuinely better, and the full C(16, 8) splits; the
committed ledger's reading is the one the research page prints, all three ways; the page's verdict
follows the numbers; the module reads only the ledger, so it is outside the engine's ``RESULT_PATHS``
(editing it must not stale 30 rows); and nothing in production imports it.
"""
from __future__ import annotations

import html
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


def _rebuilt() -> str:
    """The page as `python v2_experiments.py report` would write it, not the committed file."""
    reg, latest = _ledger()
    return rp.build(reg, latest, X.summary_for_page(reg, latest), X.detectable,
                    X.load_evaluations())


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


# --------------------------------------------------------------- the prose, figure by figure
# Closes the five `pbo-*` rows of the claims-mutation audit (opened 2026-10-29; see
# `plan/claims-mutation-audit.md`). Until 2026-10-31 the section's *cards* were pinned
# (`test_the_page_prints_the_ledger_s_reading_all_three_ways`) and its three prose paragraphs
# were not: the prose restates all three PBOs, adds two out-of-sample shares, the top winner's
# two era leads and the count of alternatives, and nothing read any of them. `_pbo_words` was
# probed at 0.21/0.5/0.73 and never at its own bars, 0.45 and 0.55, which are the only two
# points that decide which of the three words is printed.
#
# Both halves are here on purpose: the first two tests read the figures off the **generated**
# page against the committed ledger (the 2026-10-29 structural fix - a check against
# `rp.PAGE` is green until someone regenerates, which is a drift tripwire with a value check's
# name), and the last two feed `overfitting_section` a reading of their own, which is the only
# way to put a figure *at* a bar the committed ledger does not sit on.

def _reading(pbo_all=0.20, pbo_recent=0.72, pbo_without=0.60,
             oos_all=0.80, oos_recent=0.34, pre=0.030, hold=-0.010, configurations=28):
    """A `overfitting.check` return, shaped as `research_page.overfitting_section` reads it."""
    def era(pbo, oos, first, last, blocks, splits):
        # every key `overfitting.cscv` returns, so a sentence that starts reading a different
        # one of them fails on the figure rather than on a KeyError
        return {"configurations": configurations, "months": 176, "blocks": blocks,
                "splits": splits, "first_month": first, "last_month": last, "pbo": pbo,
                "winner_median_oos_percentile": 0.11, "winner_mean_in_sample": 0.012,
                "winner_mean_out_of_sample": 0.013,
                "winner_beat_current_out_of_sample": oos, "winner_was_current": 0.0,
                "top_winners": [{"id": "only-growth", "share": 0.50}]}
    r = {"all": era(pbo_all, oos_all, "2012-01-31", "2026-08-31", 16, 12870),
         "recent": era(pbo_recent, oos_recent, "2020-01-31", "2026-08-31", 8, 70),
         "without": era(pbo_without, 0.40, "2012-01-31", "2026-08-31", 16, 12870)}
    r["without"]["dropped"] = "only-growth"
    latest = {"only-growth": {"summary": {"presample": {"ic_vs_baseline": {"mean": pre}},
                                          "holdout": {"ic_vs_baseline": {"mean": hold}}}}}
    return r, latest


def _section(monkeypatch, **kw) -> str:
    r, latest = _reading(**kw)
    import overfitting
    monkeypatch.setattr(overfitting, "check", lambda reg, lat: r)
    return rp.overfitting_section([], latest)


# the words, written out as literals rather than read from `_pbo_words` - a check may not
# import the number it is checking (audit row 59)
WORSE = "worse than a coin toss"
LUCK = "about what luck alone gives"
BETTER = "better than luck alone would give"


def test_each_prose_pbo_carries_the_rows_own_reading_and_its_word():
    """Closes `pbo-all-era-prose`, `pbo-recent-era-prose` and `pbo-without-winner-prose` on the
    PBO value and the word beside it. The word is derived here from literal bars, so a moved bar
    in `_pbo_words` changes the page and not the expectation."""
    reg, latest = _ledger()
    r = O.check(reg, latest)
    page = _rebuilt()

    def word(p: float) -> str:
        return WORSE if p >= 0.55 else (LUCK if p > 0.45 else BETTER)

    a, rec, wo = r["all"], r["recent"], r["without"]
    assert (f'the PBO was <span class="num">{a["pbo"]:.2f}</span>, {word(a["pbo"])}, and the '
            "winner beat the current weighting on the other half") in page
    assert (f'alone the PBO was <span class="num">{rec["pbo"]:.2f}</span>, {word(rec["pbo"])}, '
            "and the winner beat the current weighting out of sample") in page
    assert (f"Drop {html.escape(rp.label(wo['dropped']))} from the set and the PBO over every "
            f'year is <span class="num">{wo["pbo"]:.2f}</span>, {word(wo["pbo"])}.') in page
    # the three readings are genuinely three different numbers on this ledger, so none of the
    # assertions above can be satisfied by another era's figure
    assert len({round(a["pbo"], 2), round(rec["pbo"], 2), round(wo["pbo"], 2)}) == 3


def test_the_out_of_sample_shares_and_the_configuration_count_are_the_rows_own():
    """Closes the same three rows on their remaining figures, and `pbo-configuration-count`.

    The count is the page's own subtraction (`configurations - 1`, the alternatives beside the
    current weighting) and `summary['alternatives']` is a different number from a different
    source, so the page could print either and nothing would notice."""
    reg, latest = _ledger()
    r = O.check(reg, latest)
    page = _rebuilt()
    a, rec = r["all"], r["recent"]
    assert (f'on the other half in <span class="num">{a["winner_beat_current_out_of_sample"] * 100:.0f}%</span>'
            " of trials.") in page
    assert (f'out of sample in <span class="num">{rec["winner_beat_current_out_of_sample"] * 100:.0f}%</span>'
            " of trials.") in page
    assert f"current weighting and the {a['configurations'] - 1} alternatives above" in page
    assert a["configurations"] >= 2, "the subtraction has to be of something"
    assert f"and the {a['configurations']} alternatives above" not in page, \
        "the count must exclude the current weighting"


def test_the_top_winners_two_era_leads_and_the_clause_that_depends_on_their_signs(monkeypatch):
    """Closes `pbo-top-winner-lead`. Both means and the clause - ", where it fell behind." is
    printed only when the held-out lead is negative while the 2012-2019 lead is positive, so the
    sentence has to be read at the sign boundary, which the committed ledger does not sit on."""
    reg, latest = _ledger()
    r = O.check(reg, latest)
    top = r["all"]["top_winners"][0]
    s = latest[top["id"]]["summary"]
    pre = s["presample"]["ic_vs_baseline"]["mean"]
    hold = s["holdout"]["ic_vs_baseline"]["mean"]
    page = _rebuilt()
    assert (f'Its lead over the current weighting was <span class="num">{rp.num(pre)}</span> over '
            f'2012&ndash;2019 and <span class="num">{rp.num(hold)}</span> over the held-out months'
            ) in page

    # the clause, at and either side of its boundary
    assert ", where it fell behind." in _section(monkeypatch, pre=0.030, hold=-0.010)
    for pre_, hold_ in ((0.030, 0.000), (0.030, 0.010), (-0.030, -0.010)):
        out = _section(monkeypatch, pre=pre_, hold=hold_)
        assert ", where it fell behind." not in out, (pre_, hold_)
        assert "over the held-out months." in out


def test_the_word_beside_each_pbo_turns_at_its_own_bars(monkeypatch):
    """The two bars, 0.45 and 0.55, as literals and from both sides - on the page, not on
    `_pbo_words`. Probing 0.21/0.5/0.73 (as the only check did until 2026-10-31) passes for any
    pair of bars between 0.21 and 0.5 and between 0.5 and 0.73."""
    for pbo, want in ((0.44, BETTER), (0.45, BETTER), (0.46, LUCK),
                      (0.54, LUCK), (0.55, WORSE), (0.56, WORSE)):
        out = _section(monkeypatch, pbo_all=pbo)
        assert f'<span class="num">{pbo:.2f}</span>, {want},' in out, (pbo, want)
        for other in (WORSE, LUCK, BETTER):
            if other != want:
                assert f'<span class="num">{pbo:.2f}</span>, {other},' not in out
    # and the function itself at the same points, so a caller added later inherits the bars
    assert [rp._pbo_words(p) for p in (0.45, 0.4501, 0.5499, 0.55)] == [BETTER, LUCK, LUCK, WORSE]


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
