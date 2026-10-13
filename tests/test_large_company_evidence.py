"""What the methodology says about value and momentum among large companies.

`research/2026-10-12-value-and-momentum-among-large-companies.md`: the screener ranks 500 large
companies, and the premia its Valuation and Momentum sections once cited were all-stock figures.
Each section now says what the published evidence is among the largest stocks and, for momentum,
what the public data shows since 2000. The generator cannot read the Factor Lab (no pipeline
module may), so the figures are written out there and held to the lab's data here: a refresh
that changes the picture fails this instead of leaving the page wrong.

These read the generator's output, not the committed page (CLAUDE.md rule 10).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import factor_lab as fl

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "research" / "2026-10-12-value-and-momentum-among-large-companies.md"


@pytest.fixture(scope="module")
def overview() -> str:
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return run_screener.build_screener_overview(cfg)


def section(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i: text.index(end, i)]


@pytest.fixture(scope="module")
def split() -> dict:
    return fl.load_published()["split"]


def test_valuation_says_the_premium_is_mostly_a_small_company_result(overview):
    sec = section(overview, "### 1. Valuation", "### 2.")
    assert "**Among large companies.**" in sec and "mostly a small-company result" in sec
    assert "Israel & Moskowitz 2013" in sec and "Asness, Frazzini, Israel & Moskowitz" in sec
    assert "lab.html#big-and-small" in sec and NOTE.name in sec
    # the figures are the paper's (Table 3, largest size quintile, book-to-market), as the note records them
    assert "3.7% a year" in sec and "2.2% after adjusting for market risk" in sec
    note = NOTE.read_text(encoding="utf-8")
    assert "| High minus low, return | 3.70 (1.90) | 7.49 (2.95) |" in note
    assert "| High minus low, CAPM alpha | 2.19 (1.14) | 10.24 (4.23) |" in note


def test_book_value_carries_no_weight_outside_banks():
    """ "those are the measures weighted here; book value carries no weight outside banks" """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    val = cfg["metric_weights"]["valuation"]
    assert val["pb_ratio"] == 0
    earnings_and_cash = val["fcf_yield"] + val["earnings_yield"] + val["ev_ebitda"]
    assert earnings_and_cash >= 80, val


def test_momentum_says_what_the_last_quarter_century_shows(overview, split):
    sec = section(overview, "### 4. Momentum", "### 5.")
    assert "**Among large companies, and lately.**" in sec and "Israel & Moskowitz 2013" in sec
    assert "7.5% a year" in sec and "about 38%" in sec
    assert "not on the last quarter-century" in sec and NOTE.name in sec
    # "about 0.6% a month from 1963 to 1999 and about 0.1% a month since 2000"
    early = fl.split_run(split, "196307", "199912")["factors"]["momentum"]["big"]
    late = fl.split_run(split, "200001", "209912")["factors"]["momentum"]["big"]
    assert "about 0.6% a month from 1963 to 1999" in sec and 0.55 <= early["mean"] < 0.65
    assert "about 0.1% a month since 2000" in sec and 0.05 <= late["mean"] < 0.15
    assert abs(late["t"]) < 2 < early["t"]
    # "the lowest stretch of that length in the record"
    series, n = split["momentum"]["big"], len([m for m in split["months"] if m >= "200001"])
    windows = [sum(series[i:i + n]) / n for i in range(len(series) - n + 1)]
    assert "the lowest stretch of that length in the record" in sec
    assert windows[-1] == min(windows)


def test_momentum_outside_the_panic_state_is_as_the_page_says(overview, split):
    """ "about 0.3% a month since 2000 outside the panic state the rule below watches for" - the
    state as known before each month, at the rule's own two numbers."""
    pub = fl.load_published()
    months, trail, vol = pub["momentum"]["months"], pub["market"]["trail"]["24"], pub["market"]["vol"]["126"]
    at = {m: i for i, m in enumerate(months)}
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["momentum_regime"]
    assert (cfg["return_window_months"], cfg["vol_window_days"]) == (24, 126)
    inside, outside = [], []
    for i, m in enumerate(split["months"]):
        j = at.get(m)
        if m < "200001" or j is None or trail[j] is None or vol[j] is None:
            continue
        (inside if trail[j] < 0 and vol[j] > cfg["vol_threshold"] else outside).append(split["momentum"]["big"][i])
    out, ins = fl.mean_t(outside), fl.mean_t(inside)
    assert "about 0.3% a month since 2000 outside the panic state" in section(overview, "### 4. Momentum", "### 5.")
    assert 0.25 <= out["mean"] < 0.35 and ins["mean"] < -1 and len(inside) < 60 < len(outside)


def test_the_note_keeps_the_weights_and_says_why():
    note = NOTE.read_text(encoding="utf-8")
    assert "**No weight is proposed to change.**" in note
    assert "The lab and the literature disagree, so the weight stays" in note
    for name in ("Israel, R. & Moskowitz, T. (2013)", "Bhattacharya, D., Li, W.-H. & Sonaer, G. (2017)",
                 "Hou, K., Xue, C. & Zhang, L. (2020)", "McLean, R. D. & Pontiff, J. (2016)"):
        assert name in note, name
    assert (ROOT / "research" / "measurements" / "2026-10-12-large-company-value-momentum.py").exists()
