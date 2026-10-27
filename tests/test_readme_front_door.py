"""README.md is the repository's front door, and a club adopting the tool reads it first.

WHY THIS EXISTS - 2026-10-27 (round 2 docs pass). The README was written as an operator's note
and had drifted from the system it described: it put the fetch-failure rate at "roughly 10-25% of
tickers" (measured at 0 of 9,036 since 2026-09-25), called the workbook "up to 6-sheet" (seven,
with the ReadMe), described the correlation sheet as category scores (it is every metric's
percentile) and the sensitivity test as "+-5% perturbation" (5 percentage points), listed a
``--top-n`` flag ``run_screener.py`` does not accept, labelled Size a "small-cap premium" (the
screener's own review finds none among large companies), and named no link to the research page.

It is hand-written, so nothing regenerates it. These tests hold the figures a reader would quote
to ``config.yaml`` and the engine, the flags to the parser, the sheets to the workbook writer,
and its one sentence about history to the research engine's published summary.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claims  # noqa: E402
import factor_engine as fe  # noqa: E402
import portfolio_constructor as pc  # noqa: E402
import run_screener  # noqa: E402
import stock_summary  # noqa: E402

README = ROOT / "README.md"
SITE = "https://calebsmit.github.io/screener-dashboard/"


@pytest.fixture(scope="module")
def text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _section(text: str, title: str) -> str:
    start = text.index(f"\n## {title}\n")
    nxt = text.find("\n## ", start + 4)
    return text[start: nxt if nxt != -1 else len(text)]


def test_it_opens_on_what_it_is_and_that_it_is_not_advice(text):
    head = text[:text.index("\n## ")]
    assert "not investment advice" in head
    assert "has not been" in head and "shown to predict returns" in head
    h2 = re.findall(r"^## (.+)$", text, re.M)
    assert h2[:3] == ["Open it", "How to read it", "The eight categories"], h2


def test_it_links_every_public_page(text):
    for url in (SITE + ")", SITE + "research.html", SITE + "lab.html"):
        assert url in text, url
    assert "[SCREENER_OVERVIEW.md](SCREENER_OVERVIEW.md)" in text


def test_the_category_weights_are_the_configs(text, cfg):
    rows = re.findall(r"^\| (\w+) \| (\d+)% \|", _section(text, "The eight categories"), re.M)
    got = {name.lower(): int(w) for name, w in rows}
    assert got == {k: v for k, v in cfg["factor_weights"].items() if v}, got
    assert sum(got.values()) == 100


def test_the_metric_counts_are_the_engines(text, cfg):
    gen, bank = fe.weighted_metric_sets(cfg)
    bank_only = set(bank) - set(gen)
    candidates = len(fe.METRIC_COLS) - len(set(gen) | bank_only)
    block = re.sub(r"\s+", " ", _section(text, "The eight categories"))
    assert f"**{len(fe.METRIC_COLS)} metrics**" in block
    assert f"**{len(gen)}** general" in block and f"**{len(bank_only)}** bank-specific" in block
    assert f"scored on **{len(bank)}**" in block and f"**{candidates}** candidates" in block


def test_every_flag_it_lists_is_one_the_screener_accepts(text):
    block = _section(text, "Run it yourself")
    flags = set(re.findall(r"run_screener\.py[^\n#]*?(--[a-z-]+)", block))
    assert flags >= {"--dry-run", "--refresh", "--tickers", "--preset", "--show-weights", "--no-portfolio"}
    src = inspect.getsource(run_screener.parse_args)
    for f in flags:
        assert f'"{f}"' in src, f


def test_every_workbook_sheet_it_names_is_written(text):
    row = next(ln for ln in text.splitlines() if ln.startswith("| `factor_output.xlsx`"))
    names = re.findall(r"\b(ReadMe|FactorScores|ScreenerDashboard|ModelPortfolio|DataValidation|"
                       r"WeightSensitivity|FactorCorrelation)\b", row)
    assert "**7 sheets**" in row and len(set(names)) == 7
    src = inspect.getsource(pc)
    for n in names:
        assert f'"{n}"' in src, n
    assert f"the top {pc.DASHBOARD_TOP_N}" in row and f"the top {pc.DATA_VALIDATION_TOP_N}'s" in row


def test_none_of_the_stale_statements_is_back(text):
    for stale in ("10-25%", "--top-n", "6-sheet", "Small-cap premium", "matrix of category scores",
                  "±5% perturbation", "single-name volatility only"):
        assert stale not in re.sub(r"\s+", " ", text), stale
    assert claims.forbidden_hits(text) == []


def test_the_reading_sections_carry_no_advice_language(text):
    for title in ("Open it", "How to read it", "The eight categories", "What it cannot do"):
        block = _section(text, title)
        assert stock_summary.advice_terms_in(block) == [], (title, stock_summary.advice_terms_in(block))


def test_its_sentence_about_history_is_the_research_engines(text):
    s = json.loads((ROOT / "research" / "experiments" / "summary.json").read_text(encoding="utf-8"))
    lo, hi = s["baseline_ic_ci95"]["pooled"]
    said = "did no better than chance" in re.sub(r"\s+", " ", text)
    assert said == (lo <= 0 <= hi and int(s["better"]) == 0)
