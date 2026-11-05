"""The stock a browser test opens survives the index changing (2026-11-04, `tests/examples.py`).

Four browser modules run in the 02:00 publish gate, where a failure discards the night's data, and
they opened EXPE, APA, HST and BBY by name. Measured that evening on a copy of the payload with those
four removed: the modules as they stood failed 6 tests; routed through `examples.stock` they fail
none that open a stock by name.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests import examples  # noqa: E402

FULL = {"composite": 60.0, "price": 10.0, "pt_mean": 12.0, "financials": {"x": 1}, "raw": {"fcf_yield": 0.05}}


@pytest.fixture
def fake(monkeypatch):
    def install(detail, ranks):
        payload = {"stock_detail": detail,
                   "table_data": [{"Ticker": t, "Rank": r} for t, r in ranks.items()]}
        monkeypatch.setattr(examples, "_payload", lambda: payload)
    return install


def test_the_named_stock_comes_back_unchanged_while_the_payload_carries_it(fake):
    fake({"EXPE": {}, "AAA": FULL}, {"EXPE": 300, "AAA": 1})
    assert examples.stock("EXPE") == "EXPE", "a test must read exactly as written while its stock is there"


def test_a_missing_stock_falls_back_to_the_best_ranked_full_sheet(fake):
    bank = dict(FULL, raw={"fcf_yield": None})
    fake({"AAA": dict(FULL, pt_mean=None), "BBB": bank, "CCC": FULL, "DDD": FULL},
         {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4})
    assert examples.stock("EXPE") == "CCC"          # AAA has no target, BBB no free-cash-flow yield
    assert examples.stock("EXPE", exclude=("CCC",)) == "DDD"
    assert examples.stock("DDD", exclude=("DDD",)) == "CCC"


def test_a_name_search_types_what_finds_only_that_company(fake):
    fake({"EXPE": {"company": "Expedia Group, Inc."}, "AXP": {"company": "American Express Company"},
          "AIG": {"company": "American International Group"}}, {})
    assert examples.search_text("EXPE") == "expedia"
    assert examples.search_text("AXP") == "american express company"


# The publish gate's browser modules may open a stock by name only through `examples.stock`, apart
# from the two whose tests assert what only that company is (JPMorgan the bank metric set and its
# name) - both among the largest members of the index.
ALLOWED = {"JPM", "MSFT"}
NAMED = re.compile(r"""openStockDetail\(['"]([A-Z]{1,5})['"]\)|_sheet\(page, ['"]([A-Z]{1,5})['"]\)"""
                   r"""|D\.stock_detail\.([A-Z]{1,5})\b|#stock=([A-Z]{1,5})\b""")


def test_no_publish_gate_browser_module_opens_another_stock_by_name():
    import check_published_claims as gate
    found = []
    for rel in gate.module_paths():
        p = ROOT / rel
        if not p.name.startswith("test_") or p.name == Path(__file__).name:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "delete D.stock_detail." in line:  # a test removing a name itself builds its own premise
                continue
            for m in NAMED.finditer(line):
                t = next(g for g in m.groups() if g)
                if t not in ALLOWED:
                    found.append(f"{p.name}:{n} {t}")
    assert found == [], ("open these through tests/examples.py so a change in the index cannot "
                         f"discard a night's data: {found}")
