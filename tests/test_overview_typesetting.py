"""The methodology page is typeset in one pass (``run_screener._typeset_md``, 2026-10-16).

Written over many sessions, the page had drifted to about 130 dashes in three forms: em dashes,
hyphens used as dashes and hyphens used as minus signs. The pass sets a spaced en dash for a dash and a
true minus inside a formula, and leaves code alone, so the prose can be written plainly and still publish
consistently.
"""
import re
from pathlib import Path

import yaml

import run_screener

ROOT = Path(__file__).resolve().parent.parent


def test_a_dash_becomes_a_spaced_en_dash():
    assert run_screener._typeset_md("Quality — profitable firms - hold up.") == \
        "Quality – profitable firms – hold up."


def test_a_hyphen_inside_a_formula_becomes_a_minus():
    out = run_screener._typeset_md("(Net Income - Operating Cash Flow) / Total Assets; MC + Debt - Cash; 50 - 50/n")
    assert out.count("−") == 3 and "–" not in out


def test_code_is_left_alone():
    src = "Run `a - b` here.\n\n```bash\npy x.py --start 2024 - 1\n```\n"
    assert run_screener._typeset_md(src) == src


def test_the_published_page_has_no_em_dashes_or_spaced_hyphens():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    prose = re.sub(r"`[^`\n]*`", "", re.sub(r"```.*?```", "", text, flags=re.S))
    assert "—" not in prose
    assert not re.search(r"(?<=\S) - (?=\S)", prose)
