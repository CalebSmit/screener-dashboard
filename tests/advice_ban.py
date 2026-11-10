"""The data's own proper nouns, for an advice-language check over rendered page text.

Companion to ``stock_summary.without_data_names``. See that function for why this exists:
on 2026-11-10 a healthy data run was discarded because **Best Buy Co., Inc.** entered the
Reporting Soon window and the panel's advice-language check read the word ``Buy`` in the
company's legal name.

A check over text a *browser* rendered takes its names from the page's own payload, so the
list is whatever that run published rather than anything transcribed here - the same rule as
``tests/examples.py`` (do not name the stock today's data happened to behave a certain way
for). ``tests/test_advice_ban_scope.py`` is the sweep that keeps every such call site using it.
"""
from __future__ import annotations

NAMES_JS = "(window.D && D.table_data ? D.table_data : []).flatMap(r => [r.Ticker, r.Company || ''])"


def data_names(page) -> list[str]:
    """Every ticker and company name in the payload the page actually loaded."""
    return page.evaluate(f"(() => {NAMES_JS})()")
