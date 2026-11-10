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

#: `D` is declared `const D = window.SCREENER_DATA || {}` at script scope, so it is **not** a
#: property of `window` - the same shape as the closure-scoped `CTX_LOADED` the dashboard notes
#: warn about. The first version of this read `window.D`, got `undefined`, masked nothing, and
#: the rehearsal failed exactly as the data run had. Reach it by `typeof`, and read the raw
#: payload as a fallback so a page that has not finished loading cannot silently mask nothing.
NAMES_JS = ("(typeof D !== 'undefined' && D && D.table_data ? D.table_data "
            ": (window.SCREENER_DATA || {}).table_data || [])"
            ".flatMap(r => [r.Ticker, r.Company || ''])")


def data_names(page) -> list[str]:
    """Every ticker and company name in the payload the page actually loaded.

    Raises if the page carries no payload: an empty mask is indistinguishable from a working one
    on text that happens to be clean today, which is precisely how this check failed before."""
    names = page.evaluate(f"(() => {NAMES_JS})()")
    if not names:
        raise AssertionError(
            "the page exposes no table_data, so there are no names to mask - an advice check "
            "masking an empty list is the no-op this helper exists to prevent "
            "(tests/advice_ban.py)")
    return names
