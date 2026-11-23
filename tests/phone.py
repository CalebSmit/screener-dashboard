"""What a phone reader taps before a control or a panel is in view (the phone pass, 2026-11-22).

On a phone the ranking's filters other than search and sort sit behind a Filters button, and the
stock sheet's long sections arrive folded (`generate_dashboard._js_phone`,
`tests/test_phone_pass.py`). A browser test at 375px that sets a filter, or reads a panel inside a
folded section, has to do what the reader does first. On a wide screen both helpers do nothing, so
a test that runs at both widths can call them unconditionally.
"""
from __future__ import annotations

# Tap Filters if it is showing and its group is still folded.
OPEN_FILTERS_JS = """() => {
    const b = document.getElementById('filters-more-m');
    if (!b || getComputedStyle(b).display === 'none') return 'wide';
    const bar = b.closest('.filters-bar');
    if (!bar.classList.contains('filters-open')) b.click();
    return bar.classList.contains('filters-open') ? 'open' : 'folded';
}"""


def open_filters(page) -> str:
    return page.evaluate(OPEN_FILTERS_JS)


def open_section(page, section_id: str) -> None:
    """Open one of the sheet's sections the way its tab does (``goToModal`` unfolds its target)."""
    page.evaluate("id => goToModal(id)", section_id)
