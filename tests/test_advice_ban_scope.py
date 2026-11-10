"""The advice-language ban's *scope*: the tool's prose, never the index's membership.

WHY THIS EXISTS - 2026-11-10.

The 02:00 data loop discarded a complete, healthy run. ``check_run_health`` reported
``HEALTH: PASS`` on every line - 499 scored, 499/499 priced, Spearman 0.989 against the
published page, every dispersion within its band - and then the publish gate failed on one
browser test, ``test_reporting_soon_is_a_calendar_not_a_leaderboard``, whose last assertion
ran ``stock_summary.advice_terms_in`` over the whole rendered Reporting Soon panel. The term
it found was ``Buy``, in **Best Buy Co., Inc.** Best Buy's next report fell on 2026-11-24,
exactly fourteen days after that run date, so it entered the panel's longer window for the
first time that morning. The live site stayed on 2026-11-06's data.

The page was right. It printed the company's legal name, which is what a calendar of
forthcoming reports is for. What was wrong was the check's premise: *no company reporting
within fourteen days of this run has a banned word in its name.* That is a statement about
S&P 500 membership and the earnings calendar, not about the screener, and it went false with
no commit touching anything - ``CLAUDE.md``'s ``data->red`` row, one step further out than
the cases already recorded there. Best Buy is the **only** name in the universe that trips
the ban (measured over all 499 rows of the 2026-11-06 and 2026-11-10 payloads), which is why
a check written months earlier fired on the fourth window it could have fired on.

Two things are tested here.

1. ``stock_summary.without_data_names`` does the masking **and still catches a real
   recommendation beside a masked name** - a no-op implementation passes a test that only
   checks the clean case, which is how the no-op pre-warm helper got through in 2026-11-05's
   fix (``CLAUDE.md`` ``paint->red``).
2. The sweep: every advice-language check over text read out of a live browser page masks
   first. There are three such call sites and one of them is the one that fired; the rule is
   uniform so that the fourth cannot be written naively. A check over the tool's own copy -
   a generator string, a template, a summary built from the payload's numbers - needs no
   masking and is not in scope: those read what this project wrote.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import stock_summary as ss  # noqa: E402

# How a test reads text out of a live browser page. `page.evaluate` is here because the two
# panels checked this way read `textContent` through it rather than through `inner_text`.
BROWSER_READS = ("inner_text(", "innerText", "textContent", "inner_html(")
MASK = "without_data_names"


# ---------------------------------------------------------------------------
# the masking itself
# ---------------------------------------------------------------------------

def test_a_company_name_is_not_a_recommendation():
    """The case that discarded the run, as the panel rendered it."""
    row = "BBY Best Buy Co., Inc. 14 72.5 ±5.9% +3.1% 2 bought"
    assert ss.advice_terms_in(row) == ["Buy"]
    assert ss.advice_terms_in(ss.without_data_names(row, ["BBY", "Best Buy Co., Inc."])) == []


def test_masking_a_name_does_not_mask_the_tool_saying_buy():
    """The half that makes the helper worth having. A no-op `without_data_names` passes the
    test above by accident; one that strips every occurrence of the word passes it on purpose
    and silently ends the ban. Both fail here."""
    sentence = "Best Buy Co., Inc. is a buy at this price."
    masked = ss.without_data_names(sentence, ["Best Buy Co., Inc."])
    assert "Best Buy" not in masked
    assert ss.advice_terms_in(masked) == ["buy"]


def test_masking_never_fuses_two_words_into_a_banned_one():
    """Each name goes out as a space, not as nothing: "SELL" must not be assembled out of
    the letters either side of a removed name."""
    assert ss.advice_terms_in(ss.without_data_names("Sel Lincoln l", ["Lincoln"])) == []
    assert "  " in ss.without_data_names("a Lincoln b", ["Lincoln"]) or \
           ss.without_data_names("a Lincoln b", ["Lincoln"]) == "a   b"


def test_a_longer_name_is_masked_before_a_shorter_one_inside_it():
    """Taking "Buy" out first would strand "Best  Co., Inc." and - worse - a list holding
    both "Best Buy" and "Buy Inc" would leave whichever came later in the iteration."""
    text = "Best Buy Co., Inc. and Buy Inc both report."
    assert ss.advice_terms_in(ss.without_data_names(text, ["Buy Inc", "Best Buy Co., Inc."])) == []


def test_masking_is_case_insensitive_like_the_ban():
    """The ban matches case-insensitively, so the masking must: a panel that upper-cases a
    company name in CSS still has the payload's capitalisation in its text, but a payload
    that ever carries "BEST BUY CO., INC." must mask too."""
    assert ss.advice_terms_in(ss.without_data_names("BEST BUY CO., INC.", ["Best Buy Co., Inc."])) == []


def test_an_empty_or_blank_name_list_changes_nothing():
    """A blank entry must not become a mask that eats the whole string."""
    assert ss.without_data_names("a buy", []) == "a buy"
    assert ss.advice_terms_in(ss.without_data_names("a buy", ["", "  ", None])) == ["buy"]


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_nodes(node):
    """Every descendant of ``node`` that is not inside a nested function or class.

    Scoping matters here, and getting it wrong is how the first version of this sweep
    reported `test_case_against.py` - a module where one function assigns
    ``text = page.evaluate(...innerText)`` and a *different* one checks its own local
    ``text``, built from generated sentences. Read module-wide, those are the same name."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPES):
            continue
        yield child
        yield from _own_nodes(child)


def _assignments(scope, src: str, inherited: dict[str, str]) -> dict[str, str]:
    out = dict(inherited)
    for node in _own_nodes(scope):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) and node.value else [])
        for t in targets:
            if isinstance(t, ast.Name):
                seg = ast.get_source_segment(src, node.value) or ""
                out[t.id] = (out.get(t.id, "") + " " + seg).strip()
    return out


def _ban_call_arguments(src: str) -> list[tuple[int, str]]:
    """The source of what every ``advice_terms_in(...)`` call in ``src`` was handed.

    Parsed, not joined by bracket depth: the first version of this sweep counted brackets
    line by line, a docstring's unbalanced parenthesis ran two hundred lines together, and it
    reported four call sites that do not exist.

    A bare name argument is **followed back to its assignment in the same scope**, because
    the natural way to write the fixed call site is two statements - read the panel, then
    check it - and a rule a local variable defeats is no rule. Following it again covers
    ``text = page.inner_text(...)`` then ``panel = mask(text, ...)``; the depth is bounded so
    a cycle cannot hang the sweep."""
    tree = ast.parse(src)
    out: list[tuple[int, str]] = []

    def walk_scope(scope, inherited: dict[str, str]) -> None:
        assigned = _assignments(scope, src, inherited)

        def expand(seg: str, depth: int = 3) -> str:
            if depth <= 0:
                return seg
            extra = [assigned[n] for n in set(re.findall(r"\b\w+\b", seg)) if n in assigned]
            return seg if not extra else seg + " " + " ".join(expand(e, depth - 1) for e in extra)

        for node in _own_nodes(scope):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "advice_terms_in":
                out.append((node.lineno, expand(ast.get_source_segment(src, node.args[0]) or "")))
        for child in _nested_scopes(scope):
            walk_scope(child, assigned)

    walk_scope(tree, {})
    return out


def _nested_scopes(scope):
    """The scopes directly inside ``scope`` - a nested def sees its parent's names."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, _SCOPES):
            yield child
        else:
            yield from _nested_scopes(child)


# The named place for a check that genuinely cannot mask, with the reason - the publish
# gate's `EXCLUDED` idiom. Keyed by module and test, never by line number. Emptying it
# quietly is not an option, and nor is adding to it without a reason that reads.
EXEMPT = {
    ("tests/test_holdings_panel.py",
     "test_a_name_this_run_did_not_score_still_renders_and_survives_edits"):
        "a node harness, not a browser: the payload is the three synthetic tickers the test "
        "itself writes into A.D, so there is no company name to mask and no page to read one "
        "from. The premise is asserted by the fixture rather than by the index.",
}


def _enclosing_test(src: str, line: int) -> str:
    """The name of the test the call on ``line`` is in: the latest ``test_*`` def above it."""
    defs = [(n.lineno, n.name) for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_") and n.lineno <= line]
    return max(defs)[1] if defs else ""


def _unmasked_browser_checks() -> list[str]:
    bad = []
    for path in sorted((ROOT / "tests").glob("test_*.py")) + [ROOT / "test_screener.py"]:
        if path.name == Path(__file__).name or not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        for line, arg in _ban_call_arguments(src):
            if not any(r in arg for r in BROWSER_READS) or MASK in arg:
                continue
            if (rel, _enclosing_test(src, line)) in EXEMPT:
                continue
            bad.append(f"{rel}:{line}  {arg[:120]}")
    return bad


def test_every_advice_check_over_rendered_text_masks_the_data_names():
    """A check over text a browser rendered must call `without_data_names` on it first.

    This is the rule the Reporting Soon check broke. It is syntactic rather than behavioural
    because the failure it prevents is latent: a surface that renders no company name today
    renders one as soon as a panel gains a column, and nothing fails until the data moves."""
    bad = _unmasked_browser_checks()
    assert bad == [], (
        "an advice-language check reads text a browser rendered and does not mask the data's "
        "own names first (stock_summary.without_data_names, tests/advice_ban.data_names):\n  "
        + "\n  ".join(bad))


NAIVE_SITE = '''
def t(page):
    assert advice_terms_in(page.inner_text("#reporting-body")) == []
'''
NAIVE_VIA_A_LOCAL = '''
def t(page):
    panel = page.inner_text("#reporting-body")
    assert advice_terms_in(panel) == []
'''
FIXED_SITE = '''
def t(page):
    panel = without_data_names(page.inner_text("#reporting-body"), data_names(page))
    assert advice_terms_in(panel) == []
'''


@pytest.mark.parametrize("src,flagged", [(NAIVE_SITE, True), (NAIVE_VIA_A_LOCAL, True),
                                         (FIXED_SITE, False)])
def test_the_sweep_can_actually_fail(src, flagged):
    """The sweep's own tripwire, on the exact statement that discarded the run and on the two
    ways of writing it that must not pass: inline, and behind a local variable. A sweep that
    only ever sees compliant source passes whatever it does."""
    args = _ban_call_arguments(src)
    assert len(args) == 1
    arg = args[0][1]
    assert any(r in arg for r in BROWSER_READS), arg      # it found the browser read either way
    assert (MASK not in arg) is flagged, arg


def test_every_exemption_is_real_and_says_why():
    """An exemption that names a test which no longer exists, or no longer reads rendered
    text, is coverage the sweep is not providing - and it reads as though it were. Printed
    every run, like the publish gate prints `EXCLUDED`."""
    for (rel, test), reason in sorted(EXEMPT.items()):
        print(f"EXEMPT {rel}::{test} - {reason}")
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert len(reason) > 60, (rel, test, "give a reason that reads")
        assert f"def {test}(" in src, (rel, test, "exempt test no longer exists")
        lines = [ln for ln, arg in _ban_call_arguments(src)
                 if any(r in arg for r in BROWSER_READS) and MASK not in arg
                 and _enclosing_test(src, ln) == test]
        assert lines, (rel, test, "exemption no longer needed - remove it")


def test_the_sweep_sees_at_least_one_browser_check_to_judge():
    """If the pattern list stops matching any real call site, the sweep above has quietly
    become decoration - the failure mode `CLAUDE.md` records for the pre-warm helper."""
    found = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        for line, arg in _ban_call_arguments(path.read_text(encoding="utf-8")):
            if any(r in arg for r in BROWSER_READS):
                found.append(f"{path.name}:{line}")
    assert found, "no advice check over rendered text found; the sweep judges nothing"
    print(f"{len(found)} advice check(s) over rendered text: {found}")


# ---------------------------------------------------------------------------
# the published payload, without pinning a stock
# ---------------------------------------------------------------------------

def _payload() -> dict:
    import json
    p = ROOT / "dashboard_data.js"
    if not p.exists():
        pytest.skip("MISSING DATA: dashboard_data.js")
    txt = p.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rstrip().rstrip(";").rindex("}") + 1])


def test_the_masking_clears_every_company_name_this_run_published():
    """Whatever today's membership is, masking the payload's own names leaves no banned term
    behind in any of them.

    Deliberately **not** "Best Buy is in the index": that is the premise the broken check
    asserted. If Best Buy leaves the S&P 500 this passes vacuously and correctly, and the
    count below is reported rather than bounded."""
    payload = _payload()
    rows = payload.get("table_data") or []
    names = [str(r.get("Company") or "") for r in rows] + [str(r.get("Ticker") or "") for r in rows]
    tripping = [n for n in names if ss.advice_terms_in(n)]
    print(f"{len(tripping)} of {len(names)} published names trip the ban: {sorted(set(tripping))}")
    for name in names:
        assert ss.advice_terms_in(ss.without_data_names(name, names)) == [], name


def test_the_page_really_hands_over_its_names():
    """`data_names` must read the payload **out of the page**, not quietly return nothing.

    This is the mistake this session made: the first version read `window.D`, but the page
    declares `const D = window.SCREENER_DATA || {}` at script scope, so `D` is not a property of
    `window` - the same shape as the closure-scoped `CTX_LOADED` the dashboard notes warn about.
    It returned `[]`, masked nothing, and the rehearsal failed exactly as the 02:00 run had. A
    helper that can silently do nothing is worse than no helper: every call site reads as covered.
    """
    page_file = ROOT / "index.html"
    if not page_file.exists():
        pytest.skip("MISSING DATA: index.html")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:                                        # pragma: no cover
        pytest.skip("playwright is not installed")
    from tests.advice_ban import data_names
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:                               # noqa: BLE001 - no Chromium
            pytest.skip(f"no usable chromium: {exc}")
        page = browser.new_page()
        page.goto(page_file.as_uri(), wait_until="load", timeout=60000)
        page.wait_for_function("typeof D !== 'undefined' && D.table_data && D.table_data.length",
                               timeout=30000)
        names = data_names(page)
        rows = page.evaluate("D.table_data.length")
        browser.close()
    assert len(names) == 2 * rows, (len(names), rows)
    assert all(isinstance(n, str) for n in names)
    print(f"{len(names)} names read from the page for {rows} rows")


def test_data_names_refuses_a_page_with_no_payload():
    """The guard that makes the test above hold for every caller, without a browser."""
    from tests.advice_ban import data_names

    class _Blank:
        def evaluate(self, _js):
            return []

    with pytest.raises(AssertionError, match="no table_data"):
        data_names(_Blank())


def test_a_banned_term_in_a_name_is_the_only_way_a_name_trips_the_ban():
    """Guards the reasoning rather than the data: the masking is justified only because the
    hits come from company names, not from the panel's own copy. So a name that trips the ban
    must contain a banned term as a whole word - never a fragment of one."""
    payload = _payload()
    for row in payload.get("table_data") or []:
        name = str(row.get("Company") or "")
        for hit in ss.advice_terms_in(name):
            assert re.search(r"\b" + re.escape(hit) + r"\b", name, re.IGNORECASE), (name, hit)
