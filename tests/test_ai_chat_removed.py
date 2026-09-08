"""The "Screener AI" chat is gone, and its replacement is in place.

Owner directive 2026-08-10 (``plan/dashboard-north-star.md``, "Replace the
chatbot with generated summaries"), shipped 2026-09-08.

**Why it had to go.** It asked every visitor to paste their own Anthropic API
key into ``localStorage`` and then called ``api.anthropic.com`` from the
browser. For the investment-club audience that meant a paid API account per
student, a lesson in pasting credentials into web pages, a cost per question,
and two students asking the same question getting different answers. The last
one is the one that mattered: this tool's claim is auditability, and an
un-reproducible explainer works against it.

**Why these tests exist at all.** The model-portfolio removal on 2026-08-26
taught the lesson: nothing tested that surface, so nothing would have noticed
it silently coming back or half-leaving. And a *partial* removal is worse than
either state - an ``onclick="toggleChat()"`` left behind after the function is
deleted throws at click time, and a dangling identifier in the script body
blanks the entire page while all four ship gates stay green.

So this module asserts both halves: the chat is entirely absent, **and** the
deterministic summary that replaced it renders. Failing either is a failure.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

CATEGORIES = ["valuation", "quality", "growth", "momentum",
              "risk", "revisions", "size", "investment"]


def _frame() -> pd.DataFrame:
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC", "DDD"],
        "Company": ["Alpha Inc", "Beta Corp", "Gamma Ltd", "Delta SA"],
        "Sector": ["Information Technology", "Financials", "Energy", "Utilities"],
        "Composite": [80.0, 60.0, 55.0, 40.0],
        "Rank": [1, 2, 3, 4],
        "Value_Trap_Flag": [False, False, True, False],
        "Growth_Trap_Flag": [False, False, False, True],
        "_about": ["Alpha Inc designs and sells things worldwide.", np.nan,
                   "Gamma Ltd explores for and produces crude oil.", ""],
        "_industry": ["Software - Infrastructure", "Banks - Diversified",
                      "Oil & Gas E&P", ""],
        "_current_price": [22.05, 100.0, 50.0, 10.0],
        "_target_mean": [25.14, 90.0, 55.0, np.nan],
        "_num_analysts": [20, 3, 11, 0],
        "_metric_count": [18, 15, 12, 18],
        "_metric_total": [18, 18, 18, 18],
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = 50.0
        df[cat + "_contrib"] = 6.25
    return df


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(g.prepare_dashboard_data({
        "df": _frame(), "meta": {"run_date": "2026-09-08"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    }))


@pytest.fixture(scope="module")
def html() -> str:
    return g.generate_html()


@pytest.fixture(scope="module")
def source() -> str:
    return (REPO / "generate_dashboard.py").read_text(encoding="utf-8")


def _code_only(block: str) -> str:
    """Strip ``//`` comment lines and HTML comments.

    The generator carries deliberate prose about *why* the chat was removed;
    an assertion about behaviour must not trip over its own provenance note.
    """
    block = re.sub(r"<!--.*?-->", "", block, flags=re.S)
    return "\n".join(ln for ln in block.split("\n")
                     if not ln.lstrip().startswith("//"))


# ---------------------------------------------------------------------------
# 1. Nothing about the chat reaches the published page
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol", [
    # Functions. Any of these left referenced but undefined kills the script.
    "toggleChat", "sendMessage", "callClaude", "clearChat",
    "openApiKeyDialog", "closeApiKeyDialog", "saveSettings",
    "appendChatMsg", "parseChatMd", "showChatTyping", "removeChatTyping",
    "handleChatKeydown", "autoResizeInput", "useSuggestion",
    "initChatWelcome", "loadChatSize", "initChatResize", "updateModelBadge",
    "buildSystemPrompt", "buildUserContext", "buildStockCtx",
    "detectQueryType", "extractTickers", "extractSectors",
    "chatState", "STARTER_QUESTIONS",
    # Element ids and classes.
    "chat-panel", "chat-fab", "chat-messages", "chat-input", "chat-send-btn",
    "chat-api-dialog", "chat-api-input", "chat-model-select",
    "chat-header-model", "chat-suggestions", "chat-resize-handle",
    # Animations that existed only for it.
    "chatSlideUp", "chatFabPulse", "chatDotBlink",
])
def test_no_chat_symbol_survives_in_the_page(html, symbol):
    assert symbol not in _code_only(html)


@pytest.mark.parametrize("secret", [
    "api.anthropic.com",
    "console.anthropic.com",
    "screener_anthropic_api_key",
    "screener_chat_model",
    "sk-ant-",
    "x-api-key",
    "anthropic-version",
    "anthropic-dangerous-direct-browser-access",
])
def test_the_page_no_longer_asks_for_or_sends_an_api_key(html, secret):
    """The credential-handling behaviour is the reason the owner asked for
    this, so it is asserted separately from the cosmetic symbols above."""
    assert secret not in _code_only(html)


def test_the_page_makes_no_outbound_request_to_a_model_provider(html):
    assert "anthropic" not in _code_only(html).lower()


def test_no_model_id_is_hardcoded_in_the_page(html):
    """The chat shipped a model picker. A stale model id on a public page is a
    maintenance liability with nothing left to serve it."""
    assert not re.search(r"claude-(opus|sonnet|haiku)", _code_only(html))


def test_localstorage_is_used_only_for_layout_preferences(html):
    """No backend exists, so ``localStorage`` is legitimate - but it must not
    hold anything a user would be harmed by leaking."""
    keys = set(re.findall(r"localStorage\.\w+\(\s*'([^']+)'", _code_only(html)))
    assert not any("key" in k or "anthropic" in k or "chat" in k for k in keys), keys


# ---------------------------------------------------------------------------
# 2. No dangling references - the failure mode that blanks the page silently
# ---------------------------------------------------------------------------

def test_every_inline_event_handler_resolves_to_a_defined_function(html):
    code = _code_only(html)
    handlers = set(re.findall(r'on(?:click|input|keydown|change|mousedown)="(\w+)\(',
                              html))
    handlers -= {"if"}          # `onclick="if(event.target===this)closeModal()"`
    undefined = sorted(h for h in handlers
                       if not re.search(r"function\s+%s\s*\(" % h, code))
    assert undefined == [], undefined


def test_the_generated_script_still_parses(tmp_path):
    """A syntax error from a bad deletion blanks the public page while every
    ship gate stays green. Checked with a real parser when one is available."""
    node = _node_or_skip()
    html = g.generate_html()
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>", html)
    assert blocks, "the page has no inline script at all"
    for i, block in enumerate(blocks):
        path = tmp_path / f"block{i}.js"
        path.write_text(block, encoding="utf-8")
        result = _run_node(node, ["--check", str(path)])
        assert result.returncode == 0, result.stderr


def _node_or_skip():
    import shutil
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed; JS syntax check unavailable")
    return node


def _run_node(node, args):
    import subprocess
    return subprocess.run([node, *args], capture_output=True, text=True, timeout=60)


# A parser-free brace-balance backstop was written for this slot and removed
# the same session: JavaScript regex literals make it unsound. `escapeHtml`
# contains `.replace(/'/g, '&#39;')`, and a scanner without regex-literal
# support reads that apostrophe as the start of a string, desynchronising
# everything after it. It reported the page as unbalanced when `node --check`
# passed. A check that fires on healthy code is the failure mode CLAUDE.md
# rule 7 and the 2026-09-01 bank-metrics fix are both about: it trains a
# reader to ignore it, which is exactly when the real defect gets through.
# The node check above is the syntax gate; where node is absent it skips
# visibly rather than pretending to cover.


# ---------------------------------------------------------------------------
# 3. config_traps went with it - it had no other consumer
# ---------------------------------------------------------------------------

def test_config_traps_key_is_gone_from_the_payload(payload):
    """It existed only to fill the chat's system prompt; nothing rendered it.
    The same thresholds are published in the Methodology section, which
    ``run_screener.generate_screener_overview()`` templates from ``config.yaml``.
    Same reasoning that retired ``spx_weights`` on 2026-08-26."""
    assert "config_traps" not in payload


def test_nothing_in_the_page_reads_config_traps(html):
    assert "config_traps" not in _code_only(html)


# ---------------------------------------------------------------------------
# 4. The replacement is actually there - a half-done swap is the bad state
# ---------------------------------------------------------------------------

def test_every_stock_carries_a_summary(payload):
    detail = payload["stock_detail"]
    assert detail
    for ticker, stock in detail.items():
        assert stock.get("summary"), ticker


def test_the_summary_block_renders_in_the_drilldown(html):
    for needle in ['id="modal-summary"', 'id="modal-summary-body"',
                   "function renderSummary", "renderSummary(s)",
                   ".summary-fact", "Why it ranks here"]:
        assert needle in html, needle


def test_the_summary_is_rendered_as_escaped_text_not_html(html):
    """The chat rendered model output as HTML through ``parseChatMd``. Its
    replacement must not reintroduce an injection surface, even though the
    text is generated locally."""
    body = re.search(r"function renderSummary\(s\)[\s\S]*?\n    \}", html)
    assert body, "renderSummary not found"
    assert "escapeHtml(f.t" in body.group(0)
    assert "innerHTML = facts.map" in body.group(0)


def test_the_summary_block_says_it_is_not_advice(html):
    """``plan/dashboard-north-star.md``: decision support, not a recommendation
    engine. The disclaimer is part of the feature, not decoration."""
    assert "not investment advice" in html
    assert "never says whether to buy" in html


def test_summaries_on_the_rendered_payload_carry_no_advice_language(payload):
    import stock_summary as ss
    for ticker, stock in payload["stock_detail"].items():
        text = ss.summary_text(stock["summary"])
        assert ss.advice_terms_in(text) == [], (ticker, text)


def test_the_summary_is_built_at_run_time_not_in_the_browser(source, html):
    """Baked into the payload so what shipped is what a reader can diff - and
    so two readers cannot see different explanations, which is the whole
    reason the chat was replaced."""
    assert "from stock_summary import build_summary" in source
    assert "build_summary(" in source
    # The front end renders `s.summary`; it must not compute sentences itself.
    assert "s.summary" in html
    assert "build_summary" not in _code_only(html)


# ---------------------------------------------------------------------------
# 5. Nothing else on the page was collateral damage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("needle", [
    'id="sec-defensibility"', "sensitivity-table", "correlation-heatmap",
    "universe-table", "methodology-modal", "modal-about",
    "renderStockHistory", "renderPeerComparison", "renderProvenance",
    "renderContribVisual", "renderCategoryDetails", "renderChanged",
])
def test_protected_surfaces_survived_the_removal(html, needle):
    """CLAUDE.md rule 7: the defensibility features may be redesigned but never
    quietly dropped. The chat CSS block sat directly above the responsive rules
    and the JS block directly above the collapsible-section handlers."""
    assert needle in html


def test_the_responsive_and_print_rules_survived(html):
    """The chat's mobile media query was removed from between them."""
    assert "@media (max-width: 768px)" in html
    assert "@media print" in html


def test_the_escape_key_still_closes_the_modals(html):
    block = re.search(r"'Escape'[\s\S]{0,300}", html).group(0)
    assert "closeModal()" in block
    assert "closeMethodology()" in block
