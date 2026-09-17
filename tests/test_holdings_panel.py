"""The My Holdings panel - priority 5 / north-star gap 2, the sell-side workflow.

Until 2026-09-15 the dashboard could not answer *should I sell what I hold?* at
all: there was no watchlist, no holdings list and no way to see what the
screener knew about a name you already owned without looking it up one at a
time. `plan/dashboard-inventory.md` listed it as genuinely missing.

The panel that fills the gap has three properties that are **research
constraints, not styling**, and this module exists to stop a future session
tidying them away. All three come from
`research/2026-09-14-sell-discipline-and-hold-bands.md`:

1. **Every saved name renders, every time.** Akepanidtaworn, Di Mascio, Imas &
   Schmidt (2023, *Journal of Finance* 78(6), 3055-3098) trace an 80 bp/year
   institutional selling deficit to a restricted consideration set: PMs dispose
   of positions extreme on prior returns at rates >50% higher than middling
   ones. A queue that surfaces only the big movers is that heuristic,
   implemented and shipped as a feature.
2. **Rows are ordered by current rank, never by size of move** - same source.
   The rank change is shown for context; it is not the sort key and not a
   filter.
3. **No cost basis, share count or profit-and-loss anywhere**, in the code or
   in what it stores. Gain/loss against purchase price is the reference point
   that produces the disposition effect (Odean 1998, *JF* 53(5): PGR 0.233 vs
   PLR 0.155, a 1.50x ratio at t = -32; the winners sold beat the losers held
   by 3.41% over the following year).

The functional half drives the real emitted script under Node against a stubbed
DOM, because "the string appears in the file" is a weak check for a panel whose
whole contract is about what it renders. The `node --check` parse gate lives in
`tests/test_dashboard_js.py`; this module runs the thing.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402
import stock_summary  # noqa: E402

NODE = shutil.which("node")


# ---------------------------------------------------------------------------
# Static checks on the emitted artifact
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def html() -> str:
    return g.generate_html()


@pytest.fixture(scope="module")
def script(html) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == 1
    return blocks[0]


@pytest.mark.parametrize("hook", [
    "sec-holdings",
    "holdings-count",
    "holdings-search-input",
    "holdings-search-results",
    "holdings-clear-btn",
    "holdings-fit",
    "holdings-body",
    "holdings-footnote",
])
def test_holdings_dom_hooks_present(html, hook):
    assert f'id="{hook}"' in html, f"{hook} missing from the emitted page"


@pytest.mark.parametrize("fn", [
    "initHoldings", "renderHoldings", "holdingCard", "holdingsFitLine",
    "holdingDelta", "loadHoldings", "saveHoldings", "addHolding",
    "removeHolding", "clearHoldings", "setupHoldingsSearch",
    "renderHoldingsFootnote",
])
def test_holdings_functions_emitted(script, fn):
    assert f"function {fn}(" in script


def test_init_calls_init_holdings(script):
    """A panel nothing calls is a panel nobody sees."""
    init = script.split("// INIT")[-1]
    assert "initHoldings();" in init


def test_holdings_section_sits_above_what_changed(html):
    """Placement is part of the point: a holder should meet their own names
    before the universe-wide movers panel, which is a discovery surface and
    (measured) fires for a top-25 name 0.15% of the time."""
    assert html.index('id="sec-holdings"') < html.index('id="sec-changed"')


# ---------------------------------------------------------------------------
# Constraint 3: nothing resembling a cost basis may exist
# ---------------------------------------------------------------------------

# Matched case-insensitively against the holdings code only. Deliberately
# blunt: a false positive costs one renamed identifier, a false negative ships
# the disposition effect's reference point onto a public decision surface.
COST_BASIS_TERMS = [
    "cost_basis", "costbasis", "cost basis", "purchase price", "purchasePrice",
    "avg_price", "avgprice", "avg cost", "book cost", "entry price",
    "entryPrice", "shares", "quantity", "position_size", "positionSize",
    "unrealised", "unrealized", "gain/loss", "gainloss", "gain_loss",
    "profit", "p&l", "pnl", "paid",
]


def _holdings_code(script: str) -> str:
    """The emitted holdings block, from its banner to the stock-detail banner."""
    start = script.index("// MY HOLDINGS")
    end = script.index("// STOCK DETAIL MODAL", start)
    return script[start:end]


@pytest.mark.parametrize("term", COST_BASIS_TERMS)
def test_no_cost_basis_anywhere_in_holdings_code(script, term):
    """Odean (1998). The panel must not offer the reference point at all.

    Note the footnote deliberately *says* it stores no cost basis - that copy
    lives in renderHoldingsFootnote and is checked separately - so this looks
    at identifiers and rendered fields, excluding the footnote prose.
    """
    code = _holdings_code(script)
    # Two exclusions, both prose rather than behaviour: the footnote copy
    # deliberately *says* no cost basis is stored (checked separately below),
    # and the block comment explains why. What must stay clean is the code.
    footnote_start = code.index("function renderHoldingsFootnote(")
    footnote_end = code.index("function setupHoldingsSearch(", footnote_start)
    code = code[:footnote_start] + code[footnote_end:]
    code = "\n".join(line for line in code.splitlines()
                     if not line.lstrip().startswith("//"))
    assert term.lower() not in code.lower(), (
        f"{term!r} appears in the holdings panel - see Odean (1998); this "
        "surface must not carry a cost-basis reference point")


def test_stored_value_is_tickers_only(script):
    """Whatever is in localStorage, only strings that name a scored ticker
    survive a read - so a hand-edited key cannot smuggle a position in."""
    code = _holdings_code(script)
    assert "typeof item === 'string'" in code
    assert "!D.stock_detail[t]" in code


# ---------------------------------------------------------------------------
# Constraints 1 and 2: show everything, order by rank
# ---------------------------------------------------------------------------

def test_rows_are_sorted_by_rank_not_by_move(script):
    code = _holdings_code(script)
    render = code[code.index("function renderHoldings("):
                  code.index("function holdingsFitLine(")]
    assert ".sort(" in render
    sort_at = render.index(".sort(")
    comparator = render[sort_at:render.index("fit.innerHTML", sort_at)]
    # The only ordering key is rank.
    assert "a.s.rank" in comparator and "b.s.rank" in comparator
    for forbidden in (".dr", "Math.abs", "holdingDelta", "composite"):
        assert forbidden not in comparator, (
            f"{forbidden!r} appears in the holdings sort comparator - rows "
            "must never be ordered by size of move")
    # And no filter narrows the list: the only .filter() drops tickers this
    # run did not score, which is a data check, not a consideration-set cut.
    assert render.count(".filter(") == 1
    assert "!!r.s" in render


def test_footnote_cites_its_three_sources(script):
    code = _holdings_code(script)
    footnote = code[code.index("function renderHoldingsFootnote("):]
    for citation in ("Akepanidtaworn", "Odean", "Novy-Marx",
                     "Barber", "S&amp;P Dow Jones Indices", "MSCI"):
        assert citation in footnote, f"{citation} missing from the footnote"


def test_footnote_states_the_storage_reality(script):
    code = _holdings_code(script)
    footnote = code[code.index("function renderHoldingsFootnote("):]
    assert "localStorage" in footnote
    assert "not an account" in footnote
    assert "not backed up" in footnote


def test_no_bare_recommendation_in_panel_copy(html):
    """The north-star line with teeth: decision support, never a bare verdict.

    The page may *describe* what it declines to do, so this looks for the
    imperative forms rather than the words themselves.
    """
    lowered = html.lower()
    for phrase in ("you should sell", "you should buy", "time to sell",
                   "sell now", "buy now", "recommended action"):
        assert phrase not in lowered


# ---------------------------------------------------------------------------
# Functional: drive the emitted script under Node with a stubbed DOM
# ---------------------------------------------------------------------------

_DOM_STUB = r"""
const els = {};
function el(id) {
  if (!els[id]) {
    const classes = new Set(id === 'sec-holdings' ? ['collapsed'] : []);
    els[id] = {
      id: id, innerHTML: '', textContent: '', value: '',
      style: {}, _listeners: {},
      classList: {
        add: c => classes.add(c), remove: c => classes.delete(c),
        contains: c => classes.has(c),
        toggle: c => (classes.has(c) ? (classes.delete(c), false)
                                     : (classes.add(c), true)),
        _all: () => Array.from(classes),
      },
      addEventListener: function (ev, fn) { (this._listeners[ev] ||= []).push(fn); },
      dispatchEvent: function (e) {
        (this._listeners[e.type] || []).forEach(fn => fn.call(this, e));
      },
      insertAdjacentHTML: function () {},
    };
  }
  return els[id];
}
globalThis.__els = els;
globalThis.document = {
  getElementById: el,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => ({ style: {}, appendChild() {} }),
  addEventListener: () => {},
  body: { style: {} },
};
const store = {};
globalThis.window = {
  SCREENER_DATA: JSON.parse(process.argv[2]),
  confirm: () => true,
  localStorage: {
    getItem: k => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: k => { delete store[k]; },
  },
};
globalThis.__store = store;
globalThis.Event = class { constructor(t) { this.type = t; } };
"""

_EPILOGUE = r"""
globalThis.__api = {
  initHoldings, renderHoldings, addHolding, removeHolding, clearHoldings,
  holdingCard, holdingsFitLine, holdingDelta, loadHoldings,
  HOLDINGS_KEY, HOLDINGS_MAX, HOLDINGS_FACTS,
  D,
  current: () => holdings.slice(),
};
"""


def _payload() -> dict:
    """A small payload with the shapes the panel reads.

    ``table_data`` starts empty so the script's INIT block short-circuits and
    no chart or table render runs against the stub DOM; the test fills it in
    and calls ``initHoldings()`` itself.
    """
    def detail(ticker, company, sector, rank, composite, **kw):
        d = {
            "company": company, "sector": sector, "rank": rank,
            "composite": composite,
            "cat_scores": {c: 50.0 for c in stock_summary.CATEGORIES},
            "contrib": {c: composite / 8 for c in stock_summary.CATEGORIES},
            "vt": False, "gt": False, "summary": [],
        }
        d.update(kw)
        return d

    stock_detail = {
        "AAA": detail("AAA", "Alpha Industries", "Industrials", 3, 71.0,
                      summary=[
                          {"k": "rank", "t": "Ranks 3rd of 4."},
                          {"k": "change", "t": "Since the run of 2026-08-14 it has moved down 2 places."},
                          {"k": "change_driver", "t": "Its largest category move since the run of 2026-08-14 is Growth, with that score down 9.0 points."},
                          {"k": "peers", "t": "Peer sentence that must not appear on a holdings row."},
                          {"k": "flags", "t": "It carries no value-trap, growth-trap or accounting flag."},
                          {"k": "confidence", "t": "The score rests on 17 of 18 metrics."},
                      ]),
        "BBB": detail("BBB", "Beta Corp", "Industrials", 1, 80.0, vt=True),
        "CCC": detail("CCC", "Gamma PLC", "Health Care", 40, 52.0,
                      cat_scores={**{c: 50.0 for c in stock_summary.CATEGORIES},
                                  "momentum": None}),
        "DDD": detail("DDD", "Delta SA", "Financials", 300, 31.0),
    }
    table_data = [
        {"Ticker": t, "Company": d["company"], "Sector": d["sector"],
         "Composite": d["composite"], "Rank": d["rank"],
         "Value_Trap_Flag": d["vt"], "Growth_Trap_Flag": d["gt"]}
        for t, d in stock_detail.items()
    ]
    history = {
        "available": True,
        "current_date": "2026-09-15",
        "dates": ["2026-08-14", "2026-09-14", "2026-09-15"],
        "compare": {"prev": {"date": "2026-09-14", "gap_days": 1},
                    "m1": {"date": "2026-08-14", "gap_days": 32}},
        "delta": {
            "AAA": {"prev": {"dr": 0}, "m1": {"dr": -2, "dc": -3.1,
                                              "cat": {"growth": -9.0, "risk": 2.0}}},
            "BBB": {"prev": {"dr": 1}, "m1": {"dr": 5, "dc": 2.0,
                                              "cat": {"valuation": 4.0}}},
            "CCC": {"m1": {"new": True}},
        },
        "movers": {}, "noise": {"material_threshold": 42}, "excluded": [],
        "series": {},
    }
    return {
        "kpis": {"universe_size": 4, "value_traps": 1, "growth_traps": 0,
                 "run_timestamp": None},
        "table_data": [],
        "_rows": table_data,
        "stock_detail": stock_detail,
        "history": history,
        "weights": {"factor_weights": {c: 12.5 for c in stock_summary.CATEGORIES},
                    "metric_weights": {}},
        "metric_meta": {},
        "sectors": ["Industrials", "Health Care", "Financials"],
    }


def _run_js(script: str, body: str, tmp_path: Path) -> dict:
    payload = _payload()
    src = "\n".join([
        _DOM_STUB,
        script,
        _EPILOGUE,
        "const A = globalThis.__api;",
        "A.D.table_data.push(...A.D._rows);",
        body,
    ])
    # .cjs, not .mjs: the emitted script is a classic <script> body, and
    # ES-module strict mode rejects the duplicate top-level function
    # declarations that a browser accepts. `node --check` in
    # tests/test_dashboard_js.py parses it the same (sloppy) way a browser does.
    path = tmp_path / "harness.cjs"
    path.write_text(src, encoding="utf-8")
    proc = subprocess.run([NODE, str(path), json.dumps(payload)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[:4000]
    return json.loads(proc.stdout.strip().splitlines()[-1])


pytestmark_node = pytest.mark.skipif(NODE is None, reason="node not available")


@pytestmark_node
def test_every_saved_name_renders_in_rank_order(script, tmp_path):
    """Constraint 1 and 2 together, observed on rendered output.

    DDD ranks 300th and moved not at all; a move-ranked or move-filtered queue
    would drop it. It must appear, and it must appear last.
    """
    out = _run_js(script, """
        A.initHoldings();
        ['DDD', 'AAA', 'BBB', 'CCC'].forEach(A.addHolding);
        const html = globalThis.__els['holdings-body'].innerHTML;
        const order = [...html.matchAll(/class="holding-ticker"[^>]*>([A-Z]+)</g)]
            .map(m => m[1]);
        console.log(JSON.stringify({
            order: order,
            stored: JSON.parse(globalThis.__store[A.HOLDINGS_KEY]),
            count: globalThis.__els['holdings-count'].textContent,
        }));
    """, tmp_path)
    assert out["order"] == ["BBB", "AAA", "CCC", "DDD"]
    # Insertion order is preserved in storage; only the render is re-ordered.
    assert out["stored"] == ["DDD", "AAA", "BBB", "CCC"]
    assert out["count"] == "(4)"


@pytestmark_node
def test_a_name_with_no_history_still_renders(script, tmp_path):
    """CCC is new to the universe and has a withheld category. Neither may
    silently drop it from a review surface."""
    out = _run_js(script, """
        A.initHoldings();
        A.addHolding('CCC');
        const html = globalThis.__els['holdings-body'].innerHTML;
        console.log(JSON.stringify({
            present: html.includes('>CCC<'),
            no_history: html.includes('no comparable history'),
            no_data_cat: html.includes('no data'),
        }));
    """, tmp_path)
    assert out["present"] is True
    assert out["no_history"] is True
    assert out["no_data_cat"] is True


@pytestmark_node
def test_review_notes_are_the_baked_summary_facts(script, tmp_path):
    """The panel renders prose built by stock_summary.py at build time, which
    is what keeps it diffable and advice-screened. It must not compose its own,
    and must not spill the drilldown's other sentences onto a review row."""
    out = _run_js(script, """
        A.initHoldings();
        A.addHolding('AAA');
        const html = globalThis.__els['holdings-body'].innerHTML;
        console.log(JSON.stringify({
            facts: A.HOLDINGS_FACTS,
            has_change: html.includes('moved down 2 places'),
            has_driver: html.includes('largest category move'),
            has_flags: html.includes('no value-trap'),
            has_confidence: html.includes('17 of 18 metrics'),
            leaked_peers: html.includes('must not appear'),
            leaked_rank: html.includes('Ranks 3rd of 4'),
        }));
    """, tmp_path)
    # `input_churn` joined the list 2026-09-17 - the caveat saying part of a
    # move may be an input going missing rather than the company changing. It
    # sits directly after the two sentences it qualifies, so a reader cannot
    # meet the move without it. See tests/test_input_churn.py.
    assert out["facts"] == ["change", "change_driver", "input_churn",
                            "flags", "confidence"]
    assert out["has_change"] and out["has_driver"]
    assert out["has_flags"] and out["has_confidence"]
    assert out["leaked_peers"] is False
    assert out["leaked_rank"] is False


@pytestmark_node
def test_fit_line_reports_concentration_not_position_sizes(script, tmp_path):
    out = _run_js(script, """
        A.initHoldings();
        ['AAA', 'BBB', 'CCC'].forEach(A.addHolding);
        console.log(JSON.stringify({
            fit: globalThis.__els['holdings-fit'].innerHTML,
        }));
    """, tmp_path)
    fit = out["fit"]
    assert "3</strong> names" in fit
    assert "2</strong> sectors" in fit
    assert "Industrials" in fit and "2 of 3, 67%" in fit
    assert "inside the top 25 of 4" in fit
    assert "1</strong> carrying a trap flag" in fit
    assert "%" in fit and "$" not in fit  # no currency anywhere


@pytestmark_node
def test_round_trips_through_storage_and_survives_junk(script, tmp_path):
    """A key holding a position dict, an unknown ticker and a duplicate is read
    for its tickers and written back clean."""
    out = _run_js(script, """
        globalThis.__store[A.HOLDINGS_KEY] = JSON.stringify(
            ['AAA', 'aaa', 'ZZZ', {ticker: 'BBB', shares: 100, cost: 42.5}, 'CCC']);
        A.initHoldings();
        console.log(JSON.stringify({
            loaded: A.current(),
            rewritten: JSON.parse(globalThis.__store[A.HOLDINGS_KEY] || 'null'),
            expanded: !globalThis.__els['sec-holdings'].classList.contains('collapsed'),
        }));
    """, tmp_path)
    assert out["loaded"] == ["AAA", "CCC"]
    assert out["expanded"] is True


def test_empty_section_ships_collapsed(html):
    """The owner's landing view is Top 5 plus the full table, everything else
    one click away (OWNER_FOCUS.md, 2026-08-26). An empty holdings panel must
    not push the table down the page."""
    section = html[html.index('id="sec-holdings"') - 200:
                   html.index('id="sec-holdings"') + 20]
    assert "collapsed" in section


@pytestmark_node
def test_empty_list_explains_itself_and_does_not_expand(script, tmp_path):
    out = _run_js(script, """
        A.initHoldings();
        console.log(JSON.stringify({
            body: globalThis.__els['holdings-body'].innerHTML,
            footnote_len: globalThis.__els['holdings-footnote'].innerHTML.length,
            fit: globalThis.__els['holdings-fit'].innerHTML,
            // Never looked up, so the emitted `collapsed` class is untouched.
            touched_section: Object.prototype.hasOwnProperty.call(
                globalThis.__els, 'sec-holdings'),
            count: globalThis.__els['holdings-count'].textContent,
        }));
    """, tmp_path)
    assert "Nothing on the list yet" in out["body"]
    assert out["footnote_len"] > 800, "the sourced footnote must show when empty"
    assert out["fit"] == ""
    assert out["touched_section"] is False
    assert out["count"] == ""


@pytestmark_node
def test_remove_and_clear(script, tmp_path):
    out = _run_js(script, """
        A.initHoldings();
        ['AAA', 'BBB'].forEach(A.addHolding);
        A.removeHolding('AAA');
        const afterRemove = A.current();
        A.clearHoldings();
        console.log(JSON.stringify({
            afterRemove: afterRemove,
            afterClear: A.current(),
            stored: JSON.parse(globalThis.__store[A.HOLDINGS_KEY]),
            body: globalThis.__els['holdings-body'].innerHTML.slice(0, 60),
        }));
    """, tmp_path)
    assert out["afterRemove"] == ["BBB"]
    assert out["afterClear"] == []
    assert out["stored"] == []


@pytestmark_node
def test_unknown_ticker_is_refused(script, tmp_path):
    out = _run_js(script, """
        A.initHoldings();
        A.addHolding('NOPE');
        A.addHolding('');
        A.addHolding('aaa');
        A.addHolding('AAA');
        console.log(JSON.stringify({held: A.current()}));
    """, tmp_path)
    assert out["held"] == ["AAA"]


@pytestmark_node
def test_delta_prefers_the_one_month_window(script, tmp_path):
    """Same preference order as stock_summary._pick_comparison, so the chips
    and the sentence under them cannot describe different windows."""
    out = _run_js(script, """
        A.initHoldings();
        console.log(JSON.stringify({aaa: A.holdingDelta('AAA'), ccc: A.holdingDelta('CCC')}));
    """, tmp_path)
    assert out["aaa"]["dr"] == -2
    assert out["aaa"]["date"] == "2026-08-14"
    # CCC is new in the 1m window and has no prev entry at all.
    assert out["ccc"] is None
