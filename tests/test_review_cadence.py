"""The review cadence the tool is built for, stated on the surfaces that move.

`research/2026-09-14-sell-discipline-and-hold-bands.md` §8.2, and the first item
that note listed for the build day.

The defect: `config.yaml` has recorded a quarterly rebalance cadence since
launch, the data loop regenerates the site **every weekday**, and until
2026-09-17 `generate_dashboard.py` said nothing about cadence anywhere - a grep
for "quarterly" found one unrelated data-source label. A surface that redraws a
rank every morning implicitly invites a reader to act on it every morning.

The methodology's own answer to that, measured on this repo's snapshots: acting
on the strict top-25 rule at every run implies **121.8%** monthly one-sided
turnover against **24.0%** reviewing the same rule monthly. Novy-Marx & Velikov
(2016, *RFS* 29(1) 104-147) find anomalies under roughly **50%** monthly
one-sided turnover mostly survive trading costs and few above it do - so daily
action sits 2.4x outside the surviving region, a gap that tolerates a large
error in the estimate before it reverses.

This is a product defect, not a methodology one, and the fix is a sentence
rather than a lock. The tool does not know what a reader is doing and must not
pretend to: naming the cadence it was built for is decision support; refusing to
show a number until a date would not be.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


# ---------------------------------------------------------------------------
# config.yaml carries the cadence as data
# ---------------------------------------------------------------------------

def test_config_declares_a_review_cadence():
    """It was a bare comment until 2026-09-17. A comment cannot be read by the
    generator, which is why the dashboard could not state it."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["portfolio"]["review_cadence"] in g.CADENCE_LABELS


def test_config_cadence_cites_its_source():
    """The turnover numbers are the whole argument for the line; a future reader
    changing the cadence should meet them."""
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    block = text[text.index("review_cadence") - 1200:text.index("review_cadence")]
    assert "Novy-Marx" in block
    assert "121.8" in block and "24.0" in block


# ---------------------------------------------------------------------------
# _cadence_block
# ---------------------------------------------------------------------------

def test_block_reads_the_configured_cadence():
    block = g._cadence_block({"portfolio": {"review_cadence": "monthly"}})
    assert block["review"] == "monthly"
    assert block["label"] == "monthly"
    assert block["configured"] is True


def test_block_falls_back_to_quarterly():
    """An unknown cadence must not render as *no* cadence - "no cadence" is
    exactly the daily-action reading this block exists to correct."""
    assert g._cadence_block({})["review"] == "quarterly"


@pytest.mark.parametrize("cfg", [
    {},
    {"portfolio": {}},
    {"portfolio": None},
    {"portfolio": {"review_cadence": None}},
    {"portfolio": {"review_cadence": ""}},
    {"portfolio": {"review_cadence": "whenever"}},
])
def test_block_never_raises_and_always_names_a_cadence(cfg):
    block = g._cadence_block(cfg)
    assert block["review"] in g.CADENCE_LABELS
    assert block["label"]


@pytest.mark.parametrize("cfg", [
    {},
    {"portfolio": {}},
    {"portfolio": {"review_cadence": "whenever"}},
])
def test_fallback_is_visible_in_the_payload(cfg):
    """`configured` distinguishes a real setting from the fallback. Reporting
    True for a fallback would make an old run's config snapshot silently
    indistinguishable from one that had been set."""
    assert g._cadence_block(cfg)["configured"] is False


def test_block_is_case_and_whitespace_tolerant():
    block = g._cadence_block({"portfolio": {"review_cadence": "  Quarterly  "}})
    assert block["review"] == "quarterly" and block["configured"] is True


def test_block_carries_the_turnover_evidence():
    block = g._cadence_block({})
    assert block["turnover_every_run"] == 121.8
    assert block["turnover_monthly"] == 24.0
    assert block["turnover_ceiling"] == 50.0


def test_turnover_numbers_bracket_the_nmv_ceiling():
    """The whole argument is that daily review sits above the boundary and
    monthly review below it. If a future edit broke that ordering the copy
    would assert something the numbers no longer support."""
    block = g._cadence_block({})
    assert block["turnover_monthly"] < block["turnover_ceiling"]
    assert block["turnover_every_run"] > block["turnover_ceiling"]


def test_block_reports_the_portfolio_size_it_describes():
    block = g._cadence_block({"portfolio": {"num_stocks": 25}})
    assert block["num_stocks"] == 25


def test_block_tolerates_a_missing_portfolio_size():
    assert g._cadence_block({"portfolio": {}})["num_stocks"] is None


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------

def _minimal_run_data(cfg: dict) -> dict:
    """The smallest run `prepare_dashboard_data` will build a payload from."""
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB"],
        "Company": ["Alpha Inc", "Beta Corp"],
        "Sector": ["Information Technology", "Financials"],
        "Composite": [80.0, 60.0],
        "Rank": [1, 2],
        "Value_Trap_Flag": [False, False],
        "Growth_Trap_Flag": [False, False],
    })
    for cat in g.CATEGORIES:
        df[f"{cat}_score"] = 50.0
    return {"df": df, "meta": {"run_date": "2026-09-17"}, "weights": {},
            "sens_df": None, "corr_df": None, "cfg": cfg}

def test_payload_includes_the_cadence_block():
    """Driven through the real payload builder rather than asserted on source.

    `prepare_dashboard_data` needs a scored frame, so this uses the smallest one
    that satisfies it and checks the block arrives under the `cadence` key with
    the run's own configured value.
    """
    run_data = _minimal_run_data({"portfolio": {"review_cadence": "monthly",
                                                "num_stocks": 25}})
    payload = json.loads(g.prepare_dashboard_data(run_data))
    assert payload["cadence"]["review"] == "monthly"
    assert payload["cadence"]["configured"] is True


def test_cadence_comes_from_the_runs_own_config_snapshot():
    """Not the working tree, whose `config.yaml` says quarterly. A republished
    old run must state what *it* was configured for, the same rule the rest of
    the provenance block follows."""
    run_data = _minimal_run_data({"portfolio": {"review_cadence": "monthly"}})
    payload = json.loads(g.prepare_dashboard_data(run_data))
    tree = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert tree["portfolio"]["review_cadence"] == "quarterly"
    assert payload["cadence"]["review"] == "monthly"


def test_cadence_survives_a_config_snapshot_with_no_portfolio_section():
    payload = json.loads(g.prepare_dashboard_data(_minimal_run_data({})))
    assert payload["cadence"]["review"] == "quarterly"
    assert payload["cadence"]["configured"] is False


# ---------------------------------------------------------------------------
# The emitted script
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def html() -> str:
    return g.generate_html()


@pytest.fixture(scope="module")
def script(html) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == 1
    return blocks[0]


@pytest.mark.parametrize("fn", ["cadenceText", "cadenceLine"])
def test_cadence_functions_emitted(script, fn):
    assert f"function {fn}(" in script


def test_holdings_renders_the_cadence(script):
    """The sell surface is where a daily redraw is most likely to be read as a
    daily decision."""
    body = script[script.index("function renderHoldings("):]
    body = body[:body.index("function holdingsFitLine(")]
    assert body.count("cadenceLine(false)") == 2, "empty and populated states"


def test_movers_panel_renders_the_cadence(script):
    body = script[script.index("function renderChanged("):]
    body = body[:body.index("changed-footnote')")]
    assert "cadenceText(false)" in body


def test_holdings_footnote_carries_the_long_form(script):
    body = script[script.index("function renderHoldingsFootnote("):]
    assert "cadenceText(true)" in body


def test_long_form_cites_novy_marx(script):
    assert "Novy-Marx" in script
    assert "Review of Financial" in script


def test_copy_states_both_cadences(script):
    """Saying "built for quarterly review" without saying the page rebuilds
    daily leaves the contradiction the reader is actually looking at."""
    assert "rebuilt every weekday" in script
    assert "c.label" in script, "the cadence must come from config, not a literal"


def test_cadence_copy_is_not_an_instruction(html):
    lowered = html.lower()
    for phrase in ("do not trade", "you should wait", "do not act",
                   "you must not", "wait until"):
        assert phrase not in lowered


def test_no_bare_recommendation_in_cadence_copy(html):
    lowered = html.lower()
    for phrase in ("sell now", "buy now", "time to sell", "recommended action"):
        assert phrase not in lowered


# ---------------------------------------------------------------------------
# Functional: drive the emitted script under Node
# ---------------------------------------------------------------------------

_DOM_STUB = r"""
const els = {};
function el(id) {
  if (!els[id]) {
    const classes = new Set();
    els[id] = {
      id: id, innerHTML: '', textContent: '', value: '',
      style: {}, _listeners: {},
      classList: {
        add: c => classes.add(c), remove: c => classes.delete(c),
        contains: c => classes.has(c),
        toggle: c => (classes.has(c) ? (classes.delete(c), false)
                                     : (classes.add(c), true)),
      },
      addEventListener: function (ev, fn) { (this._listeners[ev] ||= []).push(fn); },
      dispatchEvent: function () {},
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
globalThis.Event = class { constructor(t) { this.type = t; } };
"""

_EPILOGUE = r"""
globalThis.__api = { cadenceText, cadenceLine, renderHoldings, addHolding, D };
"""


def _payload(cadence) -> dict:
    detail = {
        "company": "Alpha Industries", "sector": "Industrials", "rank": 3,
        "composite": 71.0,
        "cat_scores": {c: 50.0 for c in g.CATEGORIES},
        "contrib": {c: 8.0 for c in g.CATEGORIES},
        "vt": False, "gt": False, "summary": [],
    }
    payload = {
        "kpis": {"universe_size": 1, "value_traps": 0, "growth_traps": 0,
                 "run_timestamp": None},
        "table_data": [],
        "_rows": [{"Ticker": "AAA", "Company": "Alpha Industries",
                   "Sector": "Industrials", "Composite": 71.0, "Rank": 3,
                   "Value_Trap_Flag": False, "Growth_Trap_Flag": False}],
        "stock_detail": {"AAA": detail},
        "history": {"available": False, "dates": [], "series": {},
                    "compare": {"prev": None, "m1": None}, "movers": {},
                    "delta": {}, "excluded": [], "noise": None},
        "weights": {"factor_weights": {c: 12.5 for c in g.CATEGORIES},
                    "metric_weights": {}},
        "metric_meta": {},
        "sectors": ["Industrials"],
    }
    if cadence is not None:
        payload["cadence"] = cadence
    return payload


def _run_js(script: str, body: str, tmp_path: Path, cadence) -> dict:
    src = "\n".join([_DOM_STUB, script, _EPILOGUE,
                     "const A = globalThis.__api;",
                     "A.D.table_data.push(...A.D._rows);", body])
    path = tmp_path / "cadence_harness.cjs"
    path.write_text(src, encoding="utf-8")
    proc = subprocess.run([NODE, str(path), json.dumps(_payload(cadence))],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[:4000]
    return json.loads(proc.stdout.strip().splitlines()[-1])


REAL = {"review": "quarterly", "label": "quarterly", "configured": True,
        "num_stocks": 25, "turnover_every_run": 121.8,
        "turnover_monthly": 24.0, "turnover_ceiling": 50.0}

node_only = pytest.mark.skipif(NODE is None, reason="node not available")


@node_only
def test_short_form_names_the_cadence(tmp_path, script):
    out = _run_js(script, "console.log(JSON.stringify({t: A.cadenceText(false)}));",
                  tmp_path, REAL)
    assert "quarterly" in out["t"]
    assert "every weekday" in out["t"]


@node_only
def test_short_form_declines_to_call_a_move_a_reason(tmp_path, script):
    out = _run_js(script, "console.log(JSON.stringify({t: A.cadenceText(false)}));",
                  tmp_path, REAL)
    assert "not by itself a reason to act" in out["t"]


@node_only
def test_long_form_quotes_all_three_numbers(tmp_path, script):
    out = _run_js(script, "console.log(JSON.stringify({t: A.cadenceText(true)}));",
                  tmp_path, REAL)
    for n in ("122%", "24%", "50%"):
        assert n in out["t"], (n, out["t"])
    assert "Novy-Marx" in out["t"]


@node_only
def test_long_form_names_the_portfolio_size_from_the_payload(tmp_path, script):
    out = _run_js(script, "console.log(JSON.stringify({t: A.cadenceText(true)}));",
                  tmp_path, dict(REAL, num_stocks=40))
    assert "top 40" in out["t"]


@node_only
def test_cadence_label_follows_the_payload(tmp_path, script):
    out = _run_js(script, "console.log(JSON.stringify({t: A.cadenceText(false)}));",
                  tmp_path, dict(REAL, label="monthly", review="monthly"))
    assert "monthly" in out["t"] and "quarterly" not in out["t"]


@node_only
def test_absent_cadence_block_renders_nothing_rather_than_breaking(tmp_path, script):
    """An older `dashboard_data.js` in a cached browser must not blank the
    panel. The line is an addition; its absence degrades to silence."""
    out = _run_js(script,
                  "console.log(JSON.stringify({t: A.cadenceText(false), l: A.cadenceLine(false)}));",
                  tmp_path, None)
    assert out["t"] == "" and out["l"] == ""


@node_only
def test_holdings_shows_the_cadence_before_any_rank(tmp_path, script):
    """A reader must meet the cadence before the numbers it qualifies."""
    out = _run_js(script, """
      A.addHolding('AAA');
      A.renderHoldings();
      console.log(JSON.stringify({fit: globalThis.__els['holdings-fit'].innerHTML}));
    """, tmp_path, REAL)
    assert "quarterly" in out["fit"]
    assert "cadence-note" in out["fit"]


@node_only
def test_empty_holdings_list_still_shows_the_cadence(tmp_path, script):
    out = _run_js(script, """
      A.renderHoldings();
      console.log(JSON.stringify({fit: globalThis.__els['holdings-fit'].innerHTML}));
    """, tmp_path, REAL)
    assert "quarterly" in out["fit"]


@node_only
def test_holdings_footnote_carries_the_evidence(tmp_path, script):
    out = _run_js(script, """
      A.renderHoldings();
      console.log(JSON.stringify({f: globalThis.__els['holdings-footnote'].innerHTML}));
    """, tmp_path, REAL)
    assert "Novy-Marx" in out["f"]
    assert "122%" in out["f"] and "24%" in out["f"]


@node_only
def test_cadence_copy_carries_no_advice_language(tmp_path, script):
    import stock_summary
    out = _run_js(script,
                  "console.log(JSON.stringify({s: A.cadenceText(false), l: A.cadenceText(true)}));",
                  tmp_path, REAL)
    for key in ("s", "l"):
        text = re.sub(r"<[^>]+>", " ", out[key])
        assert stock_summary.advice_terms_in(text) == [], (key, text)
