"""A bank's ROE is its earnings yield times its P/B, so its Valuation and Quality scores are
one reading - and the weights net to no position on price-to-book by itself.

`research/2026-10-20-bank-quality-valuation-overlap.md` established that 35 points of bank
Quality on ROE are not a double count of two Valuation inputs: they duplicate earnings yield and
*cancel* price-to-book, leaving the construction bank analysts, insurer valuation research and
MSCI's multi-factor indexes build on purpose. The proposal it left open - state that on the
methodology page and guard it - was reviewed and accepted on 2026-10-21.

The zero net price-to-book position is the arithmetic of four separately chosen weights
(`-pb_ratio + roe + roa`, in composite points). Nothing recorded it as a design until that day,
so the next reweight of any of them would move the bank block onto a price-to-book bet nobody
chose, and the page would go on saying otherwise. These tests are what stops that.

**What the proposal asked for, and what the review changed.** It specified a single guard: the
first-order net loading computed from `config.yaml`, failing outside +-1.5 composite points. The
review measured that guard against the note's own rescored candidates
(`research/measurements/2026-10-21-bank-pb-roe-guard-review.py`) and found it blind in the
direction that matters and noisy in one that does not:

* **C3** - ROE's 35 points moved to ROA, which the note itself names as the mildest route to a
  deliberate price-to-book tilt - reads **0.00** first-order, exactly like the status quo, because
  ROE and ROA enter log P/B with the same +1 coefficient, so moving weight between them cannot
  move the first-order figure *at all*. Measured on the scores the engine produces, its loading on
  P/B's rank is **-0.156** against the status quo's **-0.069**: **2.3x**, and past the note's own
  criterion (b) bar of half the earnings-yield loading (0.141). The proposed guard passes it.
* **P1** - ROE 35 -> 25 with the equity ratio 15 -> 25, a reweight nobody proposed - reads **-2.2**
  points and would *fail* the +-1.5 band, while its measured loading (-0.119) leaves the page's
  sentence true.

So the first-order check stays (it is the arithmetic the page prints, and it must not go stale),
but the failing condition is the **measured** one at criterion (b)'s own pre-registered bar - the
sentence's own standard, and the one C3 trips. A claim whose check tests a proxy instead of the
assertion is an unchecked claim (CLAUDE.md, 0.11's lesson).

Everything here reads the committed payload and `config.yaml`, or the generator's own output -
never a hand-editable artifact for the page's text (rule 10). No return, information-coefficient
or backtest figure is read (CLAUDE.md rules 4 and 5).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "research" / "2026-10-20-bank-quality-valuation-overlap.md"
REVIEW = ROOT / "research" / "measurements" / "2026-10-21-bank-pb-roe-guard-review.py"

# The identity's own bar, and criterion (b)'s, both pre-registered in section 8 of the note
# before anything was built. Never widen either to pass a build.
MIN_IDENTITY_R2 = 0.90
# "no net weight on P/B by itself": the first-order net must stay inside this share of the
# net on log E/P. The page states whichever of the two readings is true of the weights.
MAX_FIRST_ORDER_SHARE = 0.25


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> dict:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


@pytest.fixture(scope="module")
def overview(cfg) -> str:
    import run_screener
    return run_screener.build_screener_overview(cfg)


@pytest.fixture(scope="module")
def section(overview) -> str:
    head = "### A bank's Valuation and Quality scores are one reading, not two"
    assert head in overview, "the bank P/B-ROE section is not on the generated page"
    rest = overview[overview.index(head) + len(head):]
    end = rest.find("\n---")
    return rest if end < 0 else rest[:end]


def first_order_points(cfg: dict) -> dict:
    """Composite points on each of the three independent log coordinates, from the weights alone.

    Renormalised within each category over the metrics that carry weight, which is what
    `compute_category_scores` does when a weight is set to zero.
    """
    fw = cfg["factor_weights"]
    bmw = cfg["bank_metric_weights"]
    pts = {}
    for cat in ("valuation", "quality"):
        total = sum(w for w in bmw[cat].values() if w)
        for metric, w in bmw[cat].items():
            pts[metric] = fw[cat] * (w / total) if (w and total) else 0.0
    return {
        "E/P": pts["earnings_yield"] + pts["roe"] + pts["roa"],
        "P/B": -pts["pb_ratio"] + pts["roe"] + pts["roa"],
        "B/A": pts["roa"] + pts["equity_ratio"],
        "_pts": pts,
    }


def bank_like(payload: dict) -> list[str]:
    """The bank-like set, from the weight profile the engine published for each stock.

    `metric_weight_profiles` is the one place weights are resolved (CLAUDE.md row 0.8), and the
    payload carries its verdict, so this is the engine's own answer rather than a second
    re-derivation of the sub-industry rule.
    """
    return sorted(t for t, d in payload["stock_detail"].items()
                  if (d.get("wp") or {}).get("valuation") == "bank")


def coords(payload: dict, tickers: list[str]) -> dict:
    """log E/P, log P/B, log B/A and log ROE for the given stocks, from the published raws."""
    out = {k: [] for k in ("ep", "pb", "ba", "roe")}
    kept = []
    for t in tickers:
        raw = payload["stock_detail"][t].get("raw") or {}
        v = [raw.get("earnings_yield"), raw.get("pb_ratio"),
             raw.get("equity_ratio"), raw.get("roe")]
        if any(x is None or not isinstance(x, (int, float)) or x <= 0 for x in v):
            continue
        for key, x in zip(("ep", "pb", "ba", "roe"), v):
            out[key].append(np.log(float(x)))
        kept.append(t)
    return {"tickers": kept, **{k: np.asarray(v, float) for k, v in out.items()}}


def _ols(y: np.ndarray, *xs: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.column_stack([np.ones(len(y)), *xs])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return beta[1:], float(1 - resid.var() / y.var())


# --------------------------------------------------------------------------------------------
# 1. the identity the page states
# --------------------------------------------------------------------------------------------

def test_a_banks_roe_is_its_earnings_yield_times_its_price_to_book(payload):
    """Criterion (a). If this falls below 0.90 the paragraph must not ship: a change to how ROE,
    earnings yield or P/B is built (an earnings basis, a tangible-book P/B) is the likely cause."""
    c = coords(payload, bank_like(payload))
    assert len(c["tickers"]) >= 30, f"only {len(c['tickers'])} bank-like stocks carry all four"
    (b_ep, b_pb), r2 = _ols(c["roe"], c["ep"], c["pb"])
    assert r2 >= MIN_IDENTITY_R2, (
        f"log ROE on log E/P and log P/B has R-squared {r2:.3f}, below {MIN_IDENTITY_R2}. "
        "The methodology section says a bank's ROE is fixed once its earnings yield and its "
        "price-to-book are. Rewrite it, or find what changed about the three metrics."
    )
    # both coefficients are 1 by construction; a drift says the inputs stopped being the
    # same two statements (preferred stock and average-vs-ending balances are the only slack)
    assert 0.8 <= b_ep <= 1.2, b_ep
    assert 0.8 <= b_pb <= 1.2, b_pb


# --------------------------------------------------------------------------------------------
# 2. no net weight on price-to-book - the measured check, which is the failing one
# --------------------------------------------------------------------------------------------

def test_the_block_carries_no_net_position_on_price_to_book(payload):
    """Criterion (b), measured on the scores the engine actually produced.

    This is the test that catches C3 - a reweight between ROE and ROA, which the first-order
    arithmetic below cannot see at all.
    """
    tickers = bank_like(payload)
    c = coords(payload, tickers)
    table = {r["Ticker"]: r for r in payload["table_data"]}
    ranks = {k: (np.argsort(np.argsort(c[k])) + 0.5) / len(c["tickers"]) * 100
             for k in ("ep", "pb", "ba")}
    block = np.asarray([(float(table[t]["valuation_score"]) + float(table[t]["quality_score"])) / 2
                        for t in c["tickers"]], float)
    (l_ep, l_pb, l_ba), _ = _ols(block, ranks["ep"], ranks["pb"], ranks["ba"])
    assert l_ep > 0, f"the block should reward earnings yield; loading {l_ep:.3f}"
    assert abs(l_pb) < abs(l_ep) / 2, (
        f"the Valuation + Quality block loads {l_pb:.3f} on price-to-book's rank against "
        f"{l_ep:.3f} on earnings yield's. Past half of the earnings-yield loading the page may "
        "no longer say 'no net weight on price-to-book by itself' - it must give the measured "
        "figure instead. See the note's criterion (b)."
    )


def test_the_first_order_arithmetic_the_page_prints_comes_from_the_config(cfg, section):
    """The page's own table of net points, held to `config.yaml`. It is generated, so this
    fails only if someone hardcodes a figure or changes the three sums."""
    fo = first_order_points(cfg)
    for label, key in (("Earnings yield (E/P)", "E/P"),
                       ("Price-to-book (P/B)", "P/B"),
                       ("Equity per dollar of assets (B/A)", "B/A")):
        row = next((ln for ln in section.splitlines() if ln.startswith(f"| {label} |")), None)
        assert row is not None, f"the page has no net-points row for {label}"
        printed = float(re.search(r"\*\*([+-]?[\d.]+)\*\*", row).group(1))
        assert printed == pytest.approx(fo[key], abs=0.05), (label, printed, fo[key])


def test_the_weights_still_net_to_neutral_on_price_to_book(cfg):
    """The first-order check. Kept because it is the arithmetic the page explains, not because
    it is sufficient - see this module's docstring for what it cannot see."""
    fo = first_order_points(cfg)
    limit = MAX_FIRST_ORDER_SHARE * fo["E/P"]
    assert abs(fo["P/B"]) <= limit, (
        f"the bank weights net to {fo['P/B']:+.1f} composite points on log price-to-book "
        f"against {fo['E/P']:+.1f} on log earnings yield. Beyond {limit:.1f} the block carries a "
        "bet on cheap (or dear) book. If that is deliberate, say so in "
        "METHODOLOGY_CHANGELOG.md and rewrite the methodology section - the config.yaml "
        "CONSTRAINT comment above bank_metric_weights names what this holds."
    )


def test_a_reweight_between_roe_and_roa_is_not_invisible_to_this_module(cfg, payload):
    """The review's own finding, held in place.

    C3 - ROE's 35 points to ROA - reads 0.00 first-order, identically to the status quo. If the
    only guard were the first-order one, the page could claim neutrality through a reweight that
    more than doubles the measured price-to-book loading. This asserts the blind spot exists (so
    nobody deletes the measured test as redundant) and that the measured test is what sees it.
    """
    import copy
    c3 = copy.deepcopy(cfg)
    c3["bank_metric_weights"]["quality"]["roe"] = 0
    c3["bank_metric_weights"]["quality"]["roa"] = 60
    assert first_order_points(c3)["P/B"] == pytest.approx(first_order_points(cfg)["P/B"], abs=1e-9), (
        "C3 no longer reads the same first-order figure as the status quo. If the coordinate "
        "algebra changed, re-read the note's section 4 before trusting either guard."
    )
    # and the measured quantity does move: ROA carries log B/A, which is correlated with log P/B
    c = coords(payload, bank_like(payload))
    assert abs(np.corrcoef(c["pb"], c["ba"])[0, 1]) > 0.1, (
        "log P/B and log B/A have decoupled across the bank set, which is the mechanism that "
        "makes a ROE/ROA reweight move the measured price-to-book loading. Re-measure with "
        "research/measurements/2026-10-21-bank-pb-roe-guard-review.py."
    )


# --------------------------------------------------------------------------------------------
# 3. the page says it, and says it honestly
# --------------------------------------------------------------------------------------------

def test_the_page_states_the_identity_and_that_the_two_scores_are_read_together(section):
    assert "earnings yield multiplied by its price-to-book" in section
    assert "one reading of the same four" in section
    assert "no net weight on price-to-book by itself" in section
    assert "its ROE relative to its price-to-book" in section
    # the three coordinates are named, not just asserted
    for coord in ("earnings yield (E/P)", "price-to-book (P/B)", "equity per dollar of assets (B/A)"):
        assert coord.lower() in section.lower(), coord


def test_the_page_names_the_practice_it_follows_and_claims_no_return(section):
    for source in ("Damodaran", "Wilcox", "Nissim", "Mercer Capital", "MSCI"):
        assert source in section, source
    assert "research/2026-10-20-bank-quality-valuation-overlap.md" in section
    # the honest limit, and rules 4 and 5: this says what the weights are, not what they earn
    assert "No return, information-coefficient or backtest" in section
    # no return claim of any kind, and in particular no all-stock premium
    # (tests/test_large_company_evidence.py's rule): that rules out Damodaran's 1981-90
    # portfolio spread and Penman & Reggiani's return table, both of which are in the note.
    for forbidden in ("outperform", "excess return", "% a year", "points a year",
                      "premium", "earns more", "beat the market"):
        assert forbidden not in section.lower(), forbidden


def test_the_two_metric_cells_no_longer_call_either_one_the_key_metric(overview):
    """The proposal's item (2). Each is read against the other, so neither is 'THE key'."""
    assert "THE key bank valuation metric" not in overview
    assert "The key bank profitability metric" not in overview
    assert "read against ROE in Quality, never on its own" in overview
    assert "read against P/B in Valuation, never on its own" in overview


def test_the_config_comment_names_the_constraint_and_the_note():
    raw = (ROOT / "config.yaml").read_text(encoding="utf-8")
    head = raw[: raw.index("bank_metric_weights:")]
    block = head[head.rindex("# CONSTRAINT"):]
    assert "research/2026-10-20-bank-quality-valuation-overlap.md" in block
    assert "tests/test_bank_pb_roe_identity.py" in block
    for coord in ("log E/P", "log P/B", "log B/A"):
        assert coord in block, coord
    # and it is above the four weights it governs, not somewhere else in the file
    assert raw.index("# CONSTRAINT") < raw.index("bank_metric_weights:")


# --------------------------------------------------------------------------------------------
# 4. the note and the review it was decided on
# --------------------------------------------------------------------------------------------

def test_the_note_records_the_decision_and_is_no_longer_open():
    note = NOTE.read_text(encoding="utf-8")
    assert "**Proposal status: OPEN**" not in note, (
        "the proposal was decided on 2026-10-21; the status line must say so, or the morning "
        "brief will keep listing it as waiting for a decision"
    )
    assert "ACCEPTED" in note
    assert "No change to any weight, metric or threshold" in note
    # the sources the page leans on are in the note, with what was actually read
    for name in ("Damodaran", "Wilcox (1984)", "Nissim (2013)", "Mercer Capital",
                 "Penman & Reggiani", "Novy-Marx (2013)"):
        assert name in note, name
    # the reading that cuts the other way is kept, not dropped
    assert "The one reading that cuts the other way" in note


def test_the_review_measurement_is_committed_and_reproducible():
    """The deciding session's own measurement, on its own run. Whoever reopens this reruns it."""
    src = REVIEW.read_text(encoding="utf-8")
    assert "latest_run_dir" in src, "the review must read a real run, not a stored figure"
    assert "compute_sector_percentiles" in src, "scoring must be the engine's own path"
    assert "baseline reproduces only" in src, "it must stop unless it reproduces the ranking"
    out = REVIEW.with_suffix(".json")
    assert out.is_file(), "the review's JSON output is not committed"
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["1_refutation_criteria"]["(a) verdict"] == "holds"
    assert d["1_refutation_criteria"]["(b) verdict"] == "holds"
    # the blind spot the review found, as a number a later session can check
    assert d["2_can_the_guard_see_it"]["C3_roe_to_roa"][
        "proposal's guard (|first-order| <= 1.5)"] == "passes"
    assert d["2_can_the_guard_see_it"]["C3_roe_to_roa"][
        "criterion (b) on the measured loading"] == "REFUTED"
