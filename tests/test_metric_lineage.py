"""The registry of where every metric comes from must be true.

WHY THIS EXISTS - 2026-10-07 (``plan/calculation-transparency.md``, stages T1-T3).

The drilldown now shows, for each metric, its formula, the reported figures behind it,
and who it was ranked against. That is only worth anything if it is *right*. These tests
hold the registry (``metric_lineage``) to the real payload:

* every metric that carries weight has an entry, and every input it names is published;
* every equation the registry says can be rebuilt **is** rebuilt - for every stock - from
  the inputs published beside it, and the stocks where it is not are listed, not hidden;
* the two scores that are sums of parts (Piotroski, Beneish) add up from their components;
* a percentile reproduces from the raw values of the peers it was ranked against.

The last three are checks against facts the 2026-10-06 audit found by reading code: an
audit that has not been turned into a test is a belief.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import metric_lineage as ml  # noqa: E402
from factor_engine import CAT_METRICS, METRIC_DIR, SECTOR_MIN_PEERS  # noqa: E402

PAYLOAD = ROOT / "dashboard_data.js"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip("dashboard_data.js not present")
    text = PAYLOAD.read_text(encoding="utf-8", errors="replace")
    d = json.loads(text[text.find("{"):text.rfind("}") + 1])
    if not any(s.get("inp") for s in d["stock_detail"].values()):
        pytest.skip("payload predates the published inputs")
    return d


# ---------------------------------------------------------------------------
# the registry itself
# ---------------------------------------------------------------------------

def _weighted_metrics(payload) -> set:
    out = set()
    for profiles in payload["weights"]["profiles"].values():
        for table in profiles.values():
            out |= {m for m, w in table.items() if w > 0}
    return out


def test_every_weighted_metric_has_a_lineage_entry(payload):
    missing = sorted(_weighted_metrics(payload) - set(ml.LINEAGE))
    assert not missing, f"scored metrics with no entry in metric_lineage.LINEAGE: {missing}"


def test_lineage_only_names_metrics_the_scorer_knows():
    known = {m for ms in CAT_METRICS.values() for m in ms}
    assert set(ml.LINEAGE) <= known, sorted(set(ml.LINEAGE) - known)


# Every format code an input may carry. `date` was added 2026-10-08 for the two closes
# `max_drawdown_1y` is measured between; it renders through `fmtInput`'s documented
# `String(v)` fallback, which the test below pins so the fallback cannot be removed under it.
VALID_FORMATS = {ml.USD, ml.PRICE, ml.PCT, ml.RATIO, ml.NUM, ml.DATE}


def test_every_entry_has_a_formula_and_valid_formats():
    for m, e in ml.LINEAGE.items():
        assert e["formula"].strip(), m
        assert e["kind"] in {"ratio", "components", "series"}, m
        for label, key, fmt in e["inputs"]:
            assert label and key
            assert fmt in VALID_FORMATS, (m, key, fmt)


# `fmtInput` formats these by name; `num` and `date` are printed verbatim by its
# `return String(v)` fallback, which is the right rendering for both.
EXPLICITLY_FORMATTED = {ml.USD, ml.PRICE, ml.PCT, ml.RATIO}


def test_the_page_can_render_every_format_code_the_registry_uses():
    """A format code the registry hands the browser that `fmtInput` handles neither by name nor
    by its fallback would print `undefined` beside a number a reader is checking."""
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    body = src[src.index("function fmtInput("):][:900]
    assert "return String(v)" in body, (
        "fmtInput lost its verbatim fallback; 'num' and 'date' inputs rely on it")
    used = {fmt for e in ml.LINEAGE.values() for _, _, fmt in e["inputs"]}
    for fmt in used:
        assert fmt in VALID_FORMATS, f"{fmt!r} is not a declared format code"
        if fmt in EXPLICITLY_FORMATTED:
            assert f"'{fmt}'" in body, f"fmtInput has no branch for {fmt!r}"


def test_every_recompute_function_has_a_lineage_entry():
    assert set(ml.RECOMPUTE) <= set(ml.LINEAGE)


# A series metric may claim an equation only when the equation's inputs are the specific
# points of the series the engine measured between, so the arithmetic on the row is the
# arithmetic the engine did. Default is deny: a new series metric fails this test until it
# is listed here with its reason, which is the whole point of the guard.
SERIES_METRICS_WITH_AN_EXACT_EQUATION = {
    # The drawdown is found by scanning ~13 months of closes, but the fall itself is one
    # division between two of them. `factor_engine` step 16d publishes the pair it chose
    # (`_mdd_peak` / `_mdd_trough`), so the row shows that division and nothing is
    # re-derived. Added 2026-10-08 with the price-path fix.
    "max_drawdown_1y",
    # Beta: the covariance and variance the slope is (`_beta_cov` / `_beta_var`), and Jensen's
    # alpha: its four CAPM terms (`_ja_*`), all published from the one computation. 2026-10-09.
    "beta", "jensens_alpha",
}


def test_a_series_metric_never_claims_an_equation():
    """Volatility, beta and friends are summaries of a whole daily price history; showing a
    two-number equation for them would be a false formula.

    The exceptions are listed above, each with the engine-published inputs that make its
    equation the real arithmetic rather than a re-derivation."""
    for m, e in ml.LINEAGE.items():
        if e["kind"] == "series" and m not in SERIES_METRICS_WITH_AN_EXACT_EQUATION:
            assert m not in ml.RECOMPUTE, m


def test_a_listed_series_exception_really_publishes_the_points_it_divides():
    """An entry in the allow-list above has to earn it: its equation may only name inputs the
    engine computed and published, never a figure the page would have to work out itself."""
    for m in SERIES_METRICS_WITH_AN_EXACT_EQUATION:
        assert m in ml.LINEAGE and ml.LINEAGE[m]["kind"] == "series", m
        assert m in ml.RECOMPUTE and m in ml.EQUATIONS, m
        exact, templates = ml.EQUATIONS[m]
        assert exact, f"{m}: listed as an exception but its equation is not exact"
        slots = {k for t in templates for k, _ in ml.template_slots(t)}
        assert slots, m
        assert slots <= set(ml.ENGINE_KEYS), (
            f"{m}: equation names {sorted(slots - set(ml.ENGINE_KEYS))}, which the engine does "
            f"not publish as its own computed figures")


def test_clamps_in_the_registry_match_config():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert tuple(cfg["metric_clamps"]["forward_eps_growth"]) == ml.FEG_CLAMP
    assert tuple(cfg["metric_clamps"]["price_target_upside"]) == ml.PTU_CLAMP


def test_published_lineage_carries_no_callables():
    json.dumps(ml.published_lineage())  # raises if a function leaked through


# ---------------------------------------------------------------------------
# every published input is real, and the payload says so
# ---------------------------------------------------------------------------

def test_inputs_the_registry_names_are_published_for_most_stocks(payload):
    """An input named in the registry but absent for nearly every stock is a broken fetch
    mapping, not a missing figure."""
    stocks = list(payload["stock_detail"].values())
    for key in ml.INPUT_KEYS:
        have = sum(1 for s in stocks if (s.get("inp") or {}).get(key) is not None)
        if key in {"bookValue", "priceToBook", "returnOnEquity", "returnOnAssets", "dividendsPaid",
                   "payoutRatio", "totalEquity_prior"}:
            continue  # legitimately sparse: bank-only or payer-only fields
        assert have / len(stocks) > 0.5, f"{key} is published for only {have} of {len(stocks)} stocks"


def test_inputs_are_display_only_and_never_enter_scored_fields(payload):
    s = next(iter(payload["stock_detail"].values()))
    for k in ("inp", "pio", "bn", "inp_bad"):
        assert k not in s["raw"] and k not in s["pct"]


# ---------------------------------------------------------------------------
# the equations reproduce - the point of the exercise
# ---------------------------------------------------------------------------

def _tol(calc):
    return 1e-4 * max(1.0, abs(calc)) + 1e-4


# Metrics whose equation is held to a near-perfect reproduction rate. Anything below the bar is
# either fixed or removed from RECOMPUTE - never displayed with a formula that does not hold.
EXACT = {
    "earnings_yield": 0.995, "size_log_mcap": 0.995, "price_target_upside": 0.99,
    "short_interest_ratio": 0.99, "forward_eps_growth": 0.99, "peg_ratio": 0.99,
    "revenue_growth": 0.99, "accruals": 0.99, "gross_profit_assets": 0.99,
    "asset_growth": 0.99, "equity_ratio": 0.99, "return_12_1": 0.99, "return_6m": 0.99,
    "ev_ebitda": 0.99, "fcf_yield": 0.99, "ev_sales": 0.99,
}


@pytest.mark.parametrize("metric", sorted(ml.RECOMPUTE))
def test_the_published_inputs_rebuild_the_published_value(payload, metric):
    ok = total = 0
    worst = []
    for t, s in payload["stock_detail"].items():
        pub = s["raw"].get(metric)
        if pub is None:
            continue
        total += 1
        calc = ml.RECOMPUTE[metric](s["inp"])
        if calc is not None and abs(calc - pub) <= _tol(calc):
            ok += 1
        else:
            worst.append((t, pub, calc))
    assert total > 0, f"{metric}: no stock has a published value to test"
    rate = ok / total
    bar = EXACT.get(metric, 0.90)
    assert rate >= bar, (f"{metric}: only {ok}/{total} ({rate:.1%}) rebuild from their inputs; "
                         f"bar {bar:.0%}. First failures: {worst[:4]}")


def test_stocks_that_do_not_rebuild_are_flagged_not_hidden(payload):
    """The build lists, per stock, the metrics whose inputs do not reproduce the value, and the
    page says so for exactly those. No stock may fail silently."""
    flagged = 0
    for t, s in payload["stock_detail"].items():
        bad = set(s.get("inp_bad") or ())
        for m, fn in ml.RECOMPUTE.items():
            pub = s["raw"].get(m)
            if pub is None:
                continue
            calc = fn(s["inp"])
            fails = calc is None or abs(calc - pub) > _tol(calc)
            assert fails == (m in bad), f"{t}/{m}: build says {m in bad}, recomputation says {fails}"
            flagged += fails
    assert flagged >= 0


def test_lineage_check_summary_matches_the_stocks(payload):
    chk = payload["lineage_check"]
    for m in ml.RECOMPUTE:
        n_have = sum(1 for s in payload["stock_detail"].values() if s["raw"].get(m) is not None)
        assert chk[m][1] == n_have, m
        assert 0 <= chk[m][0] <= chk[m][1]


# ---------------------------------------------------------------------------
# scores that are sums of parts
# ---------------------------------------------------------------------------

def test_piotroski_signals_add_up_to_the_score(payload):
    n = 0
    for t, s in payload["stock_detail"].items():
        score = s["raw"].get("piotroski_f_score")
        sig = s.get("pio")
        if score is None:
            continue
        assert sig and len(sig) == 9, t
        assert sig.count("1") == round(score), f"{t}: signals {sig} vs score {score}"
        assert 9 - sig.count("-") >= 6, f"{t}: scored with fewer than 6 testable signals"
        n += 1
    assert n > 300


def test_beneish_indices_rebuild_the_m_score(payload):
    coef = (0.920, 0.528, 0.404, 0.892, 0.115, -0.172, -0.327, 4.679)
    n = 0
    for t, s in payload["stock_detail"].items():
        m = s["raw"].get("beneish_m_score")
        bn = s.get("bn")
        if m is None:
            continue
        vals, mask = bn.split("|")
        vals = [float(x) for x in vals.split(",")]
        assert len(vals) == 8 and len(mask) == 8, t
        assert sum(c == "1" for c in mask) >= 5, f"{t}: scored with fewer than 5 real indices"
        calc = -4.84 + sum(c * v for c, v in zip(coef, vals))
        assert abs(calc - m) < 2e-3, f"{t}: published {m}, indices give {calc:.4f}"
        n += 1
    assert n > 300


# ---------------------------------------------------------------------------
# percentiles reproduce from the peers they were ranked against
# ---------------------------------------------------------------------------

def _avg_rank_pct(values, mine):
    """pandas rank(pct=True, method='average') for one value among a list."""
    below = sum(1 for v in values if v < mine)
    equal = sum(1 for v in values if v == mine)
    return ((below + (equal + 1) / 2.0) / len(values)) * 100.0


def test_percentiles_reproduce_from_the_published_peer_values(payload):
    """Sector rank (or universe rank where a sector has fewer than SECTOR_MIN_PEERS values),
    direction-flipped for lower-is-better metrics. The drilldown says exactly this; the
    check recomputes it from the raw values every stock publishes."""
    stocks = payload["stock_detail"]
    metrics = [m for m in payload["lineage"] if m in next(iter(stocks.values()))["pct"]]
    checked = bad = 0
    for m in metrics:
        universe = [(t, s["raw"][m], s["sector"]) for t, s in stocks.items() if s["raw"].get(m) is not None]
        uvals = [v for _, v, _ in universe]
        by_sector = {}
        for t, v, sec in universe:
            by_sector.setdefault(sec, []).append(v)
        flip = not METRIC_DIR.get(m, True)
        for t, v, sec in universe:
            peers = by_sector[sec] if len(by_sector[sec]) >= SECTOR_MIN_PEERS else uvals
            pct = _avg_rank_pct(peers, v)
            if flip:
                pct = 100.0 - pct
            pub = stocks[t]["pct"].get(m)
            checked += 1
            # values are published to 4dp, which can merge near-ties; allow a whisker
            if pub is None or abs(pct - pub) > 0.6:
                bad += 1
    assert checked > 10000
    assert bad / checked < 0.005, f"{bad} of {checked} percentiles do not reproduce from their peers"


def test_sector_stats_agree_with_the_raw_values(payload):
    import statistics
    stocks = payload["stock_detail"]
    stats = payload["sector_stats"]
    assert payload["sector_min_peers"] == SECTOR_MIN_PEERS
    for sector, per_metric in stats.items():
        for m, (n, q1, med, q3) in list(per_metric.items())[:8]:
            vals = [s["raw"][m] for s in stocks.values() if s["sector"] == sector and s["raw"].get(m) is not None]
            assert n == len(vals), (sector, m)
            assert math.isclose(med, statistics.median(vals), abs_tol=2e-4), (sector, m)
            assert q1 <= med <= q3


# ---------------------------------------------------------------------------
# the equation printed on every row (owner, 2026-10-07: "we don't see any numbers ...
# the numbers going into the scoring") - and the guarantee that it tracks the engine
# ---------------------------------------------------------------------------
# Each metric row in the workings prints a line like "$4.5B free cash flow ÷ $31.0B
# enterprise value = 14.4%", filled from metric_lineage.EQUATIONS with the stock's own
# published inputs. An *exact* template is real arithmetic: these tests evaluate it for
# every stock and compare it with the value the engine scored. **If a scoring formula in
# factor_engine changes and its template here is not updated in the same commit, this
# fails** - which is the point: the page must not describe a formula the engine no longer
# uses.

def test_every_weighted_metric_has_a_line_on_the_row(payload):
    missing = sorted(m for m in _weighted_metrics(payload)
                     if m not in ml.EQUATIONS and m not in ml.SOURCES)
    assert not missing, ("weighted metrics with no equation or source line in "
                         f"metric_lineage.EQUATIONS / SOURCES: {missing}")


def test_every_rebuildable_metric_has_an_equation_template():
    assert set(ml.RECOMPUTE) <= set(ml.EQUATIONS), sorted(set(ml.RECOMPUTE) - set(ml.EQUATIONS))


def test_template_slots_name_published_inputs():
    published = set(ml.INPUT_KEYS) | set(ml.ENGINE_KEYS)
    for m, (_, templates) in ml.EQUATIONS.items():
        assert templates, m
        for t in templates:
            slots = ml.template_slots(t)
            assert slots or not ml.EQUATIONS[m][0], (m, t)
            for key, label in slots:
                assert key in published, f"{m}: template names {key!r}, which the page never receives"
                assert label.strip(), (m, t)


def test_exact_templates_are_pure_arithmetic():
    """An exact template may contain only slots, numbers and operators - words would make
    it unevaluable, and so unchecked."""
    for m, (exact, templates) in ml.EQUATIONS.items():
        if not exact:
            continue
        for t in templates:
            ones = {k: 2.0 for k, _ in ml.template_slots(t)}
            ml.evaluate_template(t, ones)  # raises on leftover words


@pytest.mark.parametrize("metric", sorted(m for m, e in ml.EQUATIONS.items() if e[0]))
def test_the_equation_on_the_row_gives_the_value_that_was_scored(payload, metric):
    ok = total = 0
    worst = []
    for t, s in payload["stock_detail"].items():
        pub = s["raw"].get(metric)
        if pub is None:
            continue
        tpl = ml.choose_template(metric, s.get("inp") or {})
        if tpl is None:
            continue
        total += 1
        v = ml.evaluate_template(tpl, s["inp"])
        if v is not None and abs(v - pub) <= _tol(v):
            ok += 1
        else:
            worst.append((t, pub, v, tpl))
    assert total > 0, f"{metric}: no stock shows this equation"
    assert ok / total >= 0.99, (f"{metric}: the equation printed on the row gives the scored value "
                               f"for only {ok}/{total} stocks - did the scoring formula change "
                               f"without metric_lineage.EQUATIONS? First: {worst[:3]}")


def test_nearly_every_scored_value_gets_a_line(payload):
    shown = total = 0
    for s in payload["stock_detail"].values():
        for m in ml.EQUATIONS:
            if s["raw"].get(m) is None:
                continue
            total += 1
            shown += ml.choose_template(m, s.get("inp") or {}) is not None
    assert shown / total > 0.99, f"only {shown}/{total} scored values have an equation line"


def test_the_payload_carries_the_templates(payload):
    lin = payload["lineage"]
    for m, (exact, templates) in ml.EQUATIONS.items():
        if m in lin:
            assert lin[m].get("x") == templates and lin[m].get("xe") == (1 if exact else 0), m
    for m, text in ml.SOURCES.items():
        if m in lin:
            assert lin[m].get("src") == text, m


def test_the_nightly_prompt_still_says_scoring_changes_carry_their_frontend():
    """The owner asked that sessions be told; a deleted instruction is how a rule decays."""
    prompt = (ROOT / "prompts" / "nightly.md").read_text(encoding="utf-8")
    assert "EQUATIONS" in prompt and "same commit" in prompt.lower()


# ---------------------------------------------------------------------------
# why a listed metric is not in the score (owner, 2026-10-07)
# ---------------------------------------------------------------------------

def _config():
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_a_specific_not_used_reason_is_only_given_for_a_metric_at_zero_weight():
    """'Removed from the score' must stay true: if a later session puts weight back on PEG or
    Sharpe, this fails until the reason is removed."""
    cfg = _config()
    for m in ml.NOT_USED_BECAUSE:
        for table in ("metric_weights", "bank_metric_weights"):
            for cat, weights in (cfg.get(table) or {}).items():
                if m in (weights or {}):
                    assert weights[m] == 0, f"{m} has weight {weights[m]} in {table}.{cat} but the page says it is not used"


def test_bank_only_metrics_really_are_bank_only():
    """The page calls a metric bank-only when the bank table weights it and the generic one
    does not - check config agrees for the metrics it says it about."""
    cfg = _config()
    gen, bank = cfg["metric_weights"], cfg["bank_metric_weights"]
    for cat, weights in bank.items():
        for m, w in (weights or {}).items():
            if w and w > 0 and (gen.get(cat) or {}).get(m, 0) == 0:
                assert m in {"pb_ratio", "roe", "roa", "equity_ratio"}, f"new bank-only metric {m}: check the BANK_ONLY wording still fits"


def test_not_used_reasons_carry_no_advice_language():
    import stock_summary
    texts = [ml.BANK_ONLY, ml.NOT_FOR_BANKS, ml.CANDIDATE, *ml.NOT_USED_BECAUSE.values()]
    for t in texts:
        assert stock_summary.advice_terms_in(t) == [], t


def test_the_payload_carries_the_reasons(payload):
    assert payload.get("not_used") == ml.published_not_used()
