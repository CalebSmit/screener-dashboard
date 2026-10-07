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


def test_every_entry_has_a_formula_and_valid_formats():
    for m, e in ml.LINEAGE.items():
        assert e["formula"].strip(), m
        assert e["kind"] in {"ratio", "components", "series"}, m
        for label, key, fmt in e["inputs"]:
            assert label and key
            assert fmt in {ml.USD, ml.PRICE, ml.PCT, ml.RATIO, ml.NUM}, (m, key, fmt)


def test_every_recompute_function_has_a_lineage_entry():
    assert set(ml.RECOMPUTE) <= set(ml.LINEAGE)


def test_a_series_metric_never_claims_an_equation():
    """Volatility, beta, drawdown and friends are built from a daily price history; showing a
    two-number equation for them would be a false formula."""
    for m, e in ml.LINEAGE.items():
        if e["kind"] == "series":
            assert m not in ml.RECOMPUTE, m


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
