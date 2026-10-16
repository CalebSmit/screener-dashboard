"""The Piotroski-conditional rule: what the page says it does is what the engine does.

Two defects, found 2026-10-16 and fixed together (METHODOLOGY_CHANGELOG.md 2026-10-16):

1. **A weight profile that could never apply was published.** ``config.yaml`` set
   ``piotroski_conditional.growth_trap_enabled: true`` and the page carried a ``pio_gt`` weight
   table and label ("high growth with low quality"), but ``compute_category_scores`` scores
   Quality before Growth, so the rule's test for a ``growth_score`` column was False on the
   single pass ``run_screener`` makes. Measured on the 2026-10-16 payload: **0 of 501** stocks on
   ``pio_gt`` (235 ``pio_lowval``, 220 generic, 46 bank). It was removed rather than made to work -
   a working version is a methodology change and needs a research note first.
2. **The methodology page described the split wrongly.** It said the freed weight was
   "redistributed equally to ROIC and Gross Profit / Assets"; the engine splits it in proportion
   to their weights (ROIC 29 -> 33.55, GP/A 22 -> 25.45, not 33 and 26). The sentence is now built
   from ``factor_engine.metric_weight_profiles`` (claim ``overview.piotroski_conditional``).

The general rule these pin: **every weight profile the page publishes is applied to at least one
stock**, or it is named in ``DORMANT_WEIGHT_PROFILES`` with a reason.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

import claims
import factor_engine as fe
import run_screener

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "dashboard_data.js"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated_overview(cfg) -> str:
    """The methodology page the next data run will publish (rule 10: read the generator)."""
    return run_screener.build_screener_overview(cfg)


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip("dashboard_data.js not present")
    text = PAYLOAD.read_text(encoding="utf-8", errors="replace")
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def _assigned(payload: dict, cat: str) -> dict:
    """{profile_id: number of stocks scored with it in `cat`}. A stock with no `wp` entry for the
    category was scored on the generic table - the generator omits generic to save bytes."""
    counts: dict = {}
    for s in payload["stock_detail"].values():
        if s.get("cat_scores", {}).get(cat) is None:
            continue
        pid = (s.get("wp") or {}).get(cat, "generic")
        counts[pid] = counts.get(pid, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 1. the sentence states the split the engine applies
# ---------------------------------------------------------------------------

def test_overview_states_the_split_the_engine_applies(cfg, generated_overview):
    """Recompute the split from the profiles, independently of the sentence builder, and find
    every figure in the published paragraph."""
    prof = fe.metric_weight_profiles(cfg, "quality")
    gen, cond = prof["generic"], prof["pio_lowval"]
    changed = {m for m in gen if abs(gen[m] - cond[m]) > 1e-12}
    recipients = sorted(changed - {"piotroski_f_score"}, key=lambda m: -gen[m])
    assert set(recipients) == set(cfg["piotroski_conditional"]["redistribute_to"])

    freed = gen["piotroski_f_score"] - cond["piotroski_f_score"]
    base = sum(gen[m] for m in recipients)
    para = generated_overview[generated_overview.index("## Piotroski conditional weighting"):]
    para = para[:para.index("---")]
    for m in recipients:
        expected = gen[m] + freed * gen[m] / base           # proportional, not equal
        assert cond[m] == pytest.approx(expected)
        assert f"from {run_screener._pts(gen[m] * 100)} to {run_screener._pts(cond[m] * 100)}" in para, (
            f"{m}: the page must print the weight the engine moves it to ({cond[m] * 100:.2f})")
    assert f"from {run_screener._pts(gen['piotroski_f_score'] * 100)} to " \
           f"{run_screener._pts(cond['piotroski_f_score'] * 100)}" in para
    assert f"freed {run_screener._pts(freed * 100)} points" in para
    assert "in proportion to their own weights" in para


def test_the_page_no_longer_says_equally(generated_overview):
    assert not re.search(r"redistributed equally to", generated_overview)
    assert claims.forbidden_hits("The freed weight is redistributed equally to ROIC and Gross "
                                 "Profit / Assets"), "FORBIDDEN must catch the sentence that shipped"


def test_the_split_sentence_follows_the_config_not_a_template(cfg):
    """Change the weights and the sentence changes with them - it is built, not written."""
    import copy
    alt = copy.deepcopy(cfg)
    alt["metric_weights"]["quality"]["roic"] = 30
    alt["metric_weights"]["quality"]["gross_profit_assets"] = 21
    s = run_screener.piotroski_split_sentence(alt)
    assert "(30 : 21)" in s
    off = copy.deepcopy(cfg)
    off["piotroski_conditional"]["enabled"] = False
    assert run_screener.piotroski_split_sentence(off) == ""


# ---------------------------------------------------------------------------
# 2. no published weight profile describes a rule that never applies
# ---------------------------------------------------------------------------

def test_the_growth_trap_variant_is_gone(cfg):
    pio = cfg["piotroski_conditional"]
    assert not any(k.startswith("growth_trap") for k in pio), (
        "the growth-trap Piotroski variant was removed 2026-10-16; restoring it needs a research note")
    for cat in fe.CAT_METRICS:
        assert "pio_gt" not in fe.metric_weight_profiles(cfg, cat)
    assert "pio_gt" not in fe.WEIGHT_PROFILE_LABELS
    assert "pio_gt" in fe.RETIRED_WEIGHT_PROFILES


def test_every_profile_the_engine_defines_is_applied_to_a_stock(cfg, payload):
    """Read against the committed payload: each table today's config resolves must be the one
    some stock was scored with, unless it is named dormant with a reason."""
    for cat in fe.CAT_METRICS:
        counts = _assigned(payload, cat)
        for pid in fe.metric_weight_profiles(cfg, cat):
            if pid in fe.DORMANT_WEIGHT_PROFILES:
                continue
            assert counts.get(pid, 0) > 0, (
                f"{cat}/{pid} is a published weight table no stock is scored with. Either the rule "
                f"cannot fire (remove it, as pio_gt was on 2026-10-16) or name it in "
                f"factor_engine.DORMANT_WEIGHT_PROFILES with the reason")


def test_every_profile_in_the_payload_is_applied_or_retired(payload):
    """The payload's own tables: each one is used, dormant by declaration, or retired (a run
    that predates the removal can still carry it) - and a retired one is assigned to nobody."""
    profiles = payload["weights"]["profiles"]
    for cat, tables in profiles.items():
        counts = _assigned(payload, cat)
        for pid in tables:
            if pid in fe.RETIRED_WEIGHT_PROFILES:
                assert counts.get(pid, 0) == 0, f"{cat}/{pid} is retired but a stock was scored with it"
                continue
            if pid in fe.DORMANT_WEIGHT_PROFILES:
                continue
            assert counts.get(pid, 0) > 0, f"{cat}/{pid} is published and applied to no stock"


def test_dormant_and_retired_profiles_carry_reasons():
    for reasons in (fe.DORMANT_WEIGHT_PROFILES, fe.RETIRED_WEIGHT_PROFILES):
        assert all(isinstance(v, str) and len(v) > 40 for v in reasons.values())
    assert not set(fe.RETIRED_WEIGHT_PROFILES) & set(fe.WEIGHT_PROFILE_LABELS)


def test_removing_the_variant_changed_no_score(cfg):
    """The variant could not fire on run_screener's single pass, so removing it moves nothing.
    Score a frame the way the pipeline does, and confirm no stock lands on a retired table."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)],
                       "_is_bank_like": [i % 10 == 0 for i in range(n)]})
    for cat, ms in fe.CAT_METRICS.items():
        for m in ms:
            df[m] = rng.normal(size=n)
            df[f"{m}_pct"] = rng.uniform(0, 100, size=n)
    out = fe.compute_category_scores(df.copy(), cfg)
    for cat in fe.CAT_METRICS:
        assert not set(out[f"_wp_{cat}"]) & set(fe.RETIRED_WEIGHT_PROFILES)
    assert (out["_wp_quality"] == "pio_lowval").any()
