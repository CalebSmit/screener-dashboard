"""The positive control: can the research engine see a signal at all?

WHY THIS EXISTS - plan/research-engine.md item 4(c), OWNER_FOCUS.md 2026-10-20 item 4.

Twenty-nine registered experiments report "not distinguishable from the baseline". A reader is
entitled to ask whether that is a fact about the weightings or a fact about the test, and a null
result cannot answer that for itself: it needs a case where the answer is known. The *control* is
that case - the same weighting, the same universe, the same prices, but every month scored on the
filings as they stand at the build date, which is backtest v1's look-ahead built on purpose and
worth about +0.05 of monthly IC (research/2026-10-15-pit-record-validation.md section 5c).

The public research page has asserted "The test can see a signal when there is one" since
2026-10-15 with no number behind it, on the strength of a measurement that was never a ledger row
and whose monthly records have since been rebuilt twice. These tests pin the three things that make
the sentence true from here: the control is a *control* and not an experiment, its row names its own
data by content, and the page reports what the row actually says - including when the row does not
exist, and including when the design failed to see the signal.

Deliberately tolerant of an unrun control. A fresh checkout has no
``cache/v2_scores_lookahead.parquet`` (it is gitignored, and building it takes about half an hour),
so every test that needs a real row skips loudly - OWNER_FOCUS.md 2026-10-20 item 7, three tests
that blew up with FileNotFoundError in a worktree. What must hold without any run at all is the
shape: the registration, the refusals, and the page's silence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claims  # noqa: E402
import research_page  # noqa: E402
import v2_experiments as X  # noqa: E402

CONTROL_ID = "lookahead-filings"


@pytest.fixture(scope="module")
def entry() -> dict:
    by_id = {e["id"]: e for e in X.load_controls()}
    assert CONTROL_ID in by_id, f"{CONTROL_ID} is not registered"
    return by_id[CONTROL_ID]


@pytest.fixture(scope="module")
def row() -> dict:
    """The newest ledger row for the control, or a skip if it has never been run."""
    latest = {}
    if X.LEDGER.exists():
        for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["id"]] = r
    if CONTROL_ID not in latest:
        pytest.skip(f"{CONTROL_ID} has not been run yet - build the arm, then "
                    "`python v2_experiments.py control`")
    return latest[CONTROL_ID]


# ---------------------------------------------------------------------------
# 1. a control is a control, not an experiment
# ---------------------------------------------------------------------------
def test_the_control_is_registered_outside_experiments(entry):
    """``experiments:`` holds weighting transforms and everything downstream reads
    ``e["transform"]``; the 2026-10-23 session refused three registrations because, written as
    experiments, they raised ``KeyError: 'transform'`` in ``report``. A control has no transform, so
    it must live in its own section - and must not be smuggled into that one."""
    assert "transform" not in entry
    assert entry["id"] not in {e["id"] for e in X.load_registry()}
    assert entry["kind"] == "control"


def test_the_control_is_not_counted_in_the_multiple_testing_bar():
    """Counting the engine's own self-check as an alternative would make every real experiment's
    bar stricter because the engine checked itself."""
    reg = X.load_registry()
    n = len([e for e in reg if e["transform"]["kind"] != "baseline"])
    assert CONTROL_ID not in {e["id"] for e in reg}
    # the report's denominator is computed from load_registry() alone
    assert n == len([e for e in X.load_registry() if e["transform"]["kind"] != "baseline"])


def test_a_control_must_disclose_that_its_outcome_was_known(entry):
    """A control's answer is known before it runs - that is what makes it a control - so a control
    that does not say so is a false pre-registration wearing a real one's clothes."""
    assert entry.get("outcome_known_at_registration")
    with pytest.raises(SystemExit, match="outcome_known_at_registration"):
        X.refuse_undisclosed_controls([{"id": "undisclosed"}])
    X.refuse_undisclosed_controls([entry])          # the registered one is fine


def test_the_registration_fixes_a_rule_that_can_fail(entry):
    """A pass rule written after the number is not a pre-registration. It has to name the period,
    the direction, the minimum |t| and the band - and a failure condition that says what a failure
    would mean, because a control that could only come out well is decoration."""
    rule = entry["passes_if"]
    assert rule["period"] in ("discovery", "holdout", "all", "presample")
    assert rule["direction"] in ("positive", "negative")
    assert float(rule["min_abs_t"]) >= X.HLZ_T
    lo, hi = (float(x) for x in rule["band"])
    assert 0 < lo < hi
    assert entry.get("fails_if", "").strip(), "no failure condition registered"


def test_no_scoring_module_may_read_the_control():
    """The engine is research tooling (CLAUDE.md: never imported by production code), and a control
    is the furthest thing in it from decision-grade - it is deliberately cheating."""
    for mod in ("factor_engine.py", "run_screener.py", "generate_dashboard.py",
                "portfolio_constructor.py", "stock_summary.py", "improvement_engine.py"):
        text = (ROOT / mod).read_text(encoding="utf-8")
        assert "v2_experiments" not in text, mod
        assert "load_controls" not in text, mod
        assert "v2_scores_lookahead" not in text, mod


# ---------------------------------------------------------------------------
# 2. the verdict separates "did the design see it" from "was the prediction right"
# ---------------------------------------------------------------------------
def _res(mean: float, t: float, n: int = 80, se: float = 0.011) -> dict:
    return {"summary": {"all": {"ic": {"mean": mean, "n": n},
                                "ic_vs_baseline": {"mean": mean, "t": t, "se": se, "n": n}}}}


def test_the_verdict_fails_when_the_design_cannot_see_the_signal(entry):
    """The case the control exists to catch. A difference that does not clear the bar must come back
    as a failure in the engine's own words, not as a quiet pass."""
    v = X.control_verdict(entry, _res(0.004, 0.4), 29)
    assert v["sees"] is False
    assert "did NOT see it" in v["words"]
    assert "not distinguishable" in v["words"]          # and it says what that costs


def test_the_verdict_passes_only_in_the_registered_direction(entry):
    """A large difference the wrong way is not the signal that was registered: look-ahead that made
    the ranking worse would mean something is wired backwards."""
    assert X.control_verdict(entry, _res(0.052, 6.0), 29)["sees"] is True
    assert X.control_verdict(entry, _res(-0.052, -6.0), 29)["sees"] is False


def test_the_band_is_reported_but_never_decides_whether_the_test_works(entry):
    """Two questions, and only one of them is about the engine. A control that clears the bar but
    lands outside the band it predicted has still shown the design can see a signal."""
    v = X.control_verdict(entry, _res(0.120, 9.0), 29)
    assert v["sees"] is True and v["in_band"] is False
    assert "outside the band" in v["words"]
    inside = X.control_verdict(entry, _res(0.052, 6.0), 29)
    assert inside["in_band"] is True and "inside the band" in inside["words"]


def test_the_verdict_reports_what_the_comparison_could_detect(entry):
    """"It saw a big one" is only informative beside the smallest difference it could have seen."""
    v = X.control_verdict(entry, _res(0.052, 6.0), 29)
    assert v["detectable"] is not None and 0 < v["detectable"] < 0.052


# ---------------------------------------------------------------------------
# 3. the row names its own data, which content_hash cannot see
# ---------------------------------------------------------------------------
def test_the_arm_is_named_by_content_because_content_hash_cannot_see_it(entry, tmp_path):
    """``content_hash`` covers ``cache/v2_metrics/`` - the point-in-time records, the *baseline*
    side of the pairing. The arm's scores are a separate file built by a separate command, so
    without its SHA-256 in the row a rebuild of the arm would move every number under a row that
    still looked current."""
    assert entry["scores"] not in X.DATA_SOURCES.values()
    p = tmp_path / "arm.parquet"
    pd.DataFrame({"month": ["2020-01-31"] * 3, "Ticker": list("ABC"),
                  "Composite": [1.0, 2.0, 3.0]}).to_parquet(p)
    e = dict(entry, scores=p.name, window=["2020-01-31", "2026-08-31"])
    _, prov = X.control_scores(e, root=tmp_path)
    assert prov["scores_sha256"] == X._file_sha256(p)[:16]
    assert prov["months"] == 1 and prov["stock_months"] == 3


def test_an_unbuilt_arm_is_reported_and_never_guessed_at(entry, tmp_path):
    """And it names the command that builds it, rather than failing with a path."""
    with pytest.raises(SystemExit, match="not built"):
        X.control_scores(dict(entry, scores="nope.parquet"), root=tmp_path)


def test_months_outside_the_registered_window_are_dropped(entry, tmp_path):
    """The window was registered before the run; a later build covering more months cannot quietly
    widen the comparison it was registered on."""
    p = tmp_path / "arm.parquet"
    pd.DataFrame({"month": ["2019-12-31", "2020-01-31", "2026-09-30"],
                  "Ticker": list("ABC"), "Composite": [1.0, 2.0, 3.0]}).to_parquet(p)
    df, prov = X.control_scores(dict(entry, scores=p.name), root=tmp_path)
    assert sorted(df["month"].unique()) == ["2020-01-31"] and prov["months"] == 1


def test_the_rerun_scheduler_reruns_a_control_whose_arm_was_rebuilt():
    """The four provenance keys cannot see the arm's file, so a rebuild of it would leave the row
    looking current. The scheduler compares the recorded SHA-256 with the file on disk."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import rerun_experiments as R

    arm = ROOT / {e["id"]: e for e in X.load_controls()}[CONTROL_ID]["scores"]
    if not arm.is_file():
        pytest.skip("the arm is not built in this checkout")
    current = R.current_provenance(X)
    fresh = dict(current, **{"control_data": {"scores_sha256": X._file_sha256(arm)[:16]}})
    latest = {CONTROL_ID: fresh}
    assert R.stale_controls(X, latest, current)[0] == []
    latest[CONTROL_ID] = dict(fresh, control_data={"scores_sha256": "0" * 16})
    assert R.stale_controls(X, latest, current)[0] == [CONTROL_ID]


# ---------------------------------------------------------------------------
# 4. the page reports what the row says - the registered claim
# ---------------------------------------------------------------------------
def test_the_page_makes_no_claim_without_a_comparable_row():
    """A fresh checkout, or a control row built on other records, must leave the page saying the
    check is being re-measured - never asserting that the test can see a signal."""
    s = research_page.control_sentence(None)
    assert "being re-measured" in s
    assert "can see a signal when there is one" not in s


def test_the_page_says_so_when_the_design_did_not_see_it():
    """A page that could only report the reassuring outcome would be worthless."""
    s = research_page.control_sentence({"sees": False, "ic_vs_baseline": 0.004, "t": 0.4,
                                        "months": 80, "detectable": 0.04})
    assert "did not see a signal" in s and "cannot tell" in s


def test_the_page_quotes_the_controls_own_row(row):
    """And the figures on the page are the row's, not a re-derivation of them."""
    page = research_page.PAGE.read_text(encoding="utf-8")
    summary = json.loads((ROOT / "research" / "experiments" / "summary.json").read_text(encoding="utf-8"))
    c = summary.get("control")
    assert c, "the control ran but summary.json does not carry it - run `report`"
    assert c["id"] == CONTROL_ID
    st = row["summary"][c["period"]]["ic_vs_baseline"]
    assert c["ic_vs_baseline"] == st["mean"] and c["t"] == st["t"]
    assert c["months"] == row["months"]
    assert research_page.control_sentence(c) in page
    assert research_page.num(c["ic_vs_baseline"]) in page


def test_the_claim_is_registered_against_the_code_that_makes_it_true():
    c = {x.id: x for x in claims.CLAIMS}["research.positive_control"]
    assert c.surface == claims.RESEARCH
    assert c.resolve() is research_page.control_sentence


# ---------------------------------------------------------------------------
# 5. and the row, once it exists, is what the registration promised
# ---------------------------------------------------------------------------
def test_the_row_was_paired_on_the_months_and_stocks_both_rankings_share(row):
    """Method version 2. A control measured over its own set of stocks would be the mistake 8a
    fixed, and here it would flatter the arm: it ranks a few stocks the baseline cannot."""
    assert row["method_version"] == X.METHOD_VERSION
    assert row.get("n_common_by_month") and row.get("base_ic_by_month")
    assert set(row["ic_by_month"]) == set(row["base_ic_by_month"])


def test_the_row_names_its_arm_and_the_records_it_was_paired_against(row):
    d = row.get("control_data") or {}
    assert len(d.get("scores_sha256", "")) == 16
    assert d.get("built_by", "").startswith("python backtest_v2.py --diagnostic-filings-as-of")
    assert row.get("content_hash") and row.get("code_hash")
    assert not row["code_version"].endswith("-dirty")


def test_the_control_ran_on_the_same_records_as_the_baseline(row):
    """The whole comparison rests on this: the arm differs from the baseline in the filings it
    read and in nothing else, so the baseline side has to be these records."""
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    base = latest["baseline"]
    assert X.comparable_provenance(row, base)
    assert row["data_hash"] == base["data_hash"]
