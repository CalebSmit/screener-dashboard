"""The claims register must stay a tripwire, not become documentation.

`claims.py` holds the argument for why this exists. The short version: on
2026-10-06 the first sentence of all 502 drilldowns called the cardinal
composite a percentile - telling the stock ranked **1st of 502** that it
"scores above 74% of the universe" - and it survived because
`tests/test_stock_summary.py` contained a test *asserting* it, citing a
methodology page that contradicted itself. Nothing enumerated the set of
computational claims the site makes, so nothing could notice one was unchecked.

These tests enforce six properties:

1. every sentence builder in `stock_summary.py` has a register entry, so a new
   `_sentence_*` function fails the suite until its claim is written down;
2. every `made_true_by` resolves to real code;
3. every `checked_by` names a test that exists (this caught eleven invented
   names on the day the register was written);
4. every `detect` pattern still matches its surface, so rewording a registered
   sentence cannot quietly strand the register;
5. no `FORBIDDEN` statement appears in anything published; and
6. the two corrected claims are *positively* true of a fresh build - the rank
   share for all 502 stocks, and the methodology page no longer contradicting
   itself.

Per CLAUDE.md rule 10, claims are asserted against the **generator's output**,
with a separate tripwire for drift in the committed artifact. A test that reads
only the committed file passes on a hand-edit and says nothing about what the
next data run will publish.

Stage T0a of `plan/calculation-transparency.md`.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

import claims
import run_screener
import stock_summary

ROOT = Path(__file__).resolve().parent.parent
OVERVIEW = ROOT / "SCREENER_OVERVIEW.md"
PAYLOAD = ROOT / "dashboard_data.js"
README = ROOT / "README.md"
LIVE_PAGE = ROOT / "index.html"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated_overview(cfg) -> str:
    """The methodology page the next data run will publish."""
    return run_screener.build_screener_overview(cfg)


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip("dashboard_data.js not present")
    text = PAYLOAD.read_text(encoding="utf-8", errors="replace")
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def _test_functions(path: Path) -> set[str]:
    """Every selectable test name in a module: bare functions, and ``Class::method``
    for tests inside a class.

    The class form was added 2026-10-25 by the claims mutation audit
    (``plan/claims-mutation-audit.md``). Its 2026-10-20 pass recorded "``checked_by``
    cannot name a class method" as a known limitation, because this helper returned
    only bare names while ``test_every_checked_by_test_exists`` partitions a reference
    on its *first* ``::`` - so ``tests/test_metrics.py::TestPriceTargetUpside::test_clamp_high``
    was unregistrable and the good tests in that module were invisible to the register
    and therefore to the data loop's publish gate. The audit's 2026-10-25 pass hit it a
    second time (``TestStalenessThreshold::test_config_override_threshold`` is the only
    test in the suite that pins the configured stale-filing bar), so the limitation is
    fixed rather than worked around with a duplicate module-level test.

    Every other consumer of ``checked_by`` already takes ``ref.split("::")[0]`` for the
    module, so the longer form needs nothing from them; pytest selects it directly.

    **Only selectable names** (tightened 2026-10-26, same audit). This used ``ast.walk``, so a
    method's *bare* name counted as existing too - and five registrations
    (``overview.share_classes`` x3, ``overview.effective_weights`` x2) named class methods that
    way. ``pytest tests/test_share_classes.py::test_config_declares_one_line_per_known_pair``
    reports "not found": the register named a check nobody could run by its reference, which
    is how the mutation harness found it. A bare name now means a module-level function.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for n in cls.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(f"{cls.name}::{n.name}")
    return names


# ---------------------------------------------------------------------------
# 1. the register covers every sentence the tool emits
# ---------------------------------------------------------------------------

def test_every_sentence_builder_is_registered():
    """A new `_sentence_*` in stock_summary.py must be registered before it can
    ship. This is the structural gap that let defect 3 live: the *set* of
    claims was never enumerated, so an unchecked one was invisible.
    """
    source = ast.parse((ROOT / "stock_summary.py").read_text(encoding="utf-8"))
    builders = {n.name for n in ast.walk(source)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name.startswith("_sentence_")}
    assert builders, "no sentence builders found - has stock_summary.py moved?"

    unregistered = builders - claims.registered_builders()
    assert not unregistered, (
        "these sentence builders state something about a number and have no "
        f"entry in claims.CLAIMS: {sorted(unregistered)}. Add one - id, what it "
        "asserts, the code that makes it true, and the test that checks it."
    )


def test_register_names_no_builder_that_has_been_deleted():
    """The inverse: a register entry for code that no longer exists is a claim
    nobody is checking any more."""
    source = ast.parse((ROOT / "stock_summary.py").read_text(encoding="utf-8"))
    builders = {n.name for n in ast.walk(source)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    stale = claims.registered_builders() - builders
    assert not stale, f"claims.CLAIMS references missing builders: {sorted(stale)}"


def test_every_made_true_by_resolves():
    for claim in claims.CLAIMS:
        try:
            claim.resolve()
        except (ImportError, AttributeError) as exc:
            pytest.fail(f"{claim.id}: made_true_by={claim.made_true_by!r} "
                        f"does not resolve ({exc})")


def test_every_checked_by_test_exists():
    """Eleven of the twenty-six references were wrong when the register was
    first written - every one a plausible-sounding name that did not exist. A
    register of claims pointing at imaginary tests is worse than no register,
    because it reads as coverage.
    """
    cache: dict[str, set[str] | None] = {}
    problems = []
    for claim in claims.CLAIMS:
        for ref in claim.checked_by:
            filename, _, func = ref.partition("::")
            if filename not in cache:
                path = ROOT / filename
                cache[filename] = _test_functions(path) if path.exists() else None
            names = cache[filename]
            if names is None:
                problems.append(f"{claim.id}: {filename} does not exist")
            elif func not in names:
                problems.append(f"{claim.id}: {filename} has no {func}")
    assert not problems, "\n".join(problems)


def test_every_claim_is_complete_and_uniquely_identified():
    ids = [c.id for c in claims.CLAIMS]
    assert len(ids) == len(set(ids)), "duplicate claim ids"
    for claim in claims.CLAIMS:
        assert claim.asserts.strip(), f"{claim.id}: empty `asserts`"
        assert claim.checked_by, f"{claim.id}: no check - a claim with no check fails"
        assert claim.surface in (claims.SUMMARY, claims.OVERVIEW, claims.README, claims.DRILLDOWN,
                                 claims.RESEARCH, claims.LAB), \
            f"{claim.id}: unknown surface {claim.surface!r}"


def test_every_detect_pattern_still_matches_its_surface(generated_overview):
    """A registered sentence that has been reworded away leaves the register
    describing a page that no longer says it."""
    surfaces = {
        claims.OVERVIEW: generated_overview,
        claims.README: README.read_text(encoding="utf-8"),
    }
    for claim in claims.CLAIMS:
        if not claim.detect:
            continue
        text = surfaces.get(claim.surface)
        assert text is not None, f"{claim.id}: no text for surface {claim.surface}"
        assert re.search(claim.detect, text), (
            f"{claim.id}: detect pattern {claim.detect!r} no longer matches "
            f"{claim.surface}. Either the sentence moved (update the register) "
            f"or the claim was dropped (remove it)."
        )


# ---------------------------------------------------------------------------
# 2. the forbidden statements appear nowhere published
# ---------------------------------------------------------------------------
# NOTE: deliberately excludes NIGHTLY_LOG.md, METHODOLOGY_CHANGELOG.md,
# research/ and claims.py itself. Those *quote* the false sentences as the
# historical record, which is the point of them.

def test_no_forbidden_claim_in_any_published_artifact(generated_overview):
    artifacts: dict[str, str] = {
        "build_screener_overview() output": generated_overview,
    }
    for path in (OVERVIEW, README, LIVE_PAGE, PAYLOAD,
                 ROOT / "plan" / "investor-profiles.md"):
        if path.exists():
            artifacts[str(path.relative_to(ROOT))] = path.read_text(
                encoding="utf-8", errors="replace")

    problems = []
    for name, text in artifacts.items():
        for hit in claims.forbidden_hits(text):
            problems.append(f"{name}: {hit.pattern!r} - {hit.why} "
                            f"(fixed {hit.fixed})")
    assert not problems, "forbidden claims are published:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# 3. the corrected claims are positively true
# ---------------------------------------------------------------------------

def test_rank_share_is_exact_for_every_published_stock(payload):
    """For all ~502 stocks the printed share must equal (N - rank) / (N - 1).

    Against the pre-fix tree this failed for **75% of stocks**, by a median of
    19.6 percentage points. Asserted against `_sentence_rank` - the generator -
    for every real stock in the payload, not against a fixture.
    """
    detail = payload["stock_detail"]
    n = len(detail)
    assert n > 100, f"expected a full universe, got {n}"

    pattern = re.compile(r"ahead of (\d+)% of the other (\d+) stocks")
    failures = []
    for ticker, stock in detail.items():
        sentence = stock_summary._sentence_rank(stock, n)
        assert sentence, f"{ticker}: no rank sentence"
        match = pattern.search(sentence)
        if not match:
            failures.append(f"{ticker}: no share in {sentence!r}")
            continue
        expected = round((n - stock["rank"]) / (n - 1) * 100)
        if int(match.group(1)) != expected:
            failures.append(f"{ticker}: printed {match.group(1)}%, "
                            f"rank {stock['rank']} implies {expected}%")
        if int(match.group(2)) != n - 1:
            failures.append(f"{ticker}: denominator {match.group(2)} != {n - 1}")
    assert not failures, f"{len(failures)} of {n} wrong:\n" + "\n".join(failures[:15])


def test_no_rank_sentence_calls_the_composite_a_percentile(payload):
    detail = payload["stock_detail"]
    n = len(detail)
    offenders = [t for t, s in detail.items()
                 if "is a percentile" in (stock_summary._sentence_rank(s, n) or "")]
    assert not offenders, f"{len(offenders)} stocks still claim it: {offenders[:10]}"


def test_the_published_payload_carries_the_corrected_sentence(payload):
    """Drift tripwire. The summaries are baked into `stock_detail` at build
    time, so the committed payload keeps whatever the last run wrote. If this
    fails while the generator tests pass, the payload needs regenerating -
    the fix is not live until it is.
    """
    stale = []
    for ticker, stock in payload["stock_detail"].items():
        text = stock_summary.summary_text(stock.get("summary"))
        if "is a percentile" in text:
            stale.append(ticker)
    assert not stale, (
        f"{len(stale)} baked summaries still call the composite a percentile "
        f"(e.g. {stale[:5]}). Regenerate the dashboard - `generate_dashboard.py` "
        f"rebuilds summaries via build_summary()."
    )


def test_overview_step5_describes_a_cardinal_composite(generated_overview):
    assert "The composite is cardinal, and it is not a percentile" in generated_overview
    assert "`Composite_Pct`" in generated_overview
    # The specific misreading the old sentence taught, named and refused.
    assert re.search(r"do not read a composite of 95 as", generated_overview, re.I)


def test_overview_does_not_contradict_itself_on_the_composite(generated_overview):
    """Before 2026-10-07 Step 5 said the composite *was* a percentile and
    Limitation 8 said *"Do not read the cardinal Composite as a percentile"* -
    on the same page, both published. Limitation 8 was right."""
    assert "Do not read the cardinal Composite as a percentile" in generated_overview
    assert "raw composite is then converted to a cross-sectional percentile" \
        not in generated_overview


def test_overview_states_the_coverage_discount_from_config(generated_overview, cfg):
    """Step 5's formula omitted the coverage discount entirely, while the Data
    Quality section called it "currently enabled". Its numbers must come from
    config, not from prose that can go stale."""
    cov = cfg["data_quality"]["coverage_discount"]
    if not cov.get("enabled", False):
        pytest.skip("coverage discount disabled in config")
    threshold = int(cov["threshold"] * 100)
    rate = int(cov["penalty_rate"] * 100)
    assert "coverage discount described under Data quality safeguards" in generated_overview
    assert f"a stock below {threshold}% metric coverage" in generated_overview
    assert f"{rate}%" in generated_overview


def test_the_engine_applies_the_discount_the_page_states(generated_overview, cfg):
    """Claims ``overview.coverage_discount`` and ``drilldown.composite_chain``: the
    engine's discount is the page's formula, at the page's threshold and rate.

    Every check on these two claims read either the page's text or the payload's own
    ``disc`` field, which is whatever the engine applied - so halving the rate inside
    ``compute_composite``, or starting the discount five points below the stated
    threshold, passed all of them (mutation audit 2026-10-20,
    ``plan/claims-mutation-audit.md``). Here the threshold and rate are read off the
    generated page and the engine is run on stocks either side of that threshold."""
    import numpy as np
    import pandas as pd
    import factor_engine as fe

    cov = cfg["data_quality"]["coverage_discount"]
    if not cov.get("enabled", False):
        pytest.skip("coverage discount disabled in config")
    m = re.search(r"Below (\d+)% metric coverage, the composite is multiplied by "
                  r"1 \S \((\d+)% \S coverage\) \S (\d+)%", generated_overview)
    assert m, "the discount formula sentence has moved"
    thr, thr2, rate = (int(x) / 100 for x in m.groups())
    assert thr == thr2

    gen, _ = fe.weighted_metric_sets(cfg)
    gen = [c for c in fe.METRIC_COLS if c in gen]
    n = len(gen)
    # Present counts from full coverage down past the threshold, including every count
    # within ten points below it - where a shifted threshold would show.
    counts = sorted({n, int(np.ceil(thr * n)), *range(int(thr * n) - 3, int(thr * n) + 1),
                     int(0.6 * n)}, reverse=True)
    rows = []
    for i, k in enumerate(counts):
        row = {"Ticker": f"S{i}", "Sector": "Industrials", "_is_bank_like": False}
        for j, c in enumerate(gen):
            row[c] = 1.0 if j < k else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in fe.METRIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for col in ("valuation_score", "quality_score", "growth_score", "momentum_score",
                "risk_score", "revisions_score", "size_score", "investment_score"):
        df[col] = 50.0
    out = fe.compute_composite(df, cfg)
    seen_below = 0
    for k, (_, r) in zip(counts, out.iterrows()):
        coverage = k / n
        expect = (thr - coverage) * rate if coverage < thr else 0.0
        seen_below += coverage < thr
        assert r["_cov_discount"] == pytest.approx(expect, abs=1e-9), f"{k} of {n}"
        # the composite is published to 2dp
        assert r["Composite"] == pytest.approx(50.0 * (1 - expect), abs=0.0051), f"{k} of {n}"
    assert seen_below >= 3


def test_confidence_metric_count_is_the_discount_coverage(payload, generated_overview):
    """The drilldown's "N of M metrics" must be the coverage the discount reads.

    Found 2026-10-07 and fixed in T0b: the sentence and the provenance badge counted
    a hard-coded 18-metric list while the discount counts the metrics applicable to
    the stock (35 bank-like, 41 otherwise), so a bank read 12/18 = 67% and was not
    discounted at all. Now both come from `factor_engine.applicable_coverage`.
    """
    # Since 2026-10-09: metrics that carry weight in the stock's table
    # (factor_engine.weighted_metric_sets), read from the run's own config.
    import yaml
    from factor_engine import weighted_metric_sets

    if not (ROOT / "runs").is_dir():
        # Gitignored, so absent in a fresh clone or worktree (OWNER_FOCUS 2026-10-20 item 7).
        # Only absence skips: a runs/ folder with no usable run still fails as before.
        pytest.skip(f"MISSING DATA: {ROOT / 'runs'} is absent (gitignored) - this test reads the "
                    "newest full run and cannot run without one")
    run = max((d for d in (ROOT / "runs").iterdir() if (d / "config.yaml").exists()
               and (d / "05_final_scored.parquet").exists()),
              key=lambda d: (d / "05_final_scored.parquet").stat().st_mtime)
    gen, bnk = weighted_metric_sets(yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")))
    bank, other = len(bnk), len(gen)

    stocks = payload["stock_detail"].values()
    assert all(s.get("cov") for s in stocks), "every stock must publish its coverage"
    for s in stocks:
        assert s["metric_count"] == s["cov"]["n"]
        assert s["metric_total"] == s["cov"]["of"]
        # Financials on the generic table never get the Beneish score (2026-10-09), so it is
        # not counted as applicable to them
        fin_generic = (not s["flags"]["is_bank"]) and s.get("sector") == "Financials"
        expected = bank if s["flags"]["is_bank"] else (other - 1 if fin_generic else other)
        assert s["cov"]["of"] == expected, (
            f"{s['cov']} for a {'bank' if s['flags']['is_bank'] else 'non-bank'} stock; "
            f"expected an applicable set of {expected}")

    # The methodology page says the badge and the discount read the same figure.
    assert f"{bank}" in generated_overview and f"{other}" in generated_overview
    assert "read **this same figure**" in generated_overview


#  ---------------------------------------------------------------------------
# 4. negative controls - the tripwires must fire on input they must reject
# ---------------------------------------------------------------------------
# CLAUDE.md rule 8's own lesson: a tripwire wired to something that cannot
# move is decoration. These pin the *detection*, using the exact sentences that
# were live on 2026-10-06, so the guard cannot be loosened into a no-op while
# the suite stays green.

@pytest.mark.parametrize("sentence", [
    "Ranks 1st of 502. Its composite of 73.8 is a percentile: it scores above "
    "74% of the universe.",
    'The raw composite is then converted to a cross-sectional percentile rank '
    '(0-100), so a score of 95 means "better than 95% of stocks in the universe."',
    'converted to a **cross-sectional percentile rank** (`rank(pct=True) * 100`) '
    '- so a score of 95 means "better than 95% of the universe."',
    "### 1. `Composite` is a percentile rank, not the weighted sum",
])
def test_each_false_sentence_that_shipped_is_still_detected(sentence):
    """Every one of these was published on the live site until 2026-10-07."""
    assert claims.forbidden_hits(sentence), (
        f"FORBIDDEN no longer catches a sentence that shipped: {sentence[:80]!r}")


def test_the_forbidden_patterns_do_not_fire_on_the_corrected_page(generated_overview):
    """The other half of a usable guard: no false positives on today's text,
    or the next session will delete it to get green."""
    assert not claims.forbidden_hits(generated_overview)


def test_the_corrected_rank_sentence_is_not_itself_forbidden():
    built = stock_summary._sentence_rank(
        {"rank": 1, "composite": 73.84,
         "cat_scores": {c: 50.0 for c in stock_summary.CATEGORIES}}, 502)
    assert not claims.forbidden_hits(built)
    assert not stock_summary.advice_terms_in(built), "new sentence carries advice language"


def test_the_recorded_caveats_say_whether_fixed_or_which_stage_extends_them():
    """Every recorded caveat is either marked fixed (with the stage that fixed it) or names
    the stage that will extend it, so the next session inherits the list instead of
    rediscovering it. T0b - the stage the first version of this test was waiting on -
    shipped 2026-10-07."""
    import re as _re
    recorded = claims.open_caveats()
    assert recorded, "expected at least one recorded caveat"
    for claim in recorded:
        assert ("Fixed" in claim.caveat) or _re.search(r"\bT[0-6][ab]?\b|\bD[2-8]\b", claim.caveat), (
            f"{claim.id}: caveat neither says it is fixed nor names the stage that closes it")