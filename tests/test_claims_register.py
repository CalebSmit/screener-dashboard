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
def generated_research() -> str:
    """The research page `python v2_experiments.py report` would publish from the committed
    ledger (2026-10-29). The generator's output rather than the committed `research.html`, for
    the reason in this module's docstring: a reword in the generator has to go stale here on the
    session that makes it, not on the one that happens to regenerate the page.

    Needed because `research.html` gained registered sentences with `detect` patterns; the two
    research claims that predate them are one of three mutually exclusive states each, so they
    carry no pattern and this fixture is not built for them.
    """
    import research_page
    import v2_experiments as X
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = row
    reg = X.load_registry()
    return research_page.build(reg, latest, X.summary_for_page(reg, latest),
                               X.detectable, X.load_evaluations())


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


def test_every_detect_pattern_still_matches_its_surface(generated_overview, generated_research):
    """A registered sentence that has been reworded away leaves the register
    describing a page that no longer says it."""
    surfaces = {
        claims.OVERVIEW: generated_overview,
        claims.README: README.read_text(encoding="utf-8"),
        claims.RESEARCH: generated_research,
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


COVERAGE_COUNTS = re.compile(
    r"in that stock's score\*\* [-–] (\d+) for a bank-like stock and (\d+) for every other "
    r"\((\d+) for other financials, which are\s+never given the Beneish score\)")


def _applicable(cfg: dict, *, bank: bool, beneish_na: bool) -> int:
    """How many metrics the engine's own coverage counts as applying to one stock of that kind."""
    import pandas as pd
    from factor_engine import METRIC_COLS, applicable_coverage
    df = pd.DataFrame({m: [float("nan")] for m in METRIC_COLS})
    df["_is_bank_like"], df["_beneish_na"] = [bank], [beneish_na]
    _, applicable = applicable_coverage(df, cfg)
    return int(applicable.iloc[0])


def test_the_coverage_sentence_counts_what_the_engine_counts(cfg, generated_overview):
    """Claim ``overview.coverage_counts`` (audit row `coverage-sentence-metric-counts`). The
    methodology page says coverage is out of N for a bank-like stock, M for every other, and M - 1
    for other financials "which are never given the Beneish score". Each figure, read out of that
    sentence on the page the next run will publish, must be the denominator
    ``factor_engine.applicable_coverage`` - the function the discount reads - gives a stock of
    that kind under the same config. The third is the one nothing looked for: it is printed as
    `len(_w_gen) - 1`, an arithmetic on the generic count that is true only while Beneish carries
    generic weight and the engine excuses financials from it."""
    m = COVERAGE_COUNTS.search(generated_overview)
    assert m, "the coverage sentence is not on the generated page in its registered shape"
    bank, other, financials = (int(x) for x in m.groups())
    assert bank == _applicable(cfg, bank=True, beneish_na=False)
    assert other == _applicable(cfg, bank=False, beneish_na=False)
    assert financials == _applicable(cfg, bank=False, beneish_na=True)
    assert len({bank, other, financials}) == 3, "the three kinds are counted the same - the fixture cannot tell them apart"


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


# ---------------------------------------------------------------------------
# 5. the same sweep, for a *generated page* rather than stock_summary.py
# ---------------------------------------------------------------------------
# `test_every_sentence_builder_is_registered` above works because
# `stock_summary.py` has a naming convention: a new `_sentence_*` cannot ship
# unregistered. No generated page has one, and that is the gap this module's own
# docstring names - "nothing enumerated the set of computational claims the site
# makes". It cost three claims on 2026-10-29: the research page's "How long would
# it take to prove an edge?" section printed a monthly swing, three month counts
# and two detectable differences, every one of them arithmetic this repo
# performs, and the measurement that session recorded is that **one mutation per
# figure passed the entire suite** (3,506 passed, three times out of three). They
# were not under-checked; nothing anywhere could see them change
# (`plan/claims-mutation-audit.md`, rows 106-117).
#
# Mutation cannot find that class, because it can only mutate what the register
# names. So this sweep enumerates the other direction: every sentence a generated
# page prints that puts a number *this repo computed* into prose. Each one needs a
# verdict, and a new or reworded one fails the suite until it gets one.
#
# **Scope, stated so it cannot drift.** A sentence is in scope when it
# interpolates a value that the generator formats as a number (a numeric format
# spec) or computes (arithmetic, or a call to one of its own statistics helpers).
# A figure read verbatim out of the ledger or the registry with no formatting is
# a *reported* number, not a computed one - the page-matches-its-source tests are
# what cover those, and `reported:` records it. Two classes of published number
# are deliberately **outside** this sweep because they are not sentences: the
# headline chart's dot, band and tick positions (`_pos`), and the bot table's
# cells (`_ex`). Both are named in `plan/claims-mutation-audit.md` with what they
# need; they are not covered here and this comment is the only place that says so.

_TAG = re.compile(r"</?([A-Za-z][\w-]*)[^>]*>|<[^>]*>")
# inline tags sit *inside* a sentence - a figure is wrapped in <span class="num">,
# so breaking on them would separate every number from the words around it. Every
# other tag is block-level and ends the unit, which is what keeps one key from
# swallowing a whole section (the first draft of this sweep produced a key running
# from the chart legend through the table to the next heading).
_INLINE = frozenset({"span", "a", "em", "strong", "code", "i", "b", "sup", "sub", "br", "small"})
_BREAK = "\x01"
_PROSE = re.compile(r"[A-Za-z][\w'&;-]*(?:\s+[A-Za-z][\w'&;-]*){2,}")
_HOLE = re.compile(r"\x00(\d+)\x00")
_NUMERIC_SPEC = re.compile(r"[fdeg,%]")
# the generator's own number helpers and statistics - a call to one of these is
# arithmetic performed here, whatever the value's provenance
_NUMBER_CALLS = frozenset({
    "num", "pct", "_pos", "months_needed", "detectable", "pass_bar", "t_critical",
    "p_value", "abs", "len", "sum", "round", "min", "max", "mean", "sqrt", "int", "float",
})


def _is_computed(fv: ast.FormattedValue) -> bool:
    """Whether this interpolation is a number the generator formatted or computed."""
    if fv.format_spec is not None:
        spec = "".join(c.value for c in fv.format_spec.values
                       if isinstance(c, ast.Constant) and isinstance(c.value, str))
        if _NUMERIC_SPEC.search(spec):
            return True
    if isinstance(fv.value, ast.BinOp):
        return True
    if isinstance(fv.value, ast.Call):
        func = fv.value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name in _NUMBER_CALLS
    return False


def _subtree(path: Path, within: str | None):
    """The module, or the one function named by ``within``.

    `research_page.py` exists only to write its page, so sweeping the file is the
    same as sweeping the generator. `run_screener.py` is not - it is the pipeline,
    and `build_screener_overview` is one function in it, so a file-wide sweep would
    enumerate progress lines and log messages as published sentences. The scope is
    the generator, never the file that happens to hold it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if within is None:
        return tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == within:
            return node
    raise AssertionError(f"{path.name} has no function {within} - the sweep's scope is gone")


def computed_sentences(path: Path, within: str | None = None,
                       markup: str = "html", skip_calls: frozenset = frozenset()) -> list[str]:
    """Every sentence in a generator that puts a computed number into prose.

    The key is the sentence as published, tag-stripped and whitespace-normalised,
    with each interpolation left in place as its own source expression. So a
    reword fires (as a `detect` pattern does, deliberately) **and** swapping which
    expression is printed fires - the second is the one a `{}` placeholder would
    miss, and it is how a figure gets taken off the wrong era.

    ``within`` scopes the walk to one function - see ``_subtree``.

    ``markup`` says what ends a block, and the two surfaces genuinely differ.
    In **html** a newline is whitespace - the browser collapses it - so only a
    block-level tag ends a unit. In **markdown** there are no tags to break on and
    a newline *is* the block boundary: a table row, a list item and a heading are
    each their own block. Breaking a markdown surface on tags alone produced one
    key that ran from a three-row table through the sentence after it
    (`plan/claims-mutation-audit.md`, the 2026-10-30 pass), which is the same
    swallowed-key failure the html sweep hit on its first run. Note this is safe
    only because a generator's newlines are the ones it *writes*: prose wrapped
    across source lines by implicit concatenation carries no newline into the
    output, so no sentence is split by it.
    """
    assert markup in ("html", "markdown"), markup
    out: list[str] = []
    tree = _subtree(path, within)
    # An f-string handed to one of `skip_calls` (`print`) is a log line, not a
    # published sentence - the dashboard generator's only computed f-string is one.
    skipped = {id(a) for node in ast.walk(tree) if isinstance(node, ast.Call)
               and getattr(node.func, "id", None) in skip_calls for a in node.args}
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr) or id(node) in skipped:
            continue
        parts, holes = [], []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                parts.append(f"\x00{len(holes)}\x00")
                holes.append(v)
        out.extend(_units("".join(parts), markup,
                          figure=[_is_computed(h) for h in holes],
                          shown=[ast.unparse(h.value) for h in holes],
                          brace="{"))
    return out


def _units(joined: str, markup: str, *, figure: list[bool], shown: list[str],
           brace: str, block: list[bool] | None = None, prose=None) -> list[str]:
    """Split one string-building expression into sentence keys.

    ``joined`` is its literal text with ``\\x00i\\x00`` marking hole ``i``;
    ``figure[i]`` says whether hole ``i`` is a computed number, ``shown[i]`` how it
    is printed in the key, and ``block[i]`` whether it inserts a block of markup
    (the JavaScript sweep's case - it ends the unit like a block-level tag).
    Shared by the Python and JavaScript sweeps so the two cannot drift on what a
    sentence is.
    """
    prose = prose or _PROSE
    out: list[str] = []
    if block:
        joined = _HOLE.sub(lambda m: _BREAK if block[int(m.group(1))] else m.group(0), joined)
    text = _TAG.sub(
        lambda m: " " if (m.group(1) or "").lower() in _INLINE else _BREAK, joined)
    for entity, plain in (("&mdash;", "-"), ("&ndash;", "-"), ("&nbsp;", " ")):
        text = text.replace(entity, plain)
    if markup == "markdown":
        text = text.replace("\n", _BREAK)
    units = re.split(r"(?<=[.!?:])\s+|" + _BREAK, text)
    # A stat card is a figure in one element and its caption in the next
    # (`<div class="v">{months_needed(...)}</div><div class="k">months of record
    # to show a monthly correlation of 0.05 ...</div>`). Block-splitting leaves
    # the figure with no words and the caption with no figure, so neither half
    # is a sentence and the claim disappears - which it did on the first run of
    # this sweep, for all three of the month counts registered the same morning.
    # A figure-only unit therefore takes the following unit's words.
    # A markdown table row is not a sentence, and must not be merged into one.
    # `| Earnings yield (E/P) | **+{bank_net_ep:.1f}** |` has a figure and no
    # three-word run (the "(E/P)" breaks it), so the stat-card merge below
    # swallowed the separator and the next two rows and emitted the whole table
    # as one "sentence". Table cells are already outside this sweep on the html
    # side by the same reasoning - a published figure with no prose around it -
    # and both surfaces now need the same decision about whether the register
    # covers cells (`plan/claims-mutation-audit.md`).
    if markup == "markdown":
        units = [u for u in units if not u.lstrip().startswith("|")]
    merged: list[str] = []
    for unit in units:
        if merged and _HOLE.search(merged[-1]) and not prose.search(_HOLE.sub(" ", merged[-1])):
            merged[-1] += " " + unit
        else:
            merged.append(unit)
    for raw in merged:
        idx = [int(m.group(1)) for m in _HOLE.finditer(raw)]
        if not any(figure[i] for i in idx):
            continue
        if not prose.search(_HOLE.sub(" ", raw)):
            continue                       # a bare figure with no words anywhere
        out.append(re.sub(r"\s+", " ", _HOLE.sub(
            lambda m: brace + shown[int(m.group(1))] + "}", raw)).strip())
    return out


# Every computed-number sentence on research.html, swept 2026-10-29 (13:00
# session), with what pins its arithmetic. The verdicts:
#
#   claim:<id>       a registered claim covers it, checked both ways
#   checked:<ref>    no claim, but that test recomputes the figure independently
#   reported:<ref>   the figure is read verbatim from the ledger/registry and
#                    that test pins the page against the source
#   open:<what>      at least one computed figure in the sentence is pinned by
#                    nothing. Dated below; a new sentence may not be filed here
#                    (see KNOWN_OPEN).
#
# `checked:` and `reported:` mean **every** computed figure in the sentence is
# pinned by that test. One unpinned figure makes the whole row `open:`, with a
# note saying which - otherwise a row reads as coverage it does not have, which is
# the failure this register exists to catch. Every verdict below was assigned by
# reading the named assertion, not by its name.
RESEARCH_PAGE_SENTENCES: dict[str, str] = {
    # --- the headline, and what the record could show -----------------------
    "The ranking's average monthly rank correlation with the next month's return was "
    "{num(pre_ic)} over 2012-2019":
        "checked:tests/test_research_page.py::test_its_headline_numbers_are_the_engine_s_summary",
    "Pooled over all {summary['months']['pooled']} months it was "
    "{num(summary['baseline_ic']['pooled'])} {_interval('pooled')}.":
        "checked:tests/test_research_page.py::test_every_headline_figure_carries_its_interval_from_the_ledger",
    "The pooled record could reliably detect an average of about "
    "{summary['pooled_detectable']} at a t-statistic of 3, and no smaller.":
        "checked:tests/test_research_page.py::test_every_headline_figure_carries_its_interval_from_the_ledger",
    "{months_needed(0.05, ic_sd, 1.96)} months of record to show a monthly correlation "
    "of 0.05 at the usual 5% level": "claim:research.months_to_prove",
    "{months_needed(0.05, ic_sd, 3.0)} months for the same with a t-statistic of 3":
        "claim:research.months_to_prove",
    "{months_needed(0.03, ic_sd, 3.0)} months to show 0.03 with a t-statistic of 3":
        "claim:research.months_to_prove",
    "The ranking's monthly correlation swings by about {ic_sd} from one month to the next, "
    "so any monthly ranking of large companies needs years of record before a realistic "
    "edge can be told apart from luck.": "claim:research.ic_swing",
    "Here, the smallest difference the test could reliably see was about {mde_p} over "
    "2012-2019 and {mde_h} over the held-out months.": "claim:research.detectable_difference",

    # --- the pass bar, printed twice ---------------------------------------
    "With p from Student's t, that means a t-statistic of at least {bar_p} over 2012-2019's "
    "{mid_p.get('n')} months and {bar_h} over the {mid_h.get('n')} held-out months.":
        "claim:research.pass_bar",
    "a difference counts only if it reaches beyond them, a t-statistic of {bar_p} over these "
    "{mid_p.get('n')} months, in the same direction as 2020-2026.":
        "claim:research.pass_bar",

    # --- the chart ----------------------------------------------------------
    "{''.join(dots)} {num(-LIM, 2)} {num(-LIM / 2, 2)} 0 {num(LIM / 2, 2)} {num(LIM, 2)} "
    "difference 95% interval pass bar smallest difference the test could reliably detect":
        "claim:research.chart_geometry",

    # --- the positive control ----------------------------------------------
    "Fed the filings as they stand today for every past month - information nobody had at the "
    "time - the same test puts that ranking {num(c['ic_vs_baseline'])} of rank correlation a "
    "month ahead of the real one over {c['months']} months (t {num(c['t'], 1)} )":
        "claim:research.positive_control",
    "Fed the filings as they stand today for every past month - information nobody had at the "
    "time - this test put the cheating ranking {num(c['ic_vs_baseline'])} ahead of the real one "
    "(t {num(c['t'], 1)} ), which does not clear its own bar.":
        "claim:research.positive_control",
    ", against the {num(c['detectable'])} it could reliably detect":
        "claim:research.positive_control",
    "{num(c['ic_vs_baseline'])} ahead of the real one": "claim:research.positive_control",

    # --- the universe the comparisons are measured on -----------------------
    "over 2020-2026 the current weighting ranked {b0['mean_stocks']} stocks a month on average "
    "and the narrowest alternative, {display_label(next((r['e'] for r in rows if r['id'] == "
    "m_id)))}, {m_n} .{overlap}": "claim:research.pairing_overlap",
    "The two had {c} companies a month in the overlap, and that is what each comparison is "
    "measured on.": "claim:research.pairing_overlap",
    "so pairing on the overlap holds the current weighting to that smaller set instead of "
    "crediting it with the {b0['mean_stocks'] - c} only it can score.":
        "claim:research.pairing_overlap",
    "( {only_narrow} a month are unique to it),": "claim:research.pairing_overlap",

    # --- the Size hypothesis ------------------------------------------------
    "{row['label']} shows the largest 2012-2019 difference, {num(p['mean'])} "
    "(t {num(p['t'], 2)}), {reach} the pass bar.":
        "claim:research.size_sentence",
    "The share of each month's members the test could include rose steadily, from "
    "{rel['first_cov'] * 100}% in {rel['first_year']} to {rel['last_cov'] * 100}% in "
    "{rel['last_year']} (its correlation with the calendar is {rel['r_time']} ), and within "
    "2012-2019 the Size screen's monthly result moved with that share {moved} the current "
    "weighting's did (correlations of {num(rel['pre_r_size'])} and {num(rel['pre_r_base'])} ).":
        "claim:research.size_sentence",

    # --- if you had kept the best backtest ---------------------------------
    # All five rows CLOSED 2026-10-31 (13:00; audit rows 139-143) by
    # `claim:research.overfitting_reading`, which also brings `tests/test_overfitting.py` into
    # the publish gate - it was named by no claim, so the eight checks in it, the stat cards
    # included, were outside the loop that publishes this page.
    "Over {a['first_month'][:4]}-{a['last_month'][:4]} the PBO was {a['pbo']} , "
    "{_pbo_words(a['pbo'])}, and the winner beat the current weighting on the other half in "
    "{a['winner_beat_current_out_of_sample'] * 100}% of trials.":
        "claim:research.overfitting_reading",
    "Over {rec['first_month'][:4]}-{rec['last_month'][:4]} alone the PBO was {rec['pbo']} , "
    "{_pbo_words(rec['pbo'])}, and the winner beat the current weighting out of sample in "
    "{rec['winner_beat_current_out_of_sample'] * 100}% of trials.":
        "claim:research.overfitting_reading",
    "Drop {html.escape(label(wo['dropped']))} from the set and the PBO over every year is "
    "{wo['pbo']} , {_pbo_words(wo['pbo'])}.":
        "claim:research.overfitting_reading",
    "Its lead over the current weighting was {num(pre['mean'])} over 2012-2019 and "
    "{num(hold['mean'])} over the held-out months":
        "claim:research.overfitting_reading",
    "split the history into equal blocks of months, and for every way of picking half of them "
    "( {a['splits']} ways here, from {a['blocks']} blocks), take the weighting that did best on "
    "that half and see where it ranks on the other half.":
        "checked:tests/test_overfitting.py::test_the_page_prints_the_ledger_s_reading_all_three_ways",
    "This uses the current weighting and the {a['configurations'] - 1} alternatives above, each "
    "scored month by month against the current one on the stocks both ranked, and reruns nothing.":
        "claim:research.overfitting_reading",

    # --- the trading rule ---------------------------------------------------
    "In {lab} it returned {abs(st['mean']) * 100}% a month {word} that average "
    "(t {num(st['t'], 2)}),":
        "checked:tests/test_research_page.py::"
        "test_the_trading_rules_monthly_figure_is_the_ledgers_own_number",
    "{yc(ys[0])}% of each month's members in {ys[0]}, rising to {yc(ys[-1])}% in {ys[-1]}, so "
    "the average leans toward the companies that survived.":
        "checked:tests/test_research_page.py::"
        "test_the_survivorship_coverage_percentages_are_the_two_years_means",
    "The same rule with a {sev['rule']['stop_loss_pct']}% stop-loss":
        "reported:tests/test_research_page.py::"
        "test_the_stop_loss_arm_is_shown_beside_the_rule_with_a_verdict_that_can_say_it_failed",
    "a holding whose adjusted close ends a day {sev['rule']['stop_loss_pct']}% or more below "
    "its price at the start of the month is closed out at that close, and its slot stays in "
    "cash until the next monthly rebalance.":
        "reported:tests/test_research_page.py::"
        "test_the_stop_loss_arm_is_shown_beside_the_rule_with_a_verdict_that_can_say_it_failed",
    "The paper waits a day and exits at a limit a further {sev['rule']['stop_loss_pct'] / 2}% "
    "down, where this exits at the triggering close.":
        "reported:tests/test_research_page.py::"
        "test_the_stop_loss_arm_is_shown_beside_the_rule_with_a_verdict_that_can_say_it_failed",
}

# The sentences above whose arithmetic nothing pins - twelve on 2026-10-29, ten
# after 2026-10-31 06:00 (the trading rule's monthly figure and its two coverage
# percentages), five after 2026-10-31 13:00 (the whole "best backtest" section,
# `claim:research.overfitting_reading`), three after 2026-11-02 evening (the pass
# bar printed twice, `claim:research.pass_bar`) and one after it (the Size sentence,
# `claim:research.size_sentence`), and none after the chart's geometry
# (`claim:research.chart_geometry`) the same evening - by the slug in their verdict.
# **Literals, not a comprehension over the table** -
# `frozenset(s for s, v in ... if v.startswith("open:"))` is the table compared
# with itself and passes at any value (`plan/claims-mutation-audit.md` row 59).
# This set only ever shrinks: closing one means deleting its slug and giving the
# sentence a claim. A session that has to add one says why in `NIGHTLY_LOG.md`.
KNOWN_OPEN: frozenset[str] = frozenset()


def test_every_computed_sentence_on_the_research_page_has_a_verdict():
    """A new or reworded computed-number sentence fails until it is classified.

    This is `test_every_sentence_builder_is_registered` for a page with no naming
    convention. It does not assert the numbers are right - the verdicts say what
    does - it asserts that none of them is *invisible*, which is the state all
    three of 2026-10-29's registrations were in.
    """
    found = computed_sentences(ROOT / "research_page.py")
    assert len(found) >= 25, (
        f"the sweep found only {len(found)} computed-number sentences in "
        "research_page.py - it found 29 on 2026-10-29. A sweep that stops matching "
        "passes vacuously; fix `computed_sentences`, do not trim the table."
    )
    assert len(set(found)) == len(found), "two identical sentence keys - make the key finer"

    new = [s for s in found if s not in RESEARCH_PAGE_SENTENCES]
    assert not new, (
        "these sentences put a number this repo computed into prose on research.html "
        "and have no verdict in RESEARCH_PAGE_SENTENCES. Add one - claim:, checked:, "
        "reported: or open: - after reading what actually pins the figure. If you only "
        f"reworded an existing sentence, move its line.\n\n" + "\n\n".join(new)
    )
    stale = [s for s in RESEARCH_PAGE_SENTENCES if s not in found]
    assert not stale, (
        "RESEARCH_PAGE_SENTENCES has verdicts for sentences the page no longer "
        f"prints - delete them:\n\n" + "\n\n".join(stale)
    )


def test_no_sentence_is_quietly_filed_as_unchecked():
    """`open:` is a dated list, not a bucket. A sentence born unchecked has to be
    added to KNOWN_OPEN in its own commit, where the diff shows it.
    """
    bad = {v for v in RESEARCH_PAGE_SENTENCES.values()
           if v.split(":")[0] not in {"claim", "checked", "reported", "open"}}
    assert not bad, f"unknown verdict kind: {sorted(bad)}"
    opened = {v.split(":")[1] for v in RESEARCH_PAGE_SENTENCES.values()
              if v.startswith("open:")}
    assert len(opened) == sum(1 for v in RESEARCH_PAGE_SENTENCES.values()
                              if v.startswith("open:")), "two open rows share a slug"
    assert opened == KNOWN_OPEN, (
        "the set of unchecked computed sentences has changed. Closing one: delete its "
        "slug from KNOWN_OPEN and give the sentence a claim. Opening one: say in "
        f"NIGHTLY_LOG.md why it shipped unchecked.\nnewly open: {sorted(opened - KNOWN_OPEN)}\n"
        f"no longer open: {sorted(KNOWN_OPEN - opened)}"
    )


def test_every_verdict_names_a_test_or_claim_that_exists():
    """The 2026-10-06 lesson, applied to the verdicts: eleven of the register's
    first twenty-six `checked_by` references named tests that did not exist."""
    problems = []
    ids = {c.id for c in claims.CLAIMS}
    cache: dict[str, set[str] | None] = {}
    for sentence, verdict in {**RESEARCH_PAGE_SENTENCES, **OVERVIEW_SENTENCES,
                              **DASHBOARD_SENTENCES}.items():
        kind, _, ref = verdict.partition(":")
        if kind == "claim":
            if ref not in ids:
                problems.append(f"{ref} is not a claim id ({sentence[:60]!r})")
        elif kind in ("checked", "reported"):
            filename, _, func = ref.partition("::")
            if filename not in cache:
                path = ROOT / filename
                cache[filename] = _test_functions(path) if path.exists() else None
            names = cache[filename]
            if names is None:
                problems.append(f"{filename} does not exist ({sentence[:60]!r})")
            elif func not in names:
                problems.append(f"{filename} has no {func} ({sentence[:60]!r})")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 6. the same sweep, for the public methodology page
# ---------------------------------------------------------------------------
# `SCREENER_OVERVIEW.md` is the second generated surface (the third,
# `generate_dashboard.py`, is the next session's - `plan/claims-mutation-audit.md`
# takes one per session). The yield is smaller than the research page's because
# most of this page's figures are config constants printed verbatim, which the
# scope rule calls *reported* rather than computed and the overview-claims modules
# already pin against config.
#
# Two things about markdown, both found by this sweep's own output being wrong and
# both recorded in `computed_sentences`: a newline **is** a block boundary here
# where in html it is whitespace, and a table row is not a sentence. Before the
# first fix one key ran from a three-row table through the sentence after it;
# before the second, the stat-card merge chained the table's separator and its
# three rows into a single "sentence".
#
# Verdict kinds and the one-unpinned-figure rule are the same as section 5's.
OVERVIEW_SENTENCES: dict[str, str] = {
    # --- the bank P/B-ROE pairing (2026-10-21) ------------------------------
    # The three *table* figures are pinned, cell by cell, against an independent
    # recomputation from config
    # (test_bank_pb_roe_identity.py::test_the_first_order_arithmetic_the_page_prints_comes_from_the_config
    # parses each net-points row and compares it with its own `first_order_points`).
    # The prose below restates them, and nothing reads the prose figures.
    # The three *prose* rows CLOSED 2026-10-31 (13:00; audit rows 144-146). The paragraph
    # restates the table's three figures as four more of its own, and nothing read any of them;
    # the residual clause was blinder still, because today's weights print the neutral wording,
    # so that branch was on a page no test had ever generated.
    "So the {_bp('pb_ratio') + _bp('earnings_yield') + _bp('roe') + _bp('roa') + _bp('equity_ratio')} points those five metrics carry rank a bank on **its ROE relative to its price-to-book** - which, by the identity above, is its earnings yield - and on how much equity backs each dollar of assets, with {_bank_pb_clause}.":
        "claim:overview.bank_pb_roe_identity",
    "P/B's {_bp('pb_ratio')} points are offset by ROE's {_bp('roe')} and ROA's {_bp('roa')}, because a bank that earns well on its book trades at a higher multiple of it.":
        "claim:overview.bank_pb_roe_identity",
    "a deliberate net **{abs(bank_net_pb)} points {'against' if bank_net_pb < 0 else 'on'} price-to-book**":
        "claim:overview.bank_pb_roe_identity",

    # --- the data loop's input gates (2026-10-22) ---------------------------
    # Every one of the ten figures is asserted against the gate module's own
    # constants, on the *generated* page. That pins the claim this sentence makes -
    # "the page states the thresholds the gate applies" - which is page-to-code
    # agreement; the values themselves are pinned by the replay in the same module.
    'Since 2026-10-22 it also refuses a run that ranks fewer than {_crh.MIN_SCORED} stocks or leaves more than {_crh.MAX_EXCLUDED_SHARE} of the universe unscored for missing data; one where fewer than {_crh.MIN_PRICE_ON_SESSION} of prices are dated on the latest NYSE session, or more than {_crh.MAX_UNCHANGED_ACROSS_SESSION} are unchanged from the last published run after a session has closed; and one whose ranking breaks from the last published one - a rank correlation below {_crh.RANK_MIN_SPEARMAN}, more than {_crh.RANK_MAX_NEW_IN_TOP} new names in the top {_crh.RANK_TOP_N}, or more than {_crh.RANK_MAX_FALLEN} of the top {_crh.RANK_TOP_N} falling below #{_crh.RANK_FALL_LINE} - unless the methodology version changed and the methodology changelog records the change.':
        "checked:tests/test_input_gates.py::test_the_methodology_page_states_the_thresholds_the_gate_applies",

    # --- the glossary's worked example -------------------------------------
    # Both figures: the weight from config and 60 x w / 100 recomputed in the test.
    'A Valuation score of 60 at a {_pts_example_w}% weight adds {_pts_example} points.':
        "checked:tests/test_overview_reading_order.py::test_the_glossary_points_example_and_rank_are_the_engines",

    # --- the momentum panic rule (2026-10-09) ------------------------------
    # 24, 126, 20.6% and 0.70 are literals in the test, and it re-renders the page
    # from a different config to prove the page follows it rather than agreeing by
    # coincidence - which is the strongest form in this file.
    "the index's return over the past {mr_months} months is negative, and its realised volatility over the past {mr_days} trading days is above {mr_thr}% a year (the 80th percentile of that figure over 1988-2026, fixed in `config.yaml` rather than re-estimated).":
        "checked:tests/test_momentum_panic_rule.py::test_the_overview_states_the_rule_the_engine_applies",
    "When they do, Momentum's weight is multiplied by {mr_scale} for that run and the freed weight is split equally between Quality and Valuation.":
        "checked:tests/test_momentum_panic_rule.py::test_the_overview_states_the_rule_the_engine_applies",

    # --- Data quality safeguards -------------------------------------------
    # CLOSED 2026-10-31 (rows 131-132). `..._are_the_ones_applied` said in its own
    # docstring that "the metric rows hard-code their clamps; the Data quality
    # bullet reads config. Both must be config's" - and then matched only the
    # metric-table rows (`^\| \*\*Forward EPS Growth\*\* \|`), so this bullet, the
    # second printing of all four figures, was read by nothing. It now reads the
    # bullet, pins all four against config, and asserts the two printings agree.
    "- **Output clamping (configurable):** Forward EPS growth is clamped to [{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[0] * 100)}%, +{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[1] * 100)}%]; price target upside is clamped to [{int(clamps.get('price_target_upside', [-0.5, 1.0])[0] * 100)}%, +{int(clamps.get('price_target_upside', [-0.5, 1.0])[1] * 100)}%].":
        "claim:overview.clamps_and_floors",

    # CLOSED 2026-10-31 (13:00; audit row 147), and the row's own note was half wrong, which is
    # worth recording: `test_the_discount_example_is_a_coverage_a_ranked_stock_can_have` captures
    # **seven** figures from this sentence and pins four of them (the loss, the deepest cut and
    # the pair it is quoted at). Three were read and discarded - `_ex_k`, `_n_gen`, and the
    # coverage percentage between them, which was bound to a regex group and never asserted at
    # all - so the example could have quoted any of the eight coverages the filter allows and
    # its discount would have been recomputed from whichever one it printed.
    "Below {int(cov_disc.get('threshold', 0.8) * 100)}% metric coverage, the composite is multiplied by 1 − ({int(cov_disc.get('threshold', 0.8) * 100)}% − coverage) × {int(cov_disc.get('penalty_rate', 0.15) * 100)}% (e.g., a stock with {_ex_k} of {_n_gen} metrics, {_cov_pct(_ex_k, _n_gen)}% coverage, loses {_ex_disc}% of its composite; the deepest cut the coverage filter leaves possible is {_max_disc}%, at {_bars[_floor_n]} of {_floor_n}).":
        "claim:overview.coverage_discount",

    # Until 2026-11-02 `test_confidence_metric_count_is_the_discount_coverage` pinned
    # two of the three counts as bare substrings anywhere on the page and the
    # other-financials count not at all. `test_the_coverage_sentence_counts_what_the_
    # engine_counts` reads all three out of this sentence and compares each with the
    # engine's own denominator for that kind of stock (audit rows 174-177).
    '**"Coverage" here means the share of the metrics that carry weight in that stock\'s score** - {len(_w_bank)} for a bank-like stock and {len(_w_gen)} for every other ({len(_w_gen) - 1} for other financials, which are never given the Beneish score); a metric with no weight (a candidate, or one shown for reference) cannot lower it (since 2026-10-09).':
        "claim:overview.coverage_counts",

    # --- share classes (2026-10-21) ----------------------------------------
    # CLOSED 2026-10-31 (rows 133-134). The only page check was
    # `test_share_classes_page.py::test_the_page_states_the_switching_margin_from_config`,
    # which looks for `f"{margin:.0%}"` - the config value formatted - anywhere in a
    # 2,200-character window: row 59's defect (a check may not import the number it
    # is checking) on a loose substring. `TestThePagePrintsTheMargin` now reads the
    # figure out of the sentence itself and, with
    # `TestDeclaration::test_score_once_is_on_and_the_margin_is_spdji_s`, pins it to
    # a literal 0.20 at both ends, plus the config-follows, fallback and `else` cases.
    "The choice of line is declared in `config.yaml` rather than re-picked from each run's trading volume, and a run only reports when the other line leads by more than {_sc_margin} (S&P DJI's own switching margin):":
        "claim:overview.share_classes",
}

# The one open slug above. Seven on 2026-10-30; five after 2026-10-31 13:00 moved the clamp
# bullet and the switching margin (whose checks shipped at 06:00 and whose verdicts here did
# not), and one after the same session closed the three bank P/B-ROE prose rows and the
# coverage-discount worked example. Literals, not a comprehension over the table - see
# KNOWN_OPEN. This set only ever shrinks. Empty since 2026-11-02 evening (the coverage
# counts, `claim:overview.coverage_counts`, audit rows 174-177).
OVERVIEW_KNOWN_OPEN: frozenset[str] = frozenset()


def test_every_computed_sentence_on_the_methodology_page_has_a_verdict():
    """Section 5's sweep, for `build_screener_overview`.

    The scope is the generator, not `run_screener.py` - the file is the pipeline,
    and a file-wide sweep would enumerate progress lines as published sentences.
    """
    found = computed_sentences(ROOT / "run_screener.py",
                               within="build_screener_overview", markup="markdown")
    assert len(found) >= 9, (
        f"the sweep found only {len(found)} computed-number sentences in "
        "build_screener_overview - it found 11 on 2026-10-30. A sweep that stops "
        "matching passes vacuously; fix `computed_sentences`, do not trim the table."
    )
    assert len(set(found)) == len(found), "two identical sentence keys - make the key finer"

    new = [s for s in found if s not in OVERVIEW_SENTENCES]
    assert not new, (
        "these sentences put a number this repo computed into prose on "
        "SCREENER_OVERVIEW.md and have no verdict in OVERVIEW_SENTENCES. Add one - "
        "claim:, checked:, reported: or open: - after reading what actually pins the "
        "figure. If you only reworded an existing sentence, move its line.\n\n"
        + "\n\n".join(new)
    )
    stale = [s for s in OVERVIEW_SENTENCES if s not in found]
    assert not stale, (
        "OVERVIEW_SENTENCES has verdicts for sentences the page no longer prints - "
        "delete them:\n\n" + "\n\n".join(stale)
    )


def test_no_methodology_sentence_is_quietly_filed_as_unchecked():
    """`open:` is a dated list here too, and the ratchet only turns one way."""
    bad = {v for v in OVERVIEW_SENTENCES.values()
           if v.split(":")[0] not in {"claim", "checked", "reported", "open"}}
    assert not bad, f"unknown verdict kind: {sorted(bad)}"
    opened = {v.split(":")[1] for v in OVERVIEW_SENTENCES.values() if v.startswith("open:")}
    assert len(opened) == sum(1 for v in OVERVIEW_SENTENCES.values()
                              if v.startswith("open:")), "two open rows share a slug"
    assert opened == OVERVIEW_KNOWN_OPEN, (
        "the set of unchecked computed sentences on the methodology page has changed. "
        "Closing one: delete its slug from OVERVIEW_KNOWN_OPEN and give the sentence a "
        "claim. Opening one: say in NIGHTLY_LOG.md why it shipped unchecked.\n"
        f"newly open: {sorted(opened - OVERVIEW_KNOWN_OPEN)}\n"
        f"no longer open: {sorted(OVERVIEW_KNOWN_OPEN - opened)}"
    )


def test_the_markdown_sweep_breaks_on_blocks_and_skips_table_rows(tmp_path):
    """The two markdown fixes, as a negative control.

    Both were real defects in this sweep's first run, and both produced a key that
    *looked* like coverage: a sentence glued to the table above it, and a table
    read as a sentence. A later session tidying `computed_sentences` would not
    otherwise find out.
    """
    src = tmp_path / "g.py"
    src.write_text(
        "def build():\n"
        "    n = 3\n"
        "    return (\n"
        "        f'| Coordinate | Net points |\\n|---|---|\\n'\n"
        "        f'| Earnings yield (E/P) | **+{n:.1f}** |\\n\\n'\n"
        "        f'So the {n * 2} points those metrics carry rank a bank on its own book.\\n'\n"
        "    )\n", encoding="utf-8")
    got = computed_sentences(src, within="build", markup="markdown")
    assert got == ["So the {n * 2} points those metrics carry rank a bank on its own book."], got
    # in html the same source is one unit, because a newline there is whitespace -
    # which is why the surfaces take different `markup`
    assert len(computed_sentences(src, within="build", markup="html")) == 1
    assert "Coordinate" in computed_sentences(src, within="build", markup="html")[0]


def test_the_sweeps_scope_is_the_generator_not_the_file():
    """`within` must fail loudly if the generator is renamed, rather than silently
    sweeping the whole pipeline (or nothing)."""
    with pytest.raises(AssertionError, match="no function build_screener_overview_renamed"):
        computed_sentences(ROOT / "run_screener.py",
                           within="build_screener_overview_renamed", markup="markdown")
    whole_file = computed_sentences(ROOT / "run_screener.py", markup="markdown")
    scoped = computed_sentences(ROOT / "run_screener.py",
                                within="build_screener_overview", markup="markdown")
    assert len(whole_file) > len(scoped), (
        "scoping to the generator should be narrower than the whole pipeline; if it "
        "is not, `within` is not doing anything"
    )


# ---------------------------------------------------------------------------
# 7. the same sweep, for the dashboard itself
# ---------------------------------------------------------------------------
# `generate_dashboard.py` is the third generated surface and the one a reader
# opens first. Its sweep cannot be section 5's: the Python f-strings in that file
# hold one log line and **no** published computed figure - every sentence on the
# page with a number in it is assembled in the browser by JavaScript the
# generator emits. A Python-only sweep of the file reports one sentence (the log
# line) and would pass with every JavaScript sentence unregistered.
# `tests/js_sweep.py` reads the emitted JavaScript - template literals and `+`
# chains - into the same text-with-holes shape, and `_units` splits both kinds of
# source by the same rules.
#
# **What is swept is the generator's output** (`generate_html()`'s inline
# scripts), never the committed `index.html` - rule 10, and the 2026-10-28 lesson
# that a committed artifact goes green the day it is regenerated.
#
# Scope additions, beyond section 5's rule, that the JavaScript needed:
# - a figure is a hole that calls one of the page's number formatters
#   (`js_sweep.NUMBER_HELPERS`), `toFixed`/`toLocaleString`, `Math.*` or
#   `.length`, or does arithmetic - the counterpart of a numeric format spec. A
#   ternary's **condition** is not printed, so `x.length ? ... : ''` is not a
#   figure because of its `.length`;
# - a hole that inserts markup (a literal containing a tag) ends the unit like a
#   block-level tag. Its own text is a separate entry, so nothing is lost; without
#   the rule a `.map(...)` of table rows became a "figure" glued to the caption
#   after it;
# - a "word" for the prose test has at least two characters, because an SVG path
#   (`M ... L ... Z`) has three one-letter "words";
# - text inside a tag - a `title=` tooltip, a `style=` width - is outside the
#   sweep, exactly as on the other two surfaces.
_JS_PROSE = re.compile(r"[A-Za-z][\w'&;-]+(?:\s+[A-Za-z][\w'&;-]+){2,}")


def _dashboard_scripts() -> list[str]:
    """The inline JavaScript of the page the next run will publish."""
    import generate_dashboard
    page = generate_dashboard.generate_html()
    return [m.group(2) for m in re.finditer(r"<script(\s[^>]*)?>(.*?)</script>", page, re.S)
            if "src=" not in (m.group(1) or "")]


def js_computed_sentences(scripts: list[str]) -> list[str]:
    from tests import js_sweep
    out: list[str] = []
    for src in scripts:
        assigned = js_sweep.assignments(src)
        for at, parts in js_sweep.string_exprs(src):
            joined, holes = [], []
            for kind, v in parts:
                if kind == "text":
                    joined.append(v.replace("\x00", ""))
                else:
                    joined.append(f"\x00{len(holes)}\x00")
                    holes.append(v)
            out.extend(_units("".join(joined), "html",
                              figure=[js_sweep.is_figure(h, assigned, at) for h in holes],
                              shown=[js_sweep.key_expr(h) for h in holes],
                              block=[js_sweep.is_markup(h) for h in holes],
                              brace="${", prose=_JS_PROSE))
    return out


def dashboard_sentences() -> list[str]:
    """Every computed-number sentence the dashboard generator publishes: its
    Python f-strings (log lines excluded) and the JavaScript it emits."""
    return (computed_sentences(ROOT / "generate_dashboard.py", skip_calls=frozenset({"print"}))
            + js_computed_sentences(_dashboard_scripts()))


# Every computed-number sentence the dashboard publishes, swept 2026-11-02 (owner-run night;
# `plan/claims-mutation-audit.md` rows 178+). Verdict kinds as section 5's, with one difference
# in form: an `open:` verdict is `open:<slug> - <what is unpinned>`, so the note travels with the
# row. Every verdict was assigned by reading the check it names - and, for the dashboard, one
# question decides most of them: does any registered test read **the page's printing** of the
# figure? Most dashboard claims pin the engine's number or the payload's (row 0.8's whole design),
# and the JavaScript that formats it into a sentence is a second step nothing ran. A claim whose
# checks stop at the payload does not cover the sentence; those rows say so.
#
# The sweep finds one sentence key at **two** sites (", N% on pre-set plans", printed by the
# Insider card and by the case against), so keys are compared as a set; its verdict has to hold
# for both printings, and says which one is pinned.
DASHBOARD_SENTENCES: dict[str, str] = {
    "${fw.momentum > bw.momentum ? 'raised' : 'cut'} from ${fmtWeight(bw.momentum)} by the momentum regime rule this run":
        'claim:overview.momentum_regime',
    'value / growth &middot; ${pct(k.value_traps)} and ${pct(k.growth_traps || 0)} of stocks What a flag means':
        'claim:dashboard.trap_flag_key',
    '${top}, ${Number(vt[top].rate).toFixed(1)}% of the sector':
        'claim:dashboard.trap_flag_key',
    "The median this run is ${k.median_composite.toFixed(1)}, so a composite above it is in the upper half of the ${(k.universe_size || '')} stocks.":
        'claim:dashboard.score_key',
    'Show the next ${next}':
        "open:show-next-count - the table's next-page count; nothing reads the page's printing",
    'The ranking covers the ${n} S&amp;P 500 companies this run scored.':
        'claim:dashboard.score_key',
    "It trades ${pctNearZero(sc.price_gap_pct, 1).mag}% ${(sc.price_gap_pct < 0 ? 'below ' : 'above ')}${escapeHtml(sc.scored_as)} for the same claim on the same earnings.":
        'open:share-class-price-gap - drilldown.share_class_second_line checks the ratio identity, not the printed gap',
    ', the 95th percentile of ordinary run-to-run variation measured across ${n.n_pairs} paired runs (${n.n_observations.toLocaleString()} observations).':
        "open:noise-band-pairs - the paired-run and observation counts; nothing reads the page's printing",
    '${mv.n_up} stocks moved up materially and ${mv.n_down} moved down; showing the largest ${shownUp} and ${shownDown}.':
        'claim:history.run_overview',
    '${mv.n_round_trip} of ${mv.n_up + mv.n_down} are flagged round-trip:':
        'claim:history.run_overview',
    'Built from ${H.dates.length} comparable runs, ${escapeHtml(prettyDate(H.dates[0]))} to ${escapeHtml(prettyDate(H.dates[H.dates.length - 1]))}.':
        "open:rank-history-built-from - the run count; nothing reads the page's printing",
    'Remove all ${holdings.length} names from this list?':
        'claim:dashboard.holdings_fit',
    'on this screener’s own history, acting on the top ${(c.num_stocks || 25)} at every run implies ${c.turnover_every_run.toFixed(0)}% monthly one-sided turnover against ${c.turnover_monthly.toFixed(0)}% reviewing the same rule monthly.':
        'open:cadence-turnover - the three turnover figures; no registered check reads them',
    'Novy-Marx &amp; Velikov ( Review of Financial Studies 29(1), 2016) find anomalies under roughly ${c.turnover_ceiling.toFixed(0)}% mostly survive trading costs, and few above it do.':
        "open:cadence-turnover-ceiling - nothing reads the page's printing",
    "${n} name${(n === 1 ? '' : 's')} across ${sectors.length} sector${(sectors.length === 1 ? '' : 's')}":
        'claim:dashboard.holdings_fit',
    '${inTop25} inside the top 25 of ${universe}, ${inTop100} inside the top 100':
        'claim:dashboard.holdings_fit',
    "${n} name${(n === 1 ? '' : 's')} - ${standing} of the published counts for a diversified portfolio:":
        'claim:dashboard.holdings_fit',
    'An equal split across ${n} names is ${(100 / n).toFixed(1)}% a position.':
        'claim:dashboard.holdings_concentration',
    '${escapeHtml(hi.t)} at ${(hi.v * 100).toFixed(0)}% annualised volatility against ${escapeHtml(lo.t)} at ${(lo.v * 100).toFixed(0)}% , a ${ratio.toFixed(1)}&times; spread.':
        'claim:dashboard.holdings_concentration',
    'Held in equal dollar amounts, ${escapeHtml(hi.t)} moves the value of the list about ${ratio.toFixed(1)}&times; as much as ${escapeHtml(lo.t)}.':
        'claim:dashboard.holdings_concentration',
    'For the ${n} of ${total} metrics marked ↓ better (EV/EBITDA, Beta and PEG among them) a low raw value earns a high percentile.':
        'claim:drilldown.percentile_convention',
    '${fmtDollar(price)} Avg Target ${fmtDollar(ptMean)} ${fmtPct(ptMean, price)} Low Target':
        "open:price-target-strip - nothing reads the page's printing",
    '${fmtPctChg(f.ni_growth_yoy)} fiscal year on year':
        "open:ni-growth-yoy - nothing reads the page's printing",
    '${f.short_ratio.toFixed(1)} days to cover':
        "open:short-ratio - nothing reads the page's printing",
    '${CAT_LABELS[c]} no data Not scored for this stock - its ${fmtWeight(base[c] || 0)} was shared out across the categories below.':
        'claim:drilldown.unscored_category',
    'Category points add up to ${runningTotal.toFixed(1)}':
        'claim:drilldown.composite_chain',
    'Coverage discount - values for ${cov.n} of ${cov.of} applicable metrics (${covPct}%), below the ${cd ? Math.round(cd.threshold * 100) : 80}% threshold, so the composite is reduced by ${(disc * 100).toFixed(1)}% &minus;${(runningTotal - composite).toFixed(1)}':
        'claim:drilldown.composite_chain',
    'none of its ${n} metrics has data':
        'claim:drilldown.unscored_category',
    'None of the ${n} metrics this category weights for this stock has data in this run (${listed(names)})${shared}':
        'claim:drilldown.unscored_category',
    'Configured ${fmtWeight(r.configured)} of this weighting; ${fmtWeight(r.share)} of this score after rescaling to the metrics that have data.':
        'claim:drilldown.metric_weights',
    'Configured ${fmtWeight(r.configured)}; no data for this stock, so it carries no weight here.':
        'claim:drilldown.metric_weights',
    "Ranked ${ordinal(pc0.rank)} of ${pc0.n} in ${where}, ${pc0.lower ? 'lower' : 'higher'} being better":
        'claim:drilldown.peer_rank',
    '; sector median ${fmtMetric(pc0.st[2], meta.fmt)} , middle half ${fmtMetric(pc0.st[1], meta.fmt)} to ${fmtMetric(pc0.st[3], meta.fmt)}':
        'claim:drilldown.peer_rank',
    'That is the ${ordinal(Math.round(pctv0))} percentile.':
        'claim:drilldown.peer_rank',
    '${r.pct.toFixed(2)} percentile &times; ${fmtWeight(r.share)} weight = ${r.points.toFixed(2)} points toward the category score':
        'claim:drilldown.metric_weights',
    "${sig.filter(c => c === '1').length} passed of ${sig.filter(c => c !== '-').length} testable = score ${fmtMetric(s.raw[m], 'int')}":
        'claim:drilldown.piotroski_beneish_parts',
    'The last ${q.length} quarters, oldest first':
        'claim:drilldown.row_equation',
    "Median of the ${vs.length} valid surprises (${vs.map(sp).join(', ')}) = ${fmtMetric(s.raw[m], meta.fmt)}":
        'claim:drilldown.row_equation',
    'Latest surprise ${sp(q[L - 1][3])} &minus; the quarter before ${sp(q[L - 2][3])} = ${fmtMetric(s.raw[m], meta.fmt)}':
        'claim:drilldown.row_equation',
    "${(beatW.length ? `...` : 'No quarter with data beat its estimate')} of ${allW.join(' + ')} with data, &times; 10 = ${fmtMetric(s.raw[m], meta.fmt)}":
        'claim:drilldown.row_equation',
    'Sample standard deviation of the five ROEs = ${fmtMetric(s.raw[m], meta.fmt)}':
        'claim:drilldown.row_equation',
    'the same ${fmtMetric(s.raw[m], meta.fmt)} that was scored.':
        'claim:drilldown.row_equation',
    "${sig.filter(c => c === '1').length} of ${sig.filter(c => c !== '-').length} testable signals passed = ${res}":
        'claim:drilldown.piotroski_beneish_parts',
    "${escapeHtml(a)} scores ${Math.abs(gap).toFixed(1)} composite points ${(gap >= 0 ? 'above' : 'below')} ${escapeHtml(b)} ${(lead.length ? '. The largest differences: ' + lead.join(' and ') + '.' : '.')}":
        "open:compare-gap-sentence - compare.composite_gap's page check reads the gap rows, not this sentence",
    '${vhChartSvg(y, m0, label)} ${label} ${cPlain(y.now)} of its past ${y.n} month-ends':
        'claim:context.valuation_history',
    'higher than at ${above}%':
        'claim:context.valuation_history',
    "From 52-week high ${cPct(c.px / c.hi - 1)} high ${cPrice(c.hi)}${(c.hid ? ' on ' + escapeHtml(cDate(c.hid)) : '')}, low ${cPrice(c.lo)}":
        'claim:context.price_trend',
    '&plusmn;${cPlain(c.om)} The average move options are paying for by ${escapeHtml(cDate(c.ox))} (${c.od} days)':
        'claim:context.expected_move',
    'Expected move = at-the-money call + put, divided by ${cPrice(c.osp)}, the stock&rsquo;s price when the quotes were taken:':
        'claim:context.expected_move',
    "Open-market buys, 90 days ${(ins.buy_n ? cMoney(ins.buy_value) : 'None')} ${(ins.buy_n ? ins.buy_people + ' insider' + (ins.buy_people === 1 ? '' : 's') + ', ' + ins.buy_n + ' trade' + (ins.buy_n === 1 ? '' : 's') : '&nbsp;')} Open-market sales, 90 days ${(ins.sell_n ? cMoney(ins.sell_value) : 'None')} ${(ins.sell_n ? ins.sell_people + ' insider' + (ins.sell_people === 1 ? '' : 's') + (ins.sell_planned_value != null && ins.sell_value > 0 ? ', ' + Math.round(100 * ins.sell_planned_value / ins.sell_value) + '% on pre-set plans' : '') : '&nbsp;')}":
        'claim:context.insider_trades',
    ', ${Math.round(100 * ins.sell_planned_value / ins.sell_value)}% on pre-set plans':
        'claim:context.insider_plan',
    "${cPct(c.rb * RATE_STEP, 2)} How the stock has typically moved for each +0.10 percentage point rise in the 10-year Treasury yield (a large day for the 10-year), from the past ${Math.round(c.rn / 21)} months of daily moves${(sr ? '; ' + escapeHtml(s.sector) + ' median ' + cPct(sr.median * RATE_STEP, 2) : '')}.":
        'claim:context.rate_sensitivity',
    "Rate moves explained ${cPlain(c.rr2, 0)} of its daily swings${(c.rr2 < 0.05 ? ', very little, so read the figure loosely' : '')}.":
        'claim:context.rate_sensitivity',
    'a coin toss does at least this well ${chance}% of the time':
        'claim:context.track_luck',
    'one way, on average; ${cPlain(TO.min, 0)} to ${cPlain(TO.max, 0)}':
        'claim:context.track_turnover',
    'Every holding period (${T.periods.length})':
        'claim:context.track_record',
    "${(T.unpriced[k].length)} of the ${picks[k]} picks in the ${({ top: 'top-25', q1: 'top-fifth', q5: 'bottom-fifth' })[k]} baskets":
        'claim:context.track_record',
    "${(X && X.t !== null && X.t !== undefined ? 'Measured day by day against RSP over ' + X.days + ' trading days, the top 25&rsquo;s lead is ' + Number(X.t).toFixed(1) + ' standard errors from zero; two is the usual minimum for telling skill from luck' + (X.years_to_two ? ', and even if the lead went on at the same pace it would take about ' + Number(X.years_to_two).toFixed(1) + ' years of record to get there' : '') + '. ' : '')}The methodology changelog has entries on ${(T.methodology_changes || []).length} dates inside this window (the ticks under the chart); not every entry changed how stocks are scored.":
        'claim:context.track_record',
    "${(TO ? 'No trading costs or taxes are deducted. At each rebalance the top 25 traded ' + cPlain(TO.mean, 0) + ' of its value on average, one way' + (TO.break_even_cost !== null && TO.break_even_cost !== undefined ? '; a cost of ' + cPlain(TO.break_even_cost, 2) + ' on every dollar traded would have erased its lead over RSP' : '') + '. ' : 'No trading costs or taxes are deducted. ')}${((T.days || 0) < 365 ? 'No return here is annualised, because none covers a full year. ' : '')}This record is not used to set the methodology.":
        'claim:context.track_turnover',
    "Measured day by day against RSP over ${X.days} trading days, the top 25&rsquo;s lead is ${Number(X.t).toFixed(1)} standard errors from zero; two is the usual minimum for telling skill from luck${(X.years_to_two ? ', and even if the lead went on at the same pace it would take about ' + Number(X.years_to_two).toFixed(1) + ' years of record to get there' : '')}.":
        'claim:context.track_luck',
    ', and even if the lead went on at the same pace it would take about ${Number(X.years_to_two).toFixed(1)} years of record to get there':
        'claim:context.track_luck',
    "At each rebalance the top 25 traded ${cPlain(TO.mean, 0)} of its value on average, one way${(TO.break_even_cost !== null && TO.break_even_cost !== undefined ? '; a cost of ' + cPlain(TO.break_even_cost, 2) + ' on every dollar traded would have erased its lead over RSP' : '')}.":
        'claim:context.track_turnover',
    '; a cost of ${cPlain(TO.break_even_cost, 2)} on every dollar traded would have erased its lead over RSP':
        'claim:context.track_turnover',
    '${a} report within 7 days of this run · ${b} within 14':
        "open:reporting-soon-counts - nothing reads the page's printing",
    '${Math.abs(X.t).toFixed(1)} standard errors, where two is the usual bar':
        'claim:context.track_luck',
    '${n} companies, one dot each':
        'claim:drilldown.composite_distribution',
    "${n} companies by composite score, from ${fmt(Math.min.apply(null, vals), 'score')} to ${fmt(Math.max.apply(null, vals), 'score')}${(typeof med === 'number' ? ', median ' + med.toFixed(1) : '')}.":
        'claim:drilldown.composite_distribution',
    '${mv.n_up} up of ${N} ${mv.n_down} down':
        'claim:history.run_overview',
    '${mv.n_up} of ${N} stocks moved up materially and ${mv.n_down} moved down':
        'claim:history.run_overview',
    '${p[1]} at the ${ord(S[p[0]].pct_10y)} percentile of ten years':
        'claim:context.macro_series',
    '#1 Where they rank #${N}':
        "open:holdings-rank-strip-axis - nothing reads the page's printing",
    "Your saved names rank ${ranks.sort((a, b) => a - b).map(v => '#' + v).join(', ')} of ${N}":
        "open:holdings-rank-strip - nothing reads the page's printing",
    "Composite ${fmt(score, 'score')} among the ${d.n} published composites${(hasMed ? '; the median is ' + fmt(med, 'score') : '')}":
        'claim:drilldown.composite_distribution',
    "; the median is ${fmt(med, 'score')}":
        'claim:drilldown.composite_distribution',
    "Its last close is −${gap.toFixed(gap < 0.1 ? 2 : 1)}% against its 200-day average of ${cPrice(c.s200)}${(tr ? ' (' + TREND_WORD[tr] + ')' : '')}.":
        'claim:drilldown.case_against',
    "Officers and directors made ${cMoney(ins.sell_value)} of open-market sales in the last 90 days (${ins.sell_people} insider${(ins.sell_people === 1 ? '' : 's')}${plan})${(ins.buy_n ? ', against ' + cMoney(ins.buy_value) + ' of purchases.' : ' and no open-market purchases.')} The panel gives the evidence that sales have historically said less than purchases.":
        'claim:drilldown.case_against',
    ', against ${cMoney(ins.buy_value)} of purchases.':
        'claim:drilldown.case_against',
    "${(m.ret < 0 ? '&minus;' : '+')}${pc(Math.abs(m.ret))} over ${m.return_window_months} months, with realised volatility of ${pc(m.vol)} a year over ${m.vol_window_days} trading days.":
        'claim:overview.momentum_regime',
    'The return is negative and volatility is above ${pc(m.vol_threshold)}, so the market is in the panic state and Momentum&rsquo;s weight is cut for this run.':
        'claim:overview.momentum_regime',
    'The panic state needs a negative return and volatility above ${pc(m.vol_threshold)}, so Momentum&rsquo;s weight is unchanged.':
        'claim:overview.momentum_regime',
    'Rank ${first[1]} on ${escapeHtml(prettyDate(H.dates[first[0]]))} &rarr; ${last[1]} on ${escapeHtml(prettyDate(H.dates[last[0]]))}, across ${pts.length} comparable runs':
        "open:rank-history-span - nothing reads the page's printing",
    "${rankChartSvg(ser.r)} ${catTable(d.prev, 'Since last run', (H.compare.prev || {}).date || '')} ${catTable(d.m1, 'Since ~1 month', (H.compare.m1 || {}).date || '')} Based on ${H.dates.length} comparable runs.":
        "open:rank-history-run-count - nothing reads the page's printing",
    'Its figure was ${fmtBig(evApi)} against the ${fmtBig(evUsed)} used here.':
        'claim:drilldown.ev_substituted',
    'Only ${fl.beta_overlap_pct.toFixed(0)}% of this company&rsquo;s trading days in the last year line up with the S&amp;P 500&rsquo;s, under the 80% the screener requires, so beta is left unscored and the Risk category&rsquo;s weight is shared across the measures that do have data.':
        'claim:drilldown.beta_overlap',
}

DASHBOARD_KNOWN_OPEN: frozenset[str] = frozenset({
    'cadence-turnover',
    'cadence-turnover-ceiling',
    'compare-gap-sentence',
    'holdings-rank-strip',
    'holdings-rank-strip-axis',
    'ni-growth-yoy',
    'noise-band-pairs',
    'price-target-strip',
    'rank-history-built-from',
    'rank-history-run-count',
    'rank-history-span',
    'reporting-soon-counts',
    'share-class-price-gap',
    'short-ratio',
    'show-next-count',
})


def _dashboard_slug(verdict: str) -> str:
    return verdict.partition(":")[2].split(" - ")[0]


def test_every_computed_sentence_on_the_dashboard_has_a_verdict():
    """Section 5's sweep, for the JavaScript `generate_dashboard.py` emits."""
    found = dashboard_sentences()
    assert len(set(found)) >= 75, (
        f"the sweep found only {len(set(found))} distinct computed-number sentences on the "
        "dashboard - it found 84 on 2026-11-02. A sweep that stops matching passes vacuously; fix "
        "`tests/js_sweep.py` or `computed_sentences`, do not trim the table."
    )
    new = [s for s in dict.fromkeys(found) if s not in DASHBOARD_SENTENCES]
    assert not new, (
        "these sentences put a number this repo computed into prose on the dashboard and have no "
        "verdict in DASHBOARD_SENTENCES. Add one - claim:, checked:, reported: or open:<slug> - "
        "<what is unpinned> - after reading what actually pins the figure on the page. If you "
        "only reworded an existing sentence, move its line.\n\n" + "\n\n".join(new)
    )
    stale = [s for s in DASHBOARD_SENTENCES if s not in set(found)]
    assert not stale, ("DASHBOARD_SENTENCES has verdicts for sentences the page no longer "
                       "prints - delete them:\n\n" + "\n\n".join(stale))


def test_no_dashboard_sentence_is_quietly_filed_as_unchecked():
    """`open:` is a dated list here too, and the ratchet only turns one way."""
    bad = {v for v in DASHBOARD_SENTENCES.values()
           if v.split(":")[0] not in {"claim", "checked", "reported", "open"}}
    assert not bad, f"unknown verdict kind: {sorted(bad)}"
    opened = [_dashboard_slug(v) for v in DASHBOARD_SENTENCES.values() if v.startswith("open:")]
    assert all(" - " in v for v in DASHBOARD_SENTENCES.values() if v.startswith("open:")), (
        "an open dashboard row says what is unpinned: open:<slug> - <what>")
    assert len(set(opened)) == len(opened), "two open rows share a slug"
    assert set(opened) == DASHBOARD_KNOWN_OPEN, (
        "the set of unchecked computed sentences on the dashboard has changed. Closing one: "
        "delete its slug from DASHBOARD_KNOWN_OPEN and give the sentence a claim. Opening one: "
        "say in NIGHTLY_LOG.md why it shipped unchecked.\n"
        f"newly open: {sorted(set(opened) - DASHBOARD_KNOWN_OPEN)}\n"
        f"no longer open: {sorted(DASHBOARD_KNOWN_OPEN - set(opened))}"
    )


def test_the_javascript_sweep_reads_chains_and_templates_and_not_conditions():
    """`tests/js_sweep.py`'s rules, as a negative control on a script written here.

    Each line is a case the dashboard's own script has and the first draft got wrong or would:
    a `+` chain, a template, a ternary whose only number is in its condition, a `.map()` of
    table rows ahead of a caption, an SVG path, and a figure computed one line before the
    sentence that prints it - and a ticker reusing that name later, which must not inherit it.
    """
    script = (
        "el.textContent = 'The median this run is ' + k.m.toFixed(1) + ', across ' + n + ' stocks.';\n"
        "x = `Ranked ${ordinal(r)} of ${n} in ${s}`;\n"
        "y = (a.length ? 'see ' + a.join(', ') : '') + ' plain words follow here';\n"
        "z = `<table>${rows.map(r => `<tr><td>${r.v.toFixed(1)}</td></tr>`).join('')}</table><p>Three word caption</p>`;\n"
        "p = 'M' + X(a).toFixed(1) + ' L ' + X(b).toFixed(1) + ' Z';\n"
        "const t = (v * 100).toFixed(1);\n"
        "q = 'It rose ' + t + '% over the last year';\n"
        "const t = row.ticker;\n"
        "w = t + ' added to My Holdings';\n"
    )
    assert js_computed_sentences([script]) == [
        "The median this run is ${k.m.toFixed(1)}, across ${n} stocks.",
        "Ranked ${ordinal(r)} of ${n} in ${s}",
        "It rose ${t}% over the last year",
    ]


def test_every_formatter_the_sweep_meets_is_classified():
    """A number formatter missing from `js_sweep.NUMBER_HELPERS` makes every sentence it formats
    invisible to the sweep - `cPlain`, `cPp`, `fmtRG` and three `mk*` helpers were, in the first
    draft, and eight sentences came back when they were listed. So any helper of the page's
    formatter shapes that a sentence calls must be on one of the two lists."""
    from tests import js_sweep
    unknown = set()
    for src in _dashboard_scripts():
        for _, parts in js_sweep.string_exprs(src):
            for kind, v in parts:
                if kind == "hole":
                    unknown |= {n for n in js_sweep.called_names(v)
                                if js_sweep.FORMATTER_SHAPE.match(n)
                                and n not in js_sweep.NUMBER_HELPERS | js_sweep.NOT_NUMBER_HELPERS}
    assert not unknown, (f"classify these in tests/js_sweep.py as a number or not: {sorted(unknown)}")


def test_the_dashboard_sweep_reads_the_generators_output_not_the_committed_page():
    """Rule 10 for the sweep itself: the scripts swept are `generate_html()`'s, so a sentence
    added to the generator is seen the session it is written, not the morning the page is
    regenerated."""
    import generate_dashboard
    page = generate_dashboard.generate_html()
    scripts = _dashboard_scripts()
    assert scripts and all(s in page for s in scripts)
    assert sum(len(s) for s in scripts) > 200_000, "the page's inline JavaScript has gone missing"
