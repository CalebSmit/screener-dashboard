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


def computed_sentences(path: Path) -> list[str]:
    """Every sentence in a generator that puts a computed number into prose.

    The key is the sentence as published, tag-stripped and whitespace-normalised,
    with each interpolation left in place as its own source expression. So a
    reword fires (as a `detect` pattern does, deliberately) **and** swapping which
    expression is printed fires - the second is the one a `{}` placeholder would
    miss, and it is how a figure gets taken off the wrong era.
    """
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.JoinedStr):
            continue
        parts, holes = [], []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                parts.append(f"\x00{len(holes)}\x00")
                holes.append(v)
        text = _TAG.sub(
            lambda m: " " if (m.group(1) or "").lower() in _INLINE else _BREAK, "".join(parts))
        for entity, plain in (("&mdash;", "-"), ("&ndash;", "-"), ("&nbsp;", " ")):
            text = text.replace(entity, plain)
        units = re.split(r"(?<=[.!?:])\s+|" + _BREAK, text)
        # A stat card is a figure in one element and its caption in the next
        # (`<div class="v">{months_needed(...)}</div><div class="k">months of record
        # to show a monthly correlation of 0.05 ...</div>`). Block-splitting leaves
        # the figure with no words and the caption with no figure, so neither half
        # is a sentence and the claim disappears - which it did on the first run of
        # this sweep, for all three of the month counts registered the same morning.
        # A figure-only unit therefore takes the following unit's words.
        merged: list[str] = []
        for unit in units:
            if merged and _HOLE.search(merged[-1]) and not _PROSE.search(_HOLE.sub(" ", merged[-1])):
                merged[-1] += " " + unit
            else:
                merged.append(unit)
        for raw in merged:
            idx = [int(m.group(1)) for m in _HOLE.finditer(raw)]
            if not any(_is_computed(holes[i]) for i in idx):
                continue
            if not _PROSE.search(_HOLE.sub(" ", raw)):
                continue                       # a bare figure with no words anywhere
            out.append(re.sub(r"\s+", " ", _HOLE.sub(
                lambda m: "{" + ast.unparse(holes[int(m.group(1))].value) + "}", raw)).strip())
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
        "open:pass-bar-printed:2026-10-29 - nothing asserts the page's own bar_p/bar_h. "
        "test_the_bar_at_38_held_out_months... pins X.pass_bar as a function, not that the "
        "page prints pass_bar(n, the median row's months) for each era",
    "a difference counts only if it reaches beyond them, a t-statistic of {bar_p} over these "
    "{mid_p.get('n')} months, in the same direction as 2020-2026.":
        "open:pass-bar-under-chart:2026-10-29 - the same figure, printed a second time under the chart; "
        "nothing pins either, and nothing pins that the two agree",

    # --- the chart ----------------------------------------------------------
    "{''.join(dots)} {num(-LIM, 2)} {num(-LIM / 2, 2)} 0 {num(LIM / 2, 2)} {num(LIM, 2)} "
    "difference 95% interval pass bar smallest difference the test could reliably detect":
        "open:chart-axis-and-geometry:2026-10-29 - the axis ticks (+/-LIM and its halves) and, in the same block, the "
        "dot/band/tick geometry `_pos` puts on that axis. plan/claims-mutation-audit.md names "
        "`_pos`: a sign error would put every dot on the wrong side of zero with every printed "
        "number still correct",

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
        "open:size-row-mean-t-and-bar-word:2026-10-29 - the Size row's own mean and t, and the word choosing 'short of' or "
        "'beyond' the pass bar at |t| = bar. test_the_size_sentence... pins the coverage "
        "correlations in the next sentence and not these",
    "The share of each month's members the test could include rose steadily, from "
    "{rel['first_cov'] * 100}% in {rel['first_year']} to {rel['last_cov'] * 100}% in "
    "{rel['last_year']} (its correlation with the calendar is {rel['r_time']} ), and within "
    "2012-2019 the Size screen's monthly result moved with that share {moved} the current "
    "weighting's did (correlations of {num(rel['pre_r_size'])} and {num(rel['pre_r_base'])} ).":
        "open:size-coverage-percentages:2026-10-29 - the two yearly coverage percentages are pinned by nothing; "
        "test_the_size_sentence... recomputes r_time and both within-era correlations",

    # --- if you had kept the best backtest ---------------------------------
    "Over {a['first_month'][:4]}-{a['last_month'][:4]} the PBO was {a['pbo']} , "
    "{_pbo_words(a['pbo'])}, and the winner beat the current weighting on the other half in "
    "{a['winner_beat_current_out_of_sample'] * 100}% of trials.":
        "open:pbo-all-era-prose:2026-10-29 - the out-of-sample share is pinned by nothing, and _pbo_words is "
        "probed at 0.21/0.5/0.73 but never at its own bars 0.45 and 0.55. The stat card's PBO "
        "is pinned (test_the_page_prints_the_ledger_s_reading_all_three_ways), not this copy",
    "Over {rec['first_month'][:4]}-{rec['last_month'][:4]} alone the PBO was {rec['pbo']} , "
    "{_pbo_words(rec['pbo'])}, and the winner beat the current weighting out of sample in "
    "{rec['winner_beat_current_out_of_sample'] * 100}% of trials.":
        "open:pbo-recent-era-prose:2026-10-29 - as the row above, for the recent era",
    "Drop {html.escape(label(wo['dropped']))} from the set and the PBO over every year is "
    "{wo['pbo']} , {_pbo_words(wo['pbo'])}.":
        "open:pbo-without-winner-prose:2026-10-29 - the prose PBO and its word; the card is pinned, this copy is not",
    "Its lead over the current weighting was {num(pre['mean'])} over 2012-2019 and "
    "{num(hold['mean'])} over the held-out months":
        "open:pbo-top-winner-lead:2026-10-29 - the top winner's two era means, and the clause 'where it fell "
        "behind' that depends on their signs",
    "split the history into equal blocks of months, and for every way of picking half of them "
    "( {a['splits']} ways here, from {a['blocks']} blocks), take the weighting that did best on "
    "that half and see where it ranks on the other half.":
        "checked:tests/test_overfitting.py::test_the_page_prints_the_ledger_s_reading_all_three_ways",
    "This uses the current weighting and the {a['configurations'] - 1} alternatives above, each "
    "scored month by month against the current one on the stocks both ranked, and reruns nothing.":
        "open:pbo-configuration-count:2026-10-29 - the count of alternatives the PBO was computed over is this page's "
        "own subtraction and is pinned by nothing; summary['alternatives'] is a different number "
        "from a different source",

    # --- the trading rule ---------------------------------------------------
    "In {lab} it returned {abs(st['mean']) * 100}% a month {word} that average "
    "(t {num(st['t'], 2)}),":
        "open:bot-monthly-excess-figure:2026-10-29 - test_the_trading_rule_sentence_follows_each_period_s_sign_and_t pins "
        "the words, the sign and the t, and matches the percentage as `[0-9.]+%` - so the "
        "monthly figure itself is a wildcard",
    "{yc(ys[0])}% of each month's members in {ys[0]}, rising to {yc(ys[-1])}% in {ys[-1]}, so "
    "the average leans toward the companies that survived.":
        "open:bot-coverage-by-year:2026-10-29 - the first and last year's mean coverage. "
        "test_the_trading_rule_sentence... pins only 'members still listed today'",
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

# The twelve sentences above whose arithmetic nothing pins, as of 2026-10-29, by
# the slug in their verdict. **Literals, not a comprehension over the table** -
# `frozenset(s for s, v in ... if v.startswith("open:"))` is the table compared
# with itself and passes at any value (`plan/claims-mutation-audit.md` row 59).
# This set only ever shrinks: closing one means deleting its slug and giving the
# sentence a claim. A session that has to add one says why in `NIGHTLY_LOG.md`.
KNOWN_OPEN: frozenset[str] = frozenset({
    "pass-bar-printed", "pass-bar-under-chart", "chart-axis-and-geometry",
    "size-row-mean-t-and-bar-word", "size-coverage-percentages",
    "pbo-all-era-prose", "pbo-recent-era-prose", "pbo-without-winner-prose",
    "pbo-top-winner-lead", "pbo-configuration-count",
    "bot-monthly-excess-figure", "bot-coverage-by-year",
})


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
    for sentence, verdict in RESEARCH_PAGE_SENTENCES.items():
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