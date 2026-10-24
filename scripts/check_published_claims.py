#!/usr/bin/env python3
"""Check the claims in the artifacts the data loop is about to publish.

Why this exists
---------------
The 2026-10-02 retrospective compared the two runners against each other, as
``prompts/retrospective.md`` section 1 instructs, and found that **ship gate 1
has no counterpart on the path that publishes most often.**

``nightly-screener.ps1`` runs the full suite before it will merge anything.
``data-run.ps1`` runs **no tests at all**, and it publishes to the live GitHub
Pages site five mornings a week - including ``SCREENER_OVERVIEW.md``, the public
methodology page, which ``run_screener.py`` step 11 regenerates on every full
run.

That is not hypothetical. On **2026-09-28** the 02:00 data run (``2e08f62``)
regenerated the overview, reverted the four corrections the 2026-09-25 session
had shipped, and pushed them to the public site - including a fetch-failure rate
the same session had measured at 0 of 9,036 and the page put at "10-25% of
tickers". Measured by this retrospective at that exact commit:
``pytest tests/test_overview_claims.py`` reports **12 failed, 2 passed in
0.36s**. The evidence to refuse that publish was already in the repository; the
loop doing the publishing simply never asked.

It also left the tree red, so the next code session's gate 1 failed at baseline
and the 09-28 session spent its research day on repair instead.

What is in scope
----------------
A test module belongs in this gate when both hold:

1. It asserts a property of a file in ``data-run.ps1``'s ``$DataArtifacts``
   that is **regenerated on every full run** (``PUBLISHED_ARTIFACTS`` below),
   either by reading the artifact or by asserting against the generator that
   writes it (rule 10: assert against the generator, not the committed file).
2. It needs no network, no git sandbox, no PowerShell and no concurrency, so it
   cannot flake inside an unattended loop.

Deliberately **not** the whole suite, which takes 124s against this subset's
~90s and, more importantly, would mean one unrelated red test stops *both*
loops. A red ``main`` already blocks the code loop from merging; it must not
also stop the data loop accumulating evidence. The subset is scoped to failures
that mean *this publish* is unsafe.

Membership is derived, not listed - 2026-10-16
----------------------------------------------
The 2026-10-16 retrospective found the gate had the same defect it was built to
fix, one level up. ``claims.py`` registers every sentence the site makes about
how a number is computed together with ``checked_by`` - the tests that make it
true. **Fifteen of those modules, holding 64 claim checks, were not in this
gate**, so the loop that publishes the page five mornings a week did not run the
evidence this project had already written down for it. All eight ``context.*``
claims were in that set: the context payload was split into its own file on
2026-10-08 and nothing guarding published artifacts was widened for it.

A hand-kept list of modules goes stale the moment someone registers a claim. So
``module_paths()`` is now the union of:

* ``MODULES`` - curated entries, each carrying a stated reason. Several are
  named by a claim too; the ones that are not (deleted surfaces, the overview
  generator, the weighting disclosure) are why this list still exists;
* ``claim_check_modules()`` - **every** module ``claims.py`` names in a
  ``checked_by``, derived at run time.

A registered claim whose check this gate never runs is now impossible by
construction. The browser modules are in the derived set on purpose: they are
the data loop's missing counterpart to ``scripts/rehearse_publish.py``, which
only the code loop has, and they check the page *as regenerated* rather than as
committed. They skip loudly where Playwright or Chromium is absent, so they
cannot jam the loop.

Exit codes - the runner distinguishes all three:
    0  every claim holds; safe to publish
    1  a claim fails; the artifacts about to be published are wrong
    3  pytest could not be run at all (not a verdict - the runner warns and
       continues, mirroring the ``node --check`` fallback added 2026-09-18, so a
       machine without pytest cannot jam an unattended loop)

Usage:
    python scripts/check_published_claims.py
    python scripts/check_published_claims.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The generated artifacts this gate speaks for. Kept in step with
# data-run.ps1's $DataArtifacts, narrowed to the ones a run rewrites.
#
# dashboard_context.js was missing here from the 2026-10-08 split until
# 2026-10-16: the context layer became a second published payload file and
# neither this scope nor either runner's parse check was widened for it.
PUBLISHED_ARTIFACTS = (
    "SCREENER_OVERVIEW.md",
    "index.html",
    "dashboard.html",
    "dashboard_data.js",
    "dashboard_context.js",
    "README.md",
)

# The modules, with why each one is here. tests/test_published_claims_gate.py
# pins this list and fails if a module asserting on SCREENER_OVERVIEW.md is
# added to tests/ without being added here - that is the exact class of the
# 2026-09-28 regression.
MODULES: tuple[tuple[str, str], ...] = (
    ("tests/test_claims_register.py",
     "every computational claim on the site is registered, checked, and not "
     "one of the false statements fixed on 2026-10-07"),
    ("tests/test_calculation_reproducibility.py",
     "every published category score and composite rebuilds from the published "
     "weights - the page's arithmetic adds up"),
    ("tests/test_metric_lineage.py",
     "every metric equation shown on the page rebuilds from the inputs shown beside it"),
    ("tests/test_overview_claims.py",
     "the published overview and index.html state the configured weights"),
    ("tests/test_overview_is_generated.py",
     "the committed overview equals the generator's output"),
    ("tests/test_weighting_disclosure.py",
     "the page names the portfolio weighting scheme config actually sets"),
    ("tests/test_weight_transparency.py",
     "published per-category contributions sum to the composite"),
    ("tests/test_size_tilt_is_documented_truthfully.py",
     "the size tilt is described as what the code does"),
    ("tests/test_risk_category_independence.py",
     "the risk category's documented construction matches the code"),
    # Added 2026-10-13. All four assert on the methodology page's text *through the
    # generator* rather than through the committed file, which is the better way
    # (rule 10) and was the one the gate's membership tripwire could not see. 83
    # tests, 12.9s measured.
    ("tests/test_filing_check_in_the_drilldown.py",
     "the methodology page and the drilldown say what is checked against the SEC filings, and what is not"),
    ("tests/test_growth_category.py",
     "the Growth section's claims hold for the run being published - that the "
     "ranking taken whole is close to growth-neutral, which is why the weight "
     "was kept on 2026-10-13"),
    ("tests/test_large_company_evidence.py",
     "the large-company value and momentum figures on the page match the lab's data"),
    ("tests/test_momentum_panic_rule.py",
     "the page describes the momentum regime rule the engine actually applies"),
    # Added 2026-10-15: the point-in-time limitation quotes the measured look-ahead from its JSON.
    ("tests/test_pit_record.py",
     "the page's look-ahead figure is the 2026-10-15 measurement, and the rebuild sets no weight"),
    # Added 2026-10-15 (night): the page quotes what the research engine found, from its summary.
    ("tests/test_v2_experiments.py",
     "the page's statement of the research engine's result matches the engine's published summary"),
    # Added 2026-10-16: the page is typeset in one pass; this keeps em dashes and hyphen-dashes off it.
    ("tests/test_overview_typesetting.py",
     "the methodology page publishes one dash and a true minus, with code left alone"),
    ("tests/test_factor_lab.py",
     "the page's 'Rerun the research' paragraph matches the experiments that exist, "
     "and the lab's scorecard carries the screener's own weights"),
    # Added 2026-10-16: the Piotroski paragraph is built from the weight tables, and every
    # published table is one some stock is scored with.
    ("tests/test_piotroski_conditional.py",
     "the page states the Piotroski split the engine applies, and publishes no weight table "
     "that applies to no stock"),
    ("tests/test_engine_lists_are_derived.py",
     "the page's Metric Coverage figure is the engine's per-stock coverage averaged"),
    ("tests/test_stock_summary.py",
     "no baked summary carries advice language"),
    ("tests/test_dashboard_surfaces.py",
     "deleted surfaces and payload keys stay deleted"),
    # Added 2026-10-19: the Track record panel's sentences against track_record.build - entry
    # timing read from each run's own prices, turnover and break-even cost by hand arithmetic,
    # the coin-toss and standard-error figures, and the nine corrected sentences.
    ("tests/test_track_record_claims.py",
     "the Track record panel says what track_record.build computes, and no more"),
    # Added 2026-10-19: the methodology page's computational sentences against the code - ROIC's
    # invested capital rebuilt from the published inputs, the cache window, the coverage bars,
    # the alerts, the weight tables, the percentile and trap rules, and the workbook's sheets.
    # Added 2026-10-22 (screener-bot ROADMAP 1.9): the trading bot reads dashboard_data.js as
    # Pages serves it and refuses to trade when a field it needs is missing - so a run that drops
    # one does not break this site, it silently stops the bot. The loop that publishes the payload
    # five mornings a week now checks the contract before it does.
    ("tests/test_bot_payload_contract.py",
     "every field the trading bot reads from dashboard_data.js is still published, in the "
     "shape it reads"),
    ("tests/test_input_gates.py",
     "the payload carries each price's NYSE session and the run's methodology version, which the "
     "health check's 2026-10-22 gates read, and those gates refuse what they were built to refuse"),
    # signals.json is written by this very run, after these gates, and is the only ranking the
    # bot may trade (2026-10-24). The page keeps being published by both loops; this file is
    # what says a gate stack accepted it, so the loop that writes it checks it first.
    ("tests/test_tradable_signals.py",
     "signals.json is written only after every gate and only by this loop, carries the top 40 "
     "by rank, and the code loop never stages it"),
    ("tests/test_run_fingerprint.py",
     "the payload names the run, the commit and the raw fetch it was computed from, and never "
     "borrows a neighbouring run's identity"),
    ("tests/test_run_archive.py",
     "the published run's raw fetch is kept outside the pruned tree and verified against the "
     "hash the payload publishes"),
    ("tests/test_methodology_claims_audit.py",
     "every formula, threshold, window and rule the methodology page states is the one the code "
     "applies (research/2026-10-19-methodology-claims-audit.md)"),
)

PYTEST_MISSING = 3

# A claim's check belongs in this gate unless it breaks scope rule 2 above -
# network, a git sandbox, PowerShell or concurrency. Empty today: all fifteen
# modules added on 2026-10-16 were measured running offline (392 tests, 48s;
# the two browser modules a further 26 tests in 26s, skipping loudly where
# Chromium is absent). It exists so that a future claim whose check genuinely
# cannot run unattended has a named, printed place to go rather than being
# dropped from the register's reach silently. Keep the reason specific.
EXCLUDED: dict[str, str] = {}


def claim_check_modules() -> tuple[str, ...]:
    """Every test module ``claims.py`` names as the check for a published claim.

    Derived rather than listed, because a hand-kept list goes stale the moment a
    claim is registered - which is exactly what happened between 2026-10-02 and
    2026-10-16 (see the module docstring). Raises if the register cannot be
    read; ``run()`` turns that into a refusal, never into a pass.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import claims  # noqa: PLC0415 - imported here so --list works without it on PATH

    mods: set[str] = set()
    for claim in claims.CLAIMS:
        for ref in claim.checked_by or ():
            mod = ref.split("::")[0]
            if mod.startswith("tests/") and mod not in EXCLUDED:
                mods.add(mod)
    return tuple(sorted(mods))


def module_paths() -> list[str]:
    """The curated list plus every module the claims register points at."""
    seen = []
    for mod in [m for m, _ in MODULES] + list(claim_check_modules()):
        if mod not in seen and mod not in EXCLUDED:
            seen.append(mod)
    return seen


def missing_modules() -> list[str]:
    return [m for m in module_paths() if not (ROOT / m).exists()]


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the claims in the artifacts about to be published.")
    ap.add_argument("--list", action="store_true", help="print the modules and exit")
    args = ap.parse_args(argv)

    # A register that cannot be read is not "nothing to check": every claim's
    # evidence would silently leave the gate. Refuse, loudly.
    try:
        derived = claim_check_modules()
    except Exception as exc:  # noqa: BLE001 - any failure here means refuse
        print(f"PUBLISHED CLAIMS: cannot read the claims register ({exc!r}) - "
              "refusing to publish rather than checking a shrunken gate.")
        return 1

    if args.list:
        curated = {m for m, _ in MODULES}
        for mod, why in MODULES:
            print(f"{mod}\t{why}")
        for mod in derived:
            if mod not in curated:
                print(f"{mod}\tnamed by claims.py as the check for a published claim")
        for mod, why in sorted(EXCLUDED.items()):
            print(f"{mod}\tEXCLUDED: {why}")
        return 0

    if EXCLUDED:
        # No silent caps: say what is not being checked, every run.
        for mod, why in sorted(EXCLUDED.items()):
            print(f"PUBLISHED CLAIMS: not checking {mod} - {why}")

    absent = missing_modules()
    if absent:
        # A renamed or deleted module must not silently shrink the gate.
        print("PUBLISHED CLAIMS: cannot run - these modules are missing: "
              + ", ".join(absent))
        print("Update scripts/check_published_claims.py MODULES, or restore them.")
        return 1

    cmd = [sys.executable, "-m", "pytest", *module_paths(),
           "-q", "--no-header", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        print("PUBLISHED CLAIMS: pytest is not available - claims NOT checked.")
        return PYTEST_MISSING
    except subprocess.SubprocessError as exc:
        print(f"PUBLISHED CLAIMS: could not run pytest ({exc}) - claims NOT checked.")
        return PYTEST_MISSING

    out = (proc.stdout or "") + (proc.stderr or "")
    tail = [ln for ln in out.splitlines() if ln.strip()]

    # pytest exits 4 on usage error and 5 when it collected nothing. Neither is
    # a verdict on the artifacts, and treating them as "safe to publish" is how
    # a gate quietly stops gating.
    if proc.returncode in (4, 5):
        print(f"PUBLISHED CLAIMS: pytest collected nothing or was misinvoked "
              f"(exit {proc.returncode}) - claims NOT checked.")
        for ln in tail[-5:]:
            print(f"    {ln}")
        return 1

    if proc.returncode == 0:
        summary = tail[-1] if tail else "no output"
        print(f"PUBLISHED CLAIMS: PASS ({summary})")
        return 0

    print(f"PUBLISHED CLAIMS: FAIL (pytest exit {proc.returncode})")
    for ln in [ln for ln in tail if ln.startswith("FAILED") or ln.startswith("ERROR")][-12:]:
        print(f"    {ln}")
    if tail:
        print(f"    {tail[-1]}")
    return 1


if __name__ == "__main__":
    sys.exit(run())
