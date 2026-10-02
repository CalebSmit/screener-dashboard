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
A test module belongs in ``MODULES`` when both hold:

1. It asserts a property of a file in ``data-run.ps1``'s ``$DataArtifacts``
   that is **regenerated on every full run** - ``SCREENER_OVERVIEW.md``,
   ``index.html``, ``dashboard.html``, ``dashboard_data.js``, ``README.md`` -
   either by reading the artifact or by asserting against the generator that
   writes it (rule 10: assert against the generator, not the committed file).
2. It needs no network, no git sandbox, no PowerShell and no concurrency, so it
   cannot flake inside an unattended loop.

Deliberately **not** the whole suite, which takes 124s against this subset's
15s and, more importantly, would mean one unrelated red test stops *both*
loops. A red ``main`` already blocks the code loop from merging; it must not
also stop the data loop accumulating evidence. The subset is scoped to failures
that mean *this publish* is unsafe.

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
PUBLISHED_ARTIFACTS = (
    "SCREENER_OVERVIEW.md",
    "index.html",
    "dashboard.html",
    "dashboard_data.js",
    "README.md",
)

# The modules, with why each one is here. tests/test_published_claims_gate.py
# pins this list and fails if a module asserting on SCREENER_OVERVIEW.md is
# added to tests/ without being added here - that is the exact class of the
# 2026-09-28 regression.
MODULES: tuple[tuple[str, str], ...] = (
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
    ("tests/test_stock_summary.py",
     "no baked summary carries advice language"),
    ("tests/test_dashboard_surfaces.py",
     "deleted surfaces and payload keys stay deleted"),
)

PYTEST_MISSING = 3


def module_paths() -> list[str]:
    return [m for m, _ in MODULES]


def missing_modules() -> list[str]:
    return [m for m in module_paths() if not (ROOT / m).exists()]


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the claims in the artifacts about to be published.")
    ap.add_argument("--list", action="store_true", help="print the modules and exit")
    args = ap.parse_args(argv)

    if args.list:
        for mod, why in MODULES:
            print(f"{mod}\t{why}")
        return 0

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
