"""Would anything notice if the peer table went back to building its rows two ways?

The standard from `plan/claims-mutation-audit.md` (nine passes): a test module written the same
day as its feature is the easiest place for a check to read as coverage and not be one. So each
mutation below is one a reader could plausibly ship - reintroducing the precomputed self row,
pointing the subject's row back at it, reverting the empty-state guard, or quietly giving the
subject's row its own arithmetic - and the question asked of it is whether the project *refuses*
it.

**Each mutation that touches the page's JS regenerates `index.html` and the payload before the
tests run**, which is the point of the harness. Mutating the committed page directly would be
caught by the drift tripwire ("the committed page is what the generator builds"), and the
2026-10-31 pass established that a drift catch is not arithmetic coverage: it is green in exactly
the state the change ships in.

Run:  python research/measurements/2026-11-08-peer-row-mutations.py

Edits `generate_dashboard.py` in place and restores it in a `finally`, then restores the
published page and payload from git. Refuses to run on a dirty tree, so an interrupted run is
always recoverable with `git checkout --`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generate_dashboard.py"
PUBLISHED = ("dashboard.html", "index.html", "dashboard_data.js", "dashboard_context.js",
             "dashboard_detail.js")

MODULE = "tests/test_peer_row_one_path.py"
EMPTY = "tests/test_sheet_round_three.py::test_empty_sections_say_so"
STILL = "tests/test_sheet_round_three.py::test_the_peers_panel_still_prints_the_columns_it_has"

SELF_BLOCK = """        detail["self_metrics"] = {
            "rev_growth": _safe(row.get("revenue_growth")),
            "pe_ratio": round(1.0 / row["earnings_yield"], 1) if pd.notna(row.get("earnings_yield")) and row["earnings_yield"] > 0 else None,
            "net_margin": round(_ni / _rev, 4) if (_ni is not None and _rev is not None and _rev > 0) else None,
            "roic": _safe(row.get("roic")),
            "roe": _safe(row.get("roe")),
            "debt_equity": _safe(row.get("debt_equity")),
            "div_yield": round(_div_yield, 4) if _div_yield is not None else None,
            "fcf_yield": _safe(row.get("fcf_yield")),
            "mcap": _mcap,
        }

        stock_detail[ticker] = detail"""

# The one row that is two edits at once: the realistic reopening is the pair, not either half.
PAIR = [("        stock_detail[ticker] = detail", SELF_BLOCK),
        ("        const self = buildPeerRow(ticker);",
         "        const self = s.self_metrics || buildPeerRow(ticker);")]

# (name, before, after, tests expected to refuse it, whether the page must be regenerated).
# before=None means "apply every edit in PAIR".
MUTATIONS = [
    ("the precomputed self row comes back into the payload",
     "        stock_detail[ticker] = detail", SELF_BLOCK, MODULE, True),
    # This one GOT THROUGH on the first pass and was right to: with no such block in the payload
    # the fallback is all that ever runs, so it is an equivalent mutant - and a loaded one, since
    # a later session reintroducing the block would find the page already preferring it. Closed by
    # test_the_page_contains_no_reader_of_a_precomputed_self_row, which reads the generated page
    # for a property access rather than for the word.
    ("the subject's row reads the precomputed block again (JS half only)",
     "        const self = buildPeerRow(ticker);",
     "        const self = s.self_metrics || buildPeerRow(ticker);", MODULE, True),
    ("both halves together - the block published and the page preferring it",
     None, None, MODULE, True),
    ("the subject's row gets arithmetic of its own (market cap 1% out)",
     "        const selfRow = {{ ...self, isSelf: true }};",
     "        const selfRow = {{ ...self, mcap: self.mcap === null ? null : self.mcap * 1.01, isSelf: true }};",
     MODULE, True),
    ("the subject's row rounds P/E the other way",
     "        const selfRow = {{ ...self, isSelf: true }};",
     "        const selfRow = {{ ...self, pe_ratio: self.pe_ratio === null ? null : "
     "Math.round(self.pe_ratio) * 1.0, isSelf: true }};", MODULE, True),
    ("the empty state asks whether a block exists, not whether a figure does",
     "        if (!self || !PEER_COLS.some(k => self[k] !== null && self[k] !== undefined)) {{",
     "        if (!self) {{", f"{EMPTY} {STILL}", True),
    ("a column the row prints stops being published",
     '            "net_margin": round(_ni / _rev, 4) if (_ni is not None and _rev is not None and _rev > 0) else None,\n'
     '            "fcf": round(_ocf - abs(_capex_v), 2) if (_ocf is not None and _capex_v is not None) else None,',
     '            "fcf": round(_ocf - abs(_capex_v), 2) if (_ocf is not None and _capex_v is not None) else None,',
     MODULE, False),
]


def dirty() -> bool:
    paths = [str(GEN)] + [str(ROOT / p) for p in PUBLISHED]
    out = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return bool(out)


_INSTALL = """
import sys
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import rehearse_publish as rp
run = rp.newest_full_run(ROOT / "runs")
for name, blob in rp.regenerate(run).items():
    (ROOT / name).write_bytes(blob)
"""


def regenerate() -> bool:
    """Install the page and payload as the *mutated* generator writes them, from the newest run.

    In a subprocess, so the mutated `generate_dashboard.py` is imported fresh each time - an
    already-imported module in this process would silently rehearse the unmutated code and every
    row would read CAUGHT or GOT THROUGH for the wrong reason."""
    r = subprocess.run([sys.executable, "-c", _INSTALL, str(ROOT)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"      could not regenerate: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
        return False
    return True


def restore_published() -> None:
    subprocess.run(["git", "checkout", "--", *PUBLISHED], cwd=ROOT,
                   capture_output=True, text=True)


def run(tests: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *tests.split(), "-q", "-p", "no:randomly"],
                       cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    return r.returncode != 0, (lines[-1] if lines else "")


def main() -> int:
    if dirty():
        print("refusing to run on a dirty tree: commit or stash generate_dashboard.py and the "
              "published page and payload first")
        return 2
    original = GEN.read_text(encoding="utf-8")
    caught = 0
    skipped = 0
    try:
        for name, before, after, tests, needs_page in MUTATIONS:
            edits = PAIR if before is None else [(before, after)]
            bad = [b for b, _ in edits if original.count(b) != 1]
            if bad:
                print(f"SKIP  {name}\n      an anchor does not appear exactly once - the code "
                      "moved; fix the anchor rather than trusting this row")
                skipped += 1
                continue
            mutated = original
            for b, a in edits:
                mutated = mutated.replace(b, a)
            GEN.write_text(mutated, encoding="utf-8")
            try:
                ok = regenerate() if needs_page else True
                if not ok:
                    print(f"SKIP  {name}")
                    skipped += 1
                    continue
                failed, line = run(tests)
            finally:
                GEN.write_text(original, encoding="utf-8")
                if needs_page:
                    restore_published()
            caught += failed
            print(f"{'CAUGHT' if failed else 'GOT THROUGH':<12}{name}\n            {line}")
    finally:
        GEN.write_text(original, encoding="utf-8")
        restore_published()
    total = len(MUTATIONS) - skipped
    print(f"\n{caught} of {total} caught ({skipped} skipped)")
    return 0 if caught == total and total else 1


if __name__ == "__main__":
    sys.exit(main())
