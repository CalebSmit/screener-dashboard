"""Would anything notice if the detail split were quietly broken?

The standard from `plan/claims-mutation-audit.md` (nine passes) and the 2026-11-07 per-band
gate: a test module written the same day as its feature is the easiest place for a check to
read as coverage and not be one. So each mutation below is one a reader could plausibly ship,
and the question asked of it is whether the project *refuses* it.

Two numbers are reported per mutation: whether the new module catches it, and - for the ones
where that matters - whether anything else in the suite would.

Run:  python research/measurements/2026-11-08-detail-split-mutations.py

Edits `generate_dashboard.py` and `scripts/nightly-screener.ps1` in place and restores both in a
`finally`. Refuses to run on a dirty tree, so an interrupted run can always be recovered with
`git checkout --`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generate_dashboard.py"
RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
DATA_RUNNER = ROOT / "scripts" / "data-run.ps1"
MODULE = "tests/test_detail_payload_split.py"
PARSE_GATE = "tests/test_payload_parse_gate.py"

# (name, file, before, after, the tests expected to refuse it)
MUTATIONS = [
    ("split_detail copies instead of moving (s.get for s.pop)", GEN,
     '{k: s.pop(k) for k in DETAIL_KEYS if k in s}',
     '{k: s.get(k) for k in DETAIL_KEYS if k in s}', MODULE),
    ("every stock carried, including the ones with nothing to say", GEN,
     'if any(str(v or "").strip() for v in part.values()):',
     'if True:', MODULE),
    ("the file is loaded from the start-up block as well", GEN,
     '        loadContext("./dashboard_context.js?v={ctx_version}");',
     '        loadContext("./dashboard_context.js?v={ctx_version}");\n'
     '        loadDetail("./dashboard_detail.js?v={det_version}");', MODULE),
    ("the loader loses its once-only guard", GEN,
     'if (DETAIL_LOADED || DETAIL_PENDING) return;',
     'if (DETAIL_LOADED) return;', MODULE),
    ("the waiting state goes back to an empty block", GEN,
     "        if (!about && !DETAIL_LOADED) {{",
     "        if (false) {{", MODULE),
    ("the loader stops merging into stock_detail", GEN,
     'Object.keys(P).forEach(t => { if (D.stock_detail[t]) Object.assign(D.stock_detail[t], P[t]); });',
     'Object.keys(P).forEach(t => { if (false) Object.assign(D.stock_detail[t], P[t]); });', MODULE),
    # The one that got through on the first pass, and the reason this harness exists: the file
    # is still *named* in the runner, so both the direct test and the 2026-10-16 tripwire pass -
    # while '' makes both runners skip the header check and `node --check` for it. Caught now by
    # test_both_runners_check_each_payload_by_its_assignment_not_only_by_its_name.
    ("the code runner keeps the file but turns its parse and header check off", RUNNER,
     "        @($detPath,   100000, '^window\\.SCREENER_DETAIL=')",
     "        @($detPath,   100000, '')", f"{MODULE} {PARSE_GATE}"),
    ("the data runner keeps the file but turns its parse and header check off",
     DATA_RUNNER,
     "        @($detPath,  100000, '^window\\.SCREENER_DETAIL=')",
     "        @($detPath,  100000, '')", f"{MODULE} {PARSE_GATE}"),
]


def dirty() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--",
                           str(GEN), str(RUNNER), str(DATA_RUNNER)],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return bool(out)


def run(tests: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *tests.split(), "-q", "-p", "no:randomly"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
    return r.returncode != 0, (tail[0] if tail else "")


def main() -> int:
    if dirty():
        print("refusing to run on a dirty tree: commit or stash generate_dashboard.py and "
              "both runners first")
        return 2
    originals = {p: p.read_text(encoding="utf-8") for p in (GEN, RUNNER, DATA_RUNNER)}
    caught = 0
    try:
        for name, path, before, after, tests in MUTATIONS:
            text = originals[path]
            if text.count(before) != 1:
                print(f"SKIP  {name}\n      anchor appears {text.count(before)}x - "
                      "the code moved; fix the anchor rather than trusting this row")
                continue
            path.write_text(text.replace(before, after), encoding="utf-8")
            try:
                failed, line = run(tests)
            finally:
                path.write_text(text, encoding="utf-8")
            caught += failed
            print(f"{'CAUGHT' if failed else 'GOT THROUGH':<12}{name}\n            {line}")
    finally:
        for p, t in originals.items():
            p.write_text(t, encoding="utf-8")
    print(f"\n{caught} of {len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
