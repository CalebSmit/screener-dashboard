#!/usr/bin/env python3
"""Which published artifact has which check? The retrospective's step-1 enumeration.

`prompts/retrospective.md` section 1 asks a retrospective to list every artifact
`data-run.ps1` stages and name the check standing between it and the public site. Done by
reading, that question has been answered wrongly four times - 2026-09-04, 09-18, 10-02 and
10-16 each found an artifact with no check on the path that publishes most often. This does
the crude mechanical version so the reading starts from a list rather than from memory.

What it prints, per entry in `$DataArtifacts`:

* every test module whose source names the artifact (a keyword match, so expect false
  positives - a module about the *runners* names every artifact);
* which of those the published-claims gate actually runs.

**It is a prompt for judgement, not a verdict.** The 2026-10-30 finding came out of the
`improvement` and `factor_vol_history.csv` rows: `tests/test_one_observation_per_run_date.py`
was the only module holding a tripwire on the real committed evidence files, and it was
outside the gate, because gate membership derives from `claims.py` and `claims.py` registers
published *sentences* - an evidence file carries none, so nothing could pull it in.

Not a test and not in any gate: it reports, and a session reads it.

    python scripts/retro_artifact_coverage.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def data_artifacts() -> list[str]:
    """The literal list `data-run.ps1` stages, read from the runner rather than copied."""
    src = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8-sig")
    m = re.search(r"\$DataArtifacts\s*=\s*@\((.*?)\n    \)", src, re.S)
    if not m:
        raise SystemExit("could not find $DataArtifacts in scripts/data-run.ps1")
    return re.findall(r"'([^']+)'", m.group(1))


def gate_modules() -> set[str]:
    r = subprocess.run(
        [sys.executable, "scripts/check_published_claims.py", "--list"],
        capture_output=True, text=True, cwd=ROOT,
    )
    return {line.split("\t")[0] for line in r.stdout.splitlines() if line.strip()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    arts = data_artifacts()
    in_gate = gate_modules()
    print(f"$DataArtifacts: {len(arts)} entries; publish gate: {len(in_gate)} modules")

    bodies = {}
    for t in sorted(list((ROOT / "tests").glob("test_*.py")) + [ROOT / "test_screener.py"]):
        bodies[str(t.relative_to(ROOT)).replace("\\", "/")] = t.read_text(
            encoding="utf-8", errors="replace")

    for a in arts:
        leaf = a.split("/")[-1]
        hits = [rel for rel, b in bodies.items() if a in b or (leaf != a and leaf in b)]
        outside = [h for h in hits if h not in in_gate]
        print()
        print(f"--- {a}")
        print(f"    {len(hits)} module(s) name it; {len(outside)} outside the publish gate")
        for m in outside:
            print(f"      OUT  {m}")
    print()
    print("An artifact with no module at all, or whose only real tripwire is OUT, is the finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
