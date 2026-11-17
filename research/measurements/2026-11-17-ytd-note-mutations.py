"""Break the code behind claim `drilldown.ytd_column_note` nine ways and confirm its registered
checks fail each time (`plan/claims-mutation-audit.md`'s method: a check that still passes is
checking plumbing). Each mutation is applied to the working file, the claim's `checked_by` nodes
are run, and the file is restored byte for byte - also on an error.

    python research/measurements/2026-11-17-ytd-note-mutations.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MUTATIONS = [
    ("sec_crosscheck.py", "for k in range(4):                  # the four quarter ends", "for k in range(5):                  # the four quarter ends",
     "looks at a fifth quarter end"),
    ("sec_crosscheck.py", "YTD_TOL = 0.005 ", "YTD_TOL = 0.010 ", "match tolerance doubled"),
    ("sec_crosscheck.py", "YTD_MIN_SHARE = 0.02 ", "YTD_MIN_SHARE = 0.01 ", "minimum excess halved"),
    ("sec_crosscheck.py", "if q is None or ytd is None or not pos or pos < 2:", "if q is None or ytd is None or not pos or pos < 3:",
     "six-month columns ignored"),
    ("sec_crosscheck.py", "or abs(gap - excess) > YTD_TOL * abs(run):", "or False:", "any gap matches"),
    ("sec_crosscheck.py", 'return ("-" if float(v) < 0 else "") + _usd(v)', "return _usd(v)", "an outflow printed as an inflow"),
    ("sec_crosscheck.py", 'word = "below" if entry["filed"] < entry["run"] else "above"', 'word = "below"', "direction word fixed"),
    ("sec_crosscheck.py", "return ytd - first_filed(prev[pspan == pspan.max()]), ytd, pos, when", "return ytd, ytd, pos, when",
     "the quarter is not differenced"),
    ("generate_dashboard.py", "if (differ.length > (ytdCol ? 1 : 0)) html +=", "if (differ.length) html +=",
     "'usually a definition' printed over a known cause"),
]

if __name__ == "__main__":
    import claims
    claim = next(c for c in claims.CLAIMS if c.id == "drilldown.ytd_column_note")
    nodes = list(claim.checked_by)
    caught = 0
    for name, old, new, what in MUTATIONS:
        path = ROOT / name
        original = path.read_bytes()
        text = original.decode("utf-8")
        assert text.count(old) == 1, (name, old, text.count(old))
        try:
            path.write_bytes(text.replace(old, new).encode("utf-8"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *nodes],
                               cwd=ROOT, capture_output=True, text=True)
        finally:
            path.write_bytes(original)
        failed = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
        verdict = "CAUGHT" if r.returncode == 1 and failed else f"NOT CAUGHT (exit {r.returncode})"
        caught += verdict == "CAUGHT"
        print(f"{verdict:<12} {what}: {failed[0].split('::')[-1] if failed else r.stdout.strip().splitlines()[-1]}")
    print(f"\n{caught} of {len(MUTATIONS)} mutations caught by the claim's {len(nodes)} registered checks")
