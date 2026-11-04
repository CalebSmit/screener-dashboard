"""Mutation check for the two runner fixes of 2026-11-03 (owner-run evening).

1. ``nightly-screener.ps1`` waits for the network before skipping the day (10 tries, 30 s apart).
2. ``publish-brief.ps1`` sets an unpublished brief aside so the tree stays clean for the 13:00 slot.

Each mutation reverts one part in place; the named tests must fail on every one. Run from the repo
root: ``python research/measurements/2026-11-03-runner-resilience-mutations.py``. Restores every
file byte for byte, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ["tests/test_code_loop_network_wait.py", "tests/test_brief_publish_safety.py"]

MUTATIONS = [
    ("one try, as before", "scripts/nightly-screener.ps1",
     "$RemoteTries = 10", "$RemoteTries = 1"),
    ("no wait between tries", "scripts/nightly-screener.ps1",
     "        Start-Sleep -Seconds 30\r\n    }\r\n    if ($ls.ExitCode -ne 0) {",
     "    }\r\n    if ($ls.ExitCode -ne 0) {"),
    ("a failure is reported as reachable", "scripts/nightly-screener.ps1",
     "    if ($ls.ExitCode -ne 0) {\r\n        Write-NativeOutput $ls 'ERROR'",
     "    if ($false) {\r\n        Write-NativeOutput $ls 'ERROR'"),
    ("the unpublished brief is left in the tree", "scripts/publish-brief.ps1",
     "if (-not $ok) { Set-BriefAside $RepoPath $Logger }", ""),
    ("the copy is written even where it would dirty the tree", "scripts/publish-brief.ps1",
     "if ((Invoke-BriefGit $RepoPath @('check-ignore', '-q', $aside)).ExitCode -eq 0) {",
     "if ($true) {"),
    ("the copy is skipped", "scripts/publish-brief.ps1",
     "            Copy-Item -LiteralPath", "            # Copy-Item -LiteralPath"),
]


def main() -> int:
    caught = 0
    for name, rel, old, new in MUTATIONS:
        p = ROOT / rel
        orig = p.read_bytes()
        text = orig.decode("utf-8")
        assert text.count(old) == 1, f"{name}: anchor not unique in {rel}"
        try:
            p.write_bytes(text.replace(old, new).encode("utf-8"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                               cwd=ROOT, capture_output=True, text=True)
        finally:
            p.write_bytes(orig)
        hit = r.returncode != 0
        caught += hit
        print(f"{'CAUGHT' if hit else 'MISSED'}  {name}")
    print(f"{caught} of {len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
