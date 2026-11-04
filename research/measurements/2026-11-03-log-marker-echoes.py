"""Phrase-anywhere vs runner-line log classification, on every real log and on echo cases.

2026-11-03 (owner-run evening). ``write_brief.outcome`` and ``newest_session`` used to ask whether
a marker phrase appeared anywhere in a run log; they now ask whether the runner itself wrote a
message starting with it (``write_brief.said``). This compares the two on every log in ``logs/``
(gitignored, so the result depends on the machine; 39 logs on the night, 0 differences) and on
four echo cases built from real line shapes (old: wrong on all four; new: right on all four).

    python research/measurements/2026-11-03-log-marker-echoes.py [OLD_REV]

OLD_REV is the commit holding the phrase-anywhere version (default 4abaf15, the last before it).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import write_brief as new  # noqa: E402


def _old(rev: str):
    src = subprocess.run(["git", "show", f"{rev}:scripts/write_brief.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    f = Path(tempfile.mkdtemp()) / "write_brief_old.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("write_brief_old", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(rev: str = "4abaf15") -> int:
    old = _old(rev)
    logs = sorted((ROOT / "logs").glob("nightly-*.log")) + sorted((ROOT / "logs").glob("datarun-*.log"))
    diff = 0
    for p in logs:
        lines = new.tail(p, 400)
        head = new.tail(p, 60)
        a, b = old.outcome(lines)[0], new.outcome(lines)[0]
        x = any(m in "\n".join(head) for m in old.NOT_A_SESSION)
        y = new.said(head, *new.NOT_A_SESSION)
        if a != b or x != y:
            diff += 1
            print(f"DIFF {p.name}: {a} -> {b}; not-a-session {x} -> {y}")
    print(f"{len(logs)} real logs, {diff} classified differently")

    failed = "[06:00:01] [ERROR] Cannot reach the git remote (offline, or credentials unavailable). Skipping today's run."
    cases = {
        "commit subject 'shipped to main'": "[13:53:55] [INFO]     0043f03 Log: the fix shipped to main at last",
        "commit subject 'discarded'": "[13:53:55] [INFO]     1a2b3c4 Explain why a run is discarded",
        "the health line": "[08:07:01] [INFO] Health: ... Data loop published? yes - (completed; Data loop complete)",
        "commit subject 'Standing down'": "[13:53:55] [INFO]     5d6e7f8 Standing down the extra slots on a 429",
    }
    print("\nA failed run with one echoed line (right answer: failed, and a session):")
    for name, echo in cases.items():
        lines = [failed, echo]
        print(f"  {name:34s} old={old.outcome(lines)[0]!r:24s} (session={not any(m in echo for m in old.NOT_A_SESSION)})"
              f"  new={new.outcome(lines)[0]!r} (session={not new.said(lines, *new.NOT_A_SESSION)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
