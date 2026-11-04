"""Which payload-reading tests name a real ticker that must be in today's data? (2026-11-03 night)

CLAUDE.md row "data->red": a test that reads the published payload asserts its own premise, never
which stock today's data happens to hold. Measured on the 2026-11-03 payload: 37 references in 6
modules (JPM, EXPE, APA, MSFT; HST and BBY ride along in one compare tray). Each turns `main` red
the morning that company leaves the index, with no commit touching code.

    python research/measurements/2026-11-03-named-tickers-in-payload-tests.py [REPO]
"""
import json
import re
import sys
from pathlib import Path

W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
s = (W / "dashboard_data.js").read_text(encoding="utf-8")
tickers = set(json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))["stock_detail"])
MARK = ("dashboard_data.js", "dashboard_context.js", "SCREENER_DATA", "D.stock_detail", "_payload(")
pat = re.compile(r"""(?:D\.stock_detail\.([A-Z]{1,5})\b|stock_detail\[['"]([A-Z]{1,5})['"]\]|openStockDetail\(['"]([A-Z]{1,5})['"]\)|_sheet\(page, ['"]([A-Z]{1,5})['"]\))""")
hits = {}
for p in sorted((W / "tests").glob("test_*.py")):
    src = p.read_text(encoding="utf-8", errors="replace")
    if not any(m in src for m in MARK):
        continue
    for n, line in enumerate(src.splitlines(), 1):
        for m in pat.finditer(line):
            t = next(g for g in m.groups() if g)
            if t in tickers:
                hits.setdefault(p.name, []).append((n, t))
tot = 0
for f, hs in sorted(hits.items()):
    tot += len(hs)
    print(f"{f}: {len(hs)} - {sorted(set(t for _, t in hs))}")
print("total", tot, "in", len(hits), "modules")
