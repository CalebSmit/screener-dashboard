"""Mutation check for the context cards' printed figures (2026-11-03, owner-run evening).

Each mutation breaks one slot's printing in ``generate_dashboard.py``;
``tests/test_context_cards_printed.py`` and ``tests/test_case_against.py`` must fail on every one. Run from the repo root:
``python research/measurements/2026-11-03-context-card-mutations.py``. Restores the file byte for
byte, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generate_dashboard.py"
TEST = ["tests/test_context_cards_printed.py", "tests/test_case_against.py"]

MUTATIONS = [
    ("buys print the sales figure",
     "(ins.buy_n ? cMoney(ins.buy_value) : 'None')", "(ins.buy_n ? cMoney(ins.sell_value) : 'None')"),
    ("insider and trade counts swapped",
     "ins.buy_people + ' insider' + (ins.buy_people === 1 ? '' : 's') + ', ' + ins.buy_n + ' trade'",
     "ins.buy_n + ' insider' + (ins.buy_people === 1 ? '' : 's') + ', ' + ins.buy_people + ' trade'"),
    ("the singular dropped",
     "ins.sell_people + ' insider' + (ins.sell_people === 1 ? '' : 's') + (ins.sell_planned_value != null && ins.sell_value > 0 ? ', ' + Math.round(",
     "ins.sell_people + ' insiders' + (ins.sell_planned_value != null && ins.sell_value > 0 ? ', ' + Math.round("),
    ("the card's plan share truncates",
     "<span class=\"ctx-fs\">' + (ins.sell_n ? ins.sell_people + ' insider' + (ins.sell_people === 1 ? '' : 's') + (ins.sell_planned_value != null && ins.sell_value > 0 ? ', ' + Math.round(",
     "<span class=\"ctx-fs\">' + (ins.sell_n ? ins.sell_people + ' insider' + (ins.sell_people === 1 ? '' : 's') + (ins.sell_planned_value != null && ins.sell_value > 0 ? ', ' + Math.floor("),
    ("a missing plan flag printed as a share",
     "<span class=\"ctx-fs\">' + (ins.sell_n ? ins.sell_people + ' insider' + (ins.sell_people === 1 ? '' : 's') + (ins.sell_planned_value != null && ins.sell_value > 0",
     "<span class=\"ctx-fs\">' + (ins.sell_n ? ins.sell_people + ' insider' + (ins.sell_people === 1 ? '' : 's') + (ins.sell_value > 0"),
    ("the case against's purchases print the sales",
     "', against ' + cMoney(ins.buy_value) + ' of purchases.'", "', against ' + cMoney(ins.sell_value) + ' of purchases.'"),
    ("the case against's net-sale rule loosened",
     "const net = ins.sell_n > 0 && (!ins.buy_n || (ins.sell_value || 0) > (ins.buy_value || 0));",
     "const net = ins.sell_n > 0;"),
    ("the case against's plan share truncates",
     "const plan = (ins.sell_planned_value != null && ins.sell_value > 0) ? ', ' + Math.round(",
     "const plan = (ins.sell_planned_value != null && ins.sell_value > 0) ? ', ' + Math.floor("),
    ("52-week distance measured from the low", "cPct(c.px / c.hi - 1) + '</span><span class=\"ctx-fs\">high '",
     "cPct(c.px / c.lo - 1) + '</span><span class=\"ctx-fs\">high '"),
    ("52-week low prints the high", "', low ' + cPrice(c.lo) + '</span></div>';", "', low ' + cPrice(c.hi) + '</span></div>';"),
    ("the tile's ordinal truncates", "const ord = p => { const n = Math.round(Number(p));",
     "const ord = p => { const n = Math.floor(Number(p));"),
    ("the tile labels VIX as the curve", "items = [['T10Y3M', 'Curve'], ['VIXCLS', 'VIX'],",
     "items = [['VIXCLS', 'Curve'], ['T10Y3M', 'VIX'],"),
    ("the tile's ordinal suffix ignores the teens", "const s = (n % 100 >= 11 && n % 100 <= 13) ? 'th' :",
     "const s = false ? 'th' :"),
    ("money rounds whole millions", "if (a >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';",
     "if (a >= 1e6) return '$' + (v / 1e6).toFixed(0) + 'M';"),
]


def main() -> int:
    orig = GEN.read_bytes()
    text = orig.decode("utf-8")
    caught = 0
    try:
        for name, old, new in MUTATIONS:
            assert text.count(old) == 1, f"{name}: anchor found {text.count(old)} times"
            GEN.write_bytes(text.replace(old, new).encode("utf-8"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TEST],
                               cwd=ROOT, capture_output=True, text=True)
            hit = r.returncode != 0
            caught += hit
            print(f"{'CAUGHT' if hit else 'MISSED'}  {name}")
    finally:
        GEN.write_bytes(orig)
    print(f"{caught} of {len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
