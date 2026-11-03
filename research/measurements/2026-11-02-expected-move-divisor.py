"""The expected-move card's divisor, measured on the published context file (2026-11-02 late,
plan/claims-mutation-audit.md "Results, 2026-11-02 (owner-run, late)").

The drilldown says "Expected move = at-the-money call + put, divided by the price $<c.px>", where
c.px is this run's last close (`_ctx_last_close`). The figure it prints, c.om, is
`(call mid + put mid) / spot` from `context_signals.options_context`, and the spot the 20:00 option
refresh passes is `price_latest` from the newest run *at refresh time* (`options_cache.refresh`) -
the close before the session whose quotes it records. So the divisor the page names is not the one
the figure used whenever the price moved between those two closes, and the at-the-money strike was
chosen against the older price too. The straddle cost (c.ost) is published, so the divisor the
engine actually used is c.ost / c.om. Read-only; prints counts.

    python research/measurements/2026-11-02-expected-move-divisor.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    s = (ROOT / name).read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


ctx = load("dashboard_context.js")["ctx"]
rows = []
for t, c in ctx.items():
    if c.get("om") is None or not c.get("ost") or not c.get("px"):
        continue
    used = c["ost"] / c["om"]
    rows.append((abs(used / c["px"] - 1), abs(c["ok"] / c["px"] - 1) if c.get("ok") else None, t, c["px"], used,
                 c.get("ok"), c["om"], c.get("oqd")))
rows.sort(reverse=True)
gaps = [r[0] for r in rows]
print(f"stocks with an expected move: {len(rows)}")
print(f"divisor used vs the price the page names: median gap {statistics.median(gaps):.2%}; "
      f"> 1%: {sum(g > 0.01 for g in gaps)}; > 5%: {sum(g > 0.05 for g in gaps)}; > 10%: {sum(g > 0.10 for g in gaps)}")
far = [r for r in rows if r[1] is not None and r[1] > 0.05]
print(f"'at-the-money' strike more than 5% from the price the page names: {len(far)} "
      "(options_context refuses a strike more than 5% from the spot it was given)")
print("largest:")
for g, kg, t, px, used, k, om, d in rows[:8]:
    print(f"  {t:<6} page names ${px:,.2f}; divided by ${used:,.2f} ({g:.1%}); strike {k}; move printed +/-{om:.1%}; quotes of {d}")
