"""Can a page reproduce the engine's composites under *any* weighting from the payload alone?

The open question before investor-profile sliders (plan/investor-profiles.md, M4; CLAUDE.md
priority 6): a slider composite is computed in the browser, and row 0.8 says the page shows the
engine's numbers, never its own re-derivation. It can only satisfy both if the payload carries
everything the engine's ``compute_composite`` uses, so that a browser port reproduces it exactly
and a test can hold it to that. This measures it on the published profiles - weightings the
engine computed at build time - using nothing but ``dashboard_data.js``:

  composite = sum(score_c * w_c) / sum(w_c over categories with a score) x (1 - coverage discount)

with the weights the run's momentum regime applies (``weights.momentum_regime``), ranks by
descending composite with ties sharing the lower rank. 2026-11-03, owner-run night.

    python research/measurements/2026-11-03-profile-composites-from-payload.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def composites(sd: dict, w: dict) -> dict:
    out = {}
    for t, s in sd.items():
        sc = s.get("cat_scores") or {}
        have = {c: sc[c] for c in CATS if sc.get(c) is not None and w.get(c, 0) > 0}
        tot = sum(w[c] for c in have)
        if not tot:
            continue
        disc = (s.get("cov") or {}).get("disc") or 0.0
        out[t] = sum(have[c] * w[c] for c in have) / tot * (1 - disc)
    return out


def ranks(comp: dict) -> dict:
    vals = sorted(comp.values(), reverse=True)
    first = {}
    for i, v in enumerate(vals, 1):
        first.setdefault(v, i)
    return {t: first[v] for t, v in comp.items()}


def main() -> int:
    from factor_engine import apply_momentum_regime
    s = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    D = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    sd, P = D["stock_detail"], D["profiles"]
    regime = P.get("regime")
    print(f"regime {regime}; {len(sd)} stocks")
    worst = 0.0
    for prof in P["list"]:
        k = prof["key"]
        w = apply_momentum_regime(dict(prof["weights"]), regime)
        comp = composites(sd, w)
        rk = ranks({t: round(v, 6) for t, v in comp.items()})
        pub = P["c"].get(k) if k != "balanced" else {r["Ticker"]: [r["Composite"], r["Rank"]] for r in D["table_data"]}
        if not pub:
            print(f"{k}: no published composites to compare")
            continue
        dmax = max(abs(comp[t] - v[0]) for t, v in pub.items() if t in comp)
        rank_off = sum(1 for t, v in pub.items() if t in rk and rk[t] != v[1])
        missing = sorted(set(pub) ^ set(comp))
        worst = max(worst, dmax)
        print(f"{k:9s}: {len(pub)} stocks, largest composite difference {dmax:.4f} (published to 2 dp), "
              f"{rank_off} ranks differ, {len(missing)} stocks on one side only")
    print(f"verdict: {'reproduces' if worst <= 0.005 + 1e-9 else 'does NOT reproduce'} to the published precision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
