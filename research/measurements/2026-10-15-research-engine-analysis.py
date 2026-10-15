"""The first research-engine batch, read three ways (research/2026-10-15-research-engine-first-batch.md).

Reads only ``research/experiments/ledger.jsonl`` (the latest run of each id) and the registry, so it
reproduces from committed files. Prints:

1. every experiment's held-out IC difference against the baseline, sorted, with its t;
2. the power of the design - the smallest held-out IC difference each comparison could have
   detected, at the conventional bar (80% power, two-sided 5%) and at the bar the engine applies
   (Bonferroni over the batch and |t| >= 3);
3. whether discovery and held-out agreed in sign more often than chance would give.

Not decision-grade (CLAUDE.md rule 5).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "research" / "experiments" / "ledger.jsonl"
REGISTRY = ROOT / "research" / "experiments" / "registry.yaml"


def main() -> None:
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["experiments"]
    latest = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    rows = []
    for e in reg:
        r = latest.get(e["id"])
        if not r or e["transform"]["kind"] == "baseline":
            continue
        s = r["summary"]
        rows.append((e["id"], s["discovery"]["ic"]["mean"], s["holdout"]["ic"]["mean"],
                     s["discovery"]["ic_vs_baseline"], s["holdout"]["ic_vs_baseline"],
                     s["holdout"]["top_minus_universe"]["mean"], s["holdout"]["top_vol"]))
    n = len(rows)
    print(f"{n} experiments against the baseline\n")
    print(f"{'experiment':28} {'IC disc':>8} {'IC hold':>8} {'d disc':>8} {'t':>6} {'d hold':>8} {'t':>6} "
          f"{'top-univ/mo':>11} {'top vol':>8}")
    for rid, icd, ich, dd, dh, tmu, vol in sorted(rows, key=lambda x: -x[4]["mean"]):
        print(f"{rid:28} {icd:+8.4f} {ich:+8.4f} {dd['mean']:+8.4f} {dd['t']:+6.2f} {dh['mean']:+8.4f} "
              f"{dh['t']:+6.2f} {tmu * 100:+10.2f}% {vol:8.4f}")

    # 2. power: a paired difference with standard error se is detected with 80% power at a
    # two-sided bar z when the true difference is (z + 0.84) * se
    z_bonf = _z_two_sided(0.05 / n)
    bar = max(z_bonf, 3.0)
    ses = sorted(r[4]["se"] for r in rows)
    med = ses[len(ses) // 2]
    print(f"\nheld-out paired SE: median {med:.4f}, range {ses[0]:.4f} .. {ses[-1]:.4f}")
    print(f"smallest IC difference detectable (80% power): conventional 5% -> {(1.96 + 0.84) * med:.4f}; "
          f"the engine's bar (z {bar:.2f}) -> {(bar + 0.84) * med:.4f}")

    # 3. sign agreement between the two periods
    agree = sum(1 for r in rows if math.copysign(1, r[3]["mean"]) == math.copysign(1, r[4]["mean"]))
    print(f"\ndiscovery and held-out differences agree in sign for {agree} of {n} "
          f"(chance: about {n / 2:.0f})")
    sig_d = sum(1 for r in rows if r[3]["p"] < 0.05)
    sig_h = sum(1 for r in rows if r[4]["p"] < 0.05)
    print(f"p < 0.05 against the baseline: {sig_d} in discovery, {sig_h} held out "
          f"(expected by chance alone: {0.05 * n:.1f} each)")


def _z_two_sided(alpha: float) -> float:
    """The normal quantile for a two-sided test at ``alpha``, by bisection on erfc."""
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    main()
