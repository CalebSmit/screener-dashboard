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
        if not r or e["transform"]["kind"] == "baseline" or e.get("void"):
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

    # 4. the period before the sample (registered before it was built)
    pre = [(e["id"], latest[e["id"]]["summary"]) for e in reg
           if e["id"] in latest and "presample" in latest[e["id"]]["summary"] and not e.get("void")]
    if not pre:
        return
    base = dict(pre).get("baseline")
    if base:
        b = base["presample"]
        print(f"\n2012-2019: baseline IC {b['ic']['mean']:+.4f} (t {b['ic']['t']}, n {b['ic']['n']}); "
              f"top tenth minus universe {b['top_minus_universe']['mean'] * 100:+.2f}% a month "
              f"(t {b['top_minus_universe']['t']}); 2020-2026 IC {base['all']['ic']['mean']:+.4f}")
    rows2 = [(i, s) for i, s in pre if "ic_vs_baseline" in s["presample"]]
    m = len(rows2)
    print(f"\n{'experiment':34} {'IC 12-19':>9} {'t':>6} {'d 12-19':>8} {'t':>6} {'d 20-26':>8} {'t':>6}")
    for i, s in sorted(rows2, key=lambda x: -x[1]["presample"]["ic_vs_baseline"]["mean"]):
        d, a2 = s["presample"]["ic_vs_baseline"], s["all"]["ic_vs_baseline"]
        print(f"{i:34} {s['presample']['ic']['mean']:+9.4f} {s['presample']['ic']['t']:+6.2f} "
              f"{d['mean']:+8.4f} {d['t']:+6.2f} {a2['mean']:+8.4f} {a2['t']:+6.2f}")
    agree2 = sum(1 for _, s in rows2 if math.copysign(1, s["presample"]["ic_vs_baseline"]["mean"])
                 == math.copysign(1, s["all"]["ic_vs_baseline"]["mean"]))
    sig2 = sum(1 for _, s in rows2 if s["presample"]["ic_vs_baseline"]["p"] < 0.05)
    bar2 = sum(1 for _, s in rows2 if s["presample"]["ic_vs_baseline"]["p"] < 0.05 / m
               and abs(s["presample"]["ic_vs_baseline"]["t"]) >= 3)
    ses2 = sorted(s["presample"]["ic_vs_baseline"]["se"] for _, s in rows2)
    print(f"\n2012-2019 vs 2020-2026 differences agree in sign for {agree2} of {m} (chance {m / 2:.1f}); "
          f"p < 0.05 for {sig2} (chance {0.05 * m:.1f}); past the engine's bar: {bar2}")
    med2 = ses2[len(ses2) // 2]
    print(f"2012-2019 paired SE median {med2:.4f}: detectable at the engine's bar (80% power) "
          f"{(max(_z_two_sided(0.05 / m), 3.0) + 0.84) * med2:.4f}")


def dependence(reg: list, latest: dict) -> None:
    """5. Are the months independent? Lag-1 autocorrelation of each paired difference, and how far
    the Newey-West t moves from the plain one (recorded by the engine from 2026-10-15, evening)."""
    rows = []
    for e in reg:
        r = latest.get(e["id"])
        if not r or e.get("void"):
            continue
        for p in ("holdout", "presample", "all"):
            d = r["summary"].get(p, {}).get("ic_vs_baseline") or r["summary"].get(p, {}).get("ic")
            if d and "ac1" in d:
                rows.append((e["id"], p, d["ac1"], d["t"], d["t_nw"]))
    if not rows:
        print("\n(no dependence statistics in this ledger yet)")
        return
    ac = [x[2] for x in rows]
    moved = [x for x in rows if abs(x[3]) > 0.5 and abs(x[4] - x[3]) / abs(x[3]) > 0.2]
    print(f"\nlag-1 autocorrelation of the compared series: median {sorted(ac)[len(ac) // 2]:+.3f}, "
          f"range {min(ac):+.3f} .. {max(ac):+.3f} over {len(rows)} series")
    print(f"Newey-West t differs from the plain t by more than 20% (where |t| > 0.5): {len(moved)}")
    for x in sorted(moved, key=lambda x: -abs(x[3]))[:8]:
        print(f"  {x[0]:32} {x[1]:9} ac1 {x[2]:+.2f}  t {x[3]:+.2f}  t_nw {x[4]:+.2f}")


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
    _reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["experiments"]
    _latest = {}
    for _line in LEDGER.read_text(encoding="utf-8").splitlines():
        if _line.strip():
            _r = json.loads(_line)
            _latest[_r["id"]] = _r
    dependence(_reg, _latest)
