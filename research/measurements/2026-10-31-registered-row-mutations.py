"""Claims-mutation audit, 2026-10-31 (13:00): does the *register* buy the coverage?

The 06:00 session wrote real checks for `clamp-bullet-second-printing` and
`share-class-switching-margin` and measured them with
`2026-10-31-open-row-mutations.py`, which runs **the module holding the check**. That
answers "does a test fail", which is not the question the publish gate asks. The gate's
membership is derived from `claims.py`'s `checked_by` (CLAUDE.md, row "gates"), and the
audit's own method step 3 says to run *only* that claim's `checked_by` - so a check the
register does not name is a `GAP-IN-REGISTER`, not a catch (`plan/claims-mutation-audit.md`,
"Finishing a gap means registering the fix too").

This script asks it that way: mutate the generator, then run exactly the node ids the named
claim lists. Run it before the registration to see the gap and after to see it closed.

    python research/measurements/2026-10-31-registered-row-mutations.py
    python research/measurements/2026-10-31-registered-row-mutations.py A D
    python research/measurements/2026-10-31-registered-row-mutations.py --before C D

`--before` drops the node ids this session added to `checked_by`, which is the state the
register was in when the row still read `open:`. It is how the gap is recorded rather than
asserted.

Restores `run_screener.py` in a ``finally`` and asserts it byte-for-byte.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import claims  # noqa: E402

SRC = pathlib.Path("run_screener.py")

# (id, what it breaks, find, replace, the claim whose checked_by must see it)
MUTATIONS = [
    ("A", "the clamp bullet prints the forward-EPS ceiling 10 points high",
     "+{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[1] * 100)}%]",
     "+{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[1] * 100) + 10}%]",
     "overview.clamps_and_floors"),
    ("B", "the clamp bullet prints the price-target floor as the ceiling",
     "price target upside is clamped to [{int(clamps.get('price_target_upside', [-0.50, 1.0])[0] * 100)}%",
     "price target upside is clamped to [{-int(clamps.get('price_target_upside', [-0.50, 1.0])[1] * 100)}%",
     "overview.clamps_and_floors"),
    ("C", "the switching margin is typed into the page instead of read from config",
     'f"{_sc_margin:.0%} (S&P DJI\'s own switching margin): measured over 89 "',
     '"20% (S&P DJI\'s own switching margin): measured over 89 "',
     "overview.share_classes"),
    ("D", "the page's fallback margin drifts from share_classes.CHALLENGE_MARGIN",
     '_sc_margin = float(_sc_cfg.get("challenge_margin", _sc_mod.CHALLENGE_MARGIN))',
     '_sc_margin = float(_sc_cfg.get("challenge_margin", 0.25))',
     "overview.share_classes"),
]


ADDED_TODAY = "TestThePagePrintsTheMargin"


def checked_by(claim_id: str, before: bool = False) -> list[str]:
    claim = next(c for c in claims.CLAIMS if c.id == claim_id)
    nodes = list(claim.checked_by)
    return [n for n in nodes if ADDED_TODAY not in n] if before else nodes


def run(node_ids: list[str]) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *node_ids, "-q", "--no-header",
                        "--tb=no", "-rf"], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
    out = lines[-1].strip() if lines else f"(no summary, exit {r.returncode})"
    failed = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("FAILED ")]
    if failed:
        out += "\n           failed: " + ", ".join(sorted(set(failed)))
    return bool(failed), out


def main(argv: list[str]) -> int:
    before = "--before" in argv
    only = [a for a in argv if a in {m[0] for m in MUTATIONS}]
    original = SRC.read_text(encoding="utf-8")
    worst = 0
    try:
        print("register state:", "before this session's registration" if before else "as committed")
        for mid, what, find, replace, claim_id in MUTATIONS:
            if only and mid not in only:
                continue
            nodes = checked_by(claim_id, before)
            if original.count(find) != 1:
                print(f"SKIPPED {mid}: anchor appears {original.count(find)} times - {what}")
                worst = 1
                continue
            SRC.write_text(original.replace(find, replace), encoding="utf-8")
            caught, summary = run(nodes)
            print(f"{'CAUGHT ' if caught else 'GAP    '} {mid}: {what}\n"
                  f"           claim {claim_id}, {len(nodes)} checked_by node(s)\n"
                  f"           {summary}")
            if not caught and not before:
                worst = 1
            SRC.write_text(original, encoding="utf-8")
    finally:
        SRC.write_text(original, encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") == original, "failed to restore run_screener.py"
        print("run_screener.py restored byte-for-byte")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
