"""Print the data loop's daily evidence-base readout.

Why this is a file rather than a `python -c` here-string
-------------------------------------------------------
It used to be an inline ``@'...'@`` here-string passed to ``python -c`` from
``scripts/data-run.ps1``.  On 2026-09-02 the data run logged

    Improvement engine: 30 raw IC row(s) (effective count unavailable).

which is the *fallback* branch: the inline invocation had failed, and the
fallback printed the **raw row count** - the number ``CLAUDE.md`` rule 8 calls
out as "how this went wrong the first time".  The one number the loop prints
without anyone thinking about it was the misleading one.

Two things follow.  First, a multi-line argument containing quotes is fragile
to pass to a native executable from Windows PowerShell, so it should not be
done at all.  Second, and more important: logic embedded in a here-string
cannot be unit-tested, so nothing noticed it was broken.  As a file it is both
robust to argument quoting and directly testable - see
``tests/test_evidence_readout.py``.

Output is a single line on stdout.  Exit code is 0 on success, 1 if the
effective count genuinely cannot be computed - in which case the caller should
say so plainly rather than substituting the raw row count.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_readout() -> str:
    """Return the log line, quoting the EFFECTIVE observation count.

    The engine's gates read non-overlapping observations, not rows.  Reporting
    rows overstates the evidence base - at the 1m horizon on 2026-09-02 there
    were 8 rows but only 3 effective observations, against a gate of 8.
    """
    import pandas as pd

    import improvement_engine as ie

    ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
    horizon = ie._get_governance_config()["optimization_horizon"]
    at_horizon = ic.loc[ic["horizon"].astype(str) == horizon, "run_date"]
    effective = ie._effective_observations(at_horizon, horizon)
    gate = ie._get_governance_config().get("min_observations_for_proposal", 8)

    return (
        f"{effective} effective ({len(at_horizon)} rows) at the {horizon} "
        f"horizon; {len(ic)} rows across all horizons. "
        f"{gate} effective observations are needed before the engine may "
        f"propose a weight change."
    )


def main() -> int:
    try:
        print(build_readout())
    except Exception as exc:  # noqa: BLE001 - the caller only needs the status
        # Deliberately do NOT fall back to a raw row count. A wrong number that
        # looks authoritative is worse than an honest "could not compute".
        print(f"effective observation count unavailable: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
