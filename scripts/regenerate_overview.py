"""Rewrite the committed ``SCREENER_OVERVIEW.md`` from its generator.

`SCREENER_OVERVIEW.md` is generated (CLAUDE.md rule 10): `build_screener_overview()` in
`run_screener.py` writes it at step 11 of every full run, so the 02:00 data loop overwrites it
five mornings a week. A session that changes the generator must regenerate the committed file in
the same commit, or `tests/test_overview_is_generated.py`'s drift tripwire fails - and with it
the publish rehearsal (gate 5) and the data loop's own claims gate.

Doing that used to mean a full screener run or a hand-copied snippet. This writes exactly what
step 11 writes and nothing else: no fetch, no scoring, no payload, no other artifact.

Usage:
    python scripts/regenerate_overview.py          # write it
    python scripts/regenerate_overview.py --check  # exit 1 if it would change
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OVERVIEW = ROOT / "SCREENER_OVERVIEW.md"


def generated() -> str:
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return run_screener.build_screener_overview(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report whether the committed file matches, and change nothing")
    args = ap.parse_args()

    fresh = generated()
    on_disk = OVERVIEW.read_text(encoding="utf-8") if OVERVIEW.exists() else None

    if on_disk == fresh:
        print(f"SCREENER_OVERVIEW.md is already what the generator produces "
              f"({len(fresh):,} chars)")
        return 0

    if args.check:
        print("SCREENER_OVERVIEW.md DIFFERS from build_screener_overview(cfg) - "
              "run this script without --check")
        return 1

    # newline="" so the generator's own line endings are written verbatim: step 11 writes with
    # the default newline translation, and a mismatch here would show up as a whole-file diff.
    OVERVIEW.write_text(fresh, encoding="utf-8")
    was = f"{len(on_disk):,}" if on_disk is not None else "absent"
    print(f"SCREENER_OVERVIEW.md rewritten from the generator: {was} -> {len(fresh):,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
