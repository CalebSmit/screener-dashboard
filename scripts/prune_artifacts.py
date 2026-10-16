#!/usr/bin/env python3
"""Prune unbounded local artifacts so the machine does not need a human.

`runs/` and `logs/` are gitignored working directories that every run appends
to and nothing ever removed. Measured 2026-08-25, three weeks into daily
running: `runs/` held 44 directories and **62 MB**, growing ~1.4 MB per run.
Left alone that is a disk-space problem some months out, and the first anyone
would know about it is a failed run.

Deliberately conservative about what it will touch:

* **`improvement/` is never pruned.** Snapshots, `performance_history.csv` and
  `live_ic_history.csv` are the evidence base - the thing the whole project is
  accumulating. Old snapshots are *more* valuable, not less, because forward
  returns mature with age.
* **`cache/` is never pruned.** Its freshness rules are load-bearing and were
  the subject of a real defect (2026-08-13); a second mechanism deleting from
  it would be a good way to reintroduce that.
* **`validation/` is never pruned.** Published provenance.
* The newest run directory is always kept - `generate_dashboard.py` resolves
  "latest run" from it.

Usage:
    python scripts/prune_artifacts.py [--keep-runs N] [--keep-logs N] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_context import run_start_time  # noqa: E402 - the one "when did this run start"

RUNS = ROOT / "runs"
LOGS = ROOT / "logs"

DEFAULT_KEEP_RUNS = 20
DEFAULT_KEEP_LOGS = 60
# A runs/ folder with no meta.json that nothing has written to for this long is a dry run or a
# crashed run, not one in progress (a full run takes ~20 minutes to reach meta.json).
HUSK_MIN_AGE_HOURS = 6

# Never delete these, whatever the retention count says.
PROTECTED_DIRS = {"improvement", "cache", "validation", "research", "plan", "prompts"}


def _mb(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB"


def _dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def _clear_readonly(func, path, _exc):
    """shutil.rmtree onerror hook: clear the read-only bit and retry.

    OneDrive marks synced directories read-only. `os.rmdir` honours that and
    fails with WinError 5 (Access is denied) on a directory whose contents are
    already gone, while `rmdir` from Git Bash succeeds because the POSIX layer
    clears the attribute first. Diagnosed 2026-08-25 after two wrong guesses -
    it is not a file lock and not OneDrive holding a handle.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _remove_tree(d: Path) -> bool:
    """Remove a directory and report honestly whether it went.

    `shutil.rmtree(ignore_errors=True)` deleted the *contents* but left the
    directory itself on Windows - OneDrive keeps handles open on a folder it is
    syncing. The first version of this script did exactly that and left 24
    empty husks behind while reporting success, which is the same
    silently-wrong-but-looks-fine pattern this project keeps tripping over.
    """
    shutil.rmtree(d, onerror=_clear_readonly)
    if not d.exists():
        return True

    # Contents gone, directory left behind. Note `os.listdir`, not
    # `Path.iterdir()`: iterdir returns a generator whose scandir handle stays
    # open on Windows just long enough for the rmdir below to fail with
    # "being used by another process". The first version of this script blamed
    # OneDrive for that; it was this.
    try:
        for name in sorted(os.listdir(d), reverse=True):
            child = d / name
            if child.is_dir():
                shutil.rmtree(child, onerror=_clear_readonly)
            else:
                os.chmod(child, stat.S_IWRITE)
                child.unlink(missing_ok=True)
        os.chmod(d, stat.S_IWRITE)
        os.rmdir(d)
    except OSError:
        return False
    return True


def _newest_write(d: Path) -> float:
    """The latest mtime of the folder or anything in it - a run in progress keeps writing run.log."""
    newest = d.stat().st_mtime
    try:
        for f in d.rglob("*"):
            newest = max(newest, f.stat().st_mtime)
    except OSError:
        pass
    return newest


def _run_order_key(d: Path) -> datetime:
    """Newest-first order for completed runs: ``meta.json`` ``start_time``, the same key
    ``run_context.latest_run_dir`` uses, so the run the generator calls "latest" is always
    inside the retention window. A ``meta.json`` without a readable start falls back to mtime."""
    started = run_start_time(d)
    return started if started is not None else datetime.fromtimestamp(d.stat().st_mtime)


def prune_runs(keep: int, dry: bool, now: float | None = None) -> tuple[int, int]:
    if not RUNS.exists():
        return 0, 0
    now = time.time() if now is None else now
    dirs = [d for d in RUNS.iterdir() if d.is_dir()]

    # Partition first. An emptied husk gets a FRESH mtime from the deletion
    # that emptied it, so a naive newest-first sort ranks husks above the real
    # runs and the next pass deletes the very directories the retention count
    # was protecting. That happened on 2026-08-25: a second run of this script
    # wiped the 20 populated directories it had just kept. Nothing of value was
    # lost (runs/ is gitignored scratch and the evidence base lives in
    # improvement/), but the logic was wrong.
    #
    # **Only a folder with meta.json is a run** (2026-10-16). `run_screener.py
    # --dry-run` creates a folder holding config.yaml and run.log and exits
    # before step 10 writes meta.json; so does any run that crashes early. Those
    # are non-empty, so they used to count against the retention budget as if
    # they were runs: on 2026-10-16 runs/ held 34 such husks and ONE real run,
    # so a handful more dry runs would have evicted the newest real run - the
    # one `generate_dashboard.py` regenerates from and `check_run_health.py`
    # judges. A folder without meta.json never counts, and once nothing has
    # written to it for HUSK_MIN_AGE_HOURS it is pruned (a full run writes its
    # meta.json within the hour; a run in progress keeps writing run.log, so its
    # folder is never old enough). Empty directories go immediately, as before.
    runs, husks = [], []
    for d in dirs:
        if (d / "meta.json").exists():
            runs.append(d)
        elif not os.listdir(d) or (now - _newest_write(d)) > HUSK_MIN_AGE_HOURS * 3600:
            husks.append(d)

    runs.sort(key=_run_order_key, reverse=True)
    doomed = husks + runs[keep:]
    freed = 0
    for d in doomed:
        if d.name in PROTECTED_DIRS:          # belt and braces
            continue
        size = _dir_size(d)
        freed += size
        if not dry:
            _remove_tree(d)
    return len(doomed), freed


def prune_logs(keep: int, dry: bool) -> tuple[int, int]:
    if not LOGS.exists():
        return 0, 0
    # Only the timestamped run logs. Lock files and success markers are state,
    # not history, and are managed by the runners themselves.
    files = [f for f in LOGS.iterdir()
             if f.is_file() and (f.name.startswith(("datarun-", "nightly-")))]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    doomed = files[keep:]
    freed = 0
    for f in doomed:
        freed += f.stat().st_size
        if not dry:
            try:
                f.unlink()
            except OSError:
                pass
    return len(doomed), freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-runs", type=int, default=DEFAULT_KEEP_RUNS)
    ap.add_argument("--keep-logs", type=int, default=DEFAULT_KEEP_LOGS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.keep_runs < 5 or a.keep_logs < 10:
        print("Refusing: retention floors are 5 runs and 10 logs.")
        return 2

    nr, fr = prune_runs(a.keep_runs, a.dry_run)
    nl, fl = prune_logs(a.keep_logs, a.dry_run)

    verb = "would remove" if a.dry_run else "removed"
    print(f"Prune: {verb} {nr} run dir(s) ({_mb(fr)}) and {nl} log file(s) ({_mb(fl)}). "
          f"Kept newest {a.keep_runs} runs / {a.keep_logs} logs. "
          f"improvement/, cache/ and validation/ untouched.")

    if not a.dry_run and RUNS.exists():
        husks = [d for d in RUNS.iterdir() if d.is_dir() and not os.listdir(d)]
        if husks:
            print(f"  {len(husks)} empty director(ies) could not be removed "
                  f"(handles held, typically OneDrive). Contents are gone; they "
                  f"will be retried next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
