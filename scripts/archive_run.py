"""Keep each published run's raw fetch and log where nothing deletes them.

`OWNER_FOCUS.md` 2026-10-21 item 6, second half (screener-bot ROADMAP 1.10).

``kpis.raw_sha256`` names the fetched record a published ranking was computed
from. That is only an identifier while the file it names still exists, and it
does not: ``runs/`` is gitignored working scratch and
``scripts/prune_artifacts.py`` deletes all but the newest twenty runs after
every successful publish - about four weeks. So a trade questioned two months
later cannot be checked against its inputs at all, and the hash on the page
points at nothing.

**Why outside OneDrive, and outside the repository.** The repository lives
under ``OneDrive/Documents/Screener``. That is the wrong home for an audit
archive for three separate reasons, each of which has already bitten this
project or is structural:

* OneDrive rewrites folder mtimes when it syncs - which is exactly why
  ``run_context.latest_run_dir`` stopped trusting mtime on 2026-10-16;
* Files On-Demand can leave a synced file as a placeholder, so a read months
  later may need the network and may fail;
* a synced folder has a quota and a cloud-side recycle bin, neither of which
  this project controls.

And inside the repository it would be either committed (≈1.7 MB of parquet
five mornings a week, in a repo that packs to 35 MiB) or gitignored and
therefore sitting next to the directory prune already empties.

So the default is ``%LOCALAPPDATA%\\ScreenerArchive`` - local disk, outside the
synced tree, outside git, and outside everything ``prune_artifacts.py`` can
reach (it walks only ``runs/`` and ``logs/`` under the repo root;
``tests/test_run_archive.py`` fails if that ever stops being true).
``SCREENER_ARCHIVE_DIR`` overrides it.

**Nothing here is ever pruned**, on purpose. Four weeks of retention is the
problem being fixed; a cleverer retention rule would reintroduce it. The
archive reports its size every run instead, and says so loudly past a ceiling,
so the decision to delete is a person's.

Each archived run is a folder ``<run_date>_<run_id>/`` holding the raw fetch,
``run.log`` and ``meta.json``, plus a line appended to ``manifest.jsonl``. The
copy is **verified by re-hashing it**, and the hash recorded is the one
published in ``kpis.raw_sha256`` - an archive nobody checked against the page
is a folder of parquet, not evidence.

Usage::

    python scripts/archive_run.py              # archive the latest run
    python scripts/archive_run.py --run-dir runs/abc  --dest D:/somewhere
    python scripts/archive_run.py --report     # size and count only

Never fatal to the caller: the data loop calls it after the publish that
matters, and a failure to archive must not cost a day of evidence.
``tests/test_run_archive.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_fingerprint import RAW_FETCH_NAME, fingerprint, sha256_file  # noqa: E402

# What is worth keeping: the inputs, the narrative, and the identity.
# 05_final_scored.parquet is deliberately absent - it is reproducible from the
# raw fetch and the config, and it is the thing the published page already is.
ARCHIVED_FILES = (RAW_FETCH_NAME, "run.log", "meta.json", "config.yaml", "universe.json")

MANIFEST = "manifest.jsonl"
ENV_DEST = "SCREENER_ARCHIVE_DIR"

# Not a prune threshold - a number to say out loud. At ~2 MB a weekday run
# this is roughly four years.
SIZE_WARN_GB = 5.0


def default_dest() -> Path:
    """Local disk, outside the synced tree. See the module docstring."""
    env = os.environ.get(ENV_DEST)
    if env:
        return Path(env)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "ScreenerArchive"
    return Path.home() / ".screener-archive"


def _run_date(run_dir: Path, meta: dict) -> str:
    started = meta.get("start_time") or ""
    if isinstance(started, str) and len(started) >= 10:
        return started[:10]
    return datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d")


def archive(run_dir: Path, dest: Path) -> dict:
    """Copy one run's evidence. Returns a manifest record.

    Idempotent: a run already archived with the same raw hash is left alone and
    reported as ``skipped``. Re-archiving a run whose raw fetch has *changed*
    would mean the directory was rewritten, which is worth knowing, so that
    case is recorded rather than silently overwritten.
    """
    run_dir = Path(run_dir)
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}

    fp = fingerprint(run_dir)
    run_date = _run_date(run_dir, meta)
    name = f"{run_date}_{fp['run_id']}"
    target = dest / name

    record = {
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "run_date": run_date,
        "run_id": fp["run_id"],
        "git_sha": fp["git_sha"],
        "methodology_version": fp["methodology_version"],
        "raw_sha256": fp["raw_sha256"],
        "folder": name,
    }

    if target.is_dir():
        existing = target / RAW_FETCH_NAME
        if existing.is_file() and fp["raw_sha256"] and sha256_file(existing) == fp["raw_sha256"]:
            record["status"] = "skipped"
            record["files"] = sorted(p.name for p in target.iterdir())
            return record
        record["replaced"] = True

    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in ARCHIVED_FILES:
        src = run_dir / fname
        if src.is_file():
            shutil.copy2(src, target / fname)
            copied.append(fname)
    record["files"] = copied
    record["bytes"] = sum((target / f).stat().st_size for f in copied)

    # Verify the copy against the hash the page publishes. A copy nobody
    # checked is not evidence, and a short write on a full disk is silent.
    if fp["raw_sha256"]:
        got = sha256_file(target / RAW_FETCH_NAME)
        if got != fp["raw_sha256"]:
            record["status"] = "corrupt"
            record["copied_sha256"] = got
            return record
        record["verified"] = True
    record["status"] = "archived"
    return record


def append_manifest(dest: Path, record: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / MANIFEST, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def report(dest: Path) -> dict:
    """Count and size. Never deletes anything - see the module docstring."""
    if not dest.is_dir():
        return {"runs": 0, "bytes": 0, "dest": str(dest)}
    runs = [d for d in dest.iterdir() if d.is_dir()]
    total = sum(f.stat().st_size for d in runs for f in d.rglob("*") if f.is_file())
    return {"runs": len(runs), "bytes": total, "dest": str(dest)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--report", action="store_true", help="size and count only")
    args = ap.parse_args(argv)

    dest = Path(args.dest) if args.dest else default_dest()

    if not args.report:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            from run_context import latest_run_dir
            found = latest_run_dir(ROOT / "runs")
            if found is None:
                print("archive: no completed run to archive", file=sys.stderr)
                return 1
            run_dir = found
        try:
            record = archive(run_dir, dest)
            append_manifest(dest, record)
        except OSError as exc:
            print(f"archive: NOT archived - {exc}", file=sys.stderr)
            return 1
        if record["status"] == "corrupt":
            print(f"archive: the copy of {record['folder']} does not match the published "
                  f"hash - kept for inspection", file=sys.stderr)
            return 1
        print(f"archive: {record['status']} {record['folder']} "
              f"({len(record.get('files', []))} files)")

    stats = report(dest)
    gb = stats["bytes"] / (1 << 30)
    line = f"archive: {stats['runs']} run(s), {gb:.2f} GB at {stats['dest']} (never pruned)"
    if gb > SIZE_WARN_GB:
        print(f"{line} - past {SIZE_WARN_GB} GB; deleting from it is a person's decision",
              file=sys.stderr)
    else:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
