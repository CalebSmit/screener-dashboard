"""Which run a published artifact came from, and what it was fetched from.

`OWNER_FOCUS.md` 2026-10-21, item 6 (screener-bot ROADMAP 1.10). A downstream
consumer trades the ranking this repo publishes. When a trade is questioned
afterwards the first question is always *which run produced this page, from
which code, over which raw fetch* - and until now the answer lived only in
``runs/<run_id>/meta.json``, which is gitignored scratch that
``scripts/prune_artifacts.py`` deletes after twenty runs. The page itself
carried a timestamp and (since 2026-10-22) a methodology version, and nothing
else.

So the four identifiers travel **with the page**, in ``kpis``:

``run_id``
    the ``runs/`` folder the page was generated from.
``git_sha``
    the commit the pipeline ran at, as ``run_context._get_git_sha`` recorded it.
``methodology_version``
    ``run_context.config_hash`` - every weight and rule that can move a score.
    Published since 2026-10-22; named here because it belongs to the same set.
``raw_sha256``
    the run's own ``00_raw_fetch.parquet``, the fetched record every metric is
    computed from.

**The raw hash is of this run's own fetch or it is absent.**
``generate_dashboard._find_raw_fetch`` deliberately falls back to a *previous*
run's file so a cache-hit run can still show price targets; that is right for
display and wrong for provenance - a fingerprint naming a file this run never
wrote is worse than no fingerprint. Same rule as the health gate, which has
judged only the run's own artifacts since 2026-10-16.

The hash is only an identifier while the file it names is scratch. What makes
it checkable is ``scripts/archive_run.py``, which copies each published run's
raw fetch out of the pruned working directory; the two were built together on
2026-10-24 for that reason.

``tests/test_run_fingerprint.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RAW_FETCH_NAME = "00_raw_fetch.parquet"

# A 5 MB-ish parquet hashed in 1 MB chunks: bounded memory, and the same
# number a `sha256sum` of the archived copy produces.
_CHUNK = 1 << 20


def sha256_file(path: Path | str) -> str:
    """The SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_fetch_sha256(run_dir: Path | str) -> str | None:
    """Hash of *this run's own* raw fetch, or None if it did not write one.

    No fallback to another run (see the module docstring). A cache-hit run
    honestly reports None rather than borrowing a neighbour's identity.
    """
    raw = Path(run_dir) / RAW_FETCH_NAME
    if not raw.is_file():
        return None
    return sha256_file(raw)


def fingerprint(run_dir: Path | str) -> dict:
    """The four identifiers for a run directory.

    Every value may be None: a run that crashed before ``save_metadata``, or a
    checkout with no git, still has to produce a page. A missing identifier is
    published as ``null`` and is a signal in its own right - it is never
    replaced with a guess.
    """
    run_dir = Path(run_dir)
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            meta = {}
    if not isinstance(meta, dict):
        meta = {}

    git_sha = meta.get("git_sha")
    # run_context._get_git_sha returns the string "unknown" rather than raising
    # when it cannot reach git. Publishing that word as if it were a commit
    # would read like a value; None reads like the absence it is.
    if git_sha == "unknown":
        git_sha = None

    return {
        "run_id": meta.get("run_id") or (run_dir.name or None),
        "git_sha": git_sha,
        "methodology_version": meta.get("methodology_version"),
        "raw_sha256": raw_fetch_sha256(run_dir),
    }
