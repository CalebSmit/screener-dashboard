"""The published run's inputs outlive the four-week prune.

`OWNER_FOCUS.md` 2026-10-21 item 6, second half. See
``scripts/archive_run.py`` for why the destination is outside OneDrive and
outside the repository. These tests hold three properties:

* the archive is **not reachable by the prune** that empties ``runs/``;
* the copy is **verified against the hash the page publishes**, so the archive
  and ``kpis.raw_sha256`` cannot drift apart silently;
* nothing in the archiver deletes anything.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import archive_run as A  # noqa: E402
import run_fingerprint as F  # noqa: E402


@pytest.fixture()
def run(tmp_path):
    d = tmp_path / "runs" / "abc123"
    d.mkdir(parents=True)
    (d / F.RAW_FETCH_NAME).write_bytes(b"the fetched record")
    (d / "run.log").write_text("ran\n", encoding="utf-8")
    (d / "config.yaml").write_text("weights: {}\n", encoding="utf-8")
    (d / "universe.json").write_text("{}", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "run_id": "abc123", "git_sha": "b" * 40,
        "methodology_version": "v3", "start_time": "2026-10-24T02:00:00",
    }), encoding="utf-8")
    return d


def test_it_copies_the_inputs_the_log_and_the_identity(run, tmp_path):
    dest = tmp_path / "archive"
    rec = A.archive(run, dest)
    assert rec["status"] == "archived"
    folder = dest / "2026-10-24_abc123"
    assert sorted(p.name for p in folder.iterdir()) == sorted(A.ARCHIVED_FILES)
    assert (folder / F.RAW_FETCH_NAME).read_bytes() == b"the fetched record"


def test_the_copy_is_verified_against_the_hash_the_page_publishes(run, tmp_path):
    dest = tmp_path / "archive"
    rec = A.archive(run, dest)
    assert rec["verified"] is True
    assert rec["raw_sha256"] == F.raw_fetch_sha256(run)


def test_a_short_or_damaged_copy_is_reported_not_declared_archived(run, tmp_path, monkeypatch):
    """A full disk writes less than it was asked to and says nothing."""
    dest = tmp_path / "archive"
    real = A.shutil.copy2

    def truncating(src, dst):
        real(src, dst)
        if Path(dst).name == F.RAW_FETCH_NAME:
            Path(dst).write_bytes(b"short")
    monkeypatch.setattr(A.shutil, "copy2", truncating)

    rec = A.archive(run, dest)
    assert rec["status"] == "corrupt"
    assert rec["copied_sha256"] != rec["raw_sha256"]


def test_archiving_twice_is_a_no_op(run, tmp_path):
    dest = tmp_path / "archive"
    A.archive(run, dest)
    again = A.archive(run, dest)
    assert again["status"] == "skipped"


def test_a_rewritten_run_is_recorded_as_a_replacement(run, tmp_path):
    """Not silently overwritten: a run directory whose raw fetch changed after
    it was archived is a fact worth having in the manifest."""
    dest = tmp_path / "archive"
    A.archive(run, dest)
    (run / F.RAW_FETCH_NAME).write_bytes(b"a different fetch")
    rec = A.archive(run, dest)
    assert rec["status"] == "archived" and rec["replaced"] is True


def test_the_manifest_is_append_only(run, tmp_path):
    dest = tmp_path / "archive"
    A.append_manifest(dest, A.archive(run, dest))
    A.append_manifest(dest, A.archive(run, dest))
    lines = (dest / A.MANIFEST).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "abc123"


def test_a_run_with_no_raw_fetch_archives_what_it_has(run, tmp_path):
    (run / F.RAW_FETCH_NAME).unlink()
    rec = A.archive(run, tmp_path / "archive")
    assert rec["status"] == "archived"
    assert rec["raw_sha256"] is None
    assert F.RAW_FETCH_NAME not in rec["files"]


def test_main_reports_size_and_never_prunes(run, tmp_path, capsys):
    dest = tmp_path / "archive"
    assert A.main(["--run-dir", str(run), "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "never pruned" in out
    assert (dest / "2026-10-24_abc123" / F.RAW_FETCH_NAME).exists()
    # A second invocation must not remove the first run.
    assert A.main(["--run-dir", str(run), "--dest", str(dest)]) == 0
    assert A.report(dest)["runs"] == 1


def test_nothing_in_the_archiver_deletes(tmp_path):
    src = (ROOT / "scripts" / "archive_run.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[-1]           # past the module docstring
    for forbidden in ("unlink", "rmtree", "os.remove", "shutil.move"):
        assert forbidden not in body, f"the archive must not {forbidden}"


# ---------------------------------------------------------------------------
# Where it lives
# ---------------------------------------------------------------------------

def test_the_default_destination_is_outside_the_repository_and_outside_onedrive(monkeypatch):
    monkeypatch.delenv(A.ENV_DEST, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\someone\AppData\Local")
    dest = A.default_dest()
    assert "OneDrive" not in str(dest)
    assert ROOT not in dest.parents and dest != ROOT


def test_the_destination_can_be_moved_without_editing_code(monkeypatch, tmp_path):
    monkeypatch.setenv(A.ENV_DEST, str(tmp_path / "elsewhere"))
    assert A.default_dest() == tmp_path / "elsewhere"


def test_the_prune_cannot_reach_the_archive():
    """The property that makes the archive an archive.

    ``prune_artifacts.py`` walks ``runs/`` and ``logs/`` under the repo root
    and nothing else. If it ever gained a configurable root, or the archive
    moved inside the repo, four-week retention would quietly come back.
    """
    import prune_artifacts as P
    assert P.RUNS == P.ROOT / "runs"
    assert P.LOGS == P.ROOT / "logs"

    archive = A.default_dest()
    for guarded in (P.RUNS, P.LOGS):
        assert guarded not in archive.parents and archive != guarded

    src = (ROOT / "scripts" / "prune_artifacts.py").read_text(encoding="utf-8")
    assert "ScreenerArchive" not in src and A.ENV_DEST not in src


def test_the_archive_is_never_committed():
    """It holds ~1.7 MB of parquet per weekday run. Under the repo it would be
    either a committed payload or a gitignored sibling of the pruned folder."""
    import subprocess
    dest = A.default_dest()
    try:
        inside = dest.is_relative_to(ROOT)
    except (AttributeError, ValueError):
        inside = False
    assert not inside
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", A.MANIFEST],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert tracked.returncode != 0, "the archive manifest is tracked in git"


def test_the_data_loop_archives_after_it_publishes():
    """Only a run that reached the live site is worth keeping, and archiving
    must never stand between a good run and the publish."""
    import re
    src = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8-sig")
    code = re.sub(r"(?m)#.*$", "", re.sub(r"<#.*?#>", "", src, flags=re.S))
    assert "archive_run.py" in code, "the data loop does not archive its run"
    assert code.index("Published to main") < code.index("archive_run.py")
    after = code[code.index("archive_run.py"):]
    assert "Stop-Run" not in after[:after.index("prune_artifacts.py")], \
        "a failed archive must not discard a published run"
