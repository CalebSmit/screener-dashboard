"""Tests for scripts/prune_artifacts.py.

This script deletes files unattended, so it gets tested like one that does.

Two real bugs from its first hour, both pinned below:

1. **Husks displaced real runs.** Emptying a directory gives it a fresh mtime.
   A naive newest-first sort then ranked the husks above the populated runs,
   and the next pass deleted the 20 directories the retention count had just
   protected. Nothing of value was lost - `runs/` is gitignored scratch - but
   the logic was inverted.

2. **Read-only directories.** OneDrive marks synced folders read-only.
   `os.rmdir` honours that and fails WinError 5 on an already-empty directory,
   so 44 husks accumulated while the script reported success.
"""

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "prune_artifacts", ROOT / "scripts" / "prune_artifacts.py")
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)


@pytest.fixture
def fake_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(prune, "RUNS", runs)
    monkeypatch.setattr(prune, "LOGS", tmp_path / "logs")
    return runs


def _populated(runs: Path, name: str, mtime: float) -> Path:
    d = runs / name
    d.mkdir()
    (d / "meta.json").write_text("{}", encoding="utf-8")
    os.utime(d, (mtime, mtime))
    return d


def _husk(runs: Path, name: str, mtime: float) -> Path:
    d = runs / name
    d.mkdir()
    os.utime(d, (mtime, mtime))
    return d


class TestRetention:
    def test_keeps_the_newest_populated_runs(self, fake_runs):
        for i in range(10):
            _populated(fake_runs, f"run{i:02d}", 1_000_000 + i)
        n, _ = prune.prune_runs(keep=4, dry=False)
        survivors = sorted(p.name for p in fake_runs.iterdir())
        assert n == 6
        assert survivors == ["run06", "run07", "run08", "run09"]

    def test_husks_never_displace_populated_runs(self, fake_runs):
        """The 2026-08-25 bug: fresh-mtime husks outranked the real runs."""
        for i in range(3):
            _populated(fake_runs, f"real{i}", 1_000_000 + i)
        # Husks with far NEWER mtimes, as an emptying pass would leave them.
        for i in range(5):
            _husk(fake_runs, f"husk{i}", 9_000_000 + i)

        prune.prune_runs(keep=3, dry=False)
        survivors = sorted(p.name for p in fake_runs.iterdir())
        assert survivors == ["real0", "real1", "real2"], (
            "husks with newer mtimes pushed the populated runs out of the "
            "retention window"
        )

    def test_dry_run_deletes_nothing(self, fake_runs):
        for i in range(6):
            _populated(fake_runs, f"run{i}", 1_000 + i)
        n, freed = prune.prune_runs(keep=2, dry=True)
        assert n == 4 and freed > 0
        assert len(list(fake_runs.iterdir())) == 6, "dry run deleted something"


class TestReadOnly:
    def test_removes_a_read_only_directory(self, fake_runs):
        """OneDrive marks synced folders read-only; os.rmdir refuses them."""
        d = _husk(fake_runs, "readonly", 1_000)
        os.chmod(d, stat.S_IREAD)
        try:
            prune.prune_runs(keep=0, dry=False)
            assert not d.exists(), (
                "a read-only empty directory survived - this is the WinError 5 "
                "case that left 44 husks behind"
            )
        finally:
            if d.exists():
                os.chmod(d, stat.S_IWRITE)

    def test_removes_a_directory_with_read_only_contents(self, fake_runs):
        d = _populated(fake_runs, "ro_contents", 1_000)
        os.chmod(d / "meta.json", stat.S_IREAD)
        try:
            prune.prune_runs(keep=0, dry=False)
            assert not d.exists()
        finally:
            if d.exists():
                for f in d.rglob("*"):
                    os.chmod(f, stat.S_IWRITE)


class TestSafety:
    def test_evidence_directories_are_named_as_protected(self):
        for name in ("improvement", "cache", "validation"):
            assert name in prune.PROTECTED_DIRS, (
                f"{name}/ must be protected - it holds the evidence base or "
                f"cache-freshness state"
            )

    def test_refuses_absurd_retention(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["prune", "--keep-runs", "1"])
        assert prune.main() == 2
        assert "Refusing" in capsys.readouterr().out

    def test_only_prunes_run_logs_not_state_files(self, tmp_path, monkeypatch):
        logs = tmp_path / "logs"
        logs.mkdir()
        monkeypatch.setattr(prune, "LOGS", logs)
        monkeypatch.setattr(prune, "RUNS", tmp_path / "runs")
        for i in range(5):
            f = logs / f"datarun-2026-08-{i:02d}.log"
            f.write_text("x", encoding="utf-8")
            os.utime(f, (1_000 + i, 1_000 + i))
        marker = logs / ".datarun-last-success"
        marker.write_text("2026-08-25", encoding="utf-8")
        lock = logs / ".nightly.lock"
        lock.write_text("123", encoding="utf-8")

        prune.prune_logs(keep=2, dry=False)
        assert marker.exists(), "the once-per-day success marker was deleted"
        assert lock.exists(), "the lock file was deleted"
        assert len(list(logs.glob("datarun-*.log"))) == 2
