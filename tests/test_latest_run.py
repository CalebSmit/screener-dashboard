"""One definition of "the latest run", keyed on meta.json start_time (2026-10-16).

``scripts/check_run_health.py`` picked the newest ``runs/`` folder by **filesystem mtime**;
``generate_dashboard._find_latest_run`` - which ``data-run.ps1`` runs straight after the health
gate - picked by ``meta.json`` ``start_time``, having moved off mtime because OneDrive rewrites
folder mtimes when it syncs. The gate could approve one run while the generator published
another. Both now call ``run_context.latest_run_dir``.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402
import generate_dashboard as G  # noqa: E402
import run_context as RC  # noqa: E402


def _run(runs: Path, name: str, start: str | None, mtime: float, meta: bool = True) -> Path:
    d = runs / name
    d.mkdir(parents=True)
    if meta:
        (d / "meta.json").write_text(json.dumps({"run_id": name, "start_time": start}),
                                     encoding="utf-8")
    os.utime(d, (mtime, mtime))
    return d


class TestTheRule:
    def test_start_time_wins_over_a_newer_mtime(self, tmp_path):
        """The OneDrive case: an older run's folder touched by a sync after the newer run."""
        _run(tmp_path, "older", "2026-10-15T02:00:00", mtime=9_000_000)
        _run(tmp_path, "newer", "2026-10-16T02:00:00", mtime=1_000_000)
        assert RC.latest_run_dir(tmp_path).name == "newer"

    def test_a_folder_without_meta_is_not_a_run(self, tmp_path):
        """A --dry-run husk or a run in progress has no meta.json yet."""
        _run(tmp_path, "real", "2026-10-15T02:00:00", mtime=1_000)
        husk = _run(tmp_path, "dryrun", None, mtime=9_000_000, meta=False)
        (husk / "config.yaml").write_text("x: 1", encoding="utf-8")
        assert RC.latest_run_dir(tmp_path).name == "real"

    def test_unreadable_start_time_and_test_folders_are_skipped(self, tmp_path):
        _run(tmp_path, "real", "2026-10-15T02:00:00", mtime=1_000)
        _run(tmp_path, "nostart", None, mtime=9_000_000)
        _run(tmp_path, "garbled", "not a date", mtime=9_000_000)
        _run(tmp_path, "test_fixture", "2027-01-01T00:00:00", mtime=9_000_000)
        bad = tmp_path / "badjson"
        bad.mkdir()
        (bad / "meta.json").write_text("{nope", encoding="utf-8")
        assert RC.latest_run_dir(tmp_path).name == "real"

    def test_none_when_there_is_no_run(self, tmp_path):
        assert RC.latest_run_dir(tmp_path) is None
        assert RC.latest_run_dir(tmp_path / "missing") is None

    def test_timezone_aware_stamps_compare_on_one_clock(self, tmp_path):
        _run(tmp_path, "naive", "2026-10-15T02:00:00", mtime=1_000)
        _run(tmp_path, "aware", "2026-10-16T07:00:00Z", mtime=1_000)
        assert RC.latest_run_dir(tmp_path).name == "aware"


class TestBothReadersShareIt:
    def _two_runs(self, root: Path) -> None:
        runs = root / "runs"
        _run(runs, "older", "2026-10-15T02:00:00", mtime=9_000_000)
        _run(runs, "newer", "2026-10-16T02:00:00", mtime=1_000_000)

    def test_the_health_gate_and_the_generator_name_the_same_run(self, tmp_path, monkeypatch):
        self._two_runs(tmp_path)
        monkeypatch.setattr(h, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(G, "ROOT", tmp_path)
        assert h.latest_run_dir().name == "newer"
        assert G._find_latest_run().name == "newer"

    def test_neither_reader_keeps_its_own_rule(self):
        """A second implementation is how the two drifted apart in the first place."""
        health = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
        gen = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
        h_body = health.split("def latest_run_dir(")[1].split("\ndef ")[0]
        g_body = gen.split("def _find_latest_run(")[1].split("\ndef ")[0]
        assert "run_context import latest_run_dir" in h_body and "st_mtime" not in h_body
        assert "run_context import latest_run_dir" in g_body and "st_mtime" not in g_body

    def test_the_generator_still_raises_when_there_is_no_run(self, tmp_path, monkeypatch):
        import pytest
        (tmp_path / "runs").mkdir()
        monkeypatch.setattr(G, "ROOT", tmp_path)
        with pytest.raises(FileNotFoundError):
            G._find_latest_run()
