"""`scripts/rehearse_publish.py`: see the next data run's page before the data run does.

The ship gates read the committed page, which only a full run rewrites, so a generator change
can pass all four and still change what gets published. On 2026-10-12 that put a page 8
elements over its first-paint budget four hours from the 02:00 loop. The script regenerates the
page with the working tree's code, installs it, runs the tests that read it, and restores the
published files. These tests pin the part that must never go wrong: the restore.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rehearse_publish", ROOT / "scripts" / "rehearse_publish.py")
rp = importlib.util.module_from_spec(spec)
sys.modules["rehearse_publish"] = rp
spec.loader.exec_module(rp)


def test_the_published_files_come_back_even_when_the_tests_blow_up(tmp_path):
    page, payload, fresh = tmp_path / "index.html", tmp_path / "dashboard_data.js", tmp_path / "new.js"
    page.write_bytes(b"published page")
    payload.write_bytes(b"published payload")
    seen = {}

    def boom():
        seen["page"], seen["new"] = page.read_bytes(), fresh.read_bytes()
        raise RuntimeError("a test crashed")

    with pytest.raises(RuntimeError):
        rp.with_installed({page: b"regenerated page", payload: b"regenerated payload", fresh: b"x"}, boom)
    assert seen == {"page": b"regenerated page", "new": b"x"}       # the action saw the regenerated files
    assert page.read_bytes() == b"published page" and payload.read_bytes() == b"published payload"
    assert not fresh.exists()                                       # a file that was not there is gone again


def test_a_clean_rehearsal_reports_that_everything_was_restored(tmp_path):
    page = tmp_path / "index.html"
    page.write_bytes(b"published")
    result, restored = rp.with_installed({page: b"regenerated"}, lambda: page.read_bytes())
    assert result == b"regenerated" and restored is True and page.read_bytes() == b"published"


def test_a_ticker_trial_is_never_mistaken_for_the_universe(tmp_path):
    import pandas as pd
    for name, rows in (("full_old", 501), ("subset_new", 50)):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"Ticker": [f"T{i}" for i in range(rows)]}).to_parquet(d / "05_final_scored.parquet")
        (d / "effective_weights.json").write_text("{}")
    assert rp.newest_full_run(tmp_path).name == "full_old"
    with pytest.raises(SystemExit):
        rp.newest_full_run(tmp_path / "nothing-here")


def test_it_installs_exactly_the_files_the_site_serves():
    assert set(rp.PUBLISHED) == {"dashboard.html", "index.html", "dashboard_data.js", "dashboard_context.js"}
    assert all((ROOT / t).exists() for t in rp.PAGE_TESTS), [t for t in rp.PAGE_TESTS if not (ROOT / t).exists()]
    src = (ROOT / "scripts" / "rehearse_publish.py").read_text(encoding="utf-8")
    assert "finally:" in src and "git push" not in src and "git commit" not in src      # it publishes nothing


def test_the_nightly_prompt_requires_it_when_a_generator_changes():
    prompt = (ROOT / "prompts" / "nightly.md").read_text(encoding="utf-8")
    assert "scripts/rehearse_publish.py" in prompt
