"""Build a git origin+clone sandbox once, and give each test its own copy of it.

Why (2026-10-26): three modules - test_branch_sweep, test_brief_publish_safety and
test_gate_ordering - each built the same origin+clone sandbox from scratch for every test,
nine to twenty git processes apiece. On Windows a git process costs 0.1-0.5 s to start (more
under load), and those fixtures were ~12% of the suite's whole running time. The state each
test starts from was identical every time, so it is built once per module and copied.

A copy is the same repository, not an approximation of it: every object, ref, remote-tracking
ref, the index and the config come across byte for byte. The one absolute path inside - the
clone's ``origin`` URL - is pointed at the copy's own bare repository, so a test that pushes,
deletes a branch or moves ``main`` changes only its own copy and never the template or another
test's sandbox. Each test still receives ``tmp_path / "origin.git"`` and ``tmp_path / "work"``,
exactly the paths the fixtures used to create, and every assertion is unchanged.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ORIGIN = "origin.git"
WORK = "work"


def copy_sandbox(template: Path, dest: Path) -> tuple[Path, Path]:
    """Copy ``template/origin.git`` and ``template/work`` into ``dest``; rewire the clone."""
    origin, work = dest / ORIGIN, dest / WORK
    shutil.copytree(template / ORIGIN, origin)
    shutil.copytree(template / WORK, work)
    r = subprocess.run(["git", "-C", str(work), "remote", "set-url", "origin", str(origin)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"git remote set-url failed: {r.stderr}"
    # Prove the rewiring rather than trust it: the copy must never push into the template.
    url = subprocess.run(["git", "-C", str(work), "remote", "get-url", "origin"],
                         capture_output=True, text=True, timeout=120).stdout.strip()
    assert Path(url) == origin, f"sandbox copy still points at {url}"
    return origin, work
