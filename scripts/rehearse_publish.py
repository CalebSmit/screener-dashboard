"""Would the next data run publish a page that passes the suite?

    python scripts/rehearse_publish.py            # the page's tests and the publish gate
    python scripts/rehearse_publish.py --full     # the whole suite

Why this exists. The ship gates run against the *committed* `index.html` and payload, which
only a full screener run rewrites. So a session that changes `generate_dashboard.py`, the
methodology generator or anything else the page is built from can pass all four gates and
still have changed what the 02:00 data loop will publish - and the first test run to see the
new page is the next session's baseline. On 2026-10-12 that is how a page 8 elements over its
first-paint budget came within four hours of making the 06:00 baseline red: gate 1 was green
all evening and was never looking at that evening's page. It was found by doing by hand what
this script does.

What it does:
1. copies the newest full run to a scratch folder (`logs/rehearse-publish/`, gitignored);
2. regenerates the dashboard from that copy with the working tree's code;
3. installs the regenerated page and payload over the published files;
4. runs the tests that read them, and `scripts/check_published_claims.py`;
5. **puts the published files back, whatever happened**, and checks they are byte-identical
   to what was there before.

It changes no score and publishes nothing. Run it whenever the diff touches a generator.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "logs" / "rehearse-publish"
PUBLISHED = ("dashboard.html", "index.html", "dashboard_data.js", "dashboard_context.js")
# The modules that read the generated page or payload. `--full` runs everything instead.
PAGE_TESTS = ("tests/test_dashboard_browser.py", "tests/test_dashboard_navigation.py",
              "tests/test_dashboard_surfaces.py", "tests/test_dashboard_js.py",
              "tests/test_index_is_the_dashboard.py", "tests/test_payload_parse_gate.py",
              "tests/test_bot_payload_contract.py", "tests/test_calculation_reproducibility.py",
              "tests/test_metric_lineage.py", "tests/test_holdings_panel.py",
              "tests/test_dashboard_interaction.py", "tests/test_club_readiness.py")
MIN_UNIVERSE = 495


def newest_full_run(runs: Path) -> Path:
    """The most recent run that scored the whole universe (a `--tickers` trial does not count)."""
    import pandas as pd
    full = []
    for d in runs.iterdir() if runs.exists() else []:
        scored = d / "05_final_scored.parquet"
        if scored.exists() and (d / "effective_weights.json").exists() and len(pd.read_parquet(scored)) >= MIN_UNIVERSE:
            full.append(d)
    if not full:
        raise SystemExit("no full run under runs/ to regenerate from; run the screener once first")
    return max(full, key=lambda d: (d / "05_final_scored.parquet").stat().st_mtime)


WRITE_ATTEMPTS = 6       # a synced folder can refuse a write for a moment (OneDrive, 2026-10-15)


def _write(p: Path, data: bytes | None, attempts: int = WRITE_ATTEMPTS, pause: float = 0.5) -> None:
    """Write (or, for None, remove) a file, retrying a refused write with a growing pause.

    2026-10-15: Windows refused a write to ``dashboard_data.js`` mid-rehearsal (``OSError: [Errno
    22] Invalid argument`` - the synced folder holding the file), the exception escaped the
    restore, and the published file was left as the rehearsal's version until it was restored by
    hand. A refused write is retried; one that keeps failing is still raised."""
    import time
    for i in range(attempts):
        try:
            if data is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(data)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(pause * (i + 1))


def with_installed(new_files: dict[Path, bytes], action):
    """Write `new_files`, run `action()`, and restore what was there - also when it raises.
    Returns (action's result, whether every file is back to its original bytes).

    Every file is restored on its own: a write that fails after its retries does not leave the
    files after it un-restored, and the failure is raised only once all have been tried."""
    before = {p: (p.read_bytes() if p.exists() else None) for p in new_files}
    try:
        for p, data in new_files.items():
            _write(p, data)
        result = action()
    finally:
        failed = []
        for p, data in before.items():
            try:
                _write(p, data)
            except OSError as exc:
                failed.append((p, exc))
        if failed:
            names = ", ".join(p.name for p, _ in failed)
            raise OSError(f"could not restore {names} after {WRITE_ATTEMPTS} attempts each - restore "
                          f"them with `git checkout -- <file>` before anything else") from failed[0][1]
    restored = all((p.read_bytes() if p.exists() else None) == data for p, data in before.items())
    return result, restored


def regenerate(run: Path, scratch: Path = SCRATCH) -> dict[str, bytes]:
    """The four published files as the working tree's generator would write them."""
    sys.path.insert(0, str(ROOT))
    import generate_dashboard as gd
    copy = scratch / "run"
    shutil.copytree(run, copy, dirs_exist_ok=True)       # a synced folder can refuse removal; overwrite
    site = scratch / "site"
    site.mkdir(parents=True, exist_ok=True)
    gd.generate_dashboard(copy, site / "dashboard.html")
    page = (site / "dashboard.html").read_bytes()
    out = {"dashboard.html": page, "index.html": page}
    for name in ("dashboard_data.js", "dashboard_context.js"):
        if (site / name).exists():
            out[name] = (site / name).read_bytes()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--full", action="store_true", help="run the whole suite, not only the page's tests")
    args = ap.parse_args(argv)

    run = newest_full_run(ROOT / "runs")
    print(f"regenerating from a copy of run {run.name} with the working tree's code")
    files = regenerate(run)
    for name, data in files.items():
        old = (ROOT / name).read_bytes() if (ROOT / name).exists() else b""
        print(f"  {name}: {len(data):,} bytes ({'unchanged' if data == old else 'differs from the published file'})")

    def tests() -> list[tuple[str, int]]:
        targets = ["tests/", "test_screener.py"] if args.full else [t for t in PAGE_TESTS if (ROOT / t).exists()]
        results = [("pytest " + ("(full suite)" if args.full else "(page tests)"),
                    subprocess.run([sys.executable, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider"],
                                   cwd=ROOT).returncode)]
        results.append(("check_published_claims",
                        subprocess.run([sys.executable, "scripts/check_published_claims.py"], cwd=ROOT).returncode))
        return results

    results, restored = with_installed({ROOT / n: d for n, d in files.items()}, tests)
    print()
    for name, code in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {name}")
    print(f"  {'PASS' if restored else 'FAIL'}  published files restored byte-for-byte")
    ok = restored and all(code == 0 for _, code in results)
    print("\nREHEARSAL " + ("PASSED: the next data run's page passes these tests."
                            if ok else "FAILED: fix it before the data loop publishes it."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
