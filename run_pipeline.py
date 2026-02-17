#!/usr/bin/env python3
"""
Multi-Factor Stock Screener — Full Pipeline Runner
====================================================
Single-command entry point that runs:
  1. factor_engine.py  → score the S&P 500 universe
  2. portfolio_constructor.py → build model portfolio + 3-sheet Excel

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(label: str, script: str):
    """Run a Python script as a subprocess, streaming output."""
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE STEP: {label}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n  *** FAILED: {script} exited with code {result.returncode} ***")
        sys.exit(result.returncode)


def main():
    t0 = time.time()

    print("=" * 60)
    print("  MULTI-FACTOR SCREENER — FULL PIPELINE")
    print("=" * 60)

    # Step 1: Factor Engine
    run_step("Factor Engine (scoring S&P 500)", "factor_engine.py")

    # Step 2: Portfolio Constructor + Excel
    run_step("Portfolio Constructor + Excel Dashboard", "portfolio_constructor.py")

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'=' * 60}")
    print(f"  FULL PIPELINE COMPLETE — {elapsed}s total")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
