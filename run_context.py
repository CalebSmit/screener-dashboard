#!/usr/bin/env python3
"""
Run Context — Reproducibility infrastructure for the Multi-Factor Screener.

Provides:
  - run_id generation (UUID4)
  - Config snapshot saving
  - Run metadata recording (timestamps, versions, parameters)
  - Intermediate data artifact saving
  - Structured JSON logging

Usage:
    ctx = RunContext()          # generates run_id, creates runs/{run_id}/
    ctx.save_config(cfg)       # snapshot config.yaml
    ctx.save_artifact("raw_data", df)   # save intermediate DataFrame
    ctx.log.info("message", extra={"ticker": "AAPL"})
    ctx.save_metadata({...})   # save final run metadata
"""

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        # Merge any extra fields (ticker, metric, value, etc.)
        for key in ("ticker", "metric", "value", "fetch_time_ms",
                     "phase", "step", "count", "run_id"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


class RunContext:
    """Manages a single pipeline run's metadata, artifacts, and logging."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.start_time = datetime.now()
        self.run_dir = RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Set up structured file logger
        self.log = logging.getLogger(f"screener.{self.run_id}")
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False

        # Remove existing handlers to avoid duplicates on re-init
        self.log.handlers.clear()

        # JSON file handler
        log_path = self.run_dir / "run.log"
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(_JSONFormatter())
        self.log.addHandler(fh)

        # Console handler (human-readable)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        ch.setLevel(logging.INFO)
        self.log.addHandler(ch)

        self.log.info("Run started", extra={"run_id": self.run_id})

    def save_config(self, cfg: dict) -> Path:
        """Save a snapshot of the config used for this run."""
        path = self.run_dir / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        self.log.info("Config snapshot saved", extra={"phase": "init"})
        return path

    def config_hash(self, cfg: dict) -> str:
        """Compute a deterministic hash of scoring-relevant config keys.

        Includes 'universe' so that two runs with different excluded tickers
        or different market-cap thresholds produce different cache keys.
        """
        relevant = {
            "factor_weights": cfg.get("factor_weights", {}),
            "metric_weights": cfg.get("metric_weights", {}),
            "data_quality": cfg.get("data_quality", {}),
            "sector_neutral": cfg.get("sector_neutral", {}),
            "value_trap_filters": cfg.get("value_trap_filters", {}),
            "universe": cfg.get("universe", {}),
        }
        raw = json.dumps(relevant, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def save_artifact(self, name: str, df: pd.DataFrame) -> Path:
        """Save an intermediate DataFrame as Parquet."""
        path = self.run_dir / f"{name}.parquet"
        df.to_parquet(str(path), index=False)
        self.log.info(f"Artifact saved: {name} ({len(df)} rows)",
                      extra={"phase": "artifact", "step": name, "count": len(df)})
        return path

    def save_universe(self, tickers: list, failed: list) -> Path:
        """Save the universe of tickers scored and failed."""
        data = {
            "scored": sorted(tickers),
            "failed": sorted(failed),
            "scored_count": len(tickers),
            "failed_count": len(failed),
        }
        path = self.run_dir / "universe.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def save_effective_weights(self, cfg: dict) -> Path:
        """Save the effective weights (after any auto-disable adjustments)."""
        data = {
            "factor_weights": cfg.get("factor_weights", {}),
            "metric_weights": cfg.get("metric_weights", {}),
        }
        path = self.run_dir / "effective_weights.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def save_metadata(self, extra: dict | None = None) -> Path:
        """Save run metadata (call at end of pipeline)."""
        end_time = datetime.now()
        meta = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_seconds": round((end_time - self.start_time).total_seconds(), 1),
            "git_sha": _get_git_sha(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": _get_package_versions(),
        }
        if extra:
            meta.update(extra)
        path = self.run_dir / "meta.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        self.log.info("Run metadata saved", extra={"run_id": self.run_id})
        return path


def _get_git_sha() -> str:
    """Get the current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _get_package_versions() -> dict:
    """Get versions of key dependencies."""
    versions = {}
    for pkg in ["yfinance", "pandas", "numpy", "scipy", "openpyxl",
                "pyyaml", "pyarrow", "pydantic"]:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions
