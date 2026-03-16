"""Tests for CLI argument parsing: --dry-run, --show-weights, --preset, --tickers."""
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable (conftest.py also sets this up, but be explicit)
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import parse_args


# ---------------------------------------------------------------------------
# parse_args basic flags
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = parse_args([])
    assert args.dry_run is False
    assert args.show_weights is False
    assert args.preset is None
    assert args.tickers is None
    assert args.top_n == 25


def test_parse_args_dry_run():
    args = parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_show_weights():
    args = parse_args(["--show-weights"])
    assert args.show_weights is True


def test_parse_args_preset_valid():
    for preset_name in ("balanced", "value", "growth", "momentum"):
        args = parse_args(["--preset", preset_name])
        assert args.preset == preset_name


def test_parse_args_preset_invalid():
    with pytest.raises(SystemExit):
        parse_args(["--preset", "nonexistent_preset"])


def test_parse_args_tickers():
    args = parse_args(["--tickers", "AAPL,MSFT,GOOG"])
    assert args.tickers == "AAPL,MSFT,GOOG"


def test_parse_args_top_n():
    args = parse_args(["--top-n", "50"])
    assert args.top_n == 50


def test_parse_args_combined_flags():
    args = parse_args(["--dry-run", "--show-weights", "--preset", "value", "--tickers", "AAPL"])
    assert args.dry_run is True
    assert args.show_weights is True
    assert args.preset == "value"
    assert args.tickers == "AAPL"


# ---------------------------------------------------------------------------
# Preset system
# ---------------------------------------------------------------------------

def test_preset_weights_sum_to_100():
    from presets import PRESETS
    for name, preset in PRESETS.items():
        total = sum(preset["factor_weights"].values())
        assert total == 100, f"Preset '{name}' weights sum to {total}, not 100"


def test_get_preset_returns_dict():
    from presets import get_preset
    preset = get_preset("value")
    assert preset is not None
    assert "factor_weights" in preset
    assert "name" in preset


def test_get_preset_case_insensitive():
    from presets import get_preset
    assert get_preset("VALUE") is get_preset("value")


def test_get_preset_unknown_returns_none():
    from presets import get_preset
    assert get_preset("unknown_xyz") is None


def test_list_presets():
    from presets import list_presets
    presets = list_presets()
    assert set(presets) == {"balanced", "value", "growth", "momentum"}


def test_apply_preset_modifies_weights():
    from presets import apply_preset
    cfg = {"factor_weights": {"valuation": 10, "quality": 10}}
    result = apply_preset(cfg, "value")
    assert result["factor_weights"]["valuation"] == 30
    assert result["factor_weights"]["quality"] == 25


def test_apply_preset_does_not_mutate_original():
    from presets import apply_preset
    cfg = {"factor_weights": {"valuation": 10}}
    apply_preset(cfg, "growth")
    assert cfg["factor_weights"]["valuation"] == 10


def test_apply_preset_unknown_raises():
    from presets import apply_preset
    with pytest.raises(ValueError, match="Unknown preset"):
        apply_preset({}, "bogus")
