"""
CLI argument parsing for the Multi-Factor Stock Screener.

This module handles all command-line argument parsing and validation.
"""

import argparse
from presets import list_presets


def parse_args(args=None):
    """Parse and return command-line arguments.

    Returns:
        argparse.Namespace with the following attributes:
        - refresh: bool — Force-clear cache and re-fetch
        - tickers: str — Comma-separated tickers for quick testing
        - no_portfolio: bool — Skip portfolio construction
        - dry_run: bool — Validate config and test connectivity (~10s)
        - show_weights: bool — Display effective weights and exit
        - preset: str — Apply a configuration preset (balanced/value/growth/momentum)
        - top_n: int — Number of top stocks for the portfolio
    """
    p = argparse.ArgumentParser(
        description="Multi-Factor Stock Screener v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full screener with default config
  python run_screener.py

  # Apply a growth tilt
  python run_screener.py --preset growth

  # Quick test with specific tickers
  python run_screener.py --tickers AAPL,MSFT,GOOGL

  # Validate setup without running the full pipeline (~10s)
  python run_screener.py --dry-run

  # Show effective weights for current config
  python run_screener.py --show-weights
        """)

    p.add_argument("--allow-synthetic", action="store_true",
                   help="Permit fabricated 'sector-realistic' values when the "
                        "network is unavailable. OFF by default: without this "
                        "the run refuses rather than emitting fiction that is "
                        "indistinguishable from analysis. For pipeline "
                        "validation only - never for anything published.")
    p.add_argument("--refresh", action="store_true",
                   help="Force-clear all cache and re-fetch everything")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated tickers for quick testing "
                        "(e.g. AAPL,MSFT,GOOGL)")
    p.add_argument("--no-portfolio", action="store_true",
                   help="Skip portfolio construction; only write FactorScores")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config, test network, and check output paths without running the full pipeline (~10s)")
    p.add_argument("--show-weights", action="store_true",
                   help="Display effective factor weights after config application and exit")
    p.add_argument("--preset", type=str, default=None,
                   choices=[None] + list_presets(),
                   help=f"Apply a configuration preset to override factor_weights. "
                        f"Options: {', '.join(list_presets())}")
    p.add_argument("--top-n", type=int, default=25,
                   help="Number of top stocks to include in the portfolio (default: 25)")

    return p.parse_args(args)
