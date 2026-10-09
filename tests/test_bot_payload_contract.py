"""The fields screener-bot reads from ``dashboard_data.js`` must stay published.

`CalebSmit/screener-bot` (private, separate repo, added 2026-10-09) trades
an account from this screener's rankings. It never imports this code: it reads
the payload GitHub Pages serves, through ``bot/signals.py``, and refuses to
trade when a field it needs is missing. So a refactor here that renames or
drops one of these fields would not break this site - it would silently stop
the bot. This module is the contract between the two repos.

Asserted against ``prepare_dashboard_data``'s output, not the committed
payload (CLAUDE.md rule 10). If you change a field below on purpose, change
``bot/signals.py`` in screener-bot in the same session.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402

CATEGORIES = ["valuation", "quality", "growth", "momentum",
              "risk", "revisions", "size", "investment"]

# Exactly what screener-bot's bot/signals.py reads.
TABLE_FIELDS = {"Ticker", "Rank", "Composite", "Sector",
                "Value_Trap_Flag", "Growth_Trap_Flag"}


def _frame() -> pd.DataFrame:
    df = pd.DataFrame({
        "Ticker": ["AAA", "BRK-B", "CCC"],
        "Company": ["Alpha Inc", "Berkshire", "Gamma Ltd"],
        "Sector": ["Information Technology", "Financials", "Energy"],
        "Composite": [80.0, 60.0, 60.0],
        "Rank": [1, 2, 2],
        "Value_Trap_Flag": [False, False, True],
        "Growth_Trap_Flag": [False, True, False],
        "_current_price": [101.5, 480.0, 12.25],
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = 50.0
    return df


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(g.prepare_dashboard_data({
        "df": _frame(),
        "meta": {"run_date": "2026-10-09", "start_time": "2026-10-09T02:00:00"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    }))


def test_run_timestamp_is_published_and_iso(payload):
    from datetime import datetime
    datetime.fromisoformat(payload["kpis"]["run_timestamp"])


def test_every_table_row_carries_what_the_bot_reads(payload):
    for row in payload["table_data"]:
        assert TABLE_FIELDS <= row.keys(), TABLE_FIELDS - row.keys()
        assert isinstance(row["Rank"], int)
        assert isinstance(row["Value_Trap_Flag"], bool)
        assert isinstance(row["Growth_Trap_Flag"], bool)


def test_ticker_spelling_is_the_yahoo_one(payload):
    # The bot maps BRK-B -> BRK.B for the broker; a different spelling here
    # would place orders in a symbol that does not exist.
    assert "BRK-B" in {r["Ticker"] for r in payload["table_data"]}


def test_ties_share_a_rank(payload):
    # The bot accepts competition ranking (1, 2, 2); a dense or reordered
    # ranking would change which stocks it buys.
    assert sorted(r["Rank"] for r in payload["table_data"]) == [1, 2, 2]


def test_stock_detail_carries_the_price_the_sim_trades_at(payload):
    for t, price in [("AAA", 101.5), ("BRK-B", 480.0), ("CCC", 12.25)]:
        assert payload["stock_detail"][t]["price"] == pytest.approx(price)
