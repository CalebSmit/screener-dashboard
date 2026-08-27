"""The data-run commit subject must name the actual top-ranked stocks.

The commit log is audit trail, and the subject is what reaches the owner's
GitHub notification email. A wrong "top:" line is worse than no line, because
it looks authoritative.

This existed as a regex over the raw payload text - first five matches of
``"ticker": "XXX"`` - which worked only because the lowercase key belonged to
the model-portfolio holdings and those were serialised in rank order. Removing
that surface on 2026-08-26 left the same regex matching the first stock's
**sector peers**, and the 19:38 run committed ``top: MAA DOC KIM REG UDR`` when
the real top five were ``HST EXPE APA EIX CF``. HST is a REIT, so its peers are
REITs, and the wrong answer looked plausible enough to survive a glance.

Nothing failed. No gate caught it. It is exactly the failure mode this repo
keeps rediscovering: **the system reporting success while producing garbage.**
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import commit_subject as cs  # noqa: E402


def _payload() -> dict:
    """Ranked rows deliberately out of order, with peer blocks that would trap
    a positional scrape - the top stock's peers sort alphabetically ahead of it."""
    return {
        "kpis": {"universe_size": 502},
        "table_data": [
            {"Ticker": "EXPE", "Rank": 2},
            {"Ticker": "HST", "Rank": 1},
            {"Ticker": "CF", "Rank": 5},
            {"Ticker": "APA", "Rank": 3},
            {"Ticker": "EIX", "Rank": 4},
            {"Ticker": "ZZZ", "Rank": 6},
        ],
        "stock_detail": {
            "HST": {"peers": [{"ticker": "MAA"}, {"ticker": "DOC"},
                              {"ticker": "KIM"}, {"ticker": "REG"},
                              {"ticker": "UDR"}]},
        },
    }


def test_names_the_actual_top_five_in_rank_order():
    subject = cs.build_subject("2026-08-26", _payload())
    assert subject == "data: screener run 2026-08-26 - 502 scored, top: HST EXPE APA EIX CF"


def test_does_not_name_the_top_stocks_peers():
    """The exact regression. If peers ever reappear in the subject, this fails."""
    subject = cs.build_subject("2026-08-26", _payload())
    for peer in ("MAA", "DOC", "KIM", "REG", "UDR"):
        assert peer not in subject, "sector peer %s leaked into the commit subject" % peer


def test_input_order_does_not_matter():
    """Rank decides, not serialisation order - the assumption the old scrape
    silently depended on."""
    payload = _payload()
    payload["table_data"].reverse()
    assert "top: HST EXPE APA EIX CF" in cs.build_subject("2026-08-26", payload)


def test_scored_count_comes_from_kpis():
    assert "502 scored" in cs.build_subject("2026-08-26", _payload())


def test_falls_back_to_row_count_when_kpis_missing():
    payload = _payload()
    payload["kpis"] = {}
    assert "6 scored" in cs.build_subject("2026-08-26", payload)


@pytest.mark.parametrize("payload", [
    {},
    {"table_data": []},
    {"table_data": [{"Ticker": "AAA"}]},          # no Rank at all
    {"table_data": [{"Rank": 1}]},                # no Ticker
])
def test_degrades_to_uninformative_never_to_wrong(payload):
    """An unhelpful subject is acceptable. A confidently wrong one is not."""
    subject = cs.build_subject("2026-08-26", payload)
    assert subject.startswith("data: screener run 2026-08-26")


def test_cli_never_fails_the_caller(tmp_path, capsys):
    """A crash here must not stop a healthy run from publishing."""
    junk = tmp_path / "not-a-payload.js"
    junk.write_text("this is not JSON", encoding="utf-8")
    assert cs.main(["commit_subject.py", "2026-08-26", str(junk)]) == 0
    assert capsys.readouterr().out.strip() == "data: screener run 2026-08-26"


def test_cli_handles_a_real_payload(tmp_path, capsys):
    body = json.dumps(_payload())
    path = tmp_path / "dashboard_data.js"
    path.write_text("window.SCREENER_DATA = %s;" % body, encoding="utf-8")
    assert cs.main(["commit_subject.py", "2026-08-26", str(path)]) == 0
    assert "top: HST EXPE APA EIX CF" in capsys.readouterr().out


def test_data_run_no_longer_scrapes_tickers_with_a_regex():
    text = (REPO / "scripts" / "data-run.ps1").read_text(encoding="utf-8-sig")
    assert "commit_subject.py" in text
    assert '"ticker":' not in text, "the fragile payload scrape is back"
