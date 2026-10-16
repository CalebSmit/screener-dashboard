"""The published payload files are valid JSON inside their assignment (2026-10-16).

Python's ``json.dumps`` writes a non-finite float as a bare ``NaN`` / ``Infinity`` token. A
browser's JavaScript accepts that, so the page would never show it - but JSON does not, and the
trading bot (``CalebSmit/screener-bot``) parses ``dashboard_data.js`` as served: one ``NaN``
anywhere in ~6 MB and a strict parser refuses the whole file. The writers now go through
``generate_dashboard.payload_json``, which turns every non-finite float into ``null`` and passes
``allow_nan=False`` so that anything slipping past the sanitiser raises instead of publishing.

Measured on the day: the 2026-10-16 02:00 run's payload had **0** non-finite values, and
regenerating it with the new writer was byte-identical to the old one for both files - this is a
guard on a property that held by luck, not a change to anything published.
"""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as G  # noqa: E402


def _strict_loads(text: str):
    """json.loads that refuses NaN / Infinity / -Infinity, as JSON.parse and strict parsers do."""
    def refuse(token):
        raise ValueError(f"non-JSON constant {token}")
    return json.loads(text, parse_constant=refuse)


def _body(js: str) -> str:
    return js[js.index("{"):].rstrip().rstrip(";")


NASTY = {
    "kpis": {"run_timestamp": "2026-10-16T02:00:03", "spread": float("nan")},
    "table_data": [{"Ticker": "AAA", "Composite": 71.2, "pe": float("inf")},
                   {"Ticker": "BBB", "Composite": float("-inf"), "vals": (1.0, float("nan"))}],
    "stock_detail": {"AAA": {"price": 10.5, "raw": {"x": float("nan"), "y": [float("nan"), 2]},
                             "ctx": {"iv": float("nan")}}},
}


class TestSanitiser:
    def test_non_finite_floats_become_null_at_any_depth(self):
        out = G.json_safe(NASTY)
        assert out["kpis"]["spread"] is None
        assert out["table_data"][0]["pe"] is None
        assert out["table_data"][1]["Composite"] is None
        assert out["table_data"][1]["vals"] == [1.0, None]
        assert out["stock_detail"]["AAA"]["raw"] == {"x": None, "y": [None, 2]}
        assert out["stock_detail"]["AAA"]["ctx"]["iv"] is None

    def test_finite_values_are_untouched(self):
        clean = {"a": 1, "b": 2.5, "c": "NaN", "d": None, "e": True, "f": [0.0, -3.25]}
        assert G.json_safe(clean) == clean

    def test_a_payload_with_nan_serialises_to_null_and_parses_strictly(self):
        text = G.payload_json(NASTY)
        assert "NaN" not in text and "Infinity" not in text
        parsed = _strict_loads(text)
        assert parsed["kpis"]["spread"] is None
        assert parsed["stock_detail"]["AAA"]["price"] == 10.5

    def test_the_old_writer_is_what_this_replaced(self):
        """The defect, pinned: the default writer emits a token a strict parser refuses."""
        with pytest.raises(ValueError):
            _strict_loads(json.dumps(NASTY, default=str))

    def test_a_regression_past_the_sanitiser_raises_rather_than_publishing(self, monkeypatch):
        monkeypatch.setattr(G, "json_safe", lambda obj: obj)
        with pytest.raises(ValueError):
            G.payload_json({"x": float("nan")})

    def test_separators_and_default_are_preserved(self):
        """Byte-identity with the old writer wherever there was no NaN."""
        clean = {"a": [1, 2.5], "b": {"c": "d"}}
        assert G.payload_json(clean) == json.dumps(clean, default=str)
        assert G.payload_json(clean, separators=(",", ":")) == \
            json.dumps(clean, separators=(",", ":"), default=str)


class TestTheWritersUseIt:
    def test_both_payload_files_are_written_through_payload_json(self):
        src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
        body = src.split("def generate_dashboard(")[1].split("\ndef ")[0]
        assert "data_json = payload_json(_full)" in body
        assert "ctx_json = payload_json(ctx_payload" in body
        assert "json.dumps(_full" not in body and "json.dumps(ctx_payload" not in body

    def test_generate_dashboard_writes_strict_json_files(self, tmp_path, monkeypatch):
        """End to end through the real writer, with a NaN planted in the prepared payload."""
        planted = {"kpis": {"run_timestamp": "2026-10-16T02:00:03"},
                   "table_data": [{"Ticker": "AAA", "Composite": float("nan")}],
                   "stock_detail": {"AAA": {"price": float("inf"), "ctx": {"iv": float("nan")}}}}
        monkeypatch.setattr(G, "load_run_data", lambda run_dir: {"meta": {}})
        monkeypatch.setattr(G, "prepare_dashboard_data",
                            lambda run_data: json.dumps(planted, default=str))
        monkeypatch.setattr(G, "_load_methodology_html", lambda: "<p></p>")
        monkeypatch.setattr(G, "generate_html", lambda **kw: "<html></html>")
        G.generate_dashboard(tmp_path, tmp_path / "dashboard.html")

        data = _strict_loads(_body((tmp_path / "dashboard_data.js").read_text(encoding="utf-8")))
        ctx = _strict_loads(_body((tmp_path / G.CONTEXT_FILE).read_text(encoding="utf-8")))
        assert data["table_data"][0]["Composite"] is None
        assert data["stock_detail"]["AAA"]["price"] is None
        assert ctx["ctx"]["AAA"]["iv"] is None


@pytest.mark.parametrize("name", ["dashboard_data.js", "dashboard_context.js"])
def test_the_committed_payloads_parse_as_strict_json(name):
    """What the live site serves today - checked strictly, as the bot and JSON.parse read it."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    parsed = _strict_loads(_body(path.read_text(encoding="utf-8")))
    assert isinstance(parsed, dict) and parsed
    assert not any(isinstance(v, float) and not math.isfinite(v)
                   for v in (parsed.get("kpis") or {}).values())
