"""The page shows a company's full name (2026-10-16).

The source caps ``shortName`` at 31 characters, and on the 2026-10-16 run that cut 78 of 501 names
mid-word on the public page ("Booking Holdings Inc. Common St", "Cincinnati Financial Corporatio").
``longName`` is the full name from the same ``.info`` dict the fetch already downloads, so the
engine's ``Company`` prefers it and falls back to ``shortName``, then the ticker. Display only: no
metric may read either field.
"""
import math
from pathlib import Path

import factor_engine
import pit_record

ROOT = Path(__file__).resolve().parent.parent


def _company(d):
    return factor_engine._coalesce(d, "longName", "shortName", "Ticker")


def test_company_prefers_the_full_name():
    assert _company({"Ticker": "JBHT", "shortName": "J.B. Hunt Transport Services, I",
                     "longName": "J.B. Hunt Transport Services, Inc."}) == "J.B. Hunt Transport Services, Inc."


def test_company_falls_back_when_the_full_name_is_missing_or_nan():
    assert _company({"Ticker": "X", "shortName": "X Corp"}) == "X Corp"
    assert _company({"Ticker": "X", "shortName": "X Corp", "longName": math.nan}) == "X Corp"
    assert _company({"Ticker": "X", "longName": None}) == "X"


def test_compute_metrics_builds_company_this_way():
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    assert '"Company": _coalesce(d, "longName", "shortName", "Ticker")' in src
    assert 'rec["longName"]           = _safe(info, "longName")' in src


def test_the_name_fields_are_static_and_scored_nowhere():
    assert "longName" in pit_record.FIELD_CLASSES["static"]
    # every use of the string is the fetch line or the Company line - nothing computes from it
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    uses = [ln.strip() for ln in src.splitlines() if '"longName"' in ln]
    assert len(uses) == 2, uses
