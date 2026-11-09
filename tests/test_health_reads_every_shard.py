"""The health gate judges the run's own artifacts - and since 2026-11-08 those are three files.

**What went wrong, measured rather than described.** `financials` moved out of the first-paint
payload into `dashboard_detail.js` on 2026-11-08. `sec_crosscheck.market_value_check` reads
`stock_detail[t]["financials"]["market_cap"]` - it is the share count every valuation ratio
rests on, checked against the companies' own filed count - and `check_run_health.load_payload`
read `dashboard_data.js` alone. So from the **first** run after that commit the market-value
line compared **0 stocks of 485** and printed `0/0 within 5% (0 over 10%)`, and an ATTENTION
line that fires above 60 can never reach it from zero. On the 2026-11-09 run's own artifacts:
0 compared from `dashboard_data.js`, 485 with the shards folded back, 20 over 10% - including
FOXA at ratio 1.3109, the stock the day's research note is about
(`research/measurements/2026-11-09-share-count-detector-review.py`).

**Why nothing saw it.** `tests/test_sec_crosscheck.py` builds its payload by hand with
`financials` inline, so it passes whatever the generator does with the field - the register's own
lesson (CLAUDE.md, 0.11: a claim whose check tests the plumbing instead of the assertion is an
unchecked claim) one level down, in a gate rather than a claim. And the rule for what may move
into the shard is written in terms of the **page's** read sites - "every surface that reads it is
the open sheet" - which is true of `financials` and says nothing about the payload's readers that
are not the page. `stock_detail` has those: this gate, and `sec_crosscheck` through it.

So these tests assert the fold two ways that a hand-built fixture cannot reach:

1. **round trip through the generator's own split functions** - build a payload, split it the way
   a run does, write the files a run writes, and require `load_payload` to give the payload back.
   A fourth shard, or a key moved out of one, fails here without anyone remembering to add a
   case;
2. **derived from `data-run.ps1`'s `$DataArtifacts`** - every `.js` payload file the loop
   publishes must be folded by `load_payload` or named in `KNOWN_UNFOLDED` with a reason. This is
   the parse gate's precedent (CLAUDE.md, settled row "gates": every `.js` in `$DataArtifacts` is
   named in both runners' check specs, after the context payload published unguarded for eight
   days).

And one test of the thing that actually broke: `market_value_check` compares every stock when the
payload arrives the way a run writes it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dashboard as gd  # noqa: E402
import check_run_health as h  # noqa: E402
import sec_crosscheck as X  # noqa: E402

CATEGORIES = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
# A `.js` file in $DataArtifacts that load_payload deliberately does not fold, with the reason.
# Empty, and emptying it quietly is not an option - the same convention as the publish gate's
# EXCLUDED.
KNOWN_UNFOLDED: dict[str, str] = {}


def _frame() -> pd.DataFrame:
    """Three stocks carrying a description, an industry, a market value and a price - the four
    kinds of thing the shards move, one of which is a number a gate reads."""
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC"],
        "Company": ["Alpha Inc", "Beta Corp", "Gamma Ltd"],
        "Sector": ["Information Technology", "Financials", "Energy"],
        "Composite": [80.0, 60.0, 55.0],
        "Rank": [1, 2, 3],
        "Value_Trap_Flag": [False, False, False],
        "Growth_Trap_Flag": [False, False, False],
        "_about": ["Alpha Inc designs and sells things worldwide.", np.nan,
                   "Gamma Ltd explores for and produces crude oil."],
        "_industry": ["Software - Infrastructure", "Banks - Diversified", "Oil & Gas E&P"],
        "_mcap": [2_100.0, 4_000.0, 900.0],
        "_current_price": [20.0, 40.0, 9.0],
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = 50.0
    return df


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A run folder written the way a run writes one: payload plus both shards."""
    data = json.loads(gd.prepare_dashboard_data({
        "df": _frame(), "meta": {"run_date": "2026-11-09"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    }))
    ctx = gd.split_context(data)
    det = gd.split_detail(data)
    d = tmp_path / "runs" / "deadbeef"
    d.mkdir(parents=True)
    (d / "dashboard_data.js").write_text(
        "window.SCREENER_DATA = " + json.dumps(data) + ";", encoding="utf-8")
    (d / gd.CONTEXT_FILE).write_text(
        "window.SCREENER_CONTEXT = " + json.dumps(ctx) + ";", encoding="utf-8")
    (d / gd.DETAIL_FILE).write_text(
        "window.SCREENER_DETAIL = " + json.dumps(det) + ";", encoding="utf-8")
    return d


# ------------------------------------------------------------------ 1. the round trip
def test_load_payload_gives_back_the_payload_the_generator_built(run_dir: Path):
    """The statement that does not need maintaining: what the gate reads is what the run made.

    This is the whole fold in one assertion. A key moved into a shard, a fourth shard added, or
    the fold dropped, and this fails - nobody has to remember to add a case for the field.

    **One field may legitimately not come back**, and the exception is narrow enough to state:
    `split_detail` omits a value with nothing in it (`_has_content`), because empty strings for
    500 stocks are ~10 KB of nothing and the page treats a missing entry and an empty one
    identically. `BBB` has a NaN description, which the generator cleans to `""`. So the
    assertion is: every key that comes back is unchanged, and every key that does not was empty
    before the split. A *non-empty* field that goes missing still fails.
    """
    whole = json.loads(gd.prepare_dashboard_data({
        "df": _frame(), "meta": {"run_date": "2026-11-09"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    }))
    loaded = h.load_payload(run_dir)
    assert loaded is not None
    dropped = 0
    for t, s in whole["stock_detail"].items():
        back = loaded["stock_detail"][t]
        for k, v in s.items():
            if k not in back:
                assert not gd._has_content(v), f"{t} lost {k}, which was not empty: {v!r}"
                dropped += 1
                continue
            assert back[k] == v, f"{t}'s {k} came back from the shards changed"
        assert not set(back) - set(s), f"{t} came back with keys the generator never wrote"
    assert dropped == 1, f"expected exactly BBB's empty description to be dropped, got {dropped}"
    for k in whole:
        assert k in loaded, f"the gate cannot see top-level {k}"


def test_every_field_the_shard_takes_is_folded_back(run_dir: Path):
    """Named per key, so a failure says which field went missing rather than only that one did."""
    loaded = h.load_payload(run_dir)
    missing = {t: [f for f in gd.DETAIL_KEYS if f not in s]
               for t, s in loaded["stock_detail"].items()}
    # AAA and CCC carry every field; BBB's description is NaN, so the shard omits it by design.
    assert missing["AAA"] == [] and missing["CCC"] == []
    assert "financials" in loaded["stock_detail"]["BBB"]


def test_the_fold_is_what_puts_them_back_not_the_first_file(run_dir: Path):
    """The premise of every test above: these fields really are absent before the fold.

    Settling a payload that already had the fields would pass whatever the fold does - the
    mistake `tests/test_sheet_warm_is_settled.py` records for the browser harness.
    """
    first = h._parse_payload_file(run_dir / "dashboard_data.js")
    for t, s in first["stock_detail"].items():
        for f in gd.DETAIL_KEYS:
            assert f not in s, f"{t} carries {f} before the fold - this test proves nothing"
        assert "ctx" not in s
    for k in gd.CONTEXT_KEYS:
        assert k not in first


def test_a_run_that_wrote_no_shard_still_loads(run_dir: Path):
    """Every run before 2026-10-08 wrote one payload file and every run before 2026-11-09 two.
    A shard that is simply absent is not a fault."""
    (run_dir / gd.DETAIL_FILE).unlink()
    (run_dir / gd.CONTEXT_FILE).unlink()
    loaded = h.load_payload(run_dir)
    assert loaded is not None and len(loaded["stock_detail"]) == 3


def test_a_shard_that_is_there_and_does_not_parse_raises(run_dir: Path):
    """The main payload's rule, for the same reason: this gate may not judge half a payload.
    A truncated shard is exactly the failure the size floor could not see on the main file."""
    (run_dir / gd.DETAIL_FILE).write_text("window.SCREENER_DETAIL = {\"detail\": {", encoding="utf-8")
    with pytest.raises(ValueError, match=gd.DETAIL_FILE):
        h.load_payload(run_dir)


# ------------------------------------------------------------------ 2. derived from $DataArtifacts
def _data_artifacts() -> list[str]:
    text = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8")
    block = re.search(r"\$DataArtifacts\s*=\s*@\((.*?)\)", text, re.S)
    assert block, "could not find $DataArtifacts in data-run.ps1"
    return re.findall(r"'([^']+)'", block.group(1))


def test_every_js_payload_the_loop_publishes_is_folded_or_named():
    """The parse gate's rule, for the gate that *reads* the payload rather than parsing it.

    `dashboard_context.js` was published to the live site every weekday for eight days with no
    parse and no header check because a hand-kept list did not name it. This is the same list,
    derived rather than transcribed.
    """
    folded = {getattr(gd, attr) for attr, *_ in h.SHARD_FOLD}
    js = [a for a in _data_artifacts() if a.endswith(".js")]
    assert js, "no .js artifacts found - the derivation is broken, not the gate"
    for name in js:
        if name == "dashboard_data.js":
            continue
        assert name in folded or name in KNOWN_UNFOLDED, (
            f"{name} is published every weekday and the health gate does not fold it back. "
            f"Fold it in check_run_health.SHARD_FOLD, or name it in KNOWN_UNFOLDED with the "
            f"reason a check reading its fields does not need it.")


def test_the_shard_names_come_from_the_generator_not_from_a_string():
    """Two literals for one filename are free to drift, and the gate would then read a file no
    run writes (the 2026-10-31 switching-margin finding, applied before it can happen)."""
    for attr, *_ in h.SHARD_FOLD:
        assert hasattr(gd, attr), f"check_run_health.SHARD_FOLD names {attr}, which the generator lost"
    health = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
    for name in ("dashboard_context.js", "dashboard_detail.js"):
        assert f'"{name}"' not in health and f"'{name}'" not in health, (
            f"{name} is written out in check_run_health.py as well as derived from the generator")


def test_nothing_is_excluded_quietly():
    """`KNOWN_UNFOLDED` is empty. If it ever is not, the reason is written beside the name."""
    for name, why in KNOWN_UNFOLDED.items():
        assert len(why) > 20, f"{name} is excluded without a reason"


# ------------------------------------------------------------------ 3. the check that broke
def _share_fact(t, end, val, filed="2026-11-01"):
    """One filed weighted-average share count, in the shape `sec_fundamentals` caches."""
    start = pd.Timestamp(end) - pd.Timedelta(days=90)
    return {"ticker": t, "concept": "WeightedAverageNumberOfDilutedSharesOutstanding",
            "start": start, "end": pd.Timestamp(end), "filed": pd.Timestamp(filed),
            "form": "10-Q", "fp": "", "val": float(val)}


def test_the_market_value_check_compares_every_stock_of_a_real_run(run_dir: Path):
    """The test that would have failed on 2026-11-08, and the one `_mv_payload` cannot be.

    `tests/test_sec_crosscheck.py` builds `financials` inline, so it reports the same thing
    whether or not the field is in the file the gate reads. This takes the payload the way a run
    writes it.
    """
    facts = pd.DataFrame([_share_fact("AAA", "2026-09-30", 100.0),
                          _share_fact("BBB", "2026-09-30", 100.0),
                          _share_fact("CCC", "2026-09-30", 100.0)])
    loaded = h.load_payload(run_dir)
    mv = X.market_value_check(loaded, facts)
    assert sorted(mv["ticker"]) == ["AAA", "BBB", "CCC"], (
        "the market-value check compared %d of 3 stocks - the 2026-11-08 failure" % len(mv))
    # AAA: 2,100 / 20 = 105 implied against 100 filed.
    assert float(mv.loc[mv["ticker"] == "AAA", "ratio"].iloc[0]) == pytest.approx(1.05)
    assert float(mv.loc[mv["ticker"] == "BBB", "ratio"].iloc[0]) == pytest.approx(1.00)


def test_without_the_fold_the_check_sees_nothing(run_dir: Path):
    """The mutation, run rather than argued: this is what every morning since 2026-11-08 said."""
    facts = pd.DataFrame([_share_fact(t, "2026-09-30", 100.0) for t in ("AAA", "BBB", "CCC")])
    unfolded = h._parse_payload_file(run_dir / "dashboard_data.js")
    assert X.market_value_check(unfolded, facts).empty
    assert len(X.market_value_check(h.load_payload(run_dir), facts)) == 3


def test_the_health_note_counts_what_it_compared(run_dir: Path):
    """End to end through the gate's own note, because the printed line is what a session reads -
    and `0/0 within 5% (0 over 10%)` read as *healthy* for two runs."""
    facts = pd.DataFrame([_share_fact(t, "2026-09-30", 100.0) for t in ("AAA", "BBB", "CCC")])
    res = h.Result()
    h.check_inputs_against_filings(h.load_payload(run_dir), res, facts=facts)
    note = next(n for n in res.notes if n.startswith("inputs against SEC filings: "))
    # The denominator is the whole point: it is the number of stocks the check could compare,
    # and it read 0 for two runs. The numerator is not asserted here because AAA's ratio is
    # exactly 1.05 and `<= 0.05` on a float is a boundary question of its own, not this one.
    m = re.search(r"market value / price vs filed shares (\d+)/(\d+) within 5% \((\d+) over 10%\)", note)
    assert m, f"the market-value segment is not in the note: {note}"
    assert m.group(2) == "3", f"the check compared {m.group(2)} of 3 stocks - the 2026-11-08 failure"
    assert m.group(3) == "0"
    # The other fields read 0/0 in this fixture and should: it carries share counts and no
    # revenue or income facts. Only the market-value segment is this test's subject.
