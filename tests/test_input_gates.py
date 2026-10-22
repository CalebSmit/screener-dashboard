"""The trading bot's inputs: the 2026-10-22 gates in ``scripts/check_run_health.py`` and the
data loop's git hygiene (OWNER_FOCUS 2026-10-21 items 1-4; screener-bot ROADMAP 1.3, 1.4, 1.5,
1.9).

Each rule's thresholds were replayed against every page the site published since 2026-08-21
(``research/measurements/2026-10-22-input-gates-replay.py``): none of the ordinary days fails,
and the known-bad pages of 2026-07-28, 08-06, 08-07 and 08-10 do. The tests below pin the cases
the owner named - a shuffled ranking with normal spread fails, a normal day passes, an overhaul
with its changelog entry passes; a 450-stock run fails; yesterday's prices fail and the
post-holiday case passes; a failed pull stops the run.
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402

N = 500


def _payload(order, run_ts="2026-10-14T02:00:03", version="v1", prices=None, price_date=None):
    """A payload ranking ``order`` (tickers best first), as the generator writes it."""
    table, detail = [], {}
    for i, t in enumerate(order, start=1):
        table.append({"Ticker": t, "Rank": i, "Composite": 100.0 - i / 10})
        p = (prices or {}).get(t, 10.0 + i)
        detail[t] = {"price": p, "price_date": price_date}
    return {"kpis": {"run_timestamp": run_ts, "methodology_version": version},
            "table_data": table, "stock_detail": detail}


def _names(n=N):
    return [f"T{i:03d}" for i in range(n)]


def _normal_day(order, seed=1, swaps=40, span=6):
    """An ordinary day: a few dozen names move a handful of places."""
    rng = random.Random(seed)
    o = list(order)
    for _ in range(swaps):
        i = rng.randrange(len(o) - span)
        j = i + rng.randrange(1, span)
        o[i], o[j] = o[j], o[i]
    return o


# =========================================================================== 1. rank continuity
class TestRankContinuity:
    def test_a_normal_day_passes(self):
        base = _payload(_names())
        cur = _payload(_normal_day(_names()), run_ts="2026-10-15T02:00:03")
        res = h.Result()
        h.check_rank_continuity(cur, base, res, changelog_text="")
        assert res.healthy, res.failures
        c = h.rank_continuity(cur, base)
        assert c["spearman"] > 0.99 and len(c["new_in_top"]) <= 4

    def test_a_shuffled_ranking_with_normal_spread_fails(self):
        """The 2026-07-28 / 08-07 shape: same names, same score spread, order scrambled. Every
        check of its day passes - coverage, dispersion - and only continuity can see it."""
        names = _names()
        shuffled = list(names)
        random.Random(7).shuffle(shuffled)
        base, cur = _payload(names), _payload(shuffled, run_ts="2026-10-15T02:00:03")
        # identical composite spread: the dispersion check has nothing to see
        assert [r["Composite"] for r in base["table_data"]] == [r["Composite"] for r in cur["table_data"]]
        res = h.Result()
        h.check_rank_continuity(cur, base, res, changelog_text="")
        assert not res.healthy
        assert "Spearman" in res.failures[0] and "no documented methodology change" in res.failures[0]

    @pytest.mark.parametrize("new,ok", [(8, True), (9, False)])
    def test_new_names_in_the_top_25_limit(self, new, ok):
        names = _names()
        cur = list(names)
        # swap `new` of the top 25 with names just below it - new names, none fallen
        for k in range(new):
            a, b = 24 - k, 25 + k
            cur[a], cur[b] = cur[b], cur[a]
        res = h.Result()
        h.check_rank_continuity(_payload(cur, run_ts="2026-10-15T02:00:03"), _payload(names), res,
                                changelog_text="")
        assert h.rank_continuity(_payload(cur), _payload(names))["spearman"] > 0.95
        assert res.healthy is ok, res.failures

    @pytest.mark.parametrize("fallen,ok", [(3, True), (4, False)])
    def test_top_names_falling_below_40_limit(self, fallen, ok):
        names = _names()
        cur = list(names)
        for k in range(fallen):          # top names to 41.., replaced from 26..: few new names
            a, b = k, 40 + k
            cur[a], cur[b] = cur[b], cur[a]
        res = h.Result()
        h.check_rank_continuity(_payload(cur, run_ts="2026-10-15T02:00:03"), _payload(names), res,
                                changelog_text="")
        assert res.healthy is ok, res.failures

    def test_a_top_name_that_is_no_longer_ranked_counts_as_fallen(self):
        names = _names()
        c = h.rank_continuity(_payload(names[4:]), _payload(names))
        assert c["fallen"] == names[:4]

    def test_an_overhaul_with_its_changelog_entry_passes(self):
        names = _names()
        shuffled = list(names)
        random.Random(3).shuffle(shuffled)
        base = _payload(names, run_ts="2026-10-09T02:00:03", version="aaa")
        cur = _payload(shuffled, run_ts="2026-10-09T15:00:00", version="bbb")
        res = h.Result()
        h.check_rank_continuity(cur, base, res,
                                changelog_text="## 2026-10-09 (owner-run) - the metric audit\n")
        assert res.healthy, res.failures
        assert any("allowed because" in n and "aaa -> bbb" in n for n in res.notes)

    def test_an_overhaul_without_an_entry_fails(self):
        names = _names()
        shuffled = list(names)
        random.Random(3).shuffle(shuffled)
        res = h.Result()
        h.check_rank_continuity(_payload(shuffled, run_ts="2026-10-09T15:00:00", version="bbb"),
                                _payload(names, run_ts="2026-10-09T02:00:03", version="aaa"), res,
                                changelog_text="## 2026-10-08 - yesterday's entry\n")
        assert not res.healthy

    def test_an_entry_without_a_version_change_does_not_waive_it(self):
        """A changelog entry about something else must not wave a scrambled run through."""
        names = _names()
        shuffled = list(names)
        random.Random(3).shuffle(shuffled)
        res = h.Result()
        h.check_rank_continuity(_payload(shuffled, run_ts="2026-10-09T15:00:00", version="aaa"),
                                _payload(names, run_ts="2026-10-09T02:00:03", version="aaa"), res,
                                changelog_text="## 2026-10-09 - an unrelated entry\n")
        assert not res.healthy

    def test_an_unknown_baseline_version_does_not_waive_it(self):
        names = _names()
        shuffled = list(names)
        random.Random(3).shuffle(shuffled)
        base = _payload(names, run_ts="2026-10-09T02:00:03")
        base["kpis"].pop("methodology_version")
        res = h.Result()
        h.check_rank_continuity(_payload(shuffled, run_ts="2026-10-09T15:00:00"), base, res,
                                changelog_text="## 2026-10-09 - an entry\n")
        assert not res.healthy

    def test_the_entry_may_date_from_the_published_run_onward(self):
        """A methodology change made at 06:00 and not republished is first scored by the NEXT
        day's 02:00 run. "Dated today" only would refuse that run and every one after it."""
        names = _names()
        shuffled = list(names)
        random.Random(3).shuffle(shuffled)
        base = _payload(names, run_ts="2026-10-13T02:00:03", version="aaa")
        cur = _payload(shuffled, run_ts="2026-10-14T02:00:03", version="bbb")
        assert h.continuity_override(cur, base, "## 2026-10-13 - changed at 06:00\n")
        assert h.continuity_override(cur, base, "## 2026-10-14 - today\n")
        assert not h.continuity_override(cur, base, "## 2026-10-12 - before the published run\n")
        assert not h.continuity_override(cur, base, "## 2026-10-15 - after this run\n")

    def test_a_subset_run_is_not_compared(self, tmp_path):
        names = _names()
        shuffled = list(names)
        random.Random(3).shuffle(shuffled)
        res = h.Result()
        h.check_rank_continuity(_payload(shuffled[:30]), _payload(names), res, changelog_text="",
                                run_dir=_run_dir(tmp_path, tickers="AAPL,MSFT"))
        assert res.healthy and "--tickers" in res.notes[0]

    def test_no_published_ranking_is_a_note_not_a_refusal(self):
        res = h.Result()
        h.check_rank_continuity(_payload(_names()), None, res, changelog_text="")
        assert res.healthy and any("NOT checked" in n for n in res.notes)

    def test_changelog_dates_are_the_entry_headings(self):
        text = "# Changelog\n## YYYY-MM-DD - format\n## 2026-10-09 (owner-run) - x\n### 2026-10-10 sub\n## 2026-10-21 - y\n"
        assert h.changelog_dates(text) == {"2026-10-09", "2026-10-21"}

    def test_the_real_changelog_is_dated_that_way(self):
        dates = h.changelog_dates((ROOT / "METHODOLOGY_CHANGELOG.md").read_text(encoding="utf-8"))
        assert "2026-10-09" in dates and len(dates) > 20

    def test_the_baseline_is_the_committed_payload_not_the_working_tree(self):
        """run_screener.py step 12 copies the run's payload over the root file before this gate
        runs, so the working-tree file is the run itself."""
        src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
        body = src.split("def published_payload")[1].split("\ndef ")[0]
        assert '"show", f"{ref}:dashboard_data.js"' in body
        main = src.split("def main()")[1]
        assert "published = published_payload()" in main
        assert "check_rank_continuity(payload, published, res, run_dir=run_dir)" in main


# =========================================================================== 2. scored universe
def _run_dir(tmp_path, failed=(), scored_count=None, tickers=None):
    d = tmp_path / "run"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"start_time": "2026-10-22T02:00:04",
                                             "cli_flags": {"tickers": tickers}}), encoding="utf-8")
    (d / "universe.json").write_text(json.dumps({
        "scored": [], "failed": list(failed),
        "scored_count": (503 - len(failed)) if scored_count is None else scored_count,
        "failed_count": len(failed)}), encoding="utf-8")
    return d


class TestScoredUniverse:
    def test_a_450_stock_run_fails(self, tmp_path):
        res = h.Result()
        h.check_scored_universe(_payload(_names(450)), _run_dir(tmp_path), res)
        assert not res.healthy and "only 450 stocks are ranked" in res.failures[0]

    def test_todays_run_passes(self, tmp_path):
        """2026-10-22: 498 ranked, HONA and VYLR unscored of 503."""
        res = h.Result()
        h.check_scored_universe(_payload(_names(498)), _run_dir(tmp_path, ["HONA", "VYLR"]), res)
        assert res.healthy and "2 of 503 unscored" in res.notes[0]

    @pytest.mark.parametrize("n_failed,ok", [(10, True), (11, False)])
    def test_unscored_share_limit(self, tmp_path, n_failed, ok):
        res = h.Result()
        h.check_scored_universe(_payload(_names(492)),
                                _run_dir(tmp_path, [f"X{i}" for i in range(n_failed)]), res)
        assert res.healthy is ok

    def test_a_subset_run_is_exempt(self, tmp_path):
        res = h.Result()
        h.check_scored_universe(_payload(_names(3)), _run_dir(tmp_path, tickers="AAPL,MSFT,JPM"), res)
        assert res.healthy and "--tickers" in res.notes[0]

    def test_a_run_without_universe_json_cannot_be_evaluated(self, tmp_path):
        d = _run_dir(tmp_path)
        (d / "universe.json").unlink()
        res = h.Result()
        h.check_scored_universe(_payload(_names()), d, res)
        assert res.unevaluable

    def test_the_universe_json_failed_list_is_what_the_screener_writes(self):
        """`failed` is every skipped ticker: fetch errors, empty records, the coverage filter."""
        src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
        assert 'ctx.save_universe(df["Ticker"].tolist(), skipped_tickers)' in src
        assert 'skipped_tickers += df.loc[low, "Ticker"].tolist()' in src     # coverage filter
        assert 'skipped_tickers += df.loc[skip_mask, "Ticker"].tolist()' in src  # empty records

    def test_the_data_loop_counts_them_toward_its_40_percent_gate(self):
        ps = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8-sig")
        block = ps.split("$fetchFail = @(")[1].split(")")[0]
        assert "'fetch_failure'" in block and "'Excluded from scoring'" in block
        assert "-Unique" in ps.split("$fetchFail = @(")[1].split("$fetchErr")[0]
        # and the 40% comparison still reads that count
        assert "$fetchFail.Count -gt ($universe * $maxFailPct / 100)" in ps

    def test_the_coverage_filter_logs_what_the_data_loop_counts(self):
        src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
        i = src.index("for t in df.loc[low, \"Ticker\"]:")
        assert '"Excluded from scoring"' in src[i:i + 400]


# =========================================================================== 3. price freshness
class TestNyseCalendar:
    @pytest.mark.parametrize("d", ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
                                   "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
                                   "2026-11-26", "2026-12-25", "2027-03-26", "2027-06-18",
                                   "2027-07-05", "2027-12-24", "2025-04-18"])
    def test_published_holidays(self, d):
        assert not h.is_nyse_session(date.fromisoformat(d))

    @pytest.mark.parametrize("d", ["2021-12-31", "2026-10-22", "2026-07-02", "2026-11-27"])
    def test_sessions(self, d):
        """New Year's Day 2022 fell on a Saturday and was not observed on Dec 31."""
        assert h.is_nyse_session(date.fromisoformat(d))

    @pytest.mark.parametrize("at,want", [
        ("2026-10-22T02:00:04", "2026-10-21"),   # the 02:00 Central run: yesterday's close
        ("2026-10-19T02:00:04", "2026-10-16"),   # Monday: Friday's
        ("2026-09-08T02:00:00", "2026-09-04"),   # the Tuesday after Labor Day: Friday's
        ("2026-10-22T08:29:00", "2026-10-21"),   # 09:29 Eastern, before the open
        ("2026-10-22T08:31:00", "2026-10-22"),   # after the open: today's, intraday
        ("2026-10-22T13:00:00", "2026-10-22"),   # the 13:00 session
    ])
    def test_expected_session(self, at, want):
        assert h.expected_session(at).isoformat() == want


def _fresh_payload(run_ts, date_, n=N, dated=None, prices=None):
    p = _payload(_names(n), run_ts=run_ts, price_date=date_, prices=prices)
    if dated is not None:                       # only the first `dated` carry date_
        for i, v in enumerate(p["stock_detail"].values()):
            if i >= dated:
                v["price_date"] = "2026-10-20"
    return p


class TestPriceFreshness:
    def test_todays_prices_pass(self):
        res = h.Result()
        h.check_price_freshness(_fresh_payload("2026-10-22T02:00:04", "2026-10-21"), None, res)
        assert res.healthy, res.failures

    def test_yesterdays_prices_fail(self):
        """Prices a session old - the 02:00 run on 10-22 serving 10-20's closes."""
        res = h.Result()
        h.check_price_freshness(_fresh_payload("2026-10-22T02:00:04", "2026-10-20"), None, res)
        assert not res.healthy and "0/500 on the expected session 2026-10-21" in res.failures[0]

    def test_the_post_holiday_case_passes(self):
        """Tuesday after Labor Day: Friday's closes are the latest, and unchanged from Monday's
        run - no session closed in between, so they are not compared."""
        mon = _fresh_payload("2026-09-07T02:00:00", "2026-09-04")
        tue = _fresh_payload("2026-09-08T02:00:00", "2026-09-04")
        res = h.Result()
        h.check_price_freshness(tue, mon, res)
        assert res.healthy, res.failures
        assert any("no session has closed since it" in n for n in res.notes)

    @pytest.mark.parametrize("dated,ok", [(475, True), (474, False)])
    def test_the_95_percent_line(self, dated, ok):
        res = h.Result()
        h.check_price_freshness(_fresh_payload("2026-10-22T02:00:04", "2026-10-21", dated=dated),
                                None, res)
        assert res.healthy is ok

    def test_a_stock_with_no_price_or_no_date_is_not_fresh(self):
        p = _fresh_payload("2026-10-22T02:00:04", "2026-10-21")
        for i, v in enumerate(p["stock_detail"].values()):
            if i < 30:
                v["price_date"] = None
        res = h.Result()
        h.check_price_freshness(p, None, res)
        assert not res.healthy and "none x30" in res.failures[0]

    def test_a_payload_without_price_dates_fails(self):
        """A run whose prices carry no date cannot be shown fresh - the 08-07 / 08-10 runs had
        no prices at all."""
        p = _fresh_payload("2026-10-22T02:00:04", None)
        res = h.Result()
        h.check_price_freshness(p, None, res)
        assert not res.healthy

    def test_prices_unchanged_across_a_session_fail(self):
        """Every date stamp right, every price yesterday's: the re-served-quote case."""
        prices = {t: 50.0 + i for i, t in enumerate(_names())}
        base = _fresh_payload("2026-10-21T02:00:04", "2026-10-20", prices=prices)
        cur = _fresh_payload("2026-10-22T02:00:04", "2026-10-21", prices=prices)
        res = h.Result()
        h.check_price_freshness(cur, base, res)
        assert not res.healthy and "unchanged from the published run" in res.failures[0]

    def test_a_normal_day_of_price_changes_passes(self):
        base = _fresh_payload("2026-10-21T02:00:04", "2026-10-20")
        cur = _fresh_payload("2026-10-22T02:00:04", "2026-10-21",
                             prices={t: 11.0 + i * 1.01 for i, t in enumerate(_names())})
        res = h.Result()
        h.check_price_freshness(cur, base, res)
        assert res.healthy, res.failures

    def test_a_fetch_that_straddles_the_open_is_not_refused(self, tmp_path):
        d = tmp_path / "r"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"start_time": "2026-10-22T08:20:00",
                                                 "end_time": "2026-10-22T08:38:00"}), encoding="utf-8")
        p = _fresh_payload("2026-10-22T08:20:00", "2026-10-21", dated=250)
        for i, v in enumerate(p["stock_detail"].values()):
            if i >= 250:
                v["price_date"] = "2026-10-22"
        res = h.Result()
        h.check_price_freshness(p, None, res, d)
        assert res.healthy, res.failures


class TestThePriceDateIsPublished:
    """The gate reads ``stock_detail[t].price_date``; it must be the session of the price the
    page actually publishes (``_coalesce(d, "currentPrice", "price_latest")``)."""

    def test_the_quote_is_dated_by_its_trade_time_in_eastern(self):
        import factor_engine as F
        # 2026-10-21 16:00:02 Eastern = 20:00:02 UTC; a UTC date would still be the 21st, but
        # 21:00 Eastern on the 21st is already the 22nd in UTC.
        assert F.published_price_date({"currentPrice": 10.0, "regularMarketTime": 1792612802}) == "2026-10-21"
        assert F.published_price_date({"currentPrice": 10.0, "regularMarketTime": 1792630800}) == "2026-10-21"

    def test_the_bar_is_used_exactly_when_the_quote_is_missing(self):
        import factor_engine as F
        d = {"currentPrice": np.nan, "price_latest": 9.5, "_price_bar_date": "2026-10-20",
             "regularMarketTime": 1792612802}
        assert F.published_price_date(d) == "2026-10-20"
        assert F.published_price_date({"currentPrice": None, "price_latest": np.nan}) is None
        # a quote with no time is undated - never borrowed from the bar
        assert F.published_price_date({"currentPrice": 10.0, "price_latest": 9.5,
                                       "_price_bar_date": "2026-10-20"}) is None

    def test_the_history_records_its_last_bar(self):
        import factor_engine as F
        idx = pd.bdate_range("2025-09-01", "2026-10-21", tz="America/New_York")
        hist = pd.DataFrame({"Close": np.linspace(10, 20, len(idx))}, index=idx)
        assert F.price_history_fields(hist)["_price_bar_date"] == "2026-10-21"

    def test_the_fetch_keeps_the_quote_time(self):
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        assert 'rec["regularMarketTime"]  = _safe(info, "regularMarketTime")' in src
        assert 'rec["_price_date"]        = published_price_date(d)' in src

    def test_the_generator_publishes_it(self):
        import generate_dashboard as g
        df = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Company": ["A", "B"],
                           "Sector": ["Energy", "Energy"], "Composite": [60.0, 50.0], "Rank": [1, 2],
                           "Value_Trap_Flag": [False, False], "Growth_Trap_Flag": [False, False],
                           "_current_price": [10.0, 20.0], "_price_date": ["2026-10-21", np.nan]})
        for c in ("valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"):
            df[c + "_score"] = 50.0
        out = json.loads(g.prepare_dashboard_data({
            "df": df, "meta": {"start_time": "2026-10-22T02:00:04", "methodology_version": "abc123"},
            "weights": {}, "sens_df": None, "corr_df": None, "cfg": {}}))
        assert out["stock_detail"]["AAA"]["price_date"] == "2026-10-21"
        assert out["stock_detail"]["BBB"]["price_date"] is None
        assert out["kpis"]["methodology_version"] == "abc123"

    def test_the_new_raw_fields_are_classified_for_backtest_v2(self):
        import pit_record as pr
        assert {"regularMarketTime", "_price_bar_date"} <= set(pr.FIELD_CLASSES["price_history"])


class TestMethodologyVersion:
    def test_it_is_the_scored_cache_key_recorded_by_the_run(self):
        src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
        assert "methodology_version = ctx.config_hash(cfg)" in src
        assert '"methodology_version": methodology_version,' in src
        # taken before the run mutates cfg (the auto-disables), after a preset is applied
        assert src.index("methodology_version = ctx.config_hash(cfg)") < src.index("args.dry_run:")

    def test_a_past_schema_can_be_replayed(self):
        from run_context import RunContext
        cfg = {"factor_weights": {"valuation": 22}}
        assert RunContext.config_hash(None, cfg) != RunContext.config_hash(None, cfg, 15)
        assert RunContext.config_hash(None, cfg, 15) == RunContext.config_hash(None, cfg, 15)


# =========================================================================== 4. git hygiene
def _code(path: Path) -> str:
    src = re.sub(r"<#.*?#>", "", path.read_text(encoding="utf-8-sig"), flags=re.S)
    return re.sub(r"(?m)#.*$", "", src)


DATA_RUN = ROOT / "scripts" / "data-run.ps1"


class TestDataLoopGitHygiene:
    def test_a_failed_pull_stops_the_run(self):
        code = _code(DATA_RUN)
        i = code.index("'pull', '--ff-only'")
        window = code[i:i + 400]
        assert "$pull.ExitCode -ne 0" in window and "Stop-Run" in window
        assert "| Out-Null" not in code[i - 60:i + 40], "the pull's result must not be discarded"

    def test_the_pull_and_its_check_come_before_the_screener_runs(self):
        code = _code(DATA_RUN)
        assert code.index("'pull', '--ff-only'") < code.index("$args = @('run_screener.py')")
        assert code.index("'origin/main..HEAD'") < code.index("$args = @('run_screener.py')")

    def test_a_main_ahead_of_origin_stops_the_run(self):
        code = _code(DATA_RUN)
        i = code.index("'origin/main..HEAD'")
        assert "-ne '0'" in code[i:i + 200] and "Stop-Run" in code[i:i + 500]

    def test_only_the_data_commit_is_pushed(self):
        code = _code(DATA_RUN)
        assert "'push', 'origin', 'main'" not in code, "a push of local main publishes all of it"
        assert "\"${dataCommit}:refs/heads/main\"" in code
        i = code.index("\"${dataCommit}:refs/heads/main\"")
        before = code[code.index("'commit', '-m'"):i]
        assert "-ne '1'" in before and "Stop-Run" in before, \
            "the data commit must be the only commit between origin/main and the push"

    def test_the_bot_contract_is_in_the_publish_gate(self):
        import check_published_claims as gate
        assert "tests/test_bot_payload_contract.py" in gate.module_paths()


# =========================================================================== the public page
def test_the_methodology_page_states_the_thresholds_the_gate_applies():
    """The page says what a published number has passed; it is built from the gate's constants,
    checked here against the generator's output, not the committed file (rule 10)."""
    import factor_engine as F
    import run_screener
    md = run_screener.build_screener_overview(F.load_config())
    section = md.split("trailing median.")[1].split("A number on this site has passed that gate.")[0]
    for s in (f"fewer than {h.MIN_SCORED} stocks",
              f"more than {h.MAX_EXCLUDED_SHARE:.0%} of the universe unscored",
              f"fewer than {h.MIN_PRICE_ON_SESSION:.0%} of prices are dated on the latest NYSE session",
              f"more than {h.MAX_UNCHANGED_ACROSS_SESSION:.0%} are unchanged",
              f"rank correlation below {h.RANK_MIN_SPEARMAN:.2f}",
              f"more than {h.RANK_MAX_NEW_IN_TOP} new names in the top {h.RANK_TOP_N}",
              f"more than {h.RANK_MAX_FALLEN} of the top {h.RANK_TOP_N} falling below #{h.RANK_FALL_LINE}"):
        assert s in section, s
