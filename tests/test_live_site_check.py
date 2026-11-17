"""After the data loop pushes, it checks the live site serves that run (2026-11-17).

``scripts/check_live_site.py`` reads the first bytes of the live ``dashboard_data.js`` and compares
its run_timestamp with the one just pushed. Pinned here: the parsing, the waiting (with a fake clock,
never the network), that it can never fail its caller, that a stale site produces the marker the
morning brief already surfaces, and where the data loop calls it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_live_site as cls  # noqa: E402
import write_brief as wb  # noqa: E402

HEAD = 'window.SCREENER_DATA = {"kpis": {"run_timestamp": "2026-11-16T18:00:03.456196", "universe_size": 499'


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _wait(answers, expected="2026-11-16T18:00:03.456196", budget=300, interval=30):
    seq, clock, calls = list(answers), _Clock(), []

    def fetch():
        calls.append(clock.t)
        return seq.pop(0) if len(seq) > 1 else seq[0]
    out = cls.wait_for(expected, fetch=fetch, budget_s=budget, interval_s=interval,
                       clock=clock, sleep=clock.sleep)
    return out, calls


def test_the_stamp_is_read_from_the_payload_s_opening_bytes():
    assert cls.stamp_of(HEAD) == "2026-11-16T18:00:03.456196"
    assert cls.stamp_of(None) is None and cls.stamp_of("<html>404</html>") is None
    assert cls.local_stamp() is not None                      # the committed payload opens this way


def test_live_at_once_and_live_after_a_deploy_delay():
    (state, seen, waited), calls = _wait([HEAD])
    assert state == "live" and waited == 0 and len(calls) == 1
    old = HEAD.replace("2026-11-16T18", "2026-11-13T02")
    (state, seen, waited), calls = _wait([old, old, None, HEAD])
    assert state == "live" and waited == 90 and len(calls) == 4


def test_a_site_still_on_the_old_run_is_stale_within_the_budget():
    old = HEAD.replace("2026-11-16T18", "2026-11-13T02")
    (state, seen, waited), calls = _wait([old], budget=300, interval=30)
    assert state == "stale" and seen.startswith("2026-11-13T02") and waited <= 300
    msg = cls.message(state, "2026-11-16T18:00:03.456196", seen, waited)
    assert msg.startswith(cls.STALE) and cls.DEPLOYS in msg


def test_a_site_that_never_answers_is_unreachable_not_stale():
    (state, seen, _), _ = _wait([None])
    assert state == "unreachable" and seen is None
    assert "ATTENTION" not in cls.message(state, "x", None, 300)


def test_the_brief_surfaces_a_stale_site():
    line = "[02:31:00] [WARN] " + cls.message("stale", "2026-11-16T18:00:03", "2026-11-13T02:00:01", 1200)
    verdict, notable = wb.outcome([line, "[02:40:00] [INFO] === Data loop complete ==="])
    assert any(n.startswith(cls.STALE) for n in notable)


def test_every_request_is_a_small_uncached_range_request(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return HEAD.encode()

    def fake_urlopen(req, timeout):
        seen["url"], seen["range"] = req.full_url, req.get_header("Range")
        return _Resp()
    monkeypatch.setattr(cls.urllib.request, "urlopen", fake_urlopen)
    assert cls.stamp_of(cls.fetch_head()) == "2026-11-16T18:00:03.456196"
    assert seen["range"] == f"bytes=0-{cls.HEAD_BYTES - 1}"
    assert seen["url"].startswith(cls.SITE + "dashboard_data.js?live-check=")


def test_it_never_fails_its_caller(monkeypatch, capsys):
    def boom():
        raise RuntimeError("anything")
    monkeypatch.setattr(cls, "local_stamp", boom)
    assert cls.main([]) == 0
    assert "could not check" in capsys.readouterr().out
    monkeypatch.setattr(cls, "local_stamp", lambda: None)
    assert cls.main([]) == 0


def test_the_site_address_is_the_one_the_methodology_publishes():
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    assert re.search(r'^DASHBOARD_URL = "([^"]+)"', src, re.M).group(1) == cls.SITE


def test_the_data_loop_checks_after_the_push_and_never_stops_on_it():
    ps = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8-sig")
    i = ps.index("scripts/check_live_site.py")
    assert ps.index("'push', 'origin'") < i < ps.index("[System.IO.File]::WriteAllText($SuccessMarker")
    block = ps[ps.rindex("# --- Did the live site", 0, i):ps.index("# Keep the object store packed", i)]
    code = " ".join(ln for ln in block.splitlines() if not ln.lstrip().startswith("#"))
    assert "Stop-Run" not in code and not re.search(r"(?i)(?<![a-z-])exit(?![a-z-])", code)
    assert "'WARN'" in block and "ATTENTION - LIVE SITE" in block
    assert 'Write-Log "Published to main - live dashboard refreshed."' not in ps   # a push is not a page
