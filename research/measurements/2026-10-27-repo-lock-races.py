"""Can scripts/repo-lock.ps1 let two processes hold the repo lock at once? Two measurements.

Written 2026-10-27 for `research/2026-10-27-flaky-tests.md`, after
`tests/test_loop_mutual_exclusion.py::test_exactly_one_of_several_simultaneous_starts_wins`
failed under a loaded full suite. Read-only with respect to the repo: every lock lives in a
fresh temporary directory. Windows only (it drives the real Enter-RepoLock in powershell.exe).

1. ``window`` - the winner's write window. Enter-RepoLock creates the lock with
   File.Open(CreateNew, FileShare.None) and only then writes "$PID`n$Holder`n<date>" and
   disposes the stream. Between those two moments the file exists and nobody else can read it.
   Read-RepoLockInfo swallows the read failure and leaves OwnerPid = 0; Test-RepoLockOwnerAlive(0)
   is false; so a contender arriving in that window logs "held by PID 0 (unknown), which is gone.
   Reclaiming.", loops on Remove-Item (refused while the handle is open) and deletes the lock the
   moment the winner closes it - then takes it. This plays the winner from Python with the same
   CreateNew / exclusive-share open, stays in the window for WINDOW seconds (what a winner
   descheduled under load does), then writes its own live PID and closes, exactly as the winner
   would. A correct lock never hands the contender a path while that live process owns it.

   It also times the real window: the create-format-write-dispose block run in fresh
   powershell.exe processes, alone and six at a time.

2. ``stale`` - several racers find a lock left by a crashed run (PID 999999). Each reads the dead
   PID, deletes the file and creates its own; a racer that read the dead PID before another's
   create then deletes that live lock (check-then-delete is not atomic). Every winner holds until
   all racers have returned, so a winner count above one is simultaneous ownership.

usage:
    python research/measurements/2026-10-27-repo-lock-races.py window TRIALS SECONDS
    python research/measurements/2026-10-27-repo-lock-races.py timing RUNS
    python research/measurements/2026-10-27-repo-lock-races.py stale TRIALS
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "scripts" / "repo-lock.ps1"
PS = shutil.which("powershell")
PS_ARGS = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]


def _ps(script: Path) -> subprocess.Popen:
    return subprocess.Popen([PS, *PS_ARGS, str(script)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def window(trials: int, seconds: float) -> None:
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    generic_write, create_new, share_none, normal = 0x40000000, 1, 0, 0x80
    stolen = misread = 0
    for k in range(trials):
        d = Path(tempfile.mkdtemp(prefix="lockwin-"))
        lock, ready = d / ".repo.lock", d / "ready"
        script = d / "contender.ps1"
        script.write_text(f""". '{LOCK.as_posix()}'
Set-Content -Path '{ready.as_posix()}' -Value 1
while (-not (Test-Path '{lock.as_posix()}')) {{ Start-Sleep -Milliseconds 5 }}
$p = Enter-RepoLock -LogDir '{d.as_posix()}' -Holder 'contender' -MaxWaitMinutes 0 -PollSeconds 1
if ($p) {{ Write-Host 'GOT-IT' }} else {{ Write-Host 'BLOCKED' }}
""", encoding="ascii")
        proc = _ps(script)
        while not ready.exists():
            time.sleep(0.01)
        h = k32.CreateFileW(str(lock), generic_write, share_none, None, create_new, normal, None)
        time.sleep(seconds)
        data = f"{os.getpid()}\nwinner\n2026-10-27\n".encode("ascii")
        k32.WriteFile(h, data, len(data), ctypes.byref(wintypes.DWORD()), None)
        k32.CloseHandle(h)
        out = proc.communicate(timeout=120)[0]
        got, pid0 = "GOT-IT" in out, out.count("PID 0")
        stolen += got
        misread += pid0 > 0
        print(f"trial {k}: contender {'TOOK THE LIVE LOCK' if got else 'blocked'}; "
              f"'PID 0 ... Reclaiming' lines {pid0}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    print(f"SUMMARY window={seconds}s: the contender took a live lock in {stolen} of {trials} "
          f"trials and misread the live owner as PID 0 in {misread}")


def timing(runs: int) -> None:
    d = Path(tempfile.mkdtemp(prefix="locktime-"))
    script = d / "w.ps1"
    script.write_text(r"""
$path = Join-Path $env:TEMP ("winmeasure-" + [guid]::NewGuid() + ".lock")
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$stream = [System.IO.File]::Open($path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
$t0 = $sw.Elapsed.TotalMilliseconds
try {
    $text  = "{0}`n{1}`n{2}`n" -f $PID, 'racer', (Get-Date -Format 'o')
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($text)
    $stream.Write($bytes, 0, $bytes.Length)
} finally { $stream.Dispose() }
$t1 = $sw.Elapsed.TotalMilliseconds
Remove-Item $path
Write-Host ("window_ms={0:N2}" -f ($t1 - $t0))
""", encoding="ascii")
    for conc in (1, 6):
        vals = []
        for _ in range(runs):
            procs = [_ps(script) for _ in range(conc)]
            for p in procs:
                m = re.search(r"window_ms=([\d.,]+)", p.communicate()[0])
                vals.append(float(m.group(1).replace(",", "")))
        vals.sort()
        print(f"{conc} at once: n={len(vals)} min {vals[0]:.1f} ms, median {vals[len(vals) // 2]:.1f} ms, "
              f"max {vals[-1]:.1f} ms")
    shutil.rmtree(d, ignore_errors=True)


def stale(trials: int, racers: int = 6) -> None:
    multi = 0
    for k in range(trials):
        d = Path(tempfile.mkdtemp(prefix="stale-"))
        (d / ".repo.lock").write_text("999999\ncrashed run\n2026-10-27\n", encoding="ascii")
        procs = []
        for i in range(racers):
            s = d / f"r{i}.ps1"
            s.write_text(f""". '{LOCK.as_posix()}'
$p = Enter-RepoLock -LogDir '{d.as_posix()}' -Holder 'racer {i}' -MaxWaitMinutes 0 -PollSeconds 1
if ($p) {{ Set-Content -Path '{d.as_posix()}/won{i}' -Value $PID }} else {{ Set-Content -Path '{d.as_posix()}/lost{i}' -Value $PID }}
$deadline = (Get-Date).AddSeconds(90)
if ($p) {{
  while (@(Get-ChildItem '{d.as_posix()}' -Filter 'won*').Count + @(Get-ChildItem '{d.as_posix()}' -Filter 'lost*').Count -lt {racers} -and (Get-Date) -lt $deadline) {{ Start-Sleep -Milliseconds 100 }}
  Exit-RepoLock $p
}}
""", encoding="ascii")
            procs.append(_ps(s))
        outs = [p.communicate(timeout=180)[0] for p in procs]
        won = len(list(d.glob("won*")))
        multi += won != 1
        print(f"trial {k}: {won} racer(s) held the lock at once; reclaim lines "
              f"{sum(o.count('Reclaiming') for o in outs)}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    print(f"SUMMARY: {multi} of {trials} trials ended with more than one holder")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "window":
        window(int(sys.argv[2]), float(sys.argv[3]))
    elif mode == "timing":
        timing(int(sys.argv[2]))
    elif mode == "stale":
        stale(int(sys.argv[2]))
    else:
        sys.exit(__doc__)
