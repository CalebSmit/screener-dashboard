"""Does the market backdrop's yield-curve sentence match the spread's own record?

The published panel says:

    "An inverted curve (10-year below 3-month) has preceded every US recession since the
     late 1960s, with lags of roughly 6 to 24 months and at least one false alarm - it
     describes the cycle, it does not time it (Estrella & Mishkin 1998; the New York Fed's
     recession model uses this spread)."

Three testable parts:
  (a) preceded every recession since the late 1960s
  (b) lags of roughly 6 to 24 months
  (c) at least one false alarm

Estrella & Trubin (2006, NY Fed *Current Issues* 12(5), p. 5) report (a) and (b)'s basis on
the monthly-average 10-year minus 3-month spread for 1968-2005, and report **no** false
signals on that basis over that period - so (c) cannot come from the pre-2006 record. This
script measures all three on free FRED monthly data through today, which is where a reader
of the live page stands.

Series (both free, no key):
  GS10   10-year Treasury constant maturity, monthly average
  TB3MS  3-month Treasury bill, secondary market, monthly average
  USREC  NBER recession indicator

The spread is GS10 - TB3MS. Estrella & Trubin convert the bill to a bond-equivalent basis,
which makes the spread slightly larger; the sign of a monthly average is what the claims turn
on, and the two bases differ by well under the episodes measured here (noted in the output).

Usage:  python -I -X utf8 research/measurements/2026-10-17-yield-curve-record.py
"""

from __future__ import annotations

import sys

import pandas as pd


def series(sid: str) -> pd.Series:
    """Full history of one FRED series, cached under data/measurements/ (same helper as the
    Sahm script - duplicated rather than imported because the filenames start with a digit)."""
    import io
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "data" / "measurements"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fred-{sid}-full.csv"
    if not path.exists():
        import requests

        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60)
        r.raise_for_status()
        if not r.text.startswith("observation_date"):
            raise SystemExit(f"unexpected FRED response for {sid}")
        path.write_text(r.text, encoding="utf-8")
    df = pd.read_csv(io.StringIO(path.read_text(encoding="utf-8")))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def episodes(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out, start, prev = [], None, None
    for d, v in flag.items():
        if v == 1 and start is None:
            start = d
        elif v == 0 and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, flag.index[-1]))
    return out


def bond_equivalent(discount: pd.Series, days: int = 91) -> pd.Series:
    """A 3-month bill's discount rate on a bond-equivalent basis - the basis Estrella & Trubin
    (2006, p. 3) use. BEY is *higher* than the discount rate, so the spread computed against it
    is *smaller*; this is the conservative direction for an "inverted before every recession"
    claim, which is why it is worth computing rather than assuming."""
    d = discount / 100.0
    return (365.0 * d / (360.0 - d * days)) * 100.0


def main() -> int:
    gs10, tb3 = series("GS10"), series("TB3MS")
    basis = "bond-equivalent" if "--bey" in sys.argv else "discount"
    short = bond_equivalent(tb3) if basis == "bond-equivalent" else tb3
    print(f"[3-month bill on the {basis} basis; pass --bey for the other]")
    spread = (gs10 - short).dropna()
    rec = series("USREC")
    rec = rec.loc[rec.index >= spread.index[0]]
    print(f"spread (GS10 - TB3MS): {len(spread)} months, "
          f"{spread.index[0]:%Y-%m} .. {spread.index[-1]:%Y-%m}")
    print(f"latest: {spread.iloc[-1]:+.2f}pp in {spread.index[-1]:%Y-%m}\n")

    recs = [r for r in episodes(rec) if r[0].year >= 1968]
    inv = episodes((spread < 0).astype(int))

    print("-- (a)/(b) every NBER recession since 1968: the inversion that preceded it --")
    print(f"{'recession starts':<18} {'inversion begins':<18} {'months of lead':>15} "
          f"{'negative months in the 12 before':>34}")
    leads = []
    for start, _end in recs:
        peak = start - pd.DateOffset(months=1)
        before = spread.loc[(spread.index < start) & (spread.index >= start - pd.DateOffset(months=36))]
        neg = before.loc[before < 0]
        in12 = spread.loc[(spread.index < start) & (spread.index >= start - pd.DateOffset(months=12))]
        n12 = int((in12 < 0).sum())
        if len(neg):
            # two readings of "the lag": the nearest negative run before the recession, and the
            # earliest run that begins within 24 months of it (the 2019 inversion is the signal
            # a reader would have acted on in 2020, not the one-month touch in February 2020)
            runs = [e for e in inv if e[0] <= neg.index[-1] <= e[1]]
            begin = runs[-1][0] if runs else neg.index[0]
            early = [e for e in inv if start - pd.DateOffset(months=24) <= e[0] < start]
            begin_e = early[0][0] if early else begin
            lead = (start.year - begin.year) * 12 + (start.month - begin.month)
            lead_e = (start.year - begin_e.year) * 12 + (start.month - begin_e.month)
            leads.append(lead_e)
            print(f"{start:%Y-%m}{'':<12} {begin:%Y-%m} (earliest {begin_e:%Y-%m}) "
                  f"{lead:>3d} m / {lead_e:>2d} m earliest {n12:>14d}")
        else:
            print(f"{start:%Y-%m}{'':<12} {'none in 36m':<18} {'-':>15} {n12:>34d}")
    if leads:
        print(f"\n  lead of the inversion's first month: min {min(leads)}, max {max(leads)}, "
              f"median {sorted(leads)[len(leads) // 2]} months (n={len(leads)})")
        print(f"  recessions with >= 3 negative monthly averages in the prior 12: "
              f"{sum(1 for s, _ in recs if int((spread.loc[(spread.index < s) & (spread.index >= s - pd.DateOffset(months=12))] < 0).sum()) >= 3)} "
              f"of {len(recs)}")

    # Estrella & Trubin begin at January 1968, which is also the start of the period the page's
    # sentence names ("since the late 1960s"). List from 1960 so the one episode that period
    # excludes - the 1966-67 inversion, the classic false alarm - is visible rather than filtered.
    print("\n-- (c) every inversion episode since 1960, and whether a recession followed --")
    false_alarms = []
    for start, end in inv:
        if start.year < 1960:
            continue
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        after = rec.loc[(rec.index >= start) & (rec.index <= end + pd.DateOffset(months=24))]
        followed = bool((after == 1).any())
        # the 24-month window has to have closed before "no recession followed" is a finding
        closed = end + pd.DateOffset(months=24) <= rec.index[-1]
        verdict = "yes" if followed else ("NO" if closed else "not yet decidable")
        if not followed and closed:
            false_alarms.append((start, end, months, float(spread.loc[start:end].min())))
        print(f"  {start:%Y-%m}..{end:%Y-%m}  {months:>3d} month(s)  deepest "
              f"{spread.loc[start:end].min():+.2f}pp  recession within 24m of its end: {verdict}")
    print(f"\n  inversion episodes with no recession within 24 months of their end: "
          f"{len(false_alarms)}")
    for start, end, months, low in false_alarms:
        print(f"    {start:%Y-%m}..{end:%Y-%m}, {months} months, deepest {low:+.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
