"""
Kalshi KXHIGHNY market client
=============================
Fetches the live bracket ladder + prices for the "highest temperature in NYC"
daily market, and normalizes every contract to an inclusive integer degree-F
interval so downstream probability lookup is uniform across contract types.

Settlement (from the contracts' own `rules_primary`):
    "the highest temperature recorded in Central Park, New York ... as reported
     by the National Weather Service's Climatological Report (Daily)"
That is the CLINYC product `score_predictions.fetch_cli` already retrieves, so
the scorer's ground truth and the exchange's settlement value are the same
number -- provided we use CLI *alone* and not a blend with DSM/hourly obs.

Strike semantics (verified against live data -- the field names are misleading):
    less     cap_strike=89   subtitle "88 or below"   -> [-inf, 88]   (strict)
    between  floor=89 cap=90 subtitle "89 to 90"      -> [89, 90]     (inclusive)
    greater  floor_strike=96 subtitle "97 or above"   -> [97, +inf]   (strict)
The `less`/`greater` bounds are STRICT, so the integer interval is offset by one
from the strike. `_interval` encodes that, and `check_ladder` re-derives the same
interval from the human-readable subtitle and warns on any disagreement -- if
Kalshi ever changes the convention, that mismatch is the tripwire.

Usage
-----
    python kalshi_market.py              # today's ladder (NY local)
    python kalshi_market.py 2026-08-11   # a specific date
"""
from __future__ import annotations

import math
import re
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from score_predictions import _get  # polite-retry HTTP wrapper

NY_TZ = ZoneInfo("America/New_York")
SERIES_TICKER = "KXHIGHNY"
MARKETS_URL = "https://external-api.kalshi.com/trade-api/v2/markets"

# Public market data needs no authentication.
NEG_INF, POS_INF = -math.inf, math.inf

COLUMNS = [
    "event_ticker", "ticker", "strike_type", "lo", "hi", "subtitle",
    "yes_bid", "yes_ask", "no_bid", "no_ask", "last",
    "volume", "open_interest",
]


def event_ticker(d: date) -> str:
    """date(2026, 8, 11) -> 'KXHIGHNY-26AUG11'."""
    return f"{SERIES_TICKER}-{d:%y%b%d}".upper()


def _f(val) -> float:
    """Kalshi returns numerics as strings ('0.0100', '1279.00')."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return math.nan


def _interval(strike_type: str, floor_s, cap_s) -> tuple[float, float]:
    """Inclusive integer degree-F interval [lo, hi] the contract pays on."""
    if strike_type == "less":       # strictly below cap
        return (NEG_INF, float(cap_s) - 1)
    if strike_type == "greater":    # strictly above floor
        return (float(floor_s) + 1, POS_INF)
    if strike_type == "between":    # inclusive on both ends
        return (float(floor_s), float(cap_s))
    return (math.nan, math.nan)     # unknown type -> excluded downstream


def _subtitle_interval(subtitle: str) -> tuple[float, float] | None:
    """Re-derive the interval from the human-readable subtitle, as a cross-check."""
    if not isinstance(subtitle, str):
        return None
    s = subtitle.replace("°", " ")
    m = re.search(r"(-?\d+)\D+or below", s, re.I)
    if m:
        return (NEG_INF, float(m.group(1)))
    m = re.search(r"(-?\d+)\D+or above", s, re.I)
    if m:
        return (float(m.group(1)), POS_INF)
    m = re.search(r"(-?\d+)\D+to\D+(-?\d+)", s, re.I)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def fetch_brackets(d: date | None = None, *, timeout: int = 15) -> pd.DataFrame:
    """One row per live KXHIGHNY contract for NY-local date `d` (default today).

    Returns an empty frame (never raises on "no markets") if the event has not
    been listed yet -- callers treat that as "nothing to log this hour".
    """
    d = d or datetime.now(NY_TZ).date()
    ev = event_ticker(d)
    resp = _get(MARKETS_URL, params={"event_ticker": ev, "limit": 200}, timeout=timeout)
    markets = resp.json().get("markets", [])

    rows = []
    for m in markets:
        lo, hi = _interval(m.get("strike_type", ""), m.get("floor_strike"), m.get("cap_strike"))
        if math.isnan(lo) and math.isnan(hi):
            continue  # unrecognized strike type
        rows.append({
            "event_ticker": m.get("event_ticker"),
            "ticker": m.get("ticker"),
            "strike_type": m.get("strike_type"),
            "lo": lo, "hi": hi,
            "subtitle": m.get("subtitle"),
            "yes_bid": _f(m.get("yes_bid_dollars")),
            "yes_ask": _f(m.get("yes_ask_dollars")),
            "no_bid": _f(m.get("no_bid_dollars")),
            "no_ask": _f(m.get("no_ask_dollars")),
            "last": _f(m.get("last_price_dollars")),
            "volume": _f(m.get("volume_fp")),
            "open_interest": _f(m.get("open_interest_fp")),
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values("lo", kind="stable").reset_index(drop=True)


def check_ladder(df: pd.DataFrame) -> list[str]:
    """Structural warnings about the ladder. Empty list == healthy.

    Catches the two ways this silently breaks: a strike-semantics change (our
    interval stops matching the subtitle) and a gap/overlap in the ladder (which
    would make the bracket probabilities fail to sum to 1).
    """
    problems = []
    if df.empty:
        return ["no contracts returned"]

    for _, r in df.iterrows():
        want = _subtitle_interval(r["subtitle"])
        if want is not None and (want[0] != r["lo"] or want[1] != r["hi"]):
            problems.append(
                f"{r['ticker']}: interval [{r['lo']}, {r['hi']}] from strike_type "
                f"'{r['strike_type']}' disagrees with subtitle '{r['subtitle']}' "
                f"-> [{want[0]}, {want[1]}]"
            )

    # The ladder must tile the line with no gaps and no overlaps.
    s = df.sort_values("lo")
    for (_, a), (_, b) in zip(s.iloc[:-1].iterrows(), s.iloc[1:].iterrows()):
        if b["lo"] != a["hi"] + 1:
            problems.append(
                f"ladder discontinuity between {a['ticker']} (hi={a['hi']}) and "
                f"{b['ticker']} (lo={b['lo']})"
            )
    if not math.isinf(s.iloc[0]["lo"]):
        problems.append(f"ladder not left-closed: lowest lo={s.iloc[0]['lo']}")
    if not math.isinf(s.iloc[-1]["hi"]):
        problems.append(f"ladder not right-closed: highest hi={s.iloc[-1]['hi']}")
    return problems


def round_half_up(x: float) -> int:
    """88.5 -> 89, 88.4 -> 88.

    NOT the builtin `round()`, which is banker's rounding: round(88.5) == 88 and
    round(89.5) == 90. Bracket boundaries sit exactly on the .5, so that
    inconsistency would land identical-magnitude predictions in different
    brackets depending on parity.
    """
    return math.floor(float(x) + 0.5)


def bracket_for(df: pd.DataFrame, temp_f: float):
    """The contract whose interval contains `temp_f` (rounded half-up).

    Returns the matching row, or None if the ladder does not cover it.
    """
    if df.empty or temp_f is None or (isinstance(temp_f, float) and math.isnan(temp_f)):
        return None
    t = round_half_up(temp_f)
    hit = df[(df["lo"] <= t) & (t <= df["hi"])]
    return hit.iloc[0] if not hit.empty else None


def fmt_bound(lo: float, hi: float) -> str:
    if math.isinf(lo):
        return f"<= {hi:.0f}"
    if math.isinf(hi):
        return f">= {lo:.0f}"
    return f"{lo:.0f}-{hi:.0f}"


def main() -> None:
    d = pd.to_datetime(sys.argv[1]).date() if len(sys.argv) > 1 else datetime.now(NY_TZ).date()
    df = fetch_brackets(d)
    if df.empty:
        print(f"No open contracts for {event_ticker(d)}.")
        return

    print(f"\n{event_ticker(d)}  --  {len(df)} contracts")
    print(f"{'bracket':>10}  {'type':<8} {'bid':>6} {'ask':>6} {'last':>6} "
          f"{'volume':>10} {'OI':>9}")
    print("-" * 62)
    for _, r in df.iterrows():
        print(f"{fmt_bound(r['lo'], r['hi']):>10}  {r['strike_type']:<8} "
              f"{r['yes_bid']:>6.2f} {r['yes_ask']:>6.2f} {r['last']:>6.2f} "
              f"{r['volume']:>10,.0f} {r['open_interest']:>9,.0f}")

    problems = check_ladder(df)
    print("\nladder check: " + ("OK" if not problems else "PROBLEMS"))
    for p in problems:
        print(f"  ! {p}")


if __name__ == "__main__":
    main()
