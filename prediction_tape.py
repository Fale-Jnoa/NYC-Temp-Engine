"""
Prediction tape
===============
One row per hourly prediction, in plain English:

    what the model said  ->  which Kalshi bracket that lands in
                         ->  what that bracket cost at that moment
                         ->  what the high actually turned out to be

`prediction_tape.csv` is the human-readable record. It is written live by the
bot each hour with the settlement columns blank, then backfilled the next
morning once the NWS Climate Report (CLINYC) finalizes the day -- which is the
same product Kalshi settles on, so `actual_high` here is the exchange's number.

Predicted temperatures are rounded HALF-UP (88.5 -> 89) via
`kalshi_market.round_half_up`; see that function for why the builtin `round()`
is unsuitable at bracket boundaries.

Usage
-----
    python prediction_tape.py              # print the tape
    python prediction_tape.py --backfill   # fill settled days, then print
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import kalshi_market as km
import score_predictions as sp

NY_TZ = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
TAPE_PATH = HERE / "prediction_tape.csv"

COLUMNS = [
    "valid_t", "local_date", "local_hour",
    "pred_raw", "pred_high", "obs_high", "reassessed",
    "bracket_ticker", "bracket", "lo", "hi",
    "yes_bid", "yes_ask", "last",
    "actual_high", "hit", "ret_per_dollar",
]


def append_row(valid_t: datetime, pred_raw: float, obs_high: float | None,
               reassessed: float, brackets: pd.DataFrame,
               path: Path = TAPE_PATH) -> dict | None:
    """Append one prediction to the tape. Returns the row written, or None.

    `brackets` is a `kalshi_market.fetch_brackets()` frame for the same NY-local
    date. Settlement columns are left blank for `backfill()` to fill later.
    """
    local = valid_t.astimezone(NY_TZ)
    b = km.bracket_for(brackets, pred_raw)
    row = {
        "valid_t": valid_t.isoformat(),
        "local_date": local.date().isoformat(),
        "local_hour": local.hour,
        "pred_raw": round(float(pred_raw), 2),
        "pred_high": km.round_half_up(pred_raw),
        "obs_high": obs_high,
        "reassessed": round(float(reassessed), 2),
        "bracket_ticker": b["ticker"] if b is not None else "",
        "bracket": km.fmt_bound(b["lo"], b["hi"]) if b is not None else "",
        "lo": b["lo"] if b is not None else math.nan,
        "hi": b["hi"] if b is not None else math.nan,
        "yes_bid": b["yes_bid"] if b is not None else math.nan,
        "yes_ask": b["yes_ask"] if b is not None else math.nan,
        "last": b["last"] if b is not None else math.nan,
        "actual_high": math.nan, "hit": "", "ret_per_dollar": math.nan,
    }
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )
    return row


def load(path: Path = TAPE_PATH) -> pd.DataFrame:
    """Read the tape, dropping duplicate valid_t (bot restarts re-post an hour)."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    # Force the text columns to object dtype: on a fresh tape they are entirely
    # empty, which pandas would otherwise infer as float64 and then refuse to
    # accept a string into during backfill.
    df = pd.read_csv(path, dtype={"hit": "object", "bracket": "object",
                                  "bracket_ticker": "object"})
    return (df.drop_duplicates("valid_t", keep="last")
              .sort_values("valid_t")
              .reset_index(drop=True))


def backfill(path: Path = TAPE_PATH, *, verbose: bool = True) -> int:
    """Fill actual_high / hit / ret_per_dollar for days that have settled.

    Only touches NY-local dates strictly before today -- CLINYC for the current
    day is still preliminary and would be overwritten by the evening issuance.
    Returns the number of rows filled.
    """
    df = load(path)
    if df.empty:
        return 0

    today = datetime.now(NY_TZ).date()
    df["_date"] = pd.to_datetime(df["local_date"]).dt.date
    need = df["actual_high"].isna() & (df["_date"] < today)
    if not need.any():
        if verbose:
            print("Tape already settled through yesterday.")
        return 0

    lo_d, hi_d = df.loc[need, "_date"].min(), df.loc[need, "_date"].max()
    if verbose:
        print(f"Fetching CLI settlement for {lo_d} -> {hi_d} ...")
    cli = sp.fetch_cli(datetime.combine(lo_d, datetime.min.time()),
                       datetime.combine(hi_d, datetime.min.time()) + timedelta(days=1))
    if cli.empty:
        if verbose:
            print("  no CLI records returned; nothing filled")
        return 0
    truth = dict(zip(cli["local_date"], cli["cli_max"]))

    filled = 0
    for i in df.index[need]:
        actual = truth.get(df.at[i, "_date"])
        if actual is None or pd.isna(actual):
            continue
        actual = km.round_half_up(actual)
        lo, hi = df.at[i, "lo"], df.at[i, "hi"]
        ask = df.at[i, "yes_ask"]
        df.at[i, "actual_high"] = actual
        if pd.notna(lo) and pd.notna(hi):
            hit = bool(lo <= actual <= hi)
            df.at[i, "hit"] = "Y" if hit else "N"
            # Net return on $1 bought at the ask: (1/ask - 1) on a win, -1 on a loss.
            # Skipped when the book was empty at capture time (no bid, ask pinned
            # at 1.00) — that is a placeholder quote, not a price you could trade,
            # and treating it as one would invent a 0% return out of nothing.
            bid = df.at[i, "yes_bid"]
            live_book = pd.notna(ask) and 0 < ask < 1.0 and not (pd.notna(bid) and bid == 0.0 and ask >= 1.0)
            if live_book:
                df.at[i, "ret_per_dollar"] = round((1.0 / ask - 1.0) if hit else -1.0, 3)
        filled += 1

    df.drop(columns="_date").to_csv(path, index=False)
    if verbose:
        print(f"  filled {filled} rows")
    return filled


def print_tape(df: pd.DataFrame, limit: int = 60) -> None:
    if df.empty:
        print("Tape is empty.")
        return
    show = df.tail(limit)
    print(f"\n{'date':<11} {'hr':>3} {'pred':>6} {'->':>3} {'bracket':>9} "
          f"{'ask':>5} {'actual':>7} {'hit':>4} {'ret':>7}")
    print("-" * 64)
    for _, r in show.iterrows():
        actual = f"{r['actual_high']:.0f}" if pd.notna(r["actual_high"]) else "-"
        ret = f"{r['ret_per_dollar']:+.2f}" if pd.notna(r["ret_per_dollar"]) else "-"
        ask = f"{r['yes_ask']:.2f}" if pd.notna(r["yes_ask"]) else "-"
        # NaN is truthy, so `r['hit'] or '-'` would print 'nan' on unsettled rows.
        hit = r["hit"] if pd.notna(r["hit"]) and r["hit"] else "-"
        bracket = r["bracket"] if pd.notna(r["bracket"]) else "-"
        print(f"{r['local_date']:<11} {int(r['local_hour']):>3} "
              f"{r['pred_raw']:>6.1f} {'->':>3} {int(r['pred_high']):>3} "
              f"{str(bracket):>9} {ask:>5} {actual:>7} "
              f"{str(hit):>4} {ret:>7}")
    settled = df[df["ret_per_dollar"].notna()]
    if not settled.empty:
        print(f"\n  settled rows: {len(settled)}   "
              f"hit rate: {(settled['hit'] == 'Y').mean() * 100:.0f}%   "
              f"mean return per $1: {settled['ret_per_dollar'].mean():+.2f}")
        print("  (flat-stake reference only -- not the rebalanced ROI)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prediction tape: predictions, prices, outcomes.")
    ap.add_argument("--backfill", action="store_true", help="fill settled days first")
    ap.add_argument("--path", default=str(TAPE_PATH))
    args = ap.parse_args()
    path = Path(args.path)
    if args.backfill:
        backfill(path)
    print_tape(load(path))


if __name__ == "__main__":
    main()
