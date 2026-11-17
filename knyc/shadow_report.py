"""
Side-by-side trial report: current model vs the candidate, from
logs/shadow_log.csv, scored against the official CLI high (what Kalshi
settles on). Also reports how the live guidance feeds behaved: fallback rate,
missing sources, and how late each run arrived versus the lag the features
assume (newest run seen at fetch time).

    python -m knyc.shadow_report                         # logs/shadow_log.csv
    python -m knyc.shadow_report --log data/server_pull/shadow_log.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from knyc import guidance as g
from knyc import paths
from knyc.market_scorer import HOUR_BLOCKS, settlement_truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(paths.SHADOW_LOG))
    args = ap.parse_args()

    s = pd.read_csv(args.log)
    s["valid"] = pd.to_datetime(s["valid_t"], utc=True)
    s = s.drop_duplicates("valid", keep="first")
    s["local_date"] = s["valid"].dt.tz_convert("America/New_York").dt.date
    print(f"{len(s)} logged hours, {s['local_date'].nunique()} days "
          f"({s['local_date'].min()} -> {s['local_date'].max()}), candidate {s['candidate'].iloc[-1]}")

    truth = settlement_truth(sorted(s["local_date"].unique()), verbose=False)
    s["actual"] = s["local_date"].map(truth)
    scored = s.dropna(subset=["actual"])
    print(f"settled days: {scored['local_date'].nunique()}\n")

    print("Average miss on the day's high (F), by NY-local hour block")
    print(f"{'block':>6} {'n':>5} | {'current':>8} {'v2':>8} | {'current re':>10} {'v2 re':>8}")
    rows = [(f"{a:02d}-{b:02d}", scored[scored["local_hour"].between(a, b)]) for a, b in HOUR_BLOCKS]
    rows.append(("all", scored))
    for name, d in rows:
        if d.empty:
            continue
        e = {c: (d[c] - d["actual"]).abs().mean()
             for c in ("pred_high_prod", "pred_high_cand", "reassessed_prod", "reassessed_cand")}
        print(f"{name:>6} {len(d):>5} | {e['pred_high_prod']:8.2f} {e['pred_high_cand']:8.2f} | "
              f"{e['reassessed_prod']:10.2f} {e['reassessed_cand']:8.2f}")
    days = scored.groupby("local_date").apply(
        lambda d: (d["pred_high_cand"] - d["actual"]).abs().mean() < (d["pred_high_prod"] - d["actual"]).abs().mean(),
        include_groups=False)
    print(f"\nv2 closer on {int(days.sum())} of {len(days)} days")

    print("\nLive guidance health")
    print(f"  fallback (no guidance at all): {s['fallback'].astype(bool).mean():.1%} of hours")
    for m, col in (("GFS", "gfs_tmp_t3"), ("LAMP", "lav_tmp_t3"), ("NBM", "nbs_tmp_t3")):
        if col in s:
            print(f"  {m:<4} features present: {s[col].notna().mean():.1%}")
    # Arrival: the newest run already on IEM at :55, by cycle age.
    for m, lag in (("gfs", g.GFS_AVAIL_LAG), ("lav", g.LAV_AVAIL_LAG), ("nbs", g.NBS_AVAIL_LAG)):
        col = f"{m}_newest_seen"
        if col not in s:
            continue
        seen = pd.to_datetime(s[col], utc=True, errors="coerce")
        age = ((s["valid"] + pd.Timedelta(minutes=4)) - seen) / pd.Timedelta(hours=1)
        print(f"  {m.upper():<4} newest run on IEM at post time: median {np.nanmedian(age):.1f} h old, "
              f"min {np.nanmin(age):.1f} h (lag assumed {lag / pd.Timedelta(hours=1):.0f} h)")


if __name__ == "__main__":
    main()
