"""
Feature parity check: live bot vs training builder
==================================================
The model only works if the bot computes features exactly the way
build_training_data.py did. Three silent train/serve mismatches ran in
production for two months (solar geometry pinned to 1970 by a datetime-unit
change, clear sky never reported, three upstream stations an hour stale). This
check would have caught the first two on day one.

Two parts, both against the last few days of real data (needs network):

  A. Engineering parity — feed the SAME raw IEM obs through
     knyc_discord_bot.engineer_features and build_training_data.build_features
     and diff every model feature column.
  B. Source parity — compare the bot's live feed (aviationweather.gov, what it
     actually uses) against the IEM archive the training data came from, for
     the inputs the bot parses itself (temperature, cloud cover).

    python tools/check_feature_parity.py            # exit 1 on any mismatch

Run it after any change to feature code, pandas/numpy upgrades, or a new
upstream data source.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DISCORD_TOKEN", "parity-check")
os.environ.setdefault("GUILD_ID", "1")

import knyc_discord_bot as bot  # noqa: E402
from knyc import build_training_data as btd  # noqa: E402
from knyc import guidance  # noqa: E402

DAYS = 4
TOL = 1e-6
# Columns that differ by design, not by bug:
#   tmpf_clim_global / tmpf_anomaly — training derives climatology from its
#     own rows; the bot uses the saved tmpf_climatology.csv.
#   *_drct — the bot resolves VRB/calm upstream winds to 0 where the station
#     reported a speed; training leaves them NaN (XGBoost's missing branch).
EXPECTED_DIFF = {"tmpf_clim_global", "tmpf_anomaly"} | {
    f"{s}_drct" for s in bot.UPSTREAM}
SOURCE_MIN_AGREEMENT = 0.90


def part_a(end: datetime) -> list[str]:
    start = end - timedelta(days=DAYS)
    knyc = btd.fetch_iem_asos("KNYC", start, end, primary=True)
    upstream = {s: btd.fetch_iem_asos(s, start, end, primary=False) for s in bot.UPSTREAM}

    train = btd.build_features(knyc, upstream, pd.DataFrame()).set_index("valid")
    live = bot._apply_climatology(bot.engineer_features(knyc, upstream)).set_index("valid")

    # Compare where both pipelines have a complete row (the last 24 hours of
    # history are needed for the 24h lag/roll features).
    common = train.index.intersection(live.index)[24:]
    problems = []
    print(f"A. engineering parity over {len(common)} hours, {len(bot.FEATURE_COLS)} features")
    for col in bot.FEATURE_COLS:
        a = train.loc[common, col].astype(float)
        b = live.loc[common, col].astype(float)
        both = a.notna() & b.notna()
        nan_mismatch = int((a.isna() != b.isna()).sum())
        diff = float((a[both] - b[both]).abs().max()) if both.any() else 0.0
        if diff > TOL or nan_mismatch:
            tag = "expected" if col in EXPECTED_DIFF else "MISMATCH"
            print(f"   {tag:<8} {col:<22} max|diff|={diff:.4g}  NaN-mismatch={nan_mismatch}")
            if col not in EXPECTED_DIFF:
                problems.append(col)
    if not problems:
        print("   OK - every non-exempt feature matches")

    # Guidance retrain observation feature (docs/PLAN_GUIDANCE_RETRAIN.md):
    # the training path (knyc.guidance.obs_day_features) must equal the bot's
    # live floor (_observed_high on the rows seen so far, civil date of valid_t).
    obs = train.reset_index()[["valid", "tmpf", "mxtmpf_6hr"]]
    feat = guidance.obs_day_features(obs)["obs_high_so_far"]
    ny_date = obs["valid"].dt.tz_convert("America/New_York").dt.date
    bot_vals = np.array([
        (lambda v: np.nan if v is None else v)(bot._observed_high(obs.iloc[: i + 1], ny_date.iloc[i]))
        for i in range(len(obs))])
    both = ~np.isnan(bot_vals) & feat.notna().to_numpy()
    nan_mismatch = int((np.isnan(bot_vals) != feat.isna().to_numpy()).sum())
    diff = float(np.abs(bot_vals[both] - feat.to_numpy()[both]).max()) if both.any() else 0.0
    if diff > TOL or nan_mismatch:
        print(f"   MISMATCH obs_high_so_far        max|diff|={diff:.4g}  NaN-mismatch={nan_mismatch}")
        problems.append("obs_high_so_far")
    else:
        print(f"   OK - obs_high_so_far matches the bot's floor on all {len(obs)} rows")
    return problems


def part_b(end: datetime) -> list[str]:
    hours = DAYS * 24
    live = bot.fetch_avwx("KNYC", hours_back=hours)
    iem = bot.fetch_iem("KNYC", hours_back=hours)
    m = live.merge(iem, on="valid", suffixes=("_live", "_iem"))
    print(f"\nB. source parity (aviationweather vs IEM), {len(m)} matched KNYC hours")
    if m.empty:
        return ["no overlapping hours"]

    problems = []
    t_ok = ((m["tmpf_live"] - m["tmpf_iem"]).abs() <= 1.0).mean()
    ok_live = m.apply(lambda r: bot.cloud_oktas({c: r[f"{c}_live"] for c in ("skyc1", "skyc2", "skyc3")}), axis=1)
    ok_iem = m.apply(lambda r: bot.cloud_oktas({c: r[f"{c}_iem"] for c in ("skyc1", "skyc2", "skyc3")}), axis=1)
    c_ok = (ok_live == ok_iem).mean()
    print(f"   tmpf within 1F : {t_ok:.0%}")
    print(f"   cloud oktas    : {c_ok:.0%} agree   "
          f"(live NaN {ok_live.isna().mean():.0%}, IEM clear {(ok_iem == 0).mean():.0%}, "
          f"live clear {(ok_live == 0).mean():.0%})")
    for name, val in (("tmpf", t_ok), ("cloud_oktas", c_ok)):
        if val < SOURCE_MIN_AGREEMENT:
            problems.append(f"{name} agreement {val:.0%} < {SOURCE_MIN_AGREEMENT:.0%}")
    return problems


def main() -> int:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    problems = part_a(end) + part_b(end)
    print("\nRESULT:", "PASS" if not problems else f"FAIL - {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
