"""
Offline tests for knyc/guidance.py (docs/PLAN_GUIDANCE_RETRAIN.md section 7),
on synthetic run tables. No network.

    python tests/test_guidance.py
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("GUILD_ID", "1")

from knyc import guidance as g  # noqa: E402

H = pd.Timedelta(hours=1)
GFS_LEADS = [*range(6, 61, 3), 66, 72]


def synth_gfs(start: str, days: int) -> pd.DataFrame:
    """Four runs/day. tmp = 50 + run index/10 + lead/100 (identifies the run).
    n_x sits on the leads the real archive uses (00/12Z runs: 24,36,..,72;
    06/18Z runs: 18,30,..,66), i.e. on 00Z (max) and 12Z (min) ftimes:
    n_x = 70 + run index on 00Z ftimes, 30 + run index on 12Z ftimes."""
    rows = []
    for i, rt in enumerate(pd.date_range(start, periods=days * 4, freq="6h", tz="UTC")):
        nx_leads = {24, 36, 48, 60, 72} if rt.hour in (0, 12) else {18, 30, 42, 54, 66}
        for L in GFS_LEADS:
            ft = rt + L * H
            nx = (70.0 + i if ft.hour == 0 else 30.0 + i) if L in nx_leads else np.nan
            rows.append({"runtime": rt, "ftime": ft, "lead": L,
                         "tmp": 50 + i / 10 + L / 100, "n_x": nx})
    return pd.DataFrame(rows)


def synth_lav(start: str, days: int, cycles=(0, 6, 12, 18), extra=()) -> pd.DataFrame:
    rows = []
    rts = [rt for rt in pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
           if rt.hour in cycles or rt.hour in extra]
    for i, rt in enumerate(rts):
        for L in range(1, 39):
            rows.append({"runtime": rt, "ftime": rt + L * H, "lead": L, "tmp": 40 + i / 10 + L / 100})
    return pd.DataFrame(rows)


GFS = g.clean_runs(synth_gfs("2026-01-08", 6).astype({"runtime": str, "ftime": str}), "GFS")
LAV = g.clean_runs(synth_lav("2026-01-08", 6).astype({"runtime": str, "ftime": str}), "LAV")


def rows_at(times) -> pd.DataFrame:
    v = pd.to_datetime(list(times), utc=True)
    return pd.DataFrame({"valid": v, "tmpf": np.full(len(v), 45.0)})


def feats(times, **kw) -> pd.DataFrame:
    return g.guidance_features(rows_at(times), GFS, LAV, **kw)


# 1. No leakage on random rows
def test_no_leakage():
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2026-01-09 00:51", tz="UTC")
    times = [base + int(h) * H for h in rng.integers(0, 4 * 24, 2000)]
    f = feats(times)
    v = pd.to_datetime(times, utc=True)
    for col, lag in (("gfs_nx_runtime", g.GFS_AVAIL_LAG), ("gfs_tmp_runtime", g.GFS_AVAIL_LAG),
                     ("lav_runtime", g.LAV_AVAIL_LAG)):
        rt = pd.DatetimeIndex(f[col])
        ok = rt.notna()
        assert ok.mean() > 0.9, col
        assert ((rt[ok] + lag) <= v[ok]).all(), col


# 2. Future invariance
def test_future_invariance():
    t = pd.Timestamp("2026-01-10 15:51", tz="UTC")
    full = g.guidance_features(rows_at([t]), GFS, LAV)
    gp = GFS[GFS["runtime"] + g.GFS_AVAIL_LAG <= t]
    lp = LAV[LAV["runtime"] + g.LAV_AVAIL_LAG <= t]
    past = g.guidance_features(rows_at([t]), gp, lp)
    pd.testing.assert_frame_equal(full, past)


# 3. Selection table (plan 3.1), standard time, D = 2026-01-10
def test_selection_table():
    D = "2026-01-10"
    cases = [("05:51", "00", "00", "00"), ("10:51", "00", "00", "06"),
             ("11:51", "06", "06", "06"), ("17:51", "06", "12", "12"),
             ("23:51", "06", "18", "18")]
    f = feats([f"{D} {c[0]}" for c in cases])
    for (_, nx, tmp, lav), (_, r) in zip(cases, f.iterrows()):
        assert r["gfs_nx_runtime"] == pd.Timestamp(f"{D} {nx}:00", tz="UTC"), (r, nx)
        assert r["gfs_tmp_runtime"] == pd.Timestamp(f"{D} {tmp}:00", tz="UTC"), (r, tmp)
        assert r["lav_runtime"] == pd.Timestamp(f"{D} {lav}:00", tz="UTC"), (r, lav)
    # nx age at 05:51 is 5.85 h; value = 70 + run index (00Z Jan 10 is run 8);
    # revision vs the previous run carrying D's max (18Z Jan 9, run 7).
    assert abs(f["gfs_nx_age_h"].iloc[0] - 5.85) < 1e-9
    assert f["gfs_nx_today"].iloc[0] == 78.0 and f["gfs_nx_rev"].iloc[0] == 1.0
    # 17:51Z row, 12Z Jan 10 = run 10 in both tables. GFS t3 at lead 8.85 h
    # interpolates linearly; LAMP t3 is the 21Z value (lead 9).
    assert abs(f["gfs_tmp_t3"].iloc[3] - (50 + 1.0 + 8.85 / 100)) < 1e-9
    assert abs(f["lav_tmp_t3"].iloc[3] - (40 + 1.0 + 9 / 100)) < 1e-9
    assert abs(f["lav_tmp_now_err"].iloc[3] - (45 - (40 + 1.0 + 6 / 100))) < 1e-9


def test_selection_edt_date():
    # 2026-07-15 (EDT). 12:51 EDT = 16:51Z: GFS n_x from 06Z, tmp from 06Z (12Z+5h = 17Z > 16:51Z)
    gfs = g.clean_runs(synth_gfs("2026-07-13", 4).astype({"runtime": str, "ftime": str}), "GFS")
    lav = g.clean_runs(synth_lav("2026-07-13", 4).astype({"runtime": str, "ftime": str}), "LAV")
    f = g.guidance_features(rows_at(["2026-07-15 16:51"]), gfs, lav).iloc[0]
    assert f["gfs_nx_runtime"] == pd.Timestamp("2026-07-15 06:00", tz="UTC")
    assert f["gfs_tmp_runtime"] == pd.Timestamp("2026-07-15 06:00", tz="UTC")
    assert f["lav_runtime"] == pd.Timestamp("2026-07-15 12:00", tz="UTC")


# 4. A 12Z run of D never supplies gfs_nx_today for D
def test_nx_mapping():
    f = feats(["2026-01-10 23:51", "2026-01-11 03:51"])
    for rt in f["gfs_nx_runtime"]:
        assert rt.hour in (0, 6) or rt.date() < date(2026, 1, 10)


# 5. LAMP cycle filter
def test_lav_cycle_filter():
    raw = synth_lav("2026-01-08", 6, extra=(11,)).astype({"runtime": str, "ftime": str})
    lav = g.clean_runs(raw, "LAV")
    assert set(lav["runtime"].dt.hour) == {0, 6, 12, 18}
    f = g.guidance_features(rows_at(["2026-01-10 12:51"]), GFS, lav).iloc[0]
    assert f["lav_runtime"] == pd.Timestamp("2026-01-10 06:00", tz="UTC")


# 6. Max age
def test_max_age():
    t = pd.Timestamp("2026-01-12 23:51", tz="UTC")
    old_g = GFS[GFS["runtime"] <= t - 40 * H]
    old_l = LAV[LAV["runtime"] <= t - 13 * H]
    f = g.guidance_features(rows_at([t]), old_g, old_l).iloc[0]
    assert np.isnan(f["gfs_nx_today"]) and np.isnan(f["gfs_tmp_max_rest"])
    assert np.isnan(f["lav_tmp_t3"]) and np.isnan(f["lav_age_h"])


# 7. Observation features equal the bot's _observed_high
def test_obs_features_match_bot():
    import knyc_discord_bot as bot
    v = pd.date_range("2026-03-07 00:51", "2026-03-10 23:51", freq="1h", tz="UTC")  # DST starts 03-08
    rng = np.random.default_rng(1)
    obs = pd.DataFrame({"valid": v, "tmpf": 40 + rng.normal(0, 5, len(v)), "mxtmpf_6hr": np.nan})
    six = obs["valid"].dt.hour.isin([0, 6, 12, 18]) & (obs["valid"].dt.minute == 51)
    obs.loc[six, "mxtmpf_6hr"] = obs.loc[six, "tmpf"] + 3   # includes 05:51Z groups straddling 05:00Z
    f = g.obs_day_features(obs)
    ny = obs["valid"].dt.tz_convert(g.NY_TZ)
    for i in range(24, len(obs)):
        t = obs["valid"].iloc[i]
        exp = bot._observed_high(obs[obs["valid"] <= t], ny.iloc[i].date())
        got = f["obs_high_so_far"].iloc[i]
        if exp is None:
            assert np.isnan(got), (t, got)
        else:
            assert abs(exp - got) < 1e-9, (t, exp, got)


# 8. All guidance missing -> NaN columns, model still predicts
def test_all_missing():
    f = g.guidance_features(rows_at(["2026-01-10 12:51"]), g.empty_runs("GFS"), g.empty_runs("LAV"))
    assert f[g.GUIDANCE_COLS].isna().all().all()
    from xgboost import XGBRegressor
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(200, len(g.GUIDANCE_COLS))),
                     columns=g.GUIDANCE_COLS)
    m = XGBRegressor(n_estimators=5).fit(X, X.iloc[:, 0])
    assert np.isfinite(m.predict(f[g.GUIDANCE_COLS])).all()


# Late-bulletin masking steps back exactly one run
def test_skip_newest():
    f = feats(["2026-01-10 17:51"], skip_newest=True).iloc[0]
    assert f["gfs_tmp_runtime"] == pd.Timestamp("2026-01-10 06:00", tz="UTC")
    assert f["lav_runtime"] == pd.Timestamp("2026-01-10 06:00", tz="UTC")


# ── NBM (NBS) ───────────────────────────────────────────────────────────────
def synth_nbs(start: str, days: int, cycles=(1, 7, 13, 19)) -> pd.DataFrame:
    """NBS-like runs: 3-hourly ftimes on the 00/03/.. grid, so a 01Z run's
    first lead is 5 h; max/min in `txn` (post-2021 name). tmp = 60 + i/10 + L/100."""
    rows = []
    rts = [rt for rt in pd.date_range(start, periods=days * 24, freq="1h", tz="UTC") if rt.hour in cycles]
    for i, rt in enumerate(rts):
        first = rt.ceil("3h") + 3 * H if rt.ceil("3h") - rt < 3 * H else rt.ceil("3h")
        for k in range(23):
            ft = first + 3 * k * H
            rows.append({"runtime": rt, "ftime": ft, "tmp": 60 + i / 10 + ((ft - rt) / H) / 100,
                         "txn": (80.0 + i) if ft.hour == 0 else (20.0 + i) if ft.hour == 12 else np.nan})
    return pd.DataFrame(rows)


NBS = g.clean_runs(synth_nbs("2026-01-08", 6).astype({"runtime": str, "ftime": str}), "NBS")


def test_nbs_clean_and_select():
    assert set(NBS["runtime"].dt.hour) == {1, 7, 13, 19}
    assert NBS.groupby("runtime")["lead"].min().isin([5, 6]).all()
    assert NBS["n_x"].notna().any()                      # txn -> n_x
    t = pd.Timestamp("2026-01-10 17:51", tz="UTC")      # 13Z + 3h lag = 16Z <= 17:51
    f = g.guidance_features(rows_at([t]), GFS, LAV, NBS).iloc[0]
    assert f["nbs_tmp_runtime"] == pd.Timestamp("2026-01-10 13:00", tz="UTC")
    assert (f["nbs_tmp_runtime"] + g.NBS_AVAIL_LAG) <= t
    # 13Z Jan 10 is run index 2*4 + 2 = 10. t3 at 20:51Z = lead 7.85 h,
    # between the 18Z (lead 5) and 21Z (lead 8) rows -> linear.
    assert abs(f["nbs_tmp_t3"] - (60 + 1.0 + 7.85 / 100)) < 1e-9
    # n_x for Jan 10 = value at Jan 11 00Z from the latest eligible run (13Z Jan 10).
    assert f["nbs_nx_today"] == 80.0 + 10 and f["nbs_nx_runtime"].hour == 13
    # GFS / LAMP columns are unaffected by adding NBS.
    pd.testing.assert_frame_equal(
        g.guidance_features(rows_at([t]), GFS, LAV)[g.GFS_COLS + g.LAV_COLS],
        g.guidance_features(rows_at([t]), GFS, LAV, NBS)[g.GFS_COLS + g.LAV_COLS])


def test_nbs_no_leakage():
    rng = np.random.default_rng(3)
    base = pd.Timestamp("2026-01-09 00:51", tz="UTC")
    times = [base + int(h) * H for h in rng.integers(0, 4 * 24, 1000)]
    f = g.guidance_features(rows_at(times), GFS, LAV, NBS)
    v = pd.to_datetime(times, utc=True)
    for col in ("nbs_nx_runtime", "nbs_tmp_runtime"):
        rt = pd.DatetimeIndex(f[col])
        ok = rt.notna()
        assert ok.mean() > 0.9 and ((rt[ok] + g.NBS_AVAIL_LAG) <= v[ok]).all(), col


# pandas 3 parses datetimes to microseconds; features must not depend on unit.
def test_datetime_unit_independent():
    times = [f"2026-01-10 {h:02d}:51" for h in range(0, 24, 2)]
    ref = g.guidance_features(rows_at(times), GFS, LAV, NBS)

    def as_unit(df, unit):
        d = df.copy()
        for c in ("runtime", "ftime"):
            d[c] = d[c].astype(f"datetime64[{unit}, UTC]")
        return d

    for unit in ("us", "s"):
        rows = rows_at(times)
        rows["valid"] = rows["valid"].astype(f"datetime64[{unit}, UTC]")
        got = g.guidance_features(rows, as_unit(GFS, unit), as_unit(LAV, unit), as_unit(NBS, unit))
        cols = g.GUIDANCE_COLS
        np.testing.assert_allclose(got[cols].to_numpy(float), ref[cols].to_numpy(float), equal_nan=True)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                import traceback
                print(f"FAIL {name}: {exc!r}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
