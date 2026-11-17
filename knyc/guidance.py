"""
Forecast-guidance features (GFS MOS + LAMP) for KNYC, shared by training and
the live bot so both compute identical values. See docs/PLAN_GUIDANCE_RETRAIN.md.

No-leakage rule: a run R may feed row t only if R.runtime + LAG <= valid_t,
where LAG is a conservative estimate of when the bulletin is actually out.
Each feature takes the latest eligible run able to supply it, subject to a
maximum age; past that the feature is NaN (XGBoost's missing branch).

Target date D is the NY civil date of valid_t (the existing `local_date` /
`target_high` key). D's climate window is [D 05:00Z, D+1 05:00Z): midnight to
midnight local standard time, the day the CLI report (and Kalshi) uses.

Library only: no Discord or model imports.
"""
from __future__ import annotations

import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from knyc.build_residuals import observed_so_far

NY_TZ = "America/New_York"
CLIMATE_TZ = "Etc/GMT+5"  # UTC-5 all year (NWS climate day)

MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATION = "KNYC"
HEADERS = {"User-Agent": "knyc-nowcaster/1.0"}

GFS_AVAIL_LAG = pd.Timedelta(hours=5)
LAV_AVAIL_LAG = pd.Timedelta(hours=1)
LAV_CYCLES = (0, 6, 12, 18)
NX_MAX_AGE = pd.Timedelta(hours=36)
GFS_TMP_MAX_AGE = pd.Timedelta(hours=18)
LAV_MAX_AGE = pd.Timedelta(hours=12)

GFS_LEADS = np.array([*range(6, 61, 3), 66, 72], dtype=float)  # 21 leads (hours)
LAV_LEADS = np.arange(1, 39, dtype=float)                      # 38 hourly leads
GFS_MIN_ROWS, LAV_MIN_ROWS = 21, 25

OBS_DAY_COLS = ["hours_into_climate_day", "obs_high_so_far"]
GFS_COLS = ["gfs_nx_today", "gfs_nx_age_h", "gfs_nx_rev", "gfs_tmp_max_rest",
            "gfs_tmp_age_h", "gfs_tmp_now_err", "gfs_tmp_t3", "gfs_tmp_t6"]
LAV_COLS = ["lav_tmp_max_rest", "lav_tmp_now_err", "lav_tmp_t3", "lav_tmp_t6", "lav_age_h"]
GUIDANCE_COLS = GFS_COLS + LAV_COLS
RUNTIME_COLS = ["gfs_nx_runtime", "gfs_tmp_runtime", "lav_runtime"]

HOUR = pd.Timedelta(hours=1)


# ── Fetch / clean ───────────────────────────────────────────────────────────
def fetch_runs(model: str, sts: datetime, ets: datetime, *, retries: int = 4,
               pause_s: float = 3.0, timeout_s: int = 120) -> pd.DataFrame:
    """Raw MOS rows for KNYC with runtime in [sts, ets] (both inclusive)."""
    params = {"station": STATION, "model": model, "format": "csv",
              "sts": pd.Timestamp(sts).strftime("%Y-%m-%dT%H:%MZ"),
              "ets": pd.Timestamp(ets).strftime("%Y-%m-%dT%H:%MZ")}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(MOS_URL, params=params, headers=HEADERS, timeout=timeout_s)
            if r.status_code == 200 and not r.text.startswith("Too many"):
                time.sleep(pause_s)
                return pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()
            last = f"HTTP {r.status_code}: {r.text[:80]!r}"
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = repr(exc)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"MOS fetch {model} {params['sts']}..{params['ets']} failed: {last}")


def clean_runs(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Long table: runtime, ftime (UTC), lead (h), tmp, [n_x]. Duplicates
    dropped, LAMP restricted to the four synoptic cycles, fragments dropped."""
    cols = ["runtime", "ftime", "lead", "tmp"] + (["n_x"] if model == "GFS" else [])
    if df is None or df.empty:
        return empty_runs(model)
    d = df.copy()
    d["runtime"] = pd.to_datetime(d["runtime"], utc=True)
    d["ftime"] = pd.to_datetime(d["ftime"], utc=True)
    d = d.drop_duplicates(["runtime", "ftime"])
    d["tmp"] = pd.to_numeric(d["tmp"], errors="coerce")
    if model == "GFS":
        d["n_x"] = pd.to_numeric(d["n_x"], errors="coerce")
    d["lead"] = ((d["ftime"] - d["runtime"]) / HOUR).round().astype(int)
    if model == "LAV":
        d = d[d["runtime"].dt.hour.isin(LAV_CYCLES) & (d["runtime"].dt.minute == 0)]
    first_lead, min_rows = (6, GFS_MIN_ROWS) if model == "GFS" else (1, LAV_MIN_ROWS)
    g = d.groupby("runtime")["lead"].agg(["min", "size"])
    keep = g.index[(g["min"] == first_lead) & (g["size"] >= min_rows)]
    d = d[d["runtime"].isin(keep)]
    return d[cols].sort_values(["runtime", "ftime"]).reset_index(drop=True)


def empty_runs(model: str) -> pd.DataFrame:
    """A typed, empty run table (what a failed live fetch degrades to)."""
    d = pd.DataFrame({"runtime": pd.Series(dtype="datetime64[ns, UTC]"),
                      "ftime": pd.Series(dtype="datetime64[ns, UTC]"),
                      "lead": pd.Series(dtype=int), "tmp": pd.Series(dtype=float)})
    if model == "GFS":
        d["n_x"] = pd.Series(dtype=float)
    return d


# ── Observation features ────────────────────────────────────────────────────
def target_date(valid: pd.Series) -> pd.Series:
    return valid.dt.tz_convert(NY_TZ).dt.date


def climate_window_start(valid: pd.Series) -> pd.Series:
    """D 05:00Z for D = NY civil date of valid."""
    d = pd.to_datetime(target_date(valid).astype(str))
    return d.dt.tz_localize("UTC") + pd.Timedelta(hours=5)


def obs_day_features(obs: pd.DataFrame) -> pd.DataFrame:
    """hours_into_climate_day and obs_high_so_far for each row of `obs`
    (hourly KNYC rows with valid, tmpf, mxtmpf_6hr). obs_high_so_far equals
    knyc_discord_bot._observed_high(obs[obs.valid <= valid_t], D)."""
    o = obs.sort_values("valid")
    w0 = climate_window_start(o["valid"])
    out = pd.DataFrame(index=o.index)
    out["hours_into_climate_day"] = ((o["valid"] - w0) / HOUR).to_numpy()
    run = observed_so_far(o[["valid", "tmpf", "mxtmpf_6hr"]])
    same_day = (o["valid"].dt.tz_convert(CLIMATE_TZ).dt.date == target_date(o["valid"]))
    out["obs_high_so_far"] = run.where(same_day)
    return out.reindex(obs.index)


# ── Guidance features ───────────────────────────────────────────────────────
def _run_matrix(runs: pd.DataFrame, leads: np.ndarray, field: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """runs x leads array of `field` (NaN where missing)."""
    if runs.empty:
        return pd.DatetimeIndex([], tz="UTC"), np.empty((0, len(leads)))
    piv = runs.pivot(index="runtime", columns="lead", values=field)
    piv = piv.reindex(columns=leads.astype(int))
    return pd.DatetimeIndex(piv.index), piv.to_numpy(dtype=float)


def _select_run(valid: np.ndarray, cutoff: np.ndarray, runtimes: pd.DatetimeIndex,
                lag: pd.Timedelta) -> np.ndarray:
    """Index of the latest run with runtime + lag <= cutoff (-1 if none)."""
    if len(runtimes) == 0:
        return np.full(len(cutoff), -1)
    avail = (runtimes + lag).asi8
    return np.searchsorted(avail, cutoff, side="right") - 1


def _interp(mat: np.ndarray, ridx: np.ndarray, leads: np.ndarray, at_lead: np.ndarray) -> np.ndarray:
    """Linear interpolation of mat[ridx] at fractional lead `at_lead`."""
    out = np.full(len(ridx), np.nan)
    ok = (ridx >= 0) & (at_lead >= leads[0]) & (at_lead <= leads[-1])
    if not ok.any():
        return out
    r, x = ridx[ok], at_lead[ok]
    j = np.clip(np.searchsorted(leads, x, side="right") - 1, 0, len(leads) - 2)
    x0, x1 = leads[j], leads[j + 1]
    y0, y1 = mat[r, j], mat[r, j + 1]
    w = (x - x0) / (x1 - x0)
    out[ok] = y0 + w * (y1 - y0)
    return out


def _window_max(mat: np.ndarray, ridx: np.ndarray, leads: np.ndarray,
                lo_excl: np.ndarray, lo_incl: np.ndarray, hi_excl: np.ndarray) -> np.ndarray:
    """max over leads L with L > lo_excl, L >= lo_incl, L < hi_excl."""
    out = np.full(len(ridx), np.nan)
    ok = ridx >= 0
    if not ok.any():
        return out
    vals = mat[ridx[ok]]
    L = leads[None, :]
    m = (L > lo_excl[ok, None]) & (L >= lo_incl[ok, None]) & (L < hi_excl[ok, None])
    vals = np.where(m, vals, np.nan)
    with np.errstate(all="ignore"):
        has = ~np.isnan(vals).all(axis=1)
        res = np.full(vals.shape[0], np.nan)
        res[has] = np.nanmax(vals[has], axis=1)
    out[ok] = res
    return out


def _hours(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) in hours for int64 ns arrays."""
    return (a - b) / 3.6e12


def _runtime_ns(runtimes: pd.DatetimeIndex, idx: np.ndarray) -> np.ndarray:
    """int64 ns runtime for each selected index (0 where idx < 0)."""
    if len(runtimes) == 0:
        return np.zeros(len(idx), dtype=np.int64)
    return np.where(idx >= 0, runtimes.asi8[np.maximum(idx, 0)], 0)


def _runtime_col(rt_ns: np.ndarray, ok: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(rt_ns, utc=True)).where(ok)


_NEVER = np.iinfo(np.int64).min + 1


def _cutoff(valid: np.ndarray, skip: np.ndarray, runtimes: pd.DatetimeIndex,
            lag: pd.Timedelta) -> np.ndarray:
    """Selection cutoff per row: valid_t, or (late-bulletin masking) the
    availability time of the run before the newest eligible one."""
    cutoff = valid.copy()
    if not skip.any():
        return cutoff
    newest = _select_run(valid, valid, runtimes, lag)
    back = skip & (newest >= 1)
    if back.any():
        cutoff[back] = (runtimes[newest[back] - 1] + lag).asi8
    cutoff[skip & (newest < 1)] = _NEVER
    return cutoff


def guidance_features(rows: pd.DataFrame, gfs: pd.DataFrame, lav: pd.DataFrame,
                      *, skip_newest: bool | np.ndarray = False) -> pd.DataFrame:
    """The 13 guidance columns (+ the runtimes used) for each row.

    rows: needs `valid` (UTC) and `tmpf`. gfs / lav: output of clean_runs.
    skip_newest: True (or a boolean mask per row) recomputes as if each
    model's newest eligible run had not arrived yet (late-bulletin masking).
    """
    n = len(rows)
    valid_ts = pd.DatetimeIndex(rows["valid"]).tz_convert("UTC")
    valid = valid_ts.asi8
    tmpf = rows["tmpf"].to_numpy(dtype=float)
    w0 = pd.DatetimeIndex(climate_window_start(pd.Series(valid_ts))).asi8
    w1 = w0 + 24 * 3_600_000_000_000
    skip = np.broadcast_to(np.asarray(skip_newest, dtype=bool), (n,))
    out = pd.DataFrame(index=rows.index)

    # ---------- GFS ----------
    g_rt, g_tmp = _run_matrix(gfs, GFS_LEADS, "tmp")
    cutoff = _cutoff(valid, skip, g_rt, GFS_AVAIL_LAG)
    ridx = _select_run(valid, cutoff, g_rt, GFS_AVAIL_LAG)
    rt_ns = _runtime_ns(g_rt, ridx)
    age = np.where(ridx >= 0, _hours(valid, rt_ns), np.nan)
    ridx_t = np.where(age <= GFS_TMP_MAX_AGE / HOUR, ridx, -1)
    lead_valid = np.where(ridx_t >= 0, _hours(valid, rt_ns), np.nan)
    out["gfs_tmp_max_rest"] = _window_max(g_tmp, ridx_t, GFS_LEADS, lead_valid,
                                          _hours(w0, rt_ns), _hours(w1, rt_ns))
    out["gfs_tmp_age_h"] = np.where(ridx_t >= 0, age, np.nan)
    out["gfs_tmp_t3"] = _interp(g_tmp, ridx_t, GFS_LEADS, lead_valid + 3)
    out["gfs_tmp_t6"] = _interp(g_tmp, ridx_t, GFS_LEADS, lead_valid + 6)
    out["gfs_tmp_runtime"] = _runtime_col(rt_ns, ridx_t >= 0)

    # now_err: latest eligible run with rows on both sides of valid (step back
    # while the run's first lead is still in the future).
    r_now = ridx.copy()
    for _ in range(3):
        lead_now = _hours(valid, _runtime_ns(g_rt, r_now))
        step = (r_now >= 0) & (lead_now < GFS_LEADS[0])
        if not step.any():
            break
        r_now = np.where(step, r_now - 1, r_now)
    rt_now = _runtime_ns(g_rt, r_now)
    age_now = np.where(r_now >= 0, _hours(valid, rt_now), np.nan)
    r_now = np.where(age_now <= GFS_TMP_MAX_AGE / HOUR, r_now, -1)
    out["gfs_tmp_now_err"] = tmpf - _interp(g_tmp, r_now, GFS_LEADS, _hours(valid, rt_now))

    # n_x for D: the value at ftime (D+1) 00Z, from the latest eligible run carrying it.
    if gfs.empty:
        gfs = empty_runs("GFS")
    nx = gfs[gfs["n_x"].notna() & (gfs["ftime"].dt.hour == 0)][["runtime", "ftime", "n_x"]].copy()
    nx["tdate"] = (nx["ftime"] - pd.Timedelta(days=1)).dt.date
    nx = nx.sort_values(["tdate", "runtime"])
    nx["prev_n_x"] = nx.groupby("tdate")["n_x"].shift(1)
    nx["avail"] = nx["runtime"] + GFS_AVAIL_LAG
    left = pd.DataFrame({"pos": np.arange(n),
                         "cutoff": pd.to_datetime(cutoff, utc=True),
                         "tdate": target_date(pd.Series(valid_ts)).to_numpy()})
    left = left.sort_values("cutoff")
    nx = nx.sort_values("avail")
    m = pd.merge_asof(left, nx[["avail", "tdate", "runtime", "n_x", "prev_n_x"]],
                      left_on="cutoff", right_on="avail", by="tdate", direction="backward")
    m = m.sort_values("pos")
    nx_age = ((pd.DatetimeIndex(valid_ts) - pd.DatetimeIndex(m["runtime"])) / HOUR).to_numpy(dtype=float)
    ok = ~np.isnan(nx_age) & (nx_age <= NX_MAX_AGE / HOUR)
    out["gfs_nx_today"] = np.where(ok, m["n_x"].to_numpy(dtype=float), np.nan)
    out["gfs_nx_age_h"] = np.where(ok, nx_age, np.nan)
    out["gfs_nx_rev"] = np.where(ok, (m["n_x"] - m["prev_n_x"]).to_numpy(dtype=float), np.nan)
    out["gfs_nx_runtime"] = pd.DatetimeIndex(m["runtime"]).where(ok)

    # ---------- LAMP ----------
    l_rt, l_tmp = _run_matrix(lav, LAV_LEADS, "tmp")
    cutoff_l = _cutoff(valid, skip, l_rt, LAV_AVAIL_LAG)
    lidx = _select_run(valid, cutoff_l, l_rt, LAV_AVAIL_LAG)
    lrt_ns = _runtime_ns(l_rt, lidx)
    lage = np.where(lidx >= 0, _hours(valid, lrt_ns), np.nan)
    lidx = np.where(lage <= LAV_MAX_AGE / HOUR, lidx, -1)
    llead_valid = np.where(lidx >= 0, _hours(valid, lrt_ns), np.nan)
    ceil_lead = np.ceil(llead_valid - 1e-9)          # ftime = ceil_hour(valid_t)
    out["lav_tmp_max_rest"] = _window_max(l_tmp, lidx, LAV_LEADS, llead_valid,
                                          _hours(w0, lrt_ns), _hours(w1, lrt_ns))
    out["lav_tmp_now_err"] = tmpf - _interp(l_tmp, lidx, LAV_LEADS, ceil_lead)
    out["lav_tmp_t3"] = _interp(l_tmp, lidx, LAV_LEADS, ceil_lead + 3)
    out["lav_tmp_t6"] = _interp(l_tmp, lidx, LAV_LEADS, ceil_lead + 6)
    out["lav_age_h"] = np.where(lidx >= 0, lage, np.nan)
    out["lav_runtime"] = _runtime_col(lrt_ns, lidx >= 0)

    return out[GUIDANCE_COLS + RUNTIME_COLS]
