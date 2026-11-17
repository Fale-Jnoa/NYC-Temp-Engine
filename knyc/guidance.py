"""
Forecast-guidance features (GFS MOS, LAMP, NBM) for KNYC, shared by training
and the live bot so both compute identical values. See
docs/PLAN_GUIDANCE_RETRAIN.md and docs/PLAN_GUIDANCE_V2.md.

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
HOUR = pd.Timedelta(hours=1)

GFS_AVAIL_LAG = pd.Timedelta(hours=5)
LAV_AVAIL_LAG = pd.Timedelta(hours=1)
# NBM text (NBS): no arrival time in the archive; 3 h is a conservative guess
# to be confirmed by the shadow log (docs/PLAN_GUIDANCE_V2.md).
NBS_AVAIL_LAG = pd.Timedelta(hours=3)
LAV_CYCLES = (0, 6, 12, 18)
# The NBS archive keeps four cycles a day; the hours moved over time
# (00/07/12/19Z, then 01/07/13/19Z, now 00/06/12/18Z).
NBS_CYCLES = (0, 1, 6, 7, 12, 13, 18, 19)
NX_MAX_AGE = pd.Timedelta(hours=36)
GFS_TMP_MAX_AGE = pd.Timedelta(hours=18)
NBS_TMP_MAX_AGE = pd.Timedelta(hours=18)
LAV_MAX_AGE = pd.Timedelta(hours=12)

MODEL_SPEC = {
    # first lead(s) accepted, minimum rows per run
    "GFS": {"first_leads": (6,), "min_rows": 21},
    "LAV": {"first_leads": (1,), "min_rows": 25},
    "NBS": {"first_leads": (5, 6), "min_rows": 21},
}

OBS_DAY_COLS = ["hours_into_climate_day", "obs_high_so_far"]
GFS_COLS = ["gfs_nx_today", "gfs_nx_age_h", "gfs_nx_rev", "gfs_tmp_max_rest",
            "gfs_tmp_age_h", "gfs_tmp_now_err", "gfs_tmp_t3", "gfs_tmp_t6"]
LAV_COLS = ["lav_tmp_max_rest", "lav_tmp_now_err", "lav_tmp_t3", "lav_tmp_t6", "lav_age_h"]
NBS_COLS = [c.replace("gfs_", "nbs_") for c in GFS_COLS]
SOURCE_COLS = {"GFS": GFS_COLS, "LAV": LAV_COLS, "NBS": NBS_COLS}
GUIDANCE_COLS = GFS_COLS + LAV_COLS + NBS_COLS
RUNTIME_COLS = ["gfs_nx_runtime", "gfs_tmp_runtime", "lav_runtime", "nbs_nx_runtime", "nbs_tmp_runtime"]


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


def empty_runs(model: str) -> pd.DataFrame:
    """A typed, empty run table (what a failed live fetch degrades to)."""
    d = pd.DataFrame({"runtime": pd.Series(dtype="datetime64[ns, UTC]"),
                      "ftime": pd.Series(dtype="datetime64[ns, UTC]"),
                      "lead": pd.Series(dtype=int), "tmp": pd.Series(dtype=float)})
    if model in ("GFS", "NBS"):
        d["n_x"] = pd.Series(dtype=float)
    return d


def clean_runs(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Long table: runtime, ftime (UTC), lead (h), tmp, [n_x]. Duplicates
    dropped, cycle filters applied (LAMP: synoptic four; NBS: the archived
    four of each era), fragments dropped. NBS's max/min column is `n_x`
    before 2021 and `txn` after; both land in `n_x`."""
    has_nx = model in ("GFS", "NBS")
    cols = ["runtime", "ftime", "lead", "tmp"] + (["n_x"] if has_nx else [])
    if df is None or df.empty:
        return empty_runs(model)
    d = df.copy()
    d["runtime"] = pd.to_datetime(d["runtime"], utc=True)
    d["ftime"] = pd.to_datetime(d["ftime"], utc=True)
    d = d.drop_duplicates(["runtime", "ftime"])
    d["tmp"] = pd.to_numeric(d["tmp"], errors="coerce")
    if has_nx:
        nx = pd.to_numeric(d["n_x"], errors="coerce") if "n_x" in d else pd.Series(np.nan, index=d.index)
        if "txn" in d:
            nx = pd.to_numeric(d["txn"], errors="coerce").fillna(nx)
        d["n_x"] = nx
    d["lead"] = ((d["ftime"] - d["runtime"]) / HOUR).round().astype(int)
    cycles = LAV_CYCLES if model == "LAV" else NBS_CYCLES if model == "NBS" else None
    if cycles is not None:
        d = d[d["runtime"].dt.hour.isin(cycles) & (d["runtime"].dt.minute == 0)]
    spec = MODEL_SPEC[model]
    g = d.groupby("runtime")["lead"].agg(["min", "size"])
    keep = g.index[g["min"].isin(spec["first_leads"]) & (g["size"] >= spec["min_rows"])]
    d = d[d["runtime"].isin(keep)]
    return d[cols].sort_values(["runtime", "ftime"]).reset_index(drop=True)


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


# ── Run matrices and selection ──────────────────────────────────────────────
class RunMatrix:
    """Runs as rows; each row holds that run's leads (ascending) and tmp
    values, NaN-padded. Lead sets may differ between runs (NBS cycles)."""

    def __init__(self, runs: pd.DataFrame):
        if runs.empty:
            self.runtimes = pd.DatetimeIndex([], tz="UTC")
            self.lead = self.tmp = np.empty((0, 1))
            return
        r = runs.sort_values(["runtime", "lead"])
        self.runtimes = pd.DatetimeIndex(r["runtime"].drop_duplicates())
        row = self.runtimes.get_indexer(r["runtime"])
        col = r.groupby("runtime").cumcount().to_numpy()
        K = int(col.max()) + 1
        self.lead = np.full((len(self.runtimes), K), np.nan)
        self.tmp = np.full((len(self.runtimes), K), np.nan)
        self.lead[row, col] = r["lead"].to_numpy(dtype=float)
        self.tmp[row, col] = r["tmp"].to_numpy(dtype=float)

    def __len__(self) -> int:
        return len(self.runtimes)

    def runtime_ns(self, idx: np.ndarray) -> np.ndarray:
        if len(self) == 0:
            return np.zeros(len(idx), dtype=np.int64)
        return np.where(idx >= 0, self.runtimes.asi8[np.maximum(idx, 0)], 0)

    def first_lead(self, idx: np.ndarray) -> np.ndarray:
        if len(self) == 0:
            return np.full(len(idx), np.nan)
        return np.where(idx >= 0, self.lead[np.maximum(idx, 0), 0], np.nan)

    def interp(self, idx: np.ndarray, x: np.ndarray) -> np.ndarray:
        """tmp of run idx linearly interpolated at lead x (NaN outside)."""
        out = np.full(len(idx), np.nan)
        ok = (idx >= 0) & ~np.isnan(x)
        if not ok.any() or len(self) == 0:
            return out
        L, V, xs = self.lead[idx[ok]], self.tmp[idx[ok]], x[ok]
        a = np.arange(len(xs))
        nvalid = (~np.isnan(L)).sum(axis=1)
        with np.errstate(invalid="ignore"):
            j = (L <= xs[:, None]).sum(axis=1) - 1
        last = L[a, np.maximum(nvalid - 1, 0)]
        inside = (nvalid >= 2) & (j >= 0) & (xs <= last)
        j = np.clip(j, 0, np.maximum(nvalid - 2, 0))
        x0, x1 = L[a, j], L[a, np.minimum(j + 1, L.shape[1] - 1)]
        y0, y1 = V[a, j], V[a, np.minimum(j + 1, L.shape[1] - 1)]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = (xs - x0) / (x1 - x0)
            res = y0 + w * (y1 - y0)
        out[ok] = np.where(inside, res, np.nan)
        return out

    def window_max(self, idx: np.ndarray, lo_excl: np.ndarray, lo_incl: np.ndarray,
                   hi_excl: np.ndarray) -> np.ndarray:
        """max tmp over leads L with L > lo_excl, L >= lo_incl, L < hi_excl."""
        out = np.full(len(idx), np.nan)
        ok = idx >= 0
        if not ok.any() or len(self) == 0:
            return out
        L, V = self.lead[idx[ok]], self.tmp[idx[ok]]
        with np.errstate(invalid="ignore"):
            m = (L > lo_excl[ok, None]) & (L >= lo_incl[ok, None]) & (L < hi_excl[ok, None])
        vals = np.where(m, V, np.nan)
        has = ~np.isnan(vals).all(axis=1)
        res = np.full(vals.shape[0], np.nan)
        res[has] = np.nanmax(vals[has], axis=1)
        out[ok] = res
        return out


_NEVER = np.iinfo(np.int64).min + 1


def _hours(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) in hours for int64 ns arrays."""
    return (a - b) / 3.6e12


def _select_run(cutoff: np.ndarray, runtimes: pd.DatetimeIndex, lag: pd.Timedelta) -> np.ndarray:
    """Index of the latest run with runtime + lag <= cutoff (-1 if none)."""
    if len(runtimes) == 0:
        return np.full(len(cutoff), -1)
    return np.searchsorted((runtimes + lag).asi8, cutoff, side="right") - 1


def _cutoff(valid: np.ndarray, skip: np.ndarray, runtimes: pd.DatetimeIndex,
            lag: pd.Timedelta) -> np.ndarray:
    """Selection cutoff per row: valid_t, or (late-bulletin masking) the
    availability time of the run before the newest eligible one."""
    cutoff = valid.copy()
    if not skip.any():
        return cutoff
    newest = _select_run(valid, runtimes, lag)
    back = skip & (newest >= 1)
    if back.any():
        cutoff[back] = (runtimes[newest[back] - 1] + lag).asi8
    cutoff[skip & (newest < 1)] = _NEVER
    return cutoff


def _runtime_col(rt_ns: np.ndarray, ok: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(rt_ns, utc=True)).where(ok)


# ── Feature blocks ──────────────────────────────────────────────────────────
def _mos_block(prefix: str, runs: pd.DataFrame, lag: pd.Timedelta, tmp_max_age: pd.Timedelta,
               valid_ts: pd.DatetimeIndex, tmpf: np.ndarray, skip: np.ndarray,
               w0: np.ndarray, w1: np.ndarray) -> dict:
    """GFS-MOS-style features (3-hourly tmp + daytime max n_x) for one model."""
    n = len(valid_ts)
    valid = valid_ts.asi8
    rm = RunMatrix(runs)
    cutoff = _cutoff(valid, skip, rm.runtimes, lag)
    ridx = _select_run(cutoff, rm.runtimes, lag)
    rt_ns = rm.runtime_ns(ridx)
    age = np.where(ridx >= 0, _hours(valid, rt_ns), np.nan)
    ridx_t = np.where(age <= tmp_max_age / HOUR, ridx, -1)
    lead_valid = np.where(ridx_t >= 0, _hours(valid, rt_ns), np.nan)
    f = {}
    f[f"{prefix}_tmp_max_rest"] = rm.window_max(ridx_t, lead_valid, _hours(w0, rt_ns), _hours(w1, rt_ns))
    f[f"{prefix}_tmp_age_h"] = np.where(ridx_t >= 0, age, np.nan)
    f[f"{prefix}_tmp_t3"] = rm.interp(ridx_t, lead_valid + 3)
    f[f"{prefix}_tmp_t6"] = rm.interp(ridx_t, lead_valid + 6)
    f[f"{prefix}_tmp_runtime"] = _runtime_col(rt_ns, ridx_t >= 0)

    # now_err: latest eligible run with rows on both sides of valid (step back
    # while that run's first lead is still in the future).
    r_now = ridx.copy()
    for _ in range(3):
        step = (r_now >= 0) & (_hours(valid, rm.runtime_ns(r_now)) < rm.first_lead(r_now))
        if not step.any():
            break
        r_now = np.where(step, r_now - 1, r_now)
    rt_now = rm.runtime_ns(r_now)
    age_now = np.where(r_now >= 0, _hours(valid, rt_now), np.nan)
    r_now = np.where(age_now <= tmp_max_age / HOUR, r_now, -1)
    f[f"{prefix}_tmp_now_err"] = tmpf - rm.interp(r_now, _hours(valid, rt_now))

    # n_x for D: the value at ftime (D+1) 00Z from the latest eligible run carrying it.
    if runs.empty:
        runs = empty_runs("GFS")
    nx = runs[runs["n_x"].notna() & (runs["ftime"].dt.hour == 0)][["runtime", "ftime", "n_x"]].copy()
    nx["tdate"] = (nx["ftime"] - pd.Timedelta(days=1)).dt.date
    nx = nx.sort_values(["tdate", "runtime"])
    nx["prev_n_x"] = nx.groupby("tdate")["n_x"].shift(1)
    nx["avail"] = nx["runtime"] + lag
    left = pd.DataFrame({"pos": np.arange(n), "cutoff": pd.to_datetime(cutoff, utc=True),
                         "tdate": target_date(pd.Series(valid_ts)).to_numpy()}).sort_values("cutoff")
    m = pd.merge_asof(left, nx.sort_values("avail")[["avail", "tdate", "runtime", "n_x", "prev_n_x"]],
                      left_on="cutoff", right_on="avail", by="tdate", direction="backward").sort_values("pos")
    nx_age = ((valid_ts - pd.DatetimeIndex(m["runtime"])) / HOUR).to_numpy(dtype=float)
    ok = ~np.isnan(nx_age) & (nx_age <= NX_MAX_AGE / HOUR)
    f[f"{prefix}_nx_today"] = np.where(ok, m["n_x"].to_numpy(dtype=float), np.nan)
    f[f"{prefix}_nx_age_h"] = np.where(ok, nx_age, np.nan)
    f[f"{prefix}_nx_rev"] = np.where(ok, (m["n_x"] - m["prev_n_x"]).to_numpy(dtype=float), np.nan)
    f[f"{prefix}_nx_runtime"] = pd.DatetimeIndex(m["runtime"]).where(ok)
    return f


def _lav_block(runs: pd.DataFrame, valid_ts: pd.DatetimeIndex, tmpf: np.ndarray,
               skip: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> dict:
    valid = valid_ts.asi8
    rm = RunMatrix(runs)
    cutoff = _cutoff(valid, skip, rm.runtimes, LAV_AVAIL_LAG)
    lidx = _select_run(cutoff, rm.runtimes, LAV_AVAIL_LAG)
    rt_ns = rm.runtime_ns(lidx)
    age = np.where(lidx >= 0, _hours(valid, rt_ns), np.nan)
    lidx = np.where(age <= LAV_MAX_AGE / HOUR, lidx, -1)
    lead_valid = np.where(lidx >= 0, _hours(valid, rt_ns), np.nan)
    ceil_lead = np.ceil(lead_valid - 1e-9)          # ftime = ceil_hour(valid_t)
    return {"lav_tmp_max_rest": rm.window_max(lidx, lead_valid, _hours(w0, rt_ns), _hours(w1, rt_ns)),
            "lav_tmp_now_err": tmpf - rm.interp(lidx, ceil_lead),
            "lav_tmp_t3": rm.interp(lidx, ceil_lead + 3),
            "lav_tmp_t6": rm.interp(lidx, ceil_lead + 6),
            "lav_age_h": np.where(lidx >= 0, age, np.nan),
            "lav_runtime": _runtime_col(rt_ns, lidx >= 0)}


def guidance_features(rows: pd.DataFrame, gfs: pd.DataFrame, lav: pd.DataFrame,
                      nbs: pd.DataFrame | None = None,
                      *, skip_newest: bool | np.ndarray = False) -> pd.DataFrame:
    """All guidance columns (+ the runtimes used) for each row.

    rows: needs `valid` (UTC) and `tmpf`. gfs / lav / nbs: output of
    clean_runs (nbs=None -> NBS columns NaN). skip_newest: True (or a boolean
    mask per row) recomputes as if each model's newest eligible run had not
    arrived yet (late-bulletin masking).
    """
    n = len(rows)
    valid_ts = pd.DatetimeIndex(rows["valid"]).tz_convert("UTC")
    tmpf = rows["tmpf"].to_numpy(dtype=float)
    w0 = pd.DatetimeIndex(climate_window_start(pd.Series(valid_ts))).asi8
    w1 = w0 + 24 * 3_600_000_000_000
    skip = np.broadcast_to(np.asarray(skip_newest, dtype=bool), (n,))
    cols = {}
    cols.update(_mos_block("gfs", gfs, GFS_AVAIL_LAG, GFS_TMP_MAX_AGE, valid_ts, tmpf, skip, w0, w1))
    cols.update(_lav_block(lav, valid_ts, tmpf, skip, w0, w1))
    cols.update(_mos_block("nbs", nbs if nbs is not None else empty_runs("NBS"), NBS_AVAIL_LAG,
                           NBS_TMP_MAX_AGE, valid_ts, tmpf, skip, w0, w1))
    out = pd.DataFrame(index=rows.index)
    for c in GUIDANCE_COLS + RUNTIME_COLS:
        out[c] = cols[c]
    return out
