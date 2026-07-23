"""
KNYC Training Data Builder
==========================
Rebuilds the training CSV from raw IEM ASOS sources with the fixes called out
in the code review:

  - Raw, unscaled feature units (mslp in mb, sknt in knots, relh in 0-100).
  - target_high derived from NWS Daily Climate Report (CLI) for KNYC, keyed on
    NY-local calendar date — the actual settlement value.
  - NY-local timezone-aware hour/month/dayofyear features so the model and the
    bot agree on what "noon" means.
  - Upstream station features (KEWR, KLGA, KJFK, KTEB, KPHL, KBWI, KDCA, KBOS)
    so the model has a warm-advection signal.
  - Wind direction decomposed into u/v components.
  - Cloud cover (oktas) and ceiling.
  - Hourly precipitation.
  - Solar zenith angle (proper insolation proxy).
  - Climatology baseline per (month, hour).

Run once to produce knyc_training.csv. Re-run periodically to ingest new obs.

    python build_training_data.py --start 2010-01-01 --end 2026-02-01

Outputs:
    knyc_training.csv
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

NY_TZ = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent

PRIMARY = "KNYC"
UPSTREAM = ["KEWR", "KLGA", "KJFK", "KTEB", "KPHL", "KBWI", "KDCA", "KBOS"]

# Lat/lon for solar zenith (KNYC = Central Park)
KNYC_LAT, KNYC_LON = 40.7794, -73.9692

SKY_TO_OKTAS = {"CLR": 0, "SKC": 0, "NSC": 0, "NCD": 0,
                "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8}

IEM_VARS = "tmpf,dwpf,relh,drct,sknt,mslp,p01i,skyc1,skyc2,skyc3,skyl1,feel,metar"


# ── IEM fetcher ─────────────────────────────────────────────────────────────
def fetch_iem_asos(station: str, start: datetime, end: datetime,
                   primary: bool = True) -> pd.DataFrame:
    """
    Fetch ASOS observations for one station between [start, end] UTC.

    primary=True keeps strict :51 obs (KNYC's cadence; needed for
    consistent hourly lag/roll features).
    primary=False keeps one obs per hour for stations that report at
    different minutes (BOS at :54, DCA at :52, etc.), and normalizes
    the timestamp to :51 so an outer merge against KNYC aligns.
    """
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={station}&data={IEM_VARS}"
        f"&year1={start.year}&month1={start.month:02d}&day1={start.day:02d}&hour1={start.hour:02d}"
        f"&year2={end.year}&month2={end.month:02d}&day2={end.day:02d}&hour2={end.hour:02d}"
        "&tz=UTC&format=comma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3"
    )
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            text = "\n".join(l for l in resp.text.splitlines()
                             if not l.startswith("#") and l.strip())
            df = pd.read_csv(StringIO(text))
            break
        except (requests.RequestException, pd.errors.EmptyDataError) as exc:
            if attempt == 3:
                raise RuntimeError(f"IEM fetch failed for {station}: {exc}")
            time.sleep(5 * (attempt + 1))

    df.columns = df.columns.str.strip()
    df["valid"] = pd.to_datetime(df["valid"]).dt.tz_localize("UTC")

    if primary:
        df = df[df["valid"].dt.minute == 51].copy()
    else:
        # Keep obs in the :45–:59 window and normalize to the same :51
        # timestamp as KNYC so merge('valid') works.
        df = df[df["valid"].dt.minute.between(45, 59)].copy()
        df["valid"] = df["valid"].dt.floor("1h") + pd.Timedelta(minutes=51)
        df = df.drop_duplicates("valid", keep="last")

    for col in ["tmpf", "dwpf", "relh", "drct", "sknt", "mslp", "p01i", "skyl1", "feel"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physical sanity bounds — protect against sensor glitches.
    bounds = {"tmpf": (-30, 115), "dwpf": (-40, 90), "relh": (0, 100),
              "drct": (0, 360), "sknt": (0, 90), "mslp": (940, 1060),
              "p01i": (0, 10), "feel": (-60, 130)}
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    return df.sort_values("valid").reset_index(drop=True)


def fetch_in_chunks(station: str, start: datetime, end: datetime,
                    chunk_days: int = 365, primary: bool = True) -> pd.DataFrame:
    """Chunk long IEM requests to avoid timeouts."""
    parts = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        print(f"  [{station}] {cur.date()} -> {nxt.date()}")
        parts.append(fetch_iem_asos(station, cur, nxt, primary=primary))
        cur = nxt
    return pd.concat(parts, ignore_index=True).drop_duplicates("valid")


# ── NWS CLI fetcher ─────────────────────────────────────────────────────────
def _fetch_cli_window(start: datetime, end: datetime) -> list[dict]:
    """One CLINYC request over [start, end]; returns raw {local_date, cli_max}.

    Two IEM AFOS gotchas handled here:
      * retrieve.py defaults to limit=1 (newest bulletin only); an explicit
        limit bounded by sdate/edate is required to get the whole range.
      * The day-D final report is transmitted the morning of D+1, so its
        issuance-header date is a day ahead of the data — key off the
        "CLIMATE SUMMARY FOR <date>" line, not the header, or highs land on
        the wrong date.
    """
    ndays = (end - start).days + 4
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
        "?pil=CLINYC"
        f"&sdate={start.strftime('%Y-%m-%d')}"
        f"&edate={end.strftime('%Y-%m-%d')}"
        f"&limit={max(200, ndays * 5)}"
        "&fmt=text"
    )
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    # Each bulletin contains a line like:
    #   "MAXIMUM         72   329 PM   71    78 2001   58"
    # The first integer after MAXIMUM is the daily high.
    bulletins = re.split(r"\x01|000\s*\n", resp.text)
    records = []
    for b in bulletins:
        date_m = re.search(r"CLIMATE SUMMARY FOR\s+(\w+\s+\d{1,2}\s+\d{4})", b)
        max_m = re.search(r"MAXIMUM\s+(-?\d{1,3})\s", b)
        if not (date_m and max_m):
            continue
        try:
            d = pd.to_datetime(date_m.group(1)).date()
        except (ValueError, TypeError):
            continue
        records.append({"local_date": d, "cli_max": float(max_m.group(1))})
    return records


def fetch_cli_max_temps(start: datetime, end: datetime,
                        chunk_days: int = 365) -> pd.DataFrame:
    """
    Daily-max from the NWS Climate Daily Report (CLINYC). Returns a DataFrame
    with 'local_date' and 'cli_max' (°F), one row per date.

    Chunked yearly to avoid the multi-MB single response (and server-side
    timeout) that a decade-wide request would produce — a timeout here would
    silently drop CLI entirely and fall back to the noisy METAR 6-hr max for
    every day. Multiple issuances per date (preliminary + final) are collapsed
    to the max so a partial-day preliminary can't under-report a later high.
    """
    records: list[dict] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        print(f"  [CLINYC] {cur.date()} -> {nxt.date()}")
        records.extend(_fetch_cli_window(cur, nxt))
        cur = nxt

    if not records:
        print("  WARNING: no CLI records parsed -- falling back to METAR 6h-max")
        return pd.DataFrame(columns=["local_date", "cli_max"])

    cli = (pd.DataFrame(records)
             .groupby("local_date", as_index=False)["cli_max"].max()
             .sort_values("local_date"))
    return cli


def parse_6hr_max_from_metar(metar: str) -> float:
    """Extract the 1snTTT 6-hour max group from METAR remarks (°F)."""
    if not isinstance(metar, str) or " RMK" not in metar:
        return np.nan
    rmk = metar.split(" RMK", 1)[1]
    m = re.search(r"(?:^| )1([01]\d{3})(?:$| )", rmk)
    if m is None:
        return np.nan
    raw = m.group(1)
    sign = -1 if raw[0] == "1" else 1
    temp_c = sign * int(raw[1:]) / 10.0
    if not (-50 <= temp_c <= 55):
        return np.nan
    return temp_c * 9 / 5 + 32


# ── Cloud / METAR parsing ──────────────────────────────────────────────────
def cloud_oktas_from_row(row) -> float:
    vals = []
    for c in ("skyc1", "skyc2", "skyc3"):
        v = row.get(c)
        if isinstance(v, str):
            vals.append(SKY_TO_OKTAS.get(v.strip().upper(), np.nan))
    if not vals:
        return np.nan
    arr = np.array(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmax(arr))


# ── Solar geometry ─────────────────────────────────────────────────────────
def solar_zenith_deg(valid_utc: pd.Series, lat: float, lon: float) -> np.ndarray:
    """
    NOAA solar position algorithm — accurate to ~0.1° and dependency-free.
    valid_utc must be timezone-aware UTC.
    """
    t = valid_utc.dt.tz_convert("UTC")
    # Julian day
    jd = t.astype("int64") / 86_400_000_000_000 + 2440587.5
    n = jd - 2451545.0
    # Mean longitude and anomaly (degrees)
    L = (280.460 + 0.9856474 * n) % 360
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    # Ecliptic longitude
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    # Obliquity
    eps = np.radians(23.439 - 0.0000004 * n)
    # Right ascension / declination
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    # Equation of time (minutes)
    eqt = 4 * np.degrees(
        np.radians(L) - 0.0057183 - np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    )
    # Solar time (hours)
    utc_hours = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    solar_time = (utc_hours + lon / 15 + eqt / 60) % 24
    ha = np.radians(15 * (solar_time - 12))
    lat_r = math.radians(lat)
    cos_z = (np.sin(lat_r) * np.sin(dec) + np.cos(lat_r) * np.cos(dec) * np.cos(ha))
    cos_z = np.clip(cos_z, -1, 1)
    return np.degrees(np.arccos(cos_z))


# ── Master feature engineering ─────────────────────────────────────────────
def build_features(knyc: pd.DataFrame, upstream: dict[str, pd.DataFrame],
                   cli: pd.DataFrame) -> pd.DataFrame:
    df = knyc.copy().rename(columns={c: c for c in knyc.columns})

    # NY-local time features.
    df["valid_local"] = df["valid"].dt.tz_convert(NY_TZ)
    df["local_date"] = df["valid_local"].dt.date
    df["hour"] = df["valid_local"].dt.hour
    df["month"] = df["valid_local"].dt.month
    df["dayofyear"] = df["valid_local"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Derived met. `feel` is wind-chill / heat-index from IEM; missing in
    # the temperate band where it ≈ tmpf, so fall back to tmpf.
    df["feel"] = df["feel"].fillna(df["tmpf"])
    df["feel_gap"] = df["tmpf"] - df["feel"]
    df["dew_depression"] = df["tmpf"] - df["dwpf"]
    df["dew_dep_tend_3h"] = df["dew_depression"].diff(3)

    # Wind components (meteorological convention: drct is direction FROM).
    df["sknt"] = df["sknt"].fillna(0)
    drct_rad = np.radians(df["drct"])
    df["u_wind"] = -df["sknt"] * np.sin(drct_rad)   # +u = from west
    df["v_wind"] = -df["sknt"] * np.cos(drct_rad)   # +v = from south
    df["u_wind"] = df["u_wind"].fillna(0)
    df["v_wind"] = df["v_wind"].fillna(0)

    # Cloud cover.
    df["cloud_oktas"] = df.apply(cloud_oktas_from_row, axis=1)
    df["cloud_oktas"] = df["cloud_oktas"].ffill(limit=3).fillna(4)
    df["cloud_tend_3h"] = df["cloud_oktas"].diff(3)
    df["ceiling_ft"] = df["skyl1"].fillna(25000)  # unlimited proxy

    # Precip.
    df["p01i"] = df["p01i"].fillna(0)
    df["precip_24h"] = df["p01i"].rolling(24, min_periods=1).sum()
    df["is_precip"] = (df["p01i"] > 0.005).astype(int)

    # Solar geometry.
    zen = solar_zenith_deg(df["valid"], KNYC_LAT, KNYC_LON)
    df["solar_zenith"] = zen
    df["solar_alt"] = (90 - zen).clip(lower=0)
    df["cos_zenith"] = np.cos(np.radians(zen)).clip(lower=0)

    # MSLP has occasional 1–2h gaps in older ASOS data; forward-fill briefly
    # so the tendency/rolling features don't drop every row downstream.
    df["mslp"] = df["mslp"].ffill(limit=3)

    # Pressure & temp tendencies / lags / rolls.
    for c in ("mslp", "tmpf"):
        for h in (1, 3, 6):
            df[f"{c}_tend_{h}h"] = df[c].diff(h)
    for lag in (1, 2, 3, 6, 12, 24):
        df[f"tmpf_lag_{lag}h"] = df["tmpf"].shift(lag)
        df[f"dwpf_lag_{lag}h"] = df["dwpf"].shift(lag)
    df["tmpf_roll3_mean"] = df["tmpf"].shift(1).rolling(3).mean()
    df["tmpf_roll6_mean"] = df["tmpf"].shift(1).rolling(6).mean()
    df["tmpf_roll24_mean"] = df["tmpf"].shift(1).rolling(24).mean()
    df["tmpf_roll3_std"] = df["tmpf"].shift(1).rolling(3).std()
    df["mslp_roll3_mean"] = df["mslp"].shift(1).rolling(3).mean()

    # Upstream-station features — merged on hour (rounded to :51 timestamp).
    for code, udf in upstream.items():
        if udf.empty:
            continue
        u = udf[["valid", "tmpf", "dwpf", "sknt", "drct"]].rename(
            columns={"tmpf": f"{code}_tmpf", "dwpf": f"{code}_dwpf",
                     "sknt": f"{code}_sknt", "drct": f"{code}_drct"})
        df = df.merge(u, on="valid", how="left")
        df[f"{code}_tmpf_delta"] = df[f"{code}_tmpf"] - df["tmpf"]
        df[f"{code}_tmpf_tend_3h"] = df[f"{code}_tmpf"].diff(3)

    # Climatology baseline (computed on training portion only — see notebook).
    # Filled in build_features so it's available everywhere; the notebook
    # recomputes it strictly on train rows before splitting.
    clim = df.groupby(["month", "hour"])["tmpf"].mean().rename("tmpf_clim_global")
    df = df.merge(clim, on=["month", "hour"], how="left")
    df["tmpf_anomaly"] = df["tmpf"] - df["tmpf_clim_global"]

    # ── Targets ────────────────────────────────────────────────────────────
    # t+3h, t+6h: shift tmpf forward.
    df["target_t3h"] = df["tmpf"].shift(-3)
    df["target_t6h"] = df["tmpf"].shift(-6)

    # Daily high: prefer NWS CLI; fall back to max of (hourly tmpf, METAR 6h max).
    # The 6-hr max group (00/06/12/18Z) covers the trailing 6 hours, which for
    # early-morning obs straddles local midnight — its max can belong to the
    # previous day, so only trust it when that window is fully inside the
    # same local calendar day (see the identical guard in knyc_discord_bot.py).
    df["mxtmpf_6hr"] = df["metar"].apply(parse_6hr_max_from_metar) if "metar" in df.columns else np.nan
    window_start_local = df["valid_local"] - pd.Timedelta(hours=6)
    df["mxtmpf_6hr_sameday"] = df["mxtmpf_6hr"].where(window_start_local.dt.date == df["local_date"])
    fallback = (
        df.groupby("local_date")
          .agg(hourly_max=("tmpf", "max"), six_hr_max=("mxtmpf_6hr_sameday", "max"))
          .max(axis=1)
          .rename("fallback_max")
    )

    if not cli.empty:
        cli = cli.copy()
        cli["local_date"] = pd.to_datetime(cli["local_date"]).dt.date
        daily = pd.merge(fallback.reset_index(), cli, on="local_date", how="left")
        daily["target_high"] = daily["cli_max"].fillna(daily["fallback_max"])
    else:
        daily = fallback.reset_index().rename(columns={"fallback_max": "target_high"})

    df = df.merge(daily[["local_date", "target_high"]], on="local_date", how="left")

    return df


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=str(HERE / "knyc_training.csv"))
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    print(f"Fetching {PRIMARY}...")
    knyc = fetch_in_chunks(PRIMARY, start, end)
    print(f"  -> {len(knyc):,} rows")

    upstream = {}
    for s in UPSTREAM:
        print(f"Fetching {s}...")
        try:
            upstream[s] = fetch_in_chunks(s, start, end, primary=False)
            print(f"  -> {len(upstream[s]):,} rows")
        except Exception as exc:
            print(f"  WARNING: {s} failed: {exc}")
            upstream[s] = pd.DataFrame()

    print("Fetching CLINYC daily climate reports...")
    try:
        cli = fetch_cli_max_temps(start, end)
        print(f"  -> {len(cli):,} CLI daily records")
    except Exception as exc:
        print(f"  WARNING: CLI fetch failed ({exc}); will fall back to obs max")
        cli = pd.DataFrame()

    print("Engineering features...")
    df = build_features(knyc, upstream, cli)
    print(f"  -> {df.shape[0]:,} rows x {df.shape[1]} cols")

    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
