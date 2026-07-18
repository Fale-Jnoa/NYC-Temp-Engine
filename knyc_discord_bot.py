"""
KNYC Live Nowcaster — Discord Bot (v2)
======================================
Hourly KNYC temperature predictions posted to a #predictions channel.

Setup
-----
1. pip install "discord.py>=2.0" python-dotenv requests pandas numpy xgboost joblib
2. .env in the same folder:
       DISCORD_TOKEN=...
       GUILD_ID=...
3. Train models with build_training_data.py + KNYC_Nowcaster.ipynb. They
   produce:
       knyc_model_daily_high.pkl
       knyc_model_t3h.pkl
       knyc_model_t6h.pkl
       feature_manifest.json
4. python knyc_discord_bot.py

Behavior
--------
- Posts one nowcast on startup, then every hour at xx:55 UTC (KNYC's METAR
  drops ~:51-:53 via aviationweather.gov's live feed). If the fresh ob hasn't
  landed yet, retries every 60s until it does (capped at 50 min so a dead
  feed can't stall past the next hourly tick).
- Each post: current temp, dewpoint, RH, wind, t+3h, t+6h, model daily high,
  observed high so far, floor-locked reassessed high.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import discord
from discord.ext import tasks

NY_TZ = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "nowcast_log.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("knyc-nowcaster")

# ── Credentials ────────────────────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID") or 0)
CHANNEL_NAME = "predictions"

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set in .env")
if not GUILD_ID:
    raise RuntimeError("GUILD_ID not set in .env")

# ── Models & feature manifest ──────────────────────────────────────────────
log.info("Loading models...")
model_high = joblib.load(HERE / "knyc_model_daily_high.pkl")
model_t3h = joblib.load(HERE / "knyc_model_t3h.pkl")
model_t6h = joblib.load(HERE / "knyc_model_t6h.pkl")

manifest_path = HERE / "feature_manifest.json"
if not manifest_path.exists():
    raise RuntimeError(
        "feature_manifest.json missing — re-run KNYC_Nowcaster.ipynb to regenerate."
    )
manifest = json.loads(manifest_path.read_text())
FEATURE_COLS: list[str] = manifest["feature_cols"]
UPSTREAM: list[str] = manifest["upstream_stations"]

for name, m in [("high", model_high), ("t3h", model_t3h), ("t6h", model_t6h)]:
    n_expected = getattr(m, "n_features_in_", None)
    if n_expected is not None and n_expected != len(FEATURE_COLS):
        raise RuntimeError(
            f"Model {name} expects {n_expected} features but manifest has {len(FEATURE_COLS)}"
        )
log.info("✅ Models + manifest loaded (%d features)", len(FEATURE_COLS))

# ── Constants ──────────────────────────────────────────────────────────────
KNYC_LAT, KNYC_LON = 40.7794, -73.9692
PRIMARY = "KNYC"
SKY_TO_OKTAS = {"CLR": 0, "SKC": 0, "NSC": 0, "NCD": 0,
                "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8}
SANITY = {"tmpf": (-30, 115), "dwpf": (-40, 90), "relh": (0, 100),
          "drct": (0, 360), "sknt": (0, 90), "mslp": (940, 1060),
          "p01i": (0, 10), "feel": (-60, 130)}
STALENESS_LIMIT = timedelta(minutes=90)
AVWX_URL = "https://aviationweather.gov/api/data/metar"


# ── aviationweather.gov fetch ───────────────────────────────────────────────
def fetch_avwx(station: str, hours_back: float) -> pd.DataFrame:
    """Live METAR feed (FAA/NWS text data server) — a given hour's ob is
    queryable within ~2 min of being issued, unlike IEM's archive CGI which
    re-ingests on its own, inconsistent schedule.

    Normalizes every station onto the :51 grid the models were trained on,
    since a routine METAR can post a minute or two off nominal.
    """
    params = {"ids": station, "format": "json", "hours": hours_back}
    for attempt in range(4):
        try:
            resp = requests.get(AVWX_URL, params=params, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 3:
                raise
            log.warning("aviationweather.gov fetch %s attempt %d failed: %s",
                        station, attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"aviationweather.gov fetch failed for {station} after retries")

    records = [r for r in resp.json() if r.get("metarType") == "METAR"]
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        clouds = r.get("clouds") or []
        wdir = r.get("wdir")
        temp_c, dewp_c = r.get("temp"), r.get("dewp")
        rows.append({
            "valid": datetime.fromtimestamp(r["obsTime"], tz=timezone.utc),
            "tmpf": temp_c * 9 / 5 + 32 if temp_c is not None else np.nan,
            "dwpf": dewp_c * 9 / 5 + 32 if dewp_c is not None else np.nan,
            "drct": wdir if isinstance(wdir, (int, float)) else np.nan,
            "sknt": r.get("wspd", np.nan),
            "mslp": r.get("slp", np.nan),
            "p01i": r.get("precip", np.nan),
            "skyc1": clouds[0]["cover"] if len(clouds) > 0 else np.nan,
            "skyc2": clouds[1]["cover"] if len(clouds) > 1 else np.nan,
            "skyc3": clouds[2]["cover"] if len(clouds) > 2 else np.nan,
            "skyl1": clouds[0]["base"] if len(clouds) > 0 else np.nan,
            "metar": r.get("rawOb"),
        })

    df = pd.DataFrame(rows)
    df["valid"] = df["valid"].dt.floor("1h") + pd.Timedelta(minutes=51)
    df = df.drop_duplicates("valid", keep="last")

    for col in ["tmpf", "dwpf", "drct", "sknt", "mslp", "p01i", "skyl1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, (lo, hi) in SANITY.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    return df.sort_values("valid").reset_index(drop=True)


# ── Parsing helpers ────────────────────────────────────────────────────────
def parse_6hr_max(metar: str) -> float:
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


def cloud_oktas(row) -> float:
    vals = []
    for c in ("skyc1", "skyc2", "skyc3"):
        v = row.get(c)
        if isinstance(v, str):
            vals.append(SKY_TO_OKTAS.get(v.strip().upper(), np.nan))
    if not vals:
        return np.nan
    arr = np.array(vals, dtype=float)
    return float(np.nanmax(arr)) if not np.all(np.isnan(arr)) else np.nan


def solar_zenith_deg(valid_utc: pd.Series) -> np.ndarray:
    t = valid_utc.dt.tz_convert("UTC")
    jd = t.astype("int64") / 86_400_000_000_000 + 2440587.5
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.0000004 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    eqt = 4 * np.degrees(np.radians(L) - 0.0057183
                         - np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)))
    utc_hours = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    solar_time = (utc_hours + KNYC_LON / 15 + eqt / 60) % 24
    ha = np.radians(15 * (solar_time - 12))
    lat_r = math.radians(KNYC_LAT)
    cos_z = (math.sin(lat_r) * np.sin(dec)
             + math.cos(lat_r) * np.cos(dec) * np.cos(ha))
    cos_z = np.clip(cos_z, -1, 1)
    return np.degrees(np.arccos(cos_z))


# ── Feature engineering — must match build_training_data.py exactly ────────
def engineer_features(raw: pd.DataFrame, upstream: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = raw.copy()

    df["valid_local"] = df["valid"].dt.tz_convert(NY_TZ)
    df["local_date"] = df["valid_local"].dt.date
    df["hour"] = df["valid_local"].dt.hour
    df["month"] = df["valid_local"].dt.month
    df["dayofyear"] = df["valid_local"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "feel" not in df.columns:
        df["feel"] = df["tmpf"]
    df["feel"] = df["feel"].fillna(df["tmpf"])
    df["feel_gap"] = df["tmpf"] - df["feel"]
    df["dew_depression"] = df["tmpf"] - df["dwpf"]
    df["dew_dep_tend_3h"] = df["dew_depression"].diff(3)

    df["drct"] = df.get("drct", pd.Series(np.nan, index=df.index))
    df["sknt"] = df["sknt"].fillna(0)
    drct_rad = np.radians(df["drct"])
    df["u_wind"] = (-df["sknt"] * np.sin(drct_rad)).fillna(0)
    df["v_wind"] = (-df["sknt"] * np.cos(drct_rad)).fillna(0)

    df["cloud_oktas"] = df.apply(cloud_oktas, axis=1)
    df["cloud_oktas"] = df["cloud_oktas"].ffill(limit=3).fillna(4)
    df["cloud_tend_3h"] = df["cloud_oktas"].diff(3)
    df["ceiling_ft"] = df.get("skyl1", pd.Series(np.nan, index=df.index)).fillna(25000)

    df["p01i"] = df.get("p01i", pd.Series(0.0, index=df.index)).fillna(0)
    df["precip_24h"] = df["p01i"].rolling(24, min_periods=1).sum()
    df["is_precip"] = (df["p01i"] > 0.005).astype(int)

    zen = solar_zenith_deg(df["valid"])
    df["solar_zenith"] = zen
    df["solar_alt"] = np.clip(90 - zen, 0, None)
    df["cos_zenith"] = np.clip(np.cos(np.radians(zen)), 0, None)

    df["mslp"] = df["mslp"].ffill()
    df["relh"] = df["relh"].astype(float)
    mask = df["relh"].isna()
    if mask.any():
        a, b = 17.625, 243.04
        T_C = (df.loc[mask, "tmpf"] - 32) * 5 / 9
        D_C = (df.loc[mask, "dwpf"] - 32) * 5 / 9
        df.loc[mask, "relh"] = (
            100 * np.exp(a * D_C / (b + D_C)) / np.exp(a * T_C / (b + T_C))
        )

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

    # 6-hour max temp from METAR for floor-lock logic.
    df["mxtmpf_6hr"] = (
        df["metar"].apply(parse_6hr_max) if "metar" in df.columns else np.nan
    )

    # Upstream-station features.
    for code in UPSTREAM:
        udf = upstream.get(code, pd.DataFrame())
        if udf.empty:
            df[f"{code}_tmpf"] = np.nan
            df[f"{code}_dwpf"] = np.nan
            df[f"{code}_sknt"] = np.nan
            df[f"{code}_drct"] = np.nan
        else:
            u = udf[["valid", "tmpf", "dwpf", "sknt", "drct"]].rename(
                columns={"tmpf": f"{code}_tmpf", "dwpf": f"{code}_dwpf",
                         "sknt": f"{code}_sknt", "drct": f"{code}_drct"})
            df = df.merge(u, on="valid", how="left")
        df[f"{code}_tmpf_delta"] = df[f"{code}_tmpf"] - df["tmpf"]
        df[f"{code}_tmpf_tend_3h"] = df[f"{code}_tmpf"].diff(3)

    # Climatology baseline column expected by the model. The training notebook
    # writes per-(month,hour) means; at inference we approximate with a
    # 30-day window of the most recent obs at the same hour. If that fails,
    # fall back to current tmpf (anomaly = 0).
    if "tmpf_clim_global" not in df.columns:
        df["tmpf_clim_global"] = df["tmpf"]
    df["tmpf_anomaly"] = df["tmpf"] - df["tmpf_clim_global"]

    return df


# ── Climatology (loaded from training data if available) ───────────────────
clim_path = HERE / "tmpf_climatology.csv"
if clim_path.exists():
    _clim = pd.read_csv(clim_path)
    CLIM_MAP = {(int(r.month), int(r.hour)): float(r.tmpf_clim) for r in _clim.itertuples()}
    log.info("Loaded climatology with %d (month,hour) bins", len(CLIM_MAP))
else:
    CLIM_MAP = {}
    log.warning("tmpf_climatology.csv missing — anomaly feature will be zero")


def _apply_climatology(df: pd.DataFrame) -> pd.DataFrame:
    if not CLIM_MAP:
        return df
    df["tmpf_clim_global"] = [
        CLIM_MAP.get((int(m), int(h)), float(t))
        for m, h, t in zip(df["month"], df["hour"], df["tmpf"])
    ]
    df["tmpf_anomaly"] = df["tmpf"] - df["tmpf_clim_global"]
    return df


# ── Nowcast ────────────────────────────────────────────────────────────────
def get_nowcast() -> dict:
    raw = fetch_avwx(PRIMARY, hours_back=72)
    if raw.empty:
        raise RuntimeError("aviationweather.gov returned no KNYC obs")

    upstream = {}
    for s in UPSTREAM:
        try:
            upstream[s] = fetch_avwx(s, hours_back=72)
        except Exception as exc:
            log.warning("Upstream %s fetch failed: %s — proceeding with NaN", s, exc)
            upstream[s] = pd.DataFrame()
        time.sleep(0.3)

    df = engineer_features(raw, upstream)
    df = _apply_climatology(df)

    ready = df.dropna(subset=FEATURE_COLS)
    if ready.empty:
        raise RuntimeError(
            "Insufficient recent obs to fill all features. "
            f"Latest row has missing: {df.iloc[-1][FEATURE_COLS].isna().sum()} cols."
        )

    latest = ready.iloc[[-1]]
    valid_t = latest["valid"].iloc[0]
    age = datetime.now(timezone.utc) - valid_t.to_pydatetime()
    if age > STALENESS_LIMIT:
        raise RuntimeError(
            f"Latest usable obs is {age} old (>{STALENESS_LIMIT}). Refusing to post stale forecast."
        )

    cur_temp = float(latest["tmpf"].iloc[0])
    cur_dwpf = float(latest["dwpf"].iloc[0])
    cur_relh = float(latest["relh"].iloc[0])
    cur_wind_kt = float(latest["sknt"].iloc[0])
    cur_drct = float(latest["drct"].iloc[0]) if not pd.isna(latest["drct"].iloc[0]) else None

    X = latest[FEATURE_COLS]
    pred_high = float(model_high.predict(X)[0])
    pred_t3h = float(model_t3h.predict(X)[0])
    pred_t6h = float(model_t6h.predict(X)[0])

    # Obs high floor — NY-local calendar day.
    today_local = datetime.now(NY_TZ).date()
    today_mask = df["local_date"] == today_local
    today_tmpf = df.loc[today_mask, "tmpf"].dropna()
    max_tmpf = float(today_tmpf.max()) if not today_tmpf.empty else None
    today_mxt = df.loc[today_mask, "mxtmpf_6hr"].dropna()
    max_mxt = float(today_mxt.max()) if not today_mxt.empty else None
    candidates = [v for v in (max_tmpf, max_mxt) if v is not None]
    obs_high = max(candidates) if candidates else None
    reassessed = max(obs_high if obs_high is not None else float("-inf"), pred_high)

    log.info(
        "valid=%s temp=%.1f t3h=%.1f t6h=%.1f high=%.1f obs_high=%s reassessed=%.1f age=%s",
        valid_t.isoformat(), cur_temp, pred_t3h, pred_t6h, pred_high,
        f"{obs_high:.1f}" if obs_high is not None else "-",
        reassessed, age,
    )

    # Persistent log row for postmortem analysis.
    try:
        row = {**X.iloc[0].to_dict(),
               "valid_t": valid_t.isoformat(),
               "pred_high": pred_high, "pred_t3h": pred_t3h, "pred_t6h": pred_t6h,
               "obs_high": obs_high, "reassessed": reassessed}
        pd.DataFrame([row]).to_csv(
            LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False
        )
    except Exception as exc:
        log.warning("nowcast log write failed: %s", exc)

    return {
        "valid_t": valid_t,
        "cur_temp": cur_temp,
        "cur_dwpf": cur_dwpf,
        "cur_relh": cur_relh,
        "cur_wind_kt": cur_wind_kt,
        "cur_drct": cur_drct,
        "pred_t3h": pred_t3h,
        "pred_t6h": pred_t6h,
        "pred_high": pred_high,
        "obs_high": obs_high,
        "reassessed": reassessed,
    }


# ── Discord embed ──────────────────────────────────────────────────────────
def _temp_color(t: float) -> int:
    if t < 32:  return 0x4169E1
    if t < 50:  return 0x00BFFF
    if t < 70:  return 0x00C851
    if t < 85:  return 0xFF8C00
    return 0xFF2400


def _wind_dir(deg: float | None) -> str:
    if deg is None or pd.isna(deg):
        return ""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) // 22.5) % 16]


def build_embed(data: dict) -> discord.Embed:
    color = _temp_color(data["reassessed"])
    obs_ts = data["valid_t"].astimezone(NY_TZ).strftime("%H:%M %Z")
    obs_str = f"{data['obs_high']:.1f}°F" if data["obs_high"] is not None else "—"
    obs_note = (
        "\n*(obs exceeded model)*"
        if data["obs_high"] is not None and data["obs_high"] > data["pred_high"]
        else ""
    )
    wind_str = (
        f"{_wind_dir(data['cur_drct'])} {data['cur_wind_kt']:.0f} kt"
        if data["cur_wind_kt"] is not None
        else "calm"
    )

    embed = discord.Embed(
        title="🌡️ KNYC Nowcaster Update",
        description=f"Latest METAR: **{obs_ts}** · Central Park, NYC",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Current Temp", value=f"**{data['cur_temp']:.1f}°F**", inline=True)
    embed.add_field(name="Dewpoint", value=f"{data['cur_dwpf']:.1f}°F", inline=True)
    embed.add_field(name="Rel. Humidity", value=f"{data['cur_relh']:.0f}%", inline=True)
    embed.add_field(name="Wind", value=wind_str, inline=True)
    embed.add_field(name="t+3h Forecast", value=f"{data['pred_t3h']:.1f}°F", inline=True)
    embed.add_field(name="t+6h Forecast", value=f"{data['pred_t6h']:.1f}°F", inline=True)
    embed.add_field(name="Model Daily High", value=f"{data['pred_high']:.1f}°F", inline=True)
    embed.add_field(name="Obs High So Far", value=obs_str, inline=True)
    embed.add_field(
        name="Reassessed High",
        value=f"**{data['reassessed']:.1f}°F**{obs_note}",
        inline=True,
    )
    embed.set_footer(text="XGBoost · KNYC ASOS + upstream · aviationweather.gov · NWS CLI")
    return embed


# ── Discord plumbing ───────────────────────────────────────────────────────
intents = discord.Intents.default()
bot = discord.Client(intents=intents)

last_posted_valid: datetime | None = None
FRESH_RETRY_DEADLINE = timedelta(minutes=50)
FRESH_RETRY_INTERVAL = timedelta(seconds=60)


async def post_nowcast_to_channel(*, wait_for_fresh: bool = False) -> None:
    """Fetch + post a nowcast embed.

    wait_for_fresh=True (the hourly :55 tick): if the newest available METAR
    is the same one already posted last hour, retry every 60s until a newer
    ob shows up — KNYC's METAR usually lands ~:51-:53 but can run late.
    Capped at FRESH_RETRY_DEADLINE so a dead feed can't stall past the next
    scheduled tick.
    """
    global last_posted_valid
    guild = bot.get_guild(GUILD_ID)
    if guild is None:
        log.error("Guild %s not found", GUILD_ID)
        return
    channel = discord.utils.get(guild.text_channels, name=CHANNEL_NAME)
    if channel is None:
        log.error("No #%s channel in %s", CHANNEL_NAME, guild.name)
        return

    deadline = datetime.now(timezone.utc) + FRESH_RETRY_DEADLINE if wait_for_fresh else None
    last_exc: Exception | None = None
    attempt = 0

    while True:
        attempt += 1
        try:
            data = get_nowcast()
            stale = (
                wait_for_fresh
                and last_posted_valid is not None
                and data["valid_t"] <= last_posted_valid
            )
            if not stale:
                embed = build_embed(data)
                await channel.send(embed=embed)
                last_posted_valid = data["valid_t"]
                log.info("Posted — reassessed high %.1f°F (valid %s)",
                          data["reassessed"], data["valid_t"].isoformat())
                return
            log.info("METAR still %s (no newer ob yet) — retrying in 60s",
                      data["valid_t"].isoformat())
            last_exc = None
        except Exception as exc:
            last_exc = exc
            log.warning("Nowcast attempt %d failed: %s", attempt, exc)

        if deadline is not None:
            if datetime.now(timezone.utc) >= deadline:
                await channel.send(
                    f"⚠️ No fresh METAR after {int(FRESH_RETRY_DEADLINE.total_seconds() // 60)} "
                    f"min of retrying: `{last_exc or 'still on previous obs'}`"
                )
                return
            await asyncio.sleep(FRESH_RETRY_INTERVAL.total_seconds())
        else:
            if attempt >= 3:
                await channel.send(f"⚠️ Nowcaster error after 3 attempts: `{last_exc}`")
                return
            await asyncio.sleep(120 * attempt)


@bot.event
async def on_ready() -> None:
    log.info("✅ Logged in as %s (ID %s)", bot.user, bot.user.id)
    await post_nowcast_to_channel()
    if not nowcast_loop.is_running():
        nowcast_loop.start()


@tasks.loop(hours=1)
async def nowcast_loop() -> None:
    await post_nowcast_to_channel(wait_for_fresh=True)


@nowcast_loop.before_loop
async def before_nowcast_loop() -> None:
    """Align to :55 UTC each hour — aviationweather.gov has the :51 METAR
    within ~2 min, so :55 gives a small buffer instead of IEM's old ~14 min
    ingest lag."""
    await bot.wait_until_ready()
    now = datetime.now(timezone.utc)
    next_run = now.replace(minute=55, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(hours=1)
    wait = (next_run - now).total_seconds()
    log.info("Hourly loop aligned — first tick at %s (%d min)",
             next_run.astimezone(NY_TZ).strftime("%H:%M %Z"), wait // 60)
    await asyncio.sleep(wait)


bot.run(DISCORD_TOKEN)
