"""
KNYC Nowcaster — Live Prediction Scorer
=======================================
Post-hoc accuracy scoring for the predictions the Discord bot logs to
`nowcast_log.csv`. Run it after a 1-2 week test window to grade all three
models against settled ground truth.

Ground truth sources
---------------------
- Hourly KNYC obs (IEM ASOS archive) — the authoritative settled record for
  t+3h / t+6h targets and the hourly running max.
- NWS Climate Daily Report (CLINYC) — official daily high **and the time it
  occurred**. Fetched across all intraday + evening issuances; the true
  calendar-day high is the max over issuances.
- ASOS Daily Summary Message (DSMNYC) — a second independent daily max. Some
  days the true high lands here (or in the CLI) and never shows up in the
  hourly :51 obs or the METAR 6-hour max groups; taking the max across every
  source captures those.

Scoring
-------
Daily high  (see score_daily_high for the full scheme):
  Only predictions issued BEFORE the high actually occurred count — once the
  high is in, the bot's obs-floor makes "reassessed high" trivially correct,
  so late predictions don't measure forecasting skill. Reported as MAE +
  hit-rates + a 0-100 skill score, broken out by issue-hour and by lead time.

t+3h / t+6h:
  Scored out of 24 per NY-local day — 1 point per hourly prediction landing
  within +/-1 F of the actual temp at the target time (valid_t + horizon).
  Also reports MAE, wider hit-rates, and per-hour accuracy.

All three models flag the hours at which they are most / least accurate.

Usage
-----
    python score_predictions.py                       # scores nowcast_log.csv
    python score_predictions.py --log other.csv
    python score_predictions.py --start 2026-07-01 --end 2026-07-14
    python score_predictions.py --out score_out       # write CSV breakdowns
"""
from __future__ import annotations

import argparse
import re
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
STATION = "KNYC"
HIT_TOL = 1.0  # +/-F for a t+3h / t+6h "hit"

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AFOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"

# Physically plausible KNYC bounds; reject decode/transcription garbage.
TEMP_MIN_F, TEMP_MAX_F = -20.0, 115.0


# ── HTTP with polite retry ──────────────────────────────────────────────────
def _get(url: str, *, params: dict | None = None, timeout: int = 120) -> requests.Response:
    for attempt in range(4):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 or "Too many requests" in resp.text[:80]:
            wait = 5 * (attempt + 1)
            print(f"  rate-limited -- sleeping {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"request failed after retries: {url}")


# ── Ground-truth: hourly obs ────────────────────────────────────────────────
def fetch_hourly_obs(start: datetime, end: datetime) -> pd.DataFrame:
    """KNYC :51 hourly obs (tmpf + raw METAR) from the IEM ASOS archive."""
    params = {
        "station": STATION, "data": "tmpf,metar",
        "year1": start.year, "month1": start.month, "day1": start.day, "hour1": 0,
        "year2": end.year, "month2": end.month, "day2": end.day, "hour2": 23,
        "tz": "UTC", "format": "comma", "latlon": "no", "elev": "no",
        "missing": "M", "trace": "T", "direct": "no", "report_type": "3",
    }
    resp = _get(IEM_URL, params=params)
    text = "\n".join(l for l in resp.text.splitlines()
                     if not l.startswith("#") and l.strip())
    if not text or "\n" not in text:
        return pd.DataFrame(columns=["valid", "tmpf", "metar"])

    df = pd.read_csv(StringIO(text))
    df.columns = df.columns.str.strip()
    df["valid"] = pd.to_datetime(df["valid"]).dt.tz_localize("UTC")
    df = df[df["valid"].dt.minute == 51].copy()
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df.loc[(df["tmpf"] < TEMP_MIN_F) | (df["tmpf"] > TEMP_MAX_F), "tmpf"] = np.nan
    return df.sort_values("valid").drop_duplicates("valid").reset_index(drop=True)


def parse_6hr_max(metar: str) -> float:
    """6-hour max group (1snTTT) from METAR remarks, in F."""
    if not isinstance(metar, str) or " RMK" not in metar:
        return np.nan
    rmk = metar.split(" RMK", 1)[1]
    m = re.search(r"(?:^| )1([01]\d{3})(?:$| )", rmk)
    if m is None:
        return np.nan
    raw = m.group(1)
    temp_c = (-1 if raw[0] == "1" else 1) * int(raw[1:]) / 10.0
    if not (-50 <= temp_c <= 55):
        return np.nan
    return temp_c * 9 / 5 + 32


# ── Ground-truth: NWS CLI (official daily high + time) ──────────────────────
def _parse_clock(hhmm: str, ampm: str) -> tuple[int, int] | None:
    """'212','PM' -> (14, 12).  Last two digits are minutes."""
    if len(hhmm) < 3:
        return None
    hour, minute = int(hhmm[:-2]), int(hhmm[-2:])
    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def fetch_cli(start: datetime, end: datetime) -> pd.DataFrame:
    """Per-date official high + high-time from CLINYC. Keeps the max across all
    issuances for each date (evening/next-day finals supersede intraday ones)."""
    # retrieve.py defaults to limit=1 (newest bulletin only); a generous limit
    # bounded by sdate/edate returns every issuance in the window.
    ndays = (end - start).days + 4
    params = {"pil": "CLINYC", "sdate": start.strftime("%Y-%m-%d"),
              "edate": (end + timedelta(days=2)).strftime("%Y-%m-%d"),
              "fmt": "text", "limit": str(max(50, ndays * 5))}
    text = _get(AFOS_URL, params=params).text
    bulletins = re.split(r"\x01|000\s*\n", text)

    best: dict[object, dict] = {}
    for b in bulletins:
        # Use the summary date ("...CLIMATE SUMMARY FOR JUNE 14 2026..."), NOT
        # the issuance-header date — a final report for day D is transmitted the
        # morning of D+1, so the header date is one day ahead of the data.
        dm = re.search(r"CLIMATE SUMMARY FOR\s+(\w+\s+\d{1,2}\s+\d{4})", b)
        mm = re.search(r"MAXIMUM\s+(-?\d{1,3})\s+(\d{2,4})\s+(AM|PM)", b)
        if not dm:
            continue
        try:
            d = pd.to_datetime(dm.group(1)).date()
        except (ValueError, TypeError):
            continue
        mx = re.search(r"MAXIMUM\s+(-?\d{1,3})", b)
        if not mx:
            continue
        val = float(mx.group(1))
        if not (TEMP_MIN_F <= val <= TEMP_MAX_F):
            continue
        htime = None
        if mm and float(mm.group(1)) == val:
            hm = _parse_clock(mm.group(2), mm.group(3))
            if hm:
                htime = datetime(d.year, d.month, d.day, hm[0], hm[1], tzinfo=NY_TZ)
        cur = best.get(d)
        if cur is None or val > cur["cli_max"]:
            best[d] = {"local_date": d, "cli_max": val, "cli_time": htime}

    return pd.DataFrame(best.values())


# ── Ground-truth: ASOS DSM (independent daily max) ──────────────────────────
def fetch_dsm(start: datetime, end: datetime) -> pd.DataFrame:
    """Per-date max temp + time from the ASOS Daily Summary Message (DSMNYC).

    Line form:  KNYC DS 1600 15/06 741557/ 650548// ...
                                    ^^ ^^^^  -> max 74 F at 15:57 LST
    """
    ndays = (end - start).days + 4
    params = {"pil": "DSMNYC", "sdate": start.strftime("%Y-%m-%d"),
              "edate": (end + timedelta(days=2)).strftime("%Y-%m-%d"),
              "fmt": "text", "limit": str(max(50, ndays * 5))}
    text = _get(AFOS_URL, params=params).text
    years = {start.year, end.year, (end + timedelta(days=2)).year}

    best: dict[object, dict] = {}
    for m in re.finditer(
        r"KNYC\s+DS\s+\d{3,4}\s+(\d{2})/(\d{2})\s+(-?\d{1,3})(\d{4})", text
    ):
        day, month = int(m.group(1)), int(m.group(2))
        val, tstr = float(m.group(3)), m.group(4)
        hh, mn = int(tstr[:2]), int(tstr[2:])
        if not (TEMP_MIN_F <= val <= TEMP_MAX_F and 0 <= hh <= 23 and 0 <= mn <= 59):
            continue
        # Pick the year that lands the DD/MM inside the fetch window.
        d = None
        for y in sorted(years):
            try:
                cand = datetime(y, month, day).date()
            except ValueError:
                continue
            if start.date() - timedelta(days=1) <= cand <= end.date() + timedelta(days=2):
                d = cand
                break
        if d is None:
            continue
        dtime = datetime(d.year, d.month, d.day, hh, mn, tzinfo=NY_TZ)
        cur = best.get(d)
        if cur is None or val > cur["dsm_max"]:
            best[d] = {"local_date": d, "dsm_max": val, "dsm_time": dtime}

    return pd.DataFrame(best.values())


# ── Assemble per-date actual daily high ─────────────────────────────────────
def build_daily_high(obs: pd.DataFrame, cli: pd.DataFrame, dsm: pd.DataFrame) -> pd.DataFrame:
    """One row per NY-local date: actual_high, high_time, and winning source."""
    o = obs.copy()
    o["valid_local"] = o["valid"].dt.tz_convert(NY_TZ)
    o["local_date"] = o["valid_local"].dt.date
    o["mxtmpf_6hr"] = o["metar"].apply(parse_6hr_max)

    rows = []
    for d, g in o.groupby("local_date"):
        tmax = g["tmpf"].max()
        hourly_max = float(tmax) if pd.notna(tmax) else np.nan
        hourly_time = (g.loc[g["tmpf"].idxmax(), "valid_local"]
                       if pd.notna(tmax) else None)
        s6 = g["mxtmpf_6hr"].max()
        sixhr_max = float(s6) if pd.notna(s6) else np.nan
        rows.append({"local_date": d, "hourly_max": hourly_max,
                     "hourly_time": hourly_time, "sixhr_max": sixhr_max})
    daily = pd.DataFrame(rows)

    if not cli.empty:
        daily = daily.merge(cli, on="local_date", how="left")
    else:
        daily["cli_max"], daily["cli_time"] = np.nan, None
    if not dsm.empty:
        daily = daily.merge(dsm, on="local_date", how="left")
    else:
        daily["dsm_max"], daily["dsm_time"] = np.nan, None

    def resolve(r):
        # Priority, NOT max(): the CLI is the official NWS daily high and the DSM
        # is the raw ASOS daily max — both derive from continuous 1-min monitoring
        # and already capture highs the hourly :51 obs or 6-hr max groups miss.
        # The hourly / 6-hr sources are fallback ONLY (used when no CLI/DSM):
        # the 00/06 UTC 6-hr max groups straddle the NY-local midnight boundary,
        # so letting them override the official value upward would import the
        # neighbouring day's warmth. flag_gap notes when they disagreed.
        cli, dsm = r.get("cli_max"), r.get("dsm_max")
        official = np.nanmax([v for v in (cli, dsm) if pd.notna(v)]) \
            if (pd.notna(cli) or pd.notna(dsm)) else np.nan
        fallback_vals = [v for v in (r.get("hourly_max"), r.get("sixhr_max")) if pd.notna(v)]
        fallback = max(fallback_vals) if fallback_vals else np.nan

        if pd.notna(official):
            value = float(official)
            source = "CLI" if (pd.notna(cli) and cli >= (dsm if pd.notna(dsm) else -1e9)) else "DSM"
            htime = r.get("cli_time") if source == "CLI" else r.get("dsm_time")
            if htime is None:  # value present but time unparsed -> approximate
                htime = r.get("dsm_time") or r.get("hourly_time")
        elif pd.notna(fallback):
            value, source, htime = float(fallback), "hourly/6hr", r.get("hourly_time")
        else:
            return pd.Series({"actual_high": np.nan, "high_time": None,
                              "high_source": None, "flag_gap": ""})

        # Informative flags (do not change the value):
        flags = []
        if pd.notna(cli) and pd.notna(dsm) and abs(cli - dsm) >= 2:
            flags.append(f"CLI/DSM split {cli:.0f}/{dsm:.0f}")
        if pd.notna(official) and pd.notna(r.get("hourly_max")) and official - r["hourly_max"] >= 1:
            flags.append(f"official>{r['hourly_max']:.0f} hourly by {official - r['hourly_max']:.0f}")
        return pd.Series({"actual_high": value, "high_time": htime,
                          "high_source": source, "flag_gap": "; ".join(flags)})

    daily = pd.concat([daily, daily.apply(resolve, axis=1)], axis=1)
    return daily.sort_values("local_date").reset_index(drop=True)


# ── Load predictions ────────────────────────────────────────────────────────
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"valid_t", "pred_high", "pred_t3h", "pred_t6h"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name} missing columns: {missing}")
    df = df.dropna(subset=["valid_t"]).copy()
    df["valid_utc"] = pd.to_datetime(df["valid_t"], utc=True)
    for c in ("pred_high", "pred_t3h", "pred_t6h"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # One prediction per obs time; keep the most recent write.
    df = df.drop_duplicates("valid_utc", keep="last").sort_values("valid_utc")
    df["valid_local"] = df["valid_utc"].dt.tz_convert(NY_TZ)
    df["local_date"] = df["valid_local"].dt.date
    df["local_hour"] = df["valid_local"].dt.hour
    return df.reset_index(drop=True)


# ── Scoring: daily high ─────────────────────────────────────────────────────
def _grade(err: float) -> float:
    """Graded credit for a daily-high call: 0 F err -> 1.0, >=3 F err -> 0."""
    return max(0.0, 1.0 - abs(err) / 3.0)


def score_daily_high(preds: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Score pre-high daily-high predictions.

    Scheme (my choice, since none was specified):
      * Eligible = predictions issued strictly BEFORE the day's high occurred.
        (After the high, the bot's obs-floor is trivially right — no skill.)
      * Per prediction: abs error vs the actual high, plus a graded 0..1 credit
        (1.0 at 0 F, linearly to 0 at 3 F).
      * Headline = MAE over all eligible predictions and mean graded skill x100.
      * hit-rates within +/-1/2/3 F; per-issue-hour and per-lead-time breakdowns
        surface WHEN (clock hour, and how far ahead) the model is trustworthy.
    """
    dh = daily.set_index("local_date")
    recs = []
    for _, p in preds.iterrows():
        if pd.isna(p["pred_high"]) or p["local_date"] not in dh.index:
            continue
        row = dh.loc[p["local_date"]]
        actual, htime = row["actual_high"], row["high_time"]
        if pd.isna(actual) or htime is None:
            continue
        eligible = p["valid_local"] < htime
        lead_h = (htime - p["valid_local"]).total_seconds() / 3600.0
        err = p["pred_high"] - actual
        recs.append({
            "local_date": p["local_date"], "issue_hour": p["local_hour"],
            "lead_h": lead_h, "eligible": eligible,
            "pred": p["pred_high"], "actual": actual,
            "abs_err": abs(err), "grade": _grade(err),
        })
    r = pd.DataFrame(recs)
    elig = r[r["eligible"]].copy() if not r.empty else r

    out = {"n_predictions": len(r), "n_eligible": len(elig)}
    if elig.empty:
        return out

    out["mae"] = float(elig["abs_err"].mean())
    out["rmse"] = float(np.sqrt((elig["abs_err"] ** 2).mean()))
    out["skill_pct"] = float(elig["grade"].mean() * 100)
    out["bias"] = float((elig["pred"] - elig["actual"]).mean())
    for k in (1, 2, 3):
        out[f"within_{k}"] = float((elig["abs_err"] <= k).mean() * 100)

    # Per-day: how close did each day's eligible calls get, and the earliest one.
    per_day = []
    for d, g in elig.groupby("local_date"):
        g = g.sort_values("lead_h")  # ascending lead = latest first
        earliest = g.iloc[-1]        # largest lead time
        per_day.append({
            "local_date": d, "actual_high": g["actual"].iloc[0],
            "n_eligible": len(g), "mae": g["abs_err"].mean(),
            "skill_pct": g["grade"].mean() * 100,
            "earliest_lead_h": earliest["lead_h"],
            "earliest_err": earliest["abs_err"],
        })
    out["per_day"] = pd.DataFrame(per_day).sort_values("local_date")

    out["by_hour"] = (elig.groupby("issue_hour")
                          .agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                               skill=("grade", lambda s: s.mean() * 100))
                          .reset_index())
    # Lead-time buckets (hours before the high).
    bins = [0, 2, 4, 6, 9, 12, 100]
    labels = ["0-2", "2-4", "4-6", "6-9", "9-12", "12+"]
    elig["lead_bucket"] = pd.cut(elig["lead_h"], bins=bins, labels=labels, right=False)
    out["by_lead"] = (elig.groupby("lead_bucket", observed=True)
                          .agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                               skill=("grade", lambda s: s.mean() * 100))
                          .reset_index())
    return out


# ── Scoring: t+3h / t+6h ────────────────────────────────────────────────────
def score_horizon(preds: pd.DataFrame, obs: pd.DataFrame,
                  horizon_h: int, pred_col: str) -> dict:
    """Out-of-24 daily scoring for a fixed-lead temperature forecast."""
    obs_lookup = obs.dropna(subset=["tmpf"]).set_index("valid")["tmpf"].to_dict()

    recs = []
    for _, p in preds.iterrows():
        if pd.isna(p[pred_col]):
            continue
        target = p["valid_utc"] + timedelta(hours=horizon_h)
        actual = obs_lookup.get(target)
        if actual is None:  # obs not yet settled / bot gap at target time
            continue
        err = p[pred_col] - actual
        recs.append({
            "local_date": p["local_date"], "issue_hour": p["local_hour"],
            "target_hour": target.astimezone(NY_TZ).hour,
            "pred": p[pred_col], "actual": actual,
            "abs_err": abs(err), "hit": abs(err) <= HIT_TOL,
        })
    r = pd.DataFrame(recs)
    out = {"n_scored": len(r)}
    if r.empty:
        return out

    out["mae"] = float(r["abs_err"].mean())
    out["rmse"] = float(np.sqrt((r["abs_err"] ** 2).mean()))
    out["bias"] = float((r["pred"] - r["actual"]).mean())
    out["total_points"] = int(r["hit"].sum())
    out["total_available"] = len(r)
    out["hit_rate"] = float(r["hit"].mean() * 100)
    for k in (2, 3):
        out[f"within_{k}"] = float((r["abs_err"] <= k).mean() * 100)

    # Per NY-local day: points / 24 (raw) and / available (hours the bot ran).
    per_day = (r.groupby("local_date")
                 .agg(points=("hit", "sum"), available=("hit", "size"),
                      mae=("abs_err", "mean"))
                 .reset_index())
    per_day["score_24"] = per_day["points"].astype(str) + " / 24"
    per_day["rate_pct"] = per_day["points"] / per_day["available"] * 100
    out["per_day"] = per_day.sort_values("local_date")

    out["by_issue_hour"] = (r.groupby("issue_hour")
                              .agg(n=("hit", "size"), hit_pct=("hit", lambda s: s.mean() * 100),
                                   mae=("abs_err", "mean")).reset_index())
    out["by_target_hour"] = (r.groupby("target_hour")
                               .agg(n=("hit", "size"), hit_pct=("hit", lambda s: s.mean() * 100),
                                    mae=("abs_err", "mean")).reset_index())
    return out


# ── Reporting ───────────────────────────────────────────────────────────────
def _fmt_hours(by_hour: pd.DataFrame, val_col: str, ascending: bool, n: int = 4) -> str:
    valid = by_hour[by_hour["n"] >= 2] if "n" in by_hour else by_hour
    if valid.empty:
        valid = by_hour
    best = valid.sort_values(val_col, ascending=ascending).head(n)
    hcol = "issue_hour" if "issue_hour" in best else best.columns[0]
    return ", ".join(f"{int(h):02d}:00 ({v:.2f})"
                     for h, v in zip(best[hcol], best[val_col]))


def print_report(daily: pd.DataFrame, dh: dict, s3: dict, s6: dict,
                 preds: pd.DataFrame) -> None:
    line = "=" * 70
    print(f"\n{line}\nKNYC NOWCASTER -- LIVE PREDICTION SCORECARD\n{line}")
    print(f"Predictions: {len(preds)}  |  "
          f"{preds['valid_local'].min():%Y-%m-%d %H:%M} -> "
          f"{preds['valid_local'].max():%Y-%m-%d %H:%M} (NY local)")
    print(f"Days covered: {preds['local_date'].nunique()}")

    # ── Daily high ──
    print(f"\n{'-'*70}\nDAILY HIGH MODEL  (pre-high predictions only)\n{'-'*70}")
    if dh.get("n_eligible", 0) == 0:
        print("  No eligible pre-high predictions in range.")
    else:
        print(f"  Eligible predictions : {dh['n_eligible']}  "
              f"(of {dh['n_predictions']} total; rest issued after the high)")
        print(f"  MAE                  : {dh['mae']:.2f} F")
        print(f"  RMSE                 : {dh['rmse']:.2f} F")
        print(f"  Bias (pred - actual) : {dh['bias']:+.2f} F")
        print(f"  Skill score          : {dh['skill_pct']:.1f} / 100")
        print(f"  Within +/-1 / 2 / 3 F: {dh['within_1']:.0f}% / "
              f"{dh['within_2']:.0f}% / {dh['within_3']:.0f}%")
        bh = dh["by_hour"].rename(columns={"mae": "v"})
        print(f"  Most accurate issue-hours (MAE): {_fmt_hours(bh, 'v', True)}")
        print(f"  Least accurate issue-hours(MAE): {_fmt_hours(bh, 'v', False)}")
        print("  Accuracy by lead time before the high:")
        for _, r in dh["by_lead"].iterrows():
            print(f"      {str(r['lead_bucket']):>5} h  n={int(r['n']):>3}  "
                  f"MAE={r['mae']:.2f} F  skill={r['skill']:.0f}")
        print("  Per-day:")
        for _, r in dh["per_day"].iterrows():
            print(f"      {r['local_date']}  high={r['actual_high']:.0f}F  "
                  f"n={int(r['n_eligible']):>2}  MAE={r['mae']:.2f}  "
                  f"skill={r['skill_pct']:.0f}  | earliest call "
                  f"{r['earliest_lead_h']:.1f}h out, off {r['earliest_err']:.1f}F")

    # ── t+3h / t+6h ──
    for label, s in (("t+3h", s3), ("t+6h", s6)):
        print(f"\n{'-'*70}\n{label.upper()} MODEL  (out of 24/day, hit = within "
              f"+/-{HIT_TOL:.0f} F)\n{'-'*70}")
        if s.get("n_scored", 0) == 0:
            print("  No scorable predictions (targets not yet settled?).")
            continue
        print(f"  Total score          : {s['total_points']} / {s['total_available']} "
              f"available hourly slots  ({s['hit_rate']:.1f}%)")
        print(f"  MAE                  : {s['mae']:.2f} F")
        print(f"  RMSE                 : {s['rmse']:.2f} F")
        print(f"  Bias (pred - actual) : {s['bias']:+.2f} F")
        print(f"  Within +/-1 / 2 / 3 F: {s['hit_rate']:.0f}% / "
              f"{s['within_2']:.0f}% / {s['within_3']:.0f}%")
        bi = s["by_issue_hour"].rename(columns={"hit_pct": "v"})
        bt = s["by_target_hour"].rename(columns={"hit_pct": "v"})
        print(f"  Best issue-hours  (hit%): {_fmt_hours(bi, 'v', False)}")
        print(f"  Best target-hours (hit%): {_fmt_hours(bt, 'v', False)}")
        print("  Per-day scorecard:")
        for _, r in s["per_day"].iterrows():
            print(f"      {r['local_date']}  {r['score_24']:>7}  "
                  f"(avail {int(r['available'])}, {r['rate_pct']:.0f}%)  "
                  f"MAE={r['mae']:.2f} F")
    print(line)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Score logged KNYC nowcaster predictions.")
    ap.add_argument("--log", default=str(HERE / "nowcast_log.csv"))
    ap.add_argument("--start", help="YYYY-MM-DD (NY local) inclusive lower bound")
    ap.add_argument("--end", help="YYYY-MM-DD (NY local) inclusive upper bound")
    ap.add_argument("--out", help="dir to write CSV breakdowns (optional)")
    args = ap.parse_args()

    preds = load_predictions(Path(args.log))
    if args.start:
        preds = preds[preds["local_date"] >= pd.to_datetime(args.start).date()]
    if args.end:
        preds = preds[preds["local_date"] <= pd.to_datetime(args.end).date()]
    if preds.empty:
        raise SystemExit("No predictions in the selected range.")

    # Obs window must extend +6h past the last prediction for t+6h targets.
    lo = preds["valid_utc"].min().to_pydatetime() - timedelta(hours=1)
    hi = preds["valid_utc"].max().to_pydatetime() + timedelta(hours=7)
    print(f"Fetching ground truth {lo:%Y-%m-%d} -> {hi:%Y-%m-%d} ...")
    obs = fetch_hourly_obs(lo, hi)
    print(f"  hourly obs rows: {len(obs)}")
    time.sleep(1)
    cli = fetch_cli(lo, hi)
    print(f"  CLI daily-high records: {len(cli)}")
    time.sleep(1)
    try:
        dsm = fetch_dsm(lo, hi)
    except Exception as exc:
        print(f"  DSM fetch failed ({exc}) -- continuing without it")
        dsm = pd.DataFrame()
    print(f"  DSM daily-max records: {len(dsm)}")

    daily = build_daily_high(obs, cli, dsm)
    dh = score_daily_high(preds, daily)
    s3 = score_horizon(preds, obs, 3, "pred_t3h")
    s6 = score_horizon(preds, obs, 6, "pred_t6h")

    print_report(daily, dh, s3, s6, preds)

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        daily.to_csv(outdir / "daily_high_truth.csv", index=False)
        if "per_day" in dh:
            dh["per_day"].to_csv(outdir / "daily_high_per_day.csv", index=False)
        if "per_day" in s3:
            s3["per_day"].to_csv(outdir / "t3h_per_day.csv", index=False)
        if "per_day" in s6:
            s6["per_day"].to_csv(outdir / "t6h_per_day.csv", index=False)
        print(f"\nWrote CSV breakdowns to {outdir}/")


if __name__ == "__main__":
    main()
