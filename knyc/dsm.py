"""
ASOS Daily Summary Messages for Central Park (DSMNYC), read live.

KNYC issues one around 20:15Z (max so far through 15:00 LST), one around
21:15Z (through 16:00 LST) and a final around 05:15Z the next day. Each carries
the official max so far and the LST minute it happened, i.e. the 5-minute-
averaged high that the CLI report (and Kalshi) settles on. The hourly METARs
miss that high on most days (docs/PLAN_PWS_STUDY.md), so the bot folds the
DSM max into its observed-high floor once one is out.

    KNYC DS 1500 16/11 551425/ 400656// ...   -> through 15 LST on 16 Nov:
                       ^^ ^^^^                   max 55 F at 14:25 LST
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import requests

AFOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
LST = timezone(timedelta(hours=-5))
TEMP_MIN_F, TEMP_MAX_F = -40.0, 130.0

_PRODUCT = re.compile(
    r"CXUS41 KOKX (\d{2})(\d{2})(\d{2})\s+DSMNYC\s+KNYC DS (?:COR )?(?:(\d{4}) )?"
    r"(\d{2})/(\d{2}) (-?\d{1,3})(\d{4})/")


def parse(text: str, now_utc: datetime) -> list[dict]:
    """All DSMs in `text`. Years are inferred relative to now_utc (the
    messages carry only DD/MM); the issue time comes from the WMO header."""
    out = []
    for m in _PRODUCT.finditer(text):
        hd, hh, hm, valid_hhmm, dd, mm, mx, mt = m.groups()
        cdate = None
        for y in (now_utc.year - 1, now_utc.year, now_utc.year + 1):
            try:
                c = date(y, int(mm), int(dd))
            except ValueError:
                continue
            if abs((c - now_utc.date()).days) <= 40:
                cdate = c
        if cdate is None:
            continue
        issued = None
        for k in range(3):
            cand = cdate + timedelta(days=k)
            if cand.day == int(hd):
                issued = datetime(cand.year, cand.month, cand.day, int(hh), int(hm), tzinfo=timezone.utc)
                break
        val, hhmm = float(mx), mt
        if issued is None or not (TEMP_MIN_F <= val <= TEMP_MAX_F):
            continue
        if not (0 <= int(hhmm[:2]) <= 24 and 0 <= int(hhmm[2:]) <= 59):
            continue
        t = datetime(cdate.year, cdate.month, cdate.day, tzinfo=LST) + timedelta(
            hours=int(hhmm[:2]), minutes=int(hhmm[2:]))
        out.append({"climate_date": cdate,
                    "valid_through_lst": int(valid_hhmm[:2]) if valid_hhmm else None,
                    "max_f": val, "max_time": t.astimezone(timezone.utc), "issued": issued})
    return out


def fetch_latest(now_utc: datetime | None = None, *, limit: int = 6, timeout_s: int = 10) -> list[dict]:
    """The most recent DSMs (raises on network/HTTP errors)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    r = requests.get(AFOS_URL, params={"pil": "DSMNYC", "limit": str(limit), "fmt": "text"},
                     headers={"User-Agent": "knyc-nowcaster/1.0"}, timeout=timeout_s)
    r.raise_for_status()
    return parse(r.text, now_utc)


def max_so_far(messages: list[dict], climate_date: date, now_utc: datetime) -> dict | None:
    """Highest DSM max for `climate_date` among messages already issued."""
    mine = [m for m in messages if m["climate_date"] == climate_date and m["issued"] <= now_utc]
    return max(mine, key=lambda m: (m["max_f"], m["issued"])) if mine else None
