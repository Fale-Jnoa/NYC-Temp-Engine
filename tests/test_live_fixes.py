"""
Offline regression tests for the fixes from the 2026-11 production review
(docs/REVIEW_2026-11.md). No network access needed.

    python -m pytest tests/        # or:  python tests/test_live_fixes.py

Importing knyc_discord_bot needs a .env with DISCORD_TOKEN / GUILD_ID (any
non-empty values work — the bot is never started) and the model .pkl files.
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

import knyc_discord_bot as bot  # noqa: E402
from knyc import build_training_data as btd  # noqa: E402
from knyc import score_predictions as sp  # noqa: E402


def _times(unit: str) -> pd.Series:
    t = pd.Series(pd.to_datetime(
        ["2026-06-21 17:00", "2026-11-16 17:00", "2026-03-20 12:00", "2027-01-01 05:51"],
        utc=True))
    return t.astype(f"datetime64[{unit}, UTC]")


# ── P0-1: solar geometry must not depend on datetime resolution ─────────────
def test_solar_zenith_unit_independent():
    for fn in (bot.solar_zenith_deg,
               lambda t: btd.solar_zenith_deg(t, bot.KNYC_LAT, bot.KNYC_LON)):
        ref = np.asarray(fn(_times("ns")))
        for unit in ("us", "ms", "s"):
            np.testing.assert_allclose(np.asarray(fn(_times(unit))), ref, atol=1e-9)


def test_solar_altitude_sanity():
    alt = 90 - np.asarray(bot.solar_zenith_deg(_times("us")))
    # Summer-solstice noon-ish vs mid-November noon-ish at Central Park.
    assert abs(alt[0] - 72.7) < 0.3, alt[0]
    assert abs(alt[1] - 30.2) < 0.3, alt[1]


# ── P0-2: clear sky maps to 0 oktas ─────────────────────────────────────────
def test_clear_sky_oktas():
    assert bot.cloud_oktas({"skyc1": "CLR"}) == 0
    assert bot.cloud_oktas({"skyc1": "VV "}) == 8
    assert np.isnan(bot.cloud_oktas({"skyc1": np.nan}))


# ── P0-4: evening reassessed rule ───────────────────────────────────────────
def test_reassess_high():
    r = bot.reassess_high
    assert r(60.0, None, 55.0, 10) == 60.0                 # no obs yet
    assert r(60.0, 58.0, 57.0, 14) == 60.0                 # daytime: max(obs, pred)
    assert r(57.0, 58.0, 57.0, 14) == 58.0
    assert r(60.0, 58.0, 57.0, 19) == 58.0                 # evening, temp fell >=1F: obs
    assert r(60.0, 58.0, 57.5, 20) == 60.0                 # evening, still near the high
    assert r(57.0, 58.0, 55.0, 22) == 58.0


# ── P1-1: obs floor uses the LST climate day ────────────────────────────────
def _obs_frame(rows):
    df = pd.DataFrame(rows, columns=["valid", "tmpf", "mxtmpf_6hr"])
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df


def test_observed_high_lst_day_during_dst():
    # 2026-10-17 (EDT): the 00:51 EDT ob (04:51Z) belongs to the Oct 16 climate
    # day; the CLI high for Oct 17 was 62 while that ob was 63.
    df = _obs_frame([
        ("2026-10-17 03:51", 63.5, np.nan),   # 23:51 EDT Oct 16
        ("2026-10-17 04:51", 62.96, np.nan),  # 00:51 EDT Oct 17 -> climate day Oct 16
        ("2026-10-17 05:51", 61.0, np.nan),   # 01:51 EDT -> climate day Oct 17
        ("2026-10-17 06:51", 60.0, 63.0),     # 6-hr window starts in Oct 16: ignored
        ("2026-10-17 12:51", 59.0, 61.5),     # window 06:51Z-12:51Z inside Oct 17
    ])
    assert bot._observed_high(df, date(2026, 10, 17)) == 61.5


def test_observed_high_standard_time_unchanged():
    # In EST the climate day equals the civil day.
    df = _obs_frame([
        ("2026-11-16 04:51", 50.0, np.nan),   # 23:51 EST Nov 15
        ("2026-11-16 05:51", 48.0, np.nan),   # 00:51 EST Nov 16
        ("2026-11-16 17:51", 53.0, np.nan),
    ])
    assert bot._observed_high(df, date(2026, 11, 16)) == 53.0


def test_cli_time_is_lst():
    # "MAXIMUM 62 1208 AM" on a DST day is 01:08 EDT.
    t = pd.Timestamp(2026, 10, 17, 0, 8, tz=sp.CLIMATE_TZ).tz_convert(sp.NY_TZ)
    assert (t.hour, t.minute) == (1, 8)


# ── P0-3: wait for late upstream stations, then post + log exactly once ─────
def _run_post_scenario(results: list[dict], *, last_posted, wait_for_fresh=True):
    """Drive post_nowcast_to_channel with canned compute_nowcast results.
    Returns (embeds sent, rows logged, compute calls)."""
    import asyncio
    from datetime import timedelta
    from types import SimpleNamespace

    sent, logged, calls = [], [], []

    class FakeChannel:
        name = bot.CHANNEL_NAME

        async def send(self, *args, **kwargs):
            sent.append(kwargs.get("embed") or args)

    guild = SimpleNamespace(text_channels=[FakeChannel()], name="test")
    clock = {"t": pd.Timestamp("2026-11-16 18:55", tz="UTC").to_pydatetime()}

    class FakeDatetime(bot.datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["t"] if tz is None else clock["t"].astimezone(tz)

    def fake_compute():
        calls.append(1)
        return results[min(len(calls) - 1, len(results) - 1)]

    async def fake_sleep(sec):
        clock["t"] += timedelta(seconds=sec)

    saved = (bot.compute_nowcast, bot.log_nowcast, bot.datetime, bot.asyncio.sleep,
             bot.bot.get_guild, bot.last_posted_valid, bot.last_logged_valid)
    try:
        bot.compute_nowcast = fake_compute
        bot.log_nowcast = lambda d: logged.append(d["valid_t"])
        bot.datetime = FakeDatetime
        bot.asyncio.sleep = fake_sleep
        bot.bot.get_guild = lambda gid: guild
        bot.last_posted_valid = bot.last_logged_valid = last_posted
        asyncio.run(bot.post_nowcast_to_channel(wait_for_fresh=wait_for_fresh))
    finally:
        (bot.compute_nowcast, bot.log_nowcast, bot.datetime, bot.asyncio.sleep,
         bot.bot.get_guild, bot.last_posted_valid, bot.last_logged_valid) = saved
    return sent, logged, len(calls)


def _fake_data(valid: str, stale: list[str]) -> dict:
    return {"valid_t": pd.Timestamp(valid, tz="UTC"), "today_local": date(2026, 11, 16),
            "cur_temp": 53.0, "cur_dwpf": 33.0, "cur_relh": 46.0, "cur_wind_kt": 4.0,
            "cur_drct": 270.0, "pred_t3h": 52.0, "pred_t6h": 49.0, "pred_high": 55.0,
            "obs_high": 53.0, "reassessed": 55.0, "degraded": [],
            "stale_upstream": stale, "features": {}}


def test_post_waits_for_late_upstream_then_logs_once():
    prev = pd.Timestamp("2026-11-16 17:51", tz="UTC")
    results = [_fake_data("2026-11-16 18:51", ["KPHL", "KBWI", "KBOS"]),
               _fake_data("2026-11-16 18:51", ["KBOS"]),
               _fake_data("2026-11-16 18:51", [])]
    sent, logged, n = _run_post_scenario(results, last_posted=prev)
    assert n == 3 and len(sent) == 1 and len(logged) == 1


def test_post_gives_up_waiting_after_sync_window():
    prev = pd.Timestamp("2026-11-16 17:51", tz="UTC")
    results = [_fake_data("2026-11-16 18:51", ["KBOS"])]
    sent, logged, n = _run_post_scenario(results, last_posted=prev)
    assert len(sent) == 1 and len(logged) == 1
    assert n == bot.UPSTREAM_SYNC_WAIT.total_seconds() // 60 + 1
    assert "KBOS" in sent[0].footer.text


def test_restart_does_not_relog_hour():
    # on_ready path: posts the embed, but the hour is already in the log.
    hour = pd.Timestamp("2026-11-16 18:51", tz="UTC")
    sent, logged, _ = _run_post_scenario([_fake_data("2026-11-16 18:51", [])],
                                         last_posted=hour, wait_for_fresh=False)
    assert len(sent) == 1 and logged == []


# ── 2026-11-16 layout change: runtime files move from the root into logs/ ───
def test_migrate_legacy_runtime_files():
    import tempfile
    from knyc import paths

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "logs"
        files = tuple(logs / f.name for f in paths._RUNTIME_FILES)
        saved = (paths.ROOT, paths.LOGS_DIR, paths._RUNTIME_FILES)
        try:
            paths.ROOT, paths.LOGS_DIR, paths._RUNTIME_FILES = root, logs, files
            (root / "nowcast_log.csv").write_bytes(b"valid_t\n2026-11-16T20:51:00+00:00\n")
            (root / "market_log.csv").write_bytes(b"a\n")
            logs.mkdir()
            (logs / "market_log.csv").write_bytes(b"newer\n")   # already migrated: keep both
            moved = paths.migrate_legacy_runtime_files()
            assert moved == ["nowcast_log.csv"]
            assert (logs / "nowcast_log.csv").read_bytes().endswith(b"20:51:00+00:00\n")
            assert not (root / "nowcast_log.csv").exists()
            assert (root / "market_log.csv").exists()
            assert (logs / "market_log.csv").read_bytes() == b"newer\n"
            assert paths.migrate_legacy_runtime_files() == []    # idempotent
        finally:
            paths.ROOT, paths.LOGS_DIR, paths._RUNTIME_FILES = saved


# ── Side-by-side v2 trial (docs/PLAN_GUIDANCE_V2.md) ────────────────────────
def _candidate_frame(cand) -> tuple[pd.DataFrame, pd.Timestamp]:
    """48 hourly rows carrying every production feature (values arbitrary)."""
    rng = np.random.default_rng(0)
    v = pd.date_range("2026-11-15 00:51", periods=48, freq="1h", tz="UTC")
    df = pd.DataFrame({c: rng.normal(50, 5, len(v)) for c in bot.FEATURE_COLS})
    df["valid"] = v
    df["tmpf"] = 50 + rng.normal(0, 3, len(v))
    df["mxtmpf_6hr"] = np.nan
    df["month"], df["hour"] = 11, v.tz_convert("America/New_York").hour
    return df, v[-1]


def test_candidate_falls_back_when_guidance_down():
    from knyc import candidate as cm
    from knyc import guidance as g
    cand = cm.load("guidance_v2")
    if cand is None:
        print("  (skipped: guidance_v2 artifacts not present)")
        return
    df, t = _candidate_frame(cand)
    prod = {"pred_high": 61.0, "pred_t3h": 50.0, "pred_t6h": 48.0}
    saved = g.fetch_runs
    try:
        def down(*a, **k):
            raise ConnectionError("IEM down")
        g.fetch_runs = down
        cm.FAIL_TTL_S = 0
        out = cand.predict(df, t, prod)
    finally:
        g.fetch_runs = saved
    assert out["fallback"] and out["pred_high"] == 61.0 and out["sources_ok"] == "-"
    assert "IEM down" in out["fetch_errors"]


def test_shadow_row_logged_separately():
    import tempfile
    from knyc import kalshi_market as km
    from knyc import paths
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        saved = (bot.LOG_PATH, bot.MARKET_LOG_PATH, paths.SHADOW_LOG, km.fetch_brackets)
        try:
            bot.LOG_PATH, bot.MARKET_LOG_PATH = tmp / "nowcast_log.csv", tmp / "market_log.csv"
            paths.SHADOW_LOG = tmp / "shadow_log.csv"
            km.fetch_brackets = lambda d: pd.DataFrame()
            data = _fake_data("2026-11-16 18:51", [])
            data["shadow"] = {"candidate": "guidance_v2", "pred_high": 56.2, "reassessed": 56.2,
                              "pred_t3h": 51.0, "pred_t6h": 48.5, "fallback": False,
                              "sources_ok": "GFS,LAV,NBS", "fetch_errors": "", "gfs_nx_today": 56.0}
            bot.log_nowcast(data)
            s = pd.read_csv(paths.SHADOW_LOG)
            n = pd.read_csv(bot.LOG_PATH)
        finally:
            bot.LOG_PATH, bot.MARKET_LOG_PATH, paths.SHADOW_LOG, km.fetch_brackets = saved
    assert len(s) == 1 and s["pred_high_cand"].iloc[0] == 56.2 and s["pred_high_prod"].iloc[0] == 55.0
    assert "pred_high_cand" not in n.columns          # official log unchanged


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
