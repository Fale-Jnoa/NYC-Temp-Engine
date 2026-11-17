"""
Live side-by-side ("shadow") candidate model for the bot. The production
model stays official (nowcast log, Kalshi tape, scorecard); the candidate's
prediction is shown next to it in the Discord post and logged to
logs/shadow_log.csv. See docs/PLAN_GUIDANCE_RETRAIN.md section 8.

Fallback: when no guidance at all could be fetched, the candidate repeats the
production prediction and says so (all three sources come from the same IEM
server, so they tend to fail together).

Library only: no Discord imports. Every failure is reported to the caller as
an exception; the bot wraps the whole call so a post is never blocked.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from knyc import guidance as g
from knyc import paths

log = logging.getLogger("knyc-nowcaster")

FETCH_WINDOW = timedelta(hours=40)       # covers the 36 h max age of n_x
CACHE_TTL_S = 600                        # compute_nowcast re-runs every 60 s while waiting
FAIL_TTL_S = 120                         # retry a failed source sooner
MODELS = ("GFS", "LAV", "NBS")


class GuidanceCandidate:
    def __init__(self, name: str):
        self.name = name
        self.dir = paths.CANDIDATES_DIR / name
        self.manifest = json.loads((self.dir / "feature_manifest.json").read_text())
        self.cols = self.manifest["feature_cols"]
        self.model_high = joblib.load(self.dir / "knyc_model_daily_high.pkl")
        self.model_t3h = self._maybe_load("knyc_model_t3h.pkl")
        self.model_t6h = self._maybe_load("knyc_model_t6h.pkl")
        clim = pd.read_csv(self.dir / "tmpf_climatology.csv")
        self.clim = {(int(r.month), int(r.hour)): float(r.tmpf_clim) for r in clim.itertuples()}
        self._cache: dict[str, tuple[float, pd.DataFrame, str | None]] = {}

    def _maybe_load(self, fname: str):
        p = self.dir / fname
        return joblib.load(p) if p.exists() else None

    # ── guidance fetch (cached) ────────────────────────────────────────────
    def _runs(self, model: str, now: datetime) -> tuple[pd.DataFrame, str | None]:
        hit = self._cache.get(model)
        if hit and time.monotonic() - hit[0] < (FAIL_TTL_S if hit[2] else CACHE_TTL_S):
            return hit[1], hit[2]
        err = None
        try:
            raw = g.fetch_runs(model, now - FETCH_WINDOW, now + timedelta(hours=1),
                               retries=2, pause_s=3.0, timeout_s=10, backoff_s=5.0)  # IEM rate-limits bursts
            runs = g.clean_runs(raw, model)
        except Exception as exc:  # noqa: BLE001 - an outage degrades, never raises
            runs, err = g.empty_runs(model), f"{type(exc).__name__}: {exc}"[:200]
            log.warning("guidance %s fetch failed: %s", model, err)
        self._cache[model] = (time.monotonic(), runs, err)
        return runs, err

    # ── prediction ─────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame, valid_t: pd.Timestamp, prod: dict,
                now: datetime | None = None) -> dict:
        """df: the bot's engineered frame (valid, tmpf, mxtmpf_6hr, month, hour
        and the production features). prod: production pred_high/t3h/t6h."""
        now = now or datetime.now(timezone.utc)
        runs, errs = {}, {}
        unreachable = False
        for m in MODELS:
            if unreachable:          # IEM down: don't spend another timeout per source
                runs[m], errs[m] = g.empty_runs(m), "skipped: IEM unreachable this cycle"
                continue
            runs[m], errs[m] = self._runs(m, now)
            unreachable = bool(errs[m]) and any(k in errs[m] for k in ("ConnectionError", "Timeout"))

        obs = df[df["valid"] <= valid_t][["valid", "tmpf", "mxtmpf_6hr"]].reset_index(drop=True)
        obs_feat = g.obs_day_features(obs).iloc[[-1]].reset_index(drop=True)
        row = df[df["valid"] == valid_t].iloc[[-1]].reset_index(drop=True).copy()
        key = (int(row["month"].iloc[0]), int(row["hour"].iloc[0]))
        row["tmpf_clim_global"] = self.clim.get(key, float(row["tmpf"].iloc[0]))
        row["tmpf_anomaly"] = row["tmpf"] - row["tmpf_clim_global"]
        gf = g.guidance_features(row[["valid", "tmpf"]], runs["GFS"], runs["LAV"], runs["NBS"])
        X = pd.concat([row.drop(columns=[c for c in obs_feat.columns if c in row]), obs_feat, gf], axis=1)

        guid = [c for c in self.cols if c in g.GUIDANCE_COLS]
        have = int(X[guid].notna().any(axis=None)) if guid else 0
        fallback = not have
        if fallback:
            pred_high, pred_t3h, pred_t6h = prod["pred_high"], prod["pred_t3h"], prod["pred_t6h"]
        else:
            Xc = X[self.cols]
            pred_high = float(self.model_high.predict(Xc)[0])
            pred_t3h = float(self.model_t3h.predict(Xc)[0]) if self.model_t3h else prod["pred_t3h"]
            pred_t6h = float(self.model_t6h.predict(Xc)[0]) if self.model_t6h else prod["pred_t6h"]

        out = {"candidate": self.name, "pred_high": pred_high, "pred_t3h": pred_t3h,
               "pred_t6h": pred_t6h, "fallback": fallback,
               "sources_ok": ",".join(m for m in MODELS if not runs[m].empty) or "-",
               "fetch_errors": "; ".join(f"{m}: {e}" for m, e in errs.items() if e) or ""}
        for c in g.OBS_DAY_COLS + g.GUIDANCE_COLS:
            out[c] = _num(X[c].iloc[0]) if c in X else np.nan
        for c in g.RUNTIME_COLS:
            v = gf[c].iloc[0]
            out[c] = v.isoformat() if pd.notna(v) else ""
        for m in MODELS:   # newest run on IEM at fetch time: dissemination-delay measurement
            out[f"{m.lower()}_newest_seen"] = runs[m]["runtime"].max().isoformat() if not runs[m].empty else ""
        return out


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(name: str | None) -> GuidanceCandidate | None:
    """The configured candidate, or None (disabled / missing files)."""
    if not name or name.lower() in ("off", "none", "0"):
        return None
    try:
        c = GuidanceCandidate(name)
        log.info("Side-by-side candidate %s loaded (%d features)", name, len(c.cols))
        return c
    except Exception as exc:  # noqa: BLE001
        log.warning("candidate %s not loaded: %s", name, exc)
        return None
