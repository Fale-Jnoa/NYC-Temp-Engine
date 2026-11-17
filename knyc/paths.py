"""
Where every file lives. Import paths from here instead of building them
next to `__file__`, so moving a folder is a one-line change.

    models/   trained artifacts the bot loads (tracked in git)
    data/     training data and pulled server copies (gitignored, large)
    logs/     everything the live bot writes (gitignored, per host)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"

# Trained artifacts
MODEL_DAILY_HIGH = MODELS_DIR / "knyc_model_daily_high.pkl"
MODEL_T3H = MODELS_DIR / "knyc_model_t3h.pkl"
MODEL_T6H = MODELS_DIR / "knyc_model_t6h.pkl"
FEATURE_MANIFEST = MODELS_DIR / "feature_manifest.json"
CLIMATOLOGY = MODELS_DIR / "tmpf_climatology.csv"
OFFSET_PMF = MODELS_DIR / "offset_pmf.csv"
RESIDUAL_MODEL = MODELS_DIR / "residual_model.csv"

# Training data
TRAINING_CSV = DATA_DIR / "knyc_training.csv"
TRAINING_TAIL_CSV = DATA_DIR / "knyc_training_tail.csv"   # obs after the base CSV ends
SERVER_PULL_DIR = DATA_DIR / "server_pull"

# Forecast guidance (docs/PLAN_GUIDANCE_RETRAIN.md)
GUIDANCE_DIR = DATA_DIR / "guidance"
GUIDANCE_RAW_DIR = GUIDANCE_DIR / "raw"
GUIDANCE_FEATURES = GUIDANCE_DIR / "guidance_features.csv"
CANDIDATES_DIR = MODELS_DIR / "candidates"
SHADOW_LOG = LOGS_DIR / "shadow_log.csv"   # phase 5; not a legacy runtime file

# Runtime files written by the bot
NOWCAST_LOG = LOGS_DIR / "nowcast_log.csv"
MARKET_LOG = LOGS_DIR / "market_log.csv"
PREDICTION_TAPE = LOGS_DIR / "prediction_tape.csv"
BOT_STATE = LOGS_DIR / "bot_state.json"
DAILY_CHART_PNG = LOGS_DIR / "daily_chart.png"

_RUNTIME_FILES = (NOWCAST_LOG, MARKET_LOG, PREDICTION_TAPE, BOT_STATE, DAILY_CHART_PNG)


def migrate_legacy_runtime_files() -> list[str]:
    """One-time move of runtime files from the repo root (pre-2026-11-16
    layout) into logs/. A rename only: file contents are never rewritten.
    Skips any file that already exists in logs/, so it is safe to call on
    every start. Returns the names it moved."""
    LOGS_DIR.mkdir(exist_ok=True)
    moved = []
    for new in _RUNTIME_FILES:
        old = ROOT / new.name
        if old.exists() and not new.exists():
            os.replace(old, new)
            moved.append(new.name)
        elif old.exists():
            logging.getLogger(__name__).warning(
                "both %s and %s exist - leaving the root copy alone", old, new)
    return moved
