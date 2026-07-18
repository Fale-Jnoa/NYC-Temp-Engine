# KNYC Temp Engine

A machine-learning nowcasting system for Central Park, NYC (KNYC). It trains XGBoost models on KNYC ASOS observations plus upstream-station signal to predict short-term temperature and the day's high, then posts hourly forecasts to Discord. A separate scorer grades the bot's live predictions against settled ground truth.

## What it predicts

| Model | Target | Test MAE | vs. persistence |
|---|---|---|---|
| `knyc_model_daily_high.pkl` | NWS-settled daily high temperature | **1.98 °F** | +3.71 °F better |
| `knyc_model_t3h.pkl` | Temperature 3 hours ahead | **1.30 °F** | +1.39 °F better |
| `knyc_model_t6h.pkl` | Temperature 6 hours ahead | **2.05 °F** | +2.72 °F better |

Trained on 2010–2021, validated on 2022–2023, tested on 2024 → present (~18k held-out hours). The daily-high model's warm bias is effectively eliminated (+0.05 °F on the test set). "Persistence" is the naive baseline of assuming no change (see [below](#baselines)). Exact feature list and metrics live in [feature_manifest.json](feature_manifest.json).

## How it works

1. **[build_training_data.py](build_training_data.py)** pulls historical KNYC ASOS observations from the IEM archive, along with 8 upstream stations (KEWR, KLGA, KJFK, KTEB, KPHL, KBWI, KDCA, KBOS) for warm/cold-advection signal, and NWS Daily Climate Report (CLINYC) bulletins for the true settled daily high. It engineers features — wind u/v components, cloud oktas, solar zenith/altitude, pressure and temperature tendencies/lags/rolling stats, climatology anomaly, and per-station upstream deltas — and writes `knyc_training.csv` (untracked; regenerate locally).
2. **[KNYC_Nowcaster.ipynb](KNYC_Nowcaster.ipynb)** trains the three XGBoost models on that data with time-aware splits and sample weighting toward recent years, evaluates them on the held-out test split against naive baselines, and exports the trained `.pkl` models plus [feature_manifest.json](feature_manifest.json) (the exact feature list, upstream stations, and metrics the bot relies on) and [tmpf_climatology.csv](tmpf_climatology.csv).
3. **[knyc_discord_bot.py](knyc_discord_bot.py)** runs continuously: each hour it fetches the latest KNYC + upstream obs, re-engineers the same features used in training, runs all three models, and posts an embed to a `#predictions` Discord channel with current conditions, t+3h/t+6h forecasts, model daily high, observed high so far, and a floor-locked "reassessed" high (the prediction can't go below what's already been observed). Every prediction is logged to [nowcast_log.csv](nowcast_log.csv).
4. **[score_predictions.py](score_predictions.py)** grades the logged predictions after the fact against settled ground truth (see [Scoring](#scoring-live-accuracy)).

## Data sources

- **aviationweather.gov** (FAA/NWS live METAR API) — the bot's real-time obs feed. A given hour's METAR is queryable within ~2 minutes of issuance, versus the IEM archive's inconsistent ingest lag.
- **IEM ASOS archive** (`mesonet.agron.iastate.edu`) — settled historical hourly obs, used for training and as the scorer's ground-truth temperatures.
- **NWS CLI bulletins** (CLINYC) — official settled daily high/low. The training target for the daily-high model (with a METAR 6-hr-max fallback), and the scorer's authoritative daily high + the time it occurred.
- **ASOS DSM** (DSMNYC) — the raw ASOS daily summary max, used by the scorer as an independent cross-check on the CLI high.
- [tmpf_climatology.csv](tmpf_climatology.csv) — per (month, hour) mean temperature baseline used to compute the temperature-anomaly feature.

### Baselines

The notebook reports each model against a **persistence** baseline — the naive "nothing changes" forecast:

- **t+3h / t+6h persistence** = predict the temperature N hours from now equals the temperature *right now*. Beating it shows the model has learned real thermal evolution (diurnal cycle, advection, cloud/precip effects), not just inertia.
- **Daily-high persistence** = predict today's high equals *yesterday's* high.

A model is only useful insofar as it beats persistence; on the test set all three do, comfortably. Climatology (the seasonal-average high for the date) is reported as a second, weaker baseline for the daily-high model.

## Scoring live accuracy

The bot **auto-posts a daily scorecard** to a `#score` channel every morning at **6:30 AM ET**, grading the previous day's predictions. It runs next-morning rather than at midnight because the last few t+3h/t+6h targets of a day fall after midnight and the official CLI high often finalizes overnight — 6:30 AM is the first moment the day is fully settled. Each card shows the actual high (and its source), the daily-high model's pre-high accuracy, and the t+3h/t+6h hit counts.

The same scoring logic is also available as a standalone CLI, [score_predictions.py](score_predictions.py), for grading arbitrary ranges of `nowcast_log.csv` after the fact:

```
python score_predictions.py                          # scores nowcast_log.csv
python score_predictions.py --start 2026-07-20 --end 2026-08-03 --out score_out
```

Both share the same rules:

- **Daily high** — scored only on predictions issued *before* the high actually occurred (afterward the obs-floor makes it trivially correct), using the official CLI high and its time. Reports MAE, hit-rates, a 0–100 skill score, and the issue-hours / lead-times where the model is most accurate.
- **t+3h / t+6h** — scored out of 24 per NY-local day: 1 point per hourly prediction within ±1 °F of the actual temperature at the target time, plus MAE and best/worst hours.

## Setup

```
pip install "discord.py>=2.0" python-dotenv requests pandas numpy xgboost joblib
```

Create a `.env` file next to `knyc_discord_bot.py`:

```
DISCORD_TOKEN=...
GUILD_ID=...
```

### Rebuild training data and retrain

```
python build_training_data.py --start 2010-01-01
```

This writes `knyc_training.csv` (gitignored). Then run [KNYC_Nowcaster.ipynb](KNYC_Nowcaster.ipynb) end-to-end to retrain and regenerate the three `.pkl` models, `feature_manifest.json`, and `tmpf_climatology.csv`. The bot loads all model artifacts from the repo root and reads the feature list / upstream stations from the manifest at startup, so a retrain with a changed feature set is picked up automatically.

### Run the bot

```
python knyc_discord_bot.py
```

Posts one nowcast on startup, then every hour at **:55 UTC** to `#predictions`. KNYC's METAR typically lands ~:51–:53 via the aviationweather.gov feed; if a fresh ob hasn't arrived by post time, the bot retries every 60 s until a newer METAR appears (capped at 50 min so a dead feed can't stall the next hourly tick). It also posts the daily scorecard to `#score` at 6:30 AM ET (see [Scoring](#scoring-live-accuracy)). Both channels must exist in the guild.
