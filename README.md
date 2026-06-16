# KNYC Temp Engine

A machine-learning nowcasting system for Central Park, NYC (KNYC). It trains XGBoost models on KNYC ASOS observations plus upstream-station signal to predict short-term temperature and the day's high, then posts hourly forecasts to Discord.

## What it predicts

| Model | Target | Test MAE |
|---|---|---|
| `knyc_model_daily_high.pkl` | NWS-settled daily high temperature | **2.05 °F** |
| `knyc_model_t3h.pkl` | Temperature 3 hours ahead | **1.30 °F** |
| `knyc_model_t6h.pkl` | Temperature 6 hours ahead | **2.07 °F** |

Metrics are computed on a held-out test period after training through 2021-12-31, as recorded in [feature_manifest.json](feature_manifest.json).

## How it works

1. **[build_training_data.py](build_training_data.py)** pulls historical KNYC ASOS observations from IEM, along with upstream-station obs (KEWR, KLGA, KJFK, KTEB, KPHL, KBWI, KDCA, KBOS) for warm/cold-advection signal, and NWS Daily Climate Report (CLINYC) bulletins for the true settled daily high. It engineers features — wind u/v components, cloud oktas, solar zenith/altitude, pressure and temperature tendencies/lags/rolling stats, climatology anomaly — and writes `knyc_training.csv`.
2. **[KNYC_Nowcaster.ipynb](KNYC_Nowcaster.ipynb)** trains the three XGBoost models on that data, evaluates them on the held-out test split, and exports the trained models plus [feature_manifest.json](feature_manifest.json) (the exact feature list and metrics the bot relies on).
3. **[knyc_discord_bot.py](knyc_discord_bot.py)** runs continuously: every hour, it fetches the latest KNYC + upstream obs, re-engineers the same features used in training, runs all three models, and posts an embed to a `#predictions` Discord channel with the current conditions, t+3h/t+6h forecasts, model daily high, observed high so far, and a floor-locked "reassessed" high (the model's prediction can't go below what's already been observed). Predictions are logged to [nowcast_log.csv](nowcast_log.csv) for postmortem analysis.

## Data sources

- **IEM ASOS** (`mesonet.agron.iastate.edu`) — hourly METAR observations for KNYC and upstream stations.
- **NWS CLI bulletins** (CLINYC) — official settled daily high/low, used as the ground-truth training target where available, with a METAR-derived fallback.
- [tmpf_climatology.csv](tmpf_climatology.csv) — per (month, hour) mean temperature baseline used to compute a temperature anomaly feature.

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
python build_training_data.py --start 2010-01-01 --end 2026-02-01
```

Then run [KNYC_Nowcaster.ipynb](KNYC_Nowcaster.ipynb) to retrain and regenerate the `.pkl` models and `feature_manifest.json`.

### Run the bot

```
python knyc_discord_bot.py
```

Posts one nowcast on startup, then every hour at :05 UTC (after the :51 METAR has had time to land in IEM), with retry on stale or missing observations.
