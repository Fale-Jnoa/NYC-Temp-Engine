# KNYC Temp Engine

A machine-learning nowcasting system for Central Park, NYC (KNYC). It trains XGBoost models on KNYC ASOS observations plus upstream-station signal to predict short-term temperature and the day's high, then posts hourly forecasts to Discord. A separate scorer grades the bot's live predictions against settled ground truth.

## What it predicts

| Model | Target | Test MAE | vs. persistence |
|---|---|---|---|
| `knyc_model_daily_high.pkl` | NWS-settled daily high temperature | **1.98 °F** | +3.71 °F better |
| `knyc_model_t3h.pkl` | Temperature 3 hours ahead | **1.30 °F** | +1.39 °F better |
| `knyc_model_t6h.pkl` | Temperature 6 hours ahead | **2.05 °F** | +2.72 °F better |

Trained on 2010–2021, validated on 2022–2023, tested on 2024 → present (~18k held-out hours). The daily-high model's warm bias is effectively eliminated (+0.05 °F on the test set). "Persistence" is the naive baseline of assuming no change (see [below](#baselines)). Exact feature list and metrics live in [feature_manifest.json](models/feature_manifest.json).

## How it works

1. **[build_training_data.py](knyc/build_training_data.py)** pulls historical KNYC ASOS observations from the IEM archive, along with 8 upstream stations (KEWR, KLGA, KJFK, KTEB, KPHL, KBWI, KDCA, KBOS) for warm/cold-advection signal, and NWS Daily Climate Report (CLINYC) bulletins for the true settled daily high. It engineers features — wind u/v components, cloud oktas, solar zenith/altitude, pressure and temperature tendencies/lags/rolling stats, climatology anomaly, and per-station upstream deltas — and writes `data/knyc_training.csv` (untracked; regenerate locally).
2. **[KNYC_Nowcaster.ipynb](notebooks/KNYC_Nowcaster.ipynb)** trains the three XGBoost models on that data with time-aware splits and sample weighting toward recent years, evaluates them on the held-out test split against naive baselines, and exports the trained `.pkl` models plus [feature_manifest.json](models/feature_manifest.json) (the exact feature list, upstream stations, and metrics the bot relies on) and [tmpf_climatology.csv](models/tmpf_climatology.csv).
3. **[knyc_discord_bot.py](knyc_discord_bot.py)** runs continuously: each hour it fetches the latest KNYC + upstream obs, re-engineers the same features used in training, runs all three models, and posts an embed to a `#predictions` Discord channel with current conditions, t+3h/t+6h forecasts, model daily high, observed high so far, and a floor-locked "reassessed" high (the prediction can't go below what's already been observed, and after 7 PM, once the temperature has dropped 1 °F or more below the observed high, it *is* the observed high). Every posted prediction is logged once to `logs/nowcast_log.csv`.
4. **[score_predictions.py](knyc/score_predictions.py)** grades the logged predictions after the fact against settled ground truth (see [Scoring](#scoring-live-accuracy)).
5. **[kalshi_market.py](knyc/kalshi_market.py)** fetches the live KXHIGHNY bracket ladder and prices, and **[market_scorer.py](knyc/market_scorer.py)** grades the daily-high model on trading ROI rather than degrees (see [Scoring against the market](#scoring-against-the-market)).

## Data sources

- **aviationweather.gov** (FAA/NWS live METAR API) — the bot's real-time obs feed. A given hour's METAR is queryable within ~2 minutes of issuance, versus the IEM archive's inconsistent ingest lag.
- **IEM ASOS archive** (`mesonet.agron.iastate.edu`) — settled historical hourly obs, used for training and as the scorer's ground-truth temperatures.
- **NWS CLI bulletins** (CLINYC) — official settled daily high/low. The training target for the daily-high model (with a METAR 6-hr-max fallback), and the scorer's authoritative daily high + the time it occurred.
- **ASOS DSM** (DSMNYC) — the raw ASOS daily summary max, used by the scorer as an independent cross-check on the CLI high.
- [tmpf_climatology.csv](models/tmpf_climatology.csv) — per (month, hour) mean temperature baseline used to compute the temperature-anomaly feature.

### Baselines

The notebook reports each model against a **persistence** baseline — the naive "nothing changes" forecast:

- **t+3h / t+6h persistence** = predict the temperature N hours from now equals the temperature *right now*. Beating it shows the model has learned real thermal evolution (diurnal cycle, advection, cloud/precip effects), not just inertia.
- **Daily-high persistence** = predict today's high equals *yesterday's* high.

A model is only useful insofar as it beats persistence; on the test set all three do, comfortably. Climatology (the seasonal-average high for the date) is reported as a second, weaker baseline for the daily-high model.

## Scoring live accuracy

The bot **auto-posts a daily scorecard** to a `#score` channel every morning at **6:30 AM ET**, grading the previous day's predictions. It runs next-morning rather than at midnight because the last few t+3h/t+6h targets of a day fall after midnight and the official CLI high often finalizes overnight — 6:30 AM is the first moment the day is fully settled. Each card shows the actual high (and its source), the daily-high model's pre-high accuracy, and the t+3h/t+6h hit counts.

The same scoring logic is also available as a standalone CLI, [score_predictions.py](knyc/score_predictions.py), for grading arbitrary ranges of `nowcast_log.csv` after the fact:

```
python -m knyc.score_predictions                          # scores logs/nowcast_log.csv
python -m knyc.score_predictions --start 2026-07-20 --end 2026-08-03 --out score_out
```

Both share the same rules:

- **Daily high** — scored only on predictions issued *before* the high actually occurred (afterward the obs-floor makes it trivially correct), using the official CLI high and its time. Reports MAE, hit-rates, a 0–100 ±3 °F skill score, and the issue-hours / lead-times where the model is most accurate.
- **t+3h / t+6h** — scored out of 24 per NY-local day: 1 point per hourly prediction within ±1 °F of the actual temperature at the target time, plus MAE and best/worst hours.

## Scoring against the market

The metrics above measure **degrees**. Kalshi's [KXHIGHNY](https://kalshi.com/markets/kxhighny/highest-temperature-in-nyc) contracts pay on **discrete brackets**, and the two disagree structurally: a 0.4 °F miss that straddles a bracket boundary is a total loss, while a 1.8 °F miss inside a bracket is a clean win. A distance metric cannot tell those apart, so a correct contrarian call at 10:1 odds can score mediocre.

[market_scorer.py](knyc/market_scorer.py) grades the daily-high model on money instead:

```
python -m knyc.build_residuals --training data/knyc_training.csv   # once, + after retraining
python -m knyc.market_scorer                                  # needs logged prices
python -m knyc.market_scorer --sweep                          # kelly x no-trade-band grid
```

It reports four things:

1. **Rebalanced ROI** — holds a position, re-prices hourly, buys at the ask and sells at the bid, and pays Kalshi's fee. The only accounting that *charges* for the model changing its mind mid-day.
2. **Flat ROI, net of fees**, next to a **market-favourite baseline** (buy the highest-priced bracket every hour). A model that can't beat "buy the favourite" has no edge, whatever its own ROI says.
3. **Model vs favourite** — the hours where the model's top bracket differs from the market's, and who was right. Any real edge has to show up here.
4. **Brier score, model vs market**, overall and by hour block — a proper scoring rule over the ladder that converges long before ROI does.

Confidence intervals resample whole days, because every hour of a day shares one settlement.

### Live results so far (2026-09-24 → 2026-11-15, 53 settled days)

The model has **no edge against the market** at any hour. The market favourite hit 72% of the time; the model's own bracket 33%. When they disagreed the favourite was right 69% of the time and the model 13%. Ladder Brier: model 0.55 (0.70 before the probability fix) vs market 0.27. Kalshi is treated as a **scoreboard, not a trading signal**. Closing the gap would need forecast-guidance inputs (NBM/MOS/HRRR) in a retrain. Full analysis: [docs/REVIEW_2026-11.md](docs/REVIEW_2026-11.md).

### Bracket probabilities

A point forecast can't say *how sure* it is, so it can't tell a steal from a trap at the same price. [build_residuals.py](knyc/build_residuals.py) replays the daily-high model over out-of-sample history and writes [offset_pmf.csv](models/offset_pmf.csv): the distribution of `settlement − round(max(pred, observed high so far))`, by NY-local hour **and by how far the model sits above the observed high**. Conditioning on the observed high is what makes the evening probabilities honest; the older residual quantiles ([residual_model.csv](models/residual_model.csv), still written for reference) ignored it and kept putting mass above highs that were already in. Two things matter:

- **Out-of-sample only.** Residuals are measured strictly after `feature_manifest.json:train_through`; in-sample residuals look artificially tight and would cause systematic overbetting.
- **Uncertainty is strongly hour-dependent.** The model's 80% interval spans about **11 °F at midnight** and **3 °F by late afternoon**. Against 2 °F-wide brackets that means an overnight call is spread across ~5 brackets, while a 4 PM call resolves to ~1.5. Early-morning conviction is mostly luck, and Kelly sizing scales positions down accordingly.

### Data capture

Kalshi's public market API needs no authentication. Each hour the bot appends the full bracket ladder with bid/ask to `logs/market_log.csv`, and one row per prediction to `logs/prediction_tape.csv`:

| what the model said | which bracket | what it cost | what settled |
|---|---|---|---|
| `88.6` → `89` | `89-90` | ask `0.24` | `88` → miss |

Since **2026-11-16** the tape's bracket is the one containing the **reassessed** high (which respects the observed floor); before that it used the raw model value, which hit 9% of the time after 7 PM. Hit rates and returns are **not comparable across that cut-over**.

Predicted temperatures round **half-up** (88.5 → 89); Python's builtin `round()` is banker's rounding and would send identical-magnitude predictions to different brackets depending on parity. `python -m knyc.prediction_tape --backfill` stamps the settled high onto past rows, and the 6:30 AM scorecard job does it automatically.

> **Historical prices cannot be recovered.** Once an event settles the order book is gone — a finalized day returns empty books. Any day the bot is not logging is permanently unscoreable for ROI, which is why capture runs independently of the scorer being finished. Expect ~3 weeks before ROI figures mean anything.

Settlement uses the CLI high **alone** (`settle_high`), not the CLI/DSM blend that `actual_high` uses, because that is the exact value the exchange settles on.

## Repository layout

```
knyc_discord_bot.py     live bot (entry point; systemd runs this)
knyc/                   library + command-line tools (run as python -m knyc.<tool>)
  paths.py              where every file lives
  score_predictions.py  ground-truth fetchers (CLI/DSM/IEM) and accuracy scoring
  kalshi_market.py      KXHIGHNY bracket ladder + prices
  prediction_tape.py    one row per prediction: bracket, price, settlement
  market_scorer.py      model vs market: ROI, favourite baseline, Brier
  daily_chart.py        the daily scorecard chart
  build_training_data.py  training-set builder (IEM + CLI)
  build_residuals.py    out-of-sample offset PMF for bracket probabilities
models/                 trained models, feature manifest, climatology, offset PMF
notebooks/              training notebook (+ local log-analysis notebook)
tests/                  offline regression tests
tools/                  feature-parity check (bot vs training)
docs/                   deployment guide and the 2026-11 production review
data/                   training CSV, pulled server copies       (gitignored)
logs/                   everything the bot writes: logs, tape, state (gitignored)
```

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
python -m knyc.build_training_data --start 2010-01-01
```

This writes `data/knyc_training.csv` (gitignored). Then run [KNYC_Nowcaster.ipynb](notebooks/KNYC_Nowcaster.ipynb) end-to-end to retrain and regenerate the three `.pkl` models, `feature_manifest.json`, and `tmpf_climatology.csv` in `models/`. The bot loads all model artifacts from `models/` and reads the feature list / upstream stations from the manifest at startup, so a retrain with a changed feature set is picked up automatically.

### Run the bot

```
python knyc_discord_bot.py
```

Posts one nowcast on startup, then every hour at **:55 UTC** to `#predictions`. KNYC's METAR typically lands ~:51–:53 via the aviationweather.gov feed; if a fresh ob hasn't arrived by post time, the bot retries every 60 s until a newer METAR appears (capped at 50 min so a dead feed can't stall the next hourly tick). Once KNYC is in, it also waits up to **8 minutes** for upstream stations that report later (KPHL, KBWI, KBOS issue at :54) so the model sees this hour's inputs; if one is still missing it posts anyway and says so in the footer. It also posts the daily scorecard to `#score` at 6:30 AM ET (see [Scoring](#scoring-live-accuracy)), and on startup posts any scorecard a restart caused it to miss (tracked in `logs/bot_state.json`). Both channels must exist in the guild.

### Side-by-side trial: guidance_v2

Since 2026-11-16 each post also shows **🧪 v2 High (trial)**: a candidate model that adds NWS forecast guidance (GFS MOS, LAMP and the National Blend of Models, fetched hourly from IEM) to the same observations. The current model stays official for the nowcast log, the Kalshi tape and the scorecard; v2 is logged separately to `logs/shadow_log.csv`. If no guidance can be fetched, v2 repeats the current model's number and says so. Set `KNYC_SHADOW_CANDIDATE=off` in `.env` to turn it off. Compare the two with `python -m knyc.shadow_report`. Background and gate results: [docs/PLAN_GUIDANCE_V2.md](docs/PLAN_GUIDANCE_V2.md), [models/candidates/guidance_v2/GATE_REPORT.md](models/candidates/guidance_v2/GATE_REPORT.md).

### Checks

```
python tests/test_live_fixes.py           # offline regression tests (no network)
python tools/check_feature_parity.py      # bot vs training features on live data
```

Run the parity check after any change to feature code or a pandas/numpy upgrade. The daily-high model only works if the bot computes features exactly the way training did, and three silent mismatches ran in production for two months before the 2026-11 review caught them: solar geometry pinned to January 1970 by pandas 3's microsecond datetimes, clear sky never parsed from the live feed, and three upstream stations an hour stale.

**Climate day:** the NWS climate report that Kalshi settles on runs midnight-to-midnight *local standard time* all year, i.e. 1 AM–1 AM EDT in summer. The bot's observed-high floor and the scorer's CLI/DSM times use that day, not the calendar day.
