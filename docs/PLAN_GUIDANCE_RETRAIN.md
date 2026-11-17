# Plan: guidance-aware retrain of the KNYC daily-high model (decision D2)

Written 2026-11-16. Planning only: no code, data or model was changed. Source evidence: `docs/REVIEW_2026-11.md`
(F6, F7, D2), the code at commit `9d910fb`, and live requests to IEM made today (section 2).

Naming used throughout:

- `t` / `valid_t`: the hourly KNYC observation time (`HH:51Z`), the row key in training and in the bot.
- `D` (target date): the **NY civil date of `valid_t`**. This is the existing `local_date` / `target_high` key and
  the Kalshi event date the bot posts for.
- Climate window of `D`: `[D 05:00Z, D+1 05:00Z)`, i.e. midnight-to-midnight LST (UTC-5) all year.

---

## 1. Executive summary

**What.** Train a candidate daily-high model, `guidance_v1`, that adds 15 features to the current 91: two
observation features (`obs_high_so_far`, `hours_into_climate_day`), eight from GFS MOS and five from GFS LAMP,
all for KNYC from the IEM MOS archive. Candidates are written to `models/candidates/guidance_v1/`; nothing in
`models/` is touched. This session builds the data pipeline, trains, evaluates and applies a numeric gate. Bot
integration, shadow running and the switch are later phases.

**Why.** The market's implied daily-high forecast is 2.6 to 3.6 times as accurate as the model before noon
(0.86 vs 3.06 F at 00h, 0.77 vs 1.99 F at 09h) and the model's regression weight next to the market is zero.
The model sees no forecast guidance at all.

**Expected gain (measured, not guessed).** Raw guidance scored against the CLI high using the archive pulled
today (2022-01 to 2026-07, all seasons; in brackets the same Sep 24 - Nov 15 window of 2022-2025):

| Raw input, no model | MAE F | Bias F |
|---|---|---|
| GFS MOS 00Z `n_x` (usable from about 00-01h local) | 2.21 (1.80) | -0.34 |
| GFS MOS 06Z `n_x` (usable from about 06-07h local) | 2.10 (1.78) | -0.28 |
| LAMP 06Z, max of hourly `tmp` over the climate day | 2.19 (1.45) | -1.61 |
| NAM MOS 00Z `n_x` | 2.51 (2.62) | -1.23 |
| Current model, hours 00-12, same OOS period (review F6) | 2.66 | about 0 |

A model that combines these with observations and removes the biases should land around **1.6-2.0 F overnight
and 1.2-1.5 F by mid-morning**, against 2.3-3.1 F today. That is a large improvement over production and still
well short of the market's 0.8 F.

**Honest risks.**

1. **Parity with the market is not the expected outcome; an edge even less so.** The market already prices this
   guidance plus better guidance (NBM, HRRR, ECMWF, the NWS forecast itself). MOS-class inputs close perhaps a
   third to a half of the gap. The realistic deliverable is a much better forecast and an honest probability
   ladder, not a trading signal. D1 (scoreboard only) stands regardless of this retrain.
2. **NAM MOS no longer exists.** NCEP retired the NAM and NAM MOS at 12Z on 2026-11-03 (SCN 26-47). The IEM
   archive ends with the 2026-11-03 00Z run and the live API returns 404. NAM cannot be served, so it is
   **excluded from the feature set**, which changes the scope given for D2.
3. **GFS MOS is on a sunset path.** NWS MDL's MOS page says stand-alone MOS is no longer developed and a
   phase-out "in the next couple of years" is being considered, explicitly excluding LAMP. The graceful
   degradation requirement is therefore also the retirement plan, and NBM moves from "optional" to
   "recommended next phase" (section 10).
4. **LAMP history is short.** IEM has LAMP only from 2020-07-11 and keeps only the 00/06/12/18Z cycles after
   about 7 days. The live bot must restrict itself to those four cycles or the features will not match training.
5. **Dissemination delay is not recorded in the archive.** Only the cycle time is stored. Availability is
   modelled with fixed, conservative lags (section 3.1) that the shadow phase must confirm.

---

## 2. Verified data-source findings

All requests used `User-Agent: python-requests/2.31`, 2.5-3 s apart. One 429-style refusal ("Too many requests
from your IP address") occurred when a second process ran in parallel: keep the backfill single-threaded.

### 2.1 Endpoints

| Purpose | Endpoint | Verified behaviour |
|---|---|---|
| Bulk and live (use this one for both) | `GET https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station=KNYC&model=GFS&sts=2026-11-15T00:00Z&ets=2026-11-15T13:00Z&format=csv` | CSV, one row per (run, forecast time). `sts`/`ets` filter on **run time**, both ends inclusive. A whole calendar year in one request works: GFS about 3.2 MB in 3 s, LAV about 5 MB. |
| JSON, one run | `GET https://mesonet.agron.iastate.edu/api/1/mos.json?station=KNYC&model=GFS&runtime=2026-11-15T12:00:00Z` | Same fields plus `runtime_utc`/`ftime_utc`. Without `runtime` it returns the latest run, but that response is cached: at 23:41Z it still served the 22Z LAMP run while the CSV endpoint already had 23Z. Returns 404 when nothing exists in the last 48 h (NAM today). |
| Raw text (not needed, noted as a cross-check) | `.../cgi-bin/afos/retrieve.py?pil=MAVNYC&limit=2&fmt=text` and `pil=LAVNYC` | Returns the bulletins. `METNYC` returns "Could not Find". |

Station code is `KNYC` (four letters required). Model codes accepted: `AVN, GFS, ETA, NAM, NBS, NBE, ECM, LAV, MEX`.

### 2.2 Coverage for KNYC

Counted from full-year downloads of GFS, NAM and LAV, 2010 to today (kept in the session scratchpad, not the repo).

| Model | Archive span | Cycles kept | Rows per run | Gaps |
|---|---|---|---|---|
| **GFS MOS** (`GFS`) | 2004-01 or earlier to today | 00/06/12/18Z | 21: 3-hourly +6..+60 h, then 6-hourly to +72 h | 36 of 24,656 runs missing since 2010 (0.15%), at most 5 in a year, no gap over 24 h |
| **NAM MOS** (`NAM`) | 2004-07 to **2026-11-03 00Z (retired)** | 00/12Z | 21 | 19 of 12,301 missing (0.15%); one 36 h gap 2017-10-25/27 |
| **LAMP** (`LAV`) | **2020-07-11 18Z** to today | Hourly for the last 7 days only; **00/06/12/18Z** before that | 38 hourly leads +1..+38 h (325 runs have 25; 56 runs are 13-row fragments starting at +26 h) | 84 of 9,277 four-cycle runs missing (0.9%); 72 of them in one block **2023-06-07 to 2023-08-24** |
| GFS extended (`MEX`) | between 2020-07 and 2021-01, to today | 00/12Z | not used | - |
| NBM short (`NBS`) | 2018-11 to today | four per day, but the hours changed: 00/07/12/19Z, then 01/07/13/19Z from 2020, then 00/06/12/18Z from mid-2026 | 23 | max/min column is `n_x` until 2020, `txn` from 2021 |
| NBM extended (`NBE`) | between 2020-07 and 2021-01, to today | thinned after 7 days | not used | - |
| `ETA` | 2004 to 2008 only | - | - | - |
| `AVN`, `ECM` | nothing for KNYC on the dates probed (10 July of 2003-2026 for AVN; 2003-2024 for ECM) | - | - | - |

Other facts the parser must handle:

- GFS files contain 4,284 exact duplicate `(runtime, ftime)` rows (no conflicting values): drop duplicates.
- Timestamps are naive UTC strings (`2026-11-15 00:00:00`). `runtime` is the **model cycle**, not the time the
  bulletin was sent. LAMP bulletins are stamped `HH30 UTC`; IEM stores them as `HH:00`.
- `n_x` sits only on 00Z rows (daytime maximum) and 12Z rows (overnight minimum). For target date `D` the
  maximum is the row with `ftime = (D+1) 00:00Z`.
- **Which GFS runs carry today's maximum:** 00Z of `D` (lead 24 h), 06Z of `D` (lead 18 h) and earlier days'
  runs. The **12Z and 18Z runs of `D` do not** (their first `n_x` is tonight's minimum; verified, the 18Z row at
  00Z has `n_x` empty). After about 06-07h local the newest view of today comes only from the 3-hourly `tmp`
  rows, which is why the feature set has both.
- LAMP has no max/min field, only hourly `tmp`.
- Arrival time observed today: the 23Z LAMP run was present on the CSV endpoint at 23:37Z, and the 18Z GFS run
  at 23:37Z (single spot checks; a full cycle-arrival measurement was not possible in this session). Nominal NWS timing is about cycle + 4 h 10 min for GFS MOS and cycle + 35-45 min for LAMP. These
  are not verifiable from the archive.

---

## 3. Feature specification

Fifteen new columns. Every one is a pure function of `(valid_t, hourly KNYC obs up to valid_t, guidance run
table)`, implemented once in `knyc/guidance.py` and called by both the training builder and the bot.

### 3.1 Run eligibility (the no-leakage rule)

```
GFS_AVAIL_LAG = 5 h      LAV_AVAIL_LAG = 1 h      LAV_CYCLES = {0, 6, 12, 18}
a run R is eligible for row t   <=>   R.runtime + LAG <= valid_t
```

- LAMP runs from any other cycle are discarded on load, in training **and** live.
- Each feature takes the **latest eligible run that can supply it** (defined per feature below), subject to a
  maximum age: 36 h for `n_x`, 18 h for GFS `tmp`, 12 h for LAMP. Past that the feature is NaN.
- The same filter runs live, so a run that is already on IEM but younger than the lag is ignored. Parity holds
  by construction unless a run arrives later than its lag; the shadow log records that case (section 8).
- Runs with fewer than 21 rows (GFS) or 25 rows (LAMP), or whose first lead is not +6 h / +1 h, are dropped.

Worked example for a standard-time date `D` (during EDT local hours are one later):

| `valid_t` | Local | GFS run for `n_x` | GFS run for `tmp` | LAMP run |
|---|---|---|---|---|
| D 05:51Z | 00:51 | 00Z D (lead 24) | 00Z D | 00Z D |
| D 10:51Z | 05:51 | 00Z D | 00Z D | 06Z D |
| D 11:51Z | 06:51 | **06Z D** (lead 18), last `n_x` for D | 06Z D | 06Z D |
| D 17:51Z | 12:51 | 06Z D | **12Z D** | 12Z D |
| D 23:51Z | 18:51 | 06Z D | 18Z D | 18Z D |

### 3.2 Observation features

| Name | Definition |
|---|---|
| `hours_into_climate_day` | `(valid_t - D 05:00Z)` in hours. 0.85..23.85 in standard time; -0.15..22.85 in DST (the 00:51 EDT row is before the climate day starts). |
| `obs_high_so_far` | Maximum over rows with `D 05:00Z <= valid <= valid_t` of hourly `tmpf` and of `mxtmpf_6hr` where `valid - 6 h >= D 05:00Z`. NaN when no such row exists (00:51 EDT). Must equal `knyc_discord_bot._observed_high(df[df.valid <= valid_t], D)`. In training: `build_residuals.observed_so_far(df)`, set to NaN where the LST date of `valid` differs from `D`. |

### 3.3 GFS MOS features

| Name | Definition | Run used |
|---|---|---|
| `gfs_nx_today` | `n_x` at `ftime = (D+1) 00:00Z` | latest eligible run where it is non-null |
| `gfs_nx_age_h` | `valid_t - runtime` of that run, hours | same |
| `gfs_nx_rev` | `gfs_nx_today` minus the same field from the next older eligible run | same and its predecessor |
| `gfs_tmp_max_rest` | max of `tmp` over `ftime` with `valid_t < ftime < (D+1) 05:00Z` and `ftime >= D 05:00Z`; NaN if none | latest eligible run |
| `gfs_tmp_age_h` | age of that run, hours | same |
| `gfs_tmp_now_err` | `tmpf(valid_t)` minus `tmp` linearly interpolated to `valid_t` | latest eligible run with rows on both sides of `valid_t` |
| `gfs_tmp_t3`, `gfs_tmp_t6` | `tmp` interpolated to `valid_t + 3 h` and `+ 6 h` | latest eligible run |

### 3.4 LAMP features

| Name | Definition |
|---|---|
| `lav_tmp_max_rest` | max of hourly `tmp` over the same window as `gfs_tmp_max_rest` |
| `lav_tmp_now_err` | `tmpf(valid_t)` minus `tmp` at `ftime = ceil_hour(valid_t)` (9 minutes ahead) |
| `lav_tmp_t3`, `lav_tmp_t6` | `tmp` at `ftime = ceil_hour(valid_t) + 3 h` and `+ 6 h` |
| `lav_age_h` | age of the run used, hours |

All LAMP features use the latest eligible four-cycle run. All five are NaN before 2020-07-12.

### 3.5 Not included, and why

- **NAM MOS:** retired 2026-11-03; cannot be served.
- **NBM:** kept for phase 7. Its cycle hours changed twice and the max/min column was renamed; it needs its own
  mapping work and would give only about 5 training years.
- **Any sub-hourly or non-ASOS observation:** see the architecture boundary in section 9.

---

## 4. Data pipeline

New modules, all under `knyc/`, run as `python -m knyc.<name>`. Add to `knyc/paths.py`:
`GUIDANCE_DIR = DATA_DIR / "guidance"`, `GUIDANCE_FEATURES = GUIDANCE_DIR / "guidance_features.csv"`,
`TRAINING_TAIL_CSV = DATA_DIR / "knyc_training_tail.csv"`, `CANDIDATES_DIR = MODELS_DIR / "candidates"`,
`SHADOW_LOG = LOGS_DIR / "shadow_log.csv"` (do not add `SHADOW_LOG` to `_RUNTIME_FILES`).

| Module | Role |
|---|---|
| `knyc/guidance.py` | Library. `fetch_runs(model, sts, ets)` (CSV endpoint, 4 retries with 15 s back-off on non-200, 3 s pause between calls); `clean_runs(df, model)` (dedupe, type coercion, cycle filter, fragment drop); `obs_day_features(obs_df)`; `guidance_features(rows, gfs_runs, lav_runs)` returning the 13 guidance columns plus `gfs_nx_runtime`, `gfs_tmp_runtime`, `lav_runtime` for logging; constants from 3.1; `GUIDANCE_COLS`, `OBS_DAY_COLS`. No Discord or model imports. |
| `knyc/build_guidance_archive.py` | Backfill. One request per model per year into `data/guidance/raw/{GFS,LAV}_{YYYY}.csv`; skips existing closed years, always refetches the current year. `--start-year 2010`. |
| `knyc/build_guidance_features.py` | Reads `data/knyc_training.csv` plus the tail file, computes all 15 columns for every row, writes `data/guidance/guidance_features.csv` keyed by `valid` (about 150k rows, under 20 MB). Does not rewrite the training CSV. |
| `knyc/train_guidance.py` | Training script (section 5). |
| `knyc/eval_guidance.py` | Evaluation and gate report (section 6). |

**Extending observations to the live period.** `data/knyc_training.csv` ends 2026-07-17 23:51Z, before the
server logs start. Run `python -m knyc.build_training_data --start 2026-07-10 --end 2026-11-18 --out
data/knyc_training_tail.csv` (new file; the existing CSV is not regenerated). Loaders concatenate base rows with
tail rows where `valid > 2026-07-17 23:51Z`; the 8-day overlap lets the 24 h lags and rolls warm up.

**Vectorised selection (suggested).** For `n_x`: build a table `(target_date, runtime, avail = runtime + lag,
n_x)` from 00Z-`ftime` rows and `pd.merge_asof(rows, table, left_on="valid", right_on="avail",
by="target_date")`. For `tmp` features: `merge_asof` each row to the latest eligible run, join that run's rows
(8 for GFS, up to 38 for LAMP) and aggregate. For `*_now_err` do a second as-of pass restricted to runs that
bracket `valid_t`.

**Request budget.** GFS 17 requests (about 54 MB), LAV 7 requests (about 37 MB), tail observations 9 requests,
CLI 1. Under 40 requests, about 3 minutes including pauses. Feature build: a few minutes. Training five
variants: about 10-20 minutes on a laptop.

---

## 5. Training protocol (`knyc/train_guidance.py`, a script, not the notebook)

**Target.** `target_high` unchanged (CLI maximum for `D`).

**Splits.** The notebook's split (train to 2021) would leave only 17 months of LAMP in training, so the
candidate uses later boundaries. The production model was trained to 2021-12-31, so it is out of sample on
every evaluation period below.

| Slice | Period | Use |
|---|---|---|
| Train | 2010-01-01 to 2023-12-31 | fit |
| Validation | 2024-01-01 to 2024-12-31 | early stopping, variant selection |
| Test | 2025-01-01 to 2026-09-23 | gate G1, G4; offset PMF (with validation) |
| Live holdout | 2026-09-24 to 2026-11-16 (53 settled days) | gate G2, G3; never used for fitting, selection or the PMF |

**Rows.** Drop rows with NaN in the 91 production features or the target, as the notebook does. Do **not**
drop on the 15 new columns: XGBoost's missing branch handles them.

**Hyperparameters and weights (reused from the notebook unchanged).** `XGBRegressor(n_estimators=3000,
learning_rate=0.03, max_depth=6, min_child_weight=20, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
early_stopping_rounds=50, objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=-1)`;
`sample_weight = exp(0.10 * (year - 2010))`. Climatology (`tmpf_clim_global`, `tmpf_anomaly`) is recomputed
from the train slice and saved to the candidate directory.

**Era handling: recommendation.** One model over 2010-2023 with LAMP columns NaN before 2020-07 (variant C1
below). Reasons: GFS MOS and the observation features carry most of the signal and have 14 years of history;
a LAMP-only-era model would train on 3.4 years; the recency weight already makes 2023 count 3.7x as much as
2010; and a model that has seen LAMP missing on most rows degrades cleanly when LAMP is down. The risk is that
"LAMP missing" becomes a proxy for "old data", which variant C2 checks.

**Variants.** All use the splits above.

| Id | Features | Purpose |
|---|---|---|
| A0 | the 91 production features | control: effect of new splits and more data alone |
| A1 | A0 + 2 observation features | value of the observation features alone |
| B1 | A1 + 8 GFS features | value of GFS MOS |
| C1 | B1 + 5 LAMP features | **expected candidate** |
| C2 | C1 features, train 2020-08-01 to 2023-12-31 only | era check |

Selection rule, fixed in advance: lowest validation MAE over NY-local hours 0-12; prefer the earlier-listed
variant when within 0.02 F. Test and holdout are read only for the gate.

**Guidance masking (training rows only, seed 42).** Applied per climate day so outages look contiguous:

- 10% of days: all 8 GFS features set to NaN.
- 10% of days, drawn independently: all 5 LAMP features set to NaN (about 1% of days lose both).
- A further 5% of rows: recompute the guidance features with the newest eligible run of each model removed,
  simulating a late bulletin.

Natural missingness comes on top: all LAMP before 2020-07-12, the 2023 summer LAMP gap, 0.15% of GFS runs.

**t+3h and t+6h.** Not part of the gate and not recommended for production yet. They do not affect settlement,
they already validate at 1.27 / 1.85 F after the P0 fixes, and LAMP at a 3 h lead is unlikely to beat an
observation-driven model by much. They are cheap by-products, so the script also fits them on the C1 feature
set and reports test MAE next to production. Carry them forward only if t+6h improves by at least 0.10 F and
t+3h is not worse by more than 0.02 F; otherwise the bot keeps the production t+3h/t+6h models.

**Outputs, all in `models/candidates/guidance_v1/`.** `knyc_model_daily_high.pkl`; optional
`knyc_model_t3h.pkl`, `knyc_model_t6h.pkl`; `feature_manifest.json` (`feature_cols`, `upstream_stations`,
`train_through`, `guidance_cols`, `obs_day_cols`, lag constants, `variant`, `test_metrics`);
`tmpf_climatology.csv`; `offset_pmf.csv`; `variants.json` (validation table for all five); `GATE_REPORT.md`
and `gate.json` from the evaluator.

---

## 6. Evaluation protocol (`knyc/eval_guidance.py`)

All tables are by NY-local hour and by the scorer's blocks (00-05, 06-09, 10-12, 13-15, 16-18, 19-23). Every
confidence interval is a day-clustered bootstrap (resample `local_date`, 5,000 draws, as
`market_scorer.bootstrap_ci`).

**6.1 Test period, candidate vs production.** Score both models on identical rows. The production model uses
`models/feature_manifest.json` columns and `models/tmpf_climatology.csv` exactly as the bot applies it. Report
raw MAE and bias, and MAE of `reassess_high(pred, obs_high_so_far, tmpf, hour)`. Add a reference row: MAE of
`gfs_nx_today` itself on the same rows.

**6.2 Live holdout.** The server logs hold only the production model's features, so the candidate cannot be
replayed from them. Instead:

1. Take the hours present in `data/server_pull/market_log.csv` (first snapshot per `valid_t`, `keep="first"`).
2. Join to the rebuilt feature rows (tail CSV + guidance features) on `valid_t`. Both sit on the `HH:51Z` grid.
3. Predict with the candidate and with the production model **on the same rebuilt rows** ("production replay").
   This is the like-for-like comparison. Also print the as-logged production numbers from `nowcast_log.csv`,
   which include the F1-F3 bugs for most of the period.
4. Truth: `market_scorer.settlement_truth` (CLI only), 53 settled days.

**6.3 Market comparison on the holdout.**

- Market point forecast per ladder-hour: `p_i = mid_i / sum(mid)`; contract centre is `(lo + hi) / 2` for
  `between`, `hi - 0.5` for the lower tail, `lo + 0.5` for the upper tail. First reproduce the review's table
  (0.86 / 0.81 / 0.83 / 0.77 / 0.67 / 0.52 / 0.56 / 0.54 at hours 0/3/6/9/12/15/18/21) to within 0.05 F; if it
  does not reproduce, adjust the tail convention and record what was used.
- Ladder Brier: build the candidate's offset PMF by passing its validation + test predictions through
  `build_residuals.build_offset_pmf` (import it; do not edit that module), then call
  `market_scorer.brier_scores` and `model_vs_favourite` on a frame from `load_joined` with `pred_high` and
  `obs_high` replaced by the candidate's values. The market figure must come out near 0.278. Do the same for
  the production replay with a PMF rebuilt from production predictions over the same validation + test period,
  so neither model gets a PMF fitted on more favourable data.
- Regression `actual ~ market + candidate` for 00-06h, 06-10h and 13-16h, as in the review.

**6.4 Degradation.** Re-score the test period three times with GFS features forced to NaN, LAMP forced to NaN,
and both.

### Gate thresholds

| # | Check | Pass when |
|---|---|---|
| G1 | Test, candidate vs production, raw MAE | at least **20% lower** in 00-05 and in 06-09, at least **10% lower** in 10-12, and not more than 0.03 F worse in 13-15, 16-18, 19-23 |
| G1b | Test, candidate vs raw `gfs_nx_today`, hours 0-9 | candidate MAE at least **0.15 F lower** (it must beat its own input) |
| G2 | Holdout, hours 0-11 pooled, candidate vs production replay | MAE at least **0.40 F lower**, 95% CI of the difference entirely below 0, and candidate MAE at hours 0-9 no higher than **2.00 F** |
| G3 | Holdout ladder Brier | overall at most **0.48** and at least 0.05 below the production replay; 00-05 at most 0.74; 06-09 at most 0.70; 19-23 at most 0.20 |
| G4 | Test, both guidance sources masked | candidate MAE within **0.15 F** of production in every block |
| G5 | Leakage and parity tests (section 7) | all pass |

Reported but not gating: market MAE gap by hour, model-vs-favourite hit rates on disagreement hours,
regression weights, calibration table.

**Outcomes.**

- **GO** (G1-G5 all pass): proceed to phase 4.
- **ITERATE** (G1, G4, G5 pass; G2 or G3 fails): do not integrate; go to the NBM phase first and re-gate.
- **STOP** (G1 or G1b fails): write up the result; the guidance features do not earn their operational cost.
- **Market-parity tier** (informational, not expected): candidate MAE within 0.25 F of the market at hours 0-9
  and Brier at most 0.33. Only this would justify reopening D1.

---

## 7. Parity and test additions

**New `tests/test_guidance.py` (offline, synthetic run tables).**

1. *No leakage:* for 2,000 random rows, every `*_runtime` returned satisfies `runtime + lag <= valid_t`.
2. *Future invariance:* deleting every run with `runtime + lag > valid_t` leaves that row's features unchanged.
3. *Selection table:* the five rows of the worked example in 3.1, plus one EDT date.
4. *`n_x` mapping:* a 12Z run of `D` never supplies `gfs_nx_today` for `D`.
5. *LAMP cycle filter:* an 11Z run is ignored even when it is the newest.
6. *Max age:* features go NaN past 36 / 18 / 12 h.
7. *Observation features:* `obs_day_features` equals `bot._observed_high` row by row on a frame crossing a DST
   date and a standard-time date, including a 6-hour group whose window straddles 05:00Z (must be ignored).
8. *All guidance missing:* `guidance_features` returns 13 NaN columns and the candidate model still predicts.

**Extend `tools/check_feature_parity.py`.**

- Part A: add `obs_high_so_far` and `hours_into_climate_day`, comparing the training path with the bot path
  over the same 4 days, tolerance 1e-6.
- New Part C (guidance): fetch the last 4 days of GFS and LAMP once; compute features (i) through the archive
  path used by `build_guidance_features` and (ii) through the bot's live entry point called with each
  historical `valid_t`. All 13 columns and the three `*_runtime` columns must match exactly.
- New Part D, `--shadow-log PATH` (phase 5 onward): rebuild guidance features from the archive for every logged
  hour and compare with what the bot logged. Mismatches mean a run arrived later than its lag. Pass at 99% of
  rows matching.

---

## 8. Bot integration and shadow run (later phases)

**Live fetch.** In `compute_nowcast`, after `engineer_features`: two requests to the CSV endpoint (same as the
backfill) with `sts = now - 40 h`, `ets = now + 1 h`, for `GFS` and `LAV`. Cache the result in memory for 10
minutes, because `compute_nowcast` re-runs every 60 s while waiting for late stations and IEM rate-limits. Any
fetch error yields an empty run table, hence NaN features: the post is never blocked.

**Shadow mode.** Enabled by `KNYC_SHADOW_CANDIDATE=guidance_v1` in `.env`. The bot then loads the candidate
directory next to the production models, predicts with both, and **posts and logs production exactly as today**.
`nowcast_log.csv`, `market_log.csv` and `prediction_tape.csv` get no new columns.

Candidate output goes to a new append-only file `logs/shadow_log.csv`, one row per posted hour, written through
the existing `_append_log_row` generalised to take a path. That keeps the header-aligned behaviour: on schema
drift, columns missing from the row are written empty, extra columns are dropped with a warning, and the file is
never rewritten. Columns: `valid_t`, `candidate`, `pred_high_prod`, `pred_high_cand`, `obs_high`,
`reassessed_prod`, `reassessed_cand`, the 15 new feature values, `gfs_nx_runtime`, `gfs_tmp_runtime`,
`lav_runtime`, `gfs_newest_seen`, `lav_newest_seen` (newest run on IEM at fetch time, eligible or not; this is
the dissemination-delay measurement), `guidance_fetch_ok`. The whole shadow path sits in one `try/except` that
logs and continues.

**Switch criteria.** Evaluate after **30 settled days** (extend to 45 if criterion 1 is undecided; abandon at
60). All must hold:

1. Hours 0-12: candidate MAE at least 0.40 F below production, day-clustered 95% CI excluding 0.
2. No hour block worse than production by more than 0.05 F (reassessed values).
3. Ladder Brier at least 0.05 below production overall.
4. Guidance present in at least 97% of posted hours, and Part D parity at least 99%.
5. Zero posts delayed or lost because of the shadow path.

**Switch mechanics.** Production files in `models/` stay where they are as the rollback. `paths.py` gains an
`ACTIVE_MODEL_DIR` resolved from an env var (default `models/`); switching means pointing it at the candidate
directory and restarting. The tape and scorecard cut-over date is noted in the README, as was done for D3.

---

## 9. Architecture boundary: the high-frequency station layer

Not part of this retrain. The later layer (PWS, NYC Micronet, KNYC 1-minute data) combines with this model as
`final_high = max(already_reached, still_to_come)`.

- **Interface point:** the `obs_high` argument of `reassess_high(pred_high, obs_high, cur_temp, local_hour)`
  and the floor in `market_scorer.bracket_probs`. The station layer supplies a better `already_reached`; this
  model supplies `still_to_come`.
- **Rule for this retrain:** `obs_high_so_far` as a model *feature* is computed only from hourly KNYC METARs
  and their 6-hour groups, in training and live. The station layer must not write into that feature, or the
  model would see inputs it was never trained on.
- `gfs_tmp_max_rest` and `lav_tmp_max_rest` are already "still to come" quantities, which keeps a later change
  of target (remaining maximum instead of full-day maximum) cheap.

---

## 10. Phases and tasks

**Do now (this session): phases 0-3.**

Phase 0 - setup
1. Add the five paths to `knyc/paths.py`. Create `data/guidance/raw/` and `models/candidates/guidance_v1/`.
2. Write `knyc/guidance.py` (fetch, clean, eligibility, both feature functions).
3. Write `tests/test_guidance.py` items 1-8 and get them passing before any real data is used.

Phase 1 - data
4. `python -m knyc.build_guidance_archive --start-year 2010` (GFS and LAV only).
5. Build `data/knyc_training_tail.csv` (2026-07-10 to 2026-11-18).
6. `python -m knyc.build_guidance_features`. Sanity checks to print: `gfs_nx_today` non-null on at least 99% of
   rows; LAMP columns non-null on at least 95% of rows after 2020-07-12 outside the 2023 gap; MAE of
   `gfs_nx_today` vs `target_high` for 2022 onward between 2.0 and 2.4 F (section 1 measured 2.10-2.21);
   `obs_high_so_far <= target_high + 1` on at least 99% of rows.

Phase 2 - train
7. `python -m knyc.train_guidance`: five variants, selection by the fixed rule, artifacts written to the
   candidate directory, t+3h/t+6h by-products reported.
8. Build the candidate offset PMF.

Phase 3 - evaluate and gate
9. `python -m knyc.eval_guidance`: sections 6.1-6.4, write `GATE_REPORT.md` and `gate.json`.
10. Extend `tools/check_feature_parity.py` Part A; run it and the full test suite.
11. Report the outcome (GO / ITERATE / STOP) with the tables. **Stop here and wait for the user.**

**Later (separate sessions, after a GO and user approval).**

Phase 4 - bot integration: live fetch with cache, shadow mode behind the env var, `shadow_log.csv`, parity
Part C, tests for "guidance fetch fails, post still goes out". Deploy.

Phase 5 - shadow run: 30-45 settled days. Weekly: Part D parity and the dissemination-delay table from
`*_newest_seen`. If GFS reliably arrives by cycle + 4.5 h, consider tightening `GFS_AVAIL_LAG` (requires
rebuilding features and retraining).

Phase 6 - switch: apply the criteria in section 8; flip `ACTIVE_MODEL_DIR`; rebuild `offset_pmf.csv` for the
active model; README note.

Phase 7 - NBM (recommended, and the first step if the gate says ITERATE): map NBS `n_x`/`txn` and the three
cycle-hour regimes onto the same feature definitions, add `nbs_*` features, retrain as `guidance_v2`, re-gate.
This is also the replacement path if GFS MOS is retired.

---

## 11. Open questions for the user

1. **NAM is gone.** The brief asked for NAM MOS; it was retired on 2026-11-03 and is excluded here. Confirm
   that is acceptable, or say if an offline-only NAM ablation is wanted for curiosity.
2. **GFS MOS sunset.** Given NWS's stated direction, should NBM (phase 7) be done before the shadow run rather
   than after, accepting a later start? The plan defaults to shadow-running `guidance_v1` first.
3. **Split change.** The candidate trains through 2023 instead of 2021 so that LAMP has history. Confirm, or
   ask for the notebook's split with GFS-only features.
4. **Availability lags.** 5 h for GFS MOS is deliberately conservative (nominal arrival is about 4 h 10 min),
   which delays the 06Z run's use from about 05:10 to 06:00 EST. Keep it until the shadow log has measured
   real arrival times?
5. **Target date at local hour 23.** `compute_nowcast` uses `datetime.now(NY_TZ).date()` for `today_local`, so
   a 23:51 observation posted after midnight is attributed to the next day. The candidate features use the
   civil date of `valid_t`. Fix the bot to match in phase 4, or leave production untouched and handle it only
   on the candidate path?
6. **Better guidance.** Reaching the market's 0.8 F almost certainly needs NBM plus HRRR or the NWS point
   forecast. Is a larger data project (NOMADS / AWS archives rather than IEM text products) in scope after
   phase 7, or does the project stop at "honest, clearly better than today"?
7. **Shadow length.** 30 settled days minimum is proposed. The live comparison in the review used 53; say if
   you want the longer window before any switch.
