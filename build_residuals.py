"""
Residual model builder
======================
Turns the daily-high model's point prediction into a *distribution*, which is
what bracket probabilities require.

The method: replay `knyc_model_daily_high.pkl` over historical hours, collect
`residual = actual_high - pred_high`, and bin those residuals by conditions
known at prediction time. Applying the matching residual distribution around
today's prediction gives P(high lands in any bracket).

Two choices here matter a lot:

1. **Out-of-sample only.** `feature_manifest.json:train_through` is 2021-12-31;
   residuals measured on data the model trained on are optimistically tight.
   Using them would make the model look far more confident than it is and cause
   systematic overbetting. Rows at or before `train_through` are dropped, and
   the script refuses to run if that leaves too little data.

2. **Binned on what is knowable at prediction time** — NY-local hour and month.
   NOT lead time to the high: at 9am you do not know when the high will occur,
   so a lead-time-conditioned spread cannot be evaluated live. (The scorer still
   *reports* by lead time; that is hindsight reporting, which is fine.)

Empirical quantiles rather than a fitted normal: late-day residuals are sharply
skewed because the observed high so far truncates the distribution from below.

Usage
-----
    python build_residuals.py --training knyc_training.csv
    python build_residuals.py --training knyc_training.csv --out residual_model.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "feature_manifest.json"
MODEL_PATH = HERE / "knyc_model_daily_high.pkl"
OUT_PATH = HERE / "residual_model.csv"

# Quantiles of the residual distribution to persist. Dense in the tails because
# that is where bracket probabilities for longshot contracts come from.
QUANTILES = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]

# A bin needs this many samples before its own empirical quantiles are trusted;
# below it, the bin falls back to the pooled all-hours distribution.
MIN_BIN_N = 60

# Month -> coarse season bucket. Splitting 12 months x 24 hours would shred the
# sample count per bin; seasons keep bins populated while still separating the
# regimes that actually differ (summer sea-breeze vs winter advection).
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def compute_residuals(training: Path, manifest: dict) -> pd.DataFrame:
    """Replay the daily-high model over history -> per-hour residuals."""
    feature_cols = manifest["feature_cols"]
    df = pd.read_csv(training)

    needed = set(feature_cols) | {"valid", "target_high"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(
            f"{training.name} is missing {len(missing)} required column(s): "
            f"{sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}"
        )

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df = df.dropna(subset=feature_cols + ["target_high"]).copy()

    # Out-of-sample gate.
    train_through = pd.to_datetime(manifest["train_through"], utc=True)
    n_all = len(df)
    df = df[df["valid"] > train_through].copy()
    print(f"  rows total {n_all:,} -> out-of-sample (after "
          f"{train_through:%Y-%m-%d}) {len(df):,}")
    if len(df) < 2000:
        raise SystemExit(
            f"Only {len(df)} out-of-sample rows. Rebuild the training CSV with a "
            f"range extending well past {train_through:%Y-%m-%d}, e.g.\n"
            f"    python build_training_data.py --start 2022-01-01"
        )

    model = joblib.load(MODEL_PATH)
    df["pred_high"] = model.predict(df[feature_cols])
    df["residual"] = df["target_high"] - df["pred_high"]

    ny = df["valid"].dt.tz_convert("America/New_York")
    df["local_hour"] = ny.dt.hour
    df["season"] = ny.dt.month.map(SEASON)
    return df[["valid", "local_hour", "season", "pred_high", "target_high", "residual"]]


def build_table(res: pd.DataFrame) -> pd.DataFrame:
    """Empirical residual quantiles per (season, local_hour), plus pooled rows.

    Pooled rows are written with season='ALL'/hour=-1 so the lookup always has a
    fallback for a bin that is sparse or absent at scoring time.
    """
    rows = []

    def emit(season: str, hour: int, s: pd.Series) -> None:
        rec = {"season": season, "local_hour": hour, "n": len(s),
               "mean": s.mean(), "std": s.std()}
        rec.update({f"q{int(q * 1000):04d}": s.quantile(q) for q in QUANTILES})
        rows.append(rec)

    emit("ALL", -1, res["residual"])
    for hour, g in res.groupby("local_hour"):
        emit("ALL", int(hour), g["residual"])
    for (season, hour), g in res.groupby(["season", "local_hour"]):
        if len(g) >= MIN_BIN_N:
            emit(str(season), int(hour), g["residual"])

    return pd.DataFrame(rows).sort_values(["season", "local_hour"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the daily-high residual distribution.")
    ap.add_argument("--training", default=str(HERE / "knyc_training.csv"),
                    help="training CSV from build_training_data.py")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    training = Path(args.training)
    if not training.exists():
        raise SystemExit(
            f"{training} not found. Generate it first:\n"
            f"    python build_training_data.py --start 2022-01-01"
        )

    manifest = load_manifest()
    print(f"Replaying {MODEL_PATH.name} over {training.name} ...")
    res = compute_residuals(training, manifest)

    mae = res["residual"].abs().mean()
    bias = res["residual"].mean()
    print(f"  out-of-sample MAE {mae:.2f} F   bias {bias:+.2f} F "
          f"(manifest test MAE {manifest['test_metrics']['daily_high_mae']:.2f})")

    table = build_table(res)
    table.to_csv(args.out, index=False)
    print(f"  wrote {len(table)} bins -> {args.out}")

    # Spread by hour is the headline sanity check: uncertainty must shrink as the
    # day progresses. If it does not, the residuals are not usable for sizing.
    pooled = table[table["season"] == "ALL"].set_index("local_hour")
    print("\n  hour   n      bias    p10     p50     p90    90%-width")
    for h in range(0, 24, 2):
        if h not in pooled.index:
            continue
        r = pooled.loc[h]
        print(f"  {h:>4}  {int(r['n']):>5}  {r['mean']:>+6.2f}  "
              f"{r['q0100']:>+6.2f}  {r['q0500']:>+6.2f}  {r['q0900']:>+6.2f}  "
              f"{r['q0900'] - r['q0100']:>8.2f}")


if __name__ == "__main__":
    main()
