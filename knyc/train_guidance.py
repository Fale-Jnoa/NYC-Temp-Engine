"""
Train the guidance-aware daily-high candidate (docs/PLAN_GUIDANCE_RETRAIN.md
section 5). Five variants, selection by a rule fixed in advance; the winner's
artifacts go to models/candidates/guidance_v1/. Nothing in models/ itself is
touched.

    python -m knyc.train_guidance
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from knyc import guidance as g
from knyc import paths
from knyc.build_guidance_features import load_rows
from knyc.build_residuals import build_offset_pmf, observed_so_far

CANDIDATE = "guidance_v1"
OUT_DIR = paths.CANDIDATES_DIR / CANDIDATE

TRAIN_END = date(2023, 12, 31)
VAL = (date(2024, 1, 1), date(2024, 12, 31))
TEST = (date(2025, 1, 1), date(2026, 9, 23))
HOLDOUT_START = date(2026, 9, 24)
C2_START = date(2020, 8, 1)
SEL_HOURS = range(0, 13)
SEL_TOL = 0.02
SEED = 42
MASK_GFS_DAYS, MASK_LAV_DAYS, MASK_LATE_ROWS = 0.10, 0.10, 0.05

XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=20,
                  subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                  early_stopping_rounds=50, objective="reg:squarederror",
                  tree_method="hist", random_state=SEED, n_jobs=-1)


def production_manifest() -> dict:
    return json.loads(paths.FEATURE_MANIFEST.read_text())


def variant_features(prod_cols: list[str]) -> dict[str, list[str]]:
    a1 = prod_cols + g.OBS_DAY_COLS
    b1 = a1 + g.GFS_COLS
    c1 = b1 + g.LAV_COLS
    return {"A0": prod_cols, "A1": a1, "B1": b1, "C1": c1, "C2": c1}


# ── Data ────────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """All rows (base + tail) with the guidance features, running observed
    high for the offset PMF, and split labels. Rows lacking a production
    feature or the target are kept here and dropped by callers."""
    df = load_rows()
    df["obs_so_far"] = observed_so_far(df)          # as build_residuals does (LST day)
    feats = pd.read_csv(paths.GUIDANCE_FEATURES, low_memory=False)
    feats["valid"] = pd.to_datetime(feats["valid"], utc=True)
    df = df.merge(feats, on="valid", how="left", validate="one_to_one")
    ny = df["valid"].dt.tz_convert("America/New_York")
    df["local_date"] = ny.dt.date
    df["local_hour"] = ny.dt.hour
    df["lst_matches_civil"] = ny.dt.date == df["valid"].dt.tz_convert(g.CLIMATE_TZ).dt.date
    d = df["local_date"]
    df["split"] = np.select(
        [d <= TRAIN_END, (d >= VAL[0]) & (d <= VAL[1]), (d >= TEST[0]) & (d <= TEST[1]),
         d >= HOLDOUT_START], ["train", "val", "test", "holdout"], default="none")
    return df


def climatology(train: pd.DataFrame) -> pd.Series:
    return train.groupby(["month", "hour"])["tmpf"].mean().rename("tmpf_clim")


def apply_climatology(df: pd.DataFrame, clim: pd.Series) -> pd.DataFrame:
    """tmpf_clim_global / tmpf_anomaly from a (month, hour) table, as the
    notebook and the bot do (fallback: the row's own tmpf)."""
    out = df.copy()
    key = pd.MultiIndex.from_arrays([out["month"].astype(int), out["hour"].astype(int)])
    out["tmpf_clim_global"] = pd.Series(key.map(clim), index=out.index).astype(float).fillna(out["tmpf"])
    out["tmpf_anomaly"] = out["tmpf"] - out["tmpf_clim_global"]
    return out


def production_climatology() -> pd.Series:
    c = pd.read_csv(paths.CLIMATOLOGY)
    return c.set_index(["month", "hour"])["tmpf_clim"]


def mask_guidance(train: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Plan 5: per climate day drop GFS (10%) / LAMP (10%) independently; on
    a further 5% of rows use the late-bulletin (newest run missing) values."""
    rng = np.random.default_rng(seed)
    out = train.copy()
    days = np.array(sorted(out["local_date"].unique()))
    gfs_days = set(days[rng.random(len(days)) < MASK_GFS_DAYS])
    lav_days = set(days[rng.random(len(days)) < MASK_LAV_DAYS])
    late = rng.random(len(out)) < MASK_LATE_ROWS
    for c in g.GUIDANCE_COLS:
        out.loc[late, c] = out.loc[late, f"late_{c}"]
    out.loc[out["local_date"].isin(gfs_days), g.GFS_COLS] = np.nan
    out.loc[out["local_date"].isin(lav_days), g.LAV_COLS] = np.nan
    return out


def weights(train: pd.DataFrame) -> np.ndarray:
    year = train["valid"].dt.year
    return np.exp(0.10 * (year - year.min())).to_numpy()


def fit(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], target: str) -> XGBRegressor:
    m = XGBRegressor(**XGB_PARAMS)
    m.fit(train[cols], train[target], sample_weight=weights(train),
          eval_set=[(val[cols], val[target])], verbose=False)
    return m


def mae(a, b) -> float:
    return float(np.nanmean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A0,A1,B1,C1,C2")
    args = ap.parse_args()

    prod = production_manifest()
    prod_cols = prod["feature_cols"]
    vf = variant_features(prod_cols)

    df = load_dataset()
    usable = df.dropna(subset=prod_cols + ["target_high"])
    print(f"rows {len(df):,}, usable {len(usable):,}; "
          + ", ".join(f"{s} {n:,}" for s, n in usable["split"].value_counts().items()))

    results, models, clims = {}, {}, {}
    for v in args.variants.split(","):
        cols = vf[v]
        train = usable[usable["split"] == "train"]
        if v == "C2":
            train = train[train["local_date"] >= C2_START]
        clim = climatology(train)
        train = apply_climatology(mask_guidance(train), clim)
        val = apply_climatology(usable[usable["split"] == "val"], clim)
        m = fit(train, val, cols, "target_high")
        pv = m.predict(val[cols])
        sel = val["local_hour"].isin(SEL_HOURS).to_numpy()
        results[v] = {"n_features": len(cols), "train_rows": len(train),
                      "best_iteration": int(m.best_iteration),
                      "val_mae": mae(pv, val["target_high"]),
                      "val_mae_h0_12": mae(pv[sel], val["target_high"].to_numpy()[sel])}
        models[v], clims[v] = m, clim
        print(f"  {v}: val MAE {results[v]['val_mae']:.3f}  hours 0-12 "
              f"{results[v]['val_mae_h0_12']:.3f}  (iter {m.best_iteration}, {len(cols)} feats)")

    # Selection: lowest hours-0-12 validation MAE; earlier-listed wins within SEL_TOL.
    order = [v for v in ["A0", "A1", "B1", "C1", "C2"] if v in results]
    best = min(results[v]["val_mae_h0_12"] for v in order)
    chosen = next(v for v in order if results[v]["val_mae_h0_12"] <= best + SEL_TOL)
    print(f"selected variant: {chosen}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols, model, clim = vf[chosen], models[chosen], clims[chosen]
    joblib.dump(model, OUT_DIR / "knyc_model_daily_high.pkl")
    clim.reset_index().to_csv(OUT_DIR / "tmpf_climatology.csv", index=False)

    # Test MAE (candidate vs production on identical rows) for the manifest.
    test = usable[usable["split"] == "test"]
    test_c = apply_climatology(test, clim)
    test_p = apply_climatology(test, production_climatology())
    pred_c = model.predict(test_c[cols])
    import joblib as _jl
    prod_high = _jl.load(paths.MODEL_DAILY_HIGH)
    pred_p = prod_high.predict(test_p[prod_cols])
    metrics = {"daily_high_mae": mae(pred_c, test["target_high"]),
               "production_daily_high_mae_same_rows": mae(pred_p, test["target_high"])}

    # t+3h / t+6h by-products on the C1 feature set (plan 5).
    hz = {}
    for target, prod_path, key in (("target_t3h", paths.MODEL_T3H, "t3h"),
                                   ("target_t6h", paths.MODEL_T6H, "t6h")):
        u = usable.dropna(subset=[target])
        tr = apply_climatology(mask_guidance(u[u["split"] == "train"]), clims.get("C1", clim))
        va = apply_climatology(u[u["split"] == "val"], clims.get("C1", clim))
        te = u[u["split"] == "test"]
        m = fit(tr, va, vf["C1"], target)
        c_mae = mae(m.predict(apply_climatology(te, clims.get("C1", clim))[vf["C1"]]), te[target])
        p_mae = mae(_jl.load(prod_path).predict(apply_climatology(te, production_climatology())[prod_cols]),
                    te[target])
        keep = (key == "t6h" and p_mae - c_mae >= 0.10) or (key == "t3h" and c_mae - p_mae <= 0.02)
        hz[key] = {"candidate_test_mae": c_mae, "production_test_mae": p_mae}
        print(f"  {key}: candidate {c_mae:.3f}  production {p_mae:.3f}")
        if keep:
            joblib.dump(m, OUT_DIR / f"knyc_model_{key}.pkl")
    t6_ok = hz["t6h"]["production_test_mae"] - hz["t6h"]["candidate_test_mae"] >= 0.10
    t3_ok = hz["t3h"]["candidate_test_mae"] - hz["t3h"]["production_test_mae"] <= 0.02
    hz["carry_forward"] = bool(t6_ok and t3_ok)
    if not hz["carry_forward"]:
        for key in ("t3h", "t6h"):
            (OUT_DIR / f"knyc_model_{key}.pkl").unlink(missing_ok=True)
    metrics["horizons"] = hz

    # Candidate offset PMF from validation + test predictions (never the holdout).
    vt = usable[usable["split"].isin(["val", "test"])]
    res = vt[["local_hour", "target_high", "obs_so_far", "lst_matches_civil"]].copy()
    res["pred_high"] = model.predict(apply_climatology(vt, clim)[cols])
    build_offset_pmf(res).to_csv(OUT_DIR / "offset_pmf.csv", index=False)

    manifest = {
        "candidate": CANDIDATE, "variant": chosen,
        "feature_cols": cols, "upstream_stations": prod["upstream_stations"],
        "guidance_cols": [c for c in cols if c in g.GUIDANCE_COLS],
        "obs_day_cols": [c for c in cols if c in g.OBS_DAY_COLS],
        "train_through": str(TRAIN_END), "validation": [str(VAL[0]), str(VAL[1])],
        "test": [str(TEST[0]), str(TEST[1])], "holdout_from": str(HOLDOUT_START),
        "lags_h": {"gfs": g.GFS_AVAIL_LAG / g.HOUR, "lav": g.LAV_AVAIL_LAG / g.HOUR},
        "max_age_h": {"gfs_nx": g.NX_MAX_AGE / g.HOUR, "gfs_tmp": g.GFS_TMP_MAX_AGE / g.HOUR,
                      "lav": g.LAV_MAX_AGE / g.HOUR},
        "masking": {"gfs_days": MASK_GFS_DAYS, "lav_days": MASK_LAV_DAYS, "late_rows": MASK_LATE_ROWS},
        "test_metrics": metrics,
    }
    (OUT_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "variants.json").write_text(json.dumps({"selected": chosen, "variants": results}, indent=2))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
