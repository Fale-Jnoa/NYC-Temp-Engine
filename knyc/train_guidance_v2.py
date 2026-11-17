"""
Train guidance_v2 (docs/PLAN_GUIDANCE_V2.md): GFS MOS + LAMP + NBM features,
outage-hardened masking, and a candidate fixed in advance (D1, full era).
D2 (2019-2023 only) is trained for information and cannot become the
candidate. Splits, hyperparameters and weights are v1's (train_guidance.py).

    python -m knyc.train_guidance_v2
"""
from __future__ import annotations

import json
from datetime import date

import joblib
import numpy as np
import pandas as pd

from knyc import guidance as g
from knyc import paths
from knyc.build_residuals import build_offset_pmf
from knyc.train_guidance import (HOLDOUT_START, SEED, TEST, TRAIN_END, VAL, apply_climatology,
                                 climatology, fit, load_dataset, mae, production_climatology,
                                 production_manifest)

CANDIDATE = "guidance_v2"
OUT_DIR = paths.CANDIDATES_DIR / CANDIDATE
D2_START = date(2019, 1, 1)
MASK_SOURCE_DAYS = {"GFS": 0.10, "LAV": 0.10, "NBS": 0.10}
MASK_ALL_DAYS = 0.15
MASK_LATE_ROWS = 0.05


def mask_v2(train: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = train.copy()
    days = np.array(sorted(out["local_date"].unique()))
    late = rng.random(len(out)) < MASK_LATE_ROWS
    for c in g.GUIDANCE_COLS:
        out.loc[late, c] = out.loc[late, f"late_{c}"]
    for src, frac in MASK_SOURCE_DAYS.items():
        drop = set(days[rng.random(len(days)) < frac])
        out.loc[out["local_date"].isin(drop), g.SOURCE_COLS[src]] = np.nan
    all_days = set(days[rng.random(len(days)) < MASK_ALL_DAYS])
    out.loc[out["local_date"].isin(all_days), g.GUIDANCE_COLS] = np.nan
    return out


def main() -> None:
    prod = production_manifest()
    prod_cols = prod["feature_cols"]
    cols = prod_cols + g.OBS_DAY_COLS + g.GFS_COLS + g.LAV_COLS + g.NBS_COLS

    df = load_dataset()
    usable = df.dropna(subset=prod_cols + ["target_high"])
    print(f"rows {len(df):,}, usable {len(usable):,}")

    results, fitted = {}, {}
    for v, start in (("D1", None), ("D2", D2_START)):
        train = usable[usable["split"] == "train"]
        if start is not None:
            train = train[train["local_date"] >= start]
        clim = climatology(train)
        tr = apply_climatology(mask_v2(train), clim)
        va = apply_climatology(usable[usable["split"] == "val"], clim)
        m = fit(tr, va, cols, "target_high")
        pv = m.predict(va[cols])
        sel = va["local_hour"].between(0, 12).to_numpy()
        allmiss = mae(m.predict(va[cols].assign(**{c: np.nan for c in g.GUIDANCE_COLS})), va["target_high"])
        results[v] = {"train_rows": len(tr), "best_iteration": int(m.best_iteration),
                      "val_mae": mae(pv, va["target_high"]),
                      "val_mae_h0_12": mae(pv[sel], va["target_high"].to_numpy()[sel]),
                      "val_mae_all_guidance_missing": allmiss,
                      "role": "candidate (pre-registered)" if v == "D1" else "informational only"}
        fitted[v] = (m, clim)
        print(f"  {v}: val MAE {results[v]['val_mae']:.3f}  hours 0-12 {results[v]['val_mae_h0_12']:.3f}  "
              f"all-guidance-missing {allmiss:.3f}  (iter {m.best_iteration})")

    model, clim = fitted["D1"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "knyc_model_daily_high.pkl")
    clim.reset_index().to_csv(OUT_DIR / "tmpf_climatology.csv", index=False)

    test = usable[usable["split"] == "test"]
    prod_high = joblib.load(paths.MODEL_DAILY_HIGH)
    metrics = {
        "daily_high_mae": mae(model.predict(apply_climatology(test, clim)[cols]), test["target_high"]),
        "production_daily_high_mae_same_rows":
            mae(prod_high.predict(apply_climatology(test, production_climatology())[prod_cols]), test["target_high"]),
    }

    hz = {}
    for target, prod_path, key in (("target_t3h", paths.MODEL_T3H, "t3h"), ("target_t6h", paths.MODEL_T6H, "t6h")):
        u = usable.dropna(subset=[target])
        tr = apply_climatology(mask_v2(u[u["split"] == "train"]), clim)
        va = apply_climatology(u[u["split"] == "val"], clim)
        te = u[u["split"] == "test"]
        m = fit(tr, va, cols, target)
        c_mae = mae(m.predict(apply_climatology(te, clim)[cols]), te[target])
        p_mae = mae(joblib.load(prod_path).predict(apply_climatology(te, production_climatology())[prod_cols]),
                    te[target])
        hz[key] = {"candidate_test_mae": c_mae, "production_test_mae": p_mae}
        joblib.dump(m, OUT_DIR / f"knyc_model_{key}.pkl")
        print(f"  {key}: candidate {c_mae:.3f}  production {p_mae:.3f}")
    hz["carry_forward"] = bool(hz["t6h"]["production_test_mae"] - hz["t6h"]["candidate_test_mae"] >= 0.10
                               and hz["t3h"]["candidate_test_mae"] - hz["t3h"]["production_test_mae"] <= 0.02)
    if not hz["carry_forward"]:
        for key in ("t3h", "t6h"):
            (OUT_DIR / f"knyc_model_{key}.pkl").unlink(missing_ok=True)
    metrics["horizons"] = hz

    vt = usable[usable["split"].isin(["val", "test"])]
    res = vt[["local_hour", "target_high", "obs_so_far", "lst_matches_civil"]].copy()
    res["pred_high"] = model.predict(apply_climatology(vt, clim)[cols])
    build_offset_pmf(res).to_csv(OUT_DIR / "offset_pmf.csv", index=False)

    manifest = {
        "candidate": CANDIDATE, "variant": "D1", "feature_cols": cols,
        "upstream_stations": prod["upstream_stations"],
        "guidance_cols": [c for c in cols if c in g.GUIDANCE_COLS], "obs_day_cols": g.OBS_DAY_COLS,
        "train_through": str(TRAIN_END), "validation": [str(VAL[0]), str(VAL[1])],
        "test": [str(TEST[0]), str(TEST[1])], "holdout_from": str(HOLDOUT_START),
        "lags_h": {"gfs": g.GFS_AVAIL_LAG / g.HOUR, "lav": g.LAV_AVAIL_LAG / g.HOUR, "nbs": g.NBS_AVAIL_LAG / g.HOUR},
        "masking": {"source_days": MASK_SOURCE_DAYS, "all_days": MASK_ALL_DAYS, "late_rows": MASK_LATE_ROWS},
        "test_metrics": metrics,
    }
    (OUT_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "variants.json").write_text(json.dumps({"selected": "D1 (pre-registered)", "variants": results}, indent=2))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
