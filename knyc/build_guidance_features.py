"""
Compute the guidance-retrain features (GFS MOS, LAMP, NBM + observation) for every training row and write
data/guidance/guidance_features.csv keyed by `valid`. The training CSV itself
is not rewritten.

Rows come from data/knyc_training.csv plus data/knyc_training_tail.csv (rows
after the base file ends). Also writes `late_*` copies of the guidance columns
computed as if each model's newest eligible run were missing (the plan's
late-bulletin masking draws from these).

    python -m knyc.build_guidance_features
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from knyc import guidance as g
from knyc import paths


def load_runs(model: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(paths.GUIDANCE_RAW_DIR / f"{model}_*.csv")))
    if not files:
        raise SystemExit(f"No {model} archive in {paths.GUIDANCE_RAW_DIR}; run "
                         f"python -m knyc.build_guidance_archive first.")
    raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    return g.clean_runs(raw, model)


def load_rows(columns: list[str] | None = None) -> pd.DataFrame:
    """Base training rows + tail rows after the base file ends, sorted."""
    base = pd.read_csv(paths.TRAINING_CSV, usecols=columns, low_memory=False)
    base["valid"] = pd.to_datetime(base["valid"], utc=True)
    parts = [base]
    if paths.TRAINING_TAIL_CSV.exists():
        tail = pd.read_csv(paths.TRAINING_TAIL_CSV, usecols=columns, low_memory=False)
        tail["valid"] = pd.to_datetime(tail["valid"], utc=True)
        parts.append(tail[tail["valid"] > base["valid"].max()])
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values("valid").drop_duplicates("valid").reset_index(drop=True)


def main() -> None:
    rows = load_rows(["valid", "tmpf", "mxtmpf_6hr", "target_high"])
    print(f"rows: {len(rows):,}  {rows.valid.min()} -> {rows.valid.max()}")
    gfs, lav, nbs = load_runs("GFS"), load_runs("LAV"), load_runs("NBS")
    print(f"GFS runs {gfs.runtime.nunique():,}, LAMP runs {lav.runtime.nunique():,}, "
          f"NBM runs {nbs.runtime.nunique():,}")

    out = pd.concat([rows[["valid"]], g.obs_day_features(rows),
                     g.guidance_features(rows, gfs, lav, nbs)], axis=1)
    late = g.guidance_features(rows, gfs, lav, nbs, skip_newest=True)[g.GUIDANCE_COLS]
    out = pd.concat([out, late.add_prefix("late_")], axis=1)

    paths.GUIDANCE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.GUIDANCE_FEATURES, index=False)
    print(f"wrote {paths.GUIDANCE_FEATURES} ({len(out):,} rows, {out.shape[1]} cols)")

    # Sanity checks (plan, phase 1 step 6)
    t = rows["target_high"]
    nx = out["gfs_nx_today"]
    lav_era = (rows["valid"] >= "2020-07-12") & ~rows["valid"].between("2023-06-07", "2023-08-25")
    recent = rows["valid"] >= "2022-01-01"
    ok_obs = (out["obs_high_so_far"].isna() | (out["obs_high_so_far"] <= t + 1))
    checks = {
        "gfs_nx_today non-null (>= 99%)": nx.notna().mean(),
        "LAMP non-null after 2020-07-12 outside 2023 gap (>= 95%)": out.loc[lav_era, "lav_tmp_t3"].notna().mean(),
        "MAE gfs_nx_today vs target 2022+ (2.0-2.4 F)": float((nx - t)[recent].abs().mean()),
        "MAE nbs_nx_today vs target 2022+": float((out["nbs_nx_today"] - t)[recent].abs().mean()),
        "NBS non-null after 2019-01-01": out.loc[rows["valid"] >= "2019-01-01", "nbs_tmp_t3"].notna().mean(),
        "obs_high_so_far <= target + 1 (>= 99%)": ok_obs[t.notna()].mean(),
    }
    for k, v in checks.items():
        print(f"  {k}: {v:.4f}")
    for c in g.GUIDANCE_COLS + g.OBS_DAY_COLS:
        print(f"    {c:<24} non-null {out[c].notna().mean():6.1%}  mean {np.nanmean(out[c]):8.2f}")


if __name__ == "__main__":
    main()
