"""
Evaluate the guidance candidate against production and the Kalshi market and
apply the go/no-go gate (docs/PLAN_GUIDANCE_RETRAIN.md section 6). Writes
GATE_REPORT.md and gate.json into the candidate directory.

    python -m knyc.eval_guidance                          # guidance_v1
    python -m knyc.eval_guidance --candidate guidance_v2
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from knyc import guidance as g
from knyc import market_scorer as ms
from knyc import paths
from knyc.build_residuals import build_offset_pmf
from knyc.train_guidance import (apply_climatology, load_dataset,
                                 production_climatology, production_manifest)

BLOCKS = ms.HOUR_BLOCKS
BLOCK_NAMES = [f"{a:02d}-{b:02d}" for a, b in BLOCKS]


def block_of(hours: pd.Series) -> pd.Series:
    out = pd.Series(index=hours.index, dtype=object)
    for (a, b), name in zip(BLOCKS, BLOCK_NAMES):
        out[(hours >= a) & (hours <= b)] = name
    return out


def mae_table(df: pd.DataFrame, preds: dict[str, str], truth: str) -> pd.DataFrame:
    rows = []
    for name, grp in df.groupby(block_of(df["local_hour"])):
        rec = {"block": name, "n": len(grp)}
        for label, col in preds.items():
            ok = grp[col].notna() & grp[truth].notna()
            e = grp.loc[ok, col] - grp.loc[ok, truth]
            rec[f"{label}_mae"] = float(e.abs().mean()) if ok.any() else math.nan
            rec[f"{label}_bias"] = float(e.mean()) if ok.any() else math.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("block").reindex(BLOCK_NAMES)


def reassess(pred, obs, cur, hour) -> np.ndarray:
    """Vectorised knyc_discord_bot.reassess_high."""
    pred, obs, cur, hour = (np.asarray(x, float) for x in (pred, obs, cur, hour))
    lock = (hour >= 19) & (cur <= obs - 1.0)
    out = np.where(np.isnan(obs), pred, np.where(lock, obs, np.fmax(obs, pred)))
    return out


def day_bootstrap_diff(df: pd.DataFrame, a: str, b: str, truth: str,
                       n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of |a-truth| - |b-truth| and its day-clustered 95% CI."""
    d = df.assign(diff=(df[a] - df[truth]).abs() - (df[b] - df[truth]).abs()).dropna(subset=["diff"])
    by = d.groupby("local_date")["diff"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by), size=(n_boot, len(by)))
    s, c = by["sum"].to_numpy(), by["count"].to_numpy()
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(d["diff"].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def market_point(ladder: pd.DataFrame) -> float:
    """Mid-weighted ladder centre: between = (lo+hi)/2, tails = edge -/+ 0.5."""
    mid = ((ladder["yes_bid"] + ladder["yes_ask"]) / 2.0).to_numpy(float)
    if mid.sum() <= 0:
        return math.nan
    lo, hi = ladder["lo"].to_numpy(float), ladder["hi"].to_numpy(float)
    centre = np.where(np.isinf(lo) | (lo < -900), hi - 0.5,
                      np.where(np.isinf(hi) | (hi > 900), lo + 0.5, (lo + hi) / 2.0))
    return float((mid / mid.sum() * centre).sum())


def brier_blocks(by_hour: pd.DataFrame) -> dict:
    out = {}
    for (a, b), name in zip(BLOCKS, BLOCK_NAMES):
        h = by_hour[(by_hour["local_hour"] >= a) & (by_hour["local_hour"] <= b)]
        n = h["n"].sum()
        out[name] = {"n": int(n),
                     "model": float((h["model"] * h["n"]).sum() / n) if n else math.nan,
                     "market": float((h["market"] * h["n"]).sum() / n) if n else math.nan}
    return out


def regress(df: pd.DataFrame, hours: tuple[int, int], cols: list[str]) -> dict:
    d = df[(df["local_hour"] >= hours[0]) & (df["local_hour"] < hours[1])].dropna(subset=cols + ["actual"])
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    beta, *_ = np.linalg.lstsq(X, d["actual"].to_numpy(float), rcond=None)
    return {"n": len(d), "intercept": float(beta[0]), **{c: float(v) for c, v in zip(cols, beta[1:])}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="guidance_v1")
    OUT_DIR = paths.CANDIDATES_DIR / ap.parse_args().candidate
    cman = json.loads((OUT_DIR / "feature_manifest.json").read_text())
    ccols = cman["feature_cols"]
    cand = joblib.load(OUT_DIR / "knyc_model_daily_high.pkl")
    cclim = pd.read_csv(OUT_DIR / "tmpf_climatology.csv").set_index(["month", "hour"])["tmpf_clim"]
    pcols = production_manifest()["feature_cols"]
    prod = joblib.load(paths.MODEL_DAILY_HIGH)
    pclim = production_climatology()

    df = load_dataset()
    df = df.dropna(subset=pcols).copy()
    dc, dp = apply_climatology(df, cclim), apply_climatology(df, pclim)
    df["pred_cand"] = cand.predict(dc[ccols])
    df["pred_prod"] = prod.predict(dp[pcols])
    sources = [s for s, cols in g.SOURCE_COLS.items() if any(c in ccols for c in cols)]
    off_sets = [(f"{s.lower()}_off", g.SOURCE_COLS[s]) for s in sources] + [("all_off", g.GUIDANCE_COLS)]
    for name, cols in off_sets:
        x = dc[ccols].copy()
        x[[c for c in cols if c in ccols]] = np.nan
        df[f"pred_{name}"] = cand.predict(x)
    raw_nx = [c for c in ("gfs_nx_today", "nbs_nx_today") if c in ccols]
    df["re_cand"] = reassess(df["pred_cand"], df["obs_high_so_far"], df["tmpf"], df["local_hour"])
    df["re_prod"] = reassess(df["pred_prod"], df["obs_high_so_far"], df["tmpf"], df["local_hour"])
    report, gate = [], {}

    def p(line=""):
        print(line)
        report.append(line)

    # ── 6.1 Test period ─────────────────────────────────────────────────────
    test = df[(df["split"] == "test") & df["target_high"].notna()]
    t = mae_table(test, {"cand": "pred_cand", "prod": "pred_prod", "gfs_nx": "gfs_nx_today",
                         "re_cand": "re_cand", "re_prod": "re_prod",
                         **{c.split("_")[0] + "_raw": c for c in raw_nx},
                         **{name: f"pred_{name}" for name, _ in off_sets}}, "target_high")
    p(f"# Gate report: {cman['candidate']} (variant {cman['variant']})")
    p()
    p(f"## 6.1 Test period {cman['test'][0]} to {cman['test'][1]} ({test['local_date'].nunique()} days, {len(test):,} rows)")
    p()
    p("| Block | n | Candidate MAE | Production MAE | Change | Cand bias | Prod bias | Raw GFS n_x MAE | Reassessed cand | Reassessed prod |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for b, r in t.iterrows():
        p(f"| {b} | {r.n} | {r.cand_mae:.2f} | {r.prod_mae:.2f} | {(r.cand_mae / r.prod_mae - 1) * 100:+.0f}% | "
          f"{r.cand_bias:+.2f} | {r.prod_bias:+.2f} | {r.gfs_nx_mae:.2f} | {r.re_cand_mae:.2f} | {r.re_prod_mae:.2f} |")
    red = 1 - t["cand_mae"] / t["prod_mae"]
    g1 = bool(red["00-05"] >= 0.20 and red["06-09"] >= 0.20 and red["10-12"] >= 0.10
              and all(t.loc[b, "cand_mae"] <= t.loc[b, "prod_mae"] + 0.03 for b in ("13-15", "16-18", "19-23")))
    h09 = test[test["local_hour"] <= 9].dropna(subset=raw_nx)
    g1b_c = (h09["pred_cand"] - h09["target_high"]).abs().mean()
    raw_mae = {c: float((h09[c] - h09["target_high"]).abs().mean()) for c in raw_nx}
    best_raw = min(raw_mae, key=raw_mae.get)
    g1b_g = raw_mae[best_raw]
    g1b = bool(g1b_c <= g1b_g - 0.15)
    gate["G1"] = {"pass": g1, "reduction": {b: float(v) for b, v in red.items()}}
    gate["G1b"] = {"pass": g1b, "candidate_h0_9": float(g1b_c), "raw_h0_9": raw_mae, "best_raw": best_raw}
    p()
    p(f"G1 (>=20% lower 00-05 and 06-09, >=10% 10-12, <=+0.03 F later): **{'PASS' if g1 else 'FAIL'}**")
    p(f"G1b (hours 0-9: candidate {g1b_c:.2f} vs best raw guidance max ({best_raw}) {g1b_g:.2f}, "
      f"needs >= 0.15 lower): **{'PASS' if g1b else 'FAIL'}**")
    if "nbs_raw_mae" in t:
        p(f"(raw NBM n_x MAE by block: {', '.join(f'{b} {v:.2f}' for b, v in t['nbs_raw_mae'].items())})")

    # ── 6.4 Degradation ────────────────────────────────────────────────────
    p()
    p("## 6.4 Degradation (test period, guidance forced missing)")
    p()
    labels = {"gfs_off": "GFS missing", "lav_off": "LAMP missing", "nbs_off": "NBM missing",
              "all_off": "All guidance missing"}
    p("| Block | Candidate | Production | " + " | ".join(labels[n] for n, _ in off_sets) + " |")
    p("|---|---|---|" + "---|" * len(off_sets))
    for b, r in t.iterrows():
        p(f"| {b} | {r.cand_mae:.2f} | {r.prod_mae:.2f} | " + " | ".join(f"{r[f'{n}_mae']:.2f}" for n, _ in off_sets) + " |")
    g4 = bool((t["all_off_mae"] <= t["prod_mae"] + 0.15).all())
    gate["G4"] = {"pass": g4, "worst_excess": float((t["all_off_mae"] - t["prod_mae"]).max())}
    p()
    p(f"G4 (all guidance missing within 0.15 F of production in every block): **{'PASS' if g4 else 'FAIL'}**")

    # ── 6.2 Live holdout ───────────────────────────────────────────────────
    mlog = paths.SERVER_PULL_DIR / "market_log.csv"
    nlog = paths.SERVER_PULL_DIR / "nowcast_log.csv"
    joined = ms.load_joined(mlog, nlog)
    hours = joined.drop_duplicates("valid_utc")[["valid_utc", "local_date", "pred_high"]]
    truth = ms.settlement_truth(sorted(hours["local_date"].unique()), verbose=False)
    hold = hours.merge(df[["valid", "local_hour", "tmpf", "obs_high_so_far", "obs_so_far", "pred_cand",
                           "pred_prod", "re_cand", "re_prod"]],
                       left_on="valid_utc", right_on="valid", how="inner")
    hold["actual"] = hold["local_date"].map(truth)
    hold = hold.dropna(subset=["actual"])
    hold = hold.rename(columns={"pred_high": "pred_logged"})
    mp = {v: market_point(grp) for v, grp in joined.groupby("valid_utc")}
    hold["market"] = hold["valid_utc"].map(mp)
    ht = mae_table(hold, {"cand": "pred_cand", "prod": "pred_prod", "logged": "pred_logged",
                          "market": "market"}, "actual")
    p()
    p(f"## 6.2 Live holdout ({hold['local_date'].nunique()} settled days, {len(hold):,} hours with market data)")
    p()
    p("| Block | n | Candidate | Production replay | Production as logged | Market |")
    p("|---|---|---|---|---|---|")
    for b, r in ht.iterrows():
        p(f"| {b} | {r.n} | {r.cand_mae:.2f} | {r.prod_mae:.2f} | {r.logged_mae:.2f} | {r.market_mae:.2f} |")
    h011 = hold[hold["local_hour"] <= 11]
    mean_d, lo, hi = day_bootstrap_diff(h011, "pred_cand", "pred_prod", "actual")
    c09 = (hold.loc[hold["local_hour"] <= 9, "pred_cand"] - hold.loc[hold["local_hour"] <= 9, "actual"]).abs().mean()
    g2 = bool(-mean_d >= 0.40 and hi < 0 and c09 <= 2.00)
    gate["G2"] = {"pass": g2, "diff_h0_11": mean_d, "ci": [lo, hi], "cand_mae_h0_9": float(c09)}
    p()
    p(f"G2 (hours 0-11: candidate minus production {mean_d:+.2f} F, 95% CI [{lo:+.2f}, {hi:+.2f}]; "
      f"candidate hours 0-9 {c09:.2f} F): **{'PASS' if g2 else 'FAIL'}**")

    # Market MAE by hour, reproduced from the review.
    p()
    p("Market vs candidate MAE at selected hours (review table: market 0.86/0.81/0.83/0.77/0.67/0.52/0.56/0.54):")
    p()
    p("| Hour | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 |")
    p("|---|---|---|---|---|---|---|---|---|")
    for lab, col in (("Market", "market"), ("Candidate", "pred_cand"), ("Production replay", "pred_prod")):
        vals = [(hold.loc[hold.local_hour == h, col] - hold.loc[hold.local_hour == h, "actual"]).abs().mean()
                for h in (0, 3, 6, 9, 12, 15, 18, 21)]
        p(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")

    # ── 6.3 Brier ─────────────────────────────────────────────────────────
    vt = df[df["split"].isin(["val", "test"]) & df["target_high"].notna()]
    pmf_c = ms.OffsetPMF(pd.read_csv(OUT_DIR / "offset_pmf.csv"))
    pmf_p = ms.OffsetPMF(build_offset_pmf(vt.assign(pred_high=vt["pred_prod"])[
        ["local_hour", "pred_high", "target_high", "obs_so_far", "lst_matches_civil"]]))
    keymap = hold.set_index("valid_utc")
    jj = joined[joined["valid_utc"].isin(keymap.index)].copy()
    jj["obs_high"] = jj["valid_utc"].map(keymap["obs_high_so_far"])
    bri = {}
    for lab, col, pmf in (("candidate", "pred_cand", pmf_c), ("production", "pred_prod", pmf_p)):
        x = jj.assign(pred_high=jj["valid_utc"].map(keymap[col]))
        bri[lab] = ms.brier_scores(x, truth, pmf)
        bri[lab]["blocks"] = brier_blocks(bri[lab]["by_hour"])
    vs = ms.model_vs_favourite(jj.assign(pred_high=jj["valid_utc"].map(keymap["pred_cand"])), truth, pmf_c)
    bc, bp = bri["candidate"], bri["production"]
    p()
    p(f"## 6.3 Ladder Brier on the holdout (n={bc['n']} hours; lower is better)")
    p()
    p("| Block | Candidate | Production replay | Market |")
    p("|---|---|---|---|")
    p(f"| all | {bc['model']:.3f} | {bp['model']:.3f} | {bc['market']:.3f} |")
    for b in BLOCK_NAMES:
        p(f"| {b} | {bc['blocks'][b]['model']:.3f} | {bp['blocks'][b]['model']:.3f} | {bc['blocks'][b]['market']:.3f} |")
    blk = bc["blocks"]
    g3 = bool(bc["model"] <= 0.48 and bc["model"] <= bp["model"] - 0.05 and blk["00-05"]["model"] <= 0.74
              and blk["06-09"]["model"] <= 0.70 and blk["19-23"]["model"] <= 0.20)
    gate["G3"] = {"pass": g3, "candidate": bc["model"], "production": bp["model"], "market": bc["market"],
                  "blocks": {b: blk[b]["model"] for b in BLOCK_NAMES}}
    p()
    p(f"G3 (overall <= 0.48 and >= 0.05 below production; 00-05 <= 0.74; 06-09 <= 0.70; 19-23 <= 0.20): "
      f"**{'PASS' if g3 else 'FAIL'}**")
    p()
    p(f"Model vs market favourite where they disagree ({vs['disagree']} hours): candidate right "
      f"{vs['model_right_pct']:.0f}%, favourite right {vs['fav_right_pct']:.0f}%.")
    p()
    p("Regression `actual ~ market + candidate` (with intercept):")
    p()
    for hrs in ((0, 6), (6, 10), (13, 16)):
        r = regress(hold, hrs, ["market", "pred_cand"])
        p(f"- {hrs[0]:02d}-{hrs[1]:02d}h (n={r['n']}): market {r['market']:.2f}, candidate {r['pred_cand']:.2f}, "
          f"intercept {r['intercept']:+.1f}")

    # ── G5 tests ──────────────────────────────────────────────────────────
    res = subprocess.run([sys.executable, str(paths.ROOT / "tests" / "test_guidance.py")],
                         capture_output=True, text=True)
    g5 = res.returncode == 0
    gate["G5"] = {"pass": g5, "note": "tests/test_guidance.py; live parity: tools/check_feature_parity.py"}
    p()
    p(f"G5 (leakage/parity unit tests): **{'PASS' if g5 else 'FAIL'}** "
      "(live parity check run separately: `python tools/check_feature_parity.py`)")

    # ── Outcome ───────────────────────────────────────────────────────────
    passed = {k: v["pass"] for k, v in gate.items()}
    if all(passed.values()):
        outcome = "GO"
    elif not (passed["G1"] and passed["G1b"]):
        outcome = "STOP"
    elif passed["G4"] and passed["G5"]:
        outcome = "ITERATE"
    else:
        outcome = "STOP"
    c_h09 = (hold.loc[hold.local_hour <= 9, "pred_cand"] - hold.loc[hold.local_hour <= 9, "actual"]).abs().mean()
    m_h09 = (hold.loc[hold.local_hour <= 9, "market"] - hold.loc[hold.local_hour <= 9, "actual"]).abs().mean()
    parity = bool(c_h09 <= m_h09 + 0.25 and bc["model"] <= 0.33)
    gate["outcome"] = outcome
    gate["market_parity_tier"] = {"reached": parity, "cand_h0_9": float(c_h09), "market_h0_9": float(m_h09)}
    p()
    p(f"## Outcome: **{outcome}**  ({', '.join(f'{k} {chr(10003) if v else chr(10007)}' for k, v in passed.items())})")
    p()
    p(f"Market-parity tier (informational): {'reached' if parity else 'not reached'} "
      f"(hours 0-9: candidate {c_h09:.2f} vs market {m_h09:.2f}; Brier {bc['model']:.3f} vs 0.33).")

    (OUT_DIR / "GATE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT_DIR / "gate.json").write_text(json.dumps(gate, indent=2, default=float))
    print(f"\nwrote {OUT_DIR / 'GATE_REPORT.md'}")


if __name__ == "__main__":
    main()
