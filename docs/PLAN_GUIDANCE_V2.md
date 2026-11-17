# guidance_v2: NBM + outage-hardened retrain (pre-registration)

Written 2026-11-16, **before** any v2 model was trained. Builds on `docs/PLAN_GUIDANCE_RETRAIN.md`
(v1). v1 ended in STOP: G1 (midday not better on test), G3 (holdout Brier 0.508 > 0.48) and G4 (with both
guidance feeds missing the model was far worse than production: 5.1 vs 3.2 F overnight) failed; the
selection rule picked C2, a 2020-2023-only variant that stopped after 208 trees.

## What changes from v1

1. **NBM added** (IEM model code `NBS`, archive from 2018-11-08). Eight features mirroring GFS MOS:
   `nbs_nx_today, nbs_nx_age_h, nbs_nx_rev, nbs_tmp_max_rest, nbs_tmp_age_h, nbs_tmp_now_err,
   nbs_tmp_t3, nbs_tmp_t6`. The max/min column is `n_x` before 2021 and `txn` after; both are read.
   Archived cycles moved over time (00/07/12/19Z, 01/07/13/19Z, now 00/06/12/18Z); all eight hours are
   accepted and the archive's own thinning decides which exist. Availability lag **3 h** (no arrival
   time is archived; to be measured in shadow). Max age: 36 h for `n_x`, 18 h for `tmp`.
   Measured before training: raw `nbs_nx_today` MAE 1.94 F vs GFS 2.13 F (2022 onward).
2. **No variant selection.** The candidate is fixed in advance: **D1 = production 91 + 2 observation +
   8 GFS + 5 LAMP + 8 NBM features, trained 2010-01-01 to 2023-12-31** (full era; NBM/LAMP missing
   before their archives start). One informational variant, D2 (same features, trained 2019-01-01 to
   2023-12-31), is reported but cannot become the candidate.
3. **Outage-hardened masking** (training rows only, seed 42, per climate day unless noted):
   - each source independently: GFS 10%, LAMP 10%, NBM 10% of days;
   - **all guidance missing: a further 15% of days** (v1 had about 1% by chance);
   - 5% of rows use the late-bulletin values (newest eligible run of every model removed).
4. Splits, hyperparameters, recency weights, PMF construction, holdout and gate thresholds are
   **unchanged** from v1, with two gate definitions adapted to three sources:
   - **G1b**: candidate MAE at hours 0-9 must be at least 0.15 F below the **better** of raw
     `gfs_nx_today` and raw `nbs_nx_today` (rows where both exist).
   - **G4**: with **all three** sources missing, candidate MAE within 0.15 F of production in every block.

## Caveat that cannot be removed

The test period and live holdout were already looked at for v1, and v2's changes respond to v1's
failures. A v2 GO on the same data is therefore weaker evidence than v1's numbers would have been;
any switch still requires the shadow run on **new** live days (plan section 8).
