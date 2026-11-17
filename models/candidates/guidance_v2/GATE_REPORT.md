# Gate report: guidance_v2 (variant D1)

## 6.1 Test period 2025-01-01 to 2026-09-23 (622 days, 11,993 rows)

| Block | n | Candidate MAE | Production MAE | Change | Cand bias | Prod bias | Raw GFS n_x MAE | Reassessed cand | Reassessed prod |
|---|---|---|---|---|---|---|---|---|---|
| 00-05 | 3381.0 | 1.77 | 3.17 | -44% | +0.31 | +0.24 | 2.19 | 1.76 | 3.16 |
| 06-09 | 1862.0 | 1.61 | 2.50 | -35% | +0.39 | +0.36 | 2.18 | 1.59 | 2.47 |
| 10-12 | 1171.0 | 1.23 | 1.69 | -27% | +0.29 | +0.35 | 2.23 | 1.19 | 1.61 |
| 13-15 | 1334.0 | 0.82 | 1.08 | -24% | +0.19 | +0.26 | 2.17 | 0.76 | 0.92 |
| 16-18 | 1559.0 | 0.69 | 1.02 | -33% | +0.08 | +0.21 | 2.10 | 0.60 | 0.79 |
| 19-23 | 2686.0 | 0.58 | 1.23 | -53% | +0.04 | +0.11 | 2.07 | 0.13 | 0.14 |

G1 (>=20% lower 00-05 and 06-09, >=10% 10-12, <=+0.03 F later): **PASS**
G1b (hours 0-9: candidate 1.72 vs best raw guidance max (nbs_nx_today) 2.03, needs >= 0.15 lower): **PASS**
(raw NBM n_x MAE by block: 00-05 2.00, 06-09 2.09, 10-12 2.17, 13-15 2.12, 16-18 2.02, 19-23 2.00)

## 6.4 Degradation (test period, guidance forced missing)

| Block | Candidate | Production | GFS missing | LAMP missing | NBM missing | All guidance missing |
|---|---|---|---|---|---|---|
| 00-05 | 1.77 | 3.17 | 2.63 | 1.79 | 1.91 | 3.46 |
| 06-09 | 1.61 | 2.50 | 2.24 | 1.64 | 1.69 | 2.74 |
| 10-12 | 1.23 | 1.69 | 1.50 | 1.25 | 1.26 | 1.72 |
| 13-15 | 0.82 | 1.08 | 0.93 | 0.82 | 0.83 | 0.97 |
| 16-18 | 0.69 | 1.02 | 0.78 | 0.69 | 0.71 | 0.81 |
| 19-23 | 0.58 | 1.23 | 0.70 | 0.58 | 0.58 | 0.72 |

G4 (all guidance missing within 0.15 F of production in every block): **FAIL**

## 6.2 Live holdout (54 settled days, 1,066 hours with market data)

| Block | n | Candidate | Production replay | Production as logged | Market |
|---|---|---|---|---|---|
| 00-05 | 302.0 | 1.57 | 2.53 | 2.62 | 0.87 |
| 06-09 | 180.0 | 1.63 | 2.02 | 2.17 | 0.86 |
| 10-12 | 110.0 | 1.36 | 1.63 | 1.67 | 0.76 |
| 13-15 | 105.0 | 1.00 | 1.18 | 1.20 | 0.54 |
| 16-18 | 135.0 | 0.75 | 0.88 | 0.88 | 0.53 |
| 19-23 | 234.0 | 0.75 | 1.14 | 1.10 | 0.52 |

G2 (hours 0-11: candidate minus production -0.69 F, 95% CI [-1.21, -0.21]; candidate hours 0-9 1.59 F): **PASS**

Market vs candidate MAE at selected hours (review table: market 0.86/0.81/0.83/0.77/0.67/0.52/0.56/0.54):

| Hour | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 |
|---|---|---|---|---|---|---|---|---|
| Market | 0.90 | 0.83 | 0.84 | 0.79 | 0.75 | 0.50 | 0.55 | 0.53 |
| Candidate | 1.51 | 1.54 | 1.71 | 1.56 | 1.19 | 1.01 | 0.68 | 0.77 |
| Production replay | 3.01 | 2.44 | 2.25 | 1.81 | 1.39 | 0.97 | 0.91 | 1.19 |

## 6.3 Ladder Brier on the holdout (n=1066 hours; lower is better)

| Block | Candidate | Production replay | Market |
|---|---|---|---|
| all | 0.524 | 0.611 | 0.275 |
| 00-05 | 0.698 | 0.861 | 0.453 |
| 06-09 | 0.711 | 0.826 | 0.467 |
| 10-12 | 0.593 | 0.644 | 0.337 |
| 13-15 | 0.516 | 0.476 | 0.217 |
| 16-18 | 0.338 | 0.384 | 0.064 |
| 19-23 | 0.234 | 0.301 | 0.017 |

G3 (overall <= 0.48 and >= 0.05 below production; 00-05 <= 0.74; 06-09 <= 0.70; 19-23 <= 0.20): **FAIL**

Model vs market favourite where they disagree (432 hours): candidate right 16%, favourite right 71%.

Regression `actual ~ market + candidate` (with intercept):

- 00-06h (n=302): market 0.98, candidate -0.00, intercept +1.0
- 06-10h (n=180): market 0.90, candidate 0.07, intercept +1.7
- 13-16h (n=105): market 0.95, candidate 0.05, intercept +0.0

G5 (leakage/parity unit tests): **PASS** (live parity check run separately: `python tools/check_feature_parity.py`)

## Outcome: **STOP**  (G1 ✓, G1b ✓, G4 ✗, G2 ✓, G3 ✗, G5 ✓)

Market-parity tier (informational): not reached (hours 0-9: candidate 1.59 vs market 0.87; Brier 0.524 vs 0.33).
