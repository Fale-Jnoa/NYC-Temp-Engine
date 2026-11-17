# Gate report: guidance_v1 (variant C2)

## 6.1 Test period 2025-01-01 to 2026-09-23 (622 days, 11,993 rows)

| Block | n | Candidate MAE | Production MAE | Change | Cand bias | Prod bias | Raw GFS n_x MAE | Reassessed cand | Reassessed prod |
|---|---|---|---|---|---|---|---|---|---|
| 00-05 | 3381.0 | 2.01 | 3.17 | -36% | +0.62 | +0.24 | 2.19 | 2.00 | 3.16 |
| 06-09 | 1862.0 | 1.83 | 2.50 | -27% | +0.65 | +0.36 | 2.18 | 1.81 | 2.47 |
| 10-12 | 1171.0 | 1.58 | 1.69 | -7% | +0.60 | +0.35 | 2.23 | 1.52 | 1.61 |
| 13-15 | 1334.0 | 1.14 | 1.08 | +6% | +0.43 | +0.26 | 2.17 | 1.02 | 0.92 |
| 16-18 | 1559.0 | 0.98 | 1.02 | -4% | +0.14 | +0.21 | 2.10 | 0.79 | 0.79 |
| 19-23 | 2686.0 | 0.81 | 1.23 | -34% | +0.14 | +0.11 | 2.07 | 0.13 | 0.14 |

G1 (>=20% lower 00-05 and 06-09, >=10% 10-12, <=+0.03 F later): **FAIL**
G1b (hours 0-9: candidate 1.95 vs raw GFS n_x 2.19, needs >= 0.15 lower): **PASS**

## 6.4 Degradation (test period, guidance forced missing)

| Block | Production | GFS missing | LAMP missing | Both missing |
|---|---|---|---|---|
| 00-05 | 3.17 | 2.76 | 2.20 | 5.08 |
| 06-09 | 2.50 | 2.46 | 1.96 | 4.36 |
| 10-12 | 1.69 | 2.03 | 1.58 | 2.64 |
| 13-15 | 1.08 | 1.54 | 1.15 | 1.82 |
| 16-18 | 1.02 | 1.25 | 1.02 | 1.68 |
| 19-23 | 1.23 | 1.18 | 0.89 | 1.56 |

G4 (both missing within 0.15 F of production in every block): **FAIL**

## 6.2 Live holdout (54 settled days, 1,066 hours with market data)

| Block | n | Candidate | Production replay | Production as logged | Market |
|---|---|---|---|---|---|
| 00-05 | 302.0 | 1.51 | 2.53 | 2.62 | 0.87 |
| 06-09 | 180.0 | 1.54 | 2.02 | 2.17 | 0.86 |
| 10-12 | 110.0 | 1.28 | 1.63 | 1.67 | 0.76 |
| 13-15 | 105.0 | 1.13 | 1.18 | 1.20 | 0.54 |
| 16-18 | 135.0 | 0.80 | 0.88 | 0.88 | 0.53 |
| 19-23 | 234.0 | 0.73 | 1.14 | 1.10 | 0.52 |

G2 (hours 0-11: candidate minus production -0.77 F, 95% CI [-1.29, -0.29]; candidate hours 0-9 1.52 F): **PASS**

Market vs candidate MAE at selected hours (review table: market 0.86/0.81/0.83/0.77/0.67/0.52/0.56/0.54):

| Hour | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 |
|---|---|---|---|---|---|---|---|---|
| Market | 0.90 | 0.83 | 0.84 | 0.79 | 0.75 | 0.50 | 0.55 | 0.53 |
| Candidate | 1.45 | 1.51 | 1.54 | 1.47 | 1.21 | 1.18 | 0.81 | 0.78 |
| Production replay | 3.01 | 2.44 | 2.25 | 1.81 | 1.39 | 0.97 | 0.91 | 1.19 |

## 6.3 Ladder Brier on the holdout (n=1066 hours; lower is better)

| Block | Candidate | Production replay | Market |
|---|---|---|---|
| all | 0.508 | 0.611 | 0.275 |
| 00-05 | 0.687 | 0.861 | 0.453 |
| 06-09 | 0.688 | 0.826 | 0.467 |
| 10-12 | 0.553 | 0.644 | 0.337 |
| 13-15 | 0.482 | 0.476 | 0.217 |
| 16-18 | 0.363 | 0.384 | 0.064 |
| 19-23 | 0.215 | 0.301 | 0.017 |

G3 (overall <= 0.48 and >= 0.05 below production; 00-05 <= 0.74; 06-09 <= 0.70; 19-23 <= 0.20): **FAIL**

Model vs market favourite where they disagree (462 hours): candidate right 22%, favourite right 67%.

Regression `actual ~ market + candidate` (with intercept):

- 00-06h (n=302): market 0.96, candidate 0.01, intercept +1.1
- 06-10h (n=180): market 0.91, candidate 0.06, intercept +1.6
- 13-16h (n=105): market 0.89, candidate 0.10, intercept +0.3

G5 (leakage/parity unit tests): **PASS** (live parity check run separately: `python tools/check_feature_parity.py`)

## Outcome: **STOP**  (G1 ✗, G1b ✓, G4 ✗, G2 ✓, G3 ✗, G5 ✓)

Market-parity tier (informational): not reached (hours 0-9: candidate 1.52 vs market 0.87; Brier 0.508 vs 0.33).
