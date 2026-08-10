"""
KNYC Nowcaster — market-relative scorer
======================================
Grades the daily-high model on what it is actually worth against Kalshi's
KXHIGHNY brackets, rather than on distance from truth.

Why this exists: MAE and the 0-100 skill score in `score_predictions.py` measure
degrees, but the market pays on discrete buckets. A 0.4 F miss that straddles a
bracket boundary is a total loss; a 1.8 F miss inside a bracket is a clean win.
A distance metric cannot tell those apart, so a correct contrarian call at 10:1
can score mediocre.

What it reports
---------------
1. REBALANCED ROI (primary) — hold a position, re-price every hour, and pay the
   spread every time the model changes its mind. This is the only accounting
   that charges for mid-day flip-flopping.
2. FLAT ROI (baseline) — one independent 1-unit bet per hour, no position
   tracking. The GAP between this and (1) is the cost of the model's
   instability: flat +40% / rebalanced -5% means the signal is real but
   untradeable.
3. BRIER, model vs market — a proper scoring rule over the bracket ladder.
   Thousands of comparisons per fortnight instead of ~14 settled bets, so it
   converges while ROI is still noise.

Inputs
------
    market_log.csv     bracket ladder + prices, per hour   (written by the bot)
    nowcast_log.csv    pred_high / obs_high, per hour      (written by the bot)
    residual_model.csv residual distribution               (build_residuals.py)
Ground truth is the CLI daily high — the same NWS Climatological Report the
exchange settles on.

Usage
-----
    python market_scorer.py
    python market_scorer.py --start 2026-08-01 --end 2026-08-14
    python market_scorer.py --sweep          # kelly x band grid
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import score_predictions as sp
from build_residuals import QUANTILES, SEASON

NY_TZ = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent

MARKET_LOG = HERE / "market_log.csv"
NOWCAST_LOG = HERE / "nowcast_log.csv"
RESIDUAL_MODEL = HERE / "residual_model.csv"

# Strategy defaults (sweepable — see --sweep).
EDGE_THRESHOLD = 0.05   # minimum model_prob - ask before any interest
KELLY_MULT = 0.25       # quarter-Kelly: our probability is itself an estimate
NO_TRADE_BAND = 0.02    # skip rebalances smaller than this fraction of bankroll
MAX_EXPOSURE = 1.00     # cap total staked at this fraction of bankroll
FEE_RATE = 0.07         # Kalshi trading fee coefficient: rate * C * P * (1-P)


# ── Probability model ───────────────────────────────────────────────────────
class ResidualCDF:
    """Empirical CDF of the daily-high residual, binned by (season, local_hour).

    Built from `residual_model.csv`. Bins fall back to the pooled all-hours row
    when a specific (season, hour) bin was too sparse to persist.
    """

    def __init__(self, table: pd.DataFrame):
        self.qcols = [f"q{int(q * 1000):04d}" for q in QUANTILES]
        self.probs = np.array(QUANTILES, dtype=float)
        self.bins: dict[tuple[str, int], np.ndarray] = {}
        self.stats: dict[tuple[str, int], tuple[float, float]] = {}
        for _, r in table.iterrows():
            key = (str(r["season"]), int(r["local_hour"]))
            self.bins[key] = r[self.qcols].to_numpy(dtype=float)
            self.stats[key] = (float(r["mean"]), float(r["std"]))
        if ("ALL", -1) not in self.bins:
            raise SystemExit("residual_model.csv is missing the pooled ('ALL', -1) row")

    def _lookup(self, season: str, hour: int) -> tuple[np.ndarray, tuple[float, float]]:
        for key in ((season, hour), ("ALL", hour), ("ALL", -1)):
            if key in self.bins:
                return self.bins[key], self.stats[key]
        raise KeyError(season, hour)  # unreachable: ('ALL', -1) is guaranteed

    def cdf(self, x: float, center: float, season: str, hour: int) -> float:
        """P(daily high <= x) given a point prediction `center`."""
        if math.isinf(x):
            return 0.0 if x < 0 else 1.0
        q, (mean, std) = self._lookup(season, hour)
        # Pad with wide normal-ish tails so extreme brackets get small but
        # non-zero mass; clamping to the 1st/99th pctile would assign an
        # impossible-outcome probability of exactly 0 and make any loss there
        # look like a free bet.
        std = std if std and not math.isnan(std) else 1.0
        xp = np.concatenate(([mean - 4 * std], q, [mean + 4 * std]))
        fp = np.concatenate(([0.0005], self.probs, [0.9995]))
        order = np.argsort(xp)
        return float(np.clip(np.interp(x - center, xp[order], fp[order]), 0.0, 1.0))


def bracket_probs(ladder: pd.DataFrame, pred_high: float, obs_high: float | None,
                  season: str, hour: int, rcdf: ResidualCDF) -> np.ndarray:
    """P(settlement lands in each contract's interval).

    Two corrections that materially change the numbers:

    * Continuity: the CLI reports a whole degree, so a "89 to 90" contract pays
      on {89, 90} and its probability is F(90.5) - F(88.5). Dropping the half
      degree misprices every boundary bracket.
    * Obs floor: the day's high cannot come in below what has already been
      observed. Mass below `obs_high` is zeroed and the rest renormalized. The
      distribution is centered on max(pred, obs_high) — once the observed high
      exceeds the model's call, the model is known-wrong low and the observation
      is the better point estimate (this is the bot's own `reassessed` logic).
    """
    center = pred_high if obs_high is None or math.isnan(obs_high) else max(pred_high, obs_high)
    lo = ladder["lo"].to_numpy(dtype=float)
    hi = ladder["hi"].to_numpy(dtype=float)

    upper = np.array([rcdf.cdf(h + 0.5, center, season, hour) for h in hi])
    lower = np.array([rcdf.cdf(l - 0.5, center, season, hour) for l in lo])
    p = np.clip(upper - lower, 0.0, None)

    if obs_high is not None and not math.isnan(obs_high):
        floor = float(obs_high)
        # Kill any bracket that tops out below the observed high, and clip the
        # one bracket that straddles it down to its surviving upper piece.
        floor_cdf = rcdf.cdf(floor - 0.5, center, season, hour)
        p[hi < floor] = 0.0
        straddle = (lo <= floor) & (hi >= floor)
        if straddle.any():
            surviving = np.clip(upper - np.maximum(lower, floor_cdf), 0.0, None)
            p[straddle] = surviving[straddle]

    total = p.sum()
    return p / total if total > 0 else np.full(len(p), 1.0 / len(p))


# ── Fees ────────────────────────────────────────────────────────────────────
def kalshi_fee(contracts: float, price: float, rate: float = FEE_RATE) -> float:
    """Kalshi trading fee in dollars: rate * C * P * (1 - P).

    Peaks near 50c and shrinks toward the tails, so it is comparatively cheap on
    the longshot brackets this model tends to like. Verify the live coefficient
    against Kalshi's current fee schedule before trusting absolute P&L.
    """
    return abs(contracts) * rate * price * (1.0 - price)


# ── Data assembly ───────────────────────────────────────────────────────────
def load_joined(market_log: Path, nowcast_log: Path) -> pd.DataFrame:
    """market_log (one row per contract per hour) joined to the hour's prediction."""
    for p in (market_log, nowcast_log):
        if not p.exists():
            raise SystemExit(
                f"{p.name} not found. The bot writes it hourly; there is nothing "
                f"to score until it has been running.\n"
                f"Historical Kalshi prices cannot be recovered after settlement."
            )
    mk = pd.read_csv(market_log)
    nc = pd.read_csv(nowcast_log)

    mk["valid_utc"] = pd.to_datetime(mk["valid_t"], utc=True)
    nc["valid_utc"] = pd.to_datetime(nc["valid_t"], utc=True)
    nc = nc.drop_duplicates("valid_utc", keep="last")[
        ["valid_utc", "pred_high", "obs_high", "reassessed"]
    ]
    mk = mk.drop_duplicates(["valid_utc", "ticker"], keep="last")

    df = mk.merge(nc, on="valid_utc", how="inner")
    df["valid_local"] = df["valid_utc"].dt.tz_convert(NY_TZ)
    df["local_date"] = df["valid_local"].dt.date
    df["local_hour"] = df["valid_local"].dt.hour
    df["season"] = df["valid_local"].dt.month.map(SEASON)
    return df.sort_values(["valid_utc", "lo"]).reset_index(drop=True)


def settlement_truth(dates: list, verbose: bool = True) -> dict:
    """NY-local date -> official CLI high (the value Kalshi settles on).

    CLI only, deliberately: `score_predictions.resolve` blends CLI with the ASOS
    DSM, and on a day those differ by a degree the blend would book a win the
    exchange settled as a loss.
    """
    lo, hi = min(dates), max(dates)
    if verbose:
        print(f"Fetching CLI settlement {lo} -> {hi} ...")
    cli = sp.fetch_cli(datetime.combine(lo, datetime.min.time()),
                       datetime.combine(hi, datetime.min.time()) + timedelta(days=1))
    if cli.empty:
        return {}
    return {d: float(v) for d, v in zip(cli["local_date"], cli["cli_max"])}


# ── Strategies ──────────────────────────────────────────────────────────────
def run_rebalanced(df: pd.DataFrame, truth: dict, rcdf: ResidualCDF, *,
                   kelly: float = KELLY_MULT, band: float = NO_TRADE_BAND,
                   thresh: float = EDGE_THRESHOLD, fee_rate: float = FEE_RATE,
                   bankroll: float = 1000.0) -> dict:
    """Hold positions, re-price hourly, pay the spread on every change of mind."""
    cash = bankroll
    equity_curve, trades, per_day = [], [], []
    fees_paid = spread_paid = turnover = 0.0

    for day, day_rows in df.groupby("local_date"):
        if day not in truth:
            continue  # unsettled
        actual = truth[day]
        positions: dict[str, float] = {}
        day_start_cash = cash

        for _, hour_rows in day_rows.groupby("valid_utc"):
            ladder = hour_rows.reset_index(drop=True)
            p = bracket_probs(ladder, ladder["pred_high"].iloc[0],
                              ladder["obs_high"].iloc[0], ladder["season"].iloc[0],
                              int(ladder["local_hour"].iloc[0]), rcdf)

            # Mark to market on the bid (what the position could be sold for).
            held = sum(positions.get(t, 0.0) * b
                       for t, b in zip(ladder["ticker"], ladder["yes_bid"]))
            equity = cash + held

            # Independent fractional Kelly per contract, then cap gross exposure.
            targets: dict[str, float] = {}
            want_dollars = {}
            for i, row in ladder.iterrows():
                ask = row["yes_ask"]
                if not (0 < ask < 1.0):     # empty/one-sided book: untradeable
                    continue
                edge = p[i] - ask
                if edge <= thresh:
                    continue
                f = (p[i] - ask) / (1.0 - ask)          # full Kelly fraction
                want_dollars[row["ticker"]] = max(0.0, kelly * f) * equity
            gross = sum(want_dollars.values())
            if gross > MAX_EXPOSURE * equity and gross > 0:
                scale = MAX_EXPOSURE * equity / gross
                want_dollars = {k: v * scale for k, v in want_dollars.items()}
            for i, row in ladder.iterrows():
                d = want_dollars.get(row["ticker"], 0.0)
                targets[row["ticker"]] = d / row["yes_ask"] if d > 0 else 0.0

            for _, row in ladder.iterrows():
                t = row["ticker"]
                cur, tgt = positions.get(t, 0.0), targets.get(t, 0.0)
                delta = tgt - cur
                if abs(delta) < 1e-9:
                    continue  # position already correct — not a trade
                px = row["yes_ask"] if delta > 0 else row["yes_bid"]
                # A book with no bid cannot be sold into and a 1.00 ask cannot be
                # bought; either way there is no fill to simulate.
                if not (0 < px < 1.0):
                    continue
                # Hysteresis: ignore rebalances too small to be worth the spread.
                if abs(delta) * px < band * equity:
                    continue
                mid = (row["yes_bid"] + row["yes_ask"]) / 2.0
                fee = kalshi_fee(delta, px, fee_rate)
                cash -= delta * px + fee
                spread_paid += abs(delta) * abs(px - mid)
                fees_paid += fee
                turnover += abs(delta) * px
                positions[t] = cur + delta
                trades.append({"date": day, "ticker": t, "delta": delta, "px": px})

            # Re-mark after trading: `held` above was the pre-trade value, used
            # for sizing. Recording it post-trade would understate the curve.
            held_post = sum(positions.get(t, 0.0) * b
                            for t, b in zip(ladder["ticker"], ladder["yes_bid"]))
            equity_curve.append({"valid_utc": ladder["valid_utc"].iloc[0],
                                 "local_date": day, "equity": cash + held_post})

        # Settlement: each contract pays $1 if the CLI high lands in its interval.
        last = day_rows.groupby("ticker").last().reset_index()
        for _, row in last.iterrows():
            q = positions.get(row["ticker"], 0.0)
            if q:
                cash += q * (1.0 if row["lo"] <= actual <= row["hi"] else 0.0)
        positions.clear()
        per_day.append({"local_date": day, "actual": actual,
                        "pnl": cash - day_start_cash,
                        "ret_pct": (cash - day_start_cash) / day_start_cash * 100})
        equity_curve.append({"valid_utc": day_rows["valid_utc"].max(),
                             "local_date": day, "equity": cash})

    eq = pd.DataFrame(equity_curve)
    out = {"final": cash, "start": bankroll,
           "roi_pct": (cash / bankroll - 1.0) * 100,
           "n_trades": len(trades), "turnover": turnover,
           "fees_paid": fees_paid, "spread_paid": spread_paid,
           "per_day": pd.DataFrame(per_day), "equity": eq}
    if not eq.empty:
        peak = eq["equity"].cummax()
        out["max_drawdown_pct"] = float(((eq["equity"] - peak) / peak).min() * 100)
    return out


def run_flat(df: pd.DataFrame, truth: dict, rcdf: ResidualCDF, *,
             thresh: float = EDGE_THRESHOLD) -> dict:
    """One independent 1-unit bet per hour on the single best-edge contract."""
    recs = []
    for (day, _), hour_rows in df.groupby(["local_date", "valid_utc"]):
        if day not in truth:
            continue
        ladder = hour_rows.reset_index(drop=True)
        p = bracket_probs(ladder, ladder["pred_high"].iloc[0],
                          ladder["obs_high"].iloc[0], ladder["season"].iloc[0],
                          int(ladder["local_hour"].iloc[0]), rcdf)
        tradeable = [(p[i] - r["yes_ask"], i) for i, r in ladder.iterrows()
                     if 0 < r["yes_ask"] < 1.0]
        if not tradeable:
            continue
        edge, i = max(tradeable)
        if edge <= thresh:
            continue
        row = ladder.iloc[i]
        win = bool(row["lo"] <= truth[day] <= row["hi"])
        recs.append({
            "local_date": day, "local_hour": int(row["local_hour"]),
            "ticker": row["ticker"], "ask": row["yes_ask"], "model_p": p[i],
            "edge": edge, "win": win,
            "pnl": (1.0 / row["yes_ask"] - 1.0) if win else -1.0,
        })
    r = pd.DataFrame(recs)
    if r.empty:
        return {"n_bets": 0}
    return {"n_bets": len(r), "staked": float(len(r)),
            "pnl": float(r["pnl"].sum()), "roi_pct": float(r["pnl"].mean() * 100),
            "hit_pct": float(r["win"].mean() * 100),
            "mean_ask": float(r["ask"].mean()), "bets": r}


def brier_scores(df: pd.DataFrame, truth: dict, rcdf: ResidualCDF) -> dict:
    """Multi-class Brier for model vs market, over the mutually-exclusive ladder.

    Market probabilities are mid prices normalized to sum to 1, which strips the
    bid/ask spread so the comparison is prob-vs-prob rather than prob-vs-price.
    Lower is better.
    """
    rows = []
    for (day, _), hour_rows in df.groupby(["local_date", "valid_utc"]):
        if day not in truth:
            continue
        ladder = hour_rows.reset_index(drop=True)
        p = bracket_probs(ladder, ladder["pred_high"].iloc[0],
                          ladder["obs_high"].iloc[0], ladder["season"].iloc[0],
                          int(ladder["local_hour"].iloc[0]), rcdf)
        mid = ((ladder["yes_bid"] + ladder["yes_ask"]) / 2.0).to_numpy()
        if mid.sum() <= 0:
            continue
        mkt = mid / mid.sum()
        outcome = ((ladder["lo"] <= truth[day]) & (truth[day] <= ladder["hi"])).to_numpy(float)
        if outcome.sum() != 1:
            continue  # ladder did not cover the settlement value
        rows.append({"local_date": day, "local_hour": int(ladder["local_hour"].iloc[0]),
                     "model": float(((p - outcome) ** 2).sum()),
                     "market": float(((mkt - outcome) ** 2).sum())})
    b = pd.DataFrame(rows)
    if b.empty:
        return {"n": 0}
    return {"n": len(b), "model": float(b["model"].mean()),
            "market": float(b["market"].mean()),
            "by_hour": b.groupby("local_hour")[["model", "market"]].mean().reset_index()}


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for a mean. With ~10:1 payoffs this is usually humbling."""
    if len(values) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── Report ──────────────────────────────────────────────────────────────────
def print_report(reb: dict, flat: dict, brier: dict, df: pd.DataFrame) -> None:
    line = "=" * 70
    print(f"\n{line}\nKNYC NOWCASTER -- MARKET-RELATIVE SCORECARD\n{line}")
    days = df["local_date"].nunique()
    print(f"Hours scored: {df['valid_utc'].nunique()}  |  days: {days}")

    print(f"\n{'-' * 70}\n1. REBALANCED  (position held, spread paid on every change of mind)\n{'-' * 70}")
    print(f"  Bankroll   : {reb['start']:,.0f} -> {reb['final']:,.2f}")
    print(f"  ROI        : {reb['roi_pct']:+.1f}%")
    if "max_drawdown_pct" in reb:
        print(f"  Max drawdn : {reb['max_drawdown_pct']:.1f}%")
    print(f"  Trades     : {reb['n_trades']}   turnover ${reb['turnover']:,.0f}")
    print(f"  Cost of trading: ${reb['spread_paid']:,.2f} spread + "
          f"${reb['fees_paid']:,.2f} fees")

    print(f"\n{'-' * 70}\n2. FLAT BASELINE  (1 unit/hour, no position tracking)\n{'-' * 70}")
    if flat.get("n_bets", 0) == 0:
        print("  No bets cleared the edge threshold.")
    else:
        lo, hi = bootstrap_ci(flat["bets"]["pnl"].to_numpy())
        print(f"  Bets       : {flat['n_bets']}   hit rate {flat['hit_pct']:.0f}%   "
              f"mean ask {flat['mean_ask']:.2f}")
        print(f"  ROI        : {flat['roi_pct']:+.1f}%   "
              f"95% CI [{lo * 100:+.0f}%, {hi * 100:+.0f}%]")
        if lo < 0 < hi:
            print("               ^ CI spans zero: not yet distinguishable from luck")

    if flat.get("n_bets", 0) and "roi_pct" in reb:
        gap = flat["roi_pct"] - reb["roi_pct"]
        print(f"\n  >> COST OF INSTABILITY: {gap:+.1f} pts "
              f"(flat {flat['roi_pct']:+.1f}% vs rebalanced {reb['roi_pct']:+.1f}%)")
        if gap > 15:
            print("     Signal is real but the model changes its mind too often to trade.")

    print(f"\n{'-' * 70}\n3. BRIER  (lower is better; converges faster than ROI)\n{'-' * 70}")
    if brier.get("n", 0) == 0:
        print("  Not enough settled ladder-hours.")
    else:
        verdict = "MODEL" if brier["model"] < brier["market"] else "MARKET"
        print(f"  n={brier['n']}   model {brier['model']:.4f} vs "
              f"market {brier['market']:.4f}   -> {verdict} better calibrated")

    if not reb["per_day"].empty:
        print(f"\n{'-' * 70}\nPER-DAY\n{'-' * 70}")
        for _, r in reb["per_day"].iterrows():
            print(f"  {r['local_date']}  actual {r['actual']:.0f}F   "
                  f"P&L {r['pnl']:>+9.2f}  ({r['ret_pct']:+.1f}%)")
    print(line)


def sweep(df: pd.DataFrame, truth: dict, rcdf: ResidualCDF) -> None:
    """Kelly x no-trade-band grid. A result that only works at one exact setting
    is overfit; the grid makes that visible."""
    print(f"\n{'-' * 70}\nPARAMETER SWEEP (ROI %)\n{'-' * 70}")
    bands = [0.00, 0.01, 0.02, 0.05, 0.10]
    kellys = [0.10, 0.25, 0.50, 1.00]
    print("  kelly \\ band " + "".join(f"{b:>9.2f}" for b in bands))
    for k in kellys:
        cells = []
        for b in bands:
            r = run_rebalanced(df, truth, rcdf, kelly=k, band=b)
            cells.append(f"{r['roi_pct']:>+9.1f}")
        print(f"  {k:>11.2f} " + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the daily-high model against Kalshi.")
    ap.add_argument("--market-log", default=str(MARKET_LOG))
    ap.add_argument("--nowcast-log", default=str(NOWCAST_LOG))
    ap.add_argument("--residuals", default=str(RESIDUAL_MODEL))
    ap.add_argument("--start", help="YYYY-MM-DD (NY local) inclusive")
    ap.add_argument("--end", help="YYYY-MM-DD (NY local) inclusive")
    ap.add_argument("--kelly", type=float, default=KELLY_MULT)
    ap.add_argument("--band", type=float, default=NO_TRADE_BAND)
    ap.add_argument("--edge", type=float, default=EDGE_THRESHOLD)
    ap.add_argument("--sweep", action="store_true", help="run the kelly x band grid")
    args = ap.parse_args()

    resid_path = Path(args.residuals)
    if not resid_path.exists():
        raise SystemExit(f"{resid_path.name} not found. Run: python build_residuals.py")
    rcdf = ResidualCDF(pd.read_csv(resid_path))

    df = load_joined(Path(args.market_log), Path(args.nowcast_log))
    if args.start:
        df = df[df["local_date"] >= pd.to_datetime(args.start).date()]
    if args.end:
        df = df[df["local_date"] <= pd.to_datetime(args.end).date()]
    if df.empty:
        raise SystemExit("No joined market/prediction rows in range.")

    truth = settlement_truth(sorted(df["local_date"].unique()))
    if not truth:
        raise SystemExit("No CLI settlement available yet for these dates.")

    reb = run_rebalanced(df, truth, rcdf, kelly=args.kelly, band=args.band, thresh=args.edge)
    flat = run_flat(df, truth, rcdf, thresh=args.edge)
    brier = brier_scores(df, truth, rcdf)
    print_report(reb, flat, brier, df)
    if args.sweep:
        sweep(df, truth, rcdf)


if __name__ == "__main__":
    main()
