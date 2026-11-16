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
1. REBALANCED ROI — hold a position, re-price every hour, and pay the spread
   every time the model changes its mind.
2. FLAT ROI — one independent 1-unit bet per hour on the best-edge contract,
   net of fees, next to the MARKET-FAVOURITE baseline (buy the highest-mid
   contract every hour). A model that cannot beat "buy the favourite" has no
   edge, whatever its own ROI says.
3. MODEL vs FAVOURITE — the hours where the model's top bracket differs from
   the market's, and who was right. This is where any edge has to come from.
4. BRIER, model vs market, overall and by hour block — a proper scoring rule
   over the ladder. Converges far faster than ROI.

All confidence intervals resample whole days: hours within a day share one
settlement, so a per-bet bootstrap would be several times too narrow.

Inputs
------
    market_log.csv     bracket ladder + prices, per hour   (written by the bot)
    nowcast_log.csv    pred_high / obs_high, per hour      (written by the bot)
    offset_pmf.csv     observation-conditioned offset PMF  (build_residuals.py)
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

import kalshi_market as km
import score_predictions as sp
from build_residuals import K_COLS, K_MAX, K_MIN, gap_bucket

NY_TZ = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent

MARKET_LOG = HERE / "market_log.csv"
NOWCAST_LOG = HERE / "nowcast_log.csv"
OFFSET_PMF = HERE / "offset_pmf.csv"

# Strategy defaults (sweepable — see --sweep).
EDGE_THRESHOLD = 0.05   # minimum model_prob - ask before any interest
KELLY_MULT = 0.25       # quarter-Kelly: our probability is itself an estimate
NO_TRADE_BAND = 0.02    # skip rebalances smaller than this fraction of bankroll
MAX_EXPOSURE = 1.00     # cap total staked at this fraction of bankroll
FEE_RATE = 0.07         # Kalshi trading fee coefficient: rate * C * P * (1-P)


# ── Probability model ───────────────────────────────────────────────────────
class OffsetPMF:
    """P(settlement - round(re) = k), by NY-local hour and model-minus-observed
    gap (see build_residuals.py). Falls back to the hour's pooled row when an
    (hour, gap) cell was too sparse to persist, or when there is no obs yet."""

    def __init__(self, table: pd.DataFrame):
        self.ks = np.arange(K_MIN, K_MAX + 1)
        self.cells: dict[tuple[int, str], np.ndarray] = {}
        for _, r in table.iterrows():
            w = r[K_COLS].to_numpy(dtype=float)
            self.cells[(int(r["local_hour"]), str(r["gap_bucket"]))] = w / w.sum()
        missing = [h for h in range(24) if (h, "ALL") not in self.cells]
        if missing:
            raise SystemExit(f"offset_pmf.csv lacks pooled rows for hours {missing}")

    def lookup(self, hour: int, gap: str | None) -> np.ndarray:
        return self.cells.get((hour, gap)) if gap is not None and (hour, gap) in self.cells \
            else self.cells[(hour, "ALL")]


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def bracket_probs(ladder: pd.DataFrame, pred_high: float, obs_high: float | None,
                  hour: int, pmf: OffsetPMF) -> np.ndarray:
    """P(settlement lands in each contract's interval).

    Centred on re = max(pred, obs) rounded half-up, with the offset k drawn from
    the PMF for this hour and how far the model sits above the observed high.
    Each outcome is floored at the rounded observed high (the day cannot settle
    below what has been seen). A small uniform floor (0.002) keeps every
    contract possible, so a loss is never priced as impossible.
    """
    obs = None if _isnan(obs_high) else float(obs_high)
    re = pred_high if obs is None else max(pred_high, obs)
    base = km.round_half_up(re)
    floor_int = km.round_half_up(obs) if obs is not None else None
    w = pmf.lookup(hour, gap_bucket(None if obs is None else pred_high - obs))

    lo = ladder["lo"].to_numpy(dtype=float)
    hi = ladder["hi"].to_numpy(dtype=float)
    p = np.zeros(len(ladder))
    for k, wk in zip(pmf.ks, w):
        if wk <= 0:
            continue
        v = base + int(k)
        if floor_int is not None:
            v = max(v, floor_int)
        idx = np.flatnonzero((lo <= v) & (v <= hi))
        if idx.size:
            p[idx[0]] += wk
    p += 0.002
    return p / p.sum()


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
    # keep="first": the first write is what existed when the forecast was
    # posted. Pre-2026-11 logs re-polled some hours; the later rows carry
    # prices and predictions from after the posted moment (look-ahead).
    nc = nc.drop_duplicates("valid_utc", keep="first")[
        ["valid_utc", "pred_high", "obs_high", "reassessed"]
    ]
    mk = mk.drop_duplicates(["valid_utc", "ticker"], keep="first")

    df = mk.merge(nc, on="valid_utc", how="inner")
    df["valid_local"] = df["valid_utc"].dt.tz_convert(NY_TZ)
    df["local_date"] = df["valid_local"].dt.date
    df["local_hour"] = df["valid_local"].dt.hour
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
def run_rebalanced(df: pd.DataFrame, truth: dict, pmf: OffsetPMF, *,
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
                              ladder["obs_high"].iloc[0],
                              int(ladder["local_hour"].iloc[0]), pmf)

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


def run_flat(df: pd.DataFrame, truth: dict, pmf: OffsetPMF, *,
             thresh: float = EDGE_THRESHOLD) -> dict:
    """One independent 1-unit bet per hour on the single best-edge contract."""
    recs = []
    for (day, _), hour_rows in df.groupby(["local_date", "valid_utc"]):
        if day not in truth:
            continue
        ladder = hour_rows.reset_index(drop=True)
        p = bracket_probs(ladder, ladder["pred_high"].iloc[0],
                          ladder["obs_high"].iloc[0],
                          int(ladder["local_hour"].iloc[0]), pmf)
        tradeable = [(p[i] - r["yes_ask"], i) for i, r in ladder.iterrows()
                     if 0 < r["yes_ask"] < 1.0]
        if not tradeable:
            continue
        edge, i = max(tradeable)
        if edge <= thresh:
            continue
        row = ladder.iloc[i]
        recs.append(_flat_bet(day, row, truth[day], model_p=p[i], edge=edge))
    return _flat_summary(recs)


def _flat_bet(day, row, actual: float, **extra) -> dict:
    """$1 bought at the ask, net of the Kalshi fee on 1/ask contracts."""
    ask = float(row["yes_ask"])
    win = bool(row["lo"] <= actual <= row["hi"])
    fee = kalshi_fee(1.0 / ask, ask)
    return {"local_date": day, "local_hour": int(row["local_hour"]),
            "ticker": row["ticker"], "ask": ask, "win": win,
            "pnl": ((1.0 / ask - 1.0) if win else -1.0) - fee, **extra}


def _flat_summary(recs: list[dict]) -> dict:
    r = pd.DataFrame(recs)
    if r.empty:
        return {"n_bets": 0}
    return {"n_bets": len(r), "staked": float(len(r)),
            "pnl": float(r["pnl"].sum()), "roi_pct": float(r["pnl"].mean() * 100),
            "hit_pct": float(r["win"].mean() * 100),
            "mean_ask": float(r["ask"].mean()), "bets": r}


def run_favourite(df: pd.DataFrame, truth: dict) -> dict:
    """Baseline: every hour, $1 on the contract with the highest mid price."""
    recs = []
    for (day, _), hour_rows in df.groupby(["local_date", "valid_utc"]):
        if day not in truth:
            continue
        ladder = hour_rows.reset_index(drop=True)
        mid = (ladder["yes_bid"] + ladder["yes_ask"]) / 2.0
        row = ladder.loc[mid.idxmax()]
        if not (0 < row["yes_ask"] < 1.0):
            continue
        recs.append(_flat_bet(day, row, truth[day]))
    return _flat_summary(recs)


def model_vs_favourite(df: pd.DataFrame, truth: dict, pmf: OffsetPMF) -> dict:
    """Hours where the model's most likely bracket is not the market favourite."""
    agree = disagree = model_right = fav_right = 0
    for (day, _), hour_rows in df.groupby(["local_date", "valid_utc"]):
        if day not in truth:
            continue
        ladder = hour_rows.reset_index(drop=True)
        p = bracket_probs(ladder, ladder["pred_high"].iloc[0], ladder["obs_high"].iloc[0],
                          int(ladder["local_hour"].iloc[0]), pmf)
        mid = ((ladder["yes_bid"] + ladder["yes_ask"]) / 2.0).to_numpy()
        m_i, f_i = int(np.argmax(p)), int(np.argmax(mid))
        if m_i == f_i:
            agree += 1
            continue
        disagree += 1
        a = truth[day]
        model_right += bool(ladder.at[m_i, "lo"] <= a <= ladder.at[m_i, "hi"])
        fav_right += bool(ladder.at[f_i, "lo"] <= a <= ladder.at[f_i, "hi"])
    return {"agree": agree, "disagree": disagree,
            "model_right_pct": model_right / disagree * 100 if disagree else math.nan,
            "fav_right_pct": fav_right / disagree * 100 if disagree else math.nan}


def brier_scores(df: pd.DataFrame, truth: dict, pmf: OffsetPMF) -> dict:
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
                          ladder["obs_high"].iloc[0],
                          int(ladder["local_hour"].iloc[0]), pmf)
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
            "by_hour": (b.groupby("local_hour")
                          .agg(n=("model", "size"), model=("model", "mean"),
                               market=("market", "mean"))
                          .reset_index())}


def bootstrap_ci(bets: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI for mean P&L per bet, resampling whole days (cluster bootstrap).

    Bets within a day share one settlement and are strongly correlated; a
    per-bet bootstrap treats them as independent and is far too narrow."""
    by_day = bets.groupby("local_date")["pnl"].agg(["sum", "count"])
    if len(by_day) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    sums, counts = by_day["sum"].to_numpy(), by_day["count"].to_numpy()
    idx = rng.integers(0, len(by_day), size=(n_boot, len(by_day)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── Report ──────────────────────────────────────────────────────────────────
HOUR_BLOCKS = [(0, 5), (6, 9), (10, 12), (13, 15), (16, 18), (19, 23)]


def _print_flat(label: str, flat: dict) -> tuple[float, float] | None:
    if flat.get("n_bets", 0) == 0:
        print(f"  {label:<11}: no bets")
        return None
    lo, hi = bootstrap_ci(flat["bets"])
    print(f"  {label:<11}: {flat['n_bets']:>5} bets  hit {flat['hit_pct']:>3.0f}%  "
          f"mean ask {flat['mean_ask']:.2f}  ROI {flat['roi_pct']:+6.1f}%  "
          f"95% CI [{lo * 100:+.0f}%, {hi * 100:+.0f}%]")
    return lo, hi


def print_report(reb: dict, flat: dict, fav: dict, vs: dict, brier: dict,
                 df: pd.DataFrame) -> None:
    line = "=" * 70
    print(f"\n{line}\nKNYC NOWCASTER -- MARKET-RELATIVE SCORECARD\n{line}")
    days = df["local_date"].nunique()
    print(f"Hours scored: {df['valid_utc'].nunique()}  |  days: {days}")
    print("CIs resample whole days. With this few days, ROI is mostly noise.")

    print(f"\n{'-' * 70}\n1. REBALANCED  (position held, spread paid on every change of mind)\n{'-' * 70}")
    print(f"  Bankroll   : {reb['start']:,.0f} -> {reb['final']:,.2f}")
    print(f"  ROI        : {reb['roi_pct']:+.1f}%")
    if "max_drawdown_pct" in reb:
        print(f"  Max drawdn : {reb['max_drawdown_pct']:.1f}%")
    print(f"  Trades     : {reb['n_trades']}   turnover ${reb['turnover']:,.0f}")
    print(f"  Cost of trading: ${reb['spread_paid']:,.2f} spread + "
          f"${reb['fees_paid']:,.2f} fees")

    print(f"\n{'-' * 70}\n2. FLAT, NET OF FEES  (1 unit/hour at the ask)\n{'-' * 70}")
    ci = _print_flat("Model edge", flat)
    _print_flat("Favourite", fav)
    if ci is not None:
        if ci[0] > 0:
            print("  ^ model CI is above zero")
        elif ci[1] < 0:
            print("  ^ model CI is below zero: the edge bets lose money")
        else:
            print("  ^ model CI spans zero: not distinguishable from luck")

    print(f"\n{'-' * 70}\n3. MODEL vs MARKET FAVOURITE\n{'-' * 70}")
    if vs["disagree"]:
        print(f"  Agree {vs['agree']} hours, disagree {vs['disagree']} hours. When they disagree:")
        print(f"    model's bracket right {vs['model_right_pct']:.0f}%   "
              f"favourite right {vs['fav_right_pct']:.0f}%")
    else:
        print(f"  Model and favourite agreed on all {vs['agree']} hours.")

    print(f"\n{'-' * 70}\n4. BRIER  (lower is better; uniform over 6 brackets = 0.833)\n{'-' * 70}")
    if brier.get("n", 0) == 0:
        print("  Not enough settled ladder-hours.")
    else:
        verdict = "MODEL" if brier["model"] < brier["market"] else "MARKET"
        print(f"  all hours  n={brier['n']:>5}  model {brier['model']:.3f}  "
              f"market {brier['market']:.3f}  -> {verdict} better")
        bh = brier["by_hour"]
        for a, b in HOUR_BLOCKS:
            g = bh[(bh["local_hour"] >= a) & (bh["local_hour"] <= b)]
            if g.empty:
                continue
            n = int(g["n"].sum())
            m = float((g["model"] * g["n"]).sum() / n)
            k = float((g["market"] * g["n"]).sum() / n)
            print(f"  {a:02d}-{b:02d}h    n={n:>5}  model {m:.3f}  market {k:.3f}")

    if not reb["per_day"].empty:
        print(f"\n{'-' * 70}\nPER-DAY (rebalanced)\n{'-' * 70}")
        for _, r in reb["per_day"].iterrows():
            print(f"  {r['local_date']}  actual {r['actual']:.0f}F   "
                  f"P&L {r['pnl']:>+9.2f}  ({r['ret_pct']:+.1f}%)")
    print(line)


def sweep(df: pd.DataFrame, truth: dict, pmf: OffsetPMF) -> None:
    """Kelly x no-trade-band grid. A result that only works at one exact setting
    is overfit; the grid makes that visible."""
    print(f"\n{'-' * 70}\nPARAMETER SWEEP (ROI %)\n{'-' * 70}")
    bands = [0.00, 0.01, 0.02, 0.05, 0.10]
    kellys = [0.10, 0.25, 0.50, 1.00]
    print("  kelly \\ band " + "".join(f"{b:>9.2f}" for b in bands))
    for k in kellys:
        cells = []
        for b in bands:
            r = run_rebalanced(df, truth, pmf, kelly=k, band=b)
            cells.append(f"{r['roi_pct']:>+9.1f}")
        print(f"  {k:>11.2f} " + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the daily-high model against Kalshi.")
    ap.add_argument("--market-log", default=str(MARKET_LOG))
    ap.add_argument("--nowcast-log", default=str(NOWCAST_LOG))
    ap.add_argument("--pmf", default=str(OFFSET_PMF))
    ap.add_argument("--start", help="YYYY-MM-DD (NY local) inclusive")
    ap.add_argument("--end", help="YYYY-MM-DD (NY local) inclusive")
    ap.add_argument("--kelly", type=float, default=KELLY_MULT)
    ap.add_argument("--band", type=float, default=NO_TRADE_BAND)
    ap.add_argument("--edge", type=float, default=EDGE_THRESHOLD)
    ap.add_argument("--sweep", action="store_true", help="run the kelly x band grid")
    args = ap.parse_args()

    pmf_path = Path(args.pmf)
    if not pmf_path.exists():
        raise SystemExit(f"{pmf_path.name} not found. Run: python build_residuals.py")
    pmf = OffsetPMF(pd.read_csv(pmf_path))

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

    reb = run_rebalanced(df, truth, pmf, kelly=args.kelly, band=args.band, thresh=args.edge)
    flat = run_flat(df, truth, pmf, thresh=args.edge)
    fav = run_favourite(df, truth)
    vs = model_vs_favourite(df, truth, pmf)
    brier = brier_scores(df, truth, pmf)
    print_report(reb, flat, fav, vs, brier, df)
    if args.sweep:
        sweep(df, truth, pmf)


if __name__ == "__main__":
    main()
