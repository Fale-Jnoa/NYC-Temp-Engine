"""
Daily performance chart
=======================
Renders the plain-language "how did the forecast do yesterday" picture posted to
#daily-performance each morning alongside the scorecard:

    orange  what the temperature actually did, hour by hour
    blue    what the model thought the day's high would be, at each of those hours
    green   the high the day actually reached

Written for a general audience, so it avoids MAE/bias/skill language entirely and
states the outcome in a sentence. Uses the Agg backend because the deploy target
is a headless VM with no display.

`render` returns the PNG in memory; the bot uploads it straight to Discord and
nothing is written to disk. Only the manual command below saves a file.

Usage
-----
    python daily_chart.py               # yesterday, saves daily_chart.png to preview
    python daily_chart.py 2026-09-06
"""
from __future__ import annotations

import io
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")  # headless: must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

NY_TZ = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent

PREDICTION = "#2a78d6"   # blue
TEMPERATURE = "#eb6834"  # orange
ACTUAL_HIGH = "#00875a"  # green
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e1e0d9"
SURFACE = "#ffffff"


def _clock(h: float) -> str:
    """0 -> '12am', 12 -> 'noon', 15 -> '3pm' — hours as people say them."""
    h = int(round(h)) % 24
    if h == 0:
        return "12am"
    if h == 12:
        return "noon"
    return f"{h}am" if h < 12 else f"{h - 12}pm"


def render(target_date: date, preds: pd.DataFrame, obs: pd.DataFrame,
           actual_high: float | None) -> io.BytesIO:
    """Draw the day's chart and return it as an in-memory PNG, rewound to the start.

    preds  rows from nowcast_log.csv for this NY-local date (needs local_hour,
           pred_high; tmpf is used only if `obs` is unusable).
    obs    hourly observations covering the day (needs valid, tmpf), used for the
           temperature curve so it stays continuous even across bot downtime.
    """
    p = preds.dropna(subset=["pred_high"]).sort_values("local_hour")
    pred_hours = p["local_hour"].to_numpy(dtype=float)
    pred_vals = p["pred_high"].to_numpy(dtype=float)

    temp_hours, temp_vals = _temperature_curve(target_date, preds, obs)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if actual_high is not None and not pd.isna(actual_high):
        ax.axhline(actual_high, color=ACTUAL_HIGH, lw=3, zorder=2,
                   label=f"The day's actual high  ({actual_high:.0f}°F)")
        ax.annotate(f" {actual_high:.0f}°F ", xy=(1.0, actual_high),
                    xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=12, fontweight="bold", color="white", clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=ACTUAL_HIGH,
                              edgecolor="none"))

    if len(temp_hours):
        ax.plot(temp_hours, temp_vals, color=TEMPERATURE, lw=2.5, zorder=3,
                label="Actual temperature that hour")
    if len(pred_hours):
        ax.plot(pred_hours, pred_vals, color=PREDICTION, lw=2.5, marker="o",
                ms=6, markeredgecolor="white", markeredgewidth=1.2, zorder=4,
                label="Model's guess for the day's high")

    # Title and subtitle are placed manually rather than via set_title, so the
    # two lines cannot collide regardless of how tall the verdict text wraps.
    # %-d is not portable (fails on Windows), so build the day number directly.
    pretty = f"{target_date:%A, %B} {target_date.day}, {target_date:%Y}"
    verdict = _verdict(pred_vals, actual_high)
    ax.text(0, 1.16, f"How the forecast did — {pretty}", transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=INK, va="bottom", ha="left")
    if verdict:
        ax.text(0, 1.04, verdict, transform=ax.transAxes, fontsize=11.5,
                color=MUTED, va="bottom", ha="left")

    # Headroom so the actual-high line never sits flush against the frame.
    span = [*temp_vals, *pred_vals]
    if actual_high is not None and not pd.isna(actual_high):
        span.append(float(actual_high))
    if span:
        lo_y, hi_y = min(span), max(span)
        pad = max(1.0, (hi_y - lo_y) * 0.10)
        ax.set_ylim(lo_y - pad, hi_y + pad)

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _clock(v)))
    ax.set_ylabel("Temperature (°F)", fontsize=12, color=INK)
    ax.tick_params(axis="both", labelsize=11, colors=MUTED, length=0)
    # Round, evenly-spaced degree ticks; the default locator picks gaps like
    # 60/62/65/68 once a custom ylim stretches the axis.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°"))

    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # Order the legend the way the story reads, not the order things were drawn.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: ("guess" not in labels[i], "Actual temp" not in labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
              frameon=False, fontsize=11.5, handlelength=2.4,
              labelcolor=INK, columnspacing=1.8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _temperature_curve(target_date: date, preds: pd.DataFrame,
                       obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Hourly temperature for the day, preferring `obs` so the line stays
    continuous through hours the bot was offline."""
    if obs is not None and not obs.empty and "valid" in obs.columns:
        o = obs.copy()
        local = o["valid"].dt.tz_convert(NY_TZ)
        o = o[local.dt.date == target_date]
        if not o.empty:
            o = o.assign(hour=local[local.dt.date == target_date].dt.hour)
            o = o.dropna(subset=["tmpf"]).sort_values("hour")
            if not o.empty:
                return o["hour"].to_numpy(float), o["tmpf"].to_numpy(float)

    if "tmpf" in preds.columns:
        t = preds.dropna(subset=["tmpf"]).sort_values("local_hour")
        return t["local_hour"].to_numpy(float), t["tmpf"].to_numpy(float)
    return np.array([]), np.array([])


def _verdict(pred_vals: np.ndarray, actual_high: float | None) -> str:
    """One plain sentence: what it ended up saying, and how close that was."""
    if not len(pred_vals):
        return ""
    final = pred_vals[-1]
    if actual_high is None or pd.isna(actual_high):
        return (f"The model's last guess was {final:.0f}°F. "
                f"The official high for the day hasn't been published yet.")
    off = abs(final - actual_high)
    if off < 0.5:
        how = "exactly right"
    elif off < 1.5:
        how = f"off by about {off:.0f}°F"
    else:
        how = f"off by {off:.0f}°F"
    direction = "too warm" if final > actual_high else "too cold"
    tail = "" if off < 0.5 else f" — {direction}"
    return (f"The model finished the day guessing {final:.0f}°F. "
            f"It actually hit {actual_high:.0f}°F, so the final guess was {how}{tail}.")


def main() -> None:
    import score_predictions as sp

    target = (pd.to_datetime(sys.argv[1]).date() if len(sys.argv) > 1
              else datetime.now(NY_TZ).date() - timedelta(days=1))

    preds = sp.load_predictions(HERE / "nowcast_log.csv")
    preds = preds[preds["local_date"] == target]
    if preds.empty:
        raise SystemExit(f"No predictions logged for {target}.")

    lo = preds["valid_utc"].min().to_pydatetime() - timedelta(hours=6)
    hi = preds["valid_utc"].max().to_pydatetime() + timedelta(hours=8)
    obs = sp.fetch_hourly_obs(lo, hi)
    cli = sp.fetch_cli(lo, hi)
    try:
        dsm = sp.fetch_dsm(lo, hi)
    except Exception:
        dsm = pd.DataFrame()
    daily = sp.build_daily_high(obs, cli, dsm)
    drow = daily[daily["local_date"] == target]
    actual = float(drow.iloc[0]["actual_high"]) if not drow.empty else None

    out = HERE / "daily_chart.png"
    out.write_bytes(render(target, preds, obs, actual).getvalue())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
