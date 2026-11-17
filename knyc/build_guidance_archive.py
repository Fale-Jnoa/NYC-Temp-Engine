"""
Backfill the IEM MOS archive for KNYC (GFS MOS + LAMP) into
data/guidance/raw/{GFS,LAV}_{YYYY}.csv, one request per model per year.

Closed years that already exist are skipped; the current year is always
refetched. Single-threaded with pauses: IEM rate-limits parallel requests.

    python -m knyc.build_guidance_archive --start-year 2010
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from knyc import guidance, paths

FIRST_YEAR = {"GFS": 2004, "LAV": 2020, "NBS": 2018}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--models", default="GFS,LAV")
    args = ap.parse_args()

    paths.GUIDANCE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    this_year = datetime.now(timezone.utc).year
    for model in args.models.split(","):
        for year in range(max(args.start_year, FIRST_YEAR[model]), this_year + 1):
            out = paths.GUIDANCE_RAW_DIR / f"{model}_{year}.csv"
            if out.exists() and year < this_year:
                continue
            sts = datetime(year, 1, 1, tzinfo=timezone.utc)
            ets = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)
            df = guidance.fetch_runs(model, sts, ets)
            df.to_csv(out, index=False)
            print(f"  {model} {year}: {len(df):,} rows -> {out.name}")


if __name__ == "__main__":
    main()
