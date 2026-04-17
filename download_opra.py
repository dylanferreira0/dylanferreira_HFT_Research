"""
Download SPX options data from Databento OPRA for spread optimization research.

Usage:
    python download_opra.py                  # 1 week (cheapest test)
    python download_opra.py --days 5         # 5 trading days
    python download_opra.py --start 2025-08-25 --end 2025-08-30

Requires: databento Python client + valid API key with OPRA access.
Set ``DATABENTO_API_KEY`` in the environment (never commit keys).

OPRA data is expensive -- start with 1 day to validate before buying more.
"""

import argparse
import os
import databento as db
from pathlib import Path

OUT_DIR = Path(r"C:\Users\Dylan Ferreira\OneDrive\ES Datebento\opra")


def _api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "Missing DATABENTO_API_KEY. Set it in your environment or a local .env "
            "(not tracked by git). Do not commit API keys."
        )
    return key


def download_opra(start: str, end: str, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    client = db.Historical(_api_key())

    print(f"Requesting OPRA SPX options: {start} -> {end}")
    print("Dataset: OPRA.PILLAR | Schema: mbp-1 | Symbols: SPX index options")

    cost = client.metadata.get_cost(
        dataset="OPRA.PILLAR",
        schema="mbp-1",
        symbols=["SPX.IDX"],
        stype_in="parent",
        start=start,
        end=end,
    )
    print(f"Estimated cost: ${cost / 100:.2f}")

    confirm = input("Proceed with download? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return None

    data = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        schema="mbp-1",
        symbols=["SPX.IDX"],
        stype_in="parent",
        start=start,
        end=end,
    )

    out_path = out_dir / f"spx_options_{start}_{end}.mbp1.dbn.zst"
    data.replay(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download OPRA SPX options data")
    parser.add_argument("--start", default="2025-08-25",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-08-30",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=None,
                        help="Override: download N days starting from --start")
    args = parser.parse_args()

    start = args.start
    if args.days:
        from datetime import datetime, timedelta
        end = (datetime.strptime(start, "%Y-%m-%d")
               + timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        end = args.end

    download_opra(start, end)


if __name__ == "__main__":
    main()
