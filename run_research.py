"""
MBO Research Pipeline
=====================
Processes Databento MBO data through:

  1. LOB reconstruction from order-level events
  2. Full feature extraction (46+ features from every MBO field)
  3. Model training with proper ML methodology:
     - Markov chain: regime-switching price transition matrices
     - Toxicity: purged/embargoed 5-fold CV + isotonic calibration
  4. TCA on holdout: markout, adverse selection, queue dynamics
  5. Export model params (JSON) for C++ consumption

Usage:
    python run_research.py              # 3 days
    python run_research.py --days 5     # more days
    python run_research.py --all        # all weekday sessions
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptive_mm.backtest import run_research, find_mbo_files


def main():
    parser = argparse.ArgumentParser(description="MBO Research Pipeline")
    parser.add_argument('--days', type=int, default=3,
                        help='Number of trading days to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all weekday sessions')
    parser.add_argument('--no-export', action='store_true',
                        help='Skip model export')
    args = parser.parse_args()

    files = find_mbo_files()
    weekday = [f for f in files if f.stat().st_size > 10_000_000]

    print(f"Found {len(files)} MBO files  ({len(weekday)} weekday sessions)")
    if files:
        print(f"Date range : {files[0].stem.split('-')[-1].split('.')[0]} -> "
              f"{files[-1].stem.split('-')[-1].split('.')[0]}")
        total_gb = sum(f.stat().st_size for f in files) / 1e9
        print(f"Total size : {total_gb:.2f} GB (compressed)\n")

    n_days = None if args.all else args.days

    result = run_research(
        days=n_days,
        export=not args.no_export,
    )

    features_df, reg_df, markov, toxicity_models = result
    print(f"\nDataset: {len(features_df):,} observations, "
          f"{reg_df.shape[1]} regressor columns")
    print(f"Trained {len(toxicity_models)} toxicity models: "
          f"{list(toxicity_models.keys())}")
    print("Done.")


if __name__ == "__main__":
    main()
