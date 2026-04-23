"""
Export enriched CSV from MBO data for C++ backtest consumption.

Runs the full MBO pipeline (LOB reconstruction + features + regressors + model scoring)
and outputs a CSV that extends the existing MBP-10 format with extra columns:
  - toxicity scores (fast/medium/slow/macro)
  - calibrated P(toxic)
  - Markov regime (0=low, 1=mid, 2=high vol)
  - spread optimizer recommended half-spread + skew
  - key microstructure features (OFI, imbalance, microprice, vol, etc.)

The C++ engine reads this as an augmented version of its normal MBP-10 CSV.

Usage:
    python export_enriched_csv.py                  # all days
    python export_enriched_csv.py --days 3         # first 3 days
    python export_enriched_csv.py --output enriched.csv
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from adaptive_mm.backtest import (
    find_mbo_files, load_mbo, process_mbo_day, MBO_DATA_DIR, TICK_SIZE
)
from adaptive_mm.features import compute_features
from adaptive_mm.regressors import compute_regressors, get_regressor_names
from adaptive_mm.toxicity import ToxicityModel
from adaptive_mm.markov import MarkovModel

N_LEVELS = 5
N_OC = 3


def load_trained_models(model_dir: Path = Path("model_export")):
    """Load all trained models from JSON exports."""
    models = {}

    # toxicity models
    for label in ['fast', 'medium', 'slow', 'macro']:
        model_file = model_dir / f'toxicity_{label}.json'
        if not model_file.exists():
            # try legacy name
            model_file = model_dir / 'toxicity_model.json'
        if model_file.exists():
            with open(model_file) as f:
                params = json.load(f)
            m = ToxicityModel(target=params.get('target', 'fwd_return_1000ms'))
            m.active_features = params['features']
            m.n_feat = params['n_features']
            m.w = np.array(params['weights'])
            m.mean = np.array(params['mean'])
            m.var = np.array(params['var'])
            m.oof_stats = params.get('oof_stats', {})

            # isotonic calibration
            cal_file = model_dir / f'calibration_{label}.json'
            if not cal_file.exists():
                cal_file = model_dir / 'isotonic_calibration.json'
            if cal_file.exists():
                from sklearn.isotonic import IsotonicRegression
                with open(cal_file) as f:
                    cal = json.load(f)
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                iso.X_thresholds_ = np.array(cal['X_thresholds'])
                iso.y_thresholds_ = np.array(cal['y_thresholds'])
                iso.X_min_ = iso.X_thresholds_[0]
                iso.X_max_ = iso.X_thresholds_[-1]
                iso.f_ = None
                iso.increasing_ = True
                m.calibrator = iso

            models[label] = m
            print(f"  Loaded toxicity_{label}: {m.n_feat} features")

    # markov model
    markov_file = model_dir / 'markov_model.json'
    markov = MarkovModel()
    if markov_file.exists():
        with open(markov_file) as f:
            mp = json.load(f)
        markov.probs = np.array(mp['probs'])
        markov.outcome_values = np.array(mp['outcome_values'])
        if mp.get('vol_thresholds'):
            markov.vol_thresholds = np.array(mp['vol_thresholds'])
        print(f"  Loaded Markov model: {mp['n_regimes']} regimes")

    return models, markov


def export_enriched(days=None, output_file='enriched_es_data.csv'):
    print("=" * 70)
    print("  ENRICHED CSV EXPORT FOR C++ BACKTEST")
    print("=" * 70)

    # load models
    print("\nLoading trained models...")
    tox_models, markov = load_trained_models()
    primary_model = tox_models.get('slow', list(tox_models.values())[0] if tox_models else None)

    # find MBO files
    files = find_mbo_files()
    weekday_files = [f for f in files if f.stat().st_size > 10_000_000]
    if days is not None:
        weekday_files = weekday_files[:days]
    print(f"\nProcessing {len(weekday_files)} days of MBO data...")

    all_rows = []

    for filepath in weekday_files:
        date_str = filepath.stem.split('-')[-1].split('.')[0]
        print(f"\n[{date_str}]", end=" ", flush=True)

        t0 = time.time()
        df = load_mbo(filepath)
        arrays = process_mbo_day(df)
        del df

        # build level dicts
        bid_sz_lvls = {k: arrays[f'bid_sz_L{k+1}'] for k in range(N_LEVELS)
                       if f'bid_sz_L{k+1}' in arrays}
        ask_sz_lvls = {k: arrays[f'ask_sz_L{k+1}'] for k in range(N_LEVELS)
                       if f'ask_sz_L{k+1}' in arrays}
        bid_oc_lvls = {k: arrays[f'bid_oc_L{k+1}'] for k in range(N_OC)
                       if f'bid_oc_L{k+1}' in arrays}
        ask_oc_lvls = {k: arrays[f'ask_oc_L{k+1}'] for k in range(N_OC)
                       if f'ask_oc_L{k+1}' in arrays}

        feat_df = compute_features(
            ts=arrays['trade_ts'],
            trade_price=arrays['trade_price'],
            trade_size=arrays['trade_size'],
            trade_side=arrays['trade_side'],
            mid=arrays['mid'],
            spread=arrays['spread'],
            imbalance=arrays['imbalance'],
            bid_size_l1=arrays['bid_size_l1'],
            ask_size_l1=arrays['ask_size_l1'],
            cum_add_bid_sz=arrays['cum_add_bid_sz'],
            cum_add_ask_sz=arrays['cum_add_ask_sz'],
            cum_cancel_bid_sz=arrays['cum_cancel_bid_sz'],
            cum_cancel_ask_sz=arrays['cum_cancel_ask_sz'],
            cum_cancel_bid_n=arrays['cum_cancel_bid_n'],
            cum_cancel_ask_n=arrays['cum_cancel_ask_n'],
            trade_ts_recv=arrays.get('trade_ts_recv'),
            trade_ts_in_delta=arrays.get('trade_ts_in_delta'),
            trade_flags=arrays.get('trade_flags'),
            trade_sequence=arrays.get('trade_sequence'),
            bid_depth_l2_5=arrays.get('bid_depth_l2_5'),
            ask_depth_l2_5=arrays.get('ask_depth_l2_5'),
            n_bid_levels=arrays.get('n_bid_levels'),
            n_ask_levels=arrays.get('n_ask_levels'),
            cum_modify_bid_n=arrays.get('cum_modify_bid_n'),
            cum_modify_ask_n=arrays.get('cum_modify_ask_n'),
            cum_fill_bid_n=arrays.get('cum_fill_bid_n'),
            cum_fill_ask_n=arrays.get('cum_fill_ask_n'),
            cum_fill_bid_sz=arrays.get('cum_fill_bid_sz'),
            cum_fill_ask_sz=arrays.get('cum_fill_ask_sz'),
            microprice=arrays.get('microprice'),
            bid_sz_levels=bid_sz_lvls if bid_sz_lvls else None,
            ask_sz_levels=ask_sz_lvls if ask_sz_lvls else None,
            bid_oc_levels=bid_oc_lvls if bid_oc_lvls else None,
            ask_oc_levels=ask_oc_lvls if ask_oc_lvls else None,
            cum_add_bid_L1=arrays.get('cum_add_bid_L1'),
            cum_add_bid_L2=arrays.get('cum_add_bid_L2'),
            cum_add_bid_deep=arrays.get('cum_add_bid_deep'),
            cum_add_ask_L1=arrays.get('cum_add_ask_L1'),
            cum_add_ask_L2=arrays.get('cum_add_ask_L2'),
            cum_add_ask_deep=arrays.get('cum_add_ask_deep'),
            cum_canc_bid_L1=arrays.get('cum_canc_bid_L1'),
            cum_canc_bid_L2=arrays.get('cum_canc_bid_L2'),
            cum_canc_bid_deep=arrays.get('cum_canc_bid_deep'),
            cum_canc_ask_L1=arrays.get('cum_canc_ask_L1'),
            cum_canc_ask_L2=arrays.get('cum_canc_ask_L2'),
            cum_canc_ask_deep=arrays.get('cum_canc_ask_deep'),
            cum_bid_up=arrays.get('cum_bid_up'),
            cum_bid_dn=arrays.get('cum_bid_dn'),
            cum_ask_up=arrays.get('cum_ask_up'),
            cum_ask_dn=arrays.get('cum_ask_dn'),
            cum_spread_widen=arrays.get('cum_spread_widen'),
            cum_spread_narrow=arrays.get('cum_spread_narrow'),
            cum_flicker=arrays.get('cum_flicker'),
            cum_spread_time=arrays.get('cum_spread_time'),
        )

        feat_df['date'] = date_str
        feat_df.fillna(0, inplace=True)

        # compute regressors
        reg_df = compute_regressors(feat_df)

        # score with toxicity models
        for label, model in tox_models.items():
            raw = model.predict_batch(reg_df)
            feat_df[f'tox_raw_{label}'] = raw
            feat_df[f'tox_cal_{label}'] = model.calibrate(raw)

        # Markov regime
        if 'realized_vol_500ms' in feat_df.columns:
            feat_df['markov_regime'] = markov.detect_regimes(feat_df['realized_vol_500ms'].values)
        else:
            feat_df['markov_regime'] = 1

        # build MBP-10 compatible columns from the arrays
        n = len(feat_df)
        # bid/ask prices and sizes for 10 levels (L1 = _00, L2-L10 = [0]-[8])
        feat_df['bid_px_00'] = arrays['mid'] - arrays['spread'] * TICK_SIZE / 2.0
        feat_df['ask_px_00'] = arrays['mid'] + arrays['spread'] * TICK_SIZE / 2.0
        feat_df['bid_sz_00'] = arrays['bid_size_l1']
        feat_df['ask_sz_00'] = arrays['ask_size_l1']

        for k in range(1, min(N_LEVELS, 10)):
            lbl = f'L{k+1}'
            if f'bid_sz_{lbl}' in arrays:
                feat_df[f'bid_sz_0{k}'] = arrays[f'bid_sz_{lbl}']
                feat_df[f'ask_sz_0{k}'] = arrays[f'ask_sz_{lbl}']
                feat_df[f'bid_px_0{k}'] = feat_df['bid_px_00'] - k * TICK_SIZE
                feat_df[f'ask_px_0{k}'] = feat_df['ask_px_00'] + k * TICK_SIZE

        feat_df['timestamp_ns'] = arrays['trade_ts']

        # select output columns
        base_cols = ['timestamp_ns', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']
        for k in range(1, min(N_LEVELS, 10)):
            for prefix in ['bid_px', 'ask_px', 'bid_sz', 'ask_sz']:
                col = f'{prefix}_0{k}'
                if col in feat_df.columns:
                    base_cols.append(col)

        enrichment_cols = [
            'mid', 'spread_ticks', 'microprice', 'imbalance',
            'markov_regime',
        ]
        for label in tox_models:
            enrichment_cols.extend([f'tox_raw_{label}', f'tox_cal_{label}'])

        # add key features that C++ can directly use
        feature_cols = [c for c in feat_df.columns if c.startswith('ofi_L1_') or
                        c.startswith('imb_L') or c in ('realized_vol_500ms', 'vol_imbalance_500ms')]

        output_cols = base_cols + enrichment_cols + feature_cols
        output_cols = [c for c in output_cols if c in feat_df.columns]

        all_rows.append(feat_df[output_cols])
        elapsed = time.time() - t0
        print(f"{len(feat_df):,} trades in {elapsed:.1f}s")

    if not all_rows:
        print("No data processed!")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values('timestamp_ns').reset_index(drop=True)

    combined.to_csv(output_file, index=False, float_format='%.6f')
    size_mb = Path(output_file).stat().st_size / 1e6
    print(f"\nExported: {output_file} ({len(combined):,} rows, {size_mb:.1f} MB)")
    print(f"Columns: {len(combined.columns)}")
    print(f"  Base MBP-10 columns: {len(base_cols)}")
    print(f"  Enrichment columns: {len(enrichment_cols)}")
    print(f"  Feature columns: {len(feature_cols)}")


def main():
    parser = argparse.ArgumentParser(description="Export enriched CSV for C++ backtest")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--output", default="enriched_es_data.csv")
    args = parser.parse_args()
    export_enriched(days=args.days, output_file=args.output)


if __name__ == "__main__":
    main()
