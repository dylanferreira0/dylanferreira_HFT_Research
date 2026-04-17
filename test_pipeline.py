"""Quick pipeline smoke test on 1 day -- full MBO extraction with level-by-level features."""
import sys
sys.path.insert(0, ".")

from adaptive_mm.backtest import load_mbo, process_mbo_day, find_mbo_files, N_LEVELS, N_OC
from adaptive_mm.features import compute_features
import numpy as np

files = find_mbo_files()
weekday = [f for f in files if f.stat().st_size > 10_000_000]
print(f"Found {len(weekday)} weekday MBO files\n")

f = weekday[-1]
print(f"Testing with: {f.name} ({f.stat().st_size / 1e6:.1f} MB)\n")

df = load_mbo(f)
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

print("Processing through LOB...")
arrays = process_mbo_day(df)
del df

print(f"\nExtracted {len(arrays)} arrays:")
for k, v in sorted(arrays.items()):
    if len(v) > 0:
        print(f"  {k:30s}  shape={v.shape}  dtype={v.dtype}  "
              f"range=[{v.min():.4g}, {v.max():.4g}]")
    else:
        print(f"  {k:30s}  EMPTY")

# build per-level dicts
bid_sz_lvls = {k: arrays[f'bid_sz_L{k+1}'] for k in range(N_LEVELS)
               if f'bid_sz_L{k+1}' in arrays}
ask_sz_lvls = {k: arrays[f'ask_sz_L{k+1}'] for k in range(N_LEVELS)
               if f'ask_sz_L{k+1}' in arrays}
bid_oc_lvls = {k: arrays[f'bid_oc_L{k+1}'] for k in range(N_OC)
               if f'bid_oc_L{k+1}' in arrays}
ask_oc_lvls = {k: arrays[f'ask_oc_L{k+1}'] for k in range(N_OC)
               if f'ask_oc_L{k+1}' in arrays}

print("\nComputing features...")
feat = compute_features(
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

print(f"\nFeature matrix: {feat.shape[0]:,} rows x {feat.shape[1]} columns")

# show new level features
print(f"\nNEW LEVEL-BY-LEVEL FEATURES:")
new_cols = [c for c in feat.columns if any(c.startswith(p) for p in
            ['imb_L', 'ofi_L1', 'ofi_L2', 'ofi_deep', 'book_slope',
             'book_curv', 'hhi_', 'avg_order', 'microprice',
             'resilience', 'canc_L1', 'fwd_mp_',
             'quote_flicker', 'bid_move', 'ask_move',
             'spread_widen', 'twas'])]
for col in new_cols:
    vals = feat[col].dropna()
    if len(vals) > 0:
        print(f"  {col:40s}  min={vals.min():>12.4f}  max={vals.max():>12.4f}  "
              f"mean={vals.mean():>10.4f}")

if len(feat) > 1000:
    print(f"\nSample (row 1000):")
    row = feat.iloc[1000]
    for col in ['mid', 'microprice', 'spread_ticks',
                'imb_L1', 'imb_L2', 'imb_L3',
                'ofi_L1_500ms', 'ofi_L2_500ms', 'ofi_deep_500ms',
                'book_slope_bid', 'book_slope_ask',
                'hhi_bid', 'hhi_ask',
                'avg_order_sz_bid_L1', 'avg_order_sz_ask_L1',
                'resilience_bid_500ms', 'resilience_ask_500ms',
                'canc_L1_share_500ms',
                'quote_flicker_200ms', 'bid_move_rate_200ms',
                'spread_widen_rate_1000ms', 'twas_500ms',
                'fwd_return_50ms', 'fwd_return_500ms',
                'fwd_return_1000ms', 'fwd_return_5000ms',
                'fwd_mp_1000ms']:
        if col in feat.columns:
            print(f"  {col:35s} = {row[col]:.6f}")

print("\nPipeline test PASSED")
