"""
Advanced TCA from MBO data.

Unlike the old MBP-10 TCA which had to simulate queue positions,
this operates on actual order-level events: real queue consumption,
real cancel-to-fill ratios, real passive fill sequences.

Analyses:
  1. Vectorised markout (signed, by aggressor side) at multiple horizons
  2. Adverse selection decomposition by regime (spread, vol, time-of-day)
  3. Queue dynamics: time-to-fill, fill probability by queue depth
  4. Cancel analysis: cancel-before-fill rates, spoofing detection
  5. Toxicity decile separation: does the model actually predict markout?
"""

import numpy as np
import pandas as pd

TICK = 0.25


def markout_analysis(df: pd.DataFrame,
                     windows_ms=(50, 100, 200, 500, 1000, 5000)):
    """
    Vectorised signed markout at multiple horizons.
    Conditioned on aggressor side: buy-aggressor trades should show
    positive markout (price rises), sell-aggressor negative.

    Returns DataFrame with one row per window.
    """
    ts = df['ts_ns'].values
    mid = df['mid'].values
    side = df['trade_side'].values
    spread = df['spread_ticks'].values
    n = len(ts)
    date = df['date'].values if 'date' in df.columns else None

    results = []
    for wms in windows_ms:
        wns = wms * 1_000_000
        fwd_idx = np.searchsorted(ts, ts + wns)
        valid = fwd_idx < n
        fwd_idx_c = np.minimum(fwd_idx, n - 1)

        if date is not None:
            valid &= (date[fwd_idx_c] == date)

        raw_move = (mid[fwd_idx_c] - mid) / TICK
        raw_move[~valid] = np.nan

        # signed markout: positive if price moves WITH the aggressor
        sign = np.where(side == 0, 1.0, -1.0)  # buy agg=+1, sell agg=-1
        signed = raw_move * sign
        net = signed - 0.5 * spread  # subtract half-spread cost

        v = ~np.isnan(signed)
        if v.sum() == 0:
            continue

        buy_mask = v & (side == 0)
        sell_mask = v & (side == 1)

        results.append({
            'window_ms': wms,
            'n': int(v.sum()),
            'raw_mean': float(np.nanmean(raw_move[v])),
            'raw_std': float(np.nanstd(raw_move[v])),
            'signed_mean': float(np.nanmean(signed[v])),
            'signed_median': float(np.nanmedian(signed[v])),
            'net_mean': float(np.nanmean(net[v])),
            'net_median': float(np.nanmedian(net[v])),
            'buy_agg_markout': float(np.nanmean(raw_move[buy_mask])) if buy_mask.sum() > 0 else float('nan'),
            'sell_agg_markout': float(np.nanmean(raw_move[sell_mask])) if sell_mask.sum() > 0 else float('nan'),
            'pct_adverse': float((signed[v] < 0).mean()),
        })

    return pd.DataFrame(results)


def adverse_selection_by_regime(df: pd.DataFrame):
    """
    Decompose adverse selection by microstructure regime:
      - spread state (tight vs wide)
      - volatility regime (low/mid/high)
      - L1 imbalance bucket
      - time of day (UTC hour)
      - depth imbalance

    Uses 100ms signed markout as the adverse selection measure.
    """
    out = {}

    if 'fwd_return_100ms' not in df.columns:
        return out

    markout = df['fwd_return_100ms'].values / TICK
    side = df['trade_side'].values
    sign = np.where(side == 0, 1.0, -1.0)
    adverse = markout * sign  # positive = moved with aggressor

    # by spread
    sp = df['spread_ticks'].values
    sp_tight = sp <= 1
    sp_wide = sp >= 3
    sp_normal = (~sp_tight) & (~sp_wide)
    for label, mask in [('tight_1tick', sp_tight), ('normal_2tick', sp_normal), ('wide_3plus', sp_wide)]:
        v = mask & np.isfinite(adverse)
        if v.sum() > 0:
            out[f'spread_{label}'] = {
                'n': int(v.sum()),
                'mean_ticks': float(adverse[v].mean()),
                'pct_adverse': float((adverse[v] < 0).mean()),
            }

    # by vol regime
    if 'realized_vol_500ms' in df.columns:
        rv = df['realized_vol_500ms'].values
        rv_pos = rv[rv > 0]
        if len(rv_pos) > 10:
            q33, q67 = np.quantile(rv_pos, [0.33, 0.67])
            for label, lo, hi in [('low_vol', 0, q33), ('mid_vol', q33, q67), ('high_vol', q67, 1e10)]:
                mask = (rv >= lo) & (rv < hi) & np.isfinite(adverse)
                if mask.sum() > 0:
                    out[f'vol_{label}'] = {
                        'n': int(mask.sum()),
                        'mean_ticks': float(adverse[mask].mean()),
                        'pct_adverse': float((adverse[mask] < 0).mean()),
                    }

    # by L1 imbalance
    imb = df['imbalance'].values
    for label, lo, hi in [('bid_heavy', 0.33, 2), ('balanced', -0.33, 0.33), ('ask_heavy', -2, -0.33)]:
        mask = (imb >= lo) & (imb < hi) & np.isfinite(adverse)
        if mask.sum() > 0:
            out[f'imbalance_{label}'] = {
                'n': int(mask.sum()),
                'mean_ticks': float(adverse[mask].mean()),
                'pct_adverse': float((adverse[mask] < 0).mean()),
            }

    # by UTC hour
    ts = df['ts_ns'].values
    hours = (ts // 3_600_000_000_000) % 24
    for h in sorted(set(hours)):
        mask = (hours == h) & np.isfinite(adverse)
        if mask.sum() > 100:
            out[f'hour_{h:02d}_utc'] = {
                'n': int(mask.sum()),
                'mean_ticks': float(adverse[mask].mean()),
                'pct_adverse': float((adverse[mask] < 0).mean()),
            }

    return out


def queue_dynamics_analysis(df: pd.DataFrame):
    """
    MBO-specific queue analysis that the old MBP-10 TCA couldn't do:
      - Cancel-to-fill ratio by level
      - Bid/ask depletion asymmetry before adverse moves
      - Modify intensity as HFT fingerprint
    """
    out = {}

    n = len(df)
    if n == 0:
        return out

    # cancel-to-trade ratios across windows
    for w in ['50ms', '500ms', '5000ms']:
        cr_col = f'cancel_trade_ratio_{w}'
        mr_col = f'modify_trade_ratio_{w}'
        if cr_col in df.columns:
            vals = df[cr_col].values
            out[f'cancel_rate_{w}'] = {
                'mean': float(np.nanmean(vals)),
                'p50': float(np.nanmedian(vals)),
                'p95': float(np.nanpercentile(vals, 95)),
                'p99': float(np.nanpercentile(vals, 99)),
            }
        if mr_col in df.columns:
            vals = df[mr_col].values
            out[f'modify_rate_{w}'] = {
                'mean': float(np.nanmean(vals)),
                'p50': float(np.nanmedian(vals)),
                'p95': float(np.nanpercentile(vals, 95)),
            }

    # fill asymmetry (iceberg proxy)
    for w in ['500ms', '5000ms']:
        fa_col = f'fill_asymmetry_{w}'
        if fa_col in df.columns:
            vals = df[fa_col].values
            v = np.isfinite(vals)
            if v.sum() > 0:
                out[f'fill_asymmetry_{w}'] = {
                    'mean': float(vals[v].mean()),
                    'std': float(vals[v].std()),
                    'skew_bid': float((vals[v] > 0.3).mean()),
                    'skew_ask': float((vals[v] < -0.3).mean()),
                }

    # queue depletion before large moves
    if 'fwd_return_1000ms' in df.columns and 'bid_depletion_500ms' in df.columns:
        fwd = df['fwd_return_1000ms'].values / TICK
        bd = df['bid_depletion_500ms'].values
        ad = df['ask_depletion_500ms'].values

        big_up = fwd > 2
        big_down = fwd < -2
        neutral = (fwd >= -0.5) & (fwd <= 0.5)

        if big_up.sum() > 10:
            out['depletion_before_up'] = {
                'bid_depletion': float(bd[big_up].mean()),
                'ask_depletion': float(ad[big_up].mean()),
                'n': int(big_up.sum()),
            }
        if big_down.sum() > 10:
            out['depletion_before_down'] = {
                'bid_depletion': float(bd[big_down].mean()),
                'ask_depletion': float(ad[big_down].mean()),
                'n': int(big_down.sum()),
            }
        if neutral.sum() > 10:
            out['depletion_neutral'] = {
                'bid_depletion': float(bd[neutral].mean()),
                'ask_depletion': float(ad[neutral].mean()),
                'n': int(neutral.sum()),
            }

    return out


def toxicity_decile_separation(df: pd.DataFrame, tox_scores: np.ndarray):
    """
    The real test: does the toxicity model actually separate markouts?

    Buckets trades by predicted toxicity decile and measures
    realized markout in each bucket. If the model works, high-toxicity
    deciles should show worse markouts.
    """
    if 'fwd_return_1000ms' not in df.columns:
        return pd.DataFrame()

    markout = df['fwd_return_1000ms'].values / TICK
    side = df['trade_side'].values
    sign = np.where(side == 0, 1.0, -1.0)
    signed_markout = markout * sign

    valid = np.isfinite(signed_markout) & np.isfinite(tox_scores)
    if valid.sum() < 100:
        return pd.DataFrame()

    scores_v = np.abs(tox_scores[valid])
    markout_v = signed_markout[valid]

    try:
        deciles = pd.qcut(scores_v, 10, labels=False, duplicates='drop')
    except ValueError:
        return pd.DataFrame()

    n_bins = int(deciles.max()) + 1
    rows = []
    for d in sorted(set(deciles)):
        mask = deciles == d
        rows.append({
            'decile': int(d),
            'n': int(mask.sum()),
            'tox_score_mean': float(scores_v[mask].mean()),
            'markout_mean': float(markout_v[mask].mean()),
            'markout_median': float(np.median(markout_v[mask])),
            'pct_adverse': float((markout_v[mask] < 0).mean()),
        })

    result = pd.DataFrame(rows)

    if len(result) >= 2:
        lo = result.iloc[0]['markout_mean']
        hi = result.iloc[-1]['markout_mean']
        sep = lo - hi
        d_lo = int(result.iloc[0]['decile'])
        d_hi = int(result.iloc[-1]['decile'])
        label = f"D{d_lo}-D{d_hi}" if n_bins == 10 else f"D{d_lo}-D{d_hi} ({n_bins} bins, ties collapsed)"
        print(f"  Decile separation ({label}): {sep:+.4f} ticks")
        print(f"  D{d_lo} (low tox): markout={lo:+.4f}  |  "
              f"D{d_hi} (high tox): markout={hi:+.4f}")

    return result


def exchange_latency_analysis(df: pd.DataFrame):
    """
    Analyse ts_in_delta (exchange-to-gateway latency) patterns.
    Abnormal latency spikes can indicate:
      - Quote stuffing (burst of messages saturating gateway)
      - Exchange matching engine load
      - Correlated with adverse selection?
    """
    if 'ts_in_delta_ns' not in df.columns:
        return {}

    delta = df['ts_in_delta_ns'].values
    valid = np.isfinite(delta) & (delta > 0)
    if valid.sum() < 100:
        return {}

    d = delta[valid]
    out = {
        'latency_ns': {
            'mean': float(d.mean()),
            'median': float(np.median(d)),
            'p95': float(np.percentile(d, 95)),
            'p99': float(np.percentile(d, 99)),
            'max': float(d.max()),
        }
    }

    # correlation with adverse selection
    if 'fwd_return_100ms' in df.columns:
        fwd = df['fwd_return_100ms'].values
        both_valid = valid & np.isfinite(fwd)
        if both_valid.sum() > 100:
            corr = np.corrcoef(delta[both_valid], np.abs(fwd[both_valid]))[0, 1]
            out['latency_vs_abs_markout_corr'] = float(corr)

    return out


def print_tca_report(markouts, adverse, queue, latency, deciles):
    """Print formatted TCA report."""
    print("\n" + "=" * 70)
    print("  MBO TRANSACTION COST ANALYSIS")
    print("=" * 70)

    # markouts
    print("\n  MARKOUT ANALYSIS (ticks)")
    print("  " + "-" * 66)
    if len(markouts) > 0:
        print(f"  {'Window':>8s}  {'N':>10s}  {'Signed':>8s}  {'Net':>8s}  "
              f"{'Buy Agg':>8s}  {'Sell Agg':>8s}  {'%Adverse':>8s}")
        for _, r in markouts.iterrows():
            print(f"  {r['window_ms']:>6.0f}ms  {r['n']:>10,.0f}  "
                  f"{r['signed_mean']:>+8.4f}  {r['net_mean']:>+8.4f}  "
                  f"{r['buy_agg_markout']:>+8.4f}  {r['sell_agg_markout']:>+8.4f}  "
                  f"{r['pct_adverse']:>7.1%}")

    # adverse selection by regime
    print("\n  ADVERSE SELECTION BY REGIME")
    print("  " + "-" * 66)
    for regime, stats in sorted(adverse.items()):
        print(f"  {regime:25s}  n={stats['n']:>10,}  "
              f"mean={stats['mean_ticks']:>+7.4f}  "
              f"adverse={stats['pct_adverse']:>6.1%}")

    # queue dynamics
    print("\n  QUEUE DYNAMICS (MBO-specific)")
    print("  " + "-" * 66)
    for metric, stats in queue.items():
        if 'mean' in stats:
            parts = [f"{k}={v:.2f}" for k, v in stats.items() if isinstance(v, float)]
            print(f"  {metric:30s}  {', '.join(parts)}")
        elif 'bid_depletion' in stats:
            print(f"  {metric:30s}  bid_depl={stats['bid_depletion']:+.1f}  "
                  f"ask_depl={stats['ask_depletion']:+.1f}  n={stats['n']:,}")

    # latency
    if latency:
        print("\n  EXCHANGE LATENCY")
        print("  " + "-" * 66)
        if 'latency_ns' in latency:
            l = latency['latency_ns']
            print(f"  ts_in_delta  mean={l['mean']:.0f}ns  "
                  f"median={l['median']:.0f}ns  "
                  f"p95={l['p95']:.0f}ns  p99={l['p99']:.0f}ns")
        if 'latency_vs_abs_markout_corr' in latency:
            print(f"  Latency vs |markout| correlation: "
                  f"{latency['latency_vs_abs_markout_corr']:.4f}")

    # decile separation
    if len(deciles) > 0:
        print("\n  TOXICITY MODEL DECILE SEPARATION")
        print("  " + "-" * 66)
        print(f"  {'Decile':>6s}  {'N':>8s}  {'Score':>8s}  "
              f"{'Markout':>8s}  {'Median':>8s}  {'%Adverse':>8s}")
        for _, r in deciles.iterrows():
            print(f"  {r['decile']:>6.0f}  {r['n']:>8,.0f}  "
                  f"{r['tox_score_mean']:>8.4f}  "
                  f"{r['markout_mean']:>+8.4f}  "
                  f"{r['markout_median']:>+8.4f}  "
                  f"{r['pct_adverse']:>7.1%}")

    print("\n" + "=" * 70)
