"""
Optimal spread engine: finds the spread width that maximizes
    spread_captured - adverse_markout
per (regime, toxicity_decile, vol_bucket).

Works in two modes:
  A) Futures-only: uses toxicity score + Markov regime + realized vol
  B) Options-enhanced: adds GEX regime + IV level + skew

Research pipeline:
  1. For each trade in the training set, compute the markout at multiple
     half-spread levels (0.5, 1.0, 1.5, ..., 5.0 ticks)
  2. Group by (regime, tox_decile, [options_regime]) and find the
     half-spread that maximizes expected capture
  3. Fit a simple parametric surface for interpolation
  4. Export lookup table + coefficients to JSON for C++

References:
  - Gueant, Lehalle, Fernandez-Tapia (2013): optimal MM in LOB
  - Cartea, Jaimungal, Penalva (2015): algorithmic trading textbook
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .markov import MarkovModel
from .toxicity import ToxicityModel
from .regressors import compute_regressors, get_regressor_names

TICK_SIZE = 0.25

# half-spread grid in ticks
HS_GRID = np.arange(0.5, 5.5, 0.5)  # [0.5, 1.0, 1.5, ..., 5.0]

N_TOX_BINS = 10
N_VOL_BINS = 3  # low / mid / high (matches MarkovModel)
N_GEX_BINS = 3  # negative / neutral / positive dealer gamma


def _classify_gex(gex_values: np.ndarray) -> np.ndarray:
    """Bin GEX into negative(0) / neutral(1) / positive(2)."""
    if gex_values is None or len(gex_values) == 0:
        return np.ones(0, dtype=int)
    q33, q67 = np.nanpercentile(gex_values, [33, 67])
    return np.clip(np.digitize(gex_values, [q33, q67]), 0, 2)


def compute_markout_payoffs(
    mid: np.ndarray,
    trade_side: np.ndarray,
    ts_ns: np.ndarray,
    horizons_ns: list[int] = None,
) -> dict[int, np.ndarray]:
    """
    For each trade, compute the signed markout (in ticks) at each horizon.
    Markout = sign * (mid_future - mid_now) / TICK_SIZE
    where sign = +1 for buy-initiated, -1 for sell-initiated.

    Convention matches the rest of the pipeline (see features.py, tca.py):
    trade_side == 0 is a buy aggressor (+1), trade_side == 1 is a sell
    aggressor (-1). Keeping this consistent is important for the skew sign
    exported to the spread lookup table.
    """
    if horizons_ns is None:
        horizons_ns = [100_000_000, 500_000_000, 1_000_000_000, 5_000_000_000]

    n = len(mid)
    sign = np.where(trade_side == 0, 1.0, -1.0)
    markouts = {}

    for h_ns in horizons_ns:
        future_idx = np.searchsorted(ts_ns, ts_ns + h_ns, side='right') - 1
        future_idx = np.clip(future_idx, 0, n - 1)
        mid_future = mid[future_idx]
        markout_ticks = sign * (mid_future - mid) / TICK_SIZE
        markouts[h_ns] = markout_ticks

    return markouts


def compute_optimal_spread_table(
    feat_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    toxicity_model: ToxicityModel,
    markov: MarkovModel,
    gex_values: Optional[np.ndarray] = None,
    markout_horizon_ns: int = 1_000_000_000,
) -> dict:
    """
    Build the optimal spread lookup table.

    For each (vol_regime, tox_decile, [gex_regime]):
        optimal_hs = argmax_{hs} E[hs - |markout|]
        where expectation is taken over all trades in that bin.

    The payoff for a market maker quoting at half-spread hs is:
        capture = hs  (we earn hs when filled)
        cost    = max(|markout| - hs, 0)  (adverse selection exceeding our spread)
        net     = hs - |markout|  when |markout| > 0
    Actually, the net PnL per fill is simply: hs - markout (signed).
    We want to maximize E[hs - |adverse_markout|].
    """
    mid = feat_df['mid'].values
    ts_ns = feat_df['ts_ns'].values if 'ts_ns' in feat_df.columns else reg_df['ts_ns'].values
    trade_side = feat_df['trade_side'].values if 'trade_side' in feat_df.columns else reg_df['trade_side'].values

    markouts = compute_markout_payoffs(mid, trade_side, ts_ns, [markout_horizon_ns])
    markout = markouts[markout_horizon_ns]  # signed markout in ticks

    # toxicity score
    tox_scores = toxicity_model.predict_batch(reg_df)
    tox_abs = np.abs(tox_scores)
    tox_deciles = pd.qcut(tox_abs, N_TOX_BINS, labels=False, duplicates='drop')
    tox_deciles = np.nan_to_num(tox_deciles, nan=N_TOX_BINS // 2).astype(int)

    # vol regime
    rv_col = 'realized_vol_500ms'
    if rv_col in feat_df.columns:
        vol_regime = markov.detect_regimes(feat_df[rv_col].values)
    else:
        vol_regime = np.ones(len(feat_df), dtype=int)

    # GEX regime (optional)
    has_gex = gex_values is not None and len(gex_values) == len(feat_df)
    if has_gex:
        gex_regime = _classify_gex(gex_values)
    else:
        gex_regime = np.ones(len(feat_df), dtype=int)
        n_gex = 1

    n_gex = N_GEX_BINS if has_gex else 1

    # build lookup table
    table = {}
    skew_table = {}

    for v in range(N_VOL_BINS):
        for t in range(N_TOX_BINS):
            for g in range(n_gex):
                mask = (vol_regime == v) & (tox_deciles == t)
                if has_gex:
                    mask &= (gex_regime == g)

                n_trades = mask.sum()
                if n_trades < 50:
                    table[(v, t, g)] = 1.0  # default 1 tick
                    skew_table[(v, t, g)] = 0.0
                    continue

                mo = markout[mask]
                abs_mo = np.abs(mo)

                # for each half-spread candidate, compute expected net PnL
                best_hs = 1.0
                best_pnl = -np.inf

                for hs in HS_GRID:
                    # PnL = hs - |markout| per filled trade
                    # but we only get filled when the market trades through us.
                    # approximate: we capture hs, lose markout
                    net_pnl = hs - abs_mo
                    expected = net_pnl.mean()
                    if expected > best_pnl:
                        best_pnl = expected
                        best_hs = hs

                table[(v, t, g)] = float(best_hs)

                # bid/ask skew: if average markout is directional,
                # shift quotes away from the losing side
                avg_signed_mo = mo.mean()
                # positive markout = buys are toxic = widen ask
                skew = np.clip(avg_signed_mo * 0.5, -2.0, 2.0)
                skew_table[(v, t, g)] = float(skew)

    # fit linear regression for smooth interpolation
    coeffs = _fit_spread_regression(
        feat_df, reg_df, toxicity_model, markov,
        markout, tox_abs, vol_regime, gex_values
    )

    return {
        'table': table,
        'skew_table': skew_table,
        'coefficients': coeffs,
        'n_vol_bins': N_VOL_BINS,
        'n_tox_bins': N_TOX_BINS,
        'n_gex_bins': n_gex,
        'has_gex': has_gex,
        'markout_horizon_ns': markout_horizon_ns,
        'hs_grid': HS_GRID.tolist(),
    }


def _fit_spread_regression(
    feat_df, reg_df, toxicity_model, markov,
    markout, tox_abs, vol_regime, gex_values
):
    """
    Fit: optimal_half_spread = b0 + b1*|tox| + b2*vol_regime + b3*spread
                              + b4*gex + b5*|tox|*vol_regime

    Provides smooth interpolation between table cells.
    """
    n = len(feat_df)
    abs_markout = np.abs(markout)

    # target: for each trade, the half-spread that would break even
    # conservative: set optimal_hs = max(|markout|, 0.5), clipped to 5
    target_hs = np.clip(abs_markout, 0.5, 5.0)

    X_cols = [np.ones(n)]
    col_names = ['intercept']

    X_cols.append(tox_abs)
    col_names.append('abs_toxicity')

    X_cols.append(vol_regime.astype(float))
    col_names.append('vol_regime')

    if 'spread_ticks' in feat_df.columns:
        X_cols.append(feat_df['spread_ticks'].values.astype(float))
        col_names.append('spread_ticks')

    if gex_values is not None and len(gex_values) == n:
        gex_z = (gex_values - np.nanmean(gex_values)) / (np.nanstd(gex_values) + 1e-8)
        X_cols.append(np.nan_to_num(gex_z))
        col_names.append('gex_z')

    # interaction: toxicity * vol
    X_cols.append(tox_abs * vol_regime.astype(float))
    col_names.append('tox_x_vol')

    X = np.column_stack(X_cols)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(target_hs)
    X = X[valid]
    y = target_hs[valid]

    # ridge regression
    alpha = 1.0
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, X.T @ y)

    y_hat = X @ w
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    coeffs = {name: float(w[i]) for i, name in enumerate(col_names)}
    coeffs['r2'] = float(r2)
    coeffs['n_samples'] = int(valid.sum())

    return coeffs


def export_spread_optimizer(result: dict, out_dir: Path = Path("model_export")):
    """Export spread optimizer to JSON for C++ consumption."""
    out_dir.mkdir(exist_ok=True)

    # convert tuple keys to string keys for JSON
    export = {
        'coefficients': result['coefficients'],
        'n_vol_bins': result['n_vol_bins'],
        'n_tox_bins': result['n_tox_bins'],
        'n_gex_bins': result['n_gex_bins'],
        'has_gex': result['has_gex'],
        'markout_horizon_ns': result['markout_horizon_ns'],
        'hs_grid': result['hs_grid'],
        'table': {},
        'skew_table': {},
    }

    for (v, t, g), hs in result['table'].items():
        key = f"{v}_{t}_{g}"
        export['table'][key] = hs

    for (v, t, g), skew in result['skew_table'].items():
        key = f"{v}_{t}_{g}"
        export['skew_table'][key] = skew

    path = out_dir / 'spread_optimizer.json'
    with open(path, 'w') as f:
        json.dump(export, f, indent=2)

    print(f"  spread_optimizer.json (R2={result['coefficients']['r2']:.4f}, "
          f"{result['n_vol_bins']}x{result['n_tox_bins']}x{result['n_gex_bins']} table)")
    return path


def run_spread_optimization(
    feat_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    toxicity_model: ToxicityModel,
    markov: MarkovModel,
    gex_values: Optional[np.ndarray] = None,
    export: bool = True,
) -> dict:
    """
    Full spread optimization pipeline.
    Called from run_research or standalone.
    """
    print("\n  SPREAD OPTIMIZATION")
    print("  " + "-" * 50)

    result = compute_optimal_spread_table(
        feat_df, reg_df, toxicity_model, markov,
        gex_values=gex_values,
    )

    # summary
    table = result['table']
    spreads = np.array(list(table.values()))
    print(f"  Optimal half-spread: mean={spreads.mean():.2f} "
          f"min={spreads.min():.2f} max={spreads.max():.2f} ticks")
    print(f"  Regression R2: {result['coefficients']['r2']:.4f}")

    # per-regime summary
    for v in range(N_VOL_BINS):
        regime_spreads = [table[(v, t, 0)] for t in range(N_TOX_BINS)
                         if (v, t, 0) in table]
        if regime_spreads:
            label = ['LOW-VOL', 'MID-VOL', 'HIGH-VOL'][v]
            print(f"  {label}: mean hs={np.mean(regime_spreads):.2f} "
                  f"[low-tox={regime_spreads[0]:.1f}, "
                  f"high-tox={regime_spreads[-1]:.1f}]")

    if export:
        export_spread_optimizer(result)

    return result
