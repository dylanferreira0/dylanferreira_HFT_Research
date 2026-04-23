"""
Toxicity regressors derived from microstructure theory.

V2: Level-by-level features from MBO data.

THEORETICAL FOUNDATIONS
=======================

1. Level-decomposed OFI - Cont, Kukanov, Stoikov (2014)
   OFI_L1 = (add_bid_L1 - cancel_bid_L1) - (add_ask_L1 - cancel_ask_L1)
   WHY: The original paper shows OFI at the BEST level is the dominant
   predictor. Aggregate OFI dilutes this with noise from deep levels.
   We decompose into L1 / L2 / deep to isolate the signal.

2. Per-level Imbalance - Cao, Chen, Griffin (2005)
   "Informational Content of an Open Limit Order Book"
   imb_Lk = (bid_Lk - ask_Lk) / (bid_Lk + ask_Lk)
   WHY: When L1 and L2 imbalances DISAGREE, the L2 signal is often
   right -- institutions rest orders behind L1 to hide their intent.

3. Microprice - Stoikov (2018)
   microprice = (ask_price * bid_sz + bid_price * ask_sz) / (bid_sz + ask_sz)
   WHY: The raw mid-price ignores imbalance. Microprice weights toward
   the side with LESS depth (the "true" price is closer to the thin side).
   Using microprice as target gives a less noisy label.

4. Book Slope - Cartea, Jaimungal, Penalva (2015)
   slope = linear regression of log(size_Lk) on k, for k=1..5
   WHY: The shape of the depth profile contains information.
   Steep drop-off = thin behind L1 = vulnerable to sweeps.
   Flat or increasing = "wall" = support/resistance.

5. Book Concentration / Herfindahl - O'Hara (1995)
   HHI = sum((size_Lk / total_size)^2)
   WHY: High HHI = one level dominates = possible institutional wall.
   Low HHI = evenly distributed = normal market making.

6. Order Fragmentation - MBO-specific
   avg_order_size_L1 = total_size_L1 / n_orders_L1
   WHY: One large order vs many small orders at the same level
   are very different signals. Large = institutional, small = retail/HFT.

7. Book Resilience - Bouchaud, Farmer, Lillo (2009)
   resilience = total L1 size churn per trade
   WHY: Fast L1 replenishment after fills = market makers confident =
   lower toxicity. Slow refill = MMs retreating = higher toxicity.

ROLLING Z-SCORE NORMALIZATION
=============================
All features use EMA z-scores via pandas ewm (C-level, no Python loops).
Two halflifes: fast (500 trades) for local, slow (5000) for session-level.
"""

import numpy as np
import pandas as pd


def _ema(x: np.ndarray, halflife: int) -> np.ndarray:
    return pd.Series(x, dtype=np.float64).ewm(halflife=halflife).mean().values


def _emvar(x: np.ndarray, halflife: int) -> np.ndarray:
    s = pd.Series(x, dtype=np.float64)
    return np.maximum(s.ewm(halflife=halflife).var().values, 1e-12)


def _zscore(x: np.ndarray, halflife: int) -> np.ndarray:
    s = pd.Series(x, dtype=np.float64)
    ewm = s.ewm(halflife=halflife)
    mu = ewm.mean().values
    var = np.maximum(ewm.var().values, 1e-12)
    return (x - mu) / np.sqrt(var)


def _safe_div(num, denom, fill=0.0):
    return np.where(np.abs(denom) > 1e-10, num / denom, fill)


def compute_regressors(df: pd.DataFrame,
                       z_halflife_fast: int = 500,
                       z_halflife_slow: int = 5000) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    reg = pd.DataFrame(index=df.index)
    fast = z_halflife_fast
    slow = z_halflife_slow

    reg['ts_ns'] = df['ts_ns'].values
    reg['mid'] = df['mid'].values
    reg['trade_side'] = df['trade_side'].values

    # ── 1. Level-decomposed OFI (Cont et al. 2014) ────────
    # L1 OFI at multiple timescales including ultra-fast (10ms, 100ms)
    for w in ['10ms', '50ms', '100ms', '200ms', '500ms', '1000ms', '5000ms']:
        col = f'ofi_L1_{w}'
        if col in df.columns:
            reg[f'z_ofi_L1_{w}'] = _zscore(df[col].values.astype(np.float64), fast)

    # L2 and deep OFI at key windows only
    for w in ['500ms', '5000ms']:
        col = f'ofi_L2_{w}'
        if col in df.columns:
            reg[f'z_ofi_L2_{w}'] = _zscore(df[col].values.astype(np.float64), fast)
        col = f'ofi_deep_{w}'
        if col in df.columns:
            reg[f'z_ofi_deep_{w}'] = _zscore(df[col].values.astype(np.float64), fast)

    # ── 2. Per-level Imbalance (Cao et al. 2005) ──────────
    for k in range(1, 4):  # L1, L2, L3
        col = f'imb_L{k}'
        if col in df.columns:
            reg[f'z_imb_L{k}'] = _zscore(df[col].values.astype(np.float64), fast)

    # L1-L2 sign disagreement (Cao et al.'s key finding)
    if 'imb_L1' in df.columns and 'imb_L2' in df.columns:
        l1 = df['imb_L1'].values.astype(np.float64)
        l2 = df['imb_L2'].values.astype(np.float64)
        disagree = np.sign(l1) * np.sign(l2)
        reg['z_L1_L2_disagree'] = _zscore(-disagree, fast)

    # ── 3. Microprice delta (Stoikov 2018) ─────────────────
    if 'microprice' in df.columns:
        mp = df['microprice'].values.astype(np.float64)
        mp_diff = np.diff(mp, prepend=mp[0])
        reg['z_microprice_delta'] = _zscore(mp_diff, fast)

    # ── 4. Volume imbalance (|B-S|/(B+S) over time window) ──
    # N.B. This is NOT volume-synchronized VPIN (Easley et al. 2012).
    for w in ['500ms', '5000ms']:
        col = f'vol_imbalance_{w}'
        if col in df.columns:
            z = _zscore(df[col].values.astype(np.float64), fast)
            reg[f'z_vol_imbalance_{w}'] = z
            reg[f'z_vpin_{w}'] = z  # backward-compat alias for C++ / exported models

    # ── 5. Kyle's Lambda (price impact) ───────────────────
    # Use L1 OFI for a cleaner lambda estimate
    ofi_col = 'ofi_L1_500ms' if 'ofi_L1_500ms' in df.columns else 'ofi_500ms'
    if ofi_col in df.columns and 'mid' in df.columns:
        mid_diff = np.diff(df['mid'].values, prepend=df['mid'].values[0])
        ofi_vals = df[ofi_col].values.astype(np.float64)
        cov_ema = _ema(mid_diff * ofi_vals, fast)
        var_ofi = _emvar(ofi_vals, fast)
        kyle_lambda = _safe_div(cov_ema, var_ofi)
        reg['z_kyle_lambda'] = _zscore(kyle_lambda, slow)

    # ── 6. Trade Arrival ──────────────────────────────────
    if 'trade_rate_50ms' in df.columns and 'trade_rate_5000ms' in df.columns:
        fast_rate = df['trade_rate_50ms'].values.astype(np.float64)
        slow_rate = df['trade_rate_5000ms'].values.astype(np.float64)
        reg['z_trade_accel'] = _zscore(
            np.log1p(fast_rate) - np.log1p(slow_rate), fast)

    if 'trade_rate_500ms' in df.columns:
        reg['z_trade_rate'] = _zscore(
            df['trade_rate_500ms'].values.astype(np.float64), fast)

    # ── 7. Cancel intensity ───────────────────────────────
    for w in ['500ms']:
        col = f'cancel_trade_ratio_{w}'
        if col in df.columns:
            reg[f'z_cancel_ratio_{w}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # cancel L1 share (are cancels concentrated at best?)
    for w in ['500ms']:
        col = f'canc_L1_share_{w}'
        if col in df.columns:
            reg[f'z_canc_L1_share_{w}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 8. Spread ─────────────────────────────────────────
    if 'spread_ticks' in df.columns:
        reg['spread_ticks'] = df['spread_ticks'].values.astype(np.float64)
        reg['z_spread'] = _zscore(
            df['spread_ticks'].values.astype(np.float64), slow)

    # ── 9. Book Shape (Cartea et al. 2015) ────────────────
    for side in ['bid', 'ask']:
        col = f'book_slope_{side}'
        if col in df.columns:
            reg[f'z_slope_{side}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    for side in ['bid', 'ask']:
        col = f'book_curv_{side}'
        if col in df.columns:
            reg[f'z_curv_{side}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 10. Book Concentration / HHI (O'Hara 1995) ───────
    for side in ['bid', 'ask']:
        col = f'hhi_{side}'
        if col in df.columns:
            reg[f'z_hhi_{side}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 11. Order Fragmentation (MBO-specific) ────────────
    for side in ['bid', 'ask']:
        col = f'avg_order_sz_{side}_L1'
        if col in df.columns:
            reg[f'z_avg_ord_sz_{side}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 12. Queue Depletion ───────────────────────────────
    for w in ['50ms', '500ms']:
        bd_col = f'bid_depletion_{w}'
        ad_col = f'ask_depletion_{w}'
        if bd_col in df.columns and ad_col in df.columns:
            bd = df[bd_col].values.astype(np.float64)
            ad = df[ad_col].values.astype(np.float64)
            reg[f'z_net_depletion_{w}'] = _zscore(bd - ad, fast)

    # ── 13. Modify Rate ───────────────────────────────────
    for w in ['500ms']:
        col = f'modify_trade_ratio_{w}'
        if col in df.columns:
            reg[f'z_modify_rate_{w}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 14. Realized Vol ──────────────────────────────────
    if 'realized_vol_500ms' in df.columns:
        reg['z_realized_vol'] = _zscore(
            df['realized_vol_500ms'].values.astype(np.float64), fast)

    # ── 15. Signed Volume ─────────────────────────────────
    for w in ['500ms', '5000ms']:
        col = f'signed_volume_{w}'
        if col in df.columns:
            reg[f'z_signed_vol_{w}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 16. Book Resilience (Bouchaud et al. 2009) ────────
    for w in ['500ms']:
        for side in ['bid', 'ask']:
            col = f'resilience_{side}_{w}'
            if col in df.columns:
                reg[f'z_resilience_{side}_{w}'] = _zscore(
                    df[col].values.astype(np.float64), fast)

    # ── 17. Fill Asymmetry ────────────────────────────────
    for w in ['500ms']:
        col = f'fill_asymmetry_{w}'
        if col in df.columns:
            reg[f'z_fill_asym_{w}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 18. Exchange Latency ──────────────────────────────
    if 'mean_latency_ns_500ms' in df.columns:
        reg['z_latency'] = _zscore(
            df['mean_latency_ns_500ms'].values.astype(np.float64), slow)

    # ── 19. Quote Flickering (Hasbrouck & Saar 2009) ────
    for w in ['50ms', '200ms', '1000ms']:
        col = f'quote_flicker_{w}'
        if col in df.columns:
            reg[f'z_flicker_{w}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 20. BBO Move Rate (Cont & de Larrard 2013) ─────
    for w in ['200ms', '1000ms']:
        for side in ['bid', 'ask']:
            col = f'{side}_move_rate_{w}'
            if col in df.columns:
                reg[f'z_{side}_move_{w}'] = _zscore(
                    df[col].values.astype(np.float64), fast)

    # ── 21. Spread Widen Rate (Huang & Stoll 1997) ─────
    for w in ['200ms', '1000ms']:
        col = f'spread_widen_rate_{w}'
        if col in df.columns:
            reg[f'z_widen_rate_{w}'] = _zscore(
                df[col].values.astype(np.float64), fast)

    # ── 22. Time-Weighted Avg Spread (Easley et al. 1997)
    for w in ['500ms', '5000ms']:
        col = f'twas_{w}'
        if col in df.columns:
            reg[f'z_twas_{w}'] = _zscore(
                df[col].values.astype(np.float64), slow)

    # ── INTERACTION TERMS ─────────────────────────────────
    # Products of z-scored inputs are NOT themselves z-scored (their
    # variance is ~1 + corr^2 with heavy tails), so we re-z-score them
    # before handing off to the ridge. Otherwise their effective
    # penalty under alpha*I is miscalibrated relative to the linear
    # z_* features.

    # OFI_L1 in high-vol regimes (most informative)
    if 'z_ofi_L1_500ms' in reg.columns and 'z_realized_vol' in reg.columns:
        reg['ofi_L1_x_vol'] = _zscore(
            (reg['z_ofi_L1_500ms'] * reg['z_realized_vol']).values, fast)

    # OFI_L1 when spread is wide (adverse selection peaks)
    if 'z_ofi_L1_500ms' in reg.columns and 'spread_ticks' in reg.columns:
        reg['ofi_L1_x_spread'] = _zscore(
            (reg['z_ofi_L1_500ms'] * reg['spread_ticks']).values, fast)

    # L2 imbalance in high-vol (institutional hiding behavior)
    if 'z_imb_L2' in reg.columns and 'z_realized_vol' in reg.columns:
        reg['imb_L2_x_vol'] = _zscore(
            (reg['z_imb_L2'] * reg['z_realized_vol']).values, fast)

    # Book slope during trade bursts (thin book + activity = danger)
    if 'z_slope_bid' in reg.columns and 'z_trade_accel' in reg.columns:
        slope_prod = (
            ((reg['z_slope_bid'] + reg['z_slope_ask']) / 2 * reg['z_trade_accel']).values
            if 'z_slope_ask' in reg.columns
            else (reg['z_slope_bid'] * reg['z_trade_accel']).values
        )
        reg['slope_x_accel'] = _zscore(slope_prod, fast)

    # flicker during high-vol = HFTs repricing aggressively
    if 'z_flicker_50ms' in reg.columns and 'z_realized_vol' in reg.columns:
        reg['flicker_x_vol'] = _zscore(
            (reg['z_flicker_50ms'] * reg['z_realized_vol']).values, fast)

    # spread widening + OFI = MMs fleeing informed flow
    if 'z_widen_rate_1000ms' in reg.columns and 'z_ofi_L1_500ms' in reg.columns:
        reg['widen_x_ofi'] = _zscore(
            (reg['z_widen_rate_1000ms'] * reg['z_ofi_L1_500ms']).values, fast)

    # ── Forward returns (targets) ─────────────────────────
    for col in df.columns:
        if col.startswith('fwd_return_') or col.startswith('fwd_mp_'):
            reg[col] = df[col].values

    return reg


def get_regressor_names() -> list[str]:
    """Ordered list of regressor columns (excludes ts_ns, mid, targets)."""
    return [
        # Level-decomposed OFI at multiple timescales (Cont et al.)
        'z_ofi_L1_10ms', 'z_ofi_L1_50ms', 'z_ofi_L1_100ms',
        'z_ofi_L1_200ms', 'z_ofi_L1_500ms', 'z_ofi_L1_1000ms',
        'z_ofi_L1_5000ms',
        'z_ofi_L2_500ms', 'z_ofi_L2_5000ms',
        'z_ofi_deep_500ms', 'z_ofi_deep_5000ms',
        # Per-level imbalance (Cao et al.)
        'z_imb_L1', 'z_imb_L2', 'z_imb_L3',
        'z_L1_L2_disagree',
        # Microprice (Stoikov)
        'z_microprice_delta',
        # VPIN
        'z_vol_imbalance_500ms', 'z_vol_imbalance_5000ms',
        # Kyle's Lambda
        'z_kyle_lambda',
        # Trade arrival
        'z_trade_accel', 'z_trade_rate',
        # Cancel intensity
        'z_cancel_ratio_500ms',
        'z_canc_L1_share_500ms',
        # Spread state
        'spread_ticks', 'z_spread',
        # Quote flickering (Hasbrouck & Saar 2009)
        'z_flicker_50ms', 'z_flicker_200ms', 'z_flicker_1000ms',
        # BBO move rate (Cont & de Larrard 2013)
        'z_bid_move_200ms', 'z_ask_move_200ms',
        'z_bid_move_1000ms', 'z_ask_move_1000ms',
        # Spread dynamics (Huang & Stoll 1997)
        'z_widen_rate_200ms', 'z_widen_rate_1000ms',
        # Time-weighted avg spread (Easley et al. 1997)
        'z_twas_500ms', 'z_twas_5000ms',
        # Book shape (Cartea et al.)
        'z_slope_bid', 'z_slope_ask',
        'z_curv_bid', 'z_curv_ask',
        # Book concentration (O'Hara)
        'z_hhi_bid', 'z_hhi_ask',
        # Order fragmentation
        'z_avg_ord_sz_bid', 'z_avg_ord_sz_ask',
        # Queue depletion
        'z_net_depletion_50ms', 'z_net_depletion_500ms',
        # Modify rate
        'z_modify_rate_500ms',
        # Realized vol
        'z_realized_vol',
        # Signed volume
        'z_signed_vol_500ms', 'z_signed_vol_5000ms',
        # Resilience (Bouchaud et al.)
        'z_resilience_bid_500ms', 'z_resilience_ask_500ms',
        # Fill asymmetry
        'z_fill_asym_500ms',
        # Latency
        'z_latency',
        # Interaction terms
        'ofi_L1_x_vol', 'ofi_L1_x_spread',
        'imb_L2_x_vol', 'slope_x_accel',
        'flicker_x_vol', 'widen_x_ofi',
    ]


def regressor_diagnostics(reg_df: pd.DataFrame,
                          target: str = 'fwd_return_1000ms'):
    names = [c for c in get_regressor_names() if c in reg_df.columns]
    y = reg_df[target].values
    valid = np.isfinite(y)

    print(f"\n  REGRESSOR DIAGNOSTICS (n={valid.sum():,}, features={len(names)})")
    print("  " + "-" * 60)

    print(f"\n  {'Regressor':35s}  {'Univ R2':>8s}  {'Corr(y)':>8s}  {'Mean':>8s}  {'Std':>8s}")
    univar_r2 = {}
    for name in names:
        x = reg_df[name].values
        both = valid & np.isfinite(x)
        if both.sum() < 100:
            continue
        xv, yv = x[both], y[both]
        corr = np.corrcoef(xv, yv)[0, 1]
        ss_tot = np.var(yv) * len(yv)
        # use ddof=1 in BOTH cov and var so the unit-scale factor
        # cancels (np.cov defaults to ddof=1, np.var to ddof=0)
        beta = np.cov(xv, yv, ddof=1)[0, 1] / (np.var(xv, ddof=1) + 1e-10)
        y_hat = beta * xv + yv.mean() - beta * xv.mean()
        ss_res = np.sum((yv - y_hat) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0
        univar_r2[name] = r2
        print(f"  {name:35s}  {r2:>8.4f}  {corr:>+8.4f}  "
              f"{xv.mean():>8.3f}  {xv.std():>8.3f}")

    X = reg_df[names].values
    v = np.all(np.isfinite(X), axis=1)
    if v.sum() > 100:
        corr_mat = np.corrcoef(X[v].T)
        print(f"\n  TOP REGRESSOR CORRELATIONS (multicollinearity check)")
        print(f"  {'Pair':50s}  {'Corr':>8s}")
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((abs(corr_mat[i, j]), names[i], names[j], corr_mat[i, j]))
        pairs.sort(reverse=True)
        for _, n1, n2, c in pairs[:15]:
            flag = " *** HIGH" if abs(c) > 0.7 else ""
            print(f"  {n1:24s} x {n2:24s}  {c:>+8.4f}{flag}")

    print(f"\n  VARIANCE INFLATION FACTORS")
    Xv = X[v]
    Xv_dm = Xv - Xv.mean(axis=0)
    try:
        XtX = Xv_dm.T @ Xv_dm / len(Xv_dm)
        diag_inv = np.diag(np.linalg.inv(XtX + 1e-6 * np.eye(len(names))))
        diag_var = np.var(Xv_dm, axis=0)
        vif = diag_inv * diag_var
        for i, name in enumerate(names):
            flag = " *** DROP CANDIDATE" if vif[i] > 10 else ""
            print(f"  {name:35s}  VIF={vif[i]:>6.1f}{flag}")
    except np.linalg.LinAlgError:
        print("  Could not compute VIF (singular matrix)")

    return univar_r2
