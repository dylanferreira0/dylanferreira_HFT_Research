"""
Options-derived features for ES spread optimization.

Computes from SPX options data (Databento OPRA mbp-1):
  1. ATM Implied Volatility (Black-Scholes inversion)
  2. IV term structure slope (next-weekly minus 0DTE)
  3. Put-call skew (25-delta put IV minus 25-delta call IV)
  4. Net Gamma Exposure (GEX) proxy
  5. Vanna pressure proxy
  6. 0DTE concentration

All features are aligned to ES trade timestamps via forward-fill.

References:
  - Bollen & Whaley (2004): "Does Net Buying Pressure Affect the Shape of IV?"
  - Ni, Pan, Poteshman (2008): "Volatility Information Trading"
  - Barbon & Buraschi (2021): "Gamma Fragility"
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from pathlib import Path
from typing import Optional

RISK_FREE_RATE = 0.045  # approximate Aug 2025

# ═══════════════════════════════════════════════════════════════
#  BLACK-SCHOLES PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def _bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-12)


def _bs_d2(S, K, T, r, sigma):
    return _bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call(S, K, T, r, sigma):
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, is_call=True, lo=0.01, hi=5.0):
    """Invert BS to find IV via Brent's method."""
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return np.nan
    func = bs_call if is_call else bs_put
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price < intrinsic + 1e-6:
        return np.nan
    try:
        f = lambda sig: func(S, K, T, r, sig) - price
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-6, maxiter=50)
    except (ValueError, RuntimeError):
        return np.nan


def bs_delta(S, K, T, r, sigma, is_call=True):
    d1 = _bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    d1 = _bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T) + 1e-12)


def bs_vega(S, K, T, r, sigma):
    d1 = _bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_vanna(S, K, T, r, sigma):
    """dDelta/dVol = -d2 * N'(d1) / sigma"""
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    return -norm.pdf(d1) * d2 / (sigma + 1e-12)


# ═══════════════════════════════════════════════════════════════
#  OPTIONS DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_opra_data(filepath: Path) -> pd.DataFrame:
    """Load Databento OPRA mbp-1 file and parse option metadata."""
    import databento as db

    store = db.DBNStore.from_file(str(filepath))
    df = store.to_df()

    if 'symbol' not in df.columns:
        raise ValueError("OPRA data must include symbol column")

    # parse SPX option symbols: e.g. "SPX   250829C06500000"
    # format: root(6) + expiry(6) + P/C(1) + strike*1000(8)
    sym = df['symbol'].astype(str)
    df['opt_root'] = sym.str[:6].str.strip()
    df['opt_expiry'] = pd.to_datetime(sym.str[6:12], format='%y%m%d', errors='coerce')
    df['opt_type'] = sym.str[12]  # 'C' or 'P'
    df['opt_strike'] = sym.str[13:21].astype(float) / 1000.0

    ts_col = 'ts_event' if 'ts_event' in df.columns else df.index.name
    if ts_col and ts_col in df.columns:
        df['ts_ns'] = pd.to_datetime(df[ts_col]).astype('int64')
    elif hasattr(df.index, 'dtype') and hasattr(df.index.dtype, 'tz'):
        df['ts_ns'] = df.index.astype('int64')
    else:
        df['ts_ns'] = np.arange(len(df), dtype=np.int64)

    df['opt_mid'] = (df['bid_px_00'] + df['ask_px_00']) / 2.0
    return df


# ═══════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_options_features(
    opra_df: pd.DataFrame,
    es_spot: pd.Series,
    es_ts_ns: np.ndarray,
    r: float = RISK_FREE_RATE,
    snap_interval_ns: int = 60_000_000_000,  # 1-minute snapshots
) -> pd.DataFrame:
    """
    Compute options-derived features at regular intervals, then align to ES timestamps.

    Parameters
    ----------
    opra_df : OPRA DataFrame from load_opra_data()
    es_spot : ES mid-price series (used as underlying for SPX ~= ES * multiplier)
    es_ts_ns : ES trade timestamps in nanoseconds
    r : risk-free rate
    snap_interval_ns : snapshot interval for options features

    Returns
    -------
    DataFrame with options features aligned to es_ts_ns
    """
    opra = opra_df.copy()
    opra = opra.dropna(subset=['opt_strike', 'opt_expiry', 'opt_mid'])
    opra = opra[opra['opt_mid'] > 0.05]  # filter dust quotes

    # SPX ~ ES price (close enough for feature engineering)
    # build time grid
    t_min = opra['ts_ns'].min()
    t_max = opra['ts_ns'].max()
    snap_times = np.arange(t_min, t_max, snap_interval_ns)

    features = []

    for snap_ns in snap_times:
        window = opra[
            (opra['ts_ns'] >= snap_ns - snap_interval_ns) &
            (opra['ts_ns'] < snap_ns)
        ]
        if len(window) < 10:
            features.append(_empty_feature_row(snap_ns))
            continue

        # approximate spot from most recent ES mid
        idx = np.searchsorted(es_ts_ns, snap_ns, side='right') - 1
        if idx < 0:
            idx = 0
        spot = float(es_spot.iloc[min(idx, len(es_spot) - 1)])

        # time to expiry in years
        now_dt = pd.Timestamp(snap_ns, unit='ns')
        window = window.copy()
        window['T'] = (window['opt_expiry'] - now_dt).dt.total_seconds() / (365.25 * 86400)
        window = window[window['T'] > 1 / (365.25 * 24)]  # > 1 hour to expiry

        if len(window) < 5:
            features.append(_empty_feature_row(snap_ns))
            continue

        row = _compute_snapshot_features(window, spot, r, snap_ns)
        features.append(row)

    feat_df = pd.DataFrame(features)
    feat_df = feat_df.sort_values('ts_ns').reset_index(drop=True)

    # align to ES trade timestamps via forward-fill
    aligned = _align_to_es(feat_df, es_ts_ns)
    return aligned


def _empty_feature_row(ts_ns):
    return {
        'ts_ns': ts_ns,
        'atm_iv': np.nan,
        'iv_change': np.nan,
        'term_slope': np.nan,
        'put_call_skew': np.nan,
        'net_gex': np.nan,
        'vanna_pressure': np.nan,
        'zero_dte_frac': np.nan,
    }


def _compute_snapshot_features(window, spot, r, snap_ns):
    """Compute all options features from a single time snapshot."""
    row = {'ts_ns': snap_ns}

    # separate by expiry
    expiries = sorted(window['opt_expiry'].unique())
    nearest_exp = expiries[0]
    next_exp = expiries[1] if len(expiries) > 1 else nearest_exp

    near = window[window['opt_expiry'] == nearest_exp]
    far = window[window['opt_expiry'] == next_exp] if next_exp != nearest_exp else pd.DataFrame()

    # 1) ATM IV: find strikes closest to spot
    atm_iv = _atm_iv(near, spot, r)
    row['atm_iv'] = atm_iv

    # 2) IV change: computed later as rolling diff
    row['iv_change'] = np.nan

    # 3) Term structure slope: near ATM IV - far ATM IV
    if len(far) > 0:
        far_atm_iv = _atm_iv(far, spot, r)
        row['term_slope'] = far_atm_iv - atm_iv if not np.isnan(far_atm_iv) else np.nan
    else:
        row['term_slope'] = np.nan

    # 4) Put-call skew: 25-delta put IV - 25-delta call IV
    row['put_call_skew'] = _put_call_skew(near, spot, r)

    # 5) Net GEX
    row['net_gex'] = _net_gex(near, spot, r)

    # 6) Vanna pressure
    row['vanna_pressure'] = _vanna_pressure(near, spot, r)

    # 7) 0DTE concentration
    T_near = near['T'].iloc[0] if len(near) > 0 else 1.0
    is_0dte = T_near < (1.0 / 365.25)  # < 1 day
    if is_0dte:
        n_0dte = len(near)
        n_total = len(window)
        row['zero_dte_frac'] = n_0dte / max(n_total, 1)
    else:
        row['zero_dte_frac'] = 0.0

    return row


def _atm_iv(opts, spot, r):
    """Find ATM IV from nearest strikes."""
    if len(opts) == 0:
        return np.nan

    calls = opts[opts['opt_type'] == 'C'].copy()
    if len(calls) == 0:
        return np.nan

    calls = calls.copy()
    calls['dist'] = np.abs(calls['opt_strike'] - spot)
    atm = calls.nsmallest(3, 'dist')

    ivs = []
    for _, row in atm.iterrows():
        iv = implied_vol(row['opt_mid'], spot, row['opt_strike'], row['T'], r, is_call=True)
        if not np.isnan(iv) and 0.02 < iv < 3.0:
            ivs.append(iv)

    return float(np.mean(ivs)) if ivs else np.nan


def _put_call_skew(opts, spot, r):
    """25-delta put IV minus 25-delta call IV."""
    calls = opts[opts['opt_type'] == 'C']
    puts = opts[opts['opt_type'] == 'P']

    # approximate 25-delta strikes
    otm_call_strike = spot * 1.03  # ~25-delta call
    otm_put_strike = spot * 0.97   # ~25-delta put

    call_iv = _strike_iv(calls, otm_call_strike, spot, r, is_call=True)
    put_iv = _strike_iv(puts, otm_put_strike, spot, r, is_call=False)

    if np.isnan(call_iv) or np.isnan(put_iv):
        return np.nan
    return put_iv - call_iv


def _strike_iv(opts, target_strike, spot, r, is_call):
    if len(opts) == 0:
        return np.nan
    opts = opts.copy()
    opts['dist'] = np.abs(opts['opt_strike'] - target_strike)
    nearest = opts.nsmallest(2, 'dist')

    ivs = []
    for _, row in nearest.iterrows():
        iv = implied_vol(row['opt_mid'], spot, row['opt_strike'], row['T'], r, is_call=is_call)
        if not np.isnan(iv) and 0.02 < iv < 3.0:
            ivs.append(iv)
    return float(np.mean(ivs)) if ivs else np.nan


def _net_gex(opts, spot, r):
    """
    Net Gamma Exposure proxy.
    GEX = sum(gamma_i * OI_proxy_i * contract_mult * spot * sign_i)
    sign: +1 for calls (dealers long gamma when selling calls to buyers),
          -1 for puts (dealers short gamma when selling puts to buyers).
    OI not in quote data, so we use bid_sz as a rough proxy for activity/interest.
    """
    if len(opts) == 0:
        return np.nan

    gex_total = 0.0
    for _, row in opts.iterrows():
        K = row['opt_strike']
        T = row['T']
        mid = row['opt_mid']
        if T <= 0 or mid <= 0 or K <= 0:
            continue
        iv = implied_vol(mid, spot, K, T, r, is_call=(row['opt_type'] == 'C'))
        if np.isnan(iv) or iv < 0.02:
            continue
        g = bs_gamma(spot, K, T, r, iv)
        oi_proxy = max(row.get('bid_sz_00', 1), 1)
        sign = 1.0 if row['opt_type'] == 'C' else -1.0
        gex_total += g * oi_proxy * spot * sign * 100  # 100 multiplier for SPX

    return gex_total


def _vanna_pressure(opts, spot, r):
    """Net vanna exposure: sum(vanna_i * OI_proxy_i * sign_i)."""
    if len(opts) == 0:
        return np.nan

    total = 0.0
    for _, row in opts.iterrows():
        K = row['opt_strike']
        T = row['T']
        mid = row['opt_mid']
        if T <= 0 or mid <= 0 or K <= 0:
            continue
        iv = implied_vol(mid, spot, K, T, r, is_call=(row['opt_type'] == 'C'))
        if np.isnan(iv) or iv < 0.02:
            continue
        v = bs_vanna(spot, K, T, r, iv)
        oi_proxy = max(row.get('bid_sz_00', 1), 1)
        sign = 1.0 if row['opt_type'] == 'C' else -1.0
        total += v * oi_proxy * sign

    return total


def _align_to_es(feat_df: pd.DataFrame, es_ts_ns: np.ndarray) -> pd.DataFrame:
    """Forward-fill options features to ES trade timestamps."""
    feat_ts = feat_df['ts_ns'].values
    cols = [c for c in feat_df.columns if c != 'ts_ns']

    out = {'ts_ns': es_ts_ns}
    idx = np.searchsorted(feat_ts, es_ts_ns, side='right') - 1
    idx = np.clip(idx, 0, len(feat_df) - 1)

    for col in cols:
        vals = feat_df[col].values
        out[col] = vals[idx]

    result = pd.DataFrame(out)

    # compute iv_change as rolling diff
    if 'atm_iv' in result.columns:
        result['iv_change'] = result['atm_iv'].diff().fillna(0.0)

    return result
