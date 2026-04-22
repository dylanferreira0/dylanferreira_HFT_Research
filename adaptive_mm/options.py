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


DEFAULT_RISK_FREE_RATE = 0.045

# Approximate ES-to-SPX basis in index points. The front-month ES
# future trades at SPX + (r - d) * SPX * T. With r~4.5%, d~1.3%,
# T~45 DTE, basis ~ +24 pts. This is a first-order correction; a
# proper implementation would compute it from the cost-of-carry model
# using the exact ES expiry and div schedule.
ES_SPX_BASIS_PTS = 24.0

# Trading-day fraction for T-to-expiry. Options desks typically use
# ~252 trading days (or 365.25 calendar days). For 0DTE we use a
# trading-hours fraction to avoid overweighting weekends.
TRADING_DAYS_PER_YEAR = 252.0
TRADING_SECONDS_PER_DAY = 6.5 * 3600  # RTH only

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


def implied_vol_vec(prices, S, K, T, r, is_call):
    """Vectorised IV: calls scalar implied_vol per row."""
    n = len(prices)
    ivs = np.full(n, np.nan)
    for i in range(n):
        ivs[i] = implied_vol(prices[i], S, K[i], T[i], r, is_call[i])
    return ivs


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


# vectorised greeks on arrays
def bs_gamma_vec(S, K, T, r, sigma):
    d1 = _bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T) + 1e-12)


def bs_vanna_vec(S, K, T, r, sigma):
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
    r: float = DEFAULT_RISK_FREE_RATE,
    snap_interval_ns: int = 60_000_000_000,
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
    opra = opra[opra['opt_mid'] > 0.05]

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

        idx = np.searchsorted(es_ts_ns, snap_ns, side='right') - 1
        if idx < 0:
            features.append(_empty_feature_row(snap_ns))
            continue
        es_mid = float(es_spot.iloc[min(idx, len(es_spot) - 1)])

        # SPX cash ~ ES - basis. Using raw ES mid as SPX spot
        # systematically biases IVs by the cost-of-carry basis.
        spot = max(es_mid - ES_SPX_BASIS_PTS, 1.0)

        # time to expiry in trading-day years for DTE > 1d,
        # intraday fraction for 0DTE
        now_dt = pd.Timestamp(snap_ns, unit='ns')
        window = window.copy()
        secs_to_exp = (window['opt_expiry'] - now_dt).dt.total_seconds()
        one_day_s = 86400.0
        T = np.where(
            secs_to_exp > one_day_s,
            secs_to_exp / (TRADING_DAYS_PER_YEAR * TRADING_SECONDS_PER_DAY),
            np.maximum(secs_to_exp / (TRADING_DAYS_PER_YEAR * TRADING_SECONDS_PER_DAY), 1e-6),
        )
        window['T'] = T
        window = window[window['T'] > 1e-6]

        if len(window) < 5:
            features.append(_empty_feature_row(snap_ns))
            continue

        row = _compute_snapshot_features(window, spot, r, snap_ns)
        features.append(row)

    feat_df = pd.DataFrame(features)
    feat_df = feat_df.sort_values('ts_ns').reset_index(drop=True)

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

    expiries = sorted(window['opt_expiry'].unique())
    nearest_exp = expiries[0]
    next_exp = expiries[1] if len(expiries) > 1 else nearest_exp

    near = window[window['opt_expiry'] == nearest_exp]
    far = window[window['opt_expiry'] == next_exp] if next_exp != nearest_exp else pd.DataFrame()

    # 1) ATM IV
    atm_iv = _atm_iv(near, spot, r)
    row['atm_iv'] = atm_iv

    # 2) IV change: computed later as rolling diff
    row['iv_change'] = np.nan

    # 3) Term structure slope
    if len(far) > 0:
        far_atm_iv = _atm_iv(far, spot, r)
        row['term_slope'] = far_atm_iv - atm_iv if not np.isnan(far_atm_iv) else np.nan
    else:
        row['term_slope'] = np.nan

    # 4) Put-call skew: 25-delta put IV - 25-delta call IV
    row['put_call_skew'] = _put_call_skew(near, spot, r, atm_iv)

    # 5) Net GEX
    row['net_gex'] = _net_gex(near, spot, r)

    # 6) Vanna pressure
    row['vanna_pressure'] = _vanna_pressure(near, spot, r)

    # 7) 0DTE concentration
    T_near = near['T'].iloc[0] if len(near) > 0 else 1.0
    is_0dte = T_near < (1.0 / TRADING_DAYS_PER_YEAR)
    if is_0dte:
        n_0dte = len(near)
        n_total = len(window)
        row['zero_dte_frac'] = n_0dte / max(n_total, 1)
    else:
        row['zero_dte_frac'] = 0.0

    return row


def _atm_iv(opts, spot, r):
    """Find ATM IV from nearest strikes (vectorized)."""
    if len(opts) == 0:
        return np.nan

    calls = opts[opts['opt_type'] == 'C'].copy()
    if len(calls) == 0:
        return np.nan

    calls['dist'] = np.abs(calls['opt_strike'].values - spot)
    atm = calls.nsmallest(3, 'dist')

    K = atm['opt_strike'].values
    T = atm['T'].values
    mid = atm['opt_mid'].values
    is_call = np.ones(len(atm), dtype=bool)
    ivs = implied_vol_vec(mid, spot, K, T, r, is_call)
    good = (ivs > 0.02) & (ivs < 3.0) & np.isfinite(ivs)
    return float(np.mean(ivs[good])) if good.any() else np.nan


def _find_25delta_strike(opts, spot, r, atm_iv, is_call: bool):
    """
    Find the strike closest to |delta|=0.25 by numerically evaluating
    BS delta at each strike using the ATM IV as a proxy sigma.
    Falls back to the nearest-to-0.25 absolute delta.
    """
    if len(opts) == 0 or np.isnan(atm_iv) or atm_iv <= 0.02:
        return np.nan
    K = opts['opt_strike'].values
    T = opts['T'].values
    sigma = np.full(len(K), atm_iv)
    d1 = _bs_d1(spot, K, T, r, sigma)
    if is_call:
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
    target = 0.25 if is_call else -0.25
    dist = np.abs(delta - target)
    best = np.argmin(dist)
    if dist[best] > 0.20:
        return np.nan
    return float(K[best])


def _put_call_skew(opts, spot, r, atm_iv):
    """25-delta put IV minus 25-delta call IV using numerical delta."""
    calls = opts[opts['opt_type'] == 'C']
    puts = opts[opts['opt_type'] == 'P']

    call_strike = _find_25delta_strike(calls, spot, r, atm_iv, is_call=True)
    put_strike = _find_25delta_strike(puts, spot, r, atm_iv, is_call=False)

    if np.isnan(call_strike) or np.isnan(put_strike):
        return np.nan

    call_iv = _strike_iv(calls, call_strike, spot, r, is_call=True)
    put_iv = _strike_iv(puts, put_strike, spot, r, is_call=False)

    if np.isnan(call_iv) or np.isnan(put_iv):
        return np.nan
    return put_iv - call_iv


def _strike_iv(opts, target_strike, spot, r, is_call):
    if len(opts) == 0:
        return np.nan
    opts = opts.copy()
    opts['dist'] = np.abs(opts['opt_strike'].values - target_strike)
    nearest = opts.nsmallest(2, 'dist')

    K = nearest['opt_strike'].values
    T = nearest['T'].values
    mid = nearest['opt_mid'].values
    ic = np.full(len(nearest), is_call)
    ivs = implied_vol_vec(mid, spot, K, T, r, ic)
    good = (ivs > 0.02) & (ivs < 3.0) & np.isfinite(ivs)
    return float(np.mean(ivs[good])) if good.any() else np.nan


def _net_gex(opts, spot, r):
    """
    Net Gamma Exposure proxy.

    Standard GEX convention (SpotGamma / SqueezeMetrics):
      - Dealers are PRESUMED net short calls (retail/funds buy calls)
        → dealer gamma from calls is NEGATIVE (short gamma).
      - Dealers are PRESUMED net short puts (retail/funds buy puts)
        → dealer gamma from puts is POSITIVE (short put = long gamma).
      GEX = sum_calls( -gamma * OI * S * 100 ) + sum_puts( +gamma * OI * S * 100 )

    Since we don't have OI, we use bid_sz_00 as a rough activity proxy.
    The magnitude is unreliable but the sign should track dealer positioning.
    """
    if len(opts) == 0:
        return np.nan

    K = opts['opt_strike'].values.astype(float)
    T = opts['T'].values.astype(float)
    mid = opts['opt_mid'].values.astype(float)
    is_call = (opts['opt_type'].values == 'C')
    oi_proxy = np.maximum(opts.get('bid_sz_00', pd.Series(np.ones(len(opts)))).values.astype(float), 1.0)

    ivs = implied_vol_vec(mid, spot, K, T, r, is_call)
    valid = np.isfinite(ivs) & (ivs > 0.02)
    if not valid.any():
        return np.nan

    K_v = K[valid]; T_v = T[valid]; ivs_v = ivs[valid]
    is_call_v = is_call[valid]; oi_v = oi_proxy[valid]

    gamma = bs_gamma_vec(spot, K_v, T_v, r, ivs_v)

    # sign: calls = -1 (dealers short calls = short gamma)
    #        puts  = +1 (dealers short puts  = long gamma)
    sign = np.where(is_call_v, -1.0, 1.0)
    return float(np.sum(gamma * oi_v * spot * sign * 100))


def _vanna_pressure(opts, spot, r):
    """Net vanna exposure (vectorized)."""
    if len(opts) == 0:
        return np.nan

    K = opts['opt_strike'].values.astype(float)
    T = opts['T'].values.astype(float)
    mid = opts['opt_mid'].values.astype(float)
    is_call = (opts['opt_type'].values == 'C')
    oi_proxy = np.maximum(opts.get('bid_sz_00', pd.Series(np.ones(len(opts)))).values.astype(float), 1.0)

    ivs = implied_vol_vec(mid, spot, K, T, r, is_call)
    valid = np.isfinite(ivs) & (ivs > 0.02)
    if not valid.any():
        return np.nan

    K_v = K[valid]; T_v = T[valid]; ivs_v = ivs[valid]
    is_call_v = is_call[valid]; oi_v = oi_proxy[valid]

    vanna = bs_vanna_vec(spot, K_v, T_v, r, ivs_v)
    sign = np.where(is_call_v, -1.0, 1.0)
    return float(np.sum(vanna * oi_v * sign))


def _align_to_es(feat_df: pd.DataFrame, es_ts_ns: np.ndarray) -> pd.DataFrame:
    """Forward-fill options features to ES trade timestamps.

    For ES timestamps that precede all options snapshots, features
    are NaN (no non-causal backfill).
    """
    feat_ts = feat_df['ts_ns'].values
    cols = [c for c in feat_df.columns if c != 'ts_ns']

    out = {'ts_ns': es_ts_ns}
    idx = np.searchsorted(feat_ts, es_ts_ns, side='right') - 1

    for col in cols:
        vals = feat_df[col].values
        aligned = np.full(len(es_ts_ns), np.nan, dtype=np.float64)
        valid = idx >= 0
        aligned[valid] = vals[idx[valid]]
        out[col] = aligned

    result = pd.DataFrame(out)

    if 'atm_iv' in result.columns:
        result['iv_change'] = result['atm_iv'].diff().fillna(0.0)

    return result
