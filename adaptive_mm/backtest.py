"""
MBO Research Pipeline.

1. Loads .mbo.dbn.zst files via databento
2. Reconstructs LOB message-by-message, snapshots at every trade
3. Computes full feature matrix (46+ features from all MBO fields)
4. Trains models with proper ML methodology:
   - Markov chain: regime-switching transition matrices
   - Toxicity: purged/embargoed CV, OOF isotonic calibration
5. TCA: markout analysis, adverse selection, queue dynamics, decile separation
6. Exports model parameters for C++ consumption
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import databento as db

from .lob import (LOB, TICK_SIZE, price_to_ticks, preprocess_actions_sides,
                  ACT_ADD, ACT_CANCEL, ACT_MODIFY, ACT_TRADE, ACT_FILL,
                  SIDE_BID, SIDE_ASK, SIDE_NONE)
from .features import compute_features
from .markov import MarkovModel
from .toxicity import ToxicityModel
from .tca import (markout_analysis, adverse_selection_by_regime,
                  queue_dynamics_analysis, toxicity_decile_separation,
                  exchange_latency_analysis, print_tca_report)
from .regressors import compute_regressors, regressor_diagnostics
from .spread_optimizer import run_spread_optimization

DEFAULT_MBO_DIR = Path(r"C:\Users\Dylan Ferreira\OneDrive\ES Datebento")
MBO_DATA_DIR = Path(os.environ.get("HFT_MBO_DIR", str(DEFAULT_MBO_DIR)))


# ====================================================================
#  DATA LOADING
# ====================================================================

def find_mbo_files(data_dir: Path = MBO_DATA_DIR) -> list[Path]:
    return sorted(data_dir.glob("*.mbo.dbn.zst"))


def load_mbo(filepath: Path, instrument_id: int | None = None) -> pd.DataFrame:
    print(f"  Loading {filepath.name} ...")
    store = db.DBNStore.from_file(str(filepath))

    try:
        meta = store.metadata
        if hasattr(meta, 'schema'):
            print(f"    schema = {meta.schema}")
    except Exception:
        pass

    df = store.to_df()
    print(f"    {len(df):,} raw messages")

    if 'instrument_id' in df.columns:
        if instrument_id is None:
            counts = df['instrument_id'].value_counts()
            instrument_id = int(counts.index[0])
        df = df[df['instrument_id'] == instrument_id]
        sym = df['symbol'].iloc[0] if 'symbol' in df.columns and len(df) > 0 else '?'
        print(f"    filtered -> {len(df):,} msgs  (id={instrument_id}, sym={sym})")

    return df


# ====================================================================
#  LOB PROCESSING
# ====================================================================

def _resolve_ts(df: pd.DataFrame) -> np.ndarray:
    for src in ('ts_event', 'ts_recv'):
        series = df[src] if src in df.columns else None
        if series is None and df.index.name == src:
            series = df.index.to_series()
        if series is not None:
            if hasattr(series.dtype, 'tz'):
                return series.astype("int64").values
            if str(series.dtype).startswith('datetime64'):
                return series.values.astype(np.int64)
            return series.values
    idx = df.index
    if hasattr(idx.dtype, 'tz') or str(idx.dtype).startswith('datetime64'):
        return idx.astype("int64")
    raise ValueError(
        "No timestamp column (ts_event / ts_recv) found in MBO DataFrame. "
        "Fabricating fake timestamps would silently break every rolling window."
    )


def _resolve_ts_recv(df: pd.DataFrame) -> np.ndarray:
    if df.index.name == 'ts_recv':
        idx = df.index
        if hasattr(idx.dtype, 'tz'):
            return idx.astype("int64")
        return idx.values.astype(np.int64)
    if 'ts_recv' in df.columns:
        s = df['ts_recv']
        if hasattr(s.dtype, 'tz'):
            return s.astype("int64").values
        return s.values.astype(np.int64)
    return np.zeros(len(df), dtype=np.int64)


def _grow(arr, extra):
    return np.concatenate([arr, np.empty(extra, dtype=arr.dtype)])


N_LEVELS = 5   # L1-L5 snapshots
N_OC     = 3   # order counts for L1-L3


def process_mbo_day(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Process one day of MBO through LOB.  Extracts EVERY field including
    per-level book snapshots and level-relative event counters.

    Level-by-level features (Cont et al. 2014, Cao et al. 2005):
    - bid/ask sizes at L1-L5 via O(5) tick-offset lookups
    - order counts at L1-L3 (concentration / institutional signal)
    - add/cancel sizes classified by distance from BBO (L1/L2/deep)
    - microprice (imbalance-weighted mid, Stoikov 2018)
    """
    book = LOB()

    actions_raw = df['action'].values
    sides_raw   = df['side'].values
    prices_raw  = df['price'].values
    sizes       = df['size'].values.astype(np.int64)
    order_ids   = df['order_id'].values.astype(np.int64)
    ts_event    = _resolve_ts(df)
    ts_recv     = _resolve_ts_recv(df)
    flags_all   = df['flags'].values.astype(np.uint8) if 'flags' in df.columns else np.zeros(len(df), dtype=np.uint8)
    delta_all   = df['ts_in_delta'].values.astype(np.int32) if 'ts_in_delta' in df.columns else np.zeros(len(df), dtype=np.int32)
    seq_all     = df['sequence'].values.astype(np.uint32) if 'sequence' in df.columns else np.zeros(len(df), dtype=np.uint32)
    chan_all     = df['channel_id'].values.astype(np.uint8) if 'channel_id' in df.columns else np.zeros(len(df), dtype=np.uint8)
    pub_all     = df['publisher_id'].values.astype(np.uint16) if 'publisher_id' in df.columns else np.zeros(len(df), dtype=np.uint16)

    n = len(df)

    act_codes, side_codes = preprocess_actions_sides(actions_raw, sides_raw)

    prices_f = prices_raw.astype(np.float64)
    # avoid RuntimeWarning from casting NaN/inf to int
    prices_f = np.nan_to_num(prices_f, nan=0.0, posinf=0.0, neginf=0.0)
    price_ticks_all = np.round(prices_f / TICK_SIZE).astype(np.int64)

    cap = max(n // 5, 100_000)

    # -- helper to allocate all arrays via a dict for easy grow --
    def make(dtype):
        return np.empty(cap, dtype=dtype)

    # trade-level raw fields
    A = {
        'ts': make(np.int64), 'ts_recv': make(np.int64),
        'price_t': make(np.int64), 'size': make(np.int64),
        'side': make(np.int32), 'oid': make(np.int64),
        'flags': make(np.uint8), 'delta': make(np.int32),
        'seq': make(np.uint32), 'chan': make(np.uint8), 'pub': make(np.uint16),
    }

    # book snapshots
    A['mid'] = make(np.float64)
    A['microprice'] = make(np.float64)
    A['spread'] = make(np.int32)
    A['imbalance'] = make(np.float64)
    A['n_bid_levels'] = make(np.int32)
    A['n_ask_levels'] = make(np.int32)

    # per-level sizes L1-L5
    for k in range(N_LEVELS):
        A[f'bid_sz_L{k+1}'] = make(np.int64)
        A[f'ask_sz_L{k+1}'] = make(np.int64)

    # per-level order counts L1-L3
    for k in range(N_OC):
        A[f'bid_oc_L{k+1}'] = make(np.int32)
        A[f'ask_oc_L{k+1}'] = make(np.int32)

    # aggregate cumulative counters (backward compat)
    for name in ['c_absz', 'c_aasz', 'c_cbsz', 'c_casz',
                 'c_cbn', 'c_can', 'c_mbn', 'c_man',
                 'c_fbn', 'c_fan', 'c_fbsz', 'c_fasz']:
        A[name] = make(np.float64)

    # level-relative cumulative event counters (sizes)
    for side in ('bid', 'ask'):
        for lvl in ('L1', 'L2', 'deep'):
            A[f'c_add_{side}_{lvl}'] = make(np.float64)
            A[f'c_canc_{side}_{lvl}'] = make(np.float64)
            # NEW: level-decomposed fill size counters
            # (trade-induced depletion of L1/L2/deep on passive side)
            A[f'c_fill_{side}_{lvl}'] = make(np.float64)

    # quote dynamics arrays (Hasbrouck & Saar 2009, Huang & Stoll 1997)
    for name in ['c_bid_up', 'c_bid_dn', 'c_ask_up', 'c_ask_dn',
                 'c_spread_widen', 'c_spread_narrow', 'c_flicker',
                 'c_spread_time', 'c_wall_time']:
        A[name] = make(np.float64)

    # running counters: aggregate
    r_absz = r_aasz = r_cbsz = r_casz = 0.0
    r_cbn = r_can = r_mbn = r_man = 0.0
    r_fbn = r_fan = r_fbsz = r_fasz = 0.0

    # running counters: level-relative (adds, cancels, fills)
    r_ab1 = r_ab2 = r_abd = 0.0
    r_aa1 = r_aa2 = r_aad = 0.0
    r_cb1 = r_cb2 = r_cbd = 0.0
    r_ca1 = r_ca2 = r_cad = 0.0
    r_fb1 = r_fb2 = r_fbd = 0.0
    r_fa1 = r_fa2 = r_fad = 0.0

    # running counters: quote dynamics
    r_bid_up = r_bid_dn = r_ask_up = r_ask_dn = 0.0
    r_spread_widen = r_spread_narrow = r_flicker = 0.0
    r_spread_time = 0.0
    r_wall_time = 0.0
    prev_bb = prev_ba = prev_spread = 0
    prev_ts = 0

    # proper Hasbrouck & Saar flicker: count round-trip BBO moves
    # (up->down or down->up on the same side) within FLICKER_WINDOW_NS
    FLICKER_WINDOW_NS = 50_000_000
    last_bid_chg_ts = 0
    last_bid_chg_dir = 0
    last_ask_chg_ts = 0
    last_ask_chg_dir = 0

    tidx = 0
    last_mid = 0.0
    t_start = time.time()
    last_report = t_start

    for i in range(n):
        ac = act_codes[i]
        sc = side_codes[i]
        sz = sizes[i]
        pt = price_ticks_all[i]
        ts_i = ts_event[i]

        # classify level offset BEFORE book processes the event
        if pt > 0:
            if sc == SIDE_BID:
                bb = book._best_bid
                off = (bb - pt) if bb is not None else 0
            elif sc == SIDE_ASK:
                ba = book._best_ask
                off = (pt - ba) if ba is not None else 0
            else:
                off = 99
        else:
            off = 99

        # aggregate running counters (skip SIDE_NONE admin messages)
        if sc == SIDE_NONE:
            pass
        elif ac == ACT_ADD:
            if sc == SIDE_BID:
                r_absz += sz
                if off <= 0:   r_ab1 += sz
                elif off == 1: r_ab2 += sz
                else:          r_abd += sz
            else:
                r_aasz += sz
                if off <= 0:   r_aa1 += sz
                elif off == 1: r_aa2 += sz
                else:          r_aad += sz
        elif ac == ACT_CANCEL:
            if sc == SIDE_BID:
                r_cbsz += sz; r_cbn += 1
                if off <= 0:   r_cb1 += sz
                elif off == 1: r_cb2 += sz
                else:          r_cbd += sz
            else:
                r_casz += sz; r_can += 1
                if off <= 0:   r_ca1 += sz
                elif off == 1: r_ca2 += sz
                else:          r_cad += sz
        elif ac == ACT_MODIFY:
            if sc == SIDE_BID:
                r_mbn += 1
                # decompose the cancel half of modify into level buckets
                if off <= 0:   r_cb1 += sz
                elif off == 1: r_cb2 += sz
                else:          r_cbd += sz
                # the add half goes to the new price — but we only know
                # the NEW price (pt), not the original, and the offset
                # was computed against current BBO. Since modify is
                # cancel-at-old + add-at-new, and we already classified
                # the old-side offset, we add to the same bucket. The
                # add at the new price is handled by the LOB internally.
                if off <= 0:   r_ab1 += sz
                elif off == 1: r_ab2 += sz
                else:          r_abd += sz
            else:
                r_man += 1
                if off <= 0:   r_ca1 += sz
                elif off == 1: r_ca2 += sz
                else:          r_cad += sz
                if off <= 0:   r_aa1 += sz
                elif off == 1: r_aa2 += sz
                else:          r_aad += sz
        elif ac == ACT_FILL:
            # ACT_FILL: sc IS the resting side, off is already correct.
            # Only count fills here — ACT_TRADE is a summary message and
            # counting both would double the fill counters.
            if sc == SIDE_BID:
                r_fbn += 1; r_fbsz += sz
                if off <= 0:   r_fb1 += sz
                elif off == 1: r_fb2 += sz
                else:          r_fbd += sz
            elif sc == SIDE_ASK:
                r_fan += 1; r_fasz += sz
                if off <= 0:   r_fa1 += sz
                elif off == 1: r_fa2 += sz
                else:          r_fad += sz

        trade = book.process_fast(ac, sc, pt, sz, order_ids[i], ts_i)

        # ── quote dynamics: track BBO changes on EVERY message ──
        cur_bb = book._best_bid if book._best_bid is not None else 0
        cur_ba = book._best_ask if book._best_ask is not None else 0
        cur_spread = cur_ba - cur_bb if (cur_bb > 0 and cur_ba > 0) else 0

        if prev_bb > 0 and cur_bb > 0 and cur_bb != prev_bb:
            cur_dir = 1 if cur_bb > prev_bb else -1
            if cur_dir > 0: r_bid_up += 1
            else:           r_bid_dn += 1
            # flicker = reversal within FLICKER_WINDOW_NS (Hasbrouck & Saar 2009)
            if (last_bid_chg_dir != 0
                    and cur_dir * last_bid_chg_dir < 0
                    and (ts_i - last_bid_chg_ts) <= FLICKER_WINDOW_NS):
                r_flicker += 1
            last_bid_chg_ts = ts_i
            last_bid_chg_dir = cur_dir
        if prev_ba > 0 and cur_ba > 0 and cur_ba != prev_ba:
            cur_dir = 1 if cur_ba > prev_ba else -1
            if cur_dir > 0: r_ask_up += 1
            else:           r_ask_dn += 1
            if (last_ask_chg_dir != 0
                    and cur_dir * last_ask_chg_dir < 0
                    and (ts_i - last_ask_chg_ts) <= FLICKER_WINDOW_NS):
                r_flicker += 1
            last_ask_chg_ts = ts_i
            last_ask_chg_dir = cur_dir
        if prev_spread > 0 and cur_spread > 0:
            if cur_spread > prev_spread: r_spread_widen += 1
            elif cur_spread < prev_spread: r_spread_narrow += 1

        # time-weighted spread: spread * wall_dt, plus total wall-time
        # accumulator so rolling windows can divide by elapsed ns.
        if prev_ts > 0:
            dt_ns = ts_i - prev_ts
            if dt_ns > 0:
                r_wall_time += dt_ns
                if prev_spread > 0:
                    r_spread_time += prev_spread * dt_ns

        prev_bb = cur_bb
        prev_ba = cur_ba
        prev_spread = cur_spread
        prev_ts = ts_i

        if trade is None:
            continue

        # grow all arrays if needed
        if tidx >= cap:
            for k in A:
                A[k] = _grow(A[k], cap)
            cap *= 2

        # trade-level fields
        A['ts'][tidx] = ts_event[i]
        A['ts_recv'][tidx] = ts_recv[i]
        A['price_t'][tidx] = trade.price_ticks
        A['size'][tidx] = trade.size
        A['side'][tidx] = trade.aggressor_side
        A['oid'][tidx] = order_ids[i]
        A['flags'][tidx] = flags_all[i]
        A['delta'][tidx] = delta_all[i]
        A['seq'][tidx] = seq_all[i]
        A['chan'][tidx] = chan_all[i]
        A['pub'][tidx] = pub_all[i]

        # book state
        mid = book.mid
        if mid is not None:
            last_mid = mid
        A['mid'][tidx] = last_mid
        A['spread'][tidx] = book.spread_ticks
        A['n_bid_levels'][tidx] = len(book.bid_sizes)
        A['n_ask_levels'][tidx] = len(book.ask_sizes)

        # per-level snapshot via O(5) tick-offset lookups
        b_sz, a_sz, b_oc, a_oc = book.snapshot_levels(N_LEVELS)
        for k in range(N_LEVELS):
            A[f'bid_sz_L{k+1}'][tidx] = b_sz[k]
            A[f'ask_sz_L{k+1}'][tidx] = a_sz[k]
        for k in range(N_OC):
            A[f'bid_oc_L{k+1}'][tidx] = b_oc[k]
            A[f'ask_oc_L{k+1}'][tidx] = a_oc[k]

        # imbalance from L1
        b1, a1 = b_sz[0], a_sz[0]
        total = b1 + a1
        A['imbalance'][tidx] = (b1 - a1) / total if total > 0 else 0.0

        # microprice (Stoikov 2018): mid adjusted for L1 imbalance
        if total > 0 and book._best_bid is not None and book._best_ask is not None:
            bb_p = book._best_bid * TICK_SIZE
            ba_p = book._best_ask * TICK_SIZE
            A['microprice'][tidx] = (ba_p * b1 + bb_p * a1) / total
        else:
            A['microprice'][tidx] = last_mid

        # aggregate cumulative counters
        A['c_absz'][tidx] = r_absz;  A['c_aasz'][tidx] = r_aasz
        A['c_cbsz'][tidx] = r_cbsz;  A['c_casz'][tidx] = r_casz
        A['c_cbn'][tidx]  = r_cbn;   A['c_can'][tidx]  = r_can
        A['c_mbn'][tidx]  = r_mbn;   A['c_man'][tidx]  = r_man
        A['c_fbn'][tidx]  = r_fbn;   A['c_fan'][tidx]  = r_fan
        A['c_fbsz'][tidx] = r_fbsz;  A['c_fasz'][tidx] = r_fasz

        # level-relative cumulative counters
        A['c_add_bid_L1'][tidx] = r_ab1
        A['c_add_bid_L2'][tidx] = r_ab2
        A['c_add_bid_deep'][tidx] = r_abd
        A['c_add_ask_L1'][tidx] = r_aa1
        A['c_add_ask_L2'][tidx] = r_aa2
        A['c_add_ask_deep'][tidx] = r_aad
        A['c_canc_bid_L1'][tidx] = r_cb1
        A['c_canc_bid_L2'][tidx] = r_cb2
        A['c_canc_bid_deep'][tidx] = r_cbd
        A['c_canc_ask_L1'][tidx] = r_ca1
        A['c_canc_ask_L2'][tidx] = r_ca2
        A['c_canc_ask_deep'][tidx] = r_cad
        A['c_fill_bid_L1'][tidx] = r_fb1
        A['c_fill_bid_L2'][tidx] = r_fb2
        A['c_fill_bid_deep'][tidx] = r_fbd
        A['c_fill_ask_L1'][tidx] = r_fa1
        A['c_fill_ask_L2'][tidx] = r_fa2
        A['c_fill_ask_deep'][tidx] = r_fad

        # quote dynamics cumulative counters
        A['c_bid_up'][tidx] = r_bid_up
        A['c_bid_dn'][tidx] = r_bid_dn
        A['c_ask_up'][tidx] = r_ask_up
        A['c_ask_dn'][tidx] = r_ask_dn
        A['c_spread_widen'][tidx] = r_spread_widen
        A['c_spread_narrow'][tidx] = r_spread_narrow
        A['c_flicker'][tidx] = r_flicker
        A['c_spread_time'][tidx] = r_spread_time
        A['c_wall_time'][tidx] = r_wall_time

        tidx += 1

        if time.time() - last_report > 10:
            elapsed = time.time() - t_start
            pct = i / n * 100
            rate = i / elapsed
            print(f"    {i:>12,}/{n:,} ({pct:4.1f}%)  {rate:,.0f} msg/s  "
                  f"{tidx:,} trades")
            last_report = time.time()

    s = slice(0, tidx)
    elapsed = time.time() - t_start
    print(f"  Done: {book.n_msgs:,} msgs in {elapsed:.1f}s -> "
          f"{tidx:,} trades  ({book.n_adds:,} adds, {book.n_cancels:,} cancels, "
          f"{book.n_mods:,} mods)")

    # build output dict with clean names
    out = {}
    # rename to match downstream expectations
    rename = {
        'ts': 'trade_ts', 'ts_recv': 'trade_ts_recv',
        'price_t': 'trade_price', 'size': 'trade_size',
        'side': 'trade_side', 'oid': 'trade_order_id',
        'flags': 'trade_flags', 'delta': 'trade_ts_in_delta',
        'seq': 'trade_sequence', 'chan': 'trade_channel_id',
        'pub': 'trade_publisher_id',
        'spread': 'spread', 'mid': 'mid', 'microprice': 'microprice',
        'imbalance': 'imbalance',
        'n_bid_levels': 'n_bid_levels', 'n_ask_levels': 'n_ask_levels',
    }
    for old, new in rename.items():
        out[new] = A[old][s].copy()

    # backward-compat aliases
    out['bid_size_l1'] = A['bid_sz_L1'][s].copy()
    out['ask_size_l1'] = A['ask_sz_L1'][s].copy()
    bd_rest = np.zeros(tidx, dtype=np.int64)
    ad_rest = np.zeros(tidx, dtype=np.int64)
    for k in range(1, N_LEVELS):
        bd_rest += A[f'bid_sz_L{k+1}'][s]
        ad_rest += A[f'ask_sz_L{k+1}'][s]
    out['bid_depth_l2_5'] = bd_rest
    out['ask_depth_l2_5'] = ad_rest

    # per-level sizes and order counts
    for k in range(N_LEVELS):
        out[f'bid_sz_L{k+1}'] = A[f'bid_sz_L{k+1}'][s].copy()
        out[f'ask_sz_L{k+1}'] = A[f'ask_sz_L{k+1}'][s].copy()
    for k in range(N_OC):
        out[f'bid_oc_L{k+1}'] = A[f'bid_oc_L{k+1}'][s].copy()
        out[f'ask_oc_L{k+1}'] = A[f'ask_oc_L{k+1}'][s].copy()

    # cumulative counters (aggregate) -- explicit mapping to avoid replace() bugs
    _cum_rename = {
        'c_absz': 'cum_add_bid_sz', 'c_aasz': 'cum_add_ask_sz',
        'c_cbsz': 'cum_cancel_bid_sz', 'c_casz': 'cum_cancel_ask_sz',
        'c_cbn': 'cum_cancel_bid_n', 'c_can': 'cum_cancel_ask_n',
        'c_mbn': 'cum_modify_bid_n', 'c_man': 'cum_modify_ask_n',
        'c_fbn': 'cum_fill_bid_n', 'c_fan': 'cum_fill_ask_n',
        'c_fbsz': 'cum_fill_bid_sz', 'c_fasz': 'cum_fill_ask_sz',
    }
    for short, full in _cum_rename.items():
        out[full] = A[short][s].copy()

    # level-relative event counters (adds, cancels, fills)
    for side in ('bid', 'ask'):
        for lvl in ('L1', 'L2', 'deep'):
            out[f'cum_add_{side}_{lvl}'] = A[f'c_add_{side}_{lvl}'][s].copy()
            out[f'cum_canc_{side}_{lvl}'] = A[f'c_canc_{side}_{lvl}'][s].copy()
            out[f'cum_fill_{side}_{lvl}'] = A[f'c_fill_{side}_{lvl}'][s].copy()

    # quote dynamics counters
    for name in ['c_bid_up', 'c_bid_dn', 'c_ask_up', 'c_ask_dn',
                 'c_spread_widen', 'c_spread_narrow', 'c_flicker',
                 'c_spread_time', 'c_wall_time']:
        out[f'cum_{name[2:]}'] = A[name][s].copy()

    return out


# ====================================================================
#  RESEARCH PIPELINE
# ====================================================================

def run_research(days: int | None = None, export: bool = True):
    """
    Pure research pipeline:
      1. MBO -> LOB -> features
      2. Train Markov chain + toxicity model (purged CV)
      3. TCA analysis on held-out data
      4. Export model params for C++
    """
    files = find_mbo_files()
    weekday_files = [f for f in files if f.stat().st_size > 10_000_000]
    if days is not None:
        weekday_files = weekday_files[:days]

    print(f"\n{'='*70}")
    print(f"  ADAPTIVE MARKET MAKER - MBO RESEARCH PIPELINE")
    print(f"{'='*70}")
    print(f"  Trading days : {len(weekday_files)}")
    print(f"  Data dir     : {MBO_DATA_DIR}\n")

    # ── Phase 1: MBO -> features ─────────────────────────────
    print("PHASE 1 : MBO Processing -> Feature Extraction")
    print("-" * 60)

    all_features: list[pd.DataFrame] = []

    for filepath in weekday_files:
        date_str = filepath.stem.split('-')[-1].split('.')[0]
        print(f"\n[{date_str}]")

        t0 = time.time()
        df = load_mbo(filepath)

        t1 = time.time()
        arrays = process_mbo_day(df)
        del df

        t2 = time.time()

        # build per-level dicts for compute_features
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
            cum_fill_bid_L1=arrays.get('cum_fill_bid_L1'),
            cum_fill_bid_L2=arrays.get('cum_fill_bid_L2'),
            cum_fill_bid_deep=arrays.get('cum_fill_bid_deep'),
            cum_fill_ask_L1=arrays.get('cum_fill_ask_L1'),
            cum_fill_ask_L2=arrays.get('cum_fill_ask_L2'),
            cum_fill_ask_deep=arrays.get('cum_fill_ask_deep'),
            cum_bid_up=arrays.get('cum_bid_up'),
            cum_bid_dn=arrays.get('cum_bid_dn'),
            cum_ask_up=arrays.get('cum_ask_up'),
            cum_ask_dn=arrays.get('cum_ask_dn'),
            cum_spread_widen=arrays.get('cum_spread_widen'),
            cum_spread_narrow=arrays.get('cum_spread_narrow'),
            cum_flicker=arrays.get('cum_flicker'),
            cum_spread_time=arrays.get('cum_spread_time'),
            cum_wall_time=arrays.get('cum_wall_time'),
        )
        feat_df['date'] = date_str
        feat_df['trade_price_ticks'] = arrays['trade_price']

        t3 = time.time()
        print(f"  Timings: load {t1-t0:.1f}s | LOB {t2-t1:.1f}s | feat {t3-t2:.1f}s")
        print(f"  Features: {len(feat_df):,} rows x {feat_df.shape[1]} cols")
        all_features.append(feat_df)

    full_df = pd.concat(all_features, ignore_index=True)
    del all_features

    # ── Fix per-day cumulative counter resets ──
    # Each day starts its cum_* columns at 0. After concat, rolling-window
    # deltas (p[i] - p[lb]) across day boundaries would go negative.
    # Fix: accumulate the end-of-day value as an offset to subsequent days.
    cum_cols = [c for c in full_df.columns if c.startswith('cum_')]
    if 'date' in full_df.columns and len(all_features) > 1 and cum_cols:
        date_vals = full_df['date'].values
        boundaries = np.where(date_vals[:-1] != date_vals[1:])[0]
        if len(boundaries) > 0:
            for col in cum_cols:
                arr = full_df[col].values.copy().astype(np.float64)
                offset = 0.0
                prev_end = 0
                for bc in boundaries:
                    arr[prev_end:bc + 1] += offset
                    offset = arr[bc]  # last value of this day (already shifted)
                    prev_end = bc + 1
                arr[prev_end:] += offset
                full_df[col] = arr

    # NaN-safe: zero-fill FEATURES only, preserve NaN in forward-return
    # targets (day-boundary samples whose horizon extends past end-of-day
    # MUST stay NaN so they're excluded from training / evaluation).
    target_cols = set(c for c in full_df.columns
                      if c.startswith('fwd_return_') or c.startswith('fwd_mp_'))
    for col in full_df.columns:
        if col not in target_cols and full_df[col].isna().any():
            full_df[col] = full_df[col].fillna(0)

    # ensure ts_ns is monotonically non-decreasing across concatenated
    # days. If files were not date-sorted, searchsorted / EWM order breaks.
    if not np.all(np.diff(full_df['ts_ns'].values) >= 0):
        print("  WARNING: ts_ns not monotonic after concat — sorting by ts_ns")
        full_df = full_df.sort_values('ts_ns', kind='mergesort').reset_index(drop=True)

    print(f"\nTotal dataset: {len(full_df):,} trade-level observations")

    # ── Phase 2: Model training ──────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 2 : Model Training")
    print("-" * 60)

    mid_changes = np.diff(full_df['mid'].values, prepend=full_df['mid'].values[0])
    full_df['mid_change_ticks'] = np.round(mid_changes / TICK_SIZE).astype(int)

    # ── Compute theory-grounded regressors with rolling z-scores ──
    print(f"\n  Computing regressors (rolling z-scores)...")
    reg_df = compute_regressors(full_df)
    print(f"  Regressors: {reg_df.shape[1]} columns")

    assert len(reg_df) == len(full_df) and np.array_equal(
        reg_df['ts_ns'].values, full_df['ts_ns'].values), \
        "Feature/regressor alignment broken — row count or ts_ns mismatch"

    # train on first 60%, TCA on last 40% (never seen by models)
    n_train = int(len(full_df) * 0.6)
    train_feat = full_df.iloc[:n_train]
    test_feat  = full_df.iloc[n_train:].reset_index(drop=True)
    train_reg = reg_df.iloc[:n_train]
    test_reg  = reg_df.iloc[n_train:].reset_index(drop=True)
    print(f"  Train : {len(train_reg):,}  |  Holdout (TCA) : {len(test_reg):,}")

    # --- Regressor diagnostics ---
    print(f"\n  Regressor Diagnostics (target: mid-based 1s return)")
    regressor_diagnostics(train_reg, target='fwd_return_1000ms')

    # also check microprice target if available
    if 'fwd_mp_1000ms' in train_reg.columns:
        print(f"\n  Regressor Diagnostics (target: microprice-based 1s return)")
        regressor_diagnostics(train_reg, target='fwd_mp_1000ms')

    # --- Markov chain ---
    print(f"\n  Markov Chain (regime-switching transition matrices)")
    markov = MarkovModel()
    markov.fit(
        spread_ticks=train_feat['spread_ticks'].values,
        imbalance=train_feat['imbalance'].values,
        price_change_ticks=train_feat['mid_change_ticks'].values,
        realized_vol=train_feat['realized_vol_500ms'].values,
    )
    print(markov.summary())

    # --- Multi-horizon toxicity models (Lehalle & Laruelle 2013) ---
    # Different horizons detect different adversaries:
    #   50ms  = co-located HFT
    #   500ms = fast systematic / algo
    #   1000ms = institutional sweep (our main model)
    #   5000ms = fundamental / news-driven
    HORIZON_CONFIGS = [
        ('fast',   'fwd_return_50ms',   50_000_000,    2_000_000_000,  1_000_000_000),
        ('medium', 'fwd_return_500ms',  500_000_000,   5_000_000_000,  2_500_000_000),
        ('slow',   'fwd_return_1000ms', 1_000_000_000, 10_000_000_000, 5_000_000_000),
        ('macro',  'fwd_return_5000ms', 5_000_000_000, 20_000_000_000, 10_000_000_000),
    ]

    toxicity_models = {}
    for label, target, horizon_ns, purge_ns, embargo_ns in HORIZON_CONFIGS:
        if target not in train_reg.columns:
            print(f"\n  Skipping {label} model: {target} not available")
            continue

        print(f"\n  Toxicity Model [{label}] target={target} "
              f"(purge={purge_ns/1e9:.0f}s, embargo={embargo_ns/1e9:.0f}s)")
        model = ToxicityModel(
            target=target,
            n_folds=5,
            purge_ns=purge_ns,
            embargo_ns=embargo_ns,
        )
        stats = model.fit_batch(train_reg)
        print(f"  OOF R2 = {stats['r2']:.4f}  |  OOF RMSE = {stats['rmse']:.6f}  |  "
              f"n = {stats['n_samples']:,}  |  features = {stats['n_features']}")
        if 'fold_r2s' in model.oof_stats:
            r2s = model.oof_stats['fold_r2s']
            print(f"  Per-fold R2: {['%.4f' % r for r in r2s]}")
        print("  Top features:")
        for name, w in model.feature_importance[:8]:
            print(f"    {name:35s}  |w| = {w:.5f}")
        toxicity_models[label] = model

    # primary model for TCA is 'slow' (1000ms)
    toxicity = toxicity_models.get('slow', list(toxicity_models.values())[-1])

    # ── Phase 3: TCA on holdout ──────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 3 : Transaction Cost Analysis (holdout set)")
    print("-" * 60)

    markouts = markout_analysis(test_feat)
    adverse = adverse_selection_by_regime(test_feat)
    queue = queue_dynamics_analysis(test_feat)
    latency = exchange_latency_analysis(test_feat)

    # decile separation for each horizon model
    for label, model in toxicity_models.items():
        print(f"\n  --- Decile separation [{label}] target={model.target} ---")
        tox_scores = model.predict_batch(test_reg)
        toxicity_decile_separation(test_feat, tox_scores)

    # main TCA report uses the 1s model
    tox_scores = toxicity.predict_batch(test_reg)
    deciles = toxicity_decile_separation(test_feat, tox_scores)
    print_tca_report(markouts, adverse, queue, latency, deciles)

    # ── Phase 4: Spread Optimization ─────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 4 : Spread Optimization")
    print("-" * 60)
    spread_result = run_spread_optimization(
        train_feat, train_reg, toxicity, markov, export=export)

    # ── Phase 5: Export for C++ ──────────────────────────────
    if export:
        print(f"\nPHASE 5 : Exporting model parameters")
        print("-" * 60)
        export_models(toxicity_models, markov, full_df)

    return full_df, reg_df, markov, toxicity_models


def export_models(toxicity_models: dict, markov: MarkovModel,
                  df: pd.DataFrame):
    """Export trained model params as JSON for C++ consumption."""
    out_dir = Path("model_export")
    out_dir.mkdir(exist_ok=True)

    # one JSON per toxicity horizon model
    for label, model in toxicity_models.items():
        tox_params = {
            'label': label,
            'features': model.active_features,
            'weights': model.w.tolist(),
            'mean': model.mean.tolist(),
            'var': model.var.tolist(),
            'n_features': model.n_feat,
            'target': model.target,
            'oof_stats': model.oof_stats,
        }
        with open(out_dir / f'toxicity_{label}.json', 'w') as f:
            json.dump(tox_params, f, indent=2, default=str)

        if model.calibrator is not None:
            cal = model.calibrator
            cal_params = {
                'X_thresholds': cal.X_thresholds_.tolist() if hasattr(cal, 'X_thresholds_') else [],
                'y_thresholds': cal.y_thresholds_.tolist() if hasattr(cal, 'y_thresholds_') else [],
            }
            with open(out_dir / f'calibration_{label}.json', 'w') as f:
                json.dump(cal_params, f, indent=2)

        print(f"  toxicity_{label}.json  ({model.n_feat} features, "
              f"target={model.target}, OOF R2={model.oof_stats.get('oof_r2', 0):.4f})")

    # markov transition matrices
    markov_params = {
        'n_regimes': markov.N_REGIMES,
        'n_states': markov.N_STATES,
        'n_outcomes': markov.N_OUTCOMES,
        'outcome_values': markov.outcome_values.tolist(),
        'probs': markov.probs.tolist(),
        'vol_thresholds': markov.vol_thresholds.tolist() if markov.vol_thresholds is not None else [],
    }
    with open(out_dir / 'markov_model.json', 'w') as f:
        json.dump(markov_params, f, indent=2)

    # feature sample for C++ validation
    sample = df.head(1000)
    sample.to_parquet(str(out_dir / 'feature_sample.parquet'))

    print(f"  markov_model.json       ({markov.N_REGIMES} regimes, "
          f"{markov.N_STATES} states)")
    print(f"  feature_sample.parquet  (1000 rows for validation)")
