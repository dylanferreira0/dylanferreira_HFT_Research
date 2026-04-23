"""
Vectorized feature computation from MBO-derived trade snapshots.

Level-by-level features (literature-grounded):
  - Per-level imbalance L1-L3 (Cao, Chen, Griffin 2005)
  - Level-decomposed OFI: L1 / L2 / deep (Cont, Kukanov, Stoikov 2014)
  - Book slope and curvature (Cartea, Jaimungal, Penalva 2015)
  - Book concentration / Herfindahl (O'Hara 1995)
  - Order fragmentation at L1 (MBO-specific)
  - Microprice and microprice forward returns (Stoikov 2018)
  - Book resilience (L1 refill speed, Bouchaud et al. 2009)

All heavy lifting uses numpy searchsorted for O(n log n) rolling windows.
"""

import numpy as np
import pandas as pd

TICK_SIZE = 0.25

FLAG_LAST     = 0x80
FLAG_TOB      = 0x40
FLAG_SNAPSHOT = 0x20
FLAG_MBP      = 0x10


def compute_features(
    ts: np.ndarray,
    trade_price: np.ndarray,
    trade_size: np.ndarray,
    trade_side: np.ndarray,
    mid: np.ndarray,
    spread: np.ndarray,
    imbalance: np.ndarray,
    bid_size_l1: np.ndarray,
    ask_size_l1: np.ndarray,
    cum_add_bid_sz: np.ndarray,
    cum_add_ask_sz: np.ndarray,
    cum_cancel_bid_sz: np.ndarray,
    cum_cancel_ask_sz: np.ndarray,
    cum_cancel_bid_n: np.ndarray,
    cum_cancel_ask_n: np.ndarray,
    trade_ts_recv: np.ndarray | None = None,
    trade_ts_in_delta: np.ndarray | None = None,
    trade_flags: np.ndarray | None = None,
    trade_sequence: np.ndarray | None = None,
    bid_depth_l2_5: np.ndarray | None = None,
    ask_depth_l2_5: np.ndarray | None = None,
    n_bid_levels: np.ndarray | None = None,
    n_ask_levels: np.ndarray | None = None,
    cum_modify_bid_n: np.ndarray | None = None,
    cum_modify_ask_n: np.ndarray | None = None,
    cum_fill_bid_n: np.ndarray | None = None,
    cum_fill_ask_n: np.ndarray | None = None,
    cum_fill_bid_sz: np.ndarray | None = None,
    cum_fill_ask_sz: np.ndarray | None = None,
    # per-level arrays (new)
    microprice: np.ndarray | None = None,
    bid_sz_levels: dict | None = None,
    ask_sz_levels: dict | None = None,
    bid_oc_levels: dict | None = None,
    ask_oc_levels: dict | None = None,
    # level-relative event counters (new)
    cum_add_bid_L1: np.ndarray | None = None,
    cum_add_bid_L2: np.ndarray | None = None,
    cum_add_bid_deep: np.ndarray | None = None,
    cum_add_ask_L1: np.ndarray | None = None,
    cum_add_ask_L2: np.ndarray | None = None,
    cum_add_ask_deep: np.ndarray | None = None,
    cum_canc_bid_L1: np.ndarray | None = None,
    cum_canc_bid_L2: np.ndarray | None = None,
    cum_canc_bid_deep: np.ndarray | None = None,
    cum_canc_ask_L1: np.ndarray | None = None,
    cum_canc_ask_L2: np.ndarray | None = None,
    cum_canc_ask_deep: np.ndarray | None = None,
    # level-decomposed fill sizes (trade-induced depletion per level)
    cum_fill_bid_L1: np.ndarray | None = None,
    cum_fill_bid_L2: np.ndarray | None = None,
    cum_fill_bid_deep: np.ndarray | None = None,
    cum_fill_ask_L1: np.ndarray | None = None,
    cum_fill_ask_L2: np.ndarray | None = None,
    cum_fill_ask_deep: np.ndarray | None = None,
    # quote dynamics counters (Hasbrouck & Saar 2009, Huang & Stoll 1997)
    cum_bid_up: np.ndarray | None = None,
    cum_bid_dn: np.ndarray | None = None,
    cum_ask_up: np.ndarray | None = None,
    cum_ask_dn: np.ndarray | None = None,
    cum_spread_widen: np.ndarray | None = None,
    cum_spread_narrow: np.ndarray | None = None,
    cum_flicker: np.ndarray | None = None,
    cum_spread_time: np.ndarray | None = None,
    cum_wall_time: np.ndarray | None = None,
    windows_ns=(10_000_000, 50_000_000, 100_000_000, 200_000_000,
                500_000_000, 1_000_000_000, 5_000_000_000),
    horizons_ns=(10_000_000, 50_000_000, 100_000_000, 200_000_000,
                 500_000_000, 1_000_000_000, 5_000_000_000, 10_000_000_000),
) -> pd.DataFrame:
    ts = np.asarray(ts, dtype=np.int64)
    n = len(ts)
    if n == 0:
        return pd.DataFrame()

    feat: dict[str, np.ndarray] = {}

    # ── direct book state ──────────────────────────────────
    feat['mid'] = mid
    feat['spread_ticks'] = spread.astype(np.float64)
    feat['imbalance'] = imbalance
    feat['bid_size_l1'] = bid_size_l1.astype(np.float64)
    feat['ask_size_l1'] = ask_size_l1.astype(np.float64)
    feat['trade_size'] = trade_size.astype(np.float64)
    feat['trade_side'] = trade_side.astype(np.float64)

    # ── microprice (Stoikov 2018) ──────────────────────────
    if microprice is not None:
        feat['microprice'] = microprice.astype(np.float64)

    # ── per-level imbalances (Cao et al. 2005) ─────────────
    has_levels = bid_sz_levels is not None and ask_sz_levels is not None
    if has_levels:
        n_lvl = min(len(bid_sz_levels), len(ask_sz_levels))
        for k in range(min(n_lvl, 5)):
            bk = bid_sz_levels[k].astype(np.float64)
            ak = ask_sz_levels[k].astype(np.float64)
            feat[f'bid_sz_L{k+1}'] = bk
            feat[f'ask_sz_L{k+1}'] = ak
            total = bk + ak
            imb_out = np.zeros(n, dtype=np.float64)
            np.divide(bk - ak, total, out=imb_out, where=total > 0)
            feat[f'imb_L{k+1}'] = imb_out

        # book slope: log-linear regression of depth vs level offset
        # slope < 0 = depth drops off (thin behind L1)
        # slope > 0 = depth increases (wall behind L1)
        b_stack = np.column_stack([bid_sz_levels[k].astype(np.float64) for k in range(min(n_lvl, 5))])
        a_stack = np.column_stack([ask_sz_levels[k].astype(np.float64) for k in range(min(n_lvl, 5))])

        def _book_slope(stack):
            log_sz = np.log1p(stack)
            n_lvls = log_sz.shape[1]
            x = np.arange(n_lvls, dtype=np.float64)
            x_dm = x - x.mean()
            var_x = np.sum(x_dm ** 2)
            slopes = (log_sz @ x_dm) / (var_x + 1e-10)
            return slopes

        feat['book_slope_bid'] = _book_slope(b_stack)
        feat['book_slope_ask'] = _book_slope(a_stack)

        # book curvature: second derivative (convex vs concave profile)
        if n_lvl >= 3:
            b_curv = np.log1p(b_stack[:, 2]) - 2 * np.log1p(b_stack[:, 1]) + np.log1p(b_stack[:, 0])
            a_curv = np.log1p(a_stack[:, 2]) - 2 * np.log1p(a_stack[:, 1]) + np.log1p(a_stack[:, 0])
            feat['book_curv_bid'] = b_curv
            feat['book_curv_ask'] = a_curv

        # Herfindahl concentration index (O'Hara 1995)
        for side_name, stack in [('bid', b_stack), ('ask', a_stack)]:
            row_total = stack.sum(axis=1, keepdims=True)
            shares = stack / np.maximum(row_total, 1.0)
            hhi = np.sum(shares ** 2, axis=1)
            feat[f'hhi_{side_name}'] = hhi

    # order fragmentation at L1 (avg order size = institutional signal)
    has_oc = bid_oc_levels is not None and ask_oc_levels is not None
    if has_oc and has_levels:
        b_oc1 = bid_oc_levels[0].astype(np.float64)
        a_oc1 = ask_oc_levels[0].astype(np.float64)
        feat['bid_oc_L1'] = b_oc1
        feat['ask_oc_L1'] = a_oc1
        b_sz1 = bid_sz_levels[0].astype(np.float64)
        a_sz1 = ask_sz_levels[0].astype(np.float64)
        feat['avg_order_sz_bid_L1'] = np.where(b_oc1 > 0, b_sz1 / b_oc1, 0.0)
        feat['avg_order_sz_ask_L1'] = np.where(a_oc1 > 0, a_sz1 / a_oc1, 0.0)

    # ── full depth features (backward compat) ──────────────
    if bid_depth_l2_5 is not None:
        feat['bid_depth_l2_5'] = bid_depth_l2_5.astype(np.float64)
        feat['ask_depth_l2_5'] = ask_depth_l2_5.astype(np.float64)
        total_depth = bid_depth_l2_5 + ask_depth_l2_5
        feat['depth_imbalance'] = np.where(
            total_depth > 0,
            (bid_depth_l2_5 - ask_depth_l2_5).astype(np.float64) / np.maximum(total_depth, 1),
            0.0)
    if n_bid_levels is not None:
        feat['n_bid_levels'] = n_bid_levels.astype(np.float64)
        feat['n_ask_levels'] = n_ask_levels.astype(np.float64)

    # ── raw Databento fields ───────────────────────────────
    if trade_ts_in_delta is not None:
        feat['ts_in_delta_ns'] = trade_ts_in_delta.astype(np.float64)
    if trade_flags is not None:
        feat['flag_last']     = ((trade_flags & FLAG_LAST) > 0).astype(np.float64)
        feat['flag_tob']      = ((trade_flags & FLAG_TOB) > 0).astype(np.float64)
        feat['flag_snapshot']  = ((trade_flags & FLAG_SNAPSHOT) > 0).astype(np.float64)
    if trade_sequence is not None:
        seq_diff = np.empty(n, dtype=np.float64)
        seq_diff[0] = 0
        seq_diff[1:] = np.diff(trade_sequence.astype(np.float64))
        feat['sequence_gap'] = seq_diff
    if trade_ts_recv is not None and ts is not None:
        feat['recv_event_delta_ns'] = (trade_ts_recv - ts).astype(np.float64)

    # ── cumulative trade volumes ───────────────────────────
    buy_mask = (trade_side == 0).astype(np.float64)
    sell_mask = (trade_side == 1).astype(np.float64)
    cum_buy_vol = np.cumsum(trade_size * buy_mask)
    cum_sell_vol = np.cumsum(trade_size * sell_mask)

    # ── realized vol (log returns, scale-free) ─────────────
    mid_safe = np.maximum(mid.astype(np.float64), 1e-6)
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = 0.0
    log_ret[1:] = np.log(mid_safe[1:]) - np.log(mid_safe[:-1])
    cum_mid_sq = np.cumsum(log_ret ** 2)

    # ── book resilience (L1 refill speed) ──────────────────
    if has_levels:
        b1 = bid_sz_levels[0].astype(np.float64)
        a1 = ask_sz_levels[0].astype(np.float64)
        b1_diff = np.empty(n); b1_diff[0] = 0.0; b1_diff[1:] = np.diff(b1)
        a1_diff = np.empty(n); a1_diff[0] = 0.0; a1_diff[1:] = np.diff(a1)
        feat['l1_bid_change'] = b1_diff
        feat['l1_ask_change'] = a1_diff

    # ── pad cumulative arrays ──────────────────────────────
    def pad(a):
        return np.concatenate([[0.0], a.astype(np.float64)])

    p_absz = pad(cum_add_bid_sz)
    p_aasz = pad(cum_add_ask_sz)
    p_cbsz = pad(cum_cancel_bid_sz)
    p_casz = pad(cum_cancel_ask_sz)
    p_cbn  = pad(cum_cancel_bid_n)
    p_can  = pad(cum_cancel_ask_n)
    p_bvol = pad(cum_buy_vol)
    p_svol = pad(cum_sell_vol)
    p_msq  = pad(cum_mid_sq)

    has_mods  = cum_modify_bid_n is not None
    has_fills = cum_fill_bid_n is not None
    if has_mods:
        p_mbn = pad(cum_modify_bid_n)
        p_man = pad(cum_modify_ask_n)
    if has_fills:
        p_fbn  = pad(cum_fill_bid_n)
        p_fan  = pad(cum_fill_ask_n)
        p_fbsz = pad(cum_fill_bid_sz)
        p_fasz = pad(cum_fill_ask_sz)

    p_bsz = pad(bid_size_l1.astype(np.float64))
    p_asz = pad(ask_size_l1.astype(np.float64))

    has_delta = trade_ts_in_delta is not None
    if has_delta:
        delta_f = trade_ts_in_delta.astype(np.float64)
        p_delta = pad(np.cumsum(delta_f))
        p_delta_sq = pad(np.cumsum(delta_f ** 2))

    # level-relative event pads
    has_level_events = cum_add_bid_L1 is not None
    if has_level_events:
        p_ab1 = pad(cum_add_bid_L1); p_ab2 = pad(cum_add_bid_L2); p_abd = pad(cum_add_bid_deep)
        p_aa1 = pad(cum_add_ask_L1); p_aa2 = pad(cum_add_ask_L2); p_aad = pad(cum_add_ask_deep)
        p_cb1 = pad(cum_canc_bid_L1); p_cb2 = pad(cum_canc_bid_L2); p_cbd = pad(cum_canc_bid_deep)
        p_ca1 = pad(cum_canc_ask_L1); p_ca2 = pad(cum_canc_ask_L2); p_cad = pad(cum_canc_ask_deep)

    # fill-level pads (trade depletion on passive side)
    has_fill_levels = cum_fill_bid_L1 is not None
    if has_fill_levels:
        p_fb1 = pad(cum_fill_bid_L1); p_fb2 = pad(cum_fill_bid_L2); p_fbd_lvl = pad(cum_fill_bid_deep)
        p_fa1 = pad(cum_fill_ask_L1); p_fa2 = pad(cum_fill_ask_L2); p_fad_lvl = pad(cum_fill_ask_deep)

    # quote dynamics pads (Hasbrouck & Saar 2009)
    has_quote_dyn = cum_flicker is not None
    if has_quote_dyn:
        p_bid_up = pad(cum_bid_up); p_bid_dn = pad(cum_bid_dn)
        p_ask_up = pad(cum_ask_up); p_ask_dn = pad(cum_ask_dn)
        p_sp_widen = pad(cum_spread_widen); p_sp_narrow = pad(cum_spread_narrow)
        p_flicker = pad(cum_flicker)

    # wall-time weighted TWAS: use message-resolution accumulators from
    # the backtest loop (cum_spread_time, cum_wall_time) so the spread
    # trajectory between trades is properly integrated.
    has_wall_twas = cum_spread_time is not None and cum_wall_time is not None
    if has_wall_twas:
        p_spread_time = pad(cum_spread_time.astype(np.float64))
        p_wall_time = pad(cum_wall_time.astype(np.float64))

    # trade-level TWAS fallback: cumsum of (spread * inter-trade-dt)
    trade_dt = np.empty(n, dtype=np.float64)
    trade_dt[0] = 0.0
    trade_dt[1:] = np.diff(ts).astype(np.float64)
    trade_dt = np.clip(trade_dt, 0, 60_000_000_000)
    spread_f = spread.astype(np.float64)
    spread_x_dt = np.empty(n, dtype=np.float64)
    spread_x_dt[0] = 0.0
    spread_x_dt[1:] = spread_f[:-1] * trade_dt[1:]
    p_spread_dt = pad(np.cumsum(spread_x_dt))
    p_cum_dt = pad(np.cumsum(trade_dt))

    # resilience pads
    if has_levels:
        p_b1_diff_abs = pad(np.cumsum(np.abs(b1_diff)))
        p_a1_diff_abs = pad(np.cumsum(np.abs(a1_diff)))

    idx_end = np.arange(1, n + 1)

    # ── rolling features per time window ───────────────────
    for wns in windows_ns:
        wms = wns // 1_000_000
        s = f'_{wms}ms'

        lb = np.searchsorted(ts, ts - wns)
        n_trades = (idx_end - lb).astype(np.float64)

        # aggregate OFI (backward compat)
        ofi = (
            (p_absz[idx_end] - p_absz[lb])
            - (p_cbsz[idx_end] - p_cbsz[lb])
            - (p_aasz[idx_end] - p_aasz[lb])
            + (p_casz[idx_end] - p_casz[lb])
        )
        feat[f'ofi{s}'] = ofi

        # level-decomposed OFI (Cont, Kukanov, Stoikov 2014):
        #   bid_net = adds - cancels - fills_that_hit_bid
        #   ask_net = adds - cancels - fills_that_hit_ask
        #   OFI_Lk  = bid_net_Lk - ask_net_Lk
        # Fills (trade-induced depletion) are the aggressor-side flow;
        # omitting them biases L1 OFI toward passive (quote) churn and
        # under-represents aggressive trade pressure.
        if has_level_events:
            bid_add_L1 = p_ab1[idx_end] - p_ab1[lb]
            bid_add_L2 = p_ab2[idx_end] - p_ab2[lb]
            bid_add_deep = p_abd[idx_end] - p_abd[lb]
            ask_add_L1 = p_aa1[idx_end] - p_aa1[lb]
            ask_add_L2 = p_aa2[idx_end] - p_aa2[lb]
            ask_add_deep = p_aad[idx_end] - p_aad[lb]

            bid_canc_L1 = p_cb1[idx_end] - p_cb1[lb]
            bid_canc_L2 = p_cb2[idx_end] - p_cb2[lb]
            bid_canc_deep = p_cbd[idx_end] - p_cbd[lb]
            ask_canc_L1 = p_ca1[idx_end] - p_ca1[lb]
            ask_canc_L2 = p_ca2[idx_end] - p_ca2[lb]
            ask_canc_deep = p_cad[idx_end] - p_cad[lb]

            if has_fill_levels:
                bid_fill_L1 = p_fb1[idx_end] - p_fb1[lb]
                bid_fill_L2 = p_fb2[idx_end] - p_fb2[lb]
                bid_fill_deep = p_fbd_lvl[idx_end] - p_fbd_lvl[lb]
                ask_fill_L1 = p_fa1[idx_end] - p_fa1[lb]
                ask_fill_L2 = p_fa2[idx_end] - p_fa2[lb]
                ask_fill_deep = p_fad_lvl[idx_end] - p_fad_lvl[lb]
            else:
                bid_fill_L1 = bid_fill_L2 = bid_fill_deep = 0.0
                ask_fill_L1 = ask_fill_L2 = ask_fill_deep = 0.0

            ofi_L1 = ((bid_add_L1 - bid_canc_L1 - bid_fill_L1)
                      - (ask_add_L1 - ask_canc_L1 - ask_fill_L1))
            ofi_L2 = ((bid_add_L2 - bid_canc_L2 - bid_fill_L2)
                      - (ask_add_L2 - ask_canc_L2 - ask_fill_L2))
            ofi_deep = ((bid_add_deep - bid_canc_deep - bid_fill_deep)
                        - (ask_add_deep - ask_canc_deep - ask_fill_deep))
            feat[f'ofi_L1{s}'] = ofi_L1
            feat[f'ofi_L2{s}'] = ofi_L2
            feat[f'ofi_deep{s}'] = ofi_deep

            # cancel intensity at L1 vs deep (market maker fleeing signal)
            canc_L1 = bid_canc_L1 + ask_canc_L1
            canc_deep = bid_canc_deep + ask_canc_deep
            canc_total = canc_L1 + canc_deep + 1e-10
            feat[f'canc_L1_share{s}'] = canc_L1 / canc_total

        # trade counts and volumes
        buy_vol  = p_bvol[idx_end] - p_bvol[lb]
        sell_vol = p_svol[idx_end] - p_svol[lb]
        total_vol = buy_vol + sell_vol
        signed_vol = buy_vol - sell_vol

        feat[f'n_trades{s}'] = n_trades
        feat[f'signed_volume{s}'] = signed_vol
        feat[f'vol_imbalance{s}'] = np.where(total_vol > 0,
                                              np.abs(signed_vol) / total_vol, 0.0)

        n_canc = (p_cbn[idx_end] - p_cbn[lb]) + (p_can[idx_end] - p_can[lb])
        feat[f'cancel_trade_ratio{s}'] = n_canc / np.maximum(n_trades, 1.0)

        if has_mods:
            n_mods = (p_mbn[idx_end] - p_mbn[lb]) + (p_man[idx_end] - p_man[lb])
            feat[f'modify_trade_ratio{s}'] = n_mods / np.maximum(n_trades, 1.0)

        if has_fills:
            fill_bid = p_fbn[idx_end] - p_fbn[lb]
            fill_ask = p_fan[idx_end] - p_fan[lb]
            fill_total = fill_bid + fill_ask
            fa_out = np.zeros(n, dtype=np.float64)
            np.divide(fill_bid - fill_ask, fill_total, out=fa_out, where=fill_total > 0)
            feat[f'fill_asymmetry{s}'] = fa_out
            fill_bid_sz = p_fbsz[idx_end] - p_fbsz[lb]
            fill_ask_sz = p_fasz[idx_end] - p_fasz[lb]
            fill_total_sz = fill_bid_sz + fill_ask_sz
            fsa_out = np.zeros(n, dtype=np.float64)
            np.divide(fill_bid_sz - fill_ask_sz, fill_total_sz, out=fsa_out, where=fill_total_sz > 0)
            feat[f'fill_size_asymmetry{s}'] = fsa_out

        feat[f'trade_rate{s}'] = n_trades / (wms / 1000.0)

        sum_sq = p_msq[idx_end] - p_msq[lb]
        feat[f'realized_vol{s}'] = np.sqrt(sum_sq)

        bsz_lb = bid_size_l1.astype(np.float64)[np.clip(lb, 0, n - 1)]
        bsz_now = bid_size_l1.astype(np.float64)
        asz_lb = ask_size_l1.astype(np.float64)[np.clip(lb, 0, n - 1)]
        asz_now = ask_size_l1.astype(np.float64)
        feat[f'bid_depletion{s}'] = bsz_lb - bsz_now
        feat[f'ask_depletion{s}'] = asz_lb - asz_now

        if has_delta:
            delta_sum = p_delta[idx_end] - p_delta[lb]
            feat[f'mean_latency_ns{s}'] = delta_sum / np.maximum(n_trades, 1.0)
            delta_sq_sum = p_delta_sq[idx_end] - p_delta_sq[lb]
            mean_sq = delta_sq_sum / np.maximum(n_trades, 1.0)
            mean_val = feat[f'mean_latency_ns{s}']
            feat[f'latency_std_ns{s}'] = np.sqrt(np.maximum(mean_sq - mean_val**2, 0.0))

        # book resilience: total L1 churn in window (Bouchaud et al. 2009)
        if has_levels:
            b_churn = p_b1_diff_abs[idx_end] - p_b1_diff_abs[lb]
            a_churn = p_a1_diff_abs[idx_end] - p_a1_diff_abs[lb]
            feat[f'resilience_bid{s}'] = b_churn / np.maximum(n_trades, 1.0)
            feat[f'resilience_ask{s}'] = a_churn / np.maximum(n_trades, 1.0)

        # quote dynamics features (Hasbrouck & Saar 2009, Huang & Stoll 1997)
        if has_quote_dyn:
            flicker_w = p_flicker[idx_end] - p_flicker[lb]
            feat[f'quote_flicker{s}'] = flicker_w

            bid_up_w = p_bid_up[idx_end] - p_bid_up[lb]
            bid_dn_w = p_bid_dn[idx_end] - p_bid_dn[lb]
            ask_up_w = p_ask_up[idx_end] - p_ask_up[lb]
            ask_dn_w = p_ask_dn[idx_end] - p_ask_dn[lb]

            # net BBO move rate: positive = bid moving up / ask moving up
            bbo_moves = bid_up_w + bid_dn_w + ask_up_w + ask_dn_w
            feat[f'bid_move_rate{s}'] = (bid_up_w - bid_dn_w) / np.maximum(bbo_moves, 1.0)
            feat[f'ask_move_rate{s}'] = (ask_up_w - ask_dn_w) / np.maximum(bbo_moves, 1.0)

            # spread widen rate: fraction of spread changes that were
            # widenings. Default to 0 (no widening events) instead of
            # 0.5 so z-scoring doesn't produce spikes every time the
            # window is quiet (most ES windows).
            sp_w = p_sp_widen[idx_end] - p_sp_widen[lb]
            sp_n = p_sp_narrow[idx_end] - p_sp_narrow[lb]
            sp_total = sp_w + sp_n
            widen_rate = np.zeros(n, dtype=np.float64)
            np.divide(sp_w, sp_total, out=widen_rate, where=sp_total > 0)
            feat[f'spread_widen_rate{s}'] = widen_rate

        # time-weighted average spread (Easley et al. 1997). Prefer the
        # message-resolution accumulator (cum_spread_time / cum_wall_time)
        # so we integrate the actual spread trajectory between trades,
        # not just the spread seen at trade events.
        if has_wall_twas:
            sp_t_w = p_spread_time[idx_end] - p_spread_time[lb]
            wall_w = p_wall_time[idx_end] - p_wall_time[lb]
            twas_out = np.full(n, np.nan, dtype=np.float64)
            np.divide(sp_t_w, wall_w, out=twas_out, where=wall_w > 0)
        else:
            sp_dt_w = p_spread_dt[idx_end] - p_spread_dt[lb]
            dt_w = p_cum_dt[idx_end] - p_cum_dt[lb]
            twas_out = np.full(n, np.nan, dtype=np.float64)
            np.divide(sp_dt_w, dt_w, out=twas_out, where=dt_w > 0)
        feat[f'twas{s}'] = twas_out

    # ── forward returns (mid-based) ────────────────────────
    for hns in horizons_ns:
        hms = hns // 1_000_000
        fwd_idx = np.searchsorted(ts, ts + hns)
        valid = fwd_idx < n
        fwd_idx_c = np.minimum(fwd_idx, n - 1)
        fwd_ret = mid[fwd_idx_c] - mid
        fwd_ret[~valid] = np.nan
        feat[f'fwd_return_{hms}ms'] = fwd_ret

    # ── forward returns (microprice-based, Stoikov 2018) ───
    if microprice is not None:
        mp = microprice.astype(np.float64)
        for hns in horizons_ns:
            hms = hns // 1_000_000
            fwd_idx = np.searchsorted(ts, ts + hns)
            valid = fwd_idx < n
            fwd_idx_c = np.minimum(fwd_idx, n - 1)
            fwd_ret = mp[fwd_idx_c] - mp
            fwd_ret[~valid] = np.nan
            feat[f'fwd_mp_{hms}ms'] = fwd_ret

    feat['ts_ns'] = ts
    return pd.DataFrame(feat)
