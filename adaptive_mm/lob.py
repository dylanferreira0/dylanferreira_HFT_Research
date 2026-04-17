"""
Limit Order Book reconstruction from Databento MBO messages.
Tracks every order (add/cancel/modify/fill) to maintain full book state.

Optimised for the hot path: no isinstance checks, no sorting on depth queries.
"""

from collections import defaultdict
from typing import Optional

TICK_SIZE = 0.25
POINT_VALUE = 50.0

# action codes (avoids string comparisons in hot loop)
ACT_ADD    = 0
ACT_CANCEL = 1
ACT_MODIFY = 2
ACT_TRADE  = 3
ACT_FILL   = 4
ACT_CLEAR  = 5
ACT_UNKNOWN = 6

# side codes
SIDE_BID = 0
SIDE_ASK = 1

_ACTION_MAP = {
    'A': ACT_ADD, 'a': ACT_ADD,
    'C': ACT_CANCEL, 'c': ACT_CANCEL,
    'M': ACT_MODIFY, 'm': ACT_MODIFY,
    'T': ACT_TRADE, 't': ACT_TRADE,
    'F': ACT_FILL, 'f': ACT_FILL,
    'R': ACT_CLEAR, 'r': ACT_CLEAR,
}

_SIDE_MAP = {
    'B': SIDE_BID, 'b': SIDE_BID,
    'A': SIDE_ASK, 'a': SIDE_ASK,
    'N': SIDE_ASK, 'n': SIDE_ASK,
}


def preprocess_actions_sides(actions, sides):
    """
    Vectorised pre-conversion of action/side arrays to int8 codes.
    Uses numpy boolean masks instead of a Python for-loop.
    ~12 C-level passes over the array vs 1 Python-level pass = 20-50x faster.
    """
    import numpy as np
    act_arr = np.asarray(actions)
    side_arr = np.asarray(sides)
    n = len(act_arr)

    act_codes = np.full(n, ACT_UNKNOWN, dtype=np.int8)
    for char, code in _ACTION_MAP.items():
        act_codes[act_arr == char] = code

    side_codes = np.full(n, SIDE_ASK, dtype=np.int8)
    for char, code in _SIDE_MAP.items():
        side_codes[side_arr == char] = code

    return act_codes, side_codes


def price_to_ticks(price: float) -> int:
    if price != price:  # NaN
        return 0
    return int(round(price / TICK_SIZE))


def ticks_to_price(ticks: int) -> float:
    return ticks * TICK_SIZE


class TradeEvent:
    __slots__ = ('ts_ns', 'price_ticks', 'size', 'aggressor_side')

    def __init__(self, ts_ns: int, price_ticks: int, size: int, aggressor_side: int):
        self.ts_ns = ts_ns
        self.price_ticks = price_ticks
        self.size = size
        self.aggressor_side = aggressor_side


class LOB:
    """
    Full Limit Order Book with order-level tracking.

    Optimisations vs naive approach:
    - Tracks total_bid_size / total_ask_size incrementally
      so depth queries don't need to sort or sum
    - process_fast() takes pre-converted int codes, no string parsing
    """

    __slots__ = ('orders', 'bid_orders', 'ask_orders', 'bid_sizes', 'ask_sizes',
                 '_best_bid', '_best_ask', 'total_bid_size', 'total_ask_size',
                 'n_msgs', 'n_trades', 'n_adds', 'n_cancels', 'n_mods')

    def __init__(self):
        self.orders: dict[int, tuple] = {}  # oid -> (price_ticks, size, side)
        self.bid_orders: dict[int, dict[int, int]] = defaultdict(dict)
        self.ask_orders: dict[int, dict[int, int]] = defaultdict(dict)
        self.bid_sizes: dict[int, int] = defaultdict(int)
        self.ask_sizes: dict[int, int] = defaultdict(int)
        self._best_bid: Optional[int] = None
        self._best_ask: Optional[int] = None
        self.total_bid_size = 0
        self.total_ask_size = 0
        self.n_msgs = 0
        self.n_trades = 0
        self.n_adds = 0
        self.n_cancels = 0
        self.n_mods = 0

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self._best_bid, self._best_ask
        if bb is not None and ba is not None:
            return (bb + ba) * TICK_SIZE / 2.0
        return None

    @property
    def spread_ticks(self) -> int:
        bb, ba = self._best_bid, self._best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return 0

    def process_fast(self, act_code: int, side_code: int,
                     price_ticks: int, size: int,
                     order_id: int, ts_ns: int) -> Optional[TradeEvent]:
        """
        Hot-path message processor. Takes pre-converted integer codes.
        No string parsing, no isinstance checks.
        """
        self.n_msgs += 1

        if price_ticks <= 0 and act_code not in (ACT_CLEAR, ACT_CANCEL):
            return None

        if act_code == ACT_ADD:
            self.n_adds += 1
            self._add(order_id, price_ticks, size, side_code)
        elif act_code == ACT_CANCEL:
            self.n_cancels += 1
            self._cancel(order_id)
        elif act_code == ACT_MODIFY:
            self.n_mods += 1
            self._cancel(order_id)
            self._add(order_id, price_ticks, size, side_code)
        elif act_code == ACT_TRADE:
            self.n_trades += 1
            self._fill(order_id, size)
            return TradeEvent(ts_ns, price_ticks, size, side_code)
        elif act_code == ACT_FILL:
            self._fill(order_id, size)
        elif act_code == ACT_CLEAR:
            self._clear()

        return None

    def _add(self, oid: int, pt: int, size: int, side: int):
        self.orders[oid] = (pt, size, side)
        if side == SIDE_BID:
            self.bid_orders[pt][oid] = size
            self.bid_sizes[pt] += size
            self.total_bid_size += size
            if self._best_bid is None or pt > self._best_bid:
                self._best_bid = pt
        else:
            self.ask_orders[pt][oid] = size
            self.ask_sizes[pt] += size
            self.total_ask_size += size
            if self._best_ask is None or pt < self._best_ask:
                self._best_ask = pt

    def _cancel(self, oid: int):
        entry = self.orders.pop(oid, None)
        if entry is None:
            return
        pt, _, side = entry
        if side == SIDE_BID:
            sz = self.bid_orders.get(pt, {}).pop(oid, 0)
            if sz:
                self.bid_sizes[pt] -= sz
                self.total_bid_size -= sz
                if self.bid_sizes[pt] <= 0:
                    self.bid_sizes.pop(pt, None)
                    self.bid_orders.pop(pt, None)
                    if pt == self._best_bid:
                        self._best_bid = max(self.bid_sizes) if self.bid_sizes else None
        else:
            sz = self.ask_orders.get(pt, {}).pop(oid, 0)
            if sz:
                self.ask_sizes[pt] -= sz
                self.total_ask_size -= sz
                if self.ask_sizes[pt] <= 0:
                    self.ask_sizes.pop(pt, None)
                    self.ask_orders.pop(pt, None)
                    if pt == self._best_ask:
                        self._best_ask = min(self.ask_sizes) if self.ask_sizes else None

    def _fill(self, oid: int, fill_size: int):
        entry = self.orders.get(oid)
        if entry is None:
            return
        pt, order_size, side = entry
        remaining = order_size - fill_size

        if side == SIDE_BID:
            level = self.bid_orders.get(pt, {})
            if remaining <= 0:
                level.pop(oid, None)
                self.orders.pop(oid, None)
                self.bid_sizes[pt] -= order_size
                self.total_bid_size -= order_size
            else:
                level[oid] = remaining
                self.orders[oid] = (pt, remaining, side)
                self.bid_sizes[pt] -= fill_size
                self.total_bid_size -= fill_size
            if self.bid_sizes.get(pt, 0) <= 0:
                self.bid_sizes.pop(pt, None)
                self.bid_orders.pop(pt, None)
                if pt == self._best_bid:
                    self._best_bid = max(self.bid_sizes) if self.bid_sizes else None
        else:
            level = self.ask_orders.get(pt, {})
            if remaining <= 0:
                level.pop(oid, None)
                self.orders.pop(oid, None)
                self.ask_sizes[pt] -= order_size
                self.total_ask_size -= order_size
            else:
                level[oid] = remaining
                self.orders[oid] = (pt, remaining, side)
                self.ask_sizes[pt] -= fill_size
                self.total_ask_size -= fill_size
            if self.ask_sizes.get(pt, 0) <= 0:
                self.ask_sizes.pop(pt, None)
                self.ask_orders.pop(pt, None)
                if pt == self._best_ask:
                    self._best_ask = min(self.ask_sizes) if self.ask_sizes else None

    def _clear(self):
        self.orders.clear()
        self.bid_orders.clear()
        self.ask_orders.clear()
        self.bid_sizes.clear()
        self.ask_sizes.clear()
        self._best_bid = None
        self._best_ask = None
        self.total_bid_size = 0
        self.total_ask_size = 0

    def l1_and_depth(self):
        """
        Returns (bid_l1_sz, ask_l1_sz, bid_depth_rest, ask_depth_rest)
        WITHOUT sorting.  O(1) using tracked totals.
        """
        bb, ba = self._best_bid, self._best_ask
        b1 = self.bid_sizes.get(bb, 0) if bb is not None else 0
        a1 = self.ask_sizes.get(ba, 0) if ba is not None else 0
        return b1, a1, self.total_bid_size - b1, self.total_ask_size - a1

    def snapshot_levels(self, n: int = 5):
        """
        O(n) level snapshot via tick-offset lookups.
        ES levels are exactly 1 tick apart, so L_k = best +/- k.
        No sorting needed.

        Returns:
          bid_sizes[0..n-1]  (L1 = index 0)
          ask_sizes[0..n-1]
          bid_order_counts[0..n-1]  (unique orders at each level)
          ask_order_counts[0..n-1]
        """
        bb, ba = self._best_bid, self._best_ask
        bs = self.bid_sizes
        as_ = self.ask_sizes
        bo = self.bid_orders
        ao = self.ask_orders

        if bb is not None:
            b_sz = [bs.get(bb - k, 0) for k in range(n)]
            b_oc = [len(bo.get(bb - k, ())) for k in range(n)]
        else:
            b_sz = [0] * n
            b_oc = [0] * n

        if ba is not None:
            a_sz = [as_.get(ba + k, 0) for k in range(n)]
            a_oc = [len(ao.get(ba + k, ())) for k in range(n)]
        else:
            a_sz = [0] * n
            a_oc = [0] * n

        return b_sz, a_sz, b_oc, a_oc
