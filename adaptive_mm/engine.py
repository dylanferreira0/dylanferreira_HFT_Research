"""
Spread regime formulas for the adaptive market maker.

These define the math that C++ will implement:
  S(t) = S_base * vol_ratio * f(GEX) * g(inventory) * h(toxicity)

No simulation here -- this is reference logic for the C++ implementation.
"""

import numpy as np
from .lob import TICK_SIZE, POINT_VALUE


class SpreadRegime:
    """
    S(t) = S_base * vol_ratio * f(GEX_local) * g(inventory) * h(toxicity)

    Without options data f(GEX)=1.  Everything else is live.
    """

    def __init__(self, base_spread_ticks: int = 2, max_spread_ticks: int = 8,
                 inventory_penalty: float = 0.5, toxicity_scale: float = 4.0):
        self.base = base_spread_ticks
        self.max_spread = max_spread_ticks
        self.inv_penalty = inventory_penalty
        self.tox_scale = toxicity_scale

    def compute_spread(self, vol_fast: float, vol_slow: float,
                       inventory: int, toxicity_score: float,
                       gex_local: float = 0.0) -> int:
        vol_ratio = np.clip(vol_fast / (vol_slow + 1e-10), 0.5, 3.0)
        g = 1.0 + self.inv_penalty * abs(inventory) / 10.0
        h = 1.0 + self.tox_scale * abs(toxicity_score)
        # gex_local must be pre-normalized to [-1, 1] by the caller;
        # clamp defensively so raw GEX values can't flip or collapse f.
        f = 1.0 - 0.3 * np.clip(gex_local, -1.0, 1.0)
        raw = self.base * vol_ratio * g * h * f
        return int(np.clip(round(raw), 1, self.max_spread))

    @staticmethod
    def compute_skew(inventory: int, toxicity_score: float) -> int:
        inv_skew = -np.sign(inventory) * min(abs(inventory) / 5.0, 2.0)
        tox_skew = -np.sign(toxicity_score) * min(abs(toxicity_score) / TICK_SIZE, 1.5)
        return int(round(inv_skew + tox_skew))

    def to_dict(self) -> dict:
        """Export parameters for C++ implementation."""
        return {
            'base_spread_ticks': self.base,
            'max_spread_ticks': self.max_spread,
            'inventory_penalty': self.inv_penalty,
            'toxicity_scale': self.tox_scale,
        }
