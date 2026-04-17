"""
Regime-switching Markov chain for inter-level price dynamics.

State = (spread_bucket, imbalance_bucket)
Observation = next price change in ticks
Regime = realized-vol bucket (low / medium / high)

Estimates P(price_change | state, regime) as a smoothed histogram.
Provides fair-value offsets and uncertainty bands for the quote engine.
"""

import numpy as np


class MarkovModel:

    N_SPREAD_BUCKETS    = 3   # tight(1 tick), normal(2), wide(3+)
    N_IMBALANCE_BUCKETS = 3   # bid-heavy / balanced / ask-heavy
    N_STATES            = N_SPREAD_BUCKETS * N_IMBALANCE_BUCKETS  # 9
    MAX_CHANGE          = 5   # cap at ±5 ticks
    N_OUTCOMES          = 2 * MAX_CHANGE + 1                      # 11
    N_REGIMES           = 3   # low / mid / high vol

    def __init__(self, vol_quantiles=(0.33, 0.67)):
        self.vol_quantiles = vol_quantiles
        self.vol_thresholds: np.ndarray | None = None

        self.counts = np.zeros((self.N_REGIMES, self.N_STATES, self.N_OUTCOMES))
        self.probs  = np.ones((self.N_REGIMES, self.N_STATES, self.N_OUTCOMES)) / self.N_OUTCOMES
        self.outcome_values = np.arange(-self.MAX_CHANGE, self.MAX_CHANGE + 1, dtype=np.float64)

    # -- discretisers --

    @staticmethod
    def discretize_spread(spread_ticks: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(spread_ticks, dtype=int) - 1, 0, 2)

    @staticmethod
    def discretize_imbalance(imbalance: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(imbalance, dtype=np.float64), [-0.33, 0.33])

    def state_index(self, sp_bucket: np.ndarray, imb_bucket: np.ndarray) -> np.ndarray:
        return sp_bucket * self.N_IMBALANCE_BUCKETS + imb_bucket

    def discretize_price_change(self, change_ticks: np.ndarray) -> np.ndarray:
        capped = np.clip(np.asarray(change_ticks, dtype=int),
                         -self.MAX_CHANGE, self.MAX_CHANGE)
        return capped + self.MAX_CHANGE

    def detect_regimes(self, realized_vol: np.ndarray) -> np.ndarray:
        rv = np.asarray(realized_vol, dtype=np.float64)
        if self.vol_thresholds is None:
            positive = rv[rv > 0]
            if len(positive) < 10:
                self.vol_thresholds = np.array([0.0, 1e10])
            else:
                self.vol_thresholds = np.quantile(positive, list(self.vol_quantiles))
        return np.clip(np.digitize(rv, self.vol_thresholds), 0, self.N_REGIMES - 1)

    # -- estimation --

    def fit(self, spread_ticks, imbalance, price_change_ticks, realized_vol):
        """Estimate regime-conditional price-change distributions from training data."""
        sp  = self.discretize_spread(spread_ticks)
        imb = self.discretize_imbalance(imbalance)
        states   = self.state_index(sp, imb)
        outcomes = self.discretize_price_change(price_change_ticks)
        regimes  = self.detect_regimes(realized_vol)

        self.counts[:] = 0
        n = len(states)
        for i in range(n):
            self.counts[regimes[i], states[i], outcomes[i]] += 1

        smoothed = self.counts + 1.0
        self.probs = smoothed / smoothed.sum(axis=2, keepdims=True)
        return self

    # -- prediction --

    def _lookup(self, spread_ticks_val, imbalance_val, vol_val):
        sp  = self.discretize_spread(np.array([spread_ticks_val]))[0]
        imb = self.discretize_imbalance(np.array([imbalance_val]))[0]
        s   = int(sp * self.N_IMBALANCE_BUCKETS + imb)
        r   = self.detect_regimes(np.array([vol_val]))[0]
        return int(r), s

    def predict_fair_value_offset(self, spread_ticks_val, imbalance_val,
                                  realized_vol_val, steps: int = 1) -> float:
        """Expected cumulative price change in ticks over `steps` transitions."""
        r, s = self._lookup(spread_ticks_val, imbalance_val, realized_vol_val)
        expected = float(np.dot(self.probs[r, s], self.outcome_values))
        return expected * steps

    def predict_distribution(self, spread_ticks_val, imbalance_val, realized_vol_val):
        r, s = self._lookup(spread_ticks_val, imbalance_val, realized_vol_val)
        return self.outcome_values.copy(), self.probs[r, s].copy()

    def uncertainty(self, spread_ticks_val, imbalance_val, realized_vol_val) -> float:
        """Std-dev of the predicted 1-step price change (ticks)."""
        values, probs = self.predict_distribution(
            spread_ticks_val, imbalance_val, realized_vol_val)
        mu = np.dot(probs, values)
        return float(np.sqrt(np.dot(probs, (values - mu) ** 2)))

    # -- vectorised prediction for full test set --

    def predict_offsets_batch(self, spread_ticks, imbalance, realized_vol) -> np.ndarray:
        sp  = self.discretize_spread(spread_ticks)
        imb = self.discretize_imbalance(imbalance)
        states  = self.state_index(sp, imb)
        regimes = self.detect_regimes(realized_vol)
        out = np.empty(len(states))
        for i in range(len(states)):
            out[i] = np.dot(self.probs[regimes[i], states[i]], self.outcome_values)
        return out

    def summary(self):
        """Print per-regime distribution statistics."""
        lines = []
        for r in range(self.N_REGIMES):
            avg_dist = self.probs[r].mean(axis=0)
            mu  = np.dot(avg_dist, self.outcome_values)
            std = np.sqrt(np.dot(avg_dist, self.outcome_values ** 2) - mu ** 2)
            total_obs = int(self.counts[r].sum())
            label = ['LOW-VOL', 'MID-VOL', 'HIGH-VOL'][r]
            lines.append(f"  {label}: E[Δ]={mu:+.3f} ticks  σ={std:.3f} ticks  "
                         f"({total_obs:,} obs)")
        return '\n'.join(lines)
