"""
Toxicity prediction with proper ML methodology:

Research phase:
  - Purged + embargoed time-series K-fold cross-validation
  - Out-of-fold (OOF) predictions for unbiased evaluation
  - Isotonic regression calibration (raw score -> P(toxic))
  - Ridge regression base model (interpretable, fast)

Production phase:
  - Recursive Least Squares (RLS) for online coefficient adaptation
  - Exponentially-weighted forgetting for non-stationarity

The purge window removes samples whose forward-return label
overlaps with the test fold.  The embargo adds an extra gap
to kill any autocorrelation leakage.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet

from .regressors import get_regressor_names, compute_regressors, regressor_diagnostics


def _purged_embargo_split(ts_ns, fold_start_idx, fold_end_idx,
                          purge_ns, embargo_ns):
    """
    Build train mask that excludes:
      - The test fold itself
      - Purge zone: any sample whose forward label overlaps the test fold
      - Embargo zone: extra buffer after test fold ends

    Returns boolean mask over ALL samples (True = usable for training).
    """
    n = len(ts_ns)
    mask = np.ones(n, dtype=bool)

    # exclude test fold
    mask[fold_start_idx:fold_end_idx] = False

    test_start_ts = ts_ns[fold_start_idx]
    test_end_ts = ts_ns[min(fold_end_idx - 1, n - 1)]

    # purge: train samples before test whose label horizon reaches into test
    purge_boundary = test_start_ts - purge_ns
    for i in range(fold_start_idx - 1, -1, -1):
        if ts_ns[i] < purge_boundary:
            break
        mask[i] = False

    # embargo: samples after test fold that are too close
    embargo_boundary = test_end_ts + embargo_ns
    for i in range(fold_end_idx, n):
        if ts_ns[i] > embargo_boundary:
            break
        mask[i] = False

    return mask


class ToxicityModel:
    """
    Predicts signed forward return from microstructure features.

    Training uses purged/embargoed K-fold CV with isotonic calibration.
    Online adaptation uses RLS with exponential forgetting.
    """

    def __init__(self, forgetting_factor: float = 0.999,
                 delta: float = 100.0,
                 target: str = 'fwd_return_1000ms',
                 n_folds: int = 5,
                 purge_ns: int = 10_000_000_000,   # 10s (must exceed label horizon)
                 embargo_ns: int = 5_000_000_000):  # 5s extra buffer
        self.lam = forgetting_factor
        self.target = target
        self.n_folds = n_folds
        self.purge_ns = purge_ns
        self.embargo_ns = embargo_ns

        self.n_feat = len(get_regressor_names())
        self.active_features: list[str] = []

        # RLS state (intercept + features)
        self.w = np.zeros(self.n_feat + 1)
        self.P = np.eye(self.n_feat + 1) * delta

        # normalisation params
        self.mean = np.zeros(self.n_feat)
        self.var  = np.ones(self.n_feat)
        self.n_seen = 0

        # isotonic calibrator (raw score -> calibrated probability)
        self.calibrator: IsotonicRegression | None = None

        # diagnostics
        self.cumulative_sq_error = 0.0
        self.n_updates = 0
        self.oof_stats: dict = {}

    def _resolve_features(self, df) -> list[str]:
        all_names = get_regressor_names()
        available = [f for f in all_names if f in df.columns]
        if not available:
            raise ValueError("No matching regressor columns found")
        return available

    # ══════════════════════════════════════════════
    #  RESEARCH PHASE: Purged/embargoed OOF training
    # ══════════════════════════════════════════════

    def fit_purged_cv(self, df) -> dict:
        """
        Purged + embargoed K-fold CV with OOF predictions.

        1. Split data into K time-ordered folds
        2. For each fold: purge + embargo -> train ridge -> predict OOF
        3. Fit isotonic regression on all OOF predictions
        4. Refit final model on all data for production weights
        """
        self.active_features = self._resolve_features(df)
        self.n_feat = len(self.active_features)

        X_all = df[self.active_features].values.astype(np.float64)
        y_all = df[self.target].values.astype(np.float64)
        ts_all = df['ts_ns'].values.astype(np.int64)

        valid_mask = ~np.isnan(y_all)
        X_all = np.nan_to_num(X_all)

        n = len(X_all)
        fold_size = n // self.n_folds

        # global normalisation stats
        self.mean = np.nanmean(X_all, axis=0)
        self.var  = np.nanvar(X_all, axis=0)
        self.n_seen = n

        X_norm = (X_all - self.mean) / np.sqrt(self.var + 1e-8)
        X_design = np.column_stack([np.ones(n), X_norm])

        oof_preds = np.full(n, np.nan)
        fold_r2s = []

        print(f"    Purged/embargoed {self.n_folds}-fold CV "
              f"(purge={self.purge_ns/1e9:.0f}s, embargo={self.embargo_ns/1e9:.0f}s)")

        for k in range(self.n_folds):
            fold_start = k * fold_size
            fold_end = (k + 1) * fold_size if k < self.n_folds - 1 else n

            train_mask = _purged_embargo_split(
                ts_all, fold_start, fold_end,
                self.purge_ns, self.embargo_ns)
            train_mask &= valid_mask

            test_mask = np.zeros(n, dtype=bool)
            test_mask[fold_start:fold_end] = True
            test_mask &= valid_mask

            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < 100 or n_test < 10:
                print(f"    Fold {k}: skipped (train={n_train}, test={n_test})")
                continue

            X_tr = X_design[train_mask]
            y_tr = y_all[train_mask]
            X_te = X_design[test_mask]
            y_te = y_all[test_mask]

            # ridge regression
            alpha = 1.0
            XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
            w_fold = np.linalg.solve(XtX, X_tr.T @ y_tr)

            y_hat = X_te @ w_fold
            oof_preds[test_mask] = y_hat

            res = y_te - y_hat
            ss_res = np.sum(res ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)
            fold_r2s.append(r2)

            # fold-level diagnostics: time range and vol for regime identification
            fold_ts = ts_all[test_mask]
            ts_start_s = fold_ts[0] / 1e9
            ts_end_s = fold_ts[-1] / 1e9
            duration_h = (ts_end_s - ts_start_s) / 3600
            y_std = float(np.std(y_te))
            print(f"    Fold {k}: train={n_train:,}  test={n_test:,}  "
                  f"R2={r2:.4f}  RMSE={np.sqrt(np.mean(res**2)):.6f}  "
                  f"y_std={y_std:.4f}  span={duration_h:.1f}h")

        # ── isotonic calibration on OOF predictions ───────
        oof_valid = ~np.isnan(oof_preds) & valid_mask
        n_oof = oof_valid.sum()

        if n_oof > 100:
            oof_scores = oof_preds[oof_valid]
            oof_true = y_all[oof_valid]

            # for calibration: convert continuous return to binary
            # "toxic" = adverse move exceeds half a tick
            toxic_threshold = 0.0
            oof_binary = (np.abs(oof_true) > toxic_threshold).astype(float)

            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds='clip')
            self.calibrator.fit(np.abs(oof_scores), oof_binary)
            oof_calib = self.calibrator.transform(np.abs(oof_scores))

            print(f"    Isotonic calibration: {n_oof:,} OOF samples, "
                  f"mean P(toxic)={oof_calib.mean():.3f}")

            # OOF aggregate stats
            oof_res = oof_true - oof_scores
            oof_r2 = 1.0 - np.var(oof_res) / (np.var(oof_true) + 1e-10)
            oof_rmse = float(np.sqrt(np.mean(oof_res ** 2)))
        else:
            oof_r2 = float('nan')
            oof_rmse = float('nan')
            print(f"    WARNING: only {n_oof} valid OOF samples, skipping calibration")

        # ── final model on ALL data (for production weights) ──
        all_valid = valid_mask
        X_tr = X_design[all_valid]
        y_tr = y_all[all_valid]
        alpha = 1.0
        XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
        self.w = np.linalg.solve(XtX, X_tr.T @ y_tr)
        self.P = np.linalg.inv(XtX) * len(y_tr)

        # re-init RLS state sized to actual features
        # (P already set above, w already set)

        self.oof_stats = {
            'oof_r2': float(oof_r2),
            'oof_rmse': float(oof_rmse),
            'fold_r2s': fold_r2s,
            'n_oof': int(n_oof),
            'n_features': self.n_feat,
            'n_folds': self.n_folds,
            'purge_ns': self.purge_ns,
            'embargo_ns': self.embargo_ns,
        }

        return self.oof_stats

    # kept for backward compat, delegates to CV version
    def fit_batch(self, df) -> dict:
        stats = self.fit_purged_cv(df)
        return {
            'r2': stats['oof_r2'],
            'rmse': stats['oof_rmse'],
            'n_samples': stats['n_oof'],
            'n_features': stats['n_features'],
        }

    # ══════════════════════════════════════════════
    #  ONLINE PHASE: RLS adaptation
    # ══════════════════════════════════════════════

    def _update_norm(self, x: np.ndarray):
        self.n_seen += 1
        d1 = x - self.mean
        self.mean += d1 / self.n_seen
        d2 = x - self.mean
        self.var += (d1 * d2 - self.var) / self.n_seen

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    @staticmethod
    def _design(x_norm: np.ndarray) -> np.ndarray:
        return np.concatenate([[1.0], x_norm])

    def update(self, x_raw: np.ndarray, y_true: float) -> float:
        """One-sample RLS step. Returns predicted value BEFORE update."""
        self._update_norm(x_raw)
        x = self._design(self._normalize(x_raw))

        # guard against dimension mismatch if features changed
        if len(x) != len(self.w):
            return 0.0

        Px = self.P @ x
        y_pred = float(x @ self.w)
        error = y_true - y_pred

        gain = Px / (self.lam + x @ Px)
        self.w += gain * error
        self.P = (self.P - np.outer(gain, Px)) / self.lam

        self.cumulative_sq_error += error ** 2
        self.n_updates += 1
        return y_pred

    # ══════════════════════════════════════════════
    #  PREDICTION
    # ══════════════════════════════════════════════

    def predict_single(self, x_raw: np.ndarray) -> float:
        return float(self._design(self._normalize(x_raw)) @ self.w)

    def predict_batch(self, df) -> np.ndarray:
        cols = self.active_features if self.active_features else get_regressor_names()
        available = [c for c in cols if c in df.columns]
        X_raw = df[available].values.astype(np.float64)
        X_norm = (X_raw - self.mean[:len(available)]) / np.sqrt(self.var[:len(available)] + 1e-8)
        X_norm = np.nan_to_num(X_norm)
        X = np.column_stack([np.ones(len(X_norm)), X_norm])
        return X @ self.w[:X.shape[1]]

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw regression scores to calibrated P(toxic) via isotonic."""
        if self.calibrator is None:
            return np.abs(raw_scores)
        return self.calibrator.transform(np.abs(raw_scores))

    # ══════════════════════════════════════════════
    #  DIAGNOSTICS
    # ══════════════════════════════════════════════

    @property
    def feature_importance(self) -> list[tuple[str, float]]:
        cols = self.active_features if self.active_features else get_regressor_names()
        n = min(len(cols), len(self.w) - 1)
        imp = np.abs(self.w[1:n+1])
        order = np.argsort(-imp)
        return [(cols[i], float(imp[i])) for i in order]

    @property
    def online_rmse(self) -> float:
        if self.n_updates == 0:
            return 0.0
        return float(np.sqrt(self.cumulative_sq_error / self.n_updates))
