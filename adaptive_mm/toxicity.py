"""
Toxicity prediction with proper ML methodology:

Research phase:
  - Purged + embargoed time-series K-fold cross-validation
  - Out-of-fold (OOF) predictions for unbiased evaluation
  - Ridge regression with UNPENALIZED intercept (standard form)
  - Per-fold standardization (no test->train leakage via mu/sigma)
  - Alpha selected via OOF R^2 from a log-spaced grid
  - Isotonic regression calibration on a MEANINGFUL binary label
    ("toxic" = |forward return| exceeds half a tick)

Production phase:
  - Recursive Least Squares (RLS) for online coefficient adaptation
  - Exponentially-weighted forgetting for non-stationarity
  - EW running mean/var whose decay matches the RLS forgetting factor
    (so feature normalization drifts in lockstep with the fit)

The purge window removes samples whose forward-return label
overlaps with the test fold. The embargo adds an extra gap
to kill any autocorrelation leakage.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .regressors import get_regressor_names, compute_regressors, regressor_diagnostics

TICK_SIZE = 0.25


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

    mask[fold_start_idx:fold_end_idx] = False

    test_start_ts = ts_ns[fold_start_idx]
    test_end_ts = ts_ns[min(fold_end_idx - 1, n - 1)]

    purge_boundary = test_start_ts - purge_ns
    for i in range(fold_start_idx - 1, -1, -1):
        if ts_ns[i] < purge_boundary:
            break
        mask[i] = False

    embargo_boundary = test_end_ts + embargo_ns
    for i in range(fold_end_idx, n):
        if ts_ns[i] > embargo_boundary:
            break
        mask[i] = False

    return mask


def _ridge_solve(XtX, Xty, alpha, penalty_diag):
    """Ridge solve with a per-coefficient penalty scale (0 for intercept)."""
    p = XtX.shape[0]
    A = XtX + alpha * np.diag(penalty_diag)
    return np.linalg.solve(A, Xty)


class ToxicityModel:
    """
    Predicts signed forward return from microstructure features.

    Training: purged/embargoed K-fold CV with alpha-grid selection.
    Online: RLS with exponential forgetting; feature z-scores are
    exponentially-weighted with the same decay.
    """

    def __init__(self, forgetting_factor: float = 0.999,
                 delta: float = 100.0,
                 target: str = 'fwd_return_1000ms',
                 n_folds: int = 5,
                 purge_ns: int = 10_000_000_000,
                 embargo_ns: int = 5_000_000_000,
                 alpha_grid=(1e0, 1e2, 1e3, 1e4, 1e5, 1e6),
                 toxic_threshold_ticks: float = 0.5):
        self.lam = forgetting_factor
        self.target = target
        self.n_folds = n_folds
        self.purge_ns = purge_ns
        self.embargo_ns = embargo_ns
        self.alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
        self.toxic_threshold_ticks = toxic_threshold_ticks

        self.n_feat = len(get_regressor_names())
        self.active_features: list[str] = []

        self.w = np.zeros(self.n_feat + 1)
        self.P = np.eye(self.n_feat + 1) * delta
        self._delta_init = delta

        self.mean = np.zeros(self.n_feat)
        self.var = np.ones(self.n_feat)
        self.n_seen = 0

        self.calibrator: IsotonicRegression | None = None

        self.cumulative_sq_error = 0.0
        self.n_updates = 0
        self.oof_stats: dict = {}

        self.best_alpha: float = float('nan')
        self._resid_var: float = 1.0

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

        For each candidate alpha:
          1. Split data into K time-ordered folds
          2. Per fold: standardize on TRAIN ONLY, solve ridge with
             unpenalized intercept, predict test
          3. Aggregate OOF residuals, score R^2
        Pick alpha with best OOF R^2, refit on all data, fit isotonic
        calibrator on a meaningful binary label.
        """
        self.active_features = self._resolve_features(df)
        self.n_feat = len(self.active_features)
        p = self.n_feat

        X_all = df[self.active_features].values.astype(np.float64)
        y_all = df[self.target].values.astype(np.float64)
        ts_all = df['ts_ns'].values.astype(np.int64)
        mid_all = df['mid'].values.astype(np.float64) if 'mid' in df.columns else None

        valid_mask = ~np.isnan(y_all)
        X_all = np.nan_to_num(X_all)

        n = len(X_all)
        fold_size = n // self.n_folds

        # penalty diag: unpenalized intercept at position 0
        penalty_diag = np.ones(p + 1)
        penalty_diag[0] = 0.0

        # pre-compute fold splits + per-fold train statistics to reuse
        # across the alpha sweep (XtX and Xty stay fixed per fold)
        fold_cache = []
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

            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())

            if n_train < 100 or n_test < 10:
                print(f"    Fold {k}: skipped (train={n_train}, test={n_test})")
                fold_cache.append(None)
                continue

            # standardize on TRAIN ONLY (avoid mu/sigma leakage across folds)
            X_tr_raw = X_all[train_mask]
            mu_k = X_tr_raw.mean(axis=0)
            sd_k = X_tr_raw.std(axis=0) + 1e-8

            X_tr = np.column_stack([
                np.ones(n_train),
                (X_tr_raw - mu_k) / sd_k,
            ])
            y_tr = y_all[train_mask]

            X_te = np.column_stack([
                np.ones(n_test),
                (X_all[test_mask] - mu_k) / sd_k,
            ])
            y_te = y_all[test_mask]

            XtX = X_tr.T @ X_tr
            Xty = X_tr.T @ y_tr

            fold_cache.append({
                'k': k, 'train_mask': train_mask, 'test_mask': test_mask,
                'n_train': n_train, 'n_test': n_test,
                'XtX': XtX, 'Xty': Xty,
                'X_te': X_te, 'y_te': y_te,
                'mu': mu_k, 'sd': sd_k,
            })

        # sweep alpha
        oof_preds_by_alpha: dict[float, np.ndarray] = {
            float(a): np.full(n, np.nan) for a in self.alpha_grid
        }
        for fc in fold_cache:
            if fc is None:
                continue
            for a in self.alpha_grid:
                w_fold = _ridge_solve(fc['XtX'], fc['Xty'], float(a), penalty_diag)
                y_hat = fc['X_te'] @ w_fold
                oof_preds_by_alpha[float(a)][fc['test_mask']] = y_hat

        # pick alpha by OOF R^2
        alpha_scores = {}
        best_alpha = float(self.alpha_grid[0])
        best_r2 = -np.inf
        for a in self.alpha_grid:
            preds = oof_preds_by_alpha[float(a)]
            m = ~np.isnan(preds) & valid_mask
            if m.sum() < 100:
                continue
            res = y_all[m] - preds[m]
            var_y = float(np.var(y_all[m]))
            r2 = 1.0 - float(np.var(res)) / (var_y + 1e-12)
            alpha_scores[float(a)] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = float(a)

        self.best_alpha = best_alpha
        print("    Alpha grid (OOF R^2): " + "  ".join(
            f"a={a:.0e}:{alpha_scores.get(float(a), float('nan')):+.4f}"
            for a in self.alpha_grid))
        print(f"    Selected alpha = {best_alpha:.0e}  (OOF R^2 = {best_r2:+.4f})")

        oof_preds = oof_preds_by_alpha[best_alpha]

        # per-fold diagnostics at the selected alpha
        fold_r2s = []
        for fc in fold_cache:
            if fc is None:
                continue
            idx = fc['test_mask']
            res = y_all[idx] - oof_preds[idx]
            ss_res = float(np.sum(res ** 2))
            ss_tot = float(np.sum((y_all[idx] - y_all[idx].mean()) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            fold_r2s.append(r2)
            ts_fold = ts_all[idx]
            duration_h = (ts_fold[-1] - ts_fold[0]) / 3.6e12
            print(f"    Fold {fc['k']}: train={fc['n_train']:,}  test={fc['n_test']:,}  "
                  f"R2={r2:+.4f}  RMSE={np.sqrt(np.mean(res**2)):.6f}  "
                  f"span={duration_h:.1f}h")

        # ── isotonic calibration on OOF predictions ───────
        oof_valid = ~np.isnan(oof_preds) & valid_mask
        n_oof = int(oof_valid.sum())

        if n_oof > 100:
            oof_scores = oof_preds[oof_valid]
            oof_true = y_all[oof_valid]

            # "toxic" = adverse move exceeds half-a-tick. fwd_return_* is
            # defined in features.py as (mid_future - mid) in price units
            # (dollars), so the threshold is toxic_threshold_ticks *
            # TICK_SIZE in the same units. Sanity check: if the base
            # rate collapses to 0 or 1 we pick the empirical median of
            # |y| so the isotonic fit has a non-degenerate binary label.
            thresh = self.toxic_threshold_ticks * TICK_SIZE
            oof_binary = (np.abs(oof_true) > thresh).astype(float)
            base_rate = float(oof_binary.mean())
            if base_rate < 0.05 or base_rate > 0.95:
                fallback = float(np.median(np.abs(oof_true)))
                if fallback > 0:
                    thresh = fallback
                    oof_binary = (np.abs(oof_true) > thresh).astype(float)

            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds='clip')
            self.calibrator.fit(np.abs(oof_scores), oof_binary)
            oof_calib = self.calibrator.transform(np.abs(oof_scores))

            print(f"    Isotonic calibration: {n_oof:,} OOF samples, "
                  f"threshold={thresh:.2e} ({self.toxic_threshold_ticks} ticks), "
                  f"base rate={oof_binary.mean():.3f}, "
                  f"mean P(toxic)={oof_calib.mean():.3f}")

            oof_res = oof_true - oof_scores
            oof_r2 = 1.0 - float(np.var(oof_res)) / (float(np.var(oof_true)) + 1e-12)
            oof_rmse = float(np.sqrt(np.mean(oof_res ** 2)))
        else:
            oof_r2 = float('nan')
            oof_rmse = float('nan')
            print(f"    WARNING: only {n_oof} valid OOF samples, skipping calibration")

        # ── final model on ALL data ───────────────────────
        all_valid = valid_mask
        X_tr_raw = X_all[all_valid]
        y_tr = y_all[all_valid]

        # save full-data mean/var (used by online predict / predict_batch)
        mu_all = X_tr_raw.mean(axis=0)
        sd_all = X_tr_raw.std(axis=0) + 1e-8
        self.mean = mu_all
        self.var = sd_all ** 2
        self.n_seen = int(all_valid.sum())

        X_tr = np.column_stack([
            np.ones(self.n_seen),
            (X_tr_raw - mu_all) / sd_all,
        ])

        XtX = X_tr.T @ X_tr
        Xty = X_tr.T @ y_tr
        self.w = _ridge_solve(XtX, Xty, best_alpha, penalty_diag)

        # RLS P0 = sigma^2 * (X'X + alpha*D)^-1  (standard Bayesian form)
        y_hat = X_tr @ self.w
        resid = y_tr - y_hat
        sigma2 = float(np.var(resid))
        self._resid_var = sigma2 if sigma2 > 0 else 1.0
        try:
            A = XtX + best_alpha * np.diag(penalty_diag)
            self.P = np.linalg.inv(A) * self._resid_var
        except np.linalg.LinAlgError:
            # fall back to a diagonal init if the regularized system is
            # still near-singular
            self.P = np.eye(X_tr.shape[1]) * self._delta_init

        self.oof_stats = {
            'oof_r2': float(oof_r2),
            'oof_rmse': float(oof_rmse),
            'fold_r2s': fold_r2s,
            'n_oof': int(n_oof),
            'n_features': self.n_feat,
            'n_folds': self.n_folds,
            'purge_ns': self.purge_ns,
            'embargo_ns': self.embargo_ns,
            'best_alpha': best_alpha,
            'alpha_grid': [float(a) for a in self.alpha_grid],
            'alpha_scores': alpha_scores,
            'resid_var': self._resid_var,
        }

        return self.oof_stats

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
        """
        Exponentially-weighted running mean/var with decay matching
        the RLS forgetting factor self.lam. For lam=1 this reduces to
        plain Welford; for lam<1, old samples are down-weighted in
        lockstep with the weights.
        """
        self.n_seen += 1
        if self.lam >= 1.0:
            d1 = x - self.mean
            self.mean += d1 / self.n_seen
            d2 = x - self.mean
            self.var += (d1 * d2 - self.var) / self.n_seen
            return

        alpha = 1.0 - self.lam
        delta = x - self.mean
        self.mean = self.mean + alpha * delta
        # EWMA-of-squared-deviations (Welford-EWMA hybrid)
        self.var = (1.0 - alpha) * (self.var + alpha * delta * delta)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    @staticmethod
    def _design(x_norm: np.ndarray) -> np.ndarray:
        return np.concatenate([[1.0], x_norm])

    def update(self, x_raw: np.ndarray, y_true: float) -> float:
        """One-sample RLS step. Returns predicted value BEFORE update."""
        self._update_norm(x_raw)
        x = self._design(self._normalize(x_raw))

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
