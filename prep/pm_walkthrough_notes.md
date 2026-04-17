# PM Walkthrough — Technical Prep Notes

Notes for a technical PM walkthrough. **Ground truth** for behavior and defaults is the Python under `adaptive_mm/` and `run_research.py`; **this file** summarizes intent and points to the right modules. Where the **shipped** `model_export/toxicity_model.json` (primary 1s / “slow” model in this repo) differs from the *full* regressor catalog in code, that is called out explicitly.

## 1. What the project is

This is an end-to-end research pipeline for predicting adverse selection in ES futures at the trade level. It ingests raw Databento MBO (message-by-order) data — every add, cancel, modify, and fill on CME — reconstructs the full limit order book from scratch, snapshots the book state at every trade, engineers many microstructure features, and trains ridge regression models at several forward-return horizons to estimate how informed each incoming trade is. The trained model weights, isotonic calibration mappings, Markov regime parameters, and an optimal spread lookup table are exported as JSON for consumption by a C++ market-making execution engine. The intended downstream use: a live MM engine reads the toxicity score in real time and adjusts its spread width and quote skew before the next fill arrives.

## 2. The research question

When a trade prints on ES futures, should a market maker who has a resting quote at that level cancel or widen before the next fill? The specific question: given the full state of the limit order book at the instant a trade occurs, can we predict the direction and magnitude of mid-price movement over the next 50ms to 5 seconds — the window in which a passive market maker would realize adverse selection if the flow was informed? The pipeline produces a continuous toxicity score per trade, not a binary classification. That score drives two decisions in the downstream execution engine: (1) how wide to quote (spread), and (2) which direction to lean (skew).

## 3. Data layer

**Source:** Databento MBO feed (schema `mbo`) for CME Globex. Every individual order event — add, cancel, modify, trade, fill — with its own `order_id`, nanosecond `ts_event`, `ts_recv`, `ts_in_delta` (exchange-to-gateway latency), `flags`, `sequence`, `channel_id`, and `publisher_id`.

**Symbol:** ESU5 (E-mini S&P 500, September 2025 contract), instrument_id 14160 (see `test_instruments.py` / typical runs).

**Date range:** Described in prior runs as full month of August 2025; confirm against your actual `*.mbo.dbn.zst` filenames under `MBO_DATA_DIR` in `adaptive_mm/backtest.py`.

**Why MBO:** The pipeline reconstructs the LOB order-by-order, which enables features that are impossible from MBP-10 snapshots: level-decomposed OFI (add/cancel flow classified by distance from BBO), order fragmentation (average order size at L1), queue depletion rates, fill asymmetry, cancel-to-trade ratios at L1 vs deep levels, book resilience, and individual order lifecycle tracking.

**Preprocessing:** Raw MBO messages are filtered to a single instrument, then fed through the Python LOB reconstructor (`lob.py`) which processes each message via `process_fast()` using pre-converted integer action/side codes (no string comparisons in the hot loop). At every trade event, the LOB is snapshotted for downstream features.

**Options data (partial):** `options.py` implements SPX options feature computation from Databento OPRA mbp-1 data (ATM IV, term slope, skew, GEX proxy, etc.). The spread optimizer accepts optional `gex_values`; `run_research` / `run_spread_optimization` in `backtest.py` call it **without** GEX, so the GEX dimension collapses to a single neutral bin. Options are not in the toxicity feature matrix today.

## 4. Label definition

**As implemented in `features.py` (forward mid returns, ≈415–422; microprice variants ≈424–434):**

```python
for hns in horizons_ns:
    hms = hns // 1_000_000
    fwd_idx = np.searchsorted(ts, ts + hns)
    valid = fwd_idx < n
    fwd_idx_c = np.minimum(fwd_idx, n - 1)
    fwd_ret = mid[fwd_idx_c] - mid
    fwd_ret[~valid] = np.nan
    feat[f'fwd_return_{hms}ms'] = fwd_ret
```

**Definition:** For each trade at time *t*, the label is `mid[t + H] - mid[t]` — raw change in the bid-ask midpoint over horizon H, in price units ($0.25 per tick for ES). **Signed continuous return**, not binary. **Event-time indexing:** `searchsorted` picks the first row whose `ts` is **strictly after** `ts_trade + H` (see `horizons_ns` default in `compute_features`).

**Horizons in code:** 10ms, 50ms, 100ms, 200ms, 500ms, 1000ms, 5000ms, 10000ms for mid (and matching `fwd_mp_*` when microprice exists).

**Multi-horizon training in `backtest.run_research`:** Four `ToxicityModel` instances with **per-model** purge/embargo (not the class default alone):

| Label | Target | Purge | Embargo |
|-------|--------|-------|---------|
| fast | `fwd_return_50ms` | 2s | 1s |
| medium | `fwd_return_500ms` | 5s | 2.5s |
| slow | `fwd_return_1000ms` | 10s | 5s |
| macro | `fwd_return_5000ms` | 20s | 10s |

The **primary** model for TCA and the JSON named `toxicity_model.json` in this snapshot is the **slow** (1000ms) target unless you rename exports.

**Isotonic calibration (`toxicity.py`, ≈206–213):** Binary toxic label uses `toxic_threshold = 0.0` and `oof_binary = (np.abs(oof_true) > toxic_threshold)` — i.e. any strictly positive absolute forward return counts as toxic for calibration. `IsotonicRegression` fits on `|oof_scores|` vs that binary outcome.

## 5. Feature set — code vs shipped export

### 5a. Canonical regressors (`regressors.get_regressor_names()`)

`adaptive_mm/regressors.py` defines an **ordered wish-list** of regressor columns (OFI at up to seven L1 windows, L2/deep OFI, imbalances, VPIN, quote flicker, BBO move rates, widen rates, TWAS, book shape, interactions including `flicker_x_vol` and `widen_x_ofi`, etc.). **`ToxicityModel._resolve_features()`** only keeps names **that exist as columns** in the training frame. So the trained width depends on which raw columns `compute_features` produced for your dataset.

### 5b. Shipped `model_export/toxicity_model.json` (42 features)

The JSON in this repo lists **exactly** these columns (this is the active set for that export, not the full theoretical catalog):

`z_ofi_L1_50ms`, `z_ofi_L1_500ms`, `z_ofi_L1_5000ms`, `z_ofi_L2_500ms`, `z_ofi_deep_500ms`, `z_ofi_agg_500ms`, `z_imb_L1`, `z_imb_L2`, `z_imb_L3`, `z_L1_L2_disagree`, `z_microprice_delta`, `z_vpin_500ms`, `z_vpin_5000ms`, `z_kyle_lambda`, `z_trade_accel`, `z_trade_rate`, `z_cancel_ratio_500ms`, `z_canc_L1_share_500ms`, `spread_ticks`, `z_spread`, `z_slope_bid`, `z_slope_ask`, `z_curv_bid`, `z_curv_ask`, `z_hhi_bid`, `z_hhi_ask`, `z_avg_ord_sz_bid`, `z_avg_ord_sz_ask`, `z_net_depletion_50ms`, `z_net_depletion_500ms`, `z_modify_rate_500ms`, `z_realized_vol`, `z_signed_vol_500ms`, `z_signed_vol_5000ms`, `z_resilience_bid_500ms`, `z_resilience_ask_500ms`, `z_fill_asym_500ms`, `z_latency`, `ofi_L1_x_vol`, `ofi_L1_x_spread`, `imb_L2_x_vol`, `slope_x_accel`.

**Not in this export** (but may appear in other runs when columns exist): extra `z_ofi_L1_*` windows, `z_ofi_L2_5000ms`, quote flicker / BBO move / widen / TWAS regressors, and interactions `flicker_x_vol`, `widen_x_ofi`.

### 5c. Top coefficients by |weight| (from this JSON, slow model)

1. `z_imb_L2` ≈ 0.141  
2. `z_ofi_L1_5000ms` ≈ 0.040  
3. `z_curv_bid` ≈ 0.028  
4. `z_spread` ≈ 0.027  
5. `z_imb_L1` ≈ 0.023  

`z_slope_ask` is also material (≈0.021) but slightly below `z_imb_L1` in this export. **`ofi_L1_x_vol`** |w| ≈ 0.011.

## 6. Model architecture

**Model:** Ridge regression (L2-regularized OLS) with `alpha = 1.0`. Implemented in `toxicity.py` via normal equations: `XtX = X'X + alpha*I`, `w = solve(XtX, X'y)` — no `sklearn` `Ridge` class (note: `ElasticNet` is imported in `toxicity.py` but **not** used in the current training path).

**Multi-horizon:** Four separate models in `run_research` (see table above).

**Validation (`toxicity.fit_purged_cv`, ≈113–196):** 5-fold time-ordered CV; per fold, `_purged_embargo_split` builds the train mask (purge + embargo). OOF predictions on held-out folds, then isotonic on OOF.

**Train / holdout:** First **60%** of the concatenated dataset for training (`n_train = int(len(full_df) * 0.6)` in `backtest.py`); last **40%** for TCA — models never train on the holdout.

**RLS:** `ToxicityModel` holds online RLS state (`forgetting_factor` default **0.999**); `update_rls` implements the adaptation step. Architectural; live production validation is separate.

## 7. Calibration layer

Isotonic maps `|raw_ridge_score|` → P(toxic) with the permissive binary label above. **`SpreadRegime.compute_spread`** uses **raw** toxicity magnitude via `h = 1 + toxicity_scale * |score|` (default `toxicity_scale=4.0`). Spread optimizer bins raw scores for the lookup table. Calibrated probabilities are exported (`isotonic_calibration.json`) for dashboards or C++ parity, but **spread/skew in `engine.py` use raw score scaling** (see §11).

## 8. Evaluation and results

**From `toxicity_model.json` `oof_stats` (slow / 1000ms target in this export):**

| Metric | Value |
|--------|-------|
| OOF R² | ≈ 0.0589 |
| OOF RMSE | ≈ 0.4100 |
| n OOF | 5,396,595 |
| n features | 42 |

**Per-fold R²** in JSON: ≈ 0.0146, 0.0728, 0.0772, 0.1082, 0.1104.

**TCA on holdout (`tca.py`):** Markouts, adverse selection by regime, queue dynamics, latency, toxicity decile separation — see `run_research` phase 3 calls.

**Diagnostics:** `regressor_diagnostics` in `regressors.py` (univ R², pairwise corr, VIF).

## 9. Honest limitations

1. Single symbol / period unless you expand data.  
2. No live fill simulator wired to MBO models (research + TCA on historical holdout).  
3. Linear ridge + four hand-listed interactions in the **export**; broader interactions exist in code when columns exist.  
4. Isotonic threshold at zero is permissive for “toxic” binary calibration.  
5. Options / GEX not fed into toxicity or spread optimizer in the default `run_research` call.  
6. **LOB in Python** — throughput is research-grade; production would move hot path to C++/Rust.  
7. **Depth / divide:** `features.py` uses guarded divides in several places; edge cases can still emit numpy warnings depending on data.

## 10. Sensible next steps

1. Pass real `gex_values` into `run_spread_optimization` once OPRA path is wired.  
2. Load exported ridge weights / optional isotonic in `cpp_databento_ml_optimized.cpp` and align feature vector with the JSON you ship.  
3. Multi-symbol / multi-month validation.

## 10b. Sign convention

Throughout the pipeline, `trade_side` follows **features.py / tca.py**:  
- `trade_side == 0` → **buy aggressor** (positive expected markout if flow is toxic)  
- `trade_side == 1` → **sell aggressor**  

`spread_optimizer.compute_markout_payoffs` uses the same mapping so the exported **skew** in the spread lookup table is aligned with the TCA narrative (positive signed markout → widen ask and skew quotes away from the informed side).

## 11. Math reference (matches `engine.py`)

**SpreadRegime** (`adaptive_mm/engine.py`):  
`vol_ratio = clip(vol_fast / vol_slow, 0.5, 3.0)`  
`g = 1 + inventory_penalty * |inventory| / 10` (default penalty 0.5)  
`h = 1 + toxicity_scale * |toxicity_score|` (default scale 4.0)  
`f = 1 - 0.3 * clip(gex_local, -1, 1)` → **1.0** when `gex_local` default 0  

**Skew:** `compute_skew` uses `TICK_SIZE` from `lob.py` for the toxicity term: `tox_skew = -sign(tox) * min(|tox| / TICK_SIZE, 1.5)` (ES tick 0.25).

**Optimal spread table:** `spread_optimizer.compute_optimal_spread_table` — grid over half-spreads vs expected markout; see module docstrings.

## 12. Markov regime model

`markov.py`: **3** realized-vol regimes × **9** (spread × imbalance) states × **11** discrete price-change outcomes (−5…+5 ticks). Matches the “3 × 9 × 11” description.

## 13. Drift from older resume-style bullets

| Aspect | Older story | Current code | Status |
|--------|-------------|--------------|--------|
| Base learner | Sometimes boosted trees | Ridge in `toxicity.py` | Changed |
| Features | L1-only snapshot story | MBO + many regressors; **export** is 42-column subset | Expanded / subset |
| Data | MBP-style snapshots | MBO files via `DBNStore` | Changed |
| Horizons | Various | 50ms–5000ms trained models + longer labels in features | Expanded |
| CV | Purged OOF | 5-fold purged + embargo, horizon-specific windows | Same idea, explicit |
| GEX | — | Optional in optimizer; **not** passed from `run_research` | Partial |
