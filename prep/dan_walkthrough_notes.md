# Dan Deering Walkthrough — Technical Prep Notes

## 1. What the project is

This is an end-to-end research pipeline for predicting adverse selection in ES futures at the trade level. It ingests raw Databento MBO (message-by-order) data — every add, cancel, modify, and fill on CME — reconstructs the full limit order book from scratch, snapshots the book state at every trade, engineers 46+ microstructure features grounded in the academic literature, and trains ridge regression models at four forward-return horizons to estimate how informed each incoming trade is. The trained model weights, isotonic calibration mappings, Markov regime parameters, and an optimal spread lookup table are all exported as JSON for consumption by a C++ market-making execution engine. The intended downstream use: a live MM engine reads the toxicity score in real time and adjusts its spread width and quote skew before the next fill arrives.

## 2. The research question

When a trade prints on ES futures, should a market maker who has a resting quote at that level cancel or widen before the next fill? The specific question: given the full state of the limit order book at the instant a trade occurs, can we predict the direction and magnitude of mid-price movement over the next 50ms to 5 seconds — the window in which a passive market maker would realize adverse selection if the flow was informed? The pipeline produces a continuous toxicity score per trade, not a binary classification. That score drives two decisions in the downstream execution engine: (1) how wide to quote (spread), and (2) which direction to lean (skew).

## 3. Data layer

**Source:** Databento MBO feed (schema `mbo`) for CME Globex. Every individual order event — add, cancel, modify, trade, fill — with its own `order_id`, nanosecond `ts_event`, `ts_recv`, `ts_in_delta` (exchange-to-gateway latency), `flags`, `sequence`, `channel_id`, and `publisher_id`.

**Symbol:** ESU5 (E-mini S&P 500, September 2025 contract), instrument_id 14160.

**Date range:** Full month of August 2025 (~22 trading days). Most recent confirmed run: Aug 28 (323,810 trades, 8.99M messages) and Aug 29 (368,883 trades, 10.7M messages). Aggregate training set from exported model: 5,396,595 OOF samples.

**Why MBO:** The pipeline reconstructs the LOB order-by-order, which enables features that are impossible from MBP-10 snapshots: level-decomposed OFI (add/cancel flow classified by distance from BBO), order fragmentation (average order size at L1 = institutional vs HFT signal), queue depletion rates, fill asymmetry, cancel-to-trade ratios at L1 vs deep levels, book resilience (L1 refill speed after fills), and individual order lifecycle tracking. MBP-10 snapshots aggregate these into a single size number per level, destroying the signal.

**Preprocessing:** Raw MBO messages are filtered to a single instrument, then fed through the Python LOB reconstructor (`lob.py`) which processes each message via `process_fast()` using pre-converted integer action/side codes (no string comparisons in the hot loop). At every trade event, the LOB is snapshotted: L1-L5 sizes, L1-L3 order counts, microprice, spread, imbalance, and cumulative event counters for adds/cancels/modifies/fills classified by level offset.

**Options data (partial):** `options.py` implements SPX options feature computation from Databento OPRA mbp-1 data: ATM IV, IV term structure slope, put-call skew, net GEX proxy, vanna pressure, 0DTE concentration. These features are computed at 1-minute intervals and forward-filled to ES trade timestamps. The spread optimizer has a GEX regime dimension. However, the options data path is not wired into the main `run_research` pipeline — it exists as standalone infrastructure ready to be integrated.

## 4. Label definition

**As implemented in `features.py` lines 414-422:**

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

**Definition:** For each trade at time *t*, the label is `mid[t + H] - mid[t]` — the raw change in the bid-ask midpoint over horizon H, in price units ($0.25 per tick for ES). This is a **signed continuous return**, not binary. It is **event-time**: the lookup finds the first trade whose timestamp exceeds `t + H` via `searchsorted`, so the horizon is measured in wall-clock nanoseconds from the trade timestamp but the observation is at the next trade event after that time.

**Horizons computed:** 10ms, 50ms, 100ms, 200ms, 500ms, 1000ms, 5000ms, 10000ms. Both mid-based and microprice-based forward returns are computed (the microprice versions use `microprice[t+H] - microprice[t]`).

**Four models are trained at different horizons:**

| Label | Target | Purge | Embargo | Adversary Type |
|-------|--------|-------|---------|----------------|
| fast | `fwd_return_50ms` | 2s | 1s | Co-located HFT |
| medium | `fwd_return_500ms` | 5s | 2.5s | Fast systematic/algo |
| slow | `fwd_return_1000ms` | 10s | 5s | Institutional sweep (primary) |
| macro | `fwd_return_5000ms` | 20s | 10s | Fundamental/news-driven |

**Why this label maps to the research question:** The forward mid return at a given horizon is exactly the adverse selection cost a passive market maker would realize on that fill. If you sold at the ask and mid moves up by X within 1 second, you lost X in mark-to-market. The signed return captures both direction and magnitude of information content. The multi-horizon decomposition separates different adversary speeds.

**Isotonic calibration for binary toxicity (`toxicity.py` lines 206-214):**

```python
toxic_threshold = 0.0
oof_binary = (np.abs(oof_true) > toxic_threshold).astype(float)
self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
self.calibrator.fit(np.abs(oof_scores), oof_binary)
```

The isotonic calibrator maps `|raw_score|` to P(toxic), where "toxic" is defined as any non-zero absolute forward return. This is a very permissive threshold — effectively P(any price movement). The binary label used for calibration is independent of the continuous regression target.

## 5. Feature set

**42 features in the active model** (from exported `toxicity_model.json`), grouped by category. All continuous features are converted to EMA z-scores with halflife=500 trades (fast) or 5000 trades (slow) before entering the model.

### Order Flow Imbalance (Cont, Kukanov, Stoikov 2014)
- `z_ofi_L1_{10,50,100,200,500,1000,5000}ms` — Level-1 OFI = (add_bid_L1 - cancel_bid_L1) - (add_ask_L1 - cancel_ask_L1) over rolling time window, z-scored. 7 timescales from ultra-fast to session-level.
- `z_ofi_L2_{500,5000}ms` — Level-2 OFI at two windows.
- `z_ofi_deep_{500,5000}ms` — Deep-book OFI (levels 3+) at two windows.

### Per-Level Imbalance (Cao, Chen, Griffin 2005)
- `z_imb_L1`, `z_imb_L2`, `z_imb_L3` — (bid_sz - ask_sz) / (bid_sz + ask_sz) at each level, z-scored.
- **`z_imb_L2` is the single strongest predictor by coefficient magnitude (|w| = 0.141).** Economic interpretation: institutions rest orders behind L1 to hide intent. When L2 imbalance disagrees with L1, the L2 signal tends to be right.
- `z_L1_L2_disagree` — sign(L1_imb) * sign(L2_imb), negated and z-scored. Fires when L1 and L2 point in opposite directions.

### Microprice (Stoikov 2018)
- `z_microprice_delta` — Change in microprice = (ask_px * bid_sz + bid_px * ask_sz) / (bid_sz + ask_sz). Captures price drift that mid misses.

### VPIN (Easley, Lopez de Prado, O'Hara 2012)
- `z_vpin_{500ms,5000ms}` — |buy_vol - sell_vol| / total_vol over window. Measures one-sidedness of flow.

### Kyle's Lambda (Price Impact)
- `z_kyle_lambda` — EMA(mid_change * OFI) / EMA_var(OFI). Rolling estimate of the price impact coefficient. Z-scored with slow halflife.

### Trade Arrival
- `z_trade_accel` — log(trade_rate_50ms) - log(trade_rate_5000ms). Sudden acceleration in trade arrival vs baseline.
- `z_trade_rate` — trades per second over 500ms window.

### Cancel Intensity
- `z_cancel_ratio_500ms` — cancels / trades in 500ms window. High cancel rates signal HFT activity or fleeing market makers.
- `z_canc_L1_share_500ms` — Fraction of cancels at L1 vs deep. Concentrated L1 cancels = best-quote pulling.

### Spread State
- `spread_ticks` — Raw spread in ticks (not z-scored). Enters model directly as a level feature.
- `z_spread` — Z-scored spread (slow halflife). Wide spread itself is an adverse selection signal — MMs already widened defensively.

### Quote Dynamics (Hasbrouck & Saar 2009)
- `z_flicker_{50,200,1000}ms` — Count of BBO changes in window. High flicker = HFTs repricing aggressively.
- `z_bid_move_{200,1000}ms`, `z_ask_move_{200,1000}ms` — Net BBO move rate: (up - down) / total_moves. Directional quote pressure.
- `z_widen_rate_{200,1000}ms` — Fraction of spread changes that were widenings. MMs fleeing signal.
- `z_twas_{500,5000}ms` — Time-weighted average spread. More robust than instantaneous spread.

### Book Shape (Cartea, Jaimungal, Penalva 2015)
- `z_slope_bid`, `z_slope_ask` — Linear regression of log(size) on level offset k=1..5. Negative slope = thin behind L1 = vulnerable to sweeps. **`z_slope_ask` |w| = 0.021, top 5.**
- `z_curv_bid`, `z_curv_ask` — Second derivative of log-depth profile (convex vs concave). **`z_curv_bid` |w| = 0.028, top 3.**

### Book Concentration (O'Hara 1995)
- `z_hhi_bid`, `z_hhi_ask` — Herfindahl index of depth across levels. High HHI = one level dominates = possible wall.

### Order Fragmentation (MBO-specific)
- `z_avg_ord_sz_bid`, `z_avg_ord_sz_ask` — Average order size at L1 = total_size / order_count. Large = institutional, small = retail/HFT.

### Queue Depletion
- `z_net_depletion_{50,500}ms` — (bid_L1_start - bid_L1_end) - (ask_L1_start - ask_L1_end) over window. Asymmetric depletion precedes price moves.

### Modify Rate
- `z_modify_rate_500ms` — modifies / trades. High modify rate = HFT fingerprint.

### Realized Volatility
- `z_realized_vol` — sqrt(sum(mid_diff^2) / n) over 500ms, z-scored.

### Signed Volume
- `z_signed_vol_{500,5000}ms` — (buy_vol - sell_vol) over window. Directional flow pressure.

### Book Resilience (Bouchaud, Farmer, Lillo 2009)
- `z_resilience_{bid,ask}_500ms` — Total L1 size churn per trade in window. Fast replenishment = confident MMs = lower toxicity.

### Fill Asymmetry
- `z_fill_asym_500ms` — (fill_bid_n - fill_ask_n) / fill_total. Iceberg proxy.

### Exchange Latency
- `z_latency` — Mean ts_in_delta (exchange-to-gateway latency) over 500ms, z-scored with slow halflife. Latency spikes correlate with quote stuffing.

### Interaction Terms (6 features, flagged in code as "most informative")
- **`ofi_L1_x_vol`** — OFI_L1_500ms * realized_vol. One-sided flow during high vol = peak adverse selection. **|w| = 0.011.**
- `ofi_L1_x_spread` — OFI_L1 * spread_ticks. OFI when spread is wide.
- `imb_L2_x_vol` — L2 imbalance * realized_vol. Institutional hiding during vol.
- `slope_x_accel` — Avg book slope * trade acceleration. Thin book + trade burst = danger.
- `flicker_x_vol` — Quote flicker * vol. HFTs repricing in high vol.
- `widen_x_ofi` — Spread widen rate * OFI. MMs fleeing informed flow.

### Top 5 by absolute coefficient weight:
1. **z_imb_L2** = 0.141 — L2 queue imbalance (institutional hiding)
2. **z_ofi_L1_5000ms** = 0.040 — Sustained L1 order flow over 5s
3. **z_curv_bid** = 0.028 — Bid-side book curvature
4. **z_spread** = 0.027 — Spread state (defensive widening signal)
5. **z_imb_L1** = 0.023 — L1 queue imbalance

## 6. Model architecture

**Model:** Ridge regression (L2-regularized OLS) with alpha=1.0. The code implements this directly via the normal equations: `w = (X'X + alpha*I)^{-1} X'y`. No sklearn Ridge — it's hand-rolled with `np.linalg.solve`.

**Why Ridge over LightGBM:** The old project used LightGBM. Ridge was chosen because: (a) features are heavily collinear by construction (OFI at 7 timescales, imbalance at 3 levels), and ridge handles this by shrinking correlated coefficients rather than arbitrarily selecting one; (b) coefficients remain interpretable — you can explain why L2 imbalance matters by looking at its weight; (c) inference is a single dot product, fast enough for microsecond-level online scoring; (d) the model can be updated online via RLS with a forgetting factor, adapting to regime shifts without full retraining.

**Multi-horizon architecture:** Four separate ridge models are trained, each targeting a different forward-return horizon (50ms, 500ms, 1000ms, 5000ms). These detect different types of informed flow — co-located HFT at 50ms, institutional sweeps at 1s, news-driven at 5s. The purge and embargo windows scale with the horizon.

**Validation scheme (exact code in `toxicity.py` lines 113-196):**

- **5-fold time-ordered CV** — folds are contiguous time blocks, not shuffled. Fold k = rows `[k*fold_size, (k+1)*fold_size)`.
- **Purge window:** For the primary 1s model, 10 seconds. Any training sample within 10s before the test fold boundary is excluded, because its forward-return label overlaps with test-fold timestamps.
- **Embargo window:** 5 seconds after the test fold ends. Extra buffer to kill autocorrelation leakage.
- **Implementation:** `_purged_embargo_split()` walks backward/forward from fold boundaries setting a boolean mask to False, producing a clean train set per fold.

**Loss function:** OLS (minimizes sum of squared errors on the forward return). The ridge penalty is `alpha * ||w||^2`.

**Train/test split:** First 60% of the concatenated multi-day dataset is used for model training (with purged CV within it). Last 40% is held out for TCA and never seen by any model.

**Online phase:** After research training, the model weights initialize a Recursive Least Squares (RLS) estimator with forgetting factor lambda=0.999, enabling live coefficient adaptation as market regime shifts. The RLS update is a single-sample O(d^2) step.

## 7. Calibration layer

**What it does:** Isotonic regression maps `|raw_ridge_score|` to a calibrated probability P(toxic), where "toxic" is defined as any non-zero absolute forward return (`threshold = 0.0`). This is fit on the out-of-fold predictions from the purged CV, so it never sees in-sample scores.

**Input:** Absolute value of the OOF ridge predictions (continuous, unbounded).
**Output:** Monotonically increasing probability in [0, 1] (clipped via `out_of_bounds='clip'`).

**Downstream consumption:** The calibrated probability is available via `model.calibrate(raw_scores)`. The spread optimizer (`spread_optimizer.py`) uses the raw toxicity scores (not the calibrated probabilities) to bin trades into deciles for the optimal spread lookup table. The `SpreadRegime` class in `engine.py` consumes the raw toxicity score directly in the spread formula: `h(tox) = 1 + toxicity_scale * |score|`. The isotonic calibration is exported as JSON (`isotonic_calibration.json`) for C++ to implement the same mapping if it wants a probability rather than a raw score.

**Honest status:** The calibration exists and is exported, but the raw score — not the calibrated probability — is what actually drives spread/skew decisions in both the optimizer and the engine formulas. The calibration is useful for interpretability (reporting "30% chance of adverse selection" to a dashboard) but is not load-bearing in the execution logic as currently wired.

## 8. Evaluation and results

**Primary metric:** Out-of-fold R^2 from purged/embargoed 5-fold CV.

**Most recent results (from exported `toxicity_model.json`, target `fwd_return_1000ms`):**

| Metric | Value |
|--------|-------|
| OOF R^2 | 0.0589 |
| OOF RMSE | 0.4100 |
| N (OOF samples) | 5,396,595 |
| N features | 42 |

**Per-fold R^2 breakdown:**

| Fold | R^2 |
|------|-----|
| 0 | 0.0146 |
| 1 | 0.0728 |
| 2 | 0.0772 |
| 3 | 0.1082 |
| 4 | 0.1104 |

**What these numbers mean:** An OOF R^2 of ~6% on 1-second forward returns in a liquid futures market is genuinely informative. For context: a random predictor gives R^2 = 0, and most published microstructure models report R^2 of 2-8% at similar horizons. The fold variation (1.5% to 11%) reflects regime dependence — fold 0 likely covers a low-vol period where mid barely moves and the signal-to-noise ratio is poor.

**TCA metrics computed on the 40% holdout (never seen by models):**
- Signed markout analysis at 50ms, 100ms, 200ms, 500ms, 1000ms, 5000ms
- Adverse selection decomposition by: spread state (tight/normal/wide), vol regime (low/mid/high), L1 imbalance bucket, UTC hour
- Queue dynamics: cancel-to-trade ratios, modify rates, fill asymmetry, depletion before large moves
- Exchange latency analysis and latency-vs-markout correlation
- **Toxicity decile separation:** trades binned by predicted toxicity score into 10 buckets; if the model works, high-decile trades should show worse realized markouts. This is the key validation — it tests whether the model's score actually separates adverse selection in data it never trained on.

**Additional diagnostics (from `regressor_diagnostics`):**
- Univariate R^2 and correlation with target for each feature
- Pairwise correlation matrix (multicollinearity check, flags pairs > 0.7)
- Variance Inflation Factors (flags VIF > 10 as drop candidates)

## 9. Honest limitations

1. **Single symbol.** All results are on ESU5 only. Generalization to other futures (NQ, CL, ZN) is untested. ES is the most liquid equity index future — results may not transfer to thinner products.

2. **Single month.** August 2025. No major macro events in sample (no FOMC, no flash crash). The fold-0 R^2 of 1.5% suggests the model struggles in low-vol regimes. Performance during high-event-risk periods is unknown.

3. **No live fill simulation.** The spread optimizer computes theoretical optimal half-spreads, but there is no fill simulator that accounts for queue position, partial fills, or adverse selection on your own resting orders. The C++ engine has a queue position simulator from the old MBP-10 version, but it hasn't been integrated with the new MBO-trained models.

4. **Linear model only.** Ridge regression cannot capture nonlinear interactions beyond the 6 hand-crafted interaction terms. A tree-based model or neural net might extract more signal, at the cost of interpretability and online-update simplicity.

5. **Isotonic calibration threshold.** The binary label for calibration uses threshold=0.0 (any nonzero return is "toxic"), which is extremely permissive. A half-tick threshold would be more operationally relevant. This means the calibrated P(toxic) will be high even for benign trades.

6. **Options features not integrated.** The GEX/IV/vanna/skew infrastructure exists in `options.py` but is not wired into `run_research`. The spread optimizer has a GEX dimension but currently runs with `gex_values=None`, collapsing it to a single bucket.

7. **Division-by-zero bug.** `features.py:174` has a `RuntimeWarning: invalid value encountered in divide` when total depth at L2-L5 is zero. It's handled by a `np.where` guard now but the warning still fires on some edge cases.

8. **LOB reconstruction is in Python.** Processing ~10M messages per day takes ~80-85 seconds. For a production system this would need to be in C++ or Rust. The current implementation is research-grade.

## 10. What I'd do next

1. **Wire in the options features.** The infrastructure is built: `options.py` computes ATM IV, term slope, put-call skew, GEX, and vanna from OPRA data, and the spread optimizer already has a GEX regime dimension. The concrete work: add an OPRA data loading step to `run_research`, call `compute_options_features`, pass the GEX array to `run_spread_optimization`. This would let the model widen during negative-gamma regimes (0DTE expiry, dealer short gamma) and tighten during positive-gamma environments.

2. **Connect the new MBO-trained models to the C++ engine.** The model weights and calibration are already exported as JSON. The C++ engine (`cpp_databento_ml_optimized.cpp`) already has a toxicity scoring path and spread formula — it just needs to load the new JSON weights and switch from MBP-10 snapshot features to the level-decomposed features. The LOB reconstruction would also move to C++ for production latency.

3. **Multi-symbol expansion.** Train on NQ, CL, and ZN to test whether the feature set generalizes or if coefficients are ES-specific. The pipeline is symbol-agnostic — only the `instrument_id` filter and `TICK_SIZE` constant need to change. Cross-product features (ES-NQ spread velocity, ES-CL correlation breaks) would be the next feature category to add.

## 11. The math, in plain language

### Mid-price
`mid = (best_bid_ticks + best_ask_ticks) * TICK_SIZE / 2`
For ES, TICK_SIZE = $0.25. If best bid is 5600.00 and best ask is 5600.25, mid = 5600.125.

### Microprice (Stoikov 2018)
`microprice = (ask_price * bid_size_L1 + bid_price * ask_size_L1) / (bid_size_L1 + ask_size_L1)`
Weights mid toward the thinner side. If 200 contracts on the bid and 50 on the ask, microprice shifts toward the ask — the "true" price is closer to the side with less depth.

### Level-decomposed OFI (Cont et al. 2014)
`OFI_L1(t, w) = [cum_add_bid_L1(t) - cum_add_bid_L1(t-w)] - [cum_canc_bid_L1(t) - cum_canc_bid_L1(t-w)] - [cum_add_ask_L1(t) - cum_add_ask_L1(t-w)] + [cum_canc_ask_L1(t) - cum_canc_ask_L1(t-w)]`

Positive OFI = net add pressure on bids / net cancel pressure on asks = buy pressure. The level decomposition (L1/L2/deep) isolates where the pressure is coming from. The key insight from Cont's original paper: L1 OFI is the dominant predictor; aggregate OFI dilutes the signal with noise from deep levels.

Rolling windows are computed via `np.searchsorted(ts, ts - window_ns)` for O(n log n) vectorized lookups.

### EMA Z-score normalization
For each raw feature x:
```
mu_t = EMA(x, halflife=500)
var_t = EMA_var(x, halflife=500)
z_t = (x_t - mu_t) / sqrt(var_t)
```
This adapts the feature scale to local market conditions. A cancel rate of 10:1 means something very different during a quiet afternoon vs during a news spike.

### Label formation
`y_t = mid[t + H] - mid[t]`
where the future timestamp is found via `np.searchsorted(ts, ts + H_ns)`. This is event-time: if no trade occurs in the next H nanoseconds, the observation is NaN. The label is signed and continuous — not binary.

### Ridge regression
Given design matrix X (n samples x (1 + 42 features), with intercept column) and target y:
```
w = (X'X + alpha * I)^{-1} * X'y
```
with alpha = 1.0. This is equivalent to minimizing `||Xw - y||^2 + alpha * ||w||^2`. The L2 penalty shrinks correlated coefficients toward each other rather than picking one arbitrarily (unlike L1/Lasso). Solved directly via `np.linalg.solve(XtX, X'y)` — no iterative optimizer.

### Isotonic calibration
After purged CV produces OOF predictions `s_i` and true labels `y_i`:
1. Define binary: `toxic_i = 1 if |y_i| > 0 else 0`
2. Fit isotonic regression: `P(toxic) = f(|s_i|)` where f is monotonically non-decreasing
3. At inference: `P(toxic) = f(|raw_score|)`

Isotonic regression fits a piecewise-constant non-decreasing function by solving a pool-adjacent-violators problem. The result is a step function mapping raw score magnitude to calibrated probability.

### Spread formula (engine.py)
```
S(t) = S_base * vol_ratio * g(inventory) * h(toxicity) * f(GEX)
where:
  vol_ratio = clip(vol_fast / vol_slow, 0.5, 3.0)
  g(inv) = 1 + 0.5 * |inventory| / 10
  h(tox) = 1 + 4.0 * |toxicity_score|
  f(GEX) = 1 - 0.3 * clip(GEX_local, -1, 1)  [= 1.0 without options data]
```
The skew shifts quotes away from the informed side: `skew = -sign(inv) * min(|inv|/5, 2) - sign(tox) * min(|tox|/0.25, 1.5)`.

### Optimal spread table (spread_optimizer.py)
For each bin (vol_regime, tox_decile, gex_regime):
```
optimal_hs = argmax_{hs in [0.5, 1.0, ..., 5.0]} E[hs - |markout|]
```
where markout is the signed 1-second markout over all trades in that bin. A market maker quoting at half-spread `hs` captures `hs` ticks per fill but pays `|markout|` in adverse selection. The optimizer finds the half-spread that maximizes expected net PnL per fill in each regime/toxicity bucket.

### RLS online update
After training, the model adapts online via recursive least squares with forgetting factor lambda=0.999:
```
P_{t} = (P_{t-1} - gain * x * P_{t-1}) / lambda
w_t = w_{t-1} + gain * error
gain = P_{t-1} * x / (lambda + x' * P_{t-1} * x)
```
This is a single-sample O(d^2) update that exponentially downweights old observations, allowing the model to track non-stationary microstructure.

## 12. Drift from old resume bullet

| Aspect | Old bullet | Current code | Status |
|--------|-----------|--------------|--------|
| Model | LightGBM | Ridge regression | **Changed** — moved to ridge for interpretability, online adaptation, and to handle collinear features cleanly |
| Features | L1 quote dynamics only (spread state, flicker, OFI/imbalance EWMs, microprice momentum) | 42 features across 22 categories including level-decomposed OFI, per-level imbalance, book shape/curvature/HHI, order fragmentation, queue depletion, book resilience, fill asymmetry, exchange latency, 6 interaction terms | **Expanded significantly** — moved from L1-only to full L1-L5 MBO features |
| Data | Databento historical (MBP-10 implied) | Databento MBO (order-level, schema=mbo) | **Changed** — moved from snapshot data to individual order events |
| Horizons | Multi-horizon 50-1000ms | Multi-horizon 50ms-5000ms with separate models per horizon | **Expanded** — four distinct models targeting different adversary speeds |
| CV | Purged + embargoed OOF | Purged + embargoed 5-fold time-ordered CV | **Still true**, implementation preserved |
| Calibration | Isotonic, p_effective for gating | Isotonic calibration on OOF, exported but raw score (not calibrated prob) drives spread/skew | **Still true** but calibrated probability is not the primary decision input |
| TCA | Markout with 95% CIs, macro blackouts | Markout + adverse selection by regime + queue dynamics + latency analysis + decile separation | **Expanded** — deeper TCA but 95% CIs and macro blackouts from old version are not in current code |
| Execution | Implied gating (p_effective) | Spread formula S(t) = S_base * vol * g(inv) * h(tox) * f(GEX), plus optimal spread lookup table by (regime, tox_decile, gex) | **Evolved** — from binary gate to continuous spread adjustment |
| Regime model | Not mentioned | Markov chain: 3 vol regimes x 9 (spread, imbalance) states x 11 price-change outcomes | **New** |
| Options | Not mentioned | Infrastructure built (ATM IV, GEX, vanna, skew from OPRA) but not integrated into main pipeline | **New but partial** |
| C++ engine | Not mentioned explicitly | MBP-10 sim loop with cache-aligned structs, lock-free atomics, no heap allocation | **Exists** as prior work, not yet connected to new MBO models |
| Macro blackouts | Listed | Not implemented in current code | **Removed / not yet re-implemented** |
| L1 integrity flags | Listed | Not present in current feature set | **Removed** |
| 95% CIs on markouts | Listed | Not computed in current TCA | **Removed** |
