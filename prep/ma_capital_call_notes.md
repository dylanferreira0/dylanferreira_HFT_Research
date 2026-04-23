# MA Capital Call — Prep Notes
*Internal use only. Keep open during call.*

---

## 1. One-sentence project summary

A full MBO-driven microstructure research pipeline for ES futures that reconstructs the limit order book from individual order events, engineers 46+ queue-level features, trains a purged-cross-validated ridge regression to predict signed forward mid-price returns (toxicity), and outputs calibrated adverse-selection scores used to dynamically scale spreads and skew quotes in a market-making engine.

---

## 2. How I define toxicity

**Operational definition:** Toxicity is the signed forward mid-price return at a fixed horizon after a trade executes, sampled in **event time** (one observation per trade, not per clock tick):

```
fwd_return_{H} = mid[t + H] - mid[t]
```

Four horizons trained separately: **50 ms** (fast/co-lo HFT), **500 ms** (medium), **1000 ms** (slow, primary), **5000 ms** (macro/news). The deployed model targets `fwd_return_1000ms`. For calibration, the label is binarized as toxic = `|fwd_return| > 0` (any non-zero tick move), and isotonic regression maps raw ridge scores to P(toxic) ∈ [0, 1].

**Why this over alternatives:** VPIN requires fixed volume buckets and lags real-time signals; realized adverse selection over a fixed clock window conflates informed flow with vol regimes. A mid-price forward return in event time ties directly to trade-level LOB state and is cleanly purge-able in CV. The microprice-based alternative (`fwd_mp_{H}`) is also computed but not used as the model target — mid is noisier but more standard.

**One honest caveat:** The binary threshold is `|return| > 0` — any tick movement qualifies as toxic. This is a loose definition that likely inflates the positive-class rate.

---

## 3. Feature set

All features are EMA z-scored with two halflives: fast (500 trades, local adaptation) and slow (5000 trades, session-level). **`regressors.get_regressor_names()`** lists every column the code *can* use; **`ToxicityModel`** trains only on columns that actually exist in the dataframe. The **shipped** `model_export/toxicity_model.json` in this repo is a **42-feature** slow (1s) model — that list is authoritative for what was exported from the last run, not every theoretical window.

For that export: **three** L1 OFI z-scores (`50ms`, `500ms`, `5000ms` only), **one** L2 window (`500ms` only), plus deep/agg OFI, imbalances, book shape, interactions `ofi_L1_x_vol`, `ofi_L1_x_spread`, `imb_L2_x_vol`, `slope_x_accel` — and **no** flicker / TWAS / `widen_x_ofi` columns (those live in `regressors.py` when upstream feature columns exist).

**Order flow — strongest in the shipped 42** ★

| Feature | Measures | Why predictive |
|---|---|---|
| ★ `z_ofi_L1_50ms`, `z_ofi_L1_500ms`, `z_ofi_L1_5000ms` | L1 OFI (subset of windows in this export) | Cont, Kukanov & Stoikov (2014) style signal |
| ★ `z_ofi_L2_500ms` | OFI at level 2 (500ms) | Institutional layering one tick back |
| `z_ofi_deep_500ms`, `z_ofi_agg_500ms` | Deep / aggregate OFI | Iceberg and net pressure |
| ★ `z_signed_vol_500ms`, `z_signed_vol_5000ms` | Buy vol − sell vol | Directional momentum |
| `z_vpin_500ms`, `z_vpin_5000ms` | `|signed_vol| / total_vol` in volume buckets | Easley et al. (2012) probability of informed trading |
| `z_kyle_lambda` | `cov(Δmid, OFI) / var(OFI)` | Market impact per unit of order flow |
| `z_fill_asym_500ms` | `(fill_bid − fill_ask) / (fill_bid + fill_ask)` | Iceberg / directional execution pressure |

**Order book state**

| Feature | Measures | Why predictive |
|---|---|---|
| ★ `z_imb_L1`, `z_imb_L2`, `z_imb_L3` | `(bid_sz − ask_sz) / (bid_sz + ask_sz)` per level | Per-level skew reveals intent not visible in aggregate (Cao, Chen & Griffin 2005) |
| `z_L1_L2_disagree` | `sign(imb_L1) * sign(imb_L2)` negated | Institutional layering / spoofing fingerprint |
| `spread_ticks`, `z_spread` | Raw and z-scored quoted spread | Endogenous: wide spread already reflects high adverse selection |
| `z_slope_bid`, `z_slope_ask` | Log-linear depth vs. level | Steep slope → thin book → vulnerable to sweeps (Cartea et al. 2015) |
| `z_curv_bid`, `z_curv_ask` | Second derivative of depth curve | Convex curve = liquidity wall; concave = fragile |
| `z_hhi_bid`, `z_hhi_ask` | Herfindahl concentration of depth | High HHI = one large order dominating → institutional intent (O'Hara 1995) |
| `z_avg_ord_sz_bid`, `z_avg_ord_sz_ask` | `total_L1_sz / n_orders` | Large avg size = institutional; small = retail/HFT quoting noise |

**Queue and cancellation dynamics** *(MBO-only features)*

| Feature | Measures | Why predictive |
|---|---|---|
| `z_cancel_ratio_500ms` | Cancels / trades | High cancel ratio → quote stuffing / informed repricing |
| `z_canc_L1_share_500ms` | L1 cancels / total cancels | Concentrated L1 cancels ahead of moves → informed withdrawal |
| `z_modify_rate_500ms` | Order amendments / trades | HFT repricing fingerprint |
| `z_net_depletion_50ms`, `z_net_depletion_500ms` | Directional queue size change | Queue draining on one side precedes price moves |
| `z_resilience_bid_500ms`, `z_resilience_ask_500ms` | L1 size churn / n_trades after fills | Fast refill = confident MMs; slow = informed flow retreating MMs (Bouchaud et al. 2009) |

**Microprice and volatility**

| Feature | Measures | Why predictive |
|---|---|---|
| ★ `z_microprice_delta` | Rate of change of `(ask*bid_sz + bid*ask_sz)/(bid_sz+ask_sz)` | OB-imbalance-weighted price; cleaner directional signal than mid (Stoikov 2018) |
| `z_realized_vol` | `sqrt(Σ(Δmid²) / (n−1))` in rolling window | High vol amplifies adverse selection |
| `z_latency` | Exchange-to-gateway round-trip (`ts_in_delta_ns`) | Latency spikes indicate quote stuffing or network load |

**Trade arrival**

| Feature | Measures |
|---|---|
| `z_trade_accel` | `log(fast_rate) − log(slow_rate)` — acceleration of trade arrivals |
| `z_trade_rate` | Absolute trade rate |

**Interaction terms** *(in this export)*

`ofi_L1_x_vol`, `ofi_L1_x_spread`, `imb_L2_x_vol`, `slope_x_accel`. Additional interactions (`flicker_x_vol`, `widen_x_ofi`) are built in `regressors.compute_regressors` only when the underlying z-columns exist.

---

## 4. Model and validation

**Model:** Ridge regression (L2, α = 1.0). Choice rationale: microstructure features are collinear by construction (OFI at different windows, multiple imbalance levels); ridge handles this cleanly, coefficients are interpretable, and inference is fast enough for online use. Gradient boosting was not explored — no stated reason in code, but the linear + calibration stack is sufficient for the stated goal.

**Validation scheme:** Purged + embargoed K-fold (K = 5), strictly time-ordered. For the slow (1000 ms) model:
- Purge window: 10 seconds — removes any training sample whose forward label window overlaps the test fold
- Embargo window: 5 seconds — additional buffer after the test fold to kill residual autocorrelation

This is the correct scheme for financial time-series with overlapping labels. Random CV is not used.

**Production adaptation:** Recursive Least Squares (RLS), forgetting factor λ = 0.999 (~1000-trade half-life). Coefficients adapt online without batch retraining.

**Calibration:** Isotonic regression fit on OOF predictions, mapping raw scores to P(toxic) ∈ [0, 1].

**Metrics (slow model, deployed):**

| Metric | Value |
|---|---|
| OOF R² | 0.0614 |
| OOF RMSE | 0.568 ticks |
| Training samples | 1,054,594 |
| Fold R² range | [0.018, 0.082] |

**Data:** Databento MBO, symbol ESU5 (E-mini S&P 500 Sep 2025 contract, instrument id=14160). Full month of August 2025. Typical day: ~9–11.5M raw messages → ~325–370K trades after filtering. Cancel/add ratio roughly 1:1 per day (~3.6–4.4M adds, ~3.6–4.4M cancels, ~670–735K mods).

**Honest limitations:**
- Single symbol (ESU5) and single month — no cross-symbol or cross-year generalization tested
- Fold R² variance (0.018 to 0.082) signals some regime dependence — the model generalizes unevenly across market conditions
- Binary toxic label `|return| > 0` is loose; a threshold in ticks would be more economically meaningful
- RLS online adaptation is implemented but not live-tested
- No out-of-sample test on a different symbol or year

---

## 5. Why MBO specifically

MBO gives individual order lifecycles — every add, cancel, modify, and fill with its order ID — so I can track queue depth changes at each level, compute cancel-to-trade ratios, measure how fast L1 refills after a fill (resilience), and identify order amendments as an HFT fingerprint. MBP-10 gives only price/size snapshots; you can infer net flow but cannot decompose it into adds vs. cancels at a given level, which is where most of the queue-dynamics features live.

**Honest check:** The MBO-specific advantage is real for the cancel ratio, modify rate, fill asymmetry, resilience, and queue depletion features. The OFI features could in principle be approximated from MBP-10 differences, though with less precision.

---

## 6. What I'd do next

1. **Multi-horizon ensemble with learned horizon weights.** The four models (50 ms–5 s) are trained independently; a stacked model that weights them by current vol regime would be more adaptive and is a natural extension of the existing architecture.

2. **Regime-conditional feature selection.** The fold R² variance (0.015–0.110) shows the model is not robust across regimes. The Markov regime classifier is already built; using it to gate feature subsets or train separate ridge models per regime is a concrete, low-effort improvement on top of existing infrastructure.

3. **Cross-asset signal integration.** `options.py` computes GEX, vanna, and IV skew from SPX options data. The **spread optimizer** can consume `gex_values`, but **`run_research` does not pass them today**, so the GEX bin collapses to neutral. Options features are not in the toxicity regressor matrix until wired in.

---

## 7. Things to NOT overclaim *(internal only)*

- **Single symbol, single month** — full August 2025 on ESU5 is a real dataset (~7M+ trades across ~22 trading days), but don't imply multi-symbol or multi-year robustness.
- **Fold R² variance exists** — don't present 0.061 as a stable number; it's an average of 0.018–0.082.
- **RLS online adaptation is not live-tested** — it's implemented and architecturally sound, but I haven't run it in production.
- **No live trading PnL** — the backtest and TCA are on the same dataset the model was fit on (holdout is 40%, which is reasonable but not a separate out-of-sample year).
- **The GEX/options pipeline is separate from the toxicity model** — the 42-feature toxicity model does not include any options features. GEX would only affect spread tables **after** `gex_values` is passed into `run_spread_optimization`; the default `run_research` path does not.
- **The binary toxic label (|return| > 0) is weak** — a single tick move doesn't mean much; this inflates the positive-class rate and makes calibrated probabilities hard to interpret economically.
- **No multi-symbol generalization tested** — I don't know if the feature importances hold for NQ, CL, or other futures.

---

## 8. Cheat-sheet definitions

**OFI (Order Flow Imbalance):** The signed net change in best-bid size minus best-ask size over a window; measures directional pressure from limit order arrivals and cancellations. Predictive because it proxies the arrival rate of informed vs. uninformed orders.

**Microprice:** `(ask_price × bid_sz + bid_price × ask_sz) / (bid_sz + ask_sz)` — a weighted mid-price that tilts toward the side with less depth. Less noisy than mid when imbalance is high; better estimate of fair value intra-tick.

**Queue imbalance:** `(bid_sz − ask_sz) / (bid_sz + ask_sz)` at a given price level. Measures the asymmetry of resting liquidity; high imbalance predicts short-horizon price movement in the direction of the heavier side.

**Cancel-to-trade ratio:** Number of order cancellations divided by number of trades in a window. High ratio indicates order-book layering, quote stuffing, or informed traders withdrawing liquidity before a move. Only computable from MBO data.

**Adverse selection:** The cost incurred by a liquidity provider when the trade they filled is followed by a price move against their resulting position. It's the component of the bid-ask spread that compensates the market maker for trading with better-informed counterparties.

**Toxic flow:** Order flow from counterparties who have a directional informational edge — i.e., trades where the post-trade price move is adverse to the market maker at economically relevant horizons (here: 50 ms to 5 s). Distinct from noise trading, which is directionally random.
