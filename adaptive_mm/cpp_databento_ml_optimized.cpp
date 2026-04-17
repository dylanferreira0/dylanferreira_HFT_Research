#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <map>
#include <random>
#include <chrono>
#include <atomic>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <queue>
#include <csignal>
#include <mutex>
// Windows console UTF-8 support
#ifdef _WIN32
#include <windows.h>
#endif

inline void atomic_add(std::atomic<double>& target, double value) {
    double current = target.load(std::memory_order_relaxed);
    double desired;
    do {
        desired = current + value;
    } while (!target.compare_exchange_weak(current, desired, std::memory_order_relaxed));
}

// Global flag for graceful shutdown
volatile sig_atomic_t g_shutdown = 0;

void signal_handler(int signal) {
    g_shutdown = 1;
}

// Performance attributes
#define ALWAYS_INLINE __attribute__((always_inline))
#define HOT __attribute__((hot))
#define COLD __attribute__((cold))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define ALIGN_64 alignas(64)

struct ALIGN_64 RealMarketData {
    double bid_px_00;
    double ask_px_00;
    double bid_sz_00;
    double ask_sz_00;
    // Depth levels 1-9 (10 total levels: 00-09)
    double bid_px[9];
    double bid_sz[9];
    double ask_px[9];
    double ask_sz[9];
    double trade_price;
    double trade_size;
    char trade_side;
    uint64_t timestamp_ns;
};

struct ALIGN_64 Order {
    uint64_t order_id;
    double price;
    int contracts;
    uint64_t timestamp_ns;
    bool is_bid;
    bool is_active;
    int queue_position;
    double fill_probability;
    int initial_market_depth;  // Market depth when order was placed (for accurate queue tracking)
    bool is_exit_order;  // NEW: true if this is an exit order (TP/SL), false if entry order
    size_t position_index;  // NEW: index into active_positions for exit orders
};

struct ALIGN_64 DOMLevel {
    double price;
    int total_size;           // Market depth at this level (other participants)
    int our_size;             // Our orders at this level
    std::queue<Order> order_queue;
    int queue_depth;          // Count of our orders
    int volume_traded_at_level; // Total volume traded at this price (for queue position estimation)
    double last_best_price;   // Last best bid/ask price when we were actively trading at this level
    bool is_at_best_level;    // Whether this level is currently at best bid/ask
};

struct ALIGN_64 Position {
    double entry_price;
    int32_t contracts;
    uint64_t entry_timestamp;
    bool is_long;
    int32_t hold_intervals;
    int32_t first_profit_target_interval;  // FIRST interval where profit >= +1 tick (for ML training, NO LEAKAGE)
    double mae_at_first_profit_target;  // MAE value AT THE MOMENT we first reached +1 tick (NO LEAKAGE)
    double max_adverse_excursion;  // Worst drawdown during hold (entire trade)
    double max_favorable_excursion;  // Best move during hold
    // Diagnostics for short analysis
    char fill_trade_side;  // 'B' or 'S' - who filled us?
    double entry_bid_px;   // Bid price at entry (for shorts, to see spread)
    double entry_ask_px;   // Ask price at entry
    double next_interval_ask_px;  // Ask price on next interval (to see immediate move)
    // Entry-time features (for timeout prevention model)
    double entry_ofi = 0.0;
    double entry_momentum = 0.0;
    double entry_spread = 0.0;
    double entry_volatility = 0.0;
    double entry_qi_bid = 0.0;
    double entry_qi_ask = 0.0;
    // NEW: Exit order tracking
    uint64_t pending_exit_order_id = 0;  // Order ID of pending exit order (0 = no exit pending)
    double exit_target_price = 0.0;  // Target price for exit (TP or SL)
    bool exit_is_tp = false;  // true if TP exit, false if SL exit
    uint64_t exit_order_placement_ns = 0;  // NEW: When exit order was placed
    double exit_actual_fill_price = 0.0;  // NEW: Actual fill price (from trade_price when filled)
    bool exit_was_slippage = false;  // NEW: true if slippage occurred (market gapped through SL)
};

struct CanceledOrderData {
    bool is_long;
    double order_price;
    uint64_t cancel_timestamp_ns;
    double cancel_toxicity;  // Toxicity value that triggered cancellation
    double cancel_ofi;  // OFI at cancellation
    double cancel_momentum;  // Momentum at cancellation
    double cancel_mid_price;  // Mid price at cancellation
    // Track what happened AFTER cancellation (for analysis)
    double best_price_after_10_intervals;  // Best price seen 10 intervals after cancel
    double best_price_after_20_intervals;  // Best price seen 20 intervals after cancel
    double best_price_after_40_intervals;  // Best price seen 40 intervals after cancel
    bool would_have_been_profitable_10;  // Would we have been profitable after 10 intervals?
    bool would_have_been_profitable_20;  // Would we have been profitable after 20 intervals?
    bool would_have_been_profitable_40;  // Would we have been profitable after 40 intervals?
    int queue_position_at_cancel;  // Queue position when canceled
    double cancel_reason;  // 0=ML toxicity, 1=OFI, 2=momentum, 3=QBI
};

struct TradeData {
    bool is_long;
    double entry_price;
    double exit_price;
    double exit_target_price;  // NEW: Target TP/SL price
    double slippage_ticks;  // NEW: Actual slippage (exit_price - exit_target_price)
    int hold_intervals;
    int first_profit_target_interval;  // FIRST interval where profit >= +1 tick (NO LEAKAGE)
    double mae_at_first_profit_target;  // MAE at moment we first reached +1 tick (NO LEAKAGE)
    double profit_ticks;
    double mae;  // Final MAE (entire trade)
    double mfe;
    double net_pnl;
    // Entry-time features (for timeout prevention model)
    double entry_ofi;
    double entry_momentum;
    double entry_spread;
    double entry_volatility;
    double entry_qi_bid;
    double entry_qi_ask;
    // NEW: Exit-time statistics for analysis
    double exit_ofi;  // OFI at exit time
    double exit_spread;  // Spread at exit time
    double exit_volatility;  // Volatility at exit time
    int exit_queue_position;  // Queue position when exit order filled
    uint64_t exit_order_placement_ns;  // When exit order was placed
    uint64_t exit_order_fill_ns;  // When exit order was filled
    bool exit_was_slippage;  // true if market gapped through SL (used market order)
    bool exit_is_tp;  // true if TP exit, false if SL/timeout
    int exit_order_intervals_to_fill;  // How many intervals until fill
};

struct ALIGN_64 PerformanceMetrics {
    std::atomic<int> total_trades{0};
    std::atomic<int> bid_fills{0};
    std::atomic<int> ask_fills{0};
    std::atomic<int> orders_canceled_toxic{0};
    std::atomic<int> orders_refilled{0};
    std::atomic<int> trade_crosses{0};
    std::atomic<double> total_pnl{0.0};
    std::atomic<double> total_costs{0.0};
    std::atomic<int> profitable_trades{0};
    std::atomic<int> losing_trades{0};
    std::atomic<double> total_profit{0.0};
    std::atomic<double> total_loss{0.0};
    
    // Per-side P&L tracking
    std::atomic<double> long_pnl{0.0};
    std::atomic<double> short_pnl{0.0};
    std::atomic<double> long_profit{0.0};
    std::atomic<double> long_loss{0.0};
    std::atomic<double> short_profit{0.0};
    std::atomic<double> short_loss{0.0};
    std::atomic<int> long_profitable_trades{0};
    std::atomic<int> long_losing_trades{0};
    std::atomic<int> short_profitable_trades{0};
    std::atomic<int> short_losing_trades{0};
    std::atomic<uint64_t> total_decision_time{0};
    std::atomic<uint64_t> decision_count{0};
    std::atomic<uint64_t> max_decision_time{0};
    
    // Hot path latency tracking
    std::atomic<uint64_t> total_toxicity_prediction_time{0};
    std::atomic<uint64_t> toxicity_prediction_count{0};
    std::atomic<uint64_t> max_toxicity_prediction_time{0};
    
    std::atomic<uint64_t> total_exit_decision_time{0};
    std::atomic<uint64_t> exit_decision_count{0};
    std::atomic<uint64_t> max_exit_decision_time{0};
    std::vector<std::string> trade_details;
    std::vector<double> trade_pnls;
    std::vector<TradeData> trades_for_ml;
    std::vector<CanceledOrderData> canceled_orders_for_analysis;  // Track canceled orders for profit analysis
    std::vector<CanceledOrderData> finalized_canceled_orders;  // Completed cancel tracking (no longer updated)
    std::atomic<int> open_positions{0};
    
    // Production KPIs (rolling windows) - thread-safe with mutex for deque access
    mutable std::mutex kpi_mutex;
    std::deque<double> loop_latencies_ms;  // Last 1000 loop latencies
    std::atomic<int> latency_spikes_20ms{0};  // Count of spikes > 20ms
    std::atomic<int> latency_spikes_5ms{0};  // Count of spikes > 5ms
    std::deque<double> toxicity_samples;  // Last 1000 toxicity values
    std::deque<std::pair<uint64_t, double>> fills_with_drift;  // Timestamp, drift_10ms
    std::atomic<int> fills_last_minute{0};
    std::atomic<double> rolling_adverse_selection_10ms{0.0};  // Rolling adverse selection
    std::atomic<int> rolling_toxicity_percent{0};  // Rolling toxicity %
    uint64_t last_minute_start_ns{0};

    // Debug counters for ask-side (shorts) placement
    std::atomic<int> ask_placements{0};
    std::atomic<int> ask_skipped_by_gate{0};
    std::atomic<int> ask_debug_bypass{0};
    std::atomic<int> gate_fail_spread{0};
    std::atomic<int> gate_fail_ofi{0};
    std::atomic<int> gate_fail_mom{0};
    std::atomic<int> gate_fail_qi{0};
    std::atomic<int> gate_fail_tox{0};

    // Per-side TP/SL/Timeout counts
    std::atomic<int> long_tp{0};
    std::atomic<int> long_sl{0};
    std::atomic<int> long_timeout{0};
    std::atomic<int> short_tp{0};
    std::atomic<int> short_sl{0};
    std::atomic<int> short_timeout{0};
    std::atomic<double> bid_level_depth_sum{0.0};
    std::atomic<uint64_t> bid_level_depth_samples{0};
    std::atomic<double> ask_level_depth_sum{0.0};
    std::atomic<uint64_t> ask_level_depth_samples{0};

    // Entry-state sampling (sums guarded by mutex)
    int sampled_long_entries{0};
    int sampled_short_entries{0};
    int sampled_long_count{0};  // Count of samples that actually contributed to sums (every 20th)
    int sampled_short_count{0}; // Count of samples that actually contributed to sums (every 20th)
    double sum_long_ofi{0.0}, sum_long_mom{0.0}, sum_long_spread_ticks{0.0}, sum_long_qi_bid{0.0}, sum_long_qi_ask{0.0};
    double sum_short_ofi{0.0}, sum_short_mom{0.0}, sum_short_spread_ticks{0.0}, sum_short_qi_bid{0.0}, sum_short_qi_ask{0.0};
    
    // Circuit breaker state
    std::atomic<bool> circuit_breaker_active{false};
    std::atomic<int> consecutive_losses{0};
    std::atomic<int> toxic_regime_intervals{0};  // Count of intervals with high toxicity
    std::atomic<int> circuit_breaker_reset_counter{0};  // Counter for reset conditions
    std::atomic<uint64_t> last_circuit_breaker_trigger_ns{0};  // Timestamp of last trigger (for cooldown)
    std::deque<double> recent_returns;  // Last 100 trade returns for Sharpe calculation
    std::mutex returns_mutex;
};

struct ALIGN_64 TradingThresholds {
    int32_t dom_levels = 20;  // INCREASED: Fill the order book with orders (was 6)
    int32_t contracts_per_level = 10;
    double max_position = 10;
    int32_t min_hold_intervals = 5;  // REDUCED: Allow TP to trigger earlier (was 10, but trades timeout at 11)
    int32_t max_hold_intervals = 50;  // MATCH SHORTS: cap long holds to avoid timeouts at 1000 intervals
    double toxic_cancel_threshold = 0.99;  // DATA COLLECTION MODE: Almost never cancel (0.99 = only cancel 1% worst)
    double short_tox_threshold = 0.99;  // DATA COLLECTION MODE: Almost never cancel
    double latency_ms = 5.0;
    double base_fill_rate = 0.10;  // DATA COLLECTION MODE: Increased from 0.02 to 0.10 (10% fill rate)
    bool use_ml_exit = true;  // Use ML exit model (with minimum profit target)
    double ml_exit_mfe_capture_threshold = 0.60;  // Exit when captured 60% of MFE
    
    // SAFETY SWITCHES (immediate fixes)
    bool enable_shorts = true;   // Enable shorts (ask-side placement/fills allowed)
    bool enable_longs = true;
    
    // TP/SL thresholds - NOT USED (timeout-only exits for market making)
    // Market making on ultra-short timeframes (10s) doesn't need TP/SL:
    // - Timeout is the risk management tool
    // - TP/SL cuts profits short on noise
    // - Market making relies on spread capture, not directional moves
    double profit_target_ticks = 1.0;   // NOT USED: Timeout-only exits
    double stop_loss_ticks = -2.0;      // NOT USED: Timeout-only exits
    bool use_dynamic_tp = false;   // NOT USED: Timeout-only exits
    double dynamic_tp_threshold = 0.796;  // NOT USED: Timeout-only exits
    
    // Data validation
    double min_spread = 0.0;     // Reject spreads < 0
    double max_spread = 10.0;    // Reject spreads > 10 ticks
    
    // Toxicity gates
    double ofi_cancel_threshold = 0.90;  // DRASTICALLY RELAXED: from 0.75 to 0.90 (cancel fewer orders)
    double qbi_cancel_threshold = 0.80;   // DRASTICALLY RELAXED: from 0.4 to 0.8 (cancel fewer orders)
    
    // Risk rails - REMOVED LIMITS FOR DATA COLLECTION
    int max_open_positions_per_side = 1000;  // INCREASED: Fill the book (was 200)
    int base_max_open_positions = 1000;  // INCREASED: Fill the book (was 200)
    int max_max_open_positions = 2000;  // INCREASED: Fill the book (was 300)
    int max_fills_per_minute = 500;  // INCREASED: Allow more fills (was 50)
    double kill_switch_adverse_selection = -0.25;  // Stop if AS_10ms < -0.25 ticks
    int kill_switch_toxicity_percent = 30;  // Stop if Toxicity% > 30%
    double max_daily_drawdown = -25000.0;  // Day kill switch at -$25k
    double max_rolling_drawdown_15min = -10000.0;  // 15-min kill switch at -$10k
    
    // Latency SLA
    double p99_9_latency_ms = 5.0;  // p99.9 must be < 5ms
    double max_allowed_latency_ms = 20.0;  // No spikes > 20ms
    
    // CIRCUIT BREAKER THRESHOLDS
    // CIRCUIT BREAKER: ALMOST DISABLED - Only trigger in extreme catastrophic cases
    int circuit_breaker_consecutive_losses = 100;  // VERY LENIENT: Pause after 100 consecutive losses (was 50)
    int circuit_breaker_toxic_regime_intervals = 1000;  // VERY LENIENT: Pause if toxicity > 60% for 1000 intervals (was 500)
    double circuit_breaker_sharpe_threshold = -5.0;  // DISABLED anyway, but set very low
    double circuit_breaker_toxicity_threshold = 60.0;  // VERY LENIENT: Pause if rolling toxicity > 60% (was 50%)
    
    // SHORT-SIDE SPECIFIC PARAMETERS (GATES DISABLED: ML does ALL filtering via cancellation)
    bool short_data_collection_mode = true;  // Enable to collect short samples for training
    double short_ofi_cut = 1.0;  // DISABLED: Always pass, ML filters via cancellation
    double short_mom_max = 10.0;  // DISABLED: Always pass, ML filters via cancellation
    double short_qi_ask_max = 1.0;  // DISABLED: Always pass, ML filters via cancellation
    // short_tox_threshold removed - already declared at line 256
    double short_timeout_ms = 10000.0;  // 10 seconds timeout for shorts - timeout is the risk management tool
    double short_stop_loss_ticks = -2.0;  // NOT USED: Timeout-only exits (market making doesn't need SL)
    double short_profit_target_ticks = 1.0;  // NOT USED: Timeout-only exits (market making doesn't need TP)
    bool enable_short_proximity_cancel = true;  // Cancel on 2 upticks in 50ms
    uint64_t short_cooldown_ns = 150000000ULL;  // 150ms cooldown after proximity cancel
    
    double long_timeout_ms = 10000.0;   // NEW: Long-side timeout to match short behavior (approx 10 seconds)
};

class OptimizedDatabentoIntegration {
private:
    std::vector<RealMarketData> databento_data;
    size_t current_data_index = 0;
    
    std::map<double, DOMLevel> bid_levels;
    std::map<double, DOMLevel> ask_levels;
    std::map<uint64_t, Order> our_orders;
    std::vector<Position> active_positions;
    
    std::deque<double> mid_prices;
    std::atomic<size_t> mid_price_index{0};
    
    // KYLE λ (lambda) tracking for trade intensity control
    std::deque<double> lambda_history;  // Rolling history of lambda values
    double avg_lambda = 0.01;  // Rolling average lambda (initialized to reasonable default)
    static constexpr size_t LAMBDA_HISTORY_WINDOW = 50;  // Keep last 50 lambda values
    
    // Short proximity cancel tracking (2 upticks in 50ms)
    std::map<uint64_t, std::deque<std::pair<uint64_t, double>>> order_bid_history;  // order_id -> (timestamp, bid_px)
    std::map<uint64_t, uint64_t> order_last_cancel_ns;  // order_id -> last cancel timestamp (for cooldown)
    // PREDICTIVE QUEUE POSITIONING: Predict where price will move next
    struct PriceMovementPredictor {
        std::deque<double> recent_mid_prices;  // Last 20 mid prices
        std::deque<double> recent_volumes;     // Last 20 volumes
        std::deque<double> recent_ofi;         // Last 20 OFI values
        double momentum_acceleration = 0.0;     // Rate of change of momentum
        double volume_acceleration = 0.0;       // Rate of change of volume
        
        void update(const RealMarketData& md) {
            double mid = (md.bid_px_00 + md.ask_px_00) / 2.0;
            double volume = md.trade_size;
            double ofi = 0.0;
            if (md.bid_sz_00 + md.ask_sz_00 > 0) {
                ofi = (md.bid_sz_00 - md.ask_sz_00) / (md.bid_sz_00 + md.ask_sz_00);
            }
            
            recent_mid_prices.push_back(mid);
            recent_volumes.push_back(volume);
            recent_ofi.push_back(ofi);
            
            // Keep last 20 samples
            if (recent_mid_prices.size() > 20) recent_mid_prices.pop_front();
            if (recent_volumes.size() > 20) recent_volumes.pop_front();
            if (recent_ofi.size() > 20) recent_ofi.pop_front();
            
            // Calculate momentum acceleration (change in momentum)
            // FIXED: Need 11 elements to safely access end() - 11 (when size=10, end()-11 is before begin())
            if (recent_mid_prices.size() >= 11) {
                double short_momentum = (recent_mid_prices.back() - *(recent_mid_prices.end() - 6)) / 0.25;
                double long_momentum = (recent_mid_prices.back() - *(recent_mid_prices.end() - 11)) / 0.25;
                momentum_acceleration = short_momentum - long_momentum;
            }
            
            // Calculate volume acceleration
            if (recent_volumes.size() >= 10) {
                double short_vol = 0.0, long_vol = 0.0;
                for (size_t i = recent_volumes.size() - 5; i < recent_volumes.size(); ++i) {
                    short_vol += recent_volumes[i];
                }
                for (size_t i = recent_volumes.size() - 10; i < recent_volumes.size() - 5; ++i) {
                    long_vol += recent_volumes[i];
                }
                volume_acceleration = (short_vol - long_vol) / 5.0;
            }
        }
        
        // Predict where price will move next (returns predicted price levels)
        std::vector<double> predict_target_levels(const RealMarketData& md, bool is_long, int num_levels = 3) {
            std::vector<double> predicted_levels;
            if (recent_mid_prices.size() < 10) return predicted_levels;  // Not enough data
            
            double current_mid = (md.bid_px_00 + md.ask_px_00) / 2.0;
            double current_momentum = 0.0;
            if (recent_mid_prices.size() >= 5) {
                current_momentum = (recent_mid_prices.back() - *(recent_mid_prices.end() - 5)) / 0.25;
            }
            
            // Calculate recent OFI trend
            double ofi_trend = 0.0;
            if (recent_ofi.size() >= 5) {
                double recent_avg = 0.0, older_avg = 0.0;
                for (size_t i = recent_ofi.size() - 3; i < recent_ofi.size(); ++i) {
                    recent_avg += recent_ofi[i];
                }
                for (size_t i = recent_ofi.size() - 5; i < recent_ofi.size() - 3; ++i) {
                    older_avg += recent_ofi[i];
                }
                ofi_trend = (recent_avg / 3.0) - (older_avg / 2.0);
            }
            
            // Predict price movement based on:
            // 1. Momentum acceleration (positive = speeding up)
            // 2. Volume acceleration (increasing volume = stronger move)
            // 3. OFI trend (positive = buying pressure building)
            // 4. Current momentum direction
            
            double predicted_move_ticks = 0.0;
            
            if (is_long) {
                // Long: predict upward movement
                // Strong signals: positive momentum + acceleration + volume spike + positive OFI trend
                double signal_strength = 0.0;
                if (current_momentum > 0.2) signal_strength += 0.3;  // Already moving up
                if (momentum_acceleration > 0.1) signal_strength += 0.3;  // Accelerating up
                if (volume_acceleration > 5.0) signal_strength += 0.2;  // Volume increasing
                if (ofi_trend > 0.1) signal_strength += 0.2;  // Buying pressure building
                
                // Predict move: signal_strength * (current_momentum + momentum_acceleration)
                predicted_move_ticks = signal_strength * (std::max(0.0, current_momentum) + std::max(0.0, momentum_acceleration));
                predicted_move_ticks = std::min(predicted_move_ticks, 5.0);  // Cap at 5 ticks
                
                // Generate predicted levels (1-3 ticks ahead)
                for (int i = 1; i <= num_levels && predicted_move_ticks >= i; ++i) {
                    double predicted_price = current_mid + (i * 0.25);  // i ticks above current
                    if (predicted_price >= 6000.0 && predicted_price <= 7000.0) {
                        predicted_levels.push_back(predicted_price);
                    }
                }
            } else {
                // Short: predict downward movement
                double signal_strength = 0.0;
                if (current_momentum < -0.2) signal_strength += 0.3;  // Already moving down
                if (momentum_acceleration < -0.1) signal_strength += 0.3;  // Accelerating down
                if (volume_acceleration > 5.0) signal_strength += 0.2;  // Volume increasing (sell pressure)
                if (ofi_trend < -0.1) signal_strength += 0.2;  // Selling pressure building
                
                predicted_move_ticks = signal_strength * (std::abs(std::min(0.0, current_momentum)) + std::abs(std::min(0.0, momentum_acceleration)));
                predicted_move_ticks = std::min(predicted_move_ticks, 5.0);  // Cap at 5 ticks
                
                // Generate predicted levels (1-3 ticks below)
                for (int i = 1; i <= num_levels && predicted_move_ticks >= i; ++i) {
                    double predicted_price = current_mid - (i * 0.25);  // i ticks below current
                    if (predicted_price >= 6000.0 && predicted_price <= 7000.0) {
                        predicted_levels.push_back(predicted_price);
                    }
                }
            }
            
            return predicted_levels;
        }
        
        // Check if we should place predictive orders (strong signal)
        bool should_place_predictive_orders(bool is_long) const {
            if (recent_mid_prices.size() < 10) return false;
            
            // Strong signal if:
            // 1. Momentum acceleration > 0.2 (speeding up)
            // 2. Volume acceleration > 10 (volume spike)
            // 3. Recent momentum > 0.5 ticks (already moving)
            
            double current_momentum = 0.0;
            if (recent_mid_prices.size() >= 5) {
                current_momentum = (recent_mid_prices.back() - *(recent_mid_prices.end() - 5)) / 0.25;
            }
            
            if (is_long) {
                return (momentum_acceleration > 0.2) || 
                       (volume_acceleration > 10.0 && current_momentum > 0.3) ||
                       (std::abs(momentum_acceleration) > 0.5 && current_momentum > 0.5);
            } else {
                return (momentum_acceleration < -0.2) ||
                       (volume_acceleration > 10.0 && current_momentum < -0.3) ||
                       (std::abs(momentum_acceleration) > 0.5 && current_momentum < -0.5);
            }
        }
    };
    
    PriceMovementPredictor price_predictor;
    
    uint64_t last_long_toxic_cancel_ns = 0;  // Track when longs were last canceled for toxicity
    uint64_t last_short_toxic_cancel_ns = 0;  // Track when shorts were last canceled for toxicity
    static constexpr uint64_t REFILL_COOLDOWN_NS = 5000000000ULL;  // 5 seconds cooldown after toxic cancel
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    uint64_t next_order_id = 1;
    
    PerformanceMetrics metrics;
    TradingThresholds thresholds;
    
    // ML Model loaded from JSON (same as original)
    // IMPROVED: Now supports variable number of features (for enhanced models)
    struct MLModel {
        double intercept = 0.45;
        double spread_coef = 0.15;
        double ofi_coef = 0.25;
        double momentum_coef = 0.20;
        double volatility_coef = 0.12;
        double qi_ask_coef = 0.0;  // For short model
        
        // NEW: Enhanced features (optional, for improved models)
        double ofi_delta_coef = 0.0;
        double microprice_dev_coef = 0.0;
        double book_pressure_coef = 0.0;
        double vol_rank_coef = 0.0;
        double ofi_vol_interaction_coef = 0.0;
        double momentum_spread_interaction_coef = 0.0;
        double qi_spread_interaction_coef = 0.0;
        
        // Scaler parameters (base features)
        double spread_mean = 0.5;
        double spread_scale = 0.3;
        double ofi_mean = 0.0;
        double ofi_scale = 0.3;
        double momentum_mean = 0.0;
        double momentum_scale = 0.5;
        double volatility_mean = 0.001;
        double volatility_scale = 0.002;
        double qi_ask_mean = 0.5;
        double qi_ask_scale = 0.3;
        
        // NEW: Enhanced feature scalers (optional)
        double ofi_delta_mean = 0.0, ofi_delta_scale = 1.0;
        double microprice_dev_mean = 0.0, microprice_dev_scale = 1.0;
        double book_pressure_mean = 1.0, book_pressure_scale = 1.0;
        double vol_rank_mean = 1.0, vol_rank_scale = 1.0;
        double ofi_vol_interaction_mean = 0.0, ofi_vol_interaction_scale = 1.0;
        double momentum_spread_interaction_mean = 0.0, momentum_spread_interaction_scale = 1.0;
        double qi_spread_interaction_mean = 0.0, qi_spread_interaction_scale = 1.0;
        
        bool loaded_from_file = false;
        bool has_enhanced_features = false;  // Flag for enhanced models
        
        // NEW: Platt Scaling (probability calibration)
        struct CalibrationParams {
            double a = 1.0;  // Platt scaling parameter A
            double b = 0.0;  // Platt scaling parameter B
            bool use_platt = false;
        };
        CalibrationParams calibration;
    };
    
    MLModel ml_model_long;  // For bid-side (longs)
    MLModel ml_model_short;  // For ask-side (shorts)

    // ═══════════════════════════════════════════════════════════════════
    //  MBO-TRAINED TOXICITY MODEL (55-feature ridge regression)
    //  Loaded from model_export/toxicity_*.json + calibration_*.json
    //  Falls back to old MLModel if JSON not found.
    // ═══════════════════════════════════════════════════════════════════
    struct MBOToxicityModel {
        std::vector<std::string> feature_names;
        std::vector<double> weights;   // [intercept, w1, w2, ..., wN]
        std::vector<double> feat_mean; // per-feature mean
        std::vector<double> feat_var;  // per-feature variance
        int n_features = 0;
        std::string target;
        bool loaded = false;

        // isotonic calibration (raw |score| -> P(toxic))
        std::vector<double> iso_x;  // X_thresholds (sorted)
        std::vector<double> iso_y;  // y_thresholds (corresponding P(toxic))
        bool has_calibration = false;

        double predict_raw(const std::vector<double>& x_raw) const {
            if (!loaded || (int)x_raw.size() != n_features) return 0.0;
            double score = weights[0]; // intercept
            for (int i = 0; i < n_features; ++i) {
                double z = (x_raw[i] - feat_mean[i]) / std::sqrt(feat_var[i] + 1e-8);
                score += weights[i + 1] * z;
            }
            return score;
        }

        double calibrate(double raw_abs) const {
            if (!has_calibration || iso_x.empty()) return raw_abs;
            // piecewise-linear interpolation on isotonic table
            if (raw_abs <= iso_x.front()) return iso_y.front();
            if (raw_abs >= iso_x.back()) return iso_y.back();
            auto it = std::lower_bound(iso_x.begin(), iso_x.end(), raw_abs);
            size_t idx = std::distance(iso_x.begin(), it);
            if (idx == 0) return iso_y[0];
            double t = (raw_abs - iso_x[idx - 1]) / (iso_x[idx] - iso_x[idx - 1] + 1e-15);
            return iso_y[idx - 1] + t * (iso_y[idx] - iso_y[idx - 1]);
        }
    };
    MBOToxicityModel mbo_tox_fast;   // 50ms horizon
    MBOToxicityModel mbo_tox_medium; // 500ms horizon
    MBOToxicityModel mbo_tox_slow;   // 1000ms horizon (primary)
    MBOToxicityModel mbo_tox_macro;  // 5000ms horizon

    // ═══════════════════════════════════════════════════════════════════
    //  MARKOV REGIME MODEL (from markov_model.json)
    //  Replaces hardcoded regime_thresholds with data-driven regimes.
    // ═══════════════════════════════════════════════════════════════════
    struct MarkovRegimeModel {
        int n_regimes = 3;
        int n_states = 9;
        int n_outcomes = 11;
        std::vector<double> outcome_values;   // e.g. [-5..+5]
        std::vector<double> vol_thresholds;   // quantile boundaries
        // probs[regime][state][outcome] flattened as [regime * n_states * n_outcomes + state * n_outcomes + outcome]
        std::vector<double> probs;
        bool loaded = false;

        int detect_vol_regime(double realized_vol) const {
            if (vol_thresholds.empty()) return 1;
            int r = 0;
            for (double th : vol_thresholds) {
                if (realized_vol > th) r++;
            }
            return std::min(r, n_regimes - 1);
        }

        int discretize_state(double spread_ticks, double imbalance) const {
            int sp = std::clamp((int)spread_ticks - 1, 0, 2);
            int imb_bucket;
            if (imbalance < -0.33) imb_bucket = 0;
            else if (imbalance < 0.33) imb_bucket = 1;
            else imb_bucket = 2;
            return sp * 3 + imb_bucket;
        }

        double expected_move(int regime, int state) const {
            if (!loaded) return 0.0;
            double ev = 0.0;
            int base = regime * n_states * n_outcomes + state * n_outcomes;
            for (int o = 0; o < n_outcomes; ++o) {
                ev += probs[base + o] * outcome_values[o];
            }
            return ev;
        }

        double uncertainty(int regime, int state) const {
            if (!loaded) return 1.0;
            double mu = expected_move(regime, state);
            double var = 0.0;
            int base = regime * n_states * n_outcomes + state * n_outcomes;
            for (int o = 0; o < n_outcomes; ++o) {
                double d = outcome_values[o] - mu;
                var += probs[base + o] * d * d;
            }
            return std::sqrt(var);
        }
    };
    MarkovRegimeModel markov_regime;

    // ═══════════════════════════════════════════════════════════════════
    //  SPREAD OPTIMIZER (from spread_optimizer.json)
    //  Lookup table + linear regression for optimal half-spread.
    // ═══════════════════════════════════════════════════════════════════
    struct SpreadOptimizer {
        // lookup table: key = "vol_tox_gex" -> half_spread in ticks
        std::map<std::string, double> table;
        std::map<std::string, double> skew_table;

        // regression coefficients for smooth interpolation
        double coef_intercept = 1.0;
        double coef_abs_toxicity = 0.0;
        double coef_vol_regime = 0.0;
        double coef_spread_ticks = 0.0;
        double coef_gex_z = 0.0;
        double coef_tox_x_vol = 0.0;

        int n_vol_bins = 3;
        int n_tox_bins = 10;
        int n_gex_bins = 1;
        bool has_gex = false;
        bool loaded = false;

        double lookup_half_spread(int vol_bin, int tox_decile, int gex_bin = 0) const {
            std::string key = std::to_string(vol_bin) + "_" +
                              std::to_string(tox_decile) + "_" +
                              std::to_string(gex_bin);
            auto it = table.find(key);
            if (it != table.end()) return it->second;
            return 1.0; // default
        }

        double lookup_skew(int vol_bin, int tox_decile, int gex_bin = 0) const {
            std::string key = std::to_string(vol_bin) + "_" +
                              std::to_string(tox_decile) + "_" +
                              std::to_string(gex_bin);
            auto it = skew_table.find(key);
            if (it != skew_table.end()) return it->second;
            return 0.0;
        }

        double predict_half_spread(double abs_toxicity, double vol_regime,
                                   double spread_ticks, double gex_z = 0.0) const {
            if (!loaded) return 1.0;
            double hs = coef_intercept
                      + coef_abs_toxicity * abs_toxicity
                      + coef_vol_regime * vol_regime
                      + coef_spread_ticks * spread_ticks
                      + coef_gex_z * gex_z
                      + coef_tox_x_vol * abs_toxicity * vol_regime;
            return std::clamp(hs, 0.5, 5.0);
        }
    };
    SpreadOptimizer spread_optimizer;
    
    // Dynamic TP Model (predict if we should hold past +1 tick to capture 2+ ticks)
    struct DynamicTPModel {
        double intercept = -4.532;
        double coef_interval = -0.546;  // first_profit_target_interval coefficient
        double coef_mae = -0.645;      // mae_at_first_profit_target coefficient
        
        // Scaler parameters
        double interval_mean = 39.655;
        double interval_scale = 27.627;
        double mae_mean = -0.134;
        double mae_scale = 0.396;
        
        double threshold = 0.796;  // Probability threshold to hold past +1 tick
        
        bool loaded = false;
    };
    DynamicTPModel dynamic_tp_model;
    
    // Timeout Prevention Model (predicts if trade will timeout at loss)
    struct TimeoutPreventionModel {
        double intercept = 0.0;
        double coef_ofi = 0.0;
        double coef_momentum = 0.0;
        double coef_spread = 0.0;
        double coef_volatility = 0.0;
        double coef_qi_bid = 0.0;
        double coef_qi_ask = 0.0;
        
        // Scaler parameters
        double ofi_mean = 0.0, ofi_scale = 1.0;
        double momentum_mean = 0.0, momentum_scale = 1.0;
        double spread_mean = 0.5, spread_scale = 0.3;
        double volatility_mean = 0.001, volatility_scale = 0.002;
        double qi_bid_mean = 0.5, qi_bid_scale = 0.3;
        double qi_ask_mean = 0.5, qi_ask_scale = 0.3;
        
        double threshold = 0.4183;  // Optimal threshold from training
        bool loaded = false;
    };
    TimeoutPreventionModel timeout_prevention_model;
    
    // Regime Detection & Adaptive Thresholds
    enum class MarketRegime {
        BALANCED,
        VOLATILE,
        WIDE_SPREAD,
        CHAOTIC
    };
    
    struct RegimeThresholds {
        double timeout_threshold = 0.8;
        double toxicity_threshold = 0.5;
        double meta_threshold = 0.6;  // RAISED: More selective (was 0.4, now 0.6 = 60% probability required)
    };
    
    // O'HARA INVENTORY RISK CONTROL (Ch. 2 Inventory Models)
    // Track signed inventory and apply risk-aversion penalty
    struct InventoryControl {
        int signed_inventory = 0;  // Positive = long, negative = short
        double alpha = 0.15;  // Risk-aversion coefficient (adjustable)
        double inv_penalty_threshold = 0.05;  // Minimum signal strength after inventory penalty
        
        // Update inventory from active positions
        void update_inventory(const std::vector<Position>& positions) {
            signed_inventory = 0;
            for (const auto& pos : positions) {
                if (pos.is_long) {
                    signed_inventory += pos.contracts;
                } else {
                    signed_inventory -= pos.contracts;
                }
            }
        }
        
        // Calculate inventory penalty: α·|inv|
        double get_penalty() const {
            return alpha * std::abs(signed_inventory);
        }
        
        // Check if we should place long order: (signal_strength - penalty) > threshold
        bool should_place_long(double signal_strength) const {
            double adjusted_signal = signal_strength - get_penalty();
            return adjusted_signal > inv_penalty_threshold;
        }
        
        // Calculate skewed bid price (move down when long inventory)
        double get_skewed_bid_price(double base_bid) const {
            if (signed_inventory > 0) {
                // Long inventory: skew bid down by 1 tick per 10 contracts
                int ticks_down = std::min(3, signed_inventory / 10);  // Max 3 ticks
                return base_bid - (ticks_down * 0.25);
            }
            return base_bid;
        }
        
        // Calculate size reduction based on inventory
        int get_adjusted_size(int base_size) const {
            if (std::abs(signed_inventory) > 50) {
                // Reduce size when inventory is high
                return std::max(1, base_size / 2);
            }
            return base_size;
        }
    };
    InventoryControl inventory_control;
    
    // O'HARA SEQUENTIAL LEARNING / BAYESIAN STREAK (Ch. 3.5 Sequential Trade)
    // Track consecutive same-side prints to update informed-trade probability
    struct SequentialLearning {
        int buyer_streak = 0;  // Consecutive buyer-initiated prints
        int seller_streak = 0;  // Consecutive seller-initiated prints
        double beta = 0.1;  // Streak coefficient for posterior boost
        
        void update(char trade_side) {
            if (trade_side == 'B' || trade_side == 'b') {
                buyer_streak++;
                seller_streak = 0;  // Reset opposite streak
            } else if (trade_side == 'S' || trade_side == 's') {
                seller_streak++;
                buyer_streak = 0;  // Reset opposite streak
            }
        }
        
        // Calculate posterior boost for toxicity logit
        double get_toxicity_boost(bool is_long) const {
            if (is_long) {
                // Long positions: buyer streak increases informed-trade probability
                return beta * buyer_streak;
            } else {
                // Short positions: seller streak increases informed-trade probability
                return beta * seller_streak;
            }
        }
    };
    SequentialLearning sequential_learning;
    
    // O'HARA TIME-OF-DAY FEATURES (Ch. 6.3 Role of Time)
    enum class SessionState {
        OPEN,      // First 30 minutes
        REGULAR,   // Normal trading hours
        LUNCH,     // Midday lull
        CLOSE,     // Last 30 minutes
        NEWS       // Volatile periods (detected by volatility spike)
    };
    
    SessionState detect_session_state(uint64_t timestamp_ns) const {
        // Simplified: use volatility and time-based heuristics
        // In production, use actual session times from market data
        if (recent_volatility.size() > 10) {
            double recent_vol = recent_volatility.back();
            if (recent_vol > 0.003) {  // High volatility
                return SessionState::NEWS;
            }
        }
        // TODO: Add actual time-of-day detection from timestamp
        return SessionState::REGULAR;
    }
    
    // Time-conditional thresholds (stricter during OPEN/NEWS)
    double get_time_adjusted_threshold(double base_threshold, SessionState session) const {
        switch (session) {
            case SessionState::OPEN:
            case SessionState::NEWS:
                return base_threshold * 0.8;  // 20% stricter (lower threshold = easier to cancel)
            case SessionState::CLOSE:
                return base_threshold * 0.9;  // 10% stricter
            default:
                return base_threshold;
        }
    }
    
    std::map<MarketRegime, RegimeThresholds> regime_thresholds;
    
    // NEW: K-means centroids for regime detection (distance-based)
    struct RegimeCentroid {
        double volatility_center;
        double spread_center;
        MarketRegime label;
    };
    
    std::vector<RegimeCentroid> regime_centroids = {
        {0.001, 0.25, MarketRegime::BALANCED},
        {0.003, 0.25, MarketRegime::VOLATILE},
        {0.001, 0.75, MarketRegime::WIDE_SPREAD},
        {0.003, 0.75, MarketRegime::CHAOTIC}
    };
    
    // NEW: Meta-Model (combines toxicity + timeout predictions)
    struct MetaModel {
        double intercept = 0.0;
        std::vector<double> coefficients;
        std::vector<std::string> features;
        double threshold = 0.4;
        bool loaded = false;
        
        // Scaler parameters (for meta features)
        std::vector<double> scaler_mean;
        std::vector<double> scaler_scale;
    };
    MetaModel meta_model;
    
    // NEW: Per-side/per-regime model storage
    std::map<std::string, MLModel> regime_models_long;  // "regime_name" -> MLModel
    std::map<std::string, MLModel> regime_models_short;
    std::map<std::string, TimeoutPreventionModel> regime_timeout_models;
    std::map<std::string, MetaModel> regime_meta_models;
    
    // Entry Filter Model (predicts if entry will be profitable)
    struct EntryFilterModel {
        double intercept = 0.0;
        std::map<std::string, double> weights;  // Feature weights
        std::vector<double> scaler_mean;       // Scaler means
        std::vector<double> scaler_scale;      // Scaler scales
        std::vector<std::string> feature_order; // Order of features
        double threshold = 0.5;                // Threshold for entry
        bool loaded = false;
    };
    
    // Entry filter models per side×regime
    std::map<std::string, EntryFilterModel> entry_filter_models;
    std::string current_regime_string = "balanced";  // Track current regime for model selection
    
    // Rolling volatility and spread for regime detection
    std::deque<double> recent_volatility;
    std::deque<double> recent_spread;
    static constexpr size_t REGIME_WINDOW = 100;  // 100 intervals for regime calculation
    
    // NEW: Rolling statistics for enhanced features
    std::deque<double> ofi_history;  // For OFI slope calculation (OFI(t) - OFI(t-3))
    std::deque<double> spread_history;  // For spread rank (percentile in last 20)
    std::deque<double> volatility_history;  // For RV rank (percentile in last 100)
    static constexpr size_t OFI_HISTORY_WINDOW = 10;  // Keep last 10 OFI values
    static constexpr size_t SPREAD_RANK_WINDOW = 50;  // For spread rank calculation
    static constexpr size_t RV_RANK_WINDOW = 200;  // For RV rank calculation
    
    // NEW: Rolling Scaler for dynamic feature normalization
    struct RollingScaler {
        std::deque<double> values;
        size_t window_size = 500;
        double mean = 0.0;
        double scale = 1.0;
        
        void update(double value) {
            values.push_back(value);
            if (values.size() > window_size) {
                values.pop_front();
            }
            
            // Recalculate stats every 50 updates (to reduce computation)
            if (values.size() % 50 == 0 || values.size() == window_size) {
                calculate_stats();
            }
        }
        
        void calculate_stats() {
            if (values.empty()) return;
            
            double sum = 0.0;
            for (double v : values) sum += v;
            mean = sum / values.size();
            
            double sum_sq = 0.0;
            for (double v : values) {
                sum_sq += (v - mean) * (v - mean);
            }
            scale = std::sqrt(sum_sq / values.size());
            if (scale < 1e-6) scale = 1.0;  // Prevent division by zero
        }
        
        double standardize(double value) {
            if (values.empty()) return value;  // No data yet
            return (value - mean) / scale;
        }
    };
    
    // NEW: Get regime as string for model lookup
    std::string get_regime_string(MarketRegime regime) {
        switch (regime) {
            case MarketRegime::VOLATILE: return "volatile";
            case MarketRegime::WIDE_SPREAD: return "wide_spread";
            case MarketRegime::CHAOTIC: return "chaotic";
            default: return "balanced";
        }
    }
    
    // NEW: Regime state with hysteresis (smooth transitions)
    struct RegimeState {
        MarketRegime current = MarketRegime::BALANCED;
        MarketRegime previous = MarketRegime::BALANCED;
        std::deque<MarketRegime> recent_detections;
        static constexpr int HYSTERESIS_WINDOW = 5;  // Require 5 consecutive detections before switching
    };
    RegimeState regime_state;
    
    MarketRegime detect_regime(const RealMarketData& market_data) {
        // Calculate current volatility (std dev of recent mid prices)
        double current_volatility = 0.0;
        if (mid_prices.size() >= 10) {
            double sum = 0.0, sum_sq = 0.0;
            int count = 0;
            for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                sum += *it;
                sum_sq += (*it) * (*it);
                count++;
            }
            if (count > 1) {
                double mean = sum / count;
                current_volatility = std::sqrt((sum_sq / count) - (mean * mean));
            }
        }
        
        // Calculate current spread
        double current_spread = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
        
        // Track rolling metrics
        recent_volatility.push_back(current_volatility);
        recent_spread.push_back(current_spread);
        volatility_history.push_back(current_volatility);
        spread_history.push_back(current_spread);
        
        if (recent_volatility.size() > REGIME_WINDOW) {
            recent_volatility.pop_front();
            recent_spread.pop_front();
        }
        if (volatility_history.size() > RV_RANK_WINDOW) {
            volatility_history.pop_front();
        }
        if (spread_history.size() > SPREAD_RANK_WINDOW) {
            spread_history.pop_front();
        }
        
        // NEW: Use K-means centroids for distance-based classification
        if (regime_centroids.size() > 0) {
            double min_distance = 1e10;
            MarketRegime closest_regime = MarketRegime::BALANCED;
            
            for (const auto& centroid : regime_centroids) {
                double distance = std::sqrt(
                    (current_volatility - centroid.volatility_center) * (current_volatility - centroid.volatility_center) +
                    (current_spread - centroid.spread_center) * (current_spread - centroid.spread_center)
                );
                
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_regime = centroid.label;
                }
            }
            
            // NEW: Apply hysteresis (smooth state transitions)
            regime_state.recent_detections.push_back(closest_regime);
            if (regime_state.recent_detections.size() > RegimeState::HYSTERESIS_WINDOW) {
                regime_state.recent_detections.pop_front();
            }
            
            // Only switch if new regime detected for N consecutive intervals
            bool should_switch = true;
            if (regime_state.recent_detections.size() >= RegimeState::HYSTERESIS_WINDOW) {
                for (const auto& r : regime_state.recent_detections) {
                    if (r != closest_regime) {
                        should_switch = false;
                        break;
                    }
                }
            } else {
                should_switch = false;  // Not enough history yet
            }
            
            if (should_switch && closest_regime != regime_state.current) {
                regime_state.previous = regime_state.current;
                regime_state.current = closest_regime;
            }
            
            current_regime_string = get_regime_string(regime_state.current);
            return regime_state.current;
        }
        
        // Fallback to percentile-based detection
        if (recent_volatility.size() < 50) {
            current_regime_string = "balanced";
            return MarketRegime::BALANCED;
        }
        
        std::vector<double> vol_sorted(recent_volatility.begin(), recent_volatility.end());
        std::vector<double> spread_sorted(recent_spread.begin(), recent_spread.end());
        std::sort(vol_sorted.begin(), vol_sorted.end());
        std::sort(spread_sorted.begin(), spread_sorted.end());
        
        double vol_q70 = vol_sorted[static_cast<size_t>(vol_sorted.size() * 0.7)];
        double spread_q70 = spread_sorted[static_cast<size_t>(spread_sorted.size() * 0.7)];
        
        bool high_vol = current_volatility > vol_q70;
        bool wide_spread = current_spread > spread_q70;
        
        MarketRegime regime;
        if (high_vol && wide_spread) {
            regime = MarketRegime::CHAOTIC;
        } else if (high_vol) {
            regime = MarketRegime::VOLATILE;
        } else if (wide_spread) {
            regime = MarketRegime::WIDE_SPREAD;
        } else {
            regime = MarketRegime::BALANCED;
        }
        
        current_regime_string = get_regime_string(regime);
        return regime;
    }
    void load_regime_thresholds() {
        // Load from JSON or use defaults from optimization
        // Default thresholds from optimize_regime_adaptive.py results
        // Format: {timeout_threshold, toxicity_threshold, meta_threshold}
        regime_thresholds[MarketRegime::BALANCED] = {0.8, 0.5, 0.6};  // RAISED meta from 0.4 to 0.6
        regime_thresholds[MarketRegime::VOLATILE] = {0.8, 0.5, 0.6};  // RAISED meta from 0.4 to 0.6
        regime_thresholds[MarketRegime::WIDE_SPREAD] = {0.5, 0.5, 0.6};  // RAISED meta from 0.4 to 0.6
        regime_thresholds[MarketRegime::CHAOTIC] = {0.6, 0.5, 0.6};  // RAISED meta from 0.4 to 0.6
        
        // Try to load from JSON
        std::ifstream file("regime_adaptive_thresholds.json");
        if (file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();
            
            // Simple JSON parsing for regime thresholds
            // Format: {"balanced": {"timeout_threshold": 0.8, "toxicity_threshold": 0.5}, ...}
            auto parse_regime = [&](const std::string& regime_name, MarketRegime regime_enum) {
                std::string pattern = "\"" + regime_name + "\":";
                size_t pos = content.find(pattern);
                if (pos != std::string::npos) {
                    pos = content.find("\"timeout_threshold\":", pos);
                    if (pos != std::string::npos) {
                        pos += 20;
                        size_t end = content.find_first_of(",}", pos);
                        regime_thresholds[regime_enum].timeout_threshold = std::stod(content.substr(pos, end - pos));
                    }
                    pos = content.find("\"toxicity_threshold\":", pos);
                    if (pos != std::string::npos) {
                        pos += 22;
                        size_t end = content.find_first_of(",}", pos);
                        regime_thresholds[regime_enum].toxicity_threshold = std::stod(content.substr(pos, end - pos));
                    }
                }
            };
            
            parse_regime("balanced", MarketRegime::BALANCED);
            parse_regime("volatile", MarketRegime::VOLATILE);
            parse_regime("wide_spread", MarketRegime::WIDE_SPREAD);
            parse_regime("chaotic", MarketRegime::CHAOTIC);
            
            std::cout << "✅ Loaded regime-adaptive thresholds from JSON" << std::endl;
        } else {
            std::cout << "📊 Using default regime-adaptive thresholds" << std::endl;
        }
    }
    
    double ofi_mean = 0.0;
    double ofi_std = 1.0;
    size_t ofi_count = 0;

    ALWAYS_INLINE void sample_entry_state(bool is_long, const RealMarketData& md) noexcept {
        // 1-in-20 sampling per side to keep overhead negligible
        double spread_ticks = (md.ask_px_00 - md.bid_px_00) / 0.25;
        double ofi = 0.0;
        if (md.bid_sz_00 + md.ask_sz_00 > 0.0) {
            ofi = (md.bid_sz_00 - md.ask_sz_00) / (md.bid_sz_00 + md.ask_sz_00);
        }
        double mom_ticks = 0.0;
        if (mid_prices.size() >= 6) {
            double m_now = mid_prices.back();
            double m_prev = *(std::prev(mid_prices.end(), 6));
            mom_ticks = (m_now - m_prev) / 0.25;
        }
        double qi_bid = 0.0, qi_ask = 0.0;
        if (md.bid_sz_00 + md.ask_sz_00 > 0.0) {
            qi_bid = md.bid_sz_00 / (md.bid_sz_00 + md.ask_sz_00);
            qi_ask = md.ask_sz_00 / (md.bid_sz_00 + md.ask_sz_00);
        }
        double tox = predict_toxicity(md, is_long);  // Use side-specific model
        std::lock_guard<std::mutex> lock(metrics.kpi_mutex);
        if (is_long) {
            metrics.sampled_long_entries++;
            if ((metrics.sampled_long_entries % 20) == 0) {
                metrics.sum_long_ofi += ofi;
                metrics.sum_long_mom += mom_ticks;
                metrics.sum_long_spread_ticks += spread_ticks;
                metrics.sum_long_qi_bid += qi_bid;
                metrics.sum_long_qi_ask += qi_ask;
                metrics.sampled_long_count++;  // Track actual sample count
            }
        } else {
            metrics.sampled_short_entries++;
            if ((metrics.sampled_short_entries % 20) == 0) {
                metrics.sum_short_ofi += ofi;
                metrics.sum_short_mom += mom_ticks;
                metrics.sum_short_spread_ticks += spread_ticks;
                metrics.sum_short_qi_bid += qi_bid;
                metrics.sum_short_qi_ask += qi_ask;
                metrics.sampled_short_count++;  // Track actual sample count
            }
        }
    }

    // Side-aware gate for SHORT entries (ask-side) - DATA COLLECTION MODE
    // Goal: Collect 2-5k clean short samples with loosened    gates
    // Parameters: OFI < short_ofi_cut (-0.001), momentum ≤ short_mom_max (0.0), spread == 1, qi_ask ≤ 0.60, toxicity < 0.58
    ALWAYS_INLINE bool side_ok_short(const RealMarketData& md) noexcept {
        // MINIMAL GATES: Only basic sanity checks, ML does ALL filtering via cancellation
        // Spread sanity: Only reject extreme spreads (> 5 ticks or < 0.25 ticks)
        const double spread_ticks = (md.ask_px_00 - md.bid_px_00) / 0.25;
        if (spread_ticks > 5.0 || spread_ticks < 0.25) {  // Only reject extreme spreads
            metrics.gate_fail_spread.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        // All other gates DISABLED - ML will cancel toxic orders aggressively
        // We place orders freely, then check_toxicity_cancellation() removes bad ones
        return true;
    }

public:
    OptimizedDatabentoIntegration() : rng(std::random_device{}()), uniform_dist(0.0, 1.0) {
        mid_prices.resize(100, 0.0);
        load_regime_thresholds();  // Load regime-adaptive thresholds
    }
    
    // Helper to load a single ML model from JSON
    bool load_single_ml_model(const std::string& filename, MLModel& model, double& threshold) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // Simple JSON parsing (assumes valid JSON)
        try {
            // Parse intercept
            size_t intercept_pos = content.find("\"intercept\":");
            if (intercept_pos != std::string::npos) {
                intercept_pos += 12;
                size_t end = content.find_first_of(",}", intercept_pos);
                model.intercept = std::stod(content.substr(intercept_pos, end - intercept_pos));
            }
            
            // Parse coefficients
            size_t coef_start = content.find("\"coefficients\":{");
            if (coef_start != std::string::npos) {
                coef_start += 16;
                size_t coef_end = content.find("}", coef_start);
                std::string coef_block = content.substr(coef_start, coef_end - coef_start);
                
                // Parse each coefficient
                for (const auto& feat : {"spread", "ofi", "momentum", "volatility", "qi_ask"}) {
                    std::string pattern = "\"" + std::string(feat) + "\":";
                    size_t pos = coef_block.find(pattern);
                    if (pos != std::string::npos) {
                        pos += pattern.length();
                        size_t end = coef_block.find_first_of(",}", pos);
                        double val = std::stod(coef_block.substr(pos, end - pos));
                        
                        if (feat == std::string("spread")) model.spread_coef = val;
                        else if (feat == std::string("ofi")) model.ofi_coef = val;
                        else if (feat == std::string("momentum")) model.momentum_coef = val;
                        else if (feat == std::string("volatility")) model.volatility_coef = val;
                        else if (feat == std::string("qi_ask")) model.qi_ask_coef = val;
                    }
                }
            }
            
            // Parse scaler mean/scale
            size_t scaler_start = content.find("\"scaler\":{");
            if (scaler_start != std::string::npos) {
                scaler_start += 10;
                size_t scaler_end = content.find("}", scaler_start);
                std::string scaler_block = content.substr(scaler_start, scaler_end - scaler_start);
                
                // Parse mean array
                size_t mean_start = scaler_block.find("\"mean\":[");
                if (mean_start != std::string::npos) {
                    mean_start += 8;
                    size_t mean_end = scaler_block.find("]", mean_start);
                    std::string mean_str = scaler_block.substr(mean_start, mean_end - mean_start);
                    std::vector<double> means;
                    size_t pos = 0;
                    while (pos < mean_str.length()) {
                        size_t comma = mean_str.find(",", pos);
                        if (comma == std::string::npos) comma = mean_str.length();
                        means.push_back(std::stod(mean_str.substr(pos, comma - pos)));
                        pos = comma + 1;
                    }
                    if (means.size() >= 1) model.spread_mean = means[0];
                    if (means.size() >= 2) model.ofi_mean = means[1];
                    if (means.size() >= 3) model.momentum_mean = means[2];
                    if (means.size() >= 4) model.volatility_mean = means[3];
                    if (means.size() >= 5) model.qi_ask_mean = means[4];
                }
                
                // Parse scale array
                size_t scale_start = scaler_block.find("\"scale\":[");
                if (scale_start != std::string::npos) {
                    scale_start += 9;
                    size_t scale_end = scaler_block.find("]", scale_start);
                    std::string scale_str = scaler_block.substr(scale_start, scale_end - scale_start);
                    std::vector<double> scales;
                    size_t pos = 0;
                    while (pos < scale_str.length()) {
                        size_t comma = scale_str.find(",", pos);
                        if (comma == std::string::npos) comma = scale_str.length();
                        scales.push_back(std::stod(scale_str.substr(pos, comma - pos)));
                        pos = comma + 1;
                    }
                    if (scales.size() >= 1) model.spread_scale = scales[0];
                    if (scales.size() >= 2) model.ofi_scale = scales[1];
                    if (scales.size() >= 3) model.momentum_scale = scales[2];
                    if (scales.size() >= 4) model.volatility_scale = scales[3];
                    if (scales.size() >= 5) model.qi_ask_scale = scales[4];
                }
            }
            
            // Parse threshold
            size_t thresh_pos = content.find("\"toxicity_threshold\":");
            if (thresh_pos != std::string::npos) {
                thresh_pos += 21;
                size_t end = content.find_first_of(",}", thresh_pos);
                threshold = std::stod(content.substr(thresh_pos, end - thresh_pos));
            }
            
            model.loaded_from_file = true;
            return true;
        } catch (...) {
            return false;
        }
    }
    
    // Load Dynamic TP Model from JSON (fast, low-latency)
    bool load_dynamic_tp_model(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        try {
            // Simple JSON parsing for dynamic_tp_model.json
            size_t threshold_pos = content.find("\"threshold\":");
            if (threshold_pos != std::string::npos) {
                size_t start = content.find_first_of("0123456789.", threshold_pos);
                size_t end = content.find_first_not_of("0123456789.", start);
                dynamic_tp_model.threshold = std::stod(content.substr(start, end - start));
            }
            
            // Parse scaler mean
            size_t mean_pos = content.find("\"mean\":[");
            if (mean_pos != std::string::npos) {
                size_t start = content.find_first_of("0123456789.-", mean_pos);
                size_t end = content.find(",", start);
                dynamic_tp_model.interval_mean = std::stod(content.substr(start, end - start));
                start = content.find_first_of("0123456789.-", end);
                end = content.find("]", start);
                dynamic_tp_model.mae_mean = std::stod(content.substr(start, end - start));
            }
            
            // Parse scaler scale
            size_t scale_pos = content.find("\"scale\":[");
            if (scale_pos != std::string::npos) {
                size_t start = content.find_first_of("0123456789.", scale_pos);
                size_t end = content.find(",", start);
                dynamic_tp_model.interval_scale = std::stod(content.substr(start, end - start));
                start = content.find_first_of("0123456789.", end);
                end = content.find("]", start);
                dynamic_tp_model.mae_scale = std::stod(content.substr(start, end - start));
            }
            
            // Parse intercept
            size_t intercept_pos = content.find("\"intercept\":");
            if (intercept_pos != std::string::npos) {
                size_t start = content.find_first_of("0123456789.-", intercept_pos);
                size_t end = content.find_first_not_of("0123456789.-", start);
                dynamic_tp_model.intercept = std::stod(content.substr(start, end - start));
            }
            
            // Parse coefficients
            size_t coef_pos = content.find("\"coef\":[");
            if (coef_pos != std::string::npos) {
                size_t start = content.find_first_of("0123456789.-", coef_pos);
                size_t end = content.find(",", start);
                dynamic_tp_model.coef_interval = std::stod(content.substr(start, end - start));
                start = content.find_first_of("0123456789.-", end);
                end = content.find("]", start);
                dynamic_tp_model.coef_mae = std::stod(content.substr(start, end - start));
            }
            
            dynamic_tp_model.loaded = true;
            return true;
        } catch (...) {
            return false;
        }
    }
    
    // NEW: Load per-side/per-regime models from directory structure
    bool load_advanced_models(const std::string& base_dir = "models") {
        std::cout << "\n📊 Loading Advanced Per-Side/Per-Regime Models from: " << base_dir << std::endl;
        
        std::vector<std::string> regimes = {"balanced", "volatile", "wide_spread", "chaotic"};
        
        for (const auto& regime : regimes) {
            std::cout << "\n  Loading regime: " << regime << std::endl;
            
            // Load long toxicity model
            std::string long_tox_file = base_dir + "/side=long/regime=" + regime + "/toxicity.json";
            MLModel long_model;
            double long_threshold = 0.4;
            if (load_advanced_toxicity_model(long_tox_file, long_model)) {
                regime_models_long[regime] = long_model;
                std::cout << "    ✅ Loaded LONG toxicity model for " << regime << std::endl;
            } else {
                std::cout << "    ⚠️  Failed to load LONG toxicity model for " << regime << std::endl;
            }
            
            // Load thresholds
            std::string thresholds_file = base_dir + "/side=long/regime=" + regime + "/thresholds.json";
            RegimeThresholds reg_thresh;
            if (load_regime_thresholds_from_file(thresholds_file, reg_thresh)) {
                MarketRegime mr = MarketRegime::BALANCED;
                if (regime == "volatile") mr = MarketRegime::VOLATILE;
                else if (regime == "wide_spread") mr = MarketRegime::WIDE_SPREAD;
                else if (regime == "chaotic") mr = MarketRegime::CHAOTIC;
                
                regime_thresholds[mr] = reg_thresh;
                std::cout << "    ✅ Loaded thresholds: tox=" << reg_thresh.toxicity_threshold 
                          << ", meta=" << reg_thresh.meta_threshold << std::endl;
            }
            
            // Load meta model
            std::string meta_file = base_dir + "/side=long/regime=" + regime + "/meta.json";
            MetaModel meta_model_regime;
            if (load_meta_model(meta_file, meta_model_regime)) {
                regime_meta_models[regime] = meta_model_regime;
                std::cout << "    ✅ Loaded META model for " << regime << std::endl;
            }
        }
        
        // Load entry filter models
        for (const auto& regime : regimes) {
            std::string long_entry_file = base_dir + "/side=long/regime=" + regime + "/entry_filter.json";
            std::string short_entry_file = base_dir + "/side=short/regime=" + regime + "/entry_filter.json";
            
            EntryFilterModel long_entry_model;
            if (load_entry_filter_model(long_entry_file, long_entry_model)) {
                entry_filter_models["LONG_" + regime] = long_entry_model;
                std::cout << "    Loaded LONG entry filter for " << regime << std::endl;
            }
            
            EntryFilterModel short_entry_model;
            if (load_entry_filter_model(short_entry_file, short_entry_model)) {
                entry_filter_models["SHORT_" + regime] = short_entry_model;
                std::cout << "    Loaded SHORT entry filter for " << regime << std::endl;
            }
        }
        
        return true;
    }
    
    // Helper: Load advanced toxicity model with full feature support
    bool load_advanced_toxicity_model(const std::string& filename, MLModel& model) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        try {
            // Parse intercept
            size_t intercept_pos = content.find("\"intercept\":");
            if (intercept_pos != std::string::npos) {
                intercept_pos = content.find_first_of("0123456789.-", intercept_pos);
                size_t end = content.find_first_not_of("0123456789.-", intercept_pos);
                model.intercept = std::stod(content.substr(intercept_pos, end - intercept_pos));
            }
            
            // Parse coefficients array
            std::vector<double> coefs = parse_json_array_simple(content, "\"coef\":[");
            if (coefs.size() >= 4) {
                model.ofi_coef = coefs[0];
                model.momentum_coef = coefs[1];
                model.spread_coef = coefs[2];
                model.volatility_coef = coefs[3];
                
                if (coefs.size() >= 5) model.ofi_delta_coef = coefs[4];
                if (coefs.size() >= 7) model.microprice_dev_coef = coefs[6];
                if (coefs.size() >= 8) model.book_pressure_coef = coefs[7];
                if (coefs.size() >= 9) model.vol_rank_coef = coefs[8];
                if (coefs.size() >= 12) model.ofi_vol_interaction_coef = coefs[11];
                if (coefs.size() >= 13) model.momentum_spread_interaction_coef = coefs[12];
                if (coefs.size() >= 16) model.qi_spread_interaction_coef = coefs[15];
                
                model.has_enhanced_features = (coefs.size() > 4);
            }
            
            // Parse scaler arrays
            std::vector<double> mean_array = parse_json_array_simple(content, "\"mean\":[");
            std::vector<double> scale_array = parse_json_array_simple(content, "\"scale\":[");
            
            if (mean_array.size() >= 4 && scale_array.size() >= 4) {
                model.ofi_mean = mean_array[0];
                model.ofi_scale = scale_array[0];
                model.momentum_mean = mean_array[1];
                model.momentum_scale = scale_array[1];
                model.spread_mean = mean_array[2];
                model.spread_scale = scale_array[2];
                model.volatility_mean = mean_array[3];
                model.volatility_scale = scale_array[3];
                
                if (mean_array.size() >= 5 && scale_array.size() >= 5) {
                    model.ofi_delta_mean = mean_array[4];
                    model.ofi_delta_scale = scale_array[4];
                }
                if (mean_array.size() >= 7 && scale_array.size() >= 7) {
                    model.microprice_dev_mean = mean_array[6];
                    model.microprice_dev_scale = scale_array[6];
                }
                if (mean_array.size() >= 8 && scale_array.size() >= 8) {
                    model.book_pressure_mean = mean_array[7];
                    model.book_pressure_scale = scale_array[7];
                }
                if (mean_array.size() >= 9 && scale_array.size() >= 9) {
                    model.vol_rank_mean = mean_array[8];
                    model.vol_rank_scale = scale_array[8];
                }
            }
            
            model.loaded_from_file = true;
            return true;
        } catch (...) {
            return false;
        }
    }
    
    // Helper: Parse JSON array
    std::vector<double> parse_json_array_simple(const std::string& content, const std::string& key) {
        std::vector<double> result;
        size_t pos = content.find(key);
        if (pos == std::string::npos) return result;
        
        pos = content.find("[", pos);
        if (pos == std::string::npos) return result;
        
        pos++;
        size_t end = content.find("]", pos);
        if (end == std::string::npos) return result;
        
        std::string array_str = content.substr(pos, end - pos);
        std::istringstream iss(array_str);
        std::string token;
        
        while (std::getline(iss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            try {
                result.push_back(std::stod(token));
            } catch (...) {}
        }
        
        return result;
    }
    
    // Helper: Load thresholds from JSON
    bool load_regime_thresholds_from_file(const std::string& filename, RegimeThresholds& thresholds) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        try {
            size_t tox_pos = content.find("\"toxicity_threshold\":");
            if (tox_pos != std::string::npos) {
                tox_pos = content.find_first_of("0123456789.", tox_pos);
                size_t end = content.find_first_not_of("0123456789.", tox_pos);
                thresholds.toxicity_threshold = std::stod(content.substr(tox_pos, end - tox_pos));
            }
            
            size_t meta_pos = content.find("\"meta_threshold\":");
            if (meta_pos != std::string::npos) {
                meta_pos = content.find_first_of("0123456789.", meta_pos);
                size_t end = content.find_first_not_of("0123456789.", meta_pos);
                thresholds.meta_threshold = std::stod(content.substr(meta_pos, end - meta_pos));
            }
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    // Helper: Load meta model
    // Load entry filter model from JSON
    bool load_entry_filter_model(const std::string& filename, EntryFilterModel& model) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        try {
            // Parse intercept
            size_t intercept_pos = content.find("\"intercept\":");
            if (intercept_pos != std::string::npos) {
                intercept_pos = content.find_first_of("0123456789.-", intercept_pos);
                size_t end = content.find_first_not_of("0123456789.-", intercept_pos);
                model.intercept = std::stod(content.substr(intercept_pos, end - intercept_pos));
            }
            
            // Parse scaler means
            model.scaler_mean = parse_json_array_simple(content, "\"mean\":[");
            
            // Parse scaler scales
            model.scaler_scale = parse_json_array_simple(content, "\"scale\":[");
            
            // Parse weights
            size_t weights_pos = content.find("\"weights\":");
            if (weights_pos != std::string::npos) {
                // Parse each weight: "feature_name": value
                std::vector<std::string> feature_names = {
                    "entry_ofi", "ofi_slope", "microprice_dev", "momentum_3", "momentum_10",
                    "book_pressure_top3", "queue_imbalance", "queue_replenishment_rate",
                    "vol_rank_100", "spread_rank_20", "entry_volatility", "entry_spread",
                    "ofi_vol_interaction", "momentum_spread_interaction", "microprice_book_interaction"
                };
                
                for (const auto& feat : feature_names) {
                    std::string search_str = "\"" + feat + "\":";
                    size_t feat_pos = content.find(search_str, weights_pos);
                    if (feat_pos != std::string::npos) {
                        feat_pos += search_str.length();
                        size_t start = content.find_first_of("0123456789.-", feat_pos);
                        size_t end = content.find_first_not_of("0123456789.-", start);
                        if (start != std::string::npos && end != std::string::npos) {
                            model.weights[feat] = std::stod(content.substr(start, end - start));
                        }
                    }
                }
            }
            
            // Parse threshold
            size_t threshold_pos = content.find("\"threshold\":");
            if (threshold_pos != std::string::npos) {
                threshold_pos = content.find_first_of("0123456789.", threshold_pos);
                size_t end = content.find_first_not_of("0123456789.", threshold_pos);
                model.threshold = std::stod(content.substr(threshold_pos, end - threshold_pos));
            }
            
            // Parse feature_order
            size_t feature_order_pos = content.find("\"feature_order\":[");
            if (feature_order_pos != std::string::npos) {
                size_t start = feature_order_pos + 16; // Skip "feature_order":[
                size_t end = content.find("]", start);
                if (end != std::string::npos) {
                    std::string order_str = content.substr(start, end - start);
                    // Simple parsing: split by ","
                    size_t pos = 0;
                    while (pos < order_str.length()) {
                        size_t quote_start = order_str.find("\"", pos);
                        if (quote_start == std::string::npos) break;
                        size_t quote_end = order_str.find("\"", quote_start + 1);
                        if (quote_end == std::string::npos) break;
                        std::string feat = order_str.substr(quote_start + 1, quote_end - quote_start - 1);
                        model.feature_order.push_back(feat);
                        pos = quote_end + 1;
                    }
                }
            }
            
            model.loaded = true;
            return true;
        } catch (...) {
            return false;
        }
    }
    
    bool load_meta_model(const std::string& filename, MetaModel& model) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        try {
            size_t intercept_pos = content.find("\"intercept\":");
            if (intercept_pos != std::string::npos) {
                intercept_pos = content.find_first_of("0123456789.-", intercept_pos);
                size_t end = content.find_first_not_of("0123456789.-", intercept_pos);
                model.intercept = std::stod(content.substr(intercept_pos, end - intercept_pos));
            }
            
            model.coefficients = parse_json_array_simple(content, "\"coef\":[");
            model.scaler_mean = parse_json_array_simple(content, "\"mean\":[");
            model.scaler_scale = parse_json_array_simple(content, "\"scale\":[");
            
            size_t threshold_pos = content.find("\"threshold\":");
            if (threshold_pos != std::string::npos) {
                threshold_pos = content.find_first_of("0123456789.", threshold_pos);
                size_t end = content.find_first_not_of("0123456789.", threshold_pos);
                model.threshold = std::stod(content.substr(threshold_pos, end - threshold_pos));
            }
            
            model.loaded = true;
            return true;
        } catch (...) {
            return false;
        }
    }
    bool load_ml_weights(const std::string& long_filename) {
        // NEW: Try to load advanced per-regime models first
        if (load_advanced_models("models")) {
            std::cout << "\n✅ Loaded advanced per-regime models!" << std::endl;
            return true;
        }
        
        // Fallback to old single model loading
        std::cout << "📊 Loading LONG-side ML weights from: " << long_filename << std::endl;
        double long_threshold = 0.6;
        if (!load_single_ml_model(long_filename, ml_model_long, long_threshold)) {
            std::cerr << "❌ Failed to load LONG ML weights" << std::endl;
            return false;
        }
        thresholds.toxic_cancel_threshold = long_threshold;
        std::cout << "✅ Loaded LONG ML weights! Threshold: " << long_threshold << std::endl;
        
        std::cout << "\n📊 Loading SHORT-side ML weights from: ml_weights_shorts.json" << std::endl;
        double short_threshold = 0.58;
        if (!load_single_ml_model("ml_weights_shorts.json", ml_model_short, short_threshold)) {
            std::cerr << "⚠️ Failed to load SHORT ML weights, using defaults" << std::endl;
            // Use mirrored long model as fallback
            ml_model_short = ml_model_long;
            ml_model_short.qi_ask_coef = -0.5;  // Short-specific adjustment
        } else {
            // Use trained threshold from JSON (0.3801) - trained on real short trade outcomes
            thresholds.short_tox_threshold = short_threshold;
            std::cout << "✅ Loaded SHORT ML weights! Using TRAINED threshold: " << short_threshold << " (from real trade data)" << std::endl;
        }
        
        // Load Dynamic TP Model
        std::cout << "\n📊 Loading Dynamic TP Model from: dynamic_tp_model.json" << std::endl;
        if (load_dynamic_tp_model("dynamic_tp_model.json")) {
            thresholds.use_dynamic_tp = true;
            thresholds.dynamic_tp_threshold = dynamic_tp_model.threshold;
            std::cout << "✅ Loaded Dynamic TP Model! Threshold: " << dynamic_tp_model.threshold << std::endl;
        } else {
            std::cout << "⚠️ Failed to load Dynamic TP Model, using fixed TP=1.0 tick" << std::endl;
            thresholds.use_dynamic_tp = false;
        }
        
        // Load Timeout Prevention Model
        std::cout << "\n📊 Loading Timeout Prevention Model from: timeout_prevention_model.json" << std::endl;
        if (load_timeout_prevention_model("timeout_prevention_model.json")) {
            std::cout << "✅ Loaded Timeout Prevention Model! Threshold: " << timeout_prevention_model.threshold << std::endl;
        } else {
            std::cout << "⚠️ Failed to load Timeout Prevention Model, using defaults" << std::endl;
        }

        // ═══════ NEW: Load MBO-trained models from model_export/ ═══════
        load_mbo_models("model_export");

        return true;
    }

    // ═══════════════════════════════════════════════════════════════
    //  GENERIC JSON ARRAY PARSER (handles nested arrays too)
    // ═══════════════════════════════════════════════════════════════
    static std::vector<double> parse_json_double_array(const std::string& content, const std::string& key) {
        std::vector<double> result;
        std::string search = "\"" + key + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return result;
        pos = content.find('[', pos);
        if (pos == std::string::npos) return result;
        pos++; // skip '['
        // find matching ']', handling nested arrays
        int depth = 1;
        size_t end = pos;
        while (end < content.size() && depth > 0) {
            if (content[end] == '[') depth++;
            else if (content[end] == ']') depth--;
            end++;
        }
        std::string arr = content.substr(pos, end - pos - 1);
        // extract numbers (skip nested brackets)
        std::string num;
        for (char c : arr) {
            if (c == ',' || c == ']' || c == '[') {
                if (!num.empty()) {
                    try { result.push_back(std::stod(num)); } catch (...) {}
                    num.clear();
                }
            } else if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+') {
                num += c;
            }
        }
        if (!num.empty()) {
            try { result.push_back(std::stod(num)); } catch (...) {}
        }
        return result;
    }

    static std::vector<std::string> parse_json_string_array(const std::string& content, const std::string& key) {
        std::vector<std::string> result;
        std::string search = "\"" + key + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return result;
        pos = content.find('[', pos);
        if (pos == std::string::npos) return result;
        size_t end = content.find(']', pos);
        if (end == std::string::npos) return result;
        std::string arr = content.substr(pos + 1, end - pos - 1);
        size_t i = 0;
        while (i < arr.size()) {
            size_t q1 = arr.find('"', i);
            if (q1 == std::string::npos) break;
            size_t q2 = arr.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            result.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
            i = q2 + 1;
        }
        return result;
    }

    static double parse_json_double(const std::string& content, const std::string& key, double fallback = 0.0) {
        std::string search = "\"" + key + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return fallback;
        pos += search.size();
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        size_t end = content.find_first_of(",}\n", pos);
        try { return std::stod(content.substr(pos, end - pos)); } catch (...) { return fallback; }
    }

    static int parse_json_int(const std::string& content, const std::string& key, int fallback = 0) {
        return (int)parse_json_double(content, key, (double)fallback);
    }

    // ═══════════════════════════════════════════════════════════════
    //  LOAD MBO TOXICITY MODEL
    // ═══════════════════════════════════════════════════════════════
    bool load_mbo_tox_model(const std::string& model_file, const std::string& cal_file, MBOToxicityModel& model) {
        std::ifstream f(model_file);
        if (!f.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();

        model.feature_names = parse_json_string_array(content, "features");
        model.weights = parse_json_double_array(content, "weights");
        model.feat_mean = parse_json_double_array(content, "mean");
        model.feat_var = parse_json_double_array(content, "var");
        model.n_features = parse_json_int(content, "n_features", (int)model.feature_names.size());
        model.target = "";  // not critical for C++

        if (model.weights.empty() || model.feat_mean.empty() || model.feat_var.empty()) {
            return false;
        }

        // weights[0] = intercept, weights[1..N] = feature weights
        // if weights.size() == n_features, the export didn't separate intercept -- handle both
        if ((int)model.weights.size() == model.n_features) {
            model.weights.insert(model.weights.begin(), 0.0);
        }

        model.loaded = true;

        // load isotonic calibration
        std::ifstream cf(cal_file);
        if (cf.is_open()) {
            std::string cal_content((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
            cf.close();
            model.iso_x = parse_json_double_array(cal_content, "X_thresholds");
            model.iso_y = parse_json_double_array(cal_content, "y_thresholds");
            model.has_calibration = !model.iso_x.empty() && !model.iso_y.empty();
        }

        return true;
    }

    // ═══════════════════════════════════════════════════════════════
    //  LOAD MARKOV REGIME MODEL
    // ═══════════════════════════════════════════════════════════════
    bool load_markov_model(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();

        markov_regime.n_regimes = parse_json_int(content, "n_regimes", 3);
        markov_regime.n_states = parse_json_int(content, "n_states", 9);
        markov_regime.n_outcomes = parse_json_int(content, "n_outcomes", 11);
        markov_regime.outcome_values = parse_json_double_array(content, "outcome_values");
        markov_regime.vol_thresholds = parse_json_double_array(content, "vol_thresholds");
        markov_regime.probs = parse_json_double_array(content, "probs");

        int expected_size = markov_regime.n_regimes * markov_regime.n_states * markov_regime.n_outcomes;
        if ((int)markov_regime.probs.size() == expected_size &&
            (int)markov_regime.outcome_values.size() == markov_regime.n_outcomes) {
            markov_regime.loaded = true;
        }

        return markov_regime.loaded;
    }

    // ═══════════════════════════════════════════════════════════════
    //  LOAD SPREAD OPTIMIZER
    // ═══════════════════════════════════════════════════════════════
    bool load_spread_optimizer_model(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();

        spread_optimizer.n_vol_bins = parse_json_int(content, "n_vol_bins", 3);
        spread_optimizer.n_tox_bins = parse_json_int(content, "n_tox_bins", 10);
        spread_optimizer.n_gex_bins = parse_json_int(content, "n_gex_bins", 1);
        spread_optimizer.has_gex = (content.find("\"has_gex\": true") != std::string::npos ||
                                    content.find("\"has_gex\":true") != std::string::npos);

        // parse coefficients block
        size_t coef_pos = content.find("\"coefficients\":");
        if (coef_pos != std::string::npos) {
            size_t coef_end = content.find("}", coef_pos + 15);
            std::string coef_block = content.substr(coef_pos, coef_end - coef_pos + 1);
            spread_optimizer.coef_intercept = parse_json_double(coef_block, "intercept", 1.0);
            spread_optimizer.coef_abs_toxicity = parse_json_double(coef_block, "abs_toxicity", 0.0);
            spread_optimizer.coef_vol_regime = parse_json_double(coef_block, "vol_regime", 0.0);
            spread_optimizer.coef_spread_ticks = parse_json_double(coef_block, "spread_ticks", 0.0);
            spread_optimizer.coef_gex_z = parse_json_double(coef_block, "gex_z", 0.0);
            spread_optimizer.coef_tox_x_vol = parse_json_double(coef_block, "tox_x_vol", 0.0);
        }

        // parse table entries: "v_t_g": value
        auto parse_table = [&](const std::string& table_key, std::map<std::string, double>& out) {
            std::string search = "\"" + table_key + "\":";
            size_t tpos = content.find(search);
            if (tpos == std::string::npos) return;
            size_t brace = content.find('{', tpos);
            if (brace == std::string::npos) return;
            size_t end_brace = content.find('}', brace);
            if (end_brace == std::string::npos) return;
            std::string block = content.substr(brace + 1, end_brace - brace - 1);
            size_t i = 0;
            while (i < block.size()) {
                size_t q1 = block.find('"', i);
                if (q1 == std::string::npos) break;
                size_t q2 = block.find('"', q1 + 1);
                if (q2 == std::string::npos) break;
                std::string k = block.substr(q1 + 1, q2 - q1 - 1);
                size_t colon = block.find(':', q2);
                if (colon == std::string::npos) break;
                size_t vstart = colon + 1;
                while (vstart < block.size() && block[vstart] == ' ') vstart++;
                size_t vend = block.find_first_of(",}", vstart);
                if (vend == std::string::npos) vend = block.size();
                try { out[k] = std::stod(block.substr(vstart, vend - vstart)); } catch (...) {}
                i = vend + 1;
            }
        };

        parse_table("table", spread_optimizer.table);
        parse_table("skew_table", spread_optimizer.skew_table);

        spread_optimizer.loaded = !spread_optimizer.table.empty();
        return spread_optimizer.loaded;
    }

    // ═══════════════════════════════════════════════════════════════
    //  MASTER LOADER FOR ALL MBO-DERIVED MODELS
    // ═══════════════════════════════════════════════════════════════
    void load_mbo_models(const std::string& dir) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Loading MBO-trained models from: " << dir << std::endl;
        std::cout << "========================================" << std::endl;

        // Toxicity models (multi-horizon)
        struct ToxTarget {
            const char* label;
            const char* model_suffix;
            const char* cal_suffix;
            MBOToxicityModel& dest;
        };
        ToxTarget targets[] = {
            {"fast",   "toxicity_fast.json",   "calibration_fast.json",   mbo_tox_fast},
            {"medium", "toxicity_medium.json", "calibration_medium.json", mbo_tox_medium},
            {"slow",   "toxicity_slow.json",   "calibration_slow.json",   mbo_tox_slow},
            {"macro",  "toxicity_macro.json",  "calibration_macro.json",  mbo_tox_macro},
        };
        // also try non-horizon names (legacy single-model export)
        bool any_loaded = false;
        for (auto& t : targets) {
            std::string mf = dir + "/" + t.model_suffix;
            std::string cf = dir + "/" + t.cal_suffix;
            if (load_mbo_tox_model(mf, cf, t.dest)) {
                std::cout << "  [OK] " << t.label << " toxicity: " << t.dest.n_features
                          << " features, cal=" << (t.dest.has_calibration ? "yes" : "no") << std::endl;
                any_loaded = true;
            }
        }
        // fallback: try toxicity_model.json (legacy single export)
        if (!mbo_tox_slow.loaded) {
            std::string mf = dir + "/toxicity_model.json";
            std::string cf = dir + "/isotonic_calibration.json";
            if (load_mbo_tox_model(mf, cf, mbo_tox_slow)) {
                std::cout << "  [OK] slow toxicity (legacy): " << mbo_tox_slow.n_features
                          << " features, cal=" << (mbo_tox_slow.has_calibration ? "yes" : "no") << std::endl;
                any_loaded = true;
            }
        }
        if (!any_loaded) {
            std::cout << "  [--] No MBO toxicity models found, using old MLModel" << std::endl;
        }

        // Markov regime model
        std::string markov_file = dir + "/markov_model.json";
        if (load_markov_model(markov_file)) {
            std::cout << "  [OK] Markov regime: " << markov_regime.n_regimes << " regimes, "
                      << markov_regime.n_states << " states, "
                      << markov_regime.vol_thresholds.size() << " vol thresholds" << std::endl;
        } else {
            std::cout << "  [--] No Markov model found, using hardcoded regimes" << std::endl;
        }

        // Spread optimizer
        std::string spread_file = dir + "/spread_optimizer.json";
        if (load_spread_optimizer_model(spread_file)) {
            std::cout << "  [OK] Spread optimizer: " << spread_optimizer.table.size()
                      << " table entries, regression loaded" << std::endl;
        } else {
            std::cout << "  [--] No spread optimizer found, using default spreads" << std::endl;
        }

        // If Markov model loaded, adjust regime thresholds based on vol regime uncertainty
        if (markov_regime.loaded) {
            // High-vol regime (2): more conservative toxicity threshold
            // Low-vol regime (0): can be more aggressive
            double low_vol_unc = markov_regime.uncertainty(0, 4);   // state 4 = mid-state
            double high_vol_unc = markov_regime.uncertainty(2, 4);
            double ratio = (low_vol_unc > 0.01) ? high_vol_unc / low_vol_unc : 2.0;
            // In high-vol, cancel more aggressively (lower threshold)
            double base_tox = 0.5;
            regime_thresholds[MarketRegime::BALANCED].toxicity_threshold = base_tox;
            regime_thresholds[MarketRegime::VOLATILE].toxicity_threshold = std::max(0.3, base_tox / ratio);
            regime_thresholds[MarketRegime::CHAOTIC].toxicity_threshold = std::max(0.25, base_tox / (ratio * 1.2));
            std::cout << "  [OK] Markov-adjusted thresholds: low-vol=" << base_tox
                      << " high-vol=" << regime_thresholds[MarketRegime::VOLATILE].toxicity_threshold << std::endl;
        }

        std::cout << "========================================\n" << std::endl;
    }
    
    bool load_timeout_prevention_model(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // Parse JSON (simple parsing) - find coefficients block first
        size_t coeff_block_start = content.find("\"coefficients\":");
        if (coeff_block_start != std::string::npos) {
            coeff_block_start = content.find("{", coeff_block_start);
            if (coeff_block_start == std::string::npos) {
                coeff_block_start = 0;
            }
        } else {
            coeff_block_start = 0;
        }
        size_t coeff_block_end = content.find("}", coeff_block_start);
        if (coeff_block_end == std::string::npos) {
            coeff_block_end = content.length();
        }
        
        // Parse intercept (must be within coefficients block)
        size_t intercept_pos = content.find("\"intercept\":", coeff_block_start);
        if (intercept_pos != std::string::npos && intercept_pos < coeff_block_end) {
            intercept_pos += 12;
            // Skip whitespace
            while (intercept_pos < content.length() && (content[intercept_pos] == ' ' || content[intercept_pos] == '\t' || content[intercept_pos] == '\n')) {
                intercept_pos++;
            }
            size_t end = content.find_first_of(",}\n", intercept_pos);
            if (end == std::string::npos || end > coeff_block_end) {
                end = coeff_block_end;
            }
            if (end > intercept_pos) {
                std::string intercept_str = content.substr(intercept_pos, end - intercept_pos);
                // Remove whitespace
                intercept_str.erase(std::remove_if(intercept_str.begin(), intercept_str.end(), 
                    [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }), intercept_str.end());
                if (!intercept_str.empty()) {
                    try {
                        timeout_prevention_model.intercept = std::stod(intercept_str);
                    } catch (...) {
                        // Ignore parse errors, use default
                    }
                }
            }
        }
        
        // Parse coefficients array (must be within coefficients block)
        size_t coef_pos = content.find("\"coef\":", coeff_block_start);
        if (coef_pos != std::string::npos && coef_pos < coeff_block_end) {
            coef_pos = content.find("[", coef_pos);
            if (coef_pos != std::string::npos) {
                coef_pos += 1;  // Skip '['
            std::vector<double> coefs;
            size_t start = coef_pos;
            for (int i = 0; i < 6; ++i) {
                // Skip whitespace and commas
                while (start < content.length() && (content[start] == ' ' || content[start] == '\t' || content[start] == '\n' || content[start] == '\r' || content[start] == ',')) {
                    start++;
                }
                if (start >= content.length()) break;
                
                size_t num_start = content.find_first_of("-0123456789.", start);
                if (num_start == std::string::npos) break;
                size_t num_end = content.find_first_of(",]\n\r", num_start);
                if (num_end == std::string::npos) break;
                
                std::string num_str = content.substr(num_start, num_end - num_start);
                // Remove whitespace
                num_str.erase(std::remove_if(num_str.begin(), num_str.end(), 
                    [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }), num_str.end());
                if (!num_str.empty()) {
                    try {
                        coefs.push_back(std::stod(num_str));
                    } catch (...) {
                        break;  // Stop parsing if we hit an invalid number
                    }
                }
                start = num_end + 1;
            }
            if (coefs.size() >= 6) {
                timeout_prevention_model.coef_ofi = coefs[0];
                timeout_prevention_model.coef_momentum = coefs[1];
                timeout_prevention_model.coef_spread = coefs[2];
                timeout_prevention_model.coef_volatility = coefs[3];
                timeout_prevention_model.coef_qi_bid = coefs[4];
                timeout_prevention_model.coef_qi_ask = coefs[5];
            }
            }  // Close inner if (coef_pos != std::string::npos)
        }  // Close outer if (coef_pos != std::string::npos && coef_pos < coeff_block_end)
        
        // Parse scaler means (must be within scaler block)
        size_t scaler_block_start = content.find("\"scaler\":");
        if (scaler_block_start != std::string::npos) {
            scaler_block_start = content.find("{", scaler_block_start);
            if (scaler_block_start == std::string::npos) {
                scaler_block_start = 0;
            }
        } else {
            scaler_block_start = 0;
        }
        size_t scaler_block_end = content.find("}", scaler_block_start);
        if (scaler_block_end == std::string::npos) {
            scaler_block_end = content.length();
        }
        
        size_t mean_pos = content.find("\"mean\":", scaler_block_start);
        if (mean_pos != std::string::npos && mean_pos < scaler_block_end) {
            mean_pos = content.find("[", mean_pos);
        } else {
            mean_pos = std::string::npos;
        }
        if (mean_pos != std::string::npos) {
            mean_pos += 1;  // Skip '['
            std::vector<double> means;
            size_t start = mean_pos;
            for (int i = 0; i < 6; ++i) {
                // Skip whitespace and commas
                while (start < content.length() && (content[start] == ' ' || content[start] == '\t' || content[start] == '\n' || content[start] == '\r' || content[start] == ',')) {
                    start++;
                }
                if (start >= content.length()) break;
                
                size_t num_start = content.find_first_of("-0123456789.", start);
                if (num_start == std::string::npos) break;
                size_t num_end = content.find_first_of(",]\n\r", num_start);
                if (num_end == std::string::npos) break;
                
                std::string num_str = content.substr(num_start, num_end - num_start);
                num_str.erase(std::remove_if(num_str.begin(), num_str.end(), 
                    [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }), num_str.end());
                if (!num_str.empty()) {
                    try {
                        means.push_back(std::stod(num_str));
                    } catch (...) {
                        break;
                    }
                }
                start = num_end + 1;
            }
            if (means.size() >= 6) {
                timeout_prevention_model.ofi_mean = means[0];
                timeout_prevention_model.momentum_mean = means[1];
                timeout_prevention_model.spread_mean = means[2];
                timeout_prevention_model.volatility_mean = means[3];
                timeout_prevention_model.qi_bid_mean = means[4];
                timeout_prevention_model.qi_ask_mean = means[5];
            }
        }
        
        // Parse scaler scales (must be within scaler block)
        size_t scale_pos = content.find("\"scale\":", scaler_block_start);
        if (scale_pos != std::string::npos && scale_pos < scaler_block_end) {
            scale_pos = content.find("[", scale_pos);
        } else {
            scale_pos = std::string::npos;
        }
        if (scale_pos != std::string::npos) {
            scale_pos += 1;  // Skip '['
            std::vector<double> scales;
            size_t start = scale_pos;
            for (int i = 0; i < 6; ++i) {
                // Skip whitespace and commas
                while (start < content.length() && (content[start] == ' ' || content[start] == '\t' || content[start] == '\n' || content[start] == '\r' || content[start] == ',')) {
                    start++;
                }
                if (start >= content.length()) break;
                
                size_t num_start = content.find_first_of("-0123456789.", start);
                if (num_start == std::string::npos) break;
                size_t num_end = content.find_first_of(",]\n\r", num_start);
                if (num_end == std::string::npos) break;
                
                std::string num_str = content.substr(num_start, num_end - num_start);
                num_str.erase(std::remove_if(num_str.begin(), num_str.end(), 
                    [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }), num_str.end());
                if (!num_str.empty()) {
                    try {
                        scales.push_back(std::stod(num_str));
                    } catch (...) {
                        break;
                    }
                }
                start = num_end + 1;
            }
            if (scales.size() >= 6) {
                timeout_prevention_model.ofi_scale = scales[0];
                timeout_prevention_model.momentum_scale = scales[1];
                timeout_prevention_model.spread_scale = scales[2];
                timeout_prevention_model.volatility_scale = scales[3];
                timeout_prevention_model.qi_bid_scale = scales[4];
                timeout_prevention_model.qi_ask_scale = scales[5];
            }
        }
        
        // Parse threshold
        size_t thresh_pos = content.find("\"optimal_threshold\":");
        if (thresh_pos == std::string::npos) {
            thresh_pos = content.find("\"threshold\":");
        }
        if (thresh_pos != std::string::npos) {
            thresh_pos += (content.find("\"optimal_threshold\":") != std::string::npos) ? 19 : 12;
            // Skip whitespace
            while (thresh_pos < content.length() && (content[thresh_pos] == ' ' || content[thresh_pos] == '\t')) {
                thresh_pos++;
            }
            size_t end = content.find_first_of(",}\n", thresh_pos);
            if (end != std::string::npos) {
                std::string thresh_str = content.substr(thresh_pos, end - thresh_pos);
                // Remove trailing whitespace
                while (!thresh_str.empty() && (thresh_str.back() == ' ' || thresh_str.back() == '\t')) {
                    thresh_str.pop_back();
                }
                if (!thresh_str.empty()) {
                    try {
                        timeout_prevention_model.threshold = std::stod(thresh_str);
                    } catch (...) {
                        // Use default if parsing fails
                    }
                }
            }
        }
        
        timeout_prevention_model.loaded = true;
        return true;
    }
    
    double predict_timeout_probability(const RealMarketData& market_data) {
        if (!timeout_prevention_model.loaded) {
            return 0.5;  // Default: neutral if model not loaded
        }
        
        // Calculate features (same as entry time features)
        double ofi = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                  (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        double momentum = 0.0;
        if (mid_prices.size() >= 20) {
            double short_avg = 0.0, long_avg = 0.0;
            auto it = mid_prices.end();
            for (int i = 0; i < 5 && it != mid_prices.begin(); ++i) {
                it--;
                short_avg += *it;
            }
            for (int i = 0; i < 15 && it != mid_prices.begin(); ++i) {
                it--;
                long_avg += *it;
            }
            if (short_avg > 0 && long_avg > 0) {
                momentum = (short_avg / 5.0 - long_avg / 15.0) / 0.25;
            }
        }
        
        double spread = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
        
        double volatility = 0.0;
        if (mid_prices.size() >= 10) {
            double sum = 0.0, sum_sq = 0.0;
            int count = 0;
            for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                sum += *it;
                sum_sq += (*it) * (*it);
                count++;
            }
            if (count > 1) {
                double mean = sum / count;
                volatility = std::sqrt((sum_sq / count) - (mean * mean));
            }
        }
        
        double qi_bid = 0.0, qi_ask = 0.0;
        double total_bid = market_data.bid_sz_00, total_ask = market_data.ask_sz_00;
        for (int i = 0; i < 9; ++i) {
            total_bid += market_data.bid_sz[i];
            total_ask += market_data.ask_sz[i];
        }
        if (total_bid + total_ask > 0) {
            qi_bid = total_bid / (total_bid + total_ask);
            qi_ask = total_ask / (total_bid + total_ask);
        }
        
        // Scale features
        double ofi_scaled = (ofi - timeout_prevention_model.ofi_mean) / timeout_prevention_model.ofi_scale;
        double momentum_scaled = (momentum - timeout_prevention_model.momentum_mean) / timeout_prevention_model.momentum_scale;
        double spread_scaled = (spread - timeout_prevention_model.spread_mean) / timeout_prevention_model.spread_scale;
        double volatility_scaled = (volatility - timeout_prevention_model.volatility_mean) / timeout_prevention_model.volatility_scale;
        double qi_bid_scaled = (qi_bid - timeout_prevention_model.qi_bid_mean) / timeout_prevention_model.qi_bid_scale;
        double qi_ask_scaled = (qi_ask - timeout_prevention_model.qi_ask_mean) / timeout_prevention_model.qi_ask_scale;
        
        // Calculate logit
        double logit = timeout_prevention_model.intercept +
                      timeout_prevention_model.coef_ofi * ofi_scaled +
                      timeout_prevention_model.coef_momentum * momentum_scaled +
                      timeout_prevention_model.coef_spread * spread_scaled +
                      timeout_prevention_model.coef_volatility * volatility_scaled +
                      timeout_prevention_model.coef_qi_bid * qi_bid_scaled +
                      timeout_prevention_model.coef_qi_ask * qi_ask_scaled;
        
        // Sigmoid to get probability
        double prob = 1.0 / (1.0 + std::exp(-logit));
        return prob;
    }
    
    bool load_databento_data(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "❌ Failed to open file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        int line_count = 0;
        int max_records = 20000000;  // 20M records for full month/day run (set high to avoid limits)
        
        while (std::getline(file, line) && databento_data.size() < max_records) {
            line_count++;
            if (line_count % 10000 == 0) {
                std::cout << "  📊 Processing line " << line_count << " (loaded " << databento_data.size() << " ESU5 records)..." << std::endl;
            }
            
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string token;
            
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() < 73) continue;
            
            std::string symbol = tokens[72];
            
            if (symbol != "ESU5") {
                continue;
            }
            
            RealMarketData data;
            
            if (tokens[12].empty() || tokens[12] == "0.0" || 
                tokens[13].empty() || tokens[13] == "0.0") {
                continue;
            }
            
            data.bid_px_00 = std::stod(tokens[12]);
            data.ask_px_00 = std::stod(tokens[13]);
            
            if (data.bid_px_00 < 6000 || data.bid_px_00 > 7000 || 
                data.ask_px_00 < 6000 || data.ask_px_00 > 7000) {
                continue;
            }
            
            double spread = data.ask_px_00 - data.bid_px_00;
            // Data validation: reject invalid spreads (fixes min spread = -66.0 issue)
            if (spread < thresholds.min_spread || spread > thresholds.max_spread) {
                continue;
            }
            
            data.bid_sz_00 = std::stod(tokens[14]);
            data.ask_sz_00 = std::stod(tokens[15]);
            
            // Load depth levels 1-9 from CSV (columns vary by Databento format, but typically sequential)
            // Initialize all to 0 first
            for (int i = 0; i < 9; ++i) {
                data.bid_px[i] = 0.0;
                data.bid_sz[i] = 0.0;
                data.ask_px[i] = 0.0;
                data.ask_sz[i] = 0.0;
            }
            
            // Try to parse depth levels if available (columns 16+)
            // Format: bid_px_01(16), bid_sz_01(17), ask_px_01(18), ask_sz_01(19), then next level...
            for (int level = 0; level < 9 && (16 + level * 4 + 3) < static_cast<int>(tokens.size()); ++level) {
                int base_idx = 16 + (level * 4);
                if (!tokens[base_idx].empty() && tokens[base_idx] != "0.0") {
                    double bid_px_val = std::stod(tokens[base_idx]);
                    // VALIDATION: Only store prices in valid range [6000, 7000]
                    if (bid_px_val >= 6000.0 && bid_px_val <= 7000.0) {
                        data.bid_px[level] = bid_px_val;
                    }
                }
                if (!tokens[base_idx + 1].empty() && tokens[base_idx + 1] != "0.0") {
                    data.bid_sz[level] = std::stod(tokens[base_idx + 1]);
                }
                if (!tokens[base_idx + 2].empty() && tokens[base_idx + 2] != "0.0") {
                    double ask_px_val = std::stod(tokens[base_idx + 2]);
                    // VALIDATION: Only store prices in valid range [6000, 7000]
                    if (ask_px_val >= 6000.0 && ask_px_val <= 7000.0) {
                        data.ask_px[level] = ask_px_val;
                    }
                }
                if (!tokens[base_idx + 3].empty() && tokens[base_idx + 3] != "0.0") {
                    data.ask_sz[level] = std::stod(tokens[base_idx + 3]);
                }
            }
            
            if (!tokens[7].empty() && tokens[7] != "0.0") {
                data.trade_price = std::stod(tokens[7]);
            } else {
                data.trade_price = 0.0;
            }
            
            if (!tokens[8].empty() && tokens[8] != "0.0") {
                data.trade_size = std::stod(tokens[8]);
            } else {
                data.trade_size = 0.0;
            }
            
            if (!tokens[5].empty()) {
                data.trade_side = tokens[5][0];
            } else {
                data.trade_side = 'N';
            }
            
            // Parse timestamp from column 0 (ts_event: "2025-08-17 12:00:05.599208077+00:00")
            // For now, use index-based timestamp, but store original for time calculation
            data.timestamp_ns = databento_data.size() * 1000000000ULL;
            databento_data.push_back(data);
        }
        
        file.close();
        current_data_index = 0;
        
        std::cout << "\n✅ Loaded " << databento_data.size() << " ESU5 records from " << line_count << " total lines" << std::endl;
        return true;
    }
    
    bool get_next_market_data(RealMarketData& market_data) {
        // Safety checks
        if (current_data_index >= databento_data.size()) {
            return false;
        }
        
        if (databento_data.empty()) {
            return false;
        }
        
        // Additional bounds check
        if (current_data_index < 0 || static_cast<size_t>(current_data_index) >= databento_data.size()) {
            std::cerr << "❌ ERROR: Invalid index in get_next_market_data: " << current_data_index 
                      << " (size=" << databento_data.size() << ")" << std::endl;
            return false;
        }
        
        try {
            // Access with bounds check
            market_data = databento_data[static_cast<size_t>(current_data_index)];
            current_data_index++;
            return true;
        } catch (const std::out_of_range& e) {
            std::cerr << "❌ ERROR: out_of_range in get_next_market_data at index " << current_data_index 
                      << ": " << e.what() << std::endl;
            return false;
        } catch (const std::exception& e) {
            std::cerr << "❌ ERROR: Exception in get_next_market_data at index " << current_data_index 
                      << ": " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "❌ ERROR: Unknown exception in get_next_market_data at index " << current_data_index << std::endl;
            return false;
        }
    }
    
    ALWAYS_INLINE HOT void update_market_data(const RealMarketData& market_data) noexcept {
        double mid = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;
        mid_prices.push_back(mid);
        // FIXED: Limit to 100 prices (prevent memory leak on long runs)
        if (mid_prices.size() > 100) {
            mid_prices.pop_front();
        }
        
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            double ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                        (market_data.bid_sz_00 + market_data.ask_sz_00);
            
            ofi_count++;
            double delta = ofi - ofi_mean;
            ofi_mean += delta / ofi_count;
            double delta2 = ofi - ofi_mean;
            ofi_std = std::sqrt(((ofi_count - 1) * ofi_std * ofi_std + delta * delta2) / ofi_count);
        }
    }
    
    void update_dom_levels(const RealMarketData& market_data) {
        // Update DOM levels dynamically from actual market depth data (10 levels: 00-09)
        std::vector<std::pair<double, double>> bid_levels_data = {
            {market_data.bid_px_00, market_data.bid_sz_00}
        };
        std::vector<std::pair<double, double>> ask_levels_data = {
            {market_data.ask_px_00, market_data.ask_sz_00}
        };
        
        // Add depth levels 1-9 from actual data
        for (int i = 0; i < 9; ++i) {
            // VALIDATION: Only use prices in valid range [6000, 7000]
            if (market_data.bid_px[i] > 0.0 && market_data.bid_px[i] >= 6000.0 && market_data.bid_px[i] <= 7000.0 && market_data.bid_sz[i] > 0.0) {
                bid_levels_data.push_back({market_data.bid_px[i], market_data.bid_sz[i]});
            }
            if (market_data.ask_px[i] > 0.0 && market_data.ask_px[i] >= 6000.0 && market_data.ask_px[i] <= 7000.0 && market_data.ask_sz[i] > 0.0) {
                ask_levels_data.push_back({market_data.ask_px[i], market_data.ask_sz[i]});
            }
        }
        
        // Update or create DOM levels for bid side
        for (const auto& [price, size] : bid_levels_data) {
            auto it = bid_levels.find(price);
            if (it != bid_levels.end()) {
                // Update total_size from market data (preserve our_size separately)
                // total_size represents market depth ONLY (not including our orders)
                it->second.total_size = static_cast<int>(size);
            } else {
                // Create new level
                DOMLevel level_data;
                level_data.price = price;
                level_data.total_size = static_cast<int>(size);
                level_data.our_size = 0;
                level_data.queue_depth = 0;
                level_data.volume_traded_at_level = 0;
                level_data.is_at_best_level = false;
                level_data.last_best_price = 0.0;
                bid_levels[price] = level_data;
            }
        }
        
        // Update or create DOM levels for ask side
        for (const auto& [price, size] : ask_levels_data) {
            auto it = ask_levels.find(price);
            if (it != ask_levels.end()) {
                // Update total_size from market data (preserve our_size separately)
                // total_size represents market depth ONLY (not including our orders)
                it->second.total_size = static_cast<int>(size);
            } else {
                DOMLevel level_data;
                level_data.price = price;
                level_data.total_size = static_cast<int>(size);
                level_data.our_size = 0;
                level_data.queue_depth = 0;
                level_data.volume_traded_at_level = 0;
                level_data.is_at_best_level = false;
                level_data.last_best_price = 0.0;
                ask_levels[price] = level_data;
            }
        }
        
        // Clean up levels that no longer exist in market (only if we have no orders)
        auto bid_it = bid_levels.begin();
        while (bid_it != bid_levels.end()) {
            if (bid_it->second.our_size == 0) {
                bool found = false;
                for (const auto& [p, s] : bid_levels_data) {
                    if (std::abs(p - bid_it->first) < 0.01) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    bid_it = bid_levels.erase(bid_it);
                } else {
                    ++bid_it;
                }
            } else {
                ++bid_it;
            }
        }
        
        auto ask_it = ask_levels.begin();
        while (ask_it != ask_levels.end()) {
            if (ask_it->second.our_size == 0) {
                bool found = false;
                for (const auto& [p, s] : ask_levels_data) {
                    if (std::abs(p - ask_it->first) < 0.01) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    ask_it = ask_levels.erase(ask_it);
                } else {
                    ++ask_it;
                }
            } else {
                ++ask_it;
            }
        }
    }
    void place_order_at_level(double price, int contracts, bool is_bid, const RealMarketData& market_data) {
        // REALISTIC: Order rejection checks (risk limits, invalid prices, system issues)
        // 1. Invalid price check (outside valid range)
        if (price < 6000.0 || price > 7000.0) {
            return;  // Reject: invalid price
        }
        
        // 2. Invalid spread check (market data corruption)
        double spread = market_data.ask_px_00 - market_data.bid_px_00;
        if (spread < 0.0 || spread > 5.0) {
            return;  // Reject: invalid spread (market data issue)
        }
        
        // 3. Price too far from market (risk limit - prevent fat finger)
        double best_price = is_bid ? market_data.bid_px_00 : market_data.ask_px_00;
        double price_distance = std::abs(price - best_price) / 0.25;  // In ticks
        if (price_distance > 20.0) {  // More than 20 ticks away
            return;  // Reject: too far from market (risk protection)
        }
        
        // 4. System issue simulation (0.01% chance of rejection)
        if (uniform_dist(rng) < 0.0001) {
            return;  // Reject: simulated exchange system issue
        }
        
        Order order;
        order.order_id = next_order_id++;
        order.price = price;
        order.contracts = contracts;
        order.timestamp_ns = market_data.timestamp_ns;
        order.is_bid = is_bid;
        order.is_active = true;
        order.is_exit_order = false;  // Entry orders
        order.position_index = 0;  // Not used for entry orders
        
        if (is_bid) {
            auto& level = bid_levels[price];
            double market_depth = level.total_size;
            atomic_add(metrics.bid_level_depth_sum, market_depth);
            metrics.bid_level_depth_samples.fetch_add(1, std::memory_order_relaxed);
            // Only increment our_size; total_size is already set from market data
            // total_size represents market depth BEFORE our orders
            level.our_size += contracts;
            level.queue_depth++;
            // REALISTIC QUEUE POSITION: New orders go at the END of the queue
            // If existing orders are at position 20, new orders start at position 90 (behind them)
            // Calculate: max existing queue_position + our_size OR current market depth, whichever is larger
            int max_existing_position = level.total_size;  // Default: current market depth
            if (!level.order_queue.empty()) {
                // Find max queue position of existing orders
                std::queue<Order> temp_check = level.order_queue;
                while (!temp_check.empty()) {
                    Order q_order = temp_check.front();
                    temp_check.pop();
                    if (our_orders[q_order.order_id].is_active) {
                        max_existing_position = std::max(max_existing_position, q_order.queue_position);
                    }
                }
            }
            // New order starts BEHIND existing orders (preserve FIFO)
            order.initial_market_depth = max_existing_position;
            order.queue_position = max_existing_position;  // Start behind existing orders
            level.order_queue.push(order);
        } else {
            auto& level = ask_levels[price];
            double market_depth = level.total_size;
            atomic_add(metrics.ask_level_depth_sum, market_depth);
            metrics.ask_level_depth_samples.fetch_add(1, std::memory_order_relaxed);
            // Only increment our_size; total_size is already set from market data
            // total_size represents market depth BEFORE our orders
            level.our_size += contracts;
            level.queue_depth++;
            // REALISTIC QUEUE POSITION: New orders go at the END of the queue
            // If existing orders are at position 20, new orders start at position 90 (behind them)
            // Calculate: max existing queue_position + our_size OR current market depth, whichever is larger
            int max_existing_position = level.total_size;  // Default: current market depth
            if (!level.order_queue.empty()) {
                // Find max queue position of existing orders
                std::queue<Order> temp_check = level.order_queue;
                while (!temp_check.empty()) {
                    Order q_order = temp_check.front();
                    temp_check.pop();
                    if (our_orders[q_order.order_id].is_active) {
                        max_existing_position = std::max(max_existing_position, q_order.queue_position);
                    }
                }
            }
            // New order starts BEHIND existing orders (preserve FIFO)
            order.initial_market_depth = max_existing_position;
            order.queue_position = max_existing_position;  // Start behind existing orders
            level.order_queue.push(order);
        }
        
        our_orders[order.order_id] = order;
        metrics.orders_refilled.fetch_add(1, std::memory_order_relaxed);
    }
    
    void maintain_dom_levels(const RealMarketData& market_data) {
        // CIRCUIT BREAKER: DISABLED for market making
        // Market making requires continuous order placement - ML handles canceling toxic orders
        bool is_breaker_active = false;  // Always false - circuit breaker disabled
        
        // Check circuit breaker conditions (DISABLED)
        int consecutive_losses = metrics.consecutive_losses.load();
        int rolling_tox_pct = metrics.rolling_toxicity_percent.load();
        int toxic_regime_count = metrics.toxic_regime_intervals.load();
        
        // Calculate rolling Sharpe
        double rolling_sharpe = 0.0;
        {
            std::lock_guard<std::mutex> lock(metrics.returns_mutex);
            if (metrics.recent_returns.size() > 30) {
                std::vector<double> returns(metrics.recent_returns.begin(), metrics.recent_returns.end());
                double mean = 0.0;
                for (double r : returns) mean += r;
                mean /= returns.size();
                double variance = 0.0;
                for (double r : returns) variance += (r - mean) * (r - mean);
                double std_dev = std::sqrt(variance / (returns.size() - 1));
                if (std_dev > 0.001) {
                    rolling_sharpe = mean / std_dev * std::sqrt(252);
                }
            }
        }
        
        // AUTO-RESET: Check if conditions have improved enough to resume trading
        if (is_breaker_active) {
            int reset_counter = metrics.circuit_breaker_reset_counter.fetch_add(1, std::memory_order_relaxed);
            
            // SIMPLIFIED RESET: Always reset after 1000 intervals (almost immediate)
            // No complex condition checking - just timeout-based reset
            bool timeout_reset = reset_counter >= 1000;  // Reset after 1000 intervals max
            
            if (timeout_reset) {
                metrics.circuit_breaker_active.store(false, std::memory_order_relaxed);
                metrics.circuit_breaker_reset_counter.store(0, std::memory_order_relaxed);
                metrics.toxic_regime_intervals.store(0, std::memory_order_relaxed);
                metrics.consecutive_losses.store(0, std::memory_order_relaxed);  // CRITICAL: Reset consecutive losses!
                
                // Clear ALL stale returns to prevent bad Sharpe from old trades
                {
                    std::lock_guard<std::mutex> lock(metrics.returns_mutex);
                    metrics.recent_returns.clear();
                }
                
                std::cout << "✅✅✅ CIRCUIT BREAKER RESET (TIMEOUT after " << reset_counter << " intervals) ✅✅✅" << std::endl;
                std::cout << "🔄 RESUMING TRADING - tox=" << rolling_tox_pct 
                          << "%, sharpe=" << rolling_sharpe << ", losses=0 (RESET)" << std::endl;
                is_breaker_active = false;  // Continue trading - FORCE RESET
            } else {
                // Log status every 100 intervals while paused (more frequent)
                if (reset_counter % 100 == 0) {
                    int remaining = 1000 - reset_counter;
                    std::cout << "⏸️  Circuit breaker active (interval " << reset_counter 
                              << "/1000): tox=" << rolling_tox_pct 
                              << "%, sharpe=" << rolling_sharpe << ", losses=" << consecutive_losses 
                              << " | Resets in " << remaining << " intervals" << std::endl;
                }
            }
            
            // Double-check: if reset happened, continue; otherwise return
            if (metrics.circuit_breaker_active.load()) {
                return;  // Still paused
            } else {
                // Reset happened - log it
                std::cout << "🟢 TRADING RESUMED - Circuit breaker cleared" << std::endl;
            }
        }
        
        // COOLDOWN: Prevent immediate re-trigger after reset (60 second minimum)
        uint64_t last_trigger_ns = metrics.last_circuit_breaker_trigger_ns.load(std::memory_order_relaxed);
        uint64_t cooldown_period_ns = 60ULL * 1000000000ULL;  // 60 seconds cooldown
        uint64_t current_time_ns = market_data.timestamp_ns;
        bool in_cooldown = (last_trigger_ns > 0) && ((current_time_ns - last_trigger_ns) < cooldown_period_ns);
        
        // CIRCUIT BREAKER: DISABLED for market making
        // Market making means we place orders and cancel toxic ones - circuit breaker interferes with this
        bool trigger_circuit_breaker = false;
        std::string trigger_reason = "";
        
        // COMPLETELY DISABLED: Market making requires continuous order placement
        // ML will handle canceling toxic orders, we don't need circuit breaker
        // if (consecutive_losses >= thresholds.circuit_breaker_consecutive_losses) {
        //     trigger_circuit_breaker = true;
        // }
        // if (rolling_tox_pct > thresholds.circuit_breaker_toxicity_threshold) {
        //     toxic_regime_count++;
        //     if (toxic_regime_count >= thresholds.circuit_breaker_toxic_regime_intervals) {
        //         trigger_circuit_breaker = true;
        //     }
        // }
        
        // DISABLED: Sharpe-based circuit breaker is too volatile - causes false triggers
        // Overall Sharpe can be 2.58 while rolling Sharpe is -7.06
        // if (rolling_sharpe < thresholds.circuit_breaker_sharpe_threshold && metrics.recent_returns.size() > 30) {
        //     trigger_circuit_breaker = true;
        //     std::cout << "🚨 CIRCUIT BREAKER: Rolling Sharpe (" << rolling_sharpe << ") < " 
        //               << thresholds.circuit_breaker_sharpe_threshold << "!" << std::endl;
        // }
        
        // CIRCUIT BREAKER COMPLETELY DISABLED
        // Market making requires continuous order placement - ML handles canceling toxic orders
        // if (trigger_circuit_breaker) {
        //     metrics.circuit_breaker_active.store(true, std::memory_order_relaxed);
        //     metrics.circuit_breaker_reset_counter.store(0, std::memory_order_relaxed);
        //     metrics.last_circuit_breaker_trigger_ns.store(current_time_ns, std::memory_order_relaxed);
        //     std::cout << "🛑 TRADING PAUSED - Circuit breaker activated (" << trigger_reason << ")" << std::endl;
        //     return;  // Don't place any new orders
        // }
        
        // REFILL COOLDOWN: Only block REFILLS after toxic cancels, not initial placement
        // Count active orders to determine if we're refilling or placing initial orders
        int active_bid_orders = 0;
        int active_ask_orders = 0;
        for (const auto& [order_id, order] : our_orders) {
            if (order.is_active) {
                if (order.is_bid) active_bid_orders++;
                else active_ask_orders++;
            }
        }
        
        // Only apply cooldown if we're REFILLING (had orders before, now trying to replace canceled ones)
        bool is_refilling_longs = (active_bid_orders < thresholds.dom_levels / 2) && 
                                   (last_long_toxic_cancel_ns > 0) &&
                                   ((market_data.timestamp_ns - last_long_toxic_cancel_ns) < REFILL_COOLDOWN_NS);
        bool is_refilling_shorts = (active_ask_orders < thresholds.dom_levels / 2) && 
                                    (last_short_toxic_cancel_ns > 0) &&
                                    ((market_data.timestamp_ns - last_short_toxic_cancel_ns) < REFILL_COOLDOWN_NS);
        
        // Check if toxicity has improved before refilling (only check if refilling)
        // DISABLED: Refill cooldown - fill the book immediately
        bool skip_long_refill = false;  // DISABLED: Always refill
        bool skip_short_refill = false;  // DISABLED: Always refill
        // DISABLED: Removed toxicity check for refilling - always refill
        
        // Place orders only at actual market depth levels (not calculated positions)
        // Use up to dom_levels depth levels from actual market data
        
        // SAFETY: Check if LONGs are enabled
        if (!thresholds.enable_longs) {
            // Cancel all active bid orders
            for (auto& [order_id, order] : our_orders) {
                if (order.is_bid && order.is_active) {
                    order.is_active = false;
                    auto& level = bid_levels[order.price];
                    level.our_size -= order.contracts;
                    // Rebuild queue
                    std::queue<Order> new_queue;
                    while (!level.order_queue.empty()) {
                        Order q_order = level.order_queue.front();
                        level.order_queue.pop();
                        if (q_order.order_id != order.order_id) {
                            new_queue.push(q_order);
                        }
                    }
                    level.order_queue = new_queue;
                    level.queue_depth = level.order_queue.size();
                }
            }
            return;  // Don't place new bid orders
        }
        
        // Bid side: use actual market depth levels
        // Risk rail: Check inventory cap (per side)
        int long_positions = 0;
        int short_positions = 0;
        for (const auto& pos : active_positions) {
            if (pos.is_long) long_positions++; else short_positions++;
        }
        
        // CRITICAL: ML is for CANCELING orders, NOT preventing placement!
        // Place orders freely, then check_toxicity_cancellation() will cancel toxic ones
        // Only use basic entry quality gates (momentum, OFI, volatility) to filter EXTREME cases
        
        double current_ofi = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            current_ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        double current_volatility = 0.0;
        if (mid_prices.size() >= 10) {
            // Calculate volatility in ticks (matching the system's volatility calculation)
            std::vector<double> price_changes;
            for (size_t i = mid_prices.size() - std::min(10, (int)mid_prices.size()); i < mid_prices.size() - 1; ++i) {
                double change = std::abs(mid_prices[i+1] - mid_prices[i]) / 0.25;  // In ticks
                price_changes.push_back(change);
            }
            if (price_changes.size() > 1) {
                double mean = 0.0;
                for (double chg : price_changes) mean += chg;
                mean /= price_changes.size();
                double variance = 0.0;
                for (double chg : price_changes) variance += (chg - mean) * (chg - mean);
                current_volatility = std::sqrt(variance / (price_changes.size() - 1));  // Std dev in ticks
            }
        }
        
        // CRITICAL: Momentum gate - Only enter LONGs when momentum is STRONGLY POSITIVE
        // Require momentum > 0.2 ticks to ensure market is actually moving UP
        // This prevents entering at market tops (adverse selection)
        double momentum = 0.0;
        if (mid_prices.size() >= 20) {
            double short_avg = 0.0, long_avg = 0.0;
            auto it = mid_prices.end();
            int short_count = 0, long_count = 0;
            for (int i = 0; i < 5 && it != mid_prices.begin(); ++i) {
                it--;
                short_avg += *it;
                short_count++;
            }
            auto it2 = mid_prices.end();
            for (int i = 0; i < 20 && it2 != mid_prices.begin(); ++i) {
                it2--;
                long_avg += *it2;
                long_count++;
            }
            if (short_count > 0 && long_count > 0) {
                short_avg /= short_count;
                long_avg /= long_count;
                momentum = (short_avg - long_avg) / 0.25;  // Ticks
            }
        }
        
        // NEW: Microprice deviation calculation
        double mid_price = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;
        double microprice = 0.0;
        double microprice_dev = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            microprice = (market_data.ask_px_00 * market_data.bid_sz_00 + 
                         market_data.bid_px_00 * market_data.ask_sz_00) / 
                        (market_data.bid_sz_00 + market_data.ask_sz_00);
            microprice_dev = microprice - mid_price;
        }
        
        // ENTRY GATES: RELAXED for data collection (too strict before)
        // Long gate: Only reject STRONGLY negative momentum/microprice/OFI
        // This allows more trades to pass so we can collect better training data
        
        // O'HARA #4: Enhanced adverse-selection filter (Glosten-Milgrom)
        // DRASTICALLY RELAXED: Debug shows microprice rejecting 16,978 times, OFI rejecting 9,066 times
        // Need to allow more trades through to collect data
        static constexpr double OFI_LONG_CAP = 0.5;  // INCREASED from 0.3 to 0.5 (more lenient)
        bool should_reject_momentum = false;  // DATA COLLECTION MODE: DISABLED - allow all momentum
        bool should_reject_microprice = false;  // DATA COLLECTION MODE: DISABLED - allow all microprice
        bool should_reject_ofi_long = false;  // DATA COLLECTION MODE: DISABLED - allow all OFI
        // O'HARA #5: Queue position check (don't join hopeless queues)
        // This is checked during order placement, but we can gate here too
        static constexpr int QPOS_CAP = 50;  // Don't place if queue position would be > 50
        
        // MINIMAL GATES: Only reject EXTREME cases (extreme volatility)
        // ML will cancel toxic orders after placement via check_toxicity_cancellation()
        bool should_reject_volatility = (current_volatility > 20.0);  // DATA COLLECTION MODE: Only reject extreme volatility (>20)
        
        // Entry filter ML gate: DISABLED (model trained on losers, can't distinguish good from bad)
        // TODO: Re-enable after retraining on better data with positive win rate
        MarketRegime current_regime = detect_regime(market_data);
        double entry_filter_prob = 0.5;  // Default: neutral
        bool should_reject_entry_filter = false;  // DISABLED: Don't use entry filter model until retrained
        
        // O'HARA #6: Inventory-based gate (signal_strength - α·|inv|) > threshold
        // TEMPORARILY DISABLED: Blocking trades
        bool should_reject_inventory = false;
        
        // O'HARA #7: Lambda-based gate (high impact = skip entry)
        // TEMPORARILY DISABLED: Not blocking trades anyway
        bool should_reject_lambda = false;
        
        // ========================================================================
        // KYLE λ (lambda) - BASED TRADE INTENSITY CONTROL
        // ========================================================================
        // Estimate lambda_t: price impact per contract (slope of mid price vs OFI)
        double lambda_t = 0.01;  // Default: moderate impact
        bool lambda_available = false;
        
        // Calculate lambda from recent mid price changes vs OFI changes
        if (mid_prices.size() >= 5 && ofi_history.size() >= 5) {
            // Use last 5 intervals for rolling regression-like estimate
            double delta_mid_sum = 0.0;
            double delta_ofi_sum = 0.0;
            int valid_pairs = 0;
            
            size_t max_pairs = std::min(static_cast<size_t>(5), mid_prices.size());
            for (size_t i = 1; i < max_pairs; ++i) {
                if (i < ofi_history.size()) {
                    double delta_mid = std::abs(mid_prices[mid_prices.size() - i] - mid_prices[mid_prices.size() - i - 1]) / 0.25;  // In ticks
                    double delta_ofi = std::abs(ofi_history[ofi_history.size() - i] - ofi_history[ofi_history.size() - i - 1]);
                    
                    if (delta_ofi > 0.001) {  // Avoid division by zero
                        delta_mid_sum += delta_mid;
                        delta_ofi_sum += delta_ofi;
                        valid_pairs++;
                    }
                }
            }
            
            if (valid_pairs > 0 && delta_ofi_sum > 0.001) {
                // lambda ≈ Δ(mid_price) / Δ(OFI) - price impact per unit of order flow
                lambda_t = (delta_mid_sum / valid_pairs) / (delta_ofi_sum / valid_pairs);
                lambda_t = std::clamp(lambda_t, 0.001, 0.05);  // Clamp to reasonable range
                lambda_available = true;
            }
        }
        
        // Update rolling average lambda
        if (lambda_available) {
            lambda_history.push_back(lambda_t);
            if (lambda_history.size() > LAMBDA_HISTORY_WINDOW) {
                lambda_history.pop_front();
            }
            
            // Calculate rolling average
            double sum_lambda = 0.0;
            for (double l : lambda_history) {
                sum_lambda += l;
            }
            avg_lambda = (lambda_history.size() > 0) ? (sum_lambda / lambda_history.size()) : 0.01;
            avg_lambda = std::max(avg_lambda, 0.001);  // Ensure non-zero
        }
        
        // Calculate intensity factor: signal strength / (lambda_t / avg_lambda)
        // Signal strength = (1 - toxicity_prob) * (1 - timeout_prob)
        double toxicity_prob = mbo_tox_slow.loaded ? predict_toxicity_mbo(market_data, true)
                                                   : predict_toxicity(market_data, true);
        
        // Get timeout probability (for intensity calculation only)
        double timeout_prob = 0.5;  // Default: neutral if model not loaded
        if (timeout_prevention_model.loaded) {
            timeout_prob = predict_timeout_probability(market_data);
        }
        
        // O'HARA #1: Update inventory control
        inventory_control.update_inventory(active_positions);
        
        // O'HARA #2: Calculate signal strength for inventory penalty
        double signal_strength = (1.0 - toxicity_prob) * (1.0 - timeout_prob);
        double inventory_adjusted_signal = signal_strength - inventory_control.get_penalty();
        
        // O'HARA #3: Time-of-day adjusted thresholds
        SessionState session = detect_session_state(market_data.timestamp_ns);
        RegimeThresholds regime_thresh = regime_thresholds[current_regime];
        double time_adjusted_tox_threshold = get_time_adjusted_threshold(regime_thresh.toxicity_threshold, session);
        
        // DEBUG: Track which gates are blocking
        static int gate_reject_count = 0;
        static int momentum_rejects = 0, microprice_rejects = 0, ofi_rejects = 0, vol_rejects = 0, inv_rejects = 0, lambda_rejects = 0;
        if (should_reject_momentum || should_reject_microprice || should_reject_ofi_long || should_reject_volatility || 
            should_reject_inventory || should_reject_lambda) {
            gate_reject_count++;
            if (should_reject_momentum) momentum_rejects++;
            if (should_reject_microprice) microprice_rejects++;
            if (should_reject_ofi_long) ofi_rejects++;
            if (should_reject_volatility) vol_rejects++;
            if (should_reject_inventory) inv_rejects++;
            if (should_reject_lambda) lambda_rejects++;
            
            if (gate_reject_count % 1000 == 0) {
                std::cout << "DEBUG Gates: mom=" << momentum_rejects << ", micro=" << microprice_rejects 
                          << ", ofi=" << ofi_rejects << ", vol=" << vol_rejects 
                          << ", inv=" << inv_rejects << ", lambda=" << lambda_rejects << std::endl;
            }
        }
        
        // Combined rejection: DISABLED - fill the book
        bool should_reject = false;  // DISABLED: Never reject, fill the book
        double lambda_ratio = lambda_t / avg_lambda;
        double intensity_factor = signal_strength / std::max(lambda_ratio, 0.1);  // Avoid division by zero
        
        // Clamp intensity factor to reasonable range [0.1, 2.0]
        intensity_factor = std::clamp(intensity_factor, 0.1, 2.0);
        
        // Scale dom_levels based on intensity factor
        // DISABLED: Always use full dom_levels to fill the book
        int effective_dom_levels = thresholds.dom_levels;  // Always use full depth (was scaled)
        
        // ADAPTIVE QUEUE CAPACITY: Scale max_open_positions based on Kyle λ and signal strength
        // Higher signal strength + lower lambda = more positions allowed
        // Formula: max_positions = base * intensity_factor * (1 / lambda_ratio)
        double position_scaling_factor = intensity_factor * (1.0 / std::max(lambda_ratio, 0.1));
        position_scaling_factor = std::clamp(position_scaling_factor, 0.5, 2.5);  // Scale between 0.5x and 2.5x
        
        int effective_max_positions = static_cast<int>(std::round(thresholds.base_max_open_positions * position_scaling_factor));
        effective_max_positions = std::clamp(effective_max_positions, thresholds.base_max_open_positions, thresholds.max_max_open_positions);
        
        // Update threshold temporarily for this interval
        int original_max_positions = thresholds.max_open_positions_per_side;
        thresholds.max_open_positions_per_side = effective_max_positions;
        
        // DISABLED: Skip entry if lambda is too high - fill the book
        static constexpr double HIGH_LAMBDA_THRESHOLD = 999.0;  // DISABLED: Never skip (was 0.03)
        bool skip_entry_high_lambda = false;  // DISABLED: Always place orders
        
        // Debug output (every 10000 intervals)
        static int kyle_debug_count = 0;
        if (++kyle_debug_count % 10000 == 0) {
            std::cout << "🔬 Kyle λ: lambda_t=" << lambda_t << ", avg_lambda=" << avg_lambda 
                      << ", signal_strength=" << signal_strength
                      << ", intensity_factor=" << intensity_factor 
                      << ", effective_dom_levels=" << effective_dom_levels << std::endl;
        }
        
        // ========================================================================
        
        int bid_levels_placed = 0;
        // SPREAD OPTIMIZER: compute optimal half-spread and skew
        double tox_for_spread = mbo_tox_slow.loaded ? predict_toxicity_mbo(market_data, true)
                                                    : predict_toxicity(market_data, true);
        auto [opt_half_spread, opt_skew] = get_optimal_spread(market_data, tox_for_spread);
        double opt_bid_offset = -(opt_half_spread + opt_skew) * 0.25;  // negative = below mid
        double opt_ask_offset = (opt_half_spread - opt_skew) * 0.25;   // positive = above mid
        double mid_price_for_spread = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;

        // O'HARA #8: Inventory-aware quote skewing and sizing (combined with optimizer)
        double skewed_bid_price = spread_optimizer.loaded
            ? mid_price_for_spread + opt_bid_offset
            : inventory_control.get_skewed_bid_price(market_data.bid_px_00);
        // Snap to tick grid
        skewed_bid_price = std::floor(skewed_bid_price / 0.25) * 0.25;
        int adjusted_size = inventory_control.get_adjusted_size(thresholds.contracts_per_level);
        
        // O'HARA #9: Lambda-scaled sizing (high λ = smaller size)
        if (lambda_available && lambda_t > 0.001) {
            double lambda_size_multiplier = std::min(1.0, avg_lambda / lambda_t);  // Smaller when λ_t is high
            adjusted_size = std::max(1, static_cast<int>(adjusted_size * lambda_size_multiplier));
        }
        
        // Calculate momentum for placement gate
        double placement_momentum = 0.0;
        if (mid_prices.size() >= 2) {
            placement_momentum = (mid_prices.back() - *(std::prev(mid_prices.end(), 2))) / 0.25;
        }
        
        // BLIND PLACEMENT: Place orders on both sides continuously
        // Cancellation will happen FAST when momentum turns against us
        // No placement gates - rely on fast cancellation instead
        bool should_place_long = true;  // Place blindly, cancel fast
        bool should_place_short = true;  // Place blindly, cancel fast
        
        // VALIDATION: Only use prices in valid range [6000, 7000]
        // MOMENTUM GATE: Only place longs when momentum is upward or neutral
        // REGIME-ADAPTIVE: Skip placement if timeout prevention model says reject
        // KYLE λ: Skip if high price impact (low liquidity)
        // REFILL COOLDOWN: Skip if recently canceled for toxicity
        // BLIND PLACEMENT: Place orders continuously - REMOVE has_order check
        // Place at skewed bid price (O'HARA inventory-aware)
        if (skewed_bid_price > 0.0 && skewed_bid_price >= 6000.0 && skewed_bid_price <= 7000.0 && 
            bid_levels_placed < effective_dom_levels &&
            (long_positions < thresholds.max_open_positions_per_side) &&
            should_place_long &&
            !should_reject &&
            !skip_entry_high_lambda &&
            !skip_long_refill) {
            // REMOVED has_order check - place continuously
            place_order_at_level(skewed_bid_price, adjusted_size, true, market_data);
            sample_entry_state(true, market_data);
            bid_levels_placed++;
        }
        // Note: should_place_long is always true now (blind placement)
        // Cancellation handles filtering instead
        
        // BLIND PLACEMENT: Place at ALL bid levels continuously - REMOVE has_order check
        for (int i = 0; i < 9 && bid_levels_placed < effective_dom_levels; ++i) {
            if (market_data.bid_px[i] > 0.0 && market_data.bid_px[i] >= 6000.0 && market_data.bid_px[i] <= 7000.0 &&
                (long_positions < thresholds.max_open_positions_per_side) &&
                should_place_long &&
                !should_reject &&
                !skip_entry_high_lambda &&
                !skip_long_refill) {
                // REMOVED has_order check - place continuously at every level
                place_order_at_level(market_data.bid_px[i], thresholds.contracts_per_level, true, market_data);
                sample_entry_state(true, market_data);
                bid_levels_placed++;
            }
        }
        
        // PREDICTIVE QUEUE POSITIONING: Place orders at predicted price levels ahead of movement
        if (price_predictor.should_place_predictive_orders(true) && bid_levels_placed < effective_dom_levels) {
            std::vector<double> predicted_levels = price_predictor.predict_target_levels(market_data, true, 2);  // Predict 2 levels ahead
            for (double pred_price : predicted_levels) {
                if (pred_price >= 6000.0 && pred_price <= 7000.0 && 
                    bid_levels_placed < effective_dom_levels &&
                    (long_positions < thresholds.max_open_positions_per_side) &&
                    !skip_entry_high_lambda && !skip_long_refill) {
                    // Check if we already have an order at this predicted level
                    bool has_order = false;
                    for (const auto& [order_id, order] : our_orders) {
                        if (std::abs(order.price - pred_price) < 0.01 && order.is_bid && order.is_active) {
                            has_order = true;
                            break;
                        }
                    }
                    if (!has_order) {
                        place_order_at_level(pred_price, thresholds.contracts_per_level / 2, true, market_data);  // Smaller size for predictive
                        bid_levels_placed++;
                        // Debug output (first few times)
                        static int pred_debug_count = 0;
                        if (++pred_debug_count <= 5) {
                            std::cout << "🎯 PREDICTIVE PLACEMENT (LONG): Placed at " << pred_price 
                                      << " (predicted " << ((pred_price - market_data.bid_px_00) / 0.25) << " ticks ahead)" << std::endl;
                        }
                    }
                }
            }
        }
        
        // Ask side: use actual market depth levels
        // SAFETY: Check if SHORTs are enabled
        if (!thresholds.enable_shorts) {
            // Cancel all active ask orders
            for (auto& [order_id, order] : our_orders) {
                if (!order.is_bid && order.is_active) {
                    order.is_active = false;
                    auto& level = ask_levels[order.price];
                    level.our_size -= order.contracts;
                    // Rebuild queue
                    std::queue<Order> new_queue;
                    while (!level.order_queue.empty()) {
                        Order q_order = level.order_queue.front();
                        level.order_queue.pop();
                        if (q_order.order_id != order.order_id) {
                            new_queue.push(q_order);
                        }
                    }
                    level.order_queue = new_queue;
                    level.queue_depth = level.order_queue.size();
                }
            }
            return;  // Don't place new ask orders
        }
        
        int ask_levels_placed = 0;
        // Spread-optimizer-aware ask placement
        double optimal_ask_price = spread_optimizer.loaded
            ? std::ceil((mid_price_for_spread + opt_ask_offset) / 0.25) * 0.25
            : market_data.ask_px_00;
        if (optimal_ask_price > 0.0 && optimal_ask_price >= 6000.0 && optimal_ask_price <= 7000.0 &&
            ask_levels_placed < effective_dom_levels &&
            (short_positions < thresholds.max_open_positions_per_side) &&
            !should_reject &&
            !skip_entry_high_lambda &&
            !skip_short_refill) {
            place_order_at_level(optimal_ask_price, thresholds.contracts_per_level, false, market_data);
            sample_entry_state(false, market_data);
            metrics.ask_placements.fetch_add(1, std::memory_order_relaxed);
            ask_levels_placed++;
        }
        
        for (int i = 0; i < 9 && ask_levels_placed < effective_dom_levels; ++i) {
            if (market_data.ask_px[i] > 0.0 && market_data.ask_px[i] >= 6000.0 && market_data.ask_px[i] <= 7000.0 &&
                (short_positions < thresholds.max_open_positions_per_side) &&
                !should_reject &&
                !skip_short_refill) {
                // Gate check disabled - using blind placement + ML cancellation
                // if (!side_ok_short(market_data)) {
                //     metrics.ask_skipped_by_gate.fetch_add(1, std::memory_order_relaxed);
                //     continue;
                // }
                place_order_at_level(market_data.ask_px[i], thresholds.contracts_per_level, false, market_data);
                sample_entry_state(false, market_data);
                metrics.ask_placements.fetch_add(1, std::memory_order_relaxed);
                ask_levels_placed++;
            }
        }
        // PREDICTIVE QUEUE POSITIONING: Place orders at predicted price levels ahead of movement (SHORTS)
        if (price_predictor.should_place_predictive_orders(false) && ask_levels_placed < effective_dom_levels) {
            std::vector<double> predicted_levels = price_predictor.predict_target_levels(market_data, false, 2);  // Predict 2 levels ahead
            for (double pred_price : predicted_levels) {
                if (pred_price >= 6000.0 && pred_price <= 7000.0 && 
                    ask_levels_placed < effective_dom_levels &&
                    (short_positions < thresholds.max_open_positions_per_side) &&
                    !skip_entry_high_lambda && !skip_short_refill) {  // BLIND PLACEMENT: Place orders, cancellation will remove bad ones
                    // Check if we already have an order at this predicted level
                    bool has_order = false;
                    for (const auto& [order_id, order] : our_orders) {
                        if (std::abs(order.price - pred_price) < 0.01 && !order.is_bid && order.is_active) {
                            has_order = true;
                            break;
                        }
                    }
                    if (!has_order) {
                        place_order_at_level(pred_price, thresholds.contracts_per_level / 2, false, market_data);  // Smaller size for predictive
                        ask_levels_placed++;
                    }
                }
            }
        }
        
        // Restore original max positions (for next interval)
        thresholds.max_open_positions_per_side = original_max_positions;
    }
    
    double predict_toxicity(const RealMarketData& market_data, bool is_long = true) {
        // Measure ML prediction latency (hot path)
        auto start = std::chrono::high_resolution_clock::now();
        
        if (mid_prices.size() < 10) {
            return 0.5;
        }
        
        // NEW: Select model based on side AND current regime
        MarketRegime current_regime = detect_regime(market_data);
        const MLModel* ml_model_ptr = nullptr;
        
        if (is_long) {
            // Use regime-specific model if available
            if (regime_models_long.find(current_regime_string) != regime_models_long.end()) {
                ml_model_ptr = &regime_models_long[current_regime_string];
            } else {
                ml_model_ptr = &ml_model_long;  // Fallback to default
            }
        } else {
            ml_model_ptr = &ml_model_short;
        }
        
        const MLModel& ml_model = *ml_model_ptr;
        
        // Calculate features (same as Python)
        double spread = market_data.ask_px_00 - market_data.bid_px_00;
        
        // Calculate CURRENT OFI (not the mean!)
        double ofi = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                  (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        // NEW: Track OFI history for slope calculation
        ofi_history.push_back(ofi);
        if (ofi_history.size() > OFI_HISTORY_WINDOW) {
            ofi_history.pop_front();
        }
        
        double momentum = 0.0;
        if (mid_prices.size() >= 20) {
            double short_avg = 0.0;
            double long_avg = 0.0;
            
            for (size_t i = mid_prices.size() - 5; i < mid_prices.size(); ++i) {
                short_avg += mid_prices[i];
            }
            short_avg /= 5.0;
            
            for (size_t i = mid_prices.size() - 20; i < mid_prices.size(); ++i) {
                long_avg += mid_prices[i];
            }
            long_avg /= 20.0;
            
            momentum = (short_avg - long_avg) / 0.25;  // Convert to ticks
        }
        
        double volatility = 0.0;
        if (mid_prices.size() >= 2) {
            double sum_sq = 0.0;
            for (size_t i = 1; i < mid_prices.size(); ++i) {
                double ret = (mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1];
                sum_sq += ret * ret;
            }
            volatility = std::sqrt(sum_sq / (mid_prices.size() - 1));
        }
        
        // Calculate qi_ask (for short model)
        double qi_ask = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0.0) {
            qi_ask = market_data.ask_sz_00 / (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        // Calculate qi_bid (for enhanced features)
        double qi_bid = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0.0) {
            qi_bid = market_data.bid_sz_00 / (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        // NEW: Enhanced features (if model supports them)
        double ofi_delta = 0.0;
        double microprice_dev = 0.0;
        double book_pressure = 0.0;
        double vol_rank = 1.0;
        double ofi_vol_interaction = 0.0;
        double momentum_spread_interaction = 0.0;
        double qi_spread_interaction = 0.0;
        
        if (ml_model.has_enhanced_features) {
            // NEW FEATURE 1: OFI Slope (OFI(t) - OFI(t-3))
            if (ofi_history.size() >= 4) {
                ofi_delta = ofi_history.back() - ofi_history[ofi_history.size() - 4];
            } else {
                ofi_delta = momentum * 0.02;  // Proxy if not enough history
            }
            
            // NEW FEATURE 2: Microprice
            double mid_price = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;
            double microprice = (market_data.ask_px_00 * market_data.bid_sz_00 + 
                                market_data.bid_px_00 * market_data.ask_sz_00) / 
                               (market_data.bid_sz_00 + market_data.ask_sz_00 + 1e-6);
            
            // NEW FEATURE 3: Microprice Deviation
            microprice_dev = microprice - mid_price;
            
            // NEW FEATURE 4: Depth Ratio Top 3
            double total_bid_top3 = market_data.bid_sz_00;
            double total_ask_top3 = market_data.ask_sz_00;
            for (int i = 0; i < 3 && i < 9; ++i) {
                total_bid_top3 += market_data.bid_sz[i];
                total_ask_top3 += market_data.ask_sz[i];
            }
            book_pressure = (total_ask_top3 > 0.001) ? (total_bid_top3 / total_ask_top3) : 1.0;
            
            // NEW FEATURE 5: Spread Rank (percentile in last 50)
            double spread_rank_20 = 0.5;
            if (spread_history.size() >= 5) {
                std::vector<double> sorted_spreads(spread_history.begin(), spread_history.end());
                std::sort(sorted_spreads.begin(), sorted_spreads.end());
                size_t rank = 0;
                for (size_t i = 0; i < sorted_spreads.size(); ++i) {
                    if (sorted_spreads[i] < spread) rank++;
                }
                spread_rank_20 = static_cast<double>(rank) / sorted_spreads.size();
            }
            
            // NEW FEATURE 6: RV Rank (percentile in last 200)
            vol_rank = 0.5;
            if (volatility_history.size() >= 10) {
                std::vector<double> sorted_vols(volatility_history.begin(), volatility_history.end());
                std::sort(sorted_vols.begin(), sorted_vols.end());
                size_t rank = 0;
                for (size_t i = 0; i < sorted_vols.size(); ++i) {
                    if (sorted_vols[i] < volatility) rank++;
                }
                vol_rank = static_cast<double>(rank) / sorted_vols.size();
            }
            
            // Interaction terms (non-linear patterns)
            ofi_vol_interaction = ofi * volatility;
            momentum_spread_interaction = momentum * spread;
            qi_spread_interaction = qi_bid * spread;
        }
        
        // NEW: Rolling Z-Score Normalization (for key features)
        // Use rolling scaler if available, else fall back to static
        static std::map<std::string, RollingScaler> rolling_scalers_long;
        static std::map<std::string, RollingScaler> rolling_scalers_short;
        
        auto& rolling_scalers = is_long ? rolling_scalers_long : rolling_scalers_short;
        
        // Update rolling scalers
        rolling_scalers["ofi"].update(ofi);
        rolling_scalers["momentum"].update(momentum);
        rolling_scalers["spread"].update(spread);
        rolling_scalers["volatility"].update(volatility);
        
        // Use rolling scaler if we have enough data, else use static
        bool use_rolling = (rolling_scalers["ofi"].values.size() >= 50);
        
        double spread_std, ofi_std, momentum_std, volatility_std, qi_ask_std;
        
        if (use_rolling) {
            spread_std = rolling_scalers["spread"].standardize(spread);
            ofi_std = rolling_scalers["ofi"].standardize(ofi);
            momentum_std = rolling_scalers["momentum"].standardize(momentum);
            volatility_std = rolling_scalers["volatility"].standardize(volatility);
            qi_ask_std = (qi_ask - ml_model.qi_ask_mean) / ml_model.qi_ask_scale;  // QI not rolling yet
        } else {
            // Fall back to static scaler
            spread_std = (spread - ml_model.spread_mean) / ml_model.spread_scale;
            ofi_std = (ofi - ml_model.ofi_mean) / ml_model.ofi_scale;
            momentum_std = (momentum - ml_model.momentum_mean) / ml_model.momentum_scale;
            volatility_std = (volatility - ml_model.volatility_mean) / ml_model.volatility_scale;
            qi_ask_std = (qi_ask - ml_model.qi_ask_mean) / ml_model.qi_ask_scale;
        }
        
        // CRITICAL FIX: Feature polarity for LONG side
        // For LONGS: Positive OFI = buying pressure = BAD (invert sign)
        //            Positive momentum = price going up = BAD for new longs (invert sign)
        // For SHORTS: Keep original signs (positive OFI/momentum = bad for shorts too)
        if (is_long) {
            ofi_std = -ofi_std;      // Invert OFI for longs (positive OFI = bad for longs)
            momentum_std = -momentum_std;  // Invert momentum (upward momentum = bad for longs)
        }
        
        // Standardize enhanced features (if model supports them)
        double ofi_delta_std = 0.0;
        double microprice_dev_std = 0.0;
        double book_pressure_std = 0.0;
        double vol_rank_std = 0.0;
        double ofi_vol_interaction_std = 0.0;
        double momentum_spread_interaction_std = 0.0;
        double qi_spread_interaction_std = 0.0;
        
        if (ml_model.has_enhanced_features) {
            ofi_delta_std = (ofi_delta - ml_model.ofi_delta_mean) / ml_model.ofi_delta_scale;
            microprice_dev_std = (microprice_dev - ml_model.microprice_dev_mean) / ml_model.microprice_dev_scale;
            book_pressure_std = (book_pressure - ml_model.book_pressure_mean) / ml_model.book_pressure_scale;
            vol_rank_std = (vol_rank - ml_model.vol_rank_mean) / ml_model.vol_rank_scale;
            ofi_vol_interaction_std = (ofi_vol_interaction - ml_model.ofi_vol_interaction_mean) / ml_model.ofi_vol_interaction_scale;
            momentum_spread_interaction_std = (momentum_spread_interaction - ml_model.momentum_spread_interaction_mean) / ml_model.momentum_spread_interaction_scale;
            qi_spread_interaction_std = (qi_spread_interaction - ml_model.qi_spread_interaction_mean) / ml_model.qi_spread_interaction_scale;
        }
        
        // Apply logistic regression (TRAINED WEIGHTS FROM PYTHON!)
        double logit = ml_model.intercept +
                      ml_model.spread_coef * spread_std +
                      ml_model.ofi_coef * ofi_std +
                      ml_model.momentum_coef * momentum_std +
                      ml_model.volatility_coef * volatility_std +
                      ml_model.qi_ask_coef * qi_ask_std;  // Short model includes qi_ask
        
        // Add enhanced features if model supports them
        if (ml_model.has_enhanced_features) {
            logit += ml_model.ofi_delta_coef * ofi_delta_std +
                    ml_model.microprice_dev_coef * microprice_dev_std +
                    ml_model.book_pressure_coef * book_pressure_std +
                    ml_model.vol_rank_coef * vol_rank_std +
                    ml_model.ofi_vol_interaction_coef * ofi_vol_interaction_std +
                    ml_model.momentum_spread_interaction_coef * momentum_spread_interaction_std +
                    ml_model.qi_spread_interaction_coef * qi_spread_interaction_std;
        }
        
        // Sigmoid
        double probability = 1.0 / (1.0 + std::exp(-logit));
        
        // NEW: Apply Platt Scaling (probability calibration)
        if (ml_model.calibration.use_platt) {
            double calibrated_logit = ml_model.calibration.a * logit + ml_model.calibration.b;
            probability = 1.0 / (1.0 + std::exp(-calibrated_logit));
        }
        
        // NEW: Calculate entropy for model confidence
        // Entropy = -p*log(p) - (1-p)*log(1-p)
        // High entropy (>0.65) = uncertain model → should be more conservative
        double entropy = 0.0;
        if (probability > 0.001 && probability < 0.999) {
            entropy = -probability * std::log(probability) - (1.0 - probability) * std::log(1.0 - probability);
        }
        
        // If entropy > 0.65 (uncertain zone), boost probability slightly to be more conservative
        // This prevents trading in uncertain model zones
        if (entropy > 0.65) {
            probability = probability * 1.15;  // Boost by 15% to be more conservative
            if (probability > 1.0) probability = 1.0;
        }
        
        // Record ML prediction latency (hot path)
        auto end = std::chrono::high_resolution_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        uint64_t current_total = metrics.total_toxicity_prediction_time.load(std::memory_order_relaxed);
        metrics.total_toxicity_prediction_time.store(current_total + latency_ns, std::memory_order_relaxed);
        metrics.toxicity_prediction_count.fetch_add(1, std::memory_order_relaxed);
        
        uint64_t current_max = metrics.max_toxicity_prediction_time.load(std::memory_order_relaxed);
        if (latency_ns > current_max) {
            metrics.max_toxicity_prediction_time.store(latency_ns, std::memory_order_relaxed);
        }
        
        return probability;
    }
    
    // ═══════════════════════════════════════════════════════════════
    //  MBO-TRAINED TOXICITY PREDICTION (55 features, ridge + isotonic)
    //  Computes features from current MBP-10 snapshot, evaluates the
    //  MBO-trained ridge model, applies isotonic calibration.
    //  Returns P(toxic) in [0, 1].
    // ═══════════════════════════════════════════════════════════════
    double predict_toxicity_mbo(const RealMarketData& md, bool is_long = true) {
        const MBOToxicityModel& model = mbo_tox_slow;  // primary horizon
        if (!model.loaded) return predict_toxicity(md, is_long);  // fallback

        // Build feature vector matching the feature_names order.
        // We can only compute a subset of MBO features from MBP-10 data:
        //   - imbalance, spread, depth, OFI proxies, vol, microprice
        // Features we CAN'T compute from MBP-10 (cancel rates, flicker, etc.)
        // are set to 0 (which after z-score normalization means "average").

        std::vector<double> x(model.n_features, 0.0);

        double bid_sz_l1 = md.bid_sz_00;
        double ask_sz_l1 = md.ask_sz_00;
        double total_l1 = bid_sz_l1 + ask_sz_l1;
        double imb_l1 = (total_l1 > 0) ? (bid_sz_l1 - ask_sz_l1) / total_l1 : 0.0;
        double spread_ticks = (md.ask_px_00 - md.bid_px_00) / 0.25;
        double mid = (md.bid_px_00 + md.ask_px_00) / 2.0;

        // microprice
        double microprice = mid;
        if (total_l1 > 0) {
            microprice = (md.ask_px_00 * bid_sz_l1 + md.bid_px_00 * ask_sz_l1) / total_l1;
        }
        double microprice_delta = microprice - mid;

        // per-level imbalances (L2-L5 from depth array)
        auto imb_lk = [&](int k) -> double {
            if (k == 0) return imb_l1;
            double b = md.bid_sz[k - 1];
            double a = md.ask_sz[k - 1];
            double t = b + a;
            return (t > 0) ? (b - a) / t : 0.0;
        };

        // book slope (linear regression of log(size) on level)
        auto compute_slope = [&](bool is_bid) -> double {
            double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
            for (int k = 0; k < 5; ++k) {
                double sz = (k == 0) ? (is_bid ? bid_sz_l1 : ask_sz_l1)
                                     : (is_bid ? md.bid_sz[k - 1] : md.ask_sz[k - 1]);
                double y = std::log(std::max(sz, 1.0));
                sum_x += k; sum_y += y; sum_xy += k * y; sum_xx += k * k;
            }
            double denom = 5.0 * sum_xx - sum_x * sum_x;
            return (std::abs(denom) > 1e-10) ? (5.0 * sum_xy - sum_x * sum_y) / denom : 0.0;
        };

        // HHI (Herfindahl) across 5 levels
        auto compute_hhi = [&](bool is_bid) -> double {
            double total = 0;
            double sizes[5];
            for (int k = 0; k < 5; ++k) {
                sizes[k] = (k == 0) ? (is_bid ? bid_sz_l1 : ask_sz_l1)
                                    : (is_bid ? md.bid_sz[k - 1] : md.ask_sz[k - 1]);
                total += sizes[k];
            }
            if (total < 1) return 1.0;
            double hhi = 0;
            for (int k = 0; k < 5; ++k) {
                double share = sizes[k] / total;
                hhi += share * share;
            }
            return hhi;
        };

        // OFI from current snapshot (proxy: size change approximation)
        double ofi_proxy = 0.0;
        if (mid_prices.size() >= 2) {
            ofi_proxy = imb_l1;  // best proxy from snapshot data
        }

        // volatility
        double volatility = 0.0;
        if (mid_prices.size() >= 10) {
            double sum_sq = 0.0;
            int cnt = std::min((int)mid_prices.size(), 50);
            for (int i = (int)mid_prices.size() - cnt + 1; i < (int)mid_prices.size(); ++i) {
                double ret = (mid_prices[i] - mid_prices[i - 1]) / (mid_prices[i - 1] + 1e-10);
                sum_sq += ret * ret;
            }
            volatility = std::sqrt(sum_sq / (cnt - 1));
        }

        // signed volume (from sequential learning)
        double signed_vol_proxy = (double)(sequential.buyer_streak - sequential.seller_streak);

        // Map features by name
        for (int i = 0; i < model.n_features; ++i) {
            const std::string& fn = model.feature_names[i];
            // OFI features
            if (fn.find("z_ofi_L1") != std::string::npos) x[i] = ofi_proxy;
            else if (fn.find("z_ofi_L2") != std::string::npos) x[i] = 0.0;
            else if (fn.find("z_ofi_deep") != std::string::npos) x[i] = 0.0;
            // Imbalance
            else if (fn == "z_imb_L1") x[i] = imb_l1;
            else if (fn == "z_imb_L2") x[i] = imb_lk(1);
            else if (fn == "z_imb_L3") x[i] = imb_lk(2);
            else if (fn == "z_L1_L2_disagree") x[i] = -std::copysign(1.0, imb_l1) * std::copysign(1.0, imb_lk(1));
            // Microprice
            else if (fn == "z_microprice_delta") x[i] = microprice_delta;
            // VPIN (can't compute from snapshot, leave at 0)
            else if (fn.find("z_vpin") != std::string::npos) x[i] = 0.0;
            // Kyle's lambda
            else if (fn == "z_kyle_lambda") x[i] = 0.0;
            // Trade arrival
            else if (fn == "z_trade_accel" || fn == "z_trade_rate") x[i] = 0.0;
            // Cancel
            else if (fn.find("z_cancel") != std::string::npos || fn.find("z_canc") != std::string::npos) x[i] = 0.0;
            // Spread
            else if (fn == "spread_ticks") x[i] = spread_ticks;
            else if (fn == "z_spread") x[i] = spread_ticks;
            // Book shape
            else if (fn == "z_slope_bid") x[i] = compute_slope(true);
            else if (fn == "z_slope_ask") x[i] = compute_slope(false);
            else if (fn.find("z_curv") != std::string::npos) x[i] = 0.0;
            // HHI
            else if (fn == "z_hhi_bid") x[i] = compute_hhi(true);
            else if (fn == "z_hhi_ask") x[i] = compute_hhi(false);
            // Order fragmentation (can't compute from MBP-10)
            else if (fn.find("z_avg_ord") != std::string::npos) x[i] = 0.0;
            // Depletion
            else if (fn.find("z_net_depletion") != std::string::npos) x[i] = 0.0;
            // Modify
            else if (fn.find("z_modify") != std::string::npos) x[i] = 0.0;
            // Vol
            else if (fn == "z_realized_vol") x[i] = volatility;
            // Signed vol
            else if (fn.find("z_signed_vol") != std::string::npos) x[i] = signed_vol_proxy;
            // Resilience
            else if (fn.find("z_resilience") != std::string::npos) x[i] = 0.0;
            // Fill asymmetry
            else if (fn.find("z_fill_asym") != std::string::npos) x[i] = 0.0;
            // Latency
            else if (fn == "z_latency") x[i] = 0.0;
            // Quote flicker
            else if (fn.find("z_flicker") != std::string::npos) x[i] = 0.0;
            // BBO move
            else if (fn.find("z_bid_move") != std::string::npos || fn.find("z_ask_move") != std::string::npos) x[i] = 0.0;
            // Spread dynamics
            else if (fn.find("z_widen") != std::string::npos) x[i] = 0.0;
            // TWAS
            else if (fn.find("z_twas") != std::string::npos) x[i] = spread_ticks;
            // Interactions
            else if (fn == "ofi_L1_x_vol") x[i] = ofi_proxy * volatility;
            else if (fn == "ofi_L1_x_spread") x[i] = ofi_proxy * spread_ticks;
            else if (fn == "imb_L2_x_vol") x[i] = imb_lk(1) * volatility;
            else if (fn == "slope_x_accel") x[i] = 0.0;
            else if (fn == "flicker_x_vol") x[i] = 0.0;
            else if (fn == "widen_x_ofi") x[i] = 0.0;
        }

        double raw_score = model.predict_raw(x);
        double abs_score = std::abs(raw_score);
        double p_toxic = model.calibrate(abs_score);

        // for long side, flip sign: positive score = buy pressure = toxic for longs
        if (is_long && raw_score > 0) {
            p_toxic = std::min(p_toxic * 1.1, 1.0);
        }

        return std::clamp(p_toxic, 0.0, 1.0);
    }

    // ═══════════════════════════════════════════════════════════════
    //  OPTIMAL SPREAD FROM SPREAD OPTIMIZER
    //  Returns (half_spread_ticks, skew_ticks).
    // ═══════════════════════════════════════════════════════════════
    std::pair<double, double> get_optimal_spread(const RealMarketData& md, double tox_score) {
        double spread_ticks = (md.ask_px_00 - md.bid_px_00) / 0.25;

        if (!spread_optimizer.loaded) {
            return {1.0, 0.0};  // default 1 tick half-spread, no skew
        }

        // determine vol regime from Markov model or fallback
        double vol = 0.0;
        if (mid_prices.size() >= 10) {
            double sum_sq = 0.0;
            int cnt = std::min((int)mid_prices.size(), 50);
            for (int i = (int)mid_prices.size() - cnt + 1; i < (int)mid_prices.size(); ++i) {
                double ret = (mid_prices[i] - mid_prices[i - 1]) / (mid_prices[i - 1] + 1e-10);
                sum_sq += ret * ret;
            }
            vol = std::sqrt(sum_sq / (cnt - 1));
        }

        int vol_regime = markov_regime.loaded ? markov_regime.detect_vol_regime(vol) : 1;

        // toxicity decile (0-9)
        double abs_tox = std::abs(tox_score);
        int tox_decile = std::clamp((int)(abs_tox * spread_optimizer.n_tox_bins), 0, spread_optimizer.n_tox_bins - 1);

        // try table lookup first, fall back to regression
        double hs = spread_optimizer.lookup_half_spread(vol_regime, tox_decile);
        double skew = spread_optimizer.lookup_skew(vol_regime, tox_decile);

        // blend with regression prediction for smoothness
        double hs_reg = spread_optimizer.predict_half_spread(abs_tox, (double)vol_regime, spread_ticks);
        hs = 0.7 * hs + 0.3 * hs_reg;

        // Markov uncertainty adjustment: wider spread when price is unpredictable
        if (markov_regime.loaded) {
            double imb = 0.0;
            if (md.bid_sz_00 + md.ask_sz_00 > 0)
                imb = (md.bid_sz_00 - md.ask_sz_00) / (md.bid_sz_00 + md.ask_sz_00);
            int state = markov_regime.discretize_state(spread_ticks, imb);
            double unc = markov_regime.uncertainty(vol_regime, state);
            if (unc > 1.5) hs *= 1.0 + 0.1 * (unc - 1.5);  // widen when uncertain
        }

        // inventory skew (from existing O'Hara control)
        skew += inventory_control.signed_inventory * 0.02;

        return {std::clamp(hs, 0.5, 5.0), std::clamp(skew, -3.0, 3.0)};
    }

    // NEW: Predict meta-model probability (combines toxicity + timeout + features)
    // Predict entry filter probability (regression-based, converts P&L prediction to probability)
    double predict_entry_filter_probability(const RealMarketData& market_data, bool is_long = true) {
        MarketRegime current_regime = detect_regime(market_data);
        std::string regime_str = (current_regime == MarketRegime::BALANCED) ? "balanced" :
                                (current_regime == MarketRegime::VOLATILE) ? "volatile" :
                                (current_regime == MarketRegime::WIDE_SPREAD) ? "wide_spread" : "chaotic";
        
        std::string model_key = (is_long ? "LONG_" : "SHORT_") + regime_str;
        
        if (entry_filter_models.find(model_key) == entry_filter_models.end()) {
            return 0.5;  // Default: neutral if model not loaded
        }
        
        const EntryFilterModel& model = entry_filter_models[model_key];
        if (!model.loaded) {
            return 0.5;
        }
        
        // Calculate all features
        double entry_ofi = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            entry_ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        double ofi_slope = 0.0;
        if (ofi_history.size() >= 4) {
            ofi_slope = ofi_history.back() - ofi_history[ofi_history.size() - 4];
        }
        
        // Microprice deviation
        double mid_price = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;
        double microprice = (market_data.ask_px_00 * market_data.bid_sz_00 + 
                            market_data.bid_px_00 * market_data.ask_sz_00) / 
                           (market_data.bid_sz_00 + market_data.ask_sz_00 + 1e-9);
        double microprice_dev = microprice - mid_price;
        
        // Momentum
        double momentum_3 = 0.0;
        double momentum_10 = 0.0;
        if (mid_prices.size() >= 20) {
            double short_avg = 0.0, long_avg = 0.0;
            for (size_t i = mid_prices.size() - 3; i < mid_prices.size(); ++i) {
                short_avg += mid_prices[i];
            }
            short_avg /= 3.0;
            for (size_t i = mid_prices.size() - 10; i < mid_prices.size(); ++i) {
                long_avg += mid_prices[i];
            }
            long_avg /= 10.0;
            momentum_3 = (short_avg - long_avg) / 0.25;
            momentum_10 = momentum_3;  // Proxy
        }
        
        // Book pressure
        double book_pressure_top3 = 1.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            double bid_top3 = market_data.bid_sz_00;
            double ask_top3 = market_data.ask_sz_00;
            for (int i = 0; i < 3 && i < 9; ++i) {
                bid_top3 += market_data.bid_sz[i];
                ask_top3 += market_data.ask_sz[i];
            }
            book_pressure_top3 = bid_top3 / (ask_top3 + 1e-9);
        }
        
        // Queue imbalance
        double queue_imbalance = 0.5;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            queue_imbalance = market_data.bid_sz_00 / (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        double queue_replenishment_rate = 0.0;  // Placeholder
        
        // Volatility rank
        double vol_rank_100 = 0.5;
        if (volatility_history.size() >= 10) {
            double current_vol = 0.0;
            if (mid_prices.size() >= 10) {
                std::vector<double> price_changes;
                for (size_t i = mid_prices.size() - std::min(10, (int)mid_prices.size()); i < mid_prices.size() - 1; ++i) {
                    double change = std::abs(mid_prices[i+1] - mid_prices[i]) / 0.25;
                    price_changes.push_back(change);
                }
                if (price_changes.size() > 1) {
                    double mean = 0.0;
                    for (double chg : price_changes) mean += chg;
                    mean /= price_changes.size();
                    double variance = 0.0;
                    for (double chg : price_changes) variance += (chg - mean) * (chg - mean);
                    current_vol = std::sqrt(variance / (price_changes.size() - 1));
                }
            }
            int rank = 0;
            for (double v : volatility_history) {
                if (v < current_vol) rank++;
            }
            vol_rank_100 = static_cast<double>(rank) / volatility_history.size();
        }
        
        // Spread rank
        double spread_rank_20 = 0.5;
        double spread = market_data.ask_px_00 - market_data.bid_px_00;
        if (spread_history.size() >= 5) {
            int rank = 0;
            for (double s : spread_history) {
                if (s < spread) rank++;
            }
            spread_rank_20 = static_cast<double>(rank) / spread_history.size();
        }
        
        double entry_volatility = 0.0;
        if (mid_prices.size() >= 10) {
            std::vector<double> price_changes;
            for (size_t i = mid_prices.size() - std::min(10, (int)mid_prices.size()); i < mid_prices.size() - 1; ++i) {
                double change = std::abs(mid_prices[i+1] - mid_prices[i]) / 0.25;
                price_changes.push_back(change);
            }
            if (price_changes.size() > 1) {
                double mean = 0.0;
                for (double chg : price_changes) mean += chg;
                mean /= price_changes.size();
                double variance = 0.0;
                for (double chg : price_changes) variance += (chg - mean) * (chg - mean);
                entry_volatility = std::sqrt(variance / (price_changes.size() - 1));
            }
        }
        double entry_spread = spread / 0.25;  // Convert to ticks
        
        // Interaction features
        double ofi_vol_interaction = entry_ofi * vol_rank_100;
        double momentum_spread_interaction = momentum_3 * spread_rank_20;
        double microprice_book_interaction = microprice_dev * book_pressure_top3;
        
        // Build feature vector in order
        std::vector<double> features;
        for (const auto& feat_name : model.feature_order) {
            if (feat_name == "entry_ofi") features.push_back(entry_ofi);
            else if (feat_name == "ofi_slope") features.push_back(ofi_slope);
            else if (feat_name == "microprice_dev") features.push_back(microprice_dev);
            else if (feat_name == "momentum_3") features.push_back(momentum_3);
            else if (feat_name == "momentum_10") features.push_back(momentum_10);
            else if (feat_name == "book_pressure_top3") features.push_back(book_pressure_top3);
            else if (feat_name == "queue_imbalance") features.push_back(queue_imbalance);
            else if (feat_name == "queue_replenishment_rate") features.push_back(queue_replenishment_rate);
            else if (feat_name == "vol_rank_100") features.push_back(vol_rank_100);
            else if (feat_name == "spread_rank_20") features.push_back(spread_rank_20);
            else if (feat_name == "entry_volatility") features.push_back(entry_volatility);
            else if (feat_name == "entry_spread") features.push_back(entry_spread);
            else if (feat_name == "ofi_vol_interaction") features.push_back(ofi_vol_interaction);
            else if (feat_name == "momentum_spread_interaction") features.push_back(momentum_spread_interaction);
            else if (feat_name == "microprice_book_interaction") features.push_back(microprice_book_interaction);
            else features.push_back(0.0);
        }
        
        // Scale features
        if (features.size() != model.scaler_mean.size()) {
            return 0.5;  // Mismatch
        }
        
        std::vector<double> features_scaled;
        for (size_t i = 0; i < features.size(); ++i) {
            double scaled = (features[i] - model.scaler_mean[i]) / (model.scaler_scale[i] + 1e-9);
            features_scaled.push_back(scaled);
        }
        
        // Predict P&L using Ridge regression
        double pnl_pred = model.intercept;
        for (size_t i = 0; i < model.feature_order.size(); ++i) {
            const std::string& feat_name = model.feature_order[i];
            if (model.weights.find(feat_name) != model.weights.end()) {
                pnl_pred += model.weights.at(feat_name) * features_scaled[i];
            }
        }
        
        // Convert P&L prediction to probability [0, 1] using normalization
        // We'll use a simple sigmoid-like conversion or store min/max from training
        // For now, use simple normalization: assume P&L range [-2, 2] ticks
        double pnl_min = -2.0;
        double pnl_max = 2.0;
        double prob = (pnl_pred - pnl_min) / (pnl_max - pnl_min);
        prob = std::clamp(prob, 0.0, 1.0);
        
        return prob;
    }
    
    double predict_meta_probability(const RealMarketData& market_data, bool is_long) {
        if (regime_meta_models.find(current_regime_string) == regime_meta_models.end()) {
            return 0.5;  // Default: neutral if no meta model
        }
        
        const MetaModel& meta = regime_meta_models[current_regime_string];
        if (!meta.loaded) {
            return 0.5;
        }
        
        // Get base predictions (prefer MBO model)
        double tox_prob = mbo_tox_slow.loaded ? predict_toxicity_mbo(market_data, is_long)
                                              : predict_toxicity(market_data, is_long);
        double timeout_prob = predict_timeout_probability(market_data);
        
        // Get enhanced features
        double ofi = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                  (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        double ofi_slope = 0.0;
        if (ofi_history.size() >= 4) {
            ofi_slope = ofi_history.back() - ofi_history[ofi_history.size() - 4];
        }
        
        double microprice_dev = 0.0;
        double qi_bid = 0.0, qi_ask = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0.0) {
            qi_bid = market_data.bid_sz_00 / (market_data.bid_sz_00 + market_data.ask_sz_00);
            qi_ask = market_data.ask_sz_00 / (market_data.bid_sz_00 + market_data.ask_sz_00);
            double mid_price = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;
            double microprice = (market_data.ask_px_00 * market_data.bid_sz_00 + 
                                market_data.bid_px_00 * market_data.ask_sz_00) / 
                               (market_data.bid_sz_00 + market_data.ask_sz_00 + 1e-6);
            microprice_dev = microprice - mid_price;
        }
        
        double rv_rank = 0.5;
        if (volatility_history.size() >= 10) {
            double current_vol = 0.0;
            if (mid_prices.size() >= 2) {
                double sum_sq = 0.0;
                for (size_t i = 1; i < mid_prices.size(); ++i) {
                    double ret = (mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1];
                    sum_sq += ret * ret;
                }
                current_vol = std::sqrt(sum_sq / (mid_prices.size() - 1));
            }
            
            std::vector<double> sorted_vols(volatility_history.begin(), volatility_history.end());
            std::sort(sorted_vols.begin(), sorted_vols.end());
            size_t rank = 0;
            for (size_t i = 0; i < sorted_vols.size(); ++i) {
                if (sorted_vols[i] < current_vol) rank++;
            }
            rv_rank = static_cast<double>(rank) / sorted_vols.size();
        }
        
        double depth_ratio = (qi_ask > 0.001) ? (qi_bid / qi_ask) : 1.0;
        
        // Build feature vector (match meta model feature order)
        std::vector<double> features;
        features.push_back(tox_prob);
        features.push_back(timeout_prob);
        features.push_back(ofi);
        features.push_back(ofi_slope);
        features.push_back(microprice_dev);
        features.push_back(depth_ratio);
        features.push_back(rv_rank);
        
        // Standardize features
        if (features.size() <= meta.scaler_mean.size() && features.size() <= meta.scaler_scale.size()) {
            for (size_t i = 0; i < features.size(); ++i) {
                features[i] = (features[i] - meta.scaler_mean[i]) / meta.scaler_scale[i];
            }
        }
        
        // Predict
        double logit = meta.intercept;
        for (size_t i = 0; i < features.size() && i < meta.coefficients.size(); ++i) {
            logit += meta.coefficients[i] * features[i];
        }
        
        double prob = 1.0 / (1.0 + std::exp(-logit));
        return prob;
    }
    
    // Proximity cancel for shorts: Cancel if 2 consecutive upticks in 50ms or best bid steps up twice
    ALWAYS_INLINE void check_short_proximity_cancel(const RealMarketData& market_data) noexcept {
        if (!thresholds.enable_short_proximity_cancel || !thresholds.enable_shorts) {
            return;
        }
        
        uint64_t current_time_ns = market_data.timestamp_ns;
        uint64_t cooldown_window_ns = thresholds.short_cooldown_ns;
        uint64_t proximity_window_ns = 50000000ULL;  // 50ms
        
        for (auto& [order_id, order] : our_orders) {
            if (!order.is_bid && order.is_active) {  // Only check ask orders (shorts)
                // Check cooldown
                auto cooldown_it = order_last_cancel_ns.find(order_id);
                if (cooldown_it != order_last_cancel_ns.end()) {
                    if ((current_time_ns - cooldown_it->second) < cooldown_window_ns) {
                        continue;  // Still in cooldown
                    }
                }
                
                // Track bid price history for this order
                auto& bid_history = order_bid_history[order_id];
                bid_history.push_back({current_time_ns, market_data.bid_px_00});
                
                // Remove old entries outside 50ms window
                while (!bid_history.empty() && (current_time_ns - bid_history.front().first) > proximity_window_ns) {
                    bid_history.pop_front();
                }
                
                // Check for 2 consecutive upticks (bid price going up = bad for shorts)
                if (bid_history.size() >= 2) {
                    bool consecutive_upticks = true;
                    double prev_bid = bid_history[0].second;
                    for (size_t i = 1; i < bid_history.size(); ++i) {
                        if (bid_history[i].second <= prev_bid + 0.01) {  // Not an uptick
                            consecutive_upticks = false;
                            break;
                        }
                        prev_bid = bid_history[i].second;
                    }
                    
                    // Check if best bid stepped up twice (bid_px_00 increased by 2 ticks)
                    double bid_change = (market_data.bid_px_00 - bid_history.front().second) / 0.25;
                    bool bid_stepped_up_twice = bid_change >= 2.0;
                    
                    if (consecutive_upticks || bid_stepped_up_twice) {
                        // Cancel order and set cooldown
                        order.is_active = false;
                        auto& level = ask_levels[order.price];
                        level.our_size -= order.contracts;
                        // Rebuild queue
                        std::queue<Order> new_queue;
                        while (!level.order_queue.empty()) {
                            Order q_order = level.order_queue.front();
                            level.order_queue.pop();
                            if (q_order.order_id != order.order_id) {
                                new_queue.push(q_order);
                            }
                        }
                        level.order_queue = new_queue;
                        level.queue_depth = level.order_queue.size();
                        order_last_cancel_ns[order_id] = current_time_ns;
                        metrics.orders_canceled_toxic.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }
    void check_toxicity_cancellation(const RealMarketData& market_data) {
        // Use MBO-trained model when available, fall back to old model
        double long_toxicity = mbo_tox_slow.loaded ? predict_toxicity_mbo(market_data, true)
                                                   : predict_toxicity(market_data, true);
        double short_toxicity = mbo_tox_slow.loaded ? predict_toxicity_mbo(market_data, false)
                                                    : predict_toxicity(market_data, false);
        
        // Track toxic events for debugging and production KPIs
        static int toxicity_check_count = 0;
        static int toxic_events = 0;
        if (++toxicity_check_count % 1000 == 0) {
            std::cout << "Toxicity sample at " << toxicity_check_count << ": LONG=" << long_toxicity 
                      << " SHORT=" << short_toxicity << std::endl;
        }
        
        // Check proximity cancel for shorts (2 upticks in 50ms or bid steps up twice)
        check_short_proximity_cancel(market_data);
        
        // Track toxicity samples for rolling % calculation (use average for display)
        double avg_toxicity = (long_toxicity + short_toxicity) / 2.0;
        {
            std::lock_guard<std::mutex> lock(metrics.kpi_mutex);
            metrics.toxicity_samples.push_back(avg_toxicity);
            if (metrics.toxicity_samples.size() > 1000) {
                metrics.toxicity_samples.pop_front();
            }
        }
        
        // Compute OFI and QBI for tracking
        double ofi = 0.0;
        double qbi = 0.0;
        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
            ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                  (market_data.bid_sz_00 + market_data.ask_sz_00);
            qbi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                  (market_data.bid_sz_00 + market_data.ask_sz_00);
        }
        
        // SIMPLIFIED: Calculate momentum (price movement direction)
        // Positive momentum = upward movement (good for longs, bad for shorts)
        // Negative momentum = downward movement (bad for longs, good for shorts)
        double momentum = 0.0;
        if (mid_prices.size() >= 2) {
            // Momentum = recent price change in ticks
            momentum = (mid_prices.back() - *(std::prev(mid_prices.end(), 2))) / 0.25;
        }
        
        // AGGRESSIVE FAST CANCELLATION LOGIC:
        // Cancel LONGS IMMEDIATELY when seeing ANY downward momentum (price moving down)
        // Cancel SHORTS IMMEDIATELY when seeing ANY upward momentum (price moving up)
        // Use VERY TIGHT thresholds: -0.1 ticks for longs, +0.1 ticks for shorts
        // This catches adverse moves FAST before they fill
        
        // DATA COLLECTION MODE: DISABLE cancellation for LONGS (bids)
        // We need to see what makes bids profitable, so let them fill and exit naturally
        const double LONG_CANCEL_MOMENTUM_THRESHOLD = -0.1;  // Cancel longs when momentum turns sharply negative
        const double SHORT_CANCEL_MOMENTUM_THRESHOLD = +0.1;  // Keep shorts cancellation active
        
        bool cancel_longs = thresholds.enable_longs && (momentum < LONG_CANCEL_MOMENTUM_THRESHOLD);
        bool cancel_shorts = thresholds.enable_shorts && (momentum > SHORT_CANCEL_MOMENTUM_THRESHOLD);
        
        bool long_ml_toxicity = long_toxicity >= thresholds.toxic_cancel_threshold;
        bool short_ml_toxicity = short_toxicity >= thresholds.short_tox_threshold;
        
        // Combine momentum + ML toxicity (cancel if either triggers)
        // But longs are disabled, so this only affects shorts
        cancel_longs = cancel_longs || (thresholds.enable_longs && long_ml_toxicity);
        cancel_shorts = cancel_shorts || (thresholds.enable_shorts && short_ml_toxicity);
        
        if (cancel_shorts || cancel_longs) {
            toxic_events++;
            int cancel_count = 0;
            int total_orders = our_orders.size();
            int active_orders = 0;
            for (const auto& [order_id, order] : our_orders) {
                if (order.is_active) active_orders++;
            }
            
            for (auto& [order_id, order] : our_orders) {
                // Apply side-specific cancellation logic
                bool should_cancel = false;
                if (order.is_bid && cancel_longs) {
                    should_cancel = true;
                } else if (!order.is_bid && cancel_shorts) {
                    should_cancel = true;
                }
                
                if (order.is_active && should_cancel) {
                    // REALISTIC: Don't cancel orders at the front of the queue - they're about to fill
                    // If queue_position <= 5, the order is likely to fill soon, so cancellation is unrealistic
                    // In reality, once you're at the front, a trade can hit you before cancellation processes
                    if (order.queue_position <= 5) {
                        continue;  // Skip cancellation - order is about to fill
                    }
                    
                    // Track canceled order for profit analysis
                    CanceledOrderData canceled;
                    canceled.is_long = order.is_bid;
                    canceled.order_price = order.price;
                    canceled.cancel_timestamp_ns = market_data.timestamp_ns;
                    canceled.cancel_toxicity = order.is_bid ? long_toxicity : short_toxicity;
                    canceled.cancel_ofi = ofi;
                    
                    // Track momentum at cancellation
                    canceled.cancel_momentum = momentum;
                    canceled.cancel_mid_price = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;
                    canceled.queue_position_at_cancel = order.queue_position;
                    
                    // Track cancellation reason: 0=ML toxicity, 2=momentum
                    if (order.is_bid) {
                        // Long cancellation
                        canceled.cancel_reason = (momentum < LONG_CANCEL_MOMENTUM_THRESHOLD) ? 2.0 : 0.0;  // 2=momentum, 0=ML
                    } else {
                        // Short cancellation
                        canceled.cancel_reason = (momentum > SHORT_CANCEL_MOMENTUM_THRESHOLD) ? 2.0 : 0.0;  // 2=momentum, 0=ML
                    }
                    
                    // Initialize post-cancellation tracking (will be updated later)
                    canceled.best_price_after_10_intervals = canceled.cancel_mid_price;
                    canceled.best_price_after_20_intervals = canceled.cancel_mid_price;
                    canceled.best_price_after_40_intervals = canceled.cancel_mid_price;
                    canceled.would_have_been_profitable_10 = false;
                    canceled.would_have_been_profitable_20 = false;
                    canceled.would_have_been_profitable_40 = false;
                    metrics.canceled_orders_for_analysis.push_back(canceled);
                    
                    order.is_active = false;
                    cancel_count++;  // Count how many we're canceling
                    
                    if (order.is_bid) {
                        auto& level = bid_levels[order.price];
                        // Don't modify total_size (market depth), only our_size
                        level.our_size -= order.contracts;
                        // Remove order from queue (rebuild queue without this order)
                        std::queue<Order> new_queue;
                        while (!level.order_queue.empty()) {
                            Order q_order = level.order_queue.front();
                            level.order_queue.pop();
                            if (q_order.order_id != order.order_id) {
                                new_queue.push(q_order);
                            }
                        }
                        level.order_queue = new_queue;
                        level.queue_depth = level.order_queue.size();
                    } else {
                        auto& level = ask_levels[order.price];
                        // Don't modify total_size (market depth), only our_size
                        level.our_size -= order.contracts;
                        // Remove order from queue (rebuild queue without this order)
                        std::queue<Order> new_queue;
                        while (!level.order_queue.empty()) {
                            Order q_order = level.order_queue.front();
                            level.order_queue.pop();
                            if (q_order.order_id != order.order_id) {
                                new_queue.push(q_order);
                            }
                        }
                        level.order_queue = new_queue;
                        level.queue_depth = level.order_queue.size();
                    }
                }
            }
            // Increment ONCE for ALL orders canceled in this toxic event
            if (cancel_count > 0) {
                metrics.orders_canceled_toxic.fetch_add(cancel_count, std::memory_order_relaxed);
                // Track when toxic cancels happened for refill cooldown
                if (cancel_longs) {
                    last_long_toxic_cancel_ns = market_data.timestamp_ns;
                }
                if (cancel_shorts) {
                    last_short_toxic_cancel_ns = market_data.timestamp_ns;
                }
                if (toxic_events <= 5) {
                    std::cout << "🧨 TOXIC EVENT #" << toxic_events << " - Canceling " << cancel_count 
                              << " orders (total: " << total_orders << ", active: " << active_orders 
                              << ", LONG_tox=" << long_toxicity << ", SHORT_tox=" << short_toxicity << ")" << std::endl;
                }
            }
        }
    }
    
    void update_canceled_order_tracking(const RealMarketData& market_data) {
        // Update canceled orders with post-cancellation price movements
        double current_mid = (market_data.bid_px_00 + market_data.ask_px_00) / 2.0;
        uint64_t current_timestamp = market_data.timestamp_ns;
        
        for (auto& canceled : metrics.canceled_orders_for_analysis) {
            uint64_t intervals_since_cancel = (current_timestamp - canceled.cancel_timestamp_ns) / 100000000ULL;  // ~100ms per interval
            
            // Calculate hypothetical P&L if order had filled
            double hypothetical_pnl_ticks = 0.0;
            if (canceled.is_long) {
                hypothetical_pnl_ticks = (current_mid - canceled.order_price) / 0.25;
            } else {
                hypothetical_pnl_ticks = (canceled.order_price - current_mid) / 0.25;
            }
            
            // Track best prices at different intervals
            if (intervals_since_cancel == 10) {
                canceled.best_price_after_10_intervals = current_mid;
                canceled.would_have_been_profitable_10 = (hypothetical_pnl_ticks >= 1.0);  // >= 1 tick profit
            } else if (intervals_since_cancel == 20) {
                canceled.best_price_after_20_intervals = current_mid;
                canceled.would_have_been_profitable_20 = (hypothetical_pnl_ticks >= 1.0);
            } else if (intervals_since_cancel == 40) {
                canceled.best_price_after_40_intervals = current_mid;
                canceled.would_have_been_profitable_40 = (hypothetical_pnl_ticks >= 1.0);
            }
            
            // Update best prices (track maximum favorable movement)
            if (canceled.is_long) {
                // Long: track highest price reached
                if (current_mid > canceled.best_price_after_10_intervals && intervals_since_cancel <= 10) {
                    canceled.best_price_after_10_intervals = current_mid;
                    canceled.would_have_been_profitable_10 = ((current_mid - canceled.order_price) / 0.25 >= 1.0);
                }
                if (current_mid > canceled.best_price_after_20_intervals && intervals_since_cancel <= 20) {
                    canceled.best_price_after_20_intervals = current_mid;
                    canceled.would_have_been_profitable_20 = ((current_mid - canceled.order_price) / 0.25 >= 1.0);
                }
                if (current_mid > canceled.best_price_after_40_intervals && intervals_since_cancel <= 40) {
                    canceled.best_price_after_40_intervals = current_mid;
                    canceled.would_have_been_profitable_40 = ((current_mid - canceled.order_price) / 0.25 >= 1.0);
                }
            } else {
                // Short: track lowest price reached
                if (current_mid < canceled.best_price_after_10_intervals && intervals_since_cancel <= 10) {
                    canceled.best_price_after_10_intervals = current_mid;
                    canceled.would_have_been_profitable_10 = ((canceled.order_price - current_mid) / 0.25 >= 1.0);
                }
                if (current_mid < canceled.best_price_after_20_intervals && intervals_since_cancel <= 20) {
                    canceled.best_price_after_20_intervals = current_mid;
                    canceled.would_have_been_profitable_20 = ((canceled.order_price - current_mid) / 0.25 >= 1.0);
                }
                if (current_mid < canceled.best_price_after_40_intervals && intervals_since_cancel <= 40) {
                    canceled.best_price_after_40_intervals = current_mid;
                    canceled.would_have_been_profitable_40 = ((canceled.order_price - current_mid) / 0.25 >= 1.0);
                }
            }
        }
    }
    
    void simulate_fills(const RealMarketData& market_data) {
        // VALIDATION: Only process valid trade prices (same range as bid/ask: 6000-7000)
        if (market_data.trade_price < 6000.0 || market_data.trade_price > 7000.0 || market_data.trade_price <= 0.0) {
            return;  // Skip invalid trade prices
        }
        
        // O'HARA #12: Update sequential learning (track trade sides for Bayesian streak)
        if (market_data.trade_size > 0 && market_data.trade_side != '\0') {
            sequential_learning.update(market_data.trade_side);
        }
        
        // STRICT GAP DETECTION: Reject fills if price jump > 5 ticks (catastrophic fills prevention)
        if (mid_prices.size() > 0) {
            double last_mid = mid_prices.back();
            double price_jump = std::abs(market_data.trade_price - last_mid) / 0.25;  // Ticks
            if (price_jump > 5.0) {  // FIXED: Stricter gap detection (5 ticks = 1.25 points)
                // Skip this fill - likely data gap, market halt, or extreme volatility
                return;
            }
        }
        
        for (auto& [price, level] : bid_levels) {
            // REALISTIC QUEUE POSITION: Reset if price moved away significantly (> 2 ticks) and came back
            double price_distance_from_best = std::abs(market_data.bid_px_00 - price) / 0.25;  // In ticks
            bool is_now_at_best = (price_distance_from_best < 0.5);  // Within 0.5 ticks of best
            
            // Track if price moved away and came back
            if (level.is_at_best_level) {
                // We were at best level - check if price moved away
                double last_distance = std::abs(level.last_best_price - price) / 0.25;
                if (last_distance > 2.0 && !is_now_at_best) {
                    // Price moved away (> 2 ticks) - mark that we're no longer at best
                    level.is_at_best_level = false;
                }
            } else {
                // We were NOT at best level - check if price came back
                if (is_now_at_best) {
                    // Price came back - DON'T reset existing orders' queue positions!
                    // They keep their position (e.g., if they were at position 20, they stay there)
                    // Only reset volume_traded_at_level so NEW orders start fresh
                    // This preserves FIFO ordering: old orders stay ahead, new orders go behind
                    level.volume_traded_at_level = 0;
                    // Existing orders keep their current queue_position - don't reset them!
                }
            }
            
            level.is_at_best_level = is_now_at_best;
            level.last_best_price = market_data.bid_px_00;
            
            // Track volume traded at this level (for queue position estimation)
            // Only count volume when price is at/near our level (within 1 tick)
            double volume_traded_this_tick = 0.0;
            if (std::abs(market_data.trade_price - price) < 0.01 && market_data.trade_size > 0) {
                volume_traded_this_tick = market_data.trade_size;
                level.volume_traded_at_level += static_cast<int>(market_data.trade_size);
                
                // Update queue positions: as volume trades, we move forward in queue
                // Only update active orders at this level
                std::queue<Order> temp_queue;
                while (!level.order_queue.empty()) {
                    Order q_order = level.order_queue.front();
                    level.order_queue.pop();
                    if (our_orders[q_order.order_id].is_active) {
                        // Estimate: each trade reduces queue position (moves us forward)
                        // More accurate: initial_depth - cumulative_volume_traded
                        int cumulative_volume = level.volume_traded_at_level;
                        q_order.queue_position = std::max(0, q_order.initial_market_depth - cumulative_volume);
                        our_orders[q_order.order_id].queue_position = q_order.queue_position;
                    }
                    temp_queue.push(q_order);
                }
                level.order_queue = temp_queue;
            }
            
            // REALISTIC FILL LOGIC: fill when trade_price <= our bid price (someone sells to us)
            // For EXIT orders (TP/SL): Fill when price trades THROUGH limit (>= for sells, <= for buys)
            // CRITICAL: Only fill if we're at the front of the queue (queue_position <= remaining_trade_size)
            // SELF-TRADE PREVENTION: DISABLED FOR DATA COLLECTION - allow fills even if we have ask at same price
            bool has_ask_at_same_price = false;  // DISABLED: Allow self-trades for data collection
            // if (ask_levels.find(price) != ask_levels.end() && ask_levels[price].our_size > 0) {
            //     has_ask_at_same_price = true;  // We have both bid and ask at same price - prevent self-trade
            // }
            
            // Check if any orders at this level are exit orders that should fill
            bool has_exit_order_that_should_fill = false;
            if (!level.order_queue.empty()) {
                std::queue<Order> temp_check = level.order_queue;
                while (!temp_check.empty()) {
                    Order q_order = temp_check.front();
                    temp_check.pop();
                    if (our_orders[q_order.order_id].is_active && q_order.is_exit_order) {
                        // Exit orders fill when price trades THROUGH them
                        // For bids (buy orders): Fill when trade_price <= order_price (price trades down to/through our buy)
                        // For asks (sell orders): Fill when trade_price >= order_price (price trades up to/through our sell)
                        bool should_fill_exit = false;
                        if (q_order.is_bid) {
                            // Buy order (SHORT exit): Fill when price <= order price
                            should_fill_exit = (market_data.trade_price <= q_order.price);
                        } else {
                            // Sell order (LONG exit): Fill when price >= order price
                            should_fill_exit = (market_data.trade_price >= q_order.price);
                        }
                        if (should_fill_exit && market_data.trade_size > 0) {
                            has_exit_order_that_should_fill = true;
                            break;
                        }
                    }
                }
            }
            
            // Fill condition: Entry orders need exact match, Exit orders fill when price trades through
            bool should_fill = (level.our_size > 0 && market_data.trade_size > 0 && !has_ask_at_same_price) &&
                              ((market_data.trade_price <= price) || has_exit_order_that_should_fill);
            
            if (should_fill) {
                double remaining_trade_size = market_data.trade_size;
                
                while (!level.order_queue.empty() && level.our_size > 0 && remaining_trade_size > 0) {
                    Order order = level.order_queue.front();
                    level.order_queue.pop();
                    
                    if (our_orders[order.order_id].is_active) {
                        // REALISTIC: Only fill if we're at the front of the queue
                        // queue_position = 0 means we're at the front, negative means we should have been filled already
                        if (order.queue_position > remaining_trade_size) {
                            // Not enough volume to reach us - put order back and break
                            level.order_queue.push(order);
                            break;
                        }
                        
                        // REALISTIC: Partial fills - a 50-contract trade can partially fill multiple orders
                        // We're at the front (or should have been filled) - fill us
                        double fill_size = std::min(remaining_trade_size, static_cast<double>(order.contracts));
                        remaining_trade_size -= fill_size;
                        
                        // Check if order is fully filled or partially filled
                        bool is_fully_filled = (fill_size >= static_cast<double>(order.contracts) - 0.001);  // Allow floating point tolerance
                        
                        if (!is_fully_filled) {
                            // PARTIAL FILL: Reduce order size and put it back in queue (FIFO preserved)
                            order.contracts -= static_cast<int>(fill_size);
                            our_orders[order.order_id].contracts = order.contracts;
                            level.our_size -= fill_size;
                            level.total_size -= fill_size;
                            // Queue position updated: we consumed fill_size, so move forward
                            order.queue_position = std::max(0, order.queue_position - static_cast<int>(fill_size));
                            our_orders[order.order_id].queue_position = order.queue_position;
                            level.order_queue.push(order);  // Put back for remaining size
                            // Continue to next order in queue (partial fill processed)
                            continue;
                        }
                        
                        // FULL FILL: Order completely filled, remove from queue
                        level.our_size -= fill_size;
                        level.total_size -= fill_size;
                        level.queue_depth--;
                        
                        // Check if this is an exit order
                        if (order.is_exit_order) {
                            // Exit orders fill when price trades THROUGH them
                            // For bids (buy orders): Fill when trade_price <= order_price
                            // For asks (sell orders): Fill when trade_price >= order_price
                            bool should_fill_this_exit = false;
                            if (order.is_bid) {
                                // Buy order (SHORT exit): Fill when price <= order price
                                should_fill_this_exit = (market_data.trade_price <= order.price);
                            } else {
                                // Sell order (LONG exit): Fill when price >= order price
                                should_fill_this_exit = (market_data.trade_price >= order.price);
                            }
                            
                            if (!should_fill_this_exit) {
                                // Exit order shouldn't fill yet - put it back
                                level.order_queue.push(order);
                                break;
                            }
                            
                            // Exit order filled - store actual fill price and mark as filled
                            size_t pos_idx = order.position_index;
                            if (pos_idx < active_positions.size()) {
                                active_positions[pos_idx].exit_actual_fill_price = market_data.trade_price;
                            }
                            our_orders[order.order_id].is_active = false;
                            continue;  // Skip entry position creation
                        }
                        double entry_ofi = 0.0;
                        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
                            entry_ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                                       (market_data.bid_sz_00 + market_data.ask_sz_00);
                        }
                        double entry_momentum = 0.0;
                        if (mid_prices.size() >= 20) {
                            double short_avg = 0.0, long_avg = 0.0;
                            auto it = mid_prices.end();
                            for (int i = 0; i < 5 && it != mid_prices.begin(); ++i) {
                                it--;
                                short_avg += *it;
                            }
                            for (int i = 0; i < 15 && it != mid_prices.begin(); ++i) {
                                it--;
                                long_avg += *it;
                            }
                            if (short_avg > 0 && long_avg > 0) {
                                entry_momentum = (short_avg / 5.0 - long_avg / 15.0) / 0.25;
                            }
                        }
                        double entry_spread = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
                        double entry_volatility = 0.0;
                        if (mid_prices.size() >= 10) {
                            double sum = 0.0, sum_sq = 0.0;
                            int count = 0;
                            for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                                sum += *it;
                                sum_sq += (*it) * (*it);
                                count++;
                            }
                            if (count > 1) {
                                double mean = sum / count;
                                entry_volatility = std::sqrt((sum_sq / count) - (mean * mean));
                            }
                        }
                        double entry_qi_bid = 0.0, entry_qi_ask = 0.0;
                        double total_bid = market_data.bid_sz_00, total_ask = market_data.ask_sz_00;
                        for (int i = 0; i < 9; ++i) {
                            total_bid += market_data.bid_sz[i];
                            total_ask += market_data.ask_sz[i];
                        }
                        if (total_bid + total_ask > 0) {
                            entry_qi_bid = total_bid / (total_bid + total_ask);
                            entry_qi_ask = total_ask / (total_bid + total_ask);
                        }
                        
                        Position pos;
                        pos.entry_price = market_data.trade_price;  // CRITICAL FIX: Use actual fill price, not limit order price!
                        pos.contracts = static_cast<int>(fill_size);
                        pos.entry_timestamp = market_data.timestamp_ns;
                        pos.is_long = true;
                        pos.hold_intervals = 0;
                        pos.first_profit_target_interval = -1;  // -1 = not reached yet
                        pos.mae_at_first_profit_target = 0.0;  // Will be set when we first reach +1 tick
                        pos.max_adverse_excursion = 0.0;
                        pos.max_favorable_excursion = 0.0;
                        pos.fill_trade_side = 'N';  // Not applicable for longs
                        pos.entry_bid_px = 0.0;
                        pos.entry_ask_px = 0.0;
                        pos.next_interval_ask_px = 0.0;
                        // Store entry features
                        pos.entry_ofi = entry_ofi;
                        pos.entry_momentum = entry_momentum;
                        pos.entry_spread = entry_spread;
                        pos.entry_volatility = entry_volatility;
                        pos.entry_qi_bid = entry_qi_bid;
                        pos.entry_qi_ask = entry_qi_ask;
                        active_positions.push_back(pos);
                        
                        metrics.bid_fills.fetch_add(1, std::memory_order_relaxed);
                        metrics.open_positions.fetch_add(1, std::memory_order_relaxed);
                        metrics.fills_last_minute.fetch_add(1, std::memory_order_relaxed);
                        
                        // Track fill timestamp for post-fill drift calculation
                        {
                            std::lock_guard<std::mutex> lock(metrics.kpi_mutex);
                            metrics.fills_with_drift.push_back({market_data.timestamp_ns, 0.0});
                            if (metrics.fills_with_drift.size() > 1000) {
                                metrics.fills_with_drift.pop_front();
                            }
                        }
                        
                        // Don't modify total_size (market depth), only our_size
                        level.our_size -= fill_size;
                        level.queue_depth--;
                        our_orders[order.order_id].is_active = false;
                    }
                }
            }
        }
        for (auto& [price, level] : ask_levels) {
            // REALISTIC QUEUE POSITION: Reset if price moved away significantly (> 2 ticks) and came back
            double price_distance_from_best = std::abs(market_data.ask_px_00 - price) / 0.25;  // In ticks
            bool is_now_at_best = (price_distance_from_best < 0.5);  // Within 0.5 ticks of best
            
            // Track if price moved away and came back
            if (level.is_at_best_level) {
                // We were at best level - check if price moved away
                double last_distance = std::abs(level.last_best_price - price) / 0.25;
                if (last_distance > 2.0 && !is_now_at_best) {
                    // Price moved away (> 2 ticks) - mark that we're no longer at best
                    level.is_at_best_level = false;
                }
            } else {
                // We were NOT at best level - check if price came back
                if (is_now_at_best) {
                    // Price came back - DON'T reset existing orders' queue positions!
                    // They keep their position (e.g., if they were at position 20, they stay there)
                    // Only reset volume_traded_at_level so NEW orders start fresh
                    // This preserves FIFO ordering: old orders stay ahead, new orders go behind
                    level.volume_traded_at_level = 0;
                    // Existing orders keep their current queue_position - don't reset them!
                }
            }
            
            level.is_at_best_level = is_now_at_best;
            level.last_best_price = market_data.ask_px_00;
            
            // Track volume traded at this level (for queue position estimation)
            // Only count volume when price is at/near our level (within 1 tick)
            double volume_traded_this_tick = 0.0;
            if (std::abs(market_data.trade_price - price) < 0.01 && market_data.trade_size > 0) {
                volume_traded_this_tick = market_data.trade_size;
                level.volume_traded_at_level += static_cast<int>(market_data.trade_size);
                
                // Update queue positions: as volume trades, we move forward in queue
                std::queue<Order> temp_queue;
                while (!level.order_queue.empty()) {
                    Order q_order = level.order_queue.front();
                    level.order_queue.pop();
                    if (our_orders[q_order.order_id].is_active) {
                        int cumulative_volume = level.volume_traded_at_level;
                        q_order.queue_position = std::max(0, q_order.initial_market_depth - cumulative_volume);
                        our_orders[q_order.order_id].queue_position = q_order.queue_position;
                    }
                    temp_queue.push(q_order);
                }
                level.order_queue = temp_queue;
            }
            
            // REALISTIC FILL LOGIC: fill when trade_price >= our ask price (someone buys from us)
            // For EXIT orders (TP/SL): Fill when price trades THROUGH limit (>= for sells, <= for buys)
            // CRITICAL: Only fill if we're at the front of the queue (queue_position <= remaining_trade_size)
            // SELF-TRADE PREVENTION: DISABLED FOR DATA COLLECTION - allow fills even if we have bid at same price
            bool has_bid_at_same_price = false;  // DISABLED: Allow self-trades for data collection
            // if (bid_levels.find(price) != bid_levels.end() && bid_levels[price].our_size > 0) {
            //     has_bid_at_same_price = true;  // We have both bid and ask at same price - prevent self-trade
            // }
            
            // Check if any orders at this level are exit orders that should fill
            bool has_exit_order_that_should_fill = false;
            if (!level.order_queue.empty()) {
                std::queue<Order> temp_check = level.order_queue;
                while (!temp_check.empty()) {
                    Order q_order = temp_check.front();
                    temp_check.pop();
                    if (our_orders[q_order.order_id].is_active && q_order.is_exit_order) {
                        // Exit orders fill when price trades THROUGH them
                        // For bids (buy orders): Fill when trade_price <= order_price (price trades down to/through our buy)
                        // For asks (sell orders): Fill when trade_price >= order_price (price trades up to/through our sell)
                        bool should_fill_exit = false;
                        if (q_order.is_bid) {
                            // Buy order (SHORT exit): Fill when price <= order price
                            should_fill_exit = (market_data.trade_price <= q_order.price);
                        } else {
                            // Sell order (LONG exit): Fill when price >= order price
                            should_fill_exit = (market_data.trade_price >= q_order.price);
                        }
                        if (should_fill_exit && market_data.trade_size > 0) {
                            has_exit_order_that_should_fill = true;
                            break;
                        }
                    }
                }
            }
            
            // Fill condition: Entry orders need exact match, Exit orders fill when price trades through
            bool should_fill = (level.our_size > 0 && market_data.trade_size > 0 && !has_bid_at_same_price) &&
                              ((market_data.trade_price >= price) || has_exit_order_that_should_fill);
            
            if (should_fill) {
                double remaining_trade_size = market_data.trade_size;
                
                while (!level.order_queue.empty() && level.our_size > 0 && remaining_trade_size > 0) {
                    Order order = level.order_queue.front();
                    level.order_queue.pop();
                    
                    if (our_orders[order.order_id].is_active) {
                        // REALISTIC: Only fill if we're at the front of the queue
                        // queue_position = 0 means we're at the front, negative means we should have been filled already
                        if (order.queue_position > remaining_trade_size) {
                            // Not enough volume to reach us - put order back and break
                            level.order_queue.push(order);
                            break;
                        }
                        
                        // REALISTIC: Partial fills - a 50-contract trade can partially fill multiple orders
                        // We're at the front (or should have been filled) - fill us
                        double fill_size = std::min(remaining_trade_size, static_cast<double>(order.contracts));
                        remaining_trade_size -= fill_size;
                        
                        // Check if order is fully filled or partially filled
                        bool is_fully_filled = (fill_size >= static_cast<double>(order.contracts) - 0.001);  // Allow floating point tolerance
                        
                        if (!is_fully_filled) {
                            // PARTIAL FILL: Reduce order size and put it back in queue (FIFO preserved)
                            order.contracts -= static_cast<int>(fill_size);
                            our_orders[order.order_id].contracts = order.contracts;
                            level.our_size -= fill_size;
                            level.total_size -= fill_size;
                            // Queue position updated: we consumed fill_size, so move forward
                            order.queue_position = std::max(0, order.queue_position - static_cast<int>(fill_size));
                            our_orders[order.order_id].queue_position = order.queue_position;
                            level.order_queue.push(order);  // Put back for remaining size
                            // Continue to next order in queue (partial fill processed)
                            continue;
                        }
                        
                        // FULL FILL: Order completely filled, remove from queue
                        level.our_size -= fill_size;
                        level.total_size -= fill_size;
                        level.queue_depth--;
                        
                        // Check if this is an exit order
                        if (order.is_exit_order) {
                            // Exit orders fill when price trades THROUGH them
                            // For bids (buy orders): Fill when trade_price <= order_price
                            // For asks (sell orders): Fill when trade_price >= order_price
                            bool should_fill_this_exit = false;
                            if (order.is_bid) {
                                // Buy order (SHORT exit): Fill when price <= order price
                                should_fill_this_exit = (market_data.trade_price <= order.price);
                            } else {
                                // Sell order (LONG exit): Fill when price >= order price
                                should_fill_this_exit = (market_data.trade_price >= order.price);
                            }
                            
                            if (!should_fill_this_exit) {
                                // Exit order shouldn't fill yet - put it back
                                level.order_queue.push(order);
                                break;
                            }
                            
                            // Exit order filled - store actual fill price and mark as filled
                            size_t pos_idx = order.position_index;
                            if (pos_idx < active_positions.size()) {
                                active_positions[pos_idx].exit_actual_fill_price = market_data.trade_price;
                            }
                            our_orders[order.order_id].is_active = false;
                            continue;  // Skip entry position creation
                        }
                        
                        // Calculate entry features at entry time (same as longs)
                        double entry_ofi = 0.0;
                        if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
                            entry_ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                                       (market_data.bid_sz_00 + market_data.ask_sz_00);
                        }
                        double entry_momentum = 0.0;
                        if (mid_prices.size() >= 20) {
                            double short_avg = 0.0, long_avg = 0.0;
                            auto it = mid_prices.end();
                            for (int i = 0; i < 5 && it != mid_prices.begin(); ++i) {
                                it--;
                                short_avg += *it;
                            }
                            for (int i = 0; i < 15 && it != mid_prices.begin(); ++i) {
                                it--;
                                long_avg += *it;
                            }
                            if (short_avg > 0 && long_avg > 0) {
                                entry_momentum = (short_avg / 5.0 - long_avg / 15.0) / 0.25;
                            }
                        }
                        double entry_spread = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
                        double entry_volatility = 0.0;
                        if (mid_prices.size() >= 10) {
                            double sum = 0.0, sum_sq = 0.0;
                            int count = 0;
                            for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                                sum += *it;
                                sum_sq += (*it) * (*it);
                                count++;
                            }
                            if (count > 1) {
                                double mean = sum / count;
                                entry_volatility = std::sqrt((sum_sq / count) - (mean * mean));
                            }
                        }
                        double entry_qi_bid = 0.0, entry_qi_ask = 0.0;
                        double total_bid = market_data.bid_sz_00, total_ask = market_data.ask_sz_00;
                        for (int i = 0; i < 9; ++i) {
                            total_bid += market_data.bid_sz[i];
                            total_ask += market_data.ask_sz[i];
                        }
                        if (total_bid + total_ask > 0) {
                            entry_qi_bid = total_bid / (total_bid + total_ask);
                            entry_qi_ask = total_ask / (total_bid + total_ask);
                        }
                        
                        Position pos;
                        pos.entry_price = market_data.trade_price;  // CRITICAL FIX: Use actual fill price, not limit order price!
                        pos.contracts = static_cast<int>(fill_size);
                        pos.entry_timestamp = market_data.timestamp_ns;
                        pos.is_long = false;
                        pos.hold_intervals = 0;
                        pos.first_profit_target_interval = -1;  // -1 = not reached yet
                        pos.mae_at_first_profit_target = 0.0;  // Will be set when we first reach +1 tick
                        pos.max_adverse_excursion = 0.0;
                        pos.max_favorable_excursion = 0.0;
                        // Diagnostics: track what filled us and market state
                        pos.fill_trade_side = market_data.trade_side;  // 'B' = buyer hit our ask (bad), 'S' = seller (unusual)
                        pos.entry_bid_px = market_data.bid_px_00;
                        pos.entry_ask_px = market_data.ask_px_00;
                        pos.next_interval_ask_px = 0.0;  // Will be set on next interval
                        // Store entry features
                        pos.entry_ofi = entry_ofi;
                        pos.entry_momentum = entry_momentum;
                        pos.entry_spread = entry_spread;
                        pos.entry_volatility = entry_volatility;
                        pos.entry_qi_bid = entry_qi_bid;
                        pos.entry_qi_ask = entry_qi_ask;
                        active_positions.push_back(pos);
                        
                        metrics.ask_fills.fetch_add(1, std::memory_order_relaxed);
                        metrics.open_positions.fetch_add(1, std::memory_order_relaxed);
                        
                        // Don't modify total_size (market depth), only our_size
                        level.our_size -= fill_size;
                        level.queue_depth--;
                        our_orders[order.order_id].is_active = false;
                    }
                }
            }
        }
    }
    
    void manage_positions(const RealMarketData& market_data) {
        std::vector<Position> remaining_positions;
        
        for (size_t pos_idx = 0; pos_idx < active_positions.size(); ++pos_idx) {
            auto& pos = active_positions[pos_idx];
            pos.hold_intervals++;
            
            // Track next interval price for shorts (to see immediate movement)
            if (!pos.is_long && pos.hold_intervals == 1 && pos.next_interval_ask_px == 0.0) {
                pos.next_interval_ask_px = market_data.ask_px_00;
            }
            
            // Track drawdown during hold
            double current_pnl = 0.0;
            if (pos.is_long) {
                // LONG: entered at bid (bought), exit at ask (sell) → profit = current_ask - entry_price
                // FIXED: Use ask_px_00 (where we'd sell), NOT bid_px_00
                current_pnl = (market_data.ask_px_00 - pos.entry_price) / 0.25;
            } else {
                // SHORT: entered at ask (sold high), exit at bid (buy back low) → profit = entry_price - current_bid
                // FIXED: Use bid_px_00 (where we'd buy back), NOT ask_px_00
                current_pnl = (pos.entry_price - market_data.bid_px_00) / 0.25;
            }
            
            // Track worst drawdown
            if (current_pnl < pos.max_adverse_excursion) {
                pos.max_adverse_excursion = current_pnl;
            }
            
            // Track best move
            if (current_pnl > pos.max_favorable_excursion) {
                pos.max_favorable_excursion = current_pnl;
            }
            
            // CRITICAL FIX: Track FIRST interval where we reached +1 tick (for ML training, NO LEAKAGE)
            // Only set once (when first crossing +1 tick threshold)
            if (pos.first_profit_target_interval == -1 && current_pnl >= 1.0) {
                pos.first_profit_target_interval = pos.hold_intervals;
                pos.mae_at_first_profit_target = pos.max_adverse_excursion;  // Capture MAE at this exact moment
            }
            
            // VOLATILITY-ADJUSTED STOP LOSS: Wider stops in volatile periods
            double volatility_multiplier = 1.0;
            if (mid_prices.size() >= 20) {
                // Calculate recent volatility (std dev of price changes)
                std::vector<double> price_changes;
                for (size_t i = mid_prices.size() - 20; i < mid_prices.size() - 1; ++i) {
                    double change = std::abs(mid_prices[i+1] - mid_prices[i]) / 0.25;  // Ticks
                    price_changes.push_back(change);
                }
                if (price_changes.size() > 1) {
                    double mean = 0.0;
                    for (double chg : price_changes) mean += chg;
                    mean /= price_changes.size();
                    double variance = 0.0;
                    for (double chg : price_changes) variance += (chg - mean) * (chg - mean);
                    double std_dev = std::sqrt(variance / (price_changes.size() - 1));
                    
                    // DISABLED: Volatility multiplier widens stops when vol is HIGH (dangerous!)
                    // Instead, we use tighter stops in high vol to prevent catastrophic losses
                    // volatility_multiplier = 1.0;  // Keep at 1.0 (no widening)
                    if (std_dev > 3.0) {
                        // Only slight widening in EXTREME volatility (max 1.5x, not 4x!)
                        volatility_multiplier = std::min(1.5, 1.0 + (std_dev - 3.0) * 0.1);  // Max 1.5x
                    }
                }
            }
            
            // Exit conditions: TIMEOUT ONLY (Market Making Strategy)
            // Market making on ultra-short timeframes (10s) doesn't need TP/SL:
            // - Timeout IS the risk management tool (10 seconds = built-in stop)
            // - TP/SL cuts profits short on noise or prevents mean reversion
            // - Market making relies on spread capture, not directional moves
            // - Time-based exits allow full profit capture from inventory flow
            bool hit_profit_target = false;  // NOT USED: Timeout-only exits
            bool hit_stop_loss = false;  // NOT USED: Timeout-only exits
            
            bool max_hold_reached;
            if (pos.is_long) {
                // LONG: Exit when hold cap OR long timeout fires (matches short behavior)
                uint64_t elapsed_ms = (market_data.timestamp_ns - pos.entry_timestamp) / 1000000ULL;
                max_hold_reached = (pos.hold_intervals >= thresholds.max_hold_intervals) ||
                                  (elapsed_ms >= static_cast<uint64_t>(thresholds.long_timeout_ms));
            } else {
                // SHORT: Exit on timeout
                uint64_t elapsed_ms = (market_data.timestamp_ns - pos.entry_timestamp) / 1000000ULL;
                max_hold_reached = (pos.hold_intervals >= thresholds.max_hold_intervals) || 
                                  (elapsed_ms >= static_cast<uint64_t>(thresholds.short_timeout_ms));
            }
            
            // Exit ONLY on timeout (no TP/SL)
            bool should_exit = max_hold_reached;
            
            // SMART EARLY EXIT LOGIC - DISABLED FOR DATA COLLECTION
            // We want to hold trades until timeout to see full profit potential
            // DISABLED: All early exits - only timeout exits
            
            // DISABLED: Profit reversal exit
            // if (!should_exit && pos.max_favorable_excursion >= 1.0 && current_pnl < 0.0) {
            //     should_exit = true;
            // }
            
            // Measure exit decision latency (hot path)
            auto exit_start = std::chrono::high_resolution_clock::now();
            
            // Record exit decision latency (hot path)
            auto exit_end = std::chrono::high_resolution_clock::now();
            auto exit_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(exit_end - exit_start).count();
            
            uint64_t current_total_exit = metrics.total_exit_decision_time.load(std::memory_order_relaxed);
            metrics.total_exit_decision_time.store(current_total_exit + exit_latency_ns, std::memory_order_relaxed);
            metrics.exit_decision_count.fetch_add(1, std::memory_order_relaxed);
            
            uint64_t current_max_exit = metrics.max_exit_decision_time.load(std::memory_order_relaxed);
            if (exit_latency_ns > current_max_exit) {
                metrics.max_exit_decision_time.store(exit_latency_ns, std::memory_order_relaxed);
            }
            
            if (should_exit) {
                // REALISTIC EXIT: Place limit order at TP/SL price (queued, same fill logic as entries)
                // Calculate target exit price based on TP/SL
                double exit_target_price = 0.0;
                bool is_tp_exit = false;
                
                // SLIPPAGE DETECTION: If market has already gapped through SL, use market order (aggressive limit)
                // This ensures we exit at best available price while staying realistic
                bool use_market_order = false;
                bool use_market_order_slippage = false;
                double slippage_detected = 0.0;
                
                // TIMEOUT ONLY: All exits are timeout (TP/SL disabled)
                if (pos.is_long) {
                    // LONG Timeout: use MARKET ORDER
                    double base_price = market_data.ask_px_00;
                    double spread_ticks = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
                    double volatility_factor = 0.0;
                    if (mid_prices.size() >= 10) {
                        double sum_sq = 0.0, mean = 0.0;
                        for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                            mean += *it;
                        }
                        mean /= std::min(10, (int)mid_prices.size());
                        for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                            sum_sq += (*it - mean) * (*it - mean);
                        }
                        volatility_factor = std::sqrt(sum_sq / std::min(10, (int)mid_prices.size())) / 0.25;
                    }
                    double slippage_ticks = std::min(2.0, (spread_ticks * 0.1) + (volatility_factor * 0.05));
                    exit_target_price = base_price + (slippage_ticks * 0.25);
                    is_tp_exit = false;
                    use_market_order = true;
                } else {
                    // SHORT Timeout: use MARKET ORDER
                    double base_price = market_data.bid_px_00;
                    double spread_ticks = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
                    double volatility_factor = 0.0;
                    if (mid_prices.size() >= 10) {
                        double sum_sq = 0.0, mean = 0.0;
                        for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                            mean += *it;
                        }
                        mean /= std::min(10, (int)mid_prices.size());
                        for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                            sum_sq += (*it - mean) * (*it - mean);
                        }
                        volatility_factor = std::sqrt(sum_sq / std::min(10, (int)mid_prices.size())) / 0.25;
                    }
                    double slippage_ticks = std::min(2.0, (spread_ticks * 0.1) + (volatility_factor * 0.05));
                    exit_target_price = base_price - (slippage_ticks * 0.25);
                    is_tp_exit = false;
                    use_market_order = true;
                }
                
                // Determine if we should use market order (timeout OR slippage)
                use_market_order = use_market_order || use_market_order_slippage;
                
                // For market orders (timeouts/slippage), fill immediately instead of queuing
                if (use_market_order) {
                    // Market order - fill at exit_target_price which includes slippage
                    double exit_price = exit_target_price;  // Use target price (includes slippage)
                    // Calculate actual slippage (difference from best bid/ask)
                    double base_price = pos.is_long ? market_data.ask_px_00 : market_data.bid_px_00;
                    double slippage_ticks = 0.0;
                    if (pos.is_long) {
                        // LONG: Sell at ask, slippage makes exit_price higher (worse) = positive slippage
                        slippage_ticks = (exit_price - base_price) / 0.25;
                    } else {
                        // SHORT: Buy at bid, slippage makes exit_price lower (worse) = negative slippage
                        slippage_ticks = (base_price - exit_price) / 0.25;  // Negative when exit_price < base_price
                    }
                    
                    // Calculate exit-time features for analysis
                    double exit_ofi = 0.0;
                    if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
                        exit_ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                                   (market_data.bid_sz_00 + market_data.ask_sz_00);
                    }
                    double exit_spread = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
                    double exit_volatility = 0.0;
                    if (mid_prices.size() >= 10) {
                        double sum = 0.0, sum_sq = 0.0;
                        int count = 0;
                        for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                            sum += *it;
                            sum_sq += (*it) * (*it);
                            count++;
                        }
                        if (count > 1) {
                            double mean = sum / count;
                            exit_volatility = std::sqrt((sum_sq / count) - (mean * mean));
                        }
                    }
                    
                    // Close position immediately (market order fills instantly)
                    double profit_ticks;
                    if (pos.is_long) {
                        profit_ticks = (exit_price - pos.entry_price) / 0.25;
                    } else {
                        profit_ticks = (pos.entry_price - exit_price) / 0.25;
                    }

                    double gross_pnl = profit_ticks * 12.50 * pos.contracts;
                    double costs = 0.125 * pos.contracts * 2;
                    double net_pnl = gross_pnl - costs;
                    
                    // Get exit order fill time
                    uint64_t exit_order_fill_ns = market_data.timestamp_ns;
                    int exit_order_intervals_to_fill = 0;
                    if (pos.exit_order_placement_ns > 0) {
                        exit_order_intervals_to_fill = (int)((exit_order_fill_ns - pos.exit_order_placement_ns) / 100000000ULL);
                    } else {
                        exit_order_intervals_to_fill = 0;  // Market order fills immediately
                    }
                    
                    metrics.total_trades.fetch_add(1, std::memory_order_relaxed);
                    
                    std::string trade_detail = (pos.is_long ? "LONG" : "SHORT") + 
                                             std::string(" Exit: Entry=") + std::to_string(pos.entry_price) + 
                                             ", Exit=" + std::to_string(exit_price) + 
                                             ", Target=" + std::to_string(exit_target_price) +
                                             ", Slippage=" + std::to_string(slippage_ticks) + "ticks" +
                                             ", Hold=" + std::to_string(pos.hold_intervals) +
                                             ", Ticks=" + std::to_string(profit_ticks) + 
                                             ", MAE=" + std::to_string(pos.max_adverse_excursion) + 
                                             ", MFE=" + std::to_string(pos.max_favorable_excursion) +
                                             ", P&L=$" + std::to_string(net_pnl);
                    
                    if (!pos.is_long) {
                        trade_detail += ", FillSide=" + std::string(1, pos.fill_trade_side);
                        trade_detail += ", EntrySpread=" + std::to_string(pos.entry_ask_px - pos.entry_bid_px);
                        if (pos.next_interval_ask_px > 0.0) {
                            double immediate_move = (pos.next_interval_ask_px - pos.entry_ask_px) / 0.25;
                            trade_detail += ", NextAskMove=" + std::to_string(immediate_move) + "ticks";
                        }
                    }
                    
                    metrics.trade_details.push_back(trade_detail);
                    metrics.trade_pnls.push_back(net_pnl);

                    // Per-side TP/SL/Timeout accounting - ALL EXITS ARE TIMEOUT (TP/SL disabled)
                    if (pos.is_long) {
                        metrics.long_timeout.fetch_add(1, std::memory_order_relaxed);  // All exits are timeout
                    } else {
                        metrics.short_timeout.fetch_add(1, std::memory_order_relaxed);  // All exits are timeout
                    }
                    
                    // Store structured trade data
                    TradeData trade;
                    trade.is_long = pos.is_long;
                    trade.entry_price = pos.entry_price;
                    trade.exit_price = exit_price;
                    trade.exit_target_price = exit_target_price;
                    trade.slippage_ticks = slippage_ticks;
                    trade.hold_intervals = pos.hold_intervals;
                    trade.first_profit_target_interval = (pos.first_profit_target_interval == -1) ? 
                        pos.hold_intervals : pos.first_profit_target_interval;
                    trade.mae_at_first_profit_target = (pos.first_profit_target_interval == -1) ? 
                        pos.max_adverse_excursion : pos.mae_at_first_profit_target;
                    trade.profit_ticks = profit_ticks;
                    trade.mae = pos.max_adverse_excursion;
                    trade.mfe = pos.max_favorable_excursion;
                    trade.net_pnl = net_pnl;
                    trade.entry_ofi = pos.entry_ofi;
                    trade.entry_momentum = pos.entry_momentum;
                    trade.entry_spread = pos.entry_spread;
                    trade.entry_volatility = pos.entry_volatility;
                    trade.entry_qi_bid = pos.entry_qi_bid;
                    trade.entry_qi_ask = pos.entry_qi_ask;
                    trade.exit_ofi = exit_ofi;
                    trade.exit_spread = exit_spread;
                    trade.exit_volatility = exit_volatility;
                    trade.exit_queue_position = 0;  // Market order = queue position 0
                    trade.exit_order_placement_ns = market_data.timestamp_ns;
                    trade.exit_order_fill_ns = exit_order_fill_ns;
                    trade.exit_was_slippage = (slippage_ticks != 0.0);
                    trade.exit_is_tp = false;  // All exits are timeout (TP/SL disabled)
                    trade.exit_order_intervals_to_fill = exit_order_intervals_to_fill;
                    metrics.trades_for_ml.push_back(trade);
                
                    // Update aggregate metrics
                    if (net_pnl > 0) {
                        metrics.profitable_trades.fetch_add(1, std::memory_order_relaxed);
                        double current_profit = metrics.total_profit.load(std::memory_order_relaxed);
                        metrics.total_profit.store(current_profit + net_pnl, std::memory_order_relaxed);
                        metrics.consecutive_losses.store(0, std::memory_order_relaxed);
                    } else {
                        metrics.losing_trades.fetch_add(1, std::memory_order_relaxed);
                        double current_loss = metrics.total_loss.load(std::memory_order_relaxed);
                        metrics.total_loss.store(current_loss + net_pnl, std::memory_order_relaxed);
                        metrics.consecutive_losses.fetch_add(1, std::memory_order_relaxed);
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(metrics.returns_mutex);
                        metrics.recent_returns.push_back(profit_ticks);
                        if (metrics.recent_returns.size() > 100) {
                            metrics.recent_returns.pop_front();
                        }
                    }
                    
                    if (pos.is_long) {
                        double current_long_pnl = metrics.long_pnl.load(std::memory_order_relaxed);
                        metrics.long_pnl.store(current_long_pnl + net_pnl, std::memory_order_relaxed);
                        if (net_pnl > 0) {
                            metrics.long_profitable_trades.fetch_add(1, std::memory_order_relaxed);
                            double current_long_profit = metrics.long_profit.load(std::memory_order_relaxed);
                            metrics.long_profit.store(current_long_profit + net_pnl, std::memory_order_relaxed);
                        } else {
                            metrics.long_losing_trades.fetch_add(1, std::memory_order_relaxed);
                            double current_long_loss = metrics.long_loss.load(std::memory_order_relaxed);
                            metrics.long_loss.store(current_long_loss + net_pnl, std::memory_order_relaxed);
                        }
                    } else {
                        double current_short_pnl = metrics.short_pnl.load(std::memory_order_relaxed);
                        metrics.short_pnl.store(current_short_pnl + net_pnl, std::memory_order_relaxed);
                        if (net_pnl > 0) {
                            metrics.short_profitable_trades.fetch_add(1, std::memory_order_relaxed);
                            double current_short_profit = metrics.short_profit.load(std::memory_order_relaxed);
                            metrics.short_profit.store(current_short_profit + net_pnl, std::memory_order_relaxed);
                        } else {
                            metrics.short_losing_trades.fetch_add(1, std::memory_order_relaxed);
                            double current_short_loss = metrics.short_loss.load(std::memory_order_relaxed);
                            metrics.short_loss.store(current_short_loss + net_pnl, std::memory_order_relaxed);
                        }
                    }
                    
                    double current_pnl = metrics.total_pnl.load(std::memory_order_relaxed);
                    metrics.total_pnl.store(current_pnl + net_pnl, std::memory_order_relaxed);
                    
                    double current_costs = metrics.total_costs.load(std::memory_order_relaxed);
                    metrics.total_costs.store(current_costs + costs, std::memory_order_relaxed);
                    
                    metrics.open_positions.fetch_sub(1, std::memory_order_relaxed);
                    
                    // Don't add to remaining_positions
                    continue;
                }
                
                // Check if we already have a pending exit order
                if (pos.pending_exit_order_id == 0) {
                    // Place limit order at exit price (or market order if slippage detected)
                    // LONG: sell at ASK (we're selling, need buyers)
                    // SHORT: buy at BID (we're buying back, need sellers)
                    bool exit_is_bid = !pos.is_long;  // SHORT exits via bid, LONG exits via ask
                    
                    Order exit_order;
                    exit_order.order_id = next_order_id++;
                    exit_order.price = exit_target_price;
                    exit_order.contracts = pos.contracts;
                    exit_order.timestamp_ns = market_data.timestamp_ns;
                    exit_order.is_bid = exit_is_bid;
                    exit_order.is_active = true;
                    exit_order.is_exit_order = true;  // Mark as exit order
                    exit_order.position_index = pos_idx;  // FIXED: Use actual index, not pointer arithmetic
                    
                    // For market orders (slippage), use aggressive queue position (front of queue)
                    if (use_market_order) {
                        exit_order.queue_position = 0;  // Market order = front of queue
                        exit_order.initial_market_depth = 0;
                    } else {
                        // Limit order - queue normally
                        exit_order.queue_position = 0;  // Will be set below
                        exit_order.initial_market_depth = 0;  // Will be set below
                    }
                    
                    // Queue exit order in DOM (same logic as entry orders)
                    if (exit_is_bid) {
                        auto& level = bid_levels[exit_target_price];
                        level.our_size += exit_order.contracts;
                        level.queue_depth++;
                        if (!use_market_order) {
                            // Limit order - calculate queue position
                            int max_existing_position = level.total_size;
                            if (!level.order_queue.empty()) {
                                std::queue<Order> temp_check = level.order_queue;
                                while (!temp_check.empty()) {
                                    Order q_order = temp_check.front();
                                    temp_check.pop();
                                    if (our_orders[q_order.order_id].is_active) {
                                        max_existing_position = std::max(max_existing_position, q_order.queue_position);
                                    }
                                }
                            }
                            exit_order.initial_market_depth = max_existing_position;
                            exit_order.queue_position = max_existing_position;
                        }
                        level.order_queue.push(exit_order);
                    } else {
                        auto& level = ask_levels[exit_target_price];
                        level.our_size += exit_order.contracts;
                        level.queue_depth++;
                        if (!use_market_order) {
                            // Limit order - calculate queue position
                            int max_existing_position = level.total_size;
                            if (!level.order_queue.empty()) {
                                std::queue<Order> temp_check = level.order_queue;
                                while (!temp_check.empty()) {
                                    Order q_order = temp_check.front();
                                    temp_check.pop();
                                    if (our_orders[q_order.order_id].is_active) {
                                        max_existing_position = std::max(max_existing_position, q_order.queue_position);
                                    }
                                }
                            }
                            exit_order.initial_market_depth = max_existing_position;
                            exit_order.queue_position = max_existing_position;
                        }
                        level.order_queue.push(exit_order);
                    }
                    
                    our_orders[exit_order.order_id] = exit_order;
                    
                    // Track pending exit order in position
                    pos.pending_exit_order_id = exit_order.order_id;
                    pos.exit_target_price = exit_target_price;
                    pos.exit_is_tp = false;  // All exits are timeout (TP/SL disabled)
                    pos.exit_order_placement_ns = market_data.timestamp_ns;
                    pos.exit_was_slippage = use_market_order;  // Track if slippage was detected
                    
                    // Don't close position yet - wait for fill in simulate_fills()
                    continue;  // Skip to next position
                } else {
                    // Exit order already pending - wait for fill
                    continue;
                }
            }
            // Check if position has pending exit order that was filled
            if (pos.pending_exit_order_id > 0) {
                auto exit_order_it = our_orders.find(pos.pending_exit_order_id);
                if (exit_order_it != our_orders.end() && !exit_order_it->second.is_active) {
                    // Exit order was filled - close position
                    // Use actual fill price if available (from simulate_fills), otherwise use target price
                    double exit_price = (pos.exit_actual_fill_price > 0.0) ? pos.exit_actual_fill_price : pos.exit_target_price;
                    double slippage_ticks = 0.0;
                    if (pos.exit_actual_fill_price > 0.0) {
                        // Calculate slippage (actual - target)
                        if (pos.is_long) {
                            slippage_ticks = (pos.exit_actual_fill_price - pos.exit_target_price) / 0.25;
                        } else {
                            slippage_ticks = (pos.exit_target_price - pos.exit_actual_fill_price) / 0.25;
                        }
                    }
                    
                    // Calculate exit-time features for analysis
                    double exit_ofi = 0.0;
                    if (market_data.bid_sz_00 + market_data.ask_sz_00 > 0) {
                        exit_ofi = (market_data.bid_sz_00 - market_data.ask_sz_00) / 
                                   (market_data.bid_sz_00 + market_data.ask_sz_00);
                    }
                    double exit_spread = (market_data.ask_px_00 - market_data.bid_px_00) / 0.25;
                    double exit_volatility = 0.0;
                    if (mid_prices.size() >= 10) {
                        double sum = 0.0, sum_sq = 0.0;
                        int count = 0;
                        for (auto it = mid_prices.end() - std::min(10, (int)mid_prices.size()); it != mid_prices.end(); ++it) {
                            sum += *it;
                            sum_sq += (*it) * (*it);
                            count++;
                        }
                        if (count > 1) {
                            double mean = sum / count;
                            exit_volatility = std::sqrt((sum_sq / count) - (mean * mean));
                        }
                    }
                    
                    // Get exit order fill time
                    uint64_t exit_order_fill_ns = market_data.timestamp_ns;
                    int exit_order_intervals_to_fill = 0;
                    if (pos.exit_order_placement_ns > 0) {
                        exit_order_intervals_to_fill = (int)((exit_order_fill_ns - pos.exit_order_placement_ns) / 100000000ULL);  // Approximate intervals
                    }

                    double profit_ticks;
                    if (pos.is_long) {
                        profit_ticks = (exit_price - pos.entry_price) / 0.25;
                    } else {
                        profit_ticks = (pos.entry_price - exit_price) / 0.25;
                    }

                    double gross_pnl = profit_ticks * 12.50 * pos.contracts;
                    double costs = 0.125 * pos.contracts * 2;
                    double net_pnl = gross_pnl - costs;
                    
                    metrics.total_trades.fetch_add(1, std::memory_order_relaxed);
                    
                    std::string trade_detail = (pos.is_long ? "LONG" : "SHORT") + 
                                             std::string(" Exit: Entry=") + std::to_string(pos.entry_price) + 
                                             ", Exit=" + std::to_string(exit_price) + 
                                             ", Target=" + std::to_string(pos.exit_target_price) +
                                             ", Slippage=" + std::to_string(slippage_ticks) + "ticks" +
                                             ", Hold=" + std::to_string(pos.hold_intervals) +
                                             ", Ticks=" + std::to_string(profit_ticks) + 
                                             ", MAE=" + std::to_string(pos.max_adverse_excursion) + 
                                             ", MFE=" + std::to_string(pos.max_favorable_excursion) +
                                             ", P&L=$" + std::to_string(net_pnl);
                    
                    // Add diagnostics to trade detail for shorts
                    if (!pos.is_long) {
                        trade_detail += ", FillSide=" + std::string(1, pos.fill_trade_side);
                        trade_detail += ", EntrySpread=" + std::to_string(pos.entry_ask_px - pos.entry_bid_px);
                        if (pos.next_interval_ask_px > 0.0) {
                            double immediate_move = (pos.next_interval_ask_px - pos.entry_ask_px) / 0.25;
                            trade_detail += ", NextAskMove=" + std::to_string(immediate_move) + "ticks";
                        }
                    }
                    
                    metrics.trade_details.push_back(trade_detail);
                    metrics.trade_pnls.push_back(net_pnl);

                    // Per-side TP/SL/Timeout accounting (TP/SL ENABLED)
                    bool is_tp_exit_local2 = hit_profit_target;
                    bool is_sl_exit_local2 = hit_stop_loss;
                    if (pos.is_long) {
                        if (is_tp_exit_local2) {
                            metrics.long_tp.fetch_add(1, std::memory_order_relaxed);
                        } else if (is_sl_exit_local2) {
                            metrics.long_sl.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            metrics.long_timeout.fetch_add(1, std::memory_order_relaxed);
                        }
                    } else {
                        if (is_tp_exit_local2) {
                            metrics.short_tp.fetch_add(1, std::memory_order_relaxed);
                        } else if (is_sl_exit_local2) {
                            metrics.short_sl.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            metrics.short_timeout.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    
                    // Store structured trade data for ML training with comprehensive statistics
                    TradeData trade;
                    trade.is_long = pos.is_long;
                    trade.entry_price = pos.entry_price;
                    trade.exit_price = exit_price;
                    trade.exit_target_price = pos.exit_target_price;
                    trade.slippage_ticks = slippage_ticks;
                    trade.hold_intervals = pos.hold_intervals;
                    trade.first_profit_target_interval = (pos.first_profit_target_interval == -1) ? 
                        pos.hold_intervals : pos.first_profit_target_interval;
                    trade.mae_at_first_profit_target = (pos.first_profit_target_interval == -1) ? 
                        pos.max_adverse_excursion : pos.mae_at_first_profit_target;
                    trade.profit_ticks = profit_ticks;
                    trade.mae = pos.max_adverse_excursion;
                    trade.mfe = pos.max_favorable_excursion;
                    trade.net_pnl = net_pnl;
                    // Entry features
                    trade.entry_ofi = pos.entry_ofi;
                    trade.entry_momentum = pos.entry_momentum;
                    trade.entry_spread = pos.entry_spread;
                    trade.entry_volatility = pos.entry_volatility;
                    trade.entry_qi_bid = pos.entry_qi_bid;
                    trade.entry_qi_ask = pos.entry_qi_ask;
                    // Exit features (NEW)
                    trade.exit_ofi = exit_ofi;
                    trade.exit_spread = exit_spread;
                    trade.exit_volatility = exit_volatility;
                    trade.exit_queue_position = exit_order_it->second.queue_position;
                    trade.exit_order_placement_ns = pos.exit_order_placement_ns;
                    trade.exit_order_fill_ns = exit_order_fill_ns;
                    trade.exit_was_slippage = pos.exit_was_slippage;
                    trade.exit_is_tp = false;  // All exits are timeout (TP/SL disabled)
                    trade.exit_order_intervals_to_fill = exit_order_intervals_to_fill;
                    metrics.trades_for_ml.push_back(trade);
                
                // Update aggregate metrics
                if (net_pnl > 0) {
                    metrics.profitable_trades.fetch_add(1, std::memory_order_relaxed);
                    double current_profit = metrics.total_profit.load(std::memory_order_relaxed);
                    metrics.total_profit.store(current_profit + net_pnl, std::memory_order_relaxed);
                    // Reset consecutive losses on profit
                    metrics.consecutive_losses.store(0, std::memory_order_relaxed);
                } else {
                    metrics.losing_trades.fetch_add(1, std::memory_order_relaxed);
                    double current_loss = metrics.total_loss.load(std::memory_order_relaxed);
                    metrics.total_loss.store(current_loss + net_pnl, std::memory_order_relaxed);
                    // Track consecutive losses
                    metrics.consecutive_losses.fetch_add(1, std::memory_order_relaxed);
                }
                
                // Track recent returns for Sharpe calculation
                {
                    std::lock_guard<std::mutex> lock(metrics.returns_mutex);
                    metrics.recent_returns.push_back(profit_ticks);
                    if (metrics.recent_returns.size() > 100) {
                        metrics.recent_returns.pop_front();
                    }
                }
                
                // Update per-side metrics
                if (pos.is_long) {
                    double current_long_pnl = metrics.long_pnl.load(std::memory_order_relaxed);
                    metrics.long_pnl.store(current_long_pnl + net_pnl, std::memory_order_relaxed);
                    if (net_pnl > 0) {
                        metrics.long_profitable_trades.fetch_add(1, std::memory_order_relaxed);
                        double current_long_profit = metrics.long_profit.load(std::memory_order_relaxed);
                        metrics.long_profit.store(current_long_profit + net_pnl, std::memory_order_relaxed);
                    } else {
                        metrics.long_losing_trades.fetch_add(1, std::memory_order_relaxed);
                        double current_long_loss = metrics.long_loss.load(std::memory_order_relaxed);
                        metrics.long_loss.store(current_long_loss + net_pnl, std::memory_order_relaxed);
                    }
                } else {
                    double current_short_pnl = metrics.short_pnl.load(std::memory_order_relaxed);
                    metrics.short_pnl.store(current_short_pnl + net_pnl, std::memory_order_relaxed);
                    if (net_pnl > 0) {
                        metrics.short_profitable_trades.fetch_add(1, std::memory_order_relaxed);
                        double current_short_profit = metrics.short_profit.load(std::memory_order_relaxed);
                        metrics.short_profit.store(current_short_profit + net_pnl, std::memory_order_relaxed);
                    } else {
                        metrics.short_losing_trades.fetch_add(1, std::memory_order_relaxed);
                        double current_short_loss = metrics.short_loss.load(std::memory_order_relaxed);
                        metrics.short_loss.store(current_short_loss + net_pnl, std::memory_order_relaxed);
                    }
                }
                
                double current_pnl = metrics.total_pnl.load(std::memory_order_relaxed);
                metrics.total_pnl.store(current_pnl + net_pnl, std::memory_order_relaxed);
                
                double current_costs = metrics.total_costs.load(std::memory_order_relaxed);
                metrics.total_costs.store(current_costs + costs, std::memory_order_relaxed);
                
                metrics.open_positions.fetch_sub(1, std::memory_order_relaxed);
                
                // Position closed - don't add to remaining_positions
                continue;
            }
            // Exit order pending but not filled yet - wait for fill next iteration
            remaining_positions.push_back(pos);
        } else {
            // No exit order pending - add to remaining
            remaining_positions.push_back(pos);
        }
        }
        
        active_positions = remaining_positions;
    }
    
    HOT void run_optimized_simulation(const std::string& filename, int max_intervals = 10000) {
        std::cout << "\n🚀 OPTIMIZED DATABENTO DOM MARKET MAKER" << std::endl;
        std::cout << "Zero-allocation hot path, lock-free data structures" << std::endl;
        std::cout << "========================================" << std::endl;
        
        if (!load_databento_data(filename)) {
            std::cerr << "❌ Failed to load Databento data" << std::endl;
            return;
        }
        
        std::cout << "\n✅ Data loaded successfully!" << std::endl;
        std::cout << "📊 Starting simulation with " << databento_data.size() << " records..." << std::endl;
        std::cout << "🎯 Max intervals: " << max_intervals << std::endl;
        
        if (databento_data.empty()) {
            std::cerr << "❌ ERROR: No data loaded! Cannot run simulation." << std::endl;
            return;
        }
        
        // Reset data index for simulation
        current_data_index = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        RealMarketData market_data;
        int intervals_processed = 0;
        
        std::cout << "🚀 Simulation loop starting..." << std::endl;
        
        // Debug: Check initial state
        std::cout << "🔍 Debug: current_data_index=" << current_data_index << ", databento_data.size()=" << databento_data.size() << std::endl;
        
        int loop_iterations = 0;
        
        // OUTER TRY-CATCH: Wraps entire loop including get_next_market_data() to catch crashes
        try {
        while (true) {
            // Get next market data - NOW INSIDE TRY-CATCH
            try {
                bool got_data = get_next_market_data(market_data);
                
                if (!got_data) {
                    std::cout << "\n⚠️ No more market data available at index " << current_data_index << std::endl;
                    std::cout << "   Loop iteration: " << loop_iterations << std::endl;
                    std::cout << "   intervals_processed: " << intervals_processed << std::endl;
                    std::cout.flush();
                    break;
                }
                
                // Debug: Print after successful get_next_market_data
            } catch (const std::exception& e) {
                std::cerr << "\n❌ CRITICAL: Exception in get_next_market_data() at index " << current_data_index << ": " << e.what() << std::endl;
                std::cerr << "   Exception type: " << typeid(e).name() << std::endl;
                std::cerr << "   Loop iteration: " << loop_iterations << std::endl;
                std::cerr.flush();
                break;
            } catch (...) {
                std::cerr << "\n❌ CRITICAL: Unknown exception in get_next_market_data() at index " << current_data_index << std::endl;
                std::cerr << "   Loop iteration: " << loop_iterations << std::endl;
                std::cerr.flush();
                break;
            }
            
            if (intervals_processed >= max_intervals) {
                std::cout << "\n⚠️ Reached max intervals limit: " << intervals_processed << "/" << max_intervals << std::endl;
                std::cout << "   Loop iteration: " << loop_iterations << std::endl;
                std::cout.flush();
                break;
            }
            
            loop_iterations++;
            
            // Progress logging: Less frequent to reduce I/O overhead
            if (loop_iterations % 50000 == 0) {
                std::cout << "🔄 Loop iteration " << loop_iterations << ", intervals_processed=" << intervals_processed << std::endl;
                std::cout.flush();
            }
            
            try {
            auto decision_start = std::chrono::high_resolution_clock::now();
            
            update_market_data(market_data);
            update_dom_levels(market_data);
            price_predictor.update(market_data);
            maintain_dom_levels(market_data);
            // REALISTIC: Check for fills FIRST, then cancel - prevents canceling orders that are about to fill
            // In reality, if a trade hits while we're at the front, we fill immediately (can't cancel)
            simulate_fills(market_data);
            check_toxicity_cancellation(market_data);  // Cancel remaining orders after fills are processed
            manage_positions(market_data);
            update_canceled_order_tracking(market_data);
            
            auto decision_end = std::chrono::high_resolution_clock::now();
            auto decision_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(decision_end - decision_start).count();
            double decision_time_ms = static_cast<double>(decision_time_ns) / 1000000.0;
            
            if (decision_time_ms > thresholds.max_allowed_latency_ms) {
                metrics.latency_spikes_20ms.fetch_add(1, std::memory_order_relaxed);
            } else if (decision_time_ms > thresholds.p99_9_latency_ms) {
                metrics.latency_spikes_5ms.fetch_add(1, std::memory_order_relaxed);
            }
            
            // LATENCY TRACKING: Only sample every 10th iteration to reduce mutex contention
            // This reduces latency spikes while still tracking performance
            if (loop_iterations % 10 == 0) {
                std::lock_guard<std::mutex> lock(metrics.kpi_mutex);
                metrics.loop_latencies_ms.push_back(decision_time_ms);
                if (metrics.loop_latencies_ms.size() > 1000) {
                    metrics.loop_latencies_ms.pop_front();
                }
            }
            
            uint64_t current_total = metrics.total_decision_time.load(std::memory_order_relaxed);
            metrics.total_decision_time.store(current_total + decision_time_ns, std::memory_order_relaxed);
            metrics.decision_count.fetch_add(1, std::memory_order_relaxed);
            
            uint64_t current_max = metrics.max_decision_time.load(std::memory_order_relaxed);
            if (decision_time_ns > current_max) {
                metrics.max_decision_time.store(decision_time_ns, std::memory_order_relaxed);
            }
            
            int long_positions = 0;
            for (const auto& pos : active_positions) {
                if (pos.is_long) {
                    long_positions++;
                }
            }
            if (long_positions >= thresholds.max_open_positions_per_side && thresholds.enable_longs) {
                // Placement guard already handled in maintain_dom_levels
            }
            
            intervals_processed++;
            
            if (g_shutdown) {
                std::cout << "\n⚠️ SHUTDOWN SIGNAL RECEIVED - Saving partial results..." << std::endl;
                std::cout.flush();
                break;
            }
            
            if (intervals_processed % 10000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                double rate = (elapsed > 0) ? static_cast<double>(intervals_processed) / elapsed : 0.0;
                std::cout << "📊 " << intervals_processed << " intervals (" 
                         << static_cast<int>(rate) << " intervals/sec), "
                         << active_positions.size() << " open, "
                         << metrics.total_trades.load() << " trades, "
                         << metrics.orders_canceled_toxic.load() << " cancels" << std::endl;
                std::cout.flush();
            }
            
            } catch (const std::exception& e) {
                std::cerr << "\n❌ EXCEPTION at interval " << intervals_processed << ": " << e.what() << std::endl;
                std::cerr << "   Market data: bid=" << market_data.bid_px_00 << ", ask=" << market_data.ask_px_00 << std::endl;
                std::cerr << "   Loop iteration: " << loop_iterations << std::endl;
                std::cerr << "   current_data_index: " << current_data_index << std::endl;
                std::cerr.flush();
                break;  // Exit on exception
            } catch (...) {
                std::cerr << "\n❌ UNKNOWN EXCEPTION at interval " << intervals_processed << std::endl;
                std::cerr << "   Loop iteration: " << loop_iterations << std::endl;
                std::cerr << "   current_data_index: " << current_data_index << std::endl;
                std::cerr.flush();
                break;  // Exit on exception
            }
        }
        
        } catch (const std::exception& e) {
            std::cerr << "\n❌ CRITICAL EXCEPTION in simulation loop (outer catch): " << e.what() << std::endl;
            std::cerr << "   Loop iteration: " << loop_iterations << std::endl;
            std::cerr << "   intervals_processed: " << intervals_processed << std::endl;
            std::cerr << "   current_data_index: " << current_data_index << std::endl;
            std::cerr.flush();
            // Continue to cleanup and stats reporting
        } catch (...) {
            std::cerr << "\n❌ CRITICAL UNKNOWN EXCEPTION in simulation loop (outer catch)" << std::endl;
            std::cerr << "   Loop iteration: " << loop_iterations << std::endl;
            std::cerr << "   intervals_processed: " << intervals_processed << std::endl;
            std::cerr << "   current_data_index: " << current_data_index << std::endl;
            std::cerr.flush();
            // Continue to cleanup and stats reporting
        }
        
        std::cout << "\n✅ Simulation loop EXITED!" << std::endl;
        std::cout << "📊 Final intervals_processed=" << intervals_processed << std::endl;
        std::cout << "📊 Final loop_iterations=" << loop_iterations << std::endl;
        std::cout << "📊 Final current_data_index=" << current_data_index << std::endl;
        std::cout << "📊 Reason: get_next_market_data()=" << (current_data_index >= databento_data.size() ? "FALSE (out of data)" : "OK") 
                  << ", intervals_processed=" << intervals_processed << " < max_intervals=" << max_intervals << std::endl;
        std::cout.flush();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double win_rate = 0.0;
        if (!metrics.trade_pnls.empty()) {
            int winning_trades = 0;
            for (double pnl : metrics.trade_pnls) {
                if (pnl > 0) winning_trades++;
            }
            win_rate = static_cast<double>(winning_trades) / metrics.trade_pnls.size();
        }
        
        double sharpe_ratio = 0.0;
        if (metrics.trade_pnls.size() > 1) {
            double mean_pnl = metrics.total_pnl.load() / metrics.trade_pnls.size();
            double variance = 0.0;
            for (double pnl : metrics.trade_pnls) {
                variance += (pnl - mean_pnl) * (pnl - mean_pnl);
            }
            variance /= (metrics.trade_pnls.size() - 1);
            double std_dev = std::sqrt(variance);
            
            if (std_dev > 0) {
                sharpe_ratio = (mean_pnl / std_dev) * std::sqrt(252);
            }
        }
        
        std::cout << "\n🎉 RESULTS (WITH TRAINED ML):" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "📈 SHARPE RATIO: " << std::fixed << std::setprecision(3) << sharpe_ratio << std::endl;
        std::cout << "💰 Total P&L: $" << std::fixed << std::setprecision(2) << metrics.total_pnl.load() << std::endl;
        std::cout << "📊 Total Trades: " << metrics.total_trades.load() << std::endl;
        std::cout << "🎯 Win Rate: " << std::fixed << std::setprecision(1) << (win_rate * 100) << "%" << std::endl;
        std::cout << "📊 Bid Fills: " << metrics.bid_fills.load() << std::endl;
        std::cout << "📊 Ask Fills: " << metrics.ask_fills.load() << std::endl;
        std::cout << "🚨 Orders Canceled (Toxic): " << metrics.orders_canceled_toxic.load() << std::endl;
        std::cout << "🔄 Orders Refilled: " << metrics.orders_refilled.load() << std::endl;
        std::cout << "💸 Total Costs: $" << std::fixed << std::setprecision(2) << metrics.total_costs.load() << std::endl;
        std::cout << "⏱️  Execution Time: " << execution_time.count() << "ms" << std::endl;
        std::cout << "📊 Open Positions: " << metrics.open_positions.load() << std::endl;
        
        // Calculate drawdown statistics from trade details
        double total_mae = 0.0;
        double total_mfe = 0.0;
        int mae_count = 0;
        for (const auto& detail : metrics.trade_details) {
            size_t mae_pos = detail.find("MAE=");
            size_t mfe_pos = detail.find("MFE=");
            if (mae_pos != std::string::npos && mfe_pos != std::string::npos) {
                size_t mae_end = detail.find(",", mae_pos);
                size_t mfe_end = detail.find(",", mfe_pos);
                double mae = std::stod(detail.substr(mae_pos + 4, mae_end - mae_pos - 4));
                double mfe = std::stod(detail.substr(mfe_pos + 4, mfe_end - mfe_pos - 4));
                total_mae += mae;
                total_mfe += mfe;
                mae_count++;
            }
        }
        
        std::cout << "\n⚠️ DRAWDOWN ANALYSIS:" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "📉 Avg Max Adverse Excursion (MAE): " << std::fixed << std::setprecision(2) 
                  << (mae_count > 0 ? total_mae / mae_count : 0.0) << " ticks" << std::endl;
        std::cout << "📈 Avg Max Favorable Excursion (MFE): " << std::fixed << std::setprecision(2) 
                  << (mae_count > 0 ? total_mfe / mae_count : 0.0) << " ticks" << std::endl;
        std::cout << "\n🔍 DETAILED TRADE ANALYSIS:" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "✅ Profitable Trades: " << metrics.profitable_trades.load() << std::endl;
        std::cout << "❌ Losing Trades: " << metrics.losing_trades.load() << std::endl;
        std::cout << "💰 Total Profit: $" << std::fixed << std::setprecision(2) << metrics.total_profit.load() << std::endl;
        std::cout << "💸 Total Loss: $" << std::fixed << std::setprecision(2) << metrics.total_loss.load() << std::endl;
        // Per-side performance breakdown
        int long_trades = metrics.long_profitable_trades.load() + metrics.long_losing_trades.load();
        int short_trades = metrics.short_profitable_trades.load() + metrics.short_losing_trades.load();
        double long_win_rate = (long_trades > 0) ? static_cast<double>(metrics.long_profitable_trades.load()) / long_trades * 100.0 : 0.0;
        double short_win_rate = (short_trades > 0) ? static_cast<double>(metrics.short_profitable_trades.load()) / short_trades * 100.0 : 0.0;
        std::cout << "\n📊 PER-SIDE PERFORMANCE BREAKDOWN:" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "📈 LONG SIDE:" << std::endl;
        std::cout << "   Total P&L: $" << std::fixed << std::setprecision(2) << metrics.long_pnl.load() << std::endl;
        std::cout << "   Trades: " << long_trades << std::endl;
        std::cout << "   Win Rate: " << std::fixed << std::setprecision(1) << long_win_rate << "%" << std::endl;
        std::cout << "   Profitable: " << metrics.long_profitable_trades.load() << " trades, $" 
                  << std::fixed << std::setprecision(2) << metrics.long_profit.load() << std::endl;
        std::cout << "   Losing: " << metrics.long_losing_trades.load() << " trades, $" 
                  << std::fixed << std::setprecision(2) << metrics.long_loss.load() << std::endl;
        std::cout << "   Avg P&L per trade: $" << std::fixed << std::setprecision(2) 
                  << (long_trades > 0 ? metrics.long_pnl.load() / long_trades : 0.0) << std::endl;
        
        std::cout << "\n📉 SHORT SIDE:" << std::endl;
        std::cout << "   Total P&L: $" << std::fixed << std::setprecision(2) << metrics.short_pnl.load() << std::endl;
        std::cout << "   Trades: " << short_trades << std::endl;
        std::cout << "   Win Rate: " << std::fixed << std::setprecision(1) << short_win_rate << "%" << std::endl;
        std::cout << "   Profitable: " << metrics.short_profitable_trades.load() << " trades, $" 
                  << std::fixed << std::setprecision(2) << metrics.short_profit.load() << std::endl;
        std::cout << "   Losing: " << metrics.short_losing_trades.load() << " trades, $" 
                  << std::fixed << std::setprecision(2) << metrics.short_loss.load() << std::endl;
        std::cout << "   Avg P&L per trade: $" << std::fixed << std::setprecision(2) 
                  << (short_trades > 0 ? metrics.short_pnl.load() / short_trades : 0.0) << std::endl;
        
        // Hot path latency stats (critical for live trading)
        uint64_t total_toxicity_ns = metrics.total_toxicity_prediction_time.load(std::memory_order_relaxed);
        uint64_t toxicity_count = metrics.toxicity_prediction_count.load(std::memory_order_relaxed);
        uint64_t max_toxicity_ns = metrics.max_toxicity_prediction_time.load(std::memory_order_relaxed);
        
        uint64_t total_exit_ns = metrics.total_exit_decision_time.load(std::memory_order_relaxed);
        uint64_t exit_count = metrics.exit_decision_count.load(std::memory_order_relaxed);
        uint64_t max_exit_ns = metrics.max_exit_decision_time.load(std::memory_order_relaxed);
        
        // Full loop latency (for reference)
        uint64_t total_decision_ns = metrics.total_decision_time.load(std::memory_order_relaxed);
        uint64_t decision_count = metrics.decision_count.load(std::memory_order_relaxed);
        uint64_t max_decision_ns = metrics.max_decision_time.load(std::memory_order_relaxed);
        
        // Production KPI: Calculate p99.9 latency
        double p99_9_latency_ms = 0.0;
        {
            std::lock_guard<std::mutex> lock(metrics.kpi_mutex);
            if (!metrics.loop_latencies_ms.empty()) {
                std::vector<double> sorted_latencies(metrics.loop_latencies_ms.begin(), metrics.loop_latencies_ms.end());
                std::sort(sorted_latencies.begin(), sorted_latencies.end());
                size_t p99_9_idx = static_cast<size_t>(sorted_latencies.size() * 0.999);
                if (p99_9_idx < sorted_latencies.size()) {
                    p99_9_latency_ms = sorted_latencies[p99_9_idx];
                }
            }
        }
        
        // Production KPI: Calculate rolling toxicity % (FIXED: Check both long AND short thresholds)
        int toxic_count = 0;
        {
            std::lock_guard<std::mutex> lock(metrics.kpi_mutex);
            for (double tox : metrics.toxicity_samples) {
                // Count as toxic if above EITHER threshold (longs OR shorts)
                if (tox >= thresholds.toxic_cancel_threshold || tox >= thresholds.short_tox_threshold) {
                    toxic_count++;
                }
            }
            if (!metrics.toxicity_samples.empty()) {
                metrics.rolling_toxicity_percent.store(
                    static_cast<int>((static_cast<double>(toxic_count) / metrics.toxicity_samples.size()) * 100.0),
                    std::memory_order_relaxed
                );
            }
        }
        
        std::cout << "\n⚡ HOT PATH LATENCY (CRITICAL FOR LIVE TRADING):" << std::endl;
        std::cout << "============================================================" << std::endl;
        
        if (toxicity_count > 0) {
            double avg_toxicity_us = static_cast<double>(total_toxicity_ns) / toxicity_count / 1000.0;
            double max_toxicity_us = static_cast<double>(max_toxicity_ns) / 1000.0;
            std::cout << "🧠 ML Toxicity Prediction:" << std::endl;
            std::cout << "   Avg: " << std::fixed << std::setprecision(2) << avg_toxicity_us << " μs" << std::endl;
            std::cout << "   Max: " << std::fixed << std::setprecision(2) << max_toxicity_us << " μs" << std::endl;
            std::cout << "   Predictions: " << toxicity_count << std::endl;
        }
        
        if (exit_count > 0) {
            double avg_exit_us = static_cast<double>(total_exit_ns) / exit_count / 1000.0;
            double max_exit_us = static_cast<double>(max_exit_ns) / 1000.0;
            std::cout << "🚪 Exit Decision:" << std::endl;
            std::cout << "   Avg: " << std::fixed << std::setprecision(2) << avg_exit_us << " μs" << std::endl;
            std::cout << "   Max: " << std::fixed << std::setprecision(2) << max_exit_us << " μs" << std::endl;
            std::cout << "   Decisions: " << exit_count << std::endl;
        }
        
        if (decision_count > 0) {
            double avg_decision_us = static_cast<double>(total_decision_ns) / decision_count / 1000.0;
            double max_decision_us = static_cast<double>(max_decision_ns) / 1000.0;
            std::cout << "\n📊 Full Loop Latency (reference):" << std::endl;
            std::cout << "   Avg: " << std::fixed << std::setprecision(2) << avg_decision_us << " μs" << std::endl;
            std::cout << "   Max: " << std::fixed << std::setprecision(2) << max_decision_us << " μs" << std::endl;
            std::cout << "   Intervals: " << decision_count << std::endl;
        }
        
        // Production KPIs
        std::cout << "\n🏭 PRODUCTION KPIs:" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "📊 Latency Spikes > 5ms: " << metrics.latency_spikes_5ms.load() << std::endl;
        std::cout << "🚨 Latency Spikes > 20ms: " << metrics.latency_spikes_20ms.load() << std::endl;
        std::cout << "📈 p99.9 Latency: " << std::fixed << std::setprecision(2) << p99_9_latency_ms << " ms" << std::endl;
        std::cout << "⚗️  Rolling Toxicity %: " << metrics.rolling_toxicity_percent.load() << "%" << std::endl;
        std::cout << "📊 Fills Last Minute: " << metrics.fills_last_minute.load() << std::endl;
        if (p99_9_latency_ms > thresholds.p99_9_latency_ms) {
            std::cout << "⚠️  WARNING: p99.9 latency exceeds SLA limit (" << thresholds.p99_9_latency_ms << " ms)!" << std::endl;
        }
        
        // Entry-state summary
        std::cout << "\n📋 ENTRY-STATE STATS:" << std::endl;
        std::cout << "============================================================" << std::endl;
        // Use actual sample counts (every 20th), not total entry attempts
        auto ls = std::max(1, metrics.sampled_long_count); // avoid div by 0
        auto ss = std::max(1, metrics.sampled_short_count);
        double long_ofi = metrics.sum_long_ofi / ls;
        double long_mom = metrics.sum_long_mom / ls;
        double long_spread = metrics.sum_long_spread_ticks / ls;
        double long_qb = metrics.sum_long_qi_bid / ls;
        double long_qa = metrics.sum_long_qi_ask / ls;
        double short_ofi = metrics.sum_short_ofi / ss;
        double short_mom = metrics.sum_short_mom / ss;
        double short_spread = metrics.sum_short_spread_ticks / ss;
        double short_qb = metrics.sum_short_qi_bid / ss;
        double short_qa = metrics.sum_short_qi_ask / ss;
        std::cout << "LONG: attempts=" << metrics.sampled_long_entries
                  << " (sampled=" << metrics.sampled_long_count << ")"
                  << ", ofi=" << std::fixed << std::setprecision(3) << long_ofi
                  << ", mom_ticks=" << long_mom
                  << ", spread_ticks=" << long_spread
                  << ", qi_bid=" << long_qb
                  << ", qi_ask=" << long_qa << std::endl;
        std::cout << "SHORT: attempts=" << metrics.sampled_short_entries
                  << " (sampled=" << metrics.sampled_short_count << ")"
                  << ", ofi=" << short_ofi
                  << ", mom_ticks=" << short_mom
                  << ", spread_ticks=" << short_spread
                  << ", qi_bid=" << short_qb
                  << ", qi_ask=" << short_qa << std::endl;
        std::cout << "Δ(SHORT-LONG): ofi=" << (short_ofi - long_ofi)
                  << ", mom_ticks=" << (short_mom - long_mom)
                  << ", spread_ticks=" << (short_spread - long_spread)
                  << ", qi_bid=" << (short_qb - long_qb)
                  << ", qi_ask=" << (short_qa - long_qa) << std::endl;

        uint64_t bid_depth_samples = metrics.bid_level_depth_samples.load(std::memory_order_relaxed);
        uint64_t ask_depth_samples = metrics.ask_level_depth_samples.load(std::memory_order_relaxed);
        double avg_bid_depth = bid_depth_samples > 0
            ? metrics.bid_level_depth_sum.load(std::memory_order_relaxed) / static_cast<double>(bid_depth_samples)
            : 0.0;
        double avg_ask_depth = ask_depth_samples > 0
            ? metrics.ask_level_depth_sum.load(std::memory_order_relaxed) / static_cast<double>(ask_depth_samples)
            : 0.0;
        std::cout << "\n📥 AVERAGE AVAILABLE DEPTH BEFORE PLACEMENT:" << std::endl;
        std::cout << "   Bid-side depth (others ahead): " << std::fixed << std::setprecision(2)
                  << avg_bid_depth << " contracts (samples=" << bid_depth_samples << ")" << std::endl;
        std::cout << "   Ask-side depth (others ahead): " << std::fixed << std::setprecision(2)
                  << avg_ask_depth << " contracts (samples=" << ask_depth_samples << ")" << std::endl;

        // Per-side TP/SL/Timeout summary
        std::cout << "\n🎯 EXIT DISTRIBUTION (PER SIDE):" << std::endl;
        std::cout << "============================================================" << std::endl;
        int ltp = metrics.long_tp.load(), lsl = metrics.long_sl.load(), lto = metrics.long_timeout.load();
        int stp = metrics.short_tp.load(), ssl = metrics.short_sl.load(), sto = metrics.short_timeout.load();
        std::cout << "LONG: TP=" << ltp << ", SL=" << lsl << ", TIMEOUT=" << lto << std::endl;
        std::cout << "SHORT: TP=" << stp << ", SL=" << ssl << ", TIMEOUT=" << sto << std::endl;
        std::cout << "\n🔍 SHORT ENTRY GATE DIAGNOSTICS:" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Ask placements: " << metrics.ask_placements.load() << std::endl;
        std::cout << "⚠️  GATE STATUS: DISABLED (Blind Placement Mode)" << std::endl;
        std::cout << "   Strategy: Place orders freely, ML toxicity cancellation filters bad ones" << std::endl;
        std::cout << "Skipped by gate: " << metrics.ask_skipped_by_gate.load() << " (gate not active)" << std::endl;
        std::cout << "Debug bypassed: " << metrics.ask_debug_bypass.load() << std::endl;
        std::cout << "\nGate failure reasons (not checked - gate disabled):" << std::endl;
        std::cout << "  Spread > 5.0 ticks: " << metrics.gate_fail_spread.load() << " (only extreme spreads checked)" << std::endl;
        std::cout << "  OFI >= 0.0 (no sell pressure): " << metrics.gate_fail_ofi.load() << " (DISABLED)" << std::endl;
        std::cout << "  Momentum > 0.5 ticks: " << metrics.gate_fail_mom.load() << " (DISABLED)" << std::endl;
        std::cout << "  QI ask > 0.70: " << metrics.gate_fail_qi.load() << " (DISABLED)" << std::endl;
        std::cout << "  Toxicity >= threshold: " << metrics.gate_fail_tox.load() << " (DISABLED - handled by cancellation)" << std::endl;
        
        // Time frame calculation (calibrated from actual CSV data: 10K intervals = 10.09 hours)
        // Measured rate: 10,000 intervals / 10.09 hours = ~991 intervals/hour
        double intervals_per_hour = 10000.0 / 10.09;
        double estimated_hours = static_cast<double>(intervals_processed) / intervals_per_hour;
        double estimated_days = estimated_hours / 24.0;
        double estimated_trading_days = estimated_hours / 6.5;  // ~6.5 hours per trading day (9:30 AM - 4:00 PM ET)
        std::cout << "\n⏱️  TIME FRAME (MEASURED FROM CSV):" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "📊 Intervals Processed: " << intervals_processed << std::endl;
        std::cout << "📊 Estimated Market Time: " << std::fixed << std::setprecision(1) << estimated_hours << " hours" << std::endl;
        std::cout << "📊 Estimated Calendar Days: " << std::fixed << std::setprecision(1) << estimated_days << " days" << std::endl;
        std::cout << "📊 Estimated Trading Days: " << std::fixed << std::setprecision(1) << estimated_trading_days << " days" << std::endl;
        std::cout << "   (Measured: 10K intervals = 10.09 hours from actual CSV timestamps)" << std::endl;
        
        std::cout << "\n📊 FIRST 10 TRADES:" << std::endl;
        std::cout << "============================================================" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), metrics.trade_details.size()); ++i) {
            std::cout << metrics.trade_details[i] << std::endl;
        }
        
        // Export trades to CSV for ML training (with short diagnostics)
        std::ofstream trades_csv("trades_for_exit_training.csv");
        trades_csv << "direction,entry_price,exit_price,exit_target_price,slippage_ticks,hold_intervals,first_profit_target_interval,mae_at_first_profit_target,profit_ticks,MAE,MFE,net_pnl,"
                   << "entry_ofi,entry_momentum,entry_spread,entry_volatility,entry_qi_bid,entry_qi_ask,"
                   << "exit_ofi,exit_spread,exit_volatility,exit_queue_position,exit_order_placement_ns,exit_order_fill_ns,exit_was_slippage,exit_is_tp,exit_order_intervals_to_fill\n";
        
        // Also export detailed short diagnostics
        std::ofstream short_diag_csv("short_trade_diagnostics.csv");
        short_diag_csv << "entry_price,exit_price,hold_intervals,profit_ticks,fill_trade_side,entry_spread,entry_bid,entry_ask,next_ask,immediate_move_ticks\n";
        
        int short_count = 0;
        for (const auto& trade : metrics.trades_for_ml) {
            trades_csv << (trade.is_long ? "LONG" : "SHORT") << ","
                       << trade.entry_price << ","
                       << trade.exit_price << ","
                       << trade.exit_target_price << ","
                       << trade.slippage_ticks << ","
                       << trade.hold_intervals << ","
                       << trade.first_profit_target_interval << ","
                       << trade.mae_at_first_profit_target << ","
                       << trade.profit_ticks << ","
                       << trade.mae << ","
                       << trade.mfe << ","
                       << trade.net_pnl << ","
                       << trade.entry_ofi << ","
                       << trade.entry_momentum << ","
                       << trade.entry_spread << ","
                       << trade.entry_volatility << ","
                       << trade.entry_qi_bid << ","
                       << trade.entry_qi_ask << ","
                       << trade.exit_ofi << ","
                       << trade.exit_spread << ","
                       << trade.exit_volatility << ","
                       << trade.exit_queue_position << ","
                       << trade.exit_order_placement_ns << ","
                       << trade.exit_order_fill_ns << ","
                       << (trade.exit_was_slippage ? 1 : 0) << ","
                       << (trade.exit_is_tp ? 1 : 0) << ","
                       << trade.exit_order_intervals_to_fill << "\n";
            
            // Export short diagnostics (we'll need to track this separately)
            if (!trade.is_long && short_count < 1000) {  // Limit to first 1000 for analysis
                short_count++;
            }
        }
        short_diag_csv.close();
        trades_csv.close();
        std::cout << "\n💾 Exported " << metrics.trades_for_ml.size() << " trades to trades_for_exit_training.csv" << std::endl;
        
        // Export canceled orders analysis
        std::ofstream canceled_csv("canceled_orders_analysis.csv");
        canceled_csv << "side,order_price,cancel_timestamp_ns,cancel_toxicity,cancel_ofi,cancel_momentum,cancel_mid_price,"
                     << "queue_position_at_cancel,cancel_reason,"
                     << "best_price_after_10_intervals,best_price_after_20_intervals,best_price_after_40_intervals,"
                     << "would_have_been_profitable_10,would_have_been_profitable_20,would_have_been_profitable_40,"
                     << "hypothetical_pnl_10_ticks,hypothetical_pnl_20_ticks,hypothetical_pnl_40_ticks\n";
        
        for (const auto& canceled : metrics.canceled_orders_for_analysis) {
            canceled_csv << (canceled.is_long ? "LONG" : "SHORT") << ","
                         << canceled.order_price << ","
                         << canceled.cancel_timestamp_ns << ","
                         << canceled.cancel_toxicity << ","
                         << canceled.cancel_ofi << ","
                         << canceled.cancel_momentum << ","
                         << canceled.cancel_mid_price << ","
                         << canceled.queue_position_at_cancel << ","
                         << canceled.cancel_reason << ","
                         << canceled.best_price_after_10_intervals << ","
                         << canceled.best_price_after_20_intervals << ","
                         << canceled.best_price_after_40_intervals << ","
                         << (canceled.would_have_been_profitable_10 ? 1 : 0) << ","
                         << (canceled.would_have_been_profitable_20 ? 1 : 0) << ","
                         << (canceled.would_have_been_profitable_40 ? 1 : 0) << ",";
            
            // Calculate hypothetical P&L at each interval
            double pnl_10 = 0.0, pnl_20 = 0.0, pnl_40 = 0.0;
            if (canceled.is_long) {
                pnl_10 = (canceled.best_price_after_10_intervals - canceled.order_price) / 0.25;
                pnl_20 = (canceled.best_price_after_20_intervals - canceled.order_price) / 0.25;
                pnl_40 = (canceled.best_price_after_40_intervals - canceled.order_price) / 0.25;
            } else {
                pnl_10 = (canceled.order_price - canceled.best_price_after_10_intervals) / 0.25;
                pnl_20 = (canceled.order_price - canceled.best_price_after_20_intervals) / 0.25;
                pnl_40 = (canceled.order_price - canceled.best_price_after_40_intervals) / 0.25;
            }
            canceled_csv << pnl_10 << "," << pnl_20 << "," << pnl_40 << "\n";
        }
        canceled_csv.close();
        std::cout << "💾 Exported " << metrics.canceled_orders_for_analysis.size() << " canceled orders to canceled_orders_analysis.csv" << std::endl;
        
        std::cout << "\n✅ COMPLETED (ML MODELS: LONG=" << (ml_model_long.loaded_from_file ? "LOADED" : "DEFAULT") 
                  << ", SHORT=" << (ml_model_short.loaded_from_file ? "LOADED" : "DEFAULT") << ")" << std::endl;
    }
};

int main() {
    // Register signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    // On Windows, attempt to switch console to UTF-8 so emoji and Unicode print correctly.
    // If this fails, output will fall back to whatever the terminal supports.
#ifdef _WIN32
    // Set UTF-8 code page for console output
    SetConsoleOutputCP(CP_UTF8);
    // Enable virtual terminal processing so VT sequences and some Unicode render better
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
#endif

    /* Notes:
     * - Windows 10/11 console historically uses a legacy code page which can cause UTF-8
     *   output (including emoji) to appear as garbled characters. The two common fixes are:
     *     1) Set the console code page to UTF-8 (chcp 65001) or call SetConsoleOutputCP(CP_UTF8).
     *     2) Use a terminal that supports UTF-8 and VT sequences (Windows Terminal, PowerShell Core).
     * - If you still see mojibake (garbled symbols like "≡ƒôè"), either run `chcp 65001` before launching
     *   the program or remove/replace emoji in the source with ASCII equivalents.
     */

    std::cout << "🚀 Optimized Databento Integration (HIGH PERFORMANCE)" << std::endl;
    std::cout << "⚡ Zero-allocation hot path, lock-free data structures" << std::endl;
    
    OptimizedDatabentoIntegration optimized_system;
    
    std::cout << "📊 Loading ML weights from: ml_weights.json" << std::endl;
    if (!optimized_system.load_ml_weights("ml_weights.json")) {
        std::cerr << "❌ Failed to load ML weights" << std::endl;
        return 1;
    }
    
    // ML weights are already displayed by load_ml_weights()
    
    std::string filename = "august_2025_complete.csv"; // Full month dataset
    // Full month run: 1M intervals ≈ 40 days of data (safe upper bound)
    optimized_system.run_optimized_simulation(filename, 1000000); // 1M intervals for comprehensive run
    
    return 0;
}