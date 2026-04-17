# dylanferreira_HFT_Research

Personal research repo: **Databento MBO → LOB → features → models → TCA**, with exports for downstream C++.

## Repository map

| Area | Role |
|------|------|
| [`adaptive_mm/`](adaptive_mm/) | Core library: LOB, features, regressors, toxicity, Markov regimes, spread logic, engine, TCA, options hooks |
| [`run_research.py`](run_research.py) | CLI entry: end-to-end pipeline (days / `--all`, optional `--no-export`) |
| [`export_enriched_csv.py`](export_enriched_csv.py) | Enriched CSV export utilities |
| [`download_opra.py`](download_opra.py) | OPRA / options data download helper |
| [`model_export/`](model_export/) | Serialized model artifacts (JSON) for C++ or other consumers |
| [`prep/`](prep/) | Prep notes: [`prep/pm_walkthrough_notes.md`](prep/pm_walkthrough_notes.md) (technical PM walkthrough, code-checked), [`prep/ma_capital_call_notes.md`](prep/ma_capital_call_notes.md) |
| Root tests | `test_pipeline.py`, `test_schema.py`, `test_instruments.py` |

## `adaptive_mm` modules (quick reference)

- **`lob.py`** — Limit order book reconstruction from MBO events  
- **`features.py`** — Feature extraction from MBO fields  
- **`regressors.py`** — Regressor construction / feature matrix  
- **`toxicity.py`** — Toxicity modeling (purged CV, calibration)  
- **`markov.py`** — Regime / transition structure  
- **`spread_optimizer.py`** — Spread-related optimization  
- **`backtest.py`** — Research backtest orchestration (`run_research`, file discovery)  
- **`tca.py`** — Transaction cost analysis (markouts, adverse selection, queue)  
- **`engine.py`** — Execution / simulation glue  
- **`options.py`** — Options-related extensions  
- **`cpp_databento_ml_optimized*.cpp`** — Optimized C++ reference / integration path  

## Typical workflow

1. Place MBO session files where `find_mbo_files()` expects them (see `adaptive_mm/backtest.py`).  
2. Run: `python run_research.py` or `python run_research.py --days 5` / `--all`.  
3. Consume outputs: feature frames, trained models, `model_export/*.json`, TCA summaries.  

## Data and secrets

Large raw MBO archives and generated parquet are **not** tracked here by default (see `.gitignore`). Keep API keys and paths in local env files; do not commit credentials.

## Requirements

Python 3 with dependencies implied by imports in `adaptive_mm` and the root scripts (install as you standardize the project, e.g. `requirements.txt` if you add one).
