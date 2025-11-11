# Factor Evaluation Scripts

Helper scripts for evaluating mined factors using the AlphaEval pipeline with AM/PM data.

## Overview

Three main scripts are provided:

1. **`eval_mined_factors.py`** - Evaluate individual factor pools
2. **`factor_analyzer.py`** - Analyze factor expression characteristics
3. **`batch_evaluate_pools.py`** - Batch evaluate all factor pools

## Data Sources

Factor pools are stored as JSON files in the following directories:

```
alphagen/
├── pool_3m.json    # 3-month window factors
├── pool_6m.json    # 6-month window factors
├── pool_12m.json   # 12-month window factors
└── pool_36m.json   # 36-month window factors

gfn_lstm/
├── 6m/pool_6m.json
├── 12m/pool_12m.json
└── all/pool_all.json

gfn_gnn/
├── 6m/pool_6m.json
├── 12m/pool_12m.json
└── all/pool_all.json
```

Each JSON file contains:
- `exprs`: List of factor expressions in qlib syntax
- `weights`: Optional weights for combining factors
- `window_results`: Metadata about the mining process

## AlphaEval Data Pipeline

The evaluation uses the AlphaEval framework with data from the alphagen_data_pipeline:

- **Data Format**: Parquet files partitioned by (date, session)
- **Sessions**: AM (before 12:00 ET) and PM (after 12:00 ET)
- **Features**: OHLCV data + style features (TE, CORR, IDIOVOL, etc.)
- **Storage**: Columnar format with ZSTD compression

See `alphagen_data_pipeline/README.md` for details on the data pipeline.

---

## Script 1: eval_mined_factors.py

Evaluate a single factor pool using AlphaEval.

### Usage

```bash
# Basic evaluation
python eval_mined_factors.py --pool alphagen/pool_3m.json --output results_3m.csv

# Custom date ranges
python eval_mined_factors.py --pool gfn_lstm/6m/pool_6m.json \
    --train_start 2010-01-01 --train_end 2019-12-31 \
    --test_start 2020-01-01 --test_end 2023-12-31

# Single factor mode (detailed per-factor metrics)
python eval_mined_factors.py --pool alphagen/pool_36m.json \
    --single_factor --output detailed_results.json

# Use weights from pool file
python eval_mined_factors.py --pool gfn_gnn/12m/pool_12m.json \
    --use_weights --output weighted_results.csv
```

### Arguments

- `--pool`: Path to factor pool JSON file (required)
- `--output`: Output file path (.csv or .json) (default: evaluation_results.csv)
- `--train_start`: Training start date (default: 2010-01-01)
- `--train_end`: Training end date (default: 2019-12-31)
- `--test_start`: Test start date (default: 2020-01-01)
- `--test_end`: Test end date (default: 2023-12-31)
- `--single_factor`: Evaluate each factor individually
- `--no_normalize`: Disable daily normalization
- `--use_weights`: Use weights from JSON file if available

### Output Metrics

**Combined Mode** (default):
- **IC**: Information Coefficient (correlation with future returns)
- **RankIC**: Rank Information Coefficient
- **RRE**: Rank Relative Entropy (stability measure)
- **PFS1**: Prediction Fidelity Score 1 (noise robustness - Gaussian)
- **PFS2**: Prediction Fidelity Score 2 (noise robustness - t-distribution)
- **Diversity**: Covariance entropy (factor diversity)
- **LLM_Score**: Average LLM rationality score

**Single Factor Mode**:
Per-factor metrics: IC, RankIC, RRE, PFS1, PFS2, LLM score

---

## Script 2: factor_analyzer.py

Analyze factor expression characteristics without evaluation.

### Usage

```bash
# Analyze a single pool
python factor_analyzer.py --pool alphagen/pool_3m.json

# Compare multiple pools
python factor_analyzer.py --pools alphagen/pool_3m.json \
    gfn_lstm/6m/pool_6m.json gfn_gnn/6m/pool_6m.json

# Save analysis report
python factor_analyzer.py --pool alphagen/pool_36m.json \
    --output analysis_report.txt

# Export to JSON
python factor_analyzer.py --pool gfn_gnn/12m/pool_12m.json \
    --output analysis.json

# Detailed analysis
python factor_analyzer.py --pool alphagen/pool_12m.json --detailed
```

### Arguments

- `--pool`: Path to single factor pool JSON file
- `--pools`: Paths to multiple pools for comparison
- `--output`: Output file path (.json, .txt, or .csv)
- `--detailed`: Print detailed analysis

### Analysis Metrics

- **Expression Length**: Min, max, mean, median character count
- **Expression Depth**: Min, max, mean, median nesting depth
- **Operator Usage**: Frequency of operators (Add, Mul, Log, etc.)
- **Feature Usage**: Frequency of features ($open, $close, etc.)
- **Duplicates**: Number of duplicate expressions

### Example Output

```
======================================================================
FACTOR ANALYSIS SUMMARY
======================================================================

Total Expressions: 25

--- Expression Length ---
  Min: 45, Max: 127, Mean: 78.4, Median: 76

--- Expression Depth ---
  Min: 3, Max: 8, Mean: 5.2, Median: 5

--- Operator Usage ---
  Total operators: 156
  Unique operators: 28
  Avg per expression: 6.2

  Top 10 operators:
    Mul: 24
    Add: 20
    Log: 18
    Sign: 15
    Div: 14
    ...
```

---

## Script 3: batch_evaluate_pools.py

Batch evaluate all factor pools with comprehensive reporting.

### Usage

```bash
# Evaluate all pools
python batch_evaluate_pools.py --output_dir results/

# Evaluate specific methods only
python batch_evaluate_pools.py --pools alphagen gfn_lstm \
    --output_dir results/

# Custom date ranges
python batch_evaluate_pools.py --train_start 2010-01-01 \
    --train_end 2019-12-31 --test_start 2020-01-01 \
    --test_end 2023-12-31

# Single factor mode (per-factor detailed metrics)
python batch_evaluate_pools.py --single_factor \
    --output_dir detailed_results/

# Quick comparison (no evaluation, just analysis)
python batch_evaluate_pools.py --compare_only \
    --output comparison_report.csv
```

### Arguments

- `--pools`: Filter specific methods (alphagen, gfn_lstm, gfn_gnn)
- `--output_dir`: Output directory for results (default: evaluation_results)
- `--train_start`: Training start date (default: 2010-01-01)
- `--train_end`: Training end date (default: 2019-12-31)
- `--test_start`: Test start date (default: 2020-01-01)
- `--test_end`: Test end date (default: 2023-12-31)
- `--single_factor`: Evaluate each factor individually
- `--no_weights`: Do not use weights from pool files
- `--compare_only`: Only compare pools without evaluation (fast)

### Output Files

The script generates:

1. **`summary_YYYYMMDD_HHMMSS.csv`** - Summary table with key metrics
2. **`full_results_YYYYMMDD_HHMMSS.json`** - Complete evaluation results
3. **`{pool_name}_detailed_YYYYMMDD_HHMMSS.json`** - Per-factor details (if --single_factor)

### Example Summary Output

```
====================================================================================================
EVALUATION SUMMARY
====================================================================================================
pool_name       method    window  num_factors    IC  RankIC    RRE    PFS1      PFS2  Diversity
alphagen_3m   alphagen        3m           25  0.042   0.038  0.912  0.0001  0.00008       0.87
alphagen_6m   alphagen        6m           28  0.051   0.045  0.923  0.0002  0.00012       0.89
gfn_lstm_6m   gfn_lstm        6m           20  0.048   0.041  0.918  0.0001  0.00009       0.85
...
```

---

## Workflow Examples

### Example 1: Quick Analysis of All Pools

```bash
# Compare all pools without evaluation (fast)
python batch_evaluate_pools.py --compare_only --output comparison.csv
```

### Example 2: Evaluate Specific Pool

```bash
# Evaluate 6-month GFN-LSTM pool
python eval_mined_factors.py --pool gfn_lstm/6m/pool_6m.json \
    --use_weights --output gfn_lstm_6m_results.csv
```

### Example 3: Full Evaluation Pipeline

```bash
# Step 1: Analyze all pools
python factor_analyzer.py --pools alphagen/pool_3m.json \
    alphagen/pool_6m.json alphagen/pool_12m.json

# Step 2: Batch evaluate all pools
python batch_evaluate_pools.py --output_dir results/

# Step 3: Detailed evaluation of best-performing pool
python eval_mined_factors.py --pool alphagen/pool_12m.json \
    --single_factor --output detailed_12m.json
```

### Example 4: Custom Date Range Evaluation

```bash
# Evaluate with specific train/test split
python eval_mined_factors.py --pool alphagen/pool_36m.json \
    --train_start 2015-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2024-12-31 \
    --output results_custom_dates.csv
```

---

## Requirements

Before running the scripts, ensure:

1. **Qlib is initialized** with data:
   ```python
   import qlib
   qlib.init(provider_uri="path/to/your/qlib_data", region="cn")
   ```

2. **Factor pool JSON files exist** in the specified directories

3. **Dependencies installed**:
   ```bash
   pip install pandas numpy qlib scikit-learn
   ```

4. **Update paths** in `backtest/modeltester.py`:
   - Line 54: Update `provider_uri="path/to/your/qlib_data"`

---

## Understanding the Metrics

### IC (Information Coefficient)
- Correlation between factor values and future returns
- Range: -1 to 1 (higher absolute value = better)
- Target: > 0.03 is good, > 0.05 is excellent

### RankIC (Rank Information Coefficient)
- Spearman rank correlation (more robust to outliers)
- Preferred over IC for production use

### RRE (Rank Relative Entropy)
- Measures stability of factor rankings over time
- Range: 0 to 1 (higher = more stable)
- Target: > 0.90 is good

### PFS1 & PFS2 (Prediction Fidelity Scores)
- Robustness to noise injection
- Lower values = more robust
- Target: Close to 0

### Diversity
- Measures independence of factors in the pool
- Range: 0 to 1 (higher = more diverse)
- Target: > 0.80 is good

### LLM Score
- GPT-4 evaluation of factor rationality
- Range: 0 to 100 (higher = more logical)
- Requires OpenAI API key in `backtest/modeltester.py`

---

## Troubleshooting

### Error: "No data loaded for any symbols"
- Check that qlib data path is correct
- Verify date ranges have available data

### Error: "JSON file must contain 'exprs' field"
- Ensure JSON files have correct format with "exprs" key

### Memory Error
- Reduce date range
- Use smaller factor pools
- Increase available RAM

### LLM Score Error
- Set OpenAI API key in `backtest/modeltester.py` line 237
- Or disable LLM scoring by modifying the code

---

## Next Steps

After evaluation, you can:

1. **Backtest top factors** using `backtest/modeltester.py`
2. **Combine factors** using optimal weights
3. **Deploy to production** with real-time data pipeline
4. **Monitor performance** with out-of-sample testing

For questions or issues, refer to the main AlphaEval documentation.
