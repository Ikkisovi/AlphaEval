# Pool Evaluation Guide (2024-Present)

This guide explains how to use the improved `run_pool_eval.py` script with fixes for NaN warnings and LLM evaluation support.

## What's Fixed

### 1. NaN Warnings Eliminated
- All `RuntimeWarning: All-NaN slice encountered` warnings are now suppressed
- All `RuntimeWarning: overflow encountered in cast` warnings are suppressed
- Better error handling for edge cases with insufficient data

### 2. LLM Evaluation Working
The LLM evaluation was showing `nan` because no API key was configured. Now it:
- Reads API key from environment variables (`OPENAI_API_KEY` or `LLM_API_KEY`)
- Can be passed via command line argument `--api_key`
- Gracefully falls back to NaN if no key is provided (with clear warning)
- Supports GPT-4o and other OpenAI models

## Setup

### Configure OpenAI API Key

You have three options:

**Option 1: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="your-api-key-here"
python run_pool_eval.py --enable_llm
```

**Option 2: Command Line**
```bash
python run_pool_eval.py --enable_llm --api_key "your-api-key-here"
```

**Option 3: Set permanently in your shell profile**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Usage Examples

### 1. Evaluate All Pools (2024-Present, Default)
```bash
python run_pool_eval.py --output_dir results_2024_present/
```

This evaluates all pools with:
- Training: 2020-01-01 to 2023-12-31
- Testing: 2024-01-01 to 2025-12-31 (can extend as needed)

### 2. With LLM Evaluation Enabled
```bash
export OPENAI_API_KEY="your-key"
python run_pool_eval.py --enable_llm --output_dir results_with_llm/
```

**Important**: The `--enable_llm` flag is required to enable LLM evaluation. Even if you have `OPENAI_API_KEY` set in your environment for other tools, LLM evaluation will not run unless you explicitly use this flag. This prevents unexpected API costs.

### 3. Evaluate Specific Pools Only
```bash
python run_pool_eval.py --pools alphagen gfn_lstm --output_dir results_filtered/
```

### 4. Custom Date Range
```bash
python run_pool_eval.py \
    --train_start 2018-01-01 \
    --train_end 2023-12-31 \
    --test_start 2024-01-01 \
    --test_end 2025-12-31 \
    --output_dir results_custom/
```

### 5. Single Factor Mode (Detailed Analysis)
```bash
python run_pool_eval.py --single_factor --output results_detailed/
```

### 6. Quick Comparison (No Full Evaluation)
```bash
python run_pool_eval.py --compare_only --output comparison.csv
```

### 7. All Features Combined
```bash
export OPENAI_API_KEY="your-key"
python run_pool_eval.py \
    --pools alphagen gfn_lstm gfn_gnn \
    --enable_llm \
    --llm_model gpt-4o \
    --test_start 2024-01-01 \
    --test_end 2025-12-31 \
    --output_dir comprehensive_results/
```

## Available Pools

The script will automatically find and evaluate all available pools:

- **alphagen**: pool_3m.json, pool_6m.json, pool_12m.json, pool_36m.json
- **gfn_lstm**: pool_6m.json, pool_12m.json, pool_all.json
- **gfn_gnn**: pool_6m.json, pool_12m.json, pool_all.json

## Output Files

The script generates:

1. **Summary CSV**: Quick overview of all pool results
   - `summary_TIMESTAMP.csv`

2. **Full Results JSON**: Complete evaluation data
   - `full_results_TIMESTAMP.json`

3. **Detailed Factor Results** (if `--single_factor` used):
   - `{pool_name}_detailed_TIMESTAMP.json` for each pool

## Understanding the Metrics

- **IC**: Information Coefficient (correlation with returns)
- **RankIC**: Rank Information Coefficient
- **RRE**: Rank Return Entropy (stability measure)
- **PFS1/PFS2**: Perturbation-based Factor Stability scores
- **Diversity**: Factor diversity measure (0-1)
- **LLM_Score**: LLM-based quality assessment (50-100)

## Troubleshooting

### LLM Shows NaN
If you see `LLM: nan` in the output:
1. Make sure you used `--enable_llm` flag
2. Check that your API key is set correctly:
   ```bash
   echo $OPENAI_API_KEY
   ```
3. Verify the API key is valid by testing:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### Still Seeing Warnings?
If you still see RuntimeWarnings, they may be coming from qlib or pandas internals.
The script suppresses most common warnings, but some deep library warnings may still appear.
They won't affect the results.

### Pool Not Found
If a pool file is missing, the script will skip it and show a warning.
Make sure all pool JSON files are in the correct directories relative to the script.

## Advanced Configuration

### Using Different LLM Models
```bash
# Use GPT-4 Turbo
python run_pool_eval.py --enable_llm --llm_model gpt-4-turbo

# Use GPT-3.5 (cheaper, faster)
python run_pool_eval.py --enable_llm --llm_model gpt-3.5-turbo
```

### Parallel Evaluation
For large-scale evaluation, you can run multiple instances on different pools:
```bash
# Terminal 1
python run_pool_eval.py --pools alphagen --output alphagen_results/

# Terminal 2
python run_pool_eval.py --pools gfn_lstm --output gfn_lstm_results/

# Terminal 3
python run_pool_eval.py --pools gfn_gnn --output gfn_gnn_results/
```

## Example Workflow

```bash
# 1. Set up API key
export OPENAI_API_KEY="sk-..."

# 2. First, do a quick comparison to see what's available
python run_pool_eval.py --compare_only

# 3. Run full evaluation on all pools for 2024-present
python run_pool_eval.py --enable_llm --output_dir results_2024_full/

# 4. Check the results
cat results_2024_full/summary_*.csv

# 5. For detailed analysis of best performing pool
python run_pool_eval.py \
    --pools alphagen \
    --single_factor \
    --enable_llm \
    --output_dir alphagen_detailed/
```

## Performance Notes

- **Without LLM**: ~5-10 minutes per pool
- **With LLM**: ~15-30 minutes per pool (depends on pool size and API rate limits)
- **Compare-only mode**: ~1 minute for all pools

## Changes from Original batch_evaluate_pools.py

1. ✅ Better default date ranges (2024-present)
2. ✅ Comprehensive warning suppression
3. ✅ LLM API key configuration support
4. ✅ Better error messages and user feedback
5. ✅ Graceful degradation when API key missing
6. ✅ Support for multiple OpenAI models
7. ✅ More detailed progress reporting
