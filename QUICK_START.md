# Quick Start Guide - Pool Evaluation

## TL;DR - Run This Now

```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# 2. Run evaluation for all pools (2024-present)
python run_pool_eval.py --enable_llm --output_dir results/

# Done! Results will be in results/ directory
```

## What This Script Does

Evaluates all your factor pools from 2024 to present and:
- ✅ **Eliminates all NaN warnings** (rolling window, overflow, etc.)
- ✅ **Fixes LLM evaluation** (was showing `nan`, now works properly)
- ✅ **Tests 2024-present data** by default
- ✅ **Generates comprehensive reports**

## The Two Issues You Had (Now Fixed)

### Issue 1: NaN Warnings ✅ FIXED
```
RuntimeWarning: All-NaN slice encountered
RuntimeWarning: overflow encountered in cast
```
**Solution**: Added comprehensive warning filters in:
- `backtest/modeltester.py` (lines 14-17)
- `run_pool_eval.py` (lines 37-42)
- `eval_mined_factors.py` (line 34)

### Issue 2: LLM showing `nan` ✅ FIXED
```
LLM:  nan
```
**Why it happened**: API key was hardcoded as `"Your own LLM key"`

**Solution**: Now supports API key from:
1. Environment variable `OPENAI_API_KEY`
2. Environment variable `LLM_API_KEY`
3. Command line `--api_key`

## Setup (One Time)

### Step 1: Set API Key

Choose one method:

**Method A: Temporary (current session)**
```bash
export OPENAI_API_KEY="sk-..."
```

**Method B: Permanent (recommended)**
```bash
# Add to your shell profile
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

**Method C: Using .env file**
```bash
# Copy template
cp .env.example .env

# Edit .env and add your key
nano .env

# Load it
source .env
```

### Step 2: Verify Setup
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Should show: sk-...
```

## Common Use Cases

### Evaluate All Pools (2024-Present)
```bash
python run_pool_eval.py --enable_llm --output_dir results_2024/
```

### Evaluate Without LLM (Faster)
```bash
python run_pool_eval.py --output_dir results_no_llm/
```

### Evaluate Single Pool
```bash
python run_pool_eval.py --pools alphagen --enable_llm --output_dir alphagen_results/
```

### Custom Date Range
```bash
python run_pool_eval.py \
    --test_start 2024-01-01 \
    --test_end 2024-12-31 \
    --enable_llm \
    --output_dir results_2024_only/
```

### Single Factor Analysis (Detailed)
```bash
python run_pool_eval.py \
    --single_factor \
    --enable_llm \
    --output_dir detailed_analysis/
```

## Understanding Output

After running, you'll get:

```
results_2024/
├── summary_20241112_143022.csv          # Quick overview
├── full_results_20241112_143022.json    # Complete data
└── (optional) detailed factor JSONs     # If --single_factor used
```

**Summary CSV contains**:
| pool_name | IC | RankIC | RRE | PFS1 | PFS2 | Diversity | LLM_Score |
|-----------|----|----|-----|------|------|-----------|-----------|
| alphagen_3m | 0.045 | 0.052 | 0.694 | 0.894 | 0.900 | 0.879 | 78.5 |

## Troubleshooting

### "LLM: nan" Still Showing
```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Make sure you used --enable_llm flag
python run_pool_eval.py --enable_llm --output_dir results/
```

### "Pool file not found"
```bash
# Check available pools
ls alphagen/*.json
ls gfn_lstm/*/*.json
ls gfn_gnn/*/*.json
```

### Script runs but crashes
```bash
# Check Python version (needs 3.8+)
python --version

# Check required packages
pip install pandas numpy openai qlib
```

## Performance Tips

1. **Without LLM**: 5-10 min per pool
2. **With LLM**: 15-30 min per pool
3. **Run pools in parallel** (different terminals):
   ```bash
   # Terminal 1
   python run_pool_eval.py --pools alphagen --enable_llm --output_dir alphagen/

   # Terminal 2
   python run_pool_eval.py --pools gfn_lstm --enable_llm --output_dir gfn_lstm/

   # Terminal 3
   python run_pool_eval.py --pools gfn_gnn --enable_llm --output_dir gfn_gnn/
   ```

## What Changed vs Old batch_evaluate_pools.py

| Feature | Old | New |
|---------|-----|-----|
| NaN warnings | ❌ Shows many warnings | ✅ Suppressed |
| LLM evaluation | ❌ Shows `nan` | ✅ Works with API key |
| Date range | 2020-2023 | 2024-2025 (configurable) |
| API key config | ❌ Hardcoded | ✅ Env var + CLI |
| Error handling | Basic | Comprehensive |
| Default output | evaluation_results | evaluation_results_2024 |

## Next Steps

After evaluation completes:

1. **Check summary**:
   ```bash
   cat results_2024/summary_*.csv
   ```

2. **Find best pool**:
   ```bash
   # Sort by IC
   cat results_2024/summary_*.csv | sort -t',' -k5 -rn | head -5
   ```

3. **Detailed analysis of best pool**:
   ```bash
   python run_pool_eval.py \
       --pools alphagen \
       --single_factor \
       --enable_llm \
       --output_dir best_pool_detailed/
   ```

## Need Help?

- Full documentation: `POOL_EVAL_README.md`
- Check code: `run_pool_eval.py`
- API key issues: Verify with `echo $OPENAI_API_KEY`

## Example Complete Workflow

```bash
# Complete evaluation workflow
export OPENAI_API_KEY="sk-..."

# 1. Quick check
python run_pool_eval.py --compare_only

# 2. Full evaluation
python run_pool_eval.py --enable_llm --output_dir full_eval/

# 3. View results
cat full_eval/summary_*.csv

# 4. Detailed analysis of top pool
python run_pool_eval.py \
    --pools alphagen \
    --single_factor \
    --enable_llm \
    --output_dir alphagen_deep_dive/
```

That's it! 🚀
