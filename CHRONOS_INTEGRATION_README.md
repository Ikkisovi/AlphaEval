# Chronos-2 Time Series Forecasting Integration

This directory contains an integration of Amazon's Chronos-2 pretrained time series foundation model with the qlib quantitative trading framework.

## Overview

**Chronos-2** is a state-of-the-art time series forecasting model that can generate predictions for stock prices. This integration allows you to use these predictions as factors in your alpha mining pipeline.

## Files

### Core Integration Files

1. **`chronos_predictor.py`** - Full-featured Chronos predictor
   - Downloads and caches Chronos-2 models
   - Generates rolling predictions for stocks (2022-2025)
   - Exports predictions in qlib-compatible format
   - Supports multiple model sizes (tiny to large)

2. **`chronos_qlib_integration.py`** - Simple examples
   - Minimal working examples
   - Demonstrates basic usage
   - Shows how to create custom qlib operators

3. **`test_qlib_data.py`** - Verify qlib data loading
   - Tests D.features functionality
   - Checks data integrity
   - Validates date ranges

4. **`test_qlib_expressions.py`** - Test all operators
   - Verifies custom operators work
   - Tests rolling, element-wise, pair-wise ops
   - Ensures compatibility with factor pools

## Quick Start

### 1. Install Dependencies

```bash
# Install Chronos and dependencies
pip install chronos-forecasting torch transformers

# Verify qlib is installed
pip install qlib pandas numpy
```

### 2. Run Simple Example

```bash
# Try the simple integration example
python chronos_qlib_integration.py
```

This will:
- Download the tiny Chronos model (~10MB)
- Show basic prediction example
- Demonstrate qlib factor integration

### 3. Generate Full Predictions

```bash
# Generate predictions for CSI300 stocks from 2022-2025
python chronos_predictor.py \
    --model_size small \
    --market csi300 \
    --start_date 2022-01-01 \
    --end_date 2025-12-31 \
    --context_length 60 \
    --output chronos_predictions_2022_2025.pkl
```

### 4. Use Predictions in Qlib

```python
import pandas as pd
import qlib
from qlib.data import D

# Load predictions
chronos_preds = pd.read_pickle('chronos_predictions_2022_2025.pkl')

# Load other factors
traditional_factors = D.features(
    instruments=instruments,
    fields=["$close/Ref($close,1)-1", "Mean($close,20)"],
    start_time="2022-01-01",
    end_time="2025-12-31"
)

# Combine predictions with traditional factors
combined = traditional_factors.join(chronos_preds, how='inner')

# Use in your alpha pipeline
```

## Model Sizes

Chronos-2 comes in multiple sizes. Choose based on your needs:

| Model Size | Download Size | Speed | Accuracy |
|------------|---------------|-------|----------|
| tiny       | ~10MB        | ⚡⚡⚡⚡ | ⭐⭐     |
| mini       | ~20MB        | ⚡⚡⚡   | ⭐⭐⭐   |
| small      | ~50MB        | ⚡⚡    | ⭐⭐⭐⭐  |
| base       | ~200MB       | ⚡     | ⭐⭐⭐⭐⭐|
| large      | ~700MB       | 🐌     | ⭐⭐⭐⭐⭐|

**Recommendation**: Start with `small` for good balance of speed and accuracy.

## Command-Line Options

### chronos_predictor.py

```bash
python chronos_predictor.py [OPTIONS]

Model Options:
  --model_size {tiny,mini,small,base,large}
                        Model size (default: small)
  --device {auto,cpu,cuda}
                        Device to use (default: auto)
  --cache_dir PATH      Model cache directory (default: ./chronos_models)

Prediction Options:
  --prediction_length N Prediction horizon (default: 1)
  --context_length N    Lookback window (default: 60)
  --num_samples N       Forecast samples (default: 20)

Data Options:
  --market MARKET       Market (default: csi300)
  --start_date DATE     Start date (default: 2022-01-01)
  --end_date DATE       End date (default: 2025-12-31)
  --qlib_data_path PATH Path to qlib data

Output Options:
  --output FILE         Output file (default: chronos_predictions.pkl)
  --export_qlib         Export in qlib format
```

## Advanced Usage

### Creating a Custom Chronos Factor Operator

To integrate Chronos predictions directly into qlib expressions:

```python
# Add to my_qlib/data/ops.py

from my_qlib.data.ops import ExpressionOps
import torch

class ChronosFactor(ExpressionOps):
    """Chronos prediction factor"""

    def __init__(self, feature, context_length=60):
        self.feature = feature
        self.context_length = context_length

    def _load_internal(self, instrument, start_index, end_index, *args):
        # Load price data
        series = self.feature.load(instrument, start_index, end_index, *args)

        # Generate predictions using Chronos
        # (Implementation in chronos_qlib_integration.py)

        return predictions

# Then use in expressions:
# "ChronosFactor($close, 60)"
```

### Combining with Existing Factors

```python
# Example: Combine Chronos predictions with momentum
factor_expr = "ChronosFactor($close, 60) * Slope($close, 20)"

# Or create an ensemble
factor_expr = "0.5 * ChronosFactor($close, 60) + 0.5 * ($close/Mean($close,20)-1)"
```

## Integration with Pool Evaluation

After generating Chronos predictions, integrate them into your factor pool evaluation:

```bash
# 1. Generate Chronos predictions
python chronos_predictor.py --output chronos_factor.pkl

# 2. Use in pool evaluation
# Edit your pool JSON to include Chronos-based expressions

# 3. Run pool evaluation with Chronos factors
python run_pool_eval.py --enable_llm --output_dir results_with_chronos/
```

## Performance Tips

### 1. Use GPU for Faster Predictions

```bash
# If you have CUDA available
python chronos_predictor.py --device cuda --model_size base
```

### 2. Batch Processing

For large-scale predictions, process in batches:

```python
# Split instruments into chunks
chunks = [instruments[i:i+50] for i in range(0, len(instruments), 50)]

for chunk in chunks:
    predictions = generate_predictions(chunk)
    save_predictions(predictions)
```

### 3. Caching

The model is automatically cached after first download:

```bash
# Models cached in:
./chronos_models/
  ├── models--amazon--chronos-t5-small/
  ├── models--amazon--chronos-t5-base/
  └── ...
```

## Troubleshooting

### Issue: "chronos-forecasting not found"

```bash
pip install chronos-forecasting torch transformers
```

### Issue: "CUDA out of memory"

```bash
# Use smaller model or CPU
python chronos_predictor.py --model_size tiny --device cpu
```

### Issue: "No data for date range"

Check your qlib data covers the requested period:

```python
python test_qlib_data.py  # Verify data availability
```

### Issue: Predictions are all NaN

- Check context_length is smaller than available data
- Verify price data has no gaps
- Ensure instruments are valid

## Testing

### Test 1: Verify Qlib Data

```bash
python test_qlib_data.py
```

Expected output:
```
✓ Qlib initialized
✓ Found 300 CSI300 instruments
✓ Data shape: (45000, 5)
✓ All tests passed!
```

### Test 2: Verify Operators

```bash
python test_qlib_expressions.py
```

Expected output:
```
✓ Custom operators registered
✓ 50/52 operators working (96%)
✓ All operator tests completed successfully!
```

### Test 3: Chronos Integration

```bash
python chronos_qlib_integration.py
```

Expected output:
```
✓ Model loaded!
Predicted prices (next 5 days): [110.5, 111.2, ...]
✓ All examples completed
```

## Example Workflow

Complete workflow for generating and using Chronos factors:

```bash
# Step 1: Verify qlib setup
python test_qlib_data.py
python test_qlib_expressions.py

# Step 2: Generate Chronos predictions
python chronos_predictor.py \
    --model_size small \
    --start_date 2022-01-01 \
    --end_date 2025-12-31 \
    --output chronos_2022_2025.pkl

# Step 3: Load and inspect predictions
python -c "
import pandas as pd
preds = pd.read_pickle('chronos_2022_2025.pkl')
print(preds.describe())
print(preds.head(20))
"

# Step 4: Integrate with pool evaluation
# (Add Chronos expressions to your pool JSON)

# Step 5: Run evaluation
export OPENAI_API_KEY="your-key"
python run_pool_eval.py \
    --enable_llm \
    --test_start 2024-01-01 \
    --test_end 2025-12-31 \
    --output_dir results_chronos/
```

## AlphaSAGE Integration

The `AlphaSAGE/` directory contains the state-of-the-art alpha mining framework using GFlowNets. You can combine Chronos predictions with AlphaSAGE-generated factors:

```bash
# 1. Mine factors with AlphaSAGE (see AlphaSAGE/README.md)
cd AlphaSAGE
python train_gfn.py --encoder_type gnn --pool_capacity 50

# 2. Generate Chronos predictions
cd ..
python chronos_predictor.py

# 3. Combine both in evaluation
python run_pool_eval.py --pools alphagen --enable_llm
```

## References

- **Chronos**: [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- **Qlib**: [Microsoft Qlib](https://github.com/microsoft/qlib)
- **AlphaSAGE**: [AlphaSAGE GitHub](https://github.com/BerkinChen/AlphaSAGE)

## Citation

If you use Chronos in your research:

```bibtex
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and others},
  journal={arXiv preprint arXiv:2403.07815},
  year={2024}
}
```

## License

This integration code is provided under MIT License. Chronos model weights are subject to Apache 2.0 License.

## Support

For issues:
- Chronos issues: https://github.com/amazon-science/chronos-forecasting/issues
- Qlib issues: https://github.com/microsoft/qlib/issues
- Integration issues: Create an issue in this repository
