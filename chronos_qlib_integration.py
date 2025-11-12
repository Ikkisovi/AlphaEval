#!/usr/bin/env python3
"""
Simple Chronos-2 Integration Example with Qlib
===============================================

A minimal example showing how to use Chronos-2 predictions as qlib factors.

This script demonstrates:
1. Loading Chronos-2 model
2. Generating predictions for stocks
3. Using predictions as factors in qlib expressions
4. Combining with traditional factors

Usage:
    python chronos_qlib_integration.py
"""

import numpy as np
import pandas as pd
import torch
import qlib
from qlib.data import D
import warnings

warnings.filterwarnings("ignore")

def simple_chronos_example():
    """Simple example of using Chronos for stock prediction"""
    print("\n" + "="*70)
    print("SIMPLE CHRONOS-2 EXAMPLE")
    print("="*70)

    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("Installing chronos-forecasting...")
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "chronos-forecasting", "-q"
        ])
        from chronos import ChronosPipeline

    # Load model (use tiny for speed)
    print("\nLoading Chronos-t5-tiny model...")
    print("(This will download ~10MB on first run)")

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",  # Use CPU for compatibility
        torch_dtype=torch.float32,
    )
    print("✓ Model loaded!")

    # Example: Predict next price given historical prices
    print("\nExample: Predicting stock prices")

    # Synthetic price data (replace with real data)
    historical_prices = np.array([
        100.0, 101.2, 102.5, 101.8, 103.0,
        104.1, 103.5, 105.0, 106.2, 105.5,
        107.0, 108.5, 107.8, 109.0, 110.2
    ])

    print(f"  Historical prices (last 15 days): {historical_prices[-5:]}")

    # Convert to tensor
    context = torch.tensor(historical_prices, dtype=torch.float32)

    # Generate forecast
    forecast = pipeline.predict(
        context=context,
        prediction_length=5,  # Predict next 5 days
        num_samples=20  # Number of sample paths
    )

    # Get predictions
    median_forecast = forecast.median(dim=0).values.cpu().numpy()
    mean_forecast = forecast.mean(dim=0).cpu().numpy()

    print(f"\n  Predicted prices (next 5 days):")
    print(f"    Median: {median_forecast}")
    print(f"    Mean:   {mean_forecast}")

    # Calculate predicted returns
    current_price = historical_prices[-1]
    next_day_return = (median_forecast[0] / current_price) - 1

    print(f"\n  Current price: {current_price:.2f}")
    print(f"  Predicted next day return: {next_day_return:.4f} ({next_day_return*100:.2f}%)")

    return pipeline


def qlib_integration_example(qlib_data_path="./qlib_data/cn_data/qlib_bin"):
    """Example of integrating Chronos predictions with qlib"""
    print("\n" + "="*70)
    print("QLIB INTEGRATION EXAMPLE")
    print("="*70)

    # Initialize qlib
    try:
        qlib.init(provider_uri=qlib_data_path, region="cn")
        print(f"✓ Qlib initialized")
    except Exception as e:
        print(f"Warning: Could not initialize qlib: {e}")
        print(f"  Skipping qlib integration example")
        return

    # Get some instruments
    instruments = D.list_instruments(market="csi300")[:5]
    print(f"\nTesting with {len(instruments)} instruments: {instruments}")

    # Load price data
    print(f"\nLoading price data...")
    price_data = D.features(
        instruments=instruments,
        fields=["$close", "$open", "$high", "$low", "$volume"],
        start_time="2023-01-01",
        end_time="2023-06-30",
        freq="day"
    )

    print(f"  Loaded data shape: {price_data.shape}")
    print(f"\nSample data:")
    print(price_data.head(10))

    # Example: Create a simple predictive factor
    # In practice, you would use Chronos here
    print(f"\n" + "-"*70)
    print("Creating Chronos-inspired factor (momentum + volatility)")
    print("-"*70)

    # For demonstration, create a factor that combines recent patterns
    # (In real usage, replace this with actual Chronos predictions)

    factor_exprs = [
        # Momentum-based (similar to what Chronos might capture)
        "Slope($close, 20) * Rsquare($close, 20)",

        # Volatility-adjusted return
        "Delta($close, 5) / Std($close, 20)",

        # Price-volume pattern
        "Corr($close, $volume, 10)",

        # Mean reversion signal
        "($close - Mean($close, 20)) / Std($close, 20)",
    ]

    print(f"\nTesting {len(factor_exprs)} factor expressions:")

    for i, expr in enumerate(factor_exprs, 1):
        try:
            factor_data = D.features(
                instruments=instruments,
                fields=[expr],
                start_time="2023-01-01",
                end_time="2023-06-30",
                freq="day"
            )

            print(f"\n{i}. {expr}")
            print(f"   Shape: {factor_data.shape}")
            print(f"   Mean: {factor_data.mean().values[0]:.4f}, Std: {factor_data.std().values[0]:.4f}")

        except Exception as e:
            print(f"\n{i}. {expr}")
            print(f"   Error: {e}")

    print(f"\n✓ Factor integration example completed")


def create_chronos_factor_class_example():
    """Example of how to create a custom Chronos factor for qlib"""
    print("\n" + "="*70)
    print("CUSTOM CHRONOS FACTOR CLASS EXAMPLE")
    print("="*70)

    example_code = '''
# Example: Custom Chronos Factor Operator for Qlib
# Save this in my_qlib/data/ops.py and register it

from my_qlib.data.ops import ExpressionOps
import torch
import numpy as np
import pandas as pd

class ChronosFactor(ExpressionOps):
    """
    Chronos-based prediction factor

    Usage in qlib expressions:
        "ChronosFactor($close, 60, 1)"  # Predict 1-day ahead using 60-day context
    """

    def __init__(self, feature, context_length=60, pred_length=1):
        self.feature = feature
        self.context_length = context_length
        self.pred_length = pred_length
        self._model = None

    def _load_model(self):
        """Lazy load Chronos model"""
        if self._model is None:
            from chronos import ChronosPipeline
            self._model = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        return self._model

    def _load_internal(self, instrument, start_index, end_index, *args):
        """Generate rolling predictions"""
        # Load price data
        series = self.feature.load(instrument, start_index, end_index, *args)

        if len(series) < self.context_length:
            return pd.Series(np.nan, index=series.index)

        # Get model
        model = self._load_model()

        # Generate rolling predictions
        predictions = []
        for i in range(self.context_length, len(series)):
            context = series.iloc[i-self.context_length:i].values
            context_tensor = torch.tensor(context, dtype=torch.float32)

            # Predict
            forecast = model.predict(
                context=context_tensor,
                prediction_length=self.pred_length,
                num_samples=10
            )

            # Get median prediction
            pred = forecast.median(dim=0).values[0].item()

            # Calculate return
            current = series.iloc[i]
            predicted_return = (pred / current) - 1

            predictions.append(predicted_return)

        # Pad with NaN for first context_length points
        full_preds = [np.nan] * self.context_length + predictions

        return pd.Series(full_preds, index=series.index)

    def get_longest_back_rolling(self):
        return self.context_length

    def get_extended_window_size(self):
        return self.context_length, 0


# To use this operator:
# 1. Add to my_qlib/data/ops.py
# 2. Register in OpsList
# 3. Use in expressions: "ChronosFactor($close, 60, 1)"
    '''

    print(example_code)
    print("\n✓ See above for example custom Chronos factor operator")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" "*20 + "CHRONOS-2 QLIB INTEGRATION EXAMPLES")
    print("="*80)

    # Example 1: Simple Chronos usage
    pipeline = simple_chronos_example()

    # Example 2: Qlib integration
    qlib_integration_example()

    # Example 3: Custom factor class
    create_chronos_factor_class_example()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nNext steps:")
    print("1. Install dependencies: pip install chronos-forecasting torch qlib")
    print("2. Run chronos_predictor.py for full-scale predictions")
    print("3. Integrate predictions with existing alpha factors")
    print("4. Backtest combined factors using run_pool_eval.py")


if __name__ == "__main__":
    main()
