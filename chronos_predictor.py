#!/usr/bin/env python3
"""
Chronos-2 Time Series Prediction for Qlib Factors
===================================================

This script uses Amazon's Chronos-2 pretrained time series foundation model
to generate rolling predictions for stock prices from 2022-2025.

The predictions can be integrated into qlib as custom factors for alpha mining.

Features:
- Downloads and caches the Chronos-2 model from HuggingFace
- Generates rolling predictions for a universe of stocks
- Exports predictions in qlib-compatible format
- Can be used as a standalone factor or combined with other factors

Usage:
    # Generate predictions for all CSI300 stocks
    python chronos_predictor.py --market csi300 --start_date 2022-01-01 --end_date 2025-12-31

    # Generate predictions with custom model size
    python chronos_predictor.py --model_size small --prediction_length 5

    # Export as qlib factor data
    python chronos_predictor.py --export_qlib --output predictions_factor.pkl

Requirements:
    pip install chronos-forecasting torch transformers qlib pandas numpy
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# Try to import chronos
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    print("Warning: chronos-forecasting not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chronos-forecasting", "-q"])
    try:
        from chronos import ChronosPipeline
        CHRONOS_AVAILABLE = True
    except ImportError:
        CHRONOS_AVAILABLE = False
        print("Error: Could not install chronos-forecasting")

# Add qlib support
try:
    import qlib
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("Warning: qlib not available. Some features will be limited.")


class ChronosPredictor:
    """
    Chronos-2 based time series predictor for stock prices
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        prediction_length: int = 1,
        num_samples: int = 20,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Chronos predictor

        Args:
            model_size: Model size (tiny, mini, small, base, large)
            device: Device to run on (auto, cpu, cuda)
            prediction_length: Number of steps ahead to predict
            num_samples: Number of samples for probabilistic forecasting
            cache_dir: Directory to cache model weights
        """
        self.model_size = model_size
        self.prediction_length = prediction_length
        self.num_samples = num_samples

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model name mapping
        model_names = {
            "tiny": "amazon/chronos-t5-tiny",
            "mini": "amazon/chronos-t5-mini",
            "small": "amazon/chronos-t5-small",
            "base": "amazon/chronos-t5-base",
            "large": "amazon/chronos-t5-large",
        }

        self.model_name = model_names.get(model_size, "amazon/chronos-t5-small")
        self.cache_dir = cache_dir or "./chronos_models"

        print(f"\nInitializing Chronos-2 Predictor")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Prediction length: {self.prediction_length}")
        print(f"  Cache dir: {self.cache_dir}")

        # Load model
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load or download the Chronos model"""
        try:
            print(f"\nLoading Chronos model...")
            print(f"  This may take a few minutes on first run (downloading ~{self._get_model_size()})")

            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                cache_dir=self.cache_dir
            )

            print(f"✓ Model loaded successfully!")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print(f"\nTrying to install/update dependencies...")
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "chronos-forecasting", "torch", "transformers", "-U", "-q"
            ])
            raise

    def _get_model_size(self):
        """Get approximate model size"""
        sizes = {
            "tiny": "~10MB",
            "mini": "~20MB",
            "small": "~50MB",
            "base": "~200MB",
            "large": "~700MB"
        }
        return sizes.get(self.model_size, "unknown")

    def predict(
        self,
        context: np.ndarray,
        prediction_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from context

        Args:
            context: Historical time series data (context window)
            prediction_length: Override default prediction length

        Returns:
            (mean_prediction, median_prediction): Predicted values
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        pred_length = prediction_length or self.prediction_length

        # Convert to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32)

        # Generate forecast
        forecast = self.pipeline.predict(
            context=context_tensor,
            prediction_length=pred_length,
            num_samples=self.num_samples
        )

        # Get statistics
        mean_pred = forecast.mean(dim=0).cpu().numpy()
        median_pred = forecast.median(dim=0).values.cpu().numpy()

        return mean_pred, median_pred

    def predict_returns(
        self,
        prices: np.ndarray,
        context_length: int = 60
    ) -> float:
        """
        Predict next-day return

        Args:
            prices: Historical price series
            context_length: Length of context window

        Returns:
            Predicted return for next day
        """
        if len(prices) < context_length:
            return np.nan

        # Use last context_length prices
        context = prices[-context_length:]

        # Predict next price
        mean_pred, _ = self.predict(context, prediction_length=1)
        next_price = mean_pred[0]

        # Calculate return
        current_price = prices[-1]
        predicted_return = (next_price / current_price) - 1

        return predicted_return


class ChronosFactorGenerator:
    """
    Generate Chronos-based factors for qlib
    """

    def __init__(
        self,
        predictor: ChronosPredictor,
        qlib_data_path: str,
        market: str = "csi300"
    ):
        """
        Initialize factor generator

        Args:
            predictor: ChronosPredictor instance
            qlib_data_path: Path to qlib data
            market: Market to generate factors for
        """
        self.predictor = predictor
        self.market = market

        # Initialize qlib
        if QLIB_AVAILABLE:
            try:
                qlib.init(provider_uri=qlib_data_path, region="cn")
                print(f"✓ Qlib initialized with {qlib_data_path}")
            except Exception as e:
                print(f"Warning: Qlib initialization failed: {e}")
                print(f"  Factor export will be limited")

    def generate_predictions(
        self,
        instruments: Optional[List[str]] = None,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31",
        context_length: int = 60,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate rolling predictions for all instruments

        Args:
            instruments: List of instruments (None = all in market)
            start_date: Start date for predictions
            end_date: End date for predictions
            context_length: Length of lookback window
            save_path: Path to save predictions

        Returns:
            DataFrame with predictions indexed by (datetime, instrument)
        """
        print(f"\n{'='*70}")
        print(f"GENERATING CHRONOS PREDICTIONS")
        print(f"{'='*70}")
        print(f"Market: {self.market}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Context length: {context_length}")

        # Get instruments
        if instruments is None and QLIB_AVAILABLE:
            instruments = D.list_instruments(market=self.market)
            print(f"Found {len(instruments)} instruments")
        elif instruments is None:
            raise ValueError("Instruments must be provided if qlib is not available")

        # Load price data
        print(f"\nLoading price data...")
        price_data = self._load_price_data(
            instruments,
            start_date,
            end_date,
            context_length
        )

        # Generate predictions
        print(f"\nGenerating predictions...")
        predictions = {}

        for i, instrument in enumerate(instruments, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(instruments)} instruments")

            try:
                inst_prices = price_data.xs(instrument, level='instrument')['$close']
                pred_series = self._predict_rolling(inst_prices, context_length)
                predictions[instrument] = pred_series
            except Exception as e:
                print(f"  Warning: Failed for {instrument}: {e}")
                continue

        # Combine into DataFrame
        pred_df = pd.DataFrame(predictions).stack()
        pred_df.index.names = ['datetime', 'instrument']
        pred_df.name = 'chronos_predicted_return'
        pred_df = pred_df.to_frame()

        print(f"\n✓ Generated {len(pred_df)} predictions")
        print(f"  Shape: {pred_df.shape}")
        print(f"  Date range: {pred_df.index.get_level_values('datetime').min()} to {pred_df.index.get_level_values('datetime').max()}")

        # Save if requested
        if save_path:
            pred_df.to_pickle(save_path)
            print(f"✓ Saved to {save_path}")

        return pred_df

    def _load_price_data(
        self,
        instruments: List[str],
        start_date: str,
        end_date: str,
        context_length: int
    ) -> pd.DataFrame:
        """Load price data with extended context"""
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib is required for data loading")

        # Extend start date to include context
        start_dt = pd.Timestamp(start_date)
        extended_start = (start_dt - pd.Timedelta(days=context_length * 2)).strftime("%Y-%m-%d")

        price_data = D.features(
            instruments=instruments,
            fields=["$close"],
            start_time=extended_start,
            end_time=end_date,
            freq="day"
        )

        return price_data

    def _predict_rolling(
        self,
        prices: pd.Series,
        context_length: int
    ) -> pd.Series:
        """
        Generate rolling predictions for a single instrument

        Args:
            prices: Price series
            context_length: Lookback window

        Returns:
            Series of predicted returns
        """
        predictions = []
        dates = []

        # Start from context_length
        for i in range(context_length, len(prices)):
            try:
                context_prices = prices.iloc[i-context_length:i].values
                pred_return = self.predictor.predict_returns(
                    context_prices,
                    context_length
                )
                predictions.append(pred_return)
                dates.append(prices.index[i])
            except Exception:
                predictions.append(np.nan)
                dates.append(prices.index[i])

        return pd.Series(predictions, index=dates)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate Chronos-2 predictions for stock factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model options
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "mini", "small", "base", "large"],
                        help="Chronos model size (default: small)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use (default: auto)")
    parser.add_argument("--cache_dir", type=str, default="./chronos_models",
                        help="Directory to cache models")

    # Prediction options
    parser.add_argument("--prediction_length", type=int, default=1,
                        help="Prediction horizon (default: 1)")
    parser.add_argument("--context_length", type=int, default=60,
                        help="Lookback window length (default: 60)")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples for forecasting (default: 20)")

    # Data options
    parser.add_argument("--market", type=str, default="csi300",
                        help="Market to use (default: csi300)")
    parser.add_argument("--start_date", type=str, default="2022-01-01",
                        help="Start date (default: 2022-01-01)")
    parser.add_argument("--end_date", type=str, default="2025-12-31",
                        help="End date (default: 2025-12-31)")
    parser.add_argument("--qlib_data_path", type=str,
                        default="/data/qlib_data/cn_data/qlib_bin",
                        help="Path to qlib data")

    # Output options
    parser.add_argument("--output", type=str, default="chronos_predictions.pkl",
                        help="Output file for predictions")
    parser.add_argument("--export_qlib", action="store_true",
                        help="Export in qlib factor format")

    args = parser.parse_args()

    # Check dependencies
    if not CHRONOS_AVAILABLE:
        print("Error: chronos-forecasting is not available")
        print("Install with: pip install chronos-forecasting")
        sys.exit(1)

    if not QLIB_AVAILABLE and args.export_qlib:
        print("Error: qlib is not available but --export_qlib was specified")
        sys.exit(1)

    # Initialize predictor
    print("\n" + "="*70)
    print("CHRONOS-2 TIME SERIES PREDICTION FOR QLIB")
    print("="*70)

    predictor = ChronosPredictor(
        model_size=args.model_size,
        device=args.device,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir
    )

    # Initialize factor generator
    if QLIB_AVAILABLE:
        factor_gen = ChronosFactorGenerator(
            predictor=predictor,
            qlib_data_path=args.qlib_data_path,
            market=args.market
        )

        # Generate predictions
        predictions = factor_gen.generate_predictions(
            start_date=args.start_date,
            end_date=args.end_date,
            context_length=args.context_length,
            save_path=args.output
        )

        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)
        print(predictions.describe())
        print(f"\nSample predictions:")
        print(predictions.head(20))

        print(f"\n✓ Successfully generated Chronos predictions!")
        print(f"  Saved to: {args.output}")

        if args.export_qlib:
            print(f"\nTo use as a qlib factor, load with:")
            print(f"  predictions = pd.read_pickle('{args.output}')")
            print(f"  # Then use in your qlib workflow")

    else:
        print("\nWarning: Qlib not available. Running in demo mode.")
        print("To use this script with real data, install qlib:")
        print("  pip install qlib")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
