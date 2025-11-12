"""
AlphaEval Factor Evaluation Script for Mined Factors

Example Usage:
    # Evaluate factors from alphagen pool_3m.json
    python eval_mined_factors.py --pool alphagen/pool_3m.json --output results_3m.csv

    # Evaluate with custom date ranges
    python eval_mined_factors.py --pool gfn_lstm/6m/pool_6m.json \
        --train_start 2010-01-01 --train_end 2019-12-31 \
        --test_start 2020-01-01 --test_end 2023-12-31

    # Evaluate using AM/PM feature store data
    python eval_mined_factors.py --pool gfn_gnn/12m/pool_12m.json \
        --feature_store /path/to/feature_store \
        --use_ampm_data

    # Run single factor evaluation (detailed metrics per factor)
    python eval_mined_factors.py --pool alphagen/pool_36m.json \
        --single_factor --output detailed_results.json
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add backtest module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backtest'))

from backtest.modeltester import AlphaEval


def load_factor_pool(pool_path: str) -> Dict:
    """
    Load factor expressions and weights from JSON pool file.

    Args:
        pool_path: Path to the JSON file containing factor expressions

    Returns:
        Dictionary with 'exprs' (factor expressions) and 'weights' (optional)
    """
    with open(pool_path, 'r') as f:
        data = json.load(f)

    if 'exprs' not in data:
        raise ValueError(f"JSON file {pool_path} must contain 'exprs' field")

    print(f"Loaded {len(data['exprs'])} factors from {pool_path}")

    # Print metadata if available
    if 'window_results' in data:
        print(f"  Metadata: {list(data['window_results'].keys())}")
    if 'weights' in data:
        print(f"  Weights: {len(data['weights'])} values")

    return data


def evaluate_factor_pool(
    factor_exprs: List[str],
    weights: Optional[List[float]] = None,
    train_start: str = "2010-01-01",
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
    test_end: str = "2023-12-31",
    instruments: Optional[List[str]] = None,
    daily_normalize: bool = True,
    single_factor_mode: bool = False,
    api_key: Optional[str] = None
) -> Dict:
    """
    Evaluate a pool of factors using AlphaEval framework.

    Args:
        factor_exprs: List of factor expression strings
        weights: Optional weights for combining factors
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date
        test_end: Test end date
        instruments: Optional list of instruments (default: CSI300)
        daily_normalize: Whether to apply daily normalization
        single_factor_mode: If True, evaluate each factor individually

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {len(factor_exprs)} factors")
    print(f"Train: {train_start} to {train_end}")
    print(f"Test:  {test_start} to {test_end}")
    print(f"{'='*60}\n")

    # Initialize evaluator
    evaluator = AlphaEval(
        factor_expressions=factor_exprs,
        weights=weights,
        train_start_date=train_start,
        train_end_date=train_end,
        test_start_date=test_start,
        test_end_date=test_end,
        instruments=instruments,
        daily_normalize=daily_normalize
    )

    if single_factor_mode:
        # Evaluate each factor individually
        print("Running single factor evaluation...")
        results = evaluator.run_single_factor(api_key=api_key)
        return {
            'mode': 'single_factor',
            'results': results,
            'num_factors': len(factor_exprs)
        }
    else:
        # Evaluate combined factor pool
        print("Running combined evaluation...")
        evaluator.run(api_key=api_key)

        results = {
            'mode': 'combined',
            'IC': evaluator.ic,
            'RankIC': evaluator.rankic,
            'RRE': evaluator.rre,
            'PFS1': evaluator.pfs1,
            'PFS2': evaluator.pfs2,
            'Diversity': evaluator.diversity,
            'LLM_Score': evaluator.llm_avg_score,
            'num_factors': len(factor_exprs)
        }

        evaluator.summary()
        return results


def save_results(results: Dict, output_path: str):
    """Save evaluation results to file."""
    output_path = Path(output_path)

    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    elif output_path.suffix == '.csv':
        if results['mode'] == 'single_factor':
            df = pd.DataFrame(results['results'])
        else:
            df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        raise ValueError("Output file must be .json or .csv")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate mined factors using AlphaEval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument('--pool', type=str, required=True,
                        help='Path to factor pool JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                        help='Output file path (.csv or .json)')

    # Date ranges
    parser.add_argument('--train_start', type=str, default='2010-01-01',
                        help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2019-12-31',
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default='2020-01-01',
                        help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default='2023-12-31',
                        help='Test end date (YYYY-MM-DD)')

    # Evaluation options
    parser.add_argument('--single_factor', action='store_true',
                        help='Evaluate each factor individually')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Disable daily normalization')
    parser.add_argument('--use_weights', action='store_true',
                        help='Use weights from JSON file if available')

    # AM/PM data options
    parser.add_argument('--use_ampm_data', action='store_true',
                        help='Use AM/PM feature store data (experimental)')
    parser.add_argument('--feature_store', type=str, default=None,
                        help='Path to AM/PM feature store directory')

    # LLM options
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key for LLM evaluation (or set OPENAI_API_KEY env var)')

    args = parser.parse_args()

    # Load factor pool
    pool_data = load_factor_pool(args.pool)
    factor_exprs = pool_data['exprs']

    # Get weights if requested
    weights = None
    if args.use_weights and 'weights' in pool_data:
        weights = pool_data['weights']
        print(f"Using {len(weights)} weights from pool file")

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY') or os.environ.get('LLM_API_KEY')

    # Run evaluation
    try:
        results = evaluate_factor_pool(
            factor_exprs=factor_exprs,
            weights=weights,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            daily_normalize=not args.no_normalize,
            single_factor_mode=args.single_factor,
            api_key=api_key
        )

        # Add metadata
        results['pool_path'] = args.pool
        results['train_period'] = f"{args.train_start} to {args.train_end}"
        results['test_period'] = f"{args.test_start} to {args.test_end}"

        # Save results
        save_results(results, args.output)

        print(f"\n{'='*60}")
        print("Evaluation completed successfully!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
