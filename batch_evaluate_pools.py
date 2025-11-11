"""
Batch Evaluation Script for All Mined Factor Pools

Evaluates all factor pools from alphagen/, gfn_lstm/, and gfn_gnn/ directories.

Example Usage:
    # Evaluate all pools with default settings
    python batch_evaluate_pools.py --output_dir results/

    # Evaluate specific pools only
    python batch_evaluate_pools.py --pools alphagen gfn_lstm --output_dir results/

    # Evaluate with custom date ranges
    python batch_evaluate_pools.py --train_start 2010-01-01 --train_end 2019-12-31 \
        --test_start 2020-01-01 --test_end 2023-12-31

    # Run single factor mode (detailed per-factor metrics)
    python batch_evaluate_pools.py --single_factor --output_dir detailed_results/

    # Compare all pools and generate summary report
    python batch_evaluate_pools.py --compare_only --output comparison_report.csv
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from factor_analyzer import FactorAnalyzer


def load_factor_pool_simple(pool_path: str) -> Dict:
    """Load factor pool without evaluation dependencies."""
    with open(pool_path, 'r') as f:
        data = json.load(f)
    return data


# Define all factor pool locations
POOL_CONFIGS = {
    'alphagen': {
        '3m': 'alphagen/pool_3m.json',
        '6m': 'alphagen/pool_6m.json',
        '12m': 'alphagen/pool_12m.json',
        '36m': 'alphagen/pool_36m.json',
    },
    'gfn_lstm': {
        '6m': 'gfn_lstm/6m/pool_6m.json',
        '12m': 'gfn_lstm/12m/pool_12m.json',
        'all': 'gfn_lstm/all/pool_all.json',
    },
    'gfn_gnn': {
        '6m': 'gfn_gnn/6m/pool_6m.json',
        '12m': 'gfn_gnn/12m/pool_12m.json',
        'all': 'gfn_gnn/all/pool_all.json',
    }
}


def get_pool_paths(pool_filter: List[str] = None) -> List[Dict]:
    """
    Get all pool paths with metadata.

    Args:
        pool_filter: Optional list of pool names to filter (e.g., ['alphagen', 'gfn_lstm'])

    Returns:
        List of dictionaries with pool information
    """
    pools = []

    for method, windows in POOL_CONFIGS.items():
        if pool_filter and method not in pool_filter:
            continue

        for window, path in windows.items():
            if Path(path).exists():
                pools.append({
                    'method': method,
                    'window': window,
                    'path': path,
                    'name': f"{method}_{window}"
                })
            else:
                print(f"Warning: Pool file not found: {path}")

    return pools


def analyze_pool(pool_path: str) -> Dict:
    """Analyze factor expressions in a pool."""
    pool_data = load_factor_pool_simple(pool_path)
    analyzer = FactorAnalyzer(pool_data['exprs'])
    stats = analyzer.analyze()

    return {
        'num_factors': stats['num_expressions'],
        'avg_length': stats['length_stats']['mean_length'],
        'max_length': stats['length_stats']['max_length'],
        'avg_depth': stats['depth_stats']['mean_depth'],
        'max_depth': stats['depth_stats']['max_depth'],
        'avg_operators': stats['operator_stats']['avg_operators_per_expr'],
        'unique_operators': stats['operator_stats']['unique_operators'],
        'num_duplicates': len(stats['duplicates'])
    }


def evaluate_pool(
    pool_info: Dict,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    single_factor: bool = False,
    use_weights: bool = True
) -> Dict:
    """
    Evaluate a single factor pool.

    Args:
        pool_info: Dictionary with pool metadata
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date
        test_end: Test end date
        single_factor: If True, evaluate each factor individually
        use_weights: Whether to use weights from pool file

    Returns:
        Dictionary with evaluation results
    """
    # Import evaluation modules here to avoid dependency issues in compare-only mode
    from eval_mined_factors import load_factor_pool, evaluate_factor_pool as eval_pool

    print(f"\n{'='*70}")
    print(f"Evaluating: {pool_info['name']} ({pool_info['path']})")
    print(f"{'='*70}")

    try:
        # Load pool
        pool_data = load_factor_pool(pool_info['path'])
        factor_exprs = pool_data['exprs']

        # Get weights if available and requested
        weights = None
        if use_weights and 'weights' in pool_data:
            weights = pool_data['weights']

        # Run evaluation
        results = eval_pool(
            factor_exprs=factor_exprs,
            weights=weights,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            single_factor_mode=single_factor
        )

        # Add pool metadata
        results['method'] = pool_info['method']
        results['window'] = pool_info['window']
        results['pool_name'] = pool_info['name']
        results['pool_path'] = pool_info['path']

        # Add factor analysis
        analysis = analyze_pool(pool_info['path'])
        results['analysis'] = analysis

        print(f"\n✓ Evaluation completed for {pool_info['name']}")
        return results

    except Exception as e:
        print(f"\n✗ Error evaluating {pool_info['name']}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': pool_info['method'],
            'window': pool_info['window'],
            'pool_name': pool_info['name'],
            'error': str(e)
        }


def save_batch_results(
    results: List[Dict],
    output_dir: Path,
    single_factor: bool = False
):
    """Save batch evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if single_factor:
        # Save detailed per-factor results
        for result in results:
            if 'error' in result:
                continue

            pool_name = result['pool_name']
            output_file = output_dir / f"{pool_name}_detailed_{timestamp}.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"Saved detailed results: {output_file}")

    else:
        # Save summary results
        summary_data = []
        for result in results:
            if 'error' in result:
                summary_data.append({
                    'pool_name': result['pool_name'],
                    'method': result['method'],
                    'window': result['window'],
                    'status': 'ERROR',
                    'error': result['error']
                })
            else:
                row = {
                    'pool_name': result['pool_name'],
                    'method': result['method'],
                    'window': result['window'],
                    'num_factors': result['num_factors'],
                    'IC': result.get('IC', None),
                    'RankIC': result.get('RankIC', None),
                    'RRE': result.get('RRE', None),
                    'PFS1': result.get('PFS1', None),
                    'PFS2': result.get('PFS2', None),
                    'Diversity': result.get('Diversity', None),
                    'LLM_Score': result.get('LLM_Score', None),
                }

                # Add analysis stats
                if 'analysis' in result:
                    row.update({
                        'avg_length': result['analysis']['avg_length'],
                        'avg_depth': result['analysis']['avg_depth'],
                        'unique_operators': result['analysis']['unique_operators']
                    })

                summary_data.append(row)

        # Save as CSV
        df = pd.DataFrame(summary_data)
        csv_file = output_dir / f"summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nSummary results saved: {csv_file}")

        # Save full results as JSON
        json_file = output_dir / f"full_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Full results saved: {json_file}")

        # Print summary table
        print("\n" + "="*100)
        print("EVALUATION SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        print("\n")


def compare_pools_only(pool_filter: List[str] = None) -> pd.DataFrame:
    """
    Compare factor pools based on expression analysis only (no evaluation).

    Args:
        pool_filter: Optional list of pool names to filter

    Returns:
        DataFrame with comparison results
    """
    pools = get_pool_paths(pool_filter)
    comparison_data = []

    print("\n" + "="*70)
    print("ANALYZING FACTOR POOLS (No Evaluation)")
    print("="*70 + "\n")

    for pool in pools:
        print(f"Analyzing {pool['name']}...")
        analysis = analyze_pool(pool['path'])

        row = {
            'pool_name': pool['name'],
            'method': pool['method'],
            'window': pool['window'],
            **analysis
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate all mined factor pools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Pool selection
    parser.add_argument('--pools', type=str, nargs='+',
                        choices=['alphagen', 'gfn_lstm', 'gfn_gnn'],
                        help='Filter specific pool methods (default: all)')

    # Output options
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')

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
                        help='Evaluate each factor individually (detailed mode)')
    parser.add_argument('--no_weights', action='store_true',
                        help='Do not use weights from pool files')
    parser.add_argument('--compare_only', action='store_true',
                        help='Only compare pools without evaluation (fast)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Compare only mode
    if args.compare_only:
        comparison_df = compare_pools_only(args.pools)

        print("\n" + "="*100)
        print("POOL COMPARISON")
        print("="*100 + "\n")
        print(comparison_df.to_string(index=False))
        print("\n")

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comparison_{timestamp}.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"Comparison saved to {output_file}\n")
        return

    # Full evaluation mode
    pools = get_pool_paths(args.pools)

    if not pools:
        print("No pools found to evaluate!")
        sys.exit(1)

    print(f"\nFound {len(pools)} pools to evaluate")
    for pool in pools:
        print(f"  - {pool['name']}: {pool['path']}")

    # Evaluate all pools
    results = []
    for i, pool in enumerate(pools, 1):
        print(f"\n[{i}/{len(pools)}] Processing {pool['name']}...")

        result = evaluate_pool(
            pool_info=pool,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            single_factor=args.single_factor,
            use_weights=not args.no_weights
        )
        results.append(result)

    # Save results
    save_batch_results(results, output_dir, args.single_factor)

    print("\n" + "="*70)
    print("BATCH EVALUATION COMPLETED")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
