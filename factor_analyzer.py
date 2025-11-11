"""
Factor Expression Analyzer

Analyzes factor expressions to provide statistics about complexity,
operators used, and expression characteristics.

Example Usage:
    # Analyze factors from a single pool
    python factor_analyzer.py --pool alphagen/pool_3m.json

    # Compare multiple pools
    python factor_analyzer.py --pools alphagen/pool_3m.json alphagen/pool_6m.json \
        gfn_lstm/6m/pool_6m.json

    # Generate detailed report with operator frequency
    python factor_analyzer.py --pool gfn_gnn/12m/pool_12m.json \
        --detailed --output analysis_report.txt

    # Export analysis to JSON
    python factor_analyzer.py --pool alphagen/pool_36m.json \
        --output analysis.json
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter
import pandas as pd


class FactorAnalyzer:
    """Analyze factor expression characteristics."""

    # Common operators in qlib/alphagen syntax
    OPERATORS = [
        # Arithmetic
        'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Abs', 'Sign', 'Log',
        # Time series
        'Delta', 'Sum', 'Mean', 'Std', 'Var', 'Rank', 'CSRank',
        'WMA', 'EMA', 'DEMA', 'TEMA', 'RSI', 'Max', 'Min', 'Med',
        # Statistics
        'Corr', 'Cov', 'Skew', 'Kurt', 'Mad',
        # Logical
        'Greater', 'Less', 'Ref',
    ]

    # Stock features
    FEATURES = ['$open', '$close', '$high', '$low', '$volume', '$vwap']

    def __init__(self, expressions: List[str]):
        """
        Initialize analyzer with factor expressions.

        Args:
            expressions: List of factor expression strings
        """
        self.expressions = expressions
        self.stats = {}

    def extract_operators(self, expr: str) -> List[str]:
        """Extract all operators from an expression."""
        operators = []
        for op in self.OPERATORS:
            count = len(re.findall(rf'\b{op}\(', expr))
            operators.extend([op] * count)
        return operators

    def extract_features(self, expr: str) -> List[str]:
        """Extract all features from an expression."""
        features = []
        for feat in self.FEATURES:
            count = expr.count(feat)
            features.extend([feat] * count)
        return features

    def calculate_depth(self, expr: str) -> int:
        """Calculate maximum nesting depth of an expression."""
        max_depth = 0
        current_depth = 0
        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth

    def calculate_length_stats(self) -> Dict:
        """Calculate statistics about expression lengths."""
        lengths = [len(expr) for expr in self.expressions]
        return {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': sum(lengths) / len(lengths),
            'median_length': sorted(lengths)[len(lengths) // 2]
        }

    def calculate_operator_stats(self) -> Dict:
        """Calculate statistics about operator usage."""
        all_operators = []
        operator_counts_per_expr = []

        for expr in self.expressions:
            ops = self.extract_operators(expr)
            all_operators.extend(ops)
            operator_counts_per_expr.append(len(ops))

        operator_freq = Counter(all_operators)

        return {
            'total_operators': len(all_operators),
            'unique_operators': len(operator_freq),
            'operator_frequency': dict(operator_freq.most_common()),
            'avg_operators_per_expr': (
                sum(operator_counts_per_expr) / len(operator_counts_per_expr)
            ),
            'max_operators_per_expr': max(operator_counts_per_expr),
            'min_operators_per_expr': min(operator_counts_per_expr)
        }

    def calculate_feature_stats(self) -> Dict:
        """Calculate statistics about feature usage."""
        all_features = []
        for expr in self.expressions:
            features = self.extract_features(expr)
            all_features.extend(features)

        feature_freq = Counter(all_features)

        return {
            'total_features': len(all_features),
            'feature_frequency': dict(feature_freq.most_common()),
            'avg_features_per_expr': len(all_features) / len(self.expressions)
        }

    def calculate_depth_stats(self) -> Dict:
        """Calculate statistics about expression depth."""
        depths = [self.calculate_depth(expr) for expr in self.expressions]
        return {
            'min_depth': min(depths),
            'max_depth': max(depths),
            'mean_depth': sum(depths) / len(depths),
            'median_depth': sorted(depths)[len(depths) // 2]
        }

    def find_duplicates(self) -> List[str]:
        """Find duplicate expressions."""
        expr_counts = Counter(self.expressions)
        return [expr for expr, count in expr_counts.items() if count > 1]

    def analyze(self) -> Dict:
        """
        Perform comprehensive analysis of factor expressions.

        Returns:
            Dictionary containing all analysis results
        """
        print(f"Analyzing {len(self.expressions)} factor expressions...")

        self.stats = {
            'num_expressions': len(self.expressions),
            'length_stats': self.calculate_length_stats(),
            'operator_stats': self.calculate_operator_stats(),
            'feature_stats': self.calculate_feature_stats(),
            'depth_stats': self.calculate_depth_stats(),
            'duplicates': self.find_duplicates()
        }

        return self.stats

    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        if not self.stats:
            self.analyze()

        print("\n" + "="*70)
        print("FACTOR ANALYSIS SUMMARY")
        print("="*70)

        print(f"\nTotal Expressions: {self.stats['num_expressions']}")

        print("\n--- Expression Length ---")
        ls = self.stats['length_stats']
        print(f"  Min: {ls['min_length']}, Max: {ls['max_length']}, "
              f"Mean: {ls['mean_length']:.1f}, Median: {ls['median_length']}")

        print("\n--- Expression Depth ---")
        ds = self.stats['depth_stats']
        print(f"  Min: {ds['min_depth']}, Max: {ds['max_depth']}, "
              f"Mean: {ds['mean_depth']:.1f}, Median: {ds['median_depth']}")

        print("\n--- Operator Usage ---")
        ops = self.stats['operator_stats']
        print(f"  Total operators: {ops['total_operators']}")
        print(f"  Unique operators: {ops['unique_operators']}")
        print(f"  Avg per expression: {ops['avg_operators_per_expr']:.1f}")
        print(f"\n  Top 10 operators:")
        for op, count in list(ops['operator_frequency'].items())[:10]:
            print(f"    {op}: {count}")

        print("\n--- Feature Usage ---")
        fs = self.stats['feature_stats']
        print(f"  Total features: {fs['total_features']}")
        print(f"  Avg per expression: {fs['avg_features_per_expr']:.1f}")
        print(f"\n  Feature frequency:")
        for feat, count in fs['feature_frequency'].items():
            print(f"    {feat}: {count}")

        if self.stats['duplicates']:
            print(f"\n--- Duplicates ---")
            print(f"  Found {len(self.stats['duplicates'])} duplicate expressions")

        print("\n" + "="*70 + "\n")


def load_pool(pool_path: str) -> List[str]:
    """Load factor expressions from JSON pool file."""
    with open(pool_path, 'r') as f:
        data = json.load(f)
    return data['exprs']


def compare_pools(pool_paths: List[str]) -> pd.DataFrame:
    """
    Compare statistics across multiple factor pools.

    Args:
        pool_paths: List of paths to pool JSON files

    Returns:
        DataFrame with comparison statistics
    """
    comparison = []

    for pool_path in pool_paths:
        print(f"\nAnalyzing {pool_path}...")
        expressions = load_pool(pool_path)
        analyzer = FactorAnalyzer(expressions)
        stats = analyzer.analyze()

        row = {
            'pool': Path(pool_path).name,
            'num_factors': stats['num_expressions'],
            'avg_length': stats['length_stats']['mean_length'],
            'avg_depth': stats['depth_stats']['mean_depth'],
            'avg_operators': stats['operator_stats']['avg_operators_per_expr'],
            'unique_operators': stats['operator_stats']['unique_operators'],
            'duplicates': len(stats['duplicates'])
        }
        comparison.append(row)

    return pd.DataFrame(comparison)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze factor expressions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--pool', type=str,
                        help='Path to single factor pool JSON file')
    parser.add_argument('--pools', type=str, nargs='+',
                        help='Paths to multiple pool files for comparison')
    parser.add_argument('--output', type=str,
                        help='Output file path (.json or .txt)')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed analysis')

    args = parser.parse_args()

    if not args.pool and not args.pools:
        parser.error("Must specify either --pool or --pools")

    # Single pool analysis
    if args.pool:
        expressions = load_pool(args.pool)
        analyzer = FactorAnalyzer(expressions)
        stats = analyzer.analyze()
        analyzer.print_summary()

        if args.output:
            output_path = Path(args.output)
            if output_path.suffix == '.json':
                with open(output_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Analysis saved to {output_path}")
            elif output_path.suffix == '.txt':
                # Redirect print to file
                import sys
                original_stdout = sys.stdout
                with open(output_path, 'w') as f:
                    sys.stdout = f
                    analyzer.print_summary()
                    sys.stdout = original_stdout
                print(f"Analysis saved to {output_path}")

    # Multiple pool comparison
    elif args.pools:
        comparison_df = compare_pools(args.pools)
        print("\n" + "="*70)
        print("POOL COMPARISON")
        print("="*70 + "\n")
        print(comparison_df.to_string(index=False))
        print("\n")

        if args.output:
            output_path = Path(args.output)
            if output_path.suffix == '.csv':
                comparison_df.to_csv(output_path, index=False)
                print(f"Comparison saved to {output_path}")
            elif output_path.suffix == '.json':
                comparison_df.to_json(output_path, orient='records', indent=2)
                print(f"Comparison saved to {output_path}")


if __name__ == '__main__':
    main()
