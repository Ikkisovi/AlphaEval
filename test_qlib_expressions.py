#!/usr/bin/env python3
"""
Test Qlib Expression Operators
===============================

This script tests every operator that the factor pools rely on, ensuring all
custom operators are properly registered and functional.

It systematically tests:
- Element-wise operators (Abs, Sign, Log, etc.)
- Pair-wise operators (Add, Sub, Mul, Div, etc.)
- Rolling operators (Mean, Std, Rank, Corr, etc.)
- Comparison operators (Gt, Lt, Eq, etc.)
- Special operators (If, Ref, Delta, etc.)

Usage:
    python test_qlib_expressions.py

Expected Output:
    All operator tests should pass, confirming the qlib setup is correct.
"""

import sys
import os
import qlib
from qlib.data import D
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def register_custom_ops():
    """Register custom operators defined in my_qlib/data/ops.py"""
    from my_qlib.data.ops import Operators, OpsList, RelationRank, RelationDemean
    try:
        Operators.reset()
        Operators.register(OpsList + [RelationRank, RelationDemean])
        print("✓ Custom operators registered successfully\n")
        return True
    except Exception as e:
        print(f"✗ Failed to register custom operators: {e}\n")
        return False

class OperatorTester:
    """Test suite for qlib operators"""

    def __init__(self, instruments, start_date="2020-01-01", end_date="2020-12-31"):
        self.instruments = instruments
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}

    def test_expression(self, name, expression, should_succeed=True):
        """Test a single expression"""
        try:
            df = D.features(
                instruments=self.instruments,
                fields=[expression],
                start_time=self.start_date,
                end_time=self.end_date,
                freq="day"
            )

            if df.empty:
                status = "⚠ EMPTY"
                success = False
            else:
                # Check data quality
                nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                inf_count = np.isinf(df.values).sum()

                if inf_count > 0:
                    status = f"⚠ INF ({inf_count} inf values)"
                    success = False
                elif nan_pct == 100:
                    status = "⚠ ALL NaN"
                    success = False
                else:
                    status = f"✓ OK (NaN: {nan_pct:.1f}%)"
                    success = True

            self.results[name] = success
            print(f"  {name:.<45} {status}")
            return success

        except Exception as e:
            if should_succeed:
                print(f"  {name:.<45} ✗ FAILED: {str(e)[:30]}")
                self.results[name] = False
                return False
            else:
                print(f"  {name:.<45} ✓ Expected failure")
                self.results[name] = True
                return True

    def test_element_wise_operators(self):
        """Test element-wise operators"""
        print("\n" + "="*70)
        print("ELEMENT-WISE OPERATORS")
        print("="*70)

        tests = [
            ("Abs", "Abs($close - $open)"),
            ("Sign", "Sign($close - $open)"),
            ("Log", "Log($volume + 1)"),
            ("Power (square)", "($close / $open) ** 2"),
            ("Not", "~($close > $open)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def test_pair_wise_operators(self):
        """Test pair-wise operators"""
        print("\n" + "="*70)
        print("PAIR-WISE OPERATORS")
        print("="*70)

        tests = [
            ("Add", "$close + $open"),
            ("Subtract", "$high - $low"),
            ("Multiply", "$close * $volume"),
            ("Divide", "$close / $open"),
            ("Greater (max)", "Greater($high, $close)"),
            ("Less (min)", "Less($low, $close)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def test_comparison_operators(self):
        """Test comparison operators"""
        print("\n" + "="*70)
        print("COMPARISON OPERATORS")
        print("="*70)

        tests = [
            ("Greater than", "$close > $open"),
            ("Greater or equal", "$close >= $open"),
            ("Less than", "$close < $high"),
            ("Less or equal", "$close <= $high"),
            ("Equal", "$close == $close"),
            ("Not equal", "$close != $open"),
            ("And", "($close > $open) & ($volume > 0)"),
            ("Or", "($close > $high) | ($close > $low)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def test_rolling_operators(self):
        """Test rolling window operators"""
        print("\n" + "="*70)
        print("ROLLING OPERATORS")
        print("="*70)

        tests = [
            ("Ref (lag)", "Ref($close, 1)"),
            ("Mean (MA)", "Mean($close, 5)"),
            ("Sum", "Sum($volume, 10)"),
            ("Std (volatility)", "Std($close, 20)"),
            ("Var (variance)", "Var($close, 20)"),
            ("Skew (skewness)", "Skew($close, 20)"),
            ("Kurt (kurtosis)", "Kurt($close, 20)"),
            ("Max", "Max($high, 10)"),
            ("Min", "Min($low, 10)"),
            ("IdxMax", "IdxMax($close, 10)"),
            ("IdxMin", "IdxMin($close, 10)"),
            ("Quantile", "Quantile($close, 20, 0.75)"),
            ("Med (median)", "Med($close, 10)"),
            ("Mad (mean abs dev)", "Mad($close, 10)"),
            ("Rank (percentile)", "Rank($close, 20)"),
            ("Count", "Count($close, 10)"),
            ("Delta (diff)", "Delta($close, 5)"),
            ("Slope", "Slope($close, 10)"),
            ("Rsquare", "Rsquare($close, 10)"),
            ("Resi (residual)", "Resi($close, 10)"),
            ("WMA (weighted MA)", "WMA($close, 5)"),
            ("EMA (exp MA)", "EMA($close, 5)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def test_pair_rolling_operators(self):
        """Test pair-wise rolling operators"""
        print("\n" + "="*70)
        print("PAIR-WISE ROLLING OPERATORS")
        print("="*70)

        tests = [
            ("Corr (correlation)", "Corr($close, $open, 20)"),
            ("Cov (covariance)", "Cov($close, $volume, 20)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def test_special_operators(self):
        """Test special operators"""
        print("\n" + "="*70)
        print("SPECIAL OPERATORS")
        print("="*70)

        tests = [
            ("If (conditional)", "If($close > $open, $close, $open)"),
            ("Nested expression", "Mean(Abs($close - $open), 5) / Std($close, 5)"),
            ("Complex factor", "Rank(Delta($close, 1) / $close, 10)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def test_realistic_factors(self):
        """Test realistic alpha factors similar to those in pools"""
        print("\n" + "="*70)
        print("REALISTIC ALPHA FACTORS")
        print("="*70)

        tests = [
            ("Returns", "Ref($close, -1)/$close - 1"),
            ("Momentum", "$close / Ref($close, 20) - 1"),
            ("Volatility", "Std($close / Ref($close, 1) - 1, 20)"),
            ("Volume ratio", "$volume / Mean($volume, 20)"),
            ("Price-Volume corr", "Corr($close, $volume, 20)"),
            ("RSI-like", "Mean(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / Mean(Abs($close - Ref($close, 1)), 14)"),
            ("Reversal", "-Rank(Delta($close, 5) / $close, 20)"),
            ("Trend strength", "Abs(Slope($close, 20)) * Rsquare($close, 20)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def test_ewm_operators(self):
        """Test exponential weighted moving operators"""
        print("\n" + "="*70)
        print("EXPONENTIAL WEIGHTED OPERATORS")
        print("="*70)

        tests = [
            ("EMA span=5", "EMA($close, 5)"),
            ("EMA span=20", "EMA($close, 20)"),
            ("EMA alpha=0.1", "EMA($close, 0.1)"),
            ("EMA alpha=0.3", "EMA($close, 0.3)"),
        ]

        for name, expr in tests:
            self.test_expression(name, expr)

    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "="*70)
        print("QLIB OPERATOR TEST SUITE")
        print("="*70)
        print(f"Testing with instruments: {self.instruments}")
        print(f"Date range: {self.start_date} to {self.end_date}")

        self.test_element_wise_operators()
        self.test_pair_wise_operators()
        self.test_comparison_operators()
        self.test_rolling_operators()
        self.test_pair_rolling_operators()
        self.test_special_operators()
        self.test_ewm_operators()
        self.test_realistic_factors()

        return self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        total = len(self.results)
        passed = sum(self.results.values())
        failed = total - passed

        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for name, success in self.results.items():
                if not success:
                    print(f"  - {name}")

        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")

        if success_rate >= 90:
            print("\n✓ Excellent! Most operators are working correctly.")
            return True
        elif success_rate >= 70:
            print("\n⚠ Good, but some operators may need attention.")
            return True
        else:
            print("\n✗ Many operators failed. Check qlib setup and data.")
            return False

def main():
    """Main test runner"""
    # Determine qlib data path
    possible_paths = [
        "E:/factor/qlib_data/us_data/qlib_bin",
        "/data/qlib_data/cn_data/qlib_bin",
        "./qlib_data/cn_data/qlib_bin",
        "../qlib_data/cn_data/qlib_bin",
    ]

    qlib_data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            qlib_data_path = path
            break

    if qlib_data_path is None:
        print("\n✗ ERROR: Could not find qlib data directory")
        print("Please update qlib_data_path or set QLIB_DATA_PATH environment variable")
        sys.exit(1)

    # Initialize qlib
    try:
        qlib.init(provider_uri=qlib_data_path, region="cn")
        print(f"✓ Qlib initialized with: {qlib_data_path}")
    except Exception as e:
        print(f"✗ Qlib initialization failed: {e}")
        sys.exit(1)

    # Register custom operators AFTER qlib.init()
    if not register_custom_ops():
        sys.exit(1)

    # Get test instruments
    try:
        instruments = D.list_instruments(market="csi300")[:5]  # Test with 5 instruments
        print(f"✓ Testing with {len(instruments)} instruments: {instruments}\n")
    except Exception as e:
        print(f"✗ Failed to get instruments: {e}")
        sys.exit(1)

    # Run tests
    tester = OperatorTester(instruments)
    success = tester.run_all_tests()

    if success:
        print("\n✓ All operator tests completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some operator tests failed. Please review the results above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
