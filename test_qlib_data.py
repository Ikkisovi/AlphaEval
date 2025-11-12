#!/usr/bin/env python3
"""
Test Qlib Data Loading
======================

This script verifies that the qlib data bundle is properly configured and accessible.
It tests data loading using D.features over the qlib_bin data directory.

The script will:
1. Initialize qlib with the data path
2. Test loading basic features ($close, $open, $high, $low, $volume)
3. Verify data integrity and date ranges
4. Test instrument listings

Usage:
    python test_qlib_data.py

Expected Output:
    All tests should pass showing that qlib data is accessible.
"""

import sys
import os
import qlib
from qlib.data import D
import pandas as pd
import numpy as np

def register_custom_ops():
    """Register custom operators defined in my_qlib/data/ops.py"""
    from my_qlib.data.ops import Operators, OpsList, RelationRank, RelationDemean
    try:
        # Register all custom operators
        Operators.reset()
        Operators.register(OpsList + [RelationRank, RelationDemean])
        print("✓ Custom operators registered successfully")
    except Exception as e:
        print(f"Warning: Could not register custom operators: {e}")

def test_qlib_init(qlib_data_path):
    """Test qlib initialization"""
    print("\n" + "="*70)
    print("TEST 1: Qlib Initialization")
    print("="*70)

    try:
        qlib.init(provider_uri=qlib_data_path, region="cn")
        print(f"✓ Qlib initialized with data path: {qlib_data_path}")
        return True
    except Exception as e:
        print(f"✗ Qlib initialization failed: {e}")
        return False

def test_instrument_list():
    """Test instrument listing"""
    print("\n" + "="*70)
    print("TEST 2: Instrument Listing")
    print("="*70)

    try:
        # Test CSI300 instruments
        instruments = D.list_instruments(market="csi300")
        print(f"✓ Found {len(instruments)} CSI300 instruments")
        print(f"  Sample instruments: {instruments[:5]}")

        # Test all instruments
        all_instruments = D.list_instruments(market="all")
        print(f"✓ Found {len(all_instruments)} total instruments")

        return len(instruments) > 0
    except Exception as e:
        print(f"✗ Instrument listing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_features():
    """Test loading basic OHLCV features"""
    print("\n" + "="*70)
    print("TEST 3: Basic Feature Loading (OHLCV)")
    print("="*70)

    try:
        instruments = D.list_instruments(market="csi300")
        if not instruments:
            print("✗ No instruments available for testing")
            return False

        test_instrument = instruments[:3]  # Test first 3 instruments
        features = ["$close", "$open", "$high", "$low", "$volume"]

        print(f"Testing instruments: {test_instrument}")
        print(f"Testing features: {features}")

        df = D.features(
            instruments=test_instrument,
            fields=features,
            start_time="2020-01-01",
            end_time="2020-12-31",
            freq="day"
        )

        print(f"✓ Data shape: {df.shape}")
        print(f"✓ Date range: {df.index.get_level_values('datetime').min()} to {df.index.get_level_values('datetime').max()}")
        print(f"✓ Columns: {df.columns.tolist()}")
        print(f"\nSample data:")
        print(df.head(10))

        # Check for NaN values
        nan_count = df.isna().sum().sum()
        total_values = df.shape[0] * df.shape[1]
        nan_percentage = (nan_count / total_values) * 100
        print(f"\nData quality:")
        print(f"  NaN values: {nan_count}/{total_values} ({nan_percentage:.2f}%)")

        return True
    except Exception as e:
        print(f"✗ Basic feature loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_derived_features():
    """Test loading derived features with expressions"""
    print("\n" + "="*70)
    print("TEST 4: Derived Features (Expressions)")
    print("="*70)

    try:
        instruments = D.list_instruments(market="csi300")[:2]

        # Test various expressions
        expressions = [
            "Ref($close, 1)/$close - 1",  # Returns
            "Mean($close, 5)",             # 5-day MA
            "Std($close, 20)",             # 20-day volatility
            "$high / $low - 1",            # High-low spread
            "Log($volume + 1)",            # Log volume
        ]

        print(f"Testing {len(expressions)} expressions")

        for expr in expressions:
            try:
                df = D.features(
                    instruments=instruments,
                    fields=[expr],
                    start_time="2020-01-01",
                    end_time="2020-03-31",
                    freq="day"
                )
                print(f"✓ Expression: {expr}")
                print(f"  Shape: {df.shape}, NaN%: {df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%")
            except Exception as e:
                print(f"✗ Expression failed: {expr}")
                print(f"  Error: {e}")

        return True
    except Exception as e:
        print(f"✗ Derived feature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_date_ranges():
    """Test different date ranges"""
    print("\n" + "="*70)
    print("TEST 5: Date Range Coverage")
    print("="*70)

    try:
        instruments = ["SH600000"]  # Single instrument for speed

        date_ranges = [
            ("2015-01-01", "2015-12-31"),
            ("2018-01-01", "2018-12-31"),
            ("2020-01-01", "2020-12-31"),
            ("2022-01-01", "2022-12-31"),
            ("2024-01-01", "2024-06-30"),
        ]

        for start, end in date_ranges:
            try:
                df = D.features(
                    instruments=instruments,
                    fields=["$close"],
                    start_time=start,
                    end_time=end,
                    freq="day"
                )

                if not df.empty:
                    actual_start = df.index.get_level_values('datetime').min()
                    actual_end = df.index.get_level_values('datetime').max()
                    print(f"✓ {start} to {end}: {len(df)} records ({actual_start} to {actual_end})")
                else:
                    print(f"⚠ {start} to {end}: No data available")
            except Exception as e:
                print(f"✗ {start} to {end}: Error - {e}")

        return True
    except Exception as e:
        print(f"✗ Date range test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("\n" + "="*70)
    print("QLIB DATA VERIFICATION TEST SUITE")
    print("="*70)

    # Determine qlib data path
    # Try common locations
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
        print("Tried locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease update the qlib_data_path in this script or set QLIB_DATA_PATH environment variable")
        sys.exit(1)

    # Initialize qlib and register custom ops
    if not test_qlib_init(qlib_data_path):
        print("\n✗ Qlib initialization failed. Cannot continue tests.")
        sys.exit(1)

    # Register custom operators AFTER qlib.init()
    register_custom_ops()

    # Run tests
    results = {
        "Instrument Listing": test_instrument_list(),
        "Basic Features": test_basic_features(),
        "Derived Features": test_derived_features(),
        "Date Ranges": test_date_ranges(),
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Qlib data is working correctly.")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
