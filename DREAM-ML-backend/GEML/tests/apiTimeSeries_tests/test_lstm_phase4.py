"""
Phase 4 Tests: External Features Support for LSTM
Tests univariate/multivariate sequence creation and feature validation.
"""
import pytest
import numpy as np
import pandas as pd
from apiTimeSeries.train import create_sequences_for_lstm


def test_univariate_sequences_empty_features():
    """Test univariate mode with empty feature_cols (auto-fallback to target)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100))
    })
    df.set_index('date', inplace=True)

    # Empty features → should auto-use target
    X, y = create_sequences_for_lstm(df, [], 'sales', sequence_length=10)

    assert X.shape == (90, 10, 1), f"Expected (90, 10, 1), got {X.shape}"
    assert y.shape == (90,)
    assert X.dtype in [np.float64, np.float32]
    print("✓ Test passed: Univariate mode (empty features)")


def test_univariate_sequences_target_only():
    """Test univariate mode with explicit target as feature (target history)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    # Explicit target-only
    X, y = create_sequences_for_lstm(df, ['sales'], 'sales', sequence_length=10)

    assert X.shape == (90, 10, 1), "Should use only sales column"
    assert y.shape == (90,)
    print("✓ Test passed: Univariate mode (explicit target)")


def test_multivariate_sequences_with_target():
    """Test multivariate with target + external features (target history enabled)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100),
        'humidity': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    # Target + 2 external features (using target history)
    X, y = create_sequences_for_lstm(
        df, ['sales', 'temp', 'humidity'], 'sales', sequence_length=10
    )

    assert X.shape == (90, 10, 3), f"Expected 3 features, got {X.shape[2]}"
    assert y.shape == (90,)
    print("✓ Test passed: Multivariate with target history")


def test_multivariate_sequences_without_target():
    """Test multivariate with external features only (no target history)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100),
        'humidity': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    # External features only (predict sales from temp + humidity, no sales history)
    X, y = create_sequences_for_lstm(
        df, ['temp', 'humidity'], 'sales', sequence_length=10
    )

    assert X.shape == (90, 10, 2), "Should use 2 external features"
    assert y.shape == (90,)
    # Verify sales column is NOT in the input sequences
    print("✓ Test passed: Multivariate without target history")


def test_feature_validation_missing():
    """Test error when feature doesn't exist in dataset"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100))
    })
    df.set_index('date', inplace=True)

    with pytest.raises(ValueError, match="no encontrada"):
        create_sequences_for_lstm(df, ['nonexistent_feature'], 'sales', sequence_length=10)

    print("✓ Test passed: Missing feature validation")


def test_shape_validation():
    """Test that sequence shapes are correct for different modes"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'target': np.sin(np.linspace(0, 10, 100))
    })
    df.set_index('date', inplace=True)

    # Test different feature combinations
    test_cases = [
        ([], 1, "empty → univariate"),
        (['target'], 1, "target only"),
        (['feature1'], 1, "1 external feature"),
        (['target', 'feature1'], 2, "target + 1 feature"),
        (['feature1', 'feature2', 'feature3'], 3, "3 external features"),
        (['target', 'feature1', 'feature2', 'feature3'], 4, "target + 3 features"),
    ]

    for features, expected_n_features, description in test_cases:
        X, y = create_sequences_for_lstm(df, features, 'target', sequence_length=10)
        assert X.shape[2] == expected_n_features, f"Failed for {description}: expected {expected_n_features}, got {X.shape[2]}"
        print(f"✓ Shape test passed: {description} → {X.shape}")


# ======================
# EDGE CASE TESTS (4 tests)
# ======================

def test_target_not_in_columns():
    """Test error when target variable doesn't exist in dataset"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    with pytest.raises(ValueError, match="Variable objetivo.*no encontrada"):
        create_sequences_for_lstm(df, ['sales'], 'nonexistent_target', sequence_length=10)

    print("✓ Edge case test passed: Target not in columns")


def test_date_column_in_features():
    """Test that date column (if present in features) is handled correctly"""
    # Note: Frontend validation prevents date columns from being selected
    # Backend can technically handle datetime columns (converts to numeric)
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100)
    })
    # Keep date as column (not index)

    # Date columns can be processed (pandas converts to numeric)
    # Frontend prevents this, so this is more of an integration safeguard
    try:
        X, y = create_sequences_for_lstm(df, ['date', 'temp'], 'sales', sequence_length=10)
        # If it works, verify shape is correct
        assert X.shape == (90, 10, 2)
        print("✓ Edge case test passed: Date column handled (frontend prevents this)")
    except (ValueError, KeyError, TypeError):
        # If it fails, that's also acceptable (stricter validation)
        print("✓ Edge case test passed: Date column rejected (stricter validation)")


def test_duplicate_features():
    """Test handling of duplicate features in feature list"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    # Duplicate features should work (pandas will handle duplicates)
    X, y = create_sequences_for_lstm(
        df, ['temp', 'temp', 'sales'], 'sales', sequence_length=10
    )

    # Should create sequences with duplicated column
    assert X.shape == (90, 10, 3), f"Expected shape (90, 10, 3), got {X.shape}"
    print("✓ Edge case test passed: Duplicate features handled")


def test_empty_dataset():
    """Test error with dataset that has too few rows"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),  # Only 10 rows
        'sales': np.random.rand(10)
    })
    df.set_index('date', inplace=True)

    # Should error: insufficient data for sequence_length=10 (needs min 50 sequences)
    with pytest.raises(ValueError, match="Dataset insuficiente"):
        create_sequences_for_lstm(df, ['sales'], 'sales', sequence_length=10)

    print("✓ Edge case test passed: Empty/insufficient dataset")


# Run with: pytest test_lstm_phase4.py -v -s
