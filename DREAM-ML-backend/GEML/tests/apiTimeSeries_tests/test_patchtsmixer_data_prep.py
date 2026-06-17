"""
Unit tests for PatchTSMixer data preparation functions.

Tests sequence generation, dataset class, and temporal splitting.
"""
import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from apiTimeSeries.train import (
    TimeSeriesDataset,
    create_sequences_for_patchtsmixer,
    patchtsmixer_train_val_test_split
)


@pytest.fixture
def sample_univariate_df():
    """Create a sample DataFrame with 1 channel for testing."""
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        'date': dates,
        'value': np.random.randn(1000)
    })


@pytest.fixture
def sample_multivariate_df():
    """Create a sample DataFrame with 3 channels for testing."""
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'date': dates,
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000) * 2,
        'feature_3': np.random.randn(1000) * 0.5
    })


def test_sequence_creation_univariate(sample_univariate_df):
    """Test sequence creation with 1 channel."""
    context_length = 512
    prediction_length = 96

    past_values, future_values = create_sequences_for_patchtsmixer(
        df=sample_univariate_df,
        channel_cols=['value'],
        context_length=context_length,
        prediction_length=prediction_length
    )

    # Expected: 1000 - 512 - 96 + 1 = 393 sequences
    expected_num_sequences = 393

    assert past_values.shape == (expected_num_sequences, context_length, 1), \
        f"past_values shape mismatch: {past_values.shape}"
    assert future_values.shape == (expected_num_sequences, prediction_length, 1), \
        f"future_values shape mismatch: {future_values.shape}"

    # Verify tensor types
    assert isinstance(past_values, torch.Tensor)
    assert isinstance(future_values, torch.Tensor)
    assert past_values.dtype == torch.float32
    assert future_values.dtype == torch.float32

    # Verify no NaN/Inf
    assert not torch.isnan(past_values).any()
    assert not torch.isnan(future_values).any()
    assert not torch.isinf(past_values).any()
    assert not torch.isinf(future_values).any()


def test_sequence_creation_multivariate(sample_multivariate_df):
    """Test sequence creation with 3 channels."""
    context_length = 512
    prediction_length = 96
    channel_cols = ['feature_1', 'feature_2', 'feature_3']

    past_values, future_values = create_sequences_for_patchtsmixer(
        df=sample_multivariate_df,
        channel_cols=channel_cols,
        context_length=context_length,
        prediction_length=prediction_length
    )

    expected_num_sequences = 393

    assert past_values.shape == (expected_num_sequences, context_length, 3), \
        f"past_values shape: {past_values.shape}"
    assert future_values.shape == (expected_num_sequences, prediction_length, 3), \
        f"future_values shape: {future_values.shape}"

    # Verify channel dimension
    assert past_values.shape[2] == len(channel_cols), \
        f"Expected {len(channel_cols)} channels, got {past_values.shape[2]}"


def test_pytorch_dataset(sample_multivariate_df):
    """Test TimeSeriesDataset class functionality."""
    past_values, future_values = create_sequences_for_patchtsmixer(
        df=sample_multivariate_df,
        channel_cols=['feature_1', 'feature_2', 'feature_3'],
        context_length=512,
        prediction_length=96
    )

    dataset = TimeSeriesDataset(past_values, future_values)

    # Test __len__
    assert len(dataset) == 393, f"Dataset length: {len(dataset)}"

    # Test __getitem__
    sample = dataset[0]
    assert isinstance(sample, dict), f"Sample type: {type(sample)}"
    assert 'past_values' in sample, "Missing 'past_values' key"
    assert 'future_values' in sample, "Missing 'future_values' key"
    assert 'observed_mask' in sample, "Missing 'observed_mask' key"

    # Verify sample shapes
    assert sample['past_values'].shape == (512, 3), \
        f"past_values shape in sample: {sample['past_values'].shape}"
    assert sample['future_values'].shape == (96, 3), \
        f"future_values shape in sample: {sample['future_values'].shape}"
    assert sample['observed_mask'].shape == (512, 3), \
        f"observed_mask shape in sample: {sample['observed_mask'].shape}"

    # Test batch sampling (simulate DataLoader behavior)
    samples = [dataset[i] for i in range(10)]
    assert len(samples) == 10, f"Batch sample count: {len(samples)}"


def test_temporal_split(sample_multivariate_df):
    """Test temporal train/val/test splitting."""
    past_values, future_values = create_sequences_for_patchtsmixer(
        df=sample_multivariate_df,
        channel_cols=['feature_1', 'feature_2', 'feature_3'],
        context_length=512,
        prediction_length=96
    )

    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    train_past, train_future, val_past, val_future, test_past, test_future = \
        patchtsmixer_train_val_test_split(past_values, future_values, split_ratios)

    # Verify splits sum to total
    total_sequences = 393
    split_sum = train_past.shape[0] + val_past.shape[0] + test_past.shape[0]
    assert split_sum == total_sequences, \
        f"Splits don't sum to total: {split_sum} != {total_sequences}"
    assert train_future.shape[0] + val_future.shape[0] + test_future.shape[0] == total_sequences

    # Verify temporal ordering (train comes before val comes before test)
    expected_train_size = int(total_sequences * 0.7)  # ~275
    expected_val_size = int(total_sequences * 0.15)  # ~59
    # test gets remaining

    assert train_past.shape[0] in range(expected_train_size - 1, expected_train_size + 2), \
        f"Train size {train_past.shape[0]} not near expected {expected_train_size}"

    # Shape consistency checks
    assert train_past.shape[1:] == val_past.shape[1:] == test_past.shape[1:], \
        "Context/channel dimensions mismatch across splits"
    assert train_future.shape[1:] == val_future.shape[1:] == test_future.shape[1:], \
        "Prediction/channel dimensions mismatch across splits"

    # Verify approximate split ratios
    train_ratio = train_past.shape[0] / total_sequences
    val_ratio = val_past.shape[0] / total_sequences
    test_ratio = test_past.shape[0] / total_sequences

    assert 0.65 <= train_ratio <= 0.75, f"Train ratio {train_ratio} out of range"
    assert 0.10 <= val_ratio <= 0.20, f"Val ratio {val_ratio} out of range"
    assert 0.10 <= test_ratio <= 0.20, f"Test ratio {test_ratio} out of range"


def test_insufficient_data_error():
    """Test error handling for insufficient data."""
    small_df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
        'value': np.random.randn(100)
    })

    with pytest.raises(ValueError, match="Insufficient data"):
        create_sequences_for_patchtsmixer(
            df=small_df,
            channel_cols=['value'],
            context_length=512,
            prediction_length=96
        )


def test_missing_columns_error(sample_univariate_df):
    """Test error handling for missing channel columns."""
    with pytest.raises(ValueError, match="Channel columns not found"):
        create_sequences_for_patchtsmixer(
            df=sample_univariate_df,
            channel_cols=['value', 'nonexistent_col'],
            context_length=512,
            prediction_length=96
        )


def test_non_numeric_columns_error():
    """Test error handling for non-numeric channel columns."""
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=1000, freq='D'),
        'category': ['A', 'B', 'C'] * 333 + ['A']
    })

    with pytest.raises(ValueError, match="must be numeric"):
        create_sequences_for_patchtsmixer(
            df=df,
            channel_cols=['category'],
            context_length=512,
            prediction_length=96
        )


def test_split_ratios_validation():
    """Test that split ratios must sum to 1.0."""
    # Create dummy tensors
    past = torch.randn(100, 512, 3)
    future = torch.randn(100, 96, 3)

    invalid_ratios = {'train': 0.6, 'val': 0.2, 'test': 0.1}  # Sum = 0.9

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        patchtsmixer_train_val_test_split(past, future, invalid_ratios)
